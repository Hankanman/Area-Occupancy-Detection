"""Area Occupancy Coordinator."""

from __future__ import annotations

from datetime import datetime, timedelta

# Standard Library
import logging
from typing import Any

# Third Party
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local
from .const import (
    DEFAULT_NAME,
    DEFAULT_PRIOR,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)
from .data.config import ConfigManager
from .data.entity import EntityManager
from .data.entity_type import EntityTypeManager
from .data.prior import PriorManager
from .storage import AreaOccupancyStore
from .utils import bayesian_probability, validate_prob

_LOGGER = logging.getLogger(__name__)

DECAY_INTERVAL = 5  # seconds


class AreaOccupancyCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Manage fetching and combining data for area occupancy."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=config_entry.data.get(CONF_NAME, DEFAULT_NAME),
            update_interval=None,
            setup_method=self.setup,
            update_method=self.update,
        )
        self.hass = hass
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        self.config_manager = ConfigManager(self)
        self.config = self.config_manager.config
        self.store = AreaOccupancyStore(self)
        self.entity_types = EntityTypeManager(self)
        self.priors = PriorManager(self)
        self.entities = EntityManager(self)
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None
        self._next_prior_update: datetime | None = None
        self._last_prior_update: datetime | None = None
        self._prior_update_tracker: CALLBACK_TYPE | None = None
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._remove_state_listener: CALLBACK_TYPE | None = None

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry_id)},
            name=self.config.name,
            manufacturer=DEVICE_MANUFACTURER,
            model=DEVICE_MODEL,
            sw_version=DEVICE_SW_VERSION,
        )

    @property
    def probability(self) -> float:
        """Calculate and return the current occupancy probability (0.0-1.0)."""
        if not self.entities.entities:
            return MIN_PROBABILITY

        # Start with the prior probability (like demo app's prob_occupied)
        posterior = self.prior

        # Apply Bayesian fusion for each entity (like demo app's room-level fusion)
        for entity in self.entities.entities.values():
            # Only apply Bayesian update if sensor has active evidence
            # (currently ON or decaying from recent activity)
            if entity.evidence or entity.decay.is_decaying:
                posterior = bayesian_probability(
                    prior=posterior,
                    prob_given_true=entity.prior.prob_given_true,
                    prob_given_false=entity.prior.prob_given_false,
                    evidence=True,  # use weight * decay as fractional power
                    weight=entity.type.weight * entity.decay.decay_factor,
                    decay_factor=1.0,
                )

        return validate_prob(posterior)

    @property
    def prior(self) -> float:
        """Calculate overall area prior from entity priors."""
        if not self.entities.entities:
            return DEFAULT_PRIOR

        # Simple average of all entity priors
        prior_sum = sum(
            entity.prior.prior for entity in self.entities.entities.values()
        )
        return prior_sum / len(self.entities.entities)

    @property
    def decay(self) -> float:
        """Calculate the current decay probability (0.0-1.0)."""
        if not self.entities.entities:
            return 1.0

        decay_sum = sum(
            entity.decay.decay_factor for entity in self.entities.entities.values()
        )
        return decay_sum / len(self.entities.entities)

    @property
    def occupied(self) -> bool:
        """Return the current occupancy state (True/False)."""
        return self.probability >= self.config.threshold

    @property
    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0)."""
        return self.config.threshold if self.config else 0.5

    @property
    def binary_sensor_entity_ids(self) -> dict[str, str | None]:
        """Return the entity_ids of the binary sensors created by this integration.

        Returns:
            dict[str, str | None]: Dictionary with 'occupancy' and 'wasp' keys
                                 containing their respective entity_ids or None if not set.

        """
        return {
            "occupancy": self.occupancy_entity_id,
            "wasp": self.wasp_entity_id,
        }

    @property
    def decaying_entities(self) -> list:
        """Get list of entities currently decaying."""
        return [
            entity
            for entity in self.entities.entities.values()
            if entity.decay.is_decaying
        ]

    # --- Public Methods ---
    async def setup(self) -> None:
        """Initialize the coordinator and its components."""
        try:
            _LOGGER.debug("Starting coordinator setup for %s", self.config.name)

            # Initialize components in order
            await self.entity_types.async_initialize()
            await self.async_load_stored_data()
            await self.entities.async_initialize()
            await self.track_entity_state_changes(self.entities.entity_ids)

            # Schedule periodic prior updates
            await self._schedule_next_prior_update()

            _LOGGER.debug(
                "Successfully set up AreaOccupancyCoordinator for %s with %d entities",
                self.config.name,
                len(self.entities.entities),
            )

        except HomeAssistantError as err:
            _LOGGER.error("Failed to set up coordinator: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up coordinator: {err}") from err

    async def update(self) -> dict[str, Any]:
        """Update and return the current coordinator data."""
        # Check and manage decay timer based on current entity states
        self._manage_decay_timer()

        # Save current state to storage
        self.store.async_save_coordinator_data(self.entities)

        # Return current state data
        return {
            "probability": self.probability,
            "occupied": self.occupied,
            "threshold": self.threshold,
            "prior": self.prior,
            "decay": self.decay,
            "last_updated": dt_util.utcnow(),
            "next_prior_update": self._next_prior_update,
            "last_prior_update": self._last_prior_update,
        }

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # Cancel prior update tracker
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        # Stop global decay timer
        self._stop_decay_timer()

        # Clean up state change listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Clean up entity manager
        self.entities.cleanup()

        await super().async_shutdown()

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options."""
        # Update config
        await self.config_manager.update_config(options)
        self.config = self.config_manager.config

        # Always re-initialize entities and entity types when configuration changes
        _LOGGER.info(
            "Configuration updated, re-initializing entities for %s", self.config.name
        )

        # Update entity types with new configuration
        self.entity_types.cleanup()
        await self.entity_types.async_initialize()

        # Clean up existing entity tracking and re-initialize
        self.entities.cleanup()
        await self.entities.async_initialize()

        # Re-establish entity state tracking with new entity list
        await self.track_entity_state_changes(self.entities.entity_ids)

        # Schedule next update and refresh data
        await self._schedule_next_prior_update()

        # Force immediate save after configuration changes
        self.store.async_save_coordinator_data(self.entities)

        await self.async_request_refresh()

    async def async_load_stored_data(self) -> None:
        """Load and restore data from storage."""
        try:
            _LOGGER.debug("Loading stored data from storage")

            # Load data with compatibility checking
            (
                loaded_data,
                was_reset,
            ) = await self.store.async_load_with_compatibility_check(
                self.entry_id, self.config_entry.version if self.config_entry else 9
            )

            if loaded_data:
                last_updated_str = loaded_data.get("last_updated")
                _LOGGER.debug(
                    "Found stored data for instance %s, restoring (last saved: %s)",
                    self.entry_id,
                    last_updated_str,
                )

                # Create entity manager from loaded data
                self.entities = EntityManager.from_dict(loaded_data, self)
                if last_updated_str:
                    self._last_prior_update = dt_util.parse_datetime(last_updated_str)
                else:
                    self._last_prior_update = None

                _LOGGER.debug(
                    "Successfully restored stored data for instance %s",
                    self.entry_id,
                )

            else:
                # No data loaded (either no data exists, was reset, or had errors)
                if was_reset:
                    _LOGGER.info(
                        "Storage was reset for instance %s, will initialize with defaults",
                        self.entry_id,
                    )
                else:
                    _LOGGER.info(
                        "No stored data found for instance %s, initializing with defaults",
                        self.entry_id,
                    )
                self._last_prior_update = None

        except HomeAssistantError as err:
            _LOGGER.warning(
                "Storage error for instance %s, initializing with defaults: %s",
                self.entry_id,
                err,
            )
            self._last_prior_update = None
            # Re-raise as ConfigEntryNotReady if loading fails critically
            raise ConfigEntryNotReady(f"Failed to load stored data: {err}") from err

    # --- Entity State Tracking ---
    async def track_entity_state_changes(self, entity_ids: list[str]) -> None:
        """Track state changes for a list of entity_ids."""
        # Clean up existing listener if it exists
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Only create new listener if we have entities to track
        if entity_ids:
            self._remove_state_listener = async_track_state_change_event(
                self.hass,
                entity_ids,
                self.entities.async_state_changed_listener,
            )

    # --- Prior Update Handling ---
    async def update_learned_priors(self, history_period: int | None = None) -> None:
        """Update learned priors using historical data.

        Args:
            history_period: Optional history period in days to override config value

        """
        try:
            # Check if Home Assistant is shutting down
            if self.hass.is_stopping:
                _LOGGER.warning(
                    "Skipping prior update for area %s - Home Assistant is shutting down",
                    self.config.name,
                )
                return

            # Delegate the actual prior updating to the PriorManager
            await self.priors.update_all_entity_priors(history_period)

            # Save and refresh if priors changed
            self.store.async_save_coordinator_data(self.entities)
            await self.async_request_refresh()

        except HomeAssistantError as err:
            # Handle all errors gracefully
            if "cannot schedule new futures after shutdown" in str(err):
                _LOGGER.warning(
                    "Skipping prior update for area %s - Home Assistant is shutting down",
                    self.config.name,
                )
            else:
                _LOGGER.warning(
                    "Error updating priors for area %s: %s", self.config.name, err
                )
        finally:
            # Always reschedule the next update
            await self._schedule_next_prior_update()

    async def _schedule_next_prior_update(self) -> None:
        """Schedule the next prior update at the start of the next hour."""
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()

        now = dt_util.utcnow()
        self._next_prior_update = now.replace(
            minute=0, second=0, microsecond=0
        ) + timedelta(hours=1)

        self._prior_update_tracker = async_track_point_in_time(
            self.hass, self._handle_prior_update, self._next_prior_update
        )
        _LOGGER.debug(
            "Scheduled next prior update for %s", self._next_prior_update.isoformat()
        )

    async def _handle_prior_update(self, _now: datetime) -> None:
        """Handle the prior update task."""
        self._prior_update_tracker = None
        self._next_prior_update = None

        try:
            _LOGGER.info(
                "Performing scheduled prior update for area %s", self.config.name
            )
            await self.update_learned_priors()
        except Exception:
            _LOGGER.exception("Error during scheduled prior update")
            # Reschedule even if update failed
            await self._schedule_next_prior_update()

    # --- Decay Timer Handling ---
    def _manage_decay_timer(self) -> None:
        """Manage decay timer based on current entity states and config."""
        # Stop timer if decay is disabled globally
        if not self.config.decay.enabled:
            self._stop_decay_timer()
            return

        decaying_entities = self.decaying_entities

        # Start timer if we have decaying entities but no timer running
        if decaying_entities and self._global_decay_timer is None:
            self._start_decay_timer()
            _LOGGER.debug("Started global decay timer")
        # Stop timer if no entities are decaying but timer is running
        elif not decaying_entities and self._global_decay_timer is not None:
            self._stop_decay_timer()
            _LOGGER.debug("Stopped global decay timer - no entities decaying")

    def _start_decay_timer(self) -> None:
        """Start the global decay timer (inline implementation)."""
        if self._global_decay_timer is not None or not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(seconds=DECAY_INTERVAL)
        self._global_decay_timer = async_track_point_in_time(
            self.hass, self._handle_decay_timer, next_update
        )

    def _stop_decay_timer(self) -> None:
        """Stop the global decay timer (inline implementation)."""
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

    async def _handle_decay_timer(self, _now: datetime) -> None:
        """Handle decay timer firing - refresh coordinator and reschedule if needed."""
        self._global_decay_timer = None

        # Early exit if decay is disabled globally
        if not self.config.decay.enabled:
            _LOGGER.debug("Decay disabled globally, stopping timer")
            return

        decaying_entities = self.decaying_entities

        # Early exit if no entities are decaying
        if not decaying_entities:
            _LOGGER.debug("No entities decaying, stopping timer")
            return

        # Refresh the coordinator for UI updates
        await self.async_refresh()

        # Reschedule timer if entities are still decaying after refresh
        if self.decaying_entities:
            self._start_decay_timer()
