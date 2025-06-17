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
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.recorder import get_instance
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local
from .const import DEFAULT_NAME, DEFAULT_PRIOR, DOMAIN, MIN_PROBABILITY
from .data.config import ConfigManager
from .data.entity import EntityManager
from .data.entity_type import EntityTypeManager
from .data.prior import PriorManager
from .storage import StorageManager
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
            setup_method=self._async_coordinator_setup,
            update_method=self._async_update_data,
        )
        self.hass = hass
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        self.config_manager = ConfigManager(self)
        self.config = self.config_manager.config
        self.storage = StorageManager(self)
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
        self._storage_save_debouncer = Debouncer(
            hass=self.hass,
            logger=_LOGGER,
            cooldown=30.0,
            immediate=True,
            function=self._async_save_to_storage,
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry_id)},
            name=self.config.name,
            manufacturer="Area Occupancy",
            model="Area Occupancy Detection",
            sw_version="1.0.0",
        )

    @property
    def probability(self) -> float:
        """Calculate and return the current occupancy probability (0.0-1.0)."""
        return self._calculate_room_probability()

    @property
    def prior(self) -> float:
        """Calculate overall area prior from entity priors."""
        return self._calculate_entity_aggregates()["prior"]

    @property
    def decay(self) -> float:
        """Calculate the current decay probability (0.0-1.0)."""
        return self._calculate_entity_aggregates()["decay"]

    @property
    def next_prior_update(self) -> datetime | None:
        """Return the next scheduled prior update time."""
        return self._next_prior_update

    @property
    def last_prior_update(self) -> datetime | None:
        """Return the timestamp the priors were last calculated."""
        return self._last_prior_update

    @property
    def is_occupied(self) -> bool:
        """Return the current occupancy state (True/False)."""
        return self.probability >= self.config.threshold

    @property
    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0)."""
        return self.config.threshold if self.config else 0.5

    @property
    def last_updated(self) -> datetime | None:
        """Return the last updated timestamp."""
        return self.data.get("last_updated")

    @property
    def last_changed(self) -> datetime | None:
        """Return the last changed timestamp."""
        return self.data.get("last_changed")

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

    # --- Public Methods ---
    async def _async_coordinator_setup(self) -> None:
        """Initialize the coordinator and its components."""
        try:
            _LOGGER.debug("Starting coordinator setup for %s", self.config.name)

            # Initialize components in order
            await self.storage.async_initialize()
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

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # Cancel prior update tracker
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        # Stop global decay timer
        self._stop_global_decay_timer()

        # Clean up state change listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Shutdown storage save debouncer
        self._storage_save_debouncer.async_shutdown()

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
        await self._storage_save_debouncer.async_call()

        await self.async_request_refresh()

    async def async_load_stored_data(self) -> None:
        """Load and restore data from storage."""
        try:
            _LOGGER.debug("Loading stored data from storage")

            # Use storage manager's compatibility checking method
            (
                loaded_data,
                was_reset,
            ) = await self.storage.async_load_with_compatibility_check(
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

            # Check if recorder is available
            recorder = get_instance(self.hass)
            if recorder is None:
                _LOGGER.warning(
                    "Skipping prior update for area %s - Recorder is not available",
                    self.config.name,
                )
                return

            effective_period = history_period or self.config.history.period
            _LOGGER.info(
                "Starting learned priors update for area %s with %d days history",
                self.config.name,
                effective_period,
            )

            # Delegate the actual prior updating to the PriorManager
            updated_count = await self.priors.update_all_entity_priors(history_period)

            # Request update and save immediately if priors changed
            if updated_count > 0:
                await self._storage_save_debouncer.async_call()
                await self.async_request_refresh()

            _LOGGER.info(
                "Completed learned priors update for area %s: updated %d entities",
                self.config.name,
                updated_count,
            )

        except ValueError as err:
            _LOGGER.warning("Cannot update priors: %s", err)
        except RuntimeError as err:
            if "cannot schedule new futures after shutdown" in str(err):
                _LOGGER.warning(
                    "Skipping prior update for area %s - Home Assistant is shutting down",
                    self.config.name,
                )
            else:
                raise
        finally:
            # Always reschedule the next update
            await self._schedule_next_prior_update()

    # --- Prior Update Handling ---
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

    def _calculate_entity_aggregates(self) -> dict[str, float]:
        """Calculate probability, prior, and decay aggregates from entities in a single pass.

        Returns:
            Dict with 'probability', 'prior', and 'decay' keys

        """
        if not self.entities.entities:
            return {
                "probability": MIN_PROBABILITY,
                "prior": DEFAULT_PRIOR,
                "decay": 1.0,
            }

        # Single pass through entities
        total_weight = 0.0
        weighted_prob_sum = 0.0
        prior_sum = 0.0
        prior_count = 0
        decay_weight = 0.0
        weighted_decay_sum = 0.0

        for entity in self.entities.entities.values():
            if entity.available:
                weight = entity.type.weight
                total_weight += weight
                weighted_prob_sum += entity.probability * weight

                # For decay calculation
                if entity.decay.is_decaying:
                    decay_weight += weight
                    weighted_decay_sum += entity.decay.decay_factor * weight

            # Prior calculation (includes all entities)
            prior_sum += entity.prior.prior
            prior_count += 1

        # Calculate results
        probability = (
            (weighted_prob_sum / total_weight) if total_weight > 0 else MIN_PROBABILITY
        )
        prior = (prior_sum / prior_count) if prior_count > 0 else DEFAULT_PRIOR
        decay = (weighted_decay_sum / decay_weight) if decay_weight > 0 else 1.0

        return {
            "probability": validate_prob(probability),
            "prior": prior,
            "decay": decay,
        }

    def _calculate_room_probability(self) -> float:
        """Calculate room-level probability using fusion logic from demo app."""
        if not self.entities.entities:
            return MIN_PROBABILITY

        # Start with the prior probability (like demo app's prob_occupied)
        posterior = self.prior

        # Apply Bayesian fusion for each entity (like demo app's room-level fusion)
        for entity in self.entities.entities.values():
            # Only apply Bayesian update if sensor has active evidence
            # (currently ON or decaying from recent activity)
            if entity.is_active or entity.decay.is_decaying:
                posterior = bayesian_probability(
                    prior=posterior,
                    prob_given_true=entity.prior.prob_given_true,
                    prob_given_false=entity.prior.prob_given_false,
                    is_active=True,  # use weight * decay as fractional power
                    weight=entity.type.weight * entity.decay.decay_factor,
                    decay_factor=1.0,
                )

        return validate_prob(posterior)

    # --- Global Decay Timer Management ---
    def async_notify_decay_started(self) -> None:
        """Notify coordinator that an entity has started decaying."""
        if not self.config.decay.enabled:
            return

        if self._global_decay_timer is None:
            self._start_global_decay_timer()
            _LOGGER.debug("Started global decay timer")

    def async_notify_decay_stopped(self) -> None:
        """Notify coordinator that an entity has stopped decaying."""
        # Check if any entities are still decaying
        decaying_entities = [
            entity
            for entity in self.entities.entities.values()
            if entity.decay.is_decaying
        ]

        if not decaying_entities and self._global_decay_timer is not None:
            self._stop_global_decay_timer()
            _LOGGER.debug("Stopped global decay timer - no entities decaying")

    def _start_global_decay_timer(self) -> None:
        """Start the global decay timer."""
        if self._global_decay_timer is not None:
            # Timer already running
            return

        if not self.hass:
            _LOGGER.warning("Cannot start global decay timer: no hass instance")
            return

        self._schedule_next_global_decay_update()

    def _stop_global_decay_timer(self) -> None:
        """Stop the global decay timer."""
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

    def _schedule_next_global_decay_update(self) -> None:
        """Schedule the next global decay update."""
        if not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(seconds=DECAY_INTERVAL)
        self._global_decay_timer = async_track_point_in_time(
            self.hass, self._handle_global_decay_timer, next_update
        )

    async def _handle_global_decay_timer(self, _now: datetime) -> None:
        """Handle global decay timer firing - refresh coordinator for UI updates."""
        self._global_decay_timer = None

        # Check if decay is globally disabled
        if not self.config.decay.enabled:
            _LOGGER.debug("Decay disabled globally, stopping global timer")
            return

        # Get all decaying entities
        decaying_entities = [
            entity
            for entity in self.entities.entities.values()
            if entity.decay.is_decaying
        ]

        if not decaying_entities:
            _LOGGER.debug("No entities decaying, stopping global timer")
            return

        _LOGGER.debug(
            "Refreshing coordinator for %d decaying entities", len(decaying_entities)
        )

        # Just refresh the coordinator - decay factors are calculated on-demand
        # This updates the UI and triggers any necessary state changes
        await self.async_refresh()

        # Schedule next update if any entities are still decaying
        # (decay_factor property will auto-stop decay when factor < 0.05)
        remaining_decaying = [
            entity
            for entity in self.entities.entities.values()
            if entity.decay.is_decaying
        ]

        if remaining_decaying:
            self._schedule_next_global_decay_update()
        else:
            _LOGGER.debug("No entities still decaying, stopping global timer")

    async def _async_update_data(self) -> dict[str, Any]:
        """Update coordinator state and trigger debounced storage save.

        Returns:
            dict[str, Any]: The updated data dictionary

        Raises:
            HomeAssistantError: If there's an error updating the data

        """
        try:
            _LOGGER.debug(
                "Starting coordinator data update for area %s", self.config.name
            )

            # Update internal coordinator state - the main purpose of this method
            current_data = {
                "last_updated": dt_util.utcnow().isoformat(),
                "probability": self.probability,
                "is_occupied": self.is_occupied,
                "prior": self.prior,
                "threshold": self.threshold,
                "entity_ids": self.entities.entity_ids,
                "entity_types": self.entity_types.entity_types,
            }

            # Trigger debounced storage save (non-blocking)
            await self._storage_save_debouncer.async_call()

            _LOGGER.debug(
                "Coordinator data update completed: probability=%.3f, is_occupied=%s, will notify %d listeners",
                self.probability,
                self.is_occupied,
                len(self._listeners),
            )

        except (HomeAssistantError, ValueError) as err:
            _LOGGER.error("Error updating coordinator data: %s", err)
            raise HomeAssistantError(f"Error updating data: {err}") from err

        else:
            return current_data

    async def _async_save_to_storage(self) -> None:
        """Save coordinator data to storage (debounced)."""
        try:
            await self.storage.async_save_instance_data(
                self.entry_id,
                self.entities,
            )
            _LOGGER.debug("Debounced storage save completed for %s", self.config.name)
        except HomeAssistantError as err:
            _LOGGER.error("Failed to save data to storage: %s", err)
