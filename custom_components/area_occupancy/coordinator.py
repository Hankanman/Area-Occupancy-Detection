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
from .data.prior import Prior
from .data.purpose import PurposeManager
from .storage import AreaOccupancyStore
from .utils import overall_probability

_LOGGER = logging.getLogger(__name__)

# Global timer intervals in seconds
DECAY_INTERVAL = 10
STORAGE_INTERVAL = 300
PRIOR_INTERVAL = 3600


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
        self.prior = Prior(self)
        self.store = AreaOccupancyStore(self)
        self.entity_types = EntityTypeManager(self)
        self.purpose = PurposeManager(self)
        self.entities = EntityManager(self)
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None
        self._global_prior_timer: CALLBACK_TYPE | None = None
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._global_storage_timer: CALLBACK_TYPE | None = None
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

        return overall_probability(
            entities=self.entities.entities,
            prior=self.area_prior,
        )

    @property
    def area_prior(self) -> float:
        """Get the area's baseline occupancy prior from historical data.

        This returns the pure P(area occupied) without any sensor weighting.
        """
        if not self.entities.entities:
            return DEFAULT_PRIOR

        # Use the dedicated area baseline prior calculation
        return self.prior.current_value

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

    # --- Public Methods ---
    async def setup(self) -> None:
        """Initialize the coordinator and its components."""
        try:
            _LOGGER.debug("Starting coordinator setup for %s", self.config.name)

            # Initialize purpose manager
            await self.purpose.async_initialize()

            # Build Default Entity Types
            await self.entity_types.async_initialize()

            # Load stored data
            loaded_data = await self.store.async_load_data()

            if loaded_data:
                # Create entity manager from loaded data
                self.entities = EntityManager.from_dict(dict(loaded_data), self)
                # Update restored entities with current configuration
                await self.entities.__post_init__()

                _LOGGER.debug(
                    "Successfully restored stored data for instance %s",
                    self.entry_id,
                )
            else:
                # No stored data - initialize entities from configuration
                self.entities = EntityManager(self)
                await self.entities.__post_init__()

                _LOGGER.debug(
                    "No stored data found, created fresh entities from configuration for %s",
                    self.entry_id,
                )

            # Calculate initial area baseline prior
            if self.config.history.enabled:
                await self.prior.update()
                await self.entities.update_all_entity_likelihoods()

            # Save current state to storage
            await self.store.async_save_data(force=True)

            # Track entity state changes
            await self.track_entity_state_changes(self.entities.entity_ids)

            # Start the prior update timer
            self._start_prior_timer()

            # Start the global decay timer
            self._start_decay_timer()

            # Start the global storage timer
            self._start_storage_timer()

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
        # Save current state to storage
        await self.store.async_save_data()

        # Return current state data
        return {
            "probability": self.probability,
            "occupied": self.occupied,
            "threshold": self.threshold,
            "prior": self.prior,
            "decay": self.decay,
            "last_updated": dt_util.utcnow(),
        }

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # Cancel prior update tracker
        if self._global_prior_timer is not None:
            self._global_prior_timer()
            self._global_prior_timer = None

        # Stop global decay timer
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

        # Stop global storage timer
        if self._global_storage_timer is not None:
            self._global_storage_timer()
            self._global_storage_timer = None

        # Clean up state change listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        await self.store.async_save_data(force=True)

        # Clean up entity manager
        await self.entities.cleanup()

        # Clean up purpose manager
        self.purpose.cleanup()

        await super().async_shutdown()

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options."""
        # Update config
        await self.config_manager.update_config(options)
        self.config = self.config_manager.config

        # Update purpose with new configuration
        await self.purpose.async_initialize()

        # Always re-initialize entities and entity types when configuration changes
        _LOGGER.info(
            "Configuration updated, re-initializing entities for %s", self.config.name
        )

        # Update entity types with new configuration
        self.entity_types.cleanup()
        await self.entity_types.async_initialize()

        # Clean up existing entity tracking and re-initialize
        await self.entities.cleanup()

        # Re-establish entity state tracking with new entity list
        await self.track_entity_state_changes(self.entities.entity_ids)

        # Force immediate save after configuration changes
        await self.store.async_save_data(force=True)

        await self.async_request_refresh()

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

    # --- Prior Timer Handling ---
    def _start_prior_timer(self) -> None:
        """Start the prior update timer."""
        if self._global_prior_timer is not None or not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(seconds=PRIOR_INTERVAL)

        self._global_prior_timer = async_track_point_in_time(
            self.hass, self._handle_prior_timer, next_update
        )

    async def _handle_prior_timer(self, _now: datetime) -> None:
        """Handle the prior update timer."""
        self._global_prior_timer = None

        history_period = self.config.history.period

        # Update learned priors if history is enabled
        if self.config.history.enabled:
            # Update area baseline prior separately (unweighted)
            await self.prior.update()

            # Update individual sensor likelihoods
            await self.entities.update_all_entity_likelihoods(history_period)

        # Reschedule the timer
        self._start_prior_timer()

        # Save current state to storage
        await self.store.async_save_data(force=True)

    # --- Decay Timer Handling ---
    def _start_decay_timer(self) -> None:
        """Start the global decay timer (always-on implementation)."""
        if self._global_decay_timer is not None or not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(seconds=DECAY_INTERVAL)

        self._global_decay_timer = async_track_point_in_time(
            self.hass, self._handle_decay_timer, next_update
        )

    async def _handle_decay_timer(self, _now: datetime) -> None:
        """Handle decay timer firing - refresh coordinator and always reschedule."""
        self._global_decay_timer = None

        # Refresh the coordinator if decay is enabled
        if self.config.decay.enabled:
            await self.async_refresh()

        # Reschedule the timer
        self._start_decay_timer()

    # --- Storage Timer Handling ---
    def _start_storage_timer(self) -> None:
        """Start the global storage timer."""
        if self._global_storage_timer is not None or not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(seconds=STORAGE_INTERVAL)

        self._global_storage_timer = async_track_point_in_time(
            self.hass, self._handle_storage_timer, next_update
        )

    async def _handle_storage_timer(self, _now: datetime) -> None:
        """Handle the storage timer."""
        self._global_storage_timer = None

        # Save current state to storage
        await self.store.async_save_data(force=True)

        # Reschedule the timer
        self._start_storage_timer()
