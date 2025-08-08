"""Area Occupancy Coordinator."""

from __future__ import annotations

from datetime import datetime, timedelta

# Standard Library
import logging
from typing import Any

from custom_components.area_occupancy.db import AreaOccupancyDB

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
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)
from .data.config import Config
from .data.entity import EntityFactory, EntityManager
from .data.entity_type import InputType
from .data.prior import Prior
from .data.purpose import PurposeManager
from .utils import bayesian_probability

_LOGGER = logging.getLogger(__name__)

# Global timer intervals in seconds
DECAY_INTERVAL = 10
ANALYSIS_INTERVAL = 3600  # 1 hour in seconds


class AreaOccupancyCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Manage fetching and combining data for area occupancy."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
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
        self.db = AreaOccupancyDB(self)
        self.config = Config(self)
        self.factory = EntityFactory(self)
        self.prior = Prior(self)
        self.purpose = PurposeManager(self)
        self.entities = EntityManager(self)
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._remove_state_listener: CALLBACK_TYPE | None = None
        self._analysis_timer: CALLBACK_TYPE | None = None

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

        return bayesian_probability(
            entities=self.entities.entities,
            area_prior=self.prior.value,
            time_prior=self.prior.time_prior,
        )

    @property
    def type_probabilities(self) -> dict[str, float]:
        """Calculate and return the current occupancy probabilities for each entity type (0.0-1.0)."""
        if not self.entities.entities:
            return {}

        return {
            InputType.MOTION: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.MOTION),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.MEDIA: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.MEDIA),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.APPLIANCE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.APPLIANCE),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.DOOR: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.DOOR),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.WINDOW: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.WINDOW),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.ILLUMINANCE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(
                    InputType.ILLUMINANCE
                ),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.HUMIDITY: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.HUMIDITY),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.TEMPERATURE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(
                    InputType.TEMPERATURE
                ),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
        }

    @property
    def area_prior(self) -> float:
        """Get the area's baseline occupancy prior from historical data.

        This returns the pure P(area occupied) without any sensor weighting.
        """
        # Use the dedicated area baseline prior calculation
        return self.prior.value

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

    # --- Public Methods ---
    async def setup(self) -> None:
        """Initialize the coordinator and its components."""
        try:
            _LOGGER.debug("Starting coordinator setup for %s", self.config.name)

            # Initialize purpose manager
            await self.purpose.async_initialize()

            # Ensure area exists in database
            await self.db.save_area_data()

            # Load stored data
            await self.db.load_data()

            is_empty = self.db.is_intervals_empty()
            if is_empty:
                _LOGGER.info(
                    "State intervals table is empty for instance %s. Populating with initial data from recorder.",
                    self.entry_id,
                )
                entity_ids = [eid for eid in set(self.config.entity_ids) if eid]
                if entity_ids:
                    await self.run_analysis()
                else:
                    _LOGGER.warning(
                        "No entity IDs found in configuration to populate state intervals table for instance %s.",
                        self.entry_id,
                    )
            else:
                _LOGGER.info(
                    "State intervals table is not empty for instance %s. Data already loaded from database.",
                    self.entry_id,
                )

            # Track entity state changes
            await self.track_entity_state_changes(self.entities.entity_ids)
            # Start timers only after everything is ready
            self._start_decay_timer()
            self._start_analysis_timer()
            await self.async_refresh()
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
        # Save current state to database
        await self.db.save_data()

        # Return current state data
        return {
            "probability": self.probability,
            "occupied": self.occupied,
            "threshold": self.threshold,
            "prior": self.area_prior,
            "decay": self.decay,
            "last_updated": dt_util.utcnow(),
        }

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # Cancel prior update tracker
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

        # Clean up state change listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Clean up historical timer
        if self._analysis_timer is not None:
            self._analysis_timer()
            self._analysis_timer = None

        await self.db.save_data()

        # Clean up entity manager
        await self.entities.cleanup()

        # Clean up purpose manager
        self.purpose.cleanup()

        await super().async_shutdown()

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options."""
        # Update config
        await self.config.update_config(options)

        # Update purpose with new configuration
        await self.purpose.async_initialize()

        # Always re-initialize entities and entity types when configuration changes
        _LOGGER.info(
            "Configuration updated, re-initializing entities for %s", self.config.name
        )

        # Clean up existing entity tracking and re-initialize
        await self.entities.cleanup()

        # Re-establish entity state tracking with new entity list
        await self.track_entity_state_changes(self.entities.entity_ids)

        # Force immediate save after configuration changes
        await self.db.save_data()

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

            async def _refresh_on_state_change(event):
                entity_id = event.data.get("entity_id")
                entity = self.entities.get_entity(entity_id)
                if entity and entity.has_new_evidence():
                    await self.async_refresh()

            self._remove_state_listener = async_track_state_change_event(
                self.hass, entity_ids, _refresh_on_state_change
            )

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

    # --- Historical Timer Handling ---
    def _start_analysis_timer(self) -> None:
        """Start the historical data import timer."""
        if self._analysis_timer is not None or not self.hass:
            return

        # Run first import shortly after startup (5 minutes)
        next_update = dt_util.utcnow() + timedelta(minutes=5)

        self._analysis_timer = async_track_point_in_time(
            self.hass, self.run_analysis, next_update
        )

    async def run_analysis(self, _now: datetime | None = None) -> None:
        """Handle the historical data import timer."""
        if _now is None:
            _now = dt_util.utcnow()
        self._analysis_timer = None

        try:
            # Import recent data from recorder
            await self.db.sync_states()

            # Recalculate priors with new data
            await self.prior.update()

            # Recalculate likelihoods with new data
            await self.entities.update_likelihoods()

            # Refresh the coordinator
            await self.async_refresh()

            # Save data to database
            await self.db.save_data()

        except (ValueError, OSError) as err:
            _LOGGER.error("Analysis failed: %s", err)
            await self.async_refresh()

        # Schedule next run (1 hour)
        next_update = dt_util.utcnow() + timedelta(seconds=ANALYSIS_INTERVAL)

        self._analysis_timer = async_track_point_in_time(
            self.hass, self.run_analysis, next_update
        )
