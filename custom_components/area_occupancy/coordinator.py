"""Area Occupancy Coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta

# Standard Library
import logging
from typing import Any

# Third Party
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import async_track_point_in_time
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local
from .const import CONF_THRESHOLD, DEFAULT_NAME, DOMAIN, MIN_PROBABILITY
from .exceptions import CalculationError, StateError, StorageError
from .models.config import ConfigManager
from .models.entity import EntityManager
from .models.entity_type import EntityTypeManager
from .models.prior import PriorManager
from .storage import StorageManager
from .utils import validate_prob

_LOGGER = logging.getLogger(__name__)


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
            update_interval=timedelta(seconds=1),
        )
        self.hass = hass
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        self.config_manager = ConfigManager(self)
        self.config = self.config_manager.config
        self.storage = StorageManager(hass)
        self.entity_types = EntityTypeManager(self)
        self.priors = PriorManager(self)
        self.entities = EntityManager(self)
        self._prior_update_interval = timedelta(hours=1)  # Default to 1 hour
        self._next_prior_update: datetime | None = None
        self._last_prior_update: datetime | None = None
        self._prior_update_tracker: CALLBACK_TYPE | None = None
        self._storage_lock = (
            asyncio.Lock()
        )  # Add storage lock to prevent race conditions

    @property
    def available(self) -> bool:
        """Return if the coordinator is available."""
        return any(entity.available for entity in self.entities.entities.values())

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
    def complementary_probability(self) -> float:
        """Calculate the complementary probability across all entities.

        This represents the overall probability that the area is occupied,
        taking into account all entity probabilities and their weights.

        Returns:
            float: The complementary probability (0.0-1.0)

        """
        if not self.entities.entities:
            return MIN_PROBABILITY

        # Calculate weighted sum of all entity probabilities
        total_weight = 0.0
        weighted_sum = 0.0

        for entity in self.entities.entities.values():
            if entity.is_active:  # Only consider active entities
                weight = entity.type.weight
                total_weight += weight
                weighted_sum += entity.probability.decayed_probability * weight

        if total_weight == 0:
            return MIN_PROBABILITY

        # Calculate final probability
        final_probability = weighted_sum / total_weight
        return validate_prob(final_probability)

    @property
    def prior_update_interval(self) -> timedelta:
        """Return the interval between prior updates."""
        return self._prior_update_interval

    @property
    def next_prior_update(self) -> datetime | None:
        """Return the next scheduled prior update time."""
        return self._next_prior_update

    @property
    def last_prior_update(self) -> datetime | None:
        """Return the timestamp the priors were last calculated."""
        return self._last_prior_update

    @property
    def probability(self) -> float:
        """Return the current occupancy probability (0.0-1.0)."""
        return self.complementary_probability

    @property
    def is_occupied(self) -> bool:
        """Return the current occupancy state (True/False)."""
        return self.probability >= self.config.threshold

    @property
    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0)."""
        return self.config.threshold if self.config else 0.5

    # --- Public Methods ---
    async def async_setup(self) -> None:
        """Set up the coordinator, load data, initialize states, check priors, and schedule updates."""
        try:
            _LOGGER.debug("Starting coordinator setup for %s", self.config.name)

            # Load stored data first
            await self.async_load_stored_data()

            # Initialize states after loading stored data
            await self.entities.async_initialize()

            # Log all registered entities
            _LOGGER.debug("Registered entities for %s:", self.config.name)
            for entity in self.entities.entities.values():
                _LOGGER.debug("  - %s (type: %s)", entity.entity_id, entity.type)

            # Schedule periodic prior updates
            await self._schedule_next_prior_update()

            # Wait for any pending operations to complete before initial refresh
            await asyncio.sleep(0)

            # Trigger an initial refresh after setup is complete
            await self.async_refresh()

            _LOGGER.debug(
                "Successfully set up AreaOccupancyCoordinator for %s",
                self.config.name,
            )

        except (StorageError, StateError, CalculationError) as err:
            _LOGGER.error("Failed to set up coordinator: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up coordinator: {err}") from err

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # Cancel prior update tracker
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        # Clean up entity manager
        self.entities.cleanup()

        await super().async_shutdown()

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options."""
        # Update config
        self.config_manager.update_config(options)
        self.config = self.config_manager.config

        # Update prior update interval if changed
        if "prior_update_interval" in options:
            self._prior_update_interval = timedelta(
                hours=options["prior_update_interval"]
            )

        # Schedule next update
        await self._schedule_next_prior_update()

    async def async_update_threshold(self, value: float) -> None:
        """Update the threshold value.

        Args:
            value: The new threshold value as a percentage (1-99)

        Raises:
            ServiceValidationError: If the value is invalid
            HomeAssistantError: If there's an error updating the config entry

        """
        _LOGGER.debug("Updating threshold: %.2f", value)

        self.config_manager.update_config(
            {
                CONF_THRESHOLD: value,
            }
        )

    async def async_load_stored_data(self) -> None:
        """Load and restore data from storage."""
        try:
            _LOGGER.debug("Loading stored data from storage")

            # Load instance data
            loaded_data = await self.storage.async_load_instance_data(self.entry_id)

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
            else:
                _LOGGER.info(
                    "No stored data found for instance %s, initializing with defaults",
                    self.entry_id,
                )
                self._last_prior_update = None

            _LOGGER.debug(
                "Successfully restored stored data for instance %s",
                self.entry_id,
            )
        except StorageError as err:
            _LOGGER.warning(
                "Storage error for instance %s, initializing with defaults: %s",
                self.entry_id,
                err,
            )
            self._last_prior_update = None
            # Re-raise as ConfigEntryNotReady if loading fails critically
            raise ConfigEntryNotReady(f"Failed to load stored data: {err}") from err

    async def update_learned_priors(self, history_period: int | None = None) -> None:
        """Update learned priors using historical data."""
        # TODO: Implement prior update

        await self._schedule_next_prior_update()

    # --- Prior Update Handling ---
    async def _schedule_next_prior_update(self) -> None:
        """Schedule the next prior update at the start of the next hour."""
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        now = dt_util.utcnow()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        self._next_prior_update = next_hour

        self._prior_update_tracker = async_track_point_in_time(
            self.hass, self._handle_prior_update, self._next_prior_update
        )
        _LOGGER.debug(
            "Scheduled next prior update for %s in area %s",
            self._next_prior_update.isoformat(),
            self.config.name,
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
            _LOGGER.info(
                "Finished scheduled prior update task trigger for area %s",
                self.config.name,
            )
        except Exception:
            _LOGGER.exception(
                "Error occurred during scheduled prior update for area %s",
                self.config.name,
            )
            # Ensure rescheduling happens even if the update task itself failed unexpectedly
            if not self._prior_update_tracker:
                _LOGGER.warning(
                    "Update_learned_priors failed to reschedule, attempting fallback reschedule"
                )
                await self._schedule_next_prior_update()

    # --- Data Saving ---
    async def _async_save_data(self) -> None:
        """Save the current data to storage."""
        async with self._storage_lock:
            try:
                _LOGGER.debug("Attempting to save data")
                await self.storage.async_save_instance_data(
                    self.entry_id,
                    self.entities,
                )
                _LOGGER.debug("Data saved successfully")
            except (StorageError, HomeAssistantError) as err:
                _LOGGER.error("Failed to save data: %s", err)
                raise StorageError(f"Failed to save data: {err}") from err

    # --- Listener Handling ---
    @callback
    def _async_refresh_finished(self) -> None:
        """Handle when a refresh has finished."""
        if self.last_update_success:
            _LOGGER.debug("Coordinator refresh finished successfully")
        else:
            _LOGGER.warning("Coordinator refresh failed")

    @callback
    def async_set_updated_data(self, data: dict[str, Any]) -> None:
        """Manually update data and notify listeners."""
        super().async_set_updated_data(data)

    @callback
    def async_add_listener(
        self, update_callback: CALLBACK_TYPE, context: Any = None
    ) -> Callable[[], None]:
        """Add a listener for data updates with improved tracking."""
        return super().async_add_listener(update_callback, context)

    async def _async_update_data(self) -> dict[str, Any]:
        """Update data from entities and calculate probabilities.

        Returns:
            dict[str, Any]: The updated data dictionary

        Raises:
            HomeAssistantError: If there's an error updating the data

        """
        try:
            _LOGGER.debug("Starting data update for area %s", self.config.name)

            # Update entity states and calculate probabilities
            for entity in self.entities.entities.values():
                try:
                    await entity.async_update()
                except (StateError, ValueError) as err:
                    _LOGGER.error(
                        "Error updating entity %s: %s",
                        entity.entity_id,
                        err,
                    )
                    # Continue with other entities even if one fails
                    continue

            # Calculate overall probability
            probability = self.complementary_probability
            is_occupied = probability >= self.config.threshold

            # Prepare data dictionary
            data = {
                "probability": probability,
                "is_occupied": is_occupied,
                "threshold": self.config.threshold,
                "last_updated": dt_util.utcnow().isoformat(),
                "entities": {
                    entity_id: {
                        "probability": entity.probability.decayed_probability,
                        "is_active": entity.is_active,
                        "state": entity.state,
                        "last_updated": entity.last_updated.isoformat(),
                    }
                    for entity_id, entity in self.entities.entities.items()
                },
            }

            # Save data to storage
            await self._async_save_data()

            _LOGGER.debug(
                "Data update completed for area %s: probability=%.2f, is_occupied=%s",
                self.config.name,
                probability,
                is_occupied,
            )

        except (StateError, ValueError) as err:
            _LOGGER.error("Error updating data: %s", err)
            raise HomeAssistantError(f"Error updating data: {err}") from err
        else:
            return data
