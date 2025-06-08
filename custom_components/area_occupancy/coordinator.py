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
            update_interval=timedelta(seconds=30),  # More reasonable update interval
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
        self._prior_update_interval = timedelta(hours=1)  # Default to 1 hour
        self._next_prior_update: datetime | None = None
        self._last_prior_update: datetime | None = None
        self._prior_update_tracker: CALLBACK_TYPE | None = None
        self._last_saved_data: dict[str, Any] | None = None  # Cache for smart saving
        self._cached_probability: float | None = (
            None  # Cache for probability calculation
        )

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

    def _calculate_probability(self) -> float:
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
    def complementary_probability(self) -> float:
        """Return the cached complementary probability, calculating if needed."""
        if self._cached_probability is None:
            self._cached_probability = self._calculate_probability()
        return self._cached_probability

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

    def invalidate_probability_cache(self) -> None:
        """Invalidate the cached probability to force recalculation.

        This should be called whenever:
        - An entity's probability changes
        - An entity's active state changes
        - Entity weights or configuration changes
        - Prior probabilities are updated
        """
        if self._cached_probability is not None:
            _LOGGER.debug(
                "Invalidating cached probability for area %s", self.config.name
            )
            self._cached_probability = None

    # --- Public Methods ---
    async def _async_setup(self) -> None:
        try:
            _LOGGER.debug("Starting coordinator setup for %s", self.config.name)

            # Initialize storage and load stored data first
            await self.storage.async_initialize()
            await self.async_load_stored_data()

            # Initialize states after loading stored data
            await self.entities.async_initialize()

            # Log all registered entities
            _LOGGER.debug("Registered entities for %s:", self.config.name)
            for entity in self.entities.entities.values():
                _LOGGER.debug("  - %s (type: %s)", entity.entity_id, entity.type)

            # Schedule periodic prior updates
            await self._schedule_next_prior_update()

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
        await self.config_manager.update_config(options)
        self.config = self.config_manager.config

        # Clear cached data since configuration changed
        self.invalidate_probability_cache()
        self._last_saved_data = None

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

        await self.config_manager.update_config(
            {
                CONF_THRESHOLD: value,
            }
        )

        # Invalidate cache since threshold affects is_occupied calculation
        self.invalidate_probability_cache()

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

            # Invalidate cache since restored entities may have different probabilities
            self.invalidate_probability_cache()

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
        """Update learned priors using historical data.

        Args:
            history_period: Number of days of history to analyze (defaults to config value)

        """
        try:
            _LOGGER.info("Starting learned priors update for area %s", self.config.name)

            # Delegate the actual prior updating to the PriorManager
            updated_count = await self.priors.update_all_entity_priors(history_period)

            # Invalidate cache since priors affect probability calculations
            if updated_count > 0:
                self.invalidate_probability_cache()

            _LOGGER.info(
                "Completed learned priors update for area %s: updated %d entities",
                self.config.name,
                updated_count,
            )

        except ValueError as err:
            _LOGGER.warning("Cannot update priors: %s", err)
        finally:
            # Always reschedule the next update
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
        except (ValueError, RuntimeError, HomeAssistantError):
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
    async def _async_save_data(self, force: bool = False) -> None:
        """Save the current data to storage if it has changed.

        Args:
            force: Force save even if data hasn't changed

        """
        if not force and self._last_saved_data is not None:
            # Check if we need to save based on significant changes
            current_data = {
                "probability": round(
                    self.probability, 3
                ),  # Round to avoid float precision issues
                "is_occupied": self.is_occupied,
                "threshold": self.threshold,
            }

            if current_data == self._last_saved_data:
                _LOGGER.debug("Data unchanged, skipping save")
                return

        try:
            _LOGGER.debug("Attempting to save data")
            await self.storage.async_save_instance_data(
                self.entry_id,
                self.entities,
            )

            # Update saved data cache
            self._last_saved_data = {
                "probability": round(self.probability, 3),
                "is_occupied": self.is_occupied,
                "threshold": self.threshold,
            }

            _LOGGER.debug("Data saved successfully")
        except StorageError as err:
            _LOGGER.error("Failed to save data: %s", err)
            raise

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

            # Clear cached probability to force recalculation
            self._cached_probability = None

            # Track failed entity updates
            failed_entities = []

            # Update entity states and calculate probabilities
            # Use coordinated updates to prevent race conditions with state change events
            update_tasks = []
            for entity in self.entities.entities.values():
                try:
                    # Use the entity manager's coordinated update instead of direct update
                    task = self.entities.coordinated_entity_update(entity.entity_id)
                    update_tasks.append(task)
                except (StateError, ValueError) as err:
                    _LOGGER.warning(
                        "Error creating update task for entity %s: %s (will retry next cycle)",
                        entity.entity_id,
                        err,
                    )
                    failed_entities.append(entity.entity_id)
                    continue

            # Wait for all updates to complete
            if update_tasks:
                try:
                    await asyncio.gather(*update_tasks, return_exceptions=True)
                except (asyncio.CancelledError, RuntimeError, ValueError) as err:
                    _LOGGER.warning(
                        "Some entity updates failed during coordinator update: %s", err
                    )

            # Log failed entities if any
            if failed_entities:
                _LOGGER.debug(
                    "Failed to update %d entities: %s",
                    len(failed_entities),
                    failed_entities,
                )

            # Calculate overall probability (this will use the fresh calculation due to cache clear)
            probability = self.complementary_probability
            is_occupied = probability >= self.config.threshold

            # Prepare data dictionary
            data = {
                "probability": probability,
                "is_occupied": is_occupied,
                "threshold": self.config.threshold,
                "last_updated": dt_util.utcnow().isoformat(),
                "failed_entities": failed_entities,  # Include failed entities in data
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

            # Save data to storage (uses smart saving logic)
            await self._async_save_data()

            _LOGGER.debug(
                "Data update completed for area %s: probability=%.2f, is_occupied=%s, failed_entities=%d",
                self.config.name,
                probability,
                is_occupied,
                len(failed_entities),
            )

        except (StateError, ValueError) as err:
            _LOGGER.error("Error updating data: %s", err)
            raise HomeAssistantError(f"Error updating data: {err}") from err

        else:
            return data
