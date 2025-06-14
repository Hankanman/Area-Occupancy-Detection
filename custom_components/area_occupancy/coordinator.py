"""Area Occupancy Coordinator."""

from __future__ import annotations

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
from .const import CONF_THRESHOLD, DEFAULT_NAME, DEFAULT_PRIOR, DOMAIN, MIN_PROBABILITY
from .data.config import ConfigManager
from .data.entity import EntityManager
from .data.entity_type import EntityTypeManager
from .data.prior import PriorManager
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
            update_interval=timedelta(minutes=5),
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
        self._next_prior_update: datetime | None = None
        self._last_prior_update: datetime | None = None
        self._prior_update_tracker: CALLBACK_TYPE | None = None

        # Track specific binary sensor entity_ids
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None

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
    def probability(self) -> float:
        """Calculate and return the current occupancy probability (0.0-1.0)."""
        return self._calculate_entity_aggregates()["probability"]

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

    def request_update(self, force: bool = False, message: str = "") -> None:
        """Request an immediate coordinator refresh.

        Args:
            force: If True, bypasses debouncing and forces immediate update
            message: Optional message to log when updating

        """
        if message:
            _LOGGER.debug(message)

        if force:
            # Bypass debouncing by directly calling the update methods
            self.hass.async_create_task(self._async_immediate_update())
        else:
            self.hass.async_create_task(self.async_request_refresh())

    async def _async_immediate_update(self) -> None:
        """Force immediate coordinator update bypassing any debouncing."""
        try:
            _LOGGER.debug("Executing immediate coordinator update")
            data = await self._async_update_data()
            self.async_set_updated_data(data)
        except (HomeAssistantError, ValueError, RuntimeError, AttributeError) as err:
            _LOGGER.warning("Failed to force immediate coordinator update: %s", err)
            # Fallback to regular update if direct method fails
            self.hass.async_create_task(self.async_request_refresh())

    # --- Public Methods ---
    async def _async_setup(self) -> None:
        """Initialize the coordinator and its components."""
        try:
            _LOGGER.debug("Starting coordinator setup for %s", self.config.name)

            # Initialize components in order
            await self.storage.async_initialize()
            await self.entity_types.async_initialize()
            await self.async_load_stored_data()
            await self.entities.async_initialize()

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

        # Clean up entity manager
        self.entities.cleanup()

        # Shutdown storage manager to save any pending data
        await self.storage.async_shutdown()

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

        # Schedule next update and refresh data
        await self._schedule_next_prior_update()

        # Force immediate save after configuration changes
        await self._async_save_data(force=True)

        self.request_update(message="Options updated, requesting update")

    async def async_update_threshold(self, value: float) -> None:
        """Update the threshold value.

        Args:
            value: The new threshold value as a percentage (1-99)

        Raises:
            ServiceValidationError: If the value is invalid
            HomeAssistantError: If there's an error updating the config entry

        """
        _LOGGER.debug("Updating threshold: %.2f%% (%.3f)", value, value / 100.0)

        await self.config_manager.update_config(
            {
                CONF_THRESHOLD: value / 100.0,
            }
        )

        # Request update since threshold affects is_occupied calculation
        self.request_update(message="Threshold updated, requesting update")

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

    async def update_learned_priors(self) -> None:
        """Update learned priors using historical data."""
        try:
            _LOGGER.info("Starting learned priors update for area %s", self.config.name)

            # Delegate the actual prior updating to the PriorManager
            updated_count = await self.priors.update_all_entity_priors()

            # Request update and save immediately if priors changed
            if updated_count > 0:
                await self._async_save_data(force=True)
                self.request_update(message="Learned priors updated, requesting update")

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

            # Save data periodically
            await self._async_save_data()

            _LOGGER.debug(
                "Data update completed for area %s: probability=%.2f, is_occupied=%s",
                self.config.name,
                self.probability,
                self.is_occupied,
            )

            # Return minimal data - sensors access coordinator properties directly
            return {"last_updated": dt_util.utcnow().isoformat()}

        except (HomeAssistantError, ValueError) as err:
            _LOGGER.error("Error updating data: %s", err)
            raise HomeAssistantError(f"Error updating data: {err}") from err

    # --- Data Saving ---
    async def _async_save_data(self, force: bool = False) -> None:
        """Save the current data to storage with optimized frequency.

        Args:
            force: If True, bypass debouncing and save immediately

        """
        try:
            _LOGGER.debug("Requesting storage save for coordinator data")
            await self.storage.async_save_instance_data(
                self.entry_id,
                self.entities,
                force=force,
            )
            _LOGGER.debug("Storage save request completed")
        except HomeAssistantError as err:
            _LOGGER.error("Failed to save data: %s", err)
            # Don't re-raise in normal operations to avoid breaking the update cycle
            # Only re-raise on forced saves
            if force:
                raise

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
