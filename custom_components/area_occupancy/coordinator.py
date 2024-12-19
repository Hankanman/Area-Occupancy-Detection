"""Coordinator for Area Occupancy Detection with optimized update handling."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
import threading

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.debounce import Debouncer
from homeassistant.config_entries import ConfigEntry

from .const import (
    DOMAIN,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    CONF_NAME,
    CONF_THRESHOLD,
    CONF_APPLIANCES,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_DOOR_SENSORS,
    CONF_WINDOW_SENSORS,
    CONF_LIGHTS,
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    DEFAULT_THRESHOLD,
    CONF_HISTORY_PERIOD,
    DEFAULT_HISTORY_PERIOD,
)
from .types import (
    ProbabilityResult,
    LearnedPrior,
    DeviceInfo,
)
from .storage import AreaOccupancyStorage
from .calculate_prob import ProbabilityCalculator
from .calculate_prior import PriorCalculator
from .probabilities import Probabilities

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Manage fetching and combining data for area occupancy."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the coordinator."""
        self.config_entry = config_entry
        self.config = {**config_entry.data, **config_entry.options}

        # Initialize storage and learned_priors first
        self.storage = AreaOccupancyStorage(hass, config_entry.entry_id)
        self.learned_priors: dict[str, LearnedPrior] = {}

        self.threshold = self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD) / 100.0

        # Initialize probabilities before calculator
        self._probabilities = Probabilities(config=self.config)

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=10),
            update_method=self._async_update_data,
        )

        # Initialize calculator after super().__init__
        self._calculator = ProbabilityCalculator(
            coordinator=self,
            probabilities=self._probabilities,
        )

        self.entry_id = config_entry.entry_id

        self._state_lock = asyncio.Lock()
        self._sensor_states = {}

        self._prior_update_interval = timedelta(hours=1)

        self._last_occupied: datetime | None = None

        self._prior_calculator = PriorCalculator(
            coordinator=self,
            probabilities=self._probabilities,
            hass=self.hass,
        )

        self._storage_lock = asyncio.Lock()
        self._last_save = dt_util.utcnow()
        self._save_interval = timedelta(seconds=10)

        self._entity_ids: set[str] = set()
        self._last_positive_trigger = None

        self._remove_state_listener = None

        # Initialize the debouncer
        self._debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=0.1,
            immediate=True,
            function=self.async_refresh,
        )

        # Initialize the save debouncer
        self._save_debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=30.0,  # Wait 30 seconds before saving
            immediate=False,
            function=self._save_debounced_data,
        )

        # Initialize stored data variable
        self._stored_data = None

        self._thread_lock = threading.Lock()

    @property
    def entity_ids(self) -> set[str]:
        return self._entity_ids

    @property
    def calculator(self) -> ProbabilityCalculator:
        """Get the probability calculator."""
        return self._calculator

    @property
    def available(self) -> bool:
        motion_sensors = self.config.get(CONF_MOTION_SENSORS, [])
        if not motion_sensors:
            return False
        return any(
            self._sensor_states.get(sensor_id, {}).get("availability", False)
            for sensor_id in motion_sensors
        )

    @property
    def device_info(self) -> DeviceInfo:
        return {
            "identifiers": {(DOMAIN, self.entry_id)},
            "name": self.config[CONF_NAME],
            "manufacturer": DEVICE_MANUFACTURER,
            "model": DEVICE_MODEL,
            "sw_version": DEVICE_SW_VERSION,
        }

    async def async_setup(self) -> None:
        """Setup the coordinator and load stored data."""
        _LOGGER.debug("Setting up AreaOccupancyCoordinator")

        # Load stored data first
        await self.async_load_stored_data()

        # Initialize states after loading stored data
        await self.async_initialize_states()

        # Schedule prior updates
        self.hass.loop.create_task(self._schedule_prior_updates())

    async def async_load_stored_data(self) -> None:
        """Load and restore data from storage."""
        try:
            stored_data = await self.storage.async_load()
            if not stored_data:
                _LOGGER.debug("No stored data found, initializing fresh state")
                self._reset_state()
                return

            # Restore learned priors
            self.learned_priors = stored_data.get("learned_priors", {})

            # Restore last occupied timestamp
            last_occupied = stored_data.get("last_occupied")
            if last_occupied:
                try:
                    self._last_occupied = dt_util.parse_datetime(last_occupied)
                except (ValueError, TypeError) as err:
                    _LOGGER.warning(
                        "Failed to parse last_occupied datetime: %s", err, exc_info=True
                    )
                    self._last_occupied = None

            _LOGGER.debug("Successfully restored stored data")

        except (ValueError, TypeError, KeyError) as err:
            _LOGGER.error("Error loading stored data: %s", err, exc_info=True)
            self._reset_state()
        except HomeAssistantError as err:
            _LOGGER.error(
                "Home Assistant error loading stored data: %s", err, exc_info=True
            )
            self._reset_state()

    async def async_initialize_states(self) -> None:
        """Initialize sensor states."""
        _LOGGER.debug("Initializing sensor states")
        async with self._state_lock:
            try:
                sensors = self.get_configured_sensors()
                for entity_id in sensors:
                    state = self.hass.states.get(entity_id)
                    is_available = bool(
                        state
                        and state.state
                        not in [STATE_UNAVAILABLE, STATE_UNKNOWN, None, ""]
                    )
                    self._sensor_states[entity_id] = {
                        "state": state.state if is_available else None,
                        "last_changed": (
                            state.last_changed.isoformat()
                            if state and state.last_changed
                            else dt_util.utcnow().isoformat()
                        ),
                        "availability": is_available,
                    }

                _LOGGER.info("Initialized states for sensors: %s", sensors)
            except (HomeAssistantError, ValueError, LookupError) as err:
                _LOGGER.error("Error initializing states: %s", err)
                self._sensor_states = {}

            self._setup_entity_tracking()

    def get_configured_sensors(self) -> list[str]:
        return (
            self.config.get(CONF_MOTION_SENSORS, [])
            + self.config.get(CONF_MEDIA_DEVICES, [])
            + self.config.get(CONF_APPLIANCES, [])
            + self.config.get(CONF_ILLUMINANCE_SENSORS, [])
            + self.config.get(CONF_HUMIDITY_SENSORS, [])
            + self.config.get(CONF_TEMPERATURE_SENSORS, [])
            + self.config.get(CONF_DOOR_SENSORS, [])
            + self.config.get(CONF_WINDOW_SENSORS, [])
            + self.config.get(CONF_LIGHTS, [])
        )

    async def _schedule_prior_updates(self):
        """Schedule periodic updates of learned priors."""
        while True:
            try:
                _LOGGER.debug("Updating learned priors for all sensors")
                await self._update_learned_priors()
            except (HomeAssistantError, ValueError, RuntimeError) as err:
                _LOGGER.error("Error updating learned priors: %s", err)
            # Wait for the specified interval before the next update
            await asyncio.sleep(self._prior_update_interval.total_seconds())

    async def _update_learned_priors(self):
        """Update learned priors by calculating them for all configured sensors."""
        start_time = dt_util.utcnow() - timedelta(
            days=self.config.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD)
        )
        end_time = dt_util.utcnow()
        for entity_id in self.get_configured_sensors():
            await self._prior_calculator.calculate_prior(
                entity_id, start_time, end_time
            )

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes."""
        entities = self.get_configured_sensors()
        _LOGGER.debug("Setting up entity tracking for: %s", entities)

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        @callback
        def async_state_changed_listener(event) -> None:
            """Handle state changes."""
            entity_id = event.data.get("entity_id")
            new_state = event.data.get("new_state")

            if not new_state or entity_id not in entities:
                return

            # Update sensor states
            is_available = new_state.state not in [
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
                None,
                "",
            ]
            self._sensor_states[entity_id] = {
                "state": new_state.state if is_available else None,
                "last_changed": (
                    new_state.last_changed.isoformat()
                    if new_state.last_changed
                    else dt_util.utcnow().isoformat()
                ),
                "availability": is_available,
            }

            # Schedule the refresh
            _LOGGER.debug("Scheduling refresh due to state change of %s", entity_id)
            self.hass.async_create_task(self._debouncer.async_call())

        # Set up the state change listener
        self._remove_state_listener = async_track_state_change_event(
            self.hass,
            entities,
            async_state_changed_listener,
        )

    async def _async_update_data(self) -> ProbabilityResult:
        """Update data by recalculating current probability."""
        async with self._state_lock:
            try:
                # Offload heavy calculations to the executor
                result = await self.hass.async_add_executor_job(
                    self._calculator.calculate,
                    self._sensor_states,
                )
                now = dt_util.utcnow()

                is_occupied = result["probability"] >= self.threshold

                if is_occupied:
                    self._last_occupied = now
                await self._async_store_data()

                return result

            except (HomeAssistantError, ValueError, RuntimeError, KeyError) as err:
                _LOGGER.error("Error updating data: %s", err, exc_info=True)
                raise UpdateFailed(f"Error updating data: {err}") from err

    async def _async_store_data(self) -> None:
        """Store the latest result and schedule a debounced save."""
        try:
            # Prepare the new data
            new_data = {
                "name": self.config[CONF_NAME],
                "learned_priors": self.learned_priors,
            }

            # Only save if data has meaningfully changed
            if self._stored_data:
                old_data = self._stored_data.copy()
                new_data_compare = new_data.copy()

                # Remove timestamps before comparison
                old_data.pop("last_updated", None)
                new_data_compare.pop("last_updated", None)

                if old_data == new_data_compare:
                    _LOGGER.debug("No significant changes detected; skipping save")
                    return

            self._stored_data = new_data
            await self._save_debouncer.async_call()

        except (ValueError, TypeError, KeyError) as err:
            _LOGGER.error("Error preparing data for storage: %s", err, exc_info=True)
        except HomeAssistantError as err:
            _LOGGER.error("Error preparing data storage: %s", err, exc_info=True)

    async def async_reset(self) -> None:
        """Reset the coordinator."""
        _LOGGER.debug("Resetting coordinator")
        self._sensor_states.clear()
        self._last_occupied = None

        self._last_save = dt_util.utcnow()
        self.learned_priors.clear()

        await self.async_refresh()

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        if self._debouncer is not None:
            await self._debouncer.async_shutdown()
            self._debouncer = None

    async def async_unload(self) -> None:
        """Unload the coordinator."""
        _LOGGER.debug("Unloading coordinator")
        await self._save_debounced_data()

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        if self._debouncer is not None:
            await self._debouncer.async_shutdown()
            self._debouncer = None

    async def async_update_options(self) -> None:
        """Update options asynchronously."""
        _LOGGER.debug("Updating options")
        try:
            self.config = {
                **self.config_entry.data,
                **self.config_entry.options,
            }
            self._probabilities = Probabilities(config=self.config)
            self._calculator = ProbabilityCalculator(
                coordinator=self,
                probabilities=self._probabilities,
            )
            self._prior_calculator = PriorCalculator(
                coordinator=self,
                probabilities=self._probabilities,
                hass=self.hass,
            )

            # Re-setup entity tracking with new sensors
            self._setup_entity_tracking()
            await self._async_reinitialize_states()

        except (ValueError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error updating coordinator options: %s", err)
            raise HomeAssistantError(
                f"Failed to update coordinator options: {err}"
            ) from err

    def _reset_state(self) -> None:
        _LOGGER.debug("Resetting state")
        self._last_occupied = None
        self._last_save = dt_util.utcnow()
        self.learned_priors.clear()

    def register_entity(self, entity_id: str) -> None:
        _LOGGER.debug("Registering entity: %s", entity_id)
        self._entity_ids.add(entity_id)

    def unregister_entity(self, entity_id: str) -> None:
        _LOGGER.debug("Unregistering entity: %s", entity_id)
        self._entity_ids.discard(entity_id)

    async def async_refresh(self) -> None:
        """Refresh data."""
        await super().async_refresh()

    async def _async_reinitialize_states(self) -> None:
        """Re-initialize states after options have been updated."""
        await self.async_initialize_states()
        await self.async_refresh()

    async def async_update_threshold(self, value: float) -> None:
        """Update threshold value and persist it to config entry."""
        _LOGGER.debug("Updating threshold to %.2f", value)

        # Update runtime config
        self.config[CONF_THRESHOLD] = value

        # Update config entry options
        new_options = dict(self.config_entry.options)
        new_options[CONF_THRESHOLD] = value

        try:
            # Update the config entry with new options
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                options=new_options,
            )
            self.threshold = value / 100.0

            # Trigger an update
            await self.async_refresh()

        except ValueError as err:
            _LOGGER.error("Error updating threshold: %s", err)
            # Revert runtime config if update failed
            raise HomeAssistantError(f"Failed to update threshold: {err}") from err

    def update_learned_priors(
        self, entity_id: str, p_true: float, p_false: float, prior: float
    ) -> None:
        """Update learned priors."""
        _LOGGER.debug("Updating learned priors")
        self.learned_priors[entity_id] = {
            "prob_given_true": p_true,
            "prob_given_false": p_false,
            "prior": prior,
            "last_updated": dt_util.utcnow().isoformat(),
        }
        self.hass.async_create_task(self.async_save_state())

        # Force an update of all entities
        self.async_set_updated_data(self.data)

    async def async_save_state(self) -> None:
        _LOGGER.debug("Saving state")
        now = dt_util.utcnow()
        if now - self._last_save < self._save_interval:
            return

        async with self._storage_lock:
            try:
                storage_data = {
                    "name": self.config[CONF_NAME],
                    "learned_priors": self.learned_priors,
                }
                await self.storage.async_save(storage_data)
                self._last_save = now
            except (IOError, ValueError, HomeAssistantError) as err:
                _LOGGER.error("Failed to save state: %s", err)

    async def async_restore_state(self, stored_data: dict) -> None:
        _LOGGER.debug("Restoring state")
        try:
            if not isinstance(stored_data, dict):
                raise ValueError("Invalid storage data format")

            last_occupied = stored_data.get("last_occupied")
            self._last_occupied = (
                dt_util.parse_datetime(last_occupied) if last_occupied else None
            )

            self.learned_priors = stored_data.get("learned_priors", {})

        except (ValueError, TypeError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error restoring state: %s", err)
            self._reset_state()

    async def _save_debounced_data(self) -> None:
        """Save stored data with debouncing."""
        if not hasattr(self, "_stored_data") or self._stored_data is None:
            return

        try:
            async with self._storage_lock:
                await self.storage.async_save(self._stored_data)
                _LOGGER.debug("Successfully saved data to storage")
        except (ValueError, TypeError, KeyError) as err:
            _LOGGER.error("Error saving data to storage: %s", err, exc_info=True)
        except (IOError, HomeAssistantError) as err:
            _LOGGER.error("Storage I/O error: %s", err, exc_info=True)
        finally:
            # Clear stored data after save attempt
            self._stored_data = None

    async def calculate_sensor_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ):
        """Public method to calculate prior for a sensor."""
        return await self._prior_calculator.calculate_prior(
            entity_id, start_time, end_time
        )
