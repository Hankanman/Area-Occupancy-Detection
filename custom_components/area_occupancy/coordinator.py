"""Coordinator for Area Occupancy Detection with optimized update handling."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
import threading

from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.debounce import Debouncer

from .const import (
    DOMAIN,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    STORAGE_VERSION,
    STORAGE_VERSION_MINOR,
    CONF_NAME,
    CONF_AREA_ID,
    CONF_THRESHOLD,
    CONF_DECAY_WINDOW,
    CONF_DECAY_ENABLED,
    CONF_DECAY_MIN_DELAY,
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
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_MIN_DELAY,
)
from .types import (
    ProbabilityResult,
    Config,
    DecayConfig,
)
from .calculations import ProbabilityCalculator
from .storage import AreaOccupancyStorage

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Manage fetching and combining data for area occupancy."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        config: dict[str, Any],
    ) -> None:
        _LOGGER.debug("Initializing AreaOccupancyCoordinator")
        self.storage = AreaOccupancyStorage(hass, entry_id)
        self._last_known_values: dict[str, Any] = {}

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=10),
            update_method=self._async_update_data,
        )

        if not config.get(CONF_MOTION_SENSORS):
            raise HomeAssistantError("No motion sensors configured")

        self.entry_id = entry_id
        self.config = config

        self._state_lock = asyncio.Lock()
        self._sensor_states = {}
        self._motion_timestamps = {}

        self._last_occupied: datetime | None = None
        self._last_state_change: datetime | None = None

        self._calculator = self._create_calculator()

        self._storage_lock = asyncio.Lock()
        self._last_save = dt_util.utcnow()
        self._save_interval = timedelta(seconds=10)

        self._entity_ids: set[str] = set()
        self.learned_priors: dict[str, dict[str, Any]] = {}

        self._last_positive_trigger = None
        self._decay_window = config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)

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

    def _create_calculator(self) -> ProbabilityCalculator:
        _LOGGER.debug("Creating ProbabilityCalculator")
        return ProbabilityCalculator(
            hass=self.hass,
            coordinator=self,
            config=self.config,
        )

    async def async_setup(self) -> None:
        """Minimal setup. No heavy tasks here."""
        _LOGGER.debug("Setting up AreaOccupancyCoordinator")

    async def async_load_stored_data(self) -> None:
        """Load stored data from storage."""
        _LOGGER.debug("Loading stored data")
        try:
            stored_data = await self.storage.async_load()
            if stored_data:
                self._last_known_values = {}

                self._last_occupied = (
                    dt_util.parse_datetime(stored_data.get("last_occupied"))
                    if stored_data.get("last_occupied")
                    else None
                )

                self._last_state_change = (
                    dt_util.parse_datetime(stored_data.get("last_state_change"))
                    if stored_data.get("last_state_change")
                    else None
                )

                self.learned_priors = stored_data.get("learned_priors", {})
            else:
                self._reset_state()
        except (ValueError, TypeError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error loading stored data: %s", err)
            self._reset_state()

    async def async_initialize_states(self) -> None:
        """Initialize sensor states."""
        _LOGGER.debug("Initializing sensor states")
        async with self._state_lock:
            try:
                sensors = self._get_all_configured_sensors()
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

                    if (
                        entity_id in self.config[CONF_MOTION_SENSORS]
                        and is_available
                        and state.state == STATE_ON
                    ):
                        self._motion_timestamps[entity_id] = dt_util.utcnow()

                _LOGGER.info("Initialized states for sensors: %s", sensors)
            except (HomeAssistantError, ValueError, LookupError) as err:
                _LOGGER.error("Error initializing states: %s", err)
                self._sensor_states = {}

            self._setup_entity_tracking()

    def set_last_positive_trigger(self, timestamp: datetime) -> None:
        """Synchronously set the timestamp of the last positive trigger."""
        _LOGGER.debug("Setting last positive trigger synchronously to %s", timestamp)
        with self._thread_lock:
            self._last_positive_trigger = timestamp

    def get_last_positive_trigger(self) -> datetime | None:
        """Synchronously get the timestamp of the last positive trigger."""
        _LOGGER.debug("Getting last positive trigger synchronously")
        with self._thread_lock:
            return self._last_positive_trigger

    def get_decay_window(self) -> int:
        _LOGGER.debug("Getting decay window")
        return self._decay_window

    def get_decay_min_delay(self) -> int:
        _LOGGER.debug("Getting decay min delay")
        return self.config.get(CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY)

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes."""
        entities = self._get_all_configured_sensors()
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

            if (
                entity_id in self.config[CONF_MOTION_SENSORS]
                and is_available
                and new_state.state == STATE_ON
            ):
                self._motion_timestamps[entity_id] = dt_util.utcnow()

            # Schedule the refresh
            _LOGGER.debug("Scheduling refresh due to state change of %s", entity_id)
            self.hass.async_create_task(self._debouncer.async_call())

        # Set up the state change listener
        self._remove_state_listener = async_track_state_change_event(
            self.hass,
            entities,
            async_state_changed_listener,
        )

    async def _async_debounced_refresh(self, _now):
        """Asynchronous debounced refresh."""
        _LOGGER.debug("Debounced refresh called")
        await self.async_refresh()

    async def _async_update_data(self) -> ProbabilityResult:
        """Update data by recalculating current probability."""
        async with self._state_lock:
            try:
                # Offload heavy calculations to the executor
                result = await self.hass.async_add_executor_job(
                    self._calculator.calculate,
                    self._sensor_states,
                    self._motion_timestamps,
                )

                self._update_historical_tracking(result)
                await self._async_store_result(result)

                return result

            except (HomeAssistantError, ValueError, RuntimeError, KeyError) as err:
                _LOGGER.error("Error updating data: %s", err, exc_info=True)
                raise UpdateFailed(f"Error updating data: {err}") from err

    def _update_historical_tracking(self, result: ProbabilityResult) -> None:
        """Update historical tracking data."""
        _LOGGER.debug("Updating historical tracking")
        now = dt_util.utcnow()

        is_occupied = result["probability"] >= self.get_threshold_decimal()

        if is_occupied:
            self._last_occupied = now

    async def _async_store_result(self, result: ProbabilityResult) -> None:
        _LOGGER.debug("Storing result")
        try:
            stored_data = await self.storage.async_load() or {}
            area_id = self.config[CONF_AREA_ID]

            # Prepare the new data
            new_area_data = {
                "last_updated": dt_util.utcnow().isoformat(),
                "last_probability": result["probability"],
                "configuration": {
                    "motion_sensors": self.config[CONF_MOTION_SENSORS],
                    "media_devices": self.config.get(CONF_MEDIA_DEVICES, []),
                    "appliances": self.config.get(CONF_APPLIANCES, []),
                    "illuminance_sensors": self.config.get(
                        CONF_ILLUMINANCE_SENSORS, []
                    ),
                    "humidity_sensors": self.config.get(CONF_HUMIDITY_SENSORS, []),
                    "temperature_sensors": self.config.get(
                        CONF_TEMPERATURE_SENSORS, []
                    ),
                    "door_sensors": self.config.get(CONF_DOOR_SENSORS, []),
                    "window_sensors": self.config.get(CONF_WINDOW_SENSORS, []),
                    "lights": self.config.get(CONF_LIGHTS, []),
                },
                "last_known_values": self._last_known_values,
                "last_occupied": (
                    self._last_occupied.isoformat() if self._last_occupied else None
                ),
                "last_state_change": (
                    self._last_state_change.isoformat()
                    if self._last_state_change
                    else None
                ),
                "learned_priors": self.learned_priors,
            }

            # Compare with existing data excluding 'last_updated'
            existing_area_data = stored_data.get("areas", {}).get(area_id, {}).copy()
            existing_area_data.pop("last_updated", None)

            new_area_data_to_compare = new_area_data.copy()
            new_area_data_to_compare.pop("last_updated", None)

            if existing_area_data == new_area_data_to_compare:
                _LOGGER.debug("No significant changes detected; skipping save.")
                return  # Data hasn't changed; no need to save

            # Update the stored data with the new result
            stored_data.setdefault("areas", {})
            stored_data["areas"][area_id] = new_area_data

            # Log the stored data after the update
            _LOGGER.debug("Stored data after update: %s", stored_data)

            # Save stored_data to an instance variable for access in debounced save
            self._stored_data = stored_data

            # Instead of saving immediately, schedule a debounced save
            await self._save_debouncer.async_call()

        except (HomeAssistantError, ValueError, KeyError, IOError) as err:
            _LOGGER.error("Error storing result: %s", err)

    def _get_all_configured_sensors(self) -> list[str]:
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

    async def async_reset(self) -> None:
        """Reset the coordinator."""
        _LOGGER.debug("Resetting coordinator")
        self._sensor_states.clear()
        self._motion_timestamps.clear()
        self._last_occupied = None
        self._last_state_change = None

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

    def update_options(self, config: Config) -> None:
        _LOGGER.debug("Updating options")
        try:
            self.config = config
            self._calculator = self._create_calculator()
            self._decay_window = self.config.get(
                CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
            )

            # Re-setup entity tracking with new sensors
            self._setup_entity_tracking()

            # Re-initialize states and refresh data
            self.hass.async_create_task(self._async_reinitialize_states())

            _LOGGER.debug(
                "Updated coordinator options for %s",
                self.config[CONF_NAME],
            )
        except (ValueError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error updating coordinator options: %s", err)
            raise HomeAssistantError(
                f"Failed to update coordinator options: {err}"
            ) from err

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
    def device_info(self) -> dict[str, Any]:
        return {
            "identifiers": {(DOMAIN, self.entry_id)},
            "name": self.config[CONF_NAME],
            "manufacturer": DEVICE_MANUFACTURER,
            "model": DEVICE_MODEL,
            "sw_version": DEVICE_SW_VERSION,
        }

    def _reset_state(self) -> None:
        _LOGGER.debug("Resetting state")
        self._last_known_values = {}
        self._last_occupied = None
        self._last_state_change = None
        self._last_save = dt_util.utcnow()
        self.learned_priors.clear()

    def get_storage_data(self) -> dict[str, Any]:
        """Prepare data to be saved in storage."""
        _LOGGER.debug("Preparing storage data")

        # Construct the data to be saved
        storage_data = {
            "last_occupied": (
                self._last_occupied.isoformat() if self._last_occupied else None
            ),
            "last_state_change": (
                self._last_state_change.isoformat() if self._last_state_change else None
            ),
            "learned_priors": self.learned_priors,
            "version": STORAGE_VERSION,
            "version_minor": STORAGE_VERSION_MINOR,
            "last_updated": dt_util.utcnow().isoformat(),
        }

        return storage_data

    @property
    def entity_ids(self) -> set[str]:
        return self._entity_ids

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
        _LOGGER.debug("Updating threshold")
        if not 0 <= value <= 100:
            raise ValueError("Threshold must be between 0 and 100")
        old_threshold = self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        self.config[CONF_THRESHOLD] = value
        _LOGGER.debug(
            "Updated threshold from %.2f to %.2f (decimal: %.3f)",
            old_threshold,
            value,
            self.get_threshold_decimal(),
        )
        await self.async_refresh()

    def get_threshold_decimal(self) -> float:
        threshold = self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        return threshold / 100.0

    def get_configured_sensors(self) -> list[str]:
        return self._get_all_configured_sensors()

    def _get_decay_config(self) -> DecayConfig:
        _LOGGER.debug("Getting decay config")
        return DecayConfig(
            enabled=self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
            window=self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
            min_delay=self.config.get(CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY),
        )

    def update_learned_priors(
        self, entity_id: str, p_true: float, p_false: float, prior: float
    ) -> None:
        _LOGGER.debug("Updating learned priors")
        self.learned_priors[entity_id] = {
            "prob_given_true": p_true,
            "prob_given_false": p_false,
            "prior": prior,
            "last_updated": dt_util.utcnow().isoformat(),
        }
        self.hass.async_create_task(self.async_save_state())

    async def async_save_state(self) -> None:
        _LOGGER.debug("Saving state")
        now = dt_util.utcnow()
        if now - self._last_save < self._save_interval:
            return

        async with self._storage_lock:
            try:
                storage_data = self.get_storage_data()
                await self.storage.async_save(storage_data)
                self._last_save = now
            except (IOError, ValueError, HomeAssistantError) as err:
                _LOGGER.error("Failed to save state: %s", err)

    async def async_restore_state(self, stored_data: dict) -> None:
        _LOGGER.debug("Restoring state")
        try:
            if not isinstance(stored_data, dict):
                raise ValueError("Invalid storage data format")

            self._last_known_values = stored_data.get("last_known_values", {})

            last_occupied = stored_data.get("last_occupied")
            self._last_occupied = (
                dt_util.parse_datetime(last_occupied) if last_occupied else None
            )

            last_state_change = stored_data.get("last_state_change")
            self._last_state_change = (
                dt_util.parse_datetime(last_state_change) if last_state_change else None
            )

            self.learned_priors = stored_data.get("learned_priors", {})

        except (ValueError, TypeError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error restoring state: %s", err)
            self._reset_state()

    async def _save_debounced_data(self):
        """Debounced method to save data to storage."""
        if hasattr(self, "_stored_data"):
            try:
                await self.storage.async_save(self._stored_data)
                _LOGGER.debug("Debounced save completed")
                del self._stored_data  # Clean up after saving
            except (IOError, ValueError, HomeAssistantError) as err:
                _LOGGER.error("Error during debounced save: %s", err)
