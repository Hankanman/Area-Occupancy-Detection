"""Coordinator for Area Occupancy Detection with optimized update handling."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
import threading

from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback, State
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
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
    CONF_HISTORY_PERIOD,
    CONF_THRESHOLD,
    CONF_DECAY_WINDOW,
    CONF_APPLIANCES,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_DOOR_SENSORS,
    CONF_WINDOW_SENSORS,
    CONF_LIGHTS,
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_DECAY_ENABLED,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_THRESHOLD,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_HISTORY_PERIOD,
)
from .types import (
    ProbabilityResult,
    CoreConfig,
    OptionsConfig,
    DecayConfig,
)
from .calculations import ProbabilityCalculator
from .storage import AreaOccupancyStorage

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Manage fetching and combining data for area occupancy.

    Responsibilities:
    - Load and store learned priors for sensors.
    - Handle initial setup, including loading historical data.
    - Provide a ProbabilityCalculator to perform Bayesian updates.
    - Maintain occupancy history and timeslot data.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        core_config: CoreConfig,
        options_config: OptionsConfig,
    ) -> None:
        _LOGGER.debug("Initializing AreaOccupancyCoordinator")
        self.storage = AreaOccupancyStorage(hass, entry_id)
        self._last_known_values: dict[str, Any] = {}
        self._debounce_refresh = None

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=10),
            update_method=self._async_update_data,
        )

        if not core_config.get(CONF_MOTION_SENSORS):
            raise HomeAssistantError("No motion sensors configured")

        self.entry_id = entry_id
        self.core_config = core_config
        self.options_config = options_config

        self._state_lock = asyncio.Lock()
        self._sensor_states = {}
        self._motion_timestamps = {}

        self._last_occupied: datetime | None = None
        self._last_state_change: datetime | None = None

        self._historical_analysis_ready = asyncio.Event()
        self._calculator = self._create_calculator()

        self._storage_lock = asyncio.Lock()
        self._last_save = dt_util.utcnow()
        self._save_interval = timedelta(seconds=10)

        self._entity_ids: set[str] = set()
        self.learned_priors: dict[str, dict[str, Any]] = {}

        self._last_positive_trigger = None
        self._decay_window = options_config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)

        self._remove_periodic_task = None
        self._historical_analysis_lock = asyncio.Lock()
        self._remove_state_listener = None

        # Initialize the debouncer
        self._debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=0.1,
            immediate=True,
            function=self.async_refresh,
        )

        # Add a flag to track initialization status
        self._initialization_complete = False
        self._startup_lock = asyncio.Lock()

        self._thread_lock = threading.Lock()

        self._registered_entities: set[str] = set()

    def _create_calculator(self) -> ProbabilityCalculator:
        _LOGGER.debug("Creating ProbabilityCalculator")
        return ProbabilityCalculator(
            hass=self.hass,
            coordinator=self,
            motion_sensors=self.core_config[CONF_MOTION_SENSORS],
            media_devices=self.options_config.get(CONF_MEDIA_DEVICES, []),
            appliances=self.options_config.get(CONF_APPLIANCES, []),
            illuminance_sensors=self.options_config.get(CONF_ILLUMINANCE_SENSORS, []),
            humidity_sensors=self.options_config.get(CONF_HUMIDITY_SENSORS, []),
            temperature_sensors=self.options_config.get(CONF_TEMPERATURE_SENSORS, []),
            door_sensors=self.options_config.get(CONF_DOOR_SENSORS, []),
            window_sensors=self.options_config.get(CONF_WINDOW_SENSORS, []),
            lights=self.options_config.get(CONF_LIGHTS, []),
            decay_config=self._get_decay_config(),
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
                        entity_id in self.core_config[CONF_MOTION_SENSORS]
                        and is_available
                        and state.state == STATE_ON
                    ):
                        self._motion_timestamps[entity_id] = dt_util.utcnow()

                _LOGGER.info("Initialized states for sensors: %s", sensors)
            except (HomeAssistantError, ValueError, LookupError) as err:
                _LOGGER.error("Error initializing states: %s", err)
                self._sensor_states = {}

            self._setup_entity_tracking()

    def schedule_periodic_historical_analysis(self):
        """Schedule the historical analysis to run every 6 hours."""
        _LOGGER.debug("Scheduling periodic historical analysis")

        async def periodic_task(now):
            await self.async_run_historical_analysis_task()

        # Schedule the task to run every 6 hours
        self._remove_periodic_task = async_track_time_interval(
            self.hass, periodic_task, timedelta(hours=6)
        )

    async def async_run_historical_analysis_task(self) -> None:
        """Run historical analysis and initial priors."""
        _LOGGER.debug("Running historical analysis task")
        async with self._historical_analysis_lock:
            try:
                await self.async_refresh()
                await self._async_historical_analysis()
                await self.async_refresh()
            except (ValueError, HomeAssistantError) as err:
                _LOGGER.error("Error running historical analysis: %s", err)

    async def set_last_positive_trigger(self, timestamp: datetime) -> None:
        """Asynchronously set the timestamp of the last positive trigger."""
        _LOGGER.debug("Setting last positive trigger asynchronously to %s", timestamp)
        async with self._state_lock:
            self._last_positive_trigger = timestamp

    def set_last_positive_trigger_sync(self, timestamp: datetime) -> None:
        """Synchronously set the timestamp of the last positive trigger."""
        _LOGGER.debug("Setting last positive trigger synchronously to %s", timestamp)
        with self._thread_lock:
            self._last_positive_trigger = timestamp

    async def get_last_positive_trigger(self) -> datetime | None:
        """Asynchronously get the timestamp of the last positive trigger."""
        _LOGGER.debug("Getting last positive trigger asynchronously")
        async with self._state_lock:
            return self._last_positive_trigger

    def get_last_positive_trigger_sync(self) -> datetime | None:
        """Synchronously get the timestamp of the last positive trigger."""
        _LOGGER.debug("Getting last positive trigger synchronously")
        with self._thread_lock:
            return self._last_positive_trigger

    def get_decay_window(self) -> int:
        _LOGGER.debug("Getting decay window")
        return self._decay_window

    def get_decay_min_delay(self) -> int:
        _LOGGER.debug("Getting decay min delay")
        return self.options_config.get("decay_min_delay", 60)

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
                entity_id in self.core_config[CONF_MOTION_SENSORS]
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
                    self._calculate_probability
                )

                self._update_historical_tracking(result)
                await self._async_store_result(result)

                return result

            except (HomeAssistantError, ValueError, RuntimeError, KeyError) as err:
                _LOGGER.error("Error updating data: %s", err, exc_info=True)
                raise UpdateFailed(f"Error updating data: {err}") from err

    def _calculate_probability(self) -> ProbabilityResult:
        """Synchronous method to perform heavy calculations."""
        try:
            # Perform the heavy computation synchronously
            result = self._calculator.calculate_sync(
                self._sensor_states,
                self._motion_timestamps,
            )
            return result

        except (HomeAssistantError, ValueError, RuntimeError, KeyError) as err:
            _LOGGER.error("Error in synchronous calculation: %s", err)
            raise UpdateFailed(f"Error in synchronous calculation: {err}") from err

    def _update_historical_tracking(self, result: ProbabilityResult) -> None:
        """Update historical tracking data."""
        _LOGGER.debug("Updating historical tracking")
        now = dt_util.utcnow()

        is_occupied = result["probability"] >= self.options_config["threshold"]

        if is_occupied:
            self._last_occupied = now

    async def _async_store_result(self, result: ProbabilityResult) -> None:
        _LOGGER.debug("Storing result")
        try:
            stored_data = await self.storage.async_load() or {}
            area_id = self.core_config[CONF_AREA_ID]

            stored_data.setdefault("areas", {})
            stored_data["areas"].setdefault(area_id, {})

            stored_data["areas"][area_id].update(
                {
                    "last_updated": dt_util.utcnow().isoformat(),
                    "last_probability": result["probability"],
                    "configuration": {
                        "motion_sensors": self.core_config[CONF_MOTION_SENSORS],
                        "media_devices": self.options_config.get(
                            CONF_MEDIA_DEVICES, []
                        ),
                        "appliances": self.options_config.get(CONF_APPLIANCES, []),
                        "illuminance_sensors": self.options_config.get(
                            CONF_ILLUMINANCE_SENSORS, []
                        ),
                        "humidity_sensors": self.options_config.get(
                            CONF_HUMIDITY_SENSORS, []
                        ),
                        "temperature_sensors": self.options_config.get(
                            CONF_TEMPERATURE_SENSORS, []
                        ),
                        "door_sensors": self.options_config.get(CONF_DOOR_SENSORS, []),
                        "window_sensors": self.options_config.get(
                            CONF_WINDOW_SENSORS, []
                        ),
                        "lights": self.options_config.get(CONF_LIGHTS, []),
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
            )

            await self.storage.async_save(stored_data)

        except (HomeAssistantError, ValueError, KeyError, IOError) as err:
            _LOGGER.error("Error storing result: %s", err)

    def _get_all_configured_sensors(self) -> list[str]:
        _LOGGER.debug("Getting all configured sensors")
        return (
            self.core_config.get(CONF_MOTION_SENSORS, [])
            + self.options_config.get(CONF_MEDIA_DEVICES, [])
            + self.options_config.get(CONF_APPLIANCES, [])
            + self.options_config.get(CONF_ILLUMINANCE_SENSORS, [])
            + self.options_config.get(CONF_HUMIDITY_SENSORS, [])
            + self.options_config.get(CONF_TEMPERATURE_SENSORS, [])
            + self.options_config.get(CONF_DOOR_SENSORS, [])
            + self.options_config.get(CONF_WINDOW_SENSORS, [])
            + self.options_config.get(CONF_LIGHTS, [])
        )

    async def async_reset(self) -> None:
        """Reset the coordinator."""
        _LOGGER.debug("Resetting coordinator")
        self._sensor_states.clear()
        self._motion_timestamps.clear()
        self._last_occupied = None
        self._last_state_change = None

        self._historical_analysis_ready.clear()
        self._last_save = dt_util.utcnow()
        self.learned_priors.clear()

        await self.async_refresh()

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        if self._debouncer is not None:
            await self._debouncer.async_shutdown()
            self._debouncer = None

        if self._remove_periodic_task:
            self._remove_periodic_task()
            self._remove_periodic_task = None

    async def async_unload(self) -> None:
        """Unload the coordinator."""
        _LOGGER.debug("Unloading coordinator")
        if self.data:
            await self._save_data(self.data)

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        if self._debouncer is not None:
            await self._debouncer.async_shutdown()
            self._debouncer = None

        if self._remove_periodic_task:
            self._remove_periodic_task()
            self._remove_periodic_task = None

    def update_options(self, options_config: OptionsConfig) -> None:
        _LOGGER.debug("Updating options")
        try:
            self.options_config = options_config
            self._calculator = self._create_calculator()
            self._decay_window = self.options_config.get(
                CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
            )

            # Re-setup entity tracking with new sensors
            self._setup_entity_tracking()

            # Re-initialize states and refresh data
            self.hass.async_create_task(self._async_reinitialize_states())

            _LOGGER.debug(
                "Updated coordinator options for %s",
                self.core_config[CONF_NAME],
            )
        except (ValueError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error updating coordinator options: %s", err)
            raise HomeAssistantError(
                f"Failed to update coordinator options: {err}"
            ) from err

    @property
    def available(self) -> bool:
        motion_sensors = self.core_config.get(CONF_MOTION_SENSORS, [])
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
            "name": self.core_config[CONF_NAME],
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
        """Refresh data, deferring heavy calculations until after startup."""

        if not self._initialization_complete:
            _LOGGER.debug("Using minimal update during initialization")
            async with self._state_lock:
                result = await self._async_minimal_update()
                self.async_set_updated_data(result)
            return

        await super().async_refresh()

    async def _async_minimal_update(self) -> ProbabilityResult:
        """Perform minimal update during startup."""
        return {
            "probability": 0.0,
            "prior_probability": 0.0,
            "active_triggers": [],
            "sensor_probabilities": {},
            "device_states": {},
            "decay_status": {"global_decay": 0.0},
            "sensor_availability": {
                k: v.get("availability", False) for k, v in self._sensor_states.items()
            },
            "is_occupied": False,
        }

    async def mark_initialization_complete(self):
        """Mark the coordinator as fully initialized and start background tasks."""
        async with self._startup_lock:
            self._initialization_complete = True
            # Schedule historical analysis without blocking startup
            self.hass.create_task(self._async_historical_analysis())
            await self.async_refresh()  # Perform first full refresh

    async def async_update_threshold(self, value: float) -> None:
        _LOGGER.debug("Updating threshold")
        if not 0 <= value <= 100:
            raise ValueError("Threshold must be between 0 and 100")
        old_threshold = self.options_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        self.options_config[CONF_THRESHOLD] = value
        _LOGGER.debug(
            "Updated threshold from %.2f to %.2f (decimal: %.3f)",
            old_threshold,
            value,
            self.get_threshold_decimal(),
        )
        await self.async_refresh()

    def get_threshold_decimal(self) -> float:
        _LOGGER.debug("Getting threshold decimal")
        threshold = self.options_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        return threshold / 100.0

    def get_configured_sensors(self) -> list[str]:
        _LOGGER.debug("Getting configured sensors")
        return self._get_all_configured_sensors()

    def _get_decay_config(self) -> DecayConfig:
        _LOGGER.debug("Getting decay config")
        return DecayConfig(
            enabled=self.options_config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
            window=self.options_config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
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

    async def _compute_initial_priors(self) -> None:
        _LOGGER.debug("Computing initial priors")
        if not self.options_config.get(CONF_HISTORICAL_ANALYSIS_ENABLED, True):
            return

        history_period = self.options_config.get(
            CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
        )
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=history_period)
        sensor_ids = self._get_all_configured_sensors()

        for entity_id in sensor_ids:
            p_true, p_false, prior = await self._calculator.calculate_prior(
                entity_id, start_time, end_time
            )
            _LOGGER.debug(
                "Computed priors for %s: %s, %s, prior: %s",
                entity_id,
                p_true,
                p_false,
                prior,
            )

        await self.async_save_state()
        _LOGGER.debug("Initial priors computed and saved.")

        # Notify sensors to update
        self.async_update_listeners()

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

    async def _save_data(self, data: ProbabilityResult) -> None:
        _LOGGER.debug("Saving data")
        try:
            if not data:
                return
            await self.storage.async_save(data)
        except (IOError, ValueError, HomeAssistantError) as err:
            _LOGGER.error("Error saving data: %s", err)

    async def async_stop_periodic_task(self) -> None:
        """Stop the periodic task."""
        _LOGGER.debug("Stopping periodic task")
        if self._remove_periodic_task:
            self._remove_periodic_task()

    async def _async_historical_analysis(self) -> None:
        """Run historical analysis."""
        _LOGGER.info("Running historical analysis")
        async with self._historical_analysis_lock:
            try:
                if not self.options_config.get(CONF_HISTORICAL_ANALYSIS_ENABLED, True):
                    self._historical_analysis_ready.set()
                    return

                await self._compute_initial_priors()
                self._historical_analysis_ready.set()

            except (ValueError, TypeError, HomeAssistantError) as err:
                _LOGGER.error("Historical analysis failed: %s", err)
            finally:
                _LOGGER.info("Historical analysis complete")
                self._historical_analysis_ready.set()

    async def _async_reinitialize_states(self) -> None:
        """Re-initialize states after options have been updated."""
        await self.async_initialize_states()
        await self.async_refresh()

    async def async_register_entity(self, entity_id: str) -> None:
        """Register an entity for state change tracking."""
        if entity_id in self._registered_entities:
            _LOGGER.debug(
                "Entity %s is already registered for state tracking", entity_id
            )
            return  # Prevent duplicate registration

        _LOGGER.debug("Registering entity: %s", entity_id)
        self._entity_ids.add(entity_id)
        self._registered_entities.add(entity_id)  # Keep track of registered entities

        @callback
        def _handle_sensor_state_change(event):
            """Handle the state change of a sensor."""
            entity = event.data.get("entity_id")
            new_state = event.data.get("new_state")
            old_state = event.data.get("old_state")
            _LOGGER.debug(
                "State change detected for %s: from %s to %s",
                entity,
                old_state.state if old_state else None,
                new_state.state if new_state else None,
            )
            self._handle_sensor_state_update(entity, old_state, new_state)

        # Register the state change listener
        self.hass.helpers.event.async_track_state_change_event(
            entity_id, _handle_sensor_state_change
        )

    async def async_unregister_entity(self, entity_id: str) -> None:
        """Unregister an entity from state change tracking."""
        _LOGGER.debug("Unregistering entity: %s", entity_id)
        self._entity_ids.discard(entity_id)
        self._registered_entities.discard(entity_id)  # Remove from the set
        # Additional code to remove the event listener if needed

    def _handle_sensor_state_update(
        self,
        entity_id: str,
        old_state: Optional[State],
        new_state: Optional[State],
    ) -> None:
        """Handle updates to sensor states."""
        _LOGGER.debug(
            "Handling state update for %s: from %s to %s",
            entity_id,
            old_state.state if old_state else None,
            new_state.state if new_state else None,
        )

        # Update the sensor state in the coordinator
        self._sensor_states[entity_id] = {
            "state": new_state.state if new_state else None,
            "attributes": new_state.attributes if new_state else {},
            "availability": new_state is not None
            and new_state.state != STATE_UNAVAILABLE,
        }

        # Optionally, schedule a refresh
        if self._initialization_complete:
            self.hass.async_create_task(self.async_refresh())
