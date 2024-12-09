"""Coordinator for Area Occupancy Detection with optimized update handling."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from collections import deque

from homeassistant.const import (
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, callback, EventStateChangedData
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    CONF_AREA_ID,
)
from .types import (
    ProbabilityResult,
    SensorId,
    SensorStates,
    StorageData,
    CoreConfig,
    OptionsConfig,
    DecayConfig,
    TimeslotData,
)
from .calculations import ProbabilityCalculator
from .historical_analysis import HistoricalAnalysis

_LOGGER = logging.getLogger(__name__)

# Update frequency constants
MOTION_UPDATE_DELAY = 1.0  # seconds
ENVIRONMENTAL_UPDATE_DELAY = 5.0  # seconds
DEVICE_UPDATE_DELAY = 2.0  # seconds
TIMESLOT_UPDATE_INTERVAL = timedelta(hours=6)
MIN_TIME_BETWEEN_UPDATES = timedelta(seconds=1)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Class to manage fetching area occupancy data with optimized updates."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        core_config: CoreConfig,
        options_config: OptionsConfig,
        store: Store[StorageData],
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),  # Fallback update interval
        )

        if not core_config.get("motion_sensors"):
            raise HomeAssistantError("No motion sensors configured")

        self.entry_id = entry_id
        self.store = store
        self.core_config = core_config
        self.options_config = options_config

        # Initialize state tracking
        self._state_lock = asyncio.Lock()
        self._sensor_states: SensorStates = {}
        self._motion_timestamps: dict[SensorId, datetime] = {}
        self._pending_updates: dict[str, asyncio.Task] = {}
        self._update_scheduled = False

        # Initialize history tracking
        self._probability_history = deque(maxlen=12)
        self._occupancy_history = deque([False] * 288, maxlen=288)
        self._last_occupied: datetime | None = None
        self._last_state_change: datetime | None = None
        self._last_update_time: datetime | None = None

        # Initialize timeslot tracking
        self._timeslot_data: TimeslotData = {
            "slots": {},
            "last_updated": dt_util.utcnow(),
        }
        self._historical_analysis_ready = asyncio.Event()

        # Component initialization
        self._calculator = self._create_calculator()
        self._historical_analysis = HistoricalAnalysis(self.hass)
        self._historical_analysis_task: asyncio.Task | None = None

    def _create_calculator(self) -> ProbabilityCalculator:
        """Create probability calculator with current configuration."""
        return ProbabilityCalculator(
            motion_sensors=self.core_config["motion_sensors"],
            media_devices=self.options_config.get("media_devices", []),
            appliances=self.options_config.get("appliances", []),
            illuminance_sensors=self.options_config.get("illuminance_sensors", []),
            humidity_sensors=self.options_config.get("humidity_sensors", []),
            temperature_sensors=self.options_config.get("temperature_sensors", []),
            decay_config=DecayConfig(
                enabled=self.options_config.get("decay_enabled", True),
                window=self.options_config.get("decay_window", 600),
                type=self.options_config.get("decay_type", "linear"),
            ),
        )

    async def async_setup(self) -> None:
        """Perform setup tasks that require async operations."""
        try:
            # Initialize states
            await self._async_initialize_states()

            # Set up optimized state tracking
            self._setup_entity_tracking()

            # Start historical analysis in background
            self._historical_analysis_ready.clear()
            self._historical_analysis_task = self.hass.async_create_task(
                self._async_historical_analysis()
            )

            # Set up periodic timeslot updates
            async_track_time_interval(
                self.hass,
                self._async_update_timeslots,
                TIMESLOT_UPDATE_INTERVAL,
            )

            self.last_update_success = True

        except Exception as err:
            _LOGGER.error("Failed to setup coordinator: %s", err)
            raise HomeAssistantError(f"Coordinator setup failed: {err}") from err

    async def _async_initialize_states(self) -> None:
        """Initialize states for all configured sensors."""
        try:
            # Get all configured sensors
            sensors = self._get_all_configured_sensors()

            for entity_id in sensors:
                state = self.hass.states.get(entity_id)
                is_available = bool(
                    state
                    and state.state not in [STATE_UNAVAILABLE, STATE_UNKNOWN, None, ""]
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
                    entity_id in self.core_config["motion_sensors"]
                    and is_available
                    and state.state == STATE_ON
                ):
                    self._motion_timestamps[entity_id] = dt_util.utcnow()

        except Exception as err:
            _LOGGER.error("Error initializing states: %s", err)
            self._sensor_states = {}

    def _setup_entity_tracking(self) -> None:
        """Set up optimized entity state tracking."""
        entities_by_type = self._categorize_entities()

        # Set up motion sensor tracking
        for entity_id in entities_by_type["motion"]:
            self._async_track_entity_with_debounce(
                entity_id, MOTION_UPDATE_DELAY, self._handle_motion_update
            )

        # Set up environmental sensor tracking
        for entity_id in entities_by_type["environmental"]:
            self._async_track_entity_with_debounce(
                entity_id, ENVIRONMENTAL_UPDATE_DELAY, self._handle_environmental_update
            )

        # Set up device state tracking
        for entity_id in entities_by_type["devices"]:
            self._async_track_entity_with_debounce(
                entity_id, DEVICE_UPDATE_DELAY, self._handle_device_update
            )

    @callback
    def _async_track_entity_with_debounce(
        self, entity_id: str, delay: float, handler: callable
    ) -> None:
        """Track entity state changes with debouncing."""

        @callback
        def _async_entity_changed(event: EventStateChangedData) -> None:
            """Handle entity state change."""
            if entity_id in self._pending_updates:
                self._pending_updates[entity_id].cancel()

            self._pending_updates[entity_id] = self.hass.async_create_task(
                self._async_handle_debounced_update(entity_id, delay, handler)
            )

        self.hass.bus.async_listen(
            "state_changed",
            _async_entity_changed,
            callback_kwargs={"entity_id": entity_id},
        )

    @callback
    def _async_track_entity_with_debounce(
        self, entity_id: str, delay: float, handler: callable
    ) -> None:
        """Track entity state changes with debouncing."""

        @callback
        def _async_entity_changed(event: EventStateChangedData) -> None:
            """Handle entity state change."""
            if event.data["entity_id"] != entity_id:
                return

            if entity_id in self._pending_updates:
                self._pending_updates[entity_id].cancel()

            self._pending_updates[entity_id] = self.hass.async_create_task(
                self._async_handle_debounced_update(entity_id, delay, handler)
            )

        # Use the correct event listening pattern
        self.hass.bus.async_listen(
            "state_changed",
            _async_entity_changed,
        )

    async def _async_handle_debounced_update(
        self, entity_id: str, delay: float, handler: callable
    ) -> None:
        """Handle debounced entity update."""
        try:
            await asyncio.sleep(delay)
            await handler(entity_id)
            self._pending_updates.pop(entity_id, None)

            # Schedule coordinator update if needed
            if not self._update_scheduled:
                self._update_scheduled = True
                self.hass.async_create_task(self._async_delayed_coordinator_update())
        except asyncio.CancelledError:
            _LOGGER.debug("Update cancelled for %s", entity_id)
        except Exception as err:
            _LOGGER.error("Error handling update for %s: %s", entity_id, err)

    async def _async_delayed_coordinator_update(self) -> None:
        """Perform delayed coordinator update."""
        # Ensure minimum time between updates
        if self._last_update_time:
            time_since_update = dt_util.utcnow() - self._last_update_time
            if time_since_update < MIN_TIME_BETWEEN_UPDATES:
                await asyncio.sleep(
                    (MIN_TIME_BETWEEN_UPDATES - time_since_update).total_seconds()
                )

        await self.async_refresh()
        self._update_scheduled = False
        self._last_update_time = dt_util.utcnow()

    async def _handle_motion_update(self, entity_id: str) -> None:
        """Handle motion sensor update."""
        state = self.hass.states.get(entity_id)
        if not state:
            return

        async with self._state_lock:
            if state.state == STATE_ON:
                self._motion_timestamps[entity_id] = dt_util.utcnow()

            self._update_sensor_state(entity_id, state)

    async def _handle_environmental_update(self, entity_id: str) -> None:
        """Handle environmental sensor update."""
        state = self.hass.states.get(entity_id)
        if not state:
            return

        async with self._state_lock:
            self._update_sensor_state(entity_id, state)

    async def _handle_device_update(self, entity_id: str) -> None:
        """Handle device state update."""
        state = self.hass.states.get(entity_id)
        if not state:
            return

        async with self._state_lock:
            self._update_sensor_state(entity_id, state)

    def _update_sensor_state(self, entity_id: str, state) -> None:
        """Update sensor state with validation."""
        is_available = bool(
            state and state.state not in [STATE_UNAVAILABLE, STATE_UNKNOWN, None, ""]
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

    async def _async_historical_analysis(self) -> None:
        """Run historical analysis in background."""
        try:
            if not self.options_config.get("historical_analysis_enabled", True):
                self._historical_analysis_ready.set()
                return

            self._timeslot_data = await self._historical_analysis.calculate_timeslots(
                self._get_all_configured_sensors(),
                self.options_config["history_period"],
            )

        except Exception as err:
            _LOGGER.error("Historical analysis failed: %s", err)
        finally:
            self._historical_analysis_ready.set()

    async def _async_update_timeslots(self, _: datetime) -> None:
        """Update timeslot data periodically."""
        try:
            if not self.options_config.get("historical_analysis_enabled", True):
                return

            new_data = await self._historical_analysis.calculate_timeslots(
                self._get_all_configured_sensors(),
                self.options_config["history_period"],
            )

            if isinstance(new_data, dict) and "slots" in new_data:
                self._timeslot_data = new_data

        except Exception as err:
            _LOGGER.error("Error updating timeslots: %s", err)

    def _categorize_entities(self) -> dict[str, list[str]]:
        """Categorize entities by type for optimized tracking."""
        return {
            "motion": self.core_config.get("motion_sensors", []),
            "environmental": [
                *self.options_config.get("illuminance_sensors", []),
                *self.options_config.get("humidity_sensors", []),
                *self.options_config.get("temperature_sensors", []),
            ],
            "devices": [
                *self.options_config.get("media_devices", []),
                *self.options_config.get("appliances", []),
            ],
        }

    async def _async_update_data(self) -> ProbabilityResult:
        """Update data with optimized calculations."""
        try:
            async with self._state_lock:
                # Get current timeslot if historical analysis is ready
                current_slot = {}
                if self._historical_analysis_ready.is_set():
                    current_time = dt_util.utcnow()
                    slot_key = f"{current_time.hour:02d}:{(current_time.minute // 30) * 30:02d}"
                    current_slot = self._timeslot_data.get("slots", {}).get(
                        slot_key, {}
                    )

                result = await self._calculator.calculate(
                    self._sensor_states,
                    self._motion_timestamps,
                    current_slot,
                )

                # Update historical tracking
                self._update_historical_tracking(result)

                # Store results
                await self._async_store_result(result)

                return result

        except Exception as err:
            _LOGGER.error("Error updating data: %s", err)
            raise

    def _update_historical_tracking(self, result: ProbabilityResult) -> None:
        """Update historical tracking with rate limiting."""
        now = dt_util.utcnow()

        # Update tracking data
        self._probability_history.append(result["probability"])
        is_occupied = result["probability"] >= self.options_config["threshold"]
        self._occupancy_history.append(is_occupied)

        if is_occupied:
            self._last_occupied = now

        if not self._last_state_change or (
            bool(self._occupancy_history[-2]) != is_occupied
            if len(self._occupancy_history) > 1
            else True
        ):
            self._last_state_change = now

    async def _async_store_result(self, result: ProbabilityResult) -> None:
        """Store calculation result."""
        try:
            stored_data = await self.store.async_load() or {}
            area_id = self.core_config[CONF_AREA_ID]

            # Initialize if needed
            stored_data.setdefault("areas", {})
            stored_data["areas"].setdefault(area_id, {})

            # Update stored data
            stored_data["areas"][area_id].update(
                {
                    "last_updated": dt_util.utcnow().isoformat(),
                    "last_probability": result["probability"],
                    "historical_metrics": self._get_historical_metrics(),
                    "configuration": {
                        "motion_sensors": self.core_config["motion_sensors"],
                        "media_devices": self.options_config.get("media_devices", []),
                        "appliances": self.options_config.get("appliances", []),
                        "illuminance_sensors": self.options_config.get(
                            "illuminance_sensors", []
                        ),
                        "humidity_sensors": self.options_config.get(
                            "humidity_sensors", []
                        ),
                        "temperature_sensors": self.options_config.get(
                            "temperature_sensors", []
                        ),
                    },
                }
            )

            await self.store.async_save(stored_data)

        except Exception as err:
            _LOGGER.error("Error storing result: %s", err)

    def _get_historical_metrics(self) -> dict[str, Any]:
        """Get historical metrics for storage and reporting."""
        now = dt_util.utcnow()

        # Calculate moving average
        moving_avg = (
            sum(self._probability_history) / len(self._probability_history)
            if self._probability_history
            else 0.0
        )

        # Calculate rate of change
        rate_of_change = 0.0
        if len(self._probability_history) >= 2:
            change = self._probability_history[-1] - self._probability_history[0]
            time_window = (
                len(self._probability_history) * 5
            )  # 5 minutes between samples
            rate_of_change = (change / time_window) * 60  # per hour

        # Calculate occupancy rate
        occupancy_rate = (
            sum(1 for x in self._occupancy_history if x) / len(self._occupancy_history)
            if self._occupancy_history
            else 0.0
        )

        # Calculate state duration
        state_duration = 0.0
        if self._last_state_change:
            state_duration = (now - self._last_state_change).total_seconds()

        return {
            "moving_average": moving_avg,
            "rate_of_change": rate_of_change,
            "occupancy_rate": occupancy_rate,
            "state_duration": state_duration,
            "min_probability": (
                min(self._probability_history) if self._probability_history else 0.0
            ),
            "max_probability": (
                max(self._probability_history) if self._probability_history else 0.0
            ),
            "last_occupied": (
                self._last_occupied.isoformat() if self._last_occupied else None
            ),
        }

    def _get_all_configured_sensors(self) -> list[str]:
        """Get list of all configured sensor entity IDs."""
        return [
            *self.core_config["motion_sensors"],
            *self.options_config.get("media_devices", []),
            *self.options_config.get("appliances", []),
            *self.options_config.get("illuminance_sensors", []),
            *self.options_config.get("humidity_sensors", []),
            *self.options_config.get("temperature_sensors", []),
        ]

    async def async_reset(self) -> None:
        """Reset coordinator state."""
        self._sensor_states.clear()
        self._motion_timestamps.clear()
        self._probability_history.clear()
        self._occupancy_history.extend([False] * self._occupancy_history.maxlen)
        self._last_occupied = None
        self._last_state_change = None
        self._last_update_time = None

        # Cancel pending updates
        for task in self._pending_updates.values():
            task.cancel()
        self._pending_updates.clear()

        # Reset historical analysis
        if self._historical_analysis_task and not self._historical_analysis_task.done():
            self._historical_analysis_task.cancel()

        self._historical_analysis_ready.clear()
        self._timeslot_data = {"slots": {}, "last_updated": dt_util.utcnow()}

        # Force refresh after reset
        await self.async_refresh()

    async def async_unload(self) -> None:
        """Unload coordinator and clean up resources."""
        # Cancel all pending tasks
        for task in self._pending_updates.values():
            task.cancel()

        if self._historical_analysis_task and not self._historical_analysis_task.done():
            self._historical_analysis_task.cancel()

        # Final state save
        try:
            result = await self._async_update_data()
            await self._async_store_result(result)
        except Exception as err:
            _LOGGER.error("Error saving final state: %s", err)

    def update_options(self, options_config: OptionsConfig) -> None:
        """Update coordinator with new options."""
        try:
            self.options_config = options_config
            self._calculator = self._create_calculator()

            # Clear invalid sensor states
            current_sensors = set(self._get_all_configured_sensors())
            self._sensor_states = {
                entity_id: state
                for entity_id, state in self._sensor_states.items()
                if entity_id in current_sensors
            }
            self._motion_timestamps = {
                entity_id: timestamp
                for entity_id, timestamp in self._motion_timestamps.items()
                if entity_id in self.core_config["motion_sensors"]
            }

            # Reset tracking setup
            for task in self._pending_updates.values():
                task.cancel()
            self._pending_updates.clear()
            self._setup_entity_tracking()

            _LOGGER.debug(
                "Updated coordinator options for %s",
                self.core_config["name"],
            )
        except Exception as err:
            _LOGGER.error("Error updating coordinator options: %s", err)
            raise HomeAssistantError(
                f"Failed to update coordinator options: {err}"
            ) from err

    @property
    def available(self) -> bool:
        """Return if coordinator is available."""
        # Check if we have motion sensors reporting availability
        motion_sensors = self.core_config.get("motion_sensors", [])
        if not motion_sensors:
            return False

        # Consider coordinator available if any motion sensor is available
        return any(
            self._sensor_states.get(sensor_id, {}).get("availability", False)
            for sensor_id in motion_sensors
        )

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        return {
            "core_config": self.core_config,
            "options_config": self.options_config,
            "sensor_states": self._sensor_states,
            "historical_data": {
                "probability_history": list(self._probability_history),
                "occupancy_history": list(self._occupancy_history),
                "last_occupied": (
                    self._last_occupied.isoformat() if self._last_occupied else None
                ),
                "last_state_change": (
                    self._last_state_change.isoformat()
                    if self._last_state_change
                    else None
                ),
            },
            "timeslot_data": {
                "slot_count": len(self._timeslot_data["slots"]),
                "last_updated": self._timeslot_data["last_updated"].isoformat(),
            },
            "update_statistics": {
                "last_update_time": (
                    self._last_update_time.isoformat()
                    if self._last_update_time
                    else None
                ),
                "pending_updates": len(self._pending_updates),
                "historical_analysis_ready": self._historical_analysis_ready.is_set(),
            },
        }
