"""Coordinator for Area Occupancy Detection with optimized update handling."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from collections import deque

from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    CONF_AREA_ID,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_TYPE,
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
from .storage import AreaOccupancyStorage

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Class to manage fetching area occupancy data with optimized updates."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        core_config: CoreConfig,
        options_config: OptionsConfig,
    ) -> None:
        """Initialize the coordinator."""
        self.storage = AreaOccupancyStorage(hass, entry_id)
        self._last_known_values: dict[str, Any] = {}

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),
            update_method=self._async_update_data,
        )

        if not core_config.get("motion_sensors"):
            raise HomeAssistantError("No motion sensors configured")

        self.entry_id = entry_id
        self.core_config = core_config
        self.options_config = options_config

        # State tracking
        self._state_lock = asyncio.Lock()
        self._sensor_states: SensorStates = {}
        self._motion_timestamps: dict[SensorId, datetime] = {}

        # History tracking
        self._probability_history = deque(maxlen=12)
        self._occupancy_history = deque([False] * 288, maxlen=288)
        self._last_occupied: datetime | None = None
        self._last_state_change: datetime | None = None

        # Timeslot tracking
        self._timeslot_data: TimeslotData = {
            "slots": {},
            "last_updated": dt_util.utcnow(),
        }
        self._historical_analysis_ready = asyncio.Event()

        # Components
        self._calculator = self._create_calculator()
        self._historical_analysis = HistoricalAnalysis(self.hass)
        self._historical_analysis_task: asyncio.Task | None = None

        # Storage
        self._storage_lock = asyncio.Lock()
        self._last_save = dt_util.utcnow()
        self._save_interval = timedelta(minutes=5)

        self._entity_ids: set[str] = set()

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
                enabled=self.options_config.get("decay_enabled", DEFAULT_DECAY_ENABLED),
                window=self.options_config.get("decay_window", DEFAULT_DECAY_WINDOW),
                type=self.options_config.get("decay_type", DEFAULT_DECAY_TYPE),
            ),
        )

    async def async_setup(self) -> None:
        """Perform setup tasks that require async operations."""
        start_time = dt_util.utcnow()
        try:
            # Load stored data first
            await self._load_stored_data()

            # Set up state tracking
            self._setup_entity_tracking()

            # Start all initialization in background
            self.hass.async_create_task(self._async_background_setup())

            setup_time = (dt_util.utcnow() - start_time).total_seconds()
            _LOGGER.debug("Coordinator setup completed in %.3f seconds", setup_time)

        except (HomeAssistantError, ValueError, RuntimeError) as err:
            _LOGGER.error("Failed to setup coordinator: %s", err)
            raise HomeAssistantError(f"Coordinator setup failed: {err}") from err

    async def _async_background_setup(self) -> None:
        """Perform all heavy setup operations in background."""
        start_time = dt_util.utcnow()
        try:
            # Initialize states
            state_start = dt_util.utcnow()
            await self._async_initialize_states()
            _LOGGER.debug(
                "State initialization took %.3f seconds",
                (dt_util.utcnow() - state_start).total_seconds(),
            )

            # Set up state tracking
            tracking_start = dt_util.utcnow()
            self._setup_entity_tracking()
            _LOGGER.debug(
                "Entity tracking setup took %.3f seconds",
                (dt_util.utcnow() - tracking_start).total_seconds(),
            )

            # Start historical analysis
            self._historical_analysis_ready.clear()
            _LOGGER.debug("Starting historical analysis task")
            self._historical_analysis_task = self.hass.async_create_task(
                self._async_historical_analysis()
            )

            # Schedule first update instead of waiting
            update_start = dt_util.utcnow()
            self.hass.async_create_task(self.async_refresh())
            _LOGGER.debug(
                "Scheduled first update at %.3f seconds",
                (dt_util.utcnow() - update_start).total_seconds(),
            )

            total_time = (dt_util.utcnow() - start_time).total_seconds()
            _LOGGER.debug("Background setup completed in %.3f seconds", total_time)

        except (
            HomeAssistantError,
            ValueError,
            RuntimeError,
            asyncio.TimeoutError,
        ) as err:
            _LOGGER.error("Background setup failed: %s", err)

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

        except (HomeAssistantError, ValueError, LookupError) as err:
            _LOGGER.error("Error initializing states: %s", err)
            self._sensor_states = {}

    def _setup_entity_tracking(self) -> None:
        """Set up optimized entity state tracking."""
        entities = self._get_all_configured_sensors()
        _LOGGER.debug("Setting up entity tracking for: %s", entities)

        @callback
        def async_state_changed_listener(event) -> None:
            """Handle entity state changes."""
            entity_id = event.data.get("entity_id")
            new_state = event.data.get("new_state")

            if not new_state or entity_id not in entities:
                return

            # Update state immediately
            is_available = bool(
                new_state.state not in [STATE_UNAVAILABLE, STATE_UNKNOWN, None, ""]
            )

            self._sensor_states[entity_id] = {
                "state": new_state.state if is_available else None,
                "last_changed": (
                    new_state.last_changed.isoformat()
                    if new_state.last_changed
                    else dt_util.utcnow().isoformat()
                ),
                "availability": is_available,
            }

            # Update motion timestamps
            if (
                entity_id in self.core_config["motion_sensors"]
                and is_available
                and new_state.state == STATE_ON
            ):
                self._motion_timestamps[entity_id] = dt_util.utcnow()

            # Schedule update
            self.hass.async_create_task(self.async_refresh())

        # Set up state tracking
        async_track_state_change_event(
            self.hass,
            entities,
            async_state_changed_listener,
        )

    async def _async_historical_analysis(self) -> None:
        """Run historical analysis in background."""
        try:
            if not self.options_config.get("historical_analysis_enabled", True):
                self._historical_analysis_ready.set()
                return

            self._timeslot_data = await self._historical_analysis.calculate_timeslots(
                self._get_all_configured_sensors(),
                self.options_config.get("history_period", DEFAULT_HISTORY_PERIOD),
            )

        except (ValueError, TypeError, HomeAssistantError) as err:
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

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error updating timeslots: %s", err)

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

                # Save updated data
                await self._save_data(result)

                return result

        except Exception as err:
            _LOGGER.error("Error updating data: %s", err, exc_info=True)
            raise UpdateFailed(f"Error updating data: {err}") from err

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
            stored_data = await self.storage.async_load() or {}
            area_id = self.core_config[CONF_AREA_ID]

            stored_data.setdefault("areas", {})
            stored_data["areas"].setdefault(area_id, {})

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

            await self.storage.async_save(stored_data)

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error storing result: %s", err)

    def _get_historical_metrics(self) -> dict[str, Any]:
        """Get historical metrics for storage and reporting."""
        now = dt_util.utcnow()

        moving_avg = (
            sum(self._probability_history) / len(self._probability_history)
            if self._probability_history
            else 0.0
        )

        rate_of_change = 0.0
        if len(self._probability_history) >= 2:
            change = self._probability_history[-1] - self._probability_history[0]
            time_window = len(self._probability_history) * 5
            rate_of_change = (change / time_window) * 60

        occupancy_rate = (
            sum(1 for x in self._occupancy_history if x) / len(self._occupancy_history)
            if self._occupancy_history
            else 0.0
        )

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

        # Reset historical analysis
        if self._historical_analysis_task and not self._historical_analysis_task.done():
            self._historical_analysis_task.cancel()

        self._historical_analysis_ready.clear()
        self._timeslot_data = {"slots": {}, "last_updated": dt_util.utcnow()}

        # Force refresh after reset
        await self.async_refresh()

    async def async_unload(self) -> None:
        """Unload coordinator and clean up resources."""
        if self._historical_analysis_task and not self._historical_analysis_task.done():
            self._historical_analysis_task.cancel()

        if self.data:
            await self._save_data(self.data)

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
        }

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device info for the integration."""
        return {
            "identifiers": {(DOMAIN, self.entry_id)},
            "name": self.core_config["name"],
            "manufacturer": DEVICE_MANUFACTURER,
            "model": DEVICE_MODEL,
            "sw_version": DEVICE_SW_VERSION,
        }

    async def _load_stored_data(self) -> None:
        """Load data with migration support."""
        try:
            stored_data = await self.storage.async_load()
            self._last_known_values = stored_data["data"]
            _LOGGER.debug("Loaded stored data: %s", self._last_known_values)
        except (IOError, ValueError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error loading stored data: %s", err)
            self._last_known_values = {}

    async def _save_data(self, data: ProbabilityResult) -> None:
        """Save data with validation."""
        try:
            if not data:
                return
            await self.storage.async_save(data)
        except (IOError, ValueError, HomeAssistantError) as err:
            _LOGGER.error("Error saving data: %s", err)

    async def async_save_state(self) -> None:
        """Save state to storage with rate limiting."""
        now = dt_util.utcnow()
        if now - self._last_save < self._save_interval:
            return

        async with self._storage_lock:
            try:
                await self.storage.async_save(self.get_storage_data())
                self._last_save = now
            except (IOError, ValueError, HomeAssistantError) as err:
                _LOGGER.error("Failed to save state: %s", err)

    async def async_restore_state(self, stored_data: StorageData) -> None:
        """Restore state from storage."""
        try:
            # Validate stored data format
            if not isinstance(stored_data, dict):
                raise ValueError("Invalid storage data format")

            # Restore coordinator state
            self._last_known_values = stored_data.get("last_known_values", {})
            self._probability_history = deque(
                stored_data.get("probability_history", []), maxlen=12
            )
            # Restore occupancy history
            self._occupancy_history = deque(
                stored_data.get("occupancy_history", [False] * 288), maxlen=288
            )

            # Restore timestamps
            last_occupied = stored_data.get("last_occupied")
            self._last_occupied = (
                dt_util.parse_datetime(last_occupied) if last_occupied else None
            )

            last_state_change = stored_data.get("last_state_change")
            self._last_state_change = (
                dt_util.parse_datetime(last_state_change) if last_state_change else None
            )

        except (ValueError, TypeError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error restoring state: %s", err)
            # Reset to default state
            self._reset_state()

    def _reset_state(self) -> None:
        """Reset coordinator state to defaults."""
        self._last_known_values = {}
        self._probability_history = deque(maxlen=12)
        self._occupancy_history = deque([False] * 288, maxlen=288)
        self._last_occupied = None
        self._last_state_change = None
        self._last_save = dt_util.utcnow()

    def get_storage_data(self) -> dict:
        """Get data for storage."""
        return {
            "last_known_values": self._last_known_values,
            "probability_history": list(self._probability_history),
            "occupancy_history": list(self._occupancy_history),
            "last_occupied": (
                self._last_occupied.isoformat() if self._last_occupied else None
            ),
            "last_state_change": (
                self._last_state_change.isoformat() if self._last_state_change else None
            ),
        }

    @property
    def entity_ids(self) -> set[str]:
        """Get the set of configured entity IDs."""
        return self._entity_ids

    def register_entity(self, entity_id: str) -> None:
        """Register an entity ID with the coordinator."""
        self._entity_ids.add(entity_id)

    def unregister_entity(self, entity_id: str) -> None:
        """Unregister an entity ID from the coordinator."""
        self._entity_ids.discard(entity_id)

    async def async_refresh(self) -> None:
        """Refresh data with forced update."""
        try:
            result = await self._async_update_data()
            self.async_set_updated_data(result)
        except (HomeAssistantError, ValueError, RuntimeError) as err:
            _LOGGER.error("Error during refresh: %s", err, exc_info=True)
