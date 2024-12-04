"""Coordinator for Area Occupancy Detection."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import deque

from homeassistant.const import (
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)

from .const import (
    DOMAIN,
    CONF_AREA_ID,
    ProbabilityResult,
    SensorId,
    SensorStates,
    StorageData,
)
from .calculations import DecayConfig, ProbabilityCalculator
from .pattern_analyzer import OccupancyPatternAnalyzer
from .historical_analysis import HistoricalAnalysis
from .config_management import CoreConfig, OptionsConfig

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Class to manage fetching area occupancy data."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        core_config: CoreConfig,
        options_config: OptionsConfig,
        base_config: dict[str, Any],
        store: Store[StorageData],
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),
        )

        if not base_config or "base_probabilities" not in base_config:
            raise HomeAssistantError("Invalid base configuration provided")

        if not core_config.get("motion_sensors"):
            raise HomeAssistantError("No motion sensors configured")

        self.base_config = base_config
        self.entry_id = entry_id
        self.store = store

        self.core_config = core_config
        self.options_config = options_config

        # Initialize state tracking
        self._state_lock = asyncio.Lock()
        self._last_probability: Optional[float] = None
        self._max_change_rate = 0.3
        self._sensor_states: SensorStates = {}
        self._motion_timestamps: dict[SensorId, datetime] = {}
        self._unsubscribe_handlers: list[callable] = []

        # Initialize history tracking
        self._probability_history = deque(maxlen=12)  # 1 hour of 5-minute readings
        self._last_occupied: datetime | None = None
        self._last_state_change: datetime | None = None
        self._occupancy_history = deque([False] * 288, maxlen=288)  # 24 hours
        self._historical_data: dict[str, Any] = {}

        # Initialize components
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize coordinator components."""
        # Initialize historical analysis
        self._history_analyzer = HistoricalAnalysis(
            hass=self.hass,
            history_period=self.options_config.get("history_period", 7),
            motion_sensors=self.core_config["motion_sensors"],
            media_devices=self.options_config.get("media_devices", []),
            environmental_sensors=[
                *self.options_config.get("illuminance_sensors", []),
                *self.options_config.get("humidity_sensors", []),
                *self.options_config.get("temperature_sensors", []),
            ],
        )

        # Initialize pattern analyzer
        self._pattern_analyzer = OccupancyPatternAnalyzer()

        # Initialize calculator
        self._calculator = self._create_calculator()

    def _create_calculator(self) -> ProbabilityCalculator:
        """Create probability calculator with current configuration."""
        return ProbabilityCalculator(
            base_config=self.base_config,
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
            # Initialize historical analysis
            self._historical_data = await self._history_analyzer.initialize_history()
            _LOGGER.debug("Historical analysis initialized: %s", self._historical_data)

            # Set up state change listeners
            self._setup_state_listeners()

            # Set up history update
            self._setup_history_update()

            # Initialize states
            self._initialize_states()

        except Exception as err:
            _LOGGER.error("Failed to setup coordinator: %s", err)
            raise HomeAssistantError(f"Coordinator setup failed: {err}") from err

    def _initialize_states(self) -> None:
        """Initialize states for all configured sensors."""
        try:
            for entity_id in self._get_all_configured_sensors():
                state = self.hass.states.get(entity_id)
                if state and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    self._sensor_states[entity_id] = self._create_sensor_state(state)
        except Exception as err:
            _LOGGER.error("Error initializing states: %s", err)
            raise HomeAssistantError(f"Failed to initialize states: {err}") from err

    def _create_sensor_state(self, state: State) -> dict[str, Any]:
        """Create standardized sensor state dictionary."""
        return {
            "state": state.state,
            "last_changed": state.last_changed.isoformat(),
            "availability": state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN),
        }

    async def _async_handle_state_change(
        self, entity_id: str, new_state: State | None
    ) -> None:
        """Handle state changes for sensors."""
        async with self._state_lock:
            if not new_state or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                if entity_id in self._sensor_states:
                    self._sensor_states[entity_id] = {
                        "state": None,
                        "last_changed": dt_util.utcnow().isoformat(),
                        "availability": False,
                    }
                return

            # Handle motion sensor states
            if (
                entity_id in self.core_config["motion_sensors"]
                and new_state.state == STATE_ON
            ):
                self._motion_timestamps[entity_id] = dt_util.utcnow()

            # Update sensor state
            self._sensor_states[entity_id] = self._create_sensor_state(new_state)

            # Trigger update
            self.async_set_updated_data(await self._async_calculate_data())

    def _setup_state_listeners(self) -> None:
        """Set up state change listeners."""

        @callback
        def async_state_changed(event) -> None:
            """Handle sensor state changes."""
            entity_id: str = event.data["entity_id"]
            new_state = event.data.get("new_state")
            asyncio.create_task(self._async_handle_state_change(entity_id, new_state))

        self.unsubscribe()
        self._unsubscribe_handlers.append(
            async_track_state_change_event(
                self.hass,
                self._get_all_configured_sensors(),
                async_state_changed,
            )
        )

    def _setup_history_update(self) -> None:
        """Set up periodic history updates."""
        update_interval = timedelta(minutes=15)

        @callback
        def history_update(_: datetime) -> None:
            """Update history data periodically."""
            asyncio.create_task(self._async_update_history(self.data or {}))

        self._unsubscribe_handlers.append(
            async_track_time_interval(self.hass, history_update, update_interval)
        )

    async def _get_historical_patterns(self) -> dict[str, Any]:
        """Get historical patterns for probability calculations."""
        if not self._historical_data:
            return {}

        current_time = dt_util.utcnow()
        time_slot = current_time.strftime("%H:%M")
        day_slot = current_time.strftime("%A").lower()

        patterns = {}

        if "occupancy_patterns" in self._historical_data:
            time_slots = self._historical_data["occupancy_patterns"].get(
                "time_slots", {}
            )
            day_patterns = self._historical_data["occupancy_patterns"].get(
                "day_patterns", {}
            )

            patterns.update(
                {
                    "typical_occupancy_rate": time_slots.get(time_slot, {}).get(
                        "occupied_ratio", 0.0
                    ),
                    "day_occupancy_rate": day_patterns.get(day_slot, {}).get(
                        "occupied_ratio", 0.0
                    ),
                }
            )

        if "sensor_correlations" in self._historical_data:
            patterns["sensor_correlations"] = self._historical_data[
                "sensor_correlations"
            ]

        return patterns

    def _update_historical_data(self, probability: float, is_occupied: bool) -> None:
        """Update historical tracking data."""
        now = dt_util.utcnow()

        self._probability_history.append(probability)
        self._occupancy_history.append(is_occupied)

        if is_occupied:
            self._last_occupied = now

        current_state = is_occupied
        if not self._last_state_change or (
            bool(self._occupancy_history[-2]) != current_state
            if len(self._occupancy_history) > 1
            else True
        ):
            self._last_state_change = now

    def _get_historical_metrics(self) -> dict[str, Any]:
        """Calculate historical metrics."""
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

    async def _async_calculate_data(self) -> ProbabilityResult:
        """Calculate current occupancy data."""
        try:
            # Get historical patterns
            historical_patterns = await self._get_historical_patterns()

            # Calculate base probability
            data = self._calculator.calculate(
                self._sensor_states,
                self._motion_timestamps,
                historical_patterns=historical_patterns,
            )

            # Update historical tracking
            current_probability = data["probability"]
            current_threshold = self.options_config["threshold"]
            is_occupied = current_probability >= current_threshold

            self._update_historical_data(current_probability, is_occupied)

            # Update pattern recognition
            now = dt_util.utcnow()
            try:
                self._pattern_analyzer.update_pattern(
                    self.core_config["name"], now, is_occupied
                )
                pattern_prob, pattern_confidence = (
                    self._pattern_analyzer.get_probability_adjustment(
                        self.core_config["name"], now
                    )
                )

                # Apply pattern adjustment if confidence is sufficient
                if pattern_confidence >= self.options_config["minimum_confidence"]:
                    pattern_weight = pattern_confidence * 0.3
                    final_probability = (
                        current_probability * (1 - pattern_weight)
                        + pattern_prob * pattern_weight
                    )
                    data["probability"] = max(0.0, min(1.0, final_probability))
            except Exception as pattern_err:
                _LOGGER.warning("Error updating pattern analysis: %s", pattern_err)

            # Add historical metrics
            historical_metrics = self._get_historical_metrics()
            data.update(historical_metrics)

            return data

        except Exception as err:
            _LOGGER.error("Error calculating occupancy data: %s", err)
            raise HomeAssistantError("Failed to calculate occupancy data") from err

    async def _async_update_data(self) -> ProbabilityResult:
        """Update data from all sources."""
        try:
            # Update historical analysis if enabled
            if self.options_config.get("historical_analysis_enabled", True):
                new_historical_data = await self._history_analyzer.update_analysis()
                if new_historical_data:
                    self._historical_data.update(new_historical_data)
        except Exception as err:
            _LOGGER.error("Failed to update historical analysis: %s", err)

        return await self._async_calculate_data()

    async def _async_update_history(self, data: ProbabilityResult) -> None:
        """Update historical data storage."""
        try:
            stored_data = await self.store.async_load() or {}
            stored_data.setdefault("areas", {})
            area_id = self.core_config["name"].lower().replace(" ", "_")

            stored_data["areas"].setdefault(
                area_id,
                {
                    "last_updated": dt_util.utcnow().isoformat(),
                    "probabilities": [],
                    "patterns": {},
                },
            )

            # Update with new data
            area_data = stored_data["areas"][area_id]
            area_data["probabilities"].append(
                {
                    "timestamp": dt_util.utcnow().isoformat(),
                    "probability": data["probability"],
                    "active_triggers": data["active_triggers"],
                }
            )

            # Limit stored history
            area_data["probabilities"] = area_data["probabilities"][-1000:]

            # Save updated data
            await self.store.async_save(stored_data)

        except Exception as err:
            _LOGGER.error("Failed to update history: %s", err)

    def get_storage_data(self) -> StorageData:
        """Get data for storage before unloading."""
        try:
            # Use area_id for storage key instead of name
            area_id = self.core_config[CONF_AREA_ID]

            return {
                "version": 1,
                "last_updated": dt_util.utcnow().isoformat(),
                "areas": {
                    area_id: {
                        "last_updated": dt_util.utcnow().isoformat(),
                        "probabilities": [float(p) for p in self._probability_history],
                        "occupancy_history": list(self._occupancy_history),
                        "last_occupied": (
                            self._last_occupied.isoformat()
                            if self._last_occupied
                            else None
                        ),
                        "last_state_change": (
                            self._last_state_change.isoformat()
                            if self._last_state_change
                            else None
                        ),
                        "historical_data": self._historical_data,
                        "configuration": {
                            "motion_sensors": self.core_config["motion_sensors"],
                            "media_devices": self.options_config.get(
                                "media_devices", []
                            ),
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
                },
            }
        except Exception as err:
            _LOGGER.error("Error preparing storage data: %s", err)
            raise HomeAssistantError(f"Failed to prepare storage data: {err}") from err

    def unsubscribe(self) -> None:
        """Unsubscribe from all registered events."""
        while self._unsubscribe_handlers:
            self._unsubscribe_handlers.pop()()

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

    def _limit_probability_change(self, new_probability: float) -> float:
        """Limit the rate of probability change."""
        if self._last_probability is None:
            self._last_probability = new_probability
            return new_probability

        max_change = self._max_change_rate
        min_prob = max(0.0, self._last_probability - max_change)
        max_prob = min(1.0, self._last_probability + max_change)

        limited_prob = max(min_prob, min(new_probability, max_prob))
        self._last_probability = limited_prob
        return limited_prob

    def update_options(self, options_config: OptionsConfig) -> None:
        """Update coordinator with new options."""
        try:
            self.options_config = options_config
            self._setup_components()

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

            _LOGGER.debug(
                "Updated coordinator options for %s: %s",
                self.core_config["name"],
                options_config,
            )
        except Exception as err:
            _LOGGER.error("Error updating coordinator options: %s", err)
            raise HomeAssistantError(
                f"Failed to update coordinator options: {err}"
            ) from err

    async def async_reset_history(self) -> None:
        """Reset historical data for this area."""
        try:
            area_id = self.core_config["name"].lower().replace(" ", "_")
            stored_data = await self.store.async_load() or {}

            if "areas" in stored_data and area_id in stored_data["areas"]:
                stored_data["areas"][area_id] = {
                    "last_updated": dt_util.utcnow().isoformat(),
                    "probabilities": [],
                    "patterns": {},
                }
                await self.store.async_save(stored_data)

                self._historical_data = {}
                self._probability_history.clear()
                self._occupancy_history.extend([False] * self._occupancy_history.maxlen)
                self._last_occupied = None
                self._last_state_change = None

                await self.async_refresh()

        except Exception as err:
            _LOGGER.error("Failed to reset history: %s", err)
            raise HomeAssistantError("Failed to reset historical data") from err

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information for troubleshooting."""
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
            "patterns": self._historical_data.get("occupancy_patterns", {}),
        }
