"""Historical state analysis for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, TypedDict, List
from collections import defaultdict

from homeassistant.core import HomeAssistant
from homeassistant.components.recorder import get_instance, history
from homeassistant.const import STATE_ON, STATE_OFF, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError

_LOGGER = logging.getLogger(__name__)


class TimeSlot(TypedDict):
    """Time slot statistics."""

    total_samples: int
    active_samples: int
    average_duration: float
    confidence: float


class SensorHistory(TypedDict):
    """Historical data for a sensor."""

    total_activations: int
    average_duration: float
    time_slots: dict[str, TimeSlot]
    day_patterns: dict[str, TimeSlot]
    correlated_sensors: dict[str, float]


class HistoricalAnalysis:
    """Analyzes historical state data for area occupancy patterns."""

    def __init__(
        self,
        hass: HomeAssistant,
        history_period: int,
        motion_sensors: list[str],
        media_devices: list[str] | None = None,
        environmental_sensors: list[str] | None = None,
    ) -> None:
        """Initialize the historical analysis."""
        self.hass = hass
        self.history_period = history_period
        self.motion_sensors = motion_sensors
        self.media_devices = media_devices or []
        self.environmental_sensors = environmental_sensors or []
        self.sensor_history: dict[str, SensorHistory] = {}
        self._time_slot_size = timedelta(minutes=30)
        self._last_analysis: datetime | None = None

    async def initialize_history(self) -> dict[str, Any]:
        """Perform initial historical analysis."""
        _LOGGER.debug("Starting initial historical analysis")

        start_time = dt_util.utcnow() - timedelta(days=self.history_period)

        # Get historical states for all relevant sensors
        states = await self._get_sensor_history(start_time)

        if not states:
            _LOGGER.warning("No historical state data available")
            return {}

        # Process historical data
        analysis_results = await self._analyze_historical_states(states)
        self._last_analysis = dt_util.utcnow()

        return analysis_results

    async def _get_sensor_history(self, start_time: datetime) -> dict[str, List[Any]]:
        """Retrieve historical states from recorder."""
        all_entities = (
            self.motion_sensors + self.media_devices + self.environmental_sensors
        )

        try:
            # Query history using get_instance for better performance
            recorder = get_instance(self.hass)
            state_history = await recorder.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time,
                dt_util.utcnow(),
                all_entities,
                minimal_response=True,
                no_attributes=True,
            )

            return state_history

        except (HomeAssistantError, ValueError, TypeError) as err:
            _LOGGER.error("Error retrieving historical states: %s", err)
            return {}

    async def _analyze_historical_states(
        self, states: dict[str, List[Any]]
    ) -> dict[str, Any]:
        """Analyze historical states to extract patterns."""
        motion_patterns = await self._analyze_motion_patterns(states)
        occupancy_patterns = await self._analyze_occupancy_patterns(states)
        sensor_correlations = await self._analyze_sensor_correlations(states)

        return {
            "motion_patterns": motion_patterns,
            "occupancy_patterns": occupancy_patterns,
            "sensor_correlations": sensor_correlations,
            "last_analyzed": dt_util.utcnow().isoformat(),
        }

    async def _analyze_motion_patterns(
        self, states: dict[str, List[Any]]
    ) -> dict[str, dict[str, Any]]:
        """Analyze motion sensor activation patterns."""
        patterns = {}

        for sensor_id in self.motion_sensors:
            sensor_states = states.get(sensor_id, [])
            if not sensor_states:
                continue

            time_slots: dict[str, TimeSlot] = {}
            day_patterns: dict[str, TimeSlot] = {}
            total_activations = 0
            total_duration = timedelta()

            current_activation = None

            for state in sensor_states:
                if state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    timestamp = state.last_changed
                    time_slot = self._get_time_slot(timestamp)
                    day_slot = timestamp.strftime("%A").lower()

                    # Initialize slots if needed
                    if time_slot not in time_slots:
                        time_slots[time_slot] = {
                            "total_samples": 0,
                            "active_samples": 0,
                            "average_duration": 0.0,
                            "confidence": 0.0,
                        }
                    if day_slot not in day_patterns:
                        day_patterns[day_slot] = {
                            "total_samples": 0,
                            "active_samples": 0,
                            "average_duration": 0.0,
                            "confidence": 0.0,
                        }

                    # Track activations
                    if state.state == STATE_ON:
                        total_activations += 1
                        current_activation = timestamp
                        time_slots[time_slot]["active_samples"] += 1
                        day_patterns[day_slot]["active_samples"] += 1
                    elif state.state == STATE_OFF and current_activation:
                        duration = timestamp - current_activation
                        total_duration += duration
                        current_activation = None

                    time_slots[time_slot]["total_samples"] += 1
                    day_patterns[day_slot]["total_samples"] += 1

            # Calculate averages and confidence
            for slot_data in [*time_slots.values(), *day_patterns.values()]:
                if slot_data["total_samples"] > 0:
                    slot_data["confidence"] = min(slot_data["total_samples"] / 100, 1.0)
                    if slot_data["active_samples"] > 0:
                        slot_data["average_duration"] = (
                            total_duration.total_seconds() / slot_data["active_samples"]
                        )

            patterns[sensor_id] = {
                "total_activations": total_activations,
                "average_duration": (
                    total_duration.total_seconds() / total_activations
                    if total_activations > 0
                    else 0
                ),
                "time_slots": time_slots,
                "day_patterns": day_patterns,
            }

        return patterns

    async def _analyze_occupancy_patterns(
        self, states: dict[str, List[Any]]
    ) -> dict[str, Any]:
        """Analyze overall occupancy patterns."""
        time_slots: dict[str, dict[str, float]] = defaultdict(
            lambda: {"occupied_ratio": 0.0, "samples": 0}
        )
        day_patterns: dict[str, dict[str, float]] = defaultdict(
            lambda: {"occupied_ratio": 0.0, "samples": 0}
        )

        # Combine motion and media device states to determine occupancy
        for sensor_id in [*self.motion_sensors, *self.media_devices]:
            sensor_states = states.get(sensor_id, [])

            for state in sensor_states:
                if state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    timestamp = state.last_changed
                    time_slot = self._get_time_slot(timestamp)
                    day_slot = timestamp.strftime("%A").lower()

                    is_active = state.state == STATE_ON

                    time_slots[time_slot]["samples"] += 1
                    day_patterns[day_slot]["samples"] += 1

                    if is_active:
                        time_slots[time_slot]["occupied_ratio"] += 1
                        day_patterns[day_slot]["occupied_ratio"] += 1

        # Calculate final ratios
        for patterns in [time_slots, day_patterns]:
            for slot_data in patterns.values():
                if slot_data["samples"] > 0:
                    slot_data["occupied_ratio"] /= slot_data["samples"]

        return {"time_slots": dict(time_slots), "day_patterns": dict(day_patterns)}

    async def _analyze_sensor_correlations(
        self, states: dict[str, List[Any]]
    ) -> dict[str, dict[str, float]]:
        """Analyze correlations between different sensors."""
        correlations: dict[str, dict[str, float]] = {}

        # Calculate correlations between motion sensors and other devices
        for motion_sensor in self.motion_sensors:
            motion_states = states.get(motion_sensor, [])
            if not motion_states:
                continue

            correlations[motion_sensor] = {}

            # Look for correlations with media devices
            for device in self.media_devices:
                device_states = states.get(device, [])
                if not device_states:
                    continue

                correlation = self._calculate_state_correlation(
                    motion_states, device_states
                )
                correlations[motion_sensor][device] = correlation

            # Look for correlations with other motion sensors
            for other_sensor in self.motion_sensors:
                if other_sensor != motion_sensor:
                    other_states = states.get(other_sensor, [])
                    if not other_states:
                        continue

                    correlation = self._calculate_state_correlation(
                        motion_states, other_states
                    )
                    correlations[motion_sensor][other_sensor] = correlation

        return correlations

    def _calculate_state_correlation(
        self, states_1: List[Any], states_2: List[Any]
    ) -> float:
        """Calculate correlation between two sets of states."""
        # Convert states to time series of active periods
        series_1 = self._convert_states_to_timeseries(states_1)
        series_2 = self._convert_states_to_timeseries(states_2)

        if not series_1 or not series_2:
            return 0.0

        # Calculate overlap between active periods
        total_overlap = timedelta()
        total_active_1 = timedelta()
        total_active_2 = timedelta()

        for start_1, end_1 in series_1:
            total_active_1 += end_1 - start_1
            for start_2, end_2 in series_2:
                if start_1 <= end_2 and start_2 <= end_1:
                    overlap_start = max(start_1, start_2)
                    overlap_end = min(end_1, end_2)
                    total_overlap += overlap_end - overlap_start

        for start_2, end_2 in series_2:
            total_active_2 += end_2 - start_2

        # Calculate correlation coefficient
        if total_active_1.total_seconds() == 0 or total_active_2.total_seconds() == 0:
            return 0.0

        correlation = (
            total_overlap.total_seconds()
            * 2
            / (total_active_1.total_seconds() + total_active_2.total_seconds())
        )

        return min(max(correlation, 0.0), 1.0)

    def _convert_states_to_timeseries(
        self, states: List[Any]
    ) -> List[tuple[datetime, datetime]]:
        """Convert state history to list of active time periods."""
        active_periods = []
        current_start = None

        for state in states:
            if state.state == STATE_ON and current_start is None:
                current_start = state.last_changed
            elif state.state == STATE_OFF and current_start is not None:
                active_periods.append((current_start, state.last_changed))
                current_start = None

        # Handle currently active period
        if current_start is not None:
            active_periods.append((current_start, dt_util.utcnow()))

        return active_periods

    def _get_time_slot(self, timestamp: datetime) -> str:
        """Convert timestamp to time slot string."""
        hour = timestamp.hour
        minute = (timestamp.minute // 30) * 30
        return f"{hour:02d}:{minute:02d}"

    async def update_analysis(self) -> dict[str, Any]:
        """Update historical analysis with recent data."""
        if not self._last_analysis:
            return await self.initialize_history()

        # Only analyze data since last analysis
        start_time = self._last_analysis
        current_time = dt_util.utcnow()

        if (current_time - start_time) < timedelta(hours=1):
            _LOGGER.debug("Skipping analysis update - too soon since last update")
            return {}

        # Get recent states
        recent_states = await self._get_sensor_history(start_time)

        if not recent_states:
            return {}

        # Update analysis with new data
        new_analysis = await self._analyze_historical_states(recent_states)
        self._last_analysis = current_time

        return new_analysis
