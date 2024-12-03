"""Historical state analysis for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, TypedDict, List, Dict
from collections import defaultdict

from homeassistant.core import HomeAssistant, State
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    STATE_PLAYING,
)
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

_LOGGER = logging.getLogger(__name__)


class TimeSlot(TypedDict):
    """Time slot statistics."""

    total_samples: int
    active_samples: int
    average_duration: float
    confidence: float


class DayPattern(TypedDict):
    """Day pattern statistics."""

    occupied_ratio: float
    samples: int
    confidence: float


class SensorHistory(TypedDict):
    """Historical data for a sensor."""

    total_activations: int
    average_duration: float
    time_slots: Dict[str, TimeSlot]
    day_patterns: Dict[str, DayPattern]
    correlated_sensors: Dict[str, float]


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
        self._time_slot_size = timedelta(minutes=30)
        self._last_analysis: datetime | None = None

        # Initialize storage
        self.store = Store(
            hass,
            version=1,
            key="area_occupancy_historical_analysis",
            atomic_writes=True,
        )
        self._cached_data: dict[str, Any] = {}

    async def initialize_history(self) -> dict[str, Any]:
        """Perform initial historical analysis."""
        try:
            _LOGGER.debug("Starting initial historical analysis")

            # Load any cached data
            self._cached_data = await self.store.async_load() or {}

            # If no sensors configured, return empty data without error
            if not any(
                [self.motion_sensors, self.media_devices, self.environmental_sensors]
            ):
                _LOGGER.debug("No sensors configured, skipping historical analysis")
                return {}

            # Get historical states
            start_time = dt_util.utcnow() - timedelta(days=self.history_period)
            states = await self._get_sensor_history(start_time)

            # Even if no states returned, don't fail
            if not states:
                _LOGGER.warning("No historical state data available")
                return {}

            # Process historical data
            analysis_results = await self._analyze_historical_states(states)
            self._last_analysis = dt_util.utcnow()

            # Update cache
            self._cached_data.update(analysis_results)
            await self.store.async_save(self._cached_data)

            return analysis_results

        except Exception as err:
            _LOGGER.error("Failed to initialize historical analysis: %s", err)
            raise HomeAssistantError(
                f"Historical analysis initialization failed: {err}"
            ) from err

    async def _get_sensor_history(self, start_time: datetime) -> dict[str, List[State]]:
        """Retrieve historical states from recorder."""
        all_entities = (
            self.motion_sensors + self.media_devices + self.environmental_sensors
        )

        # If no entities are configured, return empty dict instead of failing
        if not all_entities:
            _LOGGER.debug("No entities configured for historical analysis")
            return {}

        try:
            recorder = get_instance(self.hass)
            return await recorder.async_add_executor_job(
                get_significant_states,
                self.hass,
                start_time,
                dt_util.utcnow(),
                all_entities,
            )

        except Exception as err:
            _LOGGER.error("Error retrieving historical states: %s", err)
            raise HomeAssistantError(
                f"Failed to retrieve historical states: {err}"
            ) from err

    async def _analyze_historical_states(
        self, states: dict[str, List[State]]
    ) -> dict[str, Any]:
        """Analyze historical states to extract patterns."""
        try:
            motion_patterns = await self._analyze_motion_patterns(states)
            occupancy_patterns = await self._analyze_occupancy_patterns(states)
            sensor_correlations = await self._analyze_sensor_correlations(states)

            return {
                "motion_patterns": motion_patterns,
                "occupancy_patterns": occupancy_patterns,
                "sensor_correlations": sensor_correlations,
                "last_analyzed": dt_util.utcnow().isoformat(),
            }

        except Exception as err:
            _LOGGER.error("Error analyzing historical states: %s", err)
            raise HomeAssistantError(
                f"Historical state analysis failed: {err}"
            ) from err

    async def _analyze_motion_patterns(
        self, states: dict[str, List[State]]
    ) -> dict[str, dict[str, Any]]:
        """Analyze motion sensor activation patterns."""
        patterns: dict[str, dict[str, Any]] = {}

        for sensor_id in self.motion_sensors:
            sensor_states = states.get(sensor_id, [])
            if not sensor_states:
                continue

            time_slots: dict[str, TimeSlot] = defaultdict(
                lambda: {
                    "total_samples": 0,
                    "active_samples": 0,
                    "average_duration": 0.0,
                    "confidence": 0.0,
                }
            )
            day_patterns: dict[str, DayPattern] = defaultdict(
                lambda: {
                    "occupied_ratio": 0.0,
                    "samples": 0,
                    "confidence": 0.0,
                }
            )

            total_activations = 0
            total_duration = timedelta()
            current_activation = None

            for state in sensor_states:
                if state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    continue

                timestamp = state.last_changed
                time_slot = self._get_time_slot(timestamp)
                day_slot = timestamp.strftime("%A").lower()
                slot_data = time_slots[time_slot]
                day_data = day_patterns[day_slot]

                if state.state == STATE_ON:
                    total_activations += 1
                    current_activation = timestamp
                    slot_data["active_samples"] += 1
                    day_data["samples"] += 1
                elif state.state == STATE_OFF and current_activation:
                    duration = timestamp - current_activation
                    total_duration += duration
                    current_activation = None

                slot_data["total_samples"] += 1
                day_data["samples"] += 1

            # Calculate averages and confidence
            for slot_data in time_slots.values():
                if slot_data["total_samples"] > 0:
                    slot_data["confidence"] = min(slot_data["total_samples"] / 100, 1.0)
                    if slot_data["active_samples"] > 0:
                        slot_data["average_duration"] = (
                            total_duration.total_seconds() / slot_data["active_samples"]
                        )

            # Calculate day pattern ratios
            for day_data in day_patterns.values():
                if day_data["samples"] > 0:
                    day_data["occupied_ratio"] = (
                        total_activations / day_data["samples"]
                        if day_data["samples"] > 0
                        else 0.0
                    )
                    day_data["confidence"] = min(day_data["samples"] / 50, 1.0)

            patterns[sensor_id] = {
                "total_activations": total_activations,
                "average_duration": (
                    total_duration.total_seconds() / total_activations
                    if total_activations > 0
                    else 0
                ),
                "time_slots": dict(time_slots),
                "day_patterns": dict(day_patterns),
            }

        return patterns

    async def _analyze_occupancy_patterns(
        self, states: dict[str, List[State]]
    ) -> dict[str, Any]:
        """Analyze overall occupancy patterns."""
        time_slots: dict[str, dict[str, float]] = defaultdict(
            lambda: {"occupied_ratio": 0.0, "samples": 0}
        )
        day_patterns: dict[str, dict[str, float]] = defaultdict(
            lambda: {"occupied_ratio": 0.0, "samples": 0}
        )

        for sensor_id in [*self.motion_sensors, *self.media_devices]:
            sensor_states = states.get(sensor_id, [])

            for state in sensor_states:
                if state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    continue

                timestamp = state.last_changed
                time_slot = self._get_time_slot(timestamp)
                day_slot = timestamp.strftime("%A").lower()

                is_active = state.state in (STATE_ON, STATE_PLAYING)

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

        return {
            "time_slots": dict(time_slots),
            "day_patterns": dict(day_patterns),
        }

    async def _analyze_sensor_correlations(
        self, states: dict[str, List[State]]
    ) -> dict[str, dict[str, float]]:
        """Analyze correlations between different sensors."""
        correlations: dict[str, dict[str, float]] = {}

        for motion_sensor in self.motion_sensors:
            motion_states = states.get(motion_sensor, [])
            if not motion_states:
                continue

            correlations[motion_sensor] = {}

            # Analyze correlations with media devices
            await self._analyze_device_correlations(
                motion_sensor,
                motion_states,
                self.media_devices,
                states,
                correlations[motion_sensor],
            )

            # Analyze correlations with other motion sensors
            await self._analyze_device_correlations(
                motion_sensor,
                motion_states,
                [s for s in self.motion_sensors if s != motion_sensor],
                states,
                correlations[motion_sensor],
            )

        return correlations

    async def _analyze_device_correlations(
        self,
        source_id: str,
        source_states: List[State],
        target_devices: List[str],
        all_states: dict[str, List[State]],
        correlation_dict: dict[str, float],
    ) -> None:
        """Analyze correlations between a source sensor and target devices."""
        for device_id in target_devices:
            device_states = all_states.get(device_id, [])
            if not device_states:
                continue

            correlation = self._calculate_state_correlation(
                source_states, device_states
            )
            if correlation > 0.1:  # Only store significant correlations
                correlation_dict[device_id] = correlation

    def _calculate_state_correlation(
        self, states_1: List[State], states_2: List[State]
    ) -> float:
        """Calculate correlation between two sets of states."""
        active_periods_1 = self._convert_states_to_timeseries(states_1)
        active_periods_2 = self._convert_states_to_timeseries(states_2)

        if not active_periods_1 or not active_periods_2:
            return 0.0

        total_overlap = timedelta()
        total_active_1 = timedelta()
        total_active_2 = timedelta()

        for start_1, end_1 in active_periods_1:
            total_active_1 += end_1 - start_1
            for start_2, end_2 in active_periods_2:
                if start_1 <= end_2 and start_2 <= end_1:
                    overlap_start = max(start_1, start_2)
                    overlap_end = min(end_1, end_2)
                    total_overlap += overlap_end - overlap_start

        for start_2, end_2 in active_periods_2:
            total_active_2 += end_2 - start_2

        if total_active_1.total_seconds() == 0 or total_active_2.total_seconds() == 0:
            return 0.0

        correlation = (
            total_overlap.total_seconds()
            * 2
            / (total_active_1.total_seconds() + total_active_2.total_seconds())
        )

        return min(max(correlation, 0.0), 1.0)

    def _convert_states_to_timeseries(
        self, states: List[State]
    ) -> List[tuple[datetime, datetime]]:
        """Convert state history to list of active time periods."""
        active_periods: List[tuple[datetime, datetime]] = []
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

        try:
            # Only analyze data since last analysis
            start_time = self._last_analysis
            current_time = dt_util.utcnow()

            if (current_time - start_time) < timedelta(hours=1):
                _LOGGER.debug("Skipping analysis update - too soon")
                return {}

            recent_states = await self._get_sensor_history(start_time)
            if not recent_states:
                return {}

            new_analysis = await self._analyze_historical_states(recent_states)
            self._last_analysis = current_time

            # Update cached data
            self._cached_data.update(new_analysis)
            await self.store.async_save(self._cached_data)

            return new_analysis

        except Exception as err:
            _LOGGER.error("Error updating historical analysis: %s", err)
            raise HomeAssistantError(
                f"Failed to update historical analysis: {err}"
            ) from err
