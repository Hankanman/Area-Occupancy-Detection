# custom_components/area_occupancy/core/calculators/historical.py

"""Historical data analysis for Area Occupancy Detection."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.components.recorder import get_instance, history
from homeassistant.const import STATE_ON, STATE_OFF, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError

from .. import AreaConfig, SensorState

_LOGGER = logging.getLogger(__name__)


class HistoricalAnalyzer:
    """Analyzes historical state data for occupancy patterns."""

    def __init__(
        self,
        hass: HomeAssistant,
        config: AreaConfig,
        time_window_days: int = 7,
        min_samples: int = 100,
        min_confidence: float = 0.3,
    ) -> None:
        """Initialize the analyzer."""
        self.hass = hass
        self.config = config
        self.time_window_days = time_window_days
        self.min_samples = min_samples
        self.min_confidence = min_confidence

        self._cache: dict[str, tuple[float, datetime]] = {}
        self._cache_duration = timedelta(minutes=30)
        self._last_purge = datetime.utcnow()
        self._last_analysis: Optional[datetime] = None
        self._historical_data: dict[str, Any] = {}
        self._correlation_threshold = 0.7
        self._change_threshold = 0.2

        # Lock for thread-safe access to shared state
        self._lock = asyncio.Lock()

    async def async_initialize(self) -> None:
        """Perform initial historical analysis."""
        async with self._lock:
            try:
                _LOGGER.debug(
                    "Initializing historical analysis for %s", self.config.name
                )
                start_time = dt_util.utcnow() - timedelta(days=self.time_window_days)
                states = await self._get_historical_states(start_time)

                if not states:
                    _LOGGER.warning(
                        "No historical states found for %s", self.config.name
                    )
                    return

                self._historical_data = await self._analyze_historical_states(states)
                self._last_analysis = dt_util.utcnow()

            except (HomeAssistantError, ValueError, TypeError) as err:
                _LOGGER.error(
                    "Failed to initialize historical analysis for %s: %s",
                    self.config.name,
                    err,
                    exc_info=True,
                )
                raise

    def _purge_stale_cache(self) -> None:
        """Purge stale entries from the cache."""
        now = datetime.utcnow()
        if now - self._last_purge < self._cache_duration:
            return

        self._cache = {
            key: value
            for key, value in self._cache.items()
            if value[1] > now - self._cache_duration
        }
        self._last_purge = now

    async def calculate(
        self, sensor_states: dict[str, SensorState], historical_patterns: dict[str, Any]
    ) -> float:
        """Calculate probability adjustment based on historical data."""
        self._purge_stale_cache()
        area_id = self.config.name.lower().replace(" ", "_")

        async with self._lock:
            if self._is_cache_valid(area_id):
                return self._cache[area_id][0]

            try:
                time_based_prob = self._calculate_time_based_probability(
                    historical_patterns
                )
                correlation_prob = self._calculate_correlation_probability(
                    sensor_states, historical_patterns
                )
                environmental_prob = self._calculate_environmental_probability(
                    sensor_states
                )

                weights = {
                    "time": 0.4,
                    "correlation": 0.4,
                    "environmental": 0.2,
                }

                final_prob = (
                    time_based_prob * weights["time"]
                    + correlation_prob * weights["correlation"]
                    + environmental_prob * weights["environmental"]
                )

                self._update_cache(area_id, final_prob)
                return final_prob

            except (HomeAssistantError, ValueError, TypeError) as err:
                _LOGGER.error(
                    "Error calculating historical probability for %s: %s",
                    self.config.name,
                    err,
                    exc_info=True,
                )
                return 0.0

    async def _get_historical_states(
        self, start_time: datetime
    ) -> dict[str, List[Any]]:
        """Retrieve historical states from recorder."""
        try:
            # Get all relevant entity IDs
            entity_ids = (
                self.config.motion_sensors
                + self.config.media_devices
                + self.config.appliances
                + self.config.illuminance_sensors
                + self.config.humidity_sensors
                + self.config.temperature_sensors
            )

            # Query history using recorder instance
            recorder = get_instance(self.hass)
            return await recorder.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time,
                dt_util.utcnow(),
                entity_ids,
            )

        except (HomeAssistantError, ValueError, TypeError) as err:
            _LOGGER.error("Failed to retrieve historical states: %s", err)
            raise HomeAssistantError("Failed to retrieve historical states") from err

    async def _analyze_historical_states(
        self, states: dict[str, List[Any]]
    ) -> dict[str, Any]:
        """Analyze historical states to extract patterns."""
        try:
            results = {
                "time_patterns": await self._analyze_time_patterns(states),
                "sensor_correlations": self._analyze_sensor_correlations(states),
                "environmental_baselines": self._analyze_environmental_baselines(
                    states
                ),
                "last_analyzed": dt_util.utcnow().isoformat(),
            }

            _LOGGER.debug(
                "Historical analysis completed for %s: %s patterns found",
                self.config.name,
                len(results["time_patterns"]),
            )

            return results

        except (HomeAssistantError, ValueError, TypeError) as err:
            _LOGGER.error("Error analyzing historical states: %s", err)
            return {}

    async def _analyze_time_patterns(
        self, states: dict[str, List[Any]]
    ) -> dict[str, dict[str, float]]:
        """Analyze time-based occupancy patterns."""
        patterns: Dict[str, Dict[str, Any]] = {
            "time_slots": defaultdict(lambda: {"occupied": 0, "total": 0}),
            "days": defaultdict(lambda: {"occupied": 0, "total": 0}),
        }

        for entity_id in self.config.motion_sensors:
            if entity_id not in states:
                continue

            for state in states[entity_id]:
                if state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    timestamp = dt_util.as_local(state.last_changed)
                    time_slot = self._get_time_slot(timestamp)
                    day_name = timestamp.strftime("%A").lower()

                    # Update time slot stats
                    patterns["time_slots"][time_slot]["total"] += 1
                    if state.state == STATE_ON:
                        patterns["time_slots"][time_slot]["occupied"] += 1

                    # Update day stats
                    patterns["days"][day_name]["total"] += 1
                    if state.state == STATE_ON:
                        patterns["days"][day_name]["occupied"] += 1

        # Calculate probabilities
        return {
            "time_slots": {
                slot: {
                    "probability": stats["occupied"] / stats["total"],
                    "confidence": min(stats["total"] / self.min_samples, 1.0),
                }
                for slot, stats in patterns["time_slots"].items()
                if stats["total"] > 0
            },
            "days": {
                day: {
                    "probability": stats["occupied"] / stats["total"],
                    "confidence": min(stats["total"] / self.min_samples, 1.0),
                }
                for day, stats in patterns["days"].items()
                if stats["total"] > 0
            },
        }

    def _analyze_sensor_correlations(
        self, states: dict[str, List[Any]]
    ) -> dict[str, dict[str, float]]:
        """Analyze correlations between sensors."""
        correlations: Dict[str, Dict[str, float]] = {}

        # Analyze motion sensor correlations
        for motion_sensor in self.config.motion_sensors:
            if motion_sensor not in states:
                continue

            correlations[motion_sensor] = {}

            # Calculate correlations with other sensors
            for other_sensor in (
                self.config.motion_sensors
                + self.config.media_devices
                + self.config.appliances
            ):
                if other_sensor == motion_sensor or other_sensor not in states:
                    continue

                correlation = self._calculate_state_correlation(
                    states[motion_sensor],
                    states[other_sensor],
                )
                if correlation > self._correlation_threshold:
                    correlations[motion_sensor][other_sensor] = correlation

        return correlations

    def _analyze_environmental_baselines(
        self, states: dict[str, List[Any]]
    ) -> dict[str, dict[str, float]]:
        """Calculate environmental sensor baselines."""
        baselines = {
            "illuminance": self._calculate_sensor_baseline(
                states, self.config.illuminance_sensors
            ),
            "temperature": self._calculate_sensor_baseline(
                states, self.config.temperature_sensors
            ),
            "humidity": self._calculate_sensor_baseline(
                states, self.config.humidity_sensors
            ),
        }

        return {key: value for key, value in baselines.items() if value["count"] > 0}

    def _calculate_sensor_baseline(
        self, states: dict[str, List[Any]], sensor_list: list[str]
    ) -> dict[str, float]:
        """Calculate baseline statistics for a sensor type."""
        values = []
        for sensor_id in sensor_list:
            if sensor_id not in states:
                continue

            for state in states[sensor_id]:
                try:
                    value = float(state.state)
                    values.append(value)
                except (ValueError, TypeError):
                    continue

        if not values:
            return {"average": 0.0, "std_dev": 0.0, "count": 0}

        return {
            "average": float(np.mean(values)),
            "std_dev": float(np.std(values)),
            "count": len(values),
        }

    def _calculate_state_correlation(
        self, states_1: List[Any], states_2: List[Any]
    ) -> float:
        """Calculate correlation coefficient between two sets of states."""
        if not states_1 or not states_2:
            return 0.0

        # Convert states to time series
        series_1 = self._convert_to_timeseries(states_1)
        series_2 = self._convert_to_timeseries(states_2)

        # Calculate overlap
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

        return min(
            (total_overlap.total_seconds() * 2)
            / (total_active_1.total_seconds() + total_active_2.total_seconds()),
            1.0,
        )

    def _convert_to_timeseries(
        self, states: List[Any]
    ) -> List[Tuple[datetime, datetime]]:
        """Convert state history to time periods."""
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

    def _calculate_time_based_probability(
        self, historical_patterns: dict[str, Any]
    ) -> float:
        """Calculate probability based on time patterns."""
        if not historical_patterns:
            return 0.0

        current_time = dt_util.as_local(dt_util.utcnow())
        time_slot = self._get_time_slot(current_time)
        day_name = current_time.strftime("%A").lower()

        time_slots = historical_patterns.get("time_slots", {})
        days = historical_patterns.get("days", {})

        # Get probabilities and confidences
        time_prob = time_slots.get(time_slot, {}).get("probability", 0.0)
        time_conf = time_slots.get(time_slot, {}).get("confidence", 0.0)
        day_prob = days.get(day_name, {}).get("probability", 0.0)
        day_conf = days.get(day_name, {}).get("confidence", 0.0)

        # Combine probabilities based on confidence
        if time_conf >= self.min_confidence and day_conf >= self.min_confidence:
            return (time_prob * time_conf + day_prob * day_conf) / (
                time_conf + day_conf
            )
        elif time_conf >= self.min_confidence:
            return time_prob
        elif day_conf >= self.min_confidence:
            return day_prob

        return 0.0

    def _calculate_correlation_probability(
        self,
        sensor_states: dict[str, SensorState],
        historical_patterns: dict[str, Any],
    ) -> float:
        """Calculate probability based on sensor correlations."""
        if not historical_patterns:
            return 0.0

        correlations = historical_patterns.get("sensor_correlations", {})
        active_sensors = set()

        # Find currently active sensors
        for entity_id, state in sensor_states.items():
            if state.state == STATE_ON:
                active_sensors.add(entity_id)

        total_correlation = 0.0
        correlation_count = 0

        # Calculate probability based on correlated sensors
        for sensor in active_sensors:
            if sensor in correlations:
                for correlated_sensor, correlation in correlations[sensor].items():
                    if correlated_sensor in active_sensors:
                        total_correlation += correlation
                        correlation_count += 1

        if correlation_count == 0:
            return 0.0

        return min(total_correlation / correlation_count, 1.0)

    def _calculate_environmental_probability(
        self, sensor_states: dict[str, SensorState]
    ) -> float:
        """Calculate probability based on environmental changes."""
        if not self._historical_data:
            return 0.0

        baselines = self._historical_data.get("environmental_baselines", {})
        probabilities = []

        # Check each sensor type
        for sensor_type in ["illuminance", "temperature", "humidity"]:
            baseline = baselines.get(sensor_type, {})
            if not baseline or baseline["count"] == 0:
                continue

            sensors = getattr(self.config, f"{sensor_type}_sensors", [])

            significant_changes = 0
            valid_sensors = 0

            for sensor_id in sensors:
                if sensor_id not in sensor_states:
                    continue

                state = sensor_states[sensor_id]
                if not state.available:
                    continue

                try:
                    current_value = float(state.state)
                    valid_sensors += 1

                    # Check for significant deviation from baseline
                    deviation = abs(current_value - baseline["average"])
                    if deviation > (baseline["std_dev"] * 2):  # 2 standard deviations
                        significant_changes += 1

                except (ValueError, TypeError):
                    continue

            if valid_sensors > 0:
                probabilities.append(significant_changes / valid_sensors)

        # Combine environmental probabilities
        return sum(probabilities) / len(probabilities) if probabilities else 0.0

    def _get_time_slot(self, timestamp: datetime) -> str:
        """Convert timestamp to time slot string."""
        minutes = timestamp.hour * 60 + timestamp.minute
        slot = (minutes // 30) * 30  # 30-minute slots
        return f"{slot // 60:02d}:{slot % 60:02d}"

    def _is_cache_valid(self, area_id: str) -> bool:
        """Check if cached probability is still valid."""
        if area_id not in self._cache:
            return False

        cache_time = self._cache[area_id][1]
        return dt_util.utcnow() - cache_time < self._cache_duration

    def _update_cache(self, area_id: str, probability: float) -> None:
        """Update probability cache."""
        self._cache[area_id] = (probability, dt_util.utcnow())

    async def async_update(self) -> None:
        """Update historical analysis with recent data."""
        now = dt_util.utcnow()

        async with self._lock:
            if self._last_analysis and now - self._last_analysis < timedelta(hours=1):
                return

            try:
                start_time = self._last_analysis or now - timedelta(
                    days=self.time_window_days
                )
                states = await self._get_historical_states(start_time)

                if not states:
                    return

                new_data = await self._analyze_historical_states(states)
                self._update_historical_data(new_data)
                self._last_analysis = now
                self._cache.clear()

            except (HomeAssistantError, ValueError, TypeError) as err:
                _LOGGER.error(
                    "Failed to update historical analysis for %s: %s",
                    self.config.name,
                    err,
                    exc_info=True,
                )

    def _update_historical_data(self, new_data: dict[str, Any]) -> None:
        """Update historical data with new analysis results."""
        if not self._historical_data:
            self._historical_data = new_data
            return

        # Update time patterns
        if "time_patterns" in new_data:
            existing_patterns = self._historical_data.get("time_patterns", {})
            for key in ["time_slots", "days"]:
                if key in new_data["time_patterns"]:
                    existing = existing_patterns.get(key, {})
                    for slot, data in new_data["time_patterns"][key].items():
                        if slot in existing:
                            # Weighted average with more weight to newer data
                            existing[slot] = {
                                "probability": (
                                    existing[slot]["probability"] * 0.3
                                    + data["probability"] * 0.7
                                ),
                                "confidence": min(
                                    existing[slot]["confidence"] + data["confidence"],
                                    1.0,
                                ),
                            }
                        else:
                            existing[slot] = data
                    existing_patterns[key] = existing
            self._historical_data["time_patterns"] = existing_patterns

        # Update sensor correlations
        if "sensor_correlations" in new_data:
            existing_correlations = self._historical_data.get("sensor_correlations", {})
            for sensor, correlations in new_data["sensor_correlations"].items():
                if sensor in existing_correlations:
                    for other_sensor, correlation in correlations.items():
                        existing_correlations[sensor][other_sensor] = (
                            existing_correlations[sensor].get(other_sensor, 0) * 0.3
                            + correlation * 0.7
                        )
                else:
                    existing_correlations[sensor] = correlations
            self._historical_data["sensor_correlations"] = existing_correlations

        # Update environmental baselines
        if "environmental_baselines" in new_data:
            existing_baselines = self._historical_data.get(
                "environmental_baselines", {}
            )
            for sensor_type, baseline in new_data["environmental_baselines"].items():
                if sensor_type in existing_baselines:
                    # Weighted average of baselines
                    existing_baselines[sensor_type] = {
                        "average": (
                            existing_baselines[sensor_type]["average"] * 0.7
                            + baseline["average"] * 0.3
                        ),
                        "std_dev": (
                            existing_baselines[sensor_type]["std_dev"] * 0.7
                            + baseline["std_dev"] * 0.3
                        ),
                        "count": baseline["count"],
                    }
                else:
                    existing_baselines[sensor_type] = baseline
            self._historical_data["environmental_baselines"] = existing_baselines

        # Update timestamp
        self._historical_data["last_analyzed"] = dt_util.utcnow().isoformat()

    def get_historical_patterns(self) -> dict[str, Any]:
        """Get current historical patterns."""
        return self._historical_data.copy() if self._historical_data else {}

    async def async_reset(self) -> None:
        """Reset historical analysis data."""
        async with self._lock:
            self._historical_data = {}
            self._cache.clear()
            self._last_analysis = None
            await self.async_initialize()
