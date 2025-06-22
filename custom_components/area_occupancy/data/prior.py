"""Area baseline prior calculations for Area Occupancy Detection."""

from datetime import datetime, timedelta
import logging
import statistics
from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.const import STATE_ON
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import DEFAULT_PRIOR, MAX_PROBABILITY, MIN_PROBABILITY
from ..utils import TimeInterval, get_states_from_recorder, states_to_intervals

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


def _calculate_overlap_duration(
    interval: TimeInterval, start: datetime, end: datetime
) -> float:
    """Calculate overlap duration between an interval and a time window in seconds.

    Args:
        interval: The interval to check for overlap
        start: Start of the time window
        end: End of the time window

    Returns:
        Overlap duration in seconds (0 if no overlap)
    """
    # Find the overlap period
    overlap_start = max(interval["start"], start)
    overlap_end = min(interval["end"], end)

    # Return overlap duration (0 if no overlap)
    if overlap_start < overlap_end:
        return (overlap_end - overlap_start).total_seconds()
    return 0.0


class Prior:
    """Manages area baseline prior calculations."""

    def __init__(self, coordinator: "AreaOccupancyCoordinator") -> None:
        """Initialize the prior manager."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self._last_updated: datetime | None = None
        self._area_baseline_prior: float | None = None
        self._method_used: str | None = None

    async def __post_init__(self) -> None:
        """Post init - initialize area baseline prior."""
        if self.coordinator.config.history.enabled:
            await self.calculate_area_baseline_prior()

    def time_active(self, history_period: int | None = None) -> float:
        """Calculate the time the area was active in the given history period."""
        effective_period = history_period or self.coordinator.config.history.period
        start_time = dt_util.utcnow() - timedelta(days=effective_period)
        end_time = dt_util.utcnow()
        total_seconds = (end_time - start_time).total_seconds()

        # Calculate expected active time based on baseline prior
        # This gives an estimate of how much time the area should be occupied
        # based on historical patterns
        expected_active_seconds = total_seconds * self.area_baseline_prior

        _LOGGER.debug(
            "Time active calculation: %.1f hours out of %.1f hours (%.1f%% baseline prior)",
            expected_active_seconds / 3600,
            total_seconds / 3600,
            self.area_baseline_prior * 100,
        )

        return expected_active_seconds

    async def calculate_area_baseline_prior(
        self, history_period: int | None = None
    ) -> float:
        """Calculate the area's baseline occupancy rate using robust multi-method approach.

        This is the true Bayesian prior - P(area occupied) before considering current evidence.
        Uses a comprehensive fallback strategy to handle various sensor configurations.
        """
        effective_period = history_period or self.coordinator.config.history.period
        start_time = dt_util.utcnow() - timedelta(days=effective_period)
        end_time = dt_util.utcnow()

        _LOGGER.debug(
            "Calculating area baseline prior over %d days using fallback strategy",
            effective_period,
        )

        # Check if history analysis is disabled
        if not self.coordinator.config.history.enabled:
            self._area_baseline_prior = DEFAULT_PRIOR
            self._last_updated = dt_util.utcnow()
            self._method_used = "disabled"
            return DEFAULT_PRIOR

        # Check if we have a cached prior that's still valid
        if (
            self._area_baseline_prior is not None
            and self._last_updated is not None
            and self._last_updated > start_time
        ):
            _LOGGER.debug(
                "Using cached area baseline prior: %.3f (method: %s)",
                self._area_baseline_prior,
                self._method_used,
            )
            return self._area_baseline_prior

        try:
            # Gather all available sensors
            motion_sensors = self._get_all_motion_sensors()
            media_sensors = self.coordinator.config.sensors.media
            appliance_sensors = self.coordinator.config.sensors.appliances
            other_sensors = (
                self.coordinator.config.sensors.doors
                + self.coordinator.config.sensors.windows
                + self.coordinator.config.sensors.lights
                + self.coordinator.config.sensors.illuminance
                + self.coordinator.config.sensors.humidity
                + self.coordinator.config.sensors.temperature
            )

            total_sensors = (
                len(motion_sensors)
                + len(media_sensors)
                + len(appliance_sensors)
                + len(other_sensors)
            )

            _LOGGER.debug(
                "Available sensors - Motion: %d, Media: %d, Appliance: %d, Other: %d, Total: %d",
                len(motion_sensors),
                len(media_sensors),
                len(appliance_sensors),
                len(other_sensors),
                total_sensors,
            )

            # Apply fallback strategy
            baseline_prior = None
            method_used = None

            # Method 1: Multi-Motion Sensor Consensus (≥2 motion sensors)
            if len(motion_sensors) >= 2:
                baseline_prior = await self._multi_motion_consensus(
                    motion_sensors, start_time, end_time
                )
                method_used = "multi_motion_consensus"
                _LOGGER.debug("Trying multi-motion sensor consensus")

            # Method 2: Confidence-Weighted Sensor Fusion (multiple sensor types)
            elif self._has_multiple_sensor_types(
                motion_sensors, media_sensors, appliance_sensors
            ):
                baseline_prior = await self._confidence_weighted_fusion(
                    motion_sensors,
                    media_sensors,
                    appliance_sensors,
                    start_time,
                    end_time,
                )
                method_used = "confidence_weighted_fusion"
                _LOGGER.debug("Trying confidence-weighted sensor fusion")

            # Method 3: Time Pattern Analysis (≥2 total sensors)
            elif total_sensors >= 2:
                all_sensors = (
                    motion_sensors + media_sensors + appliance_sensors + other_sensors
                )
                baseline_prior = await self._time_pattern_analysis(
                    all_sensors, start_time, end_time
                )
                method_used = "time_pattern_analysis"
                _LOGGER.debug("Trying time pattern analysis")

            # Method 4: Primary Sensor + Margin (single sensor fallback)
            elif self.coordinator.config.sensors.primary_occupancy:
                baseline_prior = await self._primary_sensor_with_margin(
                    self.coordinator.config.sensors.primary_occupancy,
                    start_time,
                    end_time,
                )
                method_used = "primary_with_margin"
                _LOGGER.debug("Falling back to primary sensor with margin")

            # Final fallback to default
            if baseline_prior is None:
                _LOGGER.warning(
                    "No viable method found for baseline prior calculation, using default"
                )
                baseline_prior = DEFAULT_PRIOR
                method_used = "default_fallback"

            # Debug logging for suspicious values
            if baseline_prior is not None:
                _LOGGER.debug(
                    "Raw baseline prior calculation: %.6f using method: %s",
                    baseline_prior,
                    method_used,
                )

                # Check for impossible values
                if baseline_prior > 0.75:
                    _LOGGER.warning(
                        "Suspiciously high baseline prior %.6f from method %s. "
                        "This suggests limited historical data or overly permissive thresholds. "
                        "Period: %d days, sensors: motion=%d, media=%d, appliance=%d, other=%d",
                        baseline_prior,
                        method_used,
                        effective_period,
                        len(motion_sensors),
                        len(media_sensors),
                        len(appliance_sensors),
                        len(other_sensors),
                    )
                    # Cap at a more reasonable maximum for baseline prior
                    baseline_prior = min(baseline_prior, 0.75)
                    _LOGGER.info(
                        "Capped baseline prior to %.3f to prevent saturation",
                        baseline_prior,
                    )

                elif baseline_prior < 0.01:
                    _LOGGER.warning(
                        "Suspiciously low baseline prior %.6f from method %s. "
                        "This suggests very limited activity or sensor issues.",
                        baseline_prior,
                        method_used,
                    )
                    # Ensure minimum reasonable baseline
                    baseline_prior = max(baseline_prior, 0.05)
                    _LOGGER.info(
                        "Raised baseline prior to %.3f to prevent underflow",
                        baseline_prior,
                    )

            # Apply bounds and cache results
            baseline_prior = max(MIN_PROBABILITY, min(baseline_prior, MAX_PROBABILITY))
            self._area_baseline_prior = baseline_prior
            self._last_updated = dt_util.utcnow()
            self._method_used = method_used

            _LOGGER.info(
                "Calculated area baseline prior: %.3f using method: %s",
                baseline_prior,
                method_used,
            )

            return baseline_prior

        except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
            _LOGGER.warning(
                "Could not calculate area baseline prior: %s. Using default", err
            )
            self._area_baseline_prior = DEFAULT_PRIOR
            self._last_updated = dt_util.utcnow()
            self._method_used = "error_fallback"
            return DEFAULT_PRIOR

    def _get_all_motion_sensors(self) -> list[str]:
        """Get all motion sensors including primary and wasp sensors."""
        motion_sensors = self.coordinator.config.sensors.motion.copy()

        # Add primary occupancy if it's not already in motion sensors
        primary = self.coordinator.config.sensors.primary_occupancy
        if primary and primary not in motion_sensors:
            motion_sensors.append(primary)

        # Add wasp sensor if enabled
        if (
            self.coordinator.config.wasp_in_box.enabled
            and self.coordinator.wasp_entity_id
            and self.coordinator.wasp_entity_id not in motion_sensors
        ):
            motion_sensors.append(self.coordinator.wasp_entity_id)

        return motion_sensors

    def _has_multiple_sensor_types(
        self,
        motion_sensors: list[str],
        media_sensors: list[str],
        appliance_sensors: list[str],
    ) -> bool:
        """Check if we have multiple sensor types for fusion method."""
        sensor_types = 0
        if motion_sensors:
            sensor_types += 1
        if media_sensors:
            sensor_types += 1
        if appliance_sensors:
            sensor_types += 1
        return sensor_types >= 2

    async def _multi_motion_consensus(
        self, motion_sensors: list[str], start_time: datetime, end_time: datetime
    ) -> float | None:
        """Method 1: Multi-Motion Sensor Consensus with 60% threshold."""
        try:
            _LOGGER.debug(
                "Starting multi-motion consensus with %d sensors", len(motion_sensors)
            )

            # Get states for all motion sensors
            all_sensor_intervals = {}
            for sensor in motion_sensors:
                states = await get_states_from_recorder(
                    self.coordinator.hass, sensor, start_time, end_time
                )
                if states:
                    state_objects = [s for s in states if isinstance(s, State)]
                    if state_objects:
                        intervals = await states_to_intervals(
                            state_objects, start_time, end_time
                        )
                        all_sensor_intervals[sensor] = intervals

            if len(all_sensor_intervals) < 2:
                _LOGGER.debug("Insufficient motion sensor data for consensus")
                return None

            # Create time buckets (5-minute intervals) for proportional analysis
            bucket_size = timedelta(minutes=5)
            current_time = start_time
            consensus_periods = []

            while current_time < end_time:
                bucket_end = min(current_time + bucket_size, end_time)
                bucket_duration = (bucket_end - current_time).total_seconds()

                # Calculate weighted active time for each sensor in this bucket
                sensor_active_times = {}
                for sensor, intervals in all_sensor_intervals.items():
                    active_time = 0.0
                    for interval in intervals:
                        if interval["state"] == STATE_ON:
                            overlap = _calculate_overlap_duration(
                                interval, current_time, bucket_end
                            )
                            active_time += overlap

                    # Convert to proportion of bucket duration
                    sensor_active_times[sensor] = (
                        active_time / bucket_duration if bucket_duration > 0 else 0.0
                    )

                # Calculate consensus based on average sensor activity (60% threshold)
                avg_activity = (
                    sum(sensor_active_times.values()) / len(sensor_active_times)
                    if sensor_active_times
                    else 0.0
                )
                is_occupied = avg_activity >= 0.6

                consensus_periods.append(
                    {
                        "start": current_time,
                        "end": bucket_end,
                        "occupied": is_occupied,
                        "consensus": avg_activity,
                    }
                )

                current_time = bucket_end

            # Calculate occupancy rate from consensus periods
            occupied_time = sum(
                (period["end"] - period["start"]).total_seconds()
                for period in consensus_periods
                if period["occupied"]
            )
            total_time = sum(
                (period["end"] - period["start"]).total_seconds()
                for period in consensus_periods
            )

            if total_time == 0:
                return None

            occupancy_rate = occupied_time / total_time

            # Debug logging for suspicious values
            occupied_periods = sum(1 for p in consensus_periods if p["occupied"])
            _LOGGER.debug(
                "Multi-motion consensus: %.3f occupancy rate from %d periods "
                "(occupied: %d, total: %d, occupied_time: %.1f hours, total_time: %.1f hours)",
                occupancy_rate,
                len(consensus_periods),
                occupied_periods,
                len(consensus_periods),
                occupied_time / 3600,
                total_time / 3600,
            )

            if occupancy_rate > 0.75:
                _LOGGER.warning(
                    "Multi-motion consensus produced suspiciously high rate %.3f. "
                    "Occupied periods: %d/%d, this may indicate limited data or sensor issues.",
                    occupancy_rate,
                    occupied_periods,
                    len(consensus_periods),
                )

            return occupancy_rate

        except Exception as err:
            _LOGGER.warning("Multi-motion consensus failed: %s", err)
            return None

    async def _confidence_weighted_fusion(
        self,
        motion_sensors: list[str],
        media_sensors: list[str],
        appliance_sensors: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> float | None:
        """Method 2: Confidence-Weighted Sensor Fusion."""
        try:
            _LOGGER.debug("Starting confidence-weighted fusion")

            # Define sensor type weights
            type_weights = {"motion": 0.7, "media": 0.2, "appliance": 0.1}

            # Gather intervals for each sensor type
            type_intervals = {}

            # Motion sensors
            if motion_sensors:
                motion_intervals = []
                for sensor in motion_sensors:
                    states = await get_states_from_recorder(
                        self.coordinator.hass, sensor, start_time, end_time
                    )
                    if states:
                        state_objects = [s for s in states if isinstance(s, State)]
                        if state_objects:
                            intervals = await states_to_intervals(
                                state_objects, start_time, end_time
                            )
                            motion_intervals.extend(intervals)
                if motion_intervals:
                    type_intervals["motion"] = motion_intervals

            # Media sensors
            if media_sensors:
                media_intervals = []
                for sensor in media_sensors:
                    states = await get_states_from_recorder(
                        self.coordinator.hass, sensor, start_time, end_time
                    )
                    if states:
                        state_objects = [s for s in states if isinstance(s, State)]
                        if state_objects:
                            intervals = await states_to_intervals(
                                state_objects, start_time, end_time
                            )
                            media_intervals.extend(intervals)
                if media_intervals:
                    type_intervals["media"] = media_intervals

            # Appliance sensors
            if appliance_sensors:
                appliance_intervals = []
                for sensor in appliance_sensors:
                    states = await get_states_from_recorder(
                        self.coordinator.hass, sensor, start_time, end_time
                    )
                    if states:
                        state_objects = [s for s in states if isinstance(s, State)]
                        if state_objects:
                            intervals = await states_to_intervals(
                                state_objects, start_time, end_time
                            )
                            appliance_intervals.extend(intervals)
                if appliance_intervals:
                    type_intervals["appliance"] = appliance_intervals

            if len(type_intervals) < 2:
                _LOGGER.debug("Insufficient sensor types for weighted fusion")
                return None

            # Create time buckets and calculate weighted confidence using proportional overlap
            bucket_size = timedelta(minutes=10)
            current_time = start_time
            confidence_periods = []

            while current_time < end_time:
                bucket_end = min(current_time + bucket_size, end_time)
                bucket_duration = (bucket_end - current_time).total_seconds()
                weighted_score = 0.0
                total_weight = 0.0

                # Calculate weighted activity score for this time bucket using proportional overlap
                for sensor_type, intervals in type_intervals.items():
                    # Calculate total active time for this sensor type in this bucket
                    type_active_time = 0.0
                    for interval in intervals:
                        if interval["state"] == STATE_ON:
                            overlap = _calculate_overlap_duration(
                                interval, current_time, bucket_end
                            )
                            type_active_time += overlap

                    # Convert to proportion of bucket duration
                    type_activity_ratio = (
                        type_active_time / bucket_duration
                        if bucket_duration > 0
                        else 0.0
                    )

                    # Weight the activity ratio by sensor type weight
                    weighted_score += type_weights[sensor_type] * type_activity_ratio
                    total_weight += type_weights[sensor_type]

                # Calculate confidence (0.0 to 1.0)
                confidence = weighted_score / total_weight if total_weight > 0 else 0.0

                # Only use high-confidence periods (>= 0.5)
                is_occupied = confidence >= 0.5

                confidence_periods.append(
                    {
                        "start": current_time,
                        "end": bucket_end,
                        "occupied": is_occupied,
                        "confidence": confidence,
                    }
                )

                current_time = bucket_end

            # Calculate occupancy rate from high-confidence periods only
            high_confidence_periods = [
                p for p in confidence_periods if p["confidence"] >= 0.3
            ]

            if not high_confidence_periods:
                _LOGGER.debug("No high-confidence periods found")
                return None

            occupied_time = sum(
                (period["end"] - period["start"]).total_seconds()
                for period in high_confidence_periods
                if period["occupied"]
            )
            total_time = sum(
                (period["end"] - period["start"]).total_seconds()
                for period in high_confidence_periods
            )

            if total_time == 0:
                return None

            occupancy_rate = occupied_time / total_time

            # Debug logging for suspicious values
            occupied_periods = sum(1 for p in high_confidence_periods if p["occupied"])
            _LOGGER.debug(
                "Confidence-weighted fusion: %.3f occupancy rate from %d high-confidence periods "
                "(occupied: %d, total: %d, occupied_time: %.1f hours, total_time: %.1f hours)",
                occupancy_rate,
                len(high_confidence_periods),
                occupied_periods,
                len(high_confidence_periods),
                occupied_time / 3600,
                total_time / 3600,
            )

            if occupancy_rate > 0.75:
                _LOGGER.warning(
                    "Confidence-weighted fusion produced suspiciously high rate %.3f. "
                    "Occupied periods: %d/%d, this may indicate limited data or sensor issues.",
                    occupancy_rate,
                    occupied_periods,
                    len(high_confidence_periods),
                )

            return occupancy_rate

        except Exception as err:
            _LOGGER.warning("Confidence-weighted fusion failed: %s", err)
            return None

    async def _time_pattern_analysis(
        self, all_sensors: list[str], start_time: datetime, end_time: datetime
    ) -> float | None:
        """Method 3: Time Pattern Analysis across all sensors."""
        try:
            _LOGGER.debug(
                "Starting time pattern analysis with %d sensors", len(all_sensors)
            )

            # Collect all sensor activity intervals
            all_intervals = []
            for sensor in all_sensors:
                states = await get_states_from_recorder(
                    self.coordinator.hass, sensor, start_time, end_time
                )
                if states:
                    state_objects = [s for s in states if isinstance(s, State)]
                    if state_objects:
                        intervals = await states_to_intervals(
                            state_objects, start_time, end_time
                        )
                        all_intervals.extend(intervals)

            if not all_intervals:
                _LOGGER.debug("No sensor activity found for pattern analysis")
                return None

            # Create larger time buckets for pattern analysis (15 minutes) using proportional overlap
            bucket_size = timedelta(minutes=15)
            current_time = start_time
            activity_periods = []

            while current_time < end_time:
                bucket_end = min(current_time + bucket_size, end_time)
                bucket_duration = (bucket_end - current_time).total_seconds()

                # Calculate total active duration for all sensors in this time bucket
                total_active_time = 0.0
                for interval in all_intervals:
                    if interval["state"] == STATE_ON:
                        overlap = _calculate_overlap_duration(
                            interval, current_time, bucket_end
                        )
                        total_active_time += overlap

                # Calculate activity ratio as proportion of total possible active time
                # Total possible = bucket_duration * number_of_sensors
                max_possible_active_time = (
                    bucket_duration * len(all_sensors) if all_sensors else 1.0
                )
                activity_ratio = (
                    total_active_time / max_possible_active_time
                    if max_possible_active_time > 0
                    else 0.0
                )

                # Use adaptive threshold based on total sensors
                if len(all_sensors) >= 5:
                    threshold = 0.15  # 15% for many sensors (reduced from 30% to account for proportional calculation)
                elif len(all_sensors) >= 3:
                    threshold = 0.20  # 20% for moderate sensors (reduced from 40%)
                else:
                    threshold = 0.25  # 25% for few sensors (reduced from 50%)

                is_occupied = activity_ratio >= threshold

                activity_periods.append(
                    {
                        "start": current_time,
                        "end": bucket_end,
                        "occupied": is_occupied,
                        "activity_ratio": activity_ratio,
                    }
                )

                current_time = bucket_end

            # Calculate occupancy rate from activity patterns
            occupied_time = sum(
                (period["end"] - period["start"]).total_seconds()
                for period in activity_periods
                if period["occupied"]
            )
            total_time = sum(
                (period["end"] - period["start"]).total_seconds()
                for period in activity_periods
            )

            if total_time == 0:
                return None

            occupancy_rate = occupied_time / total_time

            # Debug logging for suspicious values
            occupied_periods = sum(1 for p in activity_periods if p["occupied"])
            _LOGGER.debug(
                "Time pattern analysis: %.3f occupancy rate from %d periods "
                "(occupied: %d, total: %d, occupied_time: %.1f hours, total_time: %.1f hours)",
                occupancy_rate,
                len(activity_periods),
                occupied_periods,
                len(activity_periods),
                occupied_time / 3600,
                total_time / 3600,
            )

            if occupancy_rate > 0.75:
                _LOGGER.warning(
                    "Time pattern analysis produced suspiciously high rate %.3f. "
                    "Occupied periods: %d/%d, this may indicate limited data or sensor issues.",
                    occupancy_rate,
                    occupied_periods,
                    len(activity_periods),
                )

            return occupancy_rate

        except Exception as err:
            _LOGGER.warning("Time pattern analysis failed: %s", err)
            return None

    async def _primary_sensor_with_margin(
        self, primary_sensor: str, start_time: datetime, end_time: datetime
    ) -> float | None:
        """Method 4: Primary Sensor + Uncertainty Margin (fallback for single sensor)."""
        try:
            _LOGGER.debug("Using primary sensor with margin: %s", primary_sensor)

            states = await get_states_from_recorder(
                self.coordinator.hass, primary_sensor, start_time, end_time
            )

            if not states:
                _LOGGER.debug("No states found for primary sensor")
                return None

            state_objects = [s for s in states if isinstance(s, State)]
            if not state_objects:
                _LOGGER.debug("No valid states found for primary sensor")
                return None

            intervals = await states_to_intervals(state_objects, start_time, end_time)

            # Calculate basic occupancy rate
            occupied_duration = sum(
                (interval["end"] - interval["start"]).total_seconds()
                for interval in intervals
                if interval["state"] == STATE_ON
            )
            total_duration = sum(
                (interval["end"] - interval["start"]).total_seconds()
                for interval in intervals
            )

            if total_duration == 0:
                return None

            raw_occupancy_rate = occupied_duration / total_duration

            # Add uncertainty margin to break circular logic
            # Reduce confidence by adding 5% uncertainty toward 0.5 (maximum entropy)
            uncertainty_factor = 0.05
            adjusted_rate = (
                raw_occupancy_rate * (1 - uncertainty_factor) + 0.5 * uncertainty_factor
            )

            _LOGGER.debug(
                "Primary sensor with margin: %.3f -> %.3f (added %.1f%% uncertainty)",
                raw_occupancy_rate,
                adjusted_rate,
                uncertainty_factor * 100,
            )

            return adjusted_rate

        except Exception as err:
            _LOGGER.warning("Primary sensor with margin failed: %s", err)
            return None

    @property
    def area_baseline_prior(self) -> float:
        """Get the cached area baseline prior or calculate it if not available."""
        if self._area_baseline_prior is not None:
            return self._area_baseline_prior

        # No cached value, return default (async calculation will be triggered by coordinator)
        return DEFAULT_PRIOR

    @property
    def method_used(self) -> str | None:
        """Get the method used for the last calculation."""
        return self._method_used

    def clear_cache(self) -> None:
        """Clear the cached area baseline prior."""
        self._area_baseline_prior = None
        self._last_updated = None
        self._method_used = None
