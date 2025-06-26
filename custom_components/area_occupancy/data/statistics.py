"""Per-entity statistics calculation for numeric sensors with active_range bounds.

Similar to the Likelihood class, this is assigned per-entity and calculates
statistics specifically for that entity's numeric values during occupied periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import State
from homeassistant.util import dt as dt_util

from ..utils import TimeInterval, get_states_from_recorder, states_to_intervals

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Minimum samples required for statistical calculations
MIN_SAMPLES_FOR_BOUNDS = 10
MIN_SAMPLES_FOR_TREND = 20


class TrendDirection(StrEnum):
    """Trend direction enumeration."""

    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class NumericStatistics:
    """Comprehensive statistics for numeric sensor analysis."""

    # Basic bounds for active_range
    lower_bound: float
    upper_bound: float

    # Central tendency measures
    mean: float
    median: float

    # Spread measures
    std_dev: float
    range_span: float

    # Percentiles for robust analysis
    percentile_10: float
    percentile_90: float

    # Trend analysis
    trend_direction: TrendDirection
    trend_slope: float  # Units per day

    # Occupancy-specific analysis
    occupied_average: float
    unoccupied_average: float | None
    occupancy_delta: float | None  # occupied - unoccupied average

    # Data quality metrics
    total_samples: int
    occupied_samples: int
    outliers_removed: int

    # Confidence metrics
    bounds_confidence: float


class Statistics:
    """Calculate statistics and active_range bounds for a numeric sensor entity.

    This class is similar to the Likelihood class - it's assigned per-entity.
    """

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entity_id: str,
        default_lower: float,
        default_upper: float,
    ) -> None:
        """Initialize the Statistics class."""
        self.coordinator = coordinator
        self.entity_id = entity_id
        self.hass = coordinator.hass
        self.days = coordinator.config.history.period
        self.cache_ttl = timedelta(hours=2)

        # Default bounds
        self.default_lower = default_lower
        self.default_upper = default_upper

        # Cached results
        self.statistics: NumericStatistics | None = None
        self.last_updated: datetime | None = None
        self.sensor_type = self._detect_sensor_type()

    def _detect_sensor_type(self) -> str:
        """Detect sensor type from entity_id."""
        entity_id_lower = self.entity_id.lower()

        # Debug logging to help identify why detection fails
        _LOGGER.debug("Detecting sensor type for %s", self.entity_id)

        if any(word in entity_id_lower for word in ["illuminance", "light", "lux"]):
            _LOGGER.debug("Detected as illuminance sensor")
            return "illuminance"
        if any(word in entity_id_lower for word in ["humidity", "humid"]):
            _LOGGER.debug("Detected as humidity sensor")
            return "humidity"
        if any(word in entity_id_lower for word in ["temperature", "temp"]):
            _LOGGER.debug("Detected as temperature sensor")
            return "temperature"
        if any(word in entity_id_lower for word in ["pressure", "bar", "hpa"]):
            _LOGGER.debug("Detected as pressure sensor")
            return "pressure"
        if any(word in entity_id_lower for word in ["co2", "carbon"]):
            _LOGGER.debug("Detected as co2 sensor")
            return "co2"
        _LOGGER.debug("Defaulting to generic sensor type")
        return "generic"

    @property
    def active_range(self) -> tuple[float, float]:
        """Get the calculated active range bounds."""
        if self.statistics:
            return (self.statistics.lower_bound, self.statistics.upper_bound)
        return (self.default_lower, self.default_upper)

    @property
    def active_range_raw(self) -> tuple[float, float]:
        """Get the raw calculated bounds without fallback to defaults."""
        if self.statistics:
            return (self.statistics.lower_bound, self.statistics.upper_bound)
        return (0.0, 0.0)  # Indicates no calculation available

    async def detect_current_activity(self, current_value: float) -> bool:
        """Detect if current sensor value indicates active occupancy.

        This is much smarter than simple range checking - it detects activity patterns
        like shower usage, lights turning on, human presence effects, etc.
        """
        if self.statistics is None:
            # Fall back to simple range check if no statistics available
            lower, upper = self.active_range
            return lower <= current_value <= upper

        try:
            # Get recent baseline for comparison
            recent_baseline = await self._get_recent_baseline()

            if self.sensor_type == "humidity":
                return await self._detect_humidity_activity(
                    current_value, recent_baseline
                )
            if self.sensor_type == "illuminance":
                return await self._detect_illuminance_activity(
                    current_value, recent_baseline
                )
            if self.sensor_type == "temperature":
                return await self._detect_temperature_activity(
                    current_value, recent_baseline
                )
            # For unknown sensor types, use enhanced range detection
            return await self._detect_generic_activity(current_value, recent_baseline)

        except Exception as err:
            _LOGGER.warning("Activity detection failed for %s: %s", self.entity_id, err)
            # Fall back to simple range check
            lower, upper = self.active_range
            return lower <= current_value <= upper

    async def _get_recent_baseline(self) -> float:
        """Get recent baseline value for comparison (last 30 minutes of unoccupied data)."""
        try:
            # Get recent states (last 30 minutes)
            end_time = dt_util.utcnow()
            start_time = end_time - timedelta(minutes=30)

            states = await get_states_from_recorder(
                self.hass, self.entity_id, start_time, end_time
            )

            if not states:
                # No recent data, use stored unoccupied average
                if self.statistics:
                    return (
                        self.statistics.unoccupied_average
                        or self.statistics.occupied_average
                    )
                return 50.0  # Default baseline if no statistics

            # Convert to intervals and get recent unoccupied values
            intervals = await states_to_intervals(
                [s for s in states if isinstance(s, State)], start_time, end_time
            )

            unoccupied_values = []
            prior_intervals = self.coordinator.prior.prior_intervals

            for interval in intervals:
                try:
                    value = float(interval["state"])
                    # Only include if NOT overlapping with prior (unoccupied)
                    if not self._interval_overlaps_prior(interval, prior_intervals):
                        unoccupied_values.append(value)
                except (ValueError, TypeError):
                    continue

            if unoccupied_values:
                return sum(unoccupied_values) / len(unoccupied_values)
            # No recent unoccupied data, use stored average
            if self.statistics:
                return (
                    self.statistics.unoccupied_average
                    or self.statistics.occupied_average
                )
            return 50.0  # Default baseline

        except Exception:
            # Fallback to stored unoccupied average
            if self.statistics:
                return (
                    self.statistics.unoccupied_average
                    or self.statistics.occupied_average
                )
            return 50.0  # Default baseline

    async def _detect_humidity_activity(
        self, current_value: float, baseline: float
    ) -> bool:
        """Detect humidity activity (shower usage, cooking, etc.)."""
        # Calculate how much above baseline we are
        delta = current_value - baseline

        # Shower detection: significant increase above baseline
        if delta > 8.0:  # 8% above baseline indicates shower activity
            _LOGGER.debug(
                "Humidity activity detected: %.1f%% vs baseline %.1f%% (delta: +%.1f%%)",
                current_value,
                baseline,
                delta,
            )
            return True

        # Also check if we're in the upper portion of our learned active range
        if self.statistics and current_value > (
            self.statistics.lower_bound
            + 0.7 * (self.statistics.upper_bound - self.statistics.lower_bound)
        ):
            _LOGGER.debug(
                "Humidity in upper active range: %.1f%% (active range: %.1f-%.1f%%)",
                current_value,
                self.statistics.lower_bound,
                self.statistics.upper_bound,
            )
            return True

        return False

    async def _detect_illuminance_activity(
        self, current_value: float, baseline: float
    ) -> bool:
        """Detect illuminance activity (lights being turned on)."""
        # Calculate how much above baseline we are
        delta = current_value - baseline

        # Light detection: significant increase above ambient
        if delta > 3.0:  # 3+ lux above baseline indicates lights on
            _LOGGER.debug(
                "Illuminance activity detected: %.1f lux vs baseline %.1f lux (delta: +%.1f lux)",
                current_value,
                baseline,
                delta,
            )
            return True

        # Also check if we're clearly above natural ambient levels
        if current_value > max(
            10.0, baseline * 1.5
        ):  # Either >10 lux or 50% above baseline
            _LOGGER.debug(
                "Illuminance above ambient levels: %.1f lux (baseline: %.1f lux)",
                current_value,
                baseline,
            )
            return True

        return False

    async def _detect_temperature_activity(
        self, current_value: float, baseline: float
    ) -> bool:
        """Detect temperature activity (human presence warming/cooling)."""
        # Calculate deviation from baseline
        delta = abs(current_value - baseline)

        # Human presence detection: measurable temperature change
        if delta > 0.3:  # 0.3°C change indicates human presence effect
            _LOGGER.debug(
                "Temperature activity detected: %.2f°C vs baseline %.2f°C (delta: %.2f°C)",
                current_value,
                baseline,
                current_value - baseline,
            )
            return True

        # Also check if we're outside the typical "unoccupied" range
        if self.statistics:
            # If we have good unoccupied data, use tighter bounds around it
            unoccupied_std = 0.2  # Assume 0.2°C standard deviation for unoccupied
            if abs(current_value - baseline) > 2 * unoccupied_std:
                _LOGGER.debug(
                    "Temperature outside unoccupied range: %.2f°C (baseline ± %.2f°C)",
                    current_value,
                    2 * unoccupied_std,
                )
                return True

        return False

    async def _detect_generic_activity(
        self, current_value: float, baseline: float
    ) -> bool:
        """Generic activity detection for unknown sensor types."""
        if not self.statistics:
            return False

        # Use statistical outlier detection
        # If current value is more than 1 standard deviation from baseline, consider it active
        threshold = (
            self.statistics.std_dev * 0.8
        )  # Slightly less than 1 std dev for sensitivity

        if abs(current_value - baseline) > threshold:
            _LOGGER.debug(
                "Generic activity detected: %.2f vs baseline %.2f (threshold: %.2f)",
                current_value,
                baseline,
                threshold,
            )
            return True

        return False

    def detect_current_activity_sync(self, current_value: float) -> bool:
        """Synchronous version of activity detection using cached baselines"""

        if self.statistics is None:
            # Fall back to simple range check if no statistics available
            lower, upper = self.active_range
            return lower <= current_value <= upper

        # Use stored unoccupied average as baseline
        baseline = (
            self.statistics.unoccupied_average or self.statistics.occupied_average
        )

        if self.sensor_type == "humidity":
            return self._detect_humidity_activity_sync(current_value, baseline)
        if self.sensor_type == "illuminance":
            return self._detect_illuminance_activity_sync(current_value, baseline)
        if self.sensor_type == "temperature":
            return self._detect_temperature_activity_sync(current_value, baseline)
        return self._detect_generic_activity_sync(current_value, baseline)

    def _detect_humidity_activity_sync(
        self, current_value: float, baseline: float
    ) -> bool:
        """Synchronous humidity activity detection"""
        delta = current_value - baseline

        # Shower detection: significant increase above baseline
        if delta > 8.0:  # 8% above baseline indicates shower activity
            return True

        # Also check if we're in the upper portion of our learned active range
        if self.statistics and current_value > (
            self.statistics.lower_bound
            + 0.7 * (self.statistics.upper_bound - self.statistics.lower_bound)
        ):
            return True

        return False

    def _detect_illuminance_activity_sync(
        self, current_value: float, baseline: float
    ) -> bool:
        """Synchronous illuminance activity detection"""
        delta = current_value - baseline

        # Light detection: significant increase above ambient
        if delta > 3.0:  # 3+ lux above baseline indicates lights on
            return True

        # Also check if we're clearly above natural ambient levels
        if current_value > max(
            10.0, baseline * 1.5
        ):  # Either >10 lux or 50% above baseline
            return True

        return False

    def _detect_temperature_activity_sync(
        self, current_value: float, baseline: float
    ) -> bool:
        """Synchronous temperature activity detection"""
        delta = abs(current_value - baseline)

        # Human presence detection: measurable temperature change
        if delta > 0.3:  # 0.3°C change indicates human presence effect
            return True

        # Also check if we're outside the typical "unoccupied" range
        if self.statistics:
            # Use tighter bounds around baseline
            unoccupied_std = 0.2  # Assume 0.2°C standard deviation for unoccupied
            if abs(current_value - baseline) > 2 * unoccupied_std:
                return True

        return False

    def _detect_generic_activity_sync(
        self, current_value: float, baseline: float
    ) -> bool:
        """Synchronous generic activity detection."""
        if not self.statistics:
            return False

        # Use statistical outlier detection
        threshold = (
            self.statistics.std_dev * 0.8
        )  # Slightly less than 1 std dev for sensitivity

        if abs(current_value - baseline) > threshold:
            return True

        return False

    async def update(
        self, force: bool = False, history_period: int | None = None
    ) -> NumericStatistics:
        """Return statistics, re-computing if cache is stale or forced."""
        if not force and self._is_cache_valid():
            return self.statistics  # type: ignore[return-value]

        try:
            statistics = await self.calculate(history_period=history_period)
        except Exception:  # pragma: no cover
            _LOGGER.exception(
                "Statistics calculation failed for %s, using defaults", self.entity_id
            )
            statistics = self._create_default_statistics()

        self.statistics = statistics
        self.last_updated = dt_util.utcnow()
        return statistics

    def _is_cache_valid(self) -> bool:
        """Check if cached statistics are still valid."""
        if self.statistics is None or self.last_updated is None:
            return False
        return (dt_util.utcnow() - self.last_updated) <= self.cache_ttl

    async def calculate(self, history_period: int | None = None) -> NumericStatistics:
        """Calculate comprehensive sensor statistics."""

        days_to_use = history_period if history_period is not None else self.days
        start_time = dt_util.utcnow() - timedelta(days=days_to_use)
        end_time = dt_util.utcnow()

        # Get sensor states and prior intervals
        states = await get_states_from_recorder(
            self.hass, self.entity_id, start_time, end_time
        )
        prior_intervals = self.coordinator.prior.prior_intervals

        if not states or not prior_intervals:
            return self._create_default_statistics()

        # Extract numeric values during occupied/unoccupied periods
        occupied_values, unoccupied_values = await self._extract_values(
            states, prior_intervals, start_time, end_time
        )

        if len(occupied_values) < MIN_SAMPLES_FOR_BOUNDS:
            return self._create_default_statistics()

        # Calculate statistics
        return self._calculate_statistics(occupied_values, unoccupied_values)

    async def _extract_values(
        self, states, prior_intervals, start_time, end_time
    ) -> tuple[list[float], list[float]]:
        """Extract numeric values categorized by occupancy."""
        intervals = await states_to_intervals(
            [s for s in states if isinstance(s, State)], start_time, end_time
        )

        occupied_values = []
        unoccupied_values = []

        # Also store timestamped data for trend analysis
        self.occupied_timestamped = []
        self.unoccupied_timestamped = []

        for interval in intervals:
            try:
                value = float(interval["state"])
                timestamp = interval["start"]

                # Categorize by occupancy
                if self._interval_overlaps_prior(interval, prior_intervals):
                    occupied_values.append(value)
                    self.occupied_timestamped.append((timestamp, value))
                else:
                    unoccupied_values.append(value)
                    self.unoccupied_timestamped.append((timestamp, value))

            except (ValueError, TypeError):
                continue  # Skip non-numeric values

        return occupied_values, unoccupied_values

    def _interval_overlaps_prior(
        self, interval: TimeInterval, prior_intervals: list[TimeInterval]
    ) -> bool:
        """Check if interval overlaps with any prior interval."""
        for prior in prior_intervals:
            if interval["end"] > prior["start"] and interval["start"] < prior["end"]:
                return True
        return False

    def _calculate_statistics(
        self, occupied_values: list[float], unoccupied_values: list[float]
    ) -> NumericStatistics:
        """Calculate comprehensive statistics."""

        # Remove outliers
        filtered_values = self._remove_outliers(occupied_values)
        outliers_removed = len(occupied_values) - len(filtered_values)

        if not filtered_values:
            return self._create_default_statistics()

        # Basic statistics
        mean_val = sum(filtered_values) / len(filtered_values)
        sorted_vals = sorted(filtered_values)
        n = len(sorted_vals)
        median_val = (
            sorted_vals[n // 2]
            if n % 2 == 1
            else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        )

        # Standard deviation
        variance = sum((x - mean_val) ** 2 for x in filtered_values) / len(
            filtered_values
        )
        std_dev = variance**0.5

        # Percentiles
        p10 = sorted_vals[max(0, int(0.1 * n))]
        p90 = sorted_vals[min(n - 1, int(0.9 * n))]

        # Calculate activity-aware bounds
        lower_bound, upper_bound = self._calculate_activity_aware_bounds(
            filtered_values, unoccupied_values
        )

        # Occupancy comparison
        unoccupied_avg = (
            sum(unoccupied_values) / len(unoccupied_values)
            if unoccupied_values
            else None
        )
        occupancy_delta = (
            mean_val - unoccupied_avg if unoccupied_avg is not None else None
        )

        # Calculate trend analysis
        trend_direction, trend_slope = self._calculate_trend_analysis()

        # Confidence calculation
        bounds_confidence = min(
            1.0, len(filtered_values) / 100.0
        )  # Based on sample size

        return NumericStatistics(
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            mean=mean_val,
            median=median_val,
            std_dev=std_dev,
            range_span=max(filtered_values) - min(filtered_values),
            percentile_10=float(p10),
            percentile_90=float(p90),
            trend_direction=trend_direction,
            trend_slope=trend_slope,
            occupied_average=mean_val,
            unoccupied_average=unoccupied_avg,
            occupancy_delta=occupancy_delta,
            total_samples=len(occupied_values) + len(unoccupied_values),
            occupied_samples=len(occupied_values),
            outliers_removed=outliers_removed,
            bounds_confidence=bounds_confidence,
        )

    def _remove_outliers(self, values: list[float]) -> list[float]:
        """Remove outliers using IQR method."""
        if len(values) < 4:
            return values

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q25 = sorted_vals[int(0.25 * n)]
        q75 = sorted_vals[int(0.75 * n)]
        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        return [v for v in values if lower_bound <= v <= upper_bound]

    def _calculate_trend_analysis(self) -> tuple[TrendDirection, float]:
        """Calculate trend analysis from timestamped occupied data."""
        if (
            not hasattr(self, "occupied_timestamped")
            or len(self.occupied_timestamped) < MIN_SAMPLES_FOR_TREND
        ):
            return TrendDirection.INSUFFICIENT_DATA, 0.0

        try:
            # Sort by timestamp to ensure chronological order
            sorted_data = sorted(self.occupied_timestamped, key=lambda x: x[0])

            # Convert timestamps to days since start for regression
            start_time = sorted_data[0][0]
            x_data = [
                (ts - start_time).total_seconds() / 86400 for ts, _ in sorted_data
            ]  # Days
            y_data = [value for _, value in sorted_data]

            if len(set(y_data)) < 2:  # All values are the same
                return TrendDirection.STABLE, 0.0

            # Simple linear regression: y = mx + b
            n = len(x_data)
            x_mean = sum(x_data) / n
            y_mean = sum(y_data) / n

            numerator = sum(
                (x_data[i] - x_mean) * (y_data[i] - y_mean) for i in range(n)
            )
            denominator = sum((x_data[i] - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return TrendDirection.STABLE, 0.0

            slope = numerator / denominator  # Units per day

            # Determine significance based on sensor type and slope magnitude
            noise_threshold = self._get_noise_threshold()

            if abs(slope) < noise_threshold:
                return TrendDirection.STABLE, slope
            if slope > 0:
                return TrendDirection.RISING, slope
            return TrendDirection.FALLING, slope

        except Exception as err:
            _LOGGER.warning("Error in trend analysis for %s: %s", self.entity_id, err)
            return TrendDirection.INSUFFICIENT_DATA, 0.0

    def _get_noise_threshold(self) -> float:
        """Get noise threshold for trend significance based on sensor type."""
        thresholds = {
            "humidity": 0.5,  # 0.5%/day
            "temperature": 0.1,  # 0.1°C/day
            "illuminance": 2.0,  # 2 lux/day
            "pressure": 0.5,  # 0.5 hPa/day
            "co2": 10.0,  # 10 ppm/day
            "generic": 0.01,
        }
        return thresholds.get(self.sensor_type, 0.01)

    def _calculate_activity_aware_bounds(
        self, occupied_values: list[float], unoccupied_values: list[float]
    ) -> tuple[float, float]:
        """Calculate bounds that focus on activity patterns rather than just percentiles."""

        # Get unoccupied baseline for delta-based calculations
        unoccupied_baseline = (
            sum(unoccupied_values) / len(unoccupied_values)
            if unoccupied_values
            else None
        )

        # Calculate occupancy delta
        occupied_mean = sum(occupied_values) / len(occupied_values)
        occupancy_delta = (
            occupied_mean - unoccupied_baseline
            if unoccupied_baseline is not None
            else 0
        )

        sorted_vals = sorted(occupied_values)
        n = len(sorted_vals)

        if self.sensor_type == "humidity" and unoccupied_baseline is not None:
            # For humidity, detect elevated events (shower usage)
            # Look for values significantly above baseline during occupancy
            if occupancy_delta > 5.0:  # Significant humidity increase
                # Use values above baseline + threshold for upper bound
                elevated_values = [
                    v for v in occupied_values if v > unoccupied_baseline + 3.0
                ]
                if elevated_values:
                    lower = unoccupied_baseline + 5.0  # Start of "active" humidity
                    upper = sorted(elevated_values)[
                        int(0.95 * len(elevated_values))
                    ]  # 95th percentile of elevated
                    _LOGGER.debug(
                        "Humidity %s: Detected shower activity (delta=%.1f%%, %d elevated samples), "
                        "bounds=[%.1f, %.1f] vs baseline %.1f",
                        self.entity_id,
                        occupancy_delta,
                        len(elevated_values),
                        lower,
                        upper,
                        unoccupied_baseline,
                    )
                    return max(50.0, lower), min(100.0, upper)
            # Fall back to conservative percentiles if no clear elevation pattern
            lower = sorted_vals[max(0, int(0.2 * n))]
            upper = sorted_vals[min(n - 1, int(0.8 * n))]
            _LOGGER.debug(
                "Humidity %s: No clear shower pattern (delta=%.1f%%), using percentiles=[%.1f, %.1f]",
                self.entity_id,
                occupancy_delta,
                lower,
                upper,
            )
            return max(0.0, lower), min(100.0, upper)

        if self.sensor_type == "illuminance" and unoccupied_baseline is not None:
            # For illuminance, detect artificial lighting vs ambient
            if occupancy_delta > 2.0:  # Lights being turned on
                # Separate ambient from artificial lighting
                bright_values = [
                    v for v in occupied_values if v > unoccupied_baseline + 2.0
                ]
                if bright_values:
                    lower = unoccupied_baseline + 2.0  # Above ambient
                    upper = sorted(bright_values)[
                        int(0.9 * len(bright_values))
                    ]  # 90th percentile of bright
                    _LOGGER.debug(
                        "Illuminance %s: Detected lighting activity (delta=%.1f lux, %d bright samples), "
                        "bounds=[%.1f, %.1f] vs ambient %.1f",
                        self.entity_id,
                        occupancy_delta,
                        len(bright_values),
                        lower,
                        upper,
                        unoccupied_baseline,
                    )
                    return max(1.0, lower), upper
            # Fall back to wider percentiles for illuminance
            lower = sorted_vals[max(0, int(0.1 * n))]
            upper = sorted_vals[min(n - 1, int(0.9 * n))]
            _LOGGER.debug(
                "Illuminance %s: No clear lighting pattern (delta=%.1f lux), using percentiles=[%.1f, %.1f]",
                self.entity_id,
                occupancy_delta,
                lower,
                upper,
            )
            return max(0.0, lower), upper

        if self.sensor_type == "temperature" and unoccupied_baseline is not None:
            # For temperature, detect human presence warming
            if abs(occupancy_delta) > 0.2:  # Detectable temperature change
                # Use baseline +/- reasonable human presence range
                if occupancy_delta > 0:  # Warming during occupancy
                    lower = unoccupied_baseline - 0.5
                    upper = unoccupied_baseline + 2.0  # Human warming effect
                else:  # Cooling during occupancy (unusual but possible)
                    lower = unoccupied_baseline - 2.0
                    upper = unoccupied_baseline + 0.5
                _LOGGER.debug(
                    "Temperature %s: Detected presence effect (delta=%.2f°C), "
                    "bounds=[%.2f, %.2f] vs baseline %.2f",
                    self.entity_id,
                    occupancy_delta,
                    lower,
                    upper,
                    unoccupied_baseline,
                )
                return lower, upper
            # Fall back to tight IQR for temperature
            q25 = sorted_vals[max(0, int(0.25 * n))]
            q75 = sorted_vals[min(n - 1, int(0.75 * n))]
            iqr = q75 - q25
            lower, upper = q25 - 0.5 * iqr, q75 + 0.5 * iqr
            _LOGGER.debug(
                "Temperature %s: Minimal presence effect (delta=%.2f°C), using IQR=[%.2f, %.2f]",
                self.entity_id,
                occupancy_delta,
                lower,
                upper,
            )
            return lower, upper

        # Generic approach for unknown sensor types
        lower = sorted_vals[max(0, int(0.15 * n))]
        upper = sorted_vals[min(n - 1, int(0.85 * n))]
        _LOGGER.debug(
            "Generic sensor %s (%s): Using 15-85th percentiles=[%.2f, %.2f]",
            self.entity_id,
            self.sensor_type,
            lower,
            upper,
        )
        return float(lower), float(upper)

    def _create_default_statistics(self) -> NumericStatistics:
        """Create default statistics when insufficient data is available."""
        mean_val = (self.default_lower + self.default_upper) / 2.0

        return NumericStatistics(
            lower_bound=self.default_lower,
            upper_bound=self.default_upper,
            mean=mean_val,
            median=mean_val,
            std_dev=0.0,
            range_span=self.default_upper - self.default_lower,
            percentile_10=self.default_lower,
            percentile_90=self.default_upper,
            trend_direction=TrendDirection.INSUFFICIENT_DATA,
            trend_slope=0.0,
            occupied_average=mean_val,
            unoccupied_average=None,
            occupancy_delta=None,
            total_samples=0,
            occupied_samples=0,
            outliers_removed=0,
            bounds_confidence=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary for storage."""
        if not self.statistics:
            return {}

        return {
            "statistics": {
                "lower_bound": self.statistics.lower_bound,
                "upper_bound": self.statistics.upper_bound,
                "mean": self.statistics.mean,
                "median": self.statistics.median,
                "std_dev": self.statistics.std_dev,
                "range_span": self.statistics.range_span,
                "percentile_10": self.statistics.percentile_10,
                "percentile_90": self.statistics.percentile_90,
                "trend_direction": self.statistics.trend_direction.value,
                "trend_slope": self.statistics.trend_slope,
                "occupied_average": self.statistics.occupied_average,
                "unoccupied_average": self.statistics.unoccupied_average,
                "occupancy_delta": self.statistics.occupancy_delta,
                "total_samples": self.statistics.total_samples,
                "occupied_samples": self.statistics.occupied_samples,
                "outliers_removed": self.statistics.outliers_removed,
                "bounds_confidence": self.statistics.bounds_confidence,
            },
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
            "sensor_type": self.sensor_type,
            "default_lower": self.default_lower,
            "default_upper": self.default_upper,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: AreaOccupancyCoordinator, entity_id: str
    ) -> Statistics:
        """Restore statistics from stored data."""
        default_lower = data.get("default_lower", 0.0)
        default_upper = data.get("default_upper", 1.0)

        statistics_obj = cls(coordinator, entity_id, default_lower, default_upper)

        if "statistics" in data:
            stats_data = data["statistics"]

            # Convert trend_direction back to enum
            trend_direction = TrendDirection(
                stats_data.get("trend_direction", "insufficient_data")
            )
            stats_data["trend_direction"] = trend_direction

            statistics_obj.statistics = NumericStatistics(**stats_data)

        statistics_obj.last_updated = (
            datetime.fromisoformat(data["last_updated"])
            if data.get("last_updated")
            else None
        )

        # Always re-detect sensor type to ensure it's current
        # (don't restore from storage as detection logic may have improved)
        statistics_obj.sensor_type = statistics_obj._detect_sensor_type()

        return statistics_obj
