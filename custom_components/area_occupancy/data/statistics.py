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
        if any(word in entity_id_lower for word in ["illuminance", "light", "lux"]):
            return "illuminance"
        if any(word in entity_id_lower for word in ["humidity", "humid"]):
            return "humidity"
        if any(word in entity_id_lower for word in ["temperature", "temp"]):
            return "temperature"
        if any(word in entity_id_lower for word in ["pressure", "bar", "hpa"]):
            return "pressure"
        if any(word in entity_id_lower for word in ["co2", "carbon"]):
            return "co2"
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

        for interval in intervals:
            try:
                value = float(interval["state"])

                # Categorize by occupancy
                if self._interval_overlaps_prior(interval, prior_intervals):
                    occupied_values.append(value)
                else:
                    unoccupied_values.append(value)

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

        # Calculate robust bounds
        lower_bound, upper_bound = self._calculate_robust_bounds(filtered_values)

        # Occupancy comparison
        unoccupied_avg = (
            sum(unoccupied_values) / len(unoccupied_values)
            if unoccupied_values
            else None
        )
        occupancy_delta = (
            mean_val - unoccupied_avg if unoccupied_avg is not None else None
        )

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
            trend_direction=TrendDirection.INSUFFICIENT_DATA,  # Simplified for now
            trend_slope=0.0,
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

    def _calculate_robust_bounds(self, values: list[float]) -> tuple[float, float]:
        """Calculate robust bounds using sensor-type specific logic."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        if self.sensor_type == "illuminance":
            # For illuminance, use wider percentiles
            lower = sorted_vals[max(0, int(0.1 * n))]
            upper = sorted_vals[min(n - 1, int(0.9 * n))]
        elif self.sensor_type == "humidity":
            # For humidity, use tighter percentiles since range is bounded 0-100
            lower = sorted_vals[max(0, int(0.15 * n))]
            upper = sorted_vals[min(n - 1, int(0.85 * n))]
            lower = max(0, lower)
            upper = min(100, upper)
        elif self.sensor_type == "temperature":
            # For temperature, use IQR-based bounds
            q25 = sorted_vals[max(0, int(0.25 * n))]
            q75 = sorted_vals[min(n - 1, int(0.75 * n))]
            iqr = q75 - q25
            lower = q25 - 1.0 * iqr
            upper = q75 + 1.0 * iqr
        else:
            # Generic approach
            lower = sorted_vals[max(0, int(0.1 * n))]
            upper = sorted_vals[min(n - 1, int(0.9 * n))]

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

        statistics_obj.sensor_type = data.get("sensor_type", "generic")

        return statistics_obj
