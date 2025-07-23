"""Per-sensor likelihoods P(E|H) and P(E|¬H).

Computes *overlap* of each sensor's active intervals with the area's
ground-truth occupied intervals, giving informative likelihoods that
differ between H and ¬H.

For numeric sensors, uses statistical analysis to calculate likelihood
of sensor readings indicating occupancy based on historical patterns.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import State
from homeassistant.util import dt as dt_util

from ..utils import TimeInterval, get_states_from_recorder, states_to_intervals

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Interval filtering thresholds to exclude anomalous data
MIN_INTERVAL_SECONDS = 10  # Exclude intervals shorter than 10 seconds (false triggers)
MAX_INTERVAL_SECONDS = (
    13 * 3600
)  # Exclude intervals longer than 13 hours (stuck sensors)

# Statistical significance thresholds for numeric sensors
MIN_OCCUPANCY_DELTA_THRESHOLD = 0.1  # Minimum meaningful occupancy delta
MIN_CONFIDENCE_THRESHOLD = 0.1  # Minimum statistical confidence required


# ─────────────────────────────────────────────────────────────────────────────
class Likelihood:
    """Learn conditional probabilities for binary and numeric sensors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entity_id: str,
        active_states: list[str],
        default_prob_true: float,
        default_prob_false: float,
        weight: float,  # Keep weight for applying to calculated values
    ) -> None:
        """Initialize the Likelihood class."""
        self.entity_id = entity_id
        self.active_states = active_states
        self.default_prob_true = default_prob_true
        self.default_prob_false = default_prob_false
        self.weight = weight  # Store weight for applying to calculations
        self.days = coordinator.config.history.period
        self.history_enabled = coordinator.config.history.enabled
        self.hass = coordinator.hass
        self.coordinator = coordinator
        self.last_updated: datetime | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.states: list[State] | None = None
        self.intervals: list[TimeInterval] | None = None
        self.active_seconds: int | None = None
        self.inactive_seconds: int | None = None
        self.active_ratio: float | None = None
        self.inactive_ratio: float | None = None
        self.cache_ttl = timedelta(hours=2)

        # Filtering statistics for anomaly detection
        self.total_on_intervals: int | None = None
        self.filtered_short_intervals: int | None = (
            None  # Count of intervals < MIN_INTERVAL_SECONDS
        )
        self.filtered_long_intervals: int | None = (
            None  # Count of intervals > MAX_INTERVAL_SECONDS
        )
        self.valid_intervals: int | None = (
            None  # Count of intervals used in calculation
        )
        self.max_filtered_duration_seconds: float | None = (
            None  # Longest filtered interval duration
        )

        # Numeric sensor statistics
        self.is_numeric_sensor = not bool(active_states)
        self.statistics_based_calculation = False

    @property
    def is_numeric(self) -> bool:
        """Check if this is a numeric sensor (no active_states defined)."""
        return self.is_numeric_sensor

    def _apply_weight_to_probability(self, prob: float, default_prob: float) -> float:
        """Apply weight to a probability value.

        Args:
            prob: Calculated probability
            default_prob: Default probability to use when calculated is very low

        Returns:
            Weighted probability

        """
        # Threshold for meaningful calculated data
        LOW_PROB_THRESHOLD = 0.05  # 5%

        if prob < LOW_PROB_THRESHOLD:
            # No meaningful calculated data - apply weight as multiplier to defaults
            weighted_prob = default_prob * self.weight
        else:
            # Good calculated data - apply weight as interpolation from neutral
            neutral_prob = 0.5
            weighted_prob = neutral_prob + (prob - neutral_prob) * self.weight

        # Final clamping to valid probability range
        return max(0.001, min(weighted_prob, 0.999))

    def _is_cache_valid(self) -> bool:
        """Check if the cached likelihood values are still valid.

        Returns:
            bool: True if cache is valid and fresh, False if needs recalculation

        """
        # If we don't have calculated values or timestamp, cache is invalid
        if (
            self.active_ratio is None
            or self.inactive_ratio is None
            or self.last_updated is None
        ):
            return False

        # Check if cache has expired based on TTL
        if (dt_util.utcnow() - self.last_updated) > self.cache_ttl:
            return False

        # Cache is valid
        return True

    @property
    def prob_given_true(self) -> float:
        """Return the weighted probability of the sensor being active given the area is occupied."""
        calculated_prob = (
            self.active_ratio
            if self.active_ratio is not None
            else self.default_prob_true
        )
        # Pass both calculated and default values
        return self._apply_weight_to_probability(
            calculated_prob, self.default_prob_true
        )

    @property
    def prob_given_false(self) -> float:
        """Return the weighted probability of the sensor being active given the area is not occupied."""
        calculated_prob = (
            self.inactive_ratio
            if self.inactive_ratio is not None
            else self.default_prob_false
        )
        # Pass both calculated and default values
        return self._apply_weight_to_probability(
            calculated_prob, self.default_prob_false
        )

    @property
    def prob_given_true_raw(self) -> float:
        """Return the raw calculated probability for prob_given_true."""
        return (
            self.active_ratio
            if self.active_ratio is not None
            else self.default_prob_true
        )

    @property
    def prob_given_false_raw(self) -> float:
        """Return the raw calculated probability for prob_given_false."""
        return (
            self.inactive_ratio
            if self.inactive_ratio is not None
            else self.default_prob_false
        )

    async def update(
        self, force: bool = False, history_period: int | None = None
    ) -> tuple[float, float]:
        """Return a likelihood, re-computing if the cache is stale or forced.

        Args:
            force: If True, bypass cache validation and force recalculation
            history_period: Period in days for historical data (overrides coordinator default)

        Returns:
            Tuple of (prob_given_true, prob_given_false) weighted values

        """
        if not self.history_enabled:
            return self.prob_given_true, self.prob_given_false  # type: ignore[return-value]

        # Check if we can use cached values
        if not force and self._is_cache_valid():
            _LOGGER.debug(
                "Using cached likelihood values for %s (last updated: %s)",
                self.entity_id,
                self.last_updated,
            )
            return self.prob_given_true, self.prob_given_false

        _LOGGER.debug(
            "Recalculating likelihood for %s (cache invalid or stale)", self.entity_id
        )

        try:
            # Use different calculation methods for numeric vs binary sensors
            if self.is_numeric:
                active_ratio, inactive_ratio = await self.calculate_numeric_likelihood(
                    history_period=history_period
                )
            else:
                active_ratio, inactive_ratio = await self.calculate(
                    history_period=history_period
                )
        except Exception:  # pragma: no cover
            _LOGGER.exception(
                "Likelihood calculation failed, using default %.2f",
                self.default_prob_true,
            )
            active_ratio = self.default_prob_true
            inactive_ratio = self.default_prob_false

        # Store the RAW calculated values
        self.active_ratio = active_ratio
        self.inactive_ratio = inactive_ratio
        self.last_updated = dt_util.utcnow()

        # Return the WEIGHTED values for immediate use
        return self.prob_given_true, self.prob_given_false

    async def calculate_numeric_likelihood(
        self, history_period: int | None = None
    ) -> tuple[float, float]:
        """Calculate likelihood for numeric sensors using statistical analysis.

        This method leverages the Statistics class calculations to determine
        P(sensor_indicates_occupancy|occupied) and P(sensor_indicates_occupancy|unoccupied).

        Args:
            history_period: Period in days for historical data (overrides coordinator default)

        Returns:
            Tuple of (prob_given_true, prob_given_false)

        """
        # Try to get statistics from the entity (if it has them)
        entity = None
        with contextlib.suppress(ValueError):
            entity = self.coordinator.entities.get_entity(self.entity_id)

        if not entity or not entity.statistics or not entity.statistics.statistics:
            _LOGGER.warning(
                "No statistics available for numeric sensor %s, using defaults",
                self.entity_id,
            )
            return self.default_prob_true, self.default_prob_false

        stats = entity.statistics.statistics

        # Check if we have sufficient data for meaningful calculations
        if (
            stats.occupied_samples < 50
            or stats.bounds_confidence < MIN_CONFIDENCE_THRESHOLD
        ):
            return self.default_prob_true, self.default_prob_false

        # Calculate likelihood based on occupancy delta and statistical distributions
        occupancy_delta = stats.occupancy_delta
        if (
            occupancy_delta is None
            or abs(occupancy_delta) < MIN_OCCUPANCY_DELTA_THRESHOLD
        ):
            return self.default_prob_true, self.default_prob_false

        # Determine sensor type for specialized calculations
        sensor_type = entity.statistics.sensor_type

        try:
            prob_given_true, prob_given_false = self._calculate_statistical_likelihood(
                stats, sensor_type, occupancy_delta
            )

            self.statistics_based_calculation = True

        except (ValueError, TypeError, ZeroDivisionError, AttributeError) as err:
            _LOGGER.warning(
                "Failed to calculate statistical likelihood for %s: %s",
                self.entity_id,
                err,
            )
            return self.default_prob_true, self.default_prob_false
        else:
            return prob_given_true, prob_given_false

    def _calculate_statistical_likelihood(
        self, stats, sensor_type: str, occupancy_delta: float
    ) -> tuple[float, float]:
        """Calculate likelihood using statistical distributions.

        Args:
            stats: NumericStatistics object with occupancy analysis
            sensor_type: Type of sensor (illuminance, humidity, temperature, etc.)
            occupancy_delta: Difference between occupied and unoccupied averages

        Returns:
            Tuple of (prob_given_true, prob_given_false)

        """
        std_dev = max(stats.std_dev, 0.1)  # Prevent division by zero

        # Calculate normalized effect sizes
        effect_size = abs(occupancy_delta) / std_dev

        # Base probabilities on statistical significance
        if effect_size < 0.5:  # Small effect
            base_prob_active_occupied = 0.15
            base_prob_active_unoccupied = 0.10
        elif effect_size < 1.0:  # Medium effect
            base_prob_active_occupied = 0.30
            base_prob_active_unoccupied = 0.15
        elif effect_size < 2.0:  # Large effect
            base_prob_active_occupied = 0.50
            base_prob_active_unoccupied = 0.20
        else:  # Very large effect
            base_prob_active_occupied = 0.70
            base_prob_active_unoccupied = 0.25

        # Apply sensor-specific adjustments
        if sensor_type == "illuminance":
            # For illuminance, occupancy typically means lower values (lights dimmed/off)
            if occupancy_delta < 0:  # Occupied is dimmer than unoccupied
                # Higher probability of "darkness" indicating occupancy
                prob_active_occupied = min(0.8, base_prob_active_occupied + 0.2)
                prob_active_unoccupied = max(0.05, base_prob_active_unoccupied - 0.05)
            else:  # Occupied is brighter than unoccupied (lights on)
                # Higher probability of "brightness" indicating occupancy
                prob_active_occupied = min(0.8, base_prob_active_occupied + 0.1)
                prob_active_unoccupied = max(0.05, base_prob_active_unoccupied)

        elif sensor_type == "humidity":
            # For humidity, occupancy typically means higher values (human presence, showers)
            if occupancy_delta > 0:  # Occupied is more humid
                # Higher probability of elevated humidity indicating occupancy
                prob_active_occupied = min(0.85, base_prob_active_occupied + 0.3)
                prob_active_unoccupied = max(0.03, base_prob_active_unoccupied - 0.02)
            else:  # Occupied is less humid (unusual)
                prob_active_occupied = base_prob_active_occupied
                prob_active_unoccupied = base_prob_active_unoccupied

        elif sensor_type == "temperature":
            # For temperature, occupancy typically means slightly higher values (body heat)
            if occupancy_delta > 0:  # Occupied is warmer
                # Higher probability of warming indicating occupancy
                prob_active_occupied = min(0.7, base_prob_active_occupied + 0.2)
                prob_active_unoccupied = max(0.05, base_prob_active_unoccupied - 0.03)
            else:  # Occupied is cooler (unusual but possible with AC)
                prob_active_occupied = base_prob_active_occupied
                prob_active_unoccupied = base_prob_active_unoccupied

        else:  # Generic sensor
            # Use base probabilities with confidence adjustment
            confidence_multiplier = min(1.5, stats.bounds_confidence + 0.5)
            prob_active_occupied = min(
                0.8, base_prob_active_occupied * confidence_multiplier
            )
            prob_active_unoccupied = max(
                0.02, base_prob_active_unoccupied / confidence_multiplier
            )

        # Apply sample size confidence adjustment
        sample_confidence = min(
            1.0, stats.occupied_samples / 200.0
        )  # Full confidence at 200+ samples
        prob_active_occupied = (
            self.default_prob_true * (1 - sample_confidence)
            + prob_active_occupied * sample_confidence
        )
        prob_active_unoccupied = (
            self.default_prob_false * (1 - sample_confidence)
            + prob_active_unoccupied * sample_confidence
        )

        # Final bounds checking
        prob_active_occupied = max(0.01, min(0.99, prob_active_occupied))
        prob_active_unoccupied = max(0.001, min(0.98, prob_active_unoccupied))

        return prob_active_occupied, prob_active_unoccupied

    async def calculate(self, history_period: int | None = None) -> tuple[float, float]:
        """Calculate the likelihood with anomaly filtering, considering only intervals within prior_intervals.

        Args:
            history_period: Period in days for historical data (overrides coordinator default)

        """
        _LOGGER.info("Likelihood calculation for %s", self.entity_id)
        # Use provided history_period or fall back to coordinator default
        days_to_use = history_period if history_period is not None else self.days
        self.start_time = dt_util.utcnow() - timedelta(days=days_to_use)
        self.end_time = dt_util.utcnow()

        states = await get_states_from_recorder(
            self.hass, self.entity_id, self.start_time, self.end_time
        )

        prior_intervals = self.coordinator.prior.prior_intervals

        active_ratio = self.default_prob_true
        inactive_ratio = self.default_prob_false

        # Initialize filtering statistics
        self.total_on_intervals = 0
        self.filtered_short_intervals = 0
        self.filtered_long_intervals = 0
        self.valid_intervals = 0
        self.max_filtered_duration_seconds = None

        # Debug logging
        _LOGGER.debug(
            "Likelihood calculation for %s: states=%d, prior_intervals=%d",
            self.entity_id,
            len(states) if states else 0,
            len(prior_intervals),
        )

        if states and prior_intervals:
            intervals = await states_to_intervals(
                [s for s in states if isinstance(s, State)],
                self.start_time,
                self.end_time,
            )

            # Apply anomaly filtering to intervals first
            filtered_intervals = []
            for interval in intervals:
                if interval["state"] in self.active_states:
                    self.total_on_intervals += 1
                    duration_seconds = (
                        interval["end"] - interval["start"]
                    ).total_seconds()

                    if duration_seconds < MIN_INTERVAL_SECONDS:
                        self.filtered_short_intervals += 1
                        _LOGGER.debug(
                            "Likelihood %s: Filtered short interval (%.1fs) from %s to %s",
                            self.entity_id,
                            duration_seconds,
                            interval["start"],
                            interval["end"],
                        )
                    elif duration_seconds > MAX_INTERVAL_SECONDS:
                        self.filtered_long_intervals += 1
                        # Track the maximum filtered duration
                        if (
                            self.max_filtered_duration_seconds is None
                            or duration_seconds > self.max_filtered_duration_seconds
                        ):
                            self.max_filtered_duration_seconds = duration_seconds
                        _LOGGER.debug(
                            "Likelihood %s: Filtered long interval (%.1fh) from %s to %s",
                            self.entity_id,
                            duration_seconds / 3600,
                            interval["start"],
                            interval["end"],
                        )
                    else:
                        # Valid interval - keep it for calculation
                        filtered_intervals.append(interval)
                        self.valid_intervals += 1
                else:
                    # Keep non-active intervals for calculation (they don't get filtered)
                    filtered_intervals.append(interval)

            # Log filtering results
            if self.filtered_short_intervals > 0 or self.filtered_long_intervals > 0:
                _LOGGER.info(
                    "Likelihood %s: Filtered %d short and %d long intervals, kept %d valid active intervals",
                    self.entity_id,
                    self.filtered_short_intervals,
                    self.filtered_long_intervals,
                    self.valid_intervals,
                )

            # Calculate total analysis period
            total_seconds = (self.end_time - self.start_time).total_seconds()

            # Calculate occupied time (sum of prior intervals)
            occupied_seconds = sum(
                (interval["end"] - interval["start"]).total_seconds()
                for interval in prior_intervals
            )

            # Calculate not-occupied time
            not_occupied_seconds = total_seconds - occupied_seconds

            if occupied_seconds > 0 and not_occupied_seconds > 0:
                # Filter intervals by overlap with prior intervals
                def interval_overlaps_prior(interval):
                    """Return True if the interval overlaps any prior interval."""
                    for prior in prior_intervals:
                        if (
                            interval["end"] > prior["start"]
                            and interval["start"] < prior["end"]
                        ):
                            return True
                    return False

                # Split filtered intervals into occupied and not-occupied periods
                occupied_active_seconds = 0
                not_occupied_active_seconds = 0

                for interval in filtered_intervals:
                    duration = (interval["end"] - interval["start"]).total_seconds()

                    if interval["state"] in self.active_states:
                        if interval_overlaps_prior(interval):
                            occupied_active_seconds += duration
                        else:
                            not_occupied_active_seconds += duration

                # Calculate the raw likelihoods as pure probabilities
                active_ratio = (
                    occupied_active_seconds / occupied_seconds
                )  # P(active|occupied)
                inactive_ratio = (
                    not_occupied_active_seconds / not_occupied_seconds
                )  # P(active|not_occupied)

                # Don't clamp here - let the weighting function handle low values
                # Only ensure they're not negative or greater than 1
                active_ratio = max(0.0, min(active_ratio, 1.0))
                inactive_ratio = max(0.0, min(inactive_ratio, 1.0))

        # Store raw probabilities (these are the pure calculated values)
        self.active_ratio = active_ratio
        self.inactive_ratio = inactive_ratio

        # Return the RAW values (weighting happens in properties)
        return active_ratio, inactive_ratio

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Convert likelihood to dictionary for storage."""
        return {
            # Store raw calculated values, not weighted ones
            "prob_given_true": self.prob_given_true_raw,
            "prob_given_false": self.prob_given_false_raw,
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
            # Store filtering statistics
            "total_on_intervals": self.total_on_intervals,
            "filtered_short_intervals": self.filtered_short_intervals,
            "filtered_long_intervals": self.filtered_long_intervals,
            "valid_intervals": self.valid_intervals,
            "max_filtered_duration_seconds": self.max_filtered_duration_seconds,
            # Store numeric sensor metadata
            "is_numeric_sensor": self.is_numeric_sensor,
            "statistics_based_calculation": getattr(
                self, "statistics_based_calculation", False
            ),
        }

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        coordinator: AreaOccupancyCoordinator,
        entity_id: str,
        active_states: list[str],
        default_prob_true: float,
        default_prob_false: float,
        weight: float,
    ) -> Likelihood:
        """Create likelihood from dictionary."""
        likelihood = cls(
            coordinator=coordinator,
            entity_id=entity_id,
            active_states=active_states,
            default_prob_true=default_prob_true,
            default_prob_false=default_prob_false,
            weight=weight,
        )
        # Now we're correctly storing raw values as raw
        likelihood.active_ratio = data["prob_given_true"]  # Raw values
        likelihood.inactive_ratio = data["prob_given_false"]  # Raw values
        likelihood.last_updated = (
            datetime.fromisoformat(data["last_updated"])
            if data["last_updated"]
            else None
        )
        # Load filtering statistics (with backward compatibility)
        likelihood.total_on_intervals = data.get("total_on_intervals")
        likelihood.filtered_short_intervals = data.get("filtered_short_intervals")
        likelihood.filtered_long_intervals = data.get("filtered_long_intervals")
        likelihood.valid_intervals = data.get("valid_intervals")
        likelihood.max_filtered_duration_seconds = data.get(
            "max_filtered_duration_seconds"
        )
        # Load numeric sensor metadata
        likelihood.is_numeric_sensor = data.get(
            "is_numeric_sensor", not bool(active_states)
        )
        likelihood.statistics_based_calculation = data.get(
            "statistics_based_calculation", False
        )
        return likelihood
