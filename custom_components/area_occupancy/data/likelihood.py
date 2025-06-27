"""Per-sensor likelihoods P(E|H) and P(E|¬H).

Computes *overlap* of each sensor's active intervals with the area's
ground-truth occupied intervals, giving informative likelihoods that
differ between H and ¬H.
"""

from __future__ import annotations

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


# ─────────────────────────────────────────────────────────────────────────────
class Likelihood:
    """Learn conditional probabilities for a single binary sensor."""

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
        return likelihood
