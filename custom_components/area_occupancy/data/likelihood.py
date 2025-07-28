"""Per-sensor likelihoods P(E|H) and P(E|¬H).

Computes *overlap* of each sensor's active intervals with the area's
ground-truth occupied intervals, giving informative likelihoods that
differ between H and ¬H.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from ..const import HA_RECORDER_DAYS
from ..state_intervals import StateInterval
from ..utils import ensure_timezone_aware, validate_prob

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


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
        self.days = HA_RECORDER_DAYS
        self.hass = coordinator.hass
        self.coordinator = coordinator
        self.last_updated: datetime | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.intervals: list[StateInterval] | None = None
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
        """Return the probability of the sensor being active given the area is occupied."""

        calculated_prob = (
            self.active_ratio
            if self.active_ratio is not None
            else self.default_prob_true
        )

        return validate_prob(calculated_prob)

    @property
    def prob_given_false(self) -> float:
        """Return the probability of the sensor being active given the area is not occupied."""

        calculated_prob = (
            self.inactive_ratio
            if self.inactive_ratio is not None
            else self.default_prob_false
        )

        return validate_prob(calculated_prob)

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

    async def update(self) -> tuple[float, float]:
        """Return a likelihood, always re-computing (cache is always stale/forced).

        Returns:
            Tuple of (prob_given_true, prob_given_false) weighted values

        """
        # Always recalculate
        _LOGGER.debug(
            "Recalculating likelihood for %s (cache invalid or stale)", self.entity_id
        )

        try:
            active_ratio, inactive_ratio = await self.calculate()
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

    async def calculate(self) -> tuple[float, float]:
        """Calculate the likelihood with anomaly filtering, considering only intervals within prior_intervals."""
        _LOGGER.debug("Likelihood calculation for %s", self.entity_id)
        self.start_time = dt_util.utcnow() - timedelta(days=self.days)
        self.end_time = dt_util.utcnow()

        # Use only our DB for interval retrieval
        intervals = await self.coordinator.storage.get_historical_intervals(
            self.entity_id,
            self.start_time,
            self.end_time,
        )

        prior_intervals = self.coordinator.prior.prior_intervals

        active_ratio = self.default_prob_true
        inactive_ratio = self.default_prob_false

        # Debug logging
        _LOGGER.debug(
            "Likelihood calculation for %s: intervals=%d, prior_intervals=%d",
            self.entity_id,
            len(intervals) if intervals else 0,
            len(prior_intervals) if prior_intervals else 0,
        )

        # If prior intervals are not available, use default values
        if not prior_intervals:
            _LOGGER.debug(
                "Prior intervals not available for %s, using default likelihoods",
                self.entity_id,
            )
            return active_ratio, inactive_ratio

        if intervals and prior_intervals:
            # Calculate total analysis period
            total_seconds = (self.end_time - self.start_time).total_seconds()

            # Calculate occupied time (sum of prior intervals) - more efficient
            occupied_seconds = sum(
                (interval["end"] - interval["start"]).total_seconds()
                for interval in prior_intervals
            )

            # Calculate not-occupied time
            not_occupied_seconds = total_seconds - occupied_seconds

            if occupied_seconds > 0 and not_occupied_seconds > 0:
                # Optimize overlap calculation by pre-sorting and using binary search approach
                # Sort prior intervals by start time for more efficient overlap checking
                sorted_prior_intervals = sorted(
                    prior_intervals, key=lambda x: x["start"]
                )

                # Split filtered intervals into occupied and not-occupied periods
                occupied_active_seconds = 0
                not_occupied_active_seconds = 0

                # Process intervals in chunks to avoid blocking
                chunk_size = 50
                for i, interval in enumerate(intervals):
                    duration = (interval["end"] - interval["start"]).total_seconds()

                    if interval["state"] in self.active_states:
                        # More efficient overlap check using sorted intervals
                        if self._interval_overlaps_prior_optimized(
                            interval, sorted_prior_intervals
                        ):
                            occupied_active_seconds += duration
                        else:
                            not_occupied_active_seconds += duration

                    # Yield control periodically to avoid blocking
                    if i % chunk_size == 0:
                        await asyncio.sleep(0)

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

    def _interval_overlaps_prior_optimized(
        self, interval: StateInterval, sorted_prior_intervals: list[StateInterval]
    ) -> bool:
        """Optimized overlap check using sorted prior intervals.

        Args:
            interval: The interval to check
            sorted_prior_intervals: Prior intervals sorted by start time

        Returns:
            True if interval overlaps any prior interval

        """

        interval_start = ensure_timezone_aware(interval["start"])
        interval_end = ensure_timezone_aware(interval["end"])

        # Binary search approach for better performance with many prior intervals
        for prior in sorted_prior_intervals:
            prior_start = ensure_timezone_aware(prior["start"])
            prior_end = ensure_timezone_aware(prior["end"])

            # Early exit if we've passed all possible overlaps
            if prior_start > interval_end:
                break

            # Check for overlap
            if interval_end > prior_start and interval_start < prior_end:
                return True

        return False

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

        return likelihood
