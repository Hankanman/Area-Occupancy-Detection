"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from ..const import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TIME_PRIOR,
    MAX_PRIOR,
    MAX_PROBABILITY,
    MIN_PRIOR,
    MIN_PROBABILITY,
    TIME_PRIOR_MAX_BOUND,
    TIME_PRIOR_MIN_BOUND,
)
from ..data.analysis import PriorAnalyzer
from ..utils import clamp_probability, combine_priors

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Prior calculation constants
PRIOR_FACTOR = 1.05
DEFAULT_PRIOR = 0.5
SIGNIFICANT_CHANGE_THRESHOLD = 0.1

# Time slot constants
DEFAULT_SLOT_MINUTES = 60


class Prior:
    """Compute the baseline probability for an Area entity."""

    def __init__(
        self, coordinator: AreaOccupancyCoordinator, area_name: str | None = None
    ) -> None:
        """Initialize the Prior class.

        Args:
            coordinator: The coordinator instance
            area_name: Optional area name for multi-area support
        """
        self.coordinator = coordinator
        self.db = coordinator.db
        self.area_name = area_name
        # Validate area_name and retrieve config from coordinator.areas
        if not area_name:
            raise ValueError("Area name is required in multi-area architecture")
        if area_name not in coordinator.areas:
            available = list(coordinator.areas.keys())
            raise ValueError(
                f"Area '{area_name}' not found. "
                f"Available areas: {available if available else '(none)'}"
            )
        self.config = coordinator.areas[area_name].config
        self.sensor_ids = self.config.sensors.motion
        self.media_sensor_ids = self.config.sensors.media
        self.appliance_sensor_ids = self.config.sensors.appliance
        self.hass = coordinator.hass
        self.global_prior: float | None = None
        self._last_updated: datetime | None = None
        # Cache for time_prior
        self._cached_time_prior: float | None = None
        self._cached_time_prior_day: int | None = None
        self._cached_time_prior_slot: int | None = None
        # Cache for occupied intervals
        self._cached_occupied_intervals: list[tuple[datetime, datetime]] | None = None
        self._cached_intervals_timestamp: datetime | None = None

    @property
    def value(self) -> float:
        """Return the current prior value or minimum if not calculated."""
        if self.global_prior is None:
            return MIN_PRIOR

        # Use global_prior directly if time_prior is None, otherwise combine them
        if self.time_prior is None:
            prior = self.global_prior
        else:
            prior = combine_priors(self.global_prior, self.time_prior)

        # Track if we needed to clamp the prior
        was_clamped = False

        # Validate that prior is within reasonable bounds before applying factor
        if not (MIN_PROBABILITY <= prior <= MAX_PROBABILITY):
            _LOGGER.warning(
                "Prior %.10f is outside valid range [%.10f, %.10f], clamping to bounds",
                prior,
                MIN_PROBABILITY,
                MAX_PROBABILITY,
            )
            prior = clamp_probability(prior)
            was_clamped = True

        # Apply factor and clamp to bounds
        adjusted_prior = prior * PRIOR_FACTOR

        # If the prior was clamped to bounds, return the clamped prior value
        if was_clamped and prior == MIN_PROBABILITY:
            result = MIN_PRIOR
        elif was_clamped and prior == MAX_PROBABILITY:
            result = MAX_PRIOR
        else:
            result = max(MIN_PRIOR, min(MAX_PRIOR, adjusted_prior))

        # Apply minimum prior override if configured
        if self.config.min_prior_override > 0.0:
            original_result = result
            if result < self.config.min_prior_override:
                result = self.config.min_prior_override
                _LOGGER.debug(
                    "Applied minimum prior override in runtime: %.4f -> %.4f",
                    original_result,
                    result,
                )

        return result

    @property
    def time_prior(self) -> float:
        """Return the current time prior value or minimum if not calculated."""
        current_day = self.day_of_week
        current_slot = self.time_slot

        # Check if we have a valid cache for the current time slot
        if (
            self._cached_time_prior is not None
            and self._cached_time_prior_day == current_day
            and self._cached_time_prior_slot == current_slot
        ):
            return self._cached_time_prior

        # Cache miss - get from database and cache the result
        self._cached_time_prior = self.get_time_prior()
        self._cached_time_prior_day = current_day
        self._cached_time_prior_slot = current_slot

        return self._cached_time_prior

    @property
    def day_of_week(self) -> int:
        """Return the current day of week (0=Monday, 6=Sunday)."""
        return dt_util.utcnow().weekday()

    @property
    def time_slot(self) -> int:
        """Return the current time slot based on DEFAULT_SLOT_MINUTES."""
        now = dt_util.utcnow()
        return (now.hour * 60 + now.minute) // DEFAULT_SLOT_MINUTES

    @property
    def last_updated(self) -> datetime | None:
        """Return the last updated timestamp."""
        return self._last_updated

    def set_global_prior(self, prior: float) -> None:
        """Set the global prior value."""
        self.global_prior = prior
        self._invalidate_time_prior_cache()
        self._invalidate_occupied_intervals_cache()
        self._last_updated = dt_util.utcnow()

    def _invalidate_time_prior_cache(self) -> None:
        """Invalidate the time_prior cache."""
        self._cached_time_prior = None
        self._cached_time_prior_day = None
        self._cached_time_prior_slot = None

    def _invalidate_occupied_intervals_cache(self) -> None:
        """Invalidate the occupied intervals cache."""
        self._cached_occupied_intervals = None
        self._cached_intervals_timestamp = None

    def clear_cache(self) -> None:
        """Clear all cached data to release memory.

        This should be called when the area is being removed or cleaned up
        to prevent memory leaks from cached data holding references.
        """
        _LOGGER.debug("Clearing all caches for area: %s", self.area_name)
        self._invalidate_time_prior_cache()
        self._invalidate_occupied_intervals_cache()
        # Also clear global_prior and last_updated to release references
        self.global_prior = None
        self._last_updated = None

    def get_time_prior(self) -> float:
        """Get the time prior for the current time slot.

        Returns:
            float: Time prior value or default 0.5 if not found

        """
        _LOGGER.debug("Getting time prior")
        time_prior = self.db.get_time_prior(
            area_name=self.area_name,
            day_of_week=self.day_of_week,
            time_slot=self.time_slot,
            default_prior=DEFAULT_TIME_PRIOR,
        )

        # Clamp the retrieved time prior between safety bounds
        # This prevents extreme values (like 0.01 or 0.99) from disproportionately
        # affecting the global prior, while still allowing meaningful influence.
        # We bound it to [0.1, 0.9] to avoid "black hole" probabilities.
        return max(TIME_PRIOR_MIN_BOUND, min(TIME_PRIOR_MAX_BOUND, time_prior))

    def get_occupied_intervals(
        self,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        include_media: bool = False,
        include_appliance: bool = False,
    ) -> list[tuple[datetime, datetime]]:
        """Get occupied time intervals from motion sensors using unified logic.

        This method provides a single source of truth for determining occupancy
        intervals that can be used by both prior and likelihood calculations.
        Delegates to PriorAnalyzer for the actual calculation but maintains caching.

        Note: For prior calculations, include_media and include_appliance should
        always be False. Prior calculations are exclusively based on motion/presence
        sensors to ensure consistent ground truth.

        Args:
            lookback_days: Number of days to look back for interval data (default: 90)
            include_media: If True, include media player intervals (playing state only)
            include_appliance: If True, include appliance intervals (on state only)

        Returns:
            List of (start_time, end_time) tuples representing occupied periods

        """
        # Check cache first (only for motion-only queries to keep cache simple)
        now = dt_util.utcnow()
        if (
            not include_media
            and not include_appliance
            and self._cached_occupied_intervals is not None
            and self._cached_intervals_timestamp is not None
            and (now - self._cached_intervals_timestamp).total_seconds()
            < DEFAULT_CACHE_TTL_SECONDS
        ):
            _LOGGER.debug(
                "Returning cached occupied intervals (%d intervals)",
                len(self._cached_occupied_intervals),
            )
            return self._cached_occupied_intervals

        # Delegate to analyzer for calculation
        analyzer = PriorAnalyzer(self.coordinator, self.area_name)
        intervals = analyzer.get_occupied_intervals(
            lookback_days=lookback_days,
            include_media=include_media,
            include_appliance=include_appliance,
        )

        # Cache the result only if motion-only
        if not include_media and not include_appliance:
            self._cached_occupied_intervals = intervals
            self._cached_intervals_timestamp = now

        return intervals

    def to_dict(self) -> dict[str, Any]:
        """Convert prior to dictionary for storage."""
        return {
            "value": self.global_prior,
            "last_updated": (
                self._last_updated.isoformat() if self._last_updated else None
            ),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: AreaOccupancyCoordinator, area_name: str
    ) -> Prior:
        """Create prior from dictionary."""
        prior = cls(coordinator, area_name=area_name)
        prior.global_prior = data["value"]
        prior._last_updated = (
            datetime.fromisoformat(data["last_updated"])
            if data["last_updated"]
            else None
        )
        return prior
