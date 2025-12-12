"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from ..const import (
    DEFAULT_TIME_PRIOR,
    MAX_PRIOR,
    MAX_PROBABILITY,
    MIN_PRIOR,
    MIN_PROBABILITY,
    TIME_PRIOR_MAX_BOUND,
    TIME_PRIOR_MIN_BOUND,
)
from ..time_utils import to_local
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
        self.config = coordinator.areas[area_name].config
        self.sensor_ids = self.config.sensors.motion
        self.media_sensor_ids = self.config.sensors.media
        self.appliance_sensor_ids = self.config.sensors.appliance
        self.hass = coordinator.hass
        self.global_prior: float | None = None
        self._last_updated: datetime | None = None
        # Cache for all 168 time priors: (day_of_week, time_slot) -> prior_value
        self._cached_time_priors: dict[tuple[int, int], float] | None = None

    @property
    def value(self) -> float:
        """Return the current prior value or minimum if not calculated."""
        # Initialize result to MIN_PRIOR if global_prior is None
        # This allows min_prior_override to be applied even when prior hasn't been calculated
        if self.global_prior is None:
            result = MIN_PRIOR
        else:
            # Use global_prior directly if time_prior is None, otherwise combine them
            if self.time_prior is None:
                prior = self.global_prior
            else:
                prior = combine_priors(self.global_prior, self.time_prior)

            # Track if we needed to clamp the prior
            was_clamped = False

            # Validate that prior is within reasonable bounds before applying factor
            if not (MIN_PROBABILITY <= prior <= MAX_PROBABILITY):
                prior = clamp_probability(prior)
                was_clamped = True

            # Apply factor and clamp to bounds
            adjusted_prior = prior * PRIOR_FACTOR

            # If the prior was clamped to bounds, use the clamped prior value
            if was_clamped and prior == MIN_PROBABILITY:
                result = MIN_PRIOR
            elif was_clamped and prior == MAX_PROBABILITY:
                result = MAX_PRIOR
            else:
                result = max(MIN_PRIOR, min(MAX_PRIOR, adjusted_prior))

        # Apply minimum prior override if configured
        # This check must run for all code paths, including when global_prior is None
        if self.config.min_prior_override > 0.0:
            result = max(result, self.config.min_prior_override)

        return result

    @property
    def time_prior(self) -> float:
        """Return the current time prior value or minimum if not calculated."""
        # Load all time priors if cache is empty
        if self._cached_time_priors is None:
            self._load_time_priors()

        current_day = self.day_of_week
        current_slot = self.time_slot
        slot_key = (current_day, current_slot)

        # Get from cache (guaranteed to exist after _load_time_priors)
        return self._cached_time_priors.get(slot_key, DEFAULT_TIME_PRIOR)

    @property
    def day_of_week(self) -> int:
        """Return the current day of week (0=Monday, 6=Sunday)."""
        return to_local(dt_util.utcnow()).weekday()

    @property
    def time_slot(self) -> int:
        """Return the current time slot based on DEFAULT_SLOT_MINUTES."""
        now = to_local(dt_util.utcnow())
        return (now.hour * 60 + now.minute) // DEFAULT_SLOT_MINUTES

    def set_global_prior(self, prior: float) -> None:
        """Set the global prior value."""
        self.global_prior = prior
        self._invalidate_time_prior_cache()
        self._last_updated = dt_util.utcnow()

    def clear_cache(self) -> None:
        """Clear all cached data to release memory.

        This should be called when the area is being removed or cleaned up
        to prevent memory leaks from cached data holding references.
        """
        _LOGGER.debug("Clearing all caches for area: %s", self.area_name)
        self._invalidate_time_prior_cache()
        # Also clear global_prior and last_updated to release references
        self.global_prior = None
        self._last_updated = None

    def _invalidate_time_prior_cache(self) -> None:
        """Invalidate the time_prior cache."""
        self._cached_time_priors = None

    def _load_time_priors(self) -> None:
        """Load all 168 time priors from database into cache.

        This method loads time priors for the area in a single database query,
        eliminating the need for individual queries when accessing time priors.
        """
        self._cached_time_priors = self.db.get_all_time_priors(
            area_name=self.area_name,
            default_prior=DEFAULT_TIME_PRIOR,
        )
        # Apply safety bounds to all cached values
        for slot_key, prior_value in self._cached_time_priors.items():
            self._cached_time_priors[slot_key] = max(
                TIME_PRIOR_MIN_BOUND,
                min(TIME_PRIOR_MAX_BOUND, prior_value),
            )
