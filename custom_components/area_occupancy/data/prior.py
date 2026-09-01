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
    MIN_PRIOR,
    PRIOR_FLOOR_THRESHOLD_MARGIN,
    TIME_PRIOR_MAX_BOUND,
    TIME_PRIOR_MIN_BOUND,
)
from ..time_utils import to_local
from ..utils import clamp_probability, combine_priors
from .forecast import forecast_prior

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from .config import AreaConfig

_LOGGER = logging.getLogger(__name__)

# Sentinel distinguishing "caller didn't pass calculation_date" (fresh
# calculation just completed -> default to now) from "caller explicitly
# passed calculation_date=None" (loaded a legacy DB row with no recorded
# timestamp -> must stay None, not silently become now). `None` itself
# can't serve as the "not passed" default since it's also the valid explicit
# value for the legacy case.
_CALCULATION_DATE_UNSET = object()

# Prior calculation constants
PRIOR_FACTOR = 1.0
DEFAULT_PRIOR = 0.5
SIGNIFICANT_CHANGE_THRESHOLD = 0.1

# Time slot constants
DEFAULT_SLOT_MINUTES = 60


class Prior:
    """Compute the baseline probability for an Area entity."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str | None = None,
        config: AreaConfig | None = None,
    ) -> None:
        """Initialize the Prior class.

        Args:
            coordinator: The coordinator instance
            area_name: Optional area name for multi-area support
            config: Area configuration (preferred). Falls back to coordinator lookup.
        """
        self.coordinator = coordinator
        self.db = coordinator.db
        self.area_name = area_name
        if config is not None:
            self.config = config
        else:
            area = coordinator.get_area(area_name)
            if area is None:
                raise ValueError(
                    f"Area '{self.area_name}' not found in coordinator and no config provided"
                )
            self.config = area.config
        self.sensor_ids = self.config.sensors.motion
        self.media_sensor_ids = self.config.sensors.media
        self.appliance_sensor_ids = self.config.sensors.appliance
        self.hass = coordinator.hass
        self.global_prior: float | None = None
        self._last_updated: datetime | None = None
        # Timestamp of the most recent *successful* global-prior calculation
        # (set by ``set_global_prior``), as opposed to an analysis cycle that
        # skipped the update entirely because no ground-truth data exists
        # yet (#520 Bug A). Loaded from ``GlobalPriors.calculation_date`` on
        # startup so staleness detection survives a restart; see
        # ``data.health.HealthMonitor._check_insufficient_priors``.
        self.last_calculation_at: datetime | None = None
        # Cache for all 168 time priors: (day_of_week, time_slot) -> prior_value
        self._cached_time_priors: dict[tuple[int, int], float] | None = None
        # Sample counts for the same slots: (day_of_week, time_slot) -> weeks
        # of data behind the value. 0 means the slot was never learned and the
        # cached value is the neutral fallback, not an observation.
        self._cached_time_prior_points: dict[tuple[int, int], int] | None = None

    @property
    def value(self) -> float:
        """Return the current prior value or minimum if not calculated.

        The prior is calculated by combining global_prior and time_prior,
        applying PRIOR_FACTOR boost, and clamping to [MIN_PRIOR, MAX_PRIOR].

        Floors (purpose.min_prior, config.min_prior_override) can raise the
        learned prior, but they are capped strictly below the configured
        occupancy threshold so that a floor alone cannot hold an area above
        the threshold with no active evidence (see issue #435). Learned
        priors — which reflect real historical occupancy — are allowed to
        exceed the threshold.

        Returns:
            Prior probability in range [MIN_PRIOR, MAX_PRIOR].
        """
        return self._compute_value_and_floor()[0]

    def _compute_value_and_floor(self) -> tuple[float, str]:
        """Compute prior.value and report which floor (if any) applied.

        Returns:
            Tuple of (prior value, floor label). Floor label is one of
            ``"none"``, ``"purpose"``, ``"override"``. The label reflects the
            floor responsible for raising the value above the learned prior,
            or ``"none"`` if the learned prior is already at or above every
            floor.
        """
        if self.global_prior is None:
            learned = MIN_PRIOR
        else:
            if self.time_prior is None:
                prior = self.global_prior
            else:
                prior = combine_priors(self.global_prior, self.time_prior)

            adjusted_prior = prior * PRIOR_FACTOR
            learned = max(MIN_PRIOR, min(MAX_PRIOR, adjusted_prior))

        purpose_floor = 0.0
        area = self.coordinator.areas.get(self.area_name)
        if area is not None and area.purpose.min_prior > 0.0:
            purpose_floor = area.purpose.min_prior

        override_floor = 0.0
        if self.config.min_prior_override > 0.0:
            override_floor = self.config.min_prior_override

        floor_cap = max(MIN_PRIOR, self.config.threshold - PRIOR_FLOOR_THRESHOLD_MARGIN)
        capped_purpose = min(purpose_floor, floor_cap)
        capped_override = min(override_floor, floor_cap)

        result = learned
        applied = "none"
        if capped_purpose > result:
            result = capped_purpose
            applied = "purpose"
        if capped_override > result:
            result = capped_override
            applied = "override"

        return result, applied

    def diagnostic_snapshot(self) -> dict[str, float | str | None]:
        """Return a snapshot of the prior's current inputs and output.

        Exposed to users via the probability sensor's extra_state_attributes
        so they can see which term is driving the prior — especially useful
        when an area appears "stuck" occupied with no active evidence.

        Returns:
            Dict with learned components, the floor that was applied (if
            any), the effective prior value, and the configured threshold.
        """
        value, applied = self._compute_value_and_floor()
        return {
            "prior_value": value,
            "global_prior": self.global_prior,
            "time_prior": self.time_prior,
            "min_prior_floor_applied": applied,
            "threshold": self.config.threshold,
        }

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
        return self._cached_time_priors.get(slot_key, self.unlearned_slot_prior)

    @property
    def day_of_week(self) -> int:
        """Return the current day of week (0=Monday, 6=Sunday)."""
        return to_local(dt_util.utcnow()).weekday()

    @property
    def time_slot(self) -> int:
        """Return the current time slot based on DEFAULT_SLOT_MINUTES."""
        now = to_local(dt_util.utcnow())
        return (now.hour * 60 + now.minute) // DEFAULT_SLOT_MINUTES

    def all_time_priors(self) -> dict[tuple[int, int], float]:
        """Return a copy of all learned weekly time priors.

        The cache is loaded from the database on first access. Values are the
        raw per-slot time priors, bounds-clamped to
        ``[TIME_PRIOR_MIN_BOUND, TIME_PRIOR_MAX_BOUND]`` (see
        :meth:`_load_time_priors`). Keys are ``(day_of_week, time_slot)``
        with ``day_of_week`` 0=Monday…6=Sunday and ``time_slot`` in
        ``[0, 1440 // DEFAULT_SLOT_MINUTES)``.

        Unlike :attr:`time_prior`, which is locked to the current wall-clock
        slot, this exposes the full weekly matrix so a consumer can read
        occupancy priors for *arbitrary* (including future) slots.

        Returns:
            Mapping of ``(day_of_week, time_slot) -> raw time prior``.
        """
        if self._cached_time_priors is None:
            self._load_time_priors()
        return dict(self._cached_time_priors)

    def all_time_prior_points(self) -> dict[tuple[int, int], int]:
        """Return the sample count behind each weekly slot.

        Keys match :meth:`all_time_priors`. A value of ``0`` means the slot has
        never been learned and its prior is the neutral fallback rather than an
        observation — consumers should render or weight it differently instead
        of treating it as a real probability.

        Returns:
            Dict mapping ``(day_of_week, time_slot)`` to the number of distinct
            weeks of data behind that slot.
        """
        if self._cached_time_prior_points is None:
            self._load_time_priors()
        return dict(self._cached_time_prior_points or {})

    def prior_for(self, day_of_week: int, time_slot: int) -> float:
        """Return the learned occupancy-probability forecast for a given slot.

        Unlike :attr:`time_prior` (locked to the current slot) and
        :attr:`value` (evaluated for "now" and subject to configuration
        floors), this computes the forecast for any weekly slot so a
        consumer can build a forward-looking occupancy profile — for
        example to pre-heat a room before its habitual occupancy.

        The value combines the learned ``global_prior`` with the slot's
        learned time prior and clamps to ``[MIN_PRIOR, MAX_PRIOR]``, mirroring
        the learned term of :attr:`value`. Configuration floors
        (``purpose.min_prior``, ``min_prior_override``) are intentionally
        *not* applied: they are threshold-relative safety nets for the live
        estimate and are not meaningful to project onto arbitrary future
        slots. When ``global_prior`` has not been learned yet, the slot's
        raw (bounds-clamped) time prior is returned as a best-effort fallback.

        Args:
            day_of_week: 0=Monday … 6=Sunday.
            time_slot: Slot index ``(hour * 60 + minute) // DEFAULT_SLOT_MINUTES``.

        Returns:
            Forecast occupancy probability in ``[MIN_PRIOR, MAX_PRIOR]``.
        """
        if self._cached_time_priors is None:
            self._load_time_priors()
        slot_time_prior = self._cached_time_priors.get(
            (day_of_week, time_slot), self.unlearned_slot_prior
        )
        return forecast_prior(
            self.global_prior, slot_time_prior, prior_factor=PRIOR_FACTOR
        )

    def set_global_prior(
        self,
        prior: float,
        *,
        calculation_date: datetime | None = _CALCULATION_DATE_UNSET,  # type: ignore[assignment]
    ) -> None:
        """Set the global prior value.

        The prior is clamped to [MIN_PROBABILITY, MAX_PROBABILITY] to ensure
        valid probability bounds even when loading from database or external sources.

        Args:
            prior: The prior probability value (will be clamped to valid bounds)
            calculation_date: When this value was actually computed. Omitting
                this argument defaults to now (the normal case: a fresh
                calculation just completed). The data-load path passes the
                persisted ``GlobalPriors.calculation_date`` instead, so
                ``last_calculation_at`` reflects when the prior was last
                *recomputed*, not when it was last *loaded* — otherwise a
                restart would reset the staleness clock and mask a frozen
                prior (#520 Bug A) until another full grace period elapsed.
                An explicit ``None`` (a legacy DB row with no recorded
                timestamp) is preserved as ``None`` rather than defaulting to
                now — coercing it would make the legacy case indistinguishable
                from a fresh calculation and defeat the staleness check
                entirely.
        """
        self.global_prior = clamp_probability(prior)
        self._invalidate_time_prior_cache()
        now = dt_util.utcnow()
        self._last_updated = now
        self.last_calculation_at = (
            now if calculation_date is _CALCULATION_DATE_UNSET else calculation_date
        )

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
        self.last_calculation_at = None

    def invalidate_time_prior_cache(self) -> None:
        """Drop the cached weekly priors so the next read reloads from the DB.

        Public because the analysis pipeline must invalidate *after* it writes
        new priors, not only when the global prior changes.
        """
        self._invalidate_time_prior_cache()

    def _invalidate_time_prior_cache(self) -> None:
        """Invalidate the time_prior cache."""
        self._cached_time_priors = None
        self._cached_time_prior_points = None

    @property
    def unlearned_slot_prior(self) -> float:
        """Return the value to use for a slot that was never learned.

        A slot with no stored row means "no observation", which is *not* the
        same as "empty". Filling it with :data:`DEFAULT_TIME_PRIOR` (0.5) makes
        the unknown outrank every genuinely low-occupancy slot and silently
        inflates the live prior — an area with ``global_prior = 0.04`` was
        being pushed to ~0.12 purely by unlearned slots.

        The area's own ``global_prior`` is the neutral choice: it is the
        identity of :func:`combine_priors` (combining a prior with itself
        returns it unchanged), so an unlearned slot contributes no opinion in
        either direction. Before any global prior exists there is nothing
        neutral to fall back to, so the historical default is kept.

        Returns:
            The fallback time prior for unlearned slots.
        """
        if self.global_prior is None:
            return DEFAULT_TIME_PRIOR
        return max(TIME_PRIOR_MIN_BOUND, min(TIME_PRIOR_MAX_BOUND, self.global_prior))

    def _load_time_priors(self) -> None:
        """Load all 168 time priors from database into cache.

        Reads the stored slots in a single query and fills the rest of the
        weekly grid with :attr:`unlearned_slot_prior`, keeping a parallel map
        of sample counts so callers can tell learned slots from filled ones.
        """
        stored = self.db.get_stored_time_priors(area_name=self.area_name)
        fallback = self.unlearned_slot_prior

        priors: dict[tuple[int, int], float] = {}
        points: dict[tuple[int, int], int] = {}
        slots_per_day = 1440 // DEFAULT_SLOT_MINUTES
        for day_of_week in range(7):
            for time_slot in range(slots_per_day):
                slot_key = (day_of_week, time_slot)
                record = stored.get(slot_key)
                if record is None:
                    priors[slot_key] = fallback
                    points[slot_key] = 0
                    continue
                prior_value, data_points = record
                priors[slot_key] = max(
                    TIME_PRIOR_MIN_BOUND,
                    min(TIME_PRIOR_MAX_BOUND, prior_value),
                )
                points[slot_key] = data_points

        self._cached_time_priors = priors
        self._cached_time_prior_points = points
