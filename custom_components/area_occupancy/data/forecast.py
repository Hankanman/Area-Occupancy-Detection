"""Read-side forecast helpers over the learned occupancy time priors.

Kept separate from :mod:`.prior` so that exposing the learned weekly priors to
other integrations touches the core model as little as possible: the forecast
math and the service-response shaping live here, while the :class:`.prior.Prior`
class keeps only thin cache accessors (``all_time_priors`` / ``prior_for``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..const import MAX_PRIOR, MIN_PRIOR
from ..utils import combine_priors

if TYPE_CHECKING:
    from ..area.area import Area

# Decimal places used when rounding forecast priors in service responses.
FORECAST_RESPONSE_PRECISION = 4


def forecast_prior(
    global_prior: float | None,
    slot_time_prior: float,
    *,
    prior_factor: float,
) -> float:
    """Return the occupancy-probability forecast for a single weekly slot.

    Combines the learned ``global_prior`` with the slot's learned time prior
    and clamps to ``[MIN_PRIOR, MAX_PRIOR]`` — the same combination the live
    prior applies, minus the configuration floors (which are threshold-relative
    and not meaningful to project onto arbitrary future slots). When
    ``global_prior`` is unknown, the slot's own (bounds-clamped) time prior is
    returned as a best-effort fallback.

    Args:
        global_prior: Learned area-wide prior, or ``None`` if not learned yet.
        slot_time_prior: The learned time prior for the target slot.
        prior_factor: Multiplicative boost applied before clamping (mirrors
            :data:`.prior.PRIOR_FACTOR`); passed in to keep this module free of
            any import from :mod:`.prior`.

    Returns:
        Forecast occupancy probability in ``[MIN_PRIOR, MAX_PRIOR]``.
    """
    if global_prior is None:
        return max(MIN_PRIOR, min(MAX_PRIOR, slot_time_prior))
    adjusted = combine_priors(global_prior, slot_time_prior) * prior_factor
    return max(MIN_PRIOR, min(MAX_PRIOR, adjusted))


def build_area_time_priors(area: Area, slot_minutes: int) -> dict[str, Any]:
    """Shape one area's learned weekly forecast for the get_time_priors service.

    Exposes all learned weekly slots (``day_of_week`` × ``time_slot``) so an
    external consumer can read the occupancy prior for *future* slots and build
    a forward-looking occupancy profile — something the per-area
    ``occupancy_probability`` sensor (a current-state estimate) cannot do.

    Args:
        area: The area whose learned priors should be exported.
        slot_minutes: Slot resolution echoed into the response.

    Returns:
        Dict with the area's ``area_id``, learned ``global_prior``, the
        ``slot_minutes`` resolution, and ``slots`` — a mapping of ``"day,slot"``
        to the forecast occupancy probability for that slot. Assumes the area's
        time-prior cache is already warm (the caller pre-loads it off the event
        loop).
    """
    prior = area.prior
    matrix = prior.all_time_priors()
    slots = {
        f"{day},{slot}": round(prior.prior_for(day, slot), FORECAST_RESPONSE_PRECISION)
        for day, slot in sorted(matrix)
    }
    return {
        "area_id": area.config.area_id,
        "global_prior": prior.global_prior,
        "slot_minutes": slot_minutes,
        "slots": slots,
    }
