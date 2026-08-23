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
        ``slot_minutes`` resolution, and three parallel per-slot maps keyed by
        ``"day,slot"``:

        - ``slots``: the forecast probability, i.e. the slot's time prior
          combined with the area's global prior. This is the number to threshold
          against the area's occupancy threshold, but its dynamic range is
          compressed by the logit-space blend (see :func:`forecast_prior`).
        - ``slots_raw``: the learned time prior itself, uncombined, in
          ``[TIME_PRIOR_MIN_BOUND, TIME_PRIOR_MAX_BOUND]``. This is the term
          that actually carries the weekly *shape*, so it is what a consumer
          should use to rank hours against each other.
        - ``data_points``: weeks of observation behind each slot; ``0`` marks a
          slot that was never learned, whose prior is the neutral fallback and
          should not be read as a real probability.

        Assumes the area's time-prior cache is already warm (the caller
        pre-loads it off the event loop).
    """
    prior = area.prior
    matrix = prior.all_time_priors()
    points = prior.all_time_prior_points()
    slots: dict[str, float] = {}
    slots_raw: dict[str, float] = {}
    data_points: dict[str, int] = {}
    for day, slot in sorted(matrix):
        key = f"{day},{slot}"
        slots[key] = round(prior.prior_for(day, slot), FORECAST_RESPONSE_PRECISION)
        slots_raw[key] = round(matrix[(day, slot)], FORECAST_RESPONSE_PRECISION)
        data_points[key] = points.get((day, slot), 0)
    return {
        "area_id": area.config.area_id,
        "global_prior": prior.global_prior,
        "slot_minutes": slot_minutes,
        "slots": slots,
        "slots_raw": slots_raw,
        "data_points": data_points,
    }


def build_aggregate_time_priors(
    members: list[Area], slot_minutes: int, area_id: str, name: str
) -> dict[str, Any] | None:
    """Build the learned weekly forecast for an aggregate zone.

    Aggregate zones (the "All Areas" device and per-floor devices) have no
    stored priors of their own; their occupancy is derived from member areas.
    This mirrors ``AllAreas.area_prior()`` — a clamped average across members —
    but per weekly slot, producing a whole-floor / whole-home occupancy forecast
    useful for air-based devices that condition several rooms at once.

    Returns ``None`` when there are no members (nothing to aggregate). Assumes
    each member's time-prior cache is already warm.
    """
    if not members:
        return None
    keys: set[tuple[int, int]] = set()
    for member in members:
        keys.update(member.prior.all_time_priors())
    member_matrices = [m.prior.all_time_priors() for m in members]
    member_points = [m.prior.all_time_prior_points() for m in members]
    slots: dict[str, float] = {}
    slots_raw: dict[str, float] = {}
    data_points: dict[str, int] = {}
    for day, slot in sorted(keys):
        key = f"{day},{slot}"
        values = [member.prior.prior_for(day, slot) for member in members]
        avg = sum(values) / len(values)
        slots[key] = round(
            max(MIN_PRIOR, min(MAX_PRIOR, avg)), FORECAST_RESPONSE_PRECISION
        )
        raw = [m[(day, slot)] for m in member_matrices if (day, slot) in m]
        slots_raw[key] = round(
            sum(raw) / len(raw) if raw else 0.0, FORECAST_RESPONSE_PRECISION
        )
        # Weakest member wins: the zone is only as well-learned as its least
        # observed room, so a consumer never over-trusts a mixed aggregate.
        per_member = [p.get((day, slot), 0) for p in member_points]
        data_points[key] = min(per_member) if per_member else 0
    return {
        "area_id": area_id,
        "name": name,
        "aggregate": True,
        "members": [m.config.area_id for m in members if m.config.area_id],
        "slot_minutes": slot_minutes,
        "slots": slots,
        "slots_raw": slots_raw,
        "data_points": data_points,
    }
