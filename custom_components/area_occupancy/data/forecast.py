"""Read-side forecast helpers over the learned occupancy time priors.

Kept separate from :mod:`.prior` so that exposing the learned weekly priors to
other integrations touches the core model as little as possible: the forecast
math and the service-response shaping live here, while the :class:`.prior.Prior`
class keeps only thin cache accessors (``all_time_priors`` / ``prior_for``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ..const import MAX_PRIOR, MIN_PRIOR, TIME_PRIOR_PSEUDO_WEEKS
from ..utils import clamp_probability, combine_priors

if TYPE_CHECKING:
    from ..area.area import Area
    from .types import ZonePriors

# Decimal places used when rounding forecast priors in service responses.
FORECAST_RESPONSE_PRECISION = 4

# Persistence of current evidence into future slots.
#
# ``Purpose.half_life`` is an *evidence* decay constant in seconds (45s for a
# passageway, 1200s for a bedroom): it says how fast a motion event stops
# justifying the live estimate, not how long a room stays occupied. Used
# directly it would give tau < 0.35 slots everywhere, collapsing every future
# slot back onto the baseline and making the conditioned forecast pointless.
#
# So the half-life is used as an *ordering* — its ranking across purposes is
# exactly right, a corridor empties fast and a bedroom does not — remapped
# monotonically onto a plausible persistence range. This is a heuristic, not a
# measurement; PERSISTENCE_TAU_DIVISOR is the knob, and learning tau from the
# median occupied-run length per area is the principled successor.
PERSISTENCE_TAU_DIVISOR = 600.0
PERSISTENCE_TAU_MIN = 0.5
PERSISTENCE_TAU_MAX = 3.0


def persistence_tau_slots(purpose_half_life: float, slot_minutes: int) -> float:
    """Return how many slots the current evidence keeps influencing the forecast.

    Args:
        purpose_half_life: The area purpose's evidence half-life, in seconds.
        slot_minutes: Slot resolution, used to keep tau expressed in slots when
            the resolution is not the default 60 minutes.

    Returns:
        Tau in slots, clamped to ``[PERSISTENCE_TAU_MIN, PERSISTENCE_TAU_MAX]``.
        With hourly slots: passageway 0.5, working 1.0, sleeping 2.0.
    """
    tau = purpose_half_life / PERSISTENCE_TAU_DIVISOR
    if slot_minutes and slot_minutes != 60:
        tau *= 60.0 / slot_minutes
    return max(PERSISTENCE_TAU_MIN, min(PERSISTENCE_TAU_MAX, tau))


def conditioned_forecast(
    posterior: float, slot_forecast: float, slots_ahead: int, tau: float
) -> float:
    """Blend what is known *now* into the baseline forecast for a future slot.

    The learned priors are a climatology: P(occupied at T) averaged over weeks,
    blind to whether anyone is in the room right now. Occupancy is strongly
    autocorrelated, so the live posterior is real information about the next
    few slots — and it cuts both ways: a room that just emptied is *less* likely
    to be occupied next hour than its average says.

    The evidence's weight decays exponentially with the horizon, so slot 0
    returns the posterior unchanged (matching the area's
    ``occupancy_probability`` sensor exactly, which is what lets a consumer
    treat this single series as its only source) and distant slots return the
    baseline untouched.

    Args:
        posterior: The area's current occupancy probability (live estimate).
        slot_forecast: The slot's baseline forecast, from :func:`forecast_prior`.
        slots_ahead: Distance from the current slot, in slots. Never negative:
            callers wrap around the weekly grid, so "yesterday" is ~167 slots
            ahead and lands on the baseline.
        tau: Persistence in slots, from :func:`persistence_tau_slots`.

    Returns:
        Evidence-conditioned probability in ``[MIN_PRIOR, MAX_PRIOR]``.
    """
    if slots_ahead <= 0:
        return max(MIN_PRIOR, min(MAX_PRIOR, posterior))
    weight = math.exp(-slots_ahead / tau)
    if weight < 1e-6:
        return max(MIN_PRIOR, min(MAX_PRIOR, slot_forecast))
    # Blend in logit space, mirroring combine_priors: averaging probabilities
    # directly would pull mid-range values around far too aggressively.
    p_post = clamp_probability(posterior)
    p_slot = clamp_probability(slot_forecast)
    logit = weight * math.log(p_post / (1 - p_post)) + (1 - weight) * math.log(
        p_slot / (1 - p_slot)
    )
    return max(MIN_PRIOR, min(MAX_PRIOR, 1 / (1 + math.exp(-logit))))


def slots_ahead_of(
    day_of_week: int, time_slot: int, now_day: int, now_slot: int, slots_per_day: int
) -> int:
    """Return the forward distance from the current slot, wrapping the week.

    A slot earlier in the week is not "in the past" for a weekly grid — it is
    that slot next week. Wrapping keeps every value non-negative, so past slots
    land far enough ahead that the evidence weight vanishes and they show the
    plain baseline.
    """
    week = 7 * slots_per_day
    return (
        (day_of_week * slots_per_day + time_slot) - (now_day * slots_per_day + now_slot)
    ) % week


def shrink_slot_prior(
    slot_time_prior: float, weeks: int, global_prior: float | None
) -> float:
    """Pull a thinly observed slot toward the area's global prior.

    A slot's stored value is its occupied fraction over the weeks observed,
    and with one week behind it that is a single afternoon: the slot sits on
    a bound and flips the live prior across the threshold on the hour. Mixing
    in :data:`TIME_PRIOR_PSEUDO_WEEKS` weeks of the global prior is the
    standard pseudo-count estimate, ``(n·slot + k·global) / (n + k)``, and it
    fades as real weeks accumulate.

    Args:
        slot_time_prior: The slot's stored (bounds-clamped) time prior.
        weeks: Weeks of observation behind the slot. ``0`` marks an unlearned
            slot, which already holds the neutral fallback and is returned
            unchanged.
        global_prior: The area's learned global prior, or ``None``.

    Returns:
        The slot prior to combine with the global prior.
    """
    if global_prior is None or weeks <= 0:
        return slot_time_prior
    k = TIME_PRIOR_PSEUDO_WEEKS
    return (weeks * slot_time_prior + k * global_prior) / (weeks + k)


def forecast_prior(
    global_prior: float | None,
    slot_time_prior: float,
    *,
    prior_factor: float,
    weeks: int = 0,
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
        weeks: Weeks of observation behind the slot, for
            :func:`shrink_slot_prior`. The default ``0`` uses the slot as is.

    Returns:
        Forecast occupancy probability in ``[MIN_PRIOR, MAX_PRIOR]``.
    """
    if global_prior is None:
        return max(MIN_PRIOR, min(MAX_PRIOR, slot_time_prior))
    slot_time_prior = shrink_slot_prior(slot_time_prior, weeks, global_prior)
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

        - ``slots``: the **evidence-conditioned** forecast — the slot's baseline
          blended with the area's live posterior, weighted by how far ahead the
          slot is (see :func:`conditioned_forecast`). This is the single series
          a consumer should act on: at the current slot it equals the area's
          ``occupancy_probability`` exactly, and it relaxes to the baseline over
          the next few slots. Because it depends on the moment of the call, it
          is a forecast *issued now*, not a stable weekly matrix.
        - ``slots_baseline``: the same forecast **without** the live evidence —
          a stable weekly matrix that only changes when the hourly analysis
          relearns. This is the field to drive an old-fashioned thermostat
          schedule from: identical between two calls a minute apart, where
          ``slots`` deliberately is not.
        - ``slots_raw``: the learned time prior itself, uncombined and
          unconditioned, in ``[TIME_PRIOR_MIN_BOUND, TIME_PRIOR_MAX_BOUND]``.
          The underlying climatology, useful to rank hours against each other
          and to tell "lit because someone is here" from "lit out of habit".
        - ``data_points``: weeks of observation behind each slot; ``0`` marks a
          slot that was never learned, whose prior is the neutral fallback and
          should not be read as a real probability.

        Also returns ``current_slot`` (the ``"day,slot"`` key covering now, the
        anchor that aligns the live estimate to the grid), ``threshold`` and
        ``tau_slots``.

        Assumes the area's time-prior cache is already warm (the caller
        pre-loads it off the event loop).
    """
    prior = area.prior
    matrix = prior.all_time_priors()
    points = prior.all_time_prior_points()
    slots_per_day = max(1, 1440 // slot_minutes)

    posterior = area.probability()
    tau = persistence_tau_slots(area.purpose.half_life, slot_minutes)
    now_day, now_slot = prior.day_of_week, prior.time_slot

    slots: dict[str, float] = {}
    slots_baseline: dict[str, float] = {}
    slots_raw: dict[str, float] = {}
    data_points: dict[str, int] = {}
    for day, slot in sorted(matrix):
        key = f"{day},{slot}"
        ahead = slots_ahead_of(day, slot, now_day, now_slot, slots_per_day)
        baseline = prior.prior_for(day, slot)
        slots[key] = round(
            conditioned_forecast(posterior, baseline, ahead, tau),
            FORECAST_RESPONSE_PRECISION,
        )
        slots_baseline[key] = round(baseline, FORECAST_RESPONSE_PRECISION)
        slots_raw[key] = round(matrix[(day, slot)], FORECAST_RESPONSE_PRECISION)
        data_points[key] = points.get((day, slot), 0)
    return {
        "area_id": area.config.area_id,
        "global_prior": prior.global_prior,
        "slot_minutes": slot_minutes,
        "current_slot": f"{now_day},{now_slot}",
        "threshold": area.config.threshold,
        "tau_slots": round(tau, 3),
        "slots": slots,
        "slots_baseline": slots_baseline,
        "slots_raw": slots_raw,
        "data_points": data_points,
    }


def build_aggregate_time_priors(
    members: list[Area],
    slot_minutes: int,
    area_id: str,
    name: str,
    *,
    prior_factor: float,
    empirical: ZonePriors | None = None,
) -> dict[str, Any] | None:
    """Build the learned weekly forecast for an aggregate zone.

    Aggregate zones (the "All Areas" device and per-floor devices) answer
    "will anyone be in this zone", which no average of the rooms can: a room
    busy at 07:00 and another at 19:00 average to a zone that is never busy.
    So each slot's habit (``slots_baseline``) and raw prior come from the
    zone's empirical priors, measured on the union of its rooms' occupied
    history (#557). Until the first analysis has computed those, the highest
    member stands in, a lower bound on "anyone". The live series is at least
    the habit and at least any member's own live-conditioned forecast, so a
    zone reads busy whenever one of its rooms does.

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
    slots_per_day = max(1, 1440 // slot_minutes)
    # Each member is conditioned on its *own* live evidence and its own
    # persistence before averaging: a floor whose study is occupied right now
    # should read higher than its climatology, even if the other rooms are empty.
    member_ctx = [
        (
            m,
            m.probability(),
            persistence_tau_slots(m.purpose.half_life, slot_minutes),
            m.prior.day_of_week,
            m.prior.time_slot,
        )
        for m in members
    ]
    slots: dict[str, float] = {}
    slots_baseline: dict[str, float] = {}
    slots_raw: dict[str, float] = {}
    data_points: dict[str, int] = {}
    for day, slot in sorted(keys):
        key = f"{day},{slot}"
        zone_slot = (
            empirical.time_priors.get((day, slot)) if empirical is not None else None
        )
        if zone_slot is not None:
            baseline = forecast_prior(
                empirical.global_prior,
                zone_slot,
                prior_factor=prior_factor,
                weeks=empirical.data_points.get((day, slot), 0),
            )
        else:
            baseline = max(m.prior.prior_for(day, slot) for m in members)
        baseline = max(MIN_PRIOR, min(MAX_PRIOR, baseline))
        slots_baseline[key] = round(baseline, FORECAST_RESPONSE_PRECISION)
        values = [
            conditioned_forecast(
                posterior,
                member.prior.prior_for(day, slot),
                slots_ahead_of(day, slot, n_day, n_slot, slots_per_day),
                tau,
            )
            for member, posterior, tau, n_day, n_slot in member_ctx
        ]
        live = max(baseline, *values)
        slots[key] = round(
            max(MIN_PRIOR, min(MAX_PRIOR, live)), FORECAST_RESPONSE_PRECISION
        )
        if zone_slot is not None:
            slots_raw[key] = round(zone_slot, FORECAST_RESPONSE_PRECISION)
            data_points[key] = empirical.data_points.get((day, slot), 0)
        else:
            raw = [m[(day, slot)] for m in member_matrices if (day, slot) in m]
            slots_raw[key] = round(
                max(raw) if raw else 0.0, FORECAST_RESPONSE_PRECISION
            )
            # Weakest member wins: the zone is only as well-learned as its
            # least observed room, so a consumer never over-trusts it.
            per_member = [p.get((day, slot), 0) for p in member_points]
            data_points[key] = min(per_member) if per_member else 0
    first = members[0].prior
    return {
        "area_id": area_id,
        "name": name,
        "aggregate": True,
        "members": [m.config.area_id for m in members if m.config.area_id],
        "slot_minutes": slot_minutes,
        "current_slot": f"{first.day_of_week},{first.time_slot}",
        "slots": slots,
        "slots_baseline": slots_baseline,
        "slots_raw": slots_raw,
        "data_points": data_points,
    }
