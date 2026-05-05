"""Phase 4 of the adjacent-areas feature: Bayesian boost + decay modifier.

Pure math + lookup wrappers. The Bayesian update path (``area/area.py``)
calls :func:`compute_adjacency_boost` to bend the post-Bayesian
probability towards what the household's transition history suggests
should happen next; the decay path (``data/decay.py``) calls
:func:`compute_decay_modifier` to slow Y's decay when adjacent exits
have been silent since Y's last evidence.

This module deliberately avoids touching the coordinator or DB
directly. Callers gather the inputs (trajectory, current adjacent
probabilities, lookup function) and pass them in. That keeps the
math testable without spinning up SQLite, and lets Phase 5 wire the
integration sites without re-doing the math design.

See discussion #431 and PR #454 for the design rationale (symmetric
config + per-pair influence + 2-hop trajectories + time-of-day
bucketing + adjacent-silence decay modifier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..const import (
    ADJACENCY_BOOST_GAIN,
    ADJACENCY_DECAY_MODIFIER_GAIN,
    ADJACENCY_DECAY_MODIFIER_MAX,
)
from ..utils import clamp_probability, logit

if TYPE_CHECKING:
    from ..db.transitions import TransitionLookupResult


# A small Protocol matching the signature of
# ``db.transitions.lookup_transition_probability`` so callers and tests
# can pass either the real function or a stub. Keyword-only fields match
# the public API.
class TransitionLookupFn(Protocol):
    """Callable that returns ``P(to | from, mid, hour)`` with smoothing fallback."""

    def __call__(
        self,
        *,
        from_area: str,
        mid_area: str,
        to_area: str,
        hour_of_week: int,
    ) -> TransitionLookupResult:
        """Return ``P(to_area | from_area, mid_area, hour_of_week)``."""
        ...


@dataclass(frozen=True)
class Trajectory:
    """The recently-occupied adjacent context used for the boost lookup.

    Attributes:
        prev_area: The most recently active adjacent area (one hop back).
            ``None`` when no adjacent has been active within the trajectory
            window — in which case neither boost nor modifier fires.
        prev_prev_area: The adjacent area active before ``prev_area``
            (two hops back). ``None`` until a 2-hop trajectory exists.
            When set, the lookup uses the 2-hop levels; when ``None``,
            it falls through to 1-hop levels (caller passes
            ``mid_area=""``).
        hour_of_week: 0..167 bucket for time-of-day learning. Caller
            computes this from the current local time.
    """

    prev_area: str | None
    prev_prev_area: str | None
    hour_of_week: int


@dataclass
class BoostContribution:
    """Diagnostics-friendly breakdown of one area's adjacency boost.

    All fields are derived in :func:`compute_adjacency_boost` and
    surfaced under the ``current.adjacency`` block of the diagnostics
    export so users can see *why* an area is bumped above its sensor-
    only probability.
    """

    fired: bool = False
    trajectory_prev: str | None = None
    trajectory_prev_prev: str | None = None
    hour_of_week: int = 0
    raw_probability: float = 0.0  # P(this_area | trajectory, hour)
    fallback_level: str = ""
    observed_count: float = 0.0
    total_count: float = 0.0
    logit_contribution: float = 0.0  # what we added to logit(P_y)


@dataclass
class DecayModifierContribution:
    """Diagnostics-friendly breakdown of the decay modifier."""

    fired: bool = False
    silence_score: float = 0.0  # in [0, 1]
    decay_modifier: float = 1.0  # in [1, ADJACENCY_DECAY_MODIFIER_MAX]
    base_half_life_seconds: float = 0.0
    effective_half_life_seconds: float = 0.0
    silent_neighbours: list[tuple[str, float, float]] = field(default_factory=list)
    # ^^ list of (neighbour, P_neighbour_lagged, P(this→neighbour | trajectory))


# ─── Bayesian boost ─────────────────────────────────────────────────


# logit(0.5) = 0; the centring term ``- 0.5 * k * logit(0.5)`` from the
# PR plan is therefore identically zero. Keeping the comment so future
# maintainers don't think it was forgotten.

_LOGIT_HALF: float = 0.0


def compute_adjacency_boost(
    *,
    target_area: str,
    trajectory: Trajectory,
    lookup: TransitionLookupFn,
    gain: float = ADJACENCY_BOOST_GAIN,
) -> BoostContribution:
    """Compute the logit-space boost for the target area.

    The boost is ``gain × logit(P(target | trajectory, hour))``. When
    the lookup falls all the way through to the static default (no
    learned data) the boost is still applied, but its impact is small
    because ``logit(static_default ≈ 0.3)`` is close to zero in
    magnitude and the gain is < 1.

    Args:
        target_area: The area whose probability we're updating.
        trajectory: The recent-history context. If ``prev_area`` is
            ``None`` we have no relevant trajectory and the boost is
            zero (no fire).
        lookup: Returns ``TransitionLookupResult`` for a chain query.
        gain: Multiplier on the logit contribution. Defaults to the
            module-level ``ADJACENCY_BOOST_GAIN`` constant.

    Returns:
        ``BoostContribution`` carrying the contribution and full
        diagnostic breakdown. Add ``contribution.logit_contribution``
        to the area's logit(probability) before sigmoid'ing back.
    """
    out = BoostContribution(
        trajectory_prev=trajectory.prev_area,
        trajectory_prev_prev=trajectory.prev_prev_area,
        hour_of_week=trajectory.hour_of_week,
    )
    if trajectory.prev_area is None:
        return out  # No trajectory → no boost

    # Pass mid_area="" if we don't have a 2-hop trajectory yet; the
    # lookup helper will skip the 2-hop fallback levels for us.
    mid = trajectory.prev_prev_area or ""
    result = lookup(
        from_area=trajectory.prev_area,
        mid_area=mid,
        to_area=target_area,
        hour_of_week=trajectory.hour_of_week,
    )

    out.fired = True
    out.raw_probability = result.probability
    out.fallback_level = result.level
    out.observed_count = result.observed_count
    out.total_count = result.total_count
    # Centre around logit(0.5) so the static default (~0.3) doesn't
    # systematically bias every area downward when there's no learned
    # data. logit(0.5) = 0 makes the term mathematically a no-op but we
    # keep the formula explicit for clarity.
    out.logit_contribution = gain * (logit(result.probability) - _LOGIT_HALF)
    return out


def apply_logit_boost(base_probability: float, boost: BoostContribution) -> float:
    """Add ``boost.logit_contribution`` in logit space and return the new probability.

    Convenience helper — keeps the (clamp → logit → add → sigmoid →
    clamp) pattern in one place.
    """
    if not boost.fired or boost.logit_contribution == 0.0:
        return base_probability
    base = clamp_probability(base_probability)
    new_logit = logit(base) + boost.logit_contribution
    # Inline sigmoid; importing utils.sigmoid_probability would pull in
    # entity-graph dependencies we don't need.
    import math  # noqa: PLC0415  - tiny local scope; avoid module-level import cycle

    new_prob = 1.0 / (1.0 + math.exp(-new_logit))
    return clamp_probability(new_prob)


# ─── Decay modifier ─────────────────────────────────────────────────


def compute_decay_modifier(
    *,
    target_area: str,
    adjacency_index: dict[str, set[str]],
    lagged_probabilities: dict[str, float],
    trajectory: Trajectory,
    lookup: TransitionLookupFn,
    base_half_life_seconds: float,
    gain: float = ADJACENCY_DECAY_MODIFIER_GAIN,
    cap: float = ADJACENCY_DECAY_MODIFIER_MAX,
) -> DecayModifierContribution:
    """Compute how much to stretch the target's decay half-life.

    Mathematics::

        silence_score = Σ_X∈adj(target) ( (1 − P_X_lagged)
                                          × P(target → X | trajectory, hour) )
        decay_modifier = min(1 + gain × silence_score, cap)
        effective_half_life = base_half_life × decay_modifier

    ``silence_score`` is in [0, 1]: each adjacent X contributes more
    when X is currently quiet AND when the household historically
    leaves target via X. So a bedroom whose only learned exit (the
    hall) has been silent since the last evidence event gets the full
    ~``cap`` slowdown, while a hub area whose exits diverge in many
    directions gets a smaller modifier.

    Args:
        target_area: The area whose decay we're modifying.
        adjacency_index: ``{area: {neighbour, ...}}`` from
            ``AreaRelationships``. Lookups use the target's
            neighbours.
        lagged_probabilities: ``{area: P_previous_tick}``. Caller
            captures last-tick state before the current update.
        trajectory: Same as for boost; influences the per-neighbour
            lookups. ``prev_area=None`` is fine (lookup falls to 1-hop).
        lookup: Same lookup function as the boost path.
        base_half_life_seconds: The decay's current effective
            half-life before this modifier (could already be tuned
            by purpose / sleep window).
        gain: ``α`` — multiplier on the silence-score. Defaults to
            ``ADJACENCY_DECAY_MODIFIER_GAIN``.
        cap: Upper bound on ``effective_half_life / base_half_life``.
            Defaults to ``ADJACENCY_DECAY_MODIFIER_MAX``.

    Returns:
        ``DecayModifierContribution`` with diagnostic breakdown.
        Apply ``contribution.effective_half_life_seconds`` in place
        of ``base_half_life_seconds`` for the current decay tick.
    """
    out = DecayModifierContribution(
        base_half_life_seconds=base_half_life_seconds,
        effective_half_life_seconds=base_half_life_seconds,
    )

    neighbours = adjacency_index.get(target_area, set())
    if not neighbours:
        return out  # No neighbours → no silence to score

    silence_score = 0.0
    silent_breakdown: list[tuple[str, float, float]] = []
    # The 2-hop chain we're querying is ``prev_area → target_area → neighbour``,
    # so the mid hop (immediate predecessor of target_area) is prev_area, not
    # prev_prev_area. Falls back to "" — and thus the 1-hop levels — when no
    # trajectory is known.
    mid = trajectory.prev_area or ""

    for neighbour in sorted(neighbours):
        # Bound to [0, 1] but don't apply the MIN/MAX_PROBABILITY clamp
        # — the silence formula multiplies by (1 − p), so a true 0
        # (neighbour confidently empty) should yield the full silence
        # contribution. The clamp_probability helper exists to avoid
        # ``log(0)`` in logit-space code, which doesn't apply here.
        raw = lagged_probabilities.get(neighbour, 0.0)
        neighbour_prob = max(0.0, min(1.0, raw))
        # P(target → neighbour | trajectory, hour). The 1-hop fallback
        # is fine when the trajectory hasn't yet built up to 2 hops.
        result = lookup(
            from_area=target_area,
            mid_area=mid if trajectory.prev_area else "",
            to_area=neighbour,
            hour_of_week=trajectory.hour_of_week,
        )
        contribution = (1.0 - neighbour_prob) * result.probability
        silence_score += contribution
        silent_breakdown.append((neighbour, neighbour_prob, result.probability))

    # silence_score is now in [0, len(neighbours)]; clamp to [0, 1] so
    # the modifier scales sanely regardless of household density.
    silence_score = max(0.0, min(1.0, silence_score))
    modifier = min(1.0 + gain * silence_score, cap)

    out.fired = True
    out.silence_score = silence_score
    out.decay_modifier = modifier
    out.effective_half_life_seconds = base_half_life_seconds * modifier
    out.silent_neighbours = silent_breakdown
    return out
