"""Tests for the adjacency Bayesian boost + decay modifier (Phase 4).

These are pure-math tests with stubbed lookup functions. The actual
hook into ``Area.probability()`` and ``Decay.half_life`` is exercised
by integration tests in subsequent commits.
"""

from __future__ import annotations

import math

import pytest

from custom_components.area_occupancy.data.adjacency import (
    BoostContribution,
    Trajectory,
    apply_logit_boost,
    compute_adjacency_boost,
    compute_decay_modifier,
)
from custom_components.area_occupancy.db.transitions import (
    LEVEL_1HOP_HOUR_OF_WEEK,
    LEVEL_2HOP_HOUR_OF_WEEK,
    LEVEL_STATIC_DEFAULT,
    TransitionLookupResult,
)


def _stub_lookup(probability: float, level: str = LEVEL_2HOP_HOUR_OF_WEEK):
    """Build a no-arg lookup stub that always returns the same result."""

    def _fn(*, from_area, mid_area, to_area, hour_of_week):
        return TransitionLookupResult(
            probability=probability,
            level=level,
            observed_count=10.0 if level != LEVEL_STATIC_DEFAULT else 0.0,
            total_count=20.0 if level != LEVEL_STATIC_DEFAULT else 0.0,
        )

    return _fn


def _per_neighbour_lookup(by_to_area: dict[str, float]):
    """Build a stub that returns different probabilities per ``to_area``."""

    def _fn(*, from_area, mid_area, to_area, hour_of_week):
        return TransitionLookupResult(
            probability=by_to_area.get(to_area, 0.0),
            level=LEVEL_1HOP_HOUR_OF_WEEK,
            observed_count=5.0,
            total_count=10.0,
        )

    return _fn


# ─── compute_adjacency_boost ───────────────────────────────────────


class TestComputeAdjacencyBoost:
    def test_no_trajectory_yields_no_boost(self):
        """No prev_area in the trajectory → no contribution at all."""
        traj = Trajectory(prev_area=None, prev_prev_area=None, hour_of_week=10)
        out = compute_adjacency_boost(
            target_area="bedroom", trajectory=traj, lookup=_stub_lookup(0.7)
        )
        assert out.fired is False
        assert out.logit_contribution == 0.0

    def test_two_hop_trajectory_uses_mid_area(self):
        """When prev_prev is set, lookup is called with that mid_area."""
        captured: dict = {}

        def _fn(*, from_area, mid_area, to_area, hour_of_week):
            captured.update(locals())
            return TransitionLookupResult(
                probability=0.7,
                level=LEVEL_2HOP_HOUR_OF_WEEK,
                observed_count=10.0,
                total_count=20.0,
            )

        traj = Trajectory(prev_area="hall", prev_prev_area="study", hour_of_week=42)
        out = compute_adjacency_boost(
            target_area="bathroom", trajectory=traj, lookup=_fn
        )
        assert captured["from_area"] == "hall"
        assert captured["mid_area"] == "study"
        assert captured["to_area"] == "bathroom"
        assert captured["hour_of_week"] == 42
        assert out.fired is True
        assert out.fallback_level == LEVEL_2HOP_HOUR_OF_WEEK

    def test_one_hop_trajectory_passes_empty_mid(self):
        """No prev_prev → mid_area="" so the lookup falls to 1-hop levels."""
        captured: dict = {}

        def _fn(*, from_area, mid_area, to_area, hour_of_week):
            captured.update(locals())
            return TransitionLookupResult(
                probability=0.6,
                level=LEVEL_1HOP_HOUR_OF_WEEK,
                observed_count=5.0,
                total_count=10.0,
            )

        traj = Trajectory(prev_area="hall", prev_prev_area=None, hour_of_week=42)
        out = compute_adjacency_boost(
            target_area="bedroom", trajectory=traj, lookup=_fn
        )
        assert captured["mid_area"] == ""
        assert out.fired is True

    def test_logit_contribution_positive_when_prob_above_half(self):
        """P > 0.5 produces a positive logit boost."""
        traj = Trajectory(prev_area="hall", prev_prev_area=None, hour_of_week=10)
        out = compute_adjacency_boost(
            target_area="bedroom", trajectory=traj, lookup=_stub_lookup(0.8)
        )
        # logit(0.8) ≈ 1.386, gain default 0.5 → contribution ≈ 0.693
        assert out.logit_contribution == pytest.approx(
            0.5 * math.log(0.8 / 0.2), abs=1e-6
        )

    def test_logit_contribution_zero_at_half(self):
        """P = 0.5 → logit = 0 → no shift in either direction."""
        traj = Trajectory(prev_area="hall", prev_prev_area=None, hour_of_week=10)
        out = compute_adjacency_boost(
            target_area="bedroom", trajectory=traj, lookup=_stub_lookup(0.5)
        )
        assert out.logit_contribution == pytest.approx(0.0, abs=1e-9)

    def test_logit_contribution_negative_when_prob_below_half(self):
        """P < 0.5 produces a negative logit boost.

        The area is *less* likely than chance to be next, so the
        probability gets pulled down.
        """
        traj = Trajectory(prev_area="hall", prev_prev_area=None, hour_of_week=10)
        out = compute_adjacency_boost(
            target_area="bedroom", trajectory=traj, lookup=_stub_lookup(0.2)
        )
        assert out.logit_contribution < 0.0

    def test_gain_override_scales_contribution(self):
        traj = Trajectory(prev_area="hall", prev_prev_area=None, hour_of_week=10)
        out_default = compute_adjacency_boost(
            target_area="bedroom", trajectory=traj, lookup=_stub_lookup(0.8)
        )
        out_doubled = compute_adjacency_boost(
            target_area="bedroom",
            trajectory=traj,
            lookup=_stub_lookup(0.8),
            gain=1.0,
        )
        assert out_doubled.logit_contribution == pytest.approx(
            2 * out_default.logit_contribution, abs=1e-6
        )


# ─── apply_logit_boost ──────────────────────────────────────────────


class TestApplyLogitBoost:
    def test_no_boost_returns_input_unchanged(self):
        b = BoostContribution(fired=False, logit_contribution=0.0)
        assert apply_logit_boost(0.42, b) == 0.42

    def test_positive_boost_increases_probability(self):
        b = BoostContribution(fired=True, logit_contribution=1.0)
        assert apply_logit_boost(0.5, b) > 0.5

    def test_negative_boost_decreases_probability(self):
        b = BoostContribution(fired=True, logit_contribution=-1.0)
        assert apply_logit_boost(0.5, b) < 0.5

    def test_extreme_boost_clamped_to_max(self):
        b = BoostContribution(fired=True, logit_contribution=100.0)
        assert apply_logit_boost(0.5, b) <= 0.99


# ─── compute_decay_modifier ────────────────────────────────────────


class TestComputeDecayModifier:
    def test_no_neighbours_returns_unmodified_half_life(self):
        out = compute_decay_modifier(
            target_area="bedroom",
            adjacency_index={},  # bedroom has no neighbours
            lagged_probabilities={},
            trajectory=Trajectory(prev_area=None, prev_prev_area=None, hour_of_week=10),
            lookup=_stub_lookup(0.5),
            base_half_life_seconds=300.0,
        )
        assert out.fired is False
        assert out.effective_half_life_seconds == 300.0
        assert out.decay_modifier == 1.0

    def test_full_silence_with_dominant_exit_hits_cap(self):
        """Bedroom's only learned exit is the hall, hall is silent → max slowdown."""
        out = compute_decay_modifier(
            target_area="bedroom",
            adjacency_index={"bedroom": {"hall"}},
            lagged_probabilities={"hall": 0.0},  # silent
            trajectory=Trajectory(prev_area=None, prev_prev_area=None, hour_of_week=10),
            # 100% of the time bedroom exits via hall
            lookup=_per_neighbour_lookup({"hall": 1.0}),
            base_half_life_seconds=300.0,
        )
        assert out.fired is True
        # silence_score = (1 - 0.0) × 1.0 = 1.0 → modifier = 1 + 0.75 × 1.0
        # = 1.75 (capped at 1.75 by default).
        assert out.silence_score == pytest.approx(1.0)
        assert out.decay_modifier == pytest.approx(1.75)
        assert out.effective_half_life_seconds == pytest.approx(525.0)

    def test_neighbour_currently_active_reduces_silence_score(self):
        """Hall is occupied → bedroom can't be sure occupant hasn't left."""
        out = compute_decay_modifier(
            target_area="bedroom",
            adjacency_index={"bedroom": {"hall"}},
            lagged_probabilities={"hall": 1.0},  # currently occupied
            trajectory=Trajectory(prev_area=None, prev_prev_area=None, hour_of_week=10),
            lookup=_per_neighbour_lookup({"hall": 1.0}),
            base_half_life_seconds=300.0,
        )
        assert out.fired is True
        # silence_score = (1 - 1.0) × 1.0 = 0 → no modifier
        assert out.silence_score == 0.0
        assert out.decay_modifier == 1.0
        assert out.effective_half_life_seconds == 300.0

    def test_modifier_caps_at_max(self):
        """A pathological lookup that returns >1 still respects the cap."""
        out = compute_decay_modifier(
            target_area="bedroom",
            adjacency_index={"bedroom": {"a", "b", "c"}},
            lagged_probabilities={"a": 0.0, "b": 0.0, "c": 0.0},
            trajectory=Trajectory(prev_area=None, prev_prev_area=None, hour_of_week=10),
            # All three neighbours each return 1.0 — silence_score
            # would be 3.0 without clamping.
            lookup=_stub_lookup(1.0),
            base_half_life_seconds=300.0,
        )
        # Clamped to 1.0 internally → modifier 1 + 0.75 = 1.75
        assert out.silence_score == 1.0
        assert out.decay_modifier == pytest.approx(1.75)

    def test_partial_silence_yields_proportional_modifier(self):
        """Half the exits silent → roughly half the slowdown."""
        out = compute_decay_modifier(
            target_area="hall",
            adjacency_index={"hall": {"bedroom", "kitchen"}},
            lagged_probabilities={"bedroom": 0.0, "kitchen": 1.0},
            trajectory=Trajectory(prev_area=None, prev_prev_area=None, hour_of_week=10),
            # hall exits 50% to bedroom, 50% to kitchen
            lookup=_per_neighbour_lookup({"bedroom": 0.5, "kitchen": 0.5}),
            base_half_life_seconds=300.0,
        )
        # silence_score = (1-0) × 0.5 + (1-1) × 0.5 = 0.5
        assert out.silence_score == pytest.approx(0.5)
        # modifier = 1 + 0.75 × 0.5 = 1.375
        assert out.decay_modifier == pytest.approx(1.375)
        assert out.effective_half_life_seconds == pytest.approx(412.5)

    def test_trajectory_prev_area_is_used_as_mid_hop(self):
        """The 2-hop chain is ``prev_area → target → neighbour``.

        Regression: the mid_area passed to the lookup must be the
        immediate predecessor of ``target_area`` (i.e. ``prev_area``),
        not ``prev_prev_area``. Without this, the lookup queries the
        wrong 2-hop chain and silently drops to a 1-hop fallback.
        """
        captured: list[dict] = []

        def _fn(*, from_area, mid_area, to_area, hour_of_week):
            captured.append(
                {"from_area": from_area, "mid_area": mid_area, "to_area": to_area}
            )
            return TransitionLookupResult(
                probability=0.5,
                level=LEVEL_2HOP_HOUR_OF_WEEK,
                observed_count=10.0,
                total_count=20.0,
            )

        out = compute_decay_modifier(
            target_area="bedroom",
            adjacency_index={"bedroom": {"hall"}},
            lagged_probabilities={"hall": 0.0},
            trajectory=Trajectory(
                prev_area="hall", prev_prev_area="study", hour_of_week=10
            ),
            lookup=_fn,
            base_half_life_seconds=300.0,
        )
        assert captured == [
            {"from_area": "bedroom", "mid_area": "hall", "to_area": "hall"}
        ]
        assert out.fired is True

    def test_silent_neighbour_breakdown_reflects_lookup_inputs(self):
        """The diagnostic breakdown carries each neighbour's contributions."""
        out = compute_decay_modifier(
            target_area="hall",
            adjacency_index={"hall": {"bedroom", "kitchen"}},
            lagged_probabilities={"bedroom": 0.2, "kitchen": 0.0},
            trajectory=Trajectory(prev_area=None, prev_prev_area=None, hour_of_week=10),
            lookup=_per_neighbour_lookup({"bedroom": 0.4, "kitchen": 0.6}),
            base_half_life_seconds=300.0,
        )
        breakdown = {n: (lagged, trans) for n, lagged, trans in out.silent_neighbours}
        assert breakdown == {
            "bedroom": (0.2, 0.4),
            "kitchen": (0.0, 0.6),
        }
