"""Tests for the sojourn-time survival shadow (learned decay curve)."""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from custom_components.area_occupancy.data.online_prior import MAX_TICK_GAP_SECONDS
from custom_components.area_occupancy.data.sojourn import (
    SOJOURN_EPISODE_MAX_SECONDS,
    SOJOURN_EPISODES_KEPT,
    SOJOURN_EXIT_GRACE_SECONDS,
    SOJOURN_NEIGHBOR_EXIT_WINDOW_S,
    SojournEpisode,
    SojournShadow,
    SojournState,
    sojourn_context,
)
from homeassistant.util import dt as dt_util

T0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
HALF_LIFE = 600.0
# Ticks must stay under MAX_TICK_GAP_SECONDS apart or the shadow treats the
# span as downtime; 30 s matches the real ~10 s cadence closely enough.
TICK_STEP = 30.0


def _run(
    shadow: SojournShadow,
    events: list[tuple[float, dict]],
    tail: float = 0.0,
) -> None:
    """Feed ticks every TICK_STEP seconds; events override the kwargs.

    Each event is ``(seconds, observe-kwargs)``. All times are seconds
    relative to T0; the run starts 30 s before the first event so a
    ``present=True`` edge is always observed before silence begins.
    ``tail`` extends the run past the last event with that event's
    kwargs — needed so an exit candidate can sit out its grace period.
    """
    overrides = dict(events)
    first = min(overrides) - TICK_STEP
    last = max(overrides)
    # Grid ticks plus every event time, so off-grid events still fire.
    times = set(overrides)
    t = first
    while t <= last + 1e-9:
        times.add(round(t, 6))
        t += TICK_STEP
    tail_times: list[float] = []
    if tail > 0:
        t = last + TICK_STEP
        while t <= last + tail + 1e-9:
            tail_times.append(round(t, 6))
            t += TICK_STEP
    tail_kwargs = overrides[last]
    for seconds in sorted(times):
        kwargs = overrides.get(seconds, {})
        shadow.observe(
            T0 + timedelta(seconds=seconds),
            room_held=kwargs.get("room_held", True),
            present=kwargs.get("present", False),
            doors_open=kwargs.get("doors_open", False),
            neighbours=kwargs.get("neighbours", {"Hall": False}),
            half_life_seconds=HALF_LIFE,
            context=kwargs.get("context", ""),
        )
    for seconds in tail_times:
        shadow.observe(
            T0 + timedelta(seconds=seconds),
            room_held=tail_kwargs.get("room_held", True),
            present=tail_kwargs.get("present", False),
            doors_open=tail_kwargs.get("doors_open", False),
            neighbours=tail_kwargs.get("neighbours", {"Hall": False}),
            half_life_seconds=HALF_LIFE,
            context=tail_kwargs.get("context", ""),
        )


def _episode(
    shadow: SojournShadow,
    silence: float,
    tail: float | None = None,
    context: str = "",
    **close: object,
) -> None:
    """Sensors on, quiet at T0, then ``close`` kwargs at ``silence``.

    Defaults to a tail long enough for an exit candidate's grace period;
    pass ``tail=0`` for outcomes that close immediately (resumed).
    ``context`` is stamped on every tick — the episode captures it at
    its start, not at close.
    """
    _run(
        shadow,
        [
            (-TICK_STEP, {"present": True, "context": context}),
            (0.0, {"context": context}),
            (silence, {**close, "context": context}),
        ],
        tail=SOJOURN_EXIT_GRACE_SECONDS + 2 * TICK_STEP if tail is None else tail,
    )


class TestEpisodes:
    def test_resumed_closes_episode(self) -> None:
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (90.0, {"present": True}),
            ],
            tail=TICK_STEP,
        )
        assert shadow.state.episode_start is None
        assert shadow.state.episodes[-1].outcome == "resumed"
        assert shadow.state.episodes[-1].silence_seconds == pytest.approx(90)

    def test_door_open_confirms_as_exit_after_grace(self) -> None:
        shadow = SojournShadow()
        _episode(shadow, 120, doors_open=True)
        episode = shadow.state.episodes[-1]
        assert episode.outcome == "exits"
        # The room emptied at the signal, not at grace expiry.
        assert episode.silence_seconds == pytest.approx(120)
        assert episode.exit_weight == pytest.approx(1.0)

    def test_door_open_then_return_is_resumed(self) -> None:
        """Evidence back inside the grace period: the room never emptied."""
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (120.0, {"doors_open": True}),
                (150.0, {"present": True, "doors_open": False}),
            ],
            tail=TICK_STEP,
        )
        assert shadow.state.episodes[-1].outcome == "resumed"
        assert shadow.state.episodes[-1].silence_seconds == pytest.approx(150)

    def test_grace_expiry_confirms_exit(self) -> None:
        """No evidence back within the grace period: the exit stands."""
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (120.0, {"doors_open": True}),
                (120.0 + SOJOURN_EXIT_GRACE_SECONDS + TICK_STEP, {}),
            ],
            tail=TICK_STEP,
        )
        assert shadow.state.episodes[-1].outcome == "exits"

    def test_neighbour_lights_up_within_window(self) -> None:
        shadow = SojournShadow()
        _episode(
            shadow,
            30,
            neighbours={"Hall": True},
        )
        # silence=30 is inside ADJACENCY_TRANSITION_WINDOW_S -> strong exit.
        episode = shadow.state.episodes[-1]
        assert episode.outcome == "exits"
        assert episode.exit_weight == pytest.approx(1.0)

    def test_delayed_neighbour_is_weak_exit(self) -> None:
        """Beyond the transition window but inside the exit window: weak."""
        shadow = SojournShadow()
        silence = SOJOURN_NEIGHBOR_EXIT_WINDOW_S - 30.0
        assert silence > 60.0  # outside the strong window
        _episode(shadow, silence, neighbours={"Hall": True})
        episode = shadow.state.episodes[-1]
        assert episode.outcome == "exits"
        assert episode.exit_weight == pytest.approx(0.5)

    def test_very_late_neighbour_is_not_an_exit(self) -> None:
        """Past SOJOURN_NEIGHBOR_EXIT_WINDOW_S a lit neighbour means nothing."""
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (
                    SOJOURN_NEIGHBOR_EXIT_WINDOW_S + TICK_STEP,
                    {"neighbours": {"Hall": True}},
                ),
                (SOJOURN_NEIGHBOR_EXIT_WINDOW_S + 3 * TICK_STEP, {"doors_open": True}),
            ],
            tail=SOJOURN_EXIT_GRACE_SECONDS + TICK_STEP,
        )
        episode = shadow.state.episodes[-1]
        assert episode.outcome == "exits"
        # The exit is the door, not the stale neighbour signal.
        assert episode.silence_seconds == pytest.approx(
            SOJOURN_NEIGHBOR_EXIT_WINDOW_S + 3 * TICK_STEP
        )

    def test_unexplained_at_cap(self) -> None:
        shadow = SojournShadow()
        _episode(shadow, SOJOURN_EPISODE_MAX_SECONDS, tail=TICK_STEP)
        assert shadow.state.episodes[-1].outcome == "unexplained"

    def test_cap_with_pending_candidate_is_exit(self) -> None:
        """A departure signal seen before the cap still counts as an exit."""
        shadow = SojournShadow()
        door_at = SOJOURN_EPISODE_MAX_SECONDS - 2 * TICK_STEP
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (door_at, {"doors_open": True}),
            ],
            tail=SOJOURN_EXIT_GRACE_SECONDS + 2 * TICK_STEP,
        )
        assert shadow.state.episodes[-1].outcome == "exits"
        assert shadow.state.episodes[-1].silence_seconds == pytest.approx(door_at)

    def test_no_episode_when_not_held(self) -> None:
        """A room already decided empty produces no silence episode."""
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True, "room_held": False}),
                (0.0, {"room_held": False}),
                (90.0, {}),
            ],
            tail=TICK_STEP,
        )
        assert shadow.state.episode_start is None
        assert not shadow.state.episodes

    def test_no_episode_without_present_transition(self) -> None:
        """Sensors already quiet: no present edge, no episode."""
        shadow = SojournShadow()
        _run(shadow, [(0.0, {}), (90.0, {})])
        assert shadow.state.episode_start is None

    def test_tick_gap_censors_episode(self) -> None:
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (90.0, {}),
            ],
        )
        # Silence is unknown across the gap: the episode is censored at
        # the last observed tick (90 s of measured silence).
        shadow.observe(
            T0 + timedelta(seconds=90 + MAX_TICK_GAP_SECONDS + 1),
            room_held=True,
            present=False,
            doors_open=False,
            neighbours={},
            half_life_seconds=HALF_LIFE,
        )
        assert shadow.state.episodes[-1].outcome == "censored"
        assert shadow.state.episodes[-1].silence_seconds == pytest.approx(90)


class TestSurvival:
    def test_no_episodes_is_neutral(self) -> None:
        assert SojournShadow().survival_at(60) == pytest.approx(0.5)

    def test_zero_silence_is_certain(self) -> None:
        shadow = SojournShadow()
        _episode(shadow, 60, doors_open=True)
        assert shadow.survival_at(0) == 1.0

    def test_early_exits_lower_survival(self) -> None:
        """Many episodes ending in exits early push S down with silence."""
        shadow = SojournShadow()
        for i in range(20):
            base = i * 600
            _run(
                shadow,
                [
                    (base - TICK_STEP, {"present": True}),
                    (base, {}),
                    (base + 110, {"doors_open": True}),
                ],
                tail=SOJOURN_EXIT_GRACE_SECONDS + 2 * TICK_STEP,
            )
        assert shadow.survival_at(30) > shadow.survival_at(300)
        assert shadow.survival_at(300) < 1.0

    def test_resumes_only_fall_back_to_baseline(self) -> None:
        """No observed exits: the curve must not learn "people never leave"."""
        shadow = SojournShadow()
        for i in range(20):
            base = i * 400
            _run(
                shadow,
                [
                    (base - TICK_STEP, {"present": True}),
                    (base, {}),
                    (base + 90, {"present": True}),
                ],
                tail=TICK_STEP,
            )
        # Exit-blind: survival_at returns the exponential baseline.
        assert shadow.survival_at(500) == pytest.approx(0.5 ** (500 / HALF_LIFE))

    def test_exit_blind_flag_in_summary(self) -> None:
        shadow = SojournShadow()
        for i in range(25):
            base = i * 400
            _run(
                shadow,
                [
                    (base - TICK_STEP, {"present": True}),
                    (base, {}),
                    (base + 90, {"present": True}),
                ],
                tail=TICK_STEP,
            )
        summary = shadow.summary()
        assert summary is not None
        assert summary["exit_blind"] is True

    def test_monotonic_non_increasing(self) -> None:
        shadow = SojournShadow()
        for i in range(10):
            _episode(shadow, 50 + i * 30, doors_open=True)
        s = [shadow.survival_at(x) for x in (10, 100, 500, 2000, 8000)]
        assert all(a >= b for a, b in pairwise(s))

    def test_never_freezes_at_zero(self) -> None:
        """All-early-exits data still leaves positive survival in the tail."""
        shadow = SojournShadow()
        for i in range(30):
            base = i * 300
            _run(
                shadow,
                [
                    (base - TICK_STEP, {"present": True}),
                    (base, {}),
                    (base + 20, {"doors_open": True}),
                ],
                tail=SOJOURN_EXIT_GRACE_SECONDS + 2 * TICK_STEP,
            )
        assert shadow.survival_at(300) > 0.0
        assert shadow.survival_at(6000) > 0.0


class TestScoring:
    def test_resumed_scores_present_exits_scores_absent(self) -> None:
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (90.0, {"present": True}),  # resumed -> y=1
                (200.0, {}),
                (300.0, {"doors_open": True}),  # exits -> y=0
            ],
            tail=SOJOURN_EXIT_GRACE_SECONDS + 2 * TICK_STEP,
        )
        day = shadow.state.score_days["2026-11-20"]
        assert day["scored"] == 2
        assert day["scored_exits"] == 1
        assert day["resumed"] == 1
        assert day["exits"] == 1
        assert day["brier_model"] >= 0
        assert day["brier_baseline"] >= 0

    def test_weak_exits_are_not_brier_scored(self) -> None:
        """Uncertain labels feed the life table but not the score."""
        shadow = SojournShadow()
        _episode(shadow, SOJOURN_NEIGHBOR_EXIT_WINDOW_S - 30, neighbours={"Hall": True})
        day = shadow.state.score_days["2026-11-20"]
        assert day["exits"] == 1
        assert day["scored"] == 0
        assert day["scored_exits"] == 0

    def test_unexplained_and_censored_are_not_scored(self) -> None:
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (SOJOURN_EPISODE_MAX_SECONDS, {}),
            ],
            tail=TICK_STEP,
        )
        day = shadow.state.score_days["2026-11-20"]
        assert day["episodes"] == 1
        assert day["scored"] == 0

    def test_scores_are_prequential(self) -> None:
        """An episode is scored against a curve that has not seen it."""
        shadow = SojournShadow()
        _episode(shadow, 100, doors_open=True)  # first exit, scored vs empty
        _run(
            shadow,
            [
                (500 - TICK_STEP, {"present": True}),
                (500.0, {}),
                (900.0, {"doors_open": True}),
            ],
            tail=SOJOURN_EXIT_GRACE_SECONDS + 2 * TICK_STEP,
        )
        eps = shadow.state.episodes
        assert len(eps) == 2
        solo = SojournShadow()
        solo.state.episodes = [eps[0]]
        # Align the recency reference so weights match the live shadow.
        solo.state.last_tick = shadow.state.last_tick
        solo.state.last_half_life = shadow.state.last_half_life
        p = solo.survival_at(eps[1].silence_seconds)
        day = shadow.state.score_days["2026-11-20"]
        # Episode 1 scored against the empty table (neutral 0.5, y=0);
        # episode 2 against the one-exit table (p, y=0). The recency
        # reference tick differs by a few ticks, hence the loose bound.
        assert day["brier_model"] == pytest.approx(0.5**2 + p**2, rel=1e-3)


class TestContext:
    def test_sojourn_context_sleep_window_and_weekend(self) -> None:
        # The sleep window is wall-clock local time; pin UTC so the
        # expectations below don't depend on the test runner's zone.
        original = dt_util.DEFAULT_TIME_ZONE
        dt_util.DEFAULT_TIME_ZONE = dt_util.UTC
        try:
            # 2026-11-20 is a Friday; 2026-11-21/22 are the weekend.
            night = datetime(2026, 11, 20, 23, 0, tzinfo=UTC)
            day = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
            weekend_day = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
            assert sojourn_context(night, "22:00:00", "08:00:00") == "night"
            assert sojourn_context(day, "22:00:00", "08:00:00") == "day"
            assert sojourn_context(weekend_day, "22:00:00", "08:00:00") == "day_weekend"
            # No sleep window configured: the day/night split would be
            # invented, not learned.
            assert sojourn_context(night, None, None) == "any"
            assert sojourn_context(weekend_day, None, None) == "any_weekend"
            # Same-day window and unparseable bounds mean "no window".
            assert sojourn_context(night, "08:00:00", "08:00:00") == "any"
            assert sojourn_context(night, "junk", "08:00:00") == "any"
        finally:
            dt_util.DEFAULT_TIME_ZONE = original

    def test_context_split_thin_context_falls_back_to_pooled(self) -> None:
        """A context with little data answers from the pooled curve."""
        shadow = SojournShadow()
        _episode(shadow, 100, doors_open=True, context="day")
        night = shadow.survival_at(200, context="night")
        pooled = shadow.survival_at(200)
        assert night == pytest.approx(pooled)

    def test_context_is_stored_on_episode(self) -> None:
        shadow = SojournShadow()
        _episode(shadow, 100, doors_open=True, context="night")
        assert shadow.state.episodes[-1].context == "night"


class TestSummary:
    def test_empty_shadow_reports_nothing(self) -> None:
        assert SojournShadow().summary() is None

    def test_summary_reports_curve_and_scores(self) -> None:
        shadow = SojournShadow()
        for i in range(10):
            _episode(shadow, 60 + i * 30, doors_open=True)
        _run(
            shadow,
            [
                (1000 - TICK_STEP, {"present": True}),
                (1000.0, {}),
                (1100.0, {"present": True}),
            ],
            tail=TICK_STEP,
        )
        summary = shadow.summary()
        assert summary is not None
        assert summary["shadow_mode"] is True
        assert summary["engaged"] is False
        assert summary["episodes"] == 11
        assert summary["scored"] == 11
        assert summary["outcomes"]["exits"] == 10
        assert summary["outcomes"]["resumed"] == 1
        assert summary["survival"]["60s"] <= 1.0
        assert summary["open_episode"] is False
        gate = summary["gate"]
        assert gate["scored"] == 11
        assert gate["scored_exits"] == 10
        assert gate["ready"] is False
        assert gate["min_scored"] > 0

    def test_gate_ready_requires_counts_days_and_skill(self) -> None:
        """Fabricated tallies: ready only when every threshold is met."""
        shadow = SojournShadow()
        shadow.state.episodes = [
            SojournEpisode(
                start=T0,
                end=T0 + timedelta(seconds=60),
                silence_seconds=60.0,
                outcome="exits",
                exit_weight=1.0,
                context="",
                half_life_seconds=HALF_LIFE,
            )
        ]
        days = [
            (datetime(2026, 11, 1) + timedelta(days=i)).date().isoformat()
            for i in range(14)
        ]
        for day in days:
            shadow.state.score_days[day] = {
                "scored": 10.0,
                "scored_exits": 3.0,
                "brier_model": 0.1,
                "brier_baseline": 0.2,
            }
        summary = shadow.summary()
        assert summary is not None
        assert summary["gate"]["ready"] is True
        # One bad day drops the pooled skill below zero -> not ready.
        shadow.state.score_days[days[0]]["brier_model"] = 5.0
        summary = shadow.summary()
        assert summary is not None
        assert summary["gate"]["ready"] is False


class TestPersistence:
    def test_roundtrip(self) -> None:
        shadow = SojournShadow()
        _episode(shadow, 100, doors_open=True)
        _run(
            shadow,
            [
                (500 - TICK_STEP, {"present": True}),
                (500.0, {}),
            ],
            tail=TICK_STEP,
        )
        state = SojournState.from_dict(shadow.state.to_dict())
        assert len(state.episodes) == 1
        assert state.episodes[0].outcome == "exits"
        assert state.episode_start is not None
        restored = SojournShadow(state)
        assert restored.survival_at(200) == pytest.approx(shadow.survival_at(200))

    def test_roundtrip_preserves_exit_candidate(self) -> None:
        shadow = SojournShadow()
        # Tail to 180 s: candidate pending, episode still open, ticks
        # close enough together that no gap-censoring triggers.
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (120.0, {"doors_open": True}),
            ],
            tail=SOJOURN_EXIT_GRACE_SECONDS - TICK_STEP,
        )
        state = SojournState.from_dict(shadow.state.to_dict())
        restored = SojournShadow(state)
        assert restored.state.exit_candidate is not None
        # Grace expires at 210 s; 30 s after the last tick — no gap.
        restored.observe(
            T0 + timedelta(seconds=120 + SOJOURN_EXIT_GRACE_SECONDS),
            room_held=True,
            present=False,
            doors_open=True,
            neighbours={},
            half_life_seconds=HALF_LIFE,
        )
        assert restored.state.episodes[-1].outcome == "exits"

    def test_malformed_payload_falls_back(self) -> None:
        assert SojournState.from_dict({"episodes": "nope"}).episodes == []
        assert SojournState.from_dict({}).episodes == []

    def test_malformed_episodes_are_skipped(self) -> None:
        """Bad episodes don't poison the load — they're dropped."""
        good = SojournEpisode(
            start=T0,
            end=T0 + timedelta(seconds=60),
            silence_seconds=60.0,
            outcome="exits",
            exit_weight=1.0,
            context="",
            half_life_seconds=HALF_LIFE,
        ).to_dict()
        bad_nan = dict(good, silence_seconds=float("nan"))
        bad_outcome = dict(good, outcome="vanished")
        bad_negative = dict(good, silence_seconds=-5.0)
        state = SojournState.from_dict(
            {"episodes": [bad_nan, bad_outcome, bad_negative, good]}
        )
        assert len(state.episodes) == 1
        assert state.episodes[0].outcome == "exits"

    def test_episode_cap(self) -> None:
        """Closing an episode trims the retained list to the cap."""
        shadow = SojournShadow()
        shadow.state.episodes = [
            SojournEpisode(
                start=T0 + timedelta(seconds=i),
                end=T0 + timedelta(seconds=i + 10),
                silence_seconds=10.0,
                outcome="exits",
                exit_weight=1.0,
                context="",
                half_life_seconds=HALF_LIFE,
            )
            for i in range(SOJOURN_EPISODES_KEPT)
        ]
        # An open episode, closed via the public observe path.
        shadow.state.episode_start = T0
        shadow.state.episode_half_life = HALF_LIFE
        shadow.state.last_tick = T0
        shadow.observe(
            T0 + timedelta(seconds=10),
            room_held=True,
            present=True,
            doors_open=False,
            neighbours={},
            half_life_seconds=HALF_LIFE,
        )
        assert len(shadow.state.episodes) == SOJOURN_EPISODES_KEPT
        # The oldest episode was dropped; the new one is last.
        assert shadow.state.episodes[-1].outcome == "resumed"
