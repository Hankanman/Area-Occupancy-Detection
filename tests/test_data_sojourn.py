"""Tests for the sojourn-time survival shadow (learned decay curve)."""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from custom_components.area_occupancy.data.online_prior import MAX_TICK_GAP_SECONDS
from custom_components.area_occupancy.data.sojourn import (
    SOJOURN_EPISODE_MAX_SECONDS,
    SOJOURN_EPISODES_KEPT,
    SojournEpisode,
    SojournShadow,
    SojournState,
)

T0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
HALF_LIFE = 600.0
# Ticks must stay under MAX_TICK_GAP_SECONDS apart or the shadow treats the
# span as downtime; 30 s matches the real ~10 s cadence closely enough.
TICK_STEP = 30.0


def _run(shadow: SojournShadow, events: list[tuple[float, dict]]) -> None:
    """Feed ticks every TICK_STEP seconds; events override the kwargs.

    Each event is ``(seconds, observe-kwargs)``. All times are seconds
    relative to T0; the run starts 30 s before the first event so a
    ``present=True`` edge is always observed before silence begins.
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
    for seconds in sorted(times):
        kwargs = overrides.get(seconds, {})
        shadow.observe(
            T0 + timedelta(seconds=seconds),
            present=kwargs.get("present", False),
            occupied=kwargs.get("occupied", True),
            doors_open=kwargs.get("doors_open", False),
            neighbours=kwargs.get("neighbours", {"Hall": False}),
            half_life_seconds=HALF_LIFE,
        )
        t += TICK_STEP


def _episode(shadow: SojournShadow, silence: float, **close: object) -> None:
    """Sensors on, quiet at T0, then ``close`` kwargs at ``silence``."""
    _run(
        shadow,
        [
            (-TICK_STEP, {"present": True}),
            (0.0, {}),
            (silence, close),
        ],
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
        )
        assert shadow.state.episode_start is None
        assert shadow.state.episodes[-1].outcome == "resumed"
        assert shadow.state.episodes[-1].silence_seconds == pytest.approx(90)

    def test_door_open_closes_as_exit(self) -> None:
        shadow = SojournShadow()
        _episode(shadow, 120, doors_open=True)
        assert shadow.state.episodes[-1].outcome == "exits"
        assert shadow.state.episodes[-1].silence_seconds == pytest.approx(120)

    def test_neighbour_lights_up_within_window(self) -> None:
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (30.0, {"neighbours": {"Hall": True}}),
            ],
        )
        assert shadow.state.episodes[-1].outcome == "exits"

    def test_late_neighbour_is_not_an_exit(self) -> None:
        """The transition window has passed, so a lit neighbour means nothing."""
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (150.0, {"neighbours": {"Hall": True}}),
                (180.0, {"doors_open": True}),
            ],
        )
        # Still an episode until the door opens; the neighbour didn't end it.
        assert shadow.state.episodes[-1].silence_seconds == pytest.approx(180)

    def test_unexplained_at_cap(self) -> None:
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (SOJOURN_EPISODE_MAX_SECONDS, {}),
            ],
        )
        assert shadow.state.episodes[-1].outcome == "unexplained"

    def test_no_episode_when_not_occupied(self) -> None:
        """A room already decided empty produces no silence episode."""
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True, "occupied": False}),
                (0.0, {"occupied": False}),
                (90.0, {}),
            ],
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
            present=False,
            occupied=True,
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
            base = i * 400
            _run(
                shadow,
                [
                    (base - TICK_STEP, {"present": True}),
                    (base, {}),
                    (base + 110, {"doors_open": True}),
                ],
            )
        assert shadow.survival_at(30) > shadow.survival_at(300)
        assert shadow.survival_at(300) < 1.0

    def test_resumes_censor_not_events(self) -> None:
        """Resumed episodes censor; they are not observed departures."""
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
            )
        assert shadow.survival_at(500) == pytest.approx(1.0)

    def test_monotonic_non_increasing(self) -> None:
        shadow = SojournShadow()
        for i in range(10):
            _episode(shadow, 50 + i * 30, doors_open=True)
        s = [shadow.survival_at(x) for x in (10, 100, 500, 2000, 8000)]
        assert all(a >= b for a, b in pairwise(s))


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
        )
        day = shadow.state.score_days["2026-11-20"]
        assert day["scored"] == 2
        assert day["resumed"] == 1
        assert day["exits"] == 1
        assert day["brier_model"] >= 0
        assert day["brier_baseline"] >= 0

    def test_censored_episodes_are_not_scored(self) -> None:
        shadow = SojournShadow()
        _run(
            shadow,
            [
                (-TICK_STEP, {"present": True}),
                (0.0, {}),
                (SOJOURN_EPISODE_MAX_SECONDS, {}),
            ],
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
        )
        eps = shadow.state.episodes
        assert len(eps) == 2
        solo = SojournShadow()
        solo.state.episodes = [eps[0]]
        p = solo.survival_at(eps[1].silence_seconds)
        day = shadow.state.score_days["2026-11-20"]
        # Episode 1 scored against the empty table (neutral 0.5, y=0);
        # episode 2 against the one-exit table (p, y=0).
        assert day["brier_model"] == pytest.approx(0.5**2 + p**2)


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
        )
        state = SojournState.from_dict(shadow.state.to_dict())
        assert len(state.episodes) == 1
        assert state.episodes[0].outcome == "exits"
        assert state.episode_start is not None
        restored = SojournShadow(state)
        assert restored.survival_at(200) == pytest.approx(shadow.survival_at(200))

    def test_malformed_payload_falls_back(self) -> None:
        assert SojournState.from_dict({"episodes": "nope"}).episodes == []
        assert SojournState.from_dict({}).episodes == []

    def test_episode_cap(self) -> None:
        """Closing an episode trims the retained list to the cap."""
        shadow = SojournShadow()
        shadow.state.episodes = [
            SojournEpisode(
                start=T0 + timedelta(seconds=i),
                silence_seconds=10.0,
                outcome="exits",
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
            present=True,
            occupied=True,
            doors_open=False,
            neighbours={},
            half_life_seconds=HALF_LIFE,
        )
        assert len(shadow.state.episodes) == SOJOURN_EPISODES_KEPT
        # The oldest episode was dropped; the new one is last.
        assert shadow.state.episodes[-1].outcome == "resumed"
