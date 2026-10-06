"""Tests for the presence-continuity shadow (#558)."""

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.continuity_shadow import (
    CONTINUITY_MAX_HALF_LIFE,
    EPISODE_MAX_SECONDS,
    ContinuityShadow,
    ContinuityState,
    continuity_half_life,
    half_life_ratio,
)

T0 = datetime(2026, 10, 6, 12, 0, 0, tzinfo=UTC)
# #558's open-plan lounge: half-life 520 s, silence 0.85. Today's stretch is
# min(1 + 0.75 * 0.85, 1.75) = 1.6375, so 851.5 s; continuity's is
# 520 * (1 + 2 * 0.85) = 1404 s. Ratio 1404 / 851.5 = 1.648855.
LOUNGE_RATIO = 1404 / 851.5
# The lounge clears 610 s after its last motion today, so continuity
# clears at 610 * 1.648855 = 1005.80 s (the issue's 1006 s).
LIVE_CLEAR = 610
CONTINUITY_CLEAR = 610 * LOUNGE_RATIO


def _tick(
    shadow: ContinuityShadow,
    seconds: float,
    *,
    present: bool = False,
    occupied: bool = True,
    doors_open: bool = False,
    neighbour: bool = False,
) -> None:
    shadow.observe(
        T0 + timedelta(seconds=seconds),
        present=present,
        occupied=occupied,
        doors_open=doors_open,
        neighbours={"Hall": neighbour},
        half_life_ratio=LOUNGE_RATIO,
    )


def _episode(shadow: ContinuityShadow, *, live_clear: float = LIVE_CLEAR) -> None:
    """Sensors on, then quiet at T0, then today's model clears the room."""
    _tick(shadow, -10, present=True)
    _tick(shadow, 0)
    _tick(shadow, live_clear - 10)
    _tick(shadow, live_clear, occupied=False)


def _day(shadow: ContinuityShadow) -> dict:
    return shadow.state.days["2026-10-06"]


class TestHalfLife:
    def test_lounge_example(self) -> None:
        assert continuity_half_life(520, 851.5, 0.85) == pytest.approx(1404)
        assert half_life_ratio(520, 851.5, 0.85) == pytest.approx(1.648855, abs=1e-6)

    def test_capped(self) -> None:
        assert continuity_half_life(900, 900, 1.0) == CONTINUITY_MAX_HALF_LIFE

    def test_never_shorter_than_today(self) -> None:
        """A room whose own half-life is past the cap keeps today's."""
        assert continuity_half_life(2400, 2400, 0.5) == 2400

    def test_no_neighbours_changes_nothing(self) -> None:
        assert half_life_ratio(520, 520, 0.0) == 1.0


class TestOutcomes:
    def test_resumed_after_live_cleared(self) -> None:
        """Back at 800 s: today's model cleared at 610, continuity holds.

        Recovered: min(1005.80, 800) - 610 = 190 s of correct occupancy.
        """
        shadow = ContinuityShadow()
        _episode(shadow)
        _tick(shadow, 800, present=True, occupied=True)
        day = _day(shadow)
        assert day["resumed"] == 1
        assert day["live_missed"] == 1
        assert day["continuity_missed"] == 0
        assert day["recovered_seconds"] == pytest.approx(190)

    def test_resumed_after_both_cleared(self) -> None:
        """Back at 1100 s: both had cleared; continuity recovered 395.80 s."""
        shadow = ContinuityShadow()
        _episode(shadow)
        _tick(shadow, 1100, present=True)
        day = _day(shadow)
        assert day["live_missed"] == 1
        assert day["continuity_missed"] == 1
        assert day["recovered_seconds"] == pytest.approx(CONTINUITY_CLEAR - LIVE_CLEAR)

    def test_resumed_before_live_cleared_misses_nothing(self) -> None:
        shadow = ContinuityShadow()
        _tick(shadow, -10, present=True)
        _tick(shadow, 0)
        _tick(shadow, 300, present=True)
        day = _day(shadow)
        assert day["resumed"] == 1
        assert day["live_missed"] == 0
        assert day["recovered_seconds"] == 0

    def test_neighbour_lighting_up_in_the_window_is_an_exit(self) -> None:
        shadow = ContinuityShadow()
        _tick(shadow, -10, present=True)
        _tick(shadow, 0)
        _tick(shadow, 30, neighbour=True)
        assert _day(shadow)["exits"] == 1
        assert shadow.state.episode is None

    def test_neighbour_after_the_window_is_not_an_exit(self) -> None:
        """Nobody seen leaving: continuity's extra 395.80 s is a false hold."""
        shadow = ContinuityShadow()
        _episode(shadow)
        _tick(shadow, 900, occupied=False, neighbour=True)
        _tick(shadow, EPISODE_MAX_SECONDS, occupied=False, neighbour=True)
        day = _day(shadow)
        assert day["unexplained"] == 1
        assert day["false_holds"] == 1
        assert day["false_hold_seconds"] == pytest.approx(CONTINUITY_CLEAR - LIVE_CLEAR)

    def test_busy_neighbour_at_the_start_is_not_an_exit(self) -> None:
        """Someone already next door doesn't show this person leaving."""
        shadow = ContinuityShadow()
        _tick(shadow, -10, present=True, neighbour=True)
        _tick(shadow, 0, neighbour=True)
        _tick(shadow, 30, neighbour=True)
        assert shadow.state.episode is not None

    def test_door_opening_is_an_exit(self) -> None:
        shadow = ContinuityShadow()
        _tick(shadow, -10, present=True)
        _tick(shadow, 0)
        _tick(shadow, 400, doors_open=True)
        assert _day(shadow)["exits"] == 1

    def test_door_left_open_is_not_an_exit(self) -> None:
        shadow = ContinuityShadow()
        _tick(shadow, -10, present=True, doors_open=True)
        _tick(shadow, 0, doors_open=True)
        _tick(shadow, 400, doors_open=True)
        assert shadow.state.episode is not None

    def test_room_already_clear_starts_no_episode(self) -> None:
        shadow = ContinuityShadow()
        _tick(shadow, -10, present=True, occupied=False)
        _tick(shadow, 0, occupied=False)
        assert shadow.state.episode is None


class TestSummary:
    def test_rates(self) -> None:
        shadow = ContinuityShadow()
        _episode(shadow)
        _tick(shadow, 800, present=True)  # resumed: live missed, continuity held
        shadow.state.present = True
        _tick(shadow, 900)
        _tick(shadow, 930, neighbour=True)  # exit
        summary = shadow.summary()
        assert summary["episodes"] == 2
        assert summary["missed_hold_rate_live"] == 1.0
        assert summary["missed_hold_rate_continuity"] == 0.0
        assert summary["false_hold_rate"] == 0.0
        assert summary["recovered_minutes"] == pytest.approx(round(190 / 60, 1))

    def test_empty(self) -> None:
        assert ContinuityShadow().summary() is None

    def test_days_pruned(self) -> None:
        shadow = ContinuityShadow()
        for day in range(50):
            base = day * 86400
            _tick(shadow, base - 10, present=True)
            _tick(shadow, base)
            _tick(shadow, base + 30, neighbour=True)
            _tick(shadow, base + 40)
        assert len(shadow.state.days) == 45
        assert shadow.summary()["days"] == 30

    def test_round_trip(self) -> None:
        shadow = ContinuityShadow()
        _episode(shadow)
        restored = ContinuityState.from_dict(shadow.state.to_dict())
        assert restored == shadow.state

    def test_malformed_falls_back_to_empty(self) -> None:
        assert ContinuityState.from_dict({"episode": {"start": "x"}}) == (
            ContinuityState()
        )


class TestCoordinator:
    async def test_closed_door_rooms_are_not_scored(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        now = datetime.now(UTC)
        area.config.closed_door_hold = False
        coordinator._observe_continuity(area_name, area, now, True)  # noqa: SLF001
        assert coordinator.continuity_for(area_name) is not None
        area.config.closed_door_hold = True
        coordinator._observe_continuity(area_name, area, now, True)  # noqa: SLF001
        assert coordinator.continuity_for(area_name) is None

    async def test_never_touches_probability(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        before = area.probability()
        coordinator._observe_continuity(  # noqa: SLF001
            area_name, area, datetime.now(UTC), True
        )
        assert area.probability() == before
