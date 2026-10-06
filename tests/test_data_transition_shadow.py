"""Tests for live area transitions (#603)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.transition_shadow import (
    TransitionShadow,
    TransitionShadowState,
    _hour_of_week,
)
from custom_components.area_occupancy.db.transitions import (
    _AreaEvent,
    _detect_transitions,
)

T = datetime(2026, 10, 7, 12, tzinfo=UTC)
ADJACENCY = {
    "Bedroom": {"Hall"},
    "Hall": {"Bedroom", "Kitchen"},
    "Kitchen": {"Hall", "Lounge"},
    "Lounge": {"Kitchen"},
}
# (seconds, area, now occupied?) — Bedroom 0-100, Hall 110-200,
# Kitchen from 230, Lounge from 400.
STEPS = [
    (0, "Bedroom", True),
    (100, "Bedroom", False),
    (110, "Hall", True),
    (200, "Hall", False),
    (230, "Kitchen", True),
    (400, "Lounge", True),
]


def _t(seconds: float) -> datetime:
    return T + timedelta(seconds=seconds)


def _live() -> TransitionShadow:
    shadow = TransitionShadow()
    labels: dict[str, bool] = {}
    for seconds, area, occupied in STEPS:
        labels[area] = occupied
        shadow.observe(now=_t(seconds), labels=dict(labels), adjacency=ADJACENCY)
    return shadow


def _flatten(shadow: TransitionShadow) -> list[tuple[str, str, str, int]]:
    out = []
    for chain, by_hour in shadow.state.counts.items():
        from_area, mid_area = chain.split("|")
        for hour, tos in by_hour.items():
            out.extend(
                (from_area, mid_area, to_area, hour)
                for to_area, n in tos.items()
                if n > 0.5
            )
    return sorted(out)


class TestDetection:
    def test_hand_computed_walk(self) -> None:
        """Bedroom -> Hall -> Kitchen, with the 2-hop chain; Lounge too late."""
        hour = _hour_of_week(_t(110))
        assert _flatten(_live()) == sorted(
            [
                ("Bedroom", "", "Hall", hour),
                ("Hall", "", "Kitchen", hour),
                ("Bedroom", "Hall", "Kitchen", hour),
            ]
        )

    def test_matches_the_database_detector(self) -> None:
        """The same event stream through db.transitions gives the same set."""
        events = [
            _AreaEvent(timestamp=_t(s), area_name=a, is_start=occ)
            for s, a, occ in STEPS
        ]
        assert sorted(_detect_transitions(events, ADJACENCY)) == _flatten(_live())

    def test_non_adjacent_rooms_are_not_counted(self) -> None:
        shadow = TransitionShadow()
        shadow.observe(now=_t(0), labels={"Bedroom": True}, adjacency=ADJACENCY)
        shadow.observe(now=_t(10), labels={"Bedroom": False}, adjacency=ADJACENCY)
        shadow.observe(
            now=_t(20), labels={"Bedroom": False, "Lounge": True}, adjacency=ADJACENCY
        )
        assert shadow.state.counts == {}

    def test_counts_decay_with_the_recency_half_life(self) -> None:
        shadow = _live()
        before = shadow.state.counts["Bedroom|"]
        hour = next(iter(before))
        shadow.observe(now=_t(400) + timedelta(days=30), labels={}, adjacency=ADJACENCY)
        # 30 days is one half-life; the count, made at 110 s, had also
        # decayed for the 290 s before that (x0.99992).
        assert shadow.state.counts["Bedroom|"][hour]["Hall"] == pytest.approx(
            0.5, abs=1e-3
        )


class TestCompare:
    def test_next_room_shares(self) -> None:
        """Live Hall -> Kitchen 4, Bedroom 1 (0.8/0.2); DB 3/2 (0.6/0.4): 0.2."""
        shadow = TransitionShadow(
            TransitionShadowState(counts={"Hall|": {12: {"Kitchen": 4, "Bedroom": 1}}})
        )
        result = shadow.compare({"Hall|": {12: {"Kitchen": 3, "Bedroom": 2}}})
        assert result["max_diff"] == pytest.approx(0.2)

    def test_thin_chains_are_not_compared(self) -> None:
        shadow = TransitionShadow(
            TransitionShadowState(counts={"Hall|": {12: {"Kitchen": 2}}})
        )
        assert shadow.compare({"Hall|": {12: {"Bedroom": 9}}})["max_diff"] is None

    def test_round_trip(self) -> None:
        shadow = _live()
        assert TransitionShadowState.from_dict(shadow.state.to_dict()) == shadow.state


class TestCoordinatorWiring:
    async def test_ticks_feed_the_shadow(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        await coordinator.update()
        area_name = coordinator.get_area_names()[0]
        assert area_name in coordinator.transition_shadow.state.occupied
