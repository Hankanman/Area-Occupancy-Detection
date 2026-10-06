"""Tests for the away-mode shadow evidence (#584)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import PropertyMock, patch

import pytest

from custom_components.area_occupancy.const import AWAY_PRIOR
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.away_shadow import (
    AwayShadow,
    AwayShadowState,
)

T0 = datetime(2026, 10, 6, 12, tzinfo=UTC)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


class TestObserve:
    """Each tick credits the time since the last tick to the last tick's state."""

    def test_hand_computed_sequence(self) -> None:
        """10 s ticks: differ (empty) -> present -> empty -> home; then a 70 s gap.

        away 30 s, occupied 10 s, differ 10 s, differ-with-presence 0.
        """
        shadow = AwayShadow()
        steps = [
            (0, True, False, True, False),
            (10, True, True, True, True),
            (20, True, False, False, False),
            (30, False, False, False, False),
            (100, True, True, True, False),  # after a 70 s gap: nothing credited
        ]
        for seconds, away, present, live, away_occ in steps:
            shadow.observe(
                now=_t(seconds),
                away=away,
                present=present,
                live_occupied=live,
                away_occupied=away_occ,
            )

        state = shadow.state
        assert state.away_seconds == 30
        assert state.occupied_away_seconds == 10
        assert state.differ_seconds == 10
        assert state.differ_present_seconds == 0

    def test_hidden_presence_is_counted(self) -> None:
        """Away mode reading empty while someone is there: the key number."""
        shadow = AwayShadow()
        shadow.observe(
            now=_t(0), away=True, present=True, live_occupied=True, away_occupied=False
        )
        shadow.observe(
            now=_t(10), away=True, present=True, live_occupied=True, away_occupied=False
        )

        assert shadow.state.differ_present_seconds == 10
        assert shadow.snapshot()["would_hide_presence_hours"] == pytest.approx(
            10 / 3600, abs=0.01
        )

    def test_learned_away_prior_after_an_hour(self) -> None:
        """72 s occupied over 2 h away: 72 / 7200 = 0.01."""
        shadow = AwayShadow(
            AwayShadowState(away_seconds=7200, occupied_away_seconds=72)
        )
        assert shadow.learned_away_prior() == pytest.approx(0.01)
        assert (
            AwayShadow(AwayShadowState(away_seconds=1800)).learned_away_prior() is None
        )

    def test_snapshot_is_shadow(self) -> None:
        block = AwayShadow().snapshot()
        assert block["shadow_mode"] is True
        assert block["engaged"] is False
        assert block["away_prior"] == AWAY_PRIOR

    def test_round_trip(self) -> None:
        state = AwayShadowState(
            away_seconds=5, occupied_away_seconds=1, differ_seconds=2, last_tick=T0
        )
        assert AwayShadowState.from_dict(state.to_dict()) == state


class TestCoordinatorWiring:
    """The tick records the shadow only while away and never moves probability."""

    async def test_records_while_away_and_leaves_probability_alone(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        await coordinator.update()
        assert coordinator.away_shadow_for(area_name) is None
        before = area.probability()

        coordinator.hass.states.async_set("input_boolean.away", "on")
        with patch.object(
            type(coordinator.integration_config),
            "away_mode_entity",
            new_callable=PropertyMock,
            return_value="input_boolean.away",
        ):
            await coordinator.update()
            assert area.probability() == before

        assert coordinator.away_shadow_for(area_name) is not None
