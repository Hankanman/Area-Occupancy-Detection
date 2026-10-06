"""Tests for the shadow sensor likelihoods (#603)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.analysis import db_likelihoods
from custom_components.area_occupancy.data.likelihood_shadow import (
    BinaryStats,
    LikelihoodShadow,
    LikelihoodShadowState,
    NumericStats,
    _decay_factor,
)

T = datetime(2026, 10, 7, 12, tzinfo=UTC)


def _s(seconds: float) -> datetime:
    return T + timedelta(seconds=seconds)


class TestBinaryStats:
    def test_duration_ratios_like_the_database(self) -> None:
        """2 h occupied, 30 min active; 2 h empty, 12 min active: 0.25 / 0.10."""
        stats = BinaryStats(
            occupied=7200, empty=7200, active_occupied=1800, active_empty=720
        )
        assert stats.likelihoods() == pytest.approx((0.25, 0.10))

    def test_clamped_to_the_database_bounds(self) -> None:
        stats = BinaryStats(
            occupied=7200, empty=7200, active_occupied=7200, active_empty=0
        )
        assert stats.likelihoods() == pytest.approx((0.95, 0.05))

    def test_none_until_an_hour_of_each_class(self) -> None:
        assert (
            BinaryStats(occupied=1800, empty=7200, active_occupied=900).likelihoods()
            is None
        )

    def test_never_active_while_occupied_means_defaults(self) -> None:
        """As the database: no active-while-occupied time, no learned value."""
        stats = BinaryStats(occupied=7200, empty=7200, active_empty=600)
        assert stats.likelihoods() is None


class TestNumericStats:
    def test_matches_numpy_like_the_database(self) -> None:
        occupied = [20.0, 22.0] * 5
        empty = [18.0, 19.0] * 5
        stats = NumericStats()
        for value in occupied:
            stats.add(value, occupied=True)
        for value in empty:
            stats.add(value, occupied=False)

        result = stats.stats()

        assert result["mean_occupied"] == pytest.approx(21.0)
        assert result["std_occupied"] == pytest.approx(1.0)
        assert result["mean_unoccupied"] == pytest.approx(18.5)
        assert result["std_unoccupied"] == pytest.approx(0.5)
        x = np.array(occupied + empty)
        y = np.array([1.0] * 10 + [0.0] * 10)
        assert result["correlation"] == pytest.approx(np.corrcoef(x, y)[0, 1])

    def test_none_below_ten_samples_per_class(self) -> None:
        stats = NumericStats()
        for _ in range(9):
            stats.add(20.0, occupied=True)
            stats.add(18.0, occupied=False)
        assert stats.stats() is None


class TestObserve:
    def test_ticks_credit_the_previous_state(self) -> None:
        """Three ticks credit 20 s occupied, 10 s of it active.

        Occupied and active, occupied and inactive, then empty, 10 s apart.
        """
        shadow = LikelihoodShadow()
        shadow.observe(now=_s(0), label=True, binary={"door": True}, numeric={})
        shadow.observe(now=_s(10), label=True, binary={"door": False}, numeric={})
        shadow.observe(now=_s(20), label=False, binary={"door": False}, numeric={})

        stats = shadow.state.binary["door"]
        assert stats.occupied == pytest.approx(20, rel=1e-4)
        assert stats.active_occupied == pytest.approx(10, rel=1e-4)
        assert stats.empty == 0

    def test_long_gap_credits_nothing(self) -> None:
        shadow = LikelihoodShadow()
        shadow.observe(now=_s(0), label=True, binary={"door": True}, numeric={})
        shadow.observe(now=_s(120), label=True, binary={"door": True}, numeric={})
        assert (
            "door" not in shadow.state.binary
            or shadow.state.binary["door"].occupied == 0
        )

    def test_numeric_samples_only_on_change(self) -> None:
        shadow = LikelihoodShadow()
        for i, value in enumerate([20.0, 20.0, 21.0]):
            shadow.observe(now=_s(10 * i), label=True, binary={}, numeric={"t": value})
        assert shadow.state.numeric["t"].w_occupied == pytest.approx(2, rel=1e-4)

    def test_fifteen_days_halves_the_sums(self) -> None:
        assert _decay_factor(15 * 86400) == pytest.approx(0.5)


class TestCompareAndStreak:
    def test_gap_to_the_database(self) -> None:
        state = LikelihoodShadowState(
            binary={"door": BinaryStats(7200, 7200, 1800, 720)}
        )
        result = LikelihoodShadow(state).compare(
            {"door": {"p_true": 0.28, "p_false": 0.10}}
        )
        assert result["max_diff"] == pytest.approx(0.03)
        assert result["sensors"]["door"]["diff"] == pytest.approx(0.03)

    def test_daily_worst_case_and_streak(self) -> None:
        shadow = LikelihoodShadow()
        shadow.record_divergence("2026-10-05", 0.2)
        shadow.record_divergence("2026-10-06", 0.01)
        shadow.record_divergence("2026-10-07", 0.03)
        shadow.record_divergence("2026-10-07", 0.02)
        assert shadow.state.diff_history[-1]["max_diff"] == 0.03
        assert shadow.days_within_tolerance(0.05) == 2

    def test_round_trip_and_forget(self) -> None:
        shadow = LikelihoodShadow()
        shadow.observe(now=_s(0), label=True, binary={"a": True}, numeric={"t": 1.0})
        shadow.observe(now=_s(10), label=True, binary={"a": True}, numeric={"t": 2.0})
        restored = LikelihoodShadowState.from_dict(shadow.state.to_dict())
        assert restored == shadow.state
        shadow.forget_others({"t"})
        assert "a" not in shadow.state.binary


class TestCoordinatorWiring:
    async def test_ticks_record_and_db_values_compare(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        await coordinator.update()
        assert coordinator.likelihood_shadow_for(area_name) is not None

        door = area.entities.get_entity("binary_sensor.door_sensor")
        door.update_binary_likelihoods(
            {
                "prob_given_true": 0.3,
                "prob_given_false": 0.1,
                "analysis_error": None,
                "correlation_type": "binary_likelihood",
            }
        )
        values = db_likelihoods(area)
        assert values["binary_sensor.door_sensor"] == {"p_true": 0.3, "p_false": 0.1}
