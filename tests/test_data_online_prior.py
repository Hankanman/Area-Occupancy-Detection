"""Tests for the shadow-mode online prior estimator (#500)."""

from datetime import UTC, datetime, timedelta

from custom_components.area_occupancy.const import MAX_PRIOR, MIN_PRIOR
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.online_prior import (
    MAX_TICK_GAP_SECONDS,
    OnlinePriorEstimator,
    OnlinePriorState,
)

# ruff: noqa: SLF001

T0 = datetime(2026, 7, 6, 12, 0, 0, tzinfo=UTC)


def _tick(est: OnlinePriorEstimator, seconds: float, active: bool) -> datetime:
    """Observe one tick at T0+seconds; return that timestamp."""
    now = T0 + timedelta(seconds=seconds)
    est.observe(motion_active=active, now=now)
    return now


class TestOnlinePriorEstimator:
    """Sufficient-statistics accumulation and the prior ratio."""

    def test_no_observations_yields_none(self) -> None:
        """Before any tick the prior is unknown, not a default."""
        est = OnlinePriorEstimator()
        assert est.prior(T0) is None
        assert est.observed_days(T0) == 0.0

    def test_half_occupied_stream(self) -> None:
        """Motion active for the first half of the window → prior ≈ 0.5.

        Ticks every 10s for 100s; motion active at ticks 0-4 (so the
        spans 0-10 … 40-50 accrue) and inactive from tick 5 on.
        """
        est = OnlinePriorEstimator()
        for i in range(11):
            _tick(est, 10 * i, active=i < 5)

        now = T0 + timedelta(seconds=100)
        # occupied 50s over a 100s period
        assert abs(est.prior(now) - 0.5) < 1e-9

    def test_prior_clamped_to_bounds(self) -> None:
        """Always-active and never-active streams hit the clamps."""
        always = OnlinePriorEstimator()
        never = OnlinePriorEstimator()
        for i in range(11):
            _tick(always, 10 * i, active=True)
            _tick(never, 10 * i, active=False)
        now = T0 + timedelta(seconds=100)
        assert always.prior(now) == MAX_PRIOR
        assert never.prior(now) == MIN_PRIOR

    def test_downtime_gap_adds_period_but_no_occupancy(self) -> None:
        """A gap beyond MAX_TICK_GAP_SECONDS grows only the denominator.

        Mirrors the DB path: recorder gaps contribute period, never
        occupied time — even if motion was active before the outage.
        """
        est = OnlinePriorEstimator()
        _tick(est, 0, active=True)
        _tick(est, 10, active=True)  # 10s occupied
        # HA restarts; next tick is one hour later
        gap = MAX_TICK_GAP_SECONDS + 3600
        _tick(est, 10 + gap, active=True)

        now = T0 + timedelta(seconds=10 + gap)
        expected = max(MIN_PRIOR, 10 / (10 + gap))  # raw ratio is below the floor
        assert abs(est.prior(now) - expected) < 1e-9

    def test_state_round_trip_preserves_accumulators(self) -> None:
        """to_dict/from_dict survives a restart without drift."""
        est = OnlinePriorEstimator()
        _tick(est, 0, active=True)
        _tick(est, 10, active=False)

        restored = OnlinePriorEstimator(OnlinePriorState.from_dict(est.state.to_dict()))

        assert restored.state.occupied_seconds == est.state.occupied_seconds
        assert restored.state.first_observation == est.state.first_observation
        assert restored.state.last_tick == est.state.last_tick
        assert restored.state.last_motion_active is False
        now = T0 + timedelta(seconds=20)
        assert restored.prior(now) == est.prior(now)

    def test_malformed_storage_falls_back_to_empty(self) -> None:
        """Corrupt persisted state resets rather than crashing setup."""
        restored = OnlinePriorState.from_dict({"first_observation": "not-a-date"})
        assert restored.occupied_seconds == 0.0
        assert restored.first_observation is None


class TestCoordinatorShadowWiring:
    """Tick feeding, persistence, and diagnostics exposure."""

    async def test_update_feeds_estimator(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Each update() tick advances the area's estimator."""
        area_name = coordinator.get_area_names()[0]

        await coordinator.update()

        estimator = coordinator.online_prior_for(area_name)
        assert estimator is not None
        assert estimator.state.first_observation is not None

    async def test_save_and_reload_round_trip(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """async_save_online_priors persists state the Store can reload."""
        area_name = coordinator.get_area_names()[0]
        await coordinator.update()
        await coordinator.async_save_online_priors()

        stored = await coordinator._online_prior_store.async_load()

        assert area_name in stored
        restored = OnlinePriorState.from_dict(stored[area_name])
        assert restored.first_observation is not None

    async def test_online_prior_never_touches_probability(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Shadow contract: estimator state doesn't change area probability."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        await coordinator.update()
        before = area.probability()

        estimator = coordinator.online_prior_for(area_name)
        estimator.state.occupied_seconds = 999999.0

        assert area.probability() == before
