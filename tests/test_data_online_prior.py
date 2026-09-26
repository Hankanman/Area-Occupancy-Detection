"""Tests for the shadow-mode online prior estimator (#500)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from custom_components.area_occupancy.const import (
    MAX_PRIOR,
    MIN_PRIOR,
    ONLINE_PRIOR_DIFF_HISTORY_DAYS,
    ONLINE_PRIOR_STORE_KEY_PREFIX,
    ONLINE_PRIOR_STORE_VERSION,
)
from custom_components.area_occupancy.coordinator import (
    AreaOccupancyCoordinator,
    OnlinePriorStore,
)
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.data.online_prior import (
    MAX_TICK_GAP_SECONDS,
    OnlinePriorEstimator,
    OnlinePriorState,
)
from homeassistant.util import dt as dt_util

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

    async def test_async_shutdown_persists_online_prior_state(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Shutdown must save shadow state, not just the hourly pipeline.

        Regression test: previously only the hourly analysis pipeline
        called ``async_save_online_priors``, so a restart between pipeline
        runs silently dropped the occupied-seconds numerator while the
        period denominator (anchored at ``first_observation``, reloaded
        from the DB-backed prior on next setup) kept growing — biasing
        the online prior low across every restart.
        """
        area_name = coordinator.get_area_names()[0]
        await coordinator.update()
        estimator = coordinator.online_prior_for(area_name)
        assert estimator is not None
        # Force a known nonzero numerator so this test can't pass by
        # "saving zero" — asserting equality with whatever update() happened
        # to accrue wouldn't catch a save that silently dropped the value.
        estimator.state.occupied_seconds = 42.0
        occupied_seconds_before = estimator.state.occupied_seconds
        first_observation_before = estimator.state.first_observation

        await coordinator.async_shutdown()

        stored = await coordinator._online_prior_store.async_load()
        assert area_name in stored
        restored = OnlinePriorState.from_dict(stored[area_name])
        assert restored.occupied_seconds == occupied_seconds_before
        assert restored.first_observation == first_observation_before

    @pytest.mark.parametrize("input_type", [InputType.MEDIA, InputType.SLEEP])
    async def test_presence_definition_includes_media_and_sleep(
        self, coordinator: AreaOccupancyCoordinator, input_type: InputType
    ) -> None:
        """Numerator must match get_occupied_intervals' truth: motion ∪ media ∪ sleep.

        Regression test: previously only ``InputType.MOTION`` evidence
        counted, so an area occupied purely by media/sleep evidence (no
        motion sensor firing) would silently accrue zero occupied_seconds
        even though the DB ground truth it's diffed against counts it as
        occupied.
        """
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # weight=0 keeps the stub out of the fusion feature loop (#501),
        # which is not what this test exercises.
        evidence_only_entity = SimpleNamespace(
            evidence=True,
            weight=0,
            type=SimpleNamespace(input_type=input_type),
        )

        # way to stub one evidence-bearing entity without a full config.
        original_entities = area.entities._entities
        area.entities._entities = {"fake.entity": evidence_only_entity}
        try:
            coordinator._record_shadow_tick(
                area_name, area, T0, probability=0.5, is_occupied=True
            )
        finally:
            area.entities._entities = original_entities

        estimator = coordinator.online_prior_for(area_name)
        assert estimator is not None
        assert estimator.state.last_motion_active is True


class TestWeeklySlotAccumulators:
    """The 168-bucket extension (#500 phase 2)."""

    def test_slot_accrual_and_time_prior(self) -> None:
        """Hand-computed: 40 min occupied of 60 min observed in one slot.

        T0 is Monday 2026-07-06 12:00 UTC -> hour_of_week 0*24+12 = 12
        (timezone pinned to UTC). 120 ticks 30 s apart, first 80 spans
        active: occupied = 2400 s, total = 3600 s -> ratio 2/3, above
        the 3600 s observation floor exactly at the final tick.
        """
        original_tz = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(dt_util.UTC)
        try:
            est = OnlinePriorEstimator()
            _tick(est, 0, True)
            for i in range(1, 121):
                _tick(est, 30.0 * i, i < 80)
            slot = 12
            assert est.state.slot_total_seconds[slot] == pytest.approx(3600.0)
            assert est.state.slot_occupied_seconds[slot] == pytest.approx(2400.0)
            assert est.time_prior(slot) == pytest.approx(2400.0 / 3600.0)
            assert est.observed_slot_count() == 1
        finally:
            dt_util.set_default_time_zone(original_tz)

    def test_time_prior_none_below_observation_floor(self) -> None:
        """A slot under the 3600 s floor reports None, not a noisy ratio."""
        original_tz = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(dt_util.UTC)
        try:
            est = OnlinePriorEstimator()
            _tick(est, 0, True)
            _tick(est, 30, True)  # 30 s observed in slot 12
            assert est.state.slot_total_seconds[12] == pytest.approx(30.0)
            assert est.time_prior(12) is None
            assert est.observed_slot_count() == 0
        finally:
            dt_util.set_default_time_zone(original_tz)

    def test_span_attributed_to_start_hour_bucket(self) -> None:
        """A span crossing an hour boundary lands in the STARTING bucket."""
        original_tz = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(dt_util.UTC)
        try:
            est = OnlinePriorEstimator()
            # Tick at 12:59:50, next at 13:00:20: the 30 s span starts in
            # hour 12 -> slot 12, none of it in slot 13.
            base = 59 * 60 + 50
            _tick(est, base, True)
            _tick(est, base + 30, False)
            assert est.state.slot_total_seconds == {12: pytest.approx(30.0)}
            assert 13 not in est.state.slot_total_seconds
        finally:
            dt_util.set_default_time_zone(original_tz)

    def test_local_timezone_keys_buckets(self) -> None:
        """Buckets follow local wall clock, matching the DB time priors."""
        original_tz = dt_util.DEFAULT_TIME_ZONE
        # UTC+2: Monday 12:xx UTC is Monday 14:xx local -> slot 14.
        dt_util.set_default_time_zone(ZoneInfo("Europe/Helsinki"))
        try:
            est = OnlinePriorEstimator()
            _tick(est, 0, True)
            _tick(est, 30, False)
            assert 15 in est.state.slot_total_seconds  # EEST = UTC+3 in July
        finally:
            dt_util.set_default_time_zone(original_tz)

    def test_downtime_gap_contributes_to_no_slot(self) -> None:
        """A gap over MAX_TICK_GAP_SECONDS adds to no slot denominator."""
        original_tz = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(dt_util.UTC)
        try:
            est = OnlinePriorEstimator()
            _tick(est, 0, True)
            _tick(est, MAX_TICK_GAP_SECONDS + 1, True)
            assert est.state.slot_total_seconds == {}
        finally:
            dt_util.set_default_time_zone(original_tz)


class TestStateV2Serialization:
    """v1->v2 storage compatibility: never zero the scalar accumulators."""

    def test_v1_payload_restores_scalars_with_empty_buckets(self) -> None:
        v1 = {
            "occupied_seconds": 1234.5,
            "first_observation": T0.isoformat(),
            "last_tick": T0.isoformat(),
            "last_motion_active": True,
        }
        state = OnlinePriorState.from_dict(v1)
        assert state.occupied_seconds == pytest.approx(1234.5)
        assert state.first_observation is not None
        assert state.slot_occupied_seconds == {}
        assert state.slot_total_seconds == {}
        assert state.diff_history == []

    def test_v2_round_trip_preserves_buckets_and_history(self) -> None:
        state = OnlinePriorState(
            occupied_seconds=10.0,
            first_observation=T0,
            slot_occupied_seconds={12: 40.0},
            slot_total_seconds={12: 60.0, 13: 5.0},
            diff_history=[
                {
                    "date": "2026-09-25",
                    "scalar_diff": 0.01,
                    "bucket_diff": None,
                    "buckets": 0,
                }
            ],
        )
        restored = OnlinePriorState.from_dict(state.to_dict())
        assert restored.slot_occupied_seconds == {12: 40.0}
        assert restored.slot_total_seconds == {12: 60.0, 13: 5.0}
        assert restored.diff_history[0]["date"] == "2026-09-25"

    async def test_store_migration_passes_v1_data_through(self, hass) -> None:
        """Store v1 payload loads under v2 via OnlinePriorStore's migration."""
        key = f"{ONLINE_PRIOR_STORE_KEY_PREFIX}.migration_test"
        v1_store = OnlinePriorStore(hass, 1, key)
        await v1_store.async_save({"Kitchen": {"occupied_seconds": 777.0}})

        v2_store = OnlinePriorStore(hass, ONLINE_PRIOR_STORE_VERSION, key)
        loaded = await v2_store.async_load()
        assert loaded == {"Kitchen": {"occupied_seconds": 777.0}}
        state = OnlinePriorState.from_dict(loaded["Kitchen"])
        assert state.occupied_seconds == pytest.approx(777.0)


class TestDivergenceHistory:
    """Daily-collapsed shadow-diff record behind the 30-day gate."""

    def test_same_day_samples_collapse_to_daily_max(self) -> None:
        est = OnlinePriorEstimator()
        est.record_divergence(
            now=T0, scalar_diff=0.01, bucket_diff=0.02, buckets_compared=5
        )
        est.record_divergence(
            now=T0 + timedelta(hours=3),
            scalar_diff=-0.04,
            bucket_diff=0.01,
            buckets_compared=8,
        )
        assert len(est.state.diff_history) == 1
        entry = est.state.diff_history[0]
        assert entry["scalar_diff"] == pytest.approx(0.04)  # abs max
        assert entry["bucket_diff"] == pytest.approx(0.02)
        assert entry["buckets"] == 8

    def test_history_capped_and_streak_counted(self) -> None:
        est = OnlinePriorEstimator()
        # 100 days: day 60 breaches tolerance, the rest are within.
        for day in range(100):
            est.record_divergence(
                now=T0 + timedelta(days=day),
                scalar_diff=0.5 if day == 60 else 0.001,
                bucket_diff=None,
                buckets_compared=0,
            )
        assert len(est.state.diff_history) == ONLINE_PRIOR_DIFF_HISTORY_DAYS
        # Days 61..99 inclusive are within tolerance -> streak of 39.
        assert est.days_within_tolerance(0.02) == 39

    def test_streak_counts_all_when_never_breached(self) -> None:
        est = OnlinePriorEstimator()
        for day in range(31):
            est.record_divergence(
                now=T0 + timedelta(days=day),
                scalar_diff=0.005,
                bucket_diff=0.003,
                buckets_compared=168,
            )
        assert est.days_within_tolerance(0.02) == 31
