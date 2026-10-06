"""Tests for the shadow-mode learned fusion (#501)."""

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.area_occupancy.const import FUSION_MIN_SAMPLES, MAX_WEIGHT
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.fusion import (
    FusionLearner,
    FusionState,
    FusionTick,
)
from custom_components.area_occupancy.utils import logit

T0 = datetime(2026, 9, 26, 12, 0, 0, tzinfo=UTC)


def _tick(seconds: float, features: dict[str, float], bias: float) -> FusionTick:
    return FusionTick(
        timestamp=T0 + timedelta(seconds=seconds), bias=bias, features=features
    )


class TestGradientStep:
    """Hand-computed logistic gradient expectations (Law 1 discipline)."""

    def test_single_occupied_tick_raises_weight(self) -> None:
        """Hand-computed: one y=1 tick moves w from 0.5 to ~0.545063.

        bias = logit(0.3) = -0.847298; x = 1.9; w0 = default = 0.5
        z = -0.847298 + 0.5*1.9 = 0.102702 -> p = sigmoid(z) = 0.525653
        grad = (p - 1)*x + l2*(w0 - d) = -0.901259 + 0
        w1 = 0.5 - 0.05*grad = 0.545063
        """
        learner = FusionLearner()
        ticks = [_tick(0, {"binary_sensor.door": 1.9}, logit(0.3))]
        occupied = [(T0 - timedelta(seconds=1), T0 + timedelta(seconds=1))]

        consumed = learner.update(ticks, occupied, {"binary_sensor.door": 0.5})

        assert consumed == 1
        assert learner.state.samples == 1
        assert learner.state.weights["binary_sensor.door"] == pytest.approx(
            0.545063, abs=1e-5
        )

    def test_single_empty_tick_lowers_weight(self) -> None:
        """Hand-computed: one y=0 tick moves w from 0.5 to ~0.450063.

        Same setup, y = 0: grad = p*x = 0.998741; w1 = 0.450063.
        """
        learner = FusionLearner()
        ticks = [_tick(0, {"binary_sensor.door": 1.9}, logit(0.3))]

        learner.update(ticks, [], {"binary_sensor.door": 0.5})

        assert learner.state.weights["binary_sensor.door"] == pytest.approx(
            0.450063, abs=1e-5
        )

    def test_weight_clamped_to_bounds(self) -> None:
        """Weights never go negative or above MAX_WEIGHT.

        Lower bound: an empty tick with a huge feature drives the
        gradient far below zero. Upper bound: an occupied tick predicted
        near-certainly EMPTY (w starts at 0, very negative bias) with a
        huge feature gives grad ~ (p-1)*100 ~ -95 -> unclamped step
        w = 0 + 0.05*95 = 4.75, clamped to MAX_WEIGHT.
        """
        learner = FusionLearner(FusionState(weights={"s.a": 0.01, "s.b": 0.0}))
        empty_tick = [_tick(0, {"s.a": 100.0}, logit(0.9))]
        # A second later: the learner skips ticks it has already trained on.
        occupied_tick = [_tick(1, {"s.b": 100.0}, logit(0.05))]

        learner.update(empty_tick, [], {"s.a": 0.01})
        learner.update(
            occupied_tick,
            [(T0, T0 + timedelta(seconds=2))],
            {"s.b": 0.0},
        )

        assert learner.state.weights["s.a"] == 0.0
        assert learner.state.weights["s.b"] == MAX_WEIGHT

    def test_cold_start_identity(self) -> None:
        """With no ticks consumed, nothing is learned and nothing reported.

        A cold-start home must be provably identical to today's behavior:
        the state holds no weights at all (the live pipeline keeps using
        its own defaults; shadow weights only exist after evidence).
        """
        learner = FusionLearner()
        consumed = learner.update([], [], {"s.a": 0.5})
        assert consumed == 0
        assert learner.state.weights == {}
        assert learner.state.samples == 0

    def test_absent_entities_are_not_touched(self) -> None:
        """An entity with zero evidence (absent from features) keeps its weight."""
        learner = FusionLearner(FusionState(weights={"s.absent": 0.7}))
        learner.update([_tick(0, {"s.present": 1.0}, 0.0)], [], {"s.present": 0.4})
        assert learner.state.weights["s.absent"] == 0.7

    def test_l2_pulls_toward_default(self) -> None:
        """With zero prediction error the only force is the anchor.

        bias = logit(0.5) = 0 and x = 0 would give no gradient at all, so
        use a tick whose prediction is exactly right by construction:
        impossible for a hard 0/1 label with sigmoid — instead verify
        directionally that a weight far above default drifts down even
        when the evidence supports occupancy (anchor + saturated error).
        """
        learner = FusionLearner(FusionState(weights={"s.a": 0.9}))
        # Strongly-predicted occupied tick that IS occupied: tiny error,
        # so the l2 term (0.01 * (0.9 - 0.2) = 0.007) dominates.
        occupied = [(T0 - timedelta(seconds=1), T0 + timedelta(seconds=1))]
        before = learner.state.weights["s.a"]
        learner.update([_tick(0, {"s.a": 10.0}, logit(0.5))], occupied, {"s.a": 0.2})
        after = learner.state.weights["s.a"]
        assert after < before


class TestTrainedThrough:
    """Each tick is trained on once, however many windows offer it."""

    def test_overlapping_windows_train_each_tick_once(self) -> None:
        """The analysis offers its whole 24h window every hour.

        Offering the same two ticks again must change nothing: no second
        gradient step and no double-counted samples.
        """
        learner = FusionLearner()
        ticks = [
            _tick(0, {"s.a": 1.9}, logit(0.3)),
            _tick(60, {"s.a": 1.9}, logit(0.3)),
        ]

        assert learner.update(ticks, [], {"s.a": 0.5}) == 2
        weight = learner.state.weights["s.a"]
        assert learner.update(ticks, [], {"s.a": 0.5}) == 0

        assert learner.state.weights["s.a"] == weight
        assert learner.state.samples == 2
        assert learner.state.trained_through == T0 + timedelta(seconds=60)

    def test_only_newer_ticks_are_consumed(self) -> None:
        learner = FusionLearner()
        learner.update([_tick(0, {"s.a": 1.0}, 0.0)], [], {"s.a": 0.5})

        consumed = learner.update(
            [_tick(0, {"s.a": 1.0}, 0.0), _tick(30, {"s.a": 1.0}, 0.0)],
            [],
            {"s.a": 0.5},
        )

        assert consumed == 1
        assert learner.state.samples == 2


class TestSnapshot:
    """Diagnostics gating and shape."""

    def test_below_gate_reports_counters_only(self) -> None:
        learner = FusionLearner(
            FusionState(weights={"s.a": 0.4}, samples=FUSION_MIN_SAMPLES - 1)
        )
        block = learner.snapshot({"s.a": 0.5})
        assert block["shadow_mode"] is True
        assert block["engaged"] is False
        assert "weights" not in block

    def test_at_gate_reports_weights(self) -> None:
        learner = FusionLearner(
            FusionState(weights={"s.a": 0.4}, samples=FUSION_MIN_SAMPLES)
        )
        block = learner.snapshot({"s.a": 0.5})
        assert block["weights"]["s.a"] == {
            "learned_weight": 0.4,
            "effective_weight": 0.5,
            "samples": 0,
        }


class TestStateSerialization:
    """Storage round-trip."""

    def test_round_trip(self) -> None:
        state = FusionState(weights={"s.a": 0.42}, samples=123)
        restored = FusionState.from_dict(state.to_dict())
        assert restored.weights == {"s.a": 0.42}
        assert restored.samples == 123
        assert restored.trained_through is None

    def test_round_trip_keeps_the_watermark(self) -> None:
        state = FusionState(samples=5, trained_through=T0)
        assert FusionState.from_dict(state.to_dict()).trained_through == T0

    def test_payload_without_watermark_loads(self) -> None:
        """State saved before the watermark existed still loads."""
        state = FusionState.from_dict({"weights": {"s.a": 0.3}, "samples": 7})
        assert state.weights == {"s.a": 0.3}
        assert state.samples == 7
        assert state.trained_through is None

    def test_malformed_falls_back_to_empty(self) -> None:
        state = FusionState.from_dict({"weights": "garbage", "samples": "x"})
        assert state.weights == {}
        assert state.samples == 0


class TestCoordinatorWiring:
    """Fusion ticks recorded per update; shadow contract holds."""

    async def test_update_records_fusion_ticks(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        await coordinator.async_refresh()
        ticks = coordinator.fusion_ticks_for(area_name)
        assert ticks, "update() should record a fusion tick per area"
        tick = ticks[-1]
        # Bias is the logit of the area's prior at tick time — finite.
        assert tick.bias == pytest.approx(tick.bias)
        # MOTION/SLEEP never appear as features (self-labelling exclusion).
        area = coordinator.get_area(area_name)
        for entity_id in tick.features:
            input_type = area.entities.entities[entity_id].type.input_type
            assert input_type.value not in ("motion", "sleep")

    async def test_fusion_never_touches_probability_path(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Shadow contract: learned weights don't change area probability."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        before = area.probability()

        learner = coordinator.ensure_fusion_learner(area_name)
        learner.state.weights = dict.fromkeys(area.entities.entities, 0.0)
        learner.state.samples = 10**6

        assert area.probability() == before

    async def test_save_and_reload_round_trip(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        learner = coordinator.ensure_fusion_learner(area_name)
        learner.state.weights["binary_sensor.test"] = 0.33
        learner.state.samples = 42

        await coordinator.async_save_fusion_state()
        stored = await coordinator._fusion_store.async_load()  # noqa: SLF001
        restored = FusionState.from_dict(stored[area_name])
        assert restored.weights == {"binary_sensor.test": 0.33}
        assert restored.samples == 42


class TestSyncEntities:
    """Removed or re-meant sensors don't keep stale learning (live report)."""

    def test_removed_sensor_is_forgotten(self) -> None:
        state = FusionState(
            weights={"door_2": 0.99, "tv": 0.6},
            entity_samples={"door_2": 500, "tv": 800},
        )
        learner = FusionLearner(state)

        dropped = learner.sync_entities({"tv": "media|playing|None"})

        assert dropped == ["door_2"]
        assert state.weights == {"tv": 0.6}
        assert state.entity_samples == {"tv": 800}
        assert state.signatures == {"tv": "media|playing|None"}

    def test_changed_meaning_is_forgotten(self) -> None:
        state = FusionState(
            weights={"tv": 0.9},
            entity_samples={"tv": 800},
            signatures={"tv": "media|paused,playing|None"},
        )

        dropped = FusionLearner(state).sync_entities(
            {"tv": "media|buffering,playing|None"}
        )

        assert dropped == ["tv"]
        assert state.weights == {}

    def test_entity_samples_count_ticks_with_a_feature(self) -> None:
        now = datetime(2026, 10, 6, 12, tzinfo=UTC)
        ticks = [
            FusionTick(
                timestamp=now + timedelta(seconds=10 * i),
                bias=0.0,
                features={"a": 1.0} if i % 2 else {"a": 1.0, "b": 1.0},
            )
            for i in range(4)
        ]
        learner = FusionLearner()

        learner.update(ticks, [], {"a": 0.5, "b": 0.5})

        assert learner.state.samples == 4
        assert learner.state.entity_samples == {"a": 4, "b": 2}

    def test_new_fields_round_trip(self) -> None:
        state = FusionState(
            weights={"a": 0.4}, entity_samples={"a": 7}, signatures={"a": "x"}
        )
        restored = FusionState.from_dict(state.to_dict())
        assert restored.entity_samples == {"a": 7}
        assert restored.signatures == {"a": "x"}
