"""Tests for the shadow-mode accuracy metrics (trust score, #499)."""

from datetime import UTC, datetime, timedelta
import json
from unittest.mock import patch

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.analysis import _run_shadow_metrics
from custom_components.area_occupancy.data.metrics import (
    AccuracyMetrics,
    TickSample,
    compute_accuracy_metrics,
    metrics_to_diagnostics,
)
from custom_components.area_occupancy.diagnostics import (
    async_get_config_entry_diagnostics,
)
from homeassistant.core import HomeAssistant

T0 = datetime(2026, 7, 6, 12, 0, 0, tzinfo=UTC)


def _samples(spec: list[tuple[float, bool]]) -> list[TickSample]:
    """Build ticks 10s apart from (probability, occupied) pairs."""
    return [
        TickSample(timestamp=T0 + timedelta(seconds=10 * i), probability=p, occupied=o)
        for i, (p, o) in enumerate(spec)
    ]


class TestComputeAccuracyMetrics:
    """Pure-math behavior of compute_accuracy_metrics."""

    def test_empty_samples_returns_unknowns(self) -> None:
        """No data means rates are None (unknown), not zero."""
        out = compute_accuracy_metrics([], [])
        assert out.sample_count == 0
        assert out.expected_calibration_error is None
        assert out.agreement is None
        assert out.false_on_rate is None
        assert out.false_off_rate is None

    def test_well_calibrated_confident_stream(self) -> None:
        """0.9 while occupied and 0.1 while empty scores ECE=0.1, agreement=1."""
        # Truth: first 5 ticks occupied, last 5 not
        intervals = [(T0, T0 + timedelta(seconds=50))]
        samples = _samples([(0.9, True)] * 5 + [(0.1, False)] * 5)

        out = compute_accuracy_metrics(samples, intervals)

        assert out.sample_count == 10
        # Bin 0.9: mean 0.9 vs observed 1.0; bin 0.1: mean 0.1 vs 0.0
        # ECE = 0.5*0.1 + 0.5*0.1 = 0.1
        assert abs(out.expected_calibration_error - 0.1) < 1e-9
        assert out.agreement == 1.0
        assert out.false_on_rate == 0.0
        assert out.false_off_rate == 0.0
        assert out.decision_transitions == 1
        assert out.truth_transitions == 1

    def test_anti_calibrated_stream(self) -> None:
        """Confident and always wrong: agreement 0, max false rates."""
        intervals = [(T0, T0 + timedelta(seconds=50))]
        # Occupied period predicted empty; empty period predicted occupied
        samples = _samples([(0.1, False)] * 5 + [(0.9, True)] * 5)

        out = compute_accuracy_metrics(samples, intervals)

        assert out.agreement == 0.0
        assert out.false_on_rate == 1.0
        assert out.false_off_rate == 1.0
        # ECE: bin 0.1 observed 1.0 (|0.1-1.0|=0.9), bin 0.9 observed 0.0
        assert abs(out.expected_calibration_error - 0.9) < 1e-9

    def test_flip_counting_spurious_decisions(self) -> None:
        """A flappy decision stream racks up decision flips truth lacks."""
        intervals = [(T0, T0 + timedelta(seconds=100))]  # truth: always on
        samples = _samples(
            [(0.6, True), (0.4, False), (0.6, True), (0.4, False), (0.6, True)]
        )

        out = compute_accuracy_metrics(samples, intervals)

        assert out.decision_transitions == 4
        assert out.truth_transitions == 0
        assert out.false_off_rate == 2 / 5

    def test_interval_boundaries_are_half_open(self) -> None:
        """A tick exactly at interval end counts as not occupied."""
        intervals = [(T0, T0 + timedelta(seconds=10))]
        samples = _samples([(0.5, False), (0.5, False)])  # t=0 inside, t=10 at end

        out = compute_accuracy_metrics(samples, intervals)

        # First tick truth=True (false_off), second truth=False (true_off)
        assert out.false_off_rate == 1.0
        assert out.agreement == 0.5

    def test_probability_out_of_range_is_clamped_into_bins(self) -> None:
        """Probabilities outside [0,1] land in the edge bins, not crash."""
        out = compute_accuracy_metrics(
            _samples([(1.2, True), (-0.3, False)]), [(T0, T0 + timedelta(seconds=5))]
        )
        assert out.sample_count == 2
        assert sum(b.count for b in out.bins) == 2

    def test_quiet_tail_counts_as_unoccupied(self) -> None:
        """Samples after the last interval score as truth=False (#483 lesson)."""
        intervals = [(T0, T0 + timedelta(seconds=10))]
        samples = _samples([(0.9, True)] * 6)  # only the first tick is truly occupied

        out = compute_accuracy_metrics(samples, intervals)

        assert out.agreement == 1 / 6
        assert out.false_on_rate == 1.0


class TestDiagnosticsShape:
    """metrics_to_diagnostics output contract."""

    def test_json_safe_and_skips_empty_bins(self) -> None:
        """The block serializes and only carries populated bins."""
        intervals = [(T0, T0 + timedelta(seconds=50))]
        out = compute_accuracy_metrics(
            _samples([(0.9, True)] * 5 + [(0.1, False)] * 5), intervals
        )

        block = metrics_to_diagnostics(out)

        json.dumps(block)
        assert block["shadow_mode"] is True
        assert block["sample_count"] == 10
        assert len(block["calibration_bins"]) == 2
        assert block["window_start"] == T0.isoformat()


class TestShadowWiring:
    """Coordinator buffer, pipeline step, and diagnostics export."""

    async def test_update_records_tick_samples(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Every update() appends one sample per area."""
        area_name = coordinator.get_area_names()[0]

        await coordinator.update()
        await coordinator.update()

        samples = coordinator.accuracy_samples_for(area_name)
        assert len(samples) == 2
        assert 0.0 <= samples[0].probability <= 1.0
        assert isinstance(samples[0].occupied, bool)

    async def test_pipeline_step_caches_metrics(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """_run_accuracy_metrics scores buffered ticks against DB truth."""
        area_name = coordinator.get_area_names()[0]
        await coordinator.update()

        with patch.object(
            coordinator.db,
            "get_occupied_intervals",
            return_value=[],
        ):
            await _run_shadow_metrics(coordinator)

        metrics = coordinator.accuracy_metrics_for(area_name)
        assert metrics is not None
        assert metrics.sample_count >= 1
        # With no occupied intervals, truth is always False
        assert metrics.false_off_rate is None

    async def test_no_samples_no_metrics(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """The step is a no-op for areas with an empty tick buffer."""
        area_name = coordinator.get_area_names()[0]

        await _run_shadow_metrics(coordinator)

        assert coordinator.accuracy_metrics_for(area_name) is None

    async def test_diagnostics_exports_accuracy_block(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Cached metrics surface under current.accuracy and stay JSON-safe."""
        area_name = coordinator.get_area_names()[0]
        entry = coordinator.config_entry
        entry.runtime_data = coordinator

        coordinator.set_accuracy_metrics(
            area_name,
            compute_accuracy_metrics(
                _samples([(0.8, True)] * 3),
                [(T0, T0 + timedelta(seconds=100))],
            ),
        )

        result = await async_get_config_entry_diagnostics(hass, entry)

        area_snapshot = next(a for a in result["areas"] if a["area_name"] == area_name)
        accuracy = area_snapshot["current"]["accuracy"]
        assert accuracy["shadow_mode"] is True
        assert accuracy["sample_count"] == 3
        json.dumps(result)

    async def test_metrics_never_touch_probability_path(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Shadow contract: cached metrics don't change area probability."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        before = area.probability()

        coordinator.set_accuracy_metrics(area_name, AccuracyMetrics(sample_count=999))

        assert area.probability() == before
