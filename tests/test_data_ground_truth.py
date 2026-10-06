"""Tests for the learning ground-truth rule (PIR-only motion timeout)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from custom_components.area_occupancy.data.ground_truth import (
    PULSE_MIN_ACTIVATIONS,
    extend_pulses,
    pulse_like_sensors,
)
from custom_components.area_occupancy.db.utils import merge_overlapping_intervals

T = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _at(minutes: float, seconds: float = 0) -> datetime:
    return T + timedelta(minutes=minutes, seconds=seconds)


class TestPulseLikeSensors:
    def test_pir_short_pulses(self) -> None:
        assert pulse_like_sensors({"pir": [20.0] * 12}) == {"pir"}

    def test_mmwave_holds(self) -> None:
        """Median 300 s >= 120 s: a presence sensor, not extended."""
        assert pulse_like_sensors({"mmwave": [300.0] * 12}) == set()

    def test_too_little_history_counts_as_pir(self) -> None:
        assert pulse_like_sensors({"new": [600.0] * (PULSE_MIN_ACTIVATIONS - 1)}) == {
            "new"
        }


class TestExtendPulses:
    def test_pir_pulses_chain_into_one_stretch(self) -> None:
        """20 s pulses at 12:00, 12:02, 12:06 with a 300 s timeout.

        Extended: 12:00:00-12:05:20, 12:02:00-12:07:20, 12:06:00-12:11:20,
        which merge to 12:00:00-12:11:20 = 680 s (raw: 60 s in 3 pieces).
        """
        pulses = [(_at(m), _at(m, 20)) for m in (0, 2, 6)]

        merged = merge_overlapping_intervals(
            extend_pulses({"pir": pulses}, {"pir"}, timedelta(seconds=300))
        )

        assert merged == [(_at(0), _at(11, 20))]
        assert (merged[0][1] - merged[0][0]).total_seconds() == 680

    def test_presence_sensor_is_not_extended(self) -> None:
        holds = [(_at(0), _at(5)), (_at(10), _at(15))]

        merged = merge_overlapping_intervals(
            extend_pulses({"mmwave": holds}, set(), timedelta(seconds=300))
        )

        assert merged == holds
