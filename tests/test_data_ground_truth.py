"""Tests for the learning ground-truth rule (PIR-only motion timeout)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from custom_components.area_occupancy.data.ground_truth import (
    PULSE_MIN_ACTIVATIONS,
    LabelerState,
    LiveLabeler,
    extend_pulses,
    is_pulse_like,
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


class TestHysteresis:
    """Sensors near the cutoff keep their class (live: Kitchen 110 s, Dining 112 s)."""

    def test_band_keeps_previous_class(self) -> None:
        assert is_pulse_like([110.0] * 12, previous=True) is True
        assert is_pulse_like([110.0] * 12, previous=False) is False

    def test_leaving_the_band_flips(self) -> None:
        assert is_pulse_like([160.0] * 12, previous=True) is False
        assert is_pulse_like([80.0] * 12, previous=False) is True

    def test_first_classification_uses_the_cutoff(self) -> None:
        assert is_pulse_like([110.0] * 12) is True
        assert is_pulse_like([125.0] * 12) is False


def _s(seconds: float) -> datetime:
    return T + timedelta(seconds=seconds)


class TestLiveLabeler:
    """Each moment labelled as it happens, by the same rule as the database."""

    @staticmethod
    def _tick(labeler: LiveLabeler, seconds: float, motion_on: bool) -> bool:
        return labeler.observe(
            now=_s(seconds),
            motion={"pir": motion_on},
            other_present=False,
            timeout_seconds=300,
        )

    def test_pir_pulse_holds_for_the_timeout(self) -> None:
        """Pulse 0-20 s; 300 s timeout: occupied at 200 s, not at 330 s."""
        labeler = LiveLabeler()
        assert self._tick(labeler, 0, True)
        assert self._tick(labeler, 20, False)
        assert self._tick(labeler, 200, False)
        assert not self._tick(labeler, 330, False)
        assert labeler.state.durations["pir"] == [20.0]

    def test_presence_sensor_is_not_extended(self) -> None:
        """Ten 5-minute holds classify it as presence: off means off."""
        labeler = LiveLabeler()
        for i in range(10):
            start = i * 1000
            labeler.observe(
                now=_s(start),
                motion={"mm": True},
                other_present=False,
                timeout_seconds=300,
            )
            labeler.observe(
                now=_s(start + 300),
                motion={"mm": False},
                other_present=False,
                timeout_seconds=300,
            )
        assert labeler.state.pulse["mm"] is False
        assert not labeler.observe(
            now=_s(9 * 1000 + 310),
            motion={"mm": False},
            other_present=False,
            timeout_seconds=300,
        )

    def test_media_or_sleep_counts_as_present(self) -> None:
        assert LiveLabeler().observe(
            now=_s(0), motion={"pir": False}, other_present=True, timeout_seconds=300
        )

    def test_state_round_trips(self) -> None:
        labeler = LiveLabeler()
        self._tick(labeler, 0, True)
        self._tick(labeler, 20, False)
        restored = LabelerState.from_dict(labeler.state.to_dict())
        assert restored == labeler.state
