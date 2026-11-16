"""Tests for manual-reference calibration shadow state."""

from custom_components.area_occupancy.data.manual_calibration import (
    MIN_CALIBRATION_SAMPLES,
    ManualCalibrationShadow,
    ManualCalibrationState,
)


def test_calibration_stays_unavailable_until_sample_gate() -> None:
    """Calibration does not affect output with too little evidence."""
    shadow = ManualCalibrationShadow()

    for _ in range(MIN_CALIBRATION_SAMPLES - 1):
        shadow.observe(0.9, False)

    assert shadow.calibrated_probability(0.9) is None


def test_calibration_learns_overconfident_predictions() -> None:
    """Repeated false high predictions reduce the calibrated probability."""
    shadow = ManualCalibrationShadow()

    for _ in range(MIN_CALIBRATION_SAMPLES):
        shadow.observe(0.9, False)

    calibrated = shadow.calibrated_probability(0.9)

    assert calibrated is not None
    assert calibrated < 0.9


def test_calibration_state_is_bounded_and_round_trips() -> None:
    """Persisted calibration parameters are safely bounded."""
    state = ManualCalibrationState.from_dict(
        {"slope": 100, "intercept": -100, "samples": 20}
    )

    restored = ManualCalibrationState.from_dict(state.to_dict())

    assert restored == state
    assert restored.slope == 4.0
    assert restored.intercept == -2.0
