"""Shadow calibration trained from human-confirmed occupancy references."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..utils import clamp_probability, logit, sigmoid

MIN_CALIBRATION_SAMPLES = 20
CALIBRATION_LEARNING_RATE = 0.03
CALIBRATION_L2 = 0.01
MIN_SLOPE = 0.25
MAX_SLOPE = 4.0
MIN_INTERCEPT = -2.0
MAX_INTERCEPT = 2.0


@dataclass
class ManualCalibrationState:
    """Persisted Platt-style calibration parameters."""

    slope: float = 1.0
    intercept: float = 0.0
    samples: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe representation."""
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, data: object) -> ManualCalibrationState:
        """Restore valid parameters, otherwise return a cold-start state."""
        if not isinstance(data, dict):
            return cls()
        try:
            slope = float(data.get("slope", 1.0))
            intercept = float(data.get("intercept", 0.0))
            samples = int(data.get("samples", 0))
        except (TypeError, ValueError, OverflowError):
            return cls()
        if not math.isfinite(slope) or not math.isfinite(intercept) or samples < 0:
            return cls()
        return cls(
            slope=min(max(slope, MIN_SLOPE), MAX_SLOPE),
            intercept=min(max(intercept, MIN_INTERCEPT), MAX_INTERCEPT),
            samples=samples,
        )


class ManualCalibrationShadow:
    """Learn a conservative probability calibration from manual labels."""

    def __init__(self, state: ManualCalibrationState | None = None) -> None:
        """Initialize from persisted state or a neutral mapping."""
        self.state = state or ManualCalibrationState()

    def observe(self, probability: float | None, occupied: bool) -> None:
        """Update the calibration with one human-confirmed reference."""
        if probability is None:
            return
        probability = clamp_probability(probability)
        feature = logit(probability)
        prediction = sigmoid(self.state.slope * feature + self.state.intercept)
        error = prediction - float(occupied)
        self.state.slope = min(
            max(
                self.state.slope
                - CALIBRATION_LEARNING_RATE
                * (error * feature + CALIBRATION_L2 * (self.state.slope - 1.0)),
                MIN_SLOPE,
            ),
            MAX_SLOPE,
        )
        self.state.intercept = min(
            max(
                self.state.intercept
                - CALIBRATION_LEARNING_RATE
                * (error + CALIBRATION_L2 * self.state.intercept),
                MIN_INTERCEPT,
            ),
            MAX_INTERCEPT,
        )
        self.state.samples += 1

    def calibrated_probability(self, probability: float | None) -> float | None:
        """Return the calibrated probability once the sample gate is met."""
        if probability is None or self.state.samples < MIN_CALIBRATION_SAMPLES:
            return None
        return clamp_probability(
            sigmoid(self.state.slope * logit(probability) + self.state.intercept)
        )
