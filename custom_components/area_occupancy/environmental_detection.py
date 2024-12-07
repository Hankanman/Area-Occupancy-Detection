"""Environmental change detection and statistical analysis."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from collections import deque
from dataclasses import dataclass
import numpy as np


from .types import EnvironmentalReading, EnvironmentalAnalysis, SensorState

_LOGGER = logging.getLogger(__name__)


@dataclass
class SensorStats:
    """Statistical data for sensor readings."""

    mean: float
    std_dev: float
    noise_floor: float
    peak_to_peak: float
    gradient: float
    sample_count: int
    is_stable: bool


class EnvironmentalDetector:
    """Handles sophisticated environmental change detection."""

    def __init__(self, window_size: int = 100) -> None:
        """Initialize the detector."""
        self.window_size = window_size
        self._readings: dict[str, deque[EnvironmentalReading]] = {}
        self._last_analysis: dict[str, EnvironmentalAnalysis] = {}
        self._calibration_data: dict[str, SensorStats] = {}

    def add_reading(
        self, entity_id: str, state: SensorState, timestamp: datetime
    ) -> None:
        """Add a new sensor reading."""
        try:
            value = float(state["state"])

            if entity_id not in self._readings:
                self._readings[entity_id] = deque(maxlen=self.window_size)

            self._readings[entity_id].append(
                {
                    "value": value,
                    "timestamp": timestamp,
                    "baseline": self._get_current_baseline(entity_id),
                    "noise_level": self._get_noise_level(entity_id),
                    "gradient": self._calculate_gradient(entity_id),
                    "is_significant": False,  # Will be set by analysis
                }
            )

            # Update calibration periodically
            if len(self._readings[entity_id]) >= self.window_size:
                self._update_calibration(entity_id)

        except (ValueError, TypeError) as err:
            _LOGGER.debug("Invalid reading for %s: %s", entity_id, err)

    def analyze_change(self, entity_id: str) -> EnvironmentalAnalysis:
        """Analyze if recent changes are significant."""
        if entity_id not in self._readings or not self._readings[entity_id]:
            return EnvironmentalAnalysis(0.0, 0.0, 0.0, 0.0, False)

        try:
            # Get recent readings
            readings = self._readings[entity_id]
            if len(readings) < 2:
                return EnvironmentalAnalysis(0.0, 0.0, 0.0, 0.0, False)

            # Get calibration data
            stats = self._get_sensor_stats(entity_id)

            # Calculate current metrics
            current_reading = readings[-1]["value"]
            baseline = stats.mean
            noise_level = stats.noise_floor
            gradient = self._calculate_gradient(entity_id)

            # Dynamic threshold based on sensor stability
            base_threshold = stats.std_dev * 2
            if stats.is_stable:
                change_threshold = max(base_threshold, noise_level * 3)
            else:
                change_threshold = max(base_threshold, stats.peak_to_peak * 0.1)

            gradient_threshold = stats.gradient * 2

            # Determine if change is significant
            is_significant = (
                abs(current_reading - baseline) > change_threshold
                and abs(gradient) > gradient_threshold
                and abs(gradient) > noise_level
            )

            analysis = EnvironmentalAnalysis(
                baseline=baseline,
                noise_level=noise_level,
                change_threshold=change_threshold,
                gradient_threshold=gradient_threshold,
                is_significant=is_significant,
            )

            self._last_analysis[entity_id] = analysis
            return analysis

        except Exception as err:
            _LOGGER.error("Error analyzing change for %s: %s", entity_id, err)
            return self._last_analysis.get(
                entity_id, EnvironmentalAnalysis(0.0, 0.0, 0.0, 0.0, False)
            )

    def _update_calibration(self, entity_id: str) -> None:
        """Update sensor calibration data."""
        readings = self._readings[entity_id]
        if not readings:
            return

        values = [r["value"] for r in readings]
        timestamps = [r["timestamp"] for r in readings]

        # Calculate basic statistics
        mean = np.mean(values)
        std_dev = np.std(values)
        peak_to_peak = max(values) - min(values)

        # Calculate noise floor using median absolute deviation
        deviations = np.abs(values - np.median(values))
        noise_floor = (
            np.median(deviations) * 1.4826
        )  # Scale factor for normal distribution

        # Calculate average gradient
        time_diffs = [
            (t2 - t1).total_seconds() for t1, t2 in zip(timestamps[:-1], timestamps[1:])
        ]
        value_diffs = [v2 - v1 for v1, v2 in zip(values[:-1], values[1:])]
        gradients = [vd / td for vd, td in zip(value_diffs, time_diffs)]
        avg_gradient = np.mean(np.abs(gradients)) if gradients else 0.0

        # Determine stability
        stability_threshold = 0.1  # 10% variation
        is_stable = (std_dev / mean) < stability_threshold if mean != 0 else False

        self._calibration_data[entity_id] = SensorStats(
            mean=mean,
            std_dev=std_dev,
            noise_floor=noise_floor,
            peak_to_peak=peak_to_peak,
            gradient=avg_gradient,
            sample_count=len(readings),
            is_stable=is_stable,
        )

    def _get_sensor_stats(self, entity_id: str) -> SensorStats:
        """Get current sensor statistics."""
        if entity_id not in self._calibration_data:
            return SensorStats(0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
        return self._calibration_data[entity_id]

    def _get_current_baseline(self, entity_id: str) -> float:
        """Get current baseline for sensor."""
        if entity_id not in self._calibration_data:
            return 0.0
        return self._calibration_data[entity_id].mean

    def _get_noise_level(self, entity_id: str) -> float:
        """Get current noise level for sensor."""
        if entity_id not in self._calibration_data:
            return 0.0
        return self._calibration_data[entity_id].noise_floor

    def _calculate_gradient(self, entity_id: str) -> float:
        """Calculate current rate of change."""
        readings = self._readings.get(entity_id, [])
        if len(readings) < 2:
            return 0.0

        # Use last 5 readings for gradient
        recent = list(readings)[-5:]
        if len(recent) < 2:
            return 0.0

        times = [
            (r["timestamp"] - recent[0]["timestamp"]).total_seconds() for r in recent
        ]
        values = [r["value"] for r in recent]

        # Calculate linear regression slope
        try:
            slope, _ = np.polyfit(times, values, 1)
            return slope
        except Exception:
            # Fallback to simple difference
            return (values[-1] - values[0]) / (times[-1] - times[0])

    def get_confidence(self, entity_id: str) -> float:
        """Get confidence in sensor readings."""
        if entity_id not in self._calibration_data:
            return 0.0

        stats = self._calibration_data[entity_id]

        # Consider multiple factors
        factors = [
            min(stats.sample_count / self.window_size, 1.0),  # Sample size
            1.0 if stats.is_stable else 0.5,  # Stability
            (
                1.0 - min(stats.noise_floor / stats.peak_to_peak, 0.9)  # Signal quality
                if stats.peak_to_peak > 0
                else 0.0
            ),
        ]

        return min(factors)

    def get_diagnostics(self, entity_id: str) -> dict[str, Any]:
        """Get diagnostic data for sensor."""
        if entity_id not in self._calibration_data:
            return {}

        stats = self._calibration_data[entity_id]
        analysis = self._last_analysis.get(
            entity_id, EnvironmentalAnalysis(0.0, 0.0, 0.0, 0.0, False)
        )

        return {
            "calibration": {
                "mean": stats.mean,
                "std_dev": stats.std_dev,
                "noise_floor": stats.noise_floor,
                "peak_to_peak": stats.peak_to_peak,
                "gradient": stats.gradient,
                "sample_count": stats.sample_count,
                "is_stable": stats.is_stable,
            },
            "last_analysis": {
                "baseline": analysis.baseline,
                "noise_level": analysis.noise_level,
                "change_threshold": analysis.change_threshold,
                "gradient_threshold": analysis.gradient_threshold,
                "is_significant": analysis.is_significant,
            },
            "confidence": self.get_confidence(entity_id),
        }
