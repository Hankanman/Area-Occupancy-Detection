"""Conditional probability calculator for Bayesian inference."""

from __future__ import annotations

import logging
from typing import Any
from collections import defaultdict

from homeassistant.const import (
    STATE_ON,
    STATE_PLAYING,
)
from homeassistant.util import dt as dt_util

from .types import (
    BayesianProbability,
    HistoricalData,
    StoredPattern,
    SensorCorrelation,
)
from .environmental_detection import EnvironmentalDetector

_LOGGER = logging.getLogger(__name__)


class ConditionalProbabilityCalculator:
    """Calculates P(Evidence|State) probabilities for Bayesian inference."""

    def __init__(
        self,
        env_detector: EnvironmentalDetector,
        min_samples: int = 100,
        correlation_threshold: float = 0.3,
    ) -> None:
        """Initialize the calculator."""
        self.env_detector = env_detector
        self.min_samples = min_samples
        self.correlation_threshold = correlation_threshold

        # Track sensor correlations
        self._correlations: dict[str, dict[str, SensorCorrelation]] = defaultdict(dict)

        # Track evidence counts
        self._evidence_counts = {
            "motion": {"occupied": 0, "vacant": 0, "total": 0},
            "media": {"occupied": 0, "vacant": 0, "total": 0},
            "appliance": {"occupied": 0, "vacant": 0, "total": 0},
            "environmental": {"occupied": 0, "vacant": 0, "total": 0},
        }

    def update_from_pattern(self, pattern: StoredPattern) -> None:
        """Update probability data from a new pattern."""
        try:
            # Extract occupancy state
            is_occupied = pattern["occupied"]

            # Update evidence counts
            self._update_evidence_counts(pattern, is_occupied)

            # Update sensor correlations
            self._update_correlations(pattern)

        except Exception as err:
            _LOGGER.error("Error updating from pattern: %s", err)

    def calculate_motion_probabilities(
        self, historical_data: HistoricalData
    ) -> BayesianProbability:
        """Calculate P(Motion|Occupied) and P(Motion|Unoccupied)."""
        try:
            counts = self._evidence_counts["motion"]

            if counts["total"] < self.min_samples:
                return self._get_default_probability("motion")

            # Calculate basic probabilities
            p_given_true = (
                counts["occupied"] / counts["total"] if counts["total"] > 0 else 0.8
            )
            p_given_false = (
                counts["vacant"] / counts["total"] if counts["total"] > 0 else 0.2
            )

            # Adjust for sensor correlations
            correlation_factor = self._get_correlation_factor("motion")
            p_given_true = self._adjust_probability(p_given_true, correlation_factor)
            p_given_false = self._adjust_probability(p_given_false, correlation_factor)

            # Calculate confidence
            confidence = min(counts["total"] / self.min_samples, 1.0)

            return {
                "prob_given_true": p_given_true,
                "prob_given_false": p_given_false,
                "confidence": confidence,
                "last_updated": dt_util.utcnow(),
            }

        except Exception as err:
            _LOGGER.error("Error calculating motion probabilities: %s", err)
            return self._get_default_probability("motion")

    def calculate_media_probabilities(
        self, historical_data: HistoricalData
    ) -> BayesianProbability:
        """Calculate P(MediaActive|Occupied) and P(MediaActive|Unoccupied)."""
        try:
            counts = self._evidence_counts["media"]

            if counts["total"] < self.min_samples:
                return self._get_default_probability("media")

            # Calculate base probabilities
            p_given_true = (
                counts["occupied"] / counts["total"] if counts["total"] > 0 else 0.7
            )
            p_given_false = (
                counts["vacant"] / counts["total"] if counts["total"] > 0 else 0.3
            )

            # Adjust for time patterns
            time_adjustment = self._calculate_time_adjustment(historical_data)
            p_given_true = self._adjust_probability(p_given_true, time_adjustment)
            p_given_false = self._adjust_probability(p_given_false, time_adjustment)

            # Calculate confidence
            confidence = min(counts["total"] / self.min_samples, 1.0)

            return {
                "prob_given_true": p_given_true,
                "prob_given_false": p_given_false,
                "confidence": confidence,
                "last_updated": dt_util.utcnow(),
            }

        except Exception as err:
            _LOGGER.error("Error calculating media probabilities: %s", err)
            return self._get_default_probability("media")

    def calculate_appliance_probabilities(
        self, historical_data: HistoricalData
    ) -> BayesianProbability:
        """Calculate P(ApplianceActive|Occupied) and P(ApplianceActive|Unoccupied)."""
        try:
            counts = self._evidence_counts["appliance"]

            if counts["total"] < self.min_samples:
                return self._get_default_probability("appliance")

            # Calculate base probabilities
            p_given_true = (
                counts["occupied"] / counts["total"] if counts["total"] > 0 else 0.6
            )
            p_given_false = (
                counts["vacant"] / counts["total"] if counts["total"] > 0 else 0.4
            )

            # Adjust for usage patterns
            usage_adjustment = self._calculate_usage_adjustment(historical_data)
            p_given_true = self._adjust_probability(p_given_true, usage_adjustment)
            p_given_false = self._adjust_probability(p_given_false, usage_adjustment)

            # Calculate confidence
            confidence = min(counts["total"] / self.min_samples, 1.0)

            return {
                "prob_given_true": p_given_true,
                "prob_given_false": p_given_false,
                "confidence": confidence,
                "last_updated": dt_util.utcnow(),
            }

        except Exception as err:
            _LOGGER.error("Error calculating appliance probabilities: %s", err)
            return self._get_default_probability("appliance")

    def calculate_environmental_probabilities(
        self, historical_data: HistoricalData
    ) -> BayesianProbability:
        """Calculate P(EnvChange|Occupied) and P(EnvChange|Unoccupied)."""
        try:
            counts = self._evidence_counts["environmental"]

            if counts["total"] < self.min_samples:
                return self._get_default_probability("environmental")

            # Get environmental sensor confidences
            sensor_confidences = [
                self.env_detector.get_confidence(entity_id)
                for entity_id in historical_data.get("environmental_baselines", {})
            ]

            avg_confidence = (
                sum(sensor_confidences) / len(sensor_confidences)
                if sensor_confidences
                else 0.0
            )

            # Calculate base probabilities
            p_given_true = (
                counts["occupied"] / counts["total"] if counts["total"] > 0 else 0.6
            )
            p_given_false = (
                counts["vacant"] / counts["total"] if counts["total"] > 0 else 0.3
            )

            # Adjust based on sensor reliability
            p_given_true = self._adjust_probability(p_given_true, avg_confidence)
            p_given_false = self._adjust_probability(p_given_false, avg_confidence)

            # Calculate overall confidence
            count_confidence = min(counts["total"] / self.min_samples, 1.0)
            confidence = min(count_confidence, avg_confidence)

            return {
                "prob_given_true": p_given_true,
                "prob_given_false": p_given_false,
                "confidence": confidence,
                "last_updated": dt_util.utcnow(),
            }

        except Exception as err:
            _LOGGER.error("Error calculating environmental probabilities: %s", err)
            return self._get_default_probability("environmental")

    def _update_evidence_counts(
        self, pattern: StoredPattern, is_occupied: bool
    ) -> None:
        """Update evidence counts from pattern."""
        # Process each sensor type
        for entity_id, state in pattern["sensor_states"].items():
            sensor_type = self._get_sensor_type(entity_id)
            if not sensor_type:
                continue

            counts = self._evidence_counts[sensor_type]
            counts["total"] += 1

            if self._is_active_state(sensor_type, state):
                if is_occupied:
                    counts["occupied"] += 1
                else:
                    counts["vacant"] += 1

    def _update_correlations(self, pattern: StoredPattern) -> None:
        """Update sensor correlations from pattern."""
        active_sensors = set()

        # Find active sensors
        for entity_id, state in pattern["sensor_states"].items():
            sensor_type = self._get_sensor_type(entity_id)
            if sensor_type and self._is_active_state(sensor_type, state):
                active_sensors.add(entity_id)

        # Update correlations for each sensor pair
        for sensor1 in active_sensors:
            for sensor2 in active_sensors:
                if sensor1 >= sensor2:
                    continue

                if sensor2 not in self._correlations[sensor1]:
                    self._correlations[sensor1][sensor2] = {
                        "correlation": 0.0,
                        "sample_count": 0,
                        "confidence": 0.0,
                        "last_updated": dt_util.utcnow(),
                    }

                corr = self._correlations[sensor1][sensor2]
                corr["sample_count"] += 1

                # Update correlation using exponential moving average
                alpha = 0.1
                new_correlation = 1.0  # Both sensors active together
                corr["correlation"] = (
                    alpha * new_correlation + (1 - alpha) * corr["correlation"]
                )
                corr["confidence"] = min(corr["sample_count"] / self.min_samples, 1.0)
                corr["last_updated"] = dt_util.utcnow()

    def _get_correlation_factor(self, sensor_type: str) -> float:
        """Calculate correlation factor for sensor type."""
        correlations = []

        # Collect relevant correlations
        for sensor1, corr_dict in self._correlations.items():
            if self._get_sensor_type(sensor1) != sensor_type:
                continue

            for sensor2, corr in corr_dict.items():
                if (
                    corr["confidence"] >= 0.5
                    and corr["correlation"] >= self.correlation_threshold
                ):
                    correlations.append(corr["correlation"])

        # Calculate factor
        if not correlations:
            return 1.0

        # Higher correlations reduce the weight of individual sensors
        avg_correlation = sum(correlations) / len(correlations)
        return 1.0 - (avg_correlation * 0.5)  # Max 50% reduction

    def _get_sensor_type(self, entity_id: str) -> str | None:
        """Determine sensor type from entity ID."""
        # Implementation would match entity_id to configured sensors
        return "motion"  # Placeholder

    def _is_active_state(self, sensor_type: str, state: str) -> bool:
        """Determine if a state indicates activity."""
        if sensor_type == "motion":
            return state == STATE_ON
        if sensor_type == "media":
            return state == STATE_PLAYING
        if sensor_type == "appliance":
            return state == STATE_ON
        return False

    def _adjust_probability(self, prob: float, factor: float) -> float:
        """Adjust a probability by a factor while keeping it valid."""
        return max(0.0, min(1.0, prob * factor))

    def _calculate_time_adjustment(self, historical_data: HistoricalData) -> float:
        """Calculate time-based adjustment factor."""
        # Implementation would use time patterns
        return 1.0  # Placeholder

    def _calculate_usage_adjustment(self, historical_data: HistoricalData) -> float:
        """Calculate usage pattern adjustment factor."""
        # Implementation would use usage patterns
        return 1.0  # Placeholder

    def _get_default_probability(self, sensor_type: str) -> BayesianProbability:
        """Get default probabilities for sensor type."""
        defaults = {
            "motion": (0.9, 0.1),
            "media": (0.8, 0.2),
            "appliance": (0.7, 0.3),
            "environmental": (0.6, 0.4),
        }
        prob_true, prob_false = defaults.get(sensor_type, (0.5, 0.5))

        return {
            "prob_given_true": prob_true,
            "prob_given_false": prob_false,
            "confidence": 0.5,
            "last_updated": dt_util.utcnow(),
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        return {
            "evidence_counts": self._evidence_counts,
            "correlations": {
                f"{s1}-{s2}": corr
                for s1, corr_dict in self._correlations.items()
                for s2, corr in corr_dict.items()
            },
        }
