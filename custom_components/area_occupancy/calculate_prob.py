"""Probability calculations for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime

from .types import (
    ProbabilityResult,
    SensorState,
    CalculationResult,
)
from .const import (
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from .decay_handler import DecayHandler

_LOGGER = logging.getLogger(__name__)


class ProbabilityCalculator:
    """Handle probability calculations."""

    def __init__(self, coordinator, probabilities) -> None:
        """Initialize the calculator."""
        self.coordinator = coordinator
        self.probabilities = probabilities
        self.current_probability = MIN_PROBABILITY
        self.previous_probability = MIN_PROBABILITY
        self.decay_handler = DecayHandler(coordinator.config)

    def _calculate_sensor_probability(
        self, entity_id: str, state: SensorState
    ) -> CalculationResult:
        """Calculate probability contribution from a single sensor."""
        if not state.get("availability", False):
            return (
                0.0,
                False,
                {"probability": 0.0, "weight": 0.0, "weighted_probability": 0.0},
            )

        sensor_config = self.probabilities.get_sensor_config(entity_id)
        if not sensor_config:
            return (
                0.0,
                False,
                {"probability": 0.0, "weight": 0.0, "weighted_probability": 0.0},
            )

        # Get learned or default priors
        learned_data = self.coordinator.learned_priors.get(entity_id, {})
        p_true = learned_data.get("prob_given_true", sensor_config["prob_given_true"])
        p_false = learned_data.get(
            "prob_given_false", sensor_config["prob_given_false"]
        )
        prior = learned_data.get("prior", sensor_config["default_prior"])

        if not self.probabilities.is_entity_active(entity_id, state["state"]):
            return (
                0.0,
                False,
                {"probability": 0.0, "weight": 0.0, "weighted_probability": 0.0},
            )

        # Calculate probability
        unweighted_prob = update_probability(prior, p_true, p_false)
        weighted_prob = unweighted_prob * sensor_config["weight"]

        return (
            weighted_prob,
            True,
            {
                "probability": unweighted_prob,
                "weight": sensor_config["weight"],
                "weighted_probability": weighted_prob,
            },
        )

    def _calculate_base_probability(
        self,
        sensor_states: dict[str, SensorState],
        active_triggers: list,
        sensor_probs: dict,
        now: datetime,
    ) -> float:
        """Calculate the base probability using only active sensors."""
        calculated_probability = MIN_PROBABILITY

        for entity_id, state in sensor_states.items():
            weighted_prob, is_active, prob_details = self._calculate_sensor_probability(
                entity_id, state
            )
            if is_active:
                active_triggers.append(entity_id)
                sensor_probs[entity_id] = prob_details
                # Combine using complementary probability
                calculated_probability = 1.0 - (
                    (1.0 - calculated_probability) * (1.0 - weighted_prob)
                )
                calculated_probability = min(calculated_probability, MAX_PROBABILITY)

        return calculated_probability

    def perform_calculation_logic(
        self,
        active_sensor_states: dict[str, SensorState],
        now: datetime,
    ) -> ProbabilityResult:
        """Core calculation logic."""
        _LOGGER.debug("Starting occupancy probability calculation.")
        active_triggers = []
        sensor_probs = {}
        threshold = self.coordinator.threshold

        # Calculate base probability
        calculated_probability = self._calculate_base_probability(
            active_sensor_states, active_triggers, sensor_probs, now
        )

        # Apply decay if needed
        actual_probability, decay_factor = self.decay_handler.calculate_decay(
            calculated_probability,
            self.previous_probability,
            threshold,
            now,
        )

        # Get decay status
        decay_status = self.decay_handler.get_decay_status(decay_factor)

        # Update current probability for next calculation
        self.current_probability = actual_probability

        return {
            "probability": actual_probability,
            "potential_probability": calculated_probability,
            "prior_probability": 0.0,
            "active_triggers": active_triggers,
            "sensor_probabilities": sensor_probs,
            "device_states": {},
            "decay_status": decay_status,
            "sensor_availability": {
                k: v.get("availability", False) for k, v in active_sensor_states.items()
            },
            "is_occupied": actual_probability >= threshold,
        }


def update_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
) -> float:
    """Perform a Bayesian update."""
    # Clamp input probabilities
    prior = max(MIN_PROBABILITY, min(prior, MAX_PROBABILITY))
    prob_given_true = max(MIN_PROBABILITY, min(prob_given_true, MAX_PROBABILITY))
    prob_given_false = max(MIN_PROBABILITY, min(prob_given_false, MAX_PROBABILITY))

    numerator = prob_given_true * prior
    denominator = numerator + prob_given_false * (1 - prior)
    if denominator == 0:
        return prior

    # Calculate and clamp result
    result = numerator / denominator
    return max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))
