"""Probability calculations for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime

from .types import (
    ProbabilityState,
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
    """Handle probability calculations using Bayesian inference."""

    def __init__(self, coordinator, probabilities) -> None:
        """Initialize the calculator."""
        self.probabilities = probabilities
        self.current_probability = MIN_PROBABILITY
        self.previous_probability = MIN_PROBABILITY
        self.decay_handler = DecayHandler(coordinator.config)
        self.probability_state = coordinator.data
        self.learned_priors = coordinator.learned_priors
        self.type_priors = coordinator.type_priors

    def _calculate_sensor_probability(
        self, entity_id: str, state: SensorState
    ) -> CalculationResult:
        """Calculate probability contribution from a single sensor using Bayesian inference."""
        if not state.get("availability", False):
            _LOGGER.debug("Sensor %s is not available", entity_id)
            return (
                0.0,
                False,
                {"probability": 0.0, "weight": 0.0, "weighted_probability": 0.0},
            )

        sensor_config = self.probabilities.get_sensor_config(entity_id)
        if not sensor_config:
            _LOGGER.warning("No configuration found for sensor %s", entity_id)
            return (
                0.0,
                False,
                {"probability": 0.0, "weight": 0.0, "weighted_probability": 0.0},
            )

        # Get learned or default priors
        learned_data = self.probability_state.get("learned_priors", {}).get(
            entity_id, {}
        )
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

        # Calculate probability using Bayesian inference
        try:
            unweighted_prob = bayesian_update(prior, p_true, p_false)
            weighted_prob = unweighted_prob * sensor_config["weight"]

            _LOGGER.debug(
                "Sensor %s: prior=%.3f, p_true=%.3f, p_false=%.3f, prob=%.3f, weight=%.3f",
                entity_id,
                prior,
                p_true,
                p_false,
                unweighted_prob,
                sensor_config["weight"],
            )

            return (
                weighted_prob,
                True,
                {
                    "probability": unweighted_prob,
                    "weight": sensor_config["weight"],
                    "weighted_probability": weighted_prob,
                },
            )
        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error(
                "Error calculating probability for sensor %s: %s", entity_id, err
            )
            return (
                0.0,
                False,
                {"probability": 0.0, "weight": 0.0, "weighted_probability": 0.0},
            )

    def _calculate_base_probability(
        self,
        sensor_states: dict[str, SensorState],
        active_triggers: list,
        sensor_probs: dict,
    ) -> float:
        """Calculate the base probability using complementary probability for multiple sensors.

        This method uses the complementary probability approach to combine multiple sensor inputs.
        For each sensor, we calculate the probability of the area NOT being occupied given that sensor's state,
        then multiply these probabilities together. The final probability is 1 minus this product.

        This approach ensures that:
        1. Multiple positive inputs increase the probability
        2. Each sensor's weight is properly considered
        3. The result stays within valid probability bounds
        """
        calculated_probability = MIN_PROBABILITY
        complementary_prob = 1.0  # Start with 1.0 for multiplication

        for entity_id, state in sensor_states.items():
            weighted_prob, is_active, prob_details = self._calculate_sensor_probability(
                entity_id, state
            )
            if is_active:
                active_triggers.append(entity_id)
                sensor_probs[entity_id] = prob_details

                # Convert to complementary probability (probability of NOT being occupied)
                # and apply weight
                sensor_complementary = 1.0 - weighted_prob
                weight = prob_details["weight"]
                # Weight the complementary probability
                weighted_complementary = 1.0 - (weight * (1.0 - sensor_complementary))
                complementary_prob *= weighted_complementary

        # Convert back to probability of being occupied
        calculated_probability = 1.0 - complementary_prob
        calculated_probability = max(
            MIN_PROBABILITY, min(calculated_probability, MAX_PROBABILITY)
        )

        _LOGGER.debug(
            "Base probability calculation: complementary=%.3f, final=%.3f, active_sensors=%d",
            complementary_prob,
            calculated_probability,
            len(active_triggers),
        )

        return calculated_probability

    def _calculate_prior_probability(
        self, sensor_states: dict[str, SensorState]
    ) -> float:
        """Calculate the prior probability based on sensor states."""
        prior_sum = 0.0
        total_sensors = 0

        for entity_id, state in sensor_states.items():
            if not state.get("availability", False):
                continue

            sensor_config = self.probabilities.get_sensor_config(entity_id)
            if not sensor_config:
                continue

            learned_data = self.learned_priors.get(entity_id, {})
            prior = learned_data.get("prior", sensor_config["default_prior"])

            prior_sum += prior
            total_sensors += 1

        if total_sensors == 0:
            return MIN_PROBABILITY

        return max(MIN_PROBABILITY, min(prior_sum / total_sensors, MAX_PROBABILITY))

    def perform_calculation_logic(
        self,
        active_sensor_states: dict[str, SensorState],
        now: datetime,
    ) -> ProbabilityState:
        """Core calculation logic using Bayesian inference."""
        _LOGGER.debug("Starting occupancy probability calculation")
        active_triggers = []
        sensor_probs = {}

        try:
            # Calculate base probability
            calculated_probability = self._calculate_base_probability(
                active_sensor_states, active_triggers, sensor_probs
            )

            # Calculate prior probability
            prior_probability = self._calculate_prior_probability(active_sensor_states)

            # Apply decay to the calculated probability
            probability_state = ProbabilityState(
                probability=calculated_probability,
                previous_probability=self.previous_probability,
                threshold=self.probability_state["threshold"],
                prior_probability=prior_probability,
                active_triggers=active_triggers,
                sensor_probabilities=sensor_probs,
                decay_status=0.0,
                device_states={},
                sensor_availability={},
            )
            decayed_probability, decay_status = self.decay_handler.calculate_decay(
                probability_state
            )

            # Update previous probability for next calculation
            self.previous_probability = decayed_probability

            # Calculate final probability
            final_probability = max(
                MIN_PROBABILITY, min(decayed_probability, MAX_PROBABILITY)
            )

            _LOGGER.debug(
                "Final probability calculation: base=%.3f, decayed=%.3f, final=%.3f, decay_status=%.3f",
                calculated_probability,
                decayed_probability,
                final_probability,
                decay_status,
            )

            return ProbabilityState(
                probability=final_probability,
                prior_probability=prior_probability,
                active_triggers=active_triggers,
                sensor_probabilities=sensor_probs,
                decay_status=decay_status,
                device_states={
                    entity_id: {"state": state.get("state")}
                    for entity_id, state in active_sensor_states.items()
                },
                sensor_availability={
                    entity_id: state.get("availability", False)
                    for entity_id, state in active_sensor_states.items()
                },
                is_occupied=final_probability >= self.probability_state["threshold"],
            )

        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error in probability calculation: %s", err, exc_info=True)
            return ProbabilityState(
                probability=MIN_PROBABILITY,
                prior_probability=MIN_PROBABILITY,
                active_triggers=[],
                sensor_probabilities={},
                decay_status=0.0,
                device_states={},
                sensor_availability={},
                is_occupied=False,
            )


def bayesian_update(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
) -> float:
    """Perform a Bayesian update using Bayes' theorem.

    Args:
        prior: The prior probability P(H)
        prob_given_true: The likelihood P(E|H)
        prob_given_false: The likelihood P(E|¬H)

    Returns:
        The posterior probability P(H|E)

    Raises:
        ValueError: If any input probability is invalid
        ZeroDivisionError: If the denominator would be zero
    """
    # Validate input probabilities
    if not all(0 <= p <= 1 for p in (prior, prob_given_true, prob_given_false)):
        raise ValueError("All probabilities must be between 0 and 1")

    # Clamp input probabilities
    prior = max(MIN_PROBABILITY, min(prior, MAX_PROBABILITY))
    prob_given_true = max(MIN_PROBABILITY, min(prob_given_true, MAX_PROBABILITY))
    prob_given_false = max(MIN_PROBABILITY, min(prob_given_false, MAX_PROBABILITY))

    # Calculate using Bayes' theorem: P(H|E) = P(E|H)P(H) / P(E)
    # where P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)
    numerator = prob_given_true * prior
    denominator = numerator + prob_given_false * (1 - prior)

    if denominator == 0:
        raise ZeroDivisionError("Denominator would be zero in Bayesian update")

    # Calculate and clamp result
    result = numerator / denominator
    return max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))
