"""Probability calculations for Area Occupancy Detection."""

from __future__ import annotations

import logging

from .const import MAX_PROBABILITY, MIN_PROBABILITY
from .probabilities import Probabilities
from .types import (
    OccupancyCalculationResult,
    PriorData,
    PriorState,
    ProbabilityConfig,
    SensorCalculation,
    SensorInfo,
    SensorProbability,
)

_LOGGER = logging.getLogger(__name__)


class ProbabilityCalculator:
    """Handle probability calculations using Bayesian inference."""

    def __init__(
        self,
        probabilities: Probabilities,
    ) -> None:
        """Initialize the calculator."""
        self.probabilities = probabilities
        # self.prior_state is no longer stored here, passed during calculation

    def calculate_occupancy_probability(
        self,
        current_states: dict[str, SensorInfo],
        prior_state: PriorState,
    ) -> OccupancyCalculationResult:
        """Calculate the current occupancy probability using Bayesian inference.

        Args:
            current_states: Dictionary of current sensor states.
            prior_state: The current prior state object containing learned/default priors.

        Returns:
            An OccupancyCalculationResult dataclass containing:
                - calculated_probability: The calculated (undecayed) probability (0.0-1.0).
                - prior_probability: The average prior based on active sensors (0.0-1.0).
                - sensor_probabilities: Dictionary mapping entity IDs to their calculation details.

        Raises:
            ValueError: If calculation encounters invalid inputs.
            ZeroDivisionError: If a division by zero occurs during calculation.

        """
        sensor_probs: dict[str, SensorProbability] = {}
        active_sensor_states = {
            entity_id: info
            for entity_id, info in current_states.items()
            if info and info.get("availability", False)  # Ensure info exists
        }

        try:
            calculated_probability = self._calculate_complementary_probability(
                active_sensor_states, sensor_probs, prior_state
            )

            prior_probability = self._get_average_prior_for_active_sensors(
                active_sensor_states, prior_state
            )

            final_calculated_probability = max(
                MIN_PROBABILITY, min(calculated_probability, MAX_PROBABILITY)
            )

            return OccupancyCalculationResult(
                calculated_probability=final_calculated_probability,
                prior_probability=prior_probability,
                sensor_probabilities=sensor_probs,
            )

        except (ValueError, ZeroDivisionError):
            _LOGGER.exception("Error in probability calculation")
            # Re-raise to be handled by the coordinator
            raise

    def _calculate_complementary_probability(
        self,
        sensor_states: dict[str, SensorInfo],
        sensor_probs: dict[str, SensorProbability],
        prior_state: PriorState,  # Pass prior_state here
    ) -> float:
        """Calculate the complementary probability for multiple sensors."""
        complementary_prob = 1.0

        for entity_id, state in sensor_states.items():
            # Pass prior_state to the sensor calculation
            calc_result = self._calculate_sensor_probability(
                entity_id, state, prior_state
            )
            if calc_result.is_active:
                sensor_probs[entity_id] = calc_result.details

                # Convert to complementary probability and apply weight
                sensor_complementary = 1.0 - calc_result.weighted_probability
                # Complementary probability should already be weighted
                # weight = calc_result.details["weight"]
                # weighted_complementary = 1.0 - (weight * (1.0 - sensor_complementary))
                # complementary_prob *= weighted_complementary
                complementary_prob *= (
                    sensor_complementary  # Use the weighted complementary prob directly
                )

        calculated_probability = 1.0 - complementary_prob
        return max(MIN_PROBABILITY, min(calculated_probability, MAX_PROBABILITY))

    def _get_average_prior_for_active_sensors(
        self, sensor_states: dict[str, SensorInfo], prior_state: PriorState
    ) -> float:
        """Calculate the average default/learned prior based on currently active sensor states."""
        prior_sum = 0.0
        total_sensors = 0

        for entity_id, state in sensor_states.items():
            if not state or not state.get("availability", False):  # Ensure state exists
                continue

            sensor_config = self.probabilities.get_sensor_config(entity_id)
            if not sensor_config:
                continue

            # Pass prior_state to get sensor priors
            _, _, prior = self._get_sensor_priors(entity_id, sensor_config, prior_state)
            prior_sum += prior
            total_sensors += 1

        if total_sensors == 0:
            return MIN_PROBABILITY

        return max(MIN_PROBABILITY, min(prior_sum / total_sensors, MAX_PROBABILITY))

    def _get_sensor_priors(
        self,
        entity_id: str,
        sensor_config: ProbabilityConfig,
        prior_state: PriorState,
    ) -> tuple[float, float, float]:
        """Get the priors for a sensor, using learned values if available."""
        # Default values from config if no learned data is found
        default_prob_true = float(sensor_config["prob_given_true"])
        default_prob_false = float(sensor_config["prob_given_false"])
        default_prior = float(sensor_config["default_prior"])

        # Check for entity-specific learned priors
        entity_prior_data: PriorData | None = prior_state.entity_priors.get(entity_id)
        if entity_prior_data:
            # Use learned entity priors if available and valid. Fall back to defaults if attributes are None.
            prob_true = (
                entity_prior_data.prob_given_true
                if entity_prior_data.prob_given_true is not None
                else default_prob_true
            )
            prob_false = (
                entity_prior_data.prob_given_false
                if entity_prior_data.prob_given_false is not None
                else default_prob_false
            )
            prior = entity_prior_data.prior  # prior always has a value in PriorData
            return float(prob_true), float(prob_false), float(prior)

        # Fall back to learned type priors if no entity-specific ones found
        sensor_type = self.probabilities.entity_types.get(entity_id)
        if sensor_type:
            type_prior_data: PriorData | None = prior_state.type_priors.get(
                sensor_type.value
            )  # Use enum value as key
            if type_prior_data:
                # Use learned type prior if available, falling back to defaults from config if attributes are None.
                prob_true = (
                    type_prior_data.prob_given_true
                    if type_prior_data.prob_given_true is not None
                    else default_prob_true
                )
                prob_false = (
                    type_prior_data.prob_given_false
                    if type_prior_data.prob_given_false is not None
                    else default_prob_false
                )
                prior = type_prior_data.prior  # prior always has a value in PriorData
                return float(prob_true), float(prob_false), float(prior)

        # Use default configuration values if no learned priors found
        return (
            default_prob_true,
            default_prob_false,
            default_prior,
        )

    def _calculate_sensor_probability(
        self, entity_id: str, state: SensorInfo, prior_state: PriorState
    ) -> SensorCalculation:
        """Calculate probability contribution from a single sensor using Bayesian inference."""
        if not state or not state.get("availability", False):  # Ensure state exists
            _LOGGER.debug("Sensor %s unavailable", entity_id)
            return SensorCalculation.empty()

        sensor_config = self.probabilities.get_sensor_config(entity_id)
        if not sensor_config:
            _LOGGER.warning("No configuration found for sensor %s", entity_id)
            return SensorCalculation.empty()

        # Check if entity state is considered active based on its type configuration
        if not self.probabilities.is_entity_active(entity_id, state.get("state")):
            return SensorCalculation.empty()

        try:
            # Pass prior_state to get sensor priors
            p_true, p_false, prior = self._get_sensor_priors(
                entity_id, sensor_config, prior_state
            )
            unweighted_prob = bayesian_update(prior, p_true, p_false)
            # Apply weight directly here
            weight = float(sensor_config["weight"])  # Cache weight
            weighted_prob = unweighted_prob * weight
            # Clamp weighted probability
            weighted_prob = max(MIN_PROBABILITY, min(weighted_prob, MAX_PROBABILITY))

            # Use cast to satisfy SensorProbability type hint
            sensor_details = SensorProbability(
                probability=unweighted_prob,
                weight=weight,  # Use cached weight
                weighted_probability=weighted_prob,  # Store final weighted prob
            )
            _LOGGER.debug(
                "Sensor %s active: p=%.3f (w=%.3f) -> wp=%.3f",
                entity_id,
                unweighted_prob,
                weight,  # Use cached weight
                weighted_prob,  # Log the final weighted prob
            )

            return SensorCalculation(
                weighted_probability=weighted_prob,  # Return the final weighted probability
                is_active=True,
                details=sensor_details,
            )
        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error(
                "Error calculating probability for sensor %s: %s", entity_id, err
            )
            # Re-raise to be handled by the coordinator
            raise


def bayesian_update(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
) -> float:
    """Perform a Bayesian update using Bayes' theorem."""
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
        # Instead of raising, return a boundary value? Let's return MIN_PROBABILITY
        # to avoid stopping the entire process for one sensor.
        _LOGGER.warning(
            "Denominator is zero in Bayesian update, returning MIN_PROBABILITY"
        )
        return MIN_PROBABILITY
        # raise ZeroDivisionError("Denominator would be zero in Bayesian update")

    # Calculate and clamp result
    result = numerator / denominator
    return max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))
