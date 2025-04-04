"""Probability calculations for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime
import logging

from .const import MAX_PROBABILITY, MIN_PROBABILITY
from .types import ProbabilityState, SensorCalculation, SensorProbability, SensorState

_LOGGER = logging.getLogger(__name__)


class ProbabilityCalculator:
    """Handle probability calculations using Bayesian inference."""

    def __init__(
        self,
        decay_handler,
        probability_state,
        prior_state,
        probabilities,
    ) -> None:
        """Initialize the calculator.

        Args:
            decay_handler: The decay handler instance
            probability_state: The current probability state
            previous_probability: The previous probability value
            prior_state: The prior state information
            probabilities: The probabilities configuration instance

        """
        self.probabilities = probabilities
        self.decay_handler = decay_handler
        self.probability_state = probability_state
        self.previous_probability = probability_state.previous_probability
        self.prior_state = prior_state

    def calculate_occupancy_probability(
        self,
        active_sensor_states: dict[str, SensorState],
        now: datetime,
    ) -> ProbabilityState:
        """Calculate the current occupancy probability using Bayesian inference.

        This method orchestrates the entire probability calculation process:
        1. Calculates base probability from sensor states
        2. Calculates prior probability
        3. Applies decay to the probability
        4. Updates the final state

        Args:
            active_sensor_states: Dictionary of active sensor states
            now: Current timestamp

        Returns:
            Updated probability state

        """
        _LOGGER.debug("Starting probability calculation")
        sensor_probs: dict[str, SensorProbability] = {}

        try:
            # Calculate base probability
            calculated_probability = self._calculate_complementary_probability(
                active_sensor_states, sensor_probs
            )

            # Calculate prior probability
            prior_probability = self._calculate_prior_probability(active_sensor_states)

            # Apply decay to the calculated probability
            self.probability_state.update(
                probability=calculated_probability,
                prior_probability=prior_probability,
            )
            decayed_probability, decay_status = self.decay_handler.calculate_decay(
                self.probability_state
            )

            # Update previous probability for next calculation
            self.previous_probability = decayed_probability

            # Calculate final probability
            final_probability = max(
                MIN_PROBABILITY, min(decayed_probability, MAX_PROBABILITY)
            )

            _LOGGER.debug(
                "Final: base=%.3f final=%.3f decay=%.3f",
                decayed_probability,
                final_probability,
                decay_status,
            )

            # Update the state with all calculated values
            self._update_probability_state(
                final_probability=final_probability,
                previous_probability=self.previous_probability,
                prior_probability=prior_probability,
                decay_status=decay_status,
                sensor_probs=sensor_probs,
                active_sensor_states=active_sensor_states,
            )

        except (ValueError, ZeroDivisionError):
            _LOGGER.exception("Error in probability calculation: %s")
            # Reset the state to minimum values
            self._update_probability_state(
                final_probability=MIN_PROBABILITY,
                previous_probability=self.previous_probability,
                prior_probability=MIN_PROBABILITY,
                decay_status=0.0,
                sensor_probs={},
                active_sensor_states={},
            )
            return self.probability_state
        else:
            return self.probability_state

    def _calculate_complementary_probability(
        self,
        sensor_states: dict[str, SensorState],
        sensor_probs: dict[str, SensorProbability],
    ) -> float:
        """Calculate the complementary probability for multiple sensors.

        This method uses the complementary probability approach to combine multiple sensor inputs.
        For each sensor, we calculate the probability of the area NOT being occupied given that sensor's state,
        then multiply these probabilities together. The final probability is 1 minus this product.

        Args:
            sensor_states: Dictionary of sensor states
            sensor_probs: Dictionary to store sensor probability details

        Returns:
            The calculated complementary probability

        """
        complementary_prob = 1.0

        for entity_id, state in sensor_states.items():
            calc_result = self._calculate_sensor_probability(entity_id, state)
            if calc_result.is_active:
                sensor_probs[entity_id] = calc_result.details

                # Convert to complementary probability and apply weight
                sensor_complementary = 1.0 - calc_result.weighted_probability
                weight = calc_result.details["weight"]
                weighted_complementary = 1.0 - (weight * (1.0 - sensor_complementary))
                complementary_prob *= weighted_complementary

        calculated_probability = 1.0 - complementary_prob
        return max(MIN_PROBABILITY, min(calculated_probability, MAX_PROBABILITY))

    def _calculate_prior_probability(
        self, sensor_states: dict[str, SensorState]
    ) -> float:
        """Calculate the prior probability based on sensor states.

        Args:
            sensor_states: Dictionary of sensor states

        Returns:
            The calculated prior probability

        """
        prior_sum = 0.0
        total_sensors = 0

        for entity_id, state in sensor_states.items():
            if not state.get("availability", False):
                continue

            sensor_config = self.probabilities.get_sensor_config(entity_id)
            if not sensor_config:
                continue

            _, _, prior = self._get_sensor_priors(entity_id, sensor_config)
            prior_sum += prior
            total_sensors += 1

        if total_sensors == 0:
            return MIN_PROBABILITY

        return max(MIN_PROBABILITY, min(prior_sum / total_sensors, MAX_PROBABILITY))

    def _get_sensor_priors(
        self, entity_id: str, sensor_config: dict[str, float]
    ) -> tuple[float, float, float]:
        """Get the priors for a sensor, using learned values if available.

        Args:
            entity_id: The entity ID of the sensor
            sensor_config: The sensor configuration dictionary

        Returns:
            Tuple of (p_true, p_false, prior) probabilities

        """
        # Check for entity-specific learned priors
        entity_prior = self.prior_state.entity_priors.get(entity_id, {})
        if entity_prior:
            return (
                entity_prior.get("prob_given_true", sensor_config["prob_given_true"]),
                entity_prior.get("prob_given_false", sensor_config["prob_given_false"]),
                entity_prior.get("prior", sensor_config["default_prior"]),
            )

        # Fall back to type priors
        sensor_type = self.probabilities.entity_types.get(entity_id)
        if sensor_type:
            type_prior = getattr(self.prior_state, f"{sensor_type}_prior", 0.0)
            if type_prior > 0:
                return (
                    sensor_config["prob_given_true"],
                    sensor_config["prob_given_false"],
                    type_prior,
                )

        # Use default configuration values
        return (
            sensor_config["prob_given_true"],
            sensor_config["prob_given_false"],
            sensor_config["default_prior"],
        )

    def _calculate_sensor_probability(
        self, entity_id: str, state: SensorState
    ) -> SensorCalculation:
        """Calculate probability contribution from a single sensor using Bayesian inference.

        Args:
            entity_id: The entity ID of the sensor
            state: The current state of the sensor

        Returns:
            SensorCalculation containing the weighted probability and details

        """
        if not state.get("availability", False):
            _LOGGER.debug("Sensor %s unavailable", entity_id)
            return SensorCalculation.empty()

        sensor_config = self.probabilities.get_sensor_config(entity_id)
        if not sensor_config:
            _LOGGER.warning("No configuration found for sensor %s", entity_id)
            return SensorCalculation.empty()

        if not self.probabilities.is_entity_active(entity_id, state["state"]):
            return SensorCalculation.empty()

        try:
            p_true, p_false, prior = self._get_sensor_priors(entity_id, sensor_config)
            unweighted_prob = bayesian_update(prior, p_true, p_false)
            weighted_prob = unweighted_prob * sensor_config["weight"]

            _LOGGER.debug(
                "Sensor %s: p=%.3f w=%.3f",
                entity_id,
                unweighted_prob,
                sensor_config["weight"],
            )

            return SensorCalculation(
                weighted_probability=weighted_prob,
                is_active=True,
                details={
                    "probability": unweighted_prob,
                    "weight": sensor_config["weight"],
                    "weighted_probability": weighted_prob,
                },
            )
        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error(
                "Error calculating probability for sensor %s: %s", entity_id, err
            )
            return SensorCalculation.empty()

    def _update_probability_state(
        self,
        final_probability: float,
        previous_probability: float,
        prior_probability: float,
        decay_status: float,
        sensor_probs: dict[str, SensorProbability],
        active_sensor_states: dict[str, SensorState],
    ) -> None:
        """Update the probability state with all calculated values.

        Args:
            final_probability: The final calculated probability
            previous_probability: The calculated prior probability
            prior_probability: The prior probability used in the calculation
            decay_status: The current decay status
            sensor_probs: Dictionary of sensor probability details
            active_sensor_states: Dictionary of active sensor states

        """
        self.probability_state.update(
            probability=final_probability,
            previous_probability=previous_probability,
            prior_probability=prior_probability,
            sensor_probabilities=sensor_probs,
            decay_status=decay_status,
            current_states={
                entity_id: {
                    "state": state.get("state"),
                    "availability": state.get("availability", False),
                }
                for entity_id, state in active_sensor_states.items()
            },
            is_occupied=final_probability >= self.probability_state.threshold,
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
