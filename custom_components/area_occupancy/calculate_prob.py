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
        probability_state,
        prior_state,
        probabilities,
    ) -> None:
        """Initialize the calculator.

        Args:
            probability_state: The current probability state
            previous_probability: The previous probability value
            prior_state: The prior state information
            probabilities: The probabilities configuration instance

        """
        self.probabilities = probabilities
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
        sensor_probs: dict[str, SensorProbability] = {}

        try:
            # Store the probability state at the start of the calculation cycle
            initial_probability = self.probability_state.probability
            # No longer need initial decay state here
            # initial_is_decaying = self.probability_state.decaying
            # initial_decay_start_time = self.probability_state.decay_start_time
            # initial_decay_start_probability = (
            #     self.probability_state.decay_start_probability
            # )

            # Calculate base probability (this is now the undecayed probability)
            calculated_probability = self._calculate_complementary_probability(
                active_sensor_states, sensor_probs
            )

            # Calculate prior probability
            prior_probability = self._calculate_prior_probability(active_sensor_states)

            # --- Decay is no longer applied here ---

            # Store the previous probability *before* updating the state
            # The coordinator will handle the final previous probability based on its initial state
            # self.previous_probability = initial_probability # Removed - Coordinator manages previous prob context

            # Calculate final probability (clamp the *undecayed* value)
            # The coordinator will apply decay later
            final_undecayed_probability = max(
                MIN_PROBABILITY, min(calculated_probability, MAX_PROBABILITY)
            )
            # Update the state object with calculated values, but without decay application yet
            self._update_probability_state(
                undecayed_probability=final_undecayed_probability,
                prior_probability=prior_probability,
                sensor_probs=sensor_probs,
                active_sensor_states=active_sensor_states,
                # Pass the initial probability from the start of *this* calculation cycle
                # The coordinator will use its own initial value for decay calculation
                current_cycle_previous_probability=initial_probability,
            )

        except (ValueError, ZeroDivisionError):
            _LOGGER.exception("Error in probability calculation: %s")
            # Reset the state to minimum values
            self._update_probability_state(
                undecayed_probability=MIN_PROBABILITY,
                prior_probability=MIN_PROBABILITY,
                sensor_probs={},
                active_sensor_states={},
                # Provide a sensible previous probability on reset
                current_cycle_previous_probability=self.probability_state.probability,
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
        undecayed_probability: float,
        prior_probability: float,
        sensor_probs: dict[str, SensorProbability],
        active_sensor_states: dict[str, SensorState],
        current_cycle_previous_probability: float,
    ) -> None:
        """Update the probability state with calculated (but not yet decayed) values.

        Args:
            undecayed_probability: The calculated probability before decay application
            prior_probability: The prior probability used in the calculation
            sensor_probs: Dictionary of sensor probability details
            active_sensor_states: Dictionary of active sensor states
            current_cycle_previous_probability: The probability value from before this calculation cycle started.

        """
        # Note: is_occupied, decaying, decay_start_time, decay_start_probability, decay_status
        # will be set by the coordinator after applying decay.
        # We update the core probabilities and sensor details here.
        # The 'previous_probability' stored here reflects the state *before* this undecayed calculation.
        self.probability_state.update(
            probability=undecayed_probability,
            previous_probability=current_cycle_previous_probability,
            prior_probability=prior_probability,
            sensor_probabilities=sensor_probs,
            current_states={
                entity_id: {
                    "state": state.get("state"),
                    "availability": state.get("availability", False),
                }
                for entity_id, state in active_sensor_states.items()
            },
            # Keep existing decay state for now; coordinator will update it
            decay_status=self.probability_state.decay_status,
            is_occupied=self.probability_state.is_occupied,  # Keep existing; coordinator updates
            decaying=self.probability_state.decaying,
            decay_start_time=self.probability_state.decay_start_time,
            decay_start_probability=self.probability_state.decay_start_probability,
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
