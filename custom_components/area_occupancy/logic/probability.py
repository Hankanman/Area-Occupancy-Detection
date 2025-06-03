"""Probability calculations for Area Occupancy Detection."""

import logging
from typing import Dict, Optional

from homeassistant.core import HomeAssistant, State

from ..const import (
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from ..models import Feature
from ..state import PriorData, SensorConfiguration
from ..state.containers import ProbabilityState

_LOGGER = logging.getLogger(__name__)


class Probability:
    """Calculate occupancy probability based on sensor states.

    This class handles the calculation of occupancy probabilities based on
    sensor states and prior probabilities. It uses Bayesian probability
    calculations to determine the likelihood of occupancy.

    Attributes:
        hass: Home Assistant instance
        config: Sensor configuration object
        primary_sensor: Primary occupancy sensor entity ID

    """

    def __init__(
        self,
        hass: HomeAssistant,
        config: SensorConfiguration,
        primary_sensor: str,
    ) -> None:
        """Initialize the calculator.

        Args:
            hass: Home Assistant instance
            config: Sensor configuration object
            primary_sensor: Primary occupancy sensor entity ID

        """
        self.hass = hass
        self.config = config
        self.primary_sensor = primary_sensor

    def calculate_probability(
        self,
        entity_id: str,
        state: State,
        prior_data: PriorData,
        current_probability: float,
    ) -> float:
        """Calculate probability of occupancy based on sensor state.

        This method calculates the probability of occupancy based on the current
        sensor state and prior probabilities. It uses Bayesian probability
        calculations to determine the likelihood of occupancy.

        Args:
            entity_id: The entity ID to calculate probability for
            state: Current state of the entity
            prior_data: Prior probability data for the entity
            current_probability: Current probability of occupancy

        Returns:
            Updated probability of occupancy

        """
        if not state:
            return current_probability

        entity_type = self.config.get_entity_type(entity_id)
        if not entity_type:
            _LOGGER.error("Invalid entity ID format provided: %s", entity_id)
            return current_probability

        # Get probability configuration for this entity type
        prob_config = self.config.get_probability_config(entity_type.value)

        # Check if the current state is considered active
        is_active = self.config.is_active_state(entity_type.value, state.state)

        # Get probability values based on state
        if is_active:
            prob_given_true = prior_data.prob_given_true or DEFAULT_PROB_GIVEN_TRUE
            prob_given_false = prior_data.prob_given_false or DEFAULT_PROB_GIVEN_FALSE
        else:
            prob_given_true = 1.0 - (
                prior_data.prob_given_true or DEFAULT_PROB_GIVEN_TRUE
            )
            prob_given_false = 1.0 - (
                prior_data.prob_given_false or DEFAULT_PROB_GIVEN_FALSE
            )

        # Calculate probability using Bayes' theorem
        numerator = prob_given_true * current_probability
        denominator = prob_given_true * current_probability + prob_given_false * (
            1.0 - current_probability
        )

        if denominator == 0:
            return current_probability

        probability = numerator / denominator

        # Apply weight from configuration
        weight = prob_config.get("weight", 1.0)
        weighted_probability = (
            current_probability * (1.0 - weight) + probability * weight
        )

        return max(MIN_PROBABILITY, min(weighted_probability, MAX_PROBABILITY))

    def calculate_combined_probability(
        self,
        sensor_probabilities: Dict[str, Feature],
        current_probability: float,
    ) -> float:
        """Calculate combined probability from all sensor probabilities.

        This method combines probabilities from multiple sensors to determine
        the overall probability of occupancy. It uses a weighted average
        approach based on sensor weights.

        Args:
            sensor_probabilities: Dictionary of sensor probabilities
            current_probability: Current probability of occupancy

        Returns:
            Combined probability of occupancy

        """
        if not sensor_probabilities:
            return current_probability

        total_weight = 0.0
        weighted_sum = 0.0

        for sensor_prob in sensor_probabilities.values():
            if not sensor_prob:
                continue

            weight = sensor_prob.get("weight", 1.0)
            probability = sensor_prob.get("probability", current_probability)

            total_weight += weight
            weighted_sum += weight * probability

        if total_weight == 0:
            return current_probability

        combined_probability = weighted_sum / total_weight
        return max(MIN_PROBABILITY, min(combined_probability, MAX_PROBABILITY))

    def calculate_sensor_probability(
        self,
        entity_id: str,
        state: State,
        prior_data: PriorData,
        current_probability: float,
    ) -> Optional[Feature]:
        """Calculate probability for a single sensor.

        This method calculates the probability contribution of a single sensor
        based on its current state and prior probabilities.

        Args:
            entity_id: The entity ID to calculate probability for
            state: Current state of the entity
            prior_data: Prior probability data for the entity
            current_probability: Current probability of occupancy

        Returns:
            Sensor probability data if successful, None otherwise

        """
        if not state:
            return None

        entity_type = self.config.get_entity_type(entity_id)
        if not entity_type:
            _LOGGER.error("Invalid entity ID format provided: %s", entity_id)
            return None

        # Get probability configuration for this entity type
        prob_config = self.config.get_probability_config(entity_type.value)

        # Calculate probability
        probability = self.calculate_probability(
            entity_id, state, prior_data, current_probability
        )

        # Get weight from configuration
        weight = prob_config.get("weight", 1.0)

        return Feature(
            type=entity_type,
            state=state.state,
            is_active=self.config.is_active_state(entity_type.value, state.state),
            probability=probability,
            weighted_probability=probability * weight,
            last_changed=state.last_changed.isoformat(),
            available=True,
        )

    def update_probability_state(
        self,
        state: ProbabilityState,
        entity_id: str,
        sensor_state: State,
        prior_data: PriorData,
    ) -> ProbabilityState:
        """Update probability state with new sensor data.

        This method updates the probability state with new sensor data,
        recalculating probabilities and updating the state accordingly.

        Args:
            state: Current probability state
            entity_id: Entity ID being updated
            sensor_state: New sensor state
            prior_data: Prior probability data for the entity

        Returns:
            Updated probability state

        """
        # Calculate new sensor probability
        sensor_prob = self.calculate_sensor_probability(
            entity_id,
            sensor_state,
            prior_data,
            state.probability,
        )

        if not sensor_prob:
            return state

        # Update sensor probabilities
        sensor_probabilities = dict(state.sensor_probabilities)
        sensor_probabilities[entity_id] = sensor_prob

        # Calculate new combined probability
        new_probability = self.calculate_combined_probability(
            sensor_probabilities, state.probability
        )

        # Update state
        return ProbabilityState(
            probability=new_probability,
            previous_probability=state.probability,
            threshold=state.threshold,
            prior_probability=state.prior_probability,
            sensor_probabilities=sensor_probabilities,
            decay_status=state.decay_status,
            current_states={
                **state.current_states,
                entity_id: {
                    "state": sensor_state.state,
                    "last_changed": sensor_state.last_changed.isoformat(),
                    "availability": True,
                },
            },
            previous_states=state.current_states,
            is_occupied=new_probability >= state.threshold,
            decaying=state.decaying,
            decay_start_time=state.decay_start_time,
            decay_start_probability=state.decay_start_probability,
        )
