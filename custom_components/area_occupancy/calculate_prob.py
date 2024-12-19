"""Probability calculations and historical analysis for Area Occupancy Detection."""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Optional

from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError

from .types import (
    ProbabilityResult,
    SensorState,
    CalculationResult,
)
from .const import (
    DEFAULT_PRIOR,
    DECAY_LAMBDA,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_MIN_DELAY,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_MIN_DELAY,
)

_LOGGER = logging.getLogger(__name__)


class ProbabilityCalculator:
    """Handles probability calculations and historical analysis."""

    def __init__(self, coordinator, probabilities) -> None:
        """Initialize the calculator."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self.probabilities = probabilities

        self.current_probability = MIN_PROBABILITY
        self.previous_probability = MIN_PROBABILITY
        self.decay_enabled = self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        self.decay_start_time: Optional[datetime] = None

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

        # Use the helper function instead of class method
        if not self.probabilities.is_entity_active(
            entity_id,
            state["state"],
        ):
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

    def _calculate_decay(
        self,
        current_probability: float,
        previous_probability: float,
        threshold: float,
        now: datetime,
    ) -> tuple[float, float]:
        """Calculate decay factor and apply it to probability."""
        if not self.decay_start_time:
            return current_probability, 1.0

        elapsed = (now - self.decay_start_time).total_seconds()
        min_delay = self.config.get(CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY)
        actual_decay_time = max(0, elapsed - min_delay)

        if actual_decay_time <= 0:
            return previous_probability, 1.0

        decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)
        decay_factor = math.exp(-DECAY_LAMBDA * (actual_decay_time / decay_window))

        decayed_probability = max(
            MIN_PROBABILITY, min(previous_probability * decay_factor, MAX_PROBABILITY)
        )

        if decayed_probability < threshold:
            self.decay_start_time = None
            return current_probability, decay_factor

        return decayed_probability, decay_factor

    def calculate(
        self,
        sensor_states: dict[str, SensorState],
    ) -> ProbabilityResult:
        """Calculate occupancy probability."""
        _LOGGER.debug("Initiating occupancy probability calculation.")
        try:
            # Update previous probability from coordinator data
            if self.coordinator.data and "probability" in self.coordinator.data:
                self.previous_probability = self.coordinator.data["probability"]
                _LOGGER.debug(
                    "Coordinator data probability: %s", self.previous_probability
                )
            else:
                self.previous_probability = DEFAULT_PRIOR
                _LOGGER.debug(
                    "No coordinator data or probability. Using default prior: %s",
                    self.previous_probability,
                )

            now = dt_util.utcnow()
            _LOGGER.debug("Current time: %s", now)

            # Use the shared calculation logic
            result = self._perform_calculation_logic(sensor_states, now)

            return result

        except (HomeAssistantError, ValueError, AttributeError, KeyError) as err:
            _LOGGER.error("Error in probability calculation: %s", err)
            raise HomeAssistantError(
                "Failed to calculate occupancy probability"
            ) from err

    def _perform_calculation_logic(
        self,
        sensor_states: dict[str, SensorState],
        now: datetime,
    ) -> ProbabilityResult:
        """Core calculation logic."""
        _LOGGER.debug("Starting occupancy probability calculation.")
        active_triggers = []
        sensor_probs = {}
        threshold = self.coordinator.get_threshold_decimal()

        # Store the previous probability for decay logic
        self.previous_probability = self.current_probability
        _LOGGER.debug("Previous probability: %s", self.previous_probability)

        # Calculate new base probability from sensors
        calculated_probability = self._calculate_base_probability(
            sensor_states, active_triggers, sensor_probs, now
        )
        _LOGGER.debug("New calculated probability: %s", calculated_probability)

        # Determine if we need decay
        if calculated_probability >= threshold:
            # Above threshold - reset decay and use new probability
            _LOGGER.debug("Above threshold - resetting decay")
            self.decay_start_time = None
            self.current_probability = calculated_probability
            decay_status = {"global_decay": 0.0}
        else:
            # Below threshold - check if we need to start/continue decay
            if self.previous_probability >= threshold:
                # Just dropped below threshold - start decay process
                if self.decay_start_time is None:
                    _LOGGER.debug("Starting decay process")
                    self.decay_start_time = now
                    self.current_probability = self.previous_probability
                    decay_status = {"global_decay": 0.0}
                else:
                    # Check if minimum delay has passed
                    elapsed = (now - self.decay_start_time).total_seconds()
                    min_delay = self.config.get(
                        CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY
                    )

                    if elapsed <= min_delay and self.decay_enabled:
                        # Still in minimum delay period - maintain previous probability
                        _LOGGER.debug("In minimum delay period")
                        self.current_probability = self.previous_probability
                        decay_status = {"global_decay": 0.0}
                    elif self.decay_enabled:
                        # Apply decay to previous probability
                        _LOGGER.debug("Applying decay to previous probability")
                        self.current_probability, decay_factor = self._calculate_decay(
                            calculated_probability,
                            self.previous_probability,
                            threshold,
                            now,
                        )
                        decay_status = {"global_decay": round(1.0 - decay_factor, 4)}
                    else:
                        # Decay is disabled - use new calculated probability
                        _LOGGER.debug(
                            "Decay is disabled - using calculated probability"
                        )
                        self.current_probability = calculated_probability
                        decay_status = {"global_decay": 0.0}
            else:
                # Already below threshold - use new calculated probability
                _LOGGER.debug("Already below threshold - using calculated probability")
                self.current_probability = calculated_probability
                decay_status = {"global_decay": 0.0}

        _LOGGER.debug("Final probability: %s", self.current_probability)

        # Create result dictionary
        result = {
            "probability": self.current_probability,
            "prior_probability": 0.0,
            "active_triggers": active_triggers,
            "sensor_probabilities": sensor_probs,
            "device_states": {},
            "decay_status": decay_status,
            "sensor_availability": {
                k: v.get("availability", False) for k, v in sensor_states.items()
            },
            "is_occupied": self.current_probability >= threshold,
        }
        return result

    def _calculate_base_probability(
        self,
        sensor_states: dict[str, SensorState],
        active_triggers: list,
        sensor_probs: dict,
        now: datetime,
    ) -> float:
        """Calculate the base probability from sensor states."""
        calculated_probability = MIN_PROBABILITY

        for entity_id, state in sensor_states.items():
            _LOGGER.debug("Processing sensor: %s with state: %s", entity_id, state)
            weighted_prob, is_active, prob_details = self._calculate_sensor_probability(
                entity_id, state
            )
            if is_active:
                active_triggers.append(entity_id)
                sensor_probs[entity_id] = prob_details
                _LOGGER.debug(
                    "Sensor %s is active. Weighted probability: %s",
                    entity_id,
                    weighted_prob,
                )
                # Stack probabilities using complementary probability
                calculated_probability = 1.0 - (
                    (1.0 - calculated_probability) * (1.0 - weighted_prob)
                )
                calculated_probability = min(calculated_probability, MAX_PROBABILITY)
                _LOGGER.debug(
                    "Updated calculated probability: %s", calculated_probability
                )

        return calculated_probability


def update_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
) -> float:
    """Perform a Bayesian update."""
    _LOGGER.debug(
        "Updating probability with prior: %s, prob_given_true: %s, prob_given_false: %s",
        prior,
        prob_given_true,
        prob_given_false,
    )
    # Clamp input probabilities
    prior = max(MIN_PROBABILITY, min(prior, MAX_PROBABILITY))
    prob_given_true = max(MIN_PROBABILITY, min(prob_given_true, MAX_PROBABILITY))
    prob_given_false = max(MIN_PROBABILITY, min(prob_given_false, MAX_PROBABILITY))

    numerator = prob_given_true * prior
    denominator = numerator + prob_given_false * (1 - prior)
    if denominator == 0:
        return prior

    # Calculate the updated probability
    result = numerator / denominator
    # Clamp result
    return max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))
