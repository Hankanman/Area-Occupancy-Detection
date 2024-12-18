"""Probability calculations and historical analysis for Area Occupancy Detection."""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Any, Optional

from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_PLAYING,
    STATE_PAUSED,
    STATE_OPEN,
    STATE_CLOSED,
)
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError

from .types import (
    ProbabilityResult,
)
from .probabilities import (
    MOTION_PROB_GIVEN_TRUE,
    MOTION_PROB_GIVEN_FALSE,
    MEDIA_PROB_GIVEN_TRUE,
    MEDIA_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
    APPLIANCE_PROB_GIVEN_FALSE,
    DOOR_PROB_GIVEN_TRUE,
    DOOR_PROB_GIVEN_FALSE,
    WINDOW_PROB_GIVEN_TRUE,
    WINDOW_PROB_GIVEN_FALSE,
    LIGHT_PROB_GIVEN_TRUE,
    LIGHT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    DEFAULT_PROB_GIVEN_FALSE,
    DECAY_LAMBDA,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    DEFAULT_PRIOR,
    MOTION_DEFAULT_PRIOR,
    MEDIA_DEFAULT_PRIOR,
    APPLIANCE_DEFAULT_PRIOR,
    DOOR_DEFAULT_PRIOR,
    WINDOW_DEFAULT_PRIOR,
    LIGHT_DEFAULT_PRIOR,
    ENVIRONMENTAL_PROB_GIVEN_TRUE,
    ENVIRONMENTAL_PROB_GIVEN_FALSE,
    ENVIRONMENTAL_DEFAULT_PRIOR,
    SENSOR_WEIGHTS,
)
from .const import (
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_APPLIANCES,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_DOOR_SENSORS,
    CONF_WINDOW_SENSORS,
    CONF_LIGHTS,
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

    def __init__(
        self,
        coordinator,
    ) -> None:
        _LOGGER.debug("Initializing ProbabilityCalculator")
        self.coordinator = coordinator
        self.config = self.coordinator.config

        self.motion_sensors = self.config.get(CONF_MOTION_SENSORS, [])
        self.media_devices = self.config.get(CONF_MEDIA_DEVICES, [])
        self.appliances = self.config.get(CONF_APPLIANCES, [])
        self.illuminance_sensors = self.config.get(CONF_ILLUMINANCE_SENSORS, [])
        self.humidity_sensors = self.config.get(CONF_HUMIDITY_SENSORS, [])
        self.temperature_sensors = self.config.get(CONF_TEMPERATURE_SENSORS, [])
        self.door_sensors = self.config.get(CONF_DOOR_SENSORS, [])
        self.window_sensors = self.config.get(CONF_WINDOW_SENSORS, [])
        self.lights = self.config.get(CONF_LIGHTS, [])

        self.current_probability = MIN_PROBABILITY
        self.previous_probability = MIN_PROBABILITY
        self.decay_enabled = self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        self.decay_start_time: Optional[datetime] = None

    def calculate(
        self,
        sensor_states: dict[str, Any],
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
        sensor_states: dict[str, Any],
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
        sensor_states: dict,
        active_triggers: list,
        sensor_probs: dict,
        now: datetime,
    ) -> float:
        """Calculate the base probability from sensor states."""
        calculated_probability = MIN_PROBABILITY

        for entity_id, state in sensor_states.items():
            _LOGGER.debug("Processing sensor: %s with state: %s", entity_id, state)
            weighted_prob, is_active, prob_details = self._calculate_sensor_probability(
                entity_id, state, now
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

    def _calculate_sensor_probability(
        self, entity_id: str, state: dict[str, Any], now: datetime
    ) -> tuple[float, bool, dict[str, float]]:
        """Calculate probability contribution from a single sensor."""
        if not state or not state.get("availability", False):
            return 0.0, False, {}

        # Get the weight for this sensor type using the helper function
        sensor_weight = self._get_sensor_weight(entity_id)

        # Retrieve learned priors with age consideration
        learned_data = self.coordinator.learned_priors.get(entity_id, {})
        p_true = learned_data.get("prob_given_true")
        p_false = learned_data.get("prob_given_false")
        learned_prior = learned_data.get("prior")

        # Check if we have valid learned priors
        using_learned_priors = p_true is not None and p_false is not None

        if not using_learned_priors:
            # Use default priors if learned priors are not available
            p_true, p_false = self._get_sensor_priors(entity_id)
            prior_val = self._get_default_prior(entity_id)
        else:
            prior_val = (
                learned_prior
                if learned_prior is not None
                else self._get_default_prior(entity_id)
            )

        is_active = is_entity_active(entity_id, state["state"], self)
        if is_active:
            # Calculate this sensor's contribution
            unweighted_prob = update_probability(prior_val, p_true, p_false)
            # Apply weight to the sensor's contribution
            weighted_prob = unweighted_prob * sensor_weight
            prob_details = {
                "probability": unweighted_prob,
                "weight": sensor_weight,
                "weighted_probability": weighted_prob,
            }
            return weighted_prob, True, prob_details

        return 0.0, False, {}

    def _get_sensor_weight(self, entity_id: str) -> float:
        """Get the weight for a sensor based on its type."""
        if entity_id in self.motion_sensors:
            return SENSOR_WEIGHTS["motion"]
        elif entity_id in self.media_devices:
            return SENSOR_WEIGHTS["media"]
        elif entity_id in self.appliances:
            return SENSOR_WEIGHTS["appliance"]
        elif entity_id in self.door_sensors:
            return SENSOR_WEIGHTS["door"]
        elif entity_id in self.window_sensors:
            return SENSOR_WEIGHTS["window"]
        elif entity_id in self.lights:
            return SENSOR_WEIGHTS["light"]
        elif entity_id in (
            self.illuminance_sensors + self.humidity_sensors + self.temperature_sensors
        ):
            return SENSOR_WEIGHTS["environmental"]
        return 1.0

    def _get_sensor_priors(self, entity_id: str) -> tuple[float, float]:
        """Return default priors for a sensor based on its category."""
        if entity_id in self.motion_sensors:
            return MOTION_PROB_GIVEN_TRUE, MOTION_PROB_GIVEN_FALSE
        elif entity_id in self.media_devices:
            return MEDIA_PROB_GIVEN_TRUE, MEDIA_PROB_GIVEN_FALSE
        elif entity_id in self.appliances:
            return APPLIANCE_PROB_GIVEN_TRUE, APPLIANCE_PROB_GIVEN_FALSE
        elif entity_id in self.door_sensors:
            return DOOR_PROB_GIVEN_TRUE, DOOR_PROB_GIVEN_FALSE
        elif entity_id in self.window_sensors:
            return WINDOW_PROB_GIVEN_TRUE, WINDOW_PROB_GIVEN_FALSE
        elif entity_id in self.lights:
            return LIGHT_PROB_GIVEN_TRUE, LIGHT_PROB_GIVEN_FALSE
        elif (
            entity_id in self.illuminance_sensors
            or entity_id in self.humidity_sensors
            or entity_id in self.temperature_sensors
        ):
            return ENVIRONMENTAL_PROB_GIVEN_TRUE, ENVIRONMENTAL_PROB_GIVEN_FALSE
        return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE

    def _get_default_prior(self, entity_id: str) -> float:
        """Return default prior based on entity category."""
        if entity_id in self.motion_sensors:
            return MOTION_DEFAULT_PRIOR
        elif entity_id in self.media_devices:
            return MEDIA_DEFAULT_PRIOR
        elif entity_id in self.appliances:
            return APPLIANCE_DEFAULT_PRIOR
        elif entity_id in self.door_sensors:
            return DOOR_DEFAULT_PRIOR
        elif entity_id in self.window_sensors:
            return WINDOW_DEFAULT_PRIOR
        elif entity_id in self.lights:
            return LIGHT_DEFAULT_PRIOR
        elif (
            entity_id in self.illuminance_sensors
            or entity_id in self.humidity_sensors
            or entity_id in self.temperature_sensors
        ):
            return ENVIRONMENTAL_DEFAULT_PRIOR
        return DEFAULT_PRIOR

    def _calculate_decay(
        self,
        current_probability: float,
        previous_probability: float,
        threshold: float,
        now: datetime,
    ) -> tuple[float, float]:
        """Calculate decay factor and apply it to probability."""
        elapsed = (now - self.decay_start_time).total_seconds()
        _LOGGER.debug(
            "Calculating decay. Elapsed time since decay start: %s seconds", elapsed
        )

        # Subtract minimum delay from elapsed time since we only want to decay
        # after the minimum delay period
        min_delay = self.config.get(CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY)
        actual_decay_time = elapsed - min_delay

        decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)
        _LOGGER.debug("Decay window: %s seconds", decay_window)

        # Apply exponential decay based on actual decay time
        decay_factor = math.exp(-DECAY_LAMBDA * (actual_decay_time / decay_window))
        _LOGGER.debug("Decay factor: %s", decay_factor)

        # Decay from previous probability
        decayed_probability = previous_probability * decay_factor
        _LOGGER.debug("After decay: decayed_probability=%s", decayed_probability)

        # Clamp the result
        decayed_probability = max(
            MIN_PROBABILITY, min(decayed_probability, MAX_PROBABILITY)
        )

        # If decayed below threshold, reset to minimum
        if decayed_probability < threshold:
            _LOGGER.debug("Decayed below threshold - resetting to minimum")
            self.decay_start_time = None
            return current_probability, decay_factor

        return decayed_probability, decay_factor


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


def is_entity_active(entity_id: str, state: str, calc: ProbabilityCalculator) -> bool:
    if entity_id in calc.motion_sensors:
        return state == STATE_ON
    elif entity_id in calc.media_devices:
        return state in (STATE_PLAYING, STATE_PAUSED)
    elif entity_id in calc.appliances:
        return state == STATE_ON
    elif entity_id in calc.door_sensors:
        return state in (STATE_OFF, STATE_CLOSED)
    elif entity_id in calc.window_sensors:
        return state in (STATE_ON, STATE_OPEN)
    elif entity_id in calc.lights:
        return state == STATE_ON
    return False
