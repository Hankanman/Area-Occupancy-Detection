"""Probability calculations for Area Occupancy Detection."""

from __future__ import annotations

import math
import logging
from datetime import datetime

from homeassistant.const import (
    STATE_ON,
    STATE_PLAYING,
    STATE_PAUSED,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .types import (
    ProbabilityResult,
    SensorStates,
    DecayConfig,
    Timeslot,
)
from .probabilities import (
    # Core weights
    MAX_MOTION_WEIGHT,
    MAX_MEDIA_WEIGHT,
    MAX_APPLIANCE_WEIGHT,
    MAX_ENVIRONMENTAL_WEIGHT,
    DEFAULT_MOTION_WEIGHT,
    DEFAULT_MEDIA_WEIGHT,
    DEFAULT_APPLIANCE_WEIGHT,
    DEFAULT_ENVIRONMENTAL_WEIGHT,
    CONFIDENCE_WEIGHTS,
    # Environmental
    ENVIRONMENTAL_SETTINGS,
    # Thresholds
    MIN_CONFIDENCE,
    MIN_PROBABILITY,
    MAX_PROBABILITY,
)

_LOGGER = logging.getLogger(__name__)


def update_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    weight: float = 1.0,
) -> float:
    """Update probability using Bayes' rule with optional weight factor."""
    # Apply weight to the probabilities while maintaining a minimum uncertainty
    effective_prob_true = (prob_given_true * weight) + ((1 - weight) * 0.5)
    effective_prob_false = (prob_given_false * weight) + ((1 - weight) * 0.5)

    # Calculate Bayesian update
    numerator = effective_prob_true * prior
    denominator = numerator + effective_prob_false * (1 - prior)

    if denominator == 0:
        return prior

    return numerator / denominator


def get_timeslot_probabilities(
    entity_id: str,
    timeslot: Timeslot | None,
) -> tuple[float, float]:
    """Get probabilities from timeslot data for entity."""
    if not timeslot or "entities" not in timeslot:
        return 0.5, 0.5

    # Find entity data in timeslot
    entity_data = next(
        (e for e in timeslot["entities"] if e["id"] == entity_id),
        None,
    )

    if entity_data:
        return entity_data["prob_given_true"], entity_data["prob_given_false"]

    return 0.5, 0.5


def calculate_decay_factor(time_diff: float, window: float, decay_type: str) -> float:
    """Calculate decay factor based on time difference."""
    if time_diff >= window:
        return MIN_PROBABILITY

    ratio = time_diff / window
    return 1.0 - ratio if decay_type == "linear" else math.exp(-3.0 * ratio)


class ProbabilityCalculator:
    """Handles probability calculations for area occupancy."""

    def __init__(
        self,
        motion_sensors: list[str],
        media_devices: list[str] | None = None,
        appliances: list[str] | None = None,
        illuminance_sensors: list[str] | None = None,
        humidity_sensors: list[str] | None = None,
        temperature_sensors: list[str] | None = None,
        decay_config: DecayConfig | None = None,
    ) -> None:
        """Initialize the calculator with configuration."""
        self.motion_sensors = motion_sensors
        self.media_devices = media_devices or []
        self.appliances = appliances or []
        self.illuminance_sensors = illuminance_sensors or []
        self.humidity_sensors = humidity_sensors or []
        self.temperature_sensors = temperature_sensors or []
        self.decay_config = decay_config or DecayConfig()

        # Initialize and validate weights
        try:
            self._initialize_weights()
        except (KeyError, TypeError, ValueError) as err:
            _LOGGER.error("Error initializing weights: %s", err)
            self._set_default_weights()

    def _initialize_weights(self) -> None:
        """Initialize and validate weights from configuration."""
        weights = {}

        # Validate and cap weights
        self.motion_weight = min(
            weights.get("motion", DEFAULT_MOTION_WEIGHT), MAX_MOTION_WEIGHT
        )
        self.media_weight = min(
            weights.get("media", DEFAULT_MEDIA_WEIGHT), MAX_MEDIA_WEIGHT
        )
        self.appliance_weight = min(
            weights.get("appliances", DEFAULT_APPLIANCE_WEIGHT), MAX_APPLIANCE_WEIGHT
        )
        self.environmental_weight = min(
            weights.get("environmental", DEFAULT_ENVIRONMENTAL_WEIGHT),
            MAX_ENVIRONMENTAL_WEIGHT,
        )

    def _set_default_weights(self) -> None:
        """Set default weights if configuration is invalid."""
        self.motion_weight = DEFAULT_MOTION_WEIGHT
        self.media_weight = DEFAULT_MEDIA_WEIGHT
        self.appliance_weight = DEFAULT_APPLIANCE_WEIGHT
        self.environmental_weight = DEFAULT_ENVIRONMENTAL_WEIGHT

    async def calculate(
        self,
        sensor_states: SensorStates,
        motion_timestamps: dict[str, datetime],
        timeslot: Timeslot | None = None,
    ) -> ProbabilityResult:
        """Calculate overall probability using Bayesian inference."""
        try:
            # Start with a neutral prior
            current_probability = 0.5
            active_triggers = []
            decay_status = {}
            device_states = {
                "media_states": {},
                "appliance_states": {},
            }
            sensor_probs = {}

            # Process motion sensors
            for entity_id in self.motion_sensors:
                state = sensor_states.get(entity_id)
                if not state or not state.get("availability", False):
                    continue

                # Get historical probabilities from timeslot
                p_true, p_false = get_timeslot_probabilities(entity_id, timeslot)
                is_active = False

                if state["state"] == STATE_ON:
                    is_active = True
                    active_triggers.append(entity_id)
                elif self.decay_config.enabled and entity_id in motion_timestamps:
                    # Apply decay if enabled
                    time_diff = (
                        dt_util.utcnow() - motion_timestamps[entity_id]
                    ).total_seconds()
                    if time_diff < self.decay_config.window:
                        decay_factor = calculate_decay_factor(
                            time_diff,
                            self.decay_config.window,
                            self.decay_config.type,
                        )
                        if decay_factor > 0.1:
                            is_active = True
                            decay_status[entity_id] = time_diff
                            active_triggers.append(
                                f"{entity_id} (decay: {decay_factor:.2f})"
                            )
                            # Modify probabilities based on decay
                            p_true = p_true * decay_factor
                            p_false = p_false + ((1 - p_false) * (1 - decay_factor))

                if is_active:
                    current_probability = update_probability(
                        current_probability, p_true, p_false, self.motion_weight
                    )
                else:
                    current_probability = update_probability(
                        current_probability, 1 - p_true, 1 - p_false, self.motion_weight
                    )

                sensor_probs[entity_id] = current_probability

            # Process media devices
            for entity_id in self.media_devices:
                state = sensor_states.get(entity_id)
                if not state or not state.get("availability", False):
                    continue

                p_true, p_false = get_timeslot_probabilities(entity_id, timeslot)
                current_state = state["state"]
                device_states["media_states"][entity_id] = current_state

                if current_state == STATE_PLAYING:
                    active_triggers.append(f"{entity_id} (playing)")
                    current_probability = update_probability(
                        current_probability, p_true, p_false, self.media_weight
                    )
                elif current_state == STATE_PAUSED:
                    active_triggers.append(f"{entity_id} (paused)")
                    current_probability = update_probability(
                        current_probability,
                        p_true * 0.7,
                        p_false * 0.7,
                        self.media_weight,
                    )

                sensor_probs[entity_id] = current_probability

            # Process appliances
            for entity_id in self.appliances:
                state = sensor_states.get(entity_id)
                if not state or not state.get("availability", False):
                    continue

                p_true, p_false = get_timeslot_probabilities(entity_id, timeslot)
                current_state = state["state"]
                device_states["appliance_states"][entity_id] = current_state

                if current_state == STATE_ON:
                    active_triggers.append(f"{entity_id} (on)")
                    current_probability = update_probability(
                        current_probability, p_true, p_false, self.appliance_weight
                    )

                sensor_probs[entity_id] = current_probability

            # Process environmental sensors
            env_sensors = (
                self.illuminance_sensors
                + self.humidity_sensors
                + self.temperature_sensors
            )

            for entity_id in env_sensors:
                state = sensor_states.get(entity_id)
                if not state or not state.get("availability", False):
                    continue

                try:
                    current_value = float(state["state"])
                    p_true, p_false = get_timeslot_probabilities(entity_id, timeslot)

                    # Check thresholds based on sensor type
                    is_active = False
                    if entity_id in self.illuminance_sensors:
                        threshold = ENVIRONMENTAL_SETTINGS["illuminance"][
                            "change_threshold"
                        ]
                        if current_value > threshold:
                            is_active = True
                            active_triggers.append(
                                f"{entity_id} ({current_value:.1f} lx)"
                            )
                    elif entity_id in self.temperature_sensors:
                        threshold = ENVIRONMENTAL_SETTINGS["temperature"][
                            "change_threshold"
                        ]
                        baseline = ENVIRONMENTAL_SETTINGS["temperature"]["baseline"]
                        if abs(current_value - baseline) > threshold:
                            is_active = True
                            active_triggers.append(
                                f"{entity_id} ({current_value:.1f}°C)"
                            )
                    elif entity_id in self.humidity_sensors:
                        threshold = ENVIRONMENTAL_SETTINGS["humidity"][
                            "change_threshold"
                        ]
                        baseline = ENVIRONMENTAL_SETTINGS["humidity"]["baseline"]
                        if abs(current_value - baseline) > threshold:
                            is_active = True
                            active_triggers.append(
                                f"{entity_id} ({current_value:.1f}%)"
                            )

                    if is_active:
                        current_probability = update_probability(
                            current_probability,
                            p_true,
                            p_false,
                            self.environmental_weight,
                        )

                    sensor_probs[entity_id] = current_probability

                except (ValueError, TypeError):
                    continue

            # Calculate confidence score
            available_sensors = sum(
                1
                for state in sensor_states.values()
                if state.get("availability", False)
            )
            total_sensors = len(sensor_states)
            availability_score = (
                available_sensors / total_sensors
                if total_sensors > 0
                else MIN_PROBABILITY
            )

            confidence_score = (
                availability_score * CONFIDENCE_WEIGHTS["sensor_availability"]
            )

            # Apply minimum confidence threshold
            if confidence_score < MIN_CONFIDENCE:
                current_probability = max(
                    MIN_PROBABILITY, current_probability * confidence_score
                )

            # Ensure probability bounds
            final_probability = max(
                MIN_PROBABILITY, min(current_probability, MAX_PROBABILITY)
            )

            return {
                "probability": final_probability,
                "prior_probability": 0.5,  # Neutral prior
                "active_triggers": active_triggers,
                "sensor_probabilities": sensor_probs,
                "device_states": device_states,
                "decay_status": decay_status,
                "confidence_score": confidence_score,
                "sensor_availability": {
                    sensor_id: state.get("availability", False)
                    for sensor_id, state in sensor_states.items()
                },
            }

        except Exception as err:
            _LOGGER.error("Error in probability calculation: %s", err)
            raise HomeAssistantError(
                "Failed to calculate occupancy probability"
            ) from err
