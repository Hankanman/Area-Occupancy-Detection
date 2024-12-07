"""Probability calculations for Area Occupancy Detection."""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Any

from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_PLAYING,
    STATE_PAUSED,
    STATE_IDLE,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .types import (
    ProbabilityResult,
    SensorStates,
    DecayConfig,
    CalculationResult,
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
    # Motion detection
    MOTION_SINGLE_SENSOR_PROBABILITY,
    MOTION_MULTIPLE_SENSORS_PROBABILITY,
    MOTION_DECAY_FACTOR,
    MAX_COMPOUND_DECAY,
    MIN_MOTION_WEIGHT,
    # Device states
    MEDIA_STATE_PROBABILITIES,
    APPLIANCE_STATE_PROBABILITIES,
    # Environmental
    ENVIRONMENTAL_SETTINGS,
    # Thresholds
    CORRELATION_LIMIT,
    MIN_CONFIDENCE,
    MIN_PROBABILITY,
    MAX_PROBABILITY,
    MAX_WEIGHT_ADJUSTMENT,
)

_LOGGER = logging.getLogger(__name__)


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

        # Normalize weights
        total_weight = (
            self.motion_weight
            + self.media_weight
            + self.appliance_weight
            + self.environmental_weight
        )

        if total_weight > 0:
            self.motion_weight /= total_weight
            self.media_weight /= total_weight
            self.appliance_weight /= total_weight
            self.environmental_weight /= total_weight
        else:
            self._set_default_weights()

    def _set_default_weights(self) -> None:
        """Set default weights if configuration is invalid."""
        self.motion_weight = DEFAULT_MOTION_WEIGHT
        self.media_weight = DEFAULT_MEDIA_WEIGHT
        self.appliance_weight = DEFAULT_APPLIANCE_WEIGHT
        self.environmental_weight = DEFAULT_ENVIRONMENTAL_WEIGHT

    def calculate(
        self,
        sensor_states: SensorStates,
        motion_timestamps: dict[str, datetime],
        historical_patterns: dict[str, Any] | None = None,
    ) -> ProbabilityResult:
        """Calculate overall area occupancy probability."""
        try:
            # Calculate individual probabilities
            motion_result = self._calculate_motion_probability(
                sensor_states, motion_timestamps
            )
            media_result = self._calculate_media_probability(sensor_states)
            appliance_result = self._calculate_appliance_probability(sensor_states)
            env_result = self._calculate_environmental_probability(sensor_states)

            # Apply sensor correlations if available
            sensor_correlations = (
                historical_patterns.get("sensor_correlations", {})
                if historical_patterns
                else {}
            )

            # Adjust weights based on correlations
            motion_weight = self._adjust_motion_weight(
                self.motion_weight, sensor_correlations
            )
            media_weight = self.media_weight
            appliance_weight = self.appliance_weight
            env_weight = self.environmental_weight

            # Normalize adjusted weights
            total_weight = motion_weight + media_weight + appliance_weight + env_weight
            if total_weight > 0:
                weights = {
                    "motion": motion_weight / total_weight,
                    "media": media_weight / total_weight,
                    "appliance": appliance_weight / total_weight,
                    "environmental": env_weight / total_weight,
                }
            else:
                weights = {
                    "motion": DEFAULT_MOTION_WEIGHT,
                    "media": DEFAULT_MEDIA_WEIGHT,
                    "appliance": DEFAULT_APPLIANCE_WEIGHT,
                    "environmental": DEFAULT_ENVIRONMENTAL_WEIGHT,
                }

            # Calculate final probability
            final_prob = (
                motion_result["probability"] * weights["motion"]
                + media_result["probability"] * weights["media"]
                + appliance_result["probability"] * weights["appliance"]
                + env_result["probability"] * weights["environmental"]
            )

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                sensor_states, motion_result, media_result, appliance_result, env_result
            )

            return {
                "probability": max(MIN_PROBABILITY, min(final_prob, MAX_PROBABILITY)),
                "prior_probability": motion_result["probability"],
                "active_triggers": (
                    motion_result["triggers"]
                    + media_result["triggers"]
                    + appliance_result["triggers"]
                    + env_result["triggers"]
                ),
                "sensor_probabilities": {
                    "motion_probability": motion_result["probability"],
                    "media_probability": media_result["probability"],
                    "appliance_probability": appliance_result["probability"],
                    "environmental_probability": env_result["probability"],
                },
                "device_states": {
                    "media_states": media_result["states"],
                    "appliance_states": appliance_result["states"],
                },
                "decay_status": motion_result["decay_status"],
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

    def _adjust_motion_weight(
        self, base_weight: float, correlations: dict[str, dict[str, float]]
    ) -> float:
        """Adjust motion weight based on sensor correlations."""
        if not correlations:
            return base_weight

        max_correlation = 0.0
        for sensor_correlations in correlations.values():
            if sensor_correlations:
                max_correlation = max(
                    max_correlation,
                    min(max(sensor_correlations.values()), CORRELATION_LIMIT),
                )

        if max_correlation > 0.5:
            reduction = max_correlation * MAX_WEIGHT_ADJUSTMENT
            return max(MIN_MOTION_WEIGHT, base_weight * (1 - reduction))

        return base_weight

    def _calculate_motion_probability(
        self,
        sensor_states: SensorStates,
        motion_timestamps: dict[str, datetime],
    ) -> CalculationResult:
        """Calculate probability based on motion sensors."""
        active_triggers: list[str] = []
        decay_status: dict[str, float] = {}
        now = dt_util.utcnow()

        motion_probability = 0.0
        compound_decay = 0.0
        active_sensors = 0
        valid_sensors = 0

        for entity_id in self.motion_sensors:
            state = sensor_states.get(entity_id)
            if not state or not state.get("availability", False):
                continue

            valid_sensors += 1
            current_state = state["state"]

            if current_state == STATE_ON:
                active_sensors += 1
                active_triggers.append(entity_id)
            elif self.decay_config.enabled and entity_id in motion_timestamps:
                last_motion = motion_timestamps[entity_id]
                time_diff = (now - last_motion).total_seconds()

                if time_diff < self.decay_config.window:
                    current_decay = self._calculate_decay_factor(
                        time_diff,
                        self.decay_config.window,
                        self.decay_config.type,
                    )
                    remaining_decay = MAX_COMPOUND_DECAY - compound_decay
                    if remaining_decay > 0:
                        applied_decay = min(
                            current_decay * MOTION_DECAY_FACTOR, remaining_decay
                        )
                        compound_decay += applied_decay
                        motion_probability += applied_decay
                        decay_status[entity_id] = time_diff
                        if applied_decay > 0.1:
                            active_triggers.append(
                                f"{entity_id} (decay: {applied_decay:.2f})"
                            )

        if active_sensors > 0:
            motion_probability = (
                MOTION_MULTIPLE_SENSORS_PROBABILITY
                if active_sensors > 1
                else MOTION_SINGLE_SENSOR_PROBABILITY
            )

        # Calculate final probability with safety bounds
        final_probability = (
            min(motion_probability / valid_sensors, MAX_PROBABILITY)
            if valid_sensors > 0
            else MIN_PROBABILITY
        )

        return {
            "probability": final_probability,
            "triggers": active_triggers,
            "result_type": "motion",
            "decay_status": decay_status,
            "states": {},
        }

    def _calculate_media_probability(
        self,
        sensor_states: SensorStates,
    ) -> CalculationResult:
        """Calculate probability based on media device states."""
        active_triggers: list[str] = []
        device_states: dict[str, str] = {}
        total_probability = 0.0
        valid_devices = 0

        for entity_id in self.media_devices:
            state = sensor_states.get(entity_id)
            if not state or not state.get("availability", False):
                continue

            valid_devices += 1
            current_state = state["state"]
            device_states[entity_id] = current_state

            if current_state == STATE_PLAYING:
                prob = MEDIA_STATE_PROBABILITIES["playing"]
                active_triggers.append(f"{entity_id} (playing)")
            elif current_state == STATE_PAUSED:
                prob = MEDIA_STATE_PROBABILITIES["paused"]
                active_triggers.append(f"{entity_id} (paused)")
            elif current_state == STATE_IDLE:
                prob = MEDIA_STATE_PROBABILITIES["idle"]
                active_triggers.append(f"{entity_id} (idle)")
            elif current_state == STATE_OFF:
                prob = MEDIA_STATE_PROBABILITIES["off"]
            else:
                prob = MEDIA_STATE_PROBABILITIES["default"]

            total_probability += prob

        final_probability = (
            total_probability / valid_devices if valid_devices > 0 else MIN_PROBABILITY
        )

        return {
            "probability": min(final_probability, MAX_PROBABILITY),
            "triggers": active_triggers,
            "result_type": "media",
            "states": device_states,
            "decay_status": {},
        }

    def _calculate_appliance_probability(
        self,
        sensor_states: SensorStates,
    ) -> CalculationResult:
        """Calculate probability based on appliance states."""
        active_triggers: list[str] = []
        device_states: dict[str, str] = {}
        total_probability = 0.0
        valid_devices = 0

        for entity_id in self.appliances:
            state = sensor_states.get(entity_id)
            if not state or not state.get("availability", False):
                continue

            valid_devices += 1
            current_state = state["state"]
            device_states[entity_id] = current_state

            # Get probability based on state
            if current_state == STATE_ON:
                prob = APPLIANCE_STATE_PROBABILITIES["active"]
                active_triggers.append(f"{entity_id} (active)")
            elif current_state == "standby":
                prob = APPLIANCE_STATE_PROBABILITIES["standby"]
                active_triggers.append(f"{entity_id} (standby)")
            elif current_state == STATE_OFF:
                prob = APPLIANCE_STATE_PROBABILITIES["off"]
            else:
                prob = APPLIANCE_STATE_PROBABILITIES["default"]

            total_probability += prob

        final_probability = (
            total_probability / valid_devices if valid_devices > 0 else MIN_PROBABILITY
        )

        return {
            "probability": min(final_probability, MAX_PROBABILITY),
            "triggers": active_triggers,
            "result_type": "appliance",
            "states": device_states,
            "decay_status": {},
        }

    def _calculate_environmental_probability(
        self,
        sensor_states: SensorStates,
    ) -> CalculationResult:
        """Calculate probability based on environmental sensors."""
        active_triggers: list[str] = []
        total_probability = 0.0
        valid_components = 0

        # Process illuminance
        illuminance_prob = self._process_illuminance_sensors(
            sensor_states, active_triggers, ENVIRONMENTAL_SETTINGS["illuminance"]
        )
        if illuminance_prob is not None:
            total_probability += illuminance_prob
            valid_components += 1

        # Process temperature
        temp_prob = self._process_environmental_sensors(
            self.temperature_sensors,
            sensor_states,
            active_triggers,
            ENVIRONMENTAL_SETTINGS["temperature"]["baseline"],
            ENVIRONMENTAL_SETTINGS["temperature"]["change_threshold"],
            ENVIRONMENTAL_SETTINGS["temperature"]["weight"],
            "°C",
        )
        if temp_prob is not None:
            total_probability += temp_prob
            valid_components += 1

        # Process humidity
        humidity_prob = self._process_environmental_sensors(
            self.humidity_sensors,
            sensor_states,
            active_triggers,
            ENVIRONMENTAL_SETTINGS["humidity"]["baseline"],
            ENVIRONMENTAL_SETTINGS["humidity"]["change_threshold"],
            ENVIRONMENTAL_SETTINGS["humidity"]["weight"],
            "%",
        )
        if humidity_prob is not None:
            total_probability += humidity_prob
            valid_components += 1

        final_probability = (
            total_probability / valid_components
            if valid_components > 0
            else MIN_PROBABILITY
        )

        return {
            "probability": min(final_probability, MAX_PROBABILITY),
            "triggers": active_triggers,
            "result_type": "environmental",
            "states": {},
            "decay_status": {},
        }

    def _process_illuminance_sensors(
        self,
        sensor_states: SensorStates,
        active_triggers: list[str],
        config: dict[str, float],
    ) -> float | None:
        """Process illuminance sensors and update triggers."""
        valid_sensors = 0
        total_prob = 0.0

        for entity_id in self.illuminance_sensors:
            state = sensor_states.get(entity_id)
            if not state or not state.get("availability", False):
                continue

            try:
                current_value = float(state["state"])
                if current_value > config["change_threshold"]:
                    total_prob += config["significant_change_probability"]
                    active_triggers.append(f"{entity_id} ({current_value} lx)")
                else:
                    total_prob += config["minor_change_probability"]
                valid_sensors += 1
            except (ValueError, TypeError) as err:
                _LOGGER.debug("Invalid illuminance value for %s: %s", entity_id, err)
                continue

        return total_prob / valid_sensors if valid_sensors > 0 else None

    def _process_environmental_sensors(
        self,
        sensor_list: list[str],
        sensor_states: SensorStates,
        active_triggers: list[str],
        baseline: float,
        threshold: float,
        weight: float,
        unit: str,
    ) -> float | None:
        """Process environmental sensors and update triggers."""
        valid_sensors = 0
        significant_changes = 0

        for entity_id in sensor_list:
            state = sensor_states.get(entity_id)
            if not state or not state.get("availability", False):
                continue

            try:
                current_value = float(state["state"])
                valid_sensors += 1
                if abs(current_value - baseline) > threshold:
                    significant_changes += 1
                    active_triggers.append(f"{entity_id} ({current_value}{unit})")
            except (ValueError, TypeError) as err:
                _LOGGER.debug("Invalid sensor value for %s: %s", entity_id, err)
                continue

        return (
            weight * (significant_changes / valid_sensors)
            if valid_sensors > 0
            else None
        )

    def _calculate_decay_factor(
        self, time_diff: float, window: float, decay_type: str
    ) -> float:
        """Calculate decay factor based on time difference."""
        if time_diff >= window:
            return MIN_PROBABILITY
        ratio = time_diff / window
        return 1.0 - ratio if decay_type == "linear" else math.exp(-3.0 * ratio)

    def _calculate_confidence_score(
        self,
        sensor_states: SensorStates,
        motion_result: CalculationResult,
        media_result: CalculationResult,
        appliance_result: CalculationResult,
        env_result: CalculationResult,
    ) -> float:
        """Calculate overall confidence score."""
        # Sensor availability weight
        available_sensors = sum(
            1 for state in sensor_states.values() if state.get("availability", False)
        )
        total_sensors = len(sensor_states)
        availability_score = (
            available_sensors / total_sensors if total_sensors > 0 else MIN_PROBABILITY
        )

        # Trigger consistency weight
        all_probabilities = [
            motion_result["probability"],
            media_result["probability"],
            appliance_result["probability"],
            env_result["probability"],
        ]
        valid_probs = [p for p in all_probabilities if p is not None]

        if not valid_probs:
            consistency_score = MIN_PROBABILITY
        else:
            variance = sum(
                (p - sum(valid_probs) / len(valid_probs)) ** 2 for p in valid_probs
            ) / len(valid_probs)
            consistency_score = 1.0 - min(variance, MAX_PROBABILITY)

        # Combine scores with confidence weights
        final_score = (
            availability_score * CONFIDENCE_WEIGHTS["sensor_availability"]
            + consistency_score * CONFIDENCE_WEIGHTS["data_freshness"]
        )

        return max(MIN_CONFIDENCE, min(MAX_PROBABILITY, final_score))
