"""Probability calculations for Area Occupancy Detection."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_PLAYING,
    STATE_PAUSED,
    STATE_IDLE,
)
from homeassistant.util import dt as dt_util

from .const import ProbabilityResult, SensorStates

_LOGGER = logging.getLogger(__name__)


@dataclass
class DecayConfig:
    """Configuration for sensor decay calculations."""

    enabled: bool = True
    window: int = 600  # seconds
    type: str = "linear"


@dataclass
class MotionCalculationResult:
    """Result of motion probability calculation."""

    probability: float
    triggers: list[str]
    decay_status: dict[str, float]


@dataclass
class MediaCalculationResult:
    """Result of media device probability calculation."""

    probability: float
    triggers: list[str]
    states: dict[str, str]


@dataclass
class ApplianceCalculationResult:
    """Result of appliance probability calculation."""

    probability: float
    triggers: list[str]
    states: dict[str, str]


@dataclass
class EnvironmentalCalculationResult:
    """Result of environmental probability calculation."""

    probability: float
    triggers: list[str]


class ProbabilityCalculator:
    """Handles probability calculations for area occupancy."""

    def __init__(
        self,
        base_config: dict[str, Any],
        motion_sensors: list[str],
        media_devices: list[str] | None = None,
        appliances: list[str] | None = None,
        illuminance_sensors: list[str] | None = None,
        humidity_sensors: list[str] | None = None,
        temperature_sensors: list[str] | None = None,
        decay_config: DecayConfig | None = None,
    ) -> None:
        """Initialize the calculator."""
        self.base_config = base_config
        self.motion_sensors = motion_sensors
        self.media_devices = media_devices or []
        self.appliances = appliances or []
        self.illuminance_sensors = illuminance_sensors or []
        self.humidity_sensors = humidity_sensors or []
        self.temperature_sensors = temperature_sensors or []
        self.decay_config = decay_config or DecayConfig()

        # Load weights from base config
        weights = self.base_config.get("weights", {})
        self.motion_weight = weights.get("motion", 0.4)
        self.media_weight = weights.get("media", 0.3)
        self.appliance_weight = weights.get("appliances", 0.2)
        self.environmental_weight = weights.get("environmental", 0.1)

    def calculate(
        self,
        sensor_states: SensorStates,
        motion_timestamps: dict[str, datetime],
    ) -> ProbabilityResult:
        """Calculate overall area occupancy probability."""
        # Calculate individual probabilities
        motion_result = self._calculate_motion_probability(
            sensor_states, motion_timestamps
        )
        media_result = self._calculate_media_probability(sensor_states)
        appliance_result = self._calculate_appliance_probability(sensor_states)
        env_result = self._calculate_environmental_probability(sensor_states)

        # Combine probabilities using weights
        combined_prob = (
            motion_result.probability * self.motion_weight
            + media_result.probability * self.media_weight
            + appliance_result.probability * self.appliance_weight
            + env_result.probability * self.environmental_weight
        )

        # Calculate confidence based on sensor availability
        available_sensors = sum(
            1 for state in sensor_states.values() if state["availability"]
        )
        total_sensors = len(self._get_all_configured_sensors())
        confidence_score = (
            min(available_sensors / total_sensors, 1.0) if total_sensors > 0 else 0.0
        )

        return {
            "probability": combined_prob,
            "prior_probability": motion_result.probability,
            "active_triggers": (
                motion_result.triggers
                + media_result.triggers
                + appliance_result.triggers
                + env_result.triggers
            ),
            "sensor_probabilities": {
                "motion_probability": motion_result.probability,
                "media_probability": media_result.probability,
                "appliance_probability": appliance_result.probability,
                "environmental_probability": env_result.probability,
            },
            "device_states": {
                "media_states": media_result.states,
                "appliance_states": appliance_result.states,
            },
            "decay_status": motion_result.decay_status,
            "confidence_score": confidence_score,
            "sensor_availability": {
                sensor_id: state["availability"]
                for sensor_id, state in sensor_states.items()
            },
        }

    def _calculate_motion_probability(
        self,
        sensor_states: SensorStates,
        motion_timestamps: dict[str, datetime],
    ) -> MotionCalculationResult:
        """Calculate probability based on motion sensors."""
        active_triggers: list[str] = []
        decay_status: dict[str, float] = {}
        now = dt_util.utcnow()

        motion_base = self.base_config["base_probabilities"]["motion"]
        single_sensor_prob = motion_base["single_sensor"]
        multiple_sensors_prob = motion_base["multiple_sensors"]
        decay_factor = motion_base["decay_factor"]

        motion_probability = 0.0
        active_sensors = 0
        total_sensors = len(self.motion_sensors)

        if total_sensors == 0:
            return MotionCalculationResult(0.0, [], {})

        for entity_id in self.motion_sensors:
            if entity_id not in sensor_states:
                total_sensors -= 1
                continue

            state = sensor_states[entity_id]
            if not state["availability"]:
                total_sensors -= 1
                continue

            if state["state"] == STATE_ON:
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
                    motion_probability += current_decay * decay_factor
                    decay_status[entity_id] = time_diff
                    if current_decay > 0.1:
                        active_triggers.append(
                            f"{entity_id} (decay: {current_decay:.2f})"
                        )

        # Calculate final probability based on number of active sensors
        if active_sensors > 0:
            motion_probability = (
                multiple_sensors_prob if active_sensors > 1 else single_sensor_prob
            )

        final_probability = (
            motion_probability / total_sensors if total_sensors > 0 else 0.0
        )

        return MotionCalculationResult(
            probability=final_probability,
            triggers=active_triggers,
            decay_status=decay_status,
        )

    def _calculate_media_probability(
        self,
        sensor_states: SensorStates,
    ) -> MediaCalculationResult:
        """Calculate probability based on media device states."""
        active_triggers: list[str] = []
        device_states: dict[str, str] = {}
        total_probability = 0.0
        valid_devices = 0

        media_base = self.base_config["base_probabilities"]["media"]

        for entity_id in self.media_devices:
            state = sensor_states.get(entity_id)
            if not state or not state["availability"]:
                continue

            valid_devices += 1
            current_state = state["state"]
            device_states[entity_id] = current_state

            # Get probability based on state
            if current_state == STATE_PLAYING:
                prob = media_base["playing"]
                active_triggers.append(f"{entity_id} (playing)")
            elif current_state == STATE_PAUSED:
                prob = media_base["paused"]
                active_triggers.append(f"{entity_id} (paused)")
            elif current_state == STATE_IDLE:
                prob = media_base["idle"]
                active_triggers.append(f"{entity_id} (idle)")
            elif current_state == STATE_OFF:
                prob = media_base["off_state"]
            else:
                prob = media_base["default_state"]

            total_probability += prob

        final_probability = (
            total_probability / valid_devices if valid_devices > 0 else 0.0
        )

        return MediaCalculationResult(
            probability=final_probability,
            triggers=active_triggers,
            states=device_states,
        )

    def _calculate_appliance_probability(
        self,
        sensor_states: SensorStates,
    ) -> ApplianceCalculationResult:
        """Calculate probability based on appliance states."""
        active_triggers: list[str] = []
        device_states: dict[str, str] = {}
        total_probability = 0.0
        valid_devices = 0

        appliance_base = self.base_config["base_probabilities"]["appliances"]

        for entity_id in self.appliances:
            state = sensor_states.get(entity_id)
            if not state or not state["availability"]:
                continue

            valid_devices += 1
            current_state = state["state"]
            device_states[entity_id] = current_state

            # Get probability based on state
            if current_state == STATE_ON:
                prob = appliance_base["active_state"]
                active_triggers.append(f"{entity_id} (active)")
            elif current_state == "standby":
                prob = appliance_base["standby_state"]
                active_triggers.append(f"{entity_id} (standby)")
            elif current_state == STATE_OFF:
                prob = appliance_base["off_state"]
            else:
                prob = appliance_base["default_state"]

            total_probability += prob

        final_probability = (
            total_probability / valid_devices if valid_devices > 0 else 0.0
        )

        return ApplianceCalculationResult(
            probability=final_probability,
            triggers=active_triggers,
            states=device_states,
        )

    def _calculate_environmental_probability(
        self,
        sensor_states: SensorStates,
    ) -> EnvironmentalCalculationResult:
        """Calculate probability based on environmental sensors."""
        active_triggers: list[str] = []
        total_probability = 0.0
        valid_components = 0

        env_base = self.base_config["base_probabilities"]["environmental"]

        # Process illuminance
        illuminance_prob = self._process_illuminance_sensors(
            sensor_states, active_triggers, env_base["illuminance"]
        )
        if illuminance_prob is not None:
            total_probability += illuminance_prob
            valid_components += 1

        # Process temperature
        temp_prob = self._process_environmental_sensors(
            self.temperature_sensors,
            sensor_states,
            active_triggers,
            env_base["temperature"]["baseline"],
            env_base["temperature"]["change_threshold"],
            env_base["temperature"]["weight"],
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
            env_base["humidity"]["baseline"],
            env_base["humidity"]["change_threshold"],
            env_base["humidity"]["weight"],
            "%",
        )
        if humidity_prob is not None:
            total_probability += humidity_prob
            valid_components += 1

        final_probability = (
            total_probability / valid_components if valid_components > 0 else 0.0
        )

        return EnvironmentalCalculationResult(
            probability=final_probability,
            triggers=active_triggers,
        )

    def _process_illuminance_sensors(
        self,
        sensor_states: SensorStates,
        active_triggers: list[str],
        config: dict[str, Any],
    ) -> Optional[float]:
        """Process illuminance sensors and update triggers."""
        valid_sensors = 0
        total_prob = 0.0
        change_threshold = config["change_threshold"]
        significant_change_prob = config["significant_change"]
        minor_change_prob = config["minor_change"]

        for entity_id in self.illuminance_sensors:
            state = sensor_states.get(entity_id)
            if not state or not state["availability"]:
                continue

            try:
                current_value = float(state["state"])
                if current_value > change_threshold:
                    total_prob += significant_change_prob
                    active_triggers.append(f"{entity_id} ({current_value} lx)")
                else:
                    total_prob += minor_change_prob
                valid_sensors += 1
            except (ValueError, TypeError):
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
    ) -> Optional[float]:
        """Process environmental sensors and update triggers."""
        valid_sensors = 0
        significant_changes = 0

        for entity_id in sensor_list:
            state = sensor_states.get(entity_id)
            if not state or not state["availability"]:
                continue

            try:
                current_value = float(state["state"])
                valid_sensors += 1
                if abs(current_value - baseline) > threshold:
                    significant_changes += 1
                    active_triggers.append(f"{entity_id} ({current_value}{unit})")
            except (ValueError, TypeError):
                continue

        return (
            weight * (significant_changes / valid_sensors)
            if valid_sensors > 0
            else None
        )

    def _calculate_decay_factor(
        self, time_diff: float, window: float, decay_type: str
    ) -> float:
        """Calculate the decay factor based on time difference."""
        if decay_type == "linear":
            return max(0.0, 1.0 - (time_diff / window))
        # Exponential decay
        return math.exp(-3.0 * time_diff / window)

    def _get_all_configured_sensors(self) -> list[str]:
        """Get list of all configured sensor entity IDs."""
        return [
            *self.motion_sensors,
            *self.media_devices,
            *self.appliances,
            *self.illuminance_sensors,
            *self.humidity_sensors,
            *self.temperature_sensors,
        ]

    @staticmethod
    def _is_valid_number(value: Any) -> bool:
        """Check if a value can be converted to a float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _validate_probability(prob: float) -> float:
        """Validate and clamp probability to valid range."""
        return max(0.0, min(1.0, prob))
