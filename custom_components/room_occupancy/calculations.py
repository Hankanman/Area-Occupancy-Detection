"""Probability calculations for Room Occupancy Detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util

from .const import (
    EnvironmentalData,
    ProbabilityResult,
    SensorStates,
)


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
class EnvironmentalCalculationResult:
    """Result of environmental probability calculation."""

    probability: float
    triggers: list[str]


class ProbabilityCalculator:
    """Handles probability calculations for room occupancy."""

    def __init__(
        self,
        motion_sensors: list[str],
        illuminance_sensors: list[str] | None = None,
        humidity_sensors: list[str] | None = None,
        temperature_sensors: list[str] | None = None,
        device_states: list[str] | None = None,
        decay_config: DecayConfig | None = None,
    ) -> None:
        """Initialize the calculator."""
        self.motion_sensors = motion_sensors
        self.illuminance_sensors = illuminance_sensors or []
        self.humidity_sensors = humidity_sensors or []
        self.temperature_sensors = temperature_sensors or []
        self.device_states = device_states or []
        self.decay_config = decay_config or DecayConfig()

    def calculate(
        self,
        sensor_states: SensorStates,
        motion_timestamps: dict[str, datetime],
    ) -> ProbabilityResult:
        """Calculate overall room occupancy probability."""
        motion_result = self._calculate_motion_probability(
            sensor_states, motion_timestamps
        )
        env_result = self._calculate_environmental_probability(sensor_states)

        # Weight motion more heavily than environmental factors
        combined_prob = (motion_result.probability * 0.7) + (
            env_result.probability * 0.3
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
            "active_triggers": motion_result.triggers + env_result.triggers,
            "sensor_probabilities": {
                "motion_probability": motion_result.probability,
                "environmental_probability": env_result.probability,
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

        motion_probability = 0.0
        total_weight = len(self.motion_sensors)

        if total_weight == 0:
            return MotionCalculationResult(0.0, [], {})

        for entity_id in self.motion_sensors:
            if entity_id not in sensor_states:
                total_weight -= 1
                continue

            state = sensor_states[entity_id]
            if not state["availability"]:
                total_weight -= 1
                continue

            if state["state"] == STATE_ON:
                motion_probability += 1.0
                active_triggers.append(entity_id)
            elif self.decay_config.enabled and entity_id in motion_timestamps:
                last_motion = motion_timestamps[entity_id]
                time_diff = (now - last_motion).total_seconds()

                if time_diff < self.decay_config.window:
                    decay_factor = self._calculate_decay_factor(
                        time_diff,
                        self.decay_config.window,
                        self.decay_config.type,
                    )
                    motion_probability += decay_factor
                    decay_status[entity_id] = time_diff
                    if decay_factor > 0.1:
                        active_triggers.append(
                            f"{entity_id} (decay: {decay_factor:.2f})"
                        )

        final_probability = (
            motion_probability / total_weight if total_weight > 0 else 0.0
        )

        return MotionCalculationResult(
            probability=final_probability,
            triggers=active_triggers,
            decay_status=decay_status,
        )

    def _calculate_environmental_probability(
        self, sensor_states: SensorStates
    ) -> EnvironmentalCalculationResult:
        """Calculate probability based on environmental sensors."""
        active_triggers: list[str] = []
        total_probability = 0.0
        total_sensors = 0

        # Process illuminance
        illuminance_prob = self._process_illuminance_sensors(
            sensor_states, active_triggers
        )
        if illuminance_prob is not None:
            total_probability += illuminance_prob
            total_sensors += 1

        # Process temperature and humidity
        env_data = [
            EnvironmentalData(
                current_value=float(state["state"]),
                baseline=21.0,  # Temperature baseline
                threshold=1.5,
                weight=0.6,
            )
            for entity_id in self.temperature_sensors
            if (state := sensor_states.get(entity_id))
            and state["availability"]
            and self._is_valid_number(state["state"])
        ]

        temp_prob = self._process_environmental_sensors(
            self.temperature_sensors,
            env_data,
            "°C",
            sensor_states,
            active_triggers,
        )
        if temp_prob is not None:
            total_probability += temp_prob
            total_sensors += 1

        # Process humidity
        humidity_data = [
            EnvironmentalData(
                current_value=float(state["state"]),
                baseline=50.0,  # Humidity baseline
                threshold=10.0,
                weight=0.5,
            )
            for entity_id in self.humidity_sensors
            if (state := sensor_states.get(entity_id))
            and state["availability"]
            and self._is_valid_number(state["state"])
        ]

        humidity_prob = self._process_environmental_sensors(
            self.humidity_sensors,
            humidity_data,
            "%",
            sensor_states,
            active_triggers,
        )
        if humidity_prob is not None:
            total_probability += humidity_prob
            total_sensors += 1

        # Process device states
        device_prob = self._process_device_states(sensor_states, active_triggers)
        if device_prob is not None:
            total_probability += device_prob
            total_sensors += 1

        final_probability = (
            total_probability / total_sensors if total_sensors > 0 else 0.0
        )

        return EnvironmentalCalculationResult(
            probability=final_probability,
            triggers=active_triggers,
        )

    def _process_illuminance_sensors(
        self,
        sensor_states: SensorStates,
        active_triggers: list[str],
    ) -> Optional[float]:
        """Process illuminance sensors and update triggers."""
        valid_sensors = 0
        total_prob = 0.0

        for entity_id in self.illuminance_sensors:
            state = sensor_states.get(entity_id)
            if not state or not state["availability"]:
                continue

            try:
                current_value = float(state["state"])
                if current_value > 50:  # Significant light level
                    total_prob += 0.7
                    active_triggers.append(f"{entity_id} ({current_value} lx)")
                valid_sensors += 1
            except (ValueError, TypeError):
                continue

        return total_prob / valid_sensors if valid_sensors > 0 else None

    def _process_environmental_sensors(
        self,
        sensor_list: list[str],
        sensor_data: list[EnvironmentalData],
        unit: str,
        sensor_states: SensorStates,
        active_triggers: list[str],
    ) -> Optional[float]:
        """Process environmental sensors and update triggers."""
        if not sensor_data:
            return None

        significant_changes = 0
        weight = 0.0
        for entity_id, data in zip(sensor_list, sensor_data):
            if abs(data["current_value"] - data["baseline"]) > data["threshold"]:
                significant_changes += 1
                active_triggers.append(f"{entity_id} ({data['current_value']}{unit})")
                weight = data["weight"]

        return weight if significant_changes > 0 else 0.0

    def _process_device_states(
        self,
        sensor_states: SensorStates,
        active_triggers: list[str],
    ) -> Optional[float]:
        """Process device states and update triggers."""
        valid_devices = 0
        active_devices = 0

        for entity_id in self.device_states:
            state = sensor_states.get(entity_id)
            if not state or not state["availability"]:
                continue

            valid_devices += 1
            if state["state"] == STATE_ON:
                active_devices += 1
                active_triggers.append(entity_id)

        return 0.8 * (active_devices / valid_devices) if valid_devices > 0 else None

    def _calculate_decay_factor(
        self, time_diff: float, window: float, decay_type: str
    ) -> float:
        """Calculate the decay factor based on time difference."""
        if decay_type == "linear":
            return 1.0 - (time_diff / window)
        # Exponential decay
        return math.exp(-3.0 * time_diff / window)

    def _get_all_configured_sensors(self) -> list[str]:
        """Get list of all configured sensor entity IDs."""
        return [
            *self.motion_sensors,
            *self.illuminance_sensors,
            *self.humidity_sensors,
            *self.temperature_sensors,
            *self.device_states,
        ]

    @staticmethod
    def _is_valid_number(value: Any) -> bool:
        """Check if a value can be converted to a float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
