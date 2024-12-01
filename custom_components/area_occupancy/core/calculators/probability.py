# custom_components/area_occupancy/core/calculators/probability.py

"""Probability calculations for Area Occupancy Detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

from homeassistant.const import STATE_ON, STATE_OFF, STATE_PLAYING, STATE_PAUSED
from homeassistant.util import dt as dt_util

from .. import AreaConfig, SensorState, ProbabilityResult

_LOGGER = logging.getLogger(__name__)


class SensorProbability(TypedDict):
    """Type for individual sensor probability calculation results."""

    probability: float
    triggers: list[str]
    decay_status: dict[str, float]


@dataclass
class DecayConfig:
    """Configuration for probability decay calculations."""

    enabled: bool = True
    window: int = 600  # seconds
    type: str = "linear"
    max_decay: float = 0.8


class ProbabilityCalculator:
    """Calculates occupancy probabilities from sensor states."""

    def __init__(self, config: AreaConfig) -> None:
        """Initialize the calculator with configuration."""
        self.config = config
        self._decay_config = DecayConfig(
            enabled=config.decay_enabled,
            window=config.decay_window,
            type=config.decay_type,
        )
        # Safety bounds
        self._min_confidence = 0.2
        self._max_change_rate = 0.3
        self._last_probability: float | None = None

    def calculate(self, sensor_states: dict[str, SensorState]) -> ProbabilityResult:
        """Calculate overall probability based on all sensors."""
        # Calculate individual probabilities
        motion_prob = self._calculate_motion_probability(sensor_states)
        media_prob = self._calculate_media_probability(sensor_states)
        appliance_prob = self._calculate_appliance_probability(sensor_states)
        environmental_prob = self._calculate_environmental_probability(sensor_states)

        # Weight the probabilities
        weighted_probs = {
            "motion": (motion_prob, 0.4),  # 40% weight
            "media": (media_prob, 0.3),  # 30% weight
            "appliance": (appliance_prob, 0.2),  # 20% weight
            "environmental": (environmental_prob, 0.1),  # 10% weight
        }

        # Combine probabilities
        final_prob = self._combine_weighted_probabilities(weighted_probs)

        # Apply rate limiting
        final_prob = self._limit_probability_change(final_prob)

        # Calculate confidence score
        confidence = self._calculate_confidence_score(sensor_states)

        return ProbabilityResult(
            probability=final_prob,
            prior_probability=motion_prob["probability"],
            active_triggers=(
                motion_prob["triggers"]
                + media_prob["triggers"]
                + appliance_prob["triggers"]
                + environmental_prob["triggers"]
            ),
            sensor_probabilities={
                "motion_probability": motion_prob["probability"],
                "media_probability": media_prob["probability"],
                "appliance_probability": appliance_prob["probability"],
                "environmental_probability": environmental_prob["probability"],
            },
            decay_status=motion_prob["decay_status"],
            confidence_score=confidence,
            sensor_availability=self._get_sensor_availability(sensor_states),
            device_states=self._get_device_states(sensor_states),
        )

    def _calculate_motion_probability(
        self, sensor_states: dict[str, SensorState]
    ) -> SensorProbability:
        """Calculate probability based on motion sensors."""
        active_triggers: list[str] = []
        decay_status: dict[str, float] = {}
        now = dt_util.utcnow()

        # Count active sensors
        active_sensors = 0
        valid_sensors = 0

        for entity_id in self.config.motion_sensors:
            if entity_id not in sensor_states:
                continue

            state = sensor_states[entity_id]
            if not state.available:
                continue

            valid_sensors += 1

            if state.state == STATE_ON:
                active_sensors += 1
                active_triggers.append(entity_id)
            elif self._decay_config.enabled:
                # Calculate decay for recently active sensors
                time_diff = (now - state.last_changed).total_seconds()
                if time_diff < self._decay_config.window:
                    decay_factor = self._calculate_decay_factor(time_diff)
                    if decay_factor > 0.1:  # Only count significant decay
                        decay_status[entity_id] = time_diff
                        active_triggers.append(
                            f"{entity_id} (decay: {decay_factor:.2f})"
                        )

        # Calculate base probability
        if valid_sensors == 0:
            base_prob = 0.0
        elif active_sensors > 0:
            base_prob = 0.9 if active_sensors > 1 else 0.85
        else:
            base_prob = 0.0

        # Apply decay adjustments
        total_decay = sum(
            self._calculate_decay_factor(time_diff)
            for time_diff in decay_status.values()
        )
        decay_adjustment = min(total_decay * 0.7, self._decay_config.max_decay)

        final_prob = max(base_prob, decay_adjustment)

        return {
            "probability": final_prob,
            "triggers": active_triggers,
            "decay_status": decay_status,
        }

    def _calculate_media_probability(
        self, sensor_states: dict[str, SensorState]
    ) -> SensorProbability:
        """Calculate probability based on media device states."""
        active_triggers: list[str] = []
        total_prob = 0.0
        valid_devices = 0

        for entity_id in self.config.media_devices:
            if entity_id not in sensor_states:
                continue

            state = sensor_states[entity_id]
            if not state.available:
                continue

            valid_devices += 1
            current_state = state.state

            # Assign probabilities based on state
            if current_state == STATE_PLAYING:
                prob = 0.9
                active_triggers.append(f"{entity_id} (playing)")
            elif current_state == STATE_PAUSED:
                prob = 0.7
                active_triggers.append(f"{entity_id} (paused)")
            elif current_state == "idle":
                prob = 0.3
                active_triggers.append(f"{entity_id} (idle)")
            elif current_state == STATE_OFF:
                prob = 0.1
            else:
                prob = 0.0

            total_prob += prob

        final_prob = total_prob / valid_devices if valid_devices > 0 else 0.0

        return {
            "probability": final_prob,
            "triggers": active_triggers,
            "decay_status": {},
        }

    def _calculate_appliance_probability(
        self, sensor_states: dict[str, SensorState]
    ) -> SensorProbability:
        """Calculate probability based on appliance states."""
        active_triggers: list[str] = []
        total_prob = 0.0
        valid_devices = 0

        for entity_id in self.config.appliances:
            if entity_id not in sensor_states:
                continue

            state = sensor_states[entity_id]
            if not state.available:
                continue

            valid_devices += 1
            current_state = state.state

            # Assign probabilities based on state
            if current_state == STATE_ON:
                prob = 0.8
                active_triggers.append(f"{entity_id} (active)")
            elif current_state == "standby":
                prob = 0.4
                active_triggers.append(f"{entity_id} (standby)")
            else:
                prob = 0.1

            total_prob += prob

        final_prob = total_prob / valid_devices if valid_devices > 0 else 0.0

        return {
            "probability": final_prob,
            "triggers": active_triggers,
            "decay_status": {},
        }

    def _calculate_environmental_probability(
        self, sensor_states: dict[str, SensorState]
    ) -> SensorProbability:
        """Calculate probability based on environmental sensors."""
        active_triggers: list[str] = []
        env_probs: list[float] = []

        # Process illuminance sensors
        illuminance_prob = self._process_illuminance_sensors(
            sensor_states, active_triggers
        )
        if illuminance_prob is not None:
            env_probs.append(illuminance_prob)

        # Process temperature sensors
        temp_prob = self._process_temperature_sensors(sensor_states, active_triggers)
        if temp_prob is not None:
            env_probs.append(temp_prob)

        # Process humidity sensors
        humidity_prob = self._process_humidity_sensors(sensor_states, active_triggers)
        if humidity_prob is not None:
            env_probs.append(humidity_prob)

        # Calculate final environmental probability
        final_prob = sum(env_probs) / len(env_probs) if env_probs else 0.0

        return {
            "probability": final_prob,
            "triggers": active_triggers,
            "decay_status": {},
        }

    def _process_illuminance_sensors(
        self, sensor_states: dict[str, SensorState], triggers: list[str]
    ) -> float | None:
        """Process illuminance sensors and update triggers."""
        valid_sensors = 0
        total_prob = 0.0
        threshold = 50  # lux

        for entity_id in self.config.illuminance_sensors:
            if entity_id not in sensor_states:
                continue

            state = sensor_states[entity_id]
            if not state.available:
                continue

            try:
                current_value = float(state.state)
                valid_sensors += 1

                if current_value > threshold:
                    prob = 0.7
                    triggers.append(f"{entity_id} ({current_value:.1f} lx)")
                else:
                    prob = 0.3

                total_prob += prob

            except (ValueError, TypeError):
                continue

        return total_prob / valid_sensors if valid_sensors > 0 else None

    def _process_temperature_sensors(
        self, sensor_states: dict[str, SensorState], triggers: list[str]
    ) -> float | None:
        """Process temperature sensors and update triggers."""
        return self._process_environmental_sensor_type(
            sensor_states,
            self.config.temperature_sensors,
            21.0,  # baseline temperature
            1.5,  # significant change threshold
            triggers,
            "°C",
        )

    def _process_humidity_sensors(
        self, sensor_states: dict[str, SensorState], triggers: list[str]
    ) -> float | None:
        """Process humidity sensors and update triggers."""
        return self._process_environmental_sensor_type(
            sensor_states,
            self.config.humidity_sensors,
            50.0,  # baseline humidity
            10.0,  # significant change threshold
            triggers,
            "%",
        )

    def _process_environmental_sensor_type(
        self,
        sensor_states: dict[str, SensorState],
        sensor_list: list[str],
        baseline: float,
        threshold: float,
        triggers: list[str],
        unit: str,
    ) -> float | None:
        """Generic environmental sensor processing."""
        valid_sensors = 0
        significant_changes = 0

        for entity_id in sensor_list:
            if entity_id not in sensor_states:
                continue

            state = sensor_states[entity_id]
            if not state.available:
                continue

            try:
                current_value = float(state.state)
                valid_sensors += 1

                if abs(current_value - baseline) > threshold:
                    significant_changes += 1
                    triggers.append(f"{entity_id} ({current_value}{unit})")

            except (ValueError, TypeError):
                continue

        return significant_changes / valid_sensors if valid_sensors > 0 else None

    def _calculate_decay_factor(self, time_diff: float) -> float:
        """Calculate decay factor based on configuration."""
        if self._decay_config.type == "linear":
            return max(0.0, 1.0 - (time_diff / self._decay_config.window))
        else:  # exponential
            return max(0.0, pow(0.5, time_diff / (self._decay_config.window / 2)))

    def _combine_weighted_probabilities(
        self, probabilities: dict[str, tuple[SensorProbability, float]]
    ) -> float:
        """Combine weighted probabilities with validation."""
        total_weight = sum(weight for _, weight in probabilities.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            prob["probability"] * weight for prob, weight in probabilities.values()
        )

        return min(1.0, weighted_sum / total_weight)

    def _limit_probability_change(self, new_probability: float) -> float:
        """Limit the rate of probability change."""
        if self._last_probability is None:
            self._last_probability = new_probability
            return new_probability

        max_change = self._max_change_rate
        min_prob = max(0.0, self._last_probability - max_change)
        max_prob = min(1.0, self._last_probability + max_change)

        limited_prob = max(min_prob, min(new_probability, max_prob))
        self._last_probability = limited_prob
        return limited_prob

    def _calculate_confidence_score(
        self, sensor_states: dict[str, SensorState]
    ) -> float:
        """Calculate overall confidence score."""
        available_sensors = sum(
            1 for state in sensor_states.values() if state.available
        )
        total_sensors = len(sensor_states)

        if total_sensors == 0:
            return 0.0

        # Base confidence on sensor availability
        confidence = available_sensors / total_sensors

        # Apply minimum confidence threshold
        return max(self._min_confidence, confidence)

    def _get_sensor_availability(
        self, sensor_states: dict[str, SensorState]
    ) -> dict[str, bool]:
        """Get availability status for all sensors."""
        return {
            entity_id: state.available for entity_id, state in sensor_states.items()
        }

    def _get_device_states(
        self, sensor_states: dict[str, SensorState]
    ) -> dict[str, dict[str, str]]:
        """Get current states of all devices."""
        return {
            "media_states": {
                entity_id: str(sensor_states[entity_id].state)
                for entity_id in self.config.media_devices
                if entity_id in sensor_states
            },
            "appliance_states": {
                entity_id: str(sensor_states[entity_id].state)
                for entity_id in self.config.appliances
                if entity_id in sensor_states
            },
        }
