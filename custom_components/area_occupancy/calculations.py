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
    # Environmental
    ENVIRONMENTAL_SETTINGS,
    # Thresholds
    MIN_PROBABILITY,
    MAX_PROBABILITY,
    MOTION_PROB_GIVEN_TRUE,
    MOTION_PROB_GIVEN_FALSE,
    MEDIA_PROB_GIVEN_TRUE,
    MEDIA_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
    APPLIANCE_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    DEFAULT_PROB_GIVEN_FALSE,
)

_LOGGER = logging.getLogger(__name__)


def update_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
) -> float:
    """Update probability using Bayes' rule."""
    # Calculate Bayesian update
    numerator = prob_given_true * prior
    denominator = numerator + prob_given_false * (1 - prior)

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
        coordinator,
        motion_sensors: list[str],
        media_devices: list[str] | None = None,
        appliances: list[str] | None = None,
        illuminance_sensors: list[str] | None = None,
        humidity_sensors: list[str] | None = None,
        temperature_sensors: list[str] | None = None,
        decay_config: DecayConfig | None = None,
    ) -> None:
        """Initialize the calculator with configuration."""
        self.coordinator = coordinator
        self.motion_sensors = motion_sensors
        self.media_devices = media_devices or []
        self.appliances = appliances or []
        self.illuminance_sensors = illuminance_sensors or []
        self.humidity_sensors = humidity_sensors or []
        self.temperature_sensors = temperature_sensors or []
        self.decay_config = decay_config or DecayConfig()

    async def calculate(
        self,
        sensor_states: SensorStates,
        motion_timestamps: dict[str, datetime],
        timeslot: Timeslot | None = None,
    ) -> ProbabilityResult:
        """Calculate overall probability using Bayesian inference."""
        try:
            # Initialize with motion sensor state
            motion_active = any(
                sensor_states.get(sensor_id, {}).get("state") == STATE_ON
                for sensor_id in self.motion_sensors
            )

            current_probability = (
                MOTION_PROB_GIVEN_TRUE
                if motion_active
                else (1 - MOTION_PROB_GIVEN_TRUE)
            )

            active_triggers = []
            decay_status = {}
            device_states = {"media_states": {}, "appliance_states": {}}
            sensor_probs = {}

            # Process motion sensors
            for entity_id in self.motion_sensors:
                state = sensor_states.get(entity_id)
                if not state or not state.get("availability", False):
                    continue

                p_true, p_false = get_timeslot_probabilities(entity_id, timeslot)
                is_active = False

                if state["state"] == STATE_ON:
                    is_active = True
                    active_triggers.append(entity_id)
                elif self.decay_config.enabled and entity_id in motion_timestamps:
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
                            p_true *= decay_factor
                            p_false = p_false + ((1 - p_false) * (1 - decay_factor))

                current_probability = update_probability(
                    current_probability,
                    p_true if is_active else (1 - p_true),
                    p_false if is_active else (1 - p_false),
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
                        current_probability, p_true, p_false
                    )
                elif current_state == STATE_PAUSED:
                    active_triggers.append(f"{entity_id} (paused)")
                    current_probability = update_probability(
                        current_probability,
                        p_true * 0.7,
                        p_false * 0.7,
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
                        current_probability, p_true, p_false
                    )
                sensor_probs[entity_id] = current_probability

            # Process environmental sensors
            for entity_id in (
                self.illuminance_sensors
                + self.humidity_sensors
                + self.temperature_sensors
            ):
                state = sensor_states.get(entity_id)
                if not state or not state.get("availability", False):
                    continue

                try:
                    current_value = float(state["state"])
                    p_true, p_false = get_timeslot_probabilities(entity_id, timeslot)
                    is_active = False
                    value_str = f"{current_value:.1f}"

                    if entity_id in self.illuminance_sensors:
                        settings = ENVIRONMENTAL_SETTINGS["illuminance"]
                        is_active = current_value > settings["change_threshold"]
                        value_str += " lx"
                    elif entity_id in self.temperature_sensors:
                        settings = ENVIRONMENTAL_SETTINGS["temperature"]
                        is_active = (
                            abs(current_value - settings["baseline"])
                            > settings["change_threshold"]
                        )
                        value_str += "°C"
                    elif entity_id in self.humidity_sensors:
                        settings = ENVIRONMENTAL_SETTINGS["humidity"]
                        is_active = (
                            abs(current_value - settings["baseline"])
                            > settings["change_threshold"]
                        )
                        value_str += "%"

                    if is_active:
                        active_triggers.append(f"{entity_id} ({value_str})")
                        current_probability = update_probability(
                            current_probability, p_true, p_false
                        )
                    sensor_probs[entity_id] = current_probability

                except (ValueError, TypeError):
                    continue

            # Ensure probability bounds
            final_probability = max(
                MIN_PROBABILITY, min(current_probability, MAX_PROBABILITY)
            )
            threshold_decimal = self.coordinator.get_threshold_decimal()

            # Add debug logging
            _LOGGER.debug(
                "Probability calculation: final=%.3f, threshold=%.3f",
                final_probability,
                threshold_decimal,
            )

            return {
                "probability": final_probability,
                "prior_probability": (
                    MOTION_PROB_GIVEN_TRUE
                    if motion_active
                    else (1 - MOTION_PROB_GIVEN_TRUE)
                ),
                "active_triggers": active_triggers,
                "sensor_probabilities": sensor_probs,
                "device_states": device_states,
                "decay_status": decay_status,
                "sensor_availability": {
                    sensor_id: state.get("availability", False)
                    for sensor_id, state in sensor_states.items()
                },
                "is_occupied": final_probability >= threshold_decimal,
            }

        except Exception as err:
            _LOGGER.error("Error in probability calculation: %s", err)
            raise HomeAssistantError(
                "Failed to calculate occupancy probability"
            ) from err

    def get_sensor_priors(self, entity_id: str) -> tuple[float, float]:
        """Get prior probabilities for a sensor."""
        if entity_id in self.motion_sensors:
            return MOTION_PROB_GIVEN_TRUE, MOTION_PROB_GIVEN_FALSE
        elif entity_id in self.media_devices:
            return MEDIA_PROB_GIVEN_TRUE, MEDIA_PROB_GIVEN_FALSE
        elif entity_id in self.appliances:
            return APPLIANCE_PROB_GIVEN_TRUE, APPLIANCE_PROB_GIVEN_FALSE
        # Add other sensor types as needed
        return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE
