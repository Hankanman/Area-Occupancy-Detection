import logging
from datetime import datetime

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.exceptions import HomeAssistantError
from homeassistant.const import (
    STATE_ON,
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
    MIN_PROBABILITY,
    MAX_PROBABILITY,
)
from .calculations import SensorType
from .probabilities import (
    DEFAULT_PROB_GIVEN_TRUE,
    DEFAULT_PROB_GIVEN_FALSE,
)

from .helpers import is_entity_active

_LOGGER = logging.getLogger(__name__)


class PriorCalculator:
    """Calculate occupancy probability based on sensor states."""

    def __init__(self, coordinator, probabilities) -> None:
        """Initialize the calculator."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self.probabilities = probabilities
        self.entity_types: dict[str, SensorType] = {}

        self.motion_sensors = self.config.get(CONF_MOTION_SENSORS, [])
        self.media_devices = self.config.get(CONF_MEDIA_DEVICES, [])
        self.appliances = self.config.get(CONF_APPLIANCES, [])
        self.illuminance_sensors = self.config.get(CONF_ILLUMINANCE_SENSORS, [])
        self.humidity_sensors = self.config.get(CONF_HUMIDITY_SENSORS, [])
        self.temperature_sensors = self.config.get(CONF_TEMPERATURE_SENSORS, [])
        self.door_sensors = self.config.get(CONF_DOOR_SENSORS, [])
        self.window_sensors = self.config.get(CONF_WINDOW_SENSORS, [])
        self.lights = self.config.get(CONF_LIGHTS, [])

    async def calculate_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> tuple[float, float, float]:
        """Calculate prior probability for a sensor based on historical data."""
        try:
            _LOGGER.debug(
                "Calculating prior for %s from %s to %s",
                entity_id,
                start_time,
                end_time,
            )

            # Get motion sensor states from history
            motion_states = {}
            for motion_sensor in self.coordinator.config.get("motion_sensors", []):
                sensor_id = motion_sensor  # Create local variable
                states = await self.coordinator.hass.async_add_executor_job(
                    lambda s=sensor_id: get_significant_states(  # Pass sensor_id as default argument
                        self.coordinator.hass,
                        start_time,
                        end_time,
                        [s],
                        minimal_response=False,
                    )
                )
                if states and motion_sensor in states:
                    motion_states[motion_sensor] = states[motion_sensor]

            if not motion_states:
                _LOGGER.debug("No motion data available for prior calculation")
                return (
                    DEFAULT_PROB_GIVEN_TRUE,
                    DEFAULT_PROB_GIVEN_FALSE,
                    MIN_PROBABILITY,
                )

            # Get entity states
            entity_states = await self.coordinator.hass.async_add_executor_job(
                lambda: get_significant_states(
                    self.coordinator.hass,
                    start_time,
                    end_time,
                    [entity_id],
                    minimal_response=False,
                )
            )
            if not entity_states or entity_id not in entity_states:
                _LOGGER.debug("No entity data available for prior calculation")
                return (
                    DEFAULT_PROB_GIVEN_TRUE,
                    DEFAULT_PROB_GIVEN_FALSE,
                    MIN_PROBABILITY,
                )

            # Calculate motion durations
            motion_active_time = 0.0
            motion_inactive_time = 0.0
            for states in motion_states.values():
                last_state = None
                last_time = start_time
                for state in states:
                    if last_state is not None:
                        duration = (state.last_changed - last_time).total_seconds()
                        if last_state.state == STATE_ON:
                            motion_active_time += duration
                        else:
                            motion_inactive_time += duration
                    last_state = state
                    last_time = state.last_changed

                # Add final state duration
                if last_state is not None:
                    duration = (end_time - last_time).total_seconds()
                    if last_state.state == STATE_ON:
                        motion_active_time += duration
                    else:
                        motion_inactive_time += duration

            total_time = motion_active_time + motion_inactive_time
            if total_time == 0:
                return (
                    DEFAULT_PROB_GIVEN_TRUE,
                    DEFAULT_PROB_GIVEN_FALSE,
                    MIN_PROBABILITY,
                )

            # Calculate prior based on motion sensor data
            prior = max(
                MIN_PROBABILITY, min(motion_active_time / total_time, MAX_PROBABILITY)
            )

            # Calculate conditional probabilities
            entity_active_time_with_motion = 0.0
            entity_active_time_without_motion = 0.0
            entity_states = entity_states[entity_id]
            last_state = None
            last_time = start_time

            for state in entity_states:
                if last_state is not None:
                    duration = (state.last_changed - last_time).total_seconds()
                    is_active = is_entity_active(
                        entity_id,
                        last_state.state,
                        self.entity_types,
                        self.probabilities.sensor_configs,
                    )
                    if is_active:
                        # Check if any motion was active during this period
                        motion_active = any(
                            self._was_motion_active(
                                ms_states, last_time, state.last_changed
                            )
                            for ms_states in motion_states.values()
                        )
                        if motion_active:
                            entity_active_time_with_motion += duration
                        else:
                            entity_active_time_without_motion += duration
                last_state = state
                last_time = state.last_changed

            # Calculate final probabilities
            prob_given_true = (
                entity_active_time_with_motion / motion_active_time
                if motion_active_time > 0
                else DEFAULT_PROB_GIVEN_TRUE
            )
            prob_given_false = (
                entity_active_time_without_motion / motion_inactive_time
                if motion_inactive_time > 0
                else DEFAULT_PROB_GIVEN_FALSE
            )

            # Clamp probabilities
            prob_given_true = max(
                MIN_PROBABILITY, min(prob_given_true, MAX_PROBABILITY)
            )
            prob_given_false = max(
                MIN_PROBABILITY, min(prob_given_false, MAX_PROBABILITY)
            )

            # Update coordinator's learned priors
            self.coordinator.update_learned_priors(
                entity_id, prob_given_true, prob_given_false, prior
            )

            return prob_given_true, prob_given_false, prior

        except (HomeAssistantError, ValueError, RuntimeError) as err:
            _LOGGER.error("Error calculating prior for %s: %s", entity_id, err)
            return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE, MIN_PROBABILITY

    def _was_motion_active(
        self, motion_states: list, start_time: datetime, end_time: datetime
    ) -> bool:
        """Check if motion was active during the given time period."""
        for state in motion_states:
            if (
                state.last_changed >= start_time
                and state.last_changed <= end_time
                and state.state == STATE_ON
            ):
                return True
        return False
