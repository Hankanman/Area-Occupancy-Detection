import logging
from datetime import datetime
from typing import Any, Literal
from homeassistant.core import State, HomeAssistant

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.exceptions import HomeAssistantError
from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
)
from sqlalchemy.exc import SQLAlchemyError

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
    DEFAULT_PRIOR,
)

from .probabilities import (
    DEFAULT_PROB_GIVEN_TRUE,
    DEFAULT_PROB_GIVEN_FALSE,
    MOTION_DEFAULT_PRIOR,
    MEDIA_DEFAULT_PRIOR,
    APPLIANCE_DEFAULT_PRIOR,
    DOOR_DEFAULT_PRIOR,
    WINDOW_DEFAULT_PRIOR,
    LIGHT_DEFAULT_PRIOR,
    ENVIRONMENTAL_DEFAULT_PRIOR,
)

from .helpers import is_entity_active

_LOGGER = logging.getLogger(__name__)

SensorType = Literal[
    "motion",
    "media",
    "appliance",
    "door",
    "window",
    "light",
    "environmental",
]


class PriorCalculator:
    """Calculate occupancy probability based on sensor states."""

    def __init__(self, coordinator, probabilities, hass: HomeAssistant) -> None:
        """Initialize the calculator."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self.probabilities = probabilities
        self.hass = hass
        self.entity_types: dict[str, SensorType] = {}
        self._map_entities_to_types()

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
        """Calculate learned priors for a given entity."""
        _LOGGER.debug("Calculating prior for %s", entity_id)
        # Fetch motion sensor states
        motion_states = {}
        for motion_sensor in self.motion_sensors:
            ms = await self._get_states_from_recorder(
                motion_sensor, start_time, end_time
            )
            if ms:
                motion_states[motion_sensor] = ms

        if not motion_states:
            # No motion data, fallback to defaults
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                get_default_prior(entity_id, self),
            )

        # Fetch entity states
        entity_states = await self._get_states_from_recorder(
            entity_id, start_time, end_time
        )
        if not entity_states:
            # No entity data, fallback to defaults
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                get_default_prior(entity_id, self),
            )

        # Compute intervals for motion sensors once
        motion_intervals_by_sensor = {}
        for sensor_id, states in motion_states.items():
            intervals = self._states_to_intervals(states, start_time, end_time)
            motion_intervals_by_sensor[sensor_id] = intervals

        # Compute total durations for motion sensors from precomputed intervals
        motion_durations = self._compute_state_durations_from_intervals(
            motion_intervals_by_sensor
        )
        total_motion_active_time = motion_durations.get(STATE_ON, 0.0)
        total_motion_inactive_time = motion_durations.get(STATE_OFF, 0.0)
        total_motion_time = total_motion_active_time + total_motion_inactive_time

        if total_motion_time == 0:
            # No motion duration data, fallback to defaults
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                get_default_prior(entity_id, self),
            )

        # Calculate prior probability based on motion sensor and clamp it
        prior = max(
            MIN_PROBABILITY,
            min(total_motion_active_time / total_motion_time, MAX_PROBABILITY),
        )

        # Compute intervals for the entity once
        entity_intervals = self._states_to_intervals(
            entity_states, start_time, end_time
        )

        # Calculate conditional probabilities using precomputed intervals
        prob_given_true = self._calculate_conditional_probability_with_intervals(
            entity_id, entity_intervals, motion_intervals_by_sensor, STATE_ON
        )
        prob_given_false = self._calculate_conditional_probability_with_intervals(
            entity_id, entity_intervals, motion_intervals_by_sensor, STATE_OFF
        )

        # After computing the probabilities, update learned priors
        self.coordinator.update_learned_priors(
            entity_id,
            prob_given_true,
            prob_given_false,
            prior,
        )

        return prob_given_true, prob_given_false, prior

    async def _get_states_from_recorder(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> list[State] | None:
        """Fetch states history from recorder."""
        try:
            states = await get_instance(self.hass).async_add_executor_job(
                lambda: get_significant_states(
                    self.hass,
                    start_time,
                    end_time,
                    [entity_id],
                    minimal_response=False,
                )
            )
            return states.get(entity_id) if states else None
        except (HomeAssistantError, SQLAlchemyError) as err:
            _LOGGER.error("Error getting states for %s: %s", entity_id, err)
            return None

    def _states_to_intervals(
        self, states: list[State], start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime, Any]]:
        """Convert a list of states into intervals [(start, end, state), ...]."""
        intervals = []
        sorted_states = sorted(states, key=lambda s: s.last_changed)
        current_start = start
        current_state = None

        for s in sorted_states:
            s_start = s.last_changed
            if s_start < start:
                s_start = start
            if current_state is None:
                current_state = s.state
                current_start = s_start
            else:
                if s_start > current_start:
                    intervals.append((current_start, s_start, current_state))
                current_state = s.state
                current_start = s_start

        if current_start < end:
            intervals.append((current_start, end, current_state))

        return intervals

    def _compute_state_durations_from_intervals(
        self, intervals_dict: dict[str, list[tuple[datetime, datetime, Any]]]
    ) -> dict[str, float]:
        """Compute total durations for each state from precomputed intervals."""
        _LOGGER.debug("Computing state durations from intervals")
        durations = {}
        for intervals in intervals_dict.values():
            for interval_start, interval_end, state_val in intervals:
                duration = (interval_end - interval_start).total_seconds()
                durations[state_val] = durations.get(state_val, 0.0) + duration
        return durations

    def _calculate_conditional_probability_with_intervals(
        self,
        entity_id: str,
        entity_intervals: list[tuple[datetime, datetime, Any]],
        motion_intervals_by_sensor: dict[str, list[tuple[datetime, datetime, Any]]],
        motion_state_filter: str,
    ) -> float:
        """Calculate P(entity_active | motion_state) using precomputed intervals."""
        _LOGGER.debug("Calculating conditional probability for %s", entity_id)
        # Combine motion intervals for the specified motion state
        motion_intervals = []
        for intervals in motion_intervals_by_sensor.values():
            motion_intervals.extend(
                (start, end)
                for start, end, state in intervals
                if state == motion_state_filter
            )

        total_motion_duration = sum(
            (end - start).total_seconds() for start, end in motion_intervals
        )
        if total_motion_duration == 0:
            return (
                DEFAULT_PROB_GIVEN_TRUE
                if motion_state_filter == STATE_ON
                else DEFAULT_PROB_GIVEN_FALSE
            )

        # Get entity active intervals
        entity_active_intervals = [
            (start, end)
            for start, end, state in entity_intervals
            if is_entity_active(
                entity_id,
                state,
                self.entity_types,
                self.probabilities.sensor_configs,
            )
        ]

        # Calculate the overlap duration
        overlap_duration = 0.0
        for e_start, e_end in entity_active_intervals:
            for m_start, m_end in motion_intervals:
                overlap_start = max(e_start, m_start)
                overlap_end = min(e_end, m_end)
                if overlap_start < overlap_end:
                    overlap_duration += (overlap_end - overlap_start).total_seconds()

        # Clamp the final probability before returning
        result = overlap_duration / total_motion_duration
        return max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))

    def _map_entities_to_types(self) -> None:
        """Create mapping of entity IDs to their sensor types."""
        mappings = [
            (CONF_MOTION_SENSORS, "motion"),
            (CONF_MEDIA_DEVICES, "media"),
            (CONF_APPLIANCES, "appliance"),
            (CONF_DOOR_SENSORS, "door"),
            (CONF_WINDOW_SENSORS, "window"),
            (CONF_LIGHTS, "light"),
            (CONF_ILLUMINANCE_SENSORS, "environmental"),
            (CONF_HUMIDITY_SENSORS, "environmental"),
            (CONF_TEMPERATURE_SENSORS, "environmental"),
        ]

        for config_key, sensor_type in mappings:
            for entity_id in self.config.get(config_key, []):
                self.entity_types[entity_id] = sensor_type


def get_default_prior(entity_id: str, calc) -> float:
    """Return default prior based on entity category."""
    if entity_id in calc.motion_sensors:
        return MOTION_DEFAULT_PRIOR
    elif entity_id in calc.media_devices:
        return MEDIA_DEFAULT_PRIOR
    elif entity_id in calc.appliances:
        return APPLIANCE_DEFAULT_PRIOR
    elif entity_id in calc.door_sensors:
        return DOOR_DEFAULT_PRIOR
    elif entity_id in calc.window_sensors:
        return WINDOW_DEFAULT_PRIOR
    elif entity_id in calc.lights:
        return LIGHT_DEFAULT_PRIOR
    elif (
        entity_id in calc.illuminance_sensors
        or entity_id in calc.humidity_sensors
        or entity_id in calc.temperature_sensors
    ):
        return ENVIRONMENTAL_DEFAULT_PRIOR
    return DEFAULT_PRIOR
