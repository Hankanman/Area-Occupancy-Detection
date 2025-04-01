"""Prior probability calculations for Area Occupancy Detection."""

from collections.abc import Sequence
from datetime import datetime
import logging

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import (
    CONF_APPLIANCES,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_WINDOW_SENSORS,
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from .types import ProbabilityConfig, TimeInterval

_LOGGER = logging.getLogger(__name__)


class PriorCalculator:
    """Calculate occupancy probability based on sensor states."""

    def __init__(self, coordinator, probabilities, hass: HomeAssistant) -> None:
        """Initialize the calculator."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self.probabilities = probabilities
        self.hass = hass

        self.motion_sensors = self.config.get(CONF_MOTION_SENSORS, [])
        self.primary_sensor = self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
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
    ) -> ProbabilityConfig:
        """Calculate learned priors for a given entity."""
        _LOGGER.debug("Calculating prior for %s", entity_id)

        # Get the sensor type for this entity
        sensor_type = self.probabilities.entity_types.get(entity_id)
        if not sensor_type:
            _LOGGER.warning("No sensor type found for entity %s", entity_id)
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                self.probabilities.get_default_prior(entity_id),
            )

        # Fetch primary occupancy sensor states
        primary_states = await self._get_states_from_recorder(
            self.primary_sensor, start_time, end_time
        )
        if not primary_states:
            _LOGGER.warning("No primary occupancy sensor data available")
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                self.probabilities.get_default_prior(entity_id),
            )

        # Fetch entity states
        entity_states = await self._get_states_from_recorder(
            entity_id, start_time, end_time
        )
        if not entity_states:
            _LOGGER.warning("No entity data available for %s", entity_id)
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                self.probabilities.get_default_prior(entity_id),
            )

        # Compute intervals for primary sensor
        primary_intervals = self._states_to_intervals(
            primary_states, start_time, end_time
        )
        if not primary_intervals:
            _LOGGER.warning("No valid intervals found for primary sensor")
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                self.probabilities.get_default_prior(entity_id),
            )

        # Compute intervals for the entity
        entity_intervals = self._states_to_intervals(
            entity_states, start_time, end_time
        )
        if not entity_intervals:
            _LOGGER.warning("No valid intervals found for entity %s", entity_id)
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                self.probabilities.get_default_prior(entity_id),
            )

        # Calculate prior probability based on primary sensor
        primary_durations = self._compute_state_durations_from_intervals(
            {self.primary_sensor: primary_intervals}
        )
        total_primary_active_time = primary_durations.get(STATE_ON, 0.0)
        total_primary_inactive_time = primary_durations.get(STATE_OFF, 0.0)
        total_primary_time = total_primary_active_time + total_primary_inactive_time

        if total_primary_time == 0:
            _LOGGER.warning("No valid duration found for primary sensor")
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                self.probabilities.get_default_prior(entity_id),
            )

        # Calculate prior probability based on primary sensor and clamp it
        prior = max(
            MIN_PROBABILITY,
            min(total_primary_active_time / total_primary_time, MAX_PROBABILITY),
        )

        # Calculate conditional probabilities using intervals
        prob_given_true = self._calculate_conditional_probability_with_intervals(
            entity_id,
            entity_intervals,
            {self.primary_sensor: primary_intervals},
            STATE_ON,
        )
        prob_given_false = self._calculate_conditional_probability_with_intervals(
            entity_id,
            entity_intervals,
            {self.primary_sensor: primary_intervals},
            STATE_OFF,
        )

        # After computing the probabilities, update learned priors in prior_state
        timestamp = dt_util.utcnow().isoformat()
        self.coordinator.prior_state.update_entity_prior(
            entity_id,
            prob_given_true,
            prob_given_false,
            prior,
            timestamp,
        )

        # Calculate and update type priors by averaging all sensors of this type
        await self._update_type_priors(sensor_type)

        return ProbabilityConfig(
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            prior=prior,
        )

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
        self, states: Sequence[State], start: datetime, end: datetime
    ) -> list[TimeInterval]:
        """Convert a list of states into intervals."""
        intervals: list[TimeInterval] = []
        sorted_states = sorted(states, key=lambda s: s.last_changed)
        current_start = start
        current_state = None

        for s in sorted_states:
            s_start = s.last_changed
            s_start = max(s_start, start)
            if current_state is None:
                current_state = s.state
                current_start = s_start
            else:
                if s_start > current_start:
                    intervals.append(
                        {"start": current_start, "end": s_start, "state": current_state}
                    )
                current_state = s.state
                current_start = s_start

        if current_start < end:
            intervals.append(
                {"start": current_start, "end": end, "state": current_state}
            )

        return intervals

    def _compute_state_durations_from_intervals(
        self, intervals_dict: dict[str, list[TimeInterval]]
    ) -> TimeInterval:
        """Compute total durations for each state from precomputed intervals."""
        _LOGGER.debug("Computing state durations from intervals")
        durations: TimeInterval = {
            "total_active_time": 0.0,
            "total_inactive_time": 0.0,
            "total_time": 0.0,
            "start": min(
                (
                    interval["start"]
                    for intervals in intervals_dict.values()
                    for interval in intervals
                ),
                default=dt_util.utcnow(),
            ),
            "end": max(
                (
                    interval["end"]
                    for intervals in intervals_dict.values()
                    for interval in intervals
                ),
                default=dt_util.utcnow(),
            ),
        }
        for intervals in intervals_dict.values():
            for interval in intervals:
                duration = (interval["end"] - interval["start"]).total_seconds()
                if "state" in interval:
                    durations[interval["state"]] = (
                        durations.get(interval["state"], 0.0) + duration
                    )
        return durations

    def _calculate_conditional_probability_with_intervals(
        self,
        entity_id: str,
        entity_intervals: list[TimeInterval],
        motion_intervals_by_sensor: dict[str, list[TimeInterval]],
        motion_state_filter: str,
    ) -> float:
        """Calculate P(entity_active | motion_state) using precomputed intervals."""
        _LOGGER.debug("Calculating conditional probability for %s", entity_id)
        # Combine motion intervals for the specified motion state
        motion_intervals = []
        for intervals in motion_intervals_by_sensor.values():
            motion_intervals.extend(
                {"start": interval["start"], "end": interval["end"]}
                for interval in intervals
                if interval.get("state") == motion_state_filter
            )

        total_motion_duration = sum(
            (interval["end"] - interval["start"]).total_seconds()
            for interval in motion_intervals
        )
        if total_motion_duration == 0:
            return (
                DEFAULT_PROB_GIVEN_TRUE
                if motion_state_filter == STATE_ON
                else DEFAULT_PROB_GIVEN_FALSE
            )

        # Get entity active intervals
        entity_active_intervals = [
            {"start": interval["start"], "end": interval["end"]}
            for interval in entity_intervals
            if self.probabilities.is_entity_active(
                entity_id,
                interval.get("state", ""),
            )
        ]

        # Calculate the overlap duration
        overlap_duration = 0.0
        for e_interval in entity_active_intervals:
            for m_interval in motion_intervals:
                overlap_start = max(e_interval["start"], m_interval["start"])
                overlap_end = min(e_interval["end"], m_interval["end"])
                if overlap_start < overlap_end:
                    overlap_duration += (overlap_end - overlap_start).total_seconds()

        # Clamp the final probability before returning
        result = overlap_duration / total_motion_duration
        return max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))

    async def _update_type_priors(self, sensor_type: str) -> None:
        """Update type priors by averaging all sensors of the given type."""
        # Get all entities of this type
        entities = []
        if sensor_type == "motion":
            entities = self.motion_sensors
        elif sensor_type == "media":
            entities = self.media_devices
        elif sensor_type == "appliance":
            entities = self.appliances
        elif sensor_type == "door":
            entities = self.door_sensors
        elif sensor_type == "window":
            entities = self.window_sensors
        elif sensor_type == "light":
            entities = self.lights

        if not entities:
            _LOGGER.debug("No entities found for type %s", sensor_type)
            return

        # Collect all learned priors for this type
        priors = []
        prob_given_trues = []
        prob_given_falses = []

        for entity_id in entities:
            learned = self.coordinator.prior_state.entity_priors.get(entity_id)
            if learned:
                priors.append(learned["prior"])
                prob_given_trues.append(learned["prob_given_true"])
                prob_given_falses.append(learned["prob_given_false"])

        if not priors:
            _LOGGER.debug("No learned priors found for type %s", sensor_type)
            return

        # Calculate averages
        avg_prior = sum(priors) / len(priors)

        # Update type priors in prior_state
        timestamp = dt_util.utcnow().isoformat()
        self.coordinator.prior_state.update_type_prior(
            sensor_type, avg_prior, timestamp
        )

        # Recalculate overall prior
        overall_prior = self.coordinator.prior_state.calculate_overall_prior()
        self.coordinator.prior_state.update(overall_prior=overall_prior)
