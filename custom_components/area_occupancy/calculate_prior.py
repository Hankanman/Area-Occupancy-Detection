"""Prior probability calculations for Area Occupancy Detection.

This module handles the calculation of prior probabilities for area occupancy detection
based on historical sensor data. It uses Bayesian probability calculations to determine
the likelihood of occupancy based on various sensor states.
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime
import logging
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import (
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from .types import TimeInterval

_LOGGER = logging.getLogger(__name__)


class PriorCalculator:
    """Calculate occupancy probability based on sensor states.

    This class handles the calculation of prior probabilities for area occupancy
    based on historical sensor data. It uses Bayesian probability calculations
    to determine the likelihood of occupancy.

    Attributes:
        coordinator: The coordinator instance managing this calculator
        config: Configuration dictionary containing sensor settings
        probabilities: Probability configuration object
        hass: Home Assistant instance

    """

    def __init__(
        self, coordinator: Any, probabilities: Any, hass: HomeAssistant
    ) -> None:
        """Initialize the calculator.

        Args:
            coordinator: The coordinator instance managing this calculator
            probabilities: Probability configuration object
            hass: Home Assistant instance

        """
        self.coordinator = coordinator
        self.config = coordinator.config
        self.probabilities = probabilities
        self.hass = hass

        # Match cache duration to update interval and track last clear
        self._cache_duration = coordinator.prior_update_interval
        self._last_cache_clear = dt_util.utcnow()
        self._cache = {}

        # Use sensor inputs from coordinator
        self.inputs = coordinator.inputs

    def _should_clear_cache(self) -> bool:
        """Check if cache should be cleared based on next update time."""
        # Clear cache if we've passed the next scheduled update time
        if (
            self.coordinator.next_prior_update
            and dt_util.utcnow() >= self.coordinator.next_prior_update
        ):
            self._last_cache_clear = dt_util.utcnow()
            self._cache.clear()
            return True
        return False

    async def _get_states_from_recorder(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> list[State] | None:
        """Fetch states history from recorder.

        Args:
            entity_id: Entity ID to fetch history for
            start_time: Start time window
            end_time: End time window

        Returns:
            List of states if successful, None if error occurred

        Raises:
            HomeAssistantError: If recorder access fails
            SQLAlchemyError: If database query fails

        """
        try:
            _LOGGER.debug(
                "States: %s [%s -> %s]",
                entity_id,
                start_time,
                end_time,
            )

            states = await get_instance(
                self.hass
            ).async_add_executor_job(
                lambda: get_significant_states(
                    self.hass,
                    start_time,
                    end_time,
                    [entity_id],
                    minimal_response=False,  # Must be false to include last_changed attribute
                )
            )

            if states:
                _LOGGER.debug(
                    "Found %d states: %s",
                    len(states.get(entity_id, [])),
                    entity_id,
                )
            else:
                _LOGGER.debug("No states: %s", entity_id)

            return states.get(entity_id) if states else None

        except (HomeAssistantError, SQLAlchemyError) as err:
            _LOGGER.error("Error getting states for %s: %s", entity_id, err)
            return None

    async def calculate_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> tuple[float, float, float]:
        """Calculate learned priors for a given entity.

        This method analyzes historical sensor data to calculate probability values
        for a given entity. It uses the primary occupancy sensor as ground truth
        and calculates conditional probabilities based on the entity's state
        correlation with the primary sensor.

        Args:
            entity_id: The entity ID to calculate priors for
            start_time: Start time for historical data analysis
            end_time: End time for historical data analysis

        Returns:
            Tuple containing:
                - prob_given_true: P(entity active | area occupied)
                - prob_given_false: P(entity active | area not occupied)
                - prior: Base probability of area being occupied

        Raises:
            HomeAssistantError: If unable to access historical data
            ValueError: If entity ID format is invalid or time range is invalid

        """
        if not self.inputs.is_valid_entity_id(entity_id):
            raise ValueError(f"Invalid entity ID format: {entity_id}")

        if end_time <= start_time:
            raise ValueError("End time must be after start time")

        _LOGGER.debug(
            "Prior calc for instance %s: %s",
            self.coordinator.config_entry.entry_id,
            entity_id,
        )

        # If this is the primary sensor, use its own state for calculations
        is_primary = entity_id == self.inputs.primary_sensor

        # Get states for both sensors in parallel for better performance
        try:
            if is_primary:
                # For primary sensor, use its own state as ground truth
                primary_states = await self._get_states_from_recorder(
                    entity_id, start_time, end_time
                )
                entity_states = primary_states
            else:
                primary_states, entity_states = await asyncio.gather(
                    self._get_states_from_recorder(
                        self.inputs.primary_sensor, start_time, end_time
                    ),
                    self._get_states_from_recorder(entity_id, start_time, end_time),
                )
        except (HomeAssistantError, SQLAlchemyError, RuntimeError) as err:
            _LOGGER.error(
                "Error fetching states for instance %s: %s",
                self.coordinator.config_entry.entry_id,
                err,
            )
            return (
                float(DEFAULT_PROB_GIVEN_TRUE),
                float(DEFAULT_PROB_GIVEN_FALSE),
                float(self.probabilities.get_default_prior(entity_id)),
            )

        if not primary_states or not entity_states:
            _LOGGER.warning(
                "No sensor data available for instance %s",
                self.coordinator.config_entry.entry_id,
            )
            return (
                float(DEFAULT_PROB_GIVEN_TRUE),
                float(DEFAULT_PROB_GIVEN_FALSE),
                float(self.probabilities.get_default_prior(entity_id)),
            )

        # For primary sensor, use higher default probabilities since it's the ground truth
        if is_primary:
            prob_given_true = 0.9  # High confidence when sensor indicates occupancy
            prob_given_false = 0.1  # Low probability of false positives

            # Calculate prior based on total active time
            intervals = self._states_to_intervals(primary_states, start_time, end_time)
            durations = self._compute_state_durations_from_intervals(
                {entity_id: intervals}
            )
            total_active_time = float(durations.get(STATE_ON, 0.0))
            total_time = total_active_time + float(durations.get(STATE_OFF, 0.0))

            if total_time > 0:
                prior = max(
                    MIN_PROBABILITY,
                    min(total_active_time / total_time, MAX_PROBABILITY),
                )
            else:
                prior = float(self.probabilities.get_default_prior(entity_id))

            return prob_given_true, prob_given_false, prior

        # Compute intervals for primary sensor
        primary_intervals = self._states_to_intervals(
            primary_states, start_time, end_time
        )
        if not primary_intervals:
            _LOGGER.warning("No valid intervals found for primary sensor")
            return (
                float(DEFAULT_PROB_GIVEN_TRUE),
                float(DEFAULT_PROB_GIVEN_FALSE),
                float(self.probabilities.get_default_prior(entity_id)),
            )

        # Compute intervals for the entity
        entity_intervals = self._states_to_intervals(
            entity_states, start_time, end_time
        )
        if not entity_intervals:
            _LOGGER.warning("No valid intervals found for entity %s", entity_id)
            return (
                float(DEFAULT_PROB_GIVEN_TRUE),
                float(DEFAULT_PROB_GIVEN_FALSE),
                float(self.probabilities.get_default_prior(entity_id)),
            )

        # Calculate prior probability based on primary sensor
        primary_durations = self._compute_state_durations_from_intervals(
            {self.inputs.primary_sensor: primary_intervals}
        )
        total_primary_active_time = float(primary_durations.get(STATE_ON, 0.0))
        total_primary_inactive_time = float(primary_durations.get(STATE_OFF, 0.0))
        total_primary_time = total_primary_active_time + total_primary_inactive_time

        if total_primary_time == 0:
            _LOGGER.warning("No valid duration found for primary sensor")
            return (
                float(DEFAULT_PROB_GIVEN_TRUE),
                float(DEFAULT_PROB_GIVEN_FALSE),
                float(self.probabilities.get_default_prior(entity_id)),
            )

        # Calculate prior probability based on this sensor's own active time
        entity_durations = self._compute_state_durations_from_intervals(
            {entity_id: entity_intervals}
        )
        total_entity_active_time = float(entity_durations.get(STATE_ON, 0.0))
        total_entity_time = total_entity_active_time + float(
            entity_durations.get(STATE_OFF, 0.0)
        )

        # Calculate prior based on this sensor's active time ratio
        if total_entity_time > 0:
            prior = float(
                max(
                    MIN_PROBABILITY,
                    min(total_entity_active_time / total_entity_time, MAX_PROBABILITY),
                )
            )
        else:
            prior = float(self.probabilities.get_default_prior(entity_id))

        # Calculate conditional probabilities using intervals
        prob_given_true = float(
            self._calculate_conditional_probability_with_intervals(
                entity_id,
                entity_intervals,
                {self.inputs.primary_sensor: primary_intervals},
                STATE_ON,
            )
        )
        prob_given_false = float(
            self._calculate_conditional_probability_with_intervals(
                entity_id,
                entity_intervals,
                {self.inputs.primary_sensor: primary_intervals},
                STATE_OFF,
            )
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
        await self._update_type_priors(entity_id)

        return (prob_given_true, prob_given_false, prior)

    def _states_to_intervals(
        self, states: Sequence[State], start: datetime, end: datetime
    ) -> list[TimeInterval]:
        """Convert a list of states into time intervals.

        Args:
            states: Sequence of states to convert
            start: Start time boundary
            end: End time boundary

        Returns:
            List of TimeInterval objects containing:
                - start: Interval start time
                - end: Interval end time
                - state: State during interval

        Raises:
            ValueError: If end time is before start time

        """
        if end < start:
            raise ValueError("End time must be after start time")

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
    ) -> dict[str, float]:
        """Compute total durations for each state from precomputed intervals.

        Args:
            intervals_dict: Dictionary mapping entity IDs to their time intervals

        Returns:
            Dictionary containing:
                - total_active_time: Total time in active state
                - total_inactive_time: Total time in inactive state
                - total_time: Total duration analyzed
                - Additional keys for specific states encountered

        """
        _LOGGER.debug("Computing durations")
        durations: dict[str, float] = {
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
        _LOGGER.debug("Conditional prob: %s", entity_id)
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

    async def _update_type_priors(self, entity_id: str) -> None:
        """Update type priors by averaging all sensors of the given type."""
        # Get all entities of this type
        entities = []
        if entity_id in self.inputs.motion_sensors:
            entities = self.inputs.motion_sensors
            sensor_type = "motion"
        elif entity_id in self.inputs.media_devices:
            entities = self.inputs.media_devices
            sensor_type = "media"
        elif entity_id in self.inputs.appliances:
            entities = self.inputs.appliances
            sensor_type = "appliance"
        elif entity_id in self.inputs.door_sensors:
            entities = self.inputs.door_sensors
            sensor_type = "door"
        elif entity_id in self.inputs.window_sensors:
            entities = self.inputs.window_sensors
            sensor_type = "window"
        elif entity_id in self.inputs.lights:
            entities = self.inputs.lights
            sensor_type = "light"
        else:
            _LOGGER.debug("Unknown sensor type for %s", entity_id)
            return

        if not entities:
            _LOGGER.debug("No entities of type %s", sensor_type)
            return

        # Collect all learned priors for this type
        priors = []
        prob_given_trues = []
        prob_given_falses = []

        for entity in entities:
            learned = self.coordinator.prior_state.entity_priors.get(entity)
            if learned:
                priors.append(learned["prior"])
                prob_given_trues.append(learned["prob_given_true"])
                prob_given_falses.append(learned["prob_given_false"])

        if not priors:
            _LOGGER.debug("No learned priors for type %s", sensor_type)
            return

        # Calculate averages for all probability values
        avg_prior = sum(priors) / len(priors)
        avg_prob_given_true = sum(prob_given_trues) / len(prob_given_trues)
        avg_prob_given_false = sum(prob_given_falses) / len(prob_given_falses)

        # Log the averages for debugging
        _LOGGER.debug(
            "Type %s averages - prior: %.3f, p_true: %.3f, p_false: %.3f (from %d sensors)",
            sensor_type,
            avg_prior,
            avg_prob_given_true,
            avg_prob_given_false,
            len(priors),
        )

        # Update type priors in prior_state with all averaged values
        timestamp = dt_util.utcnow().isoformat()
        self.coordinator.prior_state.update_type_prior(
            sensor_type,
            avg_prior,
            timestamp,
            avg_prob_given_true,
            avg_prob_given_false,
        )
