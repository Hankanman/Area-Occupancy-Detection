"""Prior probability calculations for Area Occupancy Detection.

This module handles the calculation of prior probabilities for area occupancy detection
based on historical sensor data. It uses Bayesian probability calculations to determine
the likelihood of occupancy based on various sensor states.
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta
import logging

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from .probabilities import Probabilities  # Import Probabilities
from .types import SensorInputs, TimeInterval  # Import SensorInputs

_LOGGER = logging.getLogger(__name__)


class PriorCalculator:
    """Calculate occupancy probability based on sensor states.

    This class handles the calculation of prior probabilities for area occupancy
    based on historical sensor data. It uses Bayesian probability calculations
    to determine the likelihood of occupancy.

    Attributes:
        hass: Home Assistant instance
        probabilities: Probability configuration object
        sensor_inputs: SensorInputs object containing configured sensors

    """

    def __init__(
        self,
        hass: HomeAssistant,
        probabilities: Probabilities,
        sensor_inputs: SensorInputs,
    ) -> None:
        """Initialize the calculator.

        Args:
            hass: Home Assistant instance
            probabilities: Probability configuration object
            sensor_inputs: SensorInputs object containing configured sensors

        """
        self.hass = hass
        self.probabilities = probabilities
        self.inputs = sensor_inputs
        # Coordinator reference is removed

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
                "Fetching states: %s [%s -> %s]",
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

            entity_states = states.get(entity_id) if states else None

            if entity_states:
                _LOGGER.debug(
                    "Found %d states for %s",
                    len(entity_states),
                    entity_id,
                )
            else:
                _LOGGER.debug("No states found for %s", entity_id)

            return entity_states

        except (HomeAssistantError, SQLAlchemyError) as err:
            _LOGGER.error("Error getting states for %s: %s", entity_id, err)
            # Propagate error to be handled by the coordinator
            raise HomeAssistantError(f"Recorder error for {entity_id}: {err}") from err

    async def calculate_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> tuple[float, float, float] | None:
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
            Tuple containing (prob_given_true, prob_given_false, prior) if successful,
            None otherwise.

        Raises:
            HomeAssistantError: If unable to access historical data
            ValueError: If entity ID format is invalid or time range is invalid

        """
        if not self.inputs.is_valid_entity_id(entity_id):
            _LOGGER.error(f"Invalid entity ID format provided: {entity_id}")
            return None  # Return None for invalid entity ID

        if end_time <= start_time:
            _LOGGER.error("End time must be after start time")
            return None  # Return None for invalid time range

        _LOGGER.debug(
            "Calculating prior for entity: %s",
            entity_id,
        )

        # Use default priors as fallback values
        default_prior = self.probabilities.get_default_prior(entity_id)

        is_primary = entity_id == self.inputs.primary_sensor

        try:
            if is_primary:
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
        except HomeAssistantError as err:
            _LOGGER.warning(
                "Could not fetch states for prior calculation (%s): %s. Using defaults",
                entity_id,
                err,
            )
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                default_prior,
            )

        if not primary_states or (not is_primary and not entity_states):
            _LOGGER.warning(
                "No historical sensor data available for %s in the given period. Using defaults",
                entity_id,
            )
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                default_prior,
            )

        # For primary sensor, calculate based on its own active time
        if is_primary:
            prob_given_true = 0.9  # Higher confidence for primary
            prob_given_false = 0.1  # Lower false positive for primary

            intervals = await self._states_to_intervals(
                primary_states, start_time, end_time
            )
            durations = self._compute_state_durations_from_intervals(
                {entity_id: intervals}
            )
            total_active_time = durations.get(STATE_ON, 0.0)
            total_time = total_active_time + durations.get(STATE_OFF, 0.0)

            if total_time > 0:
                prior = max(
                    MIN_PROBABILITY,
                    min(total_active_time / total_time, MAX_PROBABILITY),
                )
            else:
                prior = default_prior

            return float(prob_given_true), float(prob_given_false), float(prior)

        # ---- Calculation for non-primary sensors ----

        # Compute intervals for primary sensor
        primary_intervals = await self._states_to_intervals(
            primary_states, start_time, end_time
        )
        if not primary_intervals:
            _LOGGER.warning(
                "No valid intervals found for primary sensor. Using defaults for %s",
                entity_id,
            )
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                default_prior,
            )

        # Compute intervals for the entity
        entity_intervals = await self._states_to_intervals(
            entity_states, start_time, end_time
        )
        if not entity_intervals:
            _LOGGER.warning(
                "No valid intervals found for entity %s. Using defaults", entity_id
            )
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                default_prior,
            )

        # Calculate prior based on this sensor's active time ratio
        entity_durations = self._compute_state_durations_from_intervals(
            {entity_id: entity_intervals}
        )
        total_entity_active_time = entity_durations.get(STATE_ON, 0.0)
        total_entity_inactive_time = entity_durations.get(STATE_OFF, 0.0)
        total_entity_time = total_entity_active_time + total_entity_inactive_time

        if total_entity_time > 0:
            prior = max(
                MIN_PROBABILITY,
                min(total_entity_active_time / total_entity_time, MAX_PROBABILITY),
            )
        else:
            prior = default_prior

        # Calculate conditional probabilities using intervals
        try:
            prob_given_true = self._calculate_conditional_probability_with_intervals(
                entity_id,
                entity_intervals,
                {self.inputs.primary_sensor: primary_intervals},
                STATE_ON,
            )
            prob_given_false = self._calculate_conditional_probability_with_intervals(
                entity_id,
                entity_intervals,
                {self.inputs.primary_sensor: primary_intervals},
                STATE_OFF,
            )
        except ValueError as err:
            _LOGGER.error(
                "Error calculating conditional probability for %s: %s. Using defaults",
                entity_id,
                err,
            )
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                default_prior,
            )

        # Return calculated values
        return float(prob_given_true), float(prob_given_false), float(prior)

    async def _states_to_intervals(
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
        if not states:  # Handle empty states list
            return intervals

        sorted_states = sorted(states, key=lambda s: s.last_changed)
        current_start = start
        # Determine initial state before the first recorded state in the window
        first_state_in_window = sorted_states[0]
        initial_state_obj = None
        if first_state_in_window.last_changed > start:
            # Get state just before the window starts
            try:
                # Fetch states in a small window just before the main start time
                prior_states = await get_instance(
                    self.hass
                ).async_add_executor_job(
                    lambda: get_significant_states(
                        self.hass,
                        start - timedelta(seconds=1),  # Small window before start
                        start,
                        [first_state_in_window.entity_id],  # Only for this entity
                        minimal_response=True,  # We only need the state
                        significant_changes_only=False,  # Get the last state regardless of change
                    )
                )
                entity_prior_states = prior_states.get(first_state_in_window.entity_id)
                if entity_prior_states:
                    initial_state_obj = entity_prior_states[
                        0
                    ]  # Get the most recent state before 'start'

            except Exception:
                _LOGGER.warning(
                    "Could not fetch state before window for %s",
                    first_state_in_window.entity_id,
                    exc_info=True,
                )

        current_state = initial_state_obj.state if initial_state_obj else None
        # If no state before window, assume the state of the first item *was* the state at window start
        if current_state is None:
            current_state = first_state_in_window.state

        for s in sorted_states:
            # Clamp state change time to be within the [start, end] window
            s_change_time = max(start, s.last_changed)
            s_change_time = min(
                end, s_change_time
            )  # Ensure change time doesn't exceed end

            # If the state change happens after the current interval start
            if s_change_time > current_start:
                # Add the previous interval, ensuring its end doesn't exceed the overall end time
                interval_end = min(s_change_time, end)
                if (
                    current_state is not None and interval_end > current_start
                ):  # Ensure valid interval
                    intervals.append(
                        {
                            "start": current_start,
                            "end": interval_end,
                            "state": current_state,
                        }
                    )
                # Start the new interval from the state change time
                current_start = s_change_time
                current_state = s.state
            # Handle cases where multiple state changes might occur at the exact same microsecond
            # or if the first state change is exactly at 'start'
            elif s_change_time == current_start:
                current_state = (
                    s.state
                )  # Update state but don't create a zero-duration interval

        # Add the final interval from the last state change to the end time
        if current_start < end and current_state is not None:
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
            STATE_ON: 0.0,
            STATE_OFF: 0.0,
        }
        for intervals in intervals_dict.values():
            for interval in intervals:
                duration = (interval["end"] - interval["start"]).total_seconds()
                state = interval.get("state")
                if state is not None:
                    # Normalize common 'on' states to STATE_ON for duration calculation
                    # This needs context from the probability config (which state is active)
                    # For simplicity here, we'll map known binary states directly
                    # A more robust solution might involve passing active_states config
                    normalized_state = STATE_ON if state == STATE_ON else STATE_OFF
                    durations[normalized_state] = (
                        durations.get(normalized_state, 0.0) + duration
                    )
        return durations

    def _calculate_conditional_probability_with_intervals(
        self,
        entity_id: str,
        entity_intervals: list[TimeInterval],
        motion_intervals_by_sensor: dict[str, list[TimeInterval]],
        motion_state_filter: str,  # Should be STATE_ON or STATE_OFF
    ) -> float:
        """Calculate P(entity_active | primary_sensor_state) using precomputed intervals."""
        _LOGGER.debug(
            "Calculating conditional prob P( %s active | primary = %s )",
            entity_id,
            motion_state_filter,
        )

        # Combine intervals for the primary sensor matching the filter state
        primary_intervals_filtered = []
        primary_sensor_id = self.inputs.primary_sensor
        if primary_sensor_id in motion_intervals_by_sensor:
            primary_intervals_filtered.extend(
                {"start": interval["start"], "end": interval["end"]}
                for interval in motion_intervals_by_sensor[primary_sensor_id]
                if interval.get("state") == motion_state_filter
            )

        total_primary_duration_filtered = sum(
            (interval["end"] - interval["start"]).total_seconds()
            for interval in primary_intervals_filtered
        )

        if total_primary_duration_filtered == 0:
            _LOGGER.debug(
                "Primary sensor total duration for state '%s' is zero",
                motion_state_filter,
            )
            # Return default based on the state we are conditioning on
            return (
                DEFAULT_PROB_GIVEN_TRUE
                if motion_state_filter == STATE_ON
                else DEFAULT_PROB_GIVEN_FALSE
            )

        # Get entity intervals considered 'active' based on probabilities config
        entity_active_intervals = [
            {"start": interval["start"], "end": interval["end"]}
            for interval in entity_intervals
            if self.probabilities.is_entity_active(
                entity_id,
                interval.get("state"),
            )
        ]

        # Calculate the total duration of overlap between entity_active and primary_filtered intervals
        overlap_duration = 0.0
        for e_interval in entity_active_intervals:
            for p_interval in primary_intervals_filtered:
                overlap_start = max(e_interval["start"], p_interval["start"])
                overlap_end = min(e_interval["end"], p_interval["end"])
                if overlap_start < overlap_end:
                    overlap_duration += (overlap_end - overlap_start).total_seconds()

        # Calculate conditional probability P(Entity Active | Primary State)
        result = overlap_duration / total_primary_duration_filtered
        clamped_result = max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))

        _LOGGER.debug(
            "Conditional prob result for %s (primary=%s): overlap=%.2fs / total=%.2fs = %.3f -> %.3f",
            entity_id,
            motion_state_filter,
            overlap_duration,
            total_primary_duration_filtered,
            result,
            clamped_result,
        )

        return clamped_result
