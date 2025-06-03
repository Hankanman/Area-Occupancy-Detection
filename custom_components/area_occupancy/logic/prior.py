"""Prior probability calculations for Area Occupancy Detection."""

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance
from sqlalchemy.exc import SQLAlchemyError

from ..const import (
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from ..models.feature import FeatureManager
from ..state.containers import PriorData

_LOGGER = logging.getLogger(__name__)


class TimeInterval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime
    state: str


async def get_prior_data(
    hass: HomeAssistant,
    feature_manager: FeatureManager,
    primary_sensor: str,
    entity_id: str,
    start_time: datetime,
    end_time: datetime,
) -> PriorData:
    """Calculate learned priors for a given entity.

    This function analyzes historical sensor data to calculate probability values
    for a given entity. It uses the primary occupancy sensor as ground truth
    and calculates conditional probabilities based on the entity's state
    correlation with the primary sensor.

    Args:
        hass: Home Assistant instance
        feature_manager: Feature manager instance
        primary_sensor: Primary occupancy sensor entity ID
        entity_id: The entity ID to calculate priors for
        start_time: Start time for historical data analysis
        end_time: End time for historical data analysis

    Returns:
        A PriorData dataclass if successful.

    Raises:
        HomeAssistantError: If unable to access historical data
        ValueError: If entity ID format is invalid or time range is invalid

    """
    entity_type = feature_manager.get_feature_type(entity_id)
    if not entity_type:
        _LOGGER.error("Invalid entity ID format provided: %s", entity_id)
        raise ValueError(f"Invalid entity ID format provided: {entity_id}")

    if end_time <= start_time:
        _LOGGER.error("End time must be after start time")
        raise ValueError("End time must be after start time")

    _LOGGER.debug(
        "Calculating prior for entity: %s",
        entity_id,
    )

    # Use default priors as fallback values
    default_prior = feature_manager.get_feature_type(entity_id)["prior"]
    fallback_prior: PriorData = PriorData(
        prob_given_true=DEFAULT_PROB_GIVEN_TRUE,
        prob_given_false=DEFAULT_PROB_GIVEN_FALSE,
        prior=default_prior,
    )

    is_primary = entity_id == primary_sensor

    try:
        if is_primary:
            primary_states = await _get_states_from_recorder(
                hass, entity_id, start_time, end_time
            )
            entity_states = primary_states
        else:
            try:
                primary_states, entity_states = await asyncio.gather(
                    _get_states_from_recorder(
                        hass, primary_sensor, start_time, end_time
                    ),
                    _get_states_from_recorder(hass, entity_id, start_time, end_time),
                )
            except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
                _LOGGER.warning(
                    "Could not fetch states for prior calculation (%s): %s. Using defaults",
                    entity_id,
                    err,
                )
                return fallback_prior
    except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
        _LOGGER.warning(
            "Could not fetch states for prior calculation (%s): %s. Using defaults",
            entity_id,
            err,
        )
        return fallback_prior

    if not primary_states or not entity_states:
        _LOGGER.warning(
            "No states found for prior calculation (%s). Using defaults",
            entity_id,
        )
        return fallback_prior

    # Ensure states are lists of State objects, filtering out dicts if necessary
    # Note: _get_states_from_recorder should return list[State] when minimal_response=False
    primary_state_objects = [
        state for state in primary_states if isinstance(state, State)
    ]
    entity_state_objects = [
        state for state in entity_states if isinstance(state, State)
    ]

    if not primary_state_objects or not entity_state_objects:
        _LOGGER.warning(
            "No valid states found for prior calculation (%s). Using defaults",
            entity_id,
        )
        return fallback_prior

    # Convert states to time intervals
    primary_intervals = await _states_to_intervals(
        primary_state_objects, start_time, end_time
    )
    entity_intervals = await _states_to_intervals(
        entity_state_objects, start_time, end_time
    )

    # Calculate conditional probabilities
    prob_given_true = _calculate_conditional_probability_with_intervals(
        entity_id,
        entity_intervals,
        {primary_sensor: primary_intervals},
        STATE_ON,
        feature_manager,
    )
    prob_given_false = _calculate_conditional_probability_with_intervals(
        entity_id,
        entity_intervals,
        {primary_sensor: primary_intervals},
        STATE_OFF,
        feature_manager,
    )

    # Calculate prior probability
    prior = _calculate_prior_probability(
        prob_given_true, prob_given_false, primary_intervals
    )

    return PriorData(
        prior=prior,
        prob_given_true=prob_given_true,
        prob_given_false=prob_given_false,
        last_updated=datetime.utcnow().isoformat(),
    )


async def _get_states_from_recorder(
    hass: HomeAssistant, entity_id: str, start_time: datetime, end_time: datetime
) -> List[State | Dict[str, Any]] | None:
    """Fetch states history from recorder.

    Args:
        hass: Home Assistant instance
        entity_id: Entity ID to fetch history for
        start_time: Start time window
        end_time: End time window

    Returns:
        List of states or minimal state dicts if successful, None if error occurred

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
            hass
        ).async_add_executor_job(
            lambda: get_significant_states(
                hass,
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

    except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
        _LOGGER.error("Error getting states for %s: %s", entity_id, err)
        # Propagate error to be handled by the coordinator
        raise HomeAssistantError(f"Recorder error for {entity_id}: {err}") from err
    else:
        return entity_states


async def _states_to_intervals(
    states: Sequence[State], start: datetime, end: datetime
) -> List[TimeInterval]:
    """Convert state history to time intervals.

    Args:
        states: List of State objects
        start: Start time for analysis
        end: End time for analysis

    Returns:
        List of TimeInterval objects

    """
    intervals: List[TimeInterval] = []
    if not states:
        return intervals

    # Sort states by last_changed
    sorted_states = sorted(states, key=lambda x: x.last_changed)

    # Create intervals from states
    for i, state in enumerate(sorted_states):
        interval_start = state.last_changed
        interval_end = (
            sorted_states[i + 1].last_changed if i < len(sorted_states) - 1 else end
        )
        intervals.append(
            TimeInterval(
                start=interval_start,
                end=interval_end,
                state=state.state,
            )
        )

    return intervals


def _calculate_conditional_probability_with_intervals(
    entity_id: str,
    entity_intervals: List[TimeInterval],
    motion_intervals_by_sensor: Dict[str, List[TimeInterval]],
    motion_state_filter: str,  # Should be STATE_ON or STATE_OFF
    feature_manager: FeatureManager,
) -> float:
    """Calculate conditional probability using time intervals.

    Args:
        entity_id: Entity ID being analyzed
        entity_intervals: List of time intervals for the entity
        motion_intervals_by_sensor: Dict of motion sensor intervals
        motion_state_filter: State to filter motion intervals by
        feature_manager: Feature manager instance

    Returns:
        Conditional probability value

    """
    # Get motion intervals for the primary sensor
    motion_intervals = motion_intervals_by_sensor.get(entity_id, [])

    # Filter motion intervals by state
    filtered_motion_intervals = [
        interval
        for interval in motion_intervals
        if interval["state"] == motion_state_filter
    ]

    if not filtered_motion_intervals:
        return (
            DEFAULT_PROB_GIVEN_TRUE
            if motion_state_filter == STATE_ON
            else DEFAULT_PROB_GIVEN_FALSE
        )

    # Calculate total duration of filtered motion intervals
    total_motion_duration = sum(
        (interval["end"] - interval["start"]).total_seconds()
        for interval in filtered_motion_intervals
    )

    if total_motion_duration == 0:
        return (
            DEFAULT_PROB_GIVEN_TRUE
            if motion_state_filter == STATE_ON
            else DEFAULT_PROB_GIVEN_FALSE
        )

    # Calculate entity state durations during filtered motion intervals
    entity_active_duration = 0
    entity_type = feature_manager.get_feature_type(entity_id)
    entity_active_states = feature_manager.get_entity_active_states(entity_id)
    if not entity_type:
        return (
            DEFAULT_PROB_GIVEN_TRUE
            if motion_state_filter == STATE_ON
            else DEFAULT_PROB_GIVEN_FALSE
        )

    for motion_interval in filtered_motion_intervals:
        for entity_interval in entity_intervals:
            # Calculate overlap between intervals
            overlap_start = max(motion_interval["start"], entity_interval["start"])
            overlap_end = min(motion_interval["end"], entity_interval["end"])
            if overlap_end > overlap_start:
                # Check if entity state is considered active
                if entity_interval["state"] in entity_active_states:
                    entity_active_duration += (
                        overlap_end - overlap_start
                    ).total_seconds()

    # Calculate conditional probability
    probability = entity_active_duration / total_motion_duration
    return max(MIN_PROBABILITY, min(probability, MAX_PROBABILITY))


def _calculate_prior_probability(
    prob_given_true: float,
    prob_given_false: float,
    primary_intervals: List[TimeInterval],
) -> float:
    """Calculate prior probability using Bayes' theorem.

    Args:
        prob_given_true: P(sensor active | area occupied)
        prob_given_false: P(sensor active | area not occupied)
        primary_intervals: List of time intervals for primary sensor

    Returns:
        Prior probability value

    """
    # Calculate total duration of occupied and unoccupied states
    occupied_duration = sum(
        (interval["end"] - interval["start"]).total_seconds()
        for interval in primary_intervals
        if interval["state"] == STATE_ON
    )
    total_duration = sum(
        (interval["end"] - interval["start"]).total_seconds()
        for interval in primary_intervals
    )

    if total_duration == 0:
        return 0.5  # Default to 0.5 if no data

    # Calculate P(area occupied)
    p_occupied = occupied_duration / total_duration

    # Calculate prior using Bayes' theorem
    prior = (prob_given_true * p_occupied) / (
        (prob_given_true * p_occupied) + (prob_given_false * (1 - p_occupied))
    )

    return max(MIN_PROBABILITY, min(prior, MAX_PROBABILITY))
