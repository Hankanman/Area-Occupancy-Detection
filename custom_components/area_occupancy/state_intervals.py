"""Database schema definitions using SQLAlchemy Core."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from logging import getLogger
from typing import Any, TypedDict

from sqlalchemy import MetaData
from sqlalchemy.exc import SQLAlchemyError

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance

_LOGGER = getLogger(__name__)

# ─────────────────── Constants ───────────────────

# Database metadata
metadata = MetaData()

# Database schema version for migrations
DB_VERSION = 2

# Interval filtering thresholds to exclude anomalous data
# Exclude intervals shorter than 5 seconds (false triggers)
MIN_INTERVAL_SECONDS = 5
# Exclude intervals longer than 13 hours (stuck sensors)
MAX_INTERVAL_SECONDS = 13 * 3600
# States to exclude from intervals
INVALID_STATES = {"unknown", "unavailable", None, "", "NaN"}

# ───────────────────────────────────── Types ───────────────────────────────────────


class Interval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime


class StateInterval(Interval):
    """Time interval with time information."""

    state: str
    entity_id: str


# ─────────────────── Utils ───────────────────


def is_valid_state(state: Any) -> bool:
    """Check if a state is valid."""
    return state not in INVALID_STATES


async def get_intervals_from_recorder(
    hass: HomeAssistant, entity_id: str, start_time: datetime, end_time: datetime
) -> list[StateInterval]:
    """Get intervals from HA recorder."""
    _LOGGER.debug(
        "Getting intervals from recorder for %s from %s to %s",
        entity_id,
        start_time,
        end_time,
    )

    states = await get_states_from_recorder(hass, entity_id, start_time, end_time)

    _LOGGER.debug(
        "Got %d states from recorder for %s", len(states) if states else 0, entity_id
    )

    intervals = await states_to_intervals(states, start_time, end_time)

    _LOGGER.debug("Converted to %d intervals for %s", len(intervals), entity_id)

    filtered_intervals = filter_intervals(intervals)

    _LOGGER.debug("Filtered to %d intervals for %s", len(filtered_intervals), entity_id)

    return filtered_intervals


async def get_states_from_recorder(
    hass: HomeAssistant, entity_id: str, start_time: datetime, end_time: datetime
) -> list[State | dict[str, Any]] | None:
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
    # Check if recorder is available
    recorder = get_instance(hass)
    if recorder is None:
        _LOGGER.debug("Recorder not available for %s", entity_id)
        return None

    try:
        states = await recorder.async_add_executor_job(
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
            _LOGGER.debug("Found %d states for %s", len(entity_states), entity_id)
        else:
            _LOGGER.debug("No states found for %s in recorder query result", entity_id)

    except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
        _LOGGER.error("Error getting states for %s: %s", entity_id, err)
        raise

    else:
        return entity_states


async def states_to_intervals(
    states: Sequence[State], start: datetime, end: datetime
) -> list[StateInterval]:
    """Convert state history to time intervals.

    Args:
        states: List of State objects
        start: Start time for analysis
        end: End time for analysis

    Returns:
        List of StateInterval objects

    """
    intervals: list[StateInterval] = []
    if not states:
        return intervals
    # Sort states chronologically
    sorted_states = sorted(states, key=lambda x: x.last_changed)

    # Determine the state that was active at the start time
    current_state = sorted_states[0].state
    for state in sorted_states:
        if state.last_changed <= start:
            current_state = state.state
        else:
            break

    current_time = start

    # Build intervals between state changes
    for state in sorted_states:
        if state.last_changed <= start:
            continue
        if state.last_changed > end:
            break

        # Append interval from the last change (or start) to this state change
        intervals.append(
            StateInterval(
                start=current_time,
                end=state.last_changed,
                state=current_state,
                entity_id=state.entity_id,
            )
        )
        current_state = state.state
        current_time = state.last_changed

    # After processing all state changes, append the final interval
    # This covers the period from the last state change up to the requested end time
    intervals.append(
        StateInterval(
            start=current_time,
            end=end,
            state=current_state,
            entity_id=state.entity_id,
        )
    )

    return intervals


def filter_intervals(
    intervals: list[StateInterval],
) -> list[StateInterval]:
    """Filter intervals to remove anomalous data and invalid states.

    - For binary sensors ("on" state): filter out intervals that are too short or too long.
    - For numeric sensors and other valid states: only remove intervals with invalid states, do not filter by duration.
    """
    filtered_intervals = []
    for interval in intervals:
        state = interval["state"]
        duration_seconds = (interval["end"] - interval["start"]).total_seconds()
        # Only apply duration filtering to binary "on" state
        if state == "on":
            if duration_seconds <= MAX_INTERVAL_SECONDS:
                filtered_intervals.append(interval)
        elif is_valid_state(state) and duration_seconds >= MIN_INTERVAL_SECONDS:
            # For all other states (including numerics), keep as long as state is valid and interval is long enough
            filtered_intervals.append(interval)

    return filtered_intervals
