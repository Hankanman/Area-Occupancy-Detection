"""Database state synchronization operations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, TypeVar

import sqlalchemy as sa

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance
from homeassistant.util import dt as dt_util

from ..const import MAX_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS, RETENTION_DAYS
from . import queries, utils

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)
_INTERVAL_LOOKUP_BATCH = 250
T = TypeVar("T")


def _chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield lists of at most `size` items from the iterable."""
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _normalize_datetime(value: datetime) -> datetime:
    """Strip timezone info for consistent comparison."""
    if value.tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def _get_existing_interval_keys(
    session: sa.orm.Session,
    db: AreaOccupancyDB,
    interval_keys: set[tuple[str, datetime, datetime]],
) -> set[tuple[str, datetime, datetime]]:
    """Return keys already stored in the database using batched tuple lookups."""
    if not interval_keys:
        return set()

    keys_list = list(interval_keys)
    interval_tuple = sa.tuple_(
        db.Intervals.entity_id, db.Intervals.start_time, db.Intervals.end_time
    )
    existing_keys: set[tuple[str, datetime, datetime]] = set()

    for chunk in _chunked(keys_list, _INTERVAL_LOOKUP_BATCH):
        matches = session.query(db.Intervals).filter(interval_tuple.in_(chunk)).all()
        for interval in matches:
            start = _normalize_datetime(interval.start_time)
            end = _normalize_datetime(interval.end_time)
            existing_keys.add((interval.entity_id, start, end))

    return existing_keys


def _states_to_intervals(
    db: AreaOccupancyDB, states: dict[str, list[State]], end_time: datetime
) -> list[dict[str, Any]]:
    """Convert states to intervals by processing consecutive state changes for each entity.

    Args:
        db: Database instance
        states: Dictionary mapping entity_id to list of State objects
        end_time: The end time for the analysis period

    Returns:
        List of interval dictionaries with proper start_time, end_time, and duration_seconds

    """
    intervals = []
    retention_time = dt_util.now() - timedelta(days=RETENTION_DAYS)

    for entity_id, state_list in states.items():
        if not state_list:
            continue

        # Sort states by last_changed time
        sorted_states = sorted(state_list, key=lambda s: s.last_changed)

        # Process each state to create intervals
        for i, state in enumerate(sorted_states):
            # Skip states outside retention period
            if state.last_changed < retention_time:
                continue

            # Determine the end time for this interval
            if i + 1 < len(sorted_states):
                # Use the start time of the next state as the end time
                interval_end = sorted_states[i + 1].last_changed
            else:
                # For the last state, use the analysis end time
                interval_end = end_time

            # Calculate duration
            duration_seconds = (interval_end - state.last_changed).total_seconds()

            # Apply filtering based on state and duration
            if state.state == "on":
                if duration_seconds <= MAX_INTERVAL_SECONDS:
                    intervals.append(
                        {
                            "entity_id": entity_id,
                            "state": state.state,
                            "start_time": state.last_changed,
                            "end_time": interval_end,
                            "duration_seconds": duration_seconds,
                            "created_at": dt_util.utcnow(),
                        }
                    )
            elif (
                utils.is_valid_state(state.state)
                and duration_seconds >= MIN_INTERVAL_SECONDS
            ):
                intervals.append(
                    {
                        "entity_id": entity_id,
                        "state": state.state,
                        "start_time": state.last_changed,
                        "end_time": interval_end,
                        "duration_seconds": duration_seconds,
                        "created_at": dt_util.utcnow(),
                    }
                )

    return intervals


async def sync_states(db: AreaOccupancyDB) -> None:
    """Fetch states history from recorder and commit to Intervals table for all areas."""
    hass = db.coordinator.hass
    recorder = get_instance(hass)
    start_time = queries.get_latest_interval(db)
    end_time = dt_util.now()

    # Collect all entity IDs from all areas
    all_entity_ids = []
    for area_name in db.coordinator.get_area_names():
        area_data = db.coordinator.get_area(area_name)
        all_entity_ids.extend(area_data.entities.entity_ids)
    entity_ids = list(set(all_entity_ids))  # Remove duplicates

    try:
        states = await recorder.async_add_executor_job(
            lambda: get_significant_states(
                hass,
                start_time,
                end_time,
                entity_ids,
                minimal_response=False,
            )
        )

        if states:
            # Convert states to proper intervals with correct duration calculation
            intervals = _states_to_intervals(db, states, end_time)
            if intervals:
                with db.get_locked_session() as session:
                    # Pre-filter duplicates using a single query for better performance
                    # Build a set of (entity_id, start_time, end_time) tuples from intervals
                    interval_keys = {
                        (
                            interval_data["entity_id"],
                            interval_data["start_time"],
                            interval_data["end_time"],
                        )
                        for interval_data in intervals
                    }

                    # Query existing intervals matching these keys in a single query
                    if interval_keys:
                        existing_keys = _get_existing_interval_keys(
                            session, db, interval_keys
                        )
                    else:
                        existing_keys = set()

                    # Filter out intervals that already exist
                    new_intervals = []
                    for interval_data in intervals:
                        start = _normalize_datetime(interval_data["start_time"])
                        end = _normalize_datetime(interval_data["end_time"])
                        if (
                            interval_data["entity_id"],
                            start,
                            end,
                        ) in existing_keys:
                            continue

                        entity_id = interval_data["entity_id"]
                        area_name = db.coordinator.find_area_for_entity(entity_id)

                        if area_name:
                            interval_data["entry_id"] = db.coordinator.entry_id
                            interval_data["area_name"] = area_name
                            new_intervals.append(interval_data)

                    # Bulk insert new intervals
                    if new_intervals:
                        # Use bulk_insert_mappings for better performance
                        session.bulk_insert_mappings(db.Intervals, new_intervals)
                        session.commit()
                        _LOGGER.info(
                            "Synced %d new intervals from recorder", len(new_intervals)
                        )

    except (
        sa.exc.SQLAlchemyError,
        HomeAssistantError,
        TimeoutError,
        OSError,
        RuntimeError,
    ) as err:
        _LOGGER.error("Failed to sync states: %s", err)
        # Don't raise - let the caller handle it if needed
        # This allows the function to be called without breaking the caller
