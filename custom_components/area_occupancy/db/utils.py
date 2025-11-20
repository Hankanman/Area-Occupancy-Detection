"""Database utility functions."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any

import sqlalchemy as sa

from homeassistant.exceptions import HomeAssistantError

from . import maintenance
from .constants import INVALID_STATES

_LOGGER = logging.getLogger(__name__)


def is_valid_state(state: Any) -> bool:
    """Check if a state is valid."""
    return state not in INVALID_STATES


def is_intervals_empty(db: Any) -> bool:
    """Check if the intervals table is empty using ORM (read-only, no lock)."""
    try:
        with db.get_session() as session:
            count = session.query(db.Intervals).count()
            return bool(count == 0)
    except (
        sa.exc.SQLAlchemyError,
        HomeAssistantError,
        TimeoutError,
        OSError,
        RuntimeError,
    ) as e:
        # If table doesn't exist, it's considered empty
        if "no such table" in str(e).lower():
            _LOGGER.debug("Intervals table doesn't exist yet, considering empty")
            return True
        _LOGGER.error("Failed to check if intervals empty: %s", e)
        # Return True as fallback to trigger data population
        return True


def safe_is_intervals_empty(db: Any) -> bool:
    """Safely check if intervals table is empty (fast, no integrity checks).

    Note: Database integrity checks are deferred to background health check
    task that runs 60 seconds after startup to avoid blocking integration loading.

    Returns:
        bool: True if intervals are empty, False if intervals exist
    """
    try:
        # Quick check - assume database is healthy during startup
        # Integrity checks will be performed by background health check task
        return is_intervals_empty(db)
    except (
        sa.exc.SQLAlchemyError,
        HomeAssistantError,
        TimeoutError,
        OSError,
        RuntimeError,
    ) as e:
        # If we hit a corruption error, log it but don't block startup
        if maintenance.is_database_corrupted(db, e):
            _LOGGER.warning(
                "Database may be corrupted (error: %s). "
                "Background health check will attempt recovery in 60 seconds.",
                e,
            )
        else:
            _LOGGER.error("Error checking intervals: %s", e)

        # Assume empty to trigger data population, but don't block startup
        return True


def merge_overlapping_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge overlapping and adjacent time intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted_intervals:
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

    return merged


def find_overlapping_motion_intervals(
    merged_interval: tuple[datetime, datetime],
    motion_intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Find all motion intervals that overlap with a merged interval."""
    merged_start, merged_end = merged_interval
    return [
        (m_start, m_end)
        for m_start, m_end in motion_intervals
        if not (merged_end < m_start or merged_start > m_end)
    ]


def segment_interval_with_motion(
    merged_interval: tuple[datetime, datetime],
    motion_intervals: list[tuple[datetime, datetime]],
    timeout_seconds: int,
) -> list[tuple[datetime, datetime]]:
    """Segment a merged interval based on motion coverage and apply timeout."""
    merged_start, merged_end = merged_interval

    overlapping_motion = find_overlapping_motion_intervals(
        merged_interval, motion_intervals
    )

    if not overlapping_motion:
        return [(merged_start, merged_end)]

    sorted_motion = sorted(overlapping_motion, key=lambda x: x[0])

    segments: list[tuple[datetime, datetime]] = []
    timeout_delta = timedelta(seconds=timeout_seconds)

    first_motion_start = sorted_motion[0][0]
    if merged_start < first_motion_start:
        segments.append((merged_start, first_motion_start))

    last_motion_timeout_end = None
    for i, (motion_start, motion_end) in enumerate(sorted_motion):
        clamped_start = max(motion_start, merged_start)
        clamped_end = min(motion_end, merged_end)

        motion_timeout_end = None
        if clamped_start < clamped_end:
            motion_timeout_end = min(clamped_end + timeout_delta, merged_end)
            segments.append((clamped_start, motion_timeout_end))
            last_motion_timeout_end = motion_timeout_end

        if i < len(sorted_motion) - 1:
            next_motion_start = sorted_motion[i + 1][0]
            gap_end = min(next_motion_start, merged_end)
            if motion_timeout_end is not None and motion_timeout_end < gap_end:
                segments.append((motion_timeout_end, gap_end))

    after_start = (
        last_motion_timeout_end
        if last_motion_timeout_end
        else min(sorted_motion[-1][1], merged_end)
    )
    if after_start < merged_end:
        segments.append((after_start, merged_end))

    return segments


def apply_motion_timeout(
    merged_intervals: list[tuple[datetime, datetime]],
    motion_intervals: list[tuple[datetime, datetime]],
    timeout_seconds: int,
) -> list[tuple[datetime, datetime]]:
    """Apply motion timeout to merged intervals and merge again."""
    extended_intervals: list[tuple[datetime, datetime]] = []

    for merged_interval in merged_intervals:
        segments = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        extended_intervals.extend(segments)

    return merge_overlapping_intervals(extended_intervals)
