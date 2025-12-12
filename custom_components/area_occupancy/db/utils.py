"""Database utility functions."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError

from homeassistant.exceptions import HomeAssistantError

from ..const import INVALID_STATES, MIN_CORRELATION_SAMPLES
from ..time_utils import from_db_utc, to_db_utc, to_utc

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


def get_occupied_intervals_for_analysis(
    db: Any,
    area_name: str,
    start_time: datetime,
    end_time: datetime,
) -> list[tuple[datetime, datetime]]:
    """Get occupied intervals from cache for analysis.

    Results are scoped by the current config entry_id and use overlap semantics:
    intervals that partially overlap the requested time range are included.
    An interval overlaps if: interval.start_time <= end_time AND interval.end_time >= start_time.

    Args:
        db: Database instance
        area_name: Area name
        start_time: Start of period
        end_time: End of period

    Returns:
        List of (start, end) tuples of occupied intervals (timezone-aware UTC)
    """
    try:
        # DB stores naive UTC; always bind naive UTC for SQL queries
        start_time_db = to_db_utc(start_time)
        end_time_db = to_db_utc(end_time)

        with db.get_session() as session:
            # Debug: Check total intervals in cache for this area
            total_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter(
                    db.OccupiedIntervalsCache.entry_id == db.coordinator.entry_id,
                    db.OccupiedIntervalsCache.area_name == area_name,
                )
                .count()
            )
            _LOGGER.debug(
                "Querying occupied intervals for area %s: period=[%s, %s], total_intervals_in_cache=%d",
                area_name,
                start_time_db,
                end_time_db,
                total_intervals,
            )

            intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter(
                    db.OccupiedIntervalsCache.entry_id == db.coordinator.entry_id,
                    db.OccupiedIntervalsCache.area_name == area_name,
                    db.OccupiedIntervalsCache.start_time <= end_time_db,
                    db.OccupiedIntervalsCache.end_time >= start_time_db,
                )
                .all()
            )

            _LOGGER.debug(
                "Found %d overlapping intervals for area %s",
                len(intervals),
                area_name,
            )

            # Debug: Log first few intervals if any found
            if intervals:
                for i, interval in enumerate(intervals[:3]):
                    _LOGGER.debug(
                        "Interval %d: [%s, %s]",
                        i,
                        interval.start_time,
                        interval.end_time,
                    )

            # Convert DB naive UTC back into aware UTC for runtime computations
            return [
                (from_db_utc(i.start_time), from_db_utc(i.end_time)) for i in intervals
            ]
    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Error getting occupied intervals for analysis: %s", e)
        return []


def is_timestamp_occupied(
    timestamp: datetime,
    occupied_intervals: list[tuple[datetime, datetime]],
) -> bool:
    """Check if timestamp falls within any occupied interval.

    Args:
        timestamp: Timestamp to check
        occupied_intervals: List of (start, end) tuples. End time is exclusive.

    Returns:
        True if timestamp is within an interval, False otherwise
    """
    # Ensure timestamp and intervals are compared in aware UTC
    timestamp_utc = to_utc(timestamp)
    return any(
        to_utc(start) <= timestamp_utc < to_utc(end)
        for start, end in occupied_intervals
    )


def validate_sample_count(
    samples: list[Any],
    min_samples: int = MIN_CORRELATION_SAMPLES,
    error_type: str = "too_few_samples",
) -> dict[str, Any] | None:
    """Validate sample count and return error dict if insufficient.

    Args:
        samples: List of samples
        min_samples: Minimum required samples
        error_type: Error string to return

    Returns:
        Error dict if invalid, None if valid
    """
    if len(samples) < min_samples:
        _LOGGER.debug(
            "Insufficient samples: %d < %d",
            len(samples),
            min_samples,
        )
        return {
            "sample_count": len(samples),
            "analysis_error": error_type,
        }
    return None


def validate_occupied_intervals(
    intervals: list[Any],
    sample_count: int,
) -> dict[str, Any] | None:
    """Validate occupied intervals exist.

    Args:
        intervals: List of occupied intervals
        sample_count: Number of samples (for reporting in error)

    Returns:
        Error dict if invalid, None if valid
    """
    if not intervals:
        _LOGGER.debug("No occupied intervals found for analysis")
        return {
            "sample_count": sample_count,
            "analysis_error": "no_occupancy_data",
        }
    return None
