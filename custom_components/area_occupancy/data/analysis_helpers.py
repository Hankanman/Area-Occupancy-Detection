"""Pure business logic functions for analysis operations.

This module contains pure functions (no database/IO dependencies) that can be
easily unit tested without mocks or database setup.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import datetime, timedelta

# Time slot constants
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR


def merge_overlapping_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge overlapping and adjacent time intervals.

    Args:
        intervals: List of (start_time, end_time) tuples, must be sorted by start_time

    Returns:
        List of merged (start_time, end_time) tuples, sorted by start_time
    """
    if not intervals:
        return []

    # Ensure intervals are sorted by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted_intervals:
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:
                # Overlapping or adjacent, merge them
                merged[-1] = (last_start, max(last_end, end))
            else:
                # Non-overlapping, add as new interval
                merged.append((start, end))

    return merged


def find_overlapping_motion_intervals(
    merged_interval: tuple[datetime, datetime],
    motion_intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Find all motion intervals that overlap with a merged interval.

    Args:
        merged_interval: (start, end) tuple representing the merged interval
        motion_intervals: List of (start, end) tuples representing motion intervals

    Returns:
        List of overlapping motion intervals
    """
    merged_start, merged_end = merged_interval
    return [
        (m_start, m_end)
        for m_start, m_end in motion_intervals
        if not (merged_end < m_start or merged_start > m_end)
    ]


def calculate_motion_union(
    motion_intervals: list[tuple[datetime, datetime]],
    merged_start: datetime,
    merged_end: datetime,
) -> tuple[datetime, datetime]:
    """Calculate the union of motion coverage within a merged interval.

    Args:
        motion_intervals: List of overlapping motion intervals (must be sorted by start)
        merged_start: Start time of the merged interval
        merged_end: End time of the merged interval

    Returns:
        Tuple of (union_start, union_end) clamped to merged interval boundaries
    """
    if not motion_intervals:
        return (merged_start, merged_end)

    # Sort by start time if not already sorted
    sorted_motion = sorted(motion_intervals, key=lambda x: x[0])

    motion_union_start = min(m_start for m_start, _ in sorted_motion)
    motion_union_end = max(m_end for _, m_end in sorted_motion)

    # Clamp to merged interval boundaries
    motion_union_start = max(motion_union_start, merged_start)
    motion_union_end = min(motion_union_end, merged_end)

    return (motion_union_start, motion_union_end)


def segment_interval_with_motion(
    merged_interval: tuple[datetime, datetime],
    motion_intervals: list[tuple[datetime, datetime]],
    timeout_seconds: int,
) -> list[tuple[datetime, datetime]]:
    """Segment a merged interval based on motion coverage and apply timeout.

    Iterates through each overlapping motion interval within the merged interval
    and creates a motion segment with timeout applied to each. Preserves gaps
    between motion intervals as separate segments. The caller should merge
    overlapping/adjacent segments afterward.

    Args:
        merged_interval: (start, end) tuple representing the merged interval
        motion_intervals: List of (start, end) tuples representing motion intervals
        timeout_seconds: Motion timeout in seconds to apply to motion segments

    Returns:
        List of segmented (start, end) tuples including:
        - Non-motion segments before, between, and after motion intervals
        - Motion segments with timeout applied (motion_start to motion_end + timeout)
    """
    merged_start, merged_end = merged_interval

    # Find overlapping motion intervals
    overlapping_motion = find_overlapping_motion_intervals(
        merged_interval, motion_intervals
    )

    if not overlapping_motion:
        # No motion overlap, return interval as-is
        return [(merged_start, merged_end)]

    # Sort motion intervals by start time
    sorted_motion = sorted(overlapping_motion, key=lambda x: x[0])

    segments: list[tuple[datetime, datetime]] = []
    timeout_delta = timedelta(seconds=timeout_seconds)

    # Add segment before first motion (if any)
    first_motion_start = sorted_motion[0][0]
    if merged_start < first_motion_start:
        segments.append((merged_start, first_motion_start))

    # Process each motion interval individually
    last_motion_timeout_end = None
    for i, (motion_start, motion_end) in enumerate(sorted_motion):
        # Clamp motion interval to merged interval boundaries
        clamped_start = max(motion_start, merged_start)
        clamped_end = min(motion_end, merged_end)

        # Create motion segment with timeout applied
        if clamped_start < clamped_end:
            motion_timeout_end = clamped_end + timeout_delta
            segments.append((clamped_start, motion_timeout_end))
            last_motion_timeout_end = motion_timeout_end

        # Add gap segment between current and next motion (if any)
        if i < len(sorted_motion) - 1:
            next_motion_start = sorted_motion[i + 1][0]
            gap_start = clamped_end
            gap_end = min(next_motion_start, merged_end)
            if gap_start < gap_end:
                segments.append((gap_start, gap_end))

    # Add segment after last motion (if any)
    # Start after the timeout applied to the last motion
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
    """Apply motion timeout to merged intervals and merge again.

    Args:
        merged_intervals: List of merged (start, end) intervals
        motion_intervals: List of raw motion (start, end) intervals
        timeout_seconds: Motion timeout in seconds

    Returns:
        List of final merged intervals with timeouts applied
    """
    extended_intervals: list[tuple[datetime, datetime]] = []

    for merged_interval in merged_intervals:
        segments = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        extended_intervals.extend(segments)

    # Merge again after extending motion intervals
    return merge_overlapping_intervals(extended_intervals)


def calculate_time_slot(timestamp: datetime, slot_minutes: int) -> int:
    """Calculate the time slot number for a given timestamp.

    Args:
        timestamp: The datetime to calculate slot for
        slot_minutes: Size of each time slot in minutes

    Returns:
        Slot number (0-based) within the day
    """
    hour = timestamp.hour
    minute = timestamp.minute
    return (hour * MINUTES_PER_HOUR + minute) // slot_minutes


def aggregate_intervals_by_slot(
    intervals: list[tuple[datetime, datetime]],
    slot_minutes: int,
) -> list[tuple[int, int, float]]:
    """Aggregate time intervals by day of week and time slot.

    Args:
        intervals: List of (start_time, end_time) tuples
        slot_minutes: Size of each time slot in minutes

    Returns:
        List of (day_of_week, time_slot, total_occupied_seconds) tuples
        where day_of_week is Python weekday (0=Monday, 6=Sunday)
    """
    slot_seconds: defaultdict[tuple[int, int], float] = defaultdict(float)

    for start_time, end_time in intervals:
        # Calculate which slots this interval covers
        current_time = start_time
        while current_time < end_time:
            day_of_week = current_time.weekday()
            slot = calculate_time_slot(current_time, slot_minutes)

            # Calculate slot boundaries from start of day
            day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            slot_start = day_start + timedelta(minutes=slot * slot_minutes)
            slot_end = slot_start + timedelta(minutes=slot_minutes)

            # Calculate overlap duration
            overlap_start = max(current_time, slot_start)
            overlap_end = min(end_time, slot_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()

            if overlap_duration > 0:
                slot_seconds[(day_of_week, slot)] += overlap_duration

            # Move to next slot
            current_time = slot_end

    # Convert to result format
    result = []
    for (day, slot), seconds in slot_seconds.items():
        result.append((day, slot, seconds))

    return result


def is_timestamp_occupied(
    timestamp: datetime, occupied_intervals: list[tuple[datetime, datetime]]
) -> bool:
    """Check if timestamp falls within any occupied interval using binary search.

    Args:
        timestamp: The datetime to check
        occupied_intervals: List of (start, end) tuples representing occupied periods
                          (should be sorted by start time for optimal performance)

    Returns:
        True if timestamp falls within any occupied interval, False otherwise
    """
    if not occupied_intervals:
        return False

    # Binary search to find the rightmost interval that starts <= ts
    idx = bisect.bisect_right([start for start, _ in occupied_intervals], timestamp)

    # Check if timestamp falls within the interval found
    if idx > 0:
        start, end = occupied_intervals[idx - 1]
        if start <= timestamp < end:
            return True

    return False
