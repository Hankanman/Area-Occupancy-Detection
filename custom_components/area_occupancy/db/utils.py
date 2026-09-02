"""Database utility functions."""

from __future__ import annotations

import bisect
from collections.abc import Collection, Iterable, Iterator
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, TypeVar

import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from homeassistant.exceptions import HomeAssistantError

from ..const import INVALID_STATES, MIN_CORRELATION_SAMPLES
from ..data.entity_type import DEFAULT_TYPES, InputType
from ..time_utils import from_db_utc, to_db_utc, to_utc
from ..utils import map_binary_state_to_semantic

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from ..data.config import SensorStates

_LOGGER = logging.getLogger(__name__)

# SQLite has a limit on the number of parameters in a single query.
# Default SQLITE_MAX_VARIABLE_NUMBER is 999, but some builds use 32766.
# Use a conservative batch size to avoid "too many SQL variables" errors.
SQLITE_BATCH_SIZE = 500

T = TypeVar("T")


# ─────────────────────────── Active-state resolution ────────────────────────────
#
# Issue #520: the stored-interval ("ground truth") path and the live-evidence
# path must agree on which states count as *occupied*, or the learned prior
# and the running probability describe different rooms.
#
# They did not. ``build_presence_query`` derived its occupied-state set from
# ``DEFAULT_TYPES`` while ``Entity.evidence`` reads the area's configured
# states, so a user who removed ``paused`` from ``CONF_MEDIA_ACTIVE_STATES``
# still had every ``paused`` stretch counted as occupancy ground truth. A
# media player parked in ``paused`` for days drove ``global_prior`` to the
# 0.99 clamp while live evidence correctly read ~0.01.
#
# The fix is to resolve active states the way ``EntityType`` does — the area's
# configured list when non-empty, the type default otherwise — from one shared
# helper, so the two paths cannot drift apart again.
# ``resolve_active_states``/``area_active_states_by_type`` answer this from the
# area config (well-defined even for a type with no entity loaded);
# ``entity_active_states`` answers it per entity from the live ``Entity``
# objects, which is what ``db.sync`` needs when deciding whether a recorded
# state was active for the entity that reported it.


def is_active_state(state: str, active_states: Collection[str]) -> bool:
    """Return whether a stored state counts as active for the given state set.

    Applies the same binary→semantic mapping as :attr:`Entity.evidence` so a
    door/window configured with ``open``/``closed`` matches the ``on``/``off``
    that Home Assistant actually records.
    """
    if not active_states:
        return False
    mapped = map_binary_state_to_semantic(state, sorted(active_states))
    return mapped in active_states


def entity_active_states(
    coordinator: AreaOccupancyCoordinator,
) -> dict[str, set[str]]:
    """Map every configured entity_id to the states that count as active for it.

    Spans all areas. An entity shared between areas takes the union of its
    per-area active states, so a state active in *any* area is treated as
    active — the conservative direction for the interval-length cap in
    :mod:`db.sync`, whose job is to bound implausibly long active stretches.

    ``active_states`` is read defensively: an entity object that doesn't
    expose it (a numeric sensor resolves an active *range* instead, and
    duck-typed stand-ins may carry neither) simply contributes nothing here,
    and :mod:`db.sync` falls back to its historic "on" semantics for it. One
    unusual entity must not take the whole sync down.
    """
    resolved: dict[str, set[str]] = {}
    for area in coordinator.areas.values():
        for entity_id, entity in area.entities.entities.items():
            states = getattr(entity, "active_states", None)
            if states:
                resolved.setdefault(entity_id, set()).update(states)
    return resolved


def resolve_active_states(
    sensor_states: SensorStates | None, input_type: InputType
) -> set[str]:
    """Return the states that count as active for an input type in an area.

    Mirrors how :class:`EntityType` resolves them for :attr:`Entity.evidence`:
    the area's configured list wins when non-empty, otherwise the type default
    applies (an empty configured list means "use defaults", not "nothing is
    active"). ``sensor_states`` has no field for every input type — sleep, for
    one — and those fall through to the defaults, exactly as the live path does.
    """
    configured = getattr(sensor_states, input_type.value, None)
    if configured:
        return set(configured)
    defaults = DEFAULT_TYPES.get(input_type)
    if defaults and defaults.get("active_states"):
        return set(defaults["active_states"])
    return set()


def area_active_states_by_type(
    coordinator: AreaOccupancyCoordinator,
    area_name: str,
    input_types: Iterable[InputType],
) -> dict[InputType, set[str]]:
    """Map each requested input type to the states counting as active in an area.

    Active states are configured per input type per area (there is no
    per-entity override — see issue #159), so every entity of a given type in
    an area resolves to the same set. Resolution keys off the area's *config*
    rather than its currently-loaded entities, so the answer is well-defined
    even for a type that has no entity loaded yet.
    """
    resolved: dict[InputType, set[str]] = {}
    area = coordinator.areas.get(area_name)
    if area is None:
        return resolved
    sensor_states = getattr(getattr(area, "config", None), "sensor_states", None)
    for input_type in input_types:
        states = resolve_active_states(sensor_states, input_type)
        if states:
            resolved[input_type] = states
    return resolved


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield lists of at most `size` items from the iterable.

    Args:
        items: Items to chunk
        size: Maximum size of each chunk

    Yields:
        Lists of items, each with at most `size` elements
    """
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def batched_delete_by_ids(
    session: Session,
    model: Any,
    ids: list[int],
    batch_size: int = SQLITE_BATCH_SIZE,
) -> int:
    """Delete records by ID in batches to avoid SQLite parameter limit.

    Args:
        session: SQLAlchemy session
        model: SQLAlchemy model class (must have an 'id' attribute)
        ids: List of IDs to delete
        batch_size: Maximum IDs per query (default: SQLITE_BATCH_SIZE)

    Returns:
        Total number of records deleted
    """
    if not ids:
        return 0

    total_deleted = 0
    for id_chunk in chunked(ids, batch_size):
        deleted = (
            session.query(model)
            .filter(model.id.in_(id_chunk))
            .delete(synchronize_session=False)
        )
        total_deleted += deleted

    return total_deleted


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

    after_start = last_motion_timeout_end or min(sorted_motion[-1][1], merged_end)
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


def prepare_occupied_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> tuple[list[datetime], list[datetime]]:
    """Build an indexed view of occupied intervals for repeated lookups.

    Pair with ``is_timestamp_in_prepared_intervals``. Each endpoint is
    converted to UTC-aware *once* and the intervals are sorted by start
    so a subsequent membership check is O(log N) instead of the O(N)
    scan ``is_timestamp_occupied`` performs (with two ``astimezone``
    calls per scanned element). Use this whenever the same interval set
    is checked against many timestamps — see ``analyze_correlation``'s
    per-chunk loop, where the unindexed path was issuing tens of
    millions of ``astimezone`` calls per analysis cycle on an RPi.

    Assumes intervals are non-overlapping; the occupied-intervals cache
    populated by the analysis pipeline merges adjacent intervals before
    persisting, so this holds for all in-tree callers. With overlapping
    inputs ``bisect`` only sees the latest-starting interval per
    timestamp, which is still a correct ``contains`` answer when intervals
    overlap fully or nest, but loses coverage of points that lie inside
    an earlier interval that ends after a later one's start. Callers that
    cannot guarantee non-overlap should fall back to ``is_timestamp_occupied``.

    Returns:
        Parallel ``(starts, ends)`` arrays sorted by start time.
    """
    if not intervals:
        return [], []
    normalised = sorted(
        ((to_utc(start), to_utc(end)) for start, end in intervals),
        key=lambda iv: iv[0],
    )
    starts = [s for s, _ in normalised]
    ends = [e for _, e in normalised]
    return starts, ends


def is_timestamp_in_prepared_intervals(
    timestamp: datetime,
    starts: list[datetime],
    ends: list[datetime],
) -> bool:
    """O(log N) membership test against a prepared interval index.

    Pair with ``prepare_occupied_intervals``. End times are exclusive,
    matching ``is_timestamp_occupied``'s semantics so the two helpers are
    interchangeable for non-overlapping interval sets.
    """
    if not starts:
        return False
    ts = to_utc(timestamp)
    idx = bisect.bisect_right(starts, ts) - 1
    if idx < 0:
        return False
    return ts < ends[idx]


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
