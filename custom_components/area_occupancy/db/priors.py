"""Prior calculation and storage using new database tables.

This module handles storing and retrieving global priors and occupied intervals
using the GlobalPriors and OccupiedIntervalsCache tables.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, union_all
from sqlalchemy.exc import (
    DataError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)

from homeassistant.const import STATE_ON, STATE_PLAYING
from homeassistant.util import dt as dt_util

from ..const import GLOBAL_PRIOR_HISTORY_COUNT
from ..data.entity_type import InputType

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)


def save_global_prior(
    db: AreaOccupancyDB,
    area_name: str,
    prior_value: float,
    data_period_start: datetime,
    data_period_end: datetime,
    total_occupied_seconds: float,
    total_period_seconds: float,
    interval_count: int,
    calculation_method: str = "interval_analysis",
    confidence: float | None = None,
) -> bool:
    """Save global prior calculation to GlobalPriors table.

    Args:
        db: Database instance
        area_name: Area name
        prior_value: Calculated prior probability
        data_period_start: Start of data period used
        data_period_end: End of data period used
        total_occupied_seconds: Total occupied time in period
        total_period_seconds: Total period duration
        interval_count: Number of intervals used
        calculation_method: Method used for calculation
        confidence: Confidence in calculation (0.0-1.0)

    Returns:
        True if saved successfully, False otherwise
    """
    _LOGGER.debug(
        "Saving global prior for area: %s, value: %.4f", area_name, prior_value
    )

    try:
        with db.get_locked_session() as session:
            # Create hash of underlying data for validation
            data_hash = _create_data_hash(
                area_name,
                data_period_start,
                data_period_end,
                total_occupied_seconds,
                interval_count,
            )

            # Check if global prior already exists for this area
            existing = (
                session.query(db.GlobalPriors).filter_by(area_name=area_name).first()
            )

            if existing:
                # Update existing record
                existing.prior_value = prior_value
                existing.calculation_date = dt_util.utcnow()
                existing.data_period_start = data_period_start
                existing.data_period_end = data_period_end
                existing.total_occupied_seconds = total_occupied_seconds
                existing.total_period_seconds = total_period_seconds
                existing.interval_count = interval_count
                existing.confidence = confidence
                existing.calculation_method = calculation_method
                existing.underlying_data_hash = data_hash
                existing.updated_at = dt_util.utcnow()
            else:
                # Create new record
                global_prior = db.GlobalPriors(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    prior_value=prior_value,
                    calculation_date=dt_util.utcnow(),
                    data_period_start=data_period_start,
                    data_period_end=data_period_end,
                    total_occupied_seconds=total_occupied_seconds,
                    total_period_seconds=total_period_seconds,
                    interval_count=interval_count,
                    confidence=confidence,
                    calculation_method=calculation_method,
                    underlying_data_hash=data_hash,
                )
                session.add(global_prior)

            session.commit()

            # Prune old global prior history (keep only last N)
            _prune_old_global_priors(db, session, area_name)

            _LOGGER.debug("Global prior saved successfully")
            return True

    except SQLAlchemyError as e:
        _LOGGER.error("Database error saving global prior: %s", e)
        session.rollback()
        return False
    except (ValueError, TypeError, RuntimeError) as e:
        _LOGGER.error("Unexpected error saving global prior: %s", e)
        session.rollback()
        return False


def get_global_prior(db: AreaOccupancyDB, area_name: str) -> dict[str, Any] | None:
    """Get the most recent global prior for an area.

    Args:
        db: Database instance
        area_name: Area name

    Returns:
        Dictionary with global prior data, or None if not found
    """
    try:
        with db.get_session() as session:
            global_prior = (
                session.query(db.GlobalPriors).filter_by(area_name=area_name).first()
            )

            if global_prior:
                return {
                    "prior_value": global_prior.prior_value,
                    "calculation_date": global_prior.calculation_date,
                    "data_period_start": global_prior.data_period_start,
                    "data_period_end": global_prior.data_period_end,
                    "total_occupied_seconds": global_prior.total_occupied_seconds,
                    "total_period_seconds": global_prior.total_period_seconds,
                    "interval_count": global_prior.interval_count,
                    "confidence": global_prior.confidence,
                    "calculation_method": global_prior.calculation_method,
                }

            return None

    except SQLAlchemyError as e:
        _LOGGER.error("Database error getting global prior: %s", e)
        return None


def save_occupied_intervals_cache(
    db: AreaOccupancyDB,
    area_name: str,
    intervals: list[tuple[datetime, datetime]],
    data_source: str = "merged",
) -> bool:
    """Save occupied intervals to OccupiedIntervalsCache table.

    Args:
        db: Database instance
        area_name: Area name
        intervals: List of (start_time, end_time) tuples
        data_source: Source of intervals ('primary_sensor', 'motion_sensors', 'merged')

    Returns:
        True if saved successfully, False otherwise
    """
    _LOGGER.debug(
        "Saving %d occupied intervals to cache for area: %s",
        len(intervals),
        area_name,
    )

    try:
        with db.get_locked_session() as session:
            calculation_date = dt_util.utcnow()

            # Delete existing cached intervals for this area
            session.query(db.OccupiedIntervalsCache).filter_by(
                area_name=area_name
            ).delete(synchronize_session=False)

            # Insert new intervals
            for start_time, end_time in intervals:
                duration_seconds = (end_time - start_time).total_seconds()

                cached_interval = db.OccupiedIntervalsCache(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration_seconds,
                    calculation_date=calculation_date,
                    data_source=data_source,
                )
                session.add(cached_interval)

            session.commit()
            _LOGGER.debug("Occupied intervals cache saved successfully")
            return True

    except SQLAlchemyError as e:
        _LOGGER.error("Database error saving occupied intervals cache: %s", e)
        session.rollback()
        return False
    except (ValueError, TypeError, RuntimeError) as e:
        _LOGGER.error("Unexpected error saving occupied intervals cache: %s", e)
        session.rollback()
        return False


def get_occupied_intervals_cache(
    db: AreaOccupancyDB,
    area_name: str,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> list[tuple[datetime, datetime]]:
    """Get occupied intervals from OccupiedIntervalsCache table.

    Args:
        db: Database instance
        area_name: Area name
        period_start: Optional start time filter
        period_end: Optional end time filter

    Returns:
        List of (start_time, end_time) tuples
    """
    try:
        with db.get_session() as session:
            query = session.query(db.OccupiedIntervalsCache).filter_by(
                area_name=area_name
            )

            if period_start:
                query = query.filter(
                    db.OccupiedIntervalsCache.start_time >= period_start
                )
            if period_end:
                query = query.filter(db.OccupiedIntervalsCache.end_time <= period_end)

            cached_intervals = query.order_by(
                db.OccupiedIntervalsCache.start_time
            ).all()

            return [
                (interval.start_time, interval.end_time)
                for interval in cached_intervals
            ]

    except SQLAlchemyError as e:
        _LOGGER.error("Database error getting occupied intervals cache: %s", e)
        return []


def is_occupied_intervals_cache_valid(
    db: AreaOccupancyDB,
    area_name: str,
    max_age_hours: int = 24,
) -> bool:
    """Check if cached occupied intervals are still valid.

    Args:
        db: Database instance
        area_name: Area name
        max_age_hours: Maximum age in hours before cache is considered stale

    Returns:
        True if cache is valid, False otherwise
    """
    try:
        with db.get_session() as session:
            latest = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .order_by(db.OccupiedIntervalsCache.calculation_date.desc())
                .first()
            )

            if not latest:
                return False

            age = (dt_util.utcnow() - latest.calculation_date).total_seconds() / 3600
            return age < max_age_hours

    except SQLAlchemyError as e:
        _LOGGER.error("Database error checking cache validity: %s", e)
        return False


def get_total_occupied_seconds(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    lookback_days: int,
    motion_timeout_seconds: int,
    include_media: bool = False,
    include_appliance: bool = False,
    media_sensor_ids: list[str] | None = None,
    appliance_sensor_ids: list[str] | None = None,
    session_provider: Callable[[], AbstractContextManager[Any]] | None = None,
) -> float:
    """Calculate total occupied seconds with SQL fast-path and Python fallback."""
    session_provider = session_provider or db.get_session

    try:
        if motion_timeout_seconds == 0 and not include_media and not include_appliance:
            total_seconds = db.get_total_occupied_seconds_sql(
                entry_id=entry_id,
                area_name=area_name,
                lookback_days=lookback_days,
                motion_timeout_seconds=0,
                include_media=False,
                include_appliance=False,
                media_sensor_ids=None,
                appliance_sensor_ids=None,
            )
            if total_seconds > 0:
                _LOGGER.debug(
                    "Total occupied seconds (SQL) for %s: %.1f",
                    area_name,
                    total_seconds,
                )
                return float(total_seconds)
    except (SQLAlchemyError, AttributeError, TypeError) as e:
        _LOGGER.debug(
            "SQL method failed for total occupied seconds, falling back to Python: %s",
            e,
            exc_info=True,
        )

    intervals = get_occupied_intervals(
        db,
        entry_id,
        area_name,
        lookback_days,
        motion_timeout_seconds,
        include_media=include_media,
        include_appliance=include_appliance,
        media_sensor_ids=media_sensor_ids,
        appliance_sensor_ids=appliance_sensor_ids,
        session_provider=session_provider,
    )

    total_seconds = 0.0
    for start_time, end_time in intervals:
        total_seconds += (end_time - start_time).total_seconds()

    _LOGGER.debug(
        "Total occupied seconds (Python) for %s: %.1f", area_name, total_seconds
    )
    return total_seconds


def get_occupied_intervals(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    lookback_days: int,
    motion_timeout_seconds: int,
    include_media: bool = False,
    include_appliance: bool = False,
    media_sensor_ids: list[str] | None = None,
    appliance_sensor_ids: list[str] | None = None,
    session_provider: Callable[[], AbstractContextManager[Any]] | None = None,
) -> list[tuple[datetime, datetime]]:
    """Fetch occupied intervals with caching and Python fallback."""
    session_provider = session_provider or db.get_session
    now = dt_util.utcnow()
    lookback_date = now - timedelta(days=lookback_days)

    if not include_media and not include_appliance:
        if is_occupied_intervals_cache_valid(db, area_name, max_age_hours=24):
            cached_intervals = get_occupied_intervals_cache(
                db, area_name, period_start=lookback_date, period_end=now
            )
            if cached_intervals:
                _LOGGER.debug(
                    "Using cached occupied intervals for %s: %d intervals",
                    area_name,
                    len(cached_intervals),
                )
                return cached_intervals

    try:
        start_time = dt_util.utcnow()
        with session_provider() as session:
            return get_occupied_intervals_from_session(
                session,
                db,
                entry_id,
                area_name,
                lookback_date,
                motion_timeout_seconds,
                include_media,
                include_appliance,
                media_sensor_ids,
                appliance_sensor_ids,
                start_time,
            )
    except (OperationalError, SQLAlchemyError, DataError, ProgrammingError) as e:
        _LOGGER.error("Database error in get_occupied_intervals: %s", e)
        return []
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Unexpected error in get_occupied_intervals: %s", e)
        return []


def get_time_bounds(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    entity_ids: list[str] | None = None,
    session_provider: Callable[[], AbstractContextManager[Any]] | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Return min/max timestamps for specified entities or area."""
    session_provider = session_provider or db.get_session
    try:
        with session_provider() as session:
            return get_time_bounds_from_session(
                session, db, entry_id, area_name, entity_ids
            )
    except (OperationalError, SQLAlchemyError, DataError, ProgrammingError) as e:
        _LOGGER.error("Database error getting time bounds: %s", e)
        return (None, None)
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Unexpected error getting time bounds: %s", e)
        return (None, None)


def get_occupied_intervals_from_session(
    session: Any,
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    lookback_date: datetime,
    motion_timeout_seconds: int,
    include_media: bool,
    include_appliance: bool,
    media_sensor_ids: list[str] | None,
    appliance_sensor_ids: list[str] | None,
    start_time: datetime,
) -> list[tuple[datetime, datetime]]:
    """Query occupied intervals using an existing session."""
    try:
        base_filters = build_base_filters(db, entry_id, lookback_date, area_name)
        motion_query = build_motion_query(session, db, base_filters)
        queries = [motion_query]

        if include_media and media_sensor_ids:
            queries.append(
                build_media_query(session, db, base_filters, media_sensor_ids)
            )

        if include_appliance and appliance_sensor_ids:
            queries.append(
                build_appliance_query(session, db, base_filters, appliance_sensor_ids)
            )

        all_results = execute_union_queries(session, db, queries)
        all_intervals, motion_raw, media_count, appliance_count = process_query_results(
            all_results
        )

        query_time = (dt_util.utcnow() - start_time).total_seconds()
        _LOGGER.debug(
            "Interval query executed in %.3fs for %s (total=%d, motion=%d, media=%d, appliance=%d)",
            query_time,
            area_name,
            len(all_intervals),
            len(motion_raw),
            media_count,
            appliance_count,
        )

        if not all_intervals:
            return []

        merged_intervals = merge_overlapping_intervals(all_intervals)
        extended_intervals = apply_motion_timeout(
            merged_intervals, motion_raw, motion_timeout_seconds
        )

        processing_time = (dt_util.utcnow() - start_time).total_seconds()
        _LOGGER.debug(
            "Unified occupancy calculation for %s: %d raw -> %d merged intervals (processing: %.3fs)",
            area_name,
            len(all_intervals),
            len(extended_intervals),
            processing_time,
        )

    except (OperationalError, SQLAlchemyError, DataError, ProgrammingError) as e:
        _LOGGER.error("Database error getting occupied intervals: %s", e)
        return []
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Unexpected error getting occupied intervals: %s", e)
        return []
    else:
        return extended_intervals


def build_base_filters(
    db: AreaOccupancyDB, entry_id: str, lookback_date: datetime, area_name: str
) -> list[Any]:
    """Construct base SQLAlchemy filters for interval queries."""
    return [
        db.Entities.entry_id == entry_id,
        db.Entities.area_name == area_name,
        db.Intervals.start_time >= lookback_date,
    ]


def build_motion_query(
    session: Any, db: AreaOccupancyDB, base_filters: list[Any]
) -> Any:
    """Create query selecting motion intervals."""
    return (
        session.query(
            db.Intervals.start_time,
            db.Intervals.end_time,
            func.literal("motion").label("sensor_type"),
        )
        .join(db.Entities, db.Intervals.entity_id == db.Entities.entity_id)
        .filter(
            *base_filters,
            db.Entities.entity_type == InputType.MOTION.value,
            db.Intervals.state == "on",
        )
    )


def build_media_query(
    session: Any,
    db: AreaOccupancyDB,
    base_filters: list[Any],
    sensor_ids: list[str],
) -> Any:
    """Create query selecting media player intervals."""
    return (
        session.query(
            db.Intervals.start_time,
            db.Intervals.end_time,
            func.literal("media").label("sensor_type"),
        )
        .join(db.Entities, db.Intervals.entity_id == db.Entities.entity_id)
        .filter(
            *base_filters,
            db.Entities.entity_type == InputType.MEDIA.value,
            db.Intervals.entity_id.in_(sensor_ids),
            db.Intervals.state == STATE_PLAYING,
        )
    )


def build_appliance_query(
    session: Any,
    db: AreaOccupancyDB,
    base_filters: list[Any],
    sensor_ids: list[str],
) -> Any:
    """Create query selecting appliance intervals."""
    return (
        session.query(
            db.Intervals.start_time,
            db.Intervals.end_time,
            func.literal("appliance").label("sensor_type"),
        )
        .join(db.Entities, db.Intervals.entity_id == db.Entities.entity_id)
        .filter(
            *base_filters,
            db.Entities.entity_type == InputType.APPLIANCE.value,
            db.Intervals.entity_id.in_(sensor_ids),
            db.Intervals.state == STATE_ON,
        )
    )


def execute_union_queries(session: Any, db: AreaOccupancyDB, queries: list[Any]) -> Any:
    """Execute one or more queries, returning combined results."""
    if len(queries) == 1:
        combined_query = queries[0].order_by(db.Intervals.start_time)
        return combined_query.all()

    union_query = union_all(*[q.subquery() for q in queries])
    combined_query = session.query(
        union_query.c.start_time,
        union_query.c.end_time,
        union_query.c.sensor_type,
    ).order_by(union_query.c.start_time)
    return combined_query.all()


def process_query_results(
    results: list[tuple[datetime, datetime, str]],
) -> tuple[list[tuple[datetime, datetime]], list[tuple[datetime, datetime]], int, int]:
    """Split query results into categories and counts."""
    motion_raw: list[tuple[datetime, datetime]] = []
    all_intervals: list[tuple[datetime, datetime]] = []
    media_count = 0
    appliance_count = 0

    for start, end, sensor_type in results:
        interval = (start, end)
        all_intervals.append(interval)
        if sensor_type == "motion":
            motion_raw.append(interval)
        elif sensor_type == "media":
            media_count += 1
        elif sensor_type == "appliance":
            appliance_count += 1

    return (all_intervals, motion_raw, media_count, appliance_count)


def get_time_bounds_from_session(
    session: Any,
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    entity_ids: list[str] | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Calculate min/max timestamps using an existing session."""
    query = session.query(
        func.min(db.Intervals.start_time).label("first"),
        func.max(db.Intervals.end_time).label("last"),
    )

    if entity_ids is not None:
        query = query.filter(db.Intervals.entity_id.in_(entity_ids))
    else:
        query = query.join(
            db.Entities, db.Intervals.entity_id == db.Entities.entity_id
        ).filter(
            db.Entities.entry_id == entry_id,
            db.Entities.area_name == area_name,
        )

    time_bounds = query.first()
    if not time_bounds:
        return (None, None)
    return (time_bounds.first, time_bounds.last)


def _create_data_hash(
    area_name: str,
    period_start: datetime,
    period_end: datetime,
    total_occupied: float,
    interval_count: int,
) -> str:
    """Create a hash of underlying data for validation.

    Args:
        area_name: Area name
        period_start: Period start time
        period_end: Period end time
        total_occupied: Total occupied seconds
        interval_count: Number of intervals

    Returns:
        Hash string
    """
    data = {
        "area_name": area_name,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "total_occupied": total_occupied,
        "interval_count": interval_count,
    }
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()


def _prune_old_global_priors(
    db: AreaOccupancyDB,
    session: Any,
    area_name: str,
) -> None:
    """Prune old global prior calculations, keeping only the most recent N.

    Args:
        db: Database instance
        session: Database session
        area_name: Area name
    """
    try:
        # Get all global priors for this area, ordered by calculation date
        # Note: GlobalPriors has unique constraint on area_name, so there's only one
        # But we keep this function for future use if we change to allow history
        priors = (
            session.query(db.GlobalPriors)
            .filter_by(area_name=area_name)
            .order_by(db.GlobalPriors.calculation_date.desc())
            .all()
        )

        # Keep only the most recent N calculations
        if len(priors) > GLOBAL_PRIOR_HISTORY_COUNT:
            to_delete = priors[GLOBAL_PRIOR_HISTORY_COUNT:]
            for prior in to_delete:
                session.delete(prior)
            session.commit()
            _LOGGER.debug(
                "Pruned %d old global prior calculations for %s",
                len(to_delete),
                area_name,
            )

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError) as e:
        _LOGGER.warning("Error pruning old global priors: %s", e)
        # Don't raise - this is cleanup, not critical


# ============================================================================
# Interval Helper Functions
# ============================================================================


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


def calculate_motion_union(
    motion_intervals: list[tuple[datetime, datetime]],
    merged_start: datetime,
    merged_end: datetime,
) -> tuple[datetime, datetime]:
    """Calculate the union of motion coverage within a merged interval."""
    if not motion_intervals:
        return (merged_start, merged_end)

    sorted_motion = sorted(motion_intervals, key=lambda x: x[0])

    motion_union_start = min(m_start for m_start, _ in sorted_motion)
    motion_union_end = max(m_end for _, m_end in sorted_motion)

    motion_union_start = max(motion_union_start, merged_start)
    motion_union_end = min(motion_union_end, merged_end)

    return (motion_union_start, motion_union_end)


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
            motion_timeout_end = clamped_end + timeout_delta
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
