"""Database query operations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import func, union_all
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import literal

from homeassistant.const import STATE_ON, STATE_PLAYING
from homeassistant.util import dt as dt_util

from ..const import DEFAULT_TIME_PRIOR
from ..data.entity_type import InputType
from .utils import apply_motion_timeout, merge_overlapping_intervals

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)


def get_area_data(db: AreaOccupancyDB, entry_id: str) -> dict[str, Any] | None:
    """Get area data for a specific entry_id (read-only, no lock)."""
    try:
        with db.get_session() as session:
            area = session.query(db.Areas).filter_by(entry_id=entry_id).first()
            if area:
                return dict(area.to_dict())
            return None
    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Failed to get area data: %s", e)
        return None


def get_latest_interval(db: AreaOccupancyDB) -> datetime:
    """Return the latest interval end time minus 1 hour, or default window if none (read-only, no lock)."""
    try:
        with db.get_session() as session:
            result = session.execute(
                sa.select(sa.func.max(db.Intervals.end_time))
            ).scalar()
            if result:
                return result - timedelta(hours=1)
            return dt_util.utcnow() - timedelta(days=10)
    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        TimeoutError,
    ) as e:
        # If table doesn't exist or any other error, return a default time
        if "no such table" in str(e).lower():
            _LOGGER.debug("Intervals table doesn't exist yet, using default time")
        else:
            _LOGGER.warning("Failed to get latest interval, using default time: %s", e)
        return dt_util.utcnow() - timedelta(days=10)


def get_time_prior(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    day_of_week: int,
    time_slot: int,
    default_prior: float = DEFAULT_TIME_PRIOR,
) -> float:
    """Get the time prior for a specific time slot.

    Args:
        db: Database instance
        entry_id: The area entry ID to filter by
        area_name: The area name to filter by
        day_of_week: Day of week (0=Monday, 6=Sunday)
        time_slot: Time slot index
        default_prior: Default prior value if not found

    Returns:
        Time prior value or default if not found
    """
    try:
        with db.get_session() as session:
            prior = (
                session.query(db.Priors)
                .filter_by(
                    entry_id=entry_id,
                    area_name=area_name,
                    day_of_week=day_of_week,
                    time_slot=time_slot,
                )
                .first()
            )
            return float(prior.prior_value) if prior else default_prior
    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error getting time prior: %s", e)
        return default_prior


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
) -> list[tuple[datetime, datetime]]:
    """Fetch occupied intervals without caching (direct query).

    For cached version with prior-specific caching, use analysis.get_occupied_intervals_with_cache.
    """
    now = dt_util.utcnow()
    lookback_date = now - timedelta(days=lookback_days)
    all_intervals: list[tuple[datetime, datetime]] = []
    motion_raw: list[tuple[datetime, datetime]] = []
    media_count = 0
    appliance_count = 0
    extended_intervals: list[tuple[datetime, datetime]] = []

    try:
        start_time = dt_util.utcnow()
        with db.get_session() as session:
            base_filters = build_base_filters(db, entry_id, lookback_date, area_name)
            motion_query = build_motion_query(session, db, base_filters)
            queries = [motion_query]

            if include_media and media_sensor_ids:
                queries.append(
                    build_media_query(session, db, base_filters, media_sensor_ids)
                )

            if include_appliance and appliance_sensor_ids:
                queries.append(
                    build_appliance_query(
                        session, db, base_filters, appliance_sensor_ids
                    )
                )

            all_results = execute_union_queries(session, db, queries)
            all_intervals, motion_raw, media_count, appliance_count = (
                process_query_results(all_results)
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

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        TimeoutError,
    ) as e:
        _LOGGER.error("Error in get_occupied_intervals: %s", e)
        return []
    else:
        return extended_intervals


def get_time_bounds(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    entity_ids: list[str] | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Return min/max timestamps for specified entities or area."""
    try:
        with db.get_session() as session:
            query = session.query(
                func.min(db.Intervals.start_time).label("first"),
                func.max(db.Intervals.end_time).label("last"),
            )

            if entity_ids is not None:
                query = query.filter(
                    db.Intervals.entity_id.in_(entity_ids),
                    db.Intervals.area_name == area_name,
                )
            else:
                query = query.join(
                    db.Entities,
                    (db.Intervals.entity_id == db.Entities.entity_id)
                    & (db.Intervals.area_name == db.Entities.area_name),
                ).filter(
                    db.Entities.entry_id == entry_id,
                    db.Entities.area_name == area_name,
                )

            time_bounds = query.first()
            if not time_bounds:
                return (None, None)
            return (time_bounds.first, time_bounds.last)
    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        TimeoutError,
    ) as e:
        _LOGGER.error("Error getting time bounds: %s", e)
        return (None, None)


def build_base_filters(
    db: AreaOccupancyDB, entry_id: str, lookback_date: datetime, area_name: str
) -> list[Any]:
    """Construct base SQLAlchemy filters for interval queries."""
    return [
        db.Entities.entry_id == entry_id,
        db.Entities.area_name == area_name,
        db.Intervals.area_name == area_name,
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
            literal("motion").label("sensor_type"),
        )
        .join(
            db.Entities,
            (db.Intervals.entity_id == db.Entities.entity_id)
            & (db.Intervals.area_name == db.Entities.area_name),
        )
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
            literal("media").label("sensor_type"),
        )
        .join(
            db.Entities,
            (db.Intervals.entity_id == db.Entities.entity_id)
            & (db.Intervals.area_name == db.Entities.area_name),
        )
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
            literal("appliance").label("sensor_type"),
        )
        .join(
            db.Entities,
            (db.Intervals.entity_id == db.Entities.entity_id)
            & (db.Intervals.area_name == db.Entities.area_name),
        )
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

    select_statements = [query.statement for query in queries]
    union_query = union_all(*select_statements).subquery()
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

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Error getting global prior: %s", e)
        return None


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

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Error getting occupied intervals cache: %s", e)
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

            # Normalize datetimes for comparison (database may return with/without tzinfo)
            now = dt_util.utcnow()
            calc_date = latest.calculation_date
            # Ensure both are timezone-aware
            if calc_date.tzinfo is None:
                # If database returned naive datetime, assume it's UTC
                calc_date = calc_date.replace(tzinfo=UTC)
            if now.tzinfo is None:
                # If now is naive (shouldn't happen with dt_util.utcnow()), make it aware
                now = now.replace(tzinfo=UTC)

            age = (now - calc_date).total_seconds() / 3600
            return age < max_age_hours

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Error checking cache validity: %s", e)
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
) -> float:
    """Calculate total occupied seconds using robust Python interval merging logic.

    This method handles all complexity (timeouts, overlapping intervals, mixed sensors)
    by fetching raw intervals and processing them consistently.
    """
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
    )

    total_seconds = 0.0
    for start_time, end_time in intervals:
        total_seconds += (end_time - start_time).total_seconds()

    _LOGGER.debug(
        "Total occupied seconds (Python) for %s: %.1f", area_name, total_seconds
    )
    return total_seconds
