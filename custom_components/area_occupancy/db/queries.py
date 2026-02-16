"""Database query operations."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import literal

from homeassistant.util import dt as dt_util

from ..const import DEFAULT_TIME_PRIOR
from ..data.entity_type import InputType
from ..time_utils import from_db_utc, to_db_utc, to_utc
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
                # DB stores naive UTC; convert to aware UTC for runtime/recorder APIs
                return from_db_utc(result) - timedelta(hours=1)
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


def get_all_time_priors(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    default_prior: float = DEFAULT_TIME_PRIOR,
) -> dict[tuple[int, int], float]:
    """Get all time priors for an area (all 168 slots).

    Args:
        db: Database instance
        entry_id: The area entry ID to filter by
        area_name: The area name to filter by
        default_prior: Default prior value for slots not found

    Returns:
        Dictionary mapping (day_of_week, time_slot) to prior_value.
        All 168 slots are included, using default_prior for missing slots.
    """
    try:
        with db.get_session() as session:
            priors = (
                session.query(db.Priors)
                .filter_by(
                    entry_id=entry_id,
                    area_name=area_name,
                )
                .all()
            )

            # Build dictionary from database results
            result: dict[tuple[int, int], float] = {}
            for prior in priors:
                slot_key = (prior.day_of_week, prior.time_slot)
                result[slot_key] = float(prior.prior_value)

            # Fill in missing slots with default (all 168 slots: 7 days × 24 hours)
            for day_of_week in range(7):  # 0-6 (Monday-Sunday)
                for time_slot in range(24):  # 0-23 (hourly slots)
                    slot_key = (day_of_week, time_slot)
                    if slot_key not in result:
                        result[slot_key] = default_prior

            return result

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error getting all time priors: %s", e)
        # Return default dict with all slots set to default_prior
        return {
            (day_of_week, time_slot): default_prior
            for day_of_week in range(7)
            for time_slot in range(24)
        }


def get_occupied_intervals(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    lookback_days: int,
    motion_timeout_seconds: int,
) -> list[tuple[datetime, datetime]]:
    """Fetch occupied intervals from presence sensors (direct query).

    Occupied intervals are determined by motion, sleep, and media sensors
    to provide comprehensive ground truth for prior calculations and
    correlation analysis. Motion timeout is only applied to motion intervals.

    For cached version with prior-specific caching, use analysis.get_occupied_intervals_with_cache.
    """
    now = dt_util.utcnow()
    lookback_date_db = to_db_utc(now - timedelta(days=lookback_days))
    all_intervals: list[tuple[datetime, datetime]] = []
    motion_raw: list[tuple[datetime, datetime]] = []
    extended_intervals: list[tuple[datetime, datetime]] = []

    try:
        start_time = dt_util.utcnow()
        with db.get_session() as session:
            base_filters = build_base_filters(db, entry_id, lookback_date_db, area_name)
            motion_query = build_motion_query(session, db, base_filters)
            presence_query = build_presence_query(session, db, base_filters)

            all_results = execute_union_queries(
                session, db, [motion_query, presence_query]
            )
            all_intervals, motion_raw = process_query_results(all_results)

        query_time = (dt_util.utcnow() - start_time).total_seconds()
        _LOGGER.debug(
            "Interval query executed in %.3fs for %s (total=%d, motion=%d)",
            query_time,
            area_name,
            len(all_intervals),
            len(motion_raw),
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
            # DB stores naive UTC; return aware UTC to callers
            first = time_bounds.first
            last = time_bounds.last
            return (
                from_db_utc(first) if first is not None else None,
                from_db_utc(last) if last is not None else None,
            )
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
    db: AreaOccupancyDB, entry_id: str, lookback_date_db: datetime, area_name: str
) -> list[Any]:
    """Construct base SQLAlchemy filters for interval queries."""
    return [
        db.Entities.entry_id == entry_id,
        db.Entities.area_name == area_name,
        db.Intervals.area_name == area_name,
        # DB stores naive UTC
        db.Intervals.start_time >= lookback_date_db,
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


def build_presence_query(
    session: Any, db: AreaOccupancyDB, base_filters: list[Any]
) -> Any:
    """Create query selecting sleep and media presence intervals.

    These sensors indicate sustained presence (e.g., sleeping, watching TV)
    that motion sensors may miss.
    """
    return (
        session.query(
            db.Intervals.start_time,
            db.Intervals.end_time,
            literal("presence").label("sensor_type"),
        )
        .join(
            db.Entities,
            (db.Intervals.entity_id == db.Entities.entity_id)
            & (db.Intervals.area_name == db.Entities.area_name),
        )
        .filter(
            *base_filters,
            db.Entities.entity_type.in_([InputType.SLEEP.value, InputType.MEDIA.value]),
            db.Intervals.state == "on",
        )
    )


def execute_union_queries(session: Any, db: AreaOccupancyDB, queries: list[Any]) -> Any:
    """Execute and union multiple interval queries, returning ordered results."""
    # Filter out None queries (presence query may return no results)
    valid_queries = [q for q in queries if q is not None]
    if not valid_queries:
        return []

    if len(valid_queries) == 1:
        combined_query = valid_queries[0].order_by(db.Intervals.start_time)
        return combined_query.all()

    # Union all queries then order
    combined = valid_queries[0].union_all(*valid_queries[1:])
    return combined.order_by(db.Intervals.start_time).all()


def process_query_results(
    results: list[tuple[datetime, datetime, str]],
) -> tuple[list[tuple[datetime, datetime]], list[tuple[datetime, datetime]]]:
    """Process query results into all intervals and motion-only intervals.

    Motion intervals are tracked separately because apply_motion_timeout()
    only extends motion segments (not sleep/media presence).
    """
    motion_raw: list[tuple[datetime, datetime]] = []
    all_intervals: list[tuple[datetime, datetime]] = []

    for start, end, sensor_type in results:
        # DB stores naive UTC; convert to aware UTC for runtime computations
        interval = (from_db_utc(start), from_db_utc(end))
        all_intervals.append(interval)
        if sensor_type == "motion":
            motion_raw.append(interval)

    return (all_intervals, motion_raw)


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
                    db.OccupiedIntervalsCache.start_time >= to_db_utc(period_start)
                )
            if period_end:
                query = query.filter(
                    db.OccupiedIntervalsCache.end_time <= to_db_utc(period_end)
                )

            cached_intervals = query.order_by(
                db.OccupiedIntervalsCache.start_time
            ).all()

            return [
                (from_db_utc(interval.start_time), from_db_utc(interval.end_time))
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
            # DB stores naive UTC; compare using aware UTC
            now = to_utc(dt_util.utcnow())
            calc_date = from_db_utc(latest.calculation_date)

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
) -> float:
    """Calculate total occupied seconds using robust Python interval merging logic.

    This method handles all complexity (timeouts, overlapping intervals) by fetching
    raw motion sensor intervals and processing them consistently.
    """
    intervals = get_occupied_intervals(
        db,
        entry_id,
        area_name,
        lookback_days,
        motion_timeout_seconds,
    )

    total_seconds = 0.0
    for start_time, end_time in intervals:
        total_seconds += (end_time - start_time).total_seconds()

    _LOGGER.debug(
        "Total occupied seconds (Python) for %s: %.1f", area_name, total_seconds
    )
    return total_seconds
