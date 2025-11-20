"""Tiered aggregation logic for intervals and numeric samples.

This module handles the aggregation of raw data into daily, weekly, and monthly
aggregates, and implements retention policies to prevent database bloat.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import (
    AGGREGATION_LEVEL_RAW,
    AGGREGATION_PERIOD_DAILY,
    AGGREGATION_PERIOD_MONTHLY,
    AGGREGATION_PERIOD_WEEKLY,
    RETENTION_DAILY_AGGREGATES_DAYS,
    RETENTION_MONTHLY_AGGREGATES_YEARS,
    RETENTION_RAW_INTERVALS_DAYS,
    RETENTION_RAW_NUMERIC_SAMPLES_DAYS,
    RETENTION_WEEKLY_AGGREGATES_DAYS,
)

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)

MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR


def calculate_time_slot(timestamp: datetime, slot_minutes: int) -> int:
    """Calculate the time slot number for a given timestamp."""
    hour = timestamp.hour
    minute = timestamp.minute
    return (hour * MINUTES_PER_HOUR + minute) // slot_minutes


def aggregate_intervals_by_slot(
    intervals: list[tuple[datetime, datetime]],
    slot_minutes: int,
) -> list[tuple[int, int, float]]:
    """Aggregate time intervals by day of week and time slot."""
    slot_seconds: defaultdict[tuple[int, int], float] = defaultdict(float)

    for start_time, end_time in intervals:
        current_time = start_time
        while current_time < end_time:
            day_of_week = current_time.weekday()
            slot = calculate_time_slot(current_time, slot_minutes)

            day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            slot_start = day_start + timedelta(minutes=slot * slot_minutes)
            slot_end = slot_start + timedelta(minutes=slot_minutes)

            overlap_start = max(current_time, slot_start)
            overlap_end = min(end_time, slot_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()

            if overlap_duration > 0:
                slot_seconds[(day_of_week, slot)] += overlap_duration

            current_time = slot_end

    result = []
    for (day, slot), seconds in slot_seconds.items():
        result.append((day, slot, seconds))

    return result


def get_interval_aggregates(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str,
    slot_minutes: int,
    lookback_days: int,
    motion_timeout_seconds: int,
    media_sensor_ids: list[str] | None,
    appliance_sensor_ids: list[str] | None,
) -> list[tuple[int, int, float]]:
    """Fetch interval aggregates via SQL aggregation.

    Note:
        motion_timeout_seconds parameter is accepted for API compatibility but
        cannot be applied in SQL aggregation. This function aggregates raw intervals
        without applying motion timeout extensions.

    Args:
        db: Database instance
        entry_id: Entry ID
        area_name: Area name
        slot_minutes: Time slot size in minutes
        lookback_days: Number of days to look back
        motion_timeout_seconds: Motion timeout in seconds (not applied in SQL)
        media_sensor_ids: List of media sensor IDs
        appliance_sensor_ids: List of appliance sensor IDs

    Returns:
        List of (day_of_week, time_slot, total_occupied_seconds) tuples
    """
    start_time = dt_util.utcnow()
    result = db.get_aggregated_intervals_by_slot(
        entry_id=entry_id,
        slot_minutes=slot_minutes,
        area_name=area_name,
        lookback_days=lookback_days,
        include_media=bool(media_sensor_ids),
        include_appliance=bool(appliance_sensor_ids),
        media_sensor_ids=media_sensor_ids,
        appliance_sensor_ids=appliance_sensor_ids,
    )
    query_time = (dt_util.utcnow() - start_time).total_seconds()
    _LOGGER.debug(
        "SQL aggregation completed in %.3fs for %s (slots=%d)",
        query_time,
        area_name,
        len(result),
    )
    return result


def aggregate_raw_to_daily(db: AreaOccupancyDB, area_name: str | None = None) -> int:
    """Aggregate raw intervals to daily aggregates.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.

    Returns:
        Number of daily aggregates created
    """
    _LOGGER.debug("Starting raw to daily aggregation for area: %s", area_name)

    session = None
    try:
        with db.get_locked_session() as session:
            # Calculate cutoff date (30 days ago)
            cutoff_date = dt_util.utcnow() - timedelta(
                days=RETENTION_RAW_INTERVALS_DAYS
            )

            # Find raw intervals older than cutoff that haven't been aggregated yet
            query = session.query(db.Intervals).filter(
                db.Intervals.aggregation_level == AGGREGATION_LEVEL_RAW,
                db.Intervals.start_time < cutoff_date,
            )

            if area_name:
                query = query.filter(db.Intervals.area_name == area_name)

            raw_intervals = query.all()

            if not raw_intervals:
                _LOGGER.debug("No raw intervals to aggregate to daily")
                return 0

            # Group by entity_id, state, and day
            aggregates: dict[tuple[str, str, datetime], dict[str, Any]] = {}

            for interval in raw_intervals:
                # Get start of day for period grouping
                period_start = interval.start_time.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                period_end = period_start + timedelta(days=1)

                key = (interval.entity_id, interval.state, period_start)

                if key not in aggregates:
                    aggregates[key] = {
                        "entry_id": interval.entry_id,
                        "area_name": interval.area_name,
                        "entity_id": interval.entity_id,
                        "aggregation_period": AGGREGATION_PERIOD_DAILY,
                        "period_start": period_start,
                        "period_end": period_end,
                        "state": interval.state,
                        "interval_count": 0,
                        "total_duration_seconds": 0.0,
                        "min_duration_seconds": None,
                        "max_duration_seconds": None,
                        "first_occurrence": None,
                        "last_occurrence": None,
                    }

                agg = aggregates[key]
                agg["interval_count"] += 1
                agg["total_duration_seconds"] += interval.duration_seconds

                if agg["min_duration_seconds"] is None:
                    agg["min_duration_seconds"] = interval.duration_seconds
                else:
                    agg["min_duration_seconds"] = min(
                        agg["min_duration_seconds"], interval.duration_seconds
                    )

                if agg["max_duration_seconds"] is None:
                    agg["max_duration_seconds"] = interval.duration_seconds
                else:
                    agg["max_duration_seconds"] = max(
                        agg["max_duration_seconds"], interval.duration_seconds
                    )

                if agg["first_occurrence"] is None:
                    agg["first_occurrence"] = interval.start_time
                else:
                    agg["first_occurrence"] = min(
                        agg["first_occurrence"], interval.start_time
                    )

                if agg["last_occurrence"] is None:
                    agg["last_occurrence"] = interval.end_time
                else:
                    agg["last_occurrence"] = max(
                        agg["last_occurrence"], interval.end_time
                    )

            # Calculate averages and create aggregate records
            created_count = 0
            for agg_data in aggregates.values():
                # Calculate average duration
                if agg_data["interval_count"] > 0:
                    agg_data["avg_duration_seconds"] = (
                        agg_data["total_duration_seconds"] / agg_data["interval_count"]
                    )

                # Check if aggregate already exists
                existing = (
                    session.query(db.IntervalAggregates)
                    .filter_by(
                        entity_id=agg_data["entity_id"],
                        aggregation_period=AGGREGATION_PERIOD_DAILY,
                        period_start=agg_data["period_start"],
                        state=agg_data["state"],
                    )
                    .first()
                )

                if not existing:
                    aggregate = db.IntervalAggregates(**agg_data)
                    session.add(aggregate)
                    created_count += 1

            # Delete raw intervals that were aggregated
            interval_ids = [interval.id for interval in raw_intervals]
            if interval_ids:
                session.query(db.Intervals).filter(
                    db.Intervals.id.in_(interval_ids)
                ).delete(synchronize_session=False)

            session.commit()
            _LOGGER.info(
                "Created %d daily aggregates from %d raw intervals for area: %s",
                created_count,
                len(raw_intervals),
                area_name or "all areas",
            )

            return created_count

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error aggregating raw to daily: %s", e)
        if session is not None:
            session.rollback()
        raise


def aggregate_daily_to_weekly(db: AreaOccupancyDB, area_name: str | None = None) -> int:
    """Aggregate daily aggregates to weekly aggregates.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.

    Returns:
        Number of weekly aggregates created
    """
    _LOGGER.debug("Starting daily to weekly aggregation for area: %s", area_name)

    session = None
    try:
        with db.get_locked_session() as session:
            # Calculate cutoff date (90 days ago)
            cutoff_date = dt_util.utcnow() - timedelta(
                days=RETENTION_DAILY_AGGREGATES_DAYS
            )

            # Find daily aggregates older than cutoff
            query = session.query(db.IntervalAggregates).filter(
                db.IntervalAggregates.aggregation_period == AGGREGATION_PERIOD_DAILY,
                db.IntervalAggregates.period_start < cutoff_date,
            )

            if area_name:
                query = query.filter(db.IntervalAggregates.area_name == area_name)

            daily_aggregates = query.all()

            if not daily_aggregates:
                _LOGGER.debug("No daily aggregates to aggregate to weekly")
                return 0

            # Group by entity_id, state, and week
            aggregates: dict[tuple[str, str, datetime], dict[str, Any]] = {}

            for daily in daily_aggregates:
                # Get start of week (Monday) for period grouping
                days_since_monday = daily.period_start.weekday()
                week_start = daily.period_start - timedelta(days=days_since_monday)
                week_start = week_start.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                week_end = week_start + timedelta(days=7)

                key = (daily.entity_id, daily.state, week_start)

                if key not in aggregates:
                    aggregates[key] = {
                        "entry_id": daily.entry_id,
                        "area_name": daily.area_name,
                        "entity_id": daily.entity_id,
                        "aggregation_period": AGGREGATION_PERIOD_WEEKLY,
                        "period_start": week_start,
                        "period_end": week_end,
                        "state": daily.state,
                        "interval_count": 0,
                        "total_duration_seconds": 0.0,
                        "min_duration_seconds": None,
                        "max_duration_seconds": None,
                        "first_occurrence": None,
                        "last_occurrence": None,
                    }

                agg = aggregates[key]
                agg["interval_count"] += daily.interval_count
                agg["total_duration_seconds"] += daily.total_duration_seconds

                if agg["min_duration_seconds"] is None:
                    agg["min_duration_seconds"] = daily.min_duration_seconds
                elif daily.min_duration_seconds is not None:
                    agg["min_duration_seconds"] = min(
                        agg["min_duration_seconds"], daily.min_duration_seconds
                    )

                if agg["max_duration_seconds"] is None:
                    agg["max_duration_seconds"] = daily.max_duration_seconds
                elif daily.max_duration_seconds is not None:
                    agg["max_duration_seconds"] = max(
                        agg["max_duration_seconds"], daily.max_duration_seconds
                    )

                if agg["first_occurrence"] is None:
                    agg["first_occurrence"] = daily.first_occurrence
                elif daily.first_occurrence is not None:
                    agg["first_occurrence"] = min(
                        agg["first_occurrence"], daily.first_occurrence
                    )

                if agg["last_occurrence"] is None:
                    agg["last_occurrence"] = daily.last_occurrence
                elif daily.last_occurrence is not None:
                    agg["last_occurrence"] = max(
                        agg["last_occurrence"], daily.last_occurrence
                    )

            # Calculate averages and create aggregate records
            created_count = 0
            for agg_data in aggregates.values():
                # Calculate average duration
                if agg_data["interval_count"] > 0:
                    agg_data["avg_duration_seconds"] = (
                        agg_data["total_duration_seconds"] / agg_data["interval_count"]
                    )

                # Check if aggregate already exists
                existing = (
                    session.query(db.IntervalAggregates)
                    .filter_by(
                        entity_id=agg_data["entity_id"],
                        aggregation_period=AGGREGATION_PERIOD_WEEKLY,
                        period_start=agg_data["period_start"],
                        state=agg_data["state"],
                    )
                    .first()
                )

                if not existing:
                    aggregate = db.IntervalAggregates(**agg_data)
                    session.add(aggregate)
                    created_count += 1

            # Delete daily aggregates that were aggregated
            aggregate_ids = [daily.id for daily in daily_aggregates]
            if aggregate_ids:
                session.query(db.IntervalAggregates).filter(
                    db.IntervalAggregates.id.in_(aggregate_ids)
                ).delete(synchronize_session=False)

            session.commit()
            _LOGGER.info(
                "Created %d weekly aggregates from %d daily aggregates for area: %s",
                created_count,
                len(daily_aggregates),
                area_name or "all areas",
            )

            return created_count

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error aggregating daily to weekly: %s", e)
        if session is not None:
            session.rollback()
        raise


def aggregate_weekly_to_monthly(
    db: AreaOccupancyDB, area_name: str | None = None
) -> int:
    """Aggregate weekly aggregates to monthly aggregates.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.

    Returns:
        Number of monthly aggregates created
    """
    _LOGGER.debug("Starting weekly to monthly aggregation for area: %s", area_name)

    session = None
    try:
        with db.get_locked_session() as session:
            # Calculate cutoff date (365 days ago)
            cutoff_date = dt_util.utcnow() - timedelta(
                days=RETENTION_WEEKLY_AGGREGATES_DAYS
            )

            # Find weekly aggregates older than cutoff
            query = session.query(db.IntervalAggregates).filter(
                db.IntervalAggregates.aggregation_period == AGGREGATION_PERIOD_WEEKLY,
                db.IntervalAggregates.period_start < cutoff_date,
            )

            if area_name:
                query = query.filter(db.IntervalAggregates.area_name == area_name)

            weekly_aggregates = query.all()

            if not weekly_aggregates:
                _LOGGER.debug("No weekly aggregates to aggregate to monthly")
                return 0

            # Group by entity_id, state, and month
            aggregates: dict[tuple[str, str, datetime], dict[str, Any]] = {}

            for weekly in weekly_aggregates:
                # Get start of month for period grouping
                month_start = weekly.period_start.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                # Calculate end of month
                if month_start.month == 12:
                    month_end = month_start.replace(year=month_start.year + 1, month=1)
                else:
                    month_end = month_start.replace(month=month_start.month + 1)

                key = (weekly.entity_id, weekly.state, month_start)

                if key not in aggregates:
                    aggregates[key] = {
                        "entry_id": weekly.entry_id,
                        "area_name": weekly.area_name,
                        "entity_id": weekly.entity_id,
                        "aggregation_period": AGGREGATION_PERIOD_MONTHLY,
                        "period_start": month_start,
                        "period_end": month_end,
                        "state": weekly.state,
                        "interval_count": 0,
                        "total_duration_seconds": 0.0,
                        "min_duration_seconds": None,
                        "max_duration_seconds": None,
                        "first_occurrence": None,
                        "last_occurrence": None,
                    }

                agg = aggregates[key]
                agg["interval_count"] += weekly.interval_count
                agg["total_duration_seconds"] += weekly.total_duration_seconds

                if agg["min_duration_seconds"] is None:
                    agg["min_duration_seconds"] = weekly.min_duration_seconds
                elif weekly.min_duration_seconds is not None:
                    agg["min_duration_seconds"] = min(
                        agg["min_duration_seconds"], weekly.min_duration_seconds
                    )

                if agg["max_duration_seconds"] is None:
                    agg["max_duration_seconds"] = weekly.max_duration_seconds
                elif weekly.max_duration_seconds is not None:
                    agg["max_duration_seconds"] = max(
                        agg["max_duration_seconds"], weekly.max_duration_seconds
                    )

                if agg["first_occurrence"] is None:
                    agg["first_occurrence"] = weekly.first_occurrence
                elif weekly.first_occurrence is not None:
                    agg["first_occurrence"] = min(
                        agg["first_occurrence"], weekly.first_occurrence
                    )

                if agg["last_occurrence"] is None:
                    agg["last_occurrence"] = weekly.last_occurrence
                elif weekly.last_occurrence is not None:
                    agg["last_occurrence"] = max(
                        agg["last_occurrence"], weekly.last_occurrence
                    )

            # Calculate averages and create aggregate records
            created_count = 0
            new_aggregates = []
            for agg_data in aggregates.values():
                # Calculate average duration
                if agg_data["interval_count"] > 0:
                    agg_data["avg_duration_seconds"] = (
                        agg_data["total_duration_seconds"] / agg_data["interval_count"]
                    )

                # Check if aggregate already exists
                existing = (
                    session.query(db.IntervalAggregates)
                    .filter_by(
                        entity_id=agg_data["entity_id"],
                        aggregation_period=AGGREGATION_PERIOD_MONTHLY,
                        period_start=agg_data["period_start"],
                        state=agg_data["state"],
                    )
                    .first()
                )

                if not existing:
                    aggregate = db.IntervalAggregates(**agg_data)
                    new_aggregates.append(aggregate)
                    created_count += 1

            # Add all new aggregates at once
            if new_aggregates:
                session.add_all(new_aggregates)
                session.flush()  # Flush before deleting to avoid identity map conflicts

            # Delete weekly aggregates that were aggregated
            aggregate_ids = [weekly.id for weekly in weekly_aggregates]
            if aggregate_ids:
                session.query(db.IntervalAggregates).filter(
                    db.IntervalAggregates.id.in_(aggregate_ids)
                ).delete(synchronize_session=False)

            session.commit()
            _LOGGER.info(
                "Created %d monthly aggregates from %d weekly aggregates for area: %s",
                created_count,
                len(weekly_aggregates),
                area_name or "all areas",
            )

            return created_count

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error aggregating weekly to monthly: %s", e)
        if session is not None:
            session.rollback()
        raise


def run_interval_aggregation(
    db: AreaOccupancyDB, area_name: str | None = None, force: bool = False
) -> dict[str, int]:
    """Run the full tiered aggregation process for intervals.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.
        force: If True, run aggregation even if recently run

    Returns:
        Dictionary with counts of aggregates created at each level
    """
    _LOGGER.info(
        "Running tiered interval aggregation for area: %s", area_name or "all areas"
    )

    results = {
        "daily": 0,
        "weekly": 0,
        "monthly": 0,
    }

    try:
        # Step 1: Aggregate raw to daily
        results["daily"] = aggregate_raw_to_daily(db, area_name)

        # Step 2: Aggregate daily to weekly
        results["weekly"] = aggregate_daily_to_weekly(db, area_name)

        # Step 3: Aggregate weekly to monthly
        results["monthly"] = aggregate_weekly_to_monthly(db, area_name)

        _LOGGER.info(
            "Interval aggregation complete: %d daily, %d weekly, %d monthly aggregates created",
            results["daily"],
            results["weekly"],
            results["monthly"],
        )

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error during interval aggregation: %s", e)
        raise

    return results


def prune_old_aggregates(
    db: AreaOccupancyDB, area_name: str | None = None
) -> dict[str, int]:
    """Prune old aggregates based on retention policies.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.

    Returns:
        Dictionary with counts of aggregates deleted at each level
    """
    _LOGGER.debug("Pruning old aggregates for area: %s", area_name or "all areas")

    results = {
        "daily": 0,
        "weekly": 0,
        "monthly": 0,
    }

    session = None
    try:
        with db.get_locked_session() as session:
            now = dt_util.utcnow()

            # Prune daily aggregates older than retention period
            daily_cutoff = now - timedelta(days=RETENTION_DAILY_AGGREGATES_DAYS)
            daily_query = session.query(db.IntervalAggregates).filter(
                db.IntervalAggregates.aggregation_period == AGGREGATION_PERIOD_DAILY,
                db.IntervalAggregates.period_start < daily_cutoff,
            )
            if area_name:
                daily_query = daily_query.filter(
                    db.IntervalAggregates.area_name == area_name
                )
            results["daily"] = daily_query.delete(synchronize_session=False)

            # Prune weekly aggregates older than retention period
            weekly_cutoff = now - timedelta(days=RETENTION_WEEKLY_AGGREGATES_DAYS)
            weekly_query = session.query(db.IntervalAggregates).filter(
                db.IntervalAggregates.aggregation_period == AGGREGATION_PERIOD_WEEKLY,
                db.IntervalAggregates.period_start < weekly_cutoff,
            )
            if area_name:
                weekly_query = weekly_query.filter(
                    db.IntervalAggregates.area_name == area_name
                )
            results["weekly"] = weekly_query.delete(synchronize_session=False)

            # Prune monthly aggregates older than retention period
            monthly_cutoff = now - timedelta(
                days=RETENTION_MONTHLY_AGGREGATES_YEARS * 365
            )
            monthly_query = session.query(db.IntervalAggregates).filter(
                db.IntervalAggregates.aggregation_period == AGGREGATION_PERIOD_MONTHLY,
                db.IntervalAggregates.period_start < monthly_cutoff,
            )
            if area_name:
                monthly_query = monthly_query.filter(
                    db.IntervalAggregates.area_name == area_name
                )
            results["monthly"] = monthly_query.delete(synchronize_session=False)

            session.commit()

            _LOGGER.info(
                "Pruned old aggregates: %d daily, %d weekly, %d monthly",
                results["daily"],
                results["weekly"],
                results["monthly"],
            )

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error pruning old aggregates: %s", e)
        if session is not None:
            session.rollback()
        raise

    return results


def prune_old_numeric_samples(db: AreaOccupancyDB, area_name: str | None = None) -> int:
    """Prune old raw numeric samples based on retention policy.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.

    Returns:
        Number of samples deleted
    """
    _LOGGER.debug("Pruning old numeric samples for area: %s", area_name or "all areas")

    session = None
    try:
        with db.get_locked_session() as session:
            cutoff_date = dt_util.utcnow() - timedelta(
                days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS
            )

            query = session.query(db.NumericSamples).filter(
                db.NumericSamples.timestamp < cutoff_date
            )

            if area_name:
                query = query.filter(db.NumericSamples.area_name == area_name)

            deleted_count = query.delete(synchronize_session=False)
            session.commit()

            _LOGGER.info(
                "Pruned %d old numeric samples for area: %s",
                deleted_count,
                area_name or "all areas",
            )

            return deleted_count

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error pruning old numeric samples: %s", e)
        if session is not None:
            session.rollback()
        raise
