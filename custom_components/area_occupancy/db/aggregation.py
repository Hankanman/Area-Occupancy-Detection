"""Tiered aggregation logic for intervals and numeric samples.

This module handles the aggregation of raw data into daily, weekly, and monthly
aggregates, and implements retention policies to prevent database bloat.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import statistics
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import (
    AGGREGATION_LEVEL_RAW,
    AGGREGATION_PERIOD_DAILY,
    AGGREGATION_PERIOD_HOURLY,
    AGGREGATION_PERIOD_MONTHLY,
    AGGREGATION_PERIOD_WEEKLY,
    RETENTION_DAILY_AGGREGATES_DAYS,
    RETENTION_HOURLY_NUMERIC_DAYS,
    RETENTION_MONTHLY_AGGREGATES_YEARS,
    RETENTION_RAW_INTERVALS_DAYS,
    RETENTION_RAW_NUMERIC_SAMPLES_DAYS,
    RETENTION_WEEKLY_AGGREGATES_DAYS,
    RETENTION_WEEKLY_NUMERIC_YEARS,
)
from ..time_utils import from_db_utc, to_db_utc, to_local
from .utils import batched_delete_by_ids

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)

MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR


def aggregate_raw_to_daily(
    db: AreaOccupancyDB, area_name: str | None = None
) -> tuple[int, list[int]]:
    """Aggregate raw intervals to daily aggregates.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.

    Returns:
        Tuple of (number of daily aggregates created, list of created aggregate IDs)
    """
    _LOGGER.debug("Starting raw to daily aggregation for area: %s", area_name)

    try:
        with db.get_session() as session:
            # Calculate cutoff date (30 days ago)
            cutoff_date = to_db_utc(
                dt_util.utcnow() - timedelta(days=RETENTION_RAW_INTERVALS_DAYS)
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
                return 0, []

            # Group by entity_id, state, and day
            aggregates: dict[tuple[str, str, datetime], dict[str, Any]] = {}

            for interval in raw_intervals:
                # DB stores naive UTC; bucket by local day boundary and persist naive UTC.
                interval_start_local = to_local(from_db_utc(interval.start_time))
                day_start_local = interval_start_local.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                period_start = to_db_utc(day_start_local)
                period_end = to_db_utc(day_start_local + timedelta(days=1))

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
            created_ids: list[int] = []
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
                    session.flush()  # Flush to get the ID
                    created_count += 1
                    created_ids.append(aggregate.id)

            # Delete raw intervals that were aggregated (batched to avoid SQLite limit)
            interval_ids = [interval.id for interval in raw_intervals]
            batched_delete_by_ids(session, db.Intervals, interval_ids)

            session.commit()
            _LOGGER.info(
                "Created %d daily aggregates from %d raw intervals for area: %s",
                created_count,
                len(raw_intervals),
                area_name or "all areas",
            )

            return created_count, created_ids

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error aggregating raw to daily: %s", e)
        raise


def aggregate_daily_to_weekly(
    db: AreaOccupancyDB,
    area_name: str | None = None,
    exclude_daily_ids: set[int] | None = None,
) -> tuple[int, list[int]]:
    """Aggregate daily aggregates to weekly aggregates.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.
        exclude_daily_ids: Optional set of daily aggregate IDs to exclude from aggregation.
                          Used to prevent cascading aggregation in the same run.

    Returns:
        Tuple of (number of weekly aggregates created, list of created aggregate IDs)
    """
    _LOGGER.debug("Starting daily to weekly aggregation for area: %s", area_name)

    try:
        with db.get_session() as session:
            # Calculate cutoff date (90 days ago)
            cutoff_date = to_db_utc(
                dt_util.utcnow() - timedelta(days=RETENTION_DAILY_AGGREGATES_DAYS)
            )

            # Find daily aggregates older than cutoff
            query = session.query(db.IntervalAggregates).filter(
                db.IntervalAggregates.aggregation_period == AGGREGATION_PERIOD_DAILY,
                db.IntervalAggregates.period_start < cutoff_date,
            )

            if area_name:
                query = query.filter(db.IntervalAggregates.area_name == area_name)

            # Exclude daily aggregates created in the current run to prevent cascading aggregation
            if exclude_daily_ids:
                query = query.filter(~db.IntervalAggregates.id.in_(exclude_daily_ids))

            daily_aggregates = query.all()

            if not daily_aggregates:
                _LOGGER.debug("No daily aggregates to aggregate to weekly")
                return 0, []

            # Group by entity_id, state, and week
            aggregates: dict[tuple[str, str, datetime], dict[str, Any]] = {}

            for daily in daily_aggregates:
                # DB stores naive UTC; bucket by local week boundary and persist naive UTC.
                daily_start_local = to_local(from_db_utc(daily.period_start))
                days_since_monday = daily_start_local.weekday()
                week_start_local = (
                    daily_start_local - timedelta(days=days_since_monday)
                ).replace(hour=0, minute=0, second=0, microsecond=0)
                week_end_local = week_start_local + timedelta(days=7)

                week_start = to_db_utc(week_start_local)
                week_end = to_db_utc(week_end_local)

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
            created_ids: list[int] = []
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
                    session.flush()  # Flush to get the ID
                    created_count += 1
                    created_ids.append(aggregate.id)

            # Delete daily aggregates that were aggregated (batched to avoid SQLite limit)
            aggregate_ids = [daily.id for daily in daily_aggregates]
            batched_delete_by_ids(session, db.IntervalAggregates, aggregate_ids)

            session.commit()
            _LOGGER.info(
                "Created %d weekly aggregates from %d daily aggregates for area: %s",
                created_count,
                len(daily_aggregates),
                area_name or "all areas",
            )

            return created_count, created_ids

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error aggregating daily to weekly: %s", e)
        raise


def aggregate_weekly_to_monthly(
    db: AreaOccupancyDB,
    area_name: str | None = None,
    exclude_weekly_ids: set[int] | None = None,
) -> int:
    """Aggregate weekly aggregates to monthly aggregates.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.
        exclude_weekly_ids: Optional set of weekly aggregate IDs to exclude from aggregation.
                          Used to prevent cascading aggregation in the same run.

    Returns:
        Number of monthly aggregates created
    """
    _LOGGER.debug("Starting weekly to monthly aggregation for area: %s", area_name)

    try:
        with db.get_session() as session:
            # Calculate cutoff date (365 days ago)
            cutoff_date = to_db_utc(
                dt_util.utcnow() - timedelta(days=RETENTION_WEEKLY_AGGREGATES_DAYS)
            )

            # Find weekly aggregates older than cutoff
            query = session.query(db.IntervalAggregates).filter(
                db.IntervalAggregates.aggregation_period == AGGREGATION_PERIOD_WEEKLY,
                db.IntervalAggregates.period_start < cutoff_date,
            )

            if area_name:
                query = query.filter(db.IntervalAggregates.area_name == area_name)

            # Exclude weekly aggregates created in the current run to prevent cascading aggregation
            if exclude_weekly_ids:
                query = query.filter(~db.IntervalAggregates.id.in_(exclude_weekly_ids))

            weekly_aggregates = query.all()

            if not weekly_aggregates:
                _LOGGER.debug("No weekly aggregates to aggregate to monthly")
                return 0

            # Group by entity_id, state, and month
            aggregates: dict[tuple[str, str, datetime], dict[str, Any]] = {}

            for weekly in weekly_aggregates:
                # DB stores naive UTC; bucket by local month boundary and persist naive UTC.
                weekly_start_local = to_local(from_db_utc(weekly.period_start))
                month_start_local = weekly_start_local.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                if month_start_local.month == 12:
                    month_end_local = month_start_local.replace(
                        year=month_start_local.year + 1, month=1
                    )
                else:
                    month_end_local = month_start_local.replace(
                        month=month_start_local.month + 1
                    )
                month_start = to_db_utc(month_start_local)
                month_end = to_db_utc(month_end_local)

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

            # Delete weekly aggregates that were aggregated (batched to avoid SQLite limit)
            aggregate_ids = [weekly.id for weekly in weekly_aggregates]
            batched_delete_by_ids(session, db.IntervalAggregates, aggregate_ids)

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
    _LOGGER.debug(
        "Running tiered interval aggregation for area: %s", area_name or "all areas"
    )

    results = {
        "daily": 0,
        "weekly": 0,
        "monthly": 0,
    }

    try:
        # Step 1: Aggregate raw to daily
        daily_count, daily_ids = aggregate_raw_to_daily(db, area_name)
        results["daily"] = daily_count

        # Step 2: Aggregate daily to weekly (exclude daily aggregates created in step 1)
        weekly_count, weekly_ids = aggregate_daily_to_weekly(
            db, area_name, exclude_daily_ids=set(daily_ids) if daily_ids else None
        )
        results["weekly"] = weekly_count

        # Step 3: Aggregate weekly to monthly (exclude weekly aggregates created in step 2)
        results["monthly"] = aggregate_weekly_to_monthly(
            db, area_name, exclude_weekly_ids=set(weekly_ids) if weekly_ids else None
        )

        _LOGGER.debug(
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

    try:
        with db.get_session() as session:
            now = dt_util.utcnow()

            # Prune daily aggregates older than retention period
            daily_cutoff = to_db_utc(
                now - timedelta(days=RETENTION_DAILY_AGGREGATES_DAYS)
            )
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
            weekly_cutoff = to_db_utc(
                now - timedelta(days=RETENTION_WEEKLY_AGGREGATES_DAYS)
            )
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
            monthly_cutoff = to_db_utc(
                now - timedelta(days=RETENTION_MONTHLY_AGGREGATES_YEARS * 365)
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

    try:
        with db.get_session() as session:
            cutoff_date = to_db_utc(
                dt_util.utcnow() - timedelta(days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS)
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
        raise


def aggregate_numeric_samples_to_hourly(
    db: AreaOccupancyDB, area_name: str | None = None
) -> tuple[int, list[int]]:
    """Aggregate raw numeric samples to hourly aggregates.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.

    Returns:
        Tuple of (number of hourly aggregates created, list of created aggregate IDs)
    """
    _LOGGER.debug(
        "Starting numeric samples to hourly aggregation for area: %s", area_name
    )

    try:
        with db.get_session() as session:
            # Calculate cutoff date
            cutoff_date = to_db_utc(
                dt_util.utcnow() - timedelta(days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS)
            )

            # Find raw samples older than cutoff
            query = session.query(db.NumericSamples).filter(
                db.NumericSamples.timestamp < cutoff_date,
            )

            if area_name:
                query = query.filter(db.NumericSamples.area_name == area_name)

            raw_samples = query.all()

            if not raw_samples:
                _LOGGER.debug("No numeric samples to aggregate to hourly")
                return 0, []

            # Group by entity_id and hour
            aggregates: dict[tuple[str, datetime], dict[str, Any]] = {}
            sample_ids_by_hour: dict[tuple[str, datetime], list[int]] = {}

            for sample in raw_samples:
                # DB stores naive UTC; bucket by local hour boundary and persist naive UTC.
                sample_local = to_local(from_db_utc(sample.timestamp))
                hour_start_local = sample_local.replace(
                    minute=0, second=0, microsecond=0
                )
                period_start = to_db_utc(hour_start_local)
                period_end = to_db_utc(hour_start_local + timedelta(hours=1))

                key = (sample.entity_id, period_start)

                if key not in aggregates:
                    aggregates[key] = {
                        "entry_id": sample.entry_id,
                        "area_name": sample.area_name,
                        "entity_id": sample.entity_id,
                        "aggregation_period": AGGREGATION_PERIOD_HOURLY,
                        "period_start": period_start,
                        "period_end": period_end,
                        "values": [],
                        "sample_count": 0,
                        "first_value": None,
                        "last_value": None,
                    }
                    sample_ids_by_hour[key] = []

                agg = aggregates[key]
                value = float(sample.value)
                agg["values"].append(value)
                agg["sample_count"] += 1
                sample_ids_by_hour[key].append(sample.id)

                if agg["first_value"] is None:
                    agg["first_value"] = value
                agg["last_value"] = value

            # Calculate statistics and create aggregate records
            created_count = 0
            created_ids: list[int] = []
            for agg_data in aggregates.values():
                values = agg_data["values"]
                if not values:
                    continue

                # Calculate statistics
                agg_data["min_value"] = min(values)
                agg_data["max_value"] = max(values)
                agg_data["avg_value"] = statistics.mean(values)
                agg_data["median_value"] = statistics.median(values)
                agg_data["sample_count"] = len(values)

                # Calculate standard deviation
                if len(values) > 1:
                    agg_data["std_deviation"] = statistics.stdev(values)
                else:
                    agg_data["std_deviation"] = 0.0

                # Remove values list (not needed in database)
                del agg_data["values"]

                # Check if aggregate already exists
                existing = (
                    session.query(db.NumericAggregates)
                    .filter_by(
                        entity_id=agg_data["entity_id"],
                        aggregation_period=AGGREGATION_PERIOD_HOURLY,
                        period_start=agg_data["period_start"],
                    )
                    .first()
                )

                if not existing:
                    aggregate = db.NumericAggregates(**agg_data)
                    session.add(aggregate)
                    session.flush()  # Flush to get the ID
                    created_count += 1
                    created_ids.append(aggregate.id)

            # Delete raw samples that were aggregated (batched to avoid SQLite limit)
            all_sample_ids: list[int] = []
            for sample_ids in sample_ids_by_hour.values():
                all_sample_ids.extend(sample_ids)
            batched_delete_by_ids(session, db.NumericSamples, all_sample_ids)

            session.commit()
            _LOGGER.info(
                "Created %d hourly aggregates from %d numeric samples for area: %s",
                created_count,
                len(raw_samples),
                area_name or "all areas",
            )

            return created_count, created_ids

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error aggregating numeric samples to hourly: %s", e)
        raise


def aggregate_hourly_to_weekly(
    db: AreaOccupancyDB,
    area_name: str | None = None,
    exclude_hourly_ids: set[int] | None = None,
) -> tuple[int, list[int]]:
    """Aggregate hourly aggregates to weekly aggregates.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.
        exclude_hourly_ids: Optional set of hourly aggregate IDs to exclude from aggregation.
                          Used to prevent cascading aggregation in the same run.

    Returns:
        Tuple of (number of weekly aggregates created, list of created aggregate IDs)
    """
    _LOGGER.debug("Starting hourly to weekly aggregation for area: %s", area_name)

    try:
        with db.get_session() as session:
            # Calculate cutoff date
            cutoff_date = to_db_utc(
                dt_util.utcnow() - timedelta(days=RETENTION_HOURLY_NUMERIC_DAYS)
            )

            # Find hourly aggregates older than cutoff
            query = session.query(db.NumericAggregates).filter(
                db.NumericAggregates.aggregation_period == AGGREGATION_PERIOD_HOURLY,
                db.NumericAggregates.period_start < cutoff_date,
            )

            if area_name:
                query = query.filter(db.NumericAggregates.area_name == area_name)

            # Exclude hourly aggregates created in the current run to prevent cascading aggregation
            if exclude_hourly_ids:
                query = query.filter(~db.NumericAggregates.id.in_(exclude_hourly_ids))

            hourly_aggregates = query.all()

            if not hourly_aggregates:
                _LOGGER.debug("No hourly aggregates to aggregate to weekly")
                return 0, []

            # Group by entity_id and week
            aggregates: dict[tuple[str, datetime], dict[str, Any]] = {}

            for hourly in hourly_aggregates:
                # DB stores naive UTC; bucket by local week boundary and persist naive UTC.
                hourly_start_local = to_local(from_db_utc(hourly.period_start))
                days_since_monday = hourly_start_local.weekday()
                week_start_local = (
                    hourly_start_local - timedelta(days=days_since_monday)
                ).replace(hour=0, minute=0, second=0, microsecond=0)
                week_end_local = week_start_local + timedelta(days=7)

                week_start = to_db_utc(week_start_local)
                week_end = to_db_utc(week_end_local)

                key = (hourly.entity_id, week_start)

                if key not in aggregates:
                    aggregates[key] = {
                        "entry_id": hourly.entry_id,
                        "area_name": hourly.area_name,
                        "entity_id": hourly.entity_id,
                        "aggregation_period": AGGREGATION_PERIOD_WEEKLY,
                        "period_start": week_start,
                        "period_end": week_end,
                        "hourly_data": [],
                        "sample_count": 0,
                        "first_value": None,
                        "last_value": None,
                    }

                agg = aggregates[key]
                agg["hourly_data"].append(hourly)
                agg["sample_count"] += hourly.sample_count

                if agg["first_value"] is None:
                    agg["first_value"] = hourly.first_value
                agg["last_value"] = hourly.last_value

            # Calculate statistics and create aggregate records
            created_count = 0
            created_ids: list[int] = []
            for agg_data in aggregates.values():
                hourly_list = agg_data["hourly_data"]
                if not hourly_list:
                    continue

                # Calculate aggregated statistics
                min_values = [
                    h.min_value for h in hourly_list if h.min_value is not None
                ]
                max_values = [
                    h.max_value for h in hourly_list if h.max_value is not None
                ]
                agg_data["min_value"] = min(min_values) if min_values else None
                agg_data["max_value"] = max(max_values) if max_values else None

                # Weighted average: sum(hourly_avg * hourly_count) / sum(hourly_count)
                total_weighted_sum = sum(
                    h.avg_value * h.sample_count
                    for h in hourly_list
                    if h.avg_value is not None and h.sample_count > 0
                )
                total_samples = sum(h.sample_count for h in hourly_list)
                if total_samples > 0:
                    agg_data["avg_value"] = total_weighted_sum / total_samples
                else:
                    agg_data["avg_value"] = None

                # Median of hourly medians
                hourly_medians = [
                    h.median_value for h in hourly_list if h.median_value is not None
                ]
                if hourly_medians:
                    agg_data["median_value"] = statistics.median(hourly_medians)
                else:
                    agg_data["median_value"] = None

                # Weighted standard deviation (simplified: average of hourly std devs weighted by sample count)
                hourly_std_devs = [
                    (h.std_deviation, h.sample_count)
                    for h in hourly_list
                    if h.std_deviation is not None and h.sample_count > 0
                ]
                if hourly_std_devs and total_samples > 0:
                    weighted_std_sum = sum(
                        std_dev * count for std_dev, count in hourly_std_devs
                    )
                    agg_data["std_deviation"] = weighted_std_sum / total_samples
                else:
                    agg_data["std_deviation"] = None

                # Remove hourly_data list (not needed in database)
                del agg_data["hourly_data"]

                # Check if aggregate already exists
                existing = (
                    session.query(db.NumericAggregates)
                    .filter_by(
                        entity_id=agg_data["entity_id"],
                        aggregation_period=AGGREGATION_PERIOD_WEEKLY,
                        period_start=agg_data["period_start"],
                    )
                    .first()
                )

                if not existing:
                    aggregate = db.NumericAggregates(**agg_data)
                    session.add(aggregate)
                    session.flush()  # Flush to get the ID
                    created_count += 1
                    created_ids.append(aggregate.id)

            # Delete hourly aggregates that were aggregated (batched to avoid SQLite limit)
            aggregate_ids = [hourly.id for hourly in hourly_aggregates]
            batched_delete_by_ids(session, db.NumericAggregates, aggregate_ids)

            session.commit()
            _LOGGER.info(
                "Created %d weekly aggregates from %d hourly aggregates for area: %s",
                created_count,
                len(hourly_aggregates),
                area_name or "all areas",
            )

            return created_count, created_ids

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error aggregating hourly to weekly: %s", e)
        raise


def run_numeric_aggregation(
    db: AreaOccupancyDB, area_name: str | None = None, force: bool = False
) -> dict[str, int]:
    """Run the full tiered aggregation process for numeric samples.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.
        force: If True, run aggregation even if recently run

    Returns:
        Dictionary with counts of aggregates created at each level
    """
    _LOGGER.debug(
        "Running tiered numeric aggregation for area: %s", area_name or "all areas"
    )

    results = {
        "hourly": 0,
        "weekly": 0,
    }

    try:
        # Step 1: Aggregate raw samples to hourly
        hourly_count, hourly_ids = aggregate_numeric_samples_to_hourly(db, area_name)
        results["hourly"] = hourly_count

        # Step 2: Aggregate hourly to weekly (exclude hourly aggregates created in step 1)
        weekly_count, _ = aggregate_hourly_to_weekly(
            db, area_name, exclude_hourly_ids=set(hourly_ids) if hourly_ids else None
        )
        results["weekly"] = weekly_count

        _LOGGER.debug(
            "Numeric aggregation complete: %d hourly, %d weekly aggregates created",
            results["hourly"],
            results["weekly"],
        )

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error during numeric aggregation: %s", e)
        raise

    return results


def prune_old_numeric_aggregates(
    db: AreaOccupancyDB, area_name: str | None = None
) -> dict[str, int]:
    """Prune old numeric aggregates based on retention policies.

    Args:
        db: Database instance
        area_name: Optional area name to filter by. If None, processes all areas.

    Returns:
        Dictionary with counts of aggregates deleted at each level
    """
    _LOGGER.debug(
        "Pruning old numeric aggregates for area: %s", area_name or "all areas"
    )

    results = {
        "hourly": 0,
        "weekly": 0,
    }

    try:
        with db.get_session() as session:
            now = dt_util.utcnow()

            # Prune hourly aggregates older than retention period
            hourly_cutoff = to_db_utc(
                now - timedelta(days=RETENTION_HOURLY_NUMERIC_DAYS)
            )
            hourly_query = session.query(db.NumericAggregates).filter(
                db.NumericAggregates.aggregation_period == AGGREGATION_PERIOD_HOURLY,
                db.NumericAggregates.period_start < hourly_cutoff,
            )
            if area_name:
                hourly_query = hourly_query.filter(
                    db.NumericAggregates.area_name == area_name
                )
            results["hourly"] = hourly_query.delete(synchronize_session=False)

            # Prune weekly aggregates older than retention period
            weekly_cutoff = to_db_utc(
                now - timedelta(days=RETENTION_WEEKLY_NUMERIC_YEARS * 365)
            )
            weekly_query = session.query(db.NumericAggregates).filter(
                db.NumericAggregates.aggregation_period == AGGREGATION_PERIOD_WEEKLY,
                db.NumericAggregates.period_start < weekly_cutoff,
            )
            if area_name:
                weekly_query = weekly_query.filter(
                    db.NumericAggregates.area_name == area_name
                )
            results["weekly"] = weekly_query.delete(synchronize_session=False)

            session.commit()

            _LOGGER.info(
                "Pruned old numeric aggregates: %d hourly, %d weekly",
                results["hourly"],
                results["weekly"],
            )

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error pruning old numeric aggregates: %s", e)
        raise

    return results
