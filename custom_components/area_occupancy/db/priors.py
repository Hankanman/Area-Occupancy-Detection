"""Prior calculation and storage using new database tables.

This module handles storing and retrieving global priors and occupied intervals
using the GlobalPriors and OccupiedIntervalsCache tables.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import GLOBAL_PRIOR_HISTORY_COUNT

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
