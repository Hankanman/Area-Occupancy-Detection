"""Database query operations."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.exc import (
    DataError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)

from homeassistant.const import STATE_ON, STATE_PLAYING
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import RETENTION_DAYS
from ..data.entity_type import InputType
from . import maintenance, operations

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
    except sa.exc.SQLAlchemyError as e:
        _LOGGER.error("Failed to get area data: %s", e)
        return None


async def ensure_area_exists(db: AreaOccupancyDB) -> None:
    """Ensure that the area record exists in the database."""
    try:
        # Check if area exists
        existing_area = get_area_data(db, db.coordinator.entry_id)
        if existing_area:
            _LOGGER.debug(
                "Area already exists for entry_id: %s", db.coordinator.entry_id
            )
            return

        # Area doesn't exist, force create it
        _LOGGER.info(
            "Area not found, forcing creation for entry_id: %s",
            db.coordinator.entry_id,
        )
        operations.save_data(db)

        # Verify it was created
        new_area = get_area_data(db, db.coordinator.entry_id)
        if new_area:
            _LOGGER.info("Successfully created area: %s", new_area)
        else:
            _LOGGER.error(
                "Failed to create area for entry_id: %s", db.coordinator.entry_id
            )

    except (sa.exc.SQLAlchemyError, HomeAssistantError, TimeoutError, OSError) as e:
        _LOGGER.error("Error ensuring area exists: %s", e)


def get_latest_interval(db: AreaOccupancyDB) -> datetime:
    """Return the latest interval end time minus 1 hour, or default window if none (read-only, no lock)."""
    try:
        with db.get_session() as session:
            result = session.execute(
                sa.select(sa.func.max(db.Intervals.end_time))
            ).scalar()
            if result:
                return result - timedelta(hours=1)
            return dt_util.now() - timedelta(days=10)
    except (sa.exc.SQLAlchemyError, HomeAssistantError, TimeoutError, OSError) as e:
        # If table doesn't exist or any other error, return a default time
        if "no such table" in str(e).lower():
            _LOGGER.debug("Intervals table doesn't exist yet, using default time")
        else:
            _LOGGER.warning("Failed to get latest interval, using default time: %s", e)
        return dt_util.now() - timedelta(days=10)


def prune_old_intervals(db: AreaOccupancyDB, force: bool = False) -> int:
    """Delete intervals older than RETENTION_DAYS (coordinated across instances).

    Args:
        db: Database instance
        force: If True, skip the recent-prune check

    Returns:
        Number of intervals deleted
    """
    cutoff_date = dt_util.utcnow() - timedelta(days=RETENTION_DAYS)
    _LOGGER.debug("Pruning intervals older than %s", cutoff_date)

    try:
        with db.get_locked_session() as session:
            # Re-check last_prune inside locked session to prevent concurrent bypass
            # This ensures the throttle cannot be bypassed by concurrent instances
            if not force:
                result = (
                    session.query(db.Metadata).filter_by(key="last_prune_time").first()
                )
                if result:
                    try:
                        last_prune = datetime.fromisoformat(result.value)
                        time_since_prune = (
                            dt_util.utcnow() - last_prune
                        ).total_seconds()
                        if time_since_prune < 3600:  # 1 hour
                            _LOGGER.debug(
                                "Skipping prune - last run was %d minutes ago",
                                int(time_since_prune / 60),
                            )
                            return 0
                    except (ValueError, AttributeError) as e:
                        _LOGGER.debug(
                            "Failed to parse last prune time, proceeding: %s", e
                        )

            # Count intervals to be deleted for logging
            count_query = session.query(func.count(db.Intervals.id)).filter(
                db.Intervals.start_time < cutoff_date
            )
            intervals_to_delete = count_query.scalar() or 0

            if intervals_to_delete == 0:
                _LOGGER.debug("No old intervals to prune")
                # Still record the prune attempt to prevent other instances from trying
                maintenance.set_last_prune_time(db, dt_util.utcnow(), session)
                return 0

            # Delete old intervals
            delete_query = session.query(db.Intervals).filter(
                db.Intervals.start_time < cutoff_date
            )
            deleted_count = delete_query.delete(synchronize_session=False)

            session.commit()

            _LOGGER.info(
                "Pruned %d intervals older than %d days (cutoff: %s)",
                deleted_count,
                RETENTION_DAYS,
                cutoff_date,
            )

            # Record successful prune
            maintenance.set_last_prune_time(db, dt_util.utcnow(), session)

            return deleted_count

    except OperationalError as e:
        _LOGGER.error("Database connection error during interval pruning: %s", e)
        return 0
    except DataError as e:
        _LOGGER.error("Database data error during interval pruning: %s", e)
        return 0
    except ProgrammingError as e:
        _LOGGER.error("Database query error during interval pruning: %s", e)
        return 0
    except SQLAlchemyError as e:
        _LOGGER.error("Database error during interval pruning: %s", e)
        return 0
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Unexpected error during interval pruning: %s", e)
        return 0


def get_aggregated_intervals_by_slot(
    db: AreaOccupancyDB,
    entry_id: str,
    slot_minutes: int = 60,
    area_name: str | None = None,
) -> list[tuple[int, int, float]]:
    """Get aggregated interval data using SQL GROUP BY for better performance.

    Args:
        db: Database instance
        entry_id: The area entry ID to filter by
        slot_minutes: Time slot size in minutes
        area_name: The area name to filter by (required for multi-area support)

    Returns:
        List of (day_of_week, time_slot, total_occupied_seconds) tuples

    """
    _LOGGER.debug("Getting aggregated intervals by slot using SQL GROUP BY")

    try:
        with db.get_session() as session:
            # Use SQLite datetime functions to group intervals by day and time slot
            # This is much more efficient than Python loops
            query = (
                session.query(
                    func.strftime("%w", db.Intervals.start_time).label("day_of_week"),
                    func.cast(
                        (
                            func.cast(
                                func.strftime("%H", db.Intervals.start_time),
                                sa.Integer,
                            )
                            * 60
                            + func.cast(
                                func.strftime("%M", db.Intervals.start_time),
                                sa.Integer,
                            )
                        )
                        // slot_minutes,
                        sa.Integer,
                    ).label("time_slot"),
                    func.sum(db.Intervals.duration_seconds).label("total_seconds"),
                )
                .join(
                    db.Entities,
                    sa.and_(
                        db.Intervals.entity_id == db.Entities.entity_id,
                        db.Intervals.area_name == db.Entities.area_name,
                    ),
                )
                .filter(
                    db.Entities.entry_id == entry_id,
                    db.Entities.entity_type
                    == "motion",  # Use string instead of InputType enum
                    db.Intervals.state == "on",
                )
            )
            # Add area_name filter if provided (required for multi-area support)
            if area_name is not None:
                query = query.filter(db.Entities.area_name == area_name)
            query = query.group_by("day_of_week", "time_slot").order_by(
                "day_of_week", "time_slot"
            )

            results = query.all()

            # Convert SQLite day_of_week (0=Sunday) to Python weekday (0=Monday)
            converted_results = []
            for day_str, slot, total_seconds in results:
                try:
                    sqlite_day = int(day_str)
                    python_weekday = (
                        sqlite_day + 6
                    ) % 7  # Convert Sunday=0 to Monday=0
                    converted_results.append(
                        (python_weekday, int(slot), float(total_seconds or 0))
                    )
                except (ValueError, TypeError) as e:
                    _LOGGER.warning(
                        "Invalid day/slot data: day=%s, slot=%s, error=%s",
                        day_str,
                        slot,
                        e,
                    )
                    continue

            _LOGGER.debug(
                "SQL aggregation returned %d time slots", len(converted_results)
            )
            return converted_results

    except OperationalError as e:
        _LOGGER.error("Database connection error during interval aggregation: %s", e)
        return []
    except DataError as e:
        _LOGGER.error("Database data error during interval aggregation: %s", e)
        return []
    except ProgrammingError as e:
        _LOGGER.error("Database query error during interval aggregation: %s", e)
        return []
    except SQLAlchemyError as e:
        _LOGGER.error("Database error during interval aggregation: %s", e)
        return []
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Unexpected error during interval aggregation: %s", e)
        return []


def get_total_occupied_seconds_sql(
    db: AreaOccupancyDB,
    entry_id: str,
    area_name: str | None = None,
    lookback_days: int = 90,
    motion_timeout_seconds: int = 0,
    include_media: bool = False,
    include_appliance: bool = False,
    media_sensor_ids: list[str] | None = None,
    appliance_sensor_ids: list[str] | None = None,
) -> float | None:
    """Get total occupied seconds using SQL aggregation for better performance.

    This is a simplified SQL version that calculates the sum directly in the database.
    For complex timeout logic, the Python version should be used.

    Args:
        db: Database instance
        entry_id: The area entry ID to filter by
        area_name: The area name to filter by (required for multi-area support)
        lookback_days: Number of days to look back for data
        motion_timeout_seconds: Motion timeout in seconds (not used in SQL version)
        include_media: Whether to include media player sensors
        include_appliance: Whether to include appliance sensors
        media_sensor_ids: List of media sensor entity IDs to include
        appliance_sensor_ids: List of appliance sensor entity IDs to include

    Returns:
        Total occupied seconds as float, or None if an error occurred

    """
    cutoff_date = dt_util.utcnow() - timedelta(days=lookback_days)

    try:
        with db.get_session() as session:
            # Base filters for time range and entry_id
            base_filters = [
                db.Entities.entry_id == entry_id,
                db.Intervals.start_time >= cutoff_date,
            ]

            # Add area_name filter if provided (required for multi-area support)
            if area_name is not None:
                base_filters.append(db.Entities.area_name == area_name)

            # Build motion query with sum expression for motion sensors
            motion_sum_expr = func.sum(db.Intervals.duration_seconds).label(
                "total_seconds"
            )

            motion_query = (
                session.query(motion_sum_expr.label("total_seconds"))
                .join(
                    db.Entities,
                    sa.and_(
                        db.Intervals.entity_id == db.Entities.entity_id,
                        db.Intervals.area_name == db.Entities.area_name,
                    ),
                )
                .filter(
                    *base_filters,
                    db.Entities.entity_type == InputType.MOTION.value,
                    db.Intervals.state == "on",
                )
            )

            # Start with motion query
            queries = [motion_query]

            # Add media player query if requested
            if include_media and media_sensor_ids:
                media_query = (
                    session.query(
                        func.sum(db.Intervals.duration_seconds).label("total_seconds")
                    )
                    .join(
                        db.Entities,
                        sa.and_(
                            db.Intervals.entity_id == db.Entities.entity_id,
                            db.Intervals.area_name == db.Entities.area_name,
                        ),
                    )
                    .filter(
                        *base_filters,
                        db.Entities.entity_type == InputType.MEDIA.value,
                        db.Intervals.entity_id.in_(media_sensor_ids),
                        db.Intervals.state == STATE_PLAYING,
                    )
                )
                queries.append(media_query)

            # Add appliance query if requested
            if include_appliance and appliance_sensor_ids:
                appliance_query = (
                    session.query(
                        func.sum(db.Intervals.duration_seconds).label("total_seconds")
                    )
                    .join(
                        db.Entities,
                        sa.and_(
                            db.Intervals.entity_id == db.Entities.entity_id,
                            db.Intervals.area_name == db.Entities.area_name,
                        ),
                    )
                    .filter(
                        *base_filters,
                        db.Entities.entity_type == InputType.APPLIANCE.value,
                        db.Intervals.entity_id.in_(appliance_sensor_ids),
                        db.Intervals.state == STATE_ON,
                    )
                )
                queries.append(appliance_query)

            # Sum all query results
            total_seconds = 0.0
            for query in queries:
                result = query.scalar()
                if result is not None:
                    total_seconds += float(result)

            _LOGGER.debug(
                "SQL aggregation returned %.2f total occupied seconds",
                total_seconds,
            )
            return total_seconds

    except OperationalError as e:
        _LOGGER.error(
            "Database connection error during total seconds calculation: %s", e
        )
        return None
    except DataError as e:
        _LOGGER.error("Database data error during total seconds calculation: %s", e)
        return None
    except ProgrammingError as e:
        _LOGGER.error("Database query error during total seconds calculation: %s", e)
        return None
    except SQLAlchemyError as e:
        _LOGGER.error("Database error during total seconds calculation: %s", e)
        return None
    except (ValueError, TypeError, RuntimeError, OSError, ImportError) as e:
        _LOGGER.error("Unexpected error during total seconds calculation: %s", e)
        return None
