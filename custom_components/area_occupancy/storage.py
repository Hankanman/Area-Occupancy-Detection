"""SQLite storage module for Area Occupancy Detection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
import logging
import time
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload, sessionmaker

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    HA_RECORDER_DAYS,
    MAX_PRIOR,
    MIN_PRIOR,
)
from .data.entity_type import _ENTITY_TYPE_DATA, InputType
from .db import AreaOccupancyDB, Base
from .state_intervals import StateInterval, get_intervals_from_recorder

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Default retention for state interval cleanup
DEFAULT_RETENTION_DAYS = 365


class DatabaseExecutor:
    """Handles database operations with retry logic and session management."""

    def __init__(self, engine: sa.engine.Engine):
        """Initialize the database executor."""
        self.engine = engine
        self.Session = sessionmaker(bind=engine)

    def execute_with_retry(
        self, func: Callable, max_retries: int = 3, initial_delay: float = 0.1
    ) -> Any:
        """Execute a function with retry logic for database lock errors."""
        for attempt in range(max_retries):
            try:
                return func()
            except sa.exc.OperationalError as err:
                if (
                    "database is locked" in str(err).lower()
                    and attempt < max_retries - 1
                ):
                    delay = initial_delay * (2**attempt)  # Exponential backoff
                    _LOGGER.debug(
                        "Database locked, retrying in %.2fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise
            except Exception:
                raise
        # This should never be reached due to the retry logic above
        raise RuntimeError("Retry logic failed unexpectedly")

    def execute_in_session(self, func: Callable[[Session], Any]) -> Any:
        """Execute a function within an ORM session with automatic commit/rollback."""
        with self.Session() as session:
            try:
                result = func(session)
                session.commit()
            except Exception:
                session.rollback()
                raise
            else:
                return result

    def execute_in_transaction(self, func: Callable) -> Any:
        """Execute a function within a database transaction using engine.begin()."""
        with self.engine.begin() as conn:
            return func(conn)


class DatabaseInitializer:
    """Handles database initialization and schema creation."""

    def __init__(self, engine: sa.engine.Engine):
        """Initialize the database initializer."""
        self.engine = engine

    def enable_wal_mode(self) -> None:
        """Enable SQLite WAL mode for better concurrent writes."""
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text("PRAGMA journal_mode=WAL"))
        except sa.exc.SQLAlchemyError as err:
            _LOGGER.debug("Failed to enable WAL mode: %s", err)

    def create_schema(self) -> None:
        """Create the database schema."""
        try:
            self.enable_wal_mode()
            Base.metadata.create_all(self.engine, checkfirst=True)
        except sa.exc.OperationalError as err:
            # Handle race condition when multiple instances try to create tables
            if "already exists" in str(err).lower():
                _LOGGER.debug(
                    "Table already exists (race condition), continuing: %s", err
                )
                # Continue - other tables might still need to be created
                # Try to create remaining tables individually
                self._create_tables_individually()
            else:
                _LOGGER.error("Database initialization failed: %s", err)
                raise
        except Exception as err:
            _LOGGER.error("Database initialization failed: %s", err)
            raise

    def _create_tables_individually(self) -> None:
        """Create tables individually to handle race conditions."""
        with self.engine.connect():
            for table in Base.metadata.tables.values():
                try:
                    table.create(self.engine, checkfirst=True)
                except sa.exc.OperationalError as err:
                    if "already exists" in str(err).lower():
                        _LOGGER.debug("Table %s already exists, skipping", table.name)
                        continue
                    raise


class DatabaseQueries:
    """Handles all database queries for Area Occupancy Detection."""

    def __init__(self, db: AreaOccupancyDB, entry_id: str):
        """Initialize the queries handler."""
        self.db = db
        self.entry_id = entry_id

    # ─────────────────── Area Occupancy Queries ───────────────────

    def save_area_occupancy(
        self, record: dict[str, Any], session: Session
    ) -> dict[str, Any]:
        """Save or update area occupancy record using ORM."""
        try:
            # Try to find existing record
            area = (
                session.query(self.db.Areas)
                .filter_by(entry_id=record["entry_id"])
                .first()
            )

            if area:
                # Update existing record
                for key, value in record.items():
                    if hasattr(area, key):
                        setattr(area, key, value)
            else:
                # Create new record using from_dict
                area = self.db.Areas.from_dict(record)
                session.add(area)

            session.flush()  # Ensure the record is saved
            return area.to_dict()
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to save area occupancy: %s", e)
            raise

    def get_area_occupancy(
        self, entry_id: str, session: Session
    ) -> dict[str, Any] | None:
        """Get area occupancy record by entry ID using ORM."""
        try:
            area = session.query(self.db.Areas).filter_by(entry_id=entry_id).first()
            return area.to_dict() if area else None
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to get area occupancy: %s", e)
            raise

    # ─────────────────── Entity Configuration Queries ───────────────────

    def save_entity_config(
        self, record: dict[str, Any], session: Session
    ) -> dict[str, Any]:
        """Save or update entity configuration using ORM."""
        try:
            # Try to find existing record
            entity = (
                session.query(self.db.Entities)
                .filter_by(entry_id=record["entry_id"], entity_id=record["entity_id"])
                .first()
            )

            if entity:
                # Update existing record
                for key, value in record.items():
                    if hasattr(entity, key):
                        setattr(entity, key, value)
            else:
                # Create new record using from_dict
                entity = self.db.Entities.from_dict(record)
                session.add(entity)

            session.flush()  # Ensure the record is saved
            return entity.to_dict()
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to save entity config: %s", e)
            raise

    def get_entity_configs(
        self, entry_id: str, session: Session
    ) -> list[dict[str, Any]]:
        """Get all entity configurations for an entry using ORM."""
        try:
            entities = (
                session.query(self.db.Entities)
                .filter_by(entry_id=entry_id)
                .order_by(self.db.Entities.entity_id)
                .all()
            )
            return [entity.to_dict() for entity in entities]
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to get entity configs: %s", e)
            raise

    # ─────────────────── State Intervals Queries ───────────────────

    def save_intervals_batch(
        self, intervals: Sequence[StateInterval], session: Session
    ) -> int:
        """Save multiple state intervals efficiently using ORM bulk operations."""
        if not intervals:
            return 0

        try:
            # First, ensure all entities exist
            unique_entities = {interval["entity_id"] for interval in intervals}
            for entity_id in unique_entities:
                # Check if entity exists
                entity = (
                    session.query(self.db.Entities)
                    .filter_by(entity_id=entity_id)
                    .first()
                )
                if not entity:
                    # Create minimal entity record
                    entity = self.db.Entities(
                        entry_id=self.entry_id,
                        entity_id=entity_id,
                        entity_type="unknown",  # Default type
                        last_updated=dt_util.utcnow(),
                        created_at=dt_util.utcnow(),
                    )
                    session.add(entity)

            # Insert intervals individually to handle unique constraint violations
            inserted_count = 0
            skipped_count = 0

            for interval in intervals:
                try:
                    # Convert StateInterval to dictionary format for ORM
                    duration_seconds = (
                        interval["end"] - interval["start"]
                    ).total_seconds()
                    interval_dict = {
                        "entity_id": interval["entity_id"],
                        "state": interval["state"],
                        "start_time": interval["start"],
                        "end_time": interval["end"],
                        "duration_seconds": duration_seconds,
                        "created_at": dt_util.utcnow(),
                    }
                    interval_obj = self.db.Intervals.from_dict(interval_dict)
                    session.add(interval_obj)
                    session.flush()  # Flush immediately to catch constraint violations
                    inserted_count += 1
                except sa.exc.IntegrityError as e:
                    # Unique constraint violation - interval already exists
                    if "UNIQUE constraint failed" in str(e):
                        session.rollback()  # Rollback the failed insert
                        skipped_count += 1
                        continue
                    # Re-raise if it's a different integrity error
                    raise
                except Exception:
                    # Re-raise other exceptions
                    raise

            _LOGGER.debug(
                "Inserted %d new intervals, skipped %d existing intervals",
                inserted_count,
                skipped_count,
            )
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to save intervals batch: %s", e)
            raise
        else:
            return inserted_count

    def get_historical_intervals(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
        state_filter: str | None = None,
        limit: int | None = None,
        page_size: int = 1000,
        session: Session | None = None,
    ) -> list[StateInterval]:
        """Get historical state intervals using ORM with pagination support."""
        try:
            query = session.query(self.db.Intervals).filter(
                self.db.Intervals.entity_id == entity_id,
                self.db.Intervals.start_time >= start_time,
                self.db.Intervals.end_time <= end_time,
            )

            if state_filter:
                query = query.filter(self.db.Intervals.state == state_filter)

            query = query.order_by(self.db.Intervals.start_time.desc())

            # Apply pagination
            if limit:
                query = query.limit(limit)
            else:
                query = query.limit(page_size)

            intervals = query.all()

            # Convert to StateInterval format
            return [
                StateInterval(
                    start=interval.start_time,
                    end=interval.end_time,
                    state=interval.state,
                    entity_id=interval.entity_id,
                )
                for interval in intervals
            ]
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to get historical intervals: %s", e)
            raise

    def cleanup_old_intervals(self, cutoff_date: datetime, session: Session) -> int:
        """Remove state intervals older than the cutoff date using ORM."""
        try:
            result = (
                session.query(self.db.Intervals)
                .filter(self.db.Intervals.end_time < cutoff_date)
                .delete()
            )
            session.flush()
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to cleanup old intervals: %s", e)
            raise
        else:
            return result

    def delete_specific_intervals(
        self, intervals: list[dict[str, Any]], session: Session
    ) -> int:
        """Delete specific intervals using ORM."""
        try:
            deleted_count = 0
            for interval in intervals:
                result = (
                    session.query(self.db.Intervals)
                    .filter(
                        and_(
                            self.db.Intervals.entity_id == interval["entity_id"],
                            self.db.Intervals.start_time == interval["start"],
                            self.db.Intervals.end_time == interval["end"],
                            self.db.Intervals.state == interval["state"],
                        )
                    )
                    .delete()
                )
                deleted_count += result

            session.flush()
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to delete specific intervals: %s", e)
            raise
        else:
            return deleted_count

    # ─────────────────── Statistics Queries ───────────────────

    def get_stats(self, session: Session) -> dict[str, Any]:
        """Get database statistics using ORM."""
        try:
            stats = {}

            # Count records in each table using ORM
            stats["areas_count"] = session.query(self.db.Areas).count()
            stats["entities_count"] = session.query(self.db.Entities).count()
            stats["intervals_count"] = session.query(self.db.Intervals).count()
            stats["priors_count"] = session.query(self.db.Priors).count()

            # Entry-specific stats using ORM
            stats[f"priors_entry_{self.entry_id}"] = (
                session.query(self.db.Priors).filter_by(entry_id=self.entry_id).count()
            )

            # Database schema info
            db_version = (
                session.query(self.db.Metadata).filter_by(key="db_version").first()
            )
            stats["database_version"] = db_version.value if db_version else None

        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to get statistics: %s", e)
            raise
        else:
            return stats

    def is_intervals_empty(self, session: Session) -> bool:
        """Check if the intervals table is empty using ORM."""
        try:
            count = session.query(self.db.Intervals).count()
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to check if intervals empty: %s", e)
            raise
        else:
            return count == 0

    def get_total_intervals_count(self, session: Session) -> int:
        """Get the total count of state intervals using ORM."""
        try:
            return session.query(self.db.Intervals).count()
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to get total intervals count: %s", e)
            raise

    # ─────────────────── Cleanup Queries ───────────────────

    def reset_entry_data(self, entry_id: str, session: Session) -> None:
        """Remove all data for a specific entry (area-specific only)."""
        try:
            # Delete area-specific data only (preserve global entities and intervals)
            session.query(self.db.Entities).filter_by(entry_id=entry_id).delete()
            session.query(self.db.Areas).filter_by(entry_id=entry_id).delete()
            # Delete time-based priors
            session.query(self.db.Priors).filter_by(entry_id=entry_id).delete()
            session.flush()
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to reset entry data: %s", e)
            raise

    # ─────────────────── ORM Relationship Queries ───────────────────

    def get_area_with_entities(
        self, entry_id: str, session: Session
    ) -> dict[str, Any] | None:
        """Get area occupancy record with all related entities using ORM relationships."""

        area = (
            session.query(self.db.Areas)
            .options(joinedload(self.db.Areas.entities))
            .filter_by(entry_id=entry_id)
            .first()
        )

        if not area:
            return None

        # Use relationship to get entities (already loaded)
        entities = area.entities
        area_dict = area.to_dict()
        area_dict["entities"] = [entity.to_dict() for entity in entities]

        return area_dict

    def get_entity_with_intervals(
        self, entity_id: str, session: Session, limit: int = 100
    ) -> dict[str, Any] | None:
        """Get entity with recent intervals using ORM relationships."""

        entity = (
            session.query(self.db.Entities)
            .options(joinedload(self.db.Entities.intervals))
            .filter_by(entity_id=entity_id)
            .first()
        )

        if not entity:
            return None

        # Use relationship to get recent intervals (already loaded)
        recent_intervals = sorted(
            entity.intervals, key=lambda x: x.start_time, reverse=True
        )[:limit]

        entity_dict = entity.to_dict()
        entity_dict["recent_intervals"] = [
            interval.to_dict() for interval in recent_intervals
        ]

        return entity_dict

    def get_area_with_priors(
        self, entry_id: str, session: Session
    ) -> dict[str, Any] | None:
        """Get area occupancy record with all related priors using ORM relationships."""

        area = (
            session.query(self.db.Areas)
            .options(joinedload(self.db.Areas.priors))
            .filter_by(entry_id=entry_id)
            .first()
        )

        if not area:
            return None

        # Use relationship to get priors (already loaded)
        priors = area.priors
        area_dict = area.to_dict()
        area_dict["priors"] = [prior.to_dict() for prior in priors]

        return area_dict

    # ─────────────────── Prior Operations ───────────────────

    def save_prior(self, record: dict[str, Any], session: Session) -> dict[str, Any]:
        """Save or update area time prior using ORM."""
        try:
            # Try to find existing record
            prior = (
                session.query(self.db.Priors)
                .filter_by(
                    entry_id=record["entry_id"],
                    day_of_week=record["day_of_week"],
                    time_slot=record["time_slot"],
                )
                .first()
            )

            if prior:
                # Update existing record
                for key, value in record.items():
                    if hasattr(prior, key):
                        setattr(prior, key, value)
            else:
                # Create new record using from_dict
                prior = self.db.Priors.from_dict(record)
                session.add(prior)

            session.flush()  # Ensure the record is saved
            return prior.to_dict()
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to save prior: %s", e)
            raise

    def get_priors(self, entry_id: str, session: Session) -> list[dict[str, Any]]:
        """Get all priors for an entry using ORM."""
        try:
            priors = (
                session.query(self.db.Priors)
                .filter_by(entry_id=entry_id)
                .order_by(self.db.Priors.day_of_week, self.db.Priors.time_slot)
                .all()
            )
            return [prior.to_dict() for prior in priors]
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to get priors: %s", e)
            raise

    def save_priors_batch(
        self, priors: Sequence[dict[str, Any]], session: Session
    ) -> int:
        """Save multiple priors efficiently using ORM bulk operations."""
        if not priors:
            return 0

        try:
            # Prepare prior objects for bulk insert/update
            prior_objects = []
            for prior_record in priors:
                # Check if prior exists
                existing_prior = (
                    session.query(self.db.Priors)
                    .filter_by(
                        entry_id=prior_record["entry_id"],
                        day_of_week=prior_record["day_of_week"],
                        time_slot=prior_record["time_slot"],
                    )
                    .first()
                )

                if existing_prior:
                    # Update existing record
                    for key, value in prior_record.items():
                        if hasattr(existing_prior, key):
                            setattr(existing_prior, key, value)
                else:
                    # Create new record using from_dict
                    prior_obj = self.db.Priors.from_dict(prior_record)
                    prior_objects.append(prior_obj)

            # Use bulk_save_objects for new records
            if prior_objects:
                session.bulk_save_objects(prior_objects, update_changed_only=False)
                session.flush()

            return (
                len(prior_objects) + len(priors) - len(prior_objects)
            )  # Total processed
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to save priors batch: %s", e)
            raise


class AreaOccupancyStorage:
    """Unified storage for Area Occupancy Detection, combining DB access and coordinator logic."""

    def __init__(
        self,
        hass: HomeAssistant = None,
        entry_id: str | None = None,
        coordinator: AreaOccupancyCoordinator | None = None,
    ) -> None:
        """Initialize SQLite storage.

        Args:
            hass: Home Assistant instance (optional for testing)
            entry_id: Unique entry ID for the area (optional for testing)
            coordinator: AreaOccupancyCoordinator instance (optional for testing)

        """
        self.hass = hass
        self.entry_id = entry_id
        self.coordinator = coordinator
        self.import_stats: dict[str, int] = {}

        # Use AreaOccupancyDB for database operations
        self.db = AreaOccupancyDB(hass=self.hass) if self.hass else None
        self.engine = self.db.engine if self.db else None

        # Initialize database components
        if self.db:
            self.executor = DatabaseExecutor(self.engine)
            self.initializer = DatabaseInitializer(self.engine)
            self.queries = DatabaseQueries(self.db, self.entry_id)

        _LOGGER.debug(
            "SQLite storage initialized for entry %s using AreaOccupancyDB", entry_id
        )

    async def async_initialize(self) -> None:
        """Initialize the database schema using SQLAlchemy."""
        await self.hass.async_add_executor_job(self.initializer.create_schema)
        _LOGGER.info("SQLite storage initialized successfully")

    # ─────────────────── Area Occupancy Methods ───────────────────

    async def save_area_occupancy(self, record: dict[str, Any]) -> dict[str, Any]:
        """Save or update area occupancy record."""
        record["updated_at"] = dt_util.utcnow()

        def _save(record: dict[str, Any]):
            return self.executor.execute_in_session(
                lambda session: self.queries.save_area_occupancy(record, session)
            )

        return await self.hass.async_add_executor_job(_save, record)

    async def get_area_occupancy(self, entry_id: str) -> dict[str, Any] | None:
        """Get area occupancy record by entry ID."""

        def _get():
            return self.executor.execute_in_session(
                lambda session: self.queries.get_area_occupancy(entry_id, session)
            )

        return await self.hass.async_add_executor_job(_get)

    # ─────────────────── Entity Configuration Methods ───────────────────

    async def save_entity_config(self, record: dict[str, Any]) -> dict[str, Any]:
        """Save or update entity configuration."""
        record["last_updated"] = dt_util.utcnow()

        def _save(record: dict[str, Any]):
            def _do_save():
                return self.executor.execute_in_session(
                    lambda session: self.queries.save_entity_config(record, session)
                )

            return self.executor.execute_with_retry(_do_save)

        return await self.hass.async_add_executor_job(_save, record)

    async def get_entity_configs(self, entry_id: str) -> list[dict[str, Any]]:
        """Get all entity configurations for an entry."""

        def _get():
            return self.executor.execute_in_session(
                lambda session: self.queries.get_entity_configs(entry_id, session)
            )

        return await self.hass.async_add_executor_job(_get)

    # ─────────────────── Global State Intervals Methods ───────────────────

    async def save_state_intervals_batch(self, intervals: list[StateInterval]) -> int:
        """Save multiple state intervals efficiently to global table."""
        if not intervals:
            _LOGGER.debug("No intervals to save")
            return 0

        _LOGGER.debug("Saving batch of %d intervals", len(intervals))

        def _save():
            def _do_save():
                try:
                    # Use ORM for bulk operations
                    return self.executor.execute_in_session(
                        lambda session: self.queries.save_intervals_batch(
                            intervals, session
                        )
                    )
                except (sa.exc.SQLAlchemyError, OSError) as err:
                    _LOGGER.warning("Failed to save batch of state intervals: %s", err)
                    return 0

            return self.executor.execute_with_retry(_do_save)

        stored_count = await self.hass.async_add_executor_job(_save)

        _LOGGER.debug(
            "Committed batch: attempted %d, stored %d intervals successfully",
            len(intervals),
            stored_count,
        )
        return stored_count

    async def get_historical_intervals(
        self,
        entity_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        state_filter: str | None = None,
        limit: int | None = None,
        page_size: int = 1000,
    ) -> list[StateInterval]:
        """Get historical state intervals from global storage with pagination support."""
        if start_time is None:
            start_time = dt_util.utcnow() - timedelta(days=30)
        if end_time is None:
            end_time = dt_util.utcnow()

        def _get():
            return self.executor.execute_in_session(
                lambda session: self.queries.get_historical_intervals(
                    entity_id,
                    start_time,
                    end_time,
                    state_filter,
                    limit,
                    page_size,
                    session,
                )
            )

        return await self.hass.async_add_executor_job(_get)

    async def cleanup_old_intervals(
        self, retention_days: int = DEFAULT_RETENTION_DAYS
    ) -> int:
        """Remove state intervals older than the retention period.

        The default retention period keeps roughly one year of history to
        balance accuracy with storage size.
        """
        cutoff_date = dt_util.utcnow() - timedelta(days=retention_days)

        def _cleanup():
            return self.executor.execute_in_session(
                lambda session: self.queries.cleanup_old_intervals(cutoff_date, session)
            )

        deleted_count = await self.hass.async_add_executor_job(_cleanup)
        _LOGGER.info(
            "Cleaned up %d state intervals older than %d days",
            deleted_count,
            retention_days,
        )
        return deleted_count

    async def import_intervals_from_recorder(
        self, entity_ids: list[str], days: int = HA_RECORDER_DAYS
    ) -> None:
        """Import state intervals from recorder into the global table.

        The default range of 10 days mirrors Home Assistant's typical recorder
        retention. Intervals with ``unavailable`` or ``unknown`` states are
        ignored to keep the database small.
        """
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=days)

        _LOGGER.info(
            "Importing recorder data for %d entities from %s to %s",
            len(entity_ids),
            start_time,
            end_time,
        )

        entities = entity_ids
        if self.coordinator.occupancy_entity_id:
            entities.append(self.coordinator.occupancy_entity_id)
        if self.coordinator.wasp_entity_id:
            entities.append(self.coordinator.wasp_entity_id)

        import_counts = {}

        for entity_id in entities:
            try:
                _LOGGER.debug("Processing entity %s for import", entity_id)

                # Get intervals from recorder
                intervals = await get_intervals_from_recorder(
                    self.hass, entity_id, start_time, end_time
                )

                _LOGGER.debug(
                    "Got %d intervals from recorder for %s",
                    len(intervals) if intervals else 0,
                    entity_id,
                )

                if intervals:
                    # Store intervals efficiently
                    count = await self.save_state_intervals_batch(intervals)
                    import_counts[entity_id] = count
                    _LOGGER.debug("Imported %d intervals for %s", count, entity_id)
                else:
                    import_counts[entity_id] = 0
                    _LOGGER.debug("No intervals found for %s", entity_id)

            except Exception:
                _LOGGER.exception(
                    "Failed to import intervals for %s",
                    entity_id,
                )
                import_counts[entity_id] = 0

        total_imported = sum(import_counts.values())
        _LOGGER.info("Import completed: %d total intervals imported", total_imported)

        self.import_stats = import_counts

    # ─────────────────── Cleanup Methods ───────────────────

    async def reset_entry_data(self, entry_id: str) -> None:
        """Remove all data for a specific entry (area-specific only)."""

        def _reset():
            return self.executor.execute_in_session(
                lambda session: self.queries.reset_entry_data(entry_id, session)
            )

        await self.hass.async_add_executor_job(_reset)
        _LOGGER.info("Reset area-specific data for entry %s", entry_id)

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""

        def _get_stats():
            stats = self.executor.execute_in_session(
                lambda session: self.queries.get_stats(session)
            )

            # Database file size
            try:
                stats["db_size_bytes"] = self.db.db_path.stat().st_size
            except (FileNotFoundError, AttributeError):
                stats["db_size_bytes"] = 0

            return stats

        return await self.hass.async_add_executor_job(_get_stats)

    async def is_state_intervals_empty(self) -> bool:
        """Check if the intervals table is empty."""

        def _check_empty():
            return self.executor.execute_in_session(
                lambda session: self.queries.is_intervals_empty(session)
            )

        return await self.hass.async_add_executor_job(_check_empty)

    async def get_total_intervals_count(self) -> int:
        """Get the total count of state intervals."""

        def _get_count():
            return self.executor.execute_in_session(
                lambda session: self.queries.get_total_intervals_count(session)
            )

        return await self.hass.async_add_executor_job(_get_count)

    async def async_save_data(self) -> None:
        """Save coordinator data to SQLite storage using normalized schema."""
        if not self.coordinator:
            raise RuntimeError("Coordinator is required for async_save_data")
        try:
            # Save area occupancy data
            area_record = {
                "entry_id": self.coordinator.entry_id,
                "area_name": self.coordinator.config.name,
                "purpose": self.coordinator.config.purpose,
                "threshold": self.coordinator.threshold,
                "area_prior": self.coordinator.area_prior,
                "created_at": dt_util.utcnow(),
                "updated_at": dt_util.utcnow(),
            }
            await self.save_area_occupancy(area_record)

            # Save entity configurations
            for entity in self.coordinator.entities.entities.values():
                entity_config = {
                    "entry_id": self.coordinator.entry_id,
                    "entity_id": entity.entity_id,
                    "entity_type": entity.type.input_type.value,
                    "weight": entity.type.weight,
                    "prob_given_true": entity.likelihood.prob_given_true_raw,
                    "prob_given_false": entity.likelihood.prob_given_false_raw,
                    "created_at": dt_util.utcnow(),
                    "last_updated": dt_util.utcnow(),
                }
                await self.save_entity_config(entity_config)

            _LOGGER.debug(
                "Successfully saved data to SQLite storage for entry %s",
                self.coordinator.entry_id,
            )
        except Exception as err:
            _LOGGER.error("Failed to save data to SQLite storage: %s", err)
            raise

    async def async_load_data(self) -> dict[str, Any] | None:
        """Load coordinator data from SQLite storage using normalized schema."""
        if not self.coordinator:
            raise RuntimeError("Coordinator is required for async_load_data")
        try:
            area_record = await self.get_area_occupancy(self.coordinator.entry_id)
            if not area_record:
                _LOGGER.info(
                    "No area occupancy data found for entry %s",
                    self.coordinator.entry_id,
                )
                return None
            entity_configs = await self.get_entity_configs(self.coordinator.entry_id)
            entities_data = {}
            for config in entity_configs:
                try:
                    input_type = InputType(config["entity_type"])
                    type_defaults = _ENTITY_TYPE_DATA.get(input_type, {})
                    valid_entity_type = config["entity_type"]
                except (ValueError, KeyError):
                    type_defaults = _ENTITY_TYPE_DATA.get(InputType.UNKNOWN, {})
                    valid_entity_type = InputType.UNKNOWN.value
                    _LOGGER.warning(
                        "Unknown entity type %s for %s, using unknown defaults",
                        config["entity_type"],
                        config["entity_id"],
                    )
                entities_data[config["entity_id"]] = {
                    "entity_id": config["entity_id"],
                    "entity_type": valid_entity_type,
                    "last_updated": config["last_updated"].isoformat(),
                    "likelihood": {
                        "prob_given_true": config["prob_given_true"],
                        "prob_given_false": config["prob_given_false"],
                        "last_updated": config["last_updated"].isoformat(),
                    },
                    "type": {
                        "input_type": valid_entity_type,
                        "weight": config["weight"],
                        "prob_true": type_defaults.get(
                            "prob_true", DEFAULT_PROB_GIVEN_TRUE
                        ),
                        "prob_false": type_defaults.get(
                            "prob_false", DEFAULT_PROB_GIVEN_FALSE
                        ),
                        "prior": type_defaults.get(
                            "prior", (MIN_PRIOR + MAX_PRIOR) / 2
                        ),
                        "active_states": type_defaults.get("active_states"),
                        "active_range": type_defaults.get("active_range"),
                    },
                    "decay": {
                        "last_trigger_ts": dt_util.utcnow().timestamp(),
                        "half_life": self.coordinator.config.decay.half_life,
                        "is_decaying": False,
                    },
                    "previous_evidence": None,
                    "previous_probability": 0.0,
                }
            return {
                "name": area_record["area_name"],
                "purpose": area_record["purpose"],
                "probability": self.coordinator.probability,
                "prior": area_record.get("area_prior", self.coordinator.area_prior),
                "threshold": area_record["threshold"],
                "last_updated": area_record["updated_at"].isoformat(),
                "entities": entities_data,
            }
        except (sa.exc.SQLAlchemyError, OSError) as err:
            _LOGGER.error("Failed to load data from SQLite storage: %s", err)
            return None

    async def async_reset(self) -> None:
        """Reset SQLite storage by removing all area-specific data for this entry."""
        if not self.coordinator:
            raise RuntimeError("Coordinator is required for async_reset")
        await self.reset_entry_data(self.coordinator.entry_id)
        _LOGGER.info("Reset SQLite storage for entry %s", self.coordinator.entry_id)

    async def async_get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        return await self.get_stats()

    async def async_close(self) -> None:
        """Dispose the SQLAlchemy engine."""

        def _dispose() -> None:
            if self.db:
                self.db.close()
                if self.db.engine:
                    self.db.engine.dispose()

        await self.hass.async_add_executor_job(_dispose)
