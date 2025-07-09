"""SQLite storage module for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import create_engine

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .data.entity_type import _ENTITY_TYPE_DATA, InputType
from .schema import (
    DB_VERSION,
    AreaEntityConfigRecord,
    AreaOccupancyRecord,
    AreaTimePriorRecord,
    EntityRecord,
    SchemaConverter,
    area_entity_config_table,
    area_occupancy_table,
    area_time_priors_table,
    entities_table,
    indexes,
    metadata,
    metadata_table,
    state_intervals_table,
)
from .utils import StateInterval, _get_intervals_from_recorder

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Database constants
DB_NAME = "area_occupancy.db"


class SQLiteStorage:
    """Normalized SQLite storage using global entities and area-specific configuration."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize SQLite storage."""
        self.hass = hass
        self.entry_id = entry_id
        self.storage_path = Path(hass.config.config_dir) / ".storage"
        self.db_path = self.storage_path / DB_NAME

        # Ensure storage directory exists
        self.storage_path.mkdir(exist_ok=True)

        # Create SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")

        _LOGGER.debug(
            "SQLite storage initialized for entry %s at %s", entry_id, self.db_path
        )

    async def async_initialize(self) -> None:
        """Initialize the database schema using SQLAlchemy."""

        def _create_schema():
            try:
                # Check which tables already exist
                with self.engine.connect() as conn:
                    existing_tables = {
                        row[0]
                        for row in conn.execute(
                            sa.text("SELECT name FROM sqlite_master WHERE type='table'")
                        ).fetchall()
                    }

                    _LOGGER.debug("Existing tables: %s", existing_tables)

                # Create tables individually if they don't exist
                for table_name, table in metadata.tables.items():
                    if table_name not in existing_tables:
                        _LOGGER.debug("Creating table: %s", table_name)
                        table.create(self.engine)
                    else:
                        _LOGGER.debug("Table %s already exists, skipping", table_name)

                # Create all indexes
                with self.engine.connect() as conn:
                    for index in indexes:
                        try:
                            index.create(conn, checkfirst=True)
                        except sa.exc.SQLAlchemyError as idx_err:
                            _LOGGER.debug(
                                "Index creation skipped (likely exists): %s", idx_err
                            )

                # Initialize/check version metadata
                with self.engine.connect() as conn:
                    # Check if version exists
                    result = conn.execute(
                        sa.select(metadata_table.c.value).where(
                            metadata_table.c.key == "db_version"
                        )
                    ).fetchone()

                    if not result:
                        conn.execute(
                            metadata_table.insert().values(
                                key="db_version", value=str(DB_VERSION)
                            )
                        )
                        conn.commit()
                        _LOGGER.info("Database initialized with version %s", DB_VERSION)
                    else:
                        current_version = int(result[0])
                        if current_version < DB_VERSION:
                            _LOGGER.info(
                                "Database schema upgrade needed: %s -> %s",
                                current_version,
                                DB_VERSION,
                            )
                            # TODO: Add migration logic here for future schema changes
                            conn.execute(
                                metadata_table.update()
                                .where(metadata_table.c.key == "db_version")
                                .values(value=str(DB_VERSION))
                            )
                            conn.commit()

                    # Verify database integrity
                    self._check_database_integrity(conn)

            except (sa.exc.SQLAlchemyError, OSError) as err:
                _LOGGER.error("Database initialization failed: %s", err)
                # Attempt to recover by recreating schema
                if "corrupt" in str(err).lower():
                    _LOGGER.warning("Database corruption detected, attempting recovery")
                    self.db_path.unlink(missing_ok=True)
                    # Recreate all tables since we deleted the database
                    for table in metadata.tables.values():
                        table.create(self.engine)
                    with self.engine.connect() as conn:
                        conn.execute(
                            metadata_table.insert().values(
                                key="db_version", value=str(DB_VERSION)
                            )
                        )
                        conn.commit()
                else:
                    raise

        await self.hass.async_add_executor_job(_create_schema)
        _LOGGER.info("SQLite storage initialized successfully")

    def _check_database_integrity(self, conn) -> None:
        """Check database integrity and log issues."""
        try:
            # Quick integrity check
            result = conn.execute(sa.text("PRAGMA integrity_check")).fetchone()
            if result and result[0] != "ok":
                _LOGGER.warning("Database integrity issue detected: %s", result[0])

            # Check if all required tables exist
            required_tables = [
                "area_occupancy",
                "entities",
                "area_entity_config",
                "state_intervals",
                "metadata",
            ]
            existing_tables = [
                row[0]
                for row in conn.execute(
                    sa.text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            ]

            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                _LOGGER.warning("Missing tables detected: %s", missing_tables)

        except (sa.exc.SQLAlchemyError, OSError) as err:
            _LOGGER.warning("Database integrity check failed: %s", err)

    # ─────────────────── Global Entity Methods ───────────────────

    async def ensure_entity_exists(
        self, entity_id: str, entity_class: str
    ) -> EntityRecord:
        """Ensure a global entity record exists, creating if necessary."""

        def _upsert_entity():
            with self.engine.connect() as conn:
                # Check if entity exists
                result = conn.execute(
                    sa.select(entities_table).where(
                        entities_table.c.entity_id == entity_id
                    )
                ).fetchone()

                if result:
                    # Update last_seen time
                    conn.execute(
                        entities_table.update()
                        .where(entities_table.c.entity_id == entity_id)
                        .values(last_seen=dt_util.utcnow())
                    )
                    conn.commit()
                    return SchemaConverter.row_to_entity(result)
                # Create new entity
                entity_record = EntityRecord(
                    entity_id=entity_id,
                    last_seen=dt_util.utcnow(),
                    created_at=dt_util.utcnow(),
                )
                values = SchemaConverter.entity_to_dict(entity_record)
                conn.execute(entities_table.insert().values(**values))
                conn.commit()
                return entity_record

        return await self.hass.async_add_executor_job(_upsert_entity)

    async def get_entity(self, entity_id: str) -> EntityRecord | None:
        """Get global entity record."""

        def _get():
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.select(entities_table).where(
                        entities_table.c.entity_id == entity_id
                    )
                ).fetchone()

                return SchemaConverter.row_to_entity(result) if result else None

        return await self.hass.async_add_executor_job(_get)

    # ─────────────────── Area Occupancy Methods ───────────────────

    async def save_area_occupancy(
        self, record: AreaOccupancyRecord
    ) -> AreaOccupancyRecord:
        """Save or update area occupancy record."""
        record.updated_at = dt_util.utcnow()

        def _save(record: AreaOccupancyRecord):
            with self.engine.connect() as conn:
                values = SchemaConverter.area_occupancy_to_dict(record)

                # Use INSERT OR REPLACE for SQLite
                stmt = sa.text("""
                    INSERT OR REPLACE INTO area_occupancy
                    (entry_id, area_name, purpose, threshold, created_at, updated_at)
                    VALUES (:entry_id, :area_name, :purpose, :threshold,
                            COALESCE((SELECT created_at FROM area_occupancy WHERE entry_id = :entry_id), :created_at),
                            :updated_at)
                """)

                conn.execute(stmt, values)
                conn.commit()
                return record

        return await self.hass.async_add_executor_job(_save, record)

    async def get_area_occupancy(self, entry_id: str) -> AreaOccupancyRecord | None:
        """Get area occupancy record by entry ID."""

        def _get():
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.select(area_occupancy_table).where(
                        area_occupancy_table.c.entry_id == entry_id
                    )
                ).fetchone()

                return SchemaConverter.row_to_area_occupancy(result) if result else None

        return await self.hass.async_add_executor_job(_get)

    # ─────────────────── Area Entity Configuration Methods ───────────────────

    async def save_area_entity_config(
        self, record: AreaEntityConfigRecord
    ) -> AreaEntityConfigRecord:
        """Save or update area-specific entity configuration."""
        record.last_updated = dt_util.utcnow()

        def _save(record: AreaEntityConfigRecord):
            # First ensure the global entity exists

            with self.engine.connect() as conn:
                # Ensure global entity exists
                conn.execute(
                    sa.text("""
                        INSERT OR IGNORE INTO entities (entity_id, last_seen, created_at)
                        VALUES (:entity_id, :now, :now)
                    """),
                    {
                        "entity_id": record.entity_id,
                        "now": dt_util.utcnow(),
                    },
                )

                # Save area-specific config
                values = SchemaConverter.area_entity_config_to_dict(record)
                stmt = sa.text("""
                    INSERT OR REPLACE INTO area_entity_config
                    (entry_id, entity_id, entity_type, weight,
                     prob_given_true, prob_given_false, last_updated)
                    VALUES (:entry_id, :entity_id, :entity_type, :weight,
                            :prob_given_true, :prob_given_false, :last_updated)
                """)

                conn.execute(stmt, values)
                conn.commit()
                return record

        return await self.hass.async_add_executor_job(_save, record)

    async def get_area_entity_configs(
        self, entry_id: str
    ) -> list[AreaEntityConfigRecord]:
        """Get all area-specific entity configurations for an entry."""

        def _get():
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.select(area_entity_config_table)
                    .where(area_entity_config_table.c.entry_id == entry_id)
                    .order_by(area_entity_config_table.c.entity_id)
                ).fetchall()

                return [
                    SchemaConverter.row_to_area_entity_config(row) for row in result
                ]

        return await self.hass.async_add_executor_job(_get)

    async def get_area_entity_config(
        self, entry_id: str, entity_id: str
    ) -> AreaEntityConfigRecord | None:
        """Get specific area entity configuration."""

        def _get():
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.select(area_entity_config_table).where(
                        sa.and_(
                            area_entity_config_table.c.entry_id == entry_id,
                            area_entity_config_table.c.entity_id == entity_id,
                        )
                    )
                ).fetchone()

                return (
                    SchemaConverter.row_to_area_entity_config(result)
                    if result
                    else None
                )

        return await self.hass.async_add_executor_job(_get)

    # ─────────────────── Time-Based Priors Methods ───────────────────

    async def save_time_prior(self, record: AreaTimePriorRecord) -> AreaTimePriorRecord:
        """Save or update a time-based prior record."""
        record.last_updated = dt_util.utcnow()

        def _save(record: AreaTimePriorRecord):
            with self.engine.connect() as conn:
                values = SchemaConverter.area_time_prior_to_dict(record)

                # Use INSERT OR REPLACE for SQLite
                stmt = sa.text("""
                    INSERT OR REPLACE INTO area_time_priors
                    (entry_id, day_of_week, time_slot, prior_value, data_points, last_updated)
                    VALUES (:entry_id, :day_of_week, :time_slot, :prior_value, :data_points, :last_updated)
                """)

                conn.execute(stmt, values)
                conn.commit()
                return record

        return await self.hass.async_add_executor_job(_save, record)

    async def save_time_priors_batch(self, records: list[AreaTimePriorRecord]) -> int:
        """Save multiple time-based prior records efficiently."""
        if not records:
            _LOGGER.debug("No time priors to save")
            return 0

        _LOGGER.debug("Saving batch of %d time priors", len(records))

        def _save_batch():
            stored_count = 0
            with self.engine.connect() as conn:
                for record in records:
                    try:
                        values = SchemaConverter.area_time_prior_to_dict(record)
                        values["last_updated"] = dt_util.utcnow()

                        stmt = sa.text("""
                            INSERT OR REPLACE INTO area_time_priors
                            (entry_id, day_of_week, time_slot, prior_value, data_points, last_updated)
                            VALUES (:entry_id, :day_of_week, :time_slot, :prior_value, :data_points, :last_updated)
                        """)

                        conn.execute(stmt, values)
                        stored_count += 1

                    except (sa.exc.SQLAlchemyError, OSError) as err:
                        _LOGGER.warning(
                            "Failed to save time prior for entry %s, day %d, slot %d: %s",
                            record.entry_id,
                            record.day_of_week,
                            record.time_slot,
                            err,
                        )

                conn.commit()
                _LOGGER.debug(
                    "Committed batch: %d time priors stored successfully", stored_count
                )
            return stored_count

        return await self.hass.async_add_executor_job(_save_batch)

    async def get_time_prior(
        self, entry_id: str, day_of_week: int, time_slot: int
    ) -> AreaTimePriorRecord | None:
        """Get a specific time-based prior record."""

        def _get():
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.select(area_time_priors_table).where(
                        sa.and_(
                            area_time_priors_table.c.entry_id == entry_id,
                            area_time_priors_table.c.day_of_week == day_of_week,
                            area_time_priors_table.c.time_slot == time_slot,
                        )
                    )
                ).fetchone()

                return (
                    SchemaConverter.row_to_area_time_prior(result) if result else None
                )

        return await self.hass.async_add_executor_job(_get)

    async def get_time_priors_for_entry(
        self, entry_id: str
    ) -> list[AreaTimePriorRecord]:
        """Get all time-based prior records for an entry."""

        def _get():
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.select(area_time_priors_table)
                    .where(area_time_priors_table.c.entry_id == entry_id)
                    .order_by(
                        area_time_priors_table.c.day_of_week,
                        area_time_priors_table.c.time_slot,
                    )
                ).fetchall()

                return [SchemaConverter.row_to_area_time_prior(row) for row in result]

        return await self.hass.async_add_executor_job(_get)

    async def get_time_priors_for_day(
        self, entry_id: str, day_of_week: int
    ) -> list[AreaTimePriorRecord]:
        """Get all time-based prior records for a specific day of week."""

        def _get():
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.select(area_time_priors_table)
                    .where(
                        sa.and_(
                            area_time_priors_table.c.entry_id == entry_id,
                            area_time_priors_table.c.day_of_week == day_of_week,
                        )
                    )
                    .order_by(area_time_priors_table.c.time_slot)
                ).fetchall()

                return [SchemaConverter.row_to_area_time_prior(row) for row in result]

        return await self.hass.async_add_executor_job(_get)

    async def delete_time_priors_for_entry(self, entry_id: str) -> int:
        """Delete all time-based prior records for an entry."""

        def _delete():
            with self.engine.connect() as conn:
                result = conn.execute(
                    area_time_priors_table.delete().where(
                        area_time_priors_table.c.entry_id == entry_id
                    )
                )
                deleted_count = result.rowcount
                conn.commit()
                return deleted_count

        deleted_count = await self.hass.async_add_executor_job(_delete)
        _LOGGER.info("Deleted %d time priors for entry %s", deleted_count, entry_id)
        return deleted_count

    async def get_recent_time_priors(
        self, entry_id: str, hours: int = 24
    ) -> list[AreaTimePriorRecord]:
        """Get time-based prior records updated within the specified hours.

        Args:
            entry_id: The entry ID to get priors for
            hours: Number of hours to look back for recent updates

        Returns:
            List of recent time prior records

        """

        def _get():
            cutoff_time = dt_util.utcnow() - timedelta(hours=hours)
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.select(area_time_priors_table)
                    .where(
                        sa.and_(
                            area_time_priors_table.c.entry_id == entry_id,
                            area_time_priors_table.c.last_updated >= cutoff_time,
                        )
                    )
                    .order_by(
                        area_time_priors_table.c.day_of_week,
                        area_time_priors_table.c.time_slot,
                    )
                ).fetchall()

                return [SchemaConverter.row_to_area_time_prior(row) for row in result]

        return await self.hass.async_add_executor_job(_get)

    # ─────────────────── Global State Intervals Methods ───────────────────

    async def save_state_intervals_batch(self, intervals: list[StateInterval]) -> int:
        """Save multiple state intervals efficiently to global table."""
        if not intervals:
            _LOGGER.debug("No intervals to save")
            return 0

        _LOGGER.debug("Saving batch of %d intervals", len(intervals))

        def _save_batch():
            stored_count = 0
            with self.engine.connect() as conn:
                # First, ensure all entities exist in a separate transaction
                unique_entities = {interval["entity_id"] for interval in intervals}
                for entity_id in unique_entities:
                    try:
                        conn.execute(
                            sa.text("""
                                INSERT OR IGNORE INTO entities (entity_id, domain, last_seen, created_at)
                                VALUES (:entity_id, 'unknown', :now, :now)
                            """),
                            {
                                "entity_id": entity_id,
                                "now": dt_util.utcnow(),
                            },
                        )
                    except (sa.exc.SQLAlchemyError, OSError):
                        _LOGGER.warning("Failed to ensure entity %s exists", entity_id)

                # Commit entities first
                conn.commit()

                # Now process intervals
                for i, interval in enumerate(intervals):
                    try:
                        values = SchemaConverter.state_interval_to_dict(interval)

                        # Try INSERT OR IGNORE first
                        stmt = sa.text("""
                            INSERT OR IGNORE INTO state_intervals
                            (entity_id, state, start_time, end_time, duration_seconds, created_at)
                            VALUES (:entity_id, :state, :start_time, :end_time, :duration_seconds, :created_at)
                        """)

                        result = conn.execute(stmt, values)

                        if result.rowcount > 0:
                            stored_count += 1
                            if stored_count <= 5:  # Log first few successes
                                _LOGGER.debug(
                                    "Successfully stored interval %d for %s: %s",
                                    stored_count,
                                    interval["entity_id"],
                                    interval["state"],
                                )
                        # If INSERT OR IGNORE failed, try to understand why
                        elif i < 3:  # Only diagnose first few failures
                            # Check if it actually exists
                            check_stmt = sa.text("""
                                    SELECT COUNT(*) FROM state_intervals
                                    WHERE entity_id = :entity_id
                                    AND start_time = :start_time
                                    AND end_time = :end_time
                                """)
                            exists_count = conn.execute(
                                check_stmt,
                                {
                                    "entity_id": interval["entity_id"],
                                    "start_time": values["start_time"],
                                    "end_time": values["end_time"],
                                },
                            ).scalar()

                            if (exists_count or 0) > 0:
                                _LOGGER.debug(
                                    "Interval %d actually exists, skipped", i + 1
                                )
                            else:
                                # Try direct INSERT to see the actual error
                                try:
                                    direct_stmt = sa.text("""
                                            INSERT INTO state_intervals
                                            (entity_id, state, start_time, end_time, duration_seconds, created_at)
                                            VALUES (:entity_id, :state, :start_time, :end_time, :duration_seconds, :created_at)
                                        """)
                                    conn.execute(direct_stmt, values)
                                    _LOGGER.warning(
                                        "Direct INSERT succeeded but OR IGNORE failed - unexpected!"
                                    )
                                    stored_count += 1
                                except (sa.exc.SQLAlchemyError, OSError):
                                    _LOGGER.warning(
                                        "Interval %d INSERT failed. Values: entity_id=%s, state=%s, times=%s to %s",
                                        i + 1,
                                        values["entity_id"],
                                        values["state"],
                                        values["start_time"],
                                        values["end_time"],
                                    )

                    except (sa.exc.SQLAlchemyError, OSError):
                        _LOGGER.warning(
                            "Failed to process interval %d for %s",
                            i + 1,
                            interval["entity_id"],
                        )

                conn.commit()
                _LOGGER.debug(
                    "Committed batch: %d intervals stored successfully", stored_count
                )
            return stored_count

        return await self.hass.async_add_executor_job(_save_batch)

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
            all_intervals = []
            offset = 0

            with self.engine.connect() as conn:
                while True:
                    query = sa.select(state_intervals_table).where(
                        sa.and_(
                            state_intervals_table.c.entity_id == entity_id,
                            state_intervals_table.c.start_time >= start_time,
                            state_intervals_table.c.end_time <= end_time,
                        )
                    )

                    if state_filter:
                        query = query.where(
                            state_intervals_table.c.state == state_filter
                        )

                    query = query.order_by(state_intervals_table.c.start_time.desc())

                    # Apply pagination
                    page_limit = (
                        min(page_size, limit - len(all_intervals))
                        if limit
                        else page_size
                    )
                    query = query.limit(page_limit).offset(offset)

                    result = conn.execute(query).fetchall()
                    if not result:
                        break

                    intervals = [
                        SchemaConverter.row_to_state_interval(row) for row in result
                    ]
                    all_intervals.extend(intervals)

                    # Check if we have enough results or reached the end
                    if len(result) < page_size or (
                        limit and len(all_intervals) >= limit
                    ):
                        break

                    offset += page_size

                return all_intervals[:limit] if limit else all_intervals

        return await self.hass.async_add_executor_job(_get)

    async def cleanup_old_intervals(self, retention_days: int = 365) -> int:
        """Remove state intervals older than retention period."""
        cutoff_date = dt_util.utcnow() - timedelta(days=retention_days)

        def _cleanup():
            with self.engine.connect() as conn:
                result = conn.execute(
                    state_intervals_table.delete().where(
                        state_intervals_table.c.end_time < cutoff_date
                    )
                )
                deleted_count = result.rowcount
                conn.commit()
                return deleted_count

        deleted_count = await self.hass.async_add_executor_job(_cleanup)
        _LOGGER.info(
            "Cleaned up %d state intervals older than %d days",
            deleted_count,
            retention_days,
        )
        return deleted_count

    async def import_intervals_from_recorder(
        self, entity_ids: list[str], days: int = 10
    ) -> dict[str, int]:
        """Import state intervals from HA recorder to global table.

        Automatically filters out 'unavailable' and 'unknown' states to avoid storing useless data.
        """
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=days)

        _LOGGER.info(
            "Importing recorder data for %d entities from %s to %s",
            len(entity_ids),
            start_time,
            end_time,
        )

        import_counts = {}

        for entity_id in entity_ids:
            try:
                _LOGGER.debug("Processing entity %s for import", entity_id)

                # Get intervals from recorder
                intervals = await _get_intervals_from_recorder(
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

        return import_counts

    # ─────────────────── Cleanup Methods ───────────────────

    async def cleanup_old_area_history(self, entry_id: str, days: int = 30) -> int:
        """Remove area history records older than specified days.

        Note: area_history table has been removed in schema simplification.
        This method is kept for compatibility but does nothing.
        """
        _LOGGER.debug(
            "cleanup_old_area_history called but area_history table no longer exists (entry %s)",
            entry_id,
        )
        return 0

    async def reset_entry_data(self, entry_id: str) -> None:
        """Remove all data for a specific entry (area-specific only)."""

        def _reset():
            with self.engine.connect() as conn:
                # Delete area-specific data only (preserve global entities and intervals)
                # Note: area_history table removed in schema simplification
                conn.execute(
                    area_entity_config_table.delete().where(
                        area_entity_config_table.c.entry_id == entry_id
                    )
                )
                conn.execute(
                    area_occupancy_table.delete().where(
                        area_occupancy_table.c.entry_id == entry_id
                    )
                )
                # Delete time-based priors
                conn.execute(
                    area_time_priors_table.delete().where(
                        area_time_priors_table.c.entry_id == entry_id
                    )
                )
                conn.commit()

        await self.hass.async_add_executor_job(_reset)
        _LOGGER.info("Reset area-specific data for entry %s", entry_id)

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""

        def _get_stats():
            stats = {}
            with self.engine.connect() as conn:
                # Count records in each table
                stats["area_occupancy_count"] = conn.execute(
                    sa.select(sa.func.count()).select_from(area_occupancy_table)
                ).scalar()
                stats["entities_count"] = conn.execute(
                    sa.select(sa.func.count()).select_from(entities_table)
                ).scalar()
                stats["area_entity_config_count"] = conn.execute(
                    sa.select(sa.func.count()).select_from(area_entity_config_table)
                ).scalar()
                stats["state_intervals_count"] = conn.execute(
                    sa.select(sa.func.count()).select_from(state_intervals_table)
                ).scalar()
                stats["area_time_priors_count"] = conn.execute(
                    sa.select(sa.func.count()).select_from(area_time_priors_table)
                ).scalar()

                # Entry-specific stats
                stats[f"area_entity_config_entry_{self.entry_id}"] = conn.execute(
                    sa.select(sa.func.count())
                    .select_from(area_entity_config_table)
                    .where(area_entity_config_table.c.entry_id == self.entry_id)
                ).scalar()
                stats[f"area_time_priors_entry_{self.entry_id}"] = conn.execute(
                    sa.select(sa.func.count())
                    .select_from(area_time_priors_table)
                    .where(area_time_priors_table.c.entry_id == self.entry_id)
                ).scalar()

                # Database schema info
                stats["database_version"] = conn.execute(
                    sa.select(metadata_table.c.value).where(
                        metadata_table.c.key == "db_version"
                    )
                ).scalar()

            # Database file size
            try:
                stats["db_size_bytes"] = self.db_path.stat().st_size
            except FileNotFoundError:
                stats["db_size_bytes"] = 0

            return stats

        return await self.hass.async_add_executor_job(_get_stats)

    async def is_state_intervals_empty(self) -> bool:
        """Check if the state_intervals table is empty."""

        def _check_empty():
            with self.engine.connect() as conn:
                count = conn.execute(
                    sa.select(sa.func.count()).select_from(state_intervals_table)
                ).scalar()
                return (count or 0) == 0

        return await self.hass.async_add_executor_job(_check_empty)

    async def get_total_intervals_count(self) -> int:
        """Get the total count of state intervals."""

        def _get_count():
            with self.engine.connect() as conn:
                count = conn.execute(
                    sa.select(sa.func.count()).select_from(state_intervals_table)
                ).scalar()
                return count or 0

        return await self.hass.async_add_executor_job(_get_count)


class AreaOccupancySQLiteStore:
    """High-level storage abstraction for Area Occupancy Detection."""

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the SQLite store."""
        self._coordinator = coordinator
        self._storage = SQLiteStorage(coordinator.hass, coordinator.entry_id)

    async def async_initialize(self) -> None:
        """Initialize the storage."""
        await self._storage.async_initialize()

    async def async_save_data(self, force: bool = False) -> None:
        """Save coordinator data to SQLite storage using normalized schema."""
        try:
            # Save area occupancy data
            area_record = AreaOccupancyRecord(
                entry_id=self._coordinator.entry_id,
                area_name=self._coordinator.config.name,
                purpose=self._coordinator.config.purpose,
                threshold=self._coordinator.threshold,
            )
            await self._storage.save_area_occupancy(area_record)

            # Save area-specific entity configurations
            for entity in self._coordinator.entities.entities.values():
                # Create area-specific entity config record
                entity_config = AreaEntityConfigRecord(
                    entry_id=self._coordinator.entry_id,
                    entity_id=entity.entity_id,
                    entity_type=entity.type.input_type.value,
                    weight=entity.type.weight,
                    prob_given_true=entity.likelihood.prob_given_true,
                    prob_given_false=entity.likelihood.prob_given_false,
                )
                await self._storage.save_area_entity_config(entity_config)

            _LOGGER.debug(
                "Successfully saved data to SQLite storage for entry %s",
                self._coordinator.entry_id,
            )

        except Exception as err:
            _LOGGER.error("Failed to save data to SQLite storage: %s", err)
            raise

    async def async_load_data(self) -> dict[str, Any] | None:
        """Load coordinator data from SQLite storage using normalized schema."""
        try:
            # Load area occupancy data
            area_record = await self._storage.get_area_occupancy(
                self._coordinator.entry_id
            )
            if not area_record:
                _LOGGER.info(
                    "No area occupancy data found for entry %s",
                    self._coordinator.entry_id,
                )
                return None

            # Load area-specific entity configurations
            entity_configs = await self._storage.get_area_entity_configs(
                self._coordinator.entry_id
            )

            # Convert to format expected by coordinator
            entities_data = {}
            for config in entity_configs:
                # Get the default entity type data for this input type
                try:
                    input_type = InputType(config.entity_type)
                    type_defaults = _ENTITY_TYPE_DATA.get(input_type, {})
                except (ValueError, KeyError):
                    # Fallback to motion type defaults if unknown type
                    type_defaults = _ENTITY_TYPE_DATA.get(InputType.MOTION, {})
                    _LOGGER.warning(
                        "Unknown entity type %s for %s, using motion defaults",
                        config.entity_type,
                        config.entity_id,
                    )

                # Create entity dict in old format for compatibility
                entities_data[config.entity_id] = {
                    "entity_id": config.entity_id,
                    "entity_type": config.entity_type,
                    "last_updated": config.last_updated.isoformat(),
                    # Include likelihood data for restoration
                    "likelihood": {
                        "prob_given_true": config.prob_given_true,
                        "prob_given_false": config.prob_given_false,
                        "last_updated": config.last_updated.isoformat(),
                    },
                    # Include type data with proper active_states/active_range from defaults
                    "type": {
                        "input_type": config.entity_type,
                        "weight": config.weight,
                        "prob_true": config.prob_given_true,
                        "prob_false": config.prob_given_false,
                        "prior": type_defaults.get("prior", 0.5),
                        "active_states": type_defaults.get("active_states"),
                        "active_range": type_defaults.get("active_range"),
                    },
                    # Include basic decay data (will be reset)
                    "decay": {
                        "last_trigger_ts": dt_util.utcnow().timestamp(),
                        "half_life": self._coordinator.config.decay.half_life,
                        "is_decaying": False,
                    },
                    "previous_evidence": None,
                    "previous_probability": 0.0,  # Use default since probability field is removed
                }

            return {
                "name": area_record.area_name,
                "purpose": area_record.purpose,
                "probability": self._coordinator.probability,  # Use current calculated value
                "prior": self._coordinator.area_prior,  # Use current calculated value
                "threshold": area_record.threshold,
                "last_updated": area_record.updated_at.isoformat(),
                "entities": entities_data,
            }

        except (sa.exc.SQLAlchemyError, OSError) as err:
            _LOGGER.error("Failed to load data from SQLite storage: %s", err)
            return None

    async def async_reset(self) -> None:
        """Reset SQLite storage by removing all area-specific data for this entry."""
        await self._storage.reset_entry_data(self._coordinator.entry_id)
        _LOGGER.info("Reset SQLite storage for entry %s", self._coordinator.entry_id)

    async def async_record_state_change(
        self,
        entity_id: str,
        probability_change: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a state change in area history."""
        # This method is no longer used as area_history_table is removed.
        # Keeping it for now to avoid breaking existing calls, but it will do nothing.
        _LOGGER.warning(
            "async_record_state_change is deprecated as area_history_table is removed."
        )

    async def async_get_history(
        self,
        entity_id: str | None = None,
        days: int = 7,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get area history records for analysis."""
        # This method is no longer used as area_history_table is removed.
        # Keeping it for now to avoid breaking existing calls, but it will return empty list.
        _LOGGER.warning(
            "async_get_history is deprecated as area_history_table is removed."
        )
        return []

    async def async_cleanup(self, days: int = 30) -> None:
        """Clean up old area history records."""
        await self._storage.cleanup_old_area_history(self._coordinator.entry_id, days)

    async def async_get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        return await self._storage.get_stats()

    async def import_intervals_from_recorder(
        self, entity_ids: list[str], days: int = 10
    ) -> dict[str, int]:
        """Import state intervals from HA recorder to global table."""
        return await self._storage.import_intervals_from_recorder(entity_ids, days)

    async def cleanup_old_intervals(self, retention_days: int = 365) -> int:
        """Remove state intervals older than retention period."""
        return await self._storage.cleanup_old_intervals(retention_days)

    async def is_state_intervals_empty(self) -> bool:
        """Check if the state_intervals table is empty."""
        return await self._storage.is_state_intervals_empty()

    async def get_total_intervals_count(self) -> int:
        """Get the total count of state intervals."""
        return await self._storage.get_total_intervals_count()

    async def get_historical_intervals(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[StateInterval]:
        """Get historical intervals for an entity (public interface)."""
        return await self._storage.get_historical_intervals(
            entity_id, start_time, end_time
        )

    # ─────────────────── Time-Based Priors Methods ───────────────────

    async def save_time_prior(self, record: AreaTimePriorRecord) -> AreaTimePriorRecord:
        """Save a time-based prior record."""
        return await self._storage.save_time_prior(record)

    async def save_time_priors_batch(self, records: list[AreaTimePriorRecord]) -> int:
        """Save multiple time-based prior records efficiently."""
        return await self._storage.save_time_priors_batch(records)

    async def get_time_prior(
        self, entry_id: str, day_of_week: int, time_slot: int
    ) -> AreaTimePriorRecord | None:
        """Get a specific time-based prior record."""
        return await self._storage.get_time_prior(entry_id, day_of_week, time_slot)

    async def get_time_priors_for_entry(
        self, entry_id: str
    ) -> list[AreaTimePriorRecord]:
        """Get all time-based prior records for an entry."""
        return await self._storage.get_time_priors_for_entry(entry_id)

    async def get_time_priors_for_day(
        self, entry_id: str, day_of_week: int
    ) -> list[AreaTimePriorRecord]:
        """Get all time-based prior records for a specific day of week."""
        return await self._storage.get_time_priors_for_day(entry_id, day_of_week)

    async def delete_time_priors_for_entry(self, entry_id: str) -> int:
        """Delete all time-based prior records for an entry."""
        return await self._storage.delete_time_priors_for_entry(entry_id)

    async def get_recent_time_priors(
        self, entry_id: str, hours: int = 24
    ) -> list[AreaTimePriorRecord]:
        """Get time-based prior records updated within the specified hours.

        Args:
            entry_id: The entry ID to get priors for
            hours: Number of hours to look back for recent updates

        Returns:
            List of recent time prior records

        """
        return await self._storage.get_recent_time_priors(entry_id, hours)
