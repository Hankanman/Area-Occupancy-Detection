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

from .schema import (
    DB_VERSION,
    AreaEntityConfigRecord,
    AreaHistoryRecord,
    AreaOccupancyRecord,
    EntityRecord,
    SchemaConverter,
    area_entity_config_table,
    area_history_table,
    area_occupancy_table,
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
                # Create all tables
                metadata.create_all(self.engine)

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
                    metadata.create_all(self.engine)
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
                "area_history",
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
                    domain=entity_class,
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
                    (entry_id, area_name, purpose, probability, prior, threshold,
                     occupied, created_at, updated_at)
                    VALUES (:entry_id, :area_name, :purpose, :probability, :prior,
                            :threshold, :occupied,
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
            ha_state = self.hass.states.get(record.entity_id)
            entity_class = ha_state.domain if ha_state else "unknown"

            with self.engine.connect() as conn:
                # Ensure global entity exists
                conn.execute(
                    sa.text("""
                        INSERT OR IGNORE INTO entities (entity_id, domain, last_seen, created_at)
                        VALUES (:entity_id, :domain, :now, :now)
                    """),
                    {
                        "entity_id": record.entity_id,
                        "domain": entity_class,
                        "now": dt_util.utcnow(),
                    },
                )

                # Save area-specific config
                values = SchemaConverter.area_entity_config_to_dict(record)
                stmt = sa.text("""
                    INSERT OR REPLACE INTO area_entity_config
                    (entry_id, entity_id, entity_type, weight, probability,
                     prob_given_true, prob_given_false, last_state, last_updated, attributes)
                    VALUES (:entry_id, :entity_id, :entity_type, :weight, :probability,
                            :prob_given_true, :prob_given_false, :last_state, :last_updated, :attributes)
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

    # ─────────────────── Area History Methods ───────────────────

    async def save_area_history(self, record: AreaHistoryRecord) -> AreaHistoryRecord:
        """Save area history record."""
        record.timestamp = dt_util.utcnow()

        def _save(record: AreaHistoryRecord):
            with self.engine.connect() as conn:
                values = SchemaConverter.area_history_to_dict(record)
                result = conn.execute(area_history_table.insert().values(**values))
                record.id = result.lastrowid
                conn.commit()
                return record

        return await self.hass.async_add_executor_job(_save, record)

    async def get_area_history(
        self,
        entry_id: str,
        entity_id: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[AreaHistoryRecord]:
        """Get area history records with optional filtering."""

        def _get():
            with self.engine.connect() as conn:
                query = sa.select(area_history_table).where(
                    area_history_table.c.entry_id == entry_id
                )

                if entity_id:
                    query = query.where(area_history_table.c.entity_id == entity_id)

                if since:
                    query = query.where(area_history_table.c.timestamp >= since)

                query = query.order_by(area_history_table.c.timestamp.desc())

                if limit:
                    query = query.limit(limit)

                result = conn.execute(query).fetchall()
                return [SchemaConverter.row_to_area_history(row) for row in result]

        return await self.hass.async_add_executor_job(_get)

    # ─────────────────── Global State Intervals Methods ───────────────────

    async def save_state_intervals_batch(self, intervals: list[StateInterval]) -> int:
        """Save multiple state intervals efficiently to global table."""
        if not intervals:
            return 0

        def _save_batch():
            stored_count = 0
            with self.engine.connect() as conn:
                for interval in intervals:
                    try:
                        # Ensure entity exists first
                        conn.execute(
                            sa.text("""
                                INSERT OR IGNORE INTO entities (entity_id, domain, last_seen, created_at)
                                VALUES (:entity_id, 'unknown', :now, :now)
                            """),
                            {
                                "entity_id": interval["entity_id"],
                                "now": dt_util.utcnow(),
                            },
                        )

                        # Save interval
                        values = SchemaConverter.state_interval_to_dict(interval)
                        stmt = sa.text("""
                            INSERT OR IGNORE INTO state_intervals
                            (entity_id, state, start_time, end_time, duration_seconds)
                            VALUES (:entity_id, :state, :start_time, :end_time, :duration_seconds)
                        """)

                        result = conn.execute(stmt, values)
                        if result.rowcount > 0:
                            stored_count += 1

                    except (sa.exc.SQLAlchemyError, OSError) as err:
                        _LOGGER.warning("Failed to store interval: %s", err)

                conn.commit()
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
        """Import state intervals from HA recorder to global table."""
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
                # Get intervals from recorder
                intervals = await _get_intervals_from_recorder(
                    self.hass, entity_id, start_time, end_time
                )

                if intervals:
                    # Store intervals efficiently
                    count = await self.save_state_intervals_batch(intervals)
                    import_counts[entity_id] = count
                    _LOGGER.debug("Imported %d intervals for %s", count, entity_id)
                else:
                    import_counts[entity_id] = 0

            except (sa.exc.SQLAlchemyError, OSError) as err:
                _LOGGER.error("Failed to import intervals for %s: %s", entity_id, err)
                import_counts[entity_id] = 0

        total_imported = sum(import_counts.values())
        _LOGGER.info("Import completed: %d total intervals imported", total_imported)

        return import_counts

    # ─────────────────── Cleanup Methods ───────────────────

    async def cleanup_old_area_history(self, entry_id: str, days: int = 30) -> int:
        """Remove area history records older than specified days."""
        cutoff_date = dt_util.utcnow() - timedelta(days=days)

        def _cleanup():
            with self.engine.connect() as conn:
                result = conn.execute(
                    area_history_table.delete().where(
                        sa.and_(
                            area_history_table.c.entry_id == entry_id,
                            area_history_table.c.timestamp < cutoff_date,
                        )
                    )
                )
                deleted_count = result.rowcount
                conn.commit()
                return deleted_count

        deleted_count = await self.hass.async_add_executor_job(_cleanup)
        _LOGGER.info(
            "Cleaned up %d area history records older than %d days for entry %s",
            deleted_count,
            days,
            entry_id,
        )
        return deleted_count

    async def reset_entry_data(self, entry_id: str) -> None:
        """Remove all data for a specific entry (area-specific only)."""

        def _reset():
            with self.engine.connect() as conn:
                # Delete area-specific data only (preserve global entities and intervals)
                conn.execute(
                    area_history_table.delete().where(
                        area_history_table.c.entry_id == entry_id
                    )
                )
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
                stats["area_history_count"] = conn.execute(
                    sa.select(sa.func.count()).select_from(area_history_table)
                ).scalar()
                stats["state_intervals_count"] = conn.execute(
                    sa.select(sa.func.count()).select_from(state_intervals_table)
                ).scalar()

                # Entry-specific stats
                stats[f"area_entity_config_entry_{self.entry_id}"] = conn.execute(
                    sa.select(sa.func.count())
                    .select_from(area_entity_config_table)
                    .where(area_entity_config_table.c.entry_id == self.entry_id)
                ).scalar()

            # Database file size
            try:
                stats["db_size_bytes"] = self.db_path.stat().st_size
            except FileNotFoundError:
                stats["db_size_bytes"] = 0

            return stats

        return await self.hass.async_add_executor_job(_get_stats)


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
                probability=self._coordinator.probability,
                prior=self._coordinator.area_prior,
                threshold=self._coordinator.threshold,
                occupied=self._coordinator.occupied,
            )
            await self._storage.save_area_occupancy(area_record)

            # Save area-specific entity configurations
            for entity in self._coordinator.entities.entities.values():
                # Get entity state from HA
                ha_state = self._coordinator.hass.states.get(entity.entity_id)
                entity_attributes = dict(ha_state.attributes) if ha_state else {}

                # Create area-specific entity config record
                entity_config = AreaEntityConfigRecord(
                    entry_id=self._coordinator.entry_id,
                    entity_id=entity.entity_id,
                    entity_type=entity.type.input_type.value,
                    weight=entity.type.weight,
                    probability=entity.probability,
                    prob_given_true=entity.likelihood.prob_given_true,
                    prob_given_false=entity.likelihood.prob_given_false,
                    last_state=str(entity.state) if entity.state is not None else None,
                    attributes=entity_attributes,
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
                # Create entity dict in old format for compatibility
                entities_data[config.entity_id] = {
                    "entity_id": config.entity_id,
                    "entity_type": config.entity_type,
                    "probability": config.probability,
                    "last_state": config.last_state,
                    "last_updated": config.last_updated.isoformat(),
                    "attributes": config.attributes,
                    # Include likelihood data for restoration
                    "likelihood": {
                        "prob_given_true": config.prob_given_true,
                        "prob_given_false": config.prob_given_false,
                        "last_updated": config.last_updated.isoformat(),
                    },
                    # Include basic type data
                    "type": {
                        "input_type": config.entity_type,
                        "weight": config.weight,
                        "prob_true": config.prob_given_true,
                        "prob_false": config.prob_given_false,
                        "prior": 0.5,  # Default prior
                        "active_states": None,  # Will be set from entity type
                        "active_range": None,  # Will be set from entity type
                    },
                    # Include basic decay data (will be reset)
                    "decay": {
                        "last_trigger_ts": dt_util.utcnow().timestamp(),
                        "half_life": self._coordinator.config.decay.half_life,
                        "is_decaying": False,
                    },
                    "previous_evidence": None,
                    "previous_probability": config.probability,
                }

            return {
                "name": area_record.area_name,
                "purpose": area_record.purpose,
                "probability": area_record.probability,
                "prior": area_record.prior,
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
        history_record = AreaHistoryRecord(
            entry_id=self._coordinator.entry_id,
            entity_id=entity_id,
            probability_change=probability_change,
            context=context or {},
        )
        await self._storage.save_area_history(history_record)

    async def async_get_history(
        self,
        entity_id: str | None = None,
        days: int = 7,
        limit: int | None = None,
    ) -> list[AreaHistoryRecord]:
        """Get area history records for analysis."""
        since = dt_util.utcnow() - timedelta(days=days)
        return await self._storage.get_area_history(
            self._coordinator.entry_id, entity_id=entity_id, since=since, limit=limit
        )

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
