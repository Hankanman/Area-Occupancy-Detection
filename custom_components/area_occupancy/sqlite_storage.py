"""SQLite storage module for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import create_engine

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import HA_RECORDER_DAYS
from .data.entity_type import _ENTITY_TYPE_DATA, InputType
from .schema import (
    AreaEntityConfigRecord,
    AreaOccupancyRecord,
    SchemaConverter,
    area_entity_config_table,
    area_occupancy_table,
    area_time_priors_table,
    entities_table,
    metadata,
    metadata_table,
    state_intervals_table,
)
from .state_intervals import StateInterval, get_intervals_from_recorder

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Database constants
DB_NAME = "area_occupancy.db"
# Default retention for state interval cleanup
DEFAULT_RETENTION_DAYS = 365


class AreaOccupancyStorage:
    """Unified storage for Area Occupancy Detection, combining DB access and coordinator logic."""

    # Merge all methods and logic from SQLiteStorage and AreaOccupancySQLiteStore here.
    # Accept both hass/entry_id and coordinator in __init__, and set up attributes accordingly.
    # All public APIs from both classes should be preserved.

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
        self.storage_path = Path(hass.config.config_dir) / ".storage" if hass else None
        self.db_path = self.storage_path / DB_NAME if self.storage_path else None
        self.import_stats: dict[str, int] = {}

        # Ensure storage directory exists
        if self.storage_path:
            self.storage_path.mkdir(exist_ok=True)

        # Create SQLAlchemy engine
        self.engine = (
            create_engine(f"sqlite:///{self.db_path}") if self.db_path else None
        )

        _LOGGER.debug(
            "SQLite storage initialized for entry %s at %s", entry_id, self.db_path
        )

    def _enable_wal_mode(self) -> None:
        """Enable SQLite WAL mode for better concurrent writes."""
        if not self.engine:
            return
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text("PRAGMA journal_mode=WAL"))
        except sa.exc.SQLAlchemyError as err:
            _LOGGER.debug("Failed to enable WAL mode: %s", err)

    async def async_initialize(self) -> None:
        """Initialize the database schema using SQLAlchemy."""

        def _create_schema():
            try:
                self._enable_wal_mode()
                metadata.create_all(self.engine, checkfirst=True)
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

        await self.hass.async_add_executor_job(_create_schema)
        _LOGGER.info("SQLite storage initialized successfully")

    def _create_tables_individually(self) -> None:
        """Create tables individually to handle race conditions."""
        with self.engine.connect():
            for table in metadata.tables.values():
                try:
                    table.create(self.engine, checkfirst=True)
                except sa.exc.OperationalError as err:
                    if "already exists" in str(err).lower():
                        _LOGGER.debug("Table %s already exists, skipping", table.name)
                        continue
                    raise

    def _execute_with_retry(
        self, func, max_retries: int = 3, initial_delay: float = 0.1
    ):
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
                stmt = sa.text(
                    """
                    INSERT OR REPLACE INTO area_occupancy
                    (entry_id, area_name, purpose, threshold, created_at, updated_at)
                    VALUES (:entry_id, :area_name, :purpose, :threshold,
                            COALESCE((SELECT created_at FROM area_occupancy WHERE entry_id = :entry_id), :created_at),
                            :updated_at)
                """
                )

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
            def _do_save():
                with self.engine.connect() as conn:
                    # Ensure global entity exists
                    conn.execute(
                        sa.text(
                            """
                            INSERT OR IGNORE INTO entities (entity_id, last_seen, created_at)
                            VALUES (:entity_id, :now, :now)
                        """
                        ),
                        {
                            "entity_id": record.entity_id,
                            "now": dt_util.utcnow(),
                        },
                    )

                    # Save area-specific config
                    values = SchemaConverter.area_entity_config_to_dict(record)
                    stmt = sa.text(
                        """
                        INSERT OR REPLACE INTO area_entity_config
                        (entry_id, entity_id, entity_type, weight,
                         prob_given_true, prob_given_false, last_updated)
                        VALUES (:entry_id, :entity_id, :entity_type, :weight,
                                :prob_given_true, :prob_given_false, :last_updated)
                    """
                    )

                    conn.execute(stmt, values)
                    conn.commit()
                    return record

            return self._execute_with_retry(_do_save)

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

    # ─────────────────── Global State Intervals Methods ───────────────────

    async def save_state_intervals_batch(self, intervals: list[StateInterval]) -> int:
        """Save multiple state intervals efficiently to global table."""
        if not intervals:
            _LOGGER.debug("No intervals to save")
            return 0

        _LOGGER.debug("Saving batch of %d intervals", len(intervals))

        def _save():
            def _do_save():
                stored_count = 0
                with self.engine.connect() as conn:
                    # First, ensure all entities exist in a single executemany call
                    unique_entities = {interval["entity_id"] for interval in intervals}
                    entity_values = [
                        {"entity_id": entity_id, "now": dt_util.utcnow()}
                        for entity_id in unique_entities
                    ]
                    if entity_values:
                        conn.execute(
                            sa.text(
                                """
                                INSERT OR IGNORE INTO entities (entity_id, last_seen, created_at)
                                VALUES (:entity_id, :now, :now)
                                """
                            ),
                            entity_values,
                        )

                    # Commit entities first
                    conn.commit()

                    # Prepare all interval values for bulk insert
                    values_list = [
                        SchemaConverter.state_interval_to_dict(interval)
                        for interval in intervals
                    ]

                    if not values_list:
                        _LOGGER.debug("No intervals to insert after conversion")
                        return 0

                    stmt = sa.text(
                        """
                        INSERT OR IGNORE INTO state_intervals
                        (entity_id, state, start_time, end_time, duration_seconds, created_at)
                        VALUES (:entity_id, :state, :start_time, :end_time, :duration_seconds, :created_at)
                        """
                    )

                    result = conn.execute(stmt, values_list)
                    conn.commit()

                    # SQLAlchemy 1.4+ result.rowcount is total rows inserted (may be -1 for some DBs)
                    if (
                        hasattr(result, "rowcount")
                        and result.rowcount is not None
                        and result.rowcount >= 0
                    ):
                        stored_count = result.rowcount
                    else:
                        # Fallback: count attempted inserts (not always accurate with OR IGNORE)
                        stored_count = len(values_list)

                    _LOGGER.debug(
                        "Committed batch: attempted %d, stored %d intervals successfully",
                        len(values_list),
                        stored_count,
                    )
                    return stored_count

            try:
                return self._execute_with_retry(_do_save)
            except (sa.exc.SQLAlchemyError, OSError) as err:
                _LOGGER.warning("Failed to save batch of state intervals: %s", err)
                return 0

        return await self.hass.async_add_executor_job(_save)

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

    async def cleanup_old_intervals(
        self, retention_days: int = DEFAULT_RETENTION_DAYS
    ) -> int:
        """Remove state intervals older than the retention period.

        The default retention period keeps roughly one year of history to
        balance accuracy with storage size.
        """
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

    async def async_save_data(self) -> None:
        """Save coordinator data to SQLite storage using normalized schema."""
        if not self.coordinator:
            raise RuntimeError("Coordinator is required for async_save_data")
        try:
            # Save area occupancy data
            area_record = AreaOccupancyRecord(
                entry_id=self.coordinator.entry_id,
                area_name=self.coordinator.config.name,
                purpose=self.coordinator.config.purpose,
                threshold=self.coordinator.threshold,
            )
            await self.save_area_occupancy(area_record)

            # Save area-specific entity configurations
            for entity in self.coordinator.entities.entities.values():
                entity_config = AreaEntityConfigRecord(
                    entry_id=self.coordinator.entry_id,
                    entity_id=entity.entity_id,
                    entity_type=entity.type.input_type.value,
                    weight=entity.type.weight,
                    prob_given_true=entity.likelihood.prob_given_true,
                    prob_given_false=entity.likelihood.prob_given_false,
                )
                await self.save_area_entity_config(entity_config)

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
            entity_configs = await self.get_area_entity_configs(
                self.coordinator.entry_id
            )
            entities_data = {}
            for config in entity_configs:
                try:
                    input_type = InputType(config.entity_type)
                    type_defaults = _ENTITY_TYPE_DATA.get(input_type, {})
                except (ValueError, KeyError):
                    type_defaults = _ENTITY_TYPE_DATA.get(InputType.MOTION, {})
                    _LOGGER.warning(
                        "Unknown entity type %s for %s, using motion defaults",
                        config.entity_type,
                        config.entity_id,
                    )
                entities_data[config.entity_id] = {
                    "entity_id": config.entity_id,
                    "entity_type": config.entity_type,
                    "last_updated": config.last_updated.isoformat(),
                    "likelihood": {
                        "prob_given_true": config.prob_given_true,
                        "prob_given_false": config.prob_given_false,
                        "last_updated": config.last_updated.isoformat(),
                    },
                    "type": {
                        "input_type": config.entity_type,
                        "weight": config.weight,
                        "prob_true": config.prob_given_true,
                        "prob_false": config.prob_given_false,
                        "prior": type_defaults.get("prior", 0.5),
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
                "name": area_record.area_name,
                "purpose": area_record.purpose,
                "probability": self.coordinator.probability,
                "prior": self.coordinator.area_prior,
                "threshold": area_record.threshold,
                "last_updated": area_record.updated_at.isoformat(),
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
            if self.engine:
                self.engine.dispose()

        await self.hass.async_add_executor_job(_dispose)
