"""Core database management functionality."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock, Timeout
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker as create_sessionmaker

from homeassistant.exceptions import HomeAssistantError

from ..const import (
    CONF_VERSION,
    DEFAULT_BACKUP_INTERVAL_HOURS,
    DEFAULT_ENABLE_AUTO_RECOVERY,
    DEFAULT_ENABLE_PERIODIC_BACKUPS,
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
)
from . import maintenance, operations, queries, sync, utils
from .constants import DB_NAME
from .schema import (
    AreaRelationships,
    Areas,
    CrossAreaStats,
    Entities,
    EntityStatistics,
    GlobalPriors,
    IntervalAggregates,
    Intervals,
    Metadata,
    NumericAggregates,
    NumericCorrelations,
    NumericSamples,
    OccupiedIntervalsCache,
    Priors,
)

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyDB:
    """A class to manage area occupancy database operations."""

    # Reference schema models as class attributes
    Areas = Areas
    Entities = Entities
    Intervals = Intervals
    Priors = Priors
    Metadata = Metadata
    IntervalAggregates = IntervalAggregates
    OccupiedIntervalsCache = OccupiedIntervalsCache
    GlobalPriors = GlobalPriors
    NumericSamples = NumericSamples
    NumericAggregates = NumericAggregates
    NumericCorrelations = NumericCorrelations
    EntityStatistics = EntityStatistics
    AreaRelationships = AreaRelationships
    CrossAreaStats = CrossAreaStats

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
    ):
        """Initialize SQLite storage.

        Args:
            coordinator: AreaOccupancyCoordinator instance
        """
        self.coordinator = coordinator
        # config_entry is always present in a properly initialized coordinator
        if coordinator.config_entry is None:
            raise ValueError("Coordinator config_entry cannot be None")
        self.conf_version = coordinator.config_entry.data.get("version", CONF_VERSION)
        self.hass = coordinator.hass
        self.storage_path = (
            Path(self.hass.config.config_dir) / ".storage" if self.hass else None
        )
        self.db_path = self.storage_path / DB_NAME if self.storage_path else None

        # Database recovery configuration - use standard constants
        self.enable_auto_recovery = DEFAULT_ENABLE_AUTO_RECOVERY
        self.max_recovery_attempts = DEFAULT_MAX_RECOVERY_ATTEMPTS
        self.enable_periodic_backups = DEFAULT_ENABLE_PERIODIC_BACKUPS
        self.backup_interval_hours = DEFAULT_BACKUP_INTERVAL_HOURS

        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            pool_pre_ping=True,
            poolclass=sa.pool.NullPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 10,
            },
        )

        # Ensure storage directory exists
        if self.storage_path:
            self.storage_path.mkdir(exist_ok=True)

        # Create session maker
        self._session_maker = create_sessionmaker(bind=self.engine)
        # Debounce timestamps
        self.last_area_save_ts: float = 0.0
        self.last_entities_save_ts: float = 0.0
        self._save_debounce_seconds: float = 1.5

        # Create model classes dictionary for ORM
        self.model_classes = {
            "Areas": self.Areas,
            "Entities": self.Entities,
            "Priors": self.Priors,
            "Intervals": self.Intervals,
            "Metadata": self.Metadata,
            "IntervalAggregates": self.IntervalAggregates,
            "OccupiedIntervalsCache": self.OccupiedIntervalsCache,
            "GlobalPriors": self.GlobalPriors,
            "NumericSamples": self.NumericSamples,
            "NumericAggregates": self.NumericAggregates,
            "NumericCorrelations": self.NumericCorrelations,
            "EntityStatistics": self.EntityStatistics,
            "AreaRelationships": self.AreaRelationships,
            "CrossAreaStats": self.CrossAreaStats,
        }

        # Initialize database lock path
        self._lock_path = (
            self.storage_path / (DB_NAME + ".lock") if self.storage_path else None
        )

        # Auto-initialize database in test environments
        if os.getenv("AREA_OCCUPANCY_AUTO_INIT_DB") == "1":
            self.initialize_database()

    def initialize_database(self) -> None:
        """Initialize the database by checking if it exists and creating it if needed.

        This method performs blocking I/O operations.
        In production environments, it should be called via
        hass.async_add_executor_job() to avoid blocking the event loop.
        In test environments (when AREA_OCCUPANCY_AUTO_INIT_DB=1 is set),
        this method may be called directly.
        """
        # Check if database exists and initialize if needed
        maintenance.ensure_db_exists(self)

    # Attach maintenance methods to the class
    def check_database_integrity(self) -> bool:
        """Check if the database is healthy and not corrupted."""
        return maintenance.check_database_integrity(self)

    def check_database_accessibility(self) -> bool:
        """Check if the database file is accessible and readable."""
        return maintenance.check_database_accessibility(self)

    def is_database_corrupted(self, error: Exception) -> bool:
        """Check if an error indicates database corruption."""
        return maintenance.is_database_corrupted(self, error)

    def attempt_database_recovery(self) -> bool:
        """Attempt to recover from database corruption."""
        return maintenance.attempt_database_recovery(self)

    def backup_database(self) -> bool:
        """Create a backup of the current database."""
        return maintenance.backup_database(self)

    def restore_database_from_backup(self) -> bool:
        """Restore database from backup if available."""
        return maintenance.restore_database_from_backup(self)

    def handle_database_corruption(self) -> bool:
        """Handle database corruption with automatic recovery attempts."""
        return maintenance.handle_database_corruption(self)

    def periodic_health_check(self) -> bool:
        """Perform periodic database health check and maintenance."""
        return maintenance.periodic_health_check(self)

    def set_db_version(self) -> None:
        """Set the database version in the metadata table."""
        maintenance.set_db_version(self)

    def get_db_version(self) -> int:
        """Get the database version from the metadata table."""
        return maintenance.get_db_version(self)

    def delete_db(self) -> None:
        """Delete the database file."""
        maintenance.delete_db(self)

    def force_reinitialize(self) -> None:
        """Force reinitialization of the database tables."""
        maintenance.force_reinitialize(self)

    def init_db(self) -> None:
        """Initialize the database with WAL mode."""
        maintenance.init_db(self)

    # Attach operations methods to the class
    async def load_data(self) -> None:
        """Load the data from the database for all areas."""
        await operations.load_data(self)

    def save_area_data(self, area_name: str | None = None) -> None:
        """Save the area data to the database."""
        operations.save_area_data(self, area_name)

    def save_entity_data(self) -> None:
        """Save the entity data to the database for all areas."""
        operations.save_entity_data(self)

    def save_data(self) -> None:
        """Save both area and entity data to the database."""
        operations.save_data(self)

    def cleanup_orphaned_entities(self) -> int:
        """Clean up entities from database that are no longer in the current configuration."""
        return operations.cleanup_orphaned_entities(self)

    def delete_area_data(self, area_name: str) -> int:
        """Delete all database data for a removed area."""
        return operations.delete_area_data(self, area_name)

    # Attach utility methods
    def is_valid_state(self, state: Any) -> bool:
        """Check if a state is valid."""
        return utils.is_valid_state(state)

    def is_intervals_empty(self) -> bool:
        """Check if the intervals table is empty using ORM (read-only, no lock)."""
        return utils.is_intervals_empty(self)

    def safe_is_intervals_empty(self) -> bool:
        """Safely check if intervals table is empty (fast, no integrity checks)."""
        return utils.safe_is_intervals_empty(self)

    # Attach query methods
    def get_area_data(self, entry_id: str) -> dict[str, Any] | None:
        """Get area data for a specific entry_id (read-only, no lock)."""
        return queries.get_area_data(self, entry_id)

    async def ensure_area_exists(self) -> None:
        """Ensure that the area record exists in the database."""
        await queries.ensure_area_exists(self)

    def get_latest_interval(self) -> datetime:
        """Return the latest interval end time minus 1 hour, or default window if none."""
        return queries.get_latest_interval(self)

    def prune_old_intervals(self, force: bool = False) -> int:
        """Delete intervals older than RETENTION_DAYS."""
        return queries.prune_old_intervals(self, force)

    def get_aggregated_intervals_by_slot(
        self,
        entry_id: str,
        slot_minutes: int = 60,
        area_name: str | None = None,
    ) -> list[tuple[int, int, float]]:
        """Get aggregated interval data using SQL GROUP BY for better performance."""
        return queries.get_aggregated_intervals_by_slot(
            self, entry_id, slot_minutes, area_name
        )

    def get_total_occupied_seconds_sql(
        self,
        entry_id: str,
        area_name: str | None = None,
        lookback_days: int = 90,
        motion_timeout_seconds: int = 0,
        include_media: bool = False,
        include_appliance: bool = False,
        media_sensor_ids: list[str] | None = None,
        appliance_sensor_ids: list[str] | None = None,
    ) -> float:
        """Get total occupied seconds using SQL aggregation for better performance."""
        return queries.get_total_occupied_seconds_sql(
            self,
            entry_id,
            area_name,
            lookback_days,
            motion_timeout_seconds,
            include_media,
            include_appliance,
            media_sensor_ids,
            appliance_sensor_ids,
        )

    # Attach sync methods
    async def sync_states(self) -> None:
        """Fetch states history from recorder and commit to Intervals table for all areas."""
        await sync.sync_states(self)

    @contextmanager
    def get_session(self) -> Any:
        """Get a database session with automatic cleanup.

        Yields:
            Session: A SQLAlchemy session

        Example:
            with self.get_session() as session:
                result = session.query(self.Areas).first()

        """
        session = self._session_maker()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def get_locked_session(self, timeout: int = 30) -> Any:
        """Get a database session with file locking to prevent concurrent access.

        Args:
            timeout: Maximum time to wait for lock acquisition in seconds

        Yields:
            Session: A SQLAlchemy session protected by file lock

        Example:
            with self.get_locked_session() as session:
                result = session.query(self.Areas).first()

        """
        if not self._lock_path:
            # Fallback to regular session if no lock path available
            with self.get_session() as session:
                yield session
            return

        try:
            with (
                FileLock(self._lock_path, timeout=timeout),
                self.get_session() as session,
            ):
                yield session
        except Timeout as e:
            _LOGGER.error("Database lock timeout after %d seconds: %s", timeout, e)
            raise HomeAssistantError(
                f"Database is busy, please try again later: {e}"
            ) from e

    # Table properties for cleaner access
    @property
    def areas(self) -> Any:
        """Get the areas table."""
        return self.Areas.__table__

    @property
    def entities(self) -> Any:
        """Get the entities table."""
        return self.Entities.__table__

    @property
    def intervals(self) -> Any:
        """Get the intervals table."""
        return self.Intervals.__table__

    @property
    def priors(self) -> Any:
        """Get the priors table."""
        return self.Priors.__table__

    @property
    def metadata(self) -> Any:
        """Get the metadata table."""
        return self.Metadata.__table__

    def get_engine(self) -> Any:
        """Get the engine for the database with optimized settings."""
        return self.engine

    def update_session_maker(self) -> None:
        """Update the session maker after engine changes (e.g., recovery/restore)."""
        self._session_maker = create_sessionmaker(bind=self.engine)
