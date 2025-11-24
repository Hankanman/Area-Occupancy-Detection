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
from homeassistant.util import dt as dt_util

from ..const import (
    CONF_VERSION,
    DEFAULT_BACKUP_INTERVAL_HOURS,
    DEFAULT_ENABLE_AUTO_RECOVERY,
    DEFAULT_ENABLE_PERIODIC_BACKUPS,
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
)
from . import (
    aggregation,
    correlation,
    maintenance,
    operations,
    queries,
    relationships,
    sync,
    utils,
)
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
        await operations.ensure_area_exists(self)

    def get_latest_interval(self) -> datetime:
        """Return the latest interval end time minus 1 hour, or default window if none."""
        return queries.get_latest_interval(self)

    def prune_old_intervals(self, force: bool = False) -> int:
        """Delete intervals older than RETENTION_DAYS."""
        return operations.prune_old_intervals(self, force)

    def get_time_prior(
        self,
        area_name: str,
        day_of_week: int,
        time_slot: int,
        default_prior: float = 0.5,
    ) -> float:
        """Get the time prior for a specific time slot.

        Args:
            area_name: The area name to filter by
            day_of_week: Day of week (0=Monday, 6=Sunday)
            time_slot: Time slot index
            default_prior: Default prior value if not found

        Returns:
            Time prior value or default if not found
        """
        return queries.get_time_prior(
            self,
            self.coordinator.entry_id,
            area_name,
            day_of_week,
            time_slot,
            default_prior,
        )

    # Attach sync methods
    async def sync_states(self) -> None:
        """Fetch states history from recorder and commit to Intervals table for all areas."""
        await sync.sync_states(self)

    # Attach aggregation methods
    def aggregate_raw_to_daily(self, area_name: str | None = None) -> int:
        """Aggregate raw intervals to daily aggregates."""
        return aggregation.aggregate_raw_to_daily(self, area_name)

    def aggregate_daily_to_weekly(self, area_name: str | None = None) -> int:
        """Aggregate daily aggregates to weekly aggregates."""
        return aggregation.aggregate_daily_to_weekly(self, area_name)

    def aggregate_weekly_to_monthly(self, area_name: str | None = None) -> int:
        """Aggregate weekly aggregates to monthly aggregates."""
        return aggregation.aggregate_weekly_to_monthly(self, area_name)

    def run_interval_aggregation(
        self, area_name: str | None = None, force: bool = False
    ) -> dict[str, int]:
        """Run the full tiered aggregation process for intervals."""
        return aggregation.run_interval_aggregation(self, area_name, force)

    def prune_old_aggregates(self, area_name: str | None = None) -> dict[str, int]:
        """Prune old aggregates based on retention policies."""
        return aggregation.prune_old_aggregates(self, area_name)

    def prune_old_numeric_samples(self, area_name: str | None = None) -> int:
        """Prune old raw numeric samples based on retention policy."""
        return aggregation.prune_old_numeric_samples(self, area_name)

    # Attach correlation methods
    def analyze_correlation(
        self,
        area_name: str,
        entity_id: str,
        analysis_period_days: int = 30,
    ) -> dict[str, Any] | None:
        """Analyze correlation between sensor values and occupancy."""
        return correlation.analyze_correlation(
            self, area_name, entity_id, analysis_period_days
        )

    def save_correlation_result(self, correlation_data: dict[str, Any]) -> bool:
        """Save correlation analysis result to database."""
        return correlation.save_correlation_result(self, correlation_data)

    def analyze_and_save_correlation(
        self,
        area_name: str,
        entity_id: str,
        analysis_period_days: int = 30,
    ) -> dict[str, Any] | None:
        """Analyze and save correlation for a numeric sensor."""
        return correlation.analyze_and_save_correlation(
            self, area_name, entity_id, analysis_period_days
        )

    def get_correlation_for_entity(
        self, area_name: str, entity_id: str
    ) -> dict[str, Any] | None:
        """Get the most recent correlation result for an entity."""
        return correlation.get_correlation_for_entity(self, area_name, entity_id)

    # Attach relationship methods
    def save_area_relationship(
        self,
        area_name: str,
        related_area_name: str,
        relationship_type: str = "adjacent",
        influence_weight: float | None = None,
        distance: float | None = None,
    ) -> bool:
        """Save or update an area relationship."""
        return relationships.save_area_relationship(
            self,
            area_name,
            related_area_name,
            relationship_type,
            influence_weight,
            distance,
        )

    def get_adjacent_areas(self, area_name: str) -> list[dict[str, Any]]:
        """Get all adjacent/related areas for an area."""
        return relationships.get_adjacent_areas(self, area_name)

    def get_influence_weight(self, area_name: str, related_area_name: str) -> float:
        """Get the influence weight between two areas."""
        return relationships.get_influence_weight(self, area_name, related_area_name)

    def calculate_adjacent_influence(
        self, area_name: str, base_probability: float
    ) -> float:
        """Calculate probability adjustment based on adjacent area occupancy."""
        return relationships.calculate_adjacent_influence(
            self, area_name, base_probability
        )

    def sync_adjacent_areas_from_config(self, area_name: str) -> bool:
        """Sync adjacent areas from area configuration to AreaRelationships table."""
        return relationships.sync_adjacent_areas_from_config(self, area_name)

    # Attach prior methods
    def save_global_prior(
        self,
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
        """Save global prior calculation to GlobalPriors table."""
        return operations.save_global_prior(
            self,
            area_name,
            prior_value,
            data_period_start,
            data_period_end,
            total_occupied_seconds,
            total_period_seconds,
            interval_count,
            calculation_method,
            confidence,
        )

    def get_global_prior(self, area_name: str) -> dict[str, Any] | None:
        """Get the most recent global prior for an area."""
        return queries.get_global_prior(self, area_name)

    def save_occupied_intervals_cache(
        self,
        area_name: str,
        intervals: list[tuple[datetime, datetime]],
        data_source: str = "merged",
    ) -> bool:
        """Save occupied intervals to OccupiedIntervalsCache table."""
        return operations.save_occupied_intervals_cache(
            self, area_name, intervals, data_source
        )

    def get_occupied_intervals_cache(
        self,
        area_name: str,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> list[tuple[datetime, datetime]]:
        """Get occupied intervals from OccupiedIntervalsCache table."""
        return queries.get_occupied_intervals_cache(
            self, area_name, period_start, period_end
        )

    def get_occupied_intervals(
        self,
        area_name: str,
        motion_sensor_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_media: bool = False,
        include_appliance: bool = False,
        media_sensor_ids: list[str] | None = None,
        appliance_sensor_ids: list[str] | None = None,
        motion_timeout: int = 300,  # Default 5 min if not specified
    ) -> list[tuple[datetime, datetime]]:
        """Get raw occupied intervals without using cache.

        This delegates to queries.get_occupied_intervals but adapts arguments
        to match what PriorAnalyzer expects.
        """
        # Determine lookback days if start_time provided
        lookback_days = 90  # default
        if start_time:
            lookback_days = (dt_util.utcnow() - start_time).days + 1

        return queries.get_occupied_intervals(
            self,
            self.coordinator.entry_id,
            area_name,
            lookback_days,
            motion_timeout,
            include_media,
            include_appliance,
            media_sensor_ids,
            appliance_sensor_ids,
        )

    def is_occupied_intervals_cache_valid(
        self, area_name: str, max_age_hours: int = 24
    ) -> bool:
        """Check if cached occupied intervals are still valid."""
        return queries.is_occupied_intervals_cache_valid(self, area_name, max_age_hours)

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
