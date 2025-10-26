"""Database schema and functions to interact with the database."""

from __future__ import annotations

# Standard library imports
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING, Any

# Third-party imports
from filelock import FileLock, Timeout
import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.exc import (
    DataError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Home Assistant imports
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance
from homeassistant.util import dt as dt_util

# Local imports
from .const import (
    CONF_VERSION,
    DEFAULT_BACKUP_INTERVAL_HOURS,
    DEFAULT_ENABLE_AUTO_RECOVERY,
    DEFAULT_ENABLE_PERIODIC_BACKUPS,
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
    MAX_INTERVAL_SECONDS,
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_INTERVAL_SECONDS,
    MIN_PROBABILITY,
    MIN_WEIGHT,
    RETENTION_DAYS,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase

    from .coordinator import AreaOccupancyCoordinator

    Base = DeclarativeBase
    Areas = "Areas"
    Entities = "Entities"
    Priors = "Priors"
    Intervals = "Intervals"
else:
    Base = declarative_base()

_LOGGER = logging.getLogger(__name__)

# States to exclude from intervals
INVALID_STATES = {"unknown", "unavailable", None, "", "NaN"}

DEFAULT_AREA_PRIOR = 0.15
DEFAULT_ENTITY_WEIGHT = 0.85
DEFAULT_ENTITY_PROB_GIVEN_TRUE = 0.8
DEFAULT_ENTITY_PROB_GIVEN_FALSE = 0.05
DB_NAME = "area_occupancy.db"

# Database metadata for Core access
metadata = MetaData()

# Database schema version for migrations
DB_VERSION = 3


class AreaOccupancyDB:
    """A class to manage area occupancy database operations."""

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
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
        )

        # Ensure storage directory exists
        if self.storage_path:
            self.storage_path.mkdir(exist_ok=True)

        # Create session maker
        self._session_maker = sessionmaker(bind=self.engine)

        # Create model classes dictionary for ORM
        self.model_classes = {
            "Areas": self.Areas,
            "Entities": self.Entities,
            "Priors": self.Priors,
            "Intervals": self.Intervals,
            "Metadata": self.Metadata,
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
        self._ensure_db_exists()

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

    class Areas(Base):
        """A table to store the area occupancy information."""

        __tablename__ = "areas"
        entry_id = Column(String, primary_key=True)
        area_name = Column(String, nullable=False)
        area_id = Column(String, nullable=False)
        purpose = Column(String, nullable=False)
        threshold = Column(Float, nullable=False)
        area_prior = Column(
            Float(precision=10),
            nullable=False,
            default=DEFAULT_AREA_PRIOR,
            server_default=str(DEFAULT_AREA_PRIOR),
        )
        created_at = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        updated_at = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        entities = relationship("Entities", back_populates="area")
        priors = relationship("Priors", back_populates="area")

        def to_dict(self) -> dict[str, Any]:
            """Convert the ORM object to a dictionary."""
            return {
                "entry_id": self.entry_id,
                "area_name": self.area_name,
                "area_id": self.area_id,
                "purpose": self.purpose,
                "threshold": self.threshold,
                "area_prior": self.area_prior,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Any:
            """Create an Areas instance from a dictionary."""
            # Handle area_id fallback - use entry_id if area_id is None or empty
            area_id = data.get("area_id")
            if not area_id:
                area_id = data["entry_id"]

            return cls(
                entry_id=data["entry_id"],
                area_name=data["area_name"],
                area_id=area_id,
                purpose=data["purpose"],
                threshold=data["threshold"],
                area_prior=data.get("area_prior", DEFAULT_AREA_PRIOR),
                created_at=data.get("created_at", dt_util.utcnow()),
                updated_at=data.get("updated_at", dt_util.utcnow()),
            )

    class Entities(Base):
        """A table to store the entity information."""

        __tablename__ = "entities"
        entry_id = Column(String, ForeignKey("areas.entry_id"), primary_key=True)
        entity_id = Column(String, primary_key=True)
        entity_type = Column(String, nullable=False)
        weight = Column(Float, nullable=False, default=DEFAULT_ENTITY_WEIGHT)
        prob_given_true = Column(
            Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_TRUE
        )
        prob_given_false = Column(
            Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_FALSE
        )
        last_updated = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        created_at = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        is_decaying = Column(Boolean, nullable=False, default=False)
        decay_start = Column(DateTime(timezone=True), nullable=True)
        evidence = Column(Boolean, nullable=False, default=False)
        intervals = relationship("Intervals", back_populates="entity")
        area = relationship("Areas", back_populates="entities")

        __table_args__ = (
            Index("idx_entities_entry", "entry_id"),
            Index("idx_entities_type", "entry_id", "entity_type"),
        )

        def to_dict(self) -> dict[str, Any]:
            """Convert the ORM object to a dictionary."""
            return {
                "entry_id": self.entry_id,
                "entity_id": self.entity_id,
                "entity_type": self.entity_type,
                "weight": self.weight,
                "prob_given_true": self.prob_given_true,
                "prob_given_false": self.prob_given_false,
                "last_updated": self.last_updated,
                "created_at": self.created_at,
                "is_decaying": self.is_decaying,
                "decay_start": self.decay_start,
                "evidence": self.evidence,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Any:
            """Create an Entities instance from a dictionary."""
            return cls(
                entry_id=data["entry_id"],
                entity_id=data["entity_id"],
                entity_type=data["entity_type"],
                weight=data.get("weight", DEFAULT_ENTITY_WEIGHT),
                prob_given_true=data.get(
                    "prob_given_true", DEFAULT_ENTITY_PROB_GIVEN_TRUE
                ),
                prob_given_false=data.get(
                    "prob_given_false", DEFAULT_ENTITY_PROB_GIVEN_FALSE
                ),
                last_updated=data.get("last_updated", dt_util.utcnow()),
                created_at=data.get("created_at", dt_util.utcnow()),
                is_decaying=data.get("is_decaying", False),
                decay_start=data.get("decay_start"),
                evidence=data.get("evidence", False),
            )

    class Priors(Base):
        """A table to store the area time priors."""

        __tablename__ = "priors"
        entry_id = Column(String, ForeignKey("areas.entry_id"), primary_key=True)
        day_of_week = Column(Integer, primary_key=True)
        time_slot = Column(Integer, primary_key=True)
        prior_value = Column(Float, nullable=False)
        data_points = Column(Integer, nullable=False)
        last_updated = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        area = relationship("Areas", back_populates="priors")

        __table_args__ = (
            Index("idx_priors_entry", "entry_id"),
            Index("idx_priors_day_slot", "day_of_week", "time_slot"),
            Index("idx_priors_last_updated", "last_updated"),
        )

        def to_dict(self) -> dict[str, Any]:
            """Convert the ORM object to a dictionary."""
            return {
                "entry_id": self.entry_id,
                "day_of_week": self.day_of_week,
                "time_slot": self.time_slot,
                "prior_value": self.prior_value,
                "data_points": self.data_points,
                "last_updated": self.last_updated,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Any:
            """Create a Priors instance from a dictionary."""
            return cls(
                entry_id=data["entry_id"],
                day_of_week=data["day_of_week"],
                time_slot=data["time_slot"],
                prior_value=data["prior_value"],
                data_points=data["data_points"],
                last_updated=data.get("last_updated", dt_util.utcnow()),
            )

    class Intervals(Base):
        """A table to store the state intervals."""

        __tablename__ = "intervals"
        id = Column(Integer, primary_key=True)
        entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
        state = Column(String, nullable=False)
        start_time = Column(DateTime(timezone=True), nullable=False)
        end_time = Column(DateTime(timezone=True), nullable=False)
        duration_seconds = Column(Float, nullable=False)
        created_at = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        entity = relationship("Entities", back_populates="intervals")

        # Add unique constraint on (entity_id, start_time, end_time)
        __table_args__ = (
            UniqueConstraint(
                "entity_id", "start_time", "end_time", name="uq_intervals_entity_time"
            ),
            # Performance indexes
            Index("idx_intervals_entity", "entity_id"),
            Index("idx_intervals_entity_time", "entity_id", "start_time", "end_time"),
            Index("idx_intervals_start_time", "start_time"),
            Index("idx_intervals_end_time", "end_time"),
        )

        def to_dict(self) -> dict[str, Any]:
            """Convert the ORM object to a dictionary."""
            return {
                "id": self.id,
                "entity_id": self.entity_id,
                "state": self.state,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_seconds": self.duration_seconds,
                "created_at": self.created_at,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Any:
            """Create an Intervals instance from a dictionary."""
            return cls(
                entity_id=data["entity_id"],
                state=data["state"],
                start_time=data["start_time"],
                end_time=data["end_time"],
                duration_seconds=data["duration_seconds"],
                created_at=data["created_at"],
            )

    class Metadata(Base):
        """A table to store the metadata."""

        __tablename__ = "metadata"
        key = Column(String, primary_key=True)
        value = Column(String, nullable=False)

    def _ensure_db_exists(self) -> None:
        """Check if the database exists and initialize it if needed.

        NOTE: This method only performs FAST validation (file existence and SQLite header).
        Heavy integrity checks are deferred to background tasks to avoid blocking startup.
        """
        try:
            # Fast validation: only check if file exists and has valid SQLite header
            if self.db_path and self.db_path.exists():
                # Quick check: read first 16 bytes to validate SQLite format
                try:
                    with open(self.db_path, "rb") as f:
                        header = f.read(16)
                        if not header.startswith(b"SQLite format 3"):
                            _LOGGER.warning(
                                "Database file is not a valid SQLite database, will be recreated"
                            )
                            self.delete_db()
                        else:
                            # File exists and is valid SQLite - use it as-is
                            # Corruption checks will be handled by background health check
                            _LOGGER.debug(
                                "Database file found, deferring integrity check to background"
                            )
                            return
                except (OSError, PermissionError) as e:
                    _LOGGER.warning("Cannot read database file: %s, will recreate", e)
                    # Will create new database below

            # Use direct engine connection during initialization
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                ).fetchone()
                if not result:
                    # No tables exist, initialize the database
                    _LOGGER.debug("No tables found, initializing database")
                    self.init_db()
                    self.set_db_version()
        except sa.exc.SQLAlchemyError as e:
            # Check if this is a corruption error
            if self.is_database_corrupted(e):
                _LOGGER.warning(
                    "Database may be corrupted (error: %s), will attempt recovery in background",
                    e,
                )
                # Don't block startup - let background health check handle it
                # Just log and continue
            else:
                # Database doesn't exist or is not initialized, create it
                _LOGGER.debug(
                    "Database error during table check, initializing database: %s", e
                )
                self.init_db()
                self.set_db_version()

    def check_database_integrity(self) -> bool:
        """Check if the database is healthy and not corrupted.

        Returns:
            bool: True if database is healthy, False if corrupted

        """
        try:
            with self.engine.connect() as conn:
                # Run SQLite integrity check
                result = conn.execute(text("PRAGMA integrity_check")).fetchone()
                if result and result[0] == "ok":
                    _LOGGER.debug("Database integrity check passed")
                    return True
                _LOGGER.error("Database integrity check failed: %s", result)
                return False
        except (sa.exc.SQLAlchemyError, OSError, PermissionError) as e:
            _LOGGER.error("Failed to run database integrity check: %s", e)
            return False

    def check_database_accessibility(self) -> bool:
        """Check if the database file is accessible and readable.

        Returns:
            bool: True if database is accessible, False otherwise

        """
        if not self.db_path or not self.db_path.exists():
            return False

        try:
            # Try to open the file to check if it's readable
            with open(self.db_path, "rb") as f:
                # Read first few bytes to check if file is accessible
                header = f.read(16)
                if not header.startswith(b"SQLite format 3"):
                    _LOGGER.error("Database file is not a valid SQLite database")
                    return False
        except (OSError, PermissionError, FileNotFoundError) as e:
            _LOGGER.error("Database file is not accessible: %s", e)
            return False
        else:
            return True

    def is_database_corrupted(self, error: Exception) -> bool:
        """Check if an error indicates database corruption.

        Args:
            error: The exception that occurred

        Returns:
            bool: True if the error indicates corruption, False otherwise

        """
        error_str = str(error).lower()
        corruption_indicators = [
            "database disk image is malformed",
            "corrupted",
            "file is not a database",
            "database or disk is full",
            "database is locked",
            "unable to open database file",
        ]

        return any(indicator in error_str for indicator in corruption_indicators)

    def attempt_database_recovery(self) -> bool:
        """Attempt to recover from database corruption.

        Returns:
            bool: True if recovery was successful, False otherwise

        """
        _LOGGER.warning("Attempting database recovery from corruption")

        try:
            # First, try to close all connections and recreate engine
            self.engine.dispose()

            # Try to enable WAL mode and run recovery
            temp_engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,
                pool_pre_ping=True,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 60,  # Longer timeout for recovery
                },
            )

            with temp_engine.connect() as conn:
                # Try to enable WAL mode
                with suppress(Exception):
                    conn.execute(text("PRAGMA journal_mode=WAL"))

                # Try to run recovery
                with suppress(Exception):
                    conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))

                # Test if we can read from the database
                result = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                ).fetchone()

                if result:
                    _LOGGER.info("Database recovery successful, database is readable")
                    # Replace the engine with the recovered one
                    self.engine = temp_engine
                    self._session_maker = sessionmaker(bind=self.engine)
                    return True
                _LOGGER.error("Database recovery failed, no tables found")
                return False

        except (sa.exc.SQLAlchemyError, OSError, PermissionError) as e:
            _LOGGER.error("Database recovery failed: %s", e)
            return False

    def backup_database(self) -> bool:
        """Create a backup of the current database.

        Returns:
            bool: True if backup was successful, False otherwise

        """
        if not self.db_path or not self.db_path.exists():
            return False

        try:
            backup_path = self.db_path.with_suffix(".db.backup")

            shutil.copy2(self.db_path, backup_path)
            _LOGGER.info("Database backup created at %s", backup_path)
        except (OSError, PermissionError, shutil.Error) as e:
            _LOGGER.error("Failed to create database backup: %s", e)
            return False
        else:
            return True

    def restore_database_from_backup(self) -> bool:
        """Restore database from backup if available.

        Returns:
            bool: True if restore was successful, False otherwise

        """
        if not self.db_path:
            return False

        backup_path = self.db_path.with_suffix(".db.backup")
        if not backup_path.exists():
            _LOGGER.warning("No backup found at %s", backup_path)
            return False

        try:
            # Close current engine
            self.engine.dispose()

            shutil.copy2(backup_path, self.db_path)

            # Recreate engine
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,
                pool_pre_ping=True,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30,
                },
            )
            self._session_maker = sessionmaker(bind=self.engine)

            _LOGGER.info("Database restored from backup")

        except (OSError, PermissionError, shutil.Error, sa.exc.SQLAlchemyError) as e:
            _LOGGER.error("Failed to restore database from backup: %s", e)
            return False
        else:
            return True

    def handle_database_corruption(self) -> bool:
        """Handle database corruption with automatic recovery attempts.

        Returns:
            bool: True if database is now healthy, False if all recovery attempts failed

        """
        if not self.enable_auto_recovery:
            _LOGGER.error("Database corruption detected but auto-recovery is disabled")
            return False

        _LOGGER.error("Database corruption detected, attempting recovery")

        # First, try to create a backup if possible
        if self.enable_periodic_backups:
            self.backup_database()

        # Try database recovery first
        if self.attempt_database_recovery():
            if self.check_database_integrity():
                _LOGGER.info("Database recovery successful")
                return True

        # If recovery failed, try to restore from backup
        if self.enable_periodic_backups and self.restore_database_from_backup():
            if self.check_database_integrity():
                _LOGGER.info("Database restore from backup successful")
                return True

        # If all else fails, delete and recreate the database
        _LOGGER.warning("All recovery attempts failed, recreating database")
        try:
            self.delete_db()
            self.init_db()
            self.set_db_version()
            _LOGGER.info("Database recreated successfully")
        except (sa.exc.SQLAlchemyError, OSError, PermissionError) as e:
            _LOGGER.error("Failed to recreate database: %s", e)
            return False
        else:
            return True

    def safe_database_operation(
        self, operation_name: str, operation_func: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Safely execute a database operation with automatic corruption handling.

        Args:
            operation_name: Name of the operation for execution
            operation_func: Function to execute
            *args: Positional arguments to pass to the operation function
            **kwargs: Keyword arguments to pass to the operation function

        Returns:
            Result of the operation function, or None if all recovery attempts failed

        """
        max_attempts = self.max_recovery_attempts
        for attempt in range(max_attempts):
            try:
                # Check database integrity before operation
                if not self.check_database_integrity():
                    _LOGGER.warning(
                        "Database integrity check failed before %s, attempting recovery",
                        operation_name,
                    )
                    if not self.handle_database_corruption():
                        _LOGGER.error(
                            "Failed to recover database after %d attempts", max_attempts
                        )
                        return None

                # Execute the operation
                return operation_func(*args, **kwargs)

            except sa.exc.DatabaseError as e:
                if self.is_database_corrupted(e):
                    _LOGGER.error(
                        "Database corruption detected during %s (attempt %d/%d): %s",
                        operation_name,
                        attempt + 1,
                        max_attempts,
                        e,
                    )

                    if attempt < max_attempts - 1:
                        if not self.handle_database_corruption():
                            _LOGGER.error(
                                "Database recovery failed, retrying operation"
                            )
                            continue
                    else:
                        _LOGGER.error(
                            "All recovery attempts failed for %s", operation_name
                        )
                        return None
                else:
                    # Re-raise non-corruption database errors
                    raise
            except Exception as e:
                _LOGGER.error("Unexpected error during %s: %s", operation_name, e)
                raise

        # If we get here, all recovery attempts failed
        return None

    def periodic_health_check(self) -> bool:
        """Perform periodic database health check and maintenance.

        Returns:
            bool: True if database is healthy, False if issues were found

        """
        try:
            # Check database integrity
            if not self.check_database_integrity():
                _LOGGER.warning("Periodic health check found database corruption")
                if self.handle_database_corruption():
                    _LOGGER.info("Database recovered during periodic health check")
                    return True
                _LOGGER.error("Failed to recover database during periodic health check")
                return False

            # Create periodic backup if enabled
            if self.enable_periodic_backups and self.db_path:
                backup_path = self.db_path.with_suffix(".db.backup")
                backup_interval_seconds = self.backup_interval_hours * 3600
                if (
                    not backup_path.exists()
                    or (time.time() - backup_path.stat().st_mtime)
                    > backup_interval_seconds
                ):
                    if self.backup_database():
                        _LOGGER.debug("Periodic database backup created")
                    else:
                        _LOGGER.warning("Failed to create periodic database backup")

            # Run database maintenance
            with suppress(Exception), self.engine.connect() as conn:
                # Optimize database
                conn.execute(text("PRAGMA optimize"))
                # Update statistics
                conn.execute(text("ANALYZE"))
                _LOGGER.debug("Database maintenance completed")

        except (sa.exc.SQLAlchemyError, OSError, PermissionError) as e:
            _LOGGER.error("Periodic health check failed: %s", e)
            return False
        else:
            return True

    def manual_recovery_trigger(self) -> bool:
        """Manually trigger database recovery for testing or manual intervention.

        Returns:
            bool: True if recovery was successful, False otherwise

        """
        _LOGGER.info("Manual database recovery triggered")

        # Force a health check and recovery
        if not self.check_database_integrity():
            _LOGGER.info("Database corruption detected during manual recovery")
            return self.handle_database_corruption()
        _LOGGER.info("Database is healthy, no recovery needed")
        return True

    def get_database_status(self) -> dict[str, Any]:
        """Get comprehensive database status information.

        Returns:
            dict: Database status information

        """
        status = {
            "database_path": str(self.db_path) if self.db_path else None,
            "database_exists": self.db_path.exists() if self.db_path else False,
            "database_accessible": self.check_database_accessibility()
            if self.db_path and self.db_path.exists()
            else False,
            "database_integrity": self.check_database_integrity()
            if self.db_path and self.db_path.exists()
            else False,
            "auto_recovery_enabled": self.enable_auto_recovery,
            "max_recovery_attempts": self.max_recovery_attempts,
            "periodic_backups_enabled": self.enable_periodic_backups,
            "backup_interval_hours": self.backup_interval_hours,
        }

        # Check for backup file
        if self.db_path:
            backup_path = self.db_path.with_suffix(".db.backup")
            status["backup_exists"] = backup_path.exists()
            if backup_path.exists():
                backup_age_hours = (time.time() - backup_path.stat().st_mtime) / 3600
                status["backup_age_hours"] = round(backup_age_hours, 2)  # type: ignore[assignment]

        return status

    def get_engine(self) -> Any:
        """Get the engine for the database with optimized settings."""
        return self.engine

    def set_db_version(self) -> None:
        """Set the database version in the metadata table."""
        # Use direct engine connection during initialization
        with self.engine.begin() as conn:
            try:
                # Try to update if exists, else insert
                result = conn.execute(
                    text("SELECT value FROM metadata WHERE key = 'db_version'")
                ).fetchone()
                if result:
                    conn.execute(
                        text(
                            "UPDATE metadata SET value = :value WHERE key = 'db_version'"
                        ),
                        {"value": str(DB_VERSION)},
                    )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO metadata (key, value) VALUES ('db_version', :value)"
                        ),
                        {"value": str(DB_VERSION)},
                    )
            except Exception as e:
                _LOGGER.error("Failed to set db_version in metadata table: %s", e)
                raise

    def get_db_version(self) -> int:
        """Get the database version from the metadata table."""
        with self.get_session() as session:
            try:
                metadata_entry = (
                    session.query(self.Metadata).filter_by(key="db_version").first()
                )
                return int(metadata_entry.value) if metadata_entry else 0
            except Exception as e:
                _LOGGER.error("Failed to get db_version from metadata table: %s", e)
                raise

    def delete_db(self) -> None:
        """Delete the database file."""
        if self.db_path and self.db_path.exists():
            try:
                self.db_path.unlink()
                _LOGGER.info("Deleted database at %s", self.db_path)
            except (OSError, PermissionError) as e:
                _LOGGER.error("Failed to delete database file: %s", e)

    def force_reinitialize(self) -> None:
        """Force reinitialization of the database tables."""
        _LOGGER.debug("Forcing database reinitialization")
        self.init_db()
        self.set_db_version()

    def _get_last_prune_time(self) -> datetime | None:
        """Get timestamp of last successful prune operation.

        Returns:
            datetime of last prune, or None if not recorded
        """
        try:
            with self.get_session() as session:
                result = (
                    session.query(self.Metadata)
                    .filter_by(key="last_prune_time")
                    .first()
                )
                if result:
                    return datetime.fromisoformat(result.value)
        except (ValueError, AttributeError, SQLAlchemyError, OSError) as e:
            _LOGGER.debug("Failed to get last prune time: %s", e)
        return None

    def _set_last_prune_time(self, timestamp: datetime, session: Any = None) -> None:
        """Record timestamp of successful prune operation.

        Args:
            timestamp: When the prune occurred
            session: Optional existing session to use (avoids nested locks)
        """
        try:
            if session is not None:
                # Use existing session to avoid nested lock acquisition
                existing = (
                    session.query(self.Metadata)
                    .filter_by(key="last_prune_time")
                    .first()
                )
                if existing:
                    existing.value = timestamp.isoformat()
                else:
                    session.add(
                        self.Metadata(
                            key="last_prune_time", value=timestamp.isoformat()
                        )
                    )
                session.commit()
            else:
                # Fallback to new locked session if not provided
                with self.get_locked_session() as new_session:
                    existing = (
                        new_session.query(self.Metadata)
                        .filter_by(key="last_prune_time")
                        .first()
                    )
                    if existing:
                        existing.value = timestamp.isoformat()
                    else:
                        new_session.add(
                            self.Metadata(
                                key="last_prune_time", value=timestamp.isoformat()
                            )
                        )
                    new_session.commit()
        except (SQLAlchemyError, OSError, ValueError) as e:
            _LOGGER.warning("Failed to record prune timestamp: %s", e)

    def init_db(self) -> None:
        """Initialize the database with WAL mode and race condition handling."""
        _LOGGER.debug("Starting database initialization")
        try:
            # Enable WAL mode for better concurrent writes
            self._enable_wal_mode()
            # Create all tables with checkfirst to avoid race conditions
            _LOGGER.debug("Creating database tables")
            Base.metadata.create_all(self.engine, checkfirst=True)
            _LOGGER.debug("Database tables created successfully")
        except sa.exc.OperationalError as err:
            # Handle race condition when multiple instances try to create tables
            if err.orig and hasattr(err.orig, "sqlite_errno"):
                if err.orig.sqlite_errno == 1:
                    _LOGGER.debug(
                        "Table already exists (race condition), continuing: %s", err
                    )
                    # Continue - other tables might still need to be created
                    # Try to create remaining tables individually
                    self._create_tables_individually()
                else:
                    _LOGGER.error("Database initialization failed: %s", err)
                    raise
            else:
                _LOGGER.error("Database initialization failed: %s", err)
                raise
        except Exception as err:
            _LOGGER.error("Database initialization failed: %s", err)
            raise

    def _enable_wal_mode(self) -> None:
        """Enable SQLite WAL mode for better concurrent writes."""
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text("PRAGMA journal_mode=WAL"))
        except sa.exc.SQLAlchemyError as err:
            _LOGGER.debug("Failed to enable WAL mode: %s", err)

    def _create_tables_individually(self) -> None:
        """Create tables individually to handle race conditions."""
        for table in Base.metadata.tables.values():
            try:
                table.create(self.engine, checkfirst=True)
            except sa.exc.OperationalError as err:
                if err.orig and hasattr(err.orig, "sqlite_errno"):
                    if err.orig.sqlite_errno == 1:
                        _LOGGER.debug("Table %s already exists, skipping", table.name)
                        continue
                raise

    # --- Load Data ---

    async def load_data(self) -> None:
        """Load the data from the database (optimized for parallel reads).

        This method uses a two-phase approach to allow multiple integration
        instances to load in parallel during startup:
        1. Phase 1: Read data without lock (parallel-safe)
        2. Phase 2: Only acquire lock if stale entities need deletion (rare)
        """

        def _read_data_operation() -> tuple[Any, list[Any], list[str]]:
            """Read data WITHOUT lock (parallel-safe)."""
            stale_entity_ids = []
            with self.get_session() as session:
                area = (
                    session.query(self.Areas)
                    .filter_by(entry_id=self.coordinator.entry_id)
                    .first()
                )
                entities = (
                    session.query(self.Entities)
                    .filter_by(entry_id=self.coordinator.entry_id)
                    .order_by(self.Entities.entity_id)
                    .all()
                )
                if entities:
                    for entity_obj in entities:
                        # Check if entity exists in current coordinator config
                        try:
                            self.coordinator.entities.get_entity(entity_obj.entity_id)
                        except ValueError:
                            # Entity not found in coordinator - identify if stale
                            should_delete = False
                            if hasattr(self.coordinator.entities, "entity_ids"):
                                current_entity_ids = set(
                                    self.coordinator.entities.entity_ids
                                )
                                if entity_obj.entity_id not in current_entity_ids:
                                    should_delete = True
                            elif hasattr(self.coordinator.entities, "entities"):
                                # Fallback for mock objects that have entities dict
                                current_entity_ids = set(
                                    self.coordinator.entities.entities.keys()
                                )
                                if entity_obj.entity_id not in current_entity_ids:
                                    should_delete = True
                            else:
                                # Can't determine current config - assume entity is stale
                                should_delete = True

                            if should_delete:
                                stale_entity_ids.append(entity_obj.entity_id)
            return area, entities, stale_entity_ids

        def _delete_stale_operation(stale_ids: list[str]) -> None:
            """Delete stale entities WITH lock (only if needed)."""
            with self.get_locked_session() as session:
                for entity_id in stale_ids:
                    _LOGGER.info(
                        "Deleting stale entity %s from database (not in current config)",
                        entity_id,
                    )
                    session.query(self.Entities).filter_by(
                        entry_id=self.coordinator.entry_id, entity_id=entity_id
                    ).delete()
                session.commit()

        try:
            # Phase 1: Read without lock (all instances in parallel)
            area, entities, stale_ids = await self.hass.async_add_executor_job(
                _read_data_operation
            )

            # Update prior from area data
            if area:
                self.coordinator.prior.set_global_prior(area.area_prior)

            # Process entities
            if entities:
                for entity_obj in entities:
                    if entity_obj.entity_id in stale_ids:
                        # Skip stale entities, will be deleted in phase 2
                        continue

                    # Try to get existing entity from coordinator
                    try:
                        existing_entity = self.coordinator.entities.get_entity(
                            entity_obj.entity_id
                        )
                        # Update existing entity with database values (preserve database timestamp)
                        existing_entity.update_decay(
                            entity_obj.decay_start,
                            entity_obj.is_decaying,
                        )
                        existing_entity.update_likelihood(
                            entity_obj.prob_given_true,
                            entity_obj.prob_given_false,
                        )
                        # DB weight takes priority over configured defaults when valid
                        if hasattr(existing_entity, "type") and hasattr(
                            existing_entity.type, "weight"
                        ):
                            try:
                                weight_val = float(entity_obj.weight)
                                if MIN_WEIGHT <= weight_val <= MAX_WEIGHT:
                                    existing_entity.type.weight = weight_val
                            except (TypeError, ValueError):
                                pass
                        existing_entity.last_updated = entity_obj.last_updated
                        existing_entity.previous_evidence = entity_obj.evidence
                        _LOGGER.debug(
                            "Updated existing entity %s with database values",
                            entity_obj.entity_id,
                        )
                    except ValueError:
                        # Entity should exist but doesn't - create it from database
                        # (This handles cases where we can't determine current config, like in tests)
                        _LOGGER.warning(
                            "Entity %s not found in coordinator but is in config, creating from database",
                            entity_obj.entity_id,
                        )
                        new_entity = self.coordinator.factory.create_from_db(entity_obj)
                        self.coordinator.entities.add_entity(new_entity)

            # Phase 2: Only lock if cleanup needed (rare)
            if stale_ids:
                await self.hass.async_add_executor_job(
                    _delete_stale_operation, stale_ids
                )

            _LOGGER.debug("Loaded area occupancy data")

        except (
            sa.exc.SQLAlchemyError,
            HomeAssistantError,
            TimeoutError,
            OSError,
            RuntimeError,
        ) as err:
            _LOGGER.error("Failed to load area occupancy data: %s", err)
            # Don't raise the error, just log it and continue
            # This allows the integration to start even if data loading fails

    # --- Save Data ---

    def save_area_data(self) -> None:
        """Save the area data to the database."""
        try:
            with self.get_locked_session() as session:
                cfg = self.coordinator.config

                area_data = {
                    "entry_id": self.coordinator.entry_id,
                    "area_name": cfg.name,
                    "area_id": cfg.area_id,
                    "purpose": cfg.purpose,
                    "threshold": cfg.threshold,
                    "area_prior": self.coordinator.area_prior,
                    "updated_at": dt_util.utcnow(),
                }

                _LOGGER.debug("Attempting to insert area data: %s", area_data)

                # Validate required fields
                if not area_data["entry_id"]:
                    _LOGGER.error("entry_id is empty or None, cannot insert area")
                    return

                if not area_data["area_name"]:
                    _LOGGER.error("area_name is empty or None, cannot insert area")
                    return

                if not area_data["purpose"]:
                    _LOGGER.error("purpose is empty or None, cannot insert area")
                    return

                if area_data["threshold"] is None:
                    _LOGGER.error("threshold is None, cannot insert area")
                    return

                if area_data["area_prior"] is None:
                    _LOGGER.error("area_prior is None, cannot insert area")
                    return

                # Handle area_id - use entry_id as fallback if area_id is None or empty
                if not area_data["area_id"]:
                    _LOGGER.info(
                        "area_id is None or empty, using entry_id as fallback: %s",
                        area_data["entry_id"],
                    )
                    area_data["area_id"] = area_data["entry_id"]

                try:
                    # Use session.merge for upsert functionality
                    area_obj = self.Areas.from_dict(area_data)
                    session.merge(area_obj)
                    session.commit()

                    _LOGGER.info(
                        "Successfully saved area data for entry_id: %s",
                        area_data["entry_id"],
                    )

                except (
                    sa.exc.SQLAlchemyError,
                    HomeAssistantError,
                    TimeoutError,
                    OSError,
                ) as insert_err:
                    _LOGGER.error("Failed to save area data: %s", insert_err)
                    session.rollback()
                    try:
                        # Fallback to direct insert
                        area_obj = self.Areas.from_dict(area_data)
                        session.add(area_obj)
                        session.commit()
                        _LOGGER.info("Direct insert succeeded")
                    except Exception as direct_err:
                        _LOGGER.error("Direct insert also failed: %s", direct_err)
                        session.rollback()
                        raise
            _LOGGER.debug("Saved area data")
        except Exception as err:
            _LOGGER.error("Failed to save area data: %s", err)
            raise

    def save_entity_data(self) -> None:
        """Save the entity data to the database."""
        try:
            with self.get_locked_session() as session:
                entities = self.coordinator.entities.entities.values()
                for entity in entities:
                    # Skip entities with missing type information
                    if not hasattr(entity, "type") or not entity.type:
                        _LOGGER.warning(
                            "Entity %s has no type information, skipping",
                            entity.entity_id,
                        )
                        continue

                    entity_type = getattr(entity.type, "input_type", None)
                    if entity_type is None:
                        _LOGGER.warning(
                            "Entity %s has no input_type, skipping", entity.entity_id
                        )
                        continue

                    # Normalize values before persisting
                    try:
                        weight = float(
                            getattr(entity.type, "weight", DEFAULT_ENTITY_WEIGHT)
                        )
                    except (TypeError, ValueError):
                        weight = DEFAULT_ENTITY_WEIGHT
                    # Clamp weight to bounds
                    weight = max(MIN_WEIGHT, min(MAX_WEIGHT, weight))

                    # Clamp likelihoods to valid probability bounds
                    prob_true = max(
                        MIN_PROBABILITY,
                        min(MAX_PROBABILITY, float(entity.prob_given_true)),
                    )
                    prob_false = max(
                        MIN_PROBABILITY,
                        min(MAX_PROBABILITY, float(entity.prob_given_false)),
                    )

                    # Evidence must be a boolean for the DB schema
                    evidence_val = (
                        bool(entity.evidence) if entity.evidence is not None else False
                    )

                    last_updated = entity.last_updated or dt_util.utcnow()

                    entity_data = {
                        "entry_id": self.coordinator.entry_id,
                        "entity_id": entity.entity_id,
                        "entity_type": entity_type,
                        "weight": weight,
                        "prob_given_true": prob_true,
                        "prob_given_false": prob_false,
                        "last_updated": last_updated,
                        "is_decaying": entity.decay.is_decaying,
                        "decay_start": entity.decay.decay_start,
                        "evidence": evidence_val,
                    }

                    # Use session.merge for upsert functionality
                    entity_obj = self.Entities.from_dict(entity_data)
                    session.merge(entity_obj)

                session.commit()
            _LOGGER.debug("Saved entity data")

            # Clean up any orphaned entities after saving current ones
            cleaned_count = self.cleanup_orphaned_entities()
            if cleaned_count > 0:
                _LOGGER.info(
                    "Cleaned up %d orphaned entities after saving", cleaned_count
                )

        except Exception as err:
            _LOGGER.error("Failed to save entity data: %s", err)
            raise

    def save_data(self) -> None:
        """Save both area and entity data to the database."""
        self.save_area_data()
        self.save_entity_data()

    def cleanup_orphaned_entities(self) -> int:
        """Clean up entities from database that are no longer in the current configuration.

        This method removes entities and their associated intervals that exist in the database
        but are no longer present in the coordinator's current entity configuration.

        Returns:
            int: Number of entities that were cleaned up
        """
        try:

            def _cleanup_operation() -> int:
                with self.get_locked_session() as session:
                    # Get all entity IDs currently configured in the coordinator
                    # Handle cases where entities might be a SimpleNamespace or mock object
                    if hasattr(self.coordinator.entities, "entity_ids"):
                        current_entity_ids = set(self.coordinator.entities.entity_ids)
                    elif hasattr(self.coordinator.entities, "entities"):
                        # Fallback for mock objects that have entities dict
                        current_entity_ids = set(
                            self.coordinator.entities.entities.keys()
                        )
                    else:
                        # If we can't determine current entities, skip cleanup
                        _LOGGER.debug(
                            "Cannot determine current entity IDs, skipping cleanup"
                        )
                        return 0

                    # Query all entities for this entry_id from database
                    db_entities = (
                        session.query(self.Entities)
                        .filter_by(entry_id=self.coordinator.entry_id)
                        .all()
                    )

                    # Find entities that exist in database but not in current config
                    orphaned_entities = [
                        entity
                        for entity in db_entities
                        if entity.entity_id not in current_entity_ids
                    ]

                    if not orphaned_entities:
                        _LOGGER.debug(
                            "No orphaned entities found for entry %s",
                            self.coordinator.entry_id,
                        )
                        return 0

                    # Delete orphaned entities and their intervals
                    orphaned_count = 0
                    for entity in orphaned_entities:
                        _LOGGER.info(
                            "Removing orphaned entity %s from database (no longer in config)",
                            entity.entity_id,
                        )

                        # First delete all intervals for this entity
                        intervals_deleted = (
                            session.query(self.Intervals)
                            .filter_by(entity_id=entity.entity_id)
                            .delete()
                        )

                        if intervals_deleted > 0:
                            _LOGGER.debug(
                                "Deleted %d intervals for orphaned entity %s",
                                intervals_deleted,
                                entity.entity_id,
                            )

                        # Then delete the entity
                        session.delete(entity)
                        orphaned_count += 1

                    session.commit()
                    _LOGGER.info(
                        "Cleaned up %d orphaned entities for entry %s",
                        orphaned_count,
                        self.coordinator.entry_id,
                    )
                    return orphaned_count

            result = self.safe_database_operation(
                "cleanup orphaned entities", _cleanup_operation
            )

        except (
            sa.exc.SQLAlchemyError,
            HomeAssistantError,
            OSError,
            RuntimeError,
        ) as err:
            _LOGGER.error("Failed to cleanup orphaned entities: %s", err)
            return 0
        else:
            return result if result is not None else 0

    # --- Sync Data from Recorder ---

    def is_valid_state(self, state: Any) -> bool:
        """Check if a state is valid."""
        return state not in INVALID_STATES

    def is_intervals_empty(self) -> bool:
        """Check if the intervals table is empty using ORM (read-only, no lock)."""

        def _check_intervals_empty() -> bool:
            with self.get_session() as session:
                count = session.query(self.Intervals).count()
                return bool(count == 0)

        try:
            result = self.safe_database_operation(
                "check intervals empty", _check_intervals_empty
            )
            return bool(result) if result is not None else True
        except (sa.exc.SQLAlchemyError, HomeAssistantError, TimeoutError, OSError) as e:
            # If table doesn't exist, it's considered empty
            if "no such table" in str(e).lower():
                _LOGGER.debug("Intervals table doesn't exist yet, considering empty")
                return True
            _LOGGER.error("Failed to check if intervals empty: %s", e)
            # Return True as fallback to trigger data population
            return True

    def safe_is_intervals_empty(self) -> bool:
        """Safely check if intervals table is empty (fast, no integrity checks).

        Note: Database integrity checks are deferred to background health check
        task that runs 60 seconds after startup to avoid blocking integration loading.

        Returns:
            bool: True if intervals are empty, False if intervals exist
        """
        try:
            # Quick check - assume database is healthy during startup
            # Integrity checks will be performed by background health check task
            return self.is_intervals_empty()
        except (sa.exc.SQLAlchemyError, HomeAssistantError, TimeoutError, OSError) as e:
            # If we hit a corruption error, log it but don't block startup
            if self.is_database_corrupted(e):
                _LOGGER.warning(
                    "Database may be corrupted (error: %s). "
                    "Background health check will attempt recovery in 60 seconds.",
                    e,
                )
            else:
                _LOGGER.error("Unexpected error checking intervals: %s", e)

            # Assume empty to trigger data population, but don't block startup
            return True

    def get_area_data(self, entry_id: str) -> dict[str, Any] | None:
        """Get area data for a specific entry_id (read-only, no lock)."""
        try:
            with self.get_session() as session:
                area = session.query(self.Areas).filter_by(entry_id=entry_id).first()
                if area:
                    return dict(area.to_dict())
                return None
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to get area data: %s", e)
            return None

    async def ensure_area_exists(self) -> None:
        """Ensure that the area record exists in the database."""
        try:
            # Check if area exists
            existing_area = self.get_area_data(self.coordinator.entry_id)
            if existing_area:
                _LOGGER.debug(
                    "Area already exists for entry_id: %s", self.coordinator.entry_id
                )
                return

            # Area doesn't exist, force create it
            _LOGGER.info(
                "Area not found, forcing creation for entry_id: %s",
                self.coordinator.entry_id,
            )
            self.save_data()

            # Verify it was created
            new_area = self.get_area_data(self.coordinator.entry_id)
            if new_area:
                _LOGGER.info("Successfully created area: %s", new_area)
            else:
                _LOGGER.error(
                    "Failed to create area for entry_id: %s", self.coordinator.entry_id
                )

        except (sa.exc.SQLAlchemyError, HomeAssistantError, TimeoutError, OSError) as e:
            _LOGGER.error("Error ensuring area exists: %s", e)

    def get_latest_interval(self) -> datetime:
        """Return the latest interval end time minus 1 hour, or default window if none (read-only, no lock)."""

        def _get_latest_interval_operation() -> datetime:
            with self.get_session() as session:
                result = session.execute(
                    sa.select(sa.func.max(self.Intervals.end_time))
                ).scalar()
                if result:
                    return result - timedelta(hours=1)
                return dt_util.now() - timedelta(days=10)

        try:
            result = self.safe_database_operation(
                "get latest interval", _get_latest_interval_operation
            )
            return result if result is not None else dt_util.now() - timedelta(days=10)
        except (sa.exc.SQLAlchemyError, HomeAssistantError, TimeoutError, OSError) as e:
            # If table doesn't exist or any other error, return a default time
            if "no such table" in str(e).lower():
                _LOGGER.debug("Intervals table doesn't exist yet, using default time")
            else:
                _LOGGER.warning(
                    "Failed to get latest interval, using default time: %s", e
                )
            return dt_util.now() - timedelta(days=10)

    def _states_to_intervals(
        self, states: dict[str, list[State]], end_time: datetime
    ) -> list[dict[str, Any]]:
        """Convert states to intervals by processing consecutive state changes for each entity.

        Args:
            states: Dictionary mapping entity_id to list of State objects
            end_time: The end time for the analysis period

        Returns:
            List of interval dictionaries with proper start_time, end_time, and duration_seconds

        """
        intervals = []
        retention_time = dt_util.now() - timedelta(days=RETENTION_DAYS)

        for entity_id, state_list in states.items():
            if not state_list:
                continue

            # Sort states by last_changed time
            sorted_states = sorted(state_list, key=lambda s: s.last_changed)

            # Process each state to create intervals
            for i, state in enumerate(sorted_states):
                # Skip states outside retention period
                if state.last_changed < retention_time:
                    continue

                # Determine the end time for this interval
                if i + 1 < len(sorted_states):
                    # Use the start time of the next state as the end time
                    interval_end = sorted_states[i + 1].last_changed
                else:
                    # For the last state, use the analysis end time
                    interval_end = end_time

                # Calculate duration
                duration_seconds = (interval_end - state.last_changed).total_seconds()

                # Apply filtering based on state and duration
                if state.state == "on":
                    if duration_seconds <= MAX_INTERVAL_SECONDS:
                        intervals.append(
                            {
                                "entity_id": entity_id,
                                "state": state.state,
                                "start_time": state.last_changed,
                                "end_time": interval_end,
                                "duration_seconds": duration_seconds,
                                "created_at": dt_util.utcnow(),
                            }
                        )
                elif (
                    self.is_valid_state(state.state)
                    and duration_seconds >= MIN_INTERVAL_SECONDS
                ):
                    intervals.append(
                        {
                            "entity_id": entity_id,
                            "state": state.state,
                            "start_time": state.last_changed,
                            "end_time": interval_end,
                            "duration_seconds": duration_seconds,
                            "created_at": dt_util.utcnow(),
                        }
                    )

        _LOGGER.debug("Created %d intervals from states", len(intervals))
        return intervals

    async def sync_states(
        self,
    ) -> None:
        """Fetch states history from recorder and commit to Intervals table."""
        hass = self.coordinator.hass
        entity_ids = self.coordinator.config.entity_ids
        recorder = get_instance(hass)
        start_time = self.get_latest_interval()
        end_time = dt_util.now()

        try:
            states = await recorder.async_add_executor_job(
                lambda: get_significant_states(
                    hass,
                    start_time,
                    end_time,
                    entity_ids,
                    minimal_response=False,
                )
            )
            _LOGGER.debug("Found %d states", len(states))

            if states:
                # Convert states to proper intervals with correct duration calculation
                intervals = self._states_to_intervals(states, end_time)  # type: ignore[arg-type]
                _LOGGER.debug("Syncing %d intervals", len(intervals))
                if intervals:
                    with self.get_locked_session() as session:
                        for interval_data in intervals:
                            # Check if interval already exists to avoid UNIQUE constraint violation
                            existing = (
                                session.query(self.Intervals)
                                .filter(
                                    self.Intervals.entity_id
                                    == interval_data["entity_id"],
                                    self.Intervals.start_time
                                    == interval_data["start_time"],
                                    self.Intervals.end_time
                                    == interval_data["end_time"],
                                )
                                .first()
                            )

                            if not existing:
                                # Only create if it doesn't exist
                                interval_obj = self.Intervals.from_dict(interval_data)
                                session.add(interval_obj)
                        session.commit()
                _LOGGER.debug("Synced %d intervals", len(intervals))
            else:
                _LOGGER.debug("No states found in recorder query result")

        except (HomeAssistantError, TimeoutError) as err:
            _LOGGER.error("Error getting states: %s", err)
            raise

    def prune_old_intervals(self, force: bool = False) -> int:
        """Delete intervals older than RETENTION_DAYS (coordinated across instances).

        Args:
            force: If True, skip the recent-prune check

        Returns:
            Number of intervals deleted
        """
        # Skip if pruned recently (within last hour) unless forced
        if not force:
            last_prune = self._get_last_prune_time()
            if last_prune:
                time_since_prune = (dt_util.utcnow() - last_prune).total_seconds()
                if time_since_prune < 3600:  # 1 hour
                    _LOGGER.debug(
                        "Skipping prune - last run was %d minutes ago",
                        int(time_since_prune / 60),
                    )
                    return 0

        cutoff_date = dt_util.utcnow() - timedelta(days=RETENTION_DAYS)
        _LOGGER.debug("Pruning intervals older than %s", cutoff_date)

        try:
            with self.get_locked_session() as session:
                # Count intervals to be deleted for logging
                count_query = session.query(func.count(self.Intervals.id)).filter(
                    self.Intervals.start_time < cutoff_date
                )
                intervals_to_delete = count_query.scalar() or 0

                if intervals_to_delete == 0:
                    _LOGGER.debug("No old intervals to prune")
                    # Still record the prune attempt to prevent other instances from trying
                    self._set_last_prune_time(dt_util.utcnow(), session)
                    return 0

                # Delete old intervals
                delete_query = session.query(self.Intervals).filter(
                    self.Intervals.start_time < cutoff_date
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
                self._set_last_prune_time(dt_util.utcnow(), session)

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
        self, entry_id: str, slot_minutes: int = 60
    ) -> list[tuple[int, int, float]]:
        """Get aggregated interval data using SQL GROUP BY for better performance.

        Args:
            entry_id: The area entry ID to filter by
            slot_minutes: Time slot size in minutes

        Returns:
            List of (day_of_week, time_slot, total_occupied_seconds) tuples

        """
        _LOGGER.debug("Getting aggregated intervals by slot using SQL GROUP BY")

        try:
            with self.get_session() as session:
                # Use SQLite datetime functions to group intervals by day and time slot
                # This is much more efficient than Python loops
                query = (
                    session.query(
                        func.strftime("%w", self.Intervals.start_time).label(
                            "day_of_week"
                        ),
                        func.cast(
                            (
                                func.cast(
                                    func.strftime("%H", self.Intervals.start_time),
                                    sa.Integer,
                                )
                                * 60
                                + func.cast(
                                    func.strftime("%M", self.Intervals.start_time),
                                    sa.Integer,
                                )
                            )
                            // slot_minutes,
                            sa.Integer,
                        ).label("time_slot"),
                        func.sum(self.Intervals.duration_seconds).label(
                            "total_seconds"
                        ),
                    )
                    .join(
                        self.Entities,
                        self.Intervals.entity_id == self.Entities.entity_id,
                    )
                    .filter(
                        self.Entities.entry_id == entry_id,
                        self.Entities.entity_type
                        == "motion",  # Use string instead of InputType enum
                        self.Intervals.state == "on",
                    )
                    .group_by("day_of_week", "time_slot")
                    .order_by("day_of_week", "time_slot")
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
            _LOGGER.error(
                "Database connection error during interval aggregation: %s", e
            )
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
