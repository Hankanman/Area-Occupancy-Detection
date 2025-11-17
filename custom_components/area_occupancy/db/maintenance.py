"""Database maintenance functions."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
import logging
import shutil
import time
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .constants import DB_VERSION
from .schema import Base

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)


def ensure_db_exists(db: AreaOccupancyDB) -> None:
    """Check if the database exists and initialize it if needed.

    NOTE: This function only performs FAST validation (file existence and SQLite header).
    Heavy integrity checks are deferred to background tasks to avoid blocking startup.
    """
    try:
        # Fast validation: only check if file exists and has valid SQLite header
        if db.db_path and db.db_path.exists():
            # Quick check: read first 16 bytes to validate SQLite format
            try:
                with open(db.db_path, "rb") as f:
                    header = f.read(16)
                    if not header.startswith(b"SQLite format 3"):
                        _LOGGER.warning(
                            "Database file is not a valid SQLite database, will be recreated"
                        )
                        delete_db(db)
                    else:
                        # File exists and is valid SQLite - verify tables exist
                        _LOGGER.debug("Database file found, verifying tables exist")
            except (OSError, PermissionError) as e:
                _LOGGER.warning("Cannot read database file: %s, will recreate", e)
                # Will create new database below

        # Always verify that all required tables exist
        # This prevents race conditions when multiple instances start simultaneously
        if not verify_all_tables_exist(db):
            _LOGGER.debug("Not all tables exist, initializing database")
            init_db(db)
            set_db_version(db)
        else:
            # Tables exist - verify schema is up to date
            _ensure_schema_up_to_date(db)
    except sa.exc.SQLAlchemyError as e:
        # Check if this is a corruption error
        if is_database_corrupted(db, e):
            _LOGGER.warning(
                "Database may be corrupted (error: %s), will attempt recovery in background",
                e,
            )
            # Don't block startup - will attempt recovery in background
            return

        # Database doesn't exist or is not initialized, create it
        _LOGGER.debug("Database error during table check, initializing database: %s", e)
        try:
            init_db(db)
            set_db_version(db)
        except (
            sa.exc.SQLAlchemyError,
            OSError,
            RuntimeError,
            PermissionError,
        ):
            # If initialization fails, log and continue
            # Will attempt recovery in background
            _LOGGER.debug("Database initialization failed, will attempt recovery")


def check_database_integrity(db: AreaOccupancyDB) -> bool:
    """Check if the database is healthy and not corrupted.

    Returns:
        bool: True if database is healthy, False if corrupted
    """
    try:
        with db.engine.connect() as conn:
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


def check_database_accessibility(db: AreaOccupancyDB) -> bool:
    """Check if the database file is accessible and readable.

    Returns:
        bool: True if database is accessible, False otherwise
    """
    if not db.db_path or not db.db_path.exists():
        return False

    try:
        # Try to open the file to check if it's readable
        with open(db.db_path, "rb") as f:
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


def verify_all_tables_exist(db: AreaOccupancyDB) -> bool:
    """Verify all required tables exist in the database.

    Uses SQLAlchemy's inspector to ensure consistent connection handling,
    avoiding race conditions with in-memory databases where different
    connections might not see tables created by other connections.

    Returns:
        bool: True if all required tables exist, False otherwise
    """
    required_tables = {
        "areas",
        "entities",
        "intervals",
        "priors",
        "metadata",
        "interval_aggregates",
        "occupied_intervals_cache",
        "global_priors",
        "numeric_samples",
        "numeric_aggregates",
        "numeric_correlations",
        "entity_statistics",
        "area_relationships",
        "cross_area_stats",
    }
    try:
        # Use inspector instead of raw connection to ensure consistent
        # connection pool usage (same as init_db uses Base.metadata.create_all)
        inspector = sa.inspect(db.engine)
        existing_tables = set(inspector.get_table_names())
        return required_tables.issubset(existing_tables)
    except sa.exc.SQLAlchemyError:
        return False


def _ensure_schema_up_to_date(db: AreaOccupancyDB) -> None:
    """Ensure database schema matches current version.

    If version doesn't match DB_VERSION, delete and recreate from scratch.
    """
    try:
        db_version = get_db_version(db)
        if db_version != DB_VERSION:
            _LOGGER.info(
                "Database version mismatch (found %d, expected %d). "
                "Deleting existing database and recreating from scratch.",
                db_version,
                DB_VERSION,
            )
            delete_db(db)
            # Recreate database with new schema
            init_db(db)
            set_db_version(db)
            _LOGGER.info(
                "Database recreated with DB_VERSION %d schema. All previous data has been cleared.",
                DB_VERSION,
            )

    except (SQLAlchemyError, OSError, RuntimeError) as e:
        _LOGGER.warning("Error checking database schema: %s", e)
        # On error, delete and recreate to be safe
        _LOGGER.info("Recreating database due to schema check error")
        try:
            delete_db(db)
            init_db(db)
            set_db_version(db)
        except Exception as recreate_err:
            _LOGGER.error("Failed to recreate database: %s", recreate_err)
            raise


def _migrate_priors_table_for_area_name(db: AreaOccupancyDB) -> None:
    """Migrate priors table to add area_name column and update primary key.

    SQLite doesn't support ALTER TABLE for primary key changes, so we need to:
    1. Create a new table with the correct schema
    2. Copy data from old table, populating area_name from areas table
    3. Drop old table
    4. Rename new table to priors
    """
    try:
        with db.get_locked_session() as session:
            # Step 1: Create new priors table with area_name in primary key
            _LOGGER.debug("Creating new priors table with area_name")
            session.execute(
                text(
                    """
                    CREATE TABLE priors_new (
                        entry_id TEXT NOT NULL,
                        area_name TEXT NOT NULL,
                        day_of_week INTEGER NOT NULL,
                        time_slot INTEGER NOT NULL,
                        prior_value REAL NOT NULL,
                        data_points INTEGER NOT NULL,
                        last_updated DATETIME NOT NULL,
                        PRIMARY KEY (entry_id, area_name, day_of_week, time_slot),
                        FOREIGN KEY (entry_id) REFERENCES areas(entry_id)
                    )
                    """
                )
            )

            # Step 2: Copy data from old table, populating area_name from areas table
            _LOGGER.debug("Copying data from old priors table with area_name")
            # Check if areas table exists
            inspector = sa.inspect(db.engine)
            if "areas" in inspector.get_table_names():
                # Use JOIN to get area_name from areas table
                session.execute(
                    text(
                        """
                        INSERT INTO priors_new (
                            entry_id, area_name, day_of_week, time_slot,
                            prior_value, data_points, last_updated
                        )
                        SELECT
                            p.entry_id,
                            COALESCE(a.area_name, '') as area_name,
                            p.day_of_week,
                            p.time_slot,
                            p.prior_value,
                            p.data_points,
                            p.last_updated
                        FROM priors p
                        LEFT JOIN areas a ON p.entry_id = a.entry_id
                        """
                    )
                )
            else:
                # No areas table, use empty string for area_name
                _LOGGER.warning(
                    "Areas table not found during priors migration, using empty area_name"
                )
                session.execute(
                    text(
                        """
                        INSERT INTO priors_new (
                            entry_id, area_name, day_of_week, time_slot,
                            prior_value, data_points, last_updated
                        )
                        SELECT
                            entry_id,
                            '' as area_name,
                            day_of_week,
                            time_slot,
                            prior_value,
                            data_points,
                            last_updated
                        FROM priors
                        """
                    )
                )

            # Step 3: Drop old table
            _LOGGER.debug("Dropping old priors table")
            session.execute(text("DROP TABLE priors"))

            # Step 4: Rename new table to priors
            _LOGGER.debug("Renaming new priors table")
            session.execute(text("ALTER TABLE priors_new RENAME TO priors"))

            # Step 5: Create indexes
            _LOGGER.debug("Creating indexes on priors table")
            session.execute(text("CREATE INDEX idx_priors_entry ON priors(entry_id)"))
            session.execute(text("CREATE INDEX idx_priors_area ON priors(area_name)"))
            session.execute(
                text(
                    "CREATE INDEX idx_priors_entry_area ON priors(entry_id, area_name)"
                )
            )
            session.execute(
                text(
                    "CREATE INDEX idx_priors_day_slot ON priors(day_of_week, time_slot)"
                )
            )
            session.execute(
                text("CREATE INDEX idx_priors_last_updated ON priors(last_updated)")
            )

            session.commit()
            _LOGGER.info(
                "Successfully migrated priors table to include area_name in primary key"
            )
    except (SQLAlchemyError, OSError, RuntimeError) as e:
        _LOGGER.error("Error migrating priors table: %s", e)
        # Try to clean up if migration failed
        try:
            with db.get_locked_session() as session:
                # Check if priors_new exists and drop it
                inspector = sa.inspect(db.engine)
                if "priors_new" in inspector.get_table_names():
                    session.execute(text("DROP TABLE priors_new"))
                    session.commit()
        except Exception:  # noqa: BLE001
            pass
        raise


def is_database_corrupted(db: AreaOccupancyDB, error: Exception) -> bool:
    """Check if an error indicates database corruption.

    Args:
        db: Database instance
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


def attempt_database_recovery(db: AreaOccupancyDB) -> bool:
    """Attempt to recover from database corruption.

    Returns:
        bool: True if recovery was successful, False otherwise
    """
    _LOGGER.warning("Attempting database recovery from corruption")

    try:
        # First, try to close all connections and recreate engine
        db.engine.dispose()

        # Try to enable WAL mode and run recovery
        temp_engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            poolclass=sa.pool.NullPool,
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
                db.engine = temp_engine
                db.update_session_maker()
                return True
            _LOGGER.error("Database recovery failed, no tables found")
            return False

    except (sa.exc.SQLAlchemyError, OSError, PermissionError) as e:
        _LOGGER.error("Database recovery failed: %s", e)
        return False


def backup_database(db: AreaOccupancyDB) -> bool:
    """Create a backup of the current database.

    Returns:
        bool: True if backup was successful, False otherwise
    """
    if not db.db_path or not db.db_path.exists():
        return False

    try:
        backup_path = db.db_path.with_suffix(".db.backup")

        shutil.copy2(db.db_path, backup_path)
        _LOGGER.info("Database backup created at %s", backup_path)
    except (OSError, PermissionError, shutil.Error) as e:
        _LOGGER.error("Failed to create database backup: %s", e)
        return False
    else:
        return True


def restore_database_from_backup(db: AreaOccupancyDB) -> bool:
    """Restore database from backup if available.

    Returns:
        bool: True if restore was successful, False otherwise
    """
    if not db.db_path:
        return False

    backup_path = db.db_path.with_suffix(".db.backup")
    if not backup_path.exists():
        _LOGGER.warning("No backup found at %s", backup_path)
        return False

    try:
        # Close current engine
        db.engine.dispose()

        shutil.copy2(backup_path, db.db_path)

        # Recreate engine
        db.engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            poolclass=sa.pool.NullPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
        )
        db.update_session_maker()

        _LOGGER.info("Database restored from backup")

    except (OSError, PermissionError, shutil.Error, sa.exc.SQLAlchemyError) as e:
        _LOGGER.error("Failed to restore database from backup: %s", e)
        return False
    else:
        return True


def handle_database_corruption(db: AreaOccupancyDB) -> bool:
    """Handle database corruption with automatic recovery attempts.

    Returns:
        bool: True if database is now healthy, False if all recovery attempts failed
    """
    if not db.enable_auto_recovery:
        _LOGGER.error("Database corruption detected but auto-recovery is disabled")
        return False

    _LOGGER.error("Database corruption detected, attempting recovery")

    # First, try to create a backup if possible
    if db.enable_periodic_backups:
        backup_database(db)

    # Try database recovery first
    if attempt_database_recovery(db):
        if check_database_integrity(db):
            _LOGGER.info("Database recovery successful")
            return True

    # If recovery failed, try to restore from backup
    if db.enable_periodic_backups and restore_database_from_backup(db):
        if check_database_integrity(db):
            _LOGGER.info("Database restore from backup successful")
            return True

    # If all else fails, delete and recreate the database
    _LOGGER.warning("All recovery attempts failed, recreating database")
    try:
        delete_db(db)
        init_db(db)
        set_db_version(db)
        _LOGGER.info("Database recreated successfully")
    except (sa.exc.SQLAlchemyError, OSError, PermissionError) as e:
        _LOGGER.error("Failed to recreate database: %s", e)
        return False
    else:
        return True


def periodic_health_check(db: AreaOccupancyDB) -> bool:
    """Perform periodic database health check and maintenance.

    Health checks are performed during analysis cycles.

    Returns:
        bool: True if database is healthy, False if issues were found
    """
    try:
        # Check database integrity
        if not check_database_integrity(db):
            _LOGGER.warning("Periodic health check found database corruption")
            if handle_database_corruption(db):
                _LOGGER.info("Database recovered during periodic health check")
                return True
            _LOGGER.error("Failed to recover database during periodic health check")
            return False

        # Create periodic backup if enabled
        if db.enable_periodic_backups and db.db_path:
            backup_path = db.db_path.with_suffix(".db.backup")
            backup_interval_seconds = db.backup_interval_hours * 3600
            if (
                not backup_path.exists()
                or (time.time() - backup_path.stat().st_mtime) > backup_interval_seconds
            ):
                if backup_database(db):
                    _LOGGER.debug("Periodic database backup created")
                else:
                    _LOGGER.warning("Failed to create periodic database backup")

        # Run database maintenance
        with suppress(Exception), db.engine.connect() as conn:
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


def set_db_version(db: AreaOccupancyDB) -> None:
    """Set the database version in the metadata table."""
    # Use session for ORM operations during initialization
    with db.get_session() as session:
        try:
            with session.begin():
                # Try to get existing metadata entry using ORM
                metadata_entry = (
                    session.query(db.Metadata).filter_by(key="db_version").first()
                )
                if metadata_entry:
                    # Update existing entry
                    metadata_entry.value = str(DB_VERSION)
                else:
                    # Insert new entry
                    session.add(db.Metadata(key="db_version", value=str(DB_VERSION)))
        except Exception as e:
            _LOGGER.error("Failed to set db_version in metadata table: %s", e)
            raise


def get_db_version(db: AreaOccupancyDB) -> int:
    """Get the database version from the metadata table.

    Returns 0 if version is not set or table doesn't exist.
    """
    try:
        with db.get_session() as session:
            try:
                metadata_entry = (
                    session.query(db.Metadata).filter_by(key="db_version").first()
                )
                return int(metadata_entry.value) if metadata_entry else 0
            except (AttributeError, ValueError, SQLAlchemyError) as e:
                _LOGGER.debug("Failed to get db_version from metadata table: %s", e)
                return 0
    except (SQLAlchemyError, OSError, RuntimeError) as e:
        _LOGGER.debug("Failed to get db_version (table may not exist): %s", e)
        return 0


def delete_db(db: AreaOccupancyDB) -> None:
    """Delete the database file."""
    # Dispose engine first to release any open file handles
    engine = getattr(db, "engine", None)
    if engine is not None:
        try:
            engine.dispose()
        except SQLAlchemyError as e:
            _LOGGER.debug("Failed to dispose engine before deleting DB: %s", e)

    if db.db_path and db.db_path.exists():
        try:
            db.db_path.unlink()
            _LOGGER.info("Deleted database at %s", db.db_path)
        except (OSError, PermissionError) as e:
            _LOGGER.error("Failed to delete database file: %s", e)


def force_reinitialize(db: AreaOccupancyDB) -> None:
    """Force reinitialization of the database tables."""
    _LOGGER.debug("Forcing database reinitialization")
    init_db(db)
    set_db_version(db)


def get_last_prune_time(db: AreaOccupancyDB) -> datetime | None:
    """Get timestamp of last successful prune operation.

    Returns:
        datetime of last prune, or None if not recorded
    """
    try:
        with db.get_session() as session:
            result = session.query(db.Metadata).filter_by(key="last_prune_time").first()
            if result:
                return datetime.fromisoformat(result.value)
    except (ValueError, AttributeError, SQLAlchemyError, OSError) as e:
        _LOGGER.debug("Failed to get last prune time: %s", e)
    return None


def set_last_prune_time(
    db: AreaOccupancyDB, timestamp: datetime, session: Any = None
) -> None:
    """Record timestamp of successful prune operation.

    Args:
        db: Database instance
        timestamp: When the prune occurred
        session: Optional existing session to use (avoids nested locks)
    """
    try:
        if session is not None:
            # Use existing session to avoid nested lock acquisition
            existing = (
                session.query(db.Metadata).filter_by(key="last_prune_time").first()
            )
            if existing:
                existing.value = timestamp.isoformat()
            else:
                session.add(
                    db.Metadata(key="last_prune_time", value=timestamp.isoformat())
                )
            session.commit()
        else:
            # Fallback to new locked session if not provided
            with db.get_locked_session() as new_session:
                existing = (
                    new_session.query(db.Metadata)
                    .filter_by(key="last_prune_time")
                    .first()
                )
                if existing:
                    existing.value = timestamp.isoformat()
                else:
                    new_session.add(
                        db.Metadata(key="last_prune_time", value=timestamp.isoformat())
                    )
                new_session.commit()
    except (SQLAlchemyError, OSError, ValueError) as e:
        _LOGGER.warning("Failed to record prune timestamp: %s", e)


def init_db(db: AreaOccupancyDB) -> None:
    """Initialize the database with WAL mode."""
    _LOGGER.debug("Starting database initialization")
    try:
        # Enable WAL mode for better concurrent writes
        _enable_wal_mode(db)
        # Create all tables with checkfirst to avoid race conditions
        _LOGGER.debug("Creating database tables")
        Base.metadata.create_all(db.engine, checkfirst=True)
        _LOGGER.debug("Database tables created successfully")
    except sa.exc.OperationalError as err:
        # Handle errors when creating tables
        if err.orig and hasattr(err.orig, "sqlite_errno"):
            if err.orig.sqlite_errno == 1:
                _LOGGER.debug(
                    "Table already exists (race condition), continuing: %s", err
                )
                # Continue - other tables might still need to be created
                # Try to create remaining tables individually
                _create_tables_individually(db)
            else:
                _LOGGER.error("Database initialization failed: %s", err)
                raise
        else:
            _LOGGER.error("Database initialization failed: %s", err)
            raise
    except Exception as err:
        _LOGGER.error("Database initialization failed: %s", err)
        raise


def _enable_wal_mode(db: AreaOccupancyDB) -> None:
    """Enable SQLite WAL mode for better concurrent writes."""
    try:
        with db.engine.connect() as conn:
            conn.execute(sa.text("PRAGMA journal_mode=WAL"))
    except sa.exc.SQLAlchemyError as err:
        _LOGGER.debug("Failed to enable WAL mode: %s", err)


def _create_tables_individually(db: AreaOccupancyDB) -> None:
    """Create tables individually to handle race conditions."""
    for table in Base.metadata.tables.values():
        try:
            table.create(db.engine, checkfirst=True)
        except sa.exc.OperationalError as err:
            if err.orig and hasattr(err.orig, "sqlite_errno"):
                if err.orig.sqlite_errno == 1:
                    _LOGGER.debug("Table %s already exists, skipping", table.name)
                    continue
            raise
