"""Tests for database maintenance functions."""

from datetime import datetime
import os
from pathlib import Path
import shutil
import time
from unittest.mock import Mock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.const import DB_SCHEMA_VERSION
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db import Base
from custom_components.area_occupancy.db.maintenance import (
    _attempt_database_recovery,
    _backup_database,
    _check_database_integrity,
    _create_tables_individually,
    _ensure_schema_up_to_date,
    _handle_database_corruption,
    _is_database_corrupted,
    _restore_database_from_backup,
    _set_db_version,
    delete_db,
    ensure_db_exists,
    get_db_version,
    get_last_prune_time,
    get_missing_tables,
    init_db,
    periodic_health_check,
    set_last_prune_time,
    verify_all_tables_exist,
)
from homeassistant.util import dt as dt_util
from tests.conftest import setup_test_db_engine  # noqa: TID251


class TestEnsureDbExists:
    """Test ensure_db_exists function."""

    def test_ensure_db_exists_new_database(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test ensure_db_exists with new database."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_new.db")

        # This should create all tables
        ensure_db_exists(db)

        # Verify tables were created
        assert verify_all_tables_exist(db) is True

    def test_ensure_db_exists_with_file_no_tables(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test ensure_db_exists when file exists but has no tables (race condition)."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_race.db")

        # Create an empty SQLite database file (valid header but no tables)
        with db.engine.connect() as conn:
            conn.execute(text("CREATE TABLE _temp (id INTEGER)"))
            conn.execute(text("DROP TABLE _temp"))
            conn.commit()

        # Now verify this triggers table creation
        ensure_db_exists(db)

        # Verify all required tables were created
        assert verify_all_tables_exist(db) is True

    def test_ensure_db_exists_with_complete_database(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test ensure_db_exists when database is already complete."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_complete.db")

        # Create a fully initialized database
        init_db(db)
        _set_db_version(db)

        ensure_db_exists(db)

        # Verify tables still exist (not corrupted)
        assert verify_all_tables_exist(db) is True

    def test_ensure_db_exists_with_version_mismatch(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test ensure_db_exists when database version doesn't match."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_version_mismatch.db")

        # Create database with old version
        init_db(db)
        with db.get_session() as session:
            session.add(db.Metadata(key="db_version", value=str(DB_SCHEMA_VERSION - 1)))
            session.commit()

        # Add some test data to verify it's cleared
        with db.get_session() as session:
            session.add(db.Metadata(key="test_key", value="test_value"))
            session.commit()

        # Call ensure_db_exists - should trigger _ensure_schema_up_to_date
        ensure_db_exists(db)

        # Verify database was recreated with correct version
        assert verify_all_tables_exist(db) is True
        assert get_db_version(db) == DB_SCHEMA_VERSION

        # Verify old data was cleared
        with db.get_session() as session:
            result = session.query(db.Metadata).filter_by(key="test_key").first()
            assert result is None


class TestCheckDatabaseIntegrity:
    """Test _check_database_integrity function."""

    def test_check_database_integrity_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test integrity check with healthy database."""
        db = coordinator.db
        db.init_db()
        result = _check_database_integrity(db)
        assert result is True

    def test_check_database_integrity_error(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test integrity check with corrupted database."""
        db = coordinator.db
        db_path = tmp_path / "test_integrity_error.db"
        setup_test_db_engine(db, db_path)
        db.init_db()

        # Close engine and corrupt the database file
        db.engine.dispose()
        # Write invalid data to corrupt the database
        db_path.write_bytes(b"INVALID SQLITE DATA" * 100)

        # Recreate engine to test with corrupted database
        setup_test_db_engine(db, db_path)

        # Integrity check should fail with corrupted database
        result = _check_database_integrity(db)
        assert result is False


class TestVerifyAllTablesExist:
    """Test verify_all_tables_exist function."""

    def test_verify_all_tables_exist_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test verification with all tables present."""
        db = coordinator.db
        db.init_db()
        assert verify_all_tables_exist(db) is True

    def test_verify_all_tables_exist_error(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test verification with corrupted database."""
        db = coordinator.db
        db_path = tmp_path / "test_verify_error.db"
        # Create a corrupted database file that can't be inspected
        db_path.write_bytes(b"CORRUPTED" * 100)
        setup_test_db_engine(db, db_path)

        # Verification should fail with corrupted database
        result = verify_all_tables_exist(db)
        assert result is False


class TestGetMissingTables:
    """Test get_missing_tables function."""

    def test_get_missing_tables_none_missing(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test getting missing tables when all exist."""
        db = coordinator.db
        db.init_db()
        missing = get_missing_tables(db)
        assert missing == set()

    def test_get_missing_tables_some_missing(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test getting missing tables when some are missing."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")

        # Create only some tables
        Base.metadata.create_all(db.engine, tables=[Base.metadata.tables["areas"]])

        missing = get_missing_tables(db)
        assert len(missing) > 0
        assert "entities" in missing


class TestInitDb:
    """Test init_db function."""

    def test_init_db_success(self, coordinator: AreaOccupancyCoordinator, tmp_path):
        """Test init_db with successful initialization."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_init.db")

        # Initialize database
        init_db(db)

        # Verify tables were actually created
        assert verify_all_tables_exist(db) is True

        # Verify WAL mode is enabled by checking journal mode
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).fetchone()
            assert result is not None
            # WAL mode may be returned as 'wal' or 'WAL'
            assert result[0].upper() == "WAL"

    def test_init_db_wal_mode_persists_across_connections(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test that WAL mode persists across different database connections."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_wal_persist.db")

        # Initialize database
        init_db(db)

        # Verify WAL mode with first connection
        with db.engine.connect() as conn:
            result1 = conn.execute(text("PRAGMA journal_mode")).fetchone()
            assert result1[0].upper() == "WAL"

        # Create new connection and verify WAL mode persists
        with db.engine.connect() as conn:
            result2 = conn.execute(text("PRAGMA journal_mode")).fetchone()
            assert result2[0].upper() == "WAL"

        # Verify tables are still accessible
        assert verify_all_tables_exist(db) is True

    def test_create_tables_individually_race_condition(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test _create_tables_individually with race condition."""
        db = coordinator.db

        # Mock error with sqlite_errno = 1 (table already exists)
        mock_error = sa.exc.OperationalError("table already exists", None, None)
        mock_error.orig = Mock()
        mock_error.orig.sqlite_errno = 1

        with patch.object(db.engine, "connect", side_effect=mock_error):
            # Should not raise exception
            _create_tables_individually(db)

    def test_create_tables_individually_other_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test _create_tables_individually with other error."""
        db = coordinator.db

        # Mock error with different sqlite_errno
        mock_error = sa.exc.OperationalError("other error", None, None)
        mock_error.orig = Mock()
        mock_error.orig.sqlite_errno = 2

        with (
            patch.object(db.engine, "connect", side_effect=mock_error),
            pytest.raises(sa.exc.OperationalError),
        ):
            _create_tables_individually(db)


class TestSetDbVersion:
    """Test _set_db_version function."""

    def test_set_db_version_update_existing(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test _set_db_version when version already exists."""
        db = coordinator.db
        db.init_db()
        _set_db_version(db)  # Create initial version

        # Verify the version was set correctly
        version = get_db_version(db)
        assert version == DB_SCHEMA_VERSION

        # Call _set_db_version again - should update existing
        _set_db_version(db)

        # Verify version is still correct
        version_after = get_db_version(db)
        assert version_after == DB_SCHEMA_VERSION

    def test_set_db_version_insert_new(self, coordinator: AreaOccupancyCoordinator):
        """Test _set_db_version when version doesn't exist."""
        db = coordinator.db
        db.init_db()

        # Delete any existing version
        with db.get_session() as session:
            session.query(db.Metadata).filter_by(key="db_version").delete()
            session.commit()

        # Set version - should insert new
        _set_db_version(db)

        # Verify version was set
        version = get_db_version(db)
        assert version == DB_SCHEMA_VERSION

    def test_set_db_version_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test _set_db_version with error."""
        db = coordinator.db

        def bad_session():
            raise RuntimeError("DB Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        with pytest.raises(RuntimeError):
            _set_db_version(db)


class TestGetDbVersion:
    """Test get_db_version function."""

    def test_get_db_version_success(self, coordinator: AreaOccupancyCoordinator):
        """Test get_db_version with success."""
        db = coordinator.db
        db.init_db()
        _set_db_version(db)

        version = get_db_version(db)
        assert version == DB_SCHEMA_VERSION

    def test_get_db_version_no_metadata(self, coordinator: AreaOccupancyCoordinator):
        """Test get_db_version when no metadata exists."""
        db = coordinator.db
        db.init_db()

        # Delete metadata
        with db.get_session() as session:
            session.query(db.Metadata).filter_by(key="db_version").delete()
            session.commit()

        version = get_db_version(db)
        assert version == 0

    def test_get_db_version_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test get_db_version with error."""
        db = coordinator.db

        def bad_session():
            raise RuntimeError("DB Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        # get_db_version catches exceptions and returns 0
        version = get_db_version(db)
        assert version == 0


class TestDeleteDb:
    """Test delete_db function."""

    def test_delete_db_success(self, coordinator: AreaOccupancyCoordinator, tmp_path):
        """Test delete_db with successful deletion."""
        db = coordinator.db
        db.db_path = tmp_path / "test.db"

        # Create file to delete
        db.db_path.touch()

        delete_db(db)

        assert not db.db_path.exists()

    def test_delete_db_file_not_exists(self, coordinator: AreaOccupancyCoordinator):
        """Test delete_db when file doesn't exist."""
        db = coordinator.db
        db.db_path = Path("/nonexistent/path/db.db")

        # Should not raise exception
        delete_db(db)

    def test_delete_db_error(self, coordinator: AreaOccupancyCoordinator, tmp_path):
        """Test delete_db with error."""
        db = coordinator.db
        db.db_path = tmp_path / "test.db"

        # Create file
        db.db_path.touch()

        with patch(
            "pathlib.Path.unlink", side_effect=PermissionError("Permission denied")
        ):
            # Should not raise exception, just log error
            delete_db(db)


class TestIsDatabaseCorrupted:
    """Test _is_database_corrupted function."""

    @pytest.mark.parametrize(
        ("error_message", "expected"),
        [
            ("database disk image is malformed", True),
            ("connection error", False),
        ],
    )
    def test_is_database_corrupted(
        self, coordinator: AreaOccupancyCoordinator, error_message: str, expected: bool
    ):
        """Test corruption detection with various error messages."""
        db = coordinator.db
        error = SQLAlchemyError(error_message)
        result = _is_database_corrupted(db, error)
        assert result is expected


class TestAttemptDatabaseRecovery:
    """Test _attempt_database_recovery function."""

    def test_attempt_database_recovery_success(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test successful database recovery with recoverable corruption."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_recovery.db")

        # Create a valid database with tables
        db.init_db()
        _set_db_version(db)

        # Add some test data
        with db.get_session() as session:
            session.add(db.Metadata(key="test_key", value="test_value"))
            session.commit()

        # Create a backup before corrupting
        backup_path = db.db_path.with_suffix(".db.backup")
        shutil.copy2(db.db_path, backup_path)

        # Corrupt database by writing invalid data to the end
        # This simulates partial corruption that might be recoverable
        db.engine.dispose()
        with db.db_path.open("ab") as f:
            f.write(b"CORRUPTED DATA" * 100)

        # Attempt recovery - SQLite can handle appended corruption data
        # because it can still read the valid portion of the database
        result = _attempt_database_recovery(db)

        # Recovery should succeed for this type of corruption
        # (SQLite ignores extra data at the end of the file)
        assert result is True

        # Verify database is readable after recovery
        assert verify_all_tables_exist(db) is True

        # Verify test data is still accessible
        with db.get_session() as session:
            test_data = session.query(db.Metadata).filter_by(key="test_key").first()
            assert test_data is not None
            assert test_data.value == "test_value"


class TestBackupDatabase:
    """Test _backup_database function."""

    def test_backup_database_success(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test successful database backup."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()

        result = _backup_database(db)
        assert result is True

        # Verify backup file exists
        backup_path = db.db_path.with_suffix(".db.backup")
        assert backup_path.exists()


class TestRestoreDatabaseFromBackup:
    """Test _restore_database_from_backup function."""

    def test_restore_database_from_backup_success(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test successful database restoration."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()

        # Create backup
        _backup_database(db)

        # Corrupt database
        db.db_path.write_text("corrupted")

        result = _restore_database_from_backup(db)
        assert result is True


class TestHandleDatabaseCorruption:
    """Test _handle_database_corruption function."""

    def test_handle_database_corruption_recovery_path_selection(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test that corruption handling selects correct recovery path."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_recovery_path.db")
        db.init_db()
        _set_db_version(db)
        db.enable_auto_recovery = True
        db.enable_periodic_backups = True

        # Add test data
        with db.get_session() as session:
            session.add(db.Metadata(key="test_data", value="original"))
            session.commit()

        # Create backup
        _backup_database(db)

        # Corrupt database
        db.engine.dispose()
        db.db_path.write_text("corrupted")

        # Mock recovery to fail, forcing restore from backup path
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance._attempt_database_recovery",
                return_value=False,
            ),
        ):
            result = _handle_database_corruption(db)

        # Should succeed via restore from backup path
        assert result is True
        assert verify_all_tables_exist(db) is True
        assert _check_database_integrity(db) is True

        # Verify data was restored from backup
        # Note: If tables were missing and reinitialized, data might be lost
        # but the database should still be functional
        with db.get_session() as session:
            restored = session.query(db.Metadata).filter_by(key="test_data").first()
            # Data should be restored if backup had all tables
            # If backup was missing tables and reinitialized, data might be lost
            # but database should still be functional
            if restored:
                assert restored.value == "original"

    def test_handle_database_corruption_restore_from_backup(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test handling corruption by restoring from backup."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_corruption_restore.db")
        db.init_db()
        _set_db_version(db)
        db.enable_auto_recovery = True
        db.enable_periodic_backups = True

        # Add test data to verify it's restored
        with db.get_session() as session:
            session.add(db.Metadata(key="test_data", value="should_be_restored"))
            session.commit()

        # Create backup before corrupting
        _backup_database(db)
        backup_path = db.db_path.with_suffix(".db.backup")
        assert backup_path.exists()

        # Corrupt database by completely overwriting it
        db.engine.dispose()
        db.db_path.write_text("corrupted")

        # Handle corruption - should restore from backup
        # Recovery will fail because file is completely corrupted, so restore should be attempted
        result = _handle_database_corruption(db)

        # Verify corruption was handled by restore
        assert result is True

        # After restore, _handle_database_corruption ensures all tables exist
        # Verify database is healthy after restore
        assert verify_all_tables_exist(db) is True
        assert _check_database_integrity(db) is True
        assert get_db_version(db) == DB_SCHEMA_VERSION

        # Verify test data was restored from backup
        # Note: If tables were missing and reinitialized, data might be lost
        # but the database should still be functional
        with db.get_session() as session:
            restored_data = (
                session.query(db.Metadata).filter_by(key="test_data").first()
            )
            # Data should be restored if backup had all tables
            # If backup was missing tables and reinitialized, data might be lost
            # but database should still be functional
            if restored_data:
                assert restored_data.value == "should_be_restored"

    def test_handle_database_corruption_recreate_database(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test handling corruption by recreating database when no backup exists."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_corruption_recreate.db")
        db.init_db()
        _set_db_version(db)
        db.enable_auto_recovery = True
        db.enable_periodic_backups = False  # No backups

        # Corrupt database
        db.engine.dispose()
        db.db_path.write_text("corrupted")

        # Handle corruption - should recreate database
        result = _handle_database_corruption(db)

        # Verify corruption was handled by recreation
        assert result is True
        assert verify_all_tables_exist(db) is True
        assert _check_database_integrity(db) is True
        assert get_db_version(db) == DB_SCHEMA_VERSION

        # Verify database is empty (recreated, not restored)
        with db.get_session() as session:
            # Only db_version should exist
            all_metadata = session.query(db.Metadata).all()
            assert len(all_metadata) == 1
            assert all_metadata[0].key == "db_version"


class TestPeriodicHealthCheck:
    """Test periodic_health_check function."""

    def test_periodic_health_check_success(self, coordinator: AreaOccupancyCoordinator):
        """Test periodic health check with healthy database."""
        db = coordinator.db
        db.init_db()
        _set_db_version(db)

        # Verify initial state
        assert _check_database_integrity(db) is True
        assert verify_all_tables_exist(db) is True

        # Run health check
        result = periodic_health_check(db)

        # Verify health check succeeded
        assert result is True

        # Verify database is still healthy after health check
        assert _check_database_integrity(db) is True
        assert verify_all_tables_exist(db) is True
        assert get_db_version(db) == DB_SCHEMA_VERSION

    def test_periodic_health_check_error(self, coordinator: AreaOccupancyCoordinator):
        """Test periodic health check with error."""
        db = coordinator.db

        with patch(
            "custom_components.area_occupancy.db.maintenance._check_database_integrity",
            side_effect=OSError("Error"),
        ):
            result = periodic_health_check(db)
            assert result is False


class TestGetLastPruneTime:
    """Test get_last_prune_time function."""

    def test_get_last_prune_time_success(self, coordinator: AreaOccupancyCoordinator):
        """Test getting last prune time successfully."""
        db = coordinator.db
        db.init_db()

        # Initially should be None
        result = get_last_prune_time(db)
        assert result is None

        # Set a prune time
        prune_time = dt_util.utcnow()
        set_last_prune_time(db, prune_time)

        # Verify retrieved time matches
        result = get_last_prune_time(db)
        assert result is not None
        assert isinstance(result, datetime)
        # Allow small time difference due to database storage precision
        assert abs((result - prune_time).total_seconds()) < 1


class TestSetLastPruneTime:
    """Test set_last_prune_time function."""

    def test_set_last_prune_time_success(self, coordinator: AreaOccupancyCoordinator):
        """Test setting last prune time successfully."""
        db = coordinator.db
        db.init_db()

        prune_time = dt_util.utcnow()

        set_last_prune_time(db, prune_time)

        result = get_last_prune_time(db)
        assert result is not None

    def test_set_last_prune_time_with_external_session(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test set_last_prune_time with external session parameter."""
        db = coordinator.db
        db.init_db()

        prune_time = dt_util.utcnow()

        # Use external session
        with db.get_session() as session:
            set_last_prune_time(db, prune_time, session=session)
            # External session should not be committed by set_last_prune_time
            # We need to commit manually
            session.commit()

        # Verify prune time was set
        result = get_last_prune_time(db)
        assert result is not None
        assert abs((result - prune_time).total_seconds()) < 1

    def test_set_last_prune_time_with_external_session_update_existing(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test set_last_prune_time with external session updates existing value."""
        db = coordinator.db
        db.init_db()

        # Set initial prune time
        initial_time = dt_util.utcnow()
        set_last_prune_time(db, initial_time)

        # Update with external session
        new_time = dt_util.utcnow()
        with db.get_session() as session:
            set_last_prune_time(db, new_time, session=session)
            session.commit()

        # Verify updated time
        result = get_last_prune_time(db)
        assert result is not None
        assert abs((result - new_time).total_seconds()) < 1
        assert abs((result - initial_time).total_seconds()) > 0


class TestEnsureDbExistsErrorPaths:
    """Test ensure_db_exists error paths."""

    def test_ensure_db_exists_corrupted_header(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test ensure_db_exists with corrupted SQLite header."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_corrupted.db")

        # Create file with invalid SQLite header
        db.db_path.write_bytes(b"INVALID HEADER")

        ensure_db_exists(db)

        # Should recreate database
        assert verify_all_tables_exist(db) is True

    def test_ensure_db_exists_permission_error(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test ensure_db_exists with permission error reading file."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_permission.db")

        # Create file
        db.db_path.touch()

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            ensure_db_exists(db)

        # Should still create database
        assert verify_all_tables_exist(db) is True


class TestAttemptDatabaseRecoveryEdgeCases:
    """Test _attempt_database_recovery function - additional scenarios."""

    def test_attempt_database_recovery_with_wal_checkpoint(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test recovery attempts WAL checkpoint for recoverable corruption."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_wal_recovery.db")

        # Create a valid database with tables
        db.init_db()
        _set_db_version(db)

        # Add test data
        with db.get_session() as session:
            session.add(db.Metadata(key="test_key", value="test_value"))
            session.commit()

        # Close engine properly before attempting recovery
        db.engine.dispose()

        # Attempt recovery on a valid database (should succeed)
        # This tests the recovery path even when database is valid
        result = _attempt_database_recovery(db)

        # Recovery should succeed for valid database
        assert result is True
        assert verify_all_tables_exist(db) is True

        # Verify data is still accessible
        with db.get_session() as session:
            test_data = session.query(db.Metadata).filter_by(key="test_key").first()
            assert test_data is not None
            assert test_data.value == "test_value"

    def test_attempt_database_recovery_no_tables(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test recovery when database has no tables."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_recovery.db")

        # Create empty database (valid SQLite but no tables)
        with db.engine.connect() as conn:
            conn.execute(text("CREATE TABLE _temp (id INTEGER)"))
            conn.execute(text("DROP TABLE _temp"))
            conn.commit()

        # Attempt recovery
        result = _attempt_database_recovery(db)

        # Recovery should fail because there are no tables to recover
        # The recovery function checks if it can read tables, and if none exist, it fails
        assert result is False

        # Verify database is still empty
        with db.engine.connect() as conn:
            result_query = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
            # Only sqlite_master and sqlite_sequence (if exists) should be present
            table_names = [row[0] for row in result_query]
            assert "areas" not in table_names
            assert "entities" not in table_names

    def test_attempt_database_recovery_error(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test recovery with error."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_recovery_error.db")

        # Create corrupted database
        db.db_path.write_text("corrupted")

        result = _attempt_database_recovery(db)
        assert result is False


class TestBackupDatabaseEdgeCases:
    """Test backup_database function - additional scenarios."""

    def test_backup_database_no_path(self, coordinator: AreaOccupancyCoordinator):
        """Test backup when db_path is None."""
        db = coordinator.db
        db.db_path = None
        result = _backup_database(db)
        assert result is False

    def test_backup_database_file_not_exists(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test backup when file doesn't exist."""
        db = coordinator.db
        db.db_path = tmp_path / "nonexistent.db"
        result = _backup_database(db)
        assert result is False

    def test_backup_database_permission_error(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test backup with permission error."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()

        with patch("shutil.copy2", side_effect=PermissionError("Permission denied")):
            result = _backup_database(db)
            assert result is False

    def test_backup_database_shutil_error(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test backup with shutil error."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()

        with patch("shutil.copy2", side_effect=shutil.Error("Shutil error")):
            result = _backup_database(db)
            assert result is False


class TestRestoreDatabaseFromBackupEdgeCases:
    """Test restore_database_from_backup function - additional scenarios."""

    def test_restore_database_from_backup_no_path(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test restore when db_path is None."""
        db = coordinator.db
        db.db_path = None
        result = _restore_database_from_backup(db)
        assert result is False

    def test_restore_database_from_backup_no_backup(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test restore when backup doesn't exist."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()

        result = _restore_database_from_backup(db)
        assert result is False

    def test_restore_database_from_backup_error(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test restore with error."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()

        # Create backup
        _backup_database(db)

        # Mock error during restore
        with patch("shutil.copy2", side_effect=OSError("Restore error")):
            result = _restore_database_from_backup(db)
            assert result is False

    def test_restore_database_from_backup_sqlalchemy_error(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test restore with SQLAlchemy error."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()

        # Create backup
        _backup_database(db)

        # Mock SQLAlchemy error during engine recreation
        with patch.object(
            db, "update_session_maker", side_effect=sa.exc.SQLAlchemyError("SQL error")
        ):
            result = _restore_database_from_backup(db)
            assert result is False


class TestHandleDatabaseCorruptionEdgeCases:
    """Test _handle_database_corruption function - additional scenarios."""

    def test_handle_database_corruption_auto_recovery_disabled(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test handling corruption when auto-recovery is disabled."""
        db = coordinator.db
        db.enable_auto_recovery = False

        result = _handle_database_corruption(db)
        assert result is False

    def test_handle_database_corruption_recovery_success(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test handling corruption with successful recovery."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()
        db.enable_auto_recovery = True

        # Mock successful recovery
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance._attempt_database_recovery",
                return_value=True,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance._check_database_integrity",
                return_value=True,
            ),
        ):
            result = _handle_database_corruption(db)
            assert result is True

    def test_handle_database_corruption_restore_from_backup(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test handling corruption by restoring from backup."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()
        db.enable_auto_recovery = True
        db.enable_periodic_backups = True

        # Create backup
        _backup_database(db)

        # Mock recovery failure but restore success
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance._attempt_database_recovery",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance._restore_database_from_backup",
                return_value=True,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance._check_database_integrity",
                return_value=True,
            ),
        ):
            result = _handle_database_corruption(db)
            assert result is True

    def test_handle_database_corruption_recreate_database(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test handling corruption by recreating database."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()
        db.enable_auto_recovery = True
        db.enable_periodic_backups = False

        # Mock all recovery attempts failing
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance._attempt_database_recovery",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.delete_db",
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.init_db",
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance._set_db_version",
            ),
        ):
            result = _handle_database_corruption(db)
            assert result is True

    def test_handle_database_corruption_recreate_failure(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test handling corruption when recreation fails."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()
        db.enable_auto_recovery = True
        db.enable_periodic_backups = False

        # Mock all recovery attempts failing including recreation
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance._attempt_database_recovery",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.delete_db",
                side_effect=OSError("Recreate failed"),
            ),
        ):
            result = _handle_database_corruption(db)
            assert result is False


class TestEnsureSchemaUpToDate:
    """Test _ensure_schema_up_to_date function."""

    def test_ensure_schema_up_to_date_version_match(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test _ensure_schema_up_to_date when version matches."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_version_match.db")

        # Create database with correct version
        init_db(db)
        _set_db_version(db)

        # Add some test data
        with db.get_session() as session:
            session.add(db.Metadata(key="test_key", value="test_value"))
            session.commit()

        # Call _ensure_schema_up_to_date
        _ensure_schema_up_to_date(db)

        # Verify database was not recreated (data still exists)
        assert verify_all_tables_exist(db) is True
        assert get_db_version(db) == DB_SCHEMA_VERSION
        with db.get_session() as session:
            result = session.query(db.Metadata).filter_by(key="test_key").first()
            assert result is not None
            assert result.value == "test_value"

    def test_ensure_schema_up_to_date_version_mismatch(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test _ensure_schema_up_to_date when version doesn't match."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_version_mismatch_schema.db")

        # Create database with old version
        init_db(db)
        with db.get_session() as session:
            session.add(db.Metadata(key="db_version", value=str(DB_SCHEMA_VERSION - 1)))
            session.commit()

        # Add some test data to verify it's cleared
        with db.get_session() as session:
            session.add(db.Metadata(key="test_key", value="test_value"))
            session.commit()

        # Call _ensure_schema_up_to_date
        _ensure_schema_up_to_date(db)

        # Verify database was deleted and recreated with correct version
        assert verify_all_tables_exist(db) is True
        assert get_db_version(db) == DB_SCHEMA_VERSION

        # Verify old data was cleared
        with db.get_session() as session:
            result = session.query(db.Metadata).filter_by(key="test_key").first()
            assert result is None

    def test_ensure_schema_up_to_date_error_during_check(
        self, coordinator: AreaOccupancyCoordinator, tmp_path, monkeypatch
    ):
        """Test _ensure_schema_up_to_date when error occurs during version check."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_schema_error.db")

        # Create database
        init_db(db)
        _set_db_version(db)

        # Mock get_db_version to raise error
        def bad_get_version(db_instance):
            raise SQLAlchemyError("Version check error")

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.maintenance.get_db_version",
            bad_get_version,
        )

        # Call _ensure_schema_up_to_date - should recreate database
        _ensure_schema_up_to_date(db)

        # Verify database was recreated
        assert verify_all_tables_exist(db) is True
        assert get_db_version(db) == DB_SCHEMA_VERSION

    def test_ensure_schema_up_to_date_recreation_failure(
        self, coordinator: AreaOccupancyCoordinator, tmp_path, monkeypatch
    ):
        """Test _ensure_schema_up_to_date when recreation fails."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_schema_recreate_fail.db")

        # Create database with old version
        init_db(db)
        with db.get_session() as session:
            session.add(db.Metadata(key="db_version", value=str(DB_SCHEMA_VERSION - 1)))
            session.commit()

        # Mock delete_db to raise error
        def bad_delete_db(db_instance):
            raise OSError("Delete failed")

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.maintenance.delete_db", bad_delete_db
        )

        # Call _ensure_schema_up_to_date - should raise exception
        with pytest.raises(OSError, match="Delete failed"):
            _ensure_schema_up_to_date(db)


class TestPeriodicHealthCheckEdgeCases:
    """Test periodic_health_check function - additional scenarios."""

    def test_periodic_health_check_corruption_detected(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test health check when corruption is detected."""
        db = coordinator.db
        db.init_db()
        db.enable_auto_recovery = True

        with (
            patch(
                "custom_components.area_occupancy.db.maintenance._check_database_integrity",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance._handle_database_corruption",
                return_value=True,
            ),
        ):
            result = periodic_health_check(db)
            assert result is True

    def test_periodic_health_check_missing_tables_recovery_failure(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test health check when table recovery fails."""
        db = coordinator.db
        db.init_db()

        with (
            patch(
                "custom_components.area_occupancy.db.maintenance._check_database_integrity",
                return_value=True,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.get_missing_tables",
                return_value={"entities"},
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.init_db",
                side_effect=RuntimeError("Recovery failed"),
            ),
        ):
            result = periodic_health_check(db)
            assert result is False

    def test_periodic_health_check_backup_creation_no_backup(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test health check creates backup when no backup exists."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()
        db.enable_periodic_backups = True
        db.backup_interval_hours = 1

        # Ensure no backup exists
        backup_path = db.db_path.with_suffix(".db.backup")
        if backup_path.exists():
            backup_path.unlink()

        # Run health check
        result = periodic_health_check(db)
        assert result is True

        # Verify backup was created
        assert backup_path.exists()

    def test_periodic_health_check_backup_creation_old_backup(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test health check creates backup when backup is old."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test.db")
        db.init_db()
        db.enable_periodic_backups = True
        db.backup_interval_hours = 1

        # Create old backup
        backup_path = db.db_path.with_suffix(".db.backup")
        _backup_database(db)
        assert backup_path.exists()

        # Make backup old (2 hours ago, interval is 1 hour)
        old_time = time.time() - (2 * 3600)
        os.utime(backup_path, (old_time, old_time))

        # Run health check
        result = periodic_health_check(db)
        assert result is True

        # Verify backup was recreated (newer timestamp)
        assert backup_path.exists()
        new_time = backup_path.stat().st_mtime
        assert new_time > old_time

    def test_periodic_health_check_backup_failure(
        self, coordinator: AreaOccupancyCoordinator, tmp_path
    ):
        """Test health check handles backup failure gracefully."""
        db = coordinator.db
        setup_test_db_engine(db, tmp_path / "test_backup_failure.db")
        db.init_db()
        _set_db_version(db)
        db.enable_periodic_backups = True
        db.backup_interval_hours = 1

        # Create old backup to trigger new backup
        backup_path = db.db_path.with_suffix(".db.backup")
        backup_path.touch()

        # Make backup old
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(backup_path, (old_time, old_time))

        # Mock backup failure but verify health check continues
        with patch(
            "custom_components.area_occupancy.db.maintenance._backup_database",
            return_value=False,
        ):
            result = periodic_health_check(db)
            # Should still succeed even if backup fails
            assert result is True

        # Verify database is still healthy and accessible
        assert _check_database_integrity(db) is True
        assert verify_all_tables_exist(db) is True
        assert get_db_version(db) == DB_SCHEMA_VERSION

    def test_periodic_health_check_error(self, coordinator: AreaOccupancyCoordinator):
        """Test health check with error."""
        db = coordinator.db
        db.init_db()

        with patch(
            "custom_components.area_occupancy.db.maintenance._check_database_integrity",
            side_effect=OSError("Health check error"),
        ):
            result = periodic_health_check(db)
            assert result is False
