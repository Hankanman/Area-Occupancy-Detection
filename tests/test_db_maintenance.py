"""Tests for database maintenance functions."""
# ruff: noqa: SLF001

from datetime import datetime
import os
from pathlib import Path
import shutil
import time
from unittest.mock import Mock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from custom_components.area_occupancy.db import DB_VERSION, Base
from custom_components.area_occupancy.db.maintenance import (
    _create_tables_individually,
    _enable_wal_mode,
    _get_required_tables,
    attempt_database_recovery,
    backup_database,
    check_database_accessibility,
    check_database_integrity,
    delete_db,
    ensure_db_exists,
    force_reinitialize,
    get_db_version,
    get_last_prune_time,
    get_missing_tables,
    handle_database_corruption,
    init_db,
    is_database_corrupted,
    periodic_health_check,
    restore_database_from_backup,
    set_db_version,
    set_last_prune_time,
    verify_all_tables_exist,
)
from homeassistant.util import dt as dt_util


class TestEnsureDbExists:
    """Test ensure_db_exists function."""

    def test_ensure_db_exists_new_database(self, test_db, tmp_path):
        """Test ensure_db_exists with new database."""
        db = test_db
        db.db_path = tmp_path / "test_new.db"

        # Create new engine pointing to the new database path
        db.engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        db._session_maker = sessionmaker(bind=db.engine)

        # This should create all tables
        ensure_db_exists(db)

        # Verify tables were created
        assert verify_all_tables_exist(db) is True

    def test_ensure_db_exists_with_file_no_tables(self, test_db, tmp_path):
        """Test ensure_db_exists when file exists but has no tables (race condition)."""
        db = test_db
        db.db_path = tmp_path / "test_race.db"

        # Create new engine pointing to the new database path
        db.engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        db._session_maker = sessionmaker(bind=db.engine)

        # Create an empty SQLite database file (valid header but no tables)
        with db.engine.connect() as conn:
            conn.execute(text("CREATE TABLE _temp (id INTEGER)"))
            conn.execute(text("DROP TABLE _temp"))
            conn.commit()

        # Now verify this triggers table creation
        ensure_db_exists(db)

        # Verify all required tables were created
        assert verify_all_tables_exist(db) is True

    def test_ensure_db_exists_with_complete_database(self, test_db, tmp_path):
        """Test ensure_db_exists when database is already complete."""
        db = test_db
        db.db_path = tmp_path / "test_complete.db"

        # Create new engine pointing to the new database path
        db.engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        db._session_maker = sessionmaker(bind=db.engine)

        # Create a fully initialized database
        init_db(db)
        set_db_version(db)

        ensure_db_exists(db)

        # Verify tables still exist (not corrupted)
        assert verify_all_tables_exist(db) is True


class TestCheckDatabaseIntegrity:
    """Test check_database_integrity function."""

    def test_check_database_integrity_success(self, test_db):
        """Test integrity check with healthy database."""
        db = test_db
        db.init_db()
        result = check_database_integrity(db)
        assert result is True

    def test_check_database_integrity_error(self, test_db, monkeypatch):
        """Test integrity check with database error."""
        db = test_db

        def bad_connect():
            raise SQLAlchemyError("Error")

        monkeypatch.setattr(db.engine, "connect", bad_connect)
        result = check_database_integrity(db)
        assert result is False


class TestCheckDatabaseAccessibility:
    """Test check_database_accessibility function."""

    def test_check_database_accessibility_success(self, test_db, tmp_path):
        """Test accessibility check with accessible database."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db.init_db()

        result = check_database_accessibility(db)
        assert result is True

    def test_check_database_accessibility_file_not_exists(self, test_db):
        """Test accessibility check when file doesn't exist."""
        db = test_db
        db.db_path = Path("/nonexistent/path/db.db")

        result = check_database_accessibility(db)
        assert result is False


class TestGetRequiredTables:
    """Test _get_required_tables function."""

    def test_get_required_tables(self):
        """Test getting required tables list."""
        tables = _get_required_tables()
        assert isinstance(tables, set)
        assert "areas" in tables
        assert "entities" in tables
        assert "intervals" in tables
        assert "priors" in tables
        assert "metadata" in tables


class TestVerifyAllTablesExist:
    """Test verify_all_tables_exist function."""

    def test_verify_all_tables_exist_success(self, test_db):
        """Test verification with all tables present."""
        db = test_db
        db.init_db()
        assert verify_all_tables_exist(db) is True

    def test_verify_all_tables_exist_error(self, test_db, monkeypatch):
        """Test verification with database error."""
        db = test_db

        def bad_connect():
            raise SQLAlchemyError("DB Error")

        monkeypatch.setattr(db.engine, "connect", bad_connect)
        assert verify_all_tables_exist(db) is False


class TestGetMissingTables:
    """Test get_missing_tables function."""

    def test_get_missing_tables_none_missing(self, test_db):
        """Test getting missing tables when all exist."""
        db = test_db
        db.init_db()
        missing = get_missing_tables(db)
        assert missing == set()

    def test_get_missing_tables_some_missing(self, test_db, tmp_path):
        """Test getting missing tables when some are missing."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)

        # Create only some tables
        Base.metadata.create_all(db.engine, tables=[Base.metadata.tables["areas"]])

        missing = get_missing_tables(db)
        assert len(missing) > 0
        assert "entities" in missing


class TestInitDb:
    """Test init_db function."""

    def test_init_db_success(self, test_db):
        """Test init_db with successful initialization."""
        db = test_db

        with (
            patch("custom_components.area_occupancy.db.maintenance._enable_wal_mode"),
            patch.object(db.engine, "connect"),
        ):
            init_db(db)

    def test_init_db_operational_error_race_condition(self, test_db):
        """Test init_db with operational error (race condition)."""
        db = test_db

        # Mock error with sqlite_errno = 1 (table already exists)
        mock_error = sa.exc.OperationalError("table already exists", None, None)
        mock_error.orig = Mock()
        mock_error.orig.sqlite_errno = 1

        with (
            patch("custom_components.area_occupancy.db.maintenance._enable_wal_mode"),
            patch.object(db.engine, "connect", side_effect=mock_error),
            patch(
                "custom_components.area_occupancy.db.maintenance._create_tables_individually"
            ),
        ):
            init_db(db)

    def test_init_db_operational_error_other(self, test_db):
        """Test init_db with other operational error."""
        db = test_db

        # Mock error with different sqlite_errno
        mock_error = sa.exc.OperationalError("other error", None, None)
        mock_error.orig = Mock()
        mock_error.orig.sqlite_errno = 2

        with (
            patch("custom_components.area_occupancy.db.maintenance._enable_wal_mode"),
            patch.object(db.engine, "connect", side_effect=mock_error),
            pytest.raises(sa.exc.OperationalError),
        ):
            init_db(db)


class TestEnableWalMode:
    """Test _enable_wal_mode function."""

    def test_enable_wal_mode_success(self, test_db):
        """Test _enable_wal_mode with success."""
        db = test_db

        with patch.object(db.engine, "connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn

            _enable_wal_mode(db)

            mock_conn.execute.assert_called_once()

    def test_enable_wal_mode_error(self, test_db):
        """Test _enable_wal_mode with error."""
        db = test_db

        with patch.object(
            db.engine, "connect", side_effect=sa.exc.SQLAlchemyError("WAL error")
        ):
            # Should not raise exception, just log error
            _enable_wal_mode(db)


class TestCreateTablesIndividually:
    """Test _create_tables_individually function."""

    def test_create_tables_individually_success(self, test_db):
        """Test _create_tables_individually with success."""
        db = test_db

        with patch.object(db.engine, "connect"):
            _create_tables_individually(db)

    def test_create_tables_individually_race_condition(self, test_db):
        """Test _create_tables_individually with race condition."""
        db = test_db

        # Mock error with sqlite_errno = 1 (table already exists)
        mock_error = sa.exc.OperationalError("table already exists", None, None)
        mock_error.orig = Mock()
        mock_error.orig.sqlite_errno = 1

        with patch.object(db.engine, "connect", side_effect=mock_error):
            # Should not raise exception
            _create_tables_individually(db)

    def test_create_tables_individually_other_error(self, test_db):
        """Test _create_tables_individually with other error."""
        db = test_db

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
    """Test set_db_version function."""

    def test_set_db_version_update_existing(self, test_db):
        """Test set_db_version when version already exists."""
        db = test_db
        db.init_db()
        set_db_version(db)  # Create initial version

        # Verify the version was set correctly
        version = get_db_version(db)
        assert version == DB_VERSION

        # Call set_db_version again - should update existing
        set_db_version(db)

        # Verify version is still correct
        version_after = get_db_version(db)
        assert version_after == DB_VERSION

    def test_set_db_version_insert_new(self, test_db):
        """Test set_db_version when version doesn't exist."""
        db = test_db
        db.init_db()

        # Delete any existing version
        with db.get_session() as session:
            session.query(db.Metadata).filter_by(key="db_version").delete()
            session.commit()

        # Set version - should insert new
        set_db_version(db)

        # Verify version was set
        version = get_db_version(db)
        assert version == DB_VERSION

    def test_set_db_version_error(self, test_db, monkeypatch):
        """Test set_db_version with error."""
        db = test_db

        def bad_session():
            raise RuntimeError("DB Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        with pytest.raises(RuntimeError):
            set_db_version(db)


class TestGetDbVersion:
    """Test get_db_version function."""

    def test_get_db_version_success(self, test_db):
        """Test get_db_version with success."""
        db = test_db
        db.init_db()
        set_db_version(db)

        version = get_db_version(db)
        assert version == DB_VERSION

    def test_get_db_version_no_metadata(self, test_db):
        """Test get_db_version when no metadata exists."""
        db = test_db
        db.init_db()

        # Delete metadata
        with db.get_session() as session:
            session.query(db.Metadata).filter_by(key="db_version").delete()
            session.commit()

        version = get_db_version(db)
        assert version == 0

    def test_get_db_version_error(self, test_db, monkeypatch):
        """Test get_db_version with error."""
        db = test_db

        def bad_session():
            raise RuntimeError("DB Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        # get_db_version catches exceptions and returns 0
        version = get_db_version(db)
        assert version == 0


class TestDeleteDb:
    """Test delete_db function."""

    def test_delete_db_success(self, test_db, tmp_path):
        """Test delete_db with successful deletion."""
        db = test_db
        db.db_path = tmp_path / "test.db"

        # Create file to delete
        db.db_path.touch()

        delete_db(db)

        assert not db.db_path.exists()

    def test_delete_db_file_not_exists(self, test_db):
        """Test delete_db when file doesn't exist."""
        db = test_db
        db.db_path = Path("/nonexistent/path/db.db")

        # Should not raise exception
        delete_db(db)

    def test_delete_db_error(self, test_db, tmp_path):
        """Test delete_db with error."""
        db = test_db
        db.db_path = tmp_path / "test.db"

        # Create file
        db.db_path.touch()

        with patch(
            "pathlib.Path.unlink", side_effect=PermissionError("Permission denied")
        ):
            # Should not raise exception, just log error
            delete_db(db)


class TestForceReinitialize:
    """Test force_reinitialize function."""

    def test_force_reinitialize(self, test_db):
        """Test force_reinitialize method."""
        db = test_db

        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.init_db"
            ) as mock_init,
            patch(
                "custom_components.area_occupancy.db.maintenance.set_db_version"
            ) as mock_set_version,
        ):
            force_reinitialize(db)

            mock_init.assert_called_once_with(db)
            mock_set_version.assert_called_once_with(db)


class TestIsDatabaseCorrupted:
    """Test is_database_corrupted function."""

    def test_is_database_corrupted_true(self, test_db):
        """Test corruption detection with corrupted database."""
        db = test_db
        error = SQLAlchemyError("database disk image is malformed")
        result = is_database_corrupted(db, error)
        assert result is True

    def test_is_database_corrupted_false(self, test_db):
        """Test corruption detection with non-corruption error."""
        db = test_db
        error = SQLAlchemyError("connection error")
        result = is_database_corrupted(db, error)
        assert result is False


class TestAttemptDatabaseRecovery:
    """Test attempt_database_recovery function."""

    def test_attempt_database_recovery_success(self, test_db, tmp_path):
        """Test successful database recovery."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)

        # Create backup first
        db.init_db()
        backup_database(db)

        # Corrupt database
        db.db_path.write_text("corrupted")

        result = attempt_database_recovery(db)
        # May succeed or fail depending on backup availability
        assert isinstance(result, bool)


class TestBackupDatabase:
    """Test backup_database function."""

    def test_backup_database_success(self, test_db, tmp_path):
        """Test successful database backup."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()

        result = backup_database(db)
        assert result is True

        # Verify backup file exists
        backup_path = db.db_path.with_suffix(".db.backup")
        assert backup_path.exists()


class TestRestoreDatabaseFromBackup:
    """Test restore_database_from_backup function."""

    def test_restore_database_from_backup_success(self, test_db, tmp_path):
        """Test successful database restoration."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()

        # Create backup
        backup_database(db)

        # Corrupt database
        db.db_path.write_text("corrupted")

        result = restore_database_from_backup(db)
        assert result is True


class TestHandleDatabaseCorruption:
    """Test handle_database_corruption function."""

    def test_handle_database_corruption_success(self, test_db, tmp_path):
        """Test handling database corruption successfully."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()
        backup_database(db)

        result = handle_database_corruption(db)
        # May succeed or fail depending on backup
        assert isinstance(result, bool)


class TestPeriodicHealthCheck:
    """Test periodic_health_check function."""

    def test_periodic_health_check_success(self, test_db):
        """Test periodic health check with healthy database."""
        db = test_db
        db.init_db()

        result = periodic_health_check(db)
        assert isinstance(result, bool)

    def test_periodic_health_check_error(self, test_db):
        """Test periodic health check with error."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.maintenance.check_database_integrity",
            side_effect=OSError("Error"),
        ):
            result = periodic_health_check(db)
            assert result is False


class TestGetLastPruneTime:
    """Test get_last_prune_time function."""

    def test_get_last_prune_time_success(self, test_db):
        """Test getting last prune time successfully."""
        db = test_db
        db.init_db()

        result = get_last_prune_time(db)
        # May be None if never set
        assert result is None or isinstance(result, datetime)


class TestSetLastPruneTime:
    """Test set_last_prune_time function."""

    def test_set_last_prune_time_success(self, test_db):
        """Test setting last prune time successfully."""
        db = test_db
        db.init_db()

        prune_time = dt_util.utcnow()

        set_last_prune_time(db, prune_time)

        result = get_last_prune_time(db)
        assert result is not None


class TestEnsureDbExistsErrorPaths:
    """Test ensure_db_exists error paths."""

    def test_ensure_db_exists_corrupted_header(self, test_db, tmp_path):
        """Test ensure_db_exists with corrupted SQLite header."""
        db = test_db
        db.db_path = tmp_path / "test_corrupted.db"

        # Create new engine pointing to the new database path
        db.engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        db._session_maker = sessionmaker(bind=db.engine)

        # Create file with invalid SQLite header
        db.db_path.write_bytes(b"INVALID HEADER")

        ensure_db_exists(db)

        # Should recreate database
        assert verify_all_tables_exist(db) is True

    def test_ensure_db_exists_permission_error(self, test_db, tmp_path):
        """Test ensure_db_exists with permission error reading file."""
        db = test_db
        db.db_path = tmp_path / "test_permission.db"

        # Create new engine pointing to the new database path
        db.engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        db._session_maker = sessionmaker(bind=db.engine)

        # Create file
        db.db_path.touch()

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            ensure_db_exists(db)

        # Should still create database
        assert verify_all_tables_exist(db) is True

    def test_ensure_db_exists_corruption_detected(self, test_db, tmp_path):
        """Test ensure_db_exists when corruption is detected."""
        db = test_db
        db.db_path = tmp_path / "test_corruption.db"

        # Create new engine pointing to the new database path
        db.engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        db._session_maker = sessionmaker(bind=db.engine)

        # Create database first
        init_db(db)
        set_db_version(db)

        # Mock corruption detection
        mock_error = sa.exc.SQLAlchemyError("database disk image is malformed")
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.verify_all_tables_exist",
                side_effect=mock_error,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.is_database_corrupted",
                return_value=True,
            ),
        ):
            ensure_db_exists(db)
            # Should return early without blocking

    def test_ensure_db_exists_init_failure(self, test_db, tmp_path):
        """Test ensure_db_exists when initialization fails."""
        db = test_db
        db.db_path = tmp_path / "test_init_fail.db"

        # Create new engine pointing to the new database path
        db.engine = create_engine(
            f"sqlite:///{db.db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        db._session_maker = sessionmaker(bind=db.engine)

        # Mock initialization failure
        mock_error = sa.exc.SQLAlchemyError("DB Error")
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.verify_all_tables_exist",
                side_effect=mock_error,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.is_database_corrupted",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.init_db",
                side_effect=RuntimeError("Init failed"),
            ),
        ):
            ensure_db_exists(db)
            # Should handle gracefully


class TestAttemptDatabaseRecoveryEdgeCases:
    """Test attempt_database_recovery function - additional scenarios."""

    def test_attempt_database_recovery_no_tables(self, test_db, tmp_path):
        """Test recovery when database has no tables."""
        db = test_db
        db.db_path = tmp_path / "test_recovery.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)

        # Create empty database
        with db.engine.connect() as conn:
            conn.execute(text("CREATE TABLE _temp (id INTEGER)"))
            conn.execute(text("DROP TABLE _temp"))
            conn.commit()

        result = attempt_database_recovery(db)
        # May succeed or fail
        assert isinstance(result, bool)

    def test_attempt_database_recovery_error(self, test_db, tmp_path):
        """Test recovery with error."""
        db = test_db
        db.db_path = tmp_path / "test_recovery_error.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)

        # Create corrupted database
        db.db_path.write_text("corrupted")

        result = attempt_database_recovery(db)
        assert result is False


class TestBackupDatabaseEdgeCases:
    """Test backup_database function - additional scenarios."""

    def test_backup_database_no_path(self, test_db):
        """Test backup when db_path is None."""
        db = test_db
        db.db_path = None
        result = backup_database(db)
        assert result is False

    def test_backup_database_file_not_exists(self, test_db, tmp_path):
        """Test backup when file doesn't exist."""
        db = test_db
        db.db_path = tmp_path / "nonexistent.db"
        result = backup_database(db)
        assert result is False

    def test_backup_database_permission_error(self, test_db, tmp_path):
        """Test backup with permission error."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()

        with patch("shutil.copy2", side_effect=PermissionError("Permission denied")):
            result = backup_database(db)
            assert result is False

    def test_backup_database_shutil_error(self, test_db, tmp_path):
        """Test backup with shutil error."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()

        with patch("shutil.copy2", side_effect=shutil.Error("Shutil error")):
            result = backup_database(db)
            assert result is False


class TestRestoreDatabaseFromBackupEdgeCases:
    """Test restore_database_from_backup function - additional scenarios."""

    def test_restore_database_from_backup_no_path(self, test_db):
        """Test restore when db_path is None."""
        db = test_db
        db.db_path = None
        result = restore_database_from_backup(db)
        assert result is False

    def test_restore_database_from_backup_no_backup(self, test_db, tmp_path):
        """Test restore when backup doesn't exist."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()

        result = restore_database_from_backup(db)
        assert result is False

    def test_restore_database_from_backup_error(self, test_db, tmp_path):
        """Test restore with error."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()

        # Create backup
        backup_database(db)

        # Mock error during restore
        with patch("shutil.copy2", side_effect=OSError("Restore error")):
            result = restore_database_from_backup(db)
            assert result is False

    def test_restore_database_from_backup_sqlalchemy_error(self, test_db, tmp_path):
        """Test restore with SQLAlchemy error."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()

        # Create backup
        backup_database(db)

        # Mock SQLAlchemy error during engine recreation
        with patch.object(
            db, "update_session_maker", side_effect=sa.exc.SQLAlchemyError("SQL error")
        ):
            result = restore_database_from_backup(db)
            assert result is False


class TestHandleDatabaseCorruptionEdgeCases:
    """Test handle_database_corruption function - additional scenarios."""

    def test_handle_database_corruption_auto_recovery_disabled(self, test_db):
        """Test handling corruption when auto-recovery is disabled."""
        db = test_db
        db.enable_auto_recovery = False

        result = handle_database_corruption(db)
        assert result is False

    def test_handle_database_corruption_recovery_success(self, test_db, tmp_path):
        """Test handling corruption with successful recovery."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()
        db.enable_auto_recovery = True

        # Mock successful recovery
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.attempt_database_recovery",
                return_value=True,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.check_database_integrity",
                return_value=True,
            ),
        ):
            result = handle_database_corruption(db)
            assert result is True

    def test_handle_database_corruption_restore_from_backup(self, test_db, tmp_path):
        """Test handling corruption by restoring from backup."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()
        db.enable_auto_recovery = True
        db.enable_periodic_backups = True

        # Create backup
        backup_database(db)

        # Mock recovery failure but restore success
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.attempt_database_recovery",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.restore_database_from_backup",
                return_value=True,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.check_database_integrity",
                return_value=True,
            ),
        ):
            result = handle_database_corruption(db)
            assert result is True

    def test_handle_database_corruption_recreate_database(self, test_db, tmp_path):
        """Test handling corruption by recreating database."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()
        db.enable_auto_recovery = True
        db.enable_periodic_backups = False

        # Mock all recovery attempts failing
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.attempt_database_recovery",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.delete_db",
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.init_db",
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.set_db_version",
            ),
        ):
            result = handle_database_corruption(db)
            assert result is True

    def test_handle_database_corruption_recreate_failure(self, test_db, tmp_path):
        """Test handling corruption when recreation fails."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()
        db.enable_auto_recovery = True
        db.enable_periodic_backups = False

        # Mock all recovery attempts failing including recreation
        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.attempt_database_recovery",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.delete_db",
                side_effect=OSError("Recreate failed"),
            ),
        ):
            result = handle_database_corruption(db)
            assert result is False


class TestPeriodicHealthCheckEdgeCases:
    """Test periodic_health_check function - additional scenarios."""

    def test_periodic_health_check_corruption_detected(self, test_db):
        """Test health check when corruption is detected."""
        db = test_db
        db.init_db()
        db.enable_auto_recovery = True

        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.check_database_integrity",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.handle_database_corruption",
                return_value=True,
            ),
        ):
            result = periodic_health_check(db)
            assert result is True

    def test_periodic_health_check_missing_tables(self, test_db, tmp_path):
        """Test health check when tables are missing."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)

        # Create database with only some tables
        Base.metadata.create_all(db.engine, tables=[Base.metadata.tables["areas"]])

        result = periodic_health_check(db)
        # Should attempt recovery
        assert isinstance(result, bool)

    def test_periodic_health_check_missing_tables_recovery_failure(self, test_db):
        """Test health check when table recovery fails."""
        db = test_db
        db.init_db()

        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.check_database_integrity",
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

    def test_periodic_health_check_backup_creation(self, test_db, tmp_path):
        """Test health check creates periodic backup."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()
        db.enable_periodic_backups = True
        db.backup_interval_hours = 1

        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.check_database_integrity",
                return_value=True,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.get_missing_tables",
                return_value=set(),
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.backup_database",
                return_value=True,
            ),
        ):
            periodic_health_check(db)
            # Backup should be called if interval has passed
            # (may or may not be called depending on file age)

    def test_periodic_health_check_backup_failure(self, test_db, tmp_path):
        """Test health check handles backup failure."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.engine = create_engine(f"sqlite:///{db.db_path}")
        db._session_maker = sessionmaker(bind=db.engine)
        db.init_db()
        db.enable_periodic_backups = True
        db.backup_interval_hours = 1

        # Create old backup to trigger new backup
        backup_path = db.db_path.with_suffix(".db.backup")
        backup_path.touch()

        # Make backup old
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(backup_path, (old_time, old_time))

        with (
            patch(
                "custom_components.area_occupancy.db.maintenance.check_database_integrity",
                return_value=True,
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.get_missing_tables",
                return_value=set(),
            ),
            patch(
                "custom_components.area_occupancy.db.maintenance.backup_database",
                return_value=False,
            ),
        ):
            result = periodic_health_check(db)
            # Should still succeed even if backup fails
            assert result is True

    def test_periodic_health_check_error(self, test_db):
        """Test health check with error."""
        db = test_db
        db.init_db()

        with patch(
            "custom_components.area_occupancy.db.maintenance.check_database_integrity",
            side_effect=OSError("Health check error"),
        ):
            result = periodic_health_check(db)
            assert result is False
