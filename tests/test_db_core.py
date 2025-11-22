"""Tests for AreaOccupancyDB core functionality - initialization, session management, locks, delegation."""
# ruff: noqa: SLF001

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

from filelock import FileLock, Timeout
import pytest
from sqlalchemy import create_engine, text

from custom_components.area_occupancy.db import AreaOccupancyDB
from homeassistant.exceptions import HomeAssistantError


class TestAreaOccupancyDBInitialization:
    """Test AreaOccupancyDB initialization."""

    def test_initialization_with_valid_coordinator(self, coordinator):
        """Test initialization with valid coordinator."""

        db = AreaOccupancyDB(coordinator=coordinator)

        assert db.coordinator is coordinator
        assert db.hass is coordinator.hass
        assert db.engine is not None
        assert db._session_maker is not None
        assert db.storage_path is not None
        assert db.db_path is not None
        assert db._lock_path is not None

    def test_initialization_with_none_config_entry(self, coordinator):
        """Test initialization fails when config_entry is None."""

        coordinator.config_entry = None

        with pytest.raises(ValueError, match="Coordinator config_entry cannot be None"):
            AreaOccupancyDB(coordinator=coordinator)

    def test_initialization_creates_storage_directory(self, coordinator, tmp_path):
        """Test that initialization creates storage directory."""

        with patch.object(coordinator.hass.config, "config_dir", str(tmp_path)):
            db = AreaOccupancyDB(coordinator=coordinator)
            assert db.storage_path.exists()
            assert db.storage_path.is_dir()

    def test_initialization_sets_recovery_config(self, coordinator):
        """Test that initialization sets recovery configuration."""
        db = AreaOccupancyDB(coordinator=coordinator)

        assert hasattr(db, "enable_auto_recovery")
        assert hasattr(db, "max_recovery_attempts")
        assert hasattr(db, "enable_periodic_backups")
        assert hasattr(db, "backup_interval_hours")

    def test_initialization_creates_model_classes_dict(self, coordinator):
        """Test that initialization creates model_classes dictionary."""
        db = AreaOccupancyDB(coordinator=coordinator)

        assert "Areas" in db.model_classes
        assert "Entities" in db.model_classes
        assert "Intervals" in db.model_classes
        assert db.model_classes["Areas"] == db.Areas

    def test_initialization_auto_init_db_env_var(self, coordinator, monkeypatch):
        """Test that AREA_OCCUPANCY_AUTO_INIT_DB env var triggers initialization."""
        monkeypatch.setenv("AREA_OCCUPANCY_AUTO_INIT_DB", "1")

        with patch(
            "custom_components.area_occupancy.db.core.maintenance.ensure_db_exists"
        ) as mock_ensure:
            db = AreaOccupancyDB(coordinator=coordinator)
            mock_ensure.assert_called_once_with(db)

    def test_initialize_database(self, coordinator):
        """Test initialize_database method."""
        db = AreaOccupancyDB(coordinator=coordinator)

        with patch(
            "custom_components.area_occupancy.db.core.maintenance.ensure_db_exists"
        ) as mock_ensure:
            db.initialize_database()
            mock_ensure.assert_called_once_with(db)


class TestSessionManagement:
    """Test session management methods."""

    def test_get_session_creates_session(self, test_db):
        """Test that get_session creates a session."""
        db = test_db

        with db.get_session() as session:
            assert session is not None
            # Session should be usable
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_get_session_rollback_on_exception(self, test_db):
        """Test that get_session rolls back on exception."""
        db = test_db

        with pytest.raises(ValueError), db.get_session():
            # Add something that will fail
            raise ValueError("Test error")
            # Session should be rolled back automatically

    def test_get_session_closes_on_exit(self, test_db):
        """Test that get_session closes session on exit."""
        db = test_db

        with db.get_session() as session:
            session_id = id(session)

        # Session should be closed after context manager exits
        # We can't directly check if it's closed, but we can verify
        # that a new session is created each time
        with db.get_session() as session2:
            assert id(session2) != session_id

    def test_get_locked_session_with_lock_path(self, test_db):
        """Test get_locked_session when lock path exists."""
        db = test_db
        assert db._lock_path is not None

        with db.get_locked_session() as session:
            assert session is not None
            # Session should be usable
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_get_locked_session_without_lock_path(self, coordinator, tmp_path):
        """Test get_locked_session falls back to regular session when no lock path."""

        db = AreaOccupancyDB(coordinator=coordinator)
        db._lock_path = None

        # Should fall back to regular session
        with db.get_locked_session() as session:
            assert session is not None

    def test_get_locked_session_timeout(self, test_db, tmp_path):
        """Test get_locked_session raises error on lock timeout."""
        db = test_db
        db._lock_path = tmp_path / "test.lock"

        # Create a lock file that's already locked
        lock_file = db._lock_path
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a file lock that will hold the lock

        existing_lock = FileLock(lock_file, timeout=0.1)
        existing_lock.acquire()

        try:
            # Try to get a locked session with short timeout
            with (
                pytest.raises(HomeAssistantError, match="Database is busy"),
                db.get_locked_session(timeout=0.1),
            ):
                pass
        finally:
            existing_lock.release()

    def test_get_locked_session_handles_timeout_exception(self, test_db, monkeypatch):
        """Test that get_locked_session properly handles Timeout exception."""
        db = test_db

        @contextmanager
        def mock_filelock(*args, **kwargs):
            raise Timeout("Lock timeout")

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.core.FileLock", mock_filelock
        )

        with (
            pytest.raises(HomeAssistantError, match="Database is busy"),
            db.get_locked_session(timeout=1),
        ):
            pass


class TestTableProperties:
    """Test table property accessors."""

    def test_table_properties(self, test_db):
        """Test table property accessors."""
        db = test_db

        assert db.areas.name == "areas"
        assert db.entities.name == "entities"
        assert db.intervals.name == "intervals"
        assert db.priors.name == "priors"
        assert db.metadata.name == "metadata"

    def test_get_engine(self, test_db):
        """Test get_engine method."""
        db = test_db

        engine = db.get_engine()
        assert engine is not None
        assert engine == db.engine

    def test_update_session_maker(self, test_db):
        """Test update_session_maker method."""
        db = test_db

        original_maker = db._session_maker

        # Create new engine
        new_engine = create_engine("sqlite:///:memory:")
        db.engine = new_engine

        # Update session maker
        db.update_session_maker()

        # Session maker should be updated
        assert db._session_maker is not original_maker
        assert db._session_maker.kw.get("bind") == new_engine


class TestModelClassReferences:
    """Test model class references."""

    def test_model_class_references(self, test_db):
        """Test that model classes are correctly referenced."""
        db = test_db

        # Test that class attributes reference schema models
        assert db.Areas is not None
        assert db.Entities is not None
        assert db.Intervals is not None
        assert db.Priors is not None
        assert db.Metadata is not None
        assert db.IntervalAggregates is not None
        assert db.OccupiedIntervalsCache is not None
        assert db.GlobalPriors is not None
        assert db.NumericSamples is not None
        assert db.NumericAggregates is not None
        assert db.NumericCorrelations is not None
        assert db.EntityStatistics is not None
        assert db.AreaRelationships is not None
        assert db.CrossAreaStats is not None


class TestDelegationCorrectness:
    """Test that core.py methods correctly delegate to underlying modules."""

    @pytest.mark.asyncio
    async def test_load_data_delegates_to_operations(self, test_db):
        """Test that load_data delegates to operations.load_data."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.operations.load_data"
        ) as mock_load:
            await db.load_data()
            mock_load.assert_called_once_with(db)

    def test_save_area_data_delegates_to_operations(self, test_db):
        """Test that save_area_data delegates to operations.save_area_data."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.operations.save_area_data"
        ) as mock_save:
            db.save_area_data("Test Area")
            mock_save.assert_called_once_with(db, "Test Area")

    def test_save_entity_data_delegates_to_operations(self, test_db):
        """Test that save_entity_data delegates to operations.save_entity_data."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.operations.save_entity_data"
        ) as mock_save:
            db.save_entity_data()
            mock_save.assert_called_once_with(db)

    def test_get_area_data_delegates_to_queries(self, test_db):
        """Test that get_area_data delegates to queries.get_area_data."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.queries.get_area_data"
        ) as mock_get:
            mock_get.return_value = {"entry_id": "test"}
            result = db.get_area_data("test_entry")
            mock_get.assert_called_once_with(db, "test_entry")
            assert result == {"entry_id": "test"}

    def test_get_latest_interval_delegates_to_queries(self, test_db):
        """Test that get_latest_interval delegates to queries.get_latest_interval."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.queries.get_latest_interval"
        ) as mock_get:
            mock_get.return_value = datetime.now()
            result = db.get_latest_interval()
            mock_get.assert_called_once_with(db)
            assert isinstance(result, datetime)

    def test_is_valid_state_delegates_to_utils(self, test_db):
        """Test that is_valid_state delegates to utils.is_valid_state."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.utils.is_valid_state"
        ) as mock_is_valid:
            mock_is_valid.return_value = True
            result = db.is_valid_state("on")
            mock_is_valid.assert_called_once_with("on")
            assert result is True

    def test_is_intervals_empty_delegates_to_utils(self, test_db):
        """Test that is_intervals_empty delegates to utils.is_intervals_empty."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.utils.is_intervals_empty"
        ) as mock_is_empty:
            mock_is_empty.return_value = True
            result = db.is_intervals_empty()
            mock_is_empty.assert_called_once_with(db)
            assert result is True

    @pytest.mark.asyncio
    async def test_sync_states_delegates_to_sync(self, test_db):
        """Test that sync_states delegates to sync.sync_states."""
        db = test_db

        with patch("custom_components.area_occupancy.db.sync.sync_states") as mock_sync:
            await db.sync_states()
            mock_sync.assert_called_once_with(db)

    def test_aggregate_raw_to_daily_delegates_to_aggregation(self, test_db):
        """Test that aggregate_raw_to_daily delegates to aggregation.aggregate_raw_to_daily."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.aggregation.aggregate_raw_to_daily"
        ) as mock_agg:
            mock_agg.return_value = 5
            result = db.aggregate_raw_to_daily("Test Area")
            mock_agg.assert_called_once_with(db, "Test Area")
            assert result == 5

    def test_check_database_integrity_delegates_to_maintenance(self, test_db):
        """Test that check_database_integrity delegates to maintenance.check_database_integrity."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.maintenance.check_database_integrity"
        ) as mock_check:
            mock_check.return_value = True
            result = db.check_database_integrity()
            mock_check.assert_called_once_with(db)
            assert result is True

    def test_save_area_relationship_delegates_to_relationships(self, test_db):
        """Test that save_area_relationship delegates to relationships.save_area_relationship."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.relationships.save_area_relationship"
        ) as mock_save:
            mock_save.return_value = True
            result = db.save_area_relationship("Area1", "Area2", "adjacent", 0.5)
            mock_save.assert_called_once_with(
                db, "Area1", "Area2", "adjacent", 0.5, None
            )
            assert result is True

    def test_analyze_numeric_correlation_delegates_to_correlation(self, test_db):
        """Test that analyze_numeric_correlation delegates to correlation.analyze_numeric_correlation."""
        db = test_db

        with patch(
            "custom_components.area_occupancy.db.correlation.analyze_numeric_correlation"
        ) as mock_analyze:
            mock_analyze.return_value = {"correlation": 0.8}
            result = db.analyze_numeric_correlation("Area1", "sensor.temp", 30)
            mock_analyze.assert_called_once_with(db, "Area1", "sensor.temp", 30)
            assert result == {"correlation": 0.8}


class TestErrorHandling:
    """Test error handling at core level."""

    def test_get_session_handles_exception(self, test_db, monkeypatch):
        """Test that get_session properly handles exceptions."""
        db = test_db

        def failing_maker():
            raise RuntimeError("Session creation failed")

        db._session_maker = failing_maker

        with (
            pytest.raises(RuntimeError, match="Session creation failed"),
            db.get_session(),
        ):
            pass

    def test_get_locked_session_handles_lock_errors(self, test_db, monkeypatch):
        """Test that get_locked_session handles lock errors gracefully."""
        db = test_db

        @contextmanager
        def mock_filelock(*args, **kwargs):
            raise Timeout("Lock timeout")

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.core.FileLock", mock_filelock
        )

        with (
            pytest.raises(HomeAssistantError, match="Database is busy"),
            db.get_locked_session(timeout=1),
        ):
            pass
