"""Tests for AreaOccupancyDB core functionality - initialization, session management, delegation."""
# ruff: noqa: SLF001

from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db import AreaOccupancyDB


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

    def test_get_session_creates_session(self, coordinator: AreaOccupancyCoordinator):
        """Test that get_session creates a session."""
        db = coordinator.db

        with db.get_session() as session:
            assert session is not None
            # Session should be usable
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_get_session_rollback_on_exception(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that get_session rolls back on exception."""
        db = coordinator.db

        with pytest.raises(ValueError), db.get_session():
            # Add something that will fail
            raise ValueError("Test error")
            # Session should be rolled back automatically

    def test_get_session_closes_on_exit(self, coordinator: AreaOccupancyCoordinator):
        """Test that get_session closes session on exit."""
        db = coordinator.db

        with db.get_session() as session:
            session_id = id(session)

        # Session should be closed after context manager exits
        # We can't directly check if it's closed, but we can verify
        # that a new session is created each time
        with db.get_session() as session2:
            assert id(session2) != session_id


class TestTableProperties:
    """Test table property accessors."""

    def test_table_properties(self, coordinator: AreaOccupancyCoordinator):
        """Test table property accessors."""
        db = coordinator.db

        assert db.areas.name == "areas"
        assert db.entities.name == "entities"
        assert db.intervals.name == "intervals"
        assert db.priors.name == "priors"
        assert db.metadata.name == "metadata"

    def test_get_engine(self, coordinator: AreaOccupancyCoordinator):
        """Test get_engine method."""
        db = coordinator.db

        engine = db.get_engine()
        assert engine is not None
        assert engine == db.engine

    def test_update_session_maker(self, coordinator: AreaOccupancyCoordinator):
        """Test update_session_maker method."""
        db = coordinator.db

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

    def test_model_class_references(self, coordinator: AreaOccupancyCoordinator):
        """Test that model classes are correctly referenced."""
        db = coordinator.db

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
        assert db.Correlations is not None
        assert db.EntityStatistics is not None
        assert db.AreaRelationships is not None
        assert db.CrossAreaStats is not None


class TestDelegationCorrectness:
    """Test that core.py methods correctly delegate to underlying modules via __getattr__."""

    def test_delegated_methods_dict_exists(self, coordinator: AreaOccupancyCoordinator):
        """Test that _delegated_methods dictionary exists and contains expected methods."""
        db = coordinator.db

        assert hasattr(db, "_delegated_methods")
        assert isinstance(db._delegated_methods, dict)
        # Verify some expected methods are in the dictionary
        assert "load_data" in db._delegated_methods
        assert "save_area_data" in db._delegated_methods
        assert "check_database_integrity" in db._delegated_methods
        assert "get_area_data" in db._delegated_methods

    def test_nonexistent_attribute_raises_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that accessing non-existent attributes raises AttributeError."""
        db = coordinator.db

        with pytest.raises(
            AttributeError, match="has no attribute 'nonexistent_method'"
        ):
            _ = db.nonexistent_method

    @pytest.mark.parametrize(
        ("method_name", "module_path", "call_args", "return_value"),
        [
            (
                "save_area_data",
                "custom_components.area_occupancy.db.operations.save_area_data",
                ("Test Area",),
                None,
            ),
            (
                "get_area_data",
                "custom_components.area_occupancy.db.queries.get_area_data",
                ("test_entry",),
                {"entry_id": "test"},
            ),
            (
                "check_database_integrity",
                "custom_components.area_occupancy.db.maintenance.check_database_integrity",
                (),
                True,
            ),
            (
                "is_intervals_empty",
                "custom_components.area_occupancy.db.utils.is_intervals_empty",
                (),
                True,
            ),
            (
                "aggregate_raw_to_daily",
                "custom_components.area_occupancy.db.aggregation.aggregate_raw_to_daily",
                ("Test Area",),
                5,
            ),
            (
                "save_area_relationship",
                "custom_components.area_occupancy.db.relationships.save_area_relationship",
                ("Area1", "Area2", "adjacent", 0.5),
                True,
            ),
            (
                "analyze_correlation",
                "custom_components.area_occupancy.db.correlation.analyze_correlation",
                ("Area1", "sensor.temp", 30, False, None),
                {"correlation": 0.8},
            ),
        ],
    )
    def test_delegated_methods_via_getattr(
        self,
        coordinator: AreaOccupancyCoordinator,
        method_name: str,
        module_path: str,
        call_args: tuple,
        return_value: Any,
    ):
        """Test that delegated methods correctly call their underlying module functions via __getattr__."""
        db = coordinator.db

        # Replace function in _delegated_methods with mock to test delegation
        original_func = db._delegated_methods[method_name]
        with patch(module_path, return_value=return_value) as mock_func:
            db._delegated_methods[method_name] = mock_func
            result = getattr(db, method_name)(*call_args)
            # Verify it was called with db as first argument followed by call_args
            mock_func.assert_called_once_with(db, *call_args)
            if return_value is not None:
                assert result == return_value
            db._delegated_methods[method_name] = original_func

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method_name", "module_path"),
        [
            ("load_data", "custom_components.area_occupancy.db.operations.load_data"),
            ("sync_states", "custom_components.area_occupancy.db.sync.sync_states"),
        ],
    )
    async def test_async_delegated_methods_via_getattr(
        self,
        coordinator: AreaOccupancyCoordinator,
        method_name: str,
        module_path: str,
    ):
        """Test that async delegated methods correctly call their underlying module functions via __getattr__."""
        db = coordinator.db

        # Replace function in _delegated_methods with mock to test delegation
        original_func = db._delegated_methods[method_name]
        with patch(module_path, return_value=None) as mock_func:
            db._delegated_methods[method_name] = mock_func
            await getattr(db, method_name)()
            mock_func.assert_called_once_with(db)
            db._delegated_methods[method_name] = original_func

    def test_is_valid_state_explicit_method(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that is_valid_state is an explicit method (doesn't follow func(db, ...) pattern)."""
        db = coordinator.db

        # is_valid_state is an explicit method (doesn't follow func(db, ...) pattern)
        assert "is_valid_state" not in db._delegated_methods
        with patch(
            "custom_components.area_occupancy.db.utils.is_valid_state",
            return_value=True,
        ) as mock_is_valid:
            result = db.is_valid_state("on")
            mock_is_valid.assert_called_once_with("on")
            assert result is True

    def test_explicit_methods_not_delegated(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that methods with added logic (get_time_prior, get_occupied_intervals) are not delegated."""
        db = coordinator.db

        # These methods should exist as real methods, not in _delegated_methods
        assert "get_time_prior" not in db._delegated_methods
        assert "get_occupied_intervals" not in db._delegated_methods
        # Verify they exist as real methods
        assert hasattr(db, "get_time_prior")
        assert hasattr(db, "get_occupied_intervals")
        assert callable(db.get_time_prior)
        assert callable(db.get_occupied_intervals)


class TestErrorHandling:
    """Test error handling at core level."""

    def test_get_session_handles_exception(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that get_session properly handles exceptions."""
        db = coordinator.db

        def failing_maker():
            raise RuntimeError("Session creation failed")

        db._session_maker = failing_maker

        with (
            pytest.raises(RuntimeError, match="Session creation failed"),
            db.get_session(),
        ):
            pass
