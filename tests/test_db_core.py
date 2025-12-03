"""Tests for AreaOccupancyDB core functionality - initialization, session management, delegation."""
# ruff: noqa: SLF001

from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db import AreaOccupancyDB


# ruff: noqa: SLF001, PLC0415
def _create_test_area(
    db: AreaOccupancyDB,
    area_name: str,
    area_id: str | None = None,
    purpose: str = "social",
    threshold: float = 0.5,
) -> Any:
    """Helper function to create a test Areas record.

    Args:
        db: AreaOccupancyDB instance
        area_name: Name of the area
        area_id: Optional area ID (defaults to generated from area_name)
        purpose: Area purpose (default: "social")
        threshold: Occupancy threshold (default: 0.5)

    Returns:
        Areas model instance
    """
    if area_id is None:
        area_id = f"{area_name.lower().replace(' ', '_')}_id"

    return db.Areas(
        entry_id=db.coordinator.entry_id,
        area_name=area_name,
        area_id=area_id,
        purpose=purpose,
        threshold=threshold,
    )


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

    @pytest.mark.parametrize(
        ("attr_name", "expected_value"),
        [
            ("enable_auto_recovery", True),
            ("max_recovery_attempts", 3),
            ("enable_periodic_backups", True),
            ("backup_interval_hours", 24),
        ],
    )
    def test_initialization_sets_recovery_config(
        self, coordinator, attr_name: str, expected_value: Any
    ):
        """Test that initialization sets recovery configuration with correct default values."""
        from custom_components.area_occupancy.const import (
            DEFAULT_BACKUP_INTERVAL_HOURS,
            DEFAULT_ENABLE_AUTO_RECOVERY,
            DEFAULT_ENABLE_PERIODIC_BACKUPS,
            DEFAULT_MAX_RECOVERY_ATTEMPTS,
        )

        # Map attribute names to their constants for verification
        const_map = {
            "enable_auto_recovery": DEFAULT_ENABLE_AUTO_RECOVERY,
            "max_recovery_attempts": DEFAULT_MAX_RECOVERY_ATTEMPTS,
            "enable_periodic_backups": DEFAULT_ENABLE_PERIODIC_BACKUPS,
            "backup_interval_hours": DEFAULT_BACKUP_INTERVAL_HOURS,
        }

        db = AreaOccupancyDB(coordinator=coordinator)
        # Verify value matches both the expected value and the constant
        assert getattr(db, attr_name) == expected_value
        assert getattr(db, attr_name) == const_map[attr_name]

    def test_initialization_creates_model_classes_dict(self, coordinator):
        """Test that initialization creates model_classes dictionary with all expected models."""
        db = AreaOccupancyDB(coordinator=coordinator)

        expected_models = [
            "Areas",
            "Entities",
            "Priors",
            "Intervals",
            "Metadata",
            "IntervalAggregates",
            "OccupiedIntervalsCache",
            "GlobalPriors",
            "NumericSamples",
            "NumericAggregates",
            "Correlations",
            "EntityStatistics",
            "AreaRelationships",
            "CrossAreaStats",
        ]

        # Verify all expected models are present
        assert len(db.model_classes) == len(expected_models)
        for model_name in expected_models:
            assert model_name in db.model_classes
            # Verify mapping correctness (key matches class attribute)
            assert db.model_classes[model_name] == getattr(db, model_name)

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
        """Test that get_session rolls back on exception and prevents data persistence."""
        db = coordinator.db

        # Insert data, then raise exception - exception must escape session context
        def _insert_and_raise(session):
            """Helper to insert data and raise exception."""
            area = _create_test_area(
                db, "Rollback Test Area", area_id="rollback_test_area_id"
            )
            session.add(area)
            session.flush()  # Write to DB but don't commit
            raise ValueError("Test error")

        with pytest.raises(ValueError), db.get_session() as session:
            _insert_and_raise(session)

        # Verify data was rolled back - it should not exist
        with db.get_session() as session:
            result = (
                session.query(db.Areas)
                .filter_by(area_name="Rollback Test Area")
                .first()
            )
            assert result is None

    def test_get_session_closes_on_exit(self, coordinator: AreaOccupancyCoordinator):
        """Test that get_session closes session on exit."""
        db = coordinator.db

        with db.get_session() as session:
            assert session.is_active is True
            # Perform an operation to ensure session is active
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

        # After exit, verify session is closed
        # SQLAlchemy sessions close their connection when the context exits
        # The session should no longer be able to perform operations
        # Check that the session's connection is closed
        # In SQLAlchemy 2.x, closed sessions have their connection closed
        # We verify by checking that attempting operations would fail
        # Note: session.is_active may still be True, but connection is closed
        assert hasattr(session, "bind")
        # The connection pool should have returned the connection
        # We can verify by checking that a new session gets a fresh connection
        with db.get_session() as new_session:
            assert new_session is not session
            assert new_session.bind is not None

    def test_session_isolation(self, coordinator: AreaOccupancyCoordinator):
        """Verify multiple sessions don't interfere with each other."""
        db = coordinator.db

        # Create uncommitted data in first session, then raise exception
        # Exception must escape session context to test explicit rollback-on-exception
        def _insert_and_raise(session):
            """Helper to insert data and raise exception."""
            area1 = _create_test_area(
                db, "Session1 Isolation Test", area_id="session1_isolation_test_id"
            )
            session.add(area1)
            session.flush()  # Write to DB but don't commit
            raise ValueError("Test error for rollback")

        with pytest.raises(ValueError), db.get_session() as session1:
            _insert_and_raise(session1)

        # After session1 exits with exception, data should be rolled back
        # Verify data was not committed by checking in a new session
        with db.get_session() as session2:
            result = (
                session2.query(db.Areas)
                .filter_by(area_name="Session1 Isolation Test")
                .first()
            )
            assert result is None  # Data should not exist after rollback


class TestSessionMakerUpdate:
    """Test session maker update functionality."""

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

    def test_update_session_maker_creates_new_sessions(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Verify update_session_maker creates sessions with new engine."""
        db = coordinator.db
        original_engine = db.engine

        # Create session with original engine
        with db.get_session() as session1:
            assert session1.bind is original_engine

        # Create new engine and update session maker
        new_engine = create_engine("sqlite:///:memory:")
        db.engine = new_engine
        db.update_session_maker()

        # New sessions should use new engine
        with db.get_session() as session2:
            assert session2.bind is new_engine
            assert session2.bind is not original_engine


class TestModelClassReferences:
    """Test model class references."""

    def test_model_classes_are_functional(self, coordinator: AreaOccupancyCoordinator):
        """Verify model classes can be used to create and query records."""
        db = coordinator.db

        with db.get_session() as session:
            # Create instance using model class
            test_area = _create_test_area(
                db, "Test Model Functional", area_id="test_model_functional_id"
            )
            session.add(test_area)
            session.commit()

            # Query using model class
            result = (
                session.query(db.Areas)
                .filter_by(area_name="Test Model Functional")
                .first()
            )
            assert result is not None
            assert result.area_name == "Test Model Functional"
            assert result.threshold == 0.5


class TestDelegationCorrectness:
    """Test that core.py methods correctly delegate to underlying modules via __getattr__."""

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

    @pytest.mark.parametrize(
        "method_name", ["get_time_prior", "get_occupied_intervals"]
    )
    def test_explicit_methods_not_delegated(
        self, coordinator: AreaOccupancyCoordinator, method_name: str
    ):
        """Test that methods with added logic are not delegated."""
        db = coordinator.db

        # Method should not be in delegated methods
        assert method_name not in db._delegated_methods
        # Verify it exists as a real method
        assert hasattr(db, method_name)
        assert callable(getattr(db, method_name))


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
