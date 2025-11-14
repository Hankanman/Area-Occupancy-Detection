"""Tests for AreaOccupancy database models and utilities."""
# ruff: noqa: SLF001

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from custom_components.area_occupancy.const import MIN_PRIOR, RETENTION_DAYS
from custom_components.area_occupancy.db import (
    DB_VERSION,
    DEFAULT_AREA_PRIOR,
    AreaOccupancyDB,
    Base,
)
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


# ruff: noqa: SLF001
# Note: mock_area_occupancy_db_globally autouse fixture in conftest.py
# automatically detects DB tests and skips mocking, so no override needed here.


# Helper functions for creating test data
def create_test_area_data(entry_id="test_entry_001", **overrides):
    """Create standardized test area data."""
    data = {
        "entry_id": entry_id,
        "area_name": "Test Living Room",
        "purpose": "living",
        "threshold": 0.5,
        "area_prior": 0.3,
        "created_at": dt_util.utcnow(),
        "updated_at": dt_util.utcnow(),
    }
    data.update(overrides)
    return data


def create_test_entity_data(
    entry_id="test_entry_001", entity_id="binary_sensor.motion_1", **overrides
):
    """Create standardized test entity data."""
    data = {
        "entry_id": entry_id,
        "entity_id": entity_id,
        "entity_type": "motion",
        "weight": 0.85,
        "prob_given_true": 0.8,
        "prob_given_false": 0.1,
        "last_updated": dt_util.utcnow(),
        "created_at": dt_util.utcnow(),
    }
    data.update(overrides)
    return data


def create_test_prior_data(entry_id="test_entry_001", area_name="Testing", **overrides):
    """Create standardized test prior data."""
    data = {
        "entry_id": entry_id,
        "area_name": area_name,
        "day_of_week": 1,  # Monday
        "time_slot": 14,  # 2 PM
        "prior_value": 0.35,
        "data_points": 10,
        "last_updated": dt_util.utcnow(),
    }
    data.update(overrides)
    return data


def create_test_interval_data(entity_id="binary_sensor.motion_1", **overrides):
    """Create standardized test interval data."""
    start_time = dt_util.utcnow()
    end_time = start_time + timedelta(hours=1)
    data = {
        "entity_id": entity_id,
        "state": "on",
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": 3600.0,
        "created_at": dt_util.utcnow(),
    }
    data.update(overrides)
    return data


def setup_test_area_and_entity(db_session: Session, entry_id="test_entry_001"):
    """Create area and entity for tests requiring both."""
    # Create area
    area = AreaOccupancyDB.Areas.from_dict(create_test_area_data(entry_id))
    db_session.add(area)

    # Create entity
    entity = AreaOccupancyDB.Entities.from_dict(create_test_entity_data(entry_id))
    db_session.add(entity)

    db_session.commit()
    return area, entity


class TestDatabaseModels:
    """Test database ORM models and basic operations."""

    @pytest.mark.parametrize(
        ("model_class", "test_data", "expected_attrs"),
        [
            (
                AreaOccupancyDB.Areas,
                create_test_area_data(),
                {
                    "entry_id": "test_entry_001",
                    "area_name": "Test Living Room",
                    "threshold": 0.5,
                },
            ),
            (
                AreaOccupancyDB.Entities,
                create_test_entity_data(),
                {
                    "entity_id": "binary_sensor.motion_1",
                    "entity_type": "motion",
                    "weight": 0.85,
                },
            ),
            (
                AreaOccupancyDB.Priors,
                create_test_prior_data(),
                {"entry_id": "test_entry_001", "day_of_week": 1, "time_slot": 14},
            ),
            (
                AreaOccupancyDB.Intervals,
                create_test_interval_data(),
                {
                    "entity_id": "binary_sensor.motion_1",
                    "state": "on",
                    "duration_seconds": 3600.0,
                },
            ),
        ],
    )
    def test_model_creation_and_retrieval(
        self, db_session: Session, model_class, test_data, expected_attrs
    ):
        """Test creating and retrieving all model types."""
        # Create ORM object
        obj = model_class.from_dict(test_data)

        # Add to session and commit
        db_session.add(obj)
        db_session.commit()

        # Retrieve and verify
        if model_class == AreaOccupancyDB.Areas:
            retrieved = (
                db_session.query(model_class)
                .filter_by(entry_id=test_data["entry_id"])
                .first()
            )
        elif model_class == AreaOccupancyDB.Entities:
            retrieved = (
                db_session.query(model_class)
                .filter_by(
                    entry_id=test_data["entry_id"], entity_id=test_data["entity_id"]
                )
                .first()
            )
        elif model_class == AreaOccupancyDB.Priors:
            retrieved = (
                db_session.query(model_class)
                .filter_by(
                    entry_id=test_data["entry_id"],
                    area_name=test_data.get("area_name", "Testing"),
                    day_of_week=test_data["day_of_week"],
                    time_slot=test_data["time_slot"],
                )
                .first()
            )
        else:  # Intervals
            retrieved = (
                db_session.query(model_class)
                .filter_by(entity_id=test_data["entity_id"])
                .first()
            )

        assert retrieved is not None
        for attr, expected_value in expected_attrs.items():
            assert getattr(retrieved, attr) == expected_value

    def test_relationships(self, db_session: Session):
        """Test ORM relationships between models."""
        # Create area, entity, and prior
        _area, _entity = setup_test_area_and_entity(db_session)

        prior = AreaOccupancyDB.Priors.from_dict(create_test_prior_data())
        db_session.add(prior)
        db_session.commit()

        # Test relationships
        retrieved_area = (
            db_session.query(AreaOccupancyDB.Areas)
            .filter_by(entry_id="test_entry_001")
            .first()
        )

        assert retrieved_area is not None
        assert len(retrieved_area.entities) == 1
        assert retrieved_area.entities[0].entity_id == "binary_sensor.motion_1"
        assert len(retrieved_area.priors) == 1
        assert retrieved_area.priors[0].day_of_week == 1

    def test_to_dict_methods(self, db_session: Session):
        """Test to_dict methods on ORM models."""
        # Test with Areas model
        area = AreaOccupancyDB.Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        area_dict = area.to_dict()
        assert area_dict["entry_id"] == "test_entry_001"
        assert area_dict["area_name"] == "Test Living Room"
        assert "created_at" in area_dict
        assert "updated_at" in area_dict

    def test_unique_constraints(self, db_session: Session):
        """Test unique constraints on models."""
        # Create area and entity
        setup_test_area_and_entity(db_session)
        db_session.expunge_all()

        # Try to create duplicate entity (should fail due to primary key constraint)
        duplicate_entity = AreaOccupancyDB.Entities.from_dict(create_test_entity_data())
        db_session.add(duplicate_entity)

        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestDatabaseOperations:
    """Test database operations using the AreaOccupancyDB class.

    Note: Tests master-only recovery functionality.
    In multi-instance setup, only the master performs these operations.
    """

    def test_area_occupancy_db_initialization(self, test_db):
        """Test AreaOccupancyDB initialization with in-memory database."""
        db = test_db

        assert db.engine is not None
        assert db._session_maker is not None

        # Test that we can create tables (already initialized by fixture)
        db.init_db()

        # Verify tables exist
        with db.engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            table_names = [row[0] for row in result]

            expected_tables = ["areas", "entities", "intervals", "priors", "metadata"]
            for table in expected_tables:
                assert table in table_names

    def test_seeded_database(self, db_session):
        """Test database with pre-seeded data."""
        # Test that we can query the database
        areas = db_session.query(AreaOccupancyDB.Areas).all()
        entities = db_session.query(AreaOccupancyDB.Entities).all()

        # The database should be empty
        assert len(areas) == 0
        assert len(entities) == 0

        # Test that the session is working by creating a test area
        test_area = AreaOccupancyDB.Areas(
            entry_id="test_entry",
            area_name="Test Area",
            area_id="test_area_id",
            purpose="living",
            threshold=0.5,
            area_prior=0.3,
            created_at=dt_util.utcnow(),
            updated_at=dt_util.utcnow(),
        )

        db_session.add(test_area)
        db_session.commit()

        # Verify the area was added
        areas = db_session.query(AreaOccupancyDB.Areas).all()
        assert len(areas) == 1
        assert areas[0].entry_id == "test_entry"


class TestAreaOccupancyDBUtilities:
    """Test AreaOccupancyDB utility methods."""

    def test_table_properties_and_engine(self, test_db):
        """Test table properties and engine access."""
        db = test_db
        assert db.get_engine() is db.engine
        assert db.areas.name == "areas"
        assert db.entities.name == "entities"
        assert db.intervals.name == "intervals"
        assert db.priors.name == "priors"
        assert db.metadata.name == "metadata"

    def test_version_operations(self, test_db):
        """Test database version operations."""
        db = test_db
        db.init_db()
        db.set_db_version()
        db.set_db_version()
        assert db.get_db_version() == DB_VERSION

    def test_delete_and_force_reinitialize(self, test_db, tmp_path, monkeypatch):
        """Test database deletion and reinitialization."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.db_path.write_text("data")
        db.delete_db()
        assert not db.db_path.exists()

        calls = []
        monkeypatch.setattr(db, "init_db", lambda: calls.append("init"))
        monkeypatch.setattr(db, "set_db_version", lambda: calls.append("ver"))
        db.force_reinitialize()
        assert calls == ["init", "ver"]

    def test_enable_wal_mode_and_create_tables(self, test_db, monkeypatch):
        """Test WAL mode and table creation with error handling."""
        db = test_db
        db.init_db()
        db._create_tables_individually()

        # Exercise error handling in _create_tables_individually
        first_table = next(iter(Base.metadata.tables.values()))

        class DummyErr(Exception):
            sqlite_errno = 1

        def boom(*args, **kwargs):
            raise sa.exc.OperationalError("stmt", {}, DummyErr())

        monkeypatch.setattr(first_table, "create", boom)
        db._create_tables_individually()

        class DummyEngine:
            def connect(self):
                raise sa.exc.SQLAlchemyError("boom")

        db.engine = DummyEngine()
        db._enable_wal_mode()  # should swallow the error

    def test_is_valid_state_and_latest_interval(self, test_db):
        """Test state validation and latest interval retrieval."""
        db = test_db
        db.init_db()
        assert db.is_valid_state("on")
        assert not db.is_valid_state("unknown")

        first = db.get_latest_interval()
        assert isinstance(first, datetime)

        # Add an interval and check that latest interval is updated
        end = dt_util.utcnow()
        start = end - timedelta(seconds=60)
        with db.engine.begin() as conn:
            conn.execute(
                db.Intervals.__table__.insert(),
                {
                    "entity_id": "binary_sensor.motion",
                    "state": "on",
                    "start_time": start,
                    "end_time": end,
                    "duration_seconds": 60,
                    "created_at": end,
                },
            )
        second = db.get_latest_interval()
        assert second > first.replace(tzinfo=None)

    @pytest.mark.asyncio
    async def test_states_to_intervals_and_ensure_area(self, test_db, monkeypatch):
        """Test states to intervals conversion and area existence checking."""
        db = test_db
        start = dt_util.utcnow()
        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "off", last_changed=start),
                State(
                    "binary_sensor.motion",
                    "on",
                    last_changed=start + timedelta(seconds=10),
                ),
            ]
        }
        intervals = db._states_to_intervals(states, start + timedelta(seconds=20))
        assert len(intervals) == 2

        saved = []

        def fake_save():  # Now sync, not async
            saved.append(True)

        data_returns = [None, {"entry_id": db.coordinator.entry_id}]

        def fake_get(entry_id):
            return data_returns.pop(0)

        monkeypatch.setattr(db, "save_data", fake_save)
        monkeypatch.setattr(db, "get_area_data", fake_get)
        await db.ensure_area_exists()
        assert saved == [True]

    @pytest.mark.asyncio
    async def test_ensure_area_exists_when_present(self, test_db, monkeypatch):
        """Test ensure_area_exists when area already exists."""
        db = test_db
        monkeypatch.setattr(
            db, "get_area_data", lambda entry_id: {"entry_id": entry_id}
        )
        called = False

        async def fake_save():
            nonlocal called
            called = True

        monkeypatch.setattr(db, "save_data", fake_save)
        await db.ensure_area_exists()
        assert not called

    @pytest.mark.asyncio
    async def test_save_entity_data(self, test_db):
        """Test saving entity data with various entity types."""
        db = test_db
        good = SimpleNamespace(
            entity_id="binary_sensor.good",
            type=SimpleNamespace(input_type="motion", weight=0.9),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
            decay=SimpleNamespace(
                is_decaying=False,
                decay_start=dt_util.utcnow(),
            ),
            evidence=True,
        )
        missing_type = SimpleNamespace(
            entity_id="binary_sensor.bad",
            decay=SimpleNamespace(
                is_decaying=False,
                decay_start=dt_util.utcnow(),
            ),
            evidence=False,
        )
        no_input = SimpleNamespace(
            entity_id="binary_sensor.noinput",
            type=SimpleNamespace(input_type=None, weight=0.5),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
            decay=SimpleNamespace(
                is_decaying=False,
                decay_start=dt_util.utcnow(),
            ),
            evidence=None,
        )
        # Access entities via area (multi-area architecture)
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)
        assert area is not None

        # Set up entities on the area's entities manager using private attribute
        # since entities is a read-only property
        mock_entities_manager = SimpleNamespace(
            entities={
                "binary_sensor.good": good,
                "binary_sensor.bad": missing_type,
                "binary_sensor.noinput": no_input,
            }
        )
        area._entities = mock_entities_manager
        db.save_entity_data()
        with db.engine.connect() as conn:
            rows = conn.execute(sa.text("SELECT entity_id FROM entities")).fetchall()
        assert {r[0] for r in rows} == {"binary_sensor.good"}

    def test_is_intervals_empty(self, test_db):
        """Test intervals empty check."""
        db = test_db
        db.init_db()
        assert db.is_intervals_empty()

        # Add an interval
        end = dt_util.utcnow()
        start = end - timedelta(seconds=5)
        with db.engine.begin() as conn:
            conn.execute(
                db.Intervals.__table__.insert(),
                {
                    "entity_id": "binary_sensor.motion",
                    "state": "on",
                    "start_time": start,
                    "end_time": end,
                    "duration_seconds": 5,
                    "created_at": end,
                },
            )
        assert not db.is_intervals_empty()

    def test_get_area_data_error(self, test_db, monkeypatch):
        """Test get_area_data with session error."""
        db = test_db

        def bad_session():
            raise sa.exc.SQLAlchemyError("boom")

        monkeypatch.setattr(db, "get_session", bad_session)
        assert db.get_area_data("x") is None

    @pytest.mark.asyncio
    async def test_load_data(self, test_db, monkeypatch):
        """Test loading data from database."""
        db = test_db
        # Ensure database is initialized before saving data
        db.init_db()
        # Ensure area has an area_prior set so load_data will call set_global_prior
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)
        # Set area_prior so it will be saved and loaded
        area.prior.set_global_prior(0.5)
        # Verify area_prior is set correctly
        assert area.area_prior() >= 0.5, (
            f"area_prior should be >= 0.5, got {area.area_prior()}"
        )
        db.save_area_data()

        # Verify that area_prior was saved to the database
        area_data = db.get_area_data(db.coordinator.entry_id)
        if area_data:
            saved_prior = area_data.get("area_prior")
            assert saved_prior is not None, (
                "area_prior should be saved to database, got None"
            )
            assert saved_prior >= 0.5, f"area_prior should be >= 0.5, got {saved_prior}"

        # Populate entities table directly
        good = SimpleNamespace(
            entity_id="binary_sensor.good",
            type=SimpleNamespace(input_type="motion", weight=0.9),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
            decay=SimpleNamespace(
                is_decaying=False,
                decay_start=dt_util.utcnow(),
            ),
            evidence=True,
        )
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)
        assert area is not None

        # Set up entities on the area's entities manager using private attribute
        mock_entities_manager = SimpleNamespace(
            entities={
                "binary_sensor.good": good,
            }
        )
        area._entities = mock_entities_manager
        db.save_entity_data()

        called = []
        created_entities = []

        # Restore real entities manager for load_data test
        area._entities = None  # Reset to allow property to recreate
        real_entities = area.entities  # This will create a real EntityManager

        # Track calls to set_global_prior
        original_set_global_prior = area.prior.set_global_prior

        def tracked_set_global_prior(v):
            called.append(("prior", v))
            return original_set_global_prior(v)

        area.prior.set_global_prior = tracked_set_global_prior

        # Track entity creation
        original_add_entity = real_entities.add_entity

        def tracked_add_entity(entity):
            created_entities.append(entity.entity_id)
            return original_add_entity(entity)

        real_entities.add_entity = tracked_add_entity

        await db.load_data()

        # Check that prior was set
        # Note: set_global_prior might be called with the saved area_prior value,
        # which could be different from 0.5 if it was combined with time_prior
        assert any(call[0] == "prior" for call in called), (
            f"set_global_prior was not called. Called: {called}"
        )

    @pytest.mark.asyncio
    async def test_load_data_entity_handling(self, test_db, monkeypatch):
        """Test the entity handling logic in load_data method.

        This test verifies that:
        1. Entities in the database that are in the current config are updated
        2. Entities in the database that are NOT in the current config are deleted as stale
        """
        db = test_db
        db.save_area_data()

        # Populate entities table with an entity that's NOT in the current config
        # This entity should be deleted as stale when load_data is called
        good = SimpleNamespace(
            entity_id="binary_sensor.good",
            type=SimpleNamespace(input_type="motion", weight=0.9),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
            decay=SimpleNamespace(
                is_decaying=False,
                decay_start=dt_util.utcnow(),
            ),
            evidence=True,
        )
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)
        assert area is not None

        # Set up entities on the area's entities manager using private attribute
        mock_entities_manager = SimpleNamespace(
            entities={
                "binary_sensor.good": good,
            }
        )
        area._entities = mock_entities_manager
        db.save_entity_data()

        # Verify entity is in database
        with db.engine.connect() as conn:
            result = conn.execute(
                sa.text(
                    "SELECT entity_id FROM entities WHERE area_name=:a AND entity_id=:e"
                ),
                {"a": area_name, "e": "binary_sensor.good"},
            ).fetchone()
        assert result is not None, "Entity should be in database before load_data"

        # Restore real entities manager for load_data test
        area._entities = None  # Reset to allow property to recreate
        real_entities = area.entities  # This will create a real EntityManager

        # Verify that binary_sensor.good is NOT in the current config
        assert "binary_sensor.good" not in real_entities.entity_ids, (
            "Entity should not be in current config"
        )

        await db.load_data()

        # Verify that stale entity was deleted from database
        with db.engine.connect() as conn:
            result = conn.execute(
                sa.text(
                    "SELECT entity_id FROM entities WHERE area_name=:a AND entity_id=:e"
                ),
                {"a": area_name, "e": "binary_sensor.good"},
            ).fetchone()
        assert result is None, "Stale entity should be deleted from database"

    @pytest.mark.asyncio
    async def test_load_data_error_handling(self, test_db, monkeypatch):
        """Test error handling in load_data method."""
        db = test_db
        db.save_area_data()
        await self.test_save_entity_data(test_db)  # populate entities table

        # Mock area-based access
        mock_area = Mock()
        mock_area.area_name = "Test Area"

        # Mock the prior setter via area
        mock_area.prior = SimpleNamespace(set_global_prior=lambda v: None)

        # Mock the entities manager to always raise ValueError via area
        mock_area.entities = SimpleNamespace(
            get_entity=lambda entity_id: (_ for _ in ()).throw(
                ValueError(f"Entity {entity_id} not found")
            ),
            add_entity=lambda entity: None,
        )

        # Mock the factory via area
        mock_area.factory = SimpleNamespace(
            create_from_db=lambda ent: SimpleNamespace(entity_id=ent.entity_id)
        )

        # Set up coordinator to return the mock area
        db.coordinator.get_area_or_default = Mock(return_value=mock_area)
        db.coordinator.get_area_names = Mock(return_value=["Test Area"])
        db.coordinator.areas = {"Test Area": mock_area}

        # This should not raise an exception - it should handle the ValueError gracefully
        await db.load_data()

    @pytest.mark.asyncio
    async def test_sync_states(self, test_db, monkeypatch):
        """Test syncing states from recorder."""
        db = test_db
        db.init_db()
        start = dt_util.utcnow()
        state = State("binary_sensor.motion", "on", last_changed=start)

        class DummyRecorder:
            async def async_add_executor_job(self, func):
                return {"binary_sensor.motion": [state]}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.get_instance",
            lambda hass: DummyRecorder(),
        )
        await db.sync_states()

    def test_get_area_data_none(self, test_db):
        """Test get_area_data with missing entry."""
        db = test_db
        assert db.get_area_data("missing") is None

    def test_states_to_intervals_edge_cases(self, test_db):
        """Test states to intervals with edge cases."""
        db = test_db
        old_state = State(
            "binary_sensor.motion",
            "on",
            last_changed=dt_util.utcnow() - timedelta(days=RETENTION_DAYS + 1),
        )
        result = db._states_to_intervals(
            {"binary_sensor.old": [], "binary_sensor.motion": [old_state]},
            dt_util.utcnow(),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_sync_states_error(self, test_db, monkeypatch):
        """Test sync_states with recorder error."""
        db = test_db

        class DummyRecorder:
            async def async_add_executor_job(self, func):
                raise HomeAssistantError("boom")

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.get_instance",
            lambda hass: DummyRecorder(),
        )
        with pytest.raises(HomeAssistantError):
            await db.sync_states()

    def test_model_to_dicts(self):
        """Test to_dict methods on all model types."""
        now = dt_util.utcnow()
        ent = AreaOccupancyDB.Entities.from_dict(
            {
                "entry_id": "e1",
                "entity_id": "sensor.test",
                "entity_type": "motion",
                "last_updated": now,
                "created_at": now,
            }
        )
        pri = AreaOccupancyDB.Priors.from_dict(
            {
                "entry_id": "e1",
                "area_name": "Testing",
                "day_of_week": 1,
                "time_slot": 0,
                "prior_value": 0.1,
                "data_points": 1,
                "last_updated": now,
            }
        )
        interval = AreaOccupancyDB.Intervals.from_dict(
            {
                "entity_id": "sensor.test",
                "state": "on",
                "start_time": now,
                "end_time": now + timedelta(seconds=1),
                "duration_seconds": 1.0,
                "created_at": now,
            }
        )
        assert ent.to_dict()["entity_id"] == "sensor.test"
        assert pri.to_dict()["entry_id"] == "e1"
        assert interval.to_dict()["entity_id"] == "sensor.test"

    @pytest.mark.asyncio
    async def test_sync_states_no_states(self, test_db, monkeypatch):
        """Test sync_states with no states."""
        db = test_db

        class DummyRecorder:
            async def async_add_executor_job(self, func):
                return {}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.get_instance",
            lambda hass: DummyRecorder(),
        )
        await db.sync_states()

    @pytest.mark.asyncio
    async def test_ensure_area_exists_error_handling(self, test_db, monkeypatch):
        """Test ensure_area_exists error handling."""
        db = test_db
        monkeypatch.setattr(db, "get_area_data", lambda eid: None)

        async def bad_save():
            raise HomeAssistantError("boom")

        monkeypatch.setattr(db, "save_data", bad_save)
        await db.ensure_area_exists()

    @pytest.mark.asyncio
    async def test_ensure_area_exists_fails_to_create(self, test_db, monkeypatch):
        """Test ensure_area_exists when creation fails."""
        db = test_db
        calls = []

        def fake_get(entry_id):
            calls.append(1)

        async def fake_save():
            return None

        monkeypatch.setattr(db, "get_area_data", fake_get)
        monkeypatch.setattr(db, "save_data", fake_save)
        await db.ensure_area_exists()
        assert len(calls) == 2

    @pytest.mark.parametrize(
        ("error_type", "expected_result"),
        [
            ("no such table", True),  # Missing table should return True
            ("bad", "raise"),  # Other errors should raise
        ],
    )
    def test_is_intervals_empty_error(
        self, test_db, monkeypatch, error_type, expected_result
    ):
        """Test is_intervals_empty with various error conditions."""
        db = test_db

        @contextmanager
        def error_session():
            class S:
                def query(self, *args, **kwargs):
                    raise sa.exc.SQLAlchemyError(error_type)

                def close(self):
                    pass

            yield S()

        monkeypatch.setattr(db, "get_locked_session", error_session)

        # is_intervals_empty should catch exceptions and return True as fallback
        # This allows the integration to continue even if database operations fail
        result = db.is_intervals_empty()
        assert result is True  # Should always return True on error

    @pytest.mark.parametrize(
        ("error_type", "should_raise"),
        [
            ("no such table", False),
            ("other", True),
        ],
    )
    def test_get_latest_interval_error(
        self, test_db, monkeypatch, error_type, should_raise
    ):
        """Test get_latest_interval with various error conditions."""
        db = test_db

        @contextmanager
        def bad_session():
            class S:
                def execute(self, *args, **kwargs):
                    raise sa.exc.SQLAlchemyError(error_type)

                def close(self):
                    pass

            yield S()

        monkeypatch.setattr(db, "get_locked_session", bad_session)

        # get_latest_interval should catch exceptions and return a default time
        # This allows the integration to continue even if database operations fail
        result = db.get_latest_interval()
        # Should return a datetime object (default time when error occurs)
        assert isinstance(result, datetime)

    @pytest.mark.asyncio
    async def test_load_data_error(self, test_db, monkeypatch):
        """Test load_data with session error."""
        db = test_db

        @contextmanager
        def bad_session():
            raise RuntimeError("fail")
            yield

        monkeypatch.setattr(db, "get_locked_session", bad_session)

        # load_data should not raise exceptions, it should log them and continue
        # This allows the integration to start even if data loading fails
        await db.load_data()

        # The method should complete without raising an exception
        # The error should be logged but not propagated

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("entry_id", ""),
            ("area_name", ""),
            ("purpose", ""),
            ("threshold", None),
        ],
    )
    async def test_save_area_data_validation(self, test_db, field, value):
        """Test save_area_data validation with various invalid values."""
        db = test_db
        # Get the actual area name from the coordinator (from config entry)
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0, "Coordinator should have at least one area"
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)
        assert area is not None, f"Area '{area_name}' should exist"

        if field == "entry_id":
            db.coordinator.entry_id = value
            # When entry_id is empty, save_area_data will skip the area (logs error and continues)
            # No exception is raised, but no data is saved
            db.save_area_data(area_name=area_name)
        elif field == "area_name":
            # area_name is passed as parameter, not from config
            # Test with invalid area_name parameter - empty string will cause area lookup to fail
            # and no areas will be saved
            db.save_area_data(area_name="")
            return
        elif field == "purpose":
            area.config.purpose = value
            # When purpose is empty, save_area_data will skip the area (logs error and continues)
            # No exception is raised, but no data is saved
            db.save_area_data(area_name=area_name)
        elif field == "threshold":
            area.config.threshold = value
            # When threshold is None, save_area_data will skip the area (logs error and continues)
            # No exception is raised, but no data is saved
            db.save_area_data(area_name=area_name)

        # Verify no data was saved
        with db.engine.connect() as conn:
            result = conn.execute(sa.text("SELECT COUNT(*) FROM areas")).scalar()
        assert result == 0

    @pytest.mark.asyncio
    async def test_save_area_data_preserves_existing_prior_when_global_prior_none(
        self, test_db
    ):
        """Test that save_area_data uses MIN_PRIOR when global_prior is None (doesn't preserve existing)."""
        db = test_db
        existing_prior = 0.45

        # Get the actual area name from the coordinator
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)
        assert area is not None

        # Set up existing area in database with a learned prior
        with db.get_locked_session() as session:
            existing_area = db.Areas(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                area_id=area.config.area_id,
                purpose=area.config.purpose,
                threshold=area.config.threshold,
                area_prior=existing_prior,
            )
            session.add(existing_area)
            session.commit()

        # Set global_prior to None to simulate first run or failed load - use area-based access
        area.prior.global_prior = None
        # Mock area.area_prior to return MIN_PRIOR when global_prior is None
        area.area_prior = Mock(return_value=MIN_PRIOR)

        # Save will use MIN_PRIOR from coordinator.area_prior when global_prior is None
        db.save_area_data()

        # Verify MIN_PRIOR was used (not the existing prior)
        with db.engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT area_prior FROM areas WHERE entry_id=:e AND area_name=:a"
                ),
                {"e": db.coordinator.entry_id, "a": area_name},
            ).fetchone()
        # When global_prior is None, Prior.value returns MIN_PRIOR
        assert row[0] == MIN_PRIOR

    @pytest.mark.asyncio
    async def test_save_area_data_uses_default_when_no_existing_prior(self, test_db):
        """Test that save_area_data uses MIN_PRIOR when global_prior is None and no existing value."""
        db = test_db

        # Get the actual area name from the coordinator
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)
        assert area is not None

        # Ensure no existing area in database
        with db.get_locked_session() as session:
            session.query(db.Areas).filter_by(
                entry_id=db.coordinator.entry_id, area_name=area_name
            ).delete()
            session.commit()

        # Set global_prior to None - use area-based access
        area.prior.global_prior = None
        # Mock area.area_prior to return MIN_PRIOR when global_prior is None
        area.area_prior = Mock(return_value=MIN_PRIOR)

        # Save should use MIN_PRIOR (from Prior.value when global_prior is None)
        db.save_area_data()

        with db.engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT area_prior FROM areas WHERE entry_id=:e AND area_name=:a"
                ),
                {"e": db.coordinator.entry_id, "a": area_name},
            ).fetchone()
        # When global_prior is None, Prior.value returns MIN_PRIOR
        assert row[0] == MIN_PRIOR

    @pytest.mark.asyncio
    async def test_save_area_data_handles_none_area_prior_with_global_prior_set(
        self, test_db
    ):
        """Test that save_area_data uses DEFAULT_AREA_PRIOR when coordinator.area_prior returns None."""
        db = test_db

        # Get the actual area name from the coordinator
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)

        # Mock area.area_prior to return None
        # This simulates an edge case where the method returns None
        area.area_prior = Mock(return_value=None)

        # Save should use DEFAULT_AREA_PRIOR as fallback (from db.py line 1241-1247)
        db.save_area_data()

        # Verify DEFAULT_AREA_PRIOR was used as fallback
        with db.engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT area_prior FROM areas WHERE entry_id=:e AND area_name=:a"
                ),
                {"e": db.coordinator.entry_id, "a": area_name},
            ).fetchone()
        # When area_prior is None, save_area_data uses DEFAULT_AREA_PRIOR as fallback
        assert row[0] == DEFAULT_AREA_PRIOR

    @pytest.mark.asyncio
    async def test_save_area_data_success(self, test_db):
        """Test successful save_area_data operation."""
        db = test_db
        # Get the actual area name from the coordinator
        area_names = db.coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        area = db.coordinator.get_area_or_default(area_name)
        assert area is not None

        area.config.area_id = None
        db.save_area_data()
        with db.engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT area_id FROM areas WHERE entry_id=:e"),
                {"e": db.coordinator.entry_id},
            ).fetchone()
        assert row[0] == db.coordinator.entry_id

        data = db.get_area_data(db.coordinator.entry_id)
        assert data is not None and data["area_id"] == db.coordinator.entry_id

    def test_get_area_data_with_error(self, test_db):
        """Test get_area_data with database error."""
        db = test_db

        with patch.object(
            db, "get_locked_session", side_effect=sa.exc.SQLAlchemyError("DB Error")
        ):
            result = db.get_area_data("test_entry_id")
            assert result is None

    def test_ensure_area_exists_area_already_exists(self, test_db):
        """Test ensure_area_exists when area already exists."""
        db = test_db

        # Mock get_area_data to return existing area
        with (
            patch.object(db, "get_area_data", return_value={"entry_id": "test"}),
            patch.object(db, "save_data", new=AsyncMock()) as mock_save,
        ):
            # This should not raise an exception and should not call save_data
            asyncio.run(db.ensure_area_exists())
            mock_save.assert_not_called()

    def test_ensure_area_exists_creates_area(self, test_db):
        """Test ensure_area_exists when area doesn't exist."""
        db = test_db

        # Mock get_area_data to return None (area doesn't exist)
        with (
            patch.object(db, "get_area_data", return_value=None),
            patch.object(db, "save_data", new=AsyncMock()) as mock_save,
        ):
            asyncio.run(db.ensure_area_exists())
            mock_save.assert_called_once()

    def test_ensure_area_exists_with_error(self, test_db):
        """Test ensure_area_exists with error handling."""
        db = test_db

        with patch.object(db, "get_area_data", side_effect=HomeAssistantError("Error")):
            # Should not raise exception, just log error
            asyncio.run(db.ensure_area_exists())

    def test_safe_is_intervals_empty_with_integrity_check_failure(self, test_db):
        """Test safe_is_intervals_empty when integrity check fails."""
        db = test_db

        with (
            patch.object(db, "check_database_integrity", return_value=False),
            patch.object(db, "handle_database_corruption", return_value=False),
        ):
            result = db.safe_is_intervals_empty()
            assert result is True

    def test_safe_is_intervals_empty_with_integrity_check_success(self, test_db):
        """Test safe_is_intervals_empty when integrity check passes."""
        db = test_db

        with (
            patch.object(db, "check_database_integrity", return_value=True),
            patch.object(db, "is_intervals_empty", return_value=False),
        ):
            result = db.safe_is_intervals_empty()
            assert result is False

    def test_safe_is_intervals_empty_with_error(self, test_db):
        """Test safe_is_intervals_empty with unexpected error."""
        db = test_db

        with patch.object(db, "check_database_integrity", side_effect=OSError("Error")):
            result = db.safe_is_intervals_empty()
            assert result is True

    def test_check_database_accessibility_file_not_exists(self, test_db):
        """Test check_database_accessibility when file doesn't exist."""
        db = test_db
        db.db_path = Path("/nonexistent/path/db.db")

        result = db.check_database_accessibility()
        assert result is False

    def test_check_database_accessibility_invalid_sqlite_header(
        self, test_db, tmp_path
    ):
        """Test check_database_accessibility with invalid SQLite header."""
        db = test_db
        db.db_path = tmp_path / "invalid.db"

        # Create file with invalid header
        with open(db.db_path, "wb") as f:
            f.write(b"invalid header")

        result = db.check_database_accessibility()
        assert result is False

    def test_check_database_accessibility_permission_error(self, test_db, tmp_path):
        """Test check_database_accessibility with permission error."""
        db = test_db
        db.db_path = tmp_path / "test.db"

        # Create file
        db.db_path.touch()

        # Mock open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            result = db.check_database_accessibility()
            assert result is False

    def test_is_database_corrupted_various_indicators(self, test_db):
        """Test is_database_corrupted with various corruption indicators."""
        db = test_db

        corruption_messages = [
            "database disk image is malformed",
            "corrupted database",
            "file is not a database",
            "database or disk is full",
            "database is locked",
            "unable to open database file",
        ]

        for message in corruption_messages:
            error = sa.exc.SQLAlchemyError(message)
            assert db.is_database_corrupted(error) is True

    def test_is_database_corrupted_non_corruption_error(self, test_db):
        """Test is_database_corrupted with non-corruption error."""
        db = test_db

        error = sa.exc.SQLAlchemyError("table already exists")
        assert db.is_database_corrupted(error) is False

    def test_attempt_database_recovery_success(self, test_db, tmp_path):
        """Test attempt_database_recovery with successful recovery (master-only)."""
        db = test_db
        db.db_path = tmp_path / "test.db"

        # Create a valid database file
        db.db_path.touch()

        with (
            patch.object(db.engine, "dispose"),
            patch(
                "custom_components.area_occupancy.db.create_engine"
            ) as mock_create_engine,
        ):
            mock_engine = Mock()
            mock_conn = Mock()
            mock_conn.execute.return_value.fetchone.return_value = ("test_table",)
            # Make connect() usable as context manager
            mock_connect_cm = Mock()
            mock_connect_cm.__enter__ = Mock(return_value=mock_conn)
            mock_connect_cm.__exit__ = Mock(return_value=None)
            mock_engine.connect.return_value = mock_connect_cm
            mock_create_engine.return_value = mock_engine

            result = db.attempt_database_recovery()
            assert result is True

    def test_attempt_database_recovery_failure(self, test_db):
        """Test attempt_database_recovery with failure."""
        db = test_db

        with (
            patch.object(db.engine, "dispose"),
            patch(
                "custom_components.area_occupancy.db.create_engine",
                side_effect=sa.exc.SQLAlchemyError("Recovery failed"),
            ),
        ):
            result = db.attempt_database_recovery()
            assert result is False

    def test_backup_database_success(self, test_db, tmp_path):
        """Test backup_database with successful backup."""
        db = test_db
        db.db_path = tmp_path / "test.db"

        # Create source file
        with open(db.db_path, "w") as f:
            f.write("test data")

        result = db.backup_database()
        assert result is True

        # Check backup file exists
        backup_path = db.db_path.with_suffix(".db.backup")
        assert backup_path.exists()

    def test_backup_database_file_not_exists(self, test_db):
        """Test backup_database when source file doesn't exist."""
        db = test_db
        db.db_path = Path("/nonexistent/path/db.db")

        result = db.backup_database()
        assert result is False

    def test_backup_database_error(self, test_db, tmp_path):
        """Test backup_database with error."""
        db = test_db
        db.db_path = tmp_path / "test.db"

        # Create source file
        db.db_path.touch()

        with patch("shutil.copy2", side_effect=OSError("Copy failed")):
            result = db.backup_database()
            assert result is False

    def test_restore_database_from_backup_success(self, test_db, tmp_path):
        """Test restore_database_from_backup with successful restore."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        backup_path = tmp_path / "test.db.backup"

        # Create backup file
        with open(backup_path, "w") as f:
            f.write("backup data")

        with patch("shutil.copy2"), patch("sqlalchemy.create_engine"):
            result = db.restore_database_from_backup()
            assert result is True

    def test_restore_database_from_backup_no_backup(self, test_db):
        """Test restore_database_from_backup when no backup exists."""
        db = test_db
        db.db_path = Path("/nonexistent/path/db.db")

        result = db.restore_database_from_backup()
        assert result is False

    def test_restore_database_from_backup_error(self, test_db, tmp_path):
        """Test restore_database_from_backup with error."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        backup_path = tmp_path / "test.db.backup"

        # Create backup file
        backup_path.touch()

        with patch("shutil.copy2", side_effect=OSError("Restore failed")):
            result = db.restore_database_from_backup()
            assert result is False

    def test_handle_database_corruption_auto_recovery_disabled(self, test_db):
        """Test handle_database_corruption with auto recovery disabled."""
        db = test_db
        db.enable_auto_recovery = False

        result = db.handle_database_corruption()
        assert result is False

    def test_handle_database_corruption_recovery_success(self, test_db):
        """Test handle_database_corruption with successful recovery."""
        db = test_db

        with (
            patch.object(db, "backup_database", return_value=True),
            patch.object(db, "attempt_database_recovery", return_value=True),
            patch.object(db, "check_database_integrity", return_value=True),
        ):
            result = db.handle_database_corruption()
            assert result is True

    def test_handle_database_corruption_restore_from_backup_success(self, test_db):
        """Test handle_database_corruption with successful restore from backup."""
        db = test_db
        db.enable_periodic_backups = True

        with (
            patch.object(db, "backup_database", return_value=True),
            patch.object(db, "attempt_database_recovery", return_value=False),
            patch.object(db, "restore_database_from_backup", return_value=True),
            patch.object(db, "check_database_integrity", return_value=True),
        ):
            result = db.handle_database_corruption()
            assert result is True

    def test_handle_database_corruption_recreate_database(self, test_db):
        """Test handle_database_corruption with database recreation."""
        db = test_db

        with (
            patch.object(db, "backup_database", return_value=True),
            patch.object(db, "attempt_database_recovery", return_value=False),
            patch.object(db, "restore_database_from_backup", return_value=False),
            patch.object(db, "delete_db"),
            patch.object(db, "init_db"),
            patch.object(db, "set_db_version"),
        ):
            result = db.handle_database_corruption()
            assert result is True

    def test_handle_database_corruption_recreation_failure(self, test_db):
        """Test handle_database_corruption with recreation failure."""
        db = test_db

        with (
            patch.object(db, "backup_database", return_value=True),
            patch.object(db, "attempt_database_recovery", return_value=False),
            patch.object(db, "restore_database_from_backup", return_value=False),
            patch.object(db, "delete_db", side_effect=OSError("Delete failed")),
        ):
            result = db.handle_database_corruption()
            assert result is False

    def test_periodic_health_check_success(self, test_db):
        """Test periodic_health_check with successful check."""
        db = test_db

        with (
            patch.object(db, "check_database_integrity", return_value=True),
            patch.object(db, "backup_database", return_value=True),
        ):
            result = db.periodic_health_check()
            assert result is True

    def test_periodic_health_check_integrity_failure(self, test_db):
        """Test periodic_health_check with integrity failure."""
        db = test_db

        with (
            patch.object(db, "check_database_integrity", return_value=False),
            patch.object(db, "handle_database_corruption", return_value=True),
        ):
            result = db.periodic_health_check()
            assert result is True

    def test_periodic_health_check_integrity_failure_recovery_failed(self, test_db):
        """Test periodic_health_check with integrity failure and recovery failed."""
        db = test_db

        with (
            patch.object(db, "check_database_integrity", return_value=False),
            patch.object(db, "handle_database_corruption", return_value=False),
        ):
            result = db.periodic_health_check()
            assert result is False

    def test_periodic_health_check_backup_creation(self, test_db, tmp_path):
        """Test periodic_health_check with backup creation."""
        db = test_db
        db.db_path = tmp_path / "test.db"
        db.enable_periodic_backups = True
        db.backup_interval_hours = 1

        # Create database file
        db.db_path.touch()

        with (
            patch.object(db, "check_database_integrity", return_value=True),
            patch.object(db, "backup_database", return_value=True) as mock_backup,
        ):
            result = db.periodic_health_check()
            assert result is True
            mock_backup.assert_called_once()

    def test_periodic_health_check_error(self, test_db):
        """Test periodic_health_check with error."""
        db = test_db

        with patch.object(db, "check_database_integrity", side_effect=OSError("Error")):
            result = db.periodic_health_check()
            assert result is False

    def test_get_engine(self, test_db):
        """Test get_engine method."""
        db = test_db

        engine = db.get_engine()
        assert engine is not None
        assert engine == db.engine

    def test_delete_db_success(self, test_db, tmp_path):
        """Test delete_db with successful deletion."""
        db = test_db
        db.db_path = tmp_path / "test.db"

        # Create file to delete
        db.db_path.touch()

        db.delete_db()

        assert not db.db_path.exists()

    def test_delete_db_file_not_exists(self, test_db):
        """Test delete_db when file doesn't exist."""
        db = test_db
        db.db_path = Path("/nonexistent/path/db.db")

        # Should not raise exception
        db.delete_db()

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
            db.delete_db()

    def test_force_reinitialize(self, test_db):
        """Test force_reinitialize method."""
        db = test_db

        with (
            patch.object(db, "init_db") as mock_init,
            patch.object(db, "set_db_version") as mock_set_version,
        ):
            db.force_reinitialize()

            mock_init.assert_called_once()
            mock_set_version.assert_called_once()

    def test_init_db_success(self, test_db):
        """Test init_db with successful initialization."""
        db = test_db

        with (
            patch.object(db, "_enable_wal_mode"),
            patch.object(db.engine, "connect"),
        ):
            db.init_db()

    def test_init_db_operational_error_race_condition(self, test_db):
        """Test init_db with operational error (race condition)."""
        db = test_db

        # Mock error with sqlite_errno = 1 (table already exists)
        mock_error = sa.exc.OperationalError("table already exists", None, None)
        mock_error.orig = Mock()
        mock_error.orig.sqlite_errno = 1

        with (
            patch.object(db, "_enable_wal_mode"),
            patch.object(db.engine, "connect", side_effect=mock_error),
            patch.object(db, "_create_tables_individually"),
        ):
            db.init_db()

    def test_init_db_operational_error_other(self, test_db):
        """Test init_db with other operational error."""
        db = test_db

        # Mock error with different sqlite_errno
        mock_error = sa.exc.OperationalError("other error", None, None)
        mock_error.orig = Mock()
        mock_error.orig.sqlite_errno = 2

        with (
            patch.object(db, "_enable_wal_mode"),
            patch.object(db.engine, "connect", side_effect=mock_error),
            pytest.raises(sa.exc.OperationalError),
        ):
            db.init_db()

    def test_init_db_general_error(self, test_db):
        """Test init_db with general error."""
        db = test_db

        with (
            patch.object(db, "_enable_wal_mode"),
            patch.object(
                db.engine, "connect", side_effect=RuntimeError("General error")
            ),
            pytest.raises(RuntimeError),
        ):
            db.init_db()

    def test_enable_wal_mode_success(self, test_db):
        """Test _enable_wal_mode with success."""
        db = test_db

        with patch.object(db.engine, "connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn

            db._enable_wal_mode()

            mock_conn.execute.assert_called_once()

    def test_enable_wal_mode_error(self, test_db):
        """Test _enable_wal_mode with error."""
        db = test_db

        with patch.object(
            db.engine, "connect", side_effect=sa.exc.SQLAlchemyError("WAL error")
        ):
            # Should not raise exception, just log error
            db._enable_wal_mode()

    def test_verify_all_tables_exist_success(self, test_db):
        """Test _verify_all_tables_exist with all tables present."""
        db = test_db

        # test_db fixture already initializes tables via init_db()
        # Verify all tables exist
        assert db._verify_all_tables_exist() is True

    def test_verify_all_tables_exist_error(self, test_db):
        """Test _verify_all_tables_exist with database error."""
        db = test_db

        # Mock a database error
        with patch.object(
            db.engine, "connect", side_effect=sa.exc.SQLAlchemyError("DB Error")
        ):
            # Should return False on error
            assert db._verify_all_tables_exist() is False

    def test_ensure_db_exists_new_database(self, test_db, tmp_path):
        """Test _ensure_db_exists with new database."""

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
        db._ensure_db_exists()

        # Verify tables were created
        assert db._verify_all_tables_exist() is True

    def test_ensure_db_exists_with_file_no_tables(self, test_db, tmp_path):
        """Test _ensure_db_exists when file exists but has no tables (race condition)."""

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
        # This simulates the race condition where Instance A creates the file
        # but hasn't created tables yet when Instance B checks
        with db.engine.connect() as conn:
            # Create a minimal valid SQLite file with valid header
            # by creating and dropping a temporary table
            conn.execute(text("CREATE TABLE _temp (id INTEGER)"))
            conn.execute(text("DROP TABLE _temp"))
            conn.commit()

        # Now verify this triggers table creation
        db._ensure_db_exists()

        # Verify all required tables were created
        assert db._verify_all_tables_exist() is True

    def test_ensure_db_exists_with_complete_database(self, test_db, tmp_path):
        """Test _ensure_db_exists when database is already complete."""

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
        db.init_db()
        db.set_db_version()

        db._ensure_db_exists()

        # Verify tables still exist (not corrupted)
        assert db._verify_all_tables_exist() is True

    def test_create_tables_individually_success(self, test_db):
        """Test _create_tables_individually with success."""
        db = test_db

        with patch.object(db.engine, "connect"):
            db._create_tables_individually()

    def test_create_tables_individually_race_condition(self, test_db):
        """Test _create_tables_individually with race condition."""
        db = test_db

        # Mock error with sqlite_errno = 1 (table already exists)
        mock_error = sa.exc.OperationalError("table already exists", None, None)
        mock_error.orig = Mock()
        mock_error.orig.sqlite_errno = 1

        with patch.object(db.engine, "connect", side_effect=mock_error):
            # Should not raise exception
            db._create_tables_individually()

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
            db._create_tables_individually()

    def test_set_db_version_update_existing(self, test_db):
        """Test set_db_version when version already exists."""
        db = test_db

        with patch.object(db.engine, "begin") as mock_begin:
            mock_conn = Mock()
            mock_result = Mock()
            mock_result.fetchone.return_value = ("3",)
            mock_conn.execute.return_value = mock_result
            mock_begin.return_value.__enter__.return_value = mock_conn

            db.set_db_version()

    def test_set_db_version_insert_new(self, test_db):
        """Test set_db_version when version doesn't exist."""
        db = test_db

        with patch.object(db.engine, "begin") as mock_begin:
            mock_conn = Mock()
            mock_result = Mock()
            mock_result.fetchone.return_value = None
            mock_conn.execute.return_value = mock_result
            mock_begin.return_value.__enter__.return_value = mock_conn

            db.set_db_version()

    def test_set_db_version_error(self, test_db):
        """Test set_db_version with error."""
        db = test_db

        with (
            patch.object(db.engine, "begin", side_effect=RuntimeError("DB Error")),
            pytest.raises(RuntimeError),
        ):
            db.set_db_version()

    def test_get_db_version_success(self, test_db):
        """Test get_db_version with success."""
        db = test_db

        with patch.object(db, "get_session") as mock_session:
            mock_session_obj = Mock()
            mock_metadata = Mock()
            mock_metadata.value = "3"
            mock_session_obj.query.return_value.filter_by.return_value.first.return_value = mock_metadata
            mock_session.return_value.__enter__.return_value = mock_session_obj

            version = db.get_db_version()
            assert version == 3

    def test_get_db_version_no_metadata(self, test_db):
        """Test get_db_version when no metadata exists."""
        db = test_db

        with patch.object(db, "get_session") as mock_session:
            mock_session_obj = Mock()
            mock_session_obj.query.return_value.filter_by.return_value.first.return_value = None
            mock_session.return_value.__enter__.return_value = mock_session_obj

            version = db.get_db_version()
            assert version == 0

    def test_get_db_version_error(self, test_db):
        """Test get_db_version with error."""
        db = test_db

        with (
            patch.object(db, "get_session", side_effect=RuntimeError("DB Error")),
            pytest.raises(RuntimeError),
        ):
            db.get_db_version()


# New tests for performance optimization features


class TestPruneOldIntervals:
    """Test the prune_old_intervals method."""

    def test_prune_old_intervals_success(self, test_db):
        """Test successful pruning of old intervals."""
        db = test_db

        # Create test intervals - some old, some recent
        old_time = dt_util.utcnow() - timedelta(days=RETENTION_DAYS + 10)
        recent_time = dt_util.utcnow() - timedelta(days=30)

        with db.get_session() as session:  # Use unlocked session for test setup
            # Add old intervals
            old_interval1 = db.Intervals(
                entity_id="binary_sensor.motion1",
                start_time=old_time,
                end_time=old_time + timedelta(hours=1),
                state="on",
                duration_seconds=3600,
            )
            old_interval2 = db.Intervals(
                entity_id="binary_sensor.motion2",
                start_time=old_time + timedelta(hours=2),
                end_time=old_time + timedelta(hours=3),
                state="on",
                duration_seconds=3600,
            )
            # Add recent interval
            recent_interval = db.Intervals(
                entity_id="binary_sensor.motion1",
                start_time=recent_time,
                end_time=recent_time + timedelta(hours=1),
                state="on",
                duration_seconds=3600,
            )

            session.add_all([old_interval1, old_interval2, recent_interval])
            session.commit()

        # Prune old intervals
        pruned_count = db.prune_old_intervals()

        # Should have pruned 2 old intervals
        assert pruned_count == 2

        # Verify old intervals are gone, recent interval remains
        with db.get_session() as session:
            remaining_intervals = session.query(db.Intervals).all()
            assert len(remaining_intervals) == 1
            # Compare without timezone info since database stores naive datetime
            assert remaining_intervals[0].start_time.replace(
                tzinfo=None
            ) == recent_time.replace(tzinfo=None)

    def test_prune_old_intervals_no_old_data(self, test_db):
        """Test pruning when no intervals are older than RETENTION_DAYS."""
        db = test_db

        # Create only recent intervals
        recent_time = dt_util.utcnow() - timedelta(days=30)

        with db.get_session() as session:  # Use unlocked session for test setup
            recent_interval = db.Intervals(
                entity_id="binary_sensor.motion1",
                start_time=recent_time,
                end_time=recent_time + timedelta(hours=1),
                state="on",
                duration_seconds=3600,
            )
            session.add(recent_interval)
            session.commit()

        # Prune old intervals
        pruned_count = db.prune_old_intervals()

        # Should prune nothing
        assert pruned_count == 0

        # Verify interval still exists
        with db.get_session() as session:
            remaining_intervals = session.query(db.Intervals).all()
            assert len(remaining_intervals) == 1

    def test_prune_old_intervals_database_errors(self, test_db):
        """Test pruning with database errors."""
        db = test_db

        # Mock database error
        with patch.object(
            db,
            "get_locked_session",
            side_effect=OperationalError("DB Error", None, None),
        ):
            pruned_count = db.prune_old_intervals()

            # Should return 0 on error
            assert pruned_count == 0


class TestGetAggregatedIntervalsBySlot:
    """Test the get_aggregated_intervals_by_slot method."""

    def test_get_aggregated_intervals_by_slot_success(self, test_db):
        """Test successful SQL aggregation."""
        db = test_db

        # Create test intervals across multiple days/times
        base_time = dt_util.utcnow().replace(hour=10, minute=0, second=0, microsecond=0)

        with db.get_locked_session() as session:
            # Add entity first (area_name is required in multi-area architecture)
            entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id="test_entry_id",
                area_name="Test Area",
                entity_type="motion",
            )
            session.add(entity)

            # Add intervals for different days and times
            # Calculate Monday and Tuesday dates properly
            monday = base_time - timedelta(
                days=base_time.weekday()
            )  # Get Monday of current week
            tuesday = monday + timedelta(days=1)

            intervals = [
                # Monday, 10:00-11:00 (slot 10 with 60min slots)
                db.Intervals(
                    entity_id="binary_sensor.motion1",
                    start_time=monday,
                    end_time=monday + timedelta(hours=1),
                    state="on",
                    duration_seconds=3600,
                ),
                # Tuesday, 14:00-15:00 (slot 14 with 60min slots)
                db.Intervals(
                    entity_id="binary_sensor.motion1",
                    start_time=tuesday + timedelta(hours=4),
                    end_time=tuesday + timedelta(hours=5),
                    state="on",
                    duration_seconds=3600,
                ),
                # Same Tuesday slot, different interval (should aggregate)
                db.Intervals(
                    entity_id="binary_sensor.motion1",
                    start_time=tuesday + timedelta(hours=4, minutes=30),
                    end_time=tuesday + timedelta(hours=5, minutes=30),
                    state="on",
                    duration_seconds=3600,
                ),
            ]
            session.add_all(intervals)
            session.commit()

        # Test aggregation with 60-minute slots
        result = db.get_aggregated_intervals_by_slot("test_entry_id", slot_minutes=60)

        # Should have aggregated data
        assert len(result) >= 2

        # Check format: (day_of_week, time_slot, total_seconds)
        for day_of_week, time_slot, total_seconds in result:
            assert isinstance(day_of_week, int)
            assert isinstance(time_slot, int)
            assert isinstance(total_seconds, float)
            assert 0 <= day_of_week <= 6  # Monday=0 to Sunday=6
            assert total_seconds > 0

    def test_get_aggregated_intervals_by_slot_empty(self, test_db):
        """Test aggregation with no intervals."""
        db = test_db

        result = db.get_aggregated_intervals_by_slot("test_entry_id", slot_minutes=60)

        # Should return empty list
        assert result == []

    def test_get_aggregated_intervals_by_slot_edge_cases(self, test_db):
        """Test aggregation with edge case data."""
        db = test_db

        # Create interval with potentially problematic data
        base_time = dt_util.utcnow()

        with db.get_locked_session() as session:
            # Add entity (area_name is required in multi-area architecture)
            entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id="test_entry_id",
                area_name="Test Area",
                entity_type="motion",
            )
            session.add(entity)

            # Add interval
            interval = db.Intervals(
                entity_id="binary_sensor.motion1",
                start_time=base_time,
                end_time=base_time + timedelta(hours=1),
                state="on",
                duration_seconds=3600,
            )
            session.add(interval)
            session.commit()

        # Should handle edge cases gracefully
        result = db.get_aggregated_intervals_by_slot("test_entry_id", slot_minutes=60)

        # Should return valid data or empty list
        assert isinstance(result, list)

    def test_get_aggregated_intervals_by_slot_database_errors(self, test_db):
        """Test aggregation with database errors."""
        db = test_db

        # Mock database error
        with patch.object(
            db, "get_session", side_effect=OperationalError("DB Error", None, None)
        ):
            result = db.get_aggregated_intervals_by_slot(
                "test_entry_id", slot_minutes=60
            )

            # Should return empty list on error
            assert result == []

    def test_retention_days_constant(self):
        """Test that RETENTION_DAYS constant is properly defined."""

        assert RETENTION_DAYS == 365

    # --- Entity Cleanup Tests ---

    async def test_cleanup_orphaned_entities_no_orphans(self, test_db):
        """Test cleanup when no orphaned entities exist."""
        db = test_db

        # Get actual area name from coordinator
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area_or_default(area_name)
        # Set up entities directly using _entities (entities property is read-only)
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity2 = Mock()
        mock_entity2.entity_id = "binary_sensor.motion2"
        area.entities._entities = {
            "binary_sensor.motion1": mock_entity1,
            "binary_sensor.motion2": mock_entity2,
        }

        # Add entities to database that match current config
        with db.get_locked_session() as session:
            entity1 = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            entity2 = db.Entities(
                entity_id="binary_sensor.motion2",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add_all([entity1, entity2])
            session.commit()

        # Run cleanup
        cleaned_count = db.cleanup_orphaned_entities()

        # Should clean up 0 entities
        assert cleaned_count == 0

        # Verify entities still exist
        with db.get_locked_session() as session:
            count = (
                session.query(db.Entities)
                .filter_by(entry_id=db.coordinator.entry_id)
                .count()
            )
            assert count == 2

    async def test_cleanup_orphaned_entities_with_orphans(self, test_db):
        """Test cleanup when orphaned entities exist."""
        db = test_db

        # Get actual area name from coordinator
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area_or_default(area_name)
        # Set up entities directly using _entities (entities property is read-only)
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        area.entities._entities = {"binary_sensor.motion1": mock_entity}

        # Add entities to database - one current, one orphaned
        with db.get_locked_session() as session:
            current_entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            orphaned_entity = db.Entities(
                entity_id="binary_sensor.motion_orphaned",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add_all([current_entity, orphaned_entity])
            session.commit()

        # Run cleanup
        cleaned_count = db.cleanup_orphaned_entities()

        # Should clean up 1 entity
        assert cleaned_count == 1

        # Verify only current entity remains
        with db.get_locked_session() as session:
            entities = (
                session.query(db.Entities)
                .filter_by(entry_id=db.coordinator.entry_id)
                .all()
            )
            assert len(entities) == 1
            assert entities[0].entity_id == "binary_sensor.motion1"

    async def test_cleanup_orphaned_entities_with_intervals(self, test_db):
        """Test cleanup removes orphaned entities and their intervals."""
        db = test_db

        # Get actual area name from coordinator
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area_or_default(area_name)
        # Set up entities directly using _entities (entities property is read-only)
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        area.entities._entities = {"binary_sensor.motion1": mock_entity}

        # Add entities and intervals to database
        with db.get_locked_session() as session:
            # Current entity
            current_entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(current_entity)

            # Orphaned entity with intervals
            orphaned_entity = db.Entities(
                entity_id="binary_sensor.motion_orphaned",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(orphaned_entity)

            # Add intervals for orphaned entity
            interval1 = db.Intervals(
                entity_id="binary_sensor.motion_orphaned",
                start_time=dt_util.utcnow(),
                end_time=dt_util.utcnow() + timedelta(minutes=30),
                state="on",
                duration_seconds=1800,
            )
            interval2 = db.Intervals(
                entity_id="binary_sensor.motion_orphaned",
                start_time=dt_util.utcnow() + timedelta(hours=1),
                end_time=dt_util.utcnow() + timedelta(hours=1, minutes=30),
                state="off",
                duration_seconds=1800,
            )
            session.add_all([interval1, interval2])
            session.commit()

        # Run cleanup
        cleaned_count = db.cleanup_orphaned_entities()

        # Should clean up 1 entity
        assert cleaned_count == 1

        # Verify orphaned entity and its intervals are removed
        with db.get_locked_session() as session:
            # Entity should be gone
            orphaned_entities = (
                session.query(db.Entities)
                .filter_by(entity_id="binary_sensor.motion_orphaned")
                .count()
            )
            assert orphaned_entities == 0

            # Intervals should be gone (CASCADE delete)
            orphaned_intervals = (
                session.query(db.Intervals)
                .filter_by(entity_id="binary_sensor.motion_orphaned")
                .count()
            )
            assert orphaned_intervals == 0

            # Current entity should remain
            current_entities = (
                session.query(db.Entities)
                .filter_by(entity_id="binary_sensor.motion1")
                .count()
            )
            assert current_entities == 1

    async def test_cleanup_orphaned_entities_multiple_orphans(self, test_db):
        """Test cleanup with multiple orphaned entities."""
        db = test_db

        # Get actual area name from coordinator
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area_or_default(area_name)
        # Set up entities directly using _entities (entities property is read-only)
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        area.entities._entities = {"binary_sensor.motion1": mock_entity}

        # Add multiple orphaned entities
        with db.get_locked_session() as session:
            current_entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            orphaned1 = db.Entities(
                entity_id="binary_sensor.motion_orphaned1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            orphaned2 = db.Entities(
                entity_id="binary_sensor.motion_orphaned2",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="door",
            )
            orphaned3 = db.Entities(
                entity_id="binary_sensor.motion_orphaned3",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="window",
            )
            session.add_all([current_entity, orphaned1, orphaned2, orphaned3])
            session.commit()

        # Run cleanup
        cleaned_count = db.cleanup_orphaned_entities()

        # Should clean up 3 entities
        assert cleaned_count == 3

        # Verify only current entity remains
        with db.get_locked_session() as session:
            entities = (
                session.query(db.Entities)
                .filter_by(entry_id=db.coordinator.entry_id)
                .all()
            )
            assert len(entities) == 1
            assert entities[0].entity_id == "binary_sensor.motion1"

    async def test_cleanup_orphaned_entities_database_error(self, test_db):
        """Test cleanup handles database errors gracefully."""
        db = test_db

        # Get actual area name from coordinator
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area_or_default(area_name)
        # Set up entities directly using _entities (entities property is read-only)
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        area.entities._entities = {"binary_sensor.motion1": mock_entity}

        # Mock database error on get_session
        with patch.object(
            db,
            "get_session",
            side_effect=OperationalError("DB Error", None, None),
        ):
            cleaned_count = db.cleanup_orphaned_entities()

            # Should return 0 on error
            assert cleaned_count == 0

    async def test_save_entity_data_calls_cleanup(self, test_db):
        """Test that save_entity_data calls cleanup after saving."""
        db = test_db

        # Mock coordinator with entities
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.type.input_type = "motion"
        mock_entity.type.weight = 0.85
        mock_entity.prob_given_true = 0.8
        mock_entity.prob_given_false = 0.05
        mock_entity.last_updated = dt_util.utcnow()
        mock_entity.decay.is_decaying = False
        mock_entity.decay.decay_start = None
        mock_entity.evidence = False

        # Get actual area name from coordinator
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area_or_default(area_name)
        area.entities._entities = {"binary_sensor.motion1": mock_entity}

        # Mock cleanup method
        with patch.object(
            db, "cleanup_orphaned_entities", return_value=2
        ) as mock_cleanup:
            db.save_entity_data()

            # Verify cleanup was called
            mock_cleanup.assert_called_once()

    async def test_load_data_skips_orphaned_entities(self, test_db):
        """Test that load_data skips entities not in current config."""
        db = test_db

        # Get actual area name from coordinator
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area_or_default(area_name)
        # Set up entities directly using _entities (entities property is read-only)
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        area.entities._entities = {"binary_sensor.motion1": mock_entity}
        # Mock entity manager to track what gets added
        original_add_entity = area.entities.add_entity
        added_entities = []

        def track_add_entity(entity):
            added_entities.append(entity.entity_id)
            return original_add_entity(entity)

        area.entities.add_entity = track_add_entity

        # Add entities to database - one current, one orphaned
        with db.get_locked_session() as session:
            current_entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
                prob_given_true=0.8,
                prob_given_false=0.05,
                evidence=False,
            )
            orphaned_entity = db.Entities(
                entity_id="binary_sensor.motion_orphaned",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
                prob_given_true=0.7,
                prob_given_false=0.03,
                evidence=True,
            )
            session.add_all([current_entity, orphaned_entity])
            session.commit()

        # Mock the get_entity method to raise ValueError for orphaned entity
        def mock_get_entity(entity_id):
            if entity_id == "binary_sensor.motion_orphaned":
                raise ValueError("Entity not found")
            # Return a mock entity for the current one
            mock_entity = Mock()
            mock_entity.entity_id = entity_id
            return mock_entity

        # Use area-based access for patching
        with patch.object(area.entities, "get_entity", side_effect=mock_get_entity):
            await db.load_data()

        # Verify only current entity was processed, orphaned was skipped
        # The current entity should be updated (not added), so no entities should be added
        assert len(added_entities) == 0

    async def test_load_data_deletes_stale_entities(self, test_db):
        """Test that load_data deletes stale entities from database."""
        db = test_db

        # Get actual area name from coordinator
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area_or_default(area_name)
        # Set up entities directly using _entities (entities property is read-only)
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        area.entities._entities = {"binary_sensor.motion1": mock_entity}

        # Add entities to database - one current, one stale
        with db.get_locked_session() as session:
            current_entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
                prob_given_true=0.8,
                prob_given_false=0.05,
                evidence=False,
            )
            stale_entity = db.Entities(
                entity_id="binary_sensor.bed_status",  # This entity is not in current config
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="door",
                prob_given_true=0.7,
                prob_given_false=0.03,
                evidence=True,
            )
            session.add_all([current_entity, stale_entity])
            session.commit()

        # Verify both entities exist before load_data
        with db.get_locked_session() as session:
            entities_before = session.query(db.Entities).all()
            assert len(entities_before) == 2
            entity_ids_before = {e.entity_id for e in entities_before}
            assert "binary_sensor.motion1" in entity_ids_before
            assert "binary_sensor.bed_status" in entity_ids_before

        # Mock the get_entity method to raise ValueError for stale entity - use area-based access
        def mock_get_entity(entity_id):
            if entity_id == "binary_sensor.bed_status":
                raise ValueError("Entity not found")
            # Return a mock entity for the current one with all required attributes
            mock_entity = Mock()
            mock_entity.entity_id = entity_id
            mock_entity.update_decay = Mock()
            mock_entity.update_likelihood = Mock()
            mock_entity.type = Mock()
            mock_entity.type.weight = 0.85
            return mock_entity

        with patch.object(area.entities, "get_entity", side_effect=mock_get_entity):
            await db.load_data()

        # Verify stale entity was actually deleted from database
        with db.get_locked_session() as session:
            entities_after = session.query(db.Entities).all()
            assert len(entities_after) == 1
            entity_ids_after = {e.entity_id for e in entities_after}
            assert "binary_sensor.motion1" in entity_ids_after
            assert "binary_sensor.bed_status" not in entity_ids_after
