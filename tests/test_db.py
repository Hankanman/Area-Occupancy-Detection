"""Tests for AreaOccupancy database models and utilities."""
# ruff: noqa: SLF001

from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from custom_components.area_occupancy.db import (
    DB_VERSION,
    RETENTION_DAYS,
    AreaOccupancyDB,
    Base,
)
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


@pytest.fixture(autouse=True)
def mock_area_occupancy_db_globally():
    """Override autouse patching from conftest to use real DB class."""
    return


@pytest.fixture
def configured_db(mock_db_with_engine):
    """Return a DB instance with a configured coordinator."""
    db = mock_db_with_engine
    db.coordinator.area_prior = 0.3
    db.coordinator.config = SimpleNamespace(
        name="Test Area",
        area_id="test_area",
        purpose="living",
        threshold=0.5,
        entity_ids=["binary_sensor.motion"],
    )
    return db


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


def create_test_prior_data(entry_id="test_entry_001", **overrides):
    """Create standardized test prior data."""
    data = {
        "entry_id": entry_id,
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
        area, entity = setup_test_area_and_entity(db_session)

        prior = AreaOccupancyDB.Priors.from_dict(create_test_prior_data())
        db_session.add(prior)
        db_session.commit()

        # Test relationships
        retrieved_area = (
            db_session.query(AreaOccupancyDB.Areas)
            .filter_by(entry_id="test_entry_001")
            .first()
        )

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
    """Test database operations using the AreaOccupancyDB class."""

    def test_area_occupancy_db_initialization(self, mock_area_occupancy_db):
        """Test AreaOccupancyDB initialization with in-memory database."""
        db = mock_area_occupancy_db

        assert db.engine is not None
        assert db.session is not None

        # Test that we can create tables
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

    def test_table_properties_and_engine(self, configured_db):
        """Test table properties and engine access."""
        db = configured_db
        assert db.get_engine() is db.engine
        assert db.areas.name == "areas"
        assert db.entities.name == "entities"
        assert db.intervals.name == "intervals"
        assert db.priors.name == "priors"
        assert db.metadata.name == "metadata"

    def test_version_operations(self, configured_db):
        """Test database version operations."""
        db = configured_db
        db.init_db()
        db.set_db_version()
        db.set_db_version()
        assert db.get_db_version() == DB_VERSION

    def test_delete_and_force_reinitialize(self, configured_db, tmp_path, monkeypatch):
        """Test database deletion and reinitialization."""
        db = configured_db
        db.db_path = tmp_path / "test.db"
        db.db_path.write_text("data")
        db.delete_db()
        assert not db.db_path.exists()

        calls = []
        monkeypatch.setattr(db, "init_db", lambda: calls.append("init"))
        monkeypatch.setattr(db, "set_db_version", lambda: calls.append("ver"))
        db.force_reinitialize()
        assert calls == ["init", "ver"]

    def test_enable_wal_mode_and_create_tables(self, configured_db, monkeypatch):
        """Test WAL mode and table creation with error handling."""
        db = configured_db
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

    def test_is_valid_state_and_latest_interval(self, configured_db):
        """Test state validation and latest interval retrieval."""
        db = configured_db
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
    async def test_states_to_intervals_and_ensure_area(
        self, configured_db, monkeypatch
    ):
        """Test states to intervals conversion and area existence checking."""
        db = configured_db
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

        async def fake_save():
            saved.append(True)

        data_returns = [None, {"entry_id": db.coordinator.entry_id}]

        def fake_get(entry_id):
            return data_returns.pop(0)

        monkeypatch.setattr(db, "save_data", fake_save)
        monkeypatch.setattr(db, "get_area_data", fake_get)
        await db.ensure_area_exists()
        assert saved == [True]

    @pytest.mark.asyncio
    async def test_ensure_area_exists_when_present(self, configured_db, monkeypatch):
        """Test ensure_area_exists when area already exists."""
        db = configured_db
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
    async def test_save_entity_data(self, configured_db):
        """Test saving entity data with various entity types."""
        db = configured_db
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
        )
        missing_type = SimpleNamespace(
            entity_id="binary_sensor.bad",
            decay=SimpleNamespace(
                is_decaying=False,
                decay_start=dt_util.utcnow(),
            ),
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
        )
        db.coordinator.entities = SimpleNamespace(
            entities={
                "binary_sensor.good": good,
                "binary_sensor.bad": missing_type,
                "binary_sensor.noinput": no_input,
            }
        )
        await db.save_entity_data()
        with db.engine.connect() as conn:
            rows = conn.execute(sa.text("SELECT entity_id FROM entities")).fetchall()
        assert {r[0] for r in rows} == {"binary_sensor.good"}

    def test_is_intervals_empty(self, configured_db):
        """Test intervals empty check."""
        db = configured_db
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

    def test_get_area_data_error(self, configured_db, monkeypatch):
        """Test get_area_data with session error."""
        db = configured_db

        def bad_session():
            raise sa.exc.SQLAlchemyError("boom")

        monkeypatch.setattr(db, "get_session", bad_session)
        assert db.get_area_data("x") is None

    @pytest.mark.asyncio
    async def test_load_data(self, configured_db, monkeypatch):
        """Test loading data from database."""
        db = configured_db
        await db.save_area_data()
        await self.test_save_entity_data(configured_db)  # populate entities table

        called = []
        created_entities = []

        # Mock the prior setter
        db.coordinator.prior = SimpleNamespace(
            set_global_prior=lambda v: called.append(("prior", v))
        )

        # Mock the entities manager
        def create_mock_entity():
            entity = SimpleNamespace(
                prob_given_true=0.5, prob_given_false=0.1, last_updated=dt_util.utcnow()
            )
            entity.update_decay = Mock()
            entity.update_likelihood = Mock()
            return entity

        db.coordinator.entities = SimpleNamespace(
            get_entity=lambda entity_id: create_mock_entity(),
            add_entity=lambda entity: created_entities.append(entity.entity_id),
        )

        # Mock the factory
        db.coordinator.factory = SimpleNamespace(
            create_from_db=lambda ent: SimpleNamespace(entity_id=ent.entity_id)
        )

        await db.load_data()

        # Check that prior was set
        assert any(call[0] == "prior" for call in called)

    @pytest.mark.asyncio
    async def test_load_data_entity_handling(self, configured_db, monkeypatch):
        """Test the entity handling logic in load_data method."""
        db = configured_db
        await db.save_area_data()
        await self.test_save_entity_data(configured_db)  # populate entities table

        created_entities = []

        # Mock the prior setter
        db.coordinator.prior = SimpleNamespace(set_global_prior=lambda v: None)

        # Mock the entities manager to simulate existing entities
        def mock_get_entity(entity_id):
            if entity_id == "binary_sensor.motion":
                # Return existing entity that should be updated
                entity = SimpleNamespace(
                    prob_given_true=0.5,
                    prob_given_false=0.1,
                    last_updated=dt_util.utcnow(),
                )
                entity.update_decay = Mock()
                entity.update_likelihood = Mock()
                return entity
            # Raise ValueError to simulate entity not found
            raise ValueError(f"Entity {entity_id} not found")

        db.coordinator.entities = SimpleNamespace(
            get_entity=mock_get_entity,
            add_entity=lambda entity: created_entities.append(entity.entity_id),
        )

        # Mock the factory to create new entities
        db.coordinator.factory = SimpleNamespace(
            create_from_db=lambda ent: SimpleNamespace(entity_id=ent.entity_id)
        )

        await db.load_data()

        # Verify that new entities were created for entities not found in coordinator
        assert len(created_entities) > 0

    @pytest.mark.asyncio
    async def test_load_data_error_handling(self, configured_db, monkeypatch):
        """Test error handling in load_data method."""
        db = configured_db
        await db.save_area_data()
        await self.test_save_entity_data(configured_db)  # populate entities table

        # Mock the prior setter
        db.coordinator.prior = SimpleNamespace(set_global_prior=lambda v: None)

        # Mock the entities manager to always raise ValueError
        db.coordinator.entities = SimpleNamespace(
            get_entity=lambda entity_id: (_ for _ in ()).throw(
                ValueError(f"Entity {entity_id} not found")
            ),
            add_entity=lambda entity: None,
        )

        # Mock the factory
        db.coordinator.factory = SimpleNamespace(
            create_from_db=lambda ent: SimpleNamespace(entity_id=ent.entity_id)
        )

        # This should not raise an exception - it should handle the ValueError gracefully
        await db.load_data()

    @pytest.mark.asyncio
    async def test_sync_states(self, configured_db, monkeypatch):
        """Test syncing states from recorder."""
        db = configured_db
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

    def test_get_area_data_none(self, configured_db):
        """Test get_area_data with missing entry."""
        db = configured_db
        assert db.get_area_data("missing") is None

    def test_states_to_intervals_edge_cases(self, configured_db):
        """Test states to intervals with edge cases."""
        db = configured_db
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
    async def test_sync_states_error(self, configured_db, monkeypatch):
        """Test sync_states with recorder error."""
        db = configured_db

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
    async def test_sync_states_no_states(self, configured_db, monkeypatch):
        """Test sync_states with no states."""
        db = configured_db

        class DummyRecorder:
            async def async_add_executor_job(self, func):
                return {}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.get_instance",
            lambda hass: DummyRecorder(),
        )
        await db.sync_states()

    @pytest.mark.asyncio
    async def test_ensure_area_exists_error_handling(self, configured_db, monkeypatch):
        """Test ensure_area_exists error handling."""
        db = configured_db
        monkeypatch.setattr(db, "get_area_data", lambda eid: None)

        async def bad_save():
            raise HomeAssistantError("boom")

        monkeypatch.setattr(db, "save_data", bad_save)
        await db.ensure_area_exists()

    @pytest.mark.asyncio
    async def test_ensure_area_exists_fails_to_create(self, configured_db, monkeypatch):
        """Test ensure_area_exists when creation fails."""
        db = configured_db
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
        self, configured_db, monkeypatch, error_type, expected_result
    ):
        """Test is_intervals_empty with various error conditions."""
        db = configured_db

        @contextmanager
        def error_session():
            class S:
                def query(self, *args, **kwargs):
                    raise sa.exc.SQLAlchemyError(error_type)

                def close(self):
                    pass

            yield S()

        monkeypatch.setattr(db, "get_session", error_session)

        if expected_result == "raise":
            with pytest.raises(sa.exc.SQLAlchemyError):
                db.is_intervals_empty()
        else:
            assert db.is_intervals_empty() is expected_result

    @pytest.mark.parametrize(
        ("error_type", "should_raise"),
        [
            ("no such table", False),
            ("other", True),
        ],
    )
    def test_get_latest_interval_error(
        self, configured_db, monkeypatch, error_type, should_raise
    ):
        """Test get_latest_interval with various error conditions."""
        db = configured_db

        @contextmanager
        def bad_session():
            class S:
                def execute(self, *args, **kwargs):
                    raise sa.exc.SQLAlchemyError(error_type)

                def close(self):
                    pass

            yield S()

        monkeypatch.setattr(db, "get_session", bad_session)

        if should_raise:
            with pytest.raises(sa.exc.SQLAlchemyError):
                db.get_latest_interval()
        else:
            db.get_latest_interval()

    @pytest.mark.asyncio
    async def test_load_data_error(self, configured_db, monkeypatch):
        """Test load_data with session error."""
        db = configured_db

        @contextmanager
        def bad_session():
            raise RuntimeError("fail")
            yield

        monkeypatch.setattr(db, "get_session", bad_session)
        with pytest.raises(RuntimeError):
            await db.load_data()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("entry_id", ""),
            ("area_name", ""),
            ("purpose", ""),
            ("threshold", None),
            ("area_prior", None),
        ],
    )
    async def test_save_area_data_validation(self, configured_db, field, value):
        """Test save_area_data validation with various invalid values."""
        db = configured_db
        if field == "entry_id":
            db.coordinator.entry_id = value
        elif field == "area_name":
            db.coordinator.config.name = value
        elif field == "purpose":
            db.coordinator.config.purpose = value
        elif field == "threshold":
            db.coordinator.config.threshold = value
        elif field == "area_prior":
            db.coordinator.area_prior = value

        await db.save_area_data()
        with db.engine.connect() as conn:
            result = conn.execute(sa.text("SELECT COUNT(*) FROM areas")).scalar()
        assert result == 0

    @pytest.mark.asyncio
    async def test_save_area_data_success(self, configured_db):
        """Test successful save_area_data operation."""
        db = configured_db
        db.coordinator.config.area_id = None
        await db.save_area_data()
        with db.engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT area_id FROM areas WHERE entry_id=:e"),
                {"e": db.coordinator.entry_id},
            ).fetchone()
        assert row[0] == db.coordinator.entry_id

        data = db.get_area_data(db.coordinator.entry_id)
        assert data is not None and data["area_id"] == db.coordinator.entry_id
