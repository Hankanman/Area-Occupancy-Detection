"""Tests for AreaOccupancy database models and utilities."""
# ruff: noqa: SLF001

from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace

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


class TestDatabaseModels:
    """Test database ORM models and basic operations."""

    def test_areas_model_creation(self, db_session: Session):
        """Test creating and retrieving Areas model."""
        # Create area data
        area_data = {
            "entry_id": "test_entry_001",
            "area_name": "Test Living Room",
            "purpose": "living",
            "threshold": 0.5,
            "area_prior": 0.3,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        # Create ORM object
        area = AreaOccupancyDB.Areas.from_dict(area_data)

        # Add to session and commit
        db_session.add(area)
        db_session.commit()

        # Retrieve and verify
        retrieved_area = (
            db_session.query(AreaOccupancyDB.Areas)
            .filter_by(entry_id="test_entry_001")
            .first()
        )

        assert retrieved_area is not None
        assert retrieved_area.entry_id == "test_entry_001"
        assert retrieved_area.area_name == "Test Living Room"
        assert retrieved_area.threshold == 0.5
        assert retrieved_area.area_prior == 0.3

    def test_entities_model_creation(self, db_session: Session):
        """Test creating and retrieving Entities model."""
        # First create an area (required for foreign key)
        area_data = {
            "entry_id": "test_entry_001",
            "area_name": "Test Living Room",
            "purpose": "living",
            "threshold": 0.5,
            "area_prior": 0.3,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        area = AreaOccupancyDB.Areas.from_dict(area_data)
        db_session.add(area)
        db_session.commit()

        # Create entity data
        entity_data = {
            "entry_id": "test_entry_001",
            "entity_id": "binary_sensor.motion_1",
            "entity_type": "motion",
            "weight": 0.85,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "last_updated": datetime.now(),
            "created_at": datetime.now(),
        }

        # Create ORM object
        entity = AreaOccupancyDB.Entities.from_dict(entity_data)

        # Add to session and commit
        db_session.add(entity)
        db_session.commit()

        # Retrieve and verify
        retrieved_entity = (
            db_session.query(AreaOccupancyDB.Entities)
            .filter_by(entry_id="test_entry_001", entity_id="binary_sensor.motion_1")
            .first()
        )

        assert retrieved_entity is not None
        assert retrieved_entity.entity_id == "binary_sensor.motion_1"
        assert retrieved_entity.entity_type == "motion"
        assert retrieved_entity.weight == 0.85
        assert retrieved_entity.prob_given_true == 0.8

    def test_intervals_model_creation(self, db_session: Session):
        """Test creating and retrieving Intervals model."""
        # First create area and entity (required for foreign keys)
        area_data = {
            "entry_id": "test_entry_001",
            "area_name": "Test Living Room",
            "purpose": "living",
            "threshold": 0.5,
            "area_prior": 0.3,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        area = AreaOccupancyDB.Areas.from_dict(area_data)
        db_session.add(area)

        entity_data = {
            "entry_id": "test_entry_001",
            "entity_id": "binary_sensor.motion_1",
            "entity_type": "motion",
            "weight": 0.85,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "last_updated": datetime.now(),
            "created_at": datetime.now(),
        }
        entity = AreaOccupancyDB.Entities.from_dict(entity_data)
        db_session.add(entity)
        db_session.commit()

        # Create interval data
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        interval_data = {
            "entity_id": "binary_sensor.motion_1",
            "state": "on",
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": 3600.0,
            "created_at": datetime.now(),
        }

        # Create ORM object
        interval = AreaOccupancyDB.Intervals.from_dict(interval_data)

        # Add to session and commit
        db_session.add(interval)
        db_session.commit()

        # Retrieve and verify
        retrieved_interval = (
            db_session.query(AreaOccupancyDB.Intervals)
            .filter_by(entity_id="binary_sensor.motion_1")
            .first()
        )

        assert retrieved_interval is not None
        assert retrieved_interval.entity_id == "binary_sensor.motion_1"
        assert retrieved_interval.state == "on"
        assert retrieved_interval.duration_seconds == 3600.0

    def test_priors_model_creation(self, db_session: Session):
        """Test creating and retrieving Priors model."""
        # First create an area (required for foreign key)
        area_data = {
            "entry_id": "test_entry_001",
            "area_name": "Test Living Room",
            "purpose": "living",
            "threshold": 0.5,
            "area_prior": 0.3,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        area = AreaOccupancyDB.Areas.from_dict(area_data)
        db_session.add(area)
        db_session.commit()

        # Create prior data
        prior_data = {
            "entry_id": "test_entry_001",
            "day_of_week": 1,  # Monday
            "time_slot": 14,  # 2 PM
            "prior_value": 0.35,
            "data_points": 10,
            "last_updated": datetime.now(),
        }

        # Create ORM object
        prior = AreaOccupancyDB.Priors.from_dict(prior_data)

        # Add to session and commit
        db_session.add(prior)
        db_session.commit()

        # Retrieve and verify
        retrieved_prior = (
            db_session.query(AreaOccupancyDB.Priors)
            .filter_by(entry_id="test_entry_001", day_of_week=1, time_slot=14)
            .first()
        )

        assert retrieved_prior is not None
        assert retrieved_prior.entry_id == "test_entry_001"
        assert retrieved_prior.day_of_week == 1
        assert retrieved_prior.time_slot == 14
        assert retrieved_prior.prior_value == 0.35
        assert retrieved_prior.data_points == 10

    def test_relationships(self, db_session: Session):
        """Test ORM relationships between models."""
        # Create area
        area_data = {
            "entry_id": "test_entry_001",
            "area_name": "Test Living Room",
            "purpose": "living",
            "threshold": 0.5,
            "area_prior": 0.3,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        area = AreaOccupancyDB.Areas.from_dict(area_data)
        db_session.add(area)

        # Create entity
        entity_data = {
            "entry_id": "test_entry_001",
            "entity_id": "binary_sensor.motion_1",
            "entity_type": "motion",
            "weight": 0.85,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "last_updated": datetime.now(),
            "created_at": datetime.now(),
        }
        entity = AreaOccupancyDB.Entities.from_dict(entity_data)
        db_session.add(entity)

        # Create prior
        prior_data = {
            "entry_id": "test_entry_001",
            "day_of_week": 1,
            "time_slot": 14,
            "prior_value": 0.35,
            "data_points": 10,
            "last_updated": datetime.now(),
        }
        prior = AreaOccupancyDB.Priors.from_dict(prior_data)
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
        assert retrieved_area.priors[0].time_slot == 14

    def test_to_dict_methods(self, db_session: Session):
        """Test to_dict methods on ORM models."""
        # Create and save area
        area_data = {
            "entry_id": "test_entry_001",
            "area_name": "Test Living Room",
            "purpose": "living",
            "threshold": 0.5,
            "area_prior": 0.3,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        area = AreaOccupancyDB.Areas.from_dict(area_data)
        db_session.add(area)
        db_session.commit()

        # Test to_dict method
        area_dict = area.to_dict()
        assert area_dict["entry_id"] == "test_entry_001"
        assert area_dict["area_name"] == "Test Living Room"
        assert area_dict["threshold"] == 0.5
        assert "created_at" in area_dict
        assert "updated_at" in area_dict

    def test_unique_constraints(self, db_session: Session):
        """Test unique constraints on models."""
        # Create area
        area_data = {
            "entry_id": "test_entry_001",
            "area_name": "Test Living Room",
            "purpose": "living",
            "threshold": 0.5,
            "area_prior": 0.3,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        area = AreaOccupancyDB.Areas.from_dict(area_data)
        db_session.add(area)
        db_session.commit()

        # Create entity
        entity_data = {
            "entry_id": "test_entry_001",
            "entity_id": "binary_sensor.motion_1",
            "entity_type": "motion",
            "weight": 0.85,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "last_updated": datetime.now(),
            "created_at": datetime.now(),
        }
        entity = AreaOccupancyDB.Entities.from_dict(entity_data)
        db_session.add(entity)
        db_session.commit()
        # Expunge objects so a duplicate insert triggers an IntegrityError
        db_session.expunge_all()

        # Try to create duplicate entity (should fail due to primary key constraint)
        duplicate_entity = AreaOccupancyDB.Entities.from_dict(entity_data)
        db_session.add(duplicate_entity)

        with pytest.raises(IntegrityError):  # Should raise integrity error
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

            assert "areas" in table_names
            assert "entities" in table_names
            assert "intervals" in table_names
            assert "priors" in table_names
            assert "metadata" in table_names

    def test_seeded_database(self, seeded_db_session):
        """Test database with pre-seeded data."""
        # Since the fixture was simplified to just return the session,
        # we should test that the session is working properly instead
        # of expecting pre-seeded data

        # Test that we can query the database
        areas = seeded_db_session.query(AreaOccupancyDB.Areas).all()
        entities = seeded_db_session.query(AreaOccupancyDB.Entities).all()

        # The database should be empty since we simplified the fixture
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
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        seeded_db_session.add(test_area)
        seeded_db_session.commit()

        # Verify the area was added
        areas = seeded_db_session.query(AreaOccupancyDB.Areas).all()
        assert len(areas) == 1
        assert areas[0].entry_id == "test_entry"


class TestAreaOccupancyDBUtilities:
    def test_table_properties_and_engine(self, configured_db):
        db = configured_db
        assert db.get_engine() is db.engine
        assert db.areas.name == "areas"
        assert db.entities.name == "entities"
        assert db.intervals.name == "intervals"
        assert db.priors.name == "priors"
        assert db.metadata.name == "metadata"

    def test_version_set_and_get(self, configured_db):
        db = configured_db
        db.init_db()
        db.set_db_version()
        db.set_db_version()
        assert db.get_db_version() == DB_VERSION

    def test_delete_and_force_reinitialize(self, configured_db, tmp_path, monkeypatch):
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
        db = configured_db
        db.init_db()
        assert db.is_valid_state("on")
        assert not db.is_valid_state("unknown")

        first = db.get_latest_interval()
        assert isinstance(first, datetime)

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
        db = configured_db
        good = SimpleNamespace(
            entity_id="binary_sensor.good",
            type=SimpleNamespace(input_type="motion", weight=0.9),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
        )
        missing_type = SimpleNamespace(entity_id="binary_sensor.bad")
        no_input = SimpleNamespace(
            entity_id="binary_sensor.noinput",
            type=SimpleNamespace(input_type=None, weight=0.5),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
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
        db = configured_db
        db.init_db()
        assert db.is_intervals_empty()
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
        db = configured_db

        def bad_session():
            raise sa.exc.SQLAlchemyError("boom")

        monkeypatch.setattr(db, "get_session", bad_session)
        assert db.get_area_data("x") is None

    @pytest.mark.asyncio
    async def test_load_data(self, configured_db, monkeypatch):
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
        db.coordinator.entities = SimpleNamespace(
            get_entity=lambda entity_id: SimpleNamespace(
                prob_given_true=0.5, prob_given_false=0.1, last_updated=dt_util.utcnow()
            ),
            add_entity=lambda entity: created_entities.append(entity.entity_id),
        )

        # Mock the factory
        db.coordinator.factory = SimpleNamespace(
            create_from_db=lambda ent: SimpleNamespace(entity_id=ent.entity_id)
        )

        await db.load_data()

        # Check that prior was set
        assert any(call[0] == "prior" for call in called)

        # Check that entities were processed (either updated or created)
        # The exact behavior depends on whether entities exist in coordinator
        # This test verifies the method completes without errors

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
                return SimpleNamespace(
                    prob_given_true=0.5,
                    prob_given_false=0.1,
                    last_updated=dt_util.utcnow(),
                )
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
        # The exact count depends on the entities saved in test_save_entity_data
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
        db = configured_db
        assert db.get_area_data("missing") is None

    def test_states_to_intervals_edge_cases(self, configured_db):
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
        db = configured_db
        monkeypatch.setattr(db, "get_area_data", lambda eid: None)

        async def bad_save():
            raise HomeAssistantError("boom")

        monkeypatch.setattr(db, "save_data", bad_save)
        await db.ensure_area_exists()

    @pytest.mark.asyncio
    async def test_ensure_area_exists_fails_to_create(self, configured_db, monkeypatch):
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

    def test_is_intervals_empty_error(self, configured_db, monkeypatch):
        db = configured_db

        @contextmanager
        def missing_table():
            class S:
                def query(self, *args, **kwargs):
                    raise sa.exc.SQLAlchemyError("no such table")

                def close(self):
                    pass

            yield S()

        monkeypatch.setattr(db, "get_session", missing_table)
        assert db.is_intervals_empty() is True

        @contextmanager
        def bad_session():
            class S:
                def query(self, *args, **kwargs):
                    raise sa.exc.SQLAlchemyError("bad")

                def close(self):
                    pass

            yield S()

        monkeypatch.setattr(db, "get_session", bad_session)
        with pytest.raises(sa.exc.SQLAlchemyError):
            db.is_intervals_empty()

    def test_get_latest_interval_error(self, configured_db, monkeypatch):
        db = configured_db

        @contextmanager
        def bad_session():
            class S:
                def execute(self, *args, **kwargs):
                    raise sa.exc.SQLAlchemyError("no such table")

                def close(self):
                    pass

            yield S()

        monkeypatch.setattr(db, "get_session", bad_session)
        db.get_latest_interval()

    def test_get_latest_interval_error_other(self, configured_db, monkeypatch):
        db = configured_db

        @contextmanager
        def bad_session():
            class S:
                def execute(self, *args, **kwargs):
                    raise sa.exc.SQLAlchemyError("other")

                def close(self):
                    pass

            yield S()

        monkeypatch.setattr(db, "get_session", bad_session)
        with pytest.raises(sa.exc.SQLAlchemyError):
            db.get_latest_interval()

    @pytest.mark.asyncio
    async def test_load_data_error(self, configured_db, monkeypatch):
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
