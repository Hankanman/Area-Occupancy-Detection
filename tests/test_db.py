"""Tests for database operations using SQLAlchemy fixtures."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from custom_components.area_occupancy.db import AreaOccupancyDB


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

    def test_session_management(self, mock_area_occupancy_db):
        """Test session management methods."""
        db = mock_area_occupancy_db

        # Test commit
        db.commit()

        # Test rollback
        db.rollback()

        # Test close
        db.close()
        assert db.session is None

        # Test refresh session
        db.refresh_session()
        assert db.session is not None

    def test_seeded_database(self, seeded_db_session):
        """Test database with pre-seeded data."""
        # Query the seeded data
        areas = seeded_db_session.query(AreaOccupancyDB.Areas).all()
        entities = seeded_db_session.query(AreaOccupancyDB.Entities).all()

        assert len(areas) == 1
        assert len(entities) == 1

        area = areas[0]
        entity = entities[0]

        assert area.entry_id == "test_entry_001"
        assert area.area_name == "Test Living Room"
        assert entity.entity_id == "binary_sensor.motion_1"
        assert entity.entity_type == "motion"


class TestStateIntervalIntegration:
    """Test StateInterval integration with database."""

    def test_state_interval_to_orm(self, db_session: Session):
        """Test converting StateInterval to ORM model."""
        # Create area and entity first
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

        # Create StateInterval (this returns a dictionary)
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        state_interval = StateInterval(
            entity_id="binary_sensor.motion_1",
            state="on",
            start=start_time,
            end=end_time,
        )

        # Convert to ORM model - use the StateInterval dictionary values
        interval_data = {
            "entity_id": state_interval["entity_id"],
            "state": state_interval["state"],
            "start_time": state_interval["start"],
            "end_time": state_interval["end"],
            "duration_seconds": (end_time - start_time).total_seconds(),
            "created_at": datetime.now(),
        }

        interval = AreaOccupancyDB.Intervals.from_dict(interval_data)
        db_session.add(interval)
        db_session.commit()

        # Verify
        retrieved_interval = (
            db_session.query(AreaOccupancyDB.Intervals)
            .filter_by(entity_id="binary_sensor.motion_1")
            .first()
        )

        assert retrieved_interval is not None
        assert retrieved_interval.entity_id == state_interval["entity_id"]
        assert retrieved_interval.state == state_interval["state"]
        assert retrieved_interval.start_time == state_interval["start"]
        assert retrieved_interval.end_time == state_interval["end"]
        assert (
            retrieved_interval.duration_seconds
            == (end_time - start_time).total_seconds()
        )
