"""Tests for storage module using SQLAlchemy fixtures."""

from datetime import timedelta
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.storage import AreaOccupancyStorage
from homeassistant.util import dt as dt_util


class TestAreaOccupancyStorage:
    """Test AreaOccupancyStorage class with in-memory database operations."""

    @pytest.fixture
    def storage_with_db(self, mock_storage_with_db):
        """Create an AreaOccupancyStorage instance with in-memory database."""
        return mock_storage_with_db

    async def test_initialization(self, mock_hass: Mock, db_engine, db_session):
        """Test AreaOccupancyStorage initialization with in-memory database."""
        # Create mock coordinator
        mock_coordinator = Mock()
        mock_coordinator.hass = mock_hass
        mock_coordinator.entry_id = "test_entry_001"

        # Create storage instance
        storage = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Override with test database
        storage.db.engine = db_engine
        storage.db.session = db_session

        assert storage.entry_id == "test_entry_001"
        assert storage.db is not None

    async def test_async_initialize_success(
        self, storage_with_db: AreaOccupancyStorage
    ):
        """Test successful async_initialize with in-memory database."""
        await storage_with_db.async_initialize()
        assert storage_with_db.db is not None
        assert storage_with_db.engine is not None

    async def test_save_entity_config(self, storage_with_db: AreaOccupancyStorage):
        """Test saving entity configuration with in-memory database."""
        await storage_with_db.async_initialize()

        record = {
            "entry_id": "test_entry_001",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": dt_util.utcnow(),
        }
        result = await storage_with_db.save_entity_config(record)
        assert result["entity_id"] == "sensor.test"
        assert result["entity_type"] == "motion"

        # Test duplicate save (should update)
        record["weight"] = 0.9
        updated_result = await storage_with_db.save_entity_config(record)
        assert updated_result["weight"] == 0.9

        # Test retrieval
        configs = await storage_with_db.get_entity_configs("test_entry_001")
        assert len(configs) == 1
        assert configs[0]["entity_id"] == "sensor.test"

    async def test_get_stats(self, storage_with_db: AreaOccupancyStorage):
        """Test getting database statistics with in-memory database."""
        await storage_with_db.async_initialize()

        # Add some test data
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(area_record)

        entity_record = {
            "entry_id": "test_entry_001",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": dt_util.utcnow(),
        }
        await storage_with_db.save_entity_config(entity_record)

        intervals = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "start": dt_util.utcnow(),
                "end": dt_util.utcnow() + timedelta(hours=1),
            }
        ]
        await storage_with_db.save_state_intervals_batch(intervals)

        # Get statistics
        stats = await storage_with_db.get_stats()
        assert stats["areas_count"] == 1
        assert stats["entities_count"] == 1
        assert stats["intervals_count"] == 1

    async def test_save_area_occupancy(self, storage_with_db: AreaOccupancyStorage):
        """Test saving area occupancy with in-memory database."""
        await storage_with_db.async_initialize()

        record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        result = await storage_with_db.save_area_occupancy(record)
        assert result["entry_id"] == "test_entry_001"
        assert result["area_name"] == "Test Area"

        # Test duplicate save (should update)
        record["area_name"] = "Updated Area"
        updated_result = await storage_with_db.save_area_occupancy(record)
        assert updated_result["area_name"] == "Updated Area"

    async def test_get_area_occupancy(self, storage_with_db: AreaOccupancyStorage):
        """Test getting area occupancy with in-memory database."""
        await storage_with_db.async_initialize()

        # Save area occupancy
        record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(record)

        # Retrieve area occupancy
        result = await storage_with_db.get_area_occupancy("test_entry_001")
        assert result is not None
        assert result["entry_id"] == "test_entry_001"
        assert result["area_name"] == "Test Area"

        # Test non-existent entry
        result_none = await storage_with_db.get_area_occupancy("non_existent")
        assert result_none is None

    async def test_save_state_intervals_batch(
        self, storage_with_db: AreaOccupancyStorage
    ):
        """Test saving state intervals batch with in-memory database."""
        await storage_with_db.async_initialize()

        # First create area and entity
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(area_record)

        entity_record = {
            "entry_id": "test_entry_001",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": dt_util.utcnow(),
        }
        await storage_with_db.save_entity_config(entity_record)

        # Create intervals
        start_time = dt_util.utcnow()
        intervals = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "start": start_time,
                "end": start_time + timedelta(hours=1),
            },
            {
                "entity_id": "sensor.test",
                "state": "off",
                "start": start_time + timedelta(hours=1),
                "end": start_time + timedelta(hours=2),
            },
        ]

        # Save intervals
        saved_count = await storage_with_db.save_state_intervals_batch(intervals)
        assert saved_count == 2

        # Verify intervals were saved
        retrieved_intervals = await storage_with_db.get_historical_intervals(
            "sensor.test"
        )
        assert len(retrieved_intervals) == 2
        assert retrieved_intervals[0]["state"] == "on"
        assert retrieved_intervals[1]["state"] == "off"

    async def test_get_historical_intervals(
        self, storage_with_db: AreaOccupancyStorage
    ):
        """Test getting historical intervals with in-memory database."""
        await storage_with_db.async_initialize()

        # Setup test data
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(area_record)

        entity_record = {
            "entry_id": "test_entry_001",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": dt_util.utcnow(),
        }
        await storage_with_db.save_entity_config(entity_record)

        # Create and save intervals
        start_time = dt_util.utcnow()
        intervals = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "start": start_time,
                "end": start_time + timedelta(hours=1),
            },
        ]
        await storage_with_db.save_state_intervals_batch(intervals)

        # Test retrieval
        retrieved_intervals = await storage_with_db.get_historical_intervals(
            "sensor.test"
        )
        assert len(retrieved_intervals) == 1
        assert retrieved_intervals[0]["entity_id"] == "sensor.test"
        assert retrieved_intervals[0]["state"] == "on"

        # Test with time filter
        filtered_intervals = await storage_with_db.get_historical_intervals(
            "sensor.test",
            start_time=start_time - timedelta(minutes=30),
            end_time=start_time + timedelta(minutes=30),
        )
        assert len(filtered_intervals) == 1

        # Test with non-existent entity
        empty_intervals = await storage_with_db.get_historical_intervals(
            "non_existent.entity"
        )
        assert len(empty_intervals) == 0

    async def test_reset_entry_data(self, storage_with_db: AreaOccupancyStorage):
        """Test resetting entry data with in-memory database."""
        await storage_with_db.async_initialize()

        # Create test data
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(area_record)

        entity_record = {
            "entry_id": "test_entry_001",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": dt_util.utcnow(),
        }
        await storage_with_db.save_entity_config(entity_record)

        # Verify data exists
        area = await storage_with_db.get_area_occupancy("test_entry_001")
        assert area is not None

        configs = await storage_with_db.get_entity_configs("test_entry_001")
        assert len(configs) == 1

        # Reset entry data
        await storage_with_db.reset_entry_data("test_entry_001")

        # Verify data was removed
        area_after = await storage_with_db.get_area_occupancy("test_entry_001")
        assert area_after is None

        configs_after = await storage_with_db.get_entity_configs("test_entry_001")
        assert len(configs_after) == 0

    async def test_cleanup_old_intervals(self, storage_with_db: AreaOccupancyStorage):
        """Test cleaning up old intervals with in-memory database."""
        await storage_with_db.async_initialize()

        # Setup test data
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(area_record)

        entity_record = {
            "entry_id": "test_entry_001",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": dt_util.utcnow(),
        }
        await storage_with_db.save_entity_config(entity_record)

        # Create old and new intervals
        old_time = dt_util.utcnow() - timedelta(days=30)
        new_time = dt_util.utcnow()

        intervals = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "start": old_time,
                "end": old_time + timedelta(hours=1),
            },
            {
                "entity_id": "sensor.test",
                "state": "on",
                "start": new_time,
                "end": new_time + timedelta(hours=1),
            },
        ]
        await storage_with_db.save_state_intervals_batch(intervals)

        # Verify both intervals exist
        all_intervals = await storage_with_db.get_historical_intervals("sensor.test")
        assert len(all_intervals) == 2

        # Clean up old intervals (older than 7 days)
        cleaned_count = await storage_with_db.cleanup_old_intervals(retention_days=7)
        assert cleaned_count == 1

        # Verify only new interval remains
        remaining_intervals = await storage_with_db.get_historical_intervals(
            "sensor.test"
        )
        assert len(remaining_intervals) == 1
        assert remaining_intervals[0]["start"] == new_time

    async def test_database_cleanup_works(self, storage_with_db: AreaOccupancyStorage):
        """Test that database cleanup works correctly between tests."""
        await storage_with_db.async_initialize()

        # Add some test data
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(area_record)

        # Verify data was added
        stats = await storage_with_db.get_stats()
        assert stats["areas_count"] == 1

        # The cleanup fixture should run after this test
        # and clean up the data for the next test

    async def test_save_intervals_count_accuracy(
        self, storage_with_db: AreaOccupancyStorage
    ):
        """Test that interval saving returns accurate counts."""
        await storage_with_db.async_initialize()

        # First create area and entity
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(area_record)

        entity_record = {
            "entry_id": "test_entry_001",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": dt_util.utcnow(),
        }
        await storage_with_db.save_entity_config(entity_record)

        # Save intervals
        intervals = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "start": dt_util.utcnow(),
                "end": dt_util.utcnow() + timedelta(hours=1),
            },
            {
                "entity_id": "sensor.test",
                "state": "off",
                "start": dt_util.utcnow() + timedelta(hours=1),
                "end": dt_util.utcnow() + timedelta(hours=2),
            },
        ]

        # Save intervals and verify count
        saved_count = await storage_with_db.save_state_intervals_batch(intervals)
        assert saved_count == 2

        # Verify stats show correct count
        stats = await storage_with_db.get_stats()
        assert stats["intervals_count"] == 2

    async def test_database_schema_works(self, storage_with_db: AreaOccupancyStorage):
        """Test that the database schema is working correctly."""
        await storage_with_db.async_initialize()

        # Test that we can create and query areas
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        result = await storage_with_db.save_area_occupancy(area_record)
        assert result["entry_id"] == "test_entry_001"

        # Test that we can create and query entities
        entity_record = {
            "entry_id": "test_entry_001",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": dt_util.utcnow(),
        }
        result = await storage_with_db.save_entity_config(entity_record)
        assert result["entity_id"] == "sensor.test"

        # Test that we can save intervals
        intervals = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "start": dt_util.utcnow(),
                "end": dt_util.utcnow() + timedelta(hours=1),
            }
        ]
        saved_count = await storage_with_db.save_state_intervals_batch(intervals)
        assert saved_count == 1

        # Test that we can query intervals
        historical_intervals = await storage_with_db.get_historical_intervals(
            "sensor.test",
            start_time=dt_util.utcnow() - timedelta(days=1),
            end_time=dt_util.utcnow() + timedelta(days=1),
        )
        assert len(historical_intervals) == 1
        assert historical_intervals[0]["entity_id"] == "sensor.test"
        assert historical_intervals[0]["state"] == "on"

    async def test_database_isolation(self, storage_with_db: AreaOccupancyStorage):
        """Test that each test starts with a clean database."""
        await storage_with_db.async_initialize()

        # Verify that the database is empty at the start
        stats = await storage_with_db.get_stats()
        assert stats["areas_count"] == 0
        assert stats["entities_count"] == 0
        assert stats["intervals_count"] == 0
        assert stats["priors_count"] == 0

        # Add some data
        area_record = {
            "entry_id": "test_entry_001",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": dt_util.utcnow(),
            "updated_at": dt_util.utcnow(),
        }
        await storage_with_db.save_area_occupancy(area_record)

        # Verify data was added
        stats = await storage_with_db.get_stats()
        assert stats["areas_count"] == 1

        # The next test should start with a clean database
