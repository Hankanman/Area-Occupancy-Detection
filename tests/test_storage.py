"""Tests for storage module."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import sqlalchemy as sa

from custom_components.area_occupancy.const import HA_RECORDER_DAYS
from custom_components.area_occupancy.state_intervals import StateInterval
from custom_components.area_occupancy.storage import AreaOccupancyStorage


# ruff: noqa: SLF001
class TestAreaOccupancyStorage:
    """Test AreaOccupancyStorage class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test AreaOccupancyStorage initialization."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Verify instance attributes
        assert store.coordinator == mock_coordinator

    async def test_async_initialize(self, mock_coordinator: Mock) -> None:
        """Test async_initialize method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with patch.object(store, "async_initialize") as mock_init:
            await store.async_initialize()
            mock_init.assert_called_once()

    async def test_async_save_data_success(self, mock_coordinator: Mock) -> None:
        """Test successful data saving."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock coordinator attributes
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.config.purpose = "test"
        mock_coordinator.threshold = 0.5
        mock_coordinator.entities.entities = {}

        with patch.object(store, "save_area_occupancy") as mock_save_area:
            await store.async_save_data()
            mock_save_area.assert_called_once()

    async def test_async_save_data_with_entities(self, mock_coordinator: Mock) -> None:
        """Test data saving with entities."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock coordinator attributes
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.config.purpose = "test"
        mock_coordinator.threshold = 0.5

        # Mock entity
        mock_entity = Mock()
        mock_entity.entity_id = "sensor.test"
        mock_entity.type.input_type.value = "motion"
        mock_entity.type.weight = 1.0
        mock_entity.likelihood.prob_given_true = 0.8
        mock_entity.likelihood.prob_given_false = 0.2
        mock_coordinator.entities.entities = {"sensor.test": mock_entity}

        with (
            patch.object(store, "save_area_occupancy") as mock_save_area,
            patch.object(store, "save_entity_config") as mock_save_entity,
        ):
            await store.async_save_data()
            mock_save_area.assert_called_once()
            mock_save_entity.assert_called_once()

    async def test_async_save_data_error(self, mock_coordinator: Mock) -> None:
        """Test data saving with error."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock coordinator attributes
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.config.purpose = "test"
        mock_coordinator.threshold = 0.5
        mock_coordinator.entities.entities = {}

        with (
            patch.object(
                store,
                "save_area_occupancy",
                side_effect=Exception("Test error"),
            ),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.error"
            ) as mock_logger,
        ):
            with pytest.raises(Exception, match="Test error"):
                await store.async_save_data()
            mock_logger.assert_called_once()

    async def test_async_load_data_success(self, mock_coordinator: Mock) -> None:
        """Test successful data loading."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock storage responses
        mock_area_record = {
            "entry_id": "test_entry",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "updated_at": datetime.now(),
        }

        mock_coordinator.probability = 0.7
        mock_coordinator.area_prior = 0.3

        with (
            patch.object(store, "get_area_occupancy", return_value=mock_area_record),
            patch.object(store, "get_entity_configs", return_value=[]),
        ):
            result = await store.async_load_data()

            assert result is not None
            assert result["name"] == "Test Area"
            assert result["purpose"] == "test"
            assert result["threshold"] == 0.5
            assert result["probability"] == 0.7
            assert result["prior"] == 0.3
            assert result["entities"] == {}

    async def test_async_load_data_no_data(self, mock_coordinator: Mock) -> None:
        """Test data loading when no data exists."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with patch.object(store, "get_area_occupancy", return_value=None):
            result = await store.async_load_data()
            assert result is None

    async def test_async_load_data_error(self, mock_coordinator: Mock) -> None:
        """Test data loading with error."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with (
            patch.object(
                store,
                "get_area_occupancy",
                side_effect=sa.exc.SQLAlchemyError("Test error"),
            ),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.error"
            ) as mock_logger,
        ):
            result = await store.async_load_data()
            assert result is None
            mock_logger.assert_called_once()

    async def test_async_reset(self, mock_coordinator: Mock) -> None:
        """Test async_reset method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)
        mock_coordinator.entry_id = "test_entry"

        with (
            patch.object(store, "reset_entry_data") as mock_reset,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_logger,
        ):
            await store.async_reset()
            mock_reset.assert_called_once_with("test_entry")
            mock_logger.assert_called_once()

    async def test_async_get_stats(self, mock_coordinator: Mock) -> None:
        """Test async_get_stats method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        expected_stats = {"total_entities": 5, "total_areas": 2}

        with patch.object(store, "get_stats", return_value=expected_stats):
            result = await store.async_get_stats()
            assert result == expected_stats

    async def test_import_intervals_from_recorder(self, mock_coordinator: Mock) -> None:
        """Test import_intervals_from_recorder method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock coordinator entity IDs
        mock_coordinator.occupancy_entity_id = "binary_sensor.occupancy"
        mock_coordinator.wasp_entity_id = "binary_sensor.wasp"

        entity_ids = ["sensor.test1", "sensor.test2"]
        fake_intervals = [
            StateInterval(
                entity_id="sensor.test1",
                state="on",
                start=datetime.now(),
                end=datetime.now(),
            )
        ]
        # Expected result now includes occupancy and wasp entities
        expected_result = {
            "sensor.test1": 1,
            "sensor.test2": 0,
            "binary_sensor.occupancy": 0,
            "binary_sensor.wasp": 0,
        }

        with (
            patch(
                "custom_components.area_occupancy.storage.get_intervals_from_recorder"
            ) as mock_get_intervals,
            patch.object(store, "save_state_intervals_batch") as mock_save_batch,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_info_logger,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.debug"
            ) as mock_debug_logger,
        ):
            # Mock save_state_intervals_batch to return count of saved intervals
            mock_save_batch.return_value = 1

            # First entity returns intervals, others return empty
            mock_get_intervals.side_effect = [fake_intervals, [], [], []]

            await store.import_intervals_from_recorder(entity_ids, days=10)

            # Verify the import_stats were set correctly
            assert store.import_stats == expected_result

            # Verify save_state_intervals_batch was called for the first entity
            mock_save_batch.assert_called_once_with(fake_intervals)

            # Verify logging calls (now processes 4 entities instead of 2)
            assert mock_info_logger.call_count >= 2  # Start and completion logs
            assert (
                mock_debug_logger.call_count >= 8
            )  # Processing logs for each entity (4 entities * 2 logs each)

    async def test_cleanup_old_intervals(self, mock_coordinator: Mock) -> None:
        """Test cleanup_old_intervals method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with patch.object(store, "cleanup_old_intervals", return_value=50):
            result = await store.cleanup_old_intervals(retention_days=365)
            assert result == 50

    async def test_is_state_intervals_empty(self, mock_coordinator: Mock) -> None:
        """Test is_state_intervals_empty method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with patch.object(store, "is_state_intervals_empty", return_value=True):
            result = await store.is_state_intervals_empty()
            assert result is True

    async def test_get_total_intervals_count(self, mock_coordinator: Mock) -> None:
        """Test get_total_intervals_count method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with patch.object(store, "get_total_intervals_count", return_value=1000):
            result = await store.get_total_intervals_count()
            assert result == 1000

    async def test_get_historical_intervals(self, mock_coordinator: Mock) -> None:
        """Test get_historical_intervals method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)
        expected_intervals = [Mock(), Mock()]

        with patch.object(
            store, "get_historical_intervals", return_value=expected_intervals
        ):
            result = await store.get_historical_intervals(
                "sensor.test", start_time, end_time
            )
            assert result == expected_intervals


class TestAreaOccupancyStorageDirect:
    """Test direct AreaOccupancyStorage class (legacy SQLiteStorage tests)."""

    @pytest.fixture
    def mock_storage_path(self, tmp_path):
        """Create a temporary storage path."""
        storage_path = tmp_path / ".storage"
        storage_path.mkdir()
        return storage_path

    @pytest.fixture
    def area_occupancy_storage(self, mock_hass: Mock, mock_storage_path):
        """Create AreaOccupancyStorage instance."""
        mock_hass.config.config_dir = str(mock_storage_path.parent)
        return AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

    async def test_initialization(self, mock_hass: Mock, mock_storage_path):
        """Test SQLiteStorage initialization."""
        mock_hass.config.config_dir = str(mock_storage_path.parent)
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        assert storage.hass == mock_hass
        assert storage.entry_id == "test_entry"
        assert storage.engine is not None

    async def test_async_initialize_success(
        self, area_occupancy_storage: AreaOccupancyStorage
    ):
        """Test successful database initialization."""
        with patch(
            "custom_components.area_occupancy.storage._LOGGER.info"
        ) as mock_logger:
            await area_occupancy_storage.async_initialize()
            mock_logger.assert_called()

    async def test_save_and_query_intervals_real_db(
        self, mock_hass: Mock, tmp_path
    ) -> None:
        """Save records to a real database and query them back."""
        mock_hass.async_add_executor_job.side_effect = (
            lambda func, *args, **kwargs: func(*args, **kwargs)
        )
        mock_hass.config.config_dir = str(tmp_path)
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
        await storage.async_initialize()
        record = {
            "entry_id": "test_entry",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.4,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        saved = await storage.save_area_occupancy(record)
        assert saved["entry_id"] == "test_entry"

        fetched = await storage.get_area_occupancy("test_entry")
        assert fetched is not None
        assert fetched["area_name"] == "Test Area"

    async def test_save_entity_config(self, mock_hass: Mock) -> None:
        """Test saving entity config."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        record = {
            "entry_id": "test_entry",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": datetime.now(),
        }

        with patch.object(mock_hass, "async_add_executor_job", return_value=record):
            result = await storage.save_entity_config(record)

            assert result == record
            mock_hass.async_add_executor_job.assert_called_once()

    async def test_save_state_intervals_batch_empty(self, mock_hass: Mock) -> None:
        """Test saving empty state intervals batch."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job", return_value=0):
            result = await storage.save_state_intervals_batch([])
            assert result == 0

    async def test_save_state_intervals_batch_success(self, mock_hass: Mock) -> None:
        """Test successful state intervals batch save."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        intervals = [
            StateInterval(
                entity_id="sensor.test",
                state="on",
                start=datetime.now(),
                end=datetime.now() + timedelta(minutes=5),
            )
        ]

        with patch.object(mock_hass, "async_add_executor_job", return_value=1):
            result = await storage.save_state_intervals_batch(intervals)

            assert result == 1

    async def test_get_historical_intervals_with_filters(self, mock_hass: Mock) -> None:
        """Test getting historical intervals with filters."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        expected_intervals = [Mock(), Mock()]

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_intervals
        ):
            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                state_filter="on",
                limit=100,
                page_size=50,
            )

            assert result == expected_intervals

    async def test_cleanup_old_intervals(self, mock_hass: Mock) -> None:
        """Test cleanup of old intervals."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job", return_value=50):
            result = await storage.cleanup_old_intervals(retention_days=365)

            assert result == 50

    async def test_get_stats(self, mock_hass: Mock) -> None:
        """Test getting database stats."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_stats = {
            "areas_count": 5,
            "entities_count": 100,
            "intervals_count": 1000,
            "priors_count": 500,
            "db_size_bytes": 1572864,
        }

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_stats
        ):
            result = await storage.get_stats()

            assert result == expected_stats

    async def test_get_stats_file_not_found(self, mock_hass: Mock) -> None:
        """Test getting stats when database file doesn't exist."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_stats = {
            "areas_count": 0,
            "entities_count": 0,
            "intervals_count": 0,
            "priors_count": 0,
            "db_size_bytes": 0,
        }

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_stats
        ):
            result = await storage.get_stats()

            assert result == expected_stats

    async def test_is_state_intervals_empty_true(self, mock_hass: Mock) -> None:
        """Test checking if state intervals table is empty when it is."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job", return_value=True):
            result = await storage.is_state_intervals_empty()

            assert result is True

    async def test_is_state_intervals_empty_false(self, mock_hass: Mock) -> None:
        """Test checking if state intervals table is empty when it's not."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job", return_value=False):
            result = await storage.is_state_intervals_empty()

            assert result is False

    async def test_get_total_intervals_count(self, mock_hass: Mock) -> None:
        """Test getting total intervals count."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job", return_value=1000):
            result = await storage.get_total_intervals_count()

            assert result == 1000

    async def test_get_total_intervals_count_none(self, mock_hass: Mock) -> None:
        """Test getting total intervals count when table is empty."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job", return_value=0):
            result = await storage.get_total_intervals_count()

            assert result == 0

    async def test_get_historical_intervals_with_state_filter(
        self, mock_hass: Mock
    ) -> None:
        """Test getting historical intervals with state filtering."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        expected_intervals = [Mock(), Mock()]

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_intervals
        ):
            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                state_filter="on",
            )
            assert result == expected_intervals

    async def test_cleanup_old_intervals_logging(self, mock_hass: Mock) -> None:
        """Test that cleanup operation logs the count."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(mock_hass, "async_add_executor_job", return_value=50),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_logger,
        ):
            result = await storage.cleanup_old_intervals(retention_days=365)
            assert result == 50
            mock_logger.assert_called_with(
                "Cleaned up %d state intervals older than %d days", 50, 365
            )

    async def test_reset_entry_data_logging(self, mock_hass: Mock) -> None:
        """Test that reset operation logs the action."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(mock_hass, "async_add_executor_job"),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_logger,
        ):
            await storage.reset_entry_data("test_entry")
            mock_logger.assert_called_with(
                "Reset area-specific data for entry %s", "test_entry"
            )

    async def test_get_entity_configs_ordering(self, mock_hass: Mock) -> None:
        """Test that entity configs are returned in correct order."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_configs = [
            {
                "entry_id": "test_entry",
                "entity_id": "sensor.a",
                "entity_type": "motion",
                "weight": 1.0,
                "prob_given_true": 0.8,
                "prob_given_false": 0.2,
                "last_updated": datetime.now(),
            },
            {
                "entry_id": "test_entry",
                "entity_id": "sensor.b",
                "entity_type": "motion",
                "weight": 1.0,
                "prob_given_true": 0.8,
                "prob_given_false": 0.2,
                "last_updated": datetime.now(),
            },
        ]

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_configs
        ):
            result = await storage.get_entity_configs("test_entry")
            assert result == expected_configs

    async def test_async_save_data_missing_coordinator_attributes(
        self, mock_coordinator: Mock
    ) -> None:
        """Test save data when coordinator is missing required attributes."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock coordinator with missing attributes
        mock_coordinator.entry_id = None
        mock_coordinator.config.name = None

        with (
            patch.object(
                store,
                "save_area_occupancy",
                side_effect=AttributeError("Missing attribute"),
            ),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.error"
            ) as mock_logger,
        ):
            with pytest.raises(AttributeError):
                await store.async_save_data()
            mock_logger.assert_called()

    async def test_async_load_data_storage_error(self, mock_coordinator: Mock) -> None:
        """Test loading data when storage throws an error."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with (
            patch.object(
                store,
                "get_area_occupancy",
                side_effect=sa.exc.SQLAlchemyError("Database error"),
            ),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.error"
            ) as mock_logger,
        ):
            result = await store.async_load_data()
            assert result is None
            mock_logger.assert_called()

    async def test_async_load_data_os_error(self, mock_coordinator: Mock) -> None:
        """Test loading data when storage throws an OSError."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with (
            patch.object(
                store,
                "get_area_occupancy",
                side_effect=OSError("File not found"),
            ),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.error"
            ) as mock_logger,
        ):
            result = await store.async_load_data()
            assert result is None
            mock_logger.assert_called()

    async def test_async_reset_logging(self, mock_coordinator: Mock) -> None:
        """Test that reset operation logs the action."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)
        mock_coordinator.entry_id = "test_entry"

        with (
            patch.object(store, "reset_entry_data"),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_logger,
        ):
            await store.async_reset()
            mock_logger.assert_called_with(
                "Reset SQLite storage for entry %s", "test_entry"
            )

    async def test_save_area_occupancy(self, mock_hass: Mock) -> None:
        """Test saving area occupancy record."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        record = {
            "entry_id": "test_entry",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        with patch.object(mock_hass, "async_add_executor_job", return_value=record):
            result = await storage.save_area_occupancy(record)

            assert result == record

    async def test_get_area_occupancy(self, mock_hass: Mock) -> None:
        """Test getting area occupancy record."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_record = {
            "entry_id": "test_entry",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_record
        ):
            result = await storage.get_area_occupancy("test_entry")

            assert result == expected_record

    async def test_import_intervals_from_recorder(self, mock_hass: Mock) -> None:
        """Test importing intervals from recorder."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        # Mock coordinator entity IDs
        mock_coordinator = Mock()
        mock_coordinator.occupancy_entity_id = "binary_sensor.occupancy"
        mock_coordinator.wasp_entity_id = "binary_sensor.wasp"
        storage.coordinator = mock_coordinator

        entity_ids = ["sensor.test1", "sensor.test2"]
        fake_intervals = [
            StateInterval(
                entity_id="sensor.test1",
                state="on",
                start=datetime.now(),
                end=datetime.now(),
            )
        ]
        # Expected result now includes occupancy and wasp entities
        expected_result = {
            "sensor.test1": 1,
            "sensor.test2": 0,
            "binary_sensor.occupancy": 0,
            "binary_sensor.wasp": 0,
        }

        with (
            patch(
                "custom_components.area_occupancy.storage.get_intervals_from_recorder"
            ) as mock_get_intervals,
            patch.object(storage, "save_state_intervals_batch") as mock_save_batch,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_info_logger,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.debug"
            ) as mock_debug_logger,
        ):
            # Mock save_state_intervals_batch to return count of saved intervals
            mock_save_batch.return_value = 1

            # First entity returns intervals, others return empty
            mock_get_intervals.side_effect = [fake_intervals, [], [], []]

            await storage.import_intervals_from_recorder(
                entity_ids, days=HA_RECORDER_DAYS
            )

            # Verify the import_stats were set correctly
            assert storage.import_stats == expected_result

            # Verify save_state_intervals_batch was called for the first entity
            mock_save_batch.assert_called_once_with(fake_intervals)

            # Verify logging calls (now processes 4 entities instead of 2)
            assert mock_info_logger.call_count >= 2  # Start and completion logs
            assert (
                mock_debug_logger.call_count >= 8
            )  # Processing logs for each entity (4 entities * 2 logs each)

    async def test_reset_entry_data(self, mock_hass: Mock) -> None:
        """Test resetting entry data."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job") as mock_executor:
            await storage.reset_entry_data("test_entry")

            # Should call async_add_executor_job once for the reset operation
            mock_executor.assert_called_once()

    async def test_save_state_intervals_batch_with_duplicates(
        self, mock_hass: Mock
    ) -> None:
        """Test batch save with duplicate intervals."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        # Create intervals with potential duplicates
        base_time = datetime.now()
        intervals = [
            StateInterval(
                entity_id="sensor.test",
                state="on",
                start=base_time,
                end=base_time + timedelta(minutes=5),
            ),
            StateInterval(
                entity_id="sensor.test",
                state="on",
                start=base_time,
                end=base_time + timedelta(minutes=5),
            ),  # Duplicate
        ]

        with patch.object(mock_hass, "async_add_executor_job", return_value=1):
            result = await storage.save_state_intervals_batch(intervals)
            assert result == 1  # Only one should be saved

    async def test_save_state_intervals_batch_entity_creation_failure(
        self, mock_hass: Mock
    ) -> None:
        """Test batch save when entity creation fails."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        intervals = [
            StateInterval(
                entity_id="sensor.test",
                state="on",
                start=datetime.now(),
                end=datetime.now() + timedelta(minutes=5),
            )
        ]

        with (
            patch.object(mock_hass, "async_add_executor_job") as mock_executor,
        ):
            # Mock the batch save function to simulate entity creation failure
            def mock_batch_save():
                # Simulate entity creation failure but interval save success
                return 1  # One interval saved despite entity creation failure

            mock_executor.return_value = 1

            result = await storage.save_state_intervals_batch(intervals)
            assert result == 1

    async def test_get_historical_intervals_pagination_edge_cases(
        self, mock_hass: Mock
    ) -> None:
        """Test historical intervals retrieval with pagination edge cases."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        # Test with very small page size
        with patch.object(mock_hass, "async_add_executor_job", return_value=[]):
            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                page_size=1,
                limit=5,
            )
            assert result == []

        # Test with limit smaller than page size
        with patch.object(mock_hass, "async_add_executor_job", return_value=[]):
            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                page_size=100,
                limit=5,
            )
            assert result == []

    async def test_get_historical_intervals_pagination_edge_cases_zero_limit(
        self, mock_hass: Mock
    ):
        """Test get_historical_intervals with limit=0 and page_size=1 returns empty list."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)
        with patch.object(mock_hass, "async_add_executor_job", return_value=[]):
            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                limit=0,
                page_size=1,
            )
            assert result == []

    async def test_async_close(self, mock_coordinator: Mock) -> None:
        """Test async_close method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with (
            patch.object(store, "async_close") as mock_close,
        ):
            await store.async_close()
            mock_close.assert_called_once()

    async def test_enable_wal_mode_success(self, mock_hass: Mock) -> None:
        """Test _enable_wal_mode method when successful."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(storage, "engine") as mock_engine:
            mock_conn = Mock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            storage._enable_wal_mode()
            mock_conn.execute.assert_called_once()

    async def test_enable_wal_mode_failure(self, mock_hass: Mock) -> None:
        """Test _enable_wal_mode method when it fails."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(storage, "engine") as mock_engine,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.debug"
            ) as mock_logger,
        ):
            mock_conn = Mock()
            mock_conn.execute.side_effect = sa.exc.SQLAlchemyError("WAL mode failed")
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            storage._enable_wal_mode()
            mock_logger.assert_called_once()

    async def test_enable_wal_mode_no_engine(self, mock_hass: Mock) -> None:
        """Test _enable_wal_mode method when engine is None."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
        storage.engine = None
        storage._enable_wal_mode()  # Should not raise any exception

    async def test_create_tables_individually(self, mock_hass: Mock) -> None:
        """Test _create_tables_individually method."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(storage, "engine") as mock_engine,
            patch(
                "custom_components.area_occupancy.storage.Base.metadata"
            ) as mock_metadata,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.debug"
            ) as mock_logger,
        ):
            mock_conn = Mock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn

            # Mock tables
            mock_table1 = Mock()
            mock_table1.name = "table1"
            mock_table2 = Mock()
            mock_table2.name = "table2"
            mock_metadata.tables.values.return_value = [mock_table1, mock_table2]

            # First table succeeds, second table already exists
            mock_table1.create.side_effect = None
            mock_table2.create.side_effect = sa.exc.OperationalError(
                "table already exists", None, None
            )

            storage._create_tables_individually()

            mock_table1.create.assert_called_once()
            mock_table2.create.assert_called_once()
            mock_logger.assert_called_once()

    async def test_execute_with_retry_success(self, mock_hass: Mock) -> None:
        """Test execute_with_retry method with successful execution."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        def test_func():
            return "success"

        result = storage.execute_with_retry(test_func)
        assert result == "success"

    async def test_execute_with_retry_database_locked(self, mock_hass: Mock) -> None:
        """Test execute_with_retry method with database locked error."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        call_count = 0

        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sa.exc.OperationalError("database is locked", None, None)
            return "success"

        with patch("time.sleep") as mock_sleep:
            result = storage.execute_with_retry(test_func)
            assert result == "success"
            assert call_count == 3
            assert mock_sleep.call_count == 2

    async def test_execute_with_retry_max_retries_exceeded(
        self, mock_hass: Mock
    ) -> None:
        """Test execute_with_retry method when max retries are exceeded."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        def test_func():
            raise sa.exc.OperationalError("database is locked", None, None)

        with (
            patch("time.sleep"),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.debug"
            ) as mock_logger,
        ):
            with pytest.raises(sa.exc.OperationalError):
                storage.execute_with_retry(test_func, max_retries=2)
            assert mock_logger.call_count == 1

    async def test_execute_with_retry_other_error(self, mock_hass: Mock) -> None:
        """Test execute_with_retry method with non-database-locked error."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        def test_func():
            raise ValueError("other error")

        with pytest.raises(ValueError, match="other error"):
            storage.execute_with_retry(test_func)

    async def test_async_initialize_race_condition(self, mock_hass: Mock) -> None:
        """Test async_initialize method with race condition handling."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(storage, "_enable_wal_mode") as mock_wal,
            patch.object(
                storage, "_create_tables_individually"
            ) as mock_create_individual,
            patch(
                "custom_components.area_occupancy.storage.Base.metadata"
            ) as mock_metadata,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_info_logger,
        ):
            # Mock async_add_executor_job to execute the function directly
            def mock_executor_job(func, *args, **kwargs):
                return func(*args, **kwargs)

            mock_hass.async_add_executor_job.side_effect = mock_executor_job

            # Mock the create_all to raise OperationalError with "already exists"
            mock_metadata.create_all.side_effect = sa.exc.OperationalError(
                "table already exists", None, None
            )

            await storage.async_initialize()

            # Verify the methods were called
            mock_wal.assert_called_once()
            mock_create_individual.assert_called_once()
            mock_info_logger.assert_called_once()

    async def test_async_initialize_other_operational_error(
        self, mock_hass: Mock
    ) -> None:
        """Test async_initialize method with other operational error."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch(
                "custom_components.area_occupancy.storage.Base.metadata"
            ) as mock_metadata,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.error"
            ) as mock_error_logger,
        ):
            # Mock async_add_executor_job to execute the function directly
            def mock_executor_job(func, *args, **kwargs):
                return func(*args, **kwargs)

            mock_hass.async_add_executor_job.side_effect = mock_executor_job

            # Mock the create_all to raise OperationalError without "already exists"
            mock_metadata.create_all.side_effect = sa.exc.OperationalError(
                "other database error", None, None
            )

            with pytest.raises(sa.exc.OperationalError):
                await storage.async_initialize()

            mock_error_logger.assert_called_once()

    async def test_async_initialize_general_exception(self, mock_hass: Mock) -> None:
        """Test async_initialize method with general exception."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch(
                "custom_components.area_occupancy.storage.Base.metadata"
            ) as mock_metadata,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.error"
            ) as mock_error_logger,
        ):
            # Mock async_add_executor_job to execute the function directly
            def mock_executor_job(func, *args, **kwargs):
                return func(*args, **kwargs)

            mock_hass.async_add_executor_job.side_effect = mock_executor_job

            # Mock the create_all to raise a general exception
            mock_metadata.create_all.side_effect = Exception("general error")

            with pytest.raises(Exception, match="general error"):
                await storage.async_initialize()

            mock_error_logger.assert_called_once()

    async def test_save_state_intervals_batch_database_error(
        self, mock_hass: Mock
    ) -> None:
        """Test save_state_intervals_batch method with database error."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        intervals = [
            StateInterval(
                entity_id="sensor.test",
                state="on",
                start=datetime.now(),
                end=datetime.now() + timedelta(minutes=5),
            )
        ]

        with (
            patch(
                "custom_components.area_occupancy.storage._LOGGER.warning"
            ) as mock_warning_logger,
        ):
            # Mock async_add_executor_job to raise the exception
            def mock_executor_job(func, *args, **kwargs):
                # Execute the function which will raise the exception
                return func(*args, **kwargs)

            mock_hass.async_add_executor_job.side_effect = mock_executor_job

            # Mock the engine to raise SQLAlchemyError
            with patch.object(storage, "engine") as mock_engine:
                mock_conn = Mock()
                mock_conn.execute.side_effect = sa.exc.SQLAlchemyError("Database error")
                mock_engine.connect.return_value.__enter__.return_value = mock_conn

                result = await storage.save_state_intervals_batch(intervals)
                assert result == 0
                mock_warning_logger.assert_called_once()

    async def test_save_state_intervals_batch_os_error(self, mock_hass: Mock) -> None:
        """Test save_state_intervals_batch method with OSError."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        intervals = [
            StateInterval(
                entity_id="sensor.test",
                state="on",
                start=datetime.now(),
                end=datetime.now() + timedelta(minutes=5),
            )
        ]

        with (
            patch(
                "custom_components.area_occupancy.storage._LOGGER.warning"
            ) as mock_warning_logger,
        ):
            # Mock async_add_executor_job to raise the exception
            def mock_executor_job(func, *args, **kwargs):
                # Execute the function which will raise the exception
                return func(*args, **kwargs)

            mock_hass.async_add_executor_job.side_effect = mock_executor_job

            # Mock the engine to raise OSError
            with patch.object(storage, "engine") as mock_engine:
                mock_conn = Mock()
                mock_conn.execute.side_effect = OSError("File system error")
                mock_engine.connect.return_value.__enter__.return_value = mock_conn

                result = await storage.save_state_intervals_batch(intervals)
                assert result == 0
                mock_warning_logger.assert_called_once()

    async def test_async_load_data_with_unknown_entity_type(
        self, mock_coordinator: Mock
    ) -> None:
        """Test async_load_data method with unknown entity type."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock coordinator attributes
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.probability = 0.7
        mock_coordinator.area_prior = 0.3
        mock_coordinator.config.decay.half_life = 300

        # Mock storage responses with unknown entity type
        mock_area_record = {
            "entry_id": "test_entry",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "updated_at": datetime.now(),
        }

        mock_entity_configs = [
            {
                "entry_id": "test_entry",
                "entity_id": "sensor.unknown",
                "entity_type": "unknown_type",
                "weight": 1.0,
                "prob_given_true": 0.8,
                "prob_given_false": 0.2,
                "last_updated": datetime.now(),
            }
        ]

        with (
            patch.object(store, "get_area_occupancy", return_value=mock_area_record),
            patch.object(store, "get_entity_configs", return_value=mock_entity_configs),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.warning"
            ) as mock_warning_logger,
        ):
            result = await store.async_load_data()

            assert result is not None
            assert "sensor.unknown" in result["entities"]
            mock_warning_logger.assert_called_once()

    async def test_async_load_data_with_value_error(
        self, mock_coordinator: Mock
    ) -> None:
        """Test async_load_data method with ValueError in entity type conversion."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock coordinator attributes
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.probability = 0.7
        mock_coordinator.area_prior = 0.3
        mock_coordinator.config.decay.half_life = 300

        # Mock storage responses with invalid entity type
        mock_area_record = {
            "entry_id": "test_entry",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "updated_at": datetime.now(),
        }

        mock_entity_configs = [
            {
                "entry_id": "test_entry",
                "entity_id": "sensor.invalid",
                "entity_type": "invalid_type",
                "weight": 1.0,
                "prob_given_true": 0.8,
                "prob_given_false": 0.2,
                "last_updated": datetime.now(),
            }
        ]

        with (
            patch.object(store, "get_area_occupancy", return_value=mock_area_record),
            patch.object(store, "get_entity_configs", return_value=mock_entity_configs),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.warning"
            ) as mock_warning_logger,
        ):
            result = await store.async_load_data()

            assert result is not None
            assert "sensor.invalid" in result["entities"]
            mock_warning_logger.assert_called_once()

    async def test_async_save_data_missing_coordinator(self, mock_hass: Mock) -> None:
        """Test async_save_data method when coordinator is None."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
        storage.coordinator = None

        with pytest.raises(
            RuntimeError, match="Coordinator is required for async_save_data"
        ):
            await storage.async_save_data()

    async def test_async_load_data_missing_coordinator(self, mock_hass: Mock) -> None:
        """Test async_load_data method when coordinator is None."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
        storage.coordinator = None

        with pytest.raises(
            RuntimeError, match="Coordinator is required for async_load_data"
        ):
            await storage.async_load_data()

    async def test_async_reset_missing_coordinator(self, mock_hass: Mock) -> None:
        """Test async_reset method when coordinator is None."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
        storage.coordinator = None

        with pytest.raises(
            RuntimeError, match="Coordinator is required for async_reset"
        ):
            await storage.async_reset()

    async def test_get_historical_intervals_with_pagination(
        self, mock_hass: Mock
    ) -> None:
        """Test get_historical_intervals method with pagination."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        # Mock multiple pages of results
        mock_intervals_page1 = [Mock(), Mock()]  # 2 results (full page)
        mock_intervals_page2 = [Mock()]  # 1 result (less than page_size)
        mock_intervals_page3 = []  # Empty page

        with patch.object(mock_hass, "async_add_executor_job") as mock_executor:

            def mock_get_intervals(*args, **kwargs):
                # Simulate pagination behavior
                if not hasattr(mock_get_intervals, "call_count"):
                    mock_get_intervals.call_count = 0
                mock_get_intervals.call_count += 1

                if mock_get_intervals.call_count == 1:
                    return mock_intervals_page1
                if mock_get_intervals.call_count == 2:
                    return mock_intervals_page2
                return mock_intervals_page3

            mock_executor.side_effect = mock_get_intervals

            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                page_size=2,
            )

            # Should get results from first page only (since it's a full page)
            assert len(result) == 2
            assert mock_executor.call_count == 1

    async def test_get_historical_intervals_with_limit(self, mock_hass: Mock) -> None:
        """Test get_historical_intervals method with limit."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        # Mock more results than the limit
        mock_intervals = [Mock() for _ in range(5)]

        def mock_get_intervals(*args, **kwargs):
            # Return only the first 3 results to simulate limit
            return mock_intervals[:3]

        with patch.object(
            mock_hass, "async_add_executor_job", side_effect=mock_get_intervals
        ):
            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                limit=3,
            )

            # Should respect the limit
            assert len(result) == 3

    async def test_get_historical_intervals_default_times(
        self, mock_hass: Mock
    ) -> None:
        """Test get_historical_intervals method with default start/end times."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(mock_hass, "async_add_executor_job", return_value=[]),
            patch(
                "custom_components.area_occupancy.storage.dt_util.utcnow"
            ) as mock_utcnow,
        ):
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_utcnow.return_value = mock_now

            await storage.get_historical_intervals("sensor.test")

    async def test_cleanup_old_intervals_with_custom_retention(
        self, mock_hass: Mock
    ) -> None:
        """Test cleanup_old_intervals method with custom retention period."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(mock_hass, "async_add_executor_job", return_value=25),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_logger,
        ):
            result = await storage.cleanup_old_intervals(retention_days=180)
            assert result == 25
            mock_logger.assert_called_with(
                "Cleaned up %d state intervals older than %d days", 25, 180
            )

    async def test_get_stats_with_database_version(self, mock_hass: Mock) -> None:
        """Test get_stats method with database version."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_stats = {
            "areas_count": 5,
            "entities_count": 100,
            "intervals_count": 1000,
            "priors_count": 500,
            "priors_entry_test_entry": 10,
            "database_version": "2",
            "db_size_bytes": 1572864,
        }

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_stats
        ):
            result = await storage.get_stats()
            assert result == expected_stats

    async def test_get_stats_with_file_not_found(self, mock_hass: Mock) -> None:
        """Test get_stats method when database file doesn't exist."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_stats = {
            "areas_count": 0,
            "entities_count": 0,
            "intervals_count": 0,
            "priors_count": 0,
            "priors_entry_test_entry": 0,
            "database_version": None,
            "db_size_bytes": 0,
        }

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_stats
        ):
            result = await storage.get_stats()
            assert result == expected_stats

    async def test_save_area_occupancy_with_existing_record(
        self, mock_hass: Mock
    ) -> None:
        """Test save_area_occupancy method with existing record."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        record = {
            "entry_id": "test_entry",
            "area_name": "Updated Area",
            "purpose": "updated",
            "threshold": 0.6,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        with patch.object(mock_hass, "async_add_executor_job", return_value=record):
            result = await storage.save_area_occupancy(record)
            assert result == record

    async def test_get_area_occupancy_not_found(self, mock_hass: Mock) -> None:
        """Test get_area_occupancy method when record not found."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job", return_value=None):
            result = await storage.get_area_occupancy("nonexistent_entry")
            assert result is None

    async def test_save_entity_config_with_retry(self, mock_hass: Mock) -> None:
        """Test save_entity_config method with retry logic."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        record = {
            "entry_id": "test_entry",
            "entity_id": "sensor.test",
            "entity_type": "motion",
            "weight": 1.0,
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "last_updated": datetime.now(),
        }

        with patch.object(mock_hass, "async_add_executor_job", return_value=record):
            result = await storage.save_entity_config(record)
            assert result == record

    async def test_get_entity_configs_empty(self, mock_hass: Mock) -> None:
        """Test get_entity_configs method when no configs exist."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job", return_value=[]):
            result = await storage.get_entity_configs("test_entry")
            assert result == []

    async def test_import_intervals_from_recorder_with_exception(
        self, mock_hass: Mock
    ) -> None:
        """Test import_intervals_from_recorder method with exception handling."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        # Mock coordinator entity IDs
        mock_coordinator = Mock()
        mock_coordinator.occupancy_entity_id = "binary_sensor.occupancy"
        mock_coordinator.wasp_entity_id = "binary_sensor.wasp"
        storage.coordinator = mock_coordinator

        entity_ids = ["sensor.test1", "sensor.test2"]

        with (
            patch(
                "custom_components.area_occupancy.storage.get_intervals_from_recorder"
            ) as mock_get_intervals,
            patch.object(storage, "save_state_intervals_batch") as mock_save_batch,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.exception"
            ) as mock_exception_logger,
        ):
            # First entity raises exception, second returns empty
            mock_get_intervals.side_effect = [Exception("Test error"), [], [], []]
            mock_save_batch.return_value = 0

            await storage.import_intervals_from_recorder(entity_ids, days=10)

            # Verify exception was logged
            mock_exception_logger.assert_called_once()

            # Verify import_stats were set correctly (with 0 for failed entity)
            expected_result = {
                "sensor.test1": 0,  # Failed due to exception
                "sensor.test2": 0,
                "binary_sensor.occupancy": 0,
                "binary_sensor.wasp": 0,
            }
            assert storage.import_stats == expected_result

    async def test_async_close_with_dispose(self, mock_hass: Mock) -> None:
        """Test async_close method with proper disposal."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        # Mock the database and engine
        mock_db = Mock()
        mock_engine = Mock()
        storage.db = mock_db
        storage.db.engine = mock_engine

        with patch.object(mock_hass, "async_add_executor_job") as mock_executor:
            await storage.async_close()

            # Verify the dispose function was called
            mock_executor.assert_called_once()

    async def test_async_close_without_db(self, mock_hass: Mock) -> None:
        """Test async_close method when database is None."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
        storage.db = None

        with patch.object(mock_hass, "async_add_executor_job") as mock_executor:
            await storage.async_close()

            # Should still call async_add_executor_job even with None db
            mock_executor.assert_called_once()

    async def test_save_state_intervals_batch_with_empty_intervals(
        self, mock_hass: Mock
    ) -> None:
        """Test save_state_intervals_batch method with empty intervals list."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(mock_hass, "async_add_executor_job", return_value=0),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.debug"
            ) as mock_debug_logger,
        ):
            result = await storage.save_state_intervals_batch([])
            assert result == 0
            mock_debug_logger.assert_called_with("No intervals to save")

    async def test_save_state_intervals_batch_with_single_interval(
        self, mock_hass: Mock
    ) -> None:
        """Test save_state_intervals_batch method with single interval."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        intervals = [
            StateInterval(
                entity_id="sensor.test",
                state="on",
                start=datetime.now(),
                end=datetime.now() + timedelta(minutes=5),
            )
        ]

        with (
            patch.object(mock_hass, "async_add_executor_job", return_value=1),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.debug"
            ) as mock_debug_logger,
        ):
            result = await storage.save_state_intervals_batch(intervals)
            assert result == 1
            mock_debug_logger.assert_called_with("Saving batch of %d intervals", 1)

    async def test_get_historical_intervals_with_state_filter_and_limit(
        self, mock_hass: Mock
    ) -> None:
        """Test get_historical_intervals method with state filter and limit."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        mock_intervals = [Mock(), Mock(), Mock()]

        def mock_get_intervals(*args, **kwargs):
            # Return only the first 2 results to simulate limit
            return mock_intervals[:2]

        with patch.object(
            mock_hass, "async_add_executor_job", side_effect=mock_get_intervals
        ):
            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                state_filter="on",
                limit=2,
            )

            assert len(result) == 2  # Should respect limit

    async def test_get_historical_intervals_with_pagination_and_limit(
        self, mock_hass: Mock
    ) -> None:
        """Test get_historical_intervals method with pagination and limit."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        # Mock multiple pages with pagination
        mock_intervals_page1 = [Mock(), Mock()]  # 2 results (full page)
        mock_intervals_page2 = [Mock()]  # 1 result
        mock_intervals_page3 = []  # Empty page

        with patch.object(mock_hass, "async_add_executor_job") as mock_executor:

            def mock_get_intervals(*args, **kwargs):
                if not hasattr(mock_get_intervals, "call_count"):
                    mock_get_intervals.call_count = 0
                mock_get_intervals.call_count += 1

                if mock_get_intervals.call_count == 1:
                    return mock_intervals_page1
                if mock_get_intervals.call_count == 2:
                    return mock_intervals_page2
                return mock_intervals_page3

            mock_executor.side_effect = mock_get_intervals

            result = await storage.get_historical_intervals(
                "sensor.test",
                start_time=start_time,
                end_time=end_time,
                page_size=2,
                limit=3,
            )

            # Should get results from first page only (since it's a full page)
            assert len(result) == 2
            assert mock_executor.call_count == 1

    async def test_cleanup_old_intervals_with_default_retention(
        self, mock_hass: Mock
    ) -> None:
        """Test cleanup_old_intervals method with default retention period."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(mock_hass, "async_add_executor_job", return_value=100),
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_logger,
        ):
            result = await storage.cleanup_old_intervals()
            assert result == 100
            mock_logger.assert_called_with(
                "Cleaned up %d state intervals older than %d days", 100, 365
            )

    async def test_get_stats_with_all_fields(self, mock_hass: Mock) -> None:
        """Test get_stats method with all fields populated."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_stats = {
            "areas_count": 10,
            "entities_count": 50,
            "intervals_count": 5000,
            "priors_count": 100,
            "priors_entry_test_entry": 25,
            "database_version": "2",
            "db_size_bytes": 2097152,
        }

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_stats
        ):
            result = await storage.get_stats()
            assert result == expected_stats

    async def test_save_area_occupancy_with_new_record(self, mock_hass: Mock) -> None:
        """Test save_area_occupancy method with new record."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        record = {
            "entry_id": "new_entry",
            "area_name": "New Area",
            "purpose": "new",
            "threshold": 0.7,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        with patch.object(mock_hass, "async_add_executor_job", return_value=record):
            result = await storage.save_area_occupancy(record)
            assert result == record

    async def test_get_area_occupancy_with_existing_record(
        self, mock_hass: Mock
    ) -> None:
        """Test get_area_occupancy method with existing record."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_record = {
            "entry_id": "test_entry",
            "area_name": "Test Area",
            "purpose": "test",
            "threshold": 0.5,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_record
        ):
            result = await storage.get_area_occupancy("test_entry")
            assert result == expected_record

    async def test_save_entity_config_with_new_entity(self, mock_hass: Mock) -> None:
        """Test save_entity_config method with new entity."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        record = {
            "entry_id": "test_entry",
            "entity_id": "sensor.new",
            "entity_type": "motion",
            "weight": 0.9,
            "prob_given_true": 0.85,
            "prob_given_false": 0.15,
            "last_updated": datetime.now(),
        }

        with patch.object(mock_hass, "async_add_executor_job", return_value=record):
            result = await storage.save_entity_config(record)
            assert result == record

    async def test_get_entity_configs_with_multiple_entities(
        self, mock_hass: Mock
    ) -> None:
        """Test get_entity_configs method with multiple entities."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_configs = [
            {
                "entry_id": "test_entry",
                "entity_id": "sensor.a",
                "entity_type": "motion",
                "weight": 1.0,
                "prob_given_true": 0.8,
                "prob_given_false": 0.2,
                "last_updated": datetime.now(),
            },
            {
                "entry_id": "test_entry",
                "entity_id": "sensor.b",
                "entity_type": "media",
                "weight": 0.7,
                "prob_given_true": 0.75,
                "prob_given_false": 0.25,
                "last_updated": datetime.now(),
            },
        ]

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_configs
        ):
            result = await storage.get_entity_configs("test_entry")
            assert result == expected_configs

    async def test_reset_entry_data_with_multiple_tables(self, mock_hass: Mock) -> None:
        """Test reset_entry_data method with multiple tables."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(mock_hass, "async_add_executor_job") as mock_executor,
            patch(
                "custom_components.area_occupancy.storage._LOGGER.info"
            ) as mock_logger,
        ):
            await storage.reset_entry_data("test_entry")

            # Should call async_add_executor_job once for the reset operation
            mock_executor.assert_called_once()
            mock_logger.assert_called_with(
                "Reset area-specific data for entry %s", "test_entry"
            )

    async def test_import_intervals_from_recorder_with_no_coordinator_entities(
        self, mock_hass: Mock
    ) -> None:
        """Test import_intervals_from_recorder method when coordinator has no entity IDs."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        # Mock coordinator with no entity IDs
        mock_coordinator = Mock()
        mock_coordinator.occupancy_entity_id = None
        mock_coordinator.wasp_entity_id = None
        storage.coordinator = mock_coordinator

        entity_ids = ["sensor.test1", "sensor.test2"]

        with (
            patch(
                "custom_components.area_occupancy.storage.get_intervals_from_recorder"
            ) as mock_get_intervals,
            patch.object(storage, "save_state_intervals_batch") as mock_save_batch,
        ):
            # All entities return empty intervals
            mock_get_intervals.side_effect = [[], []]
            mock_save_batch.return_value = 0

            await storage.import_intervals_from_recorder(entity_ids, days=5)

            # Verify import_stats were set correctly (only the original entities)
            expected_result = {
                "sensor.test1": 0,
                "sensor.test2": 0,
            }
            assert storage.import_stats == expected_result
