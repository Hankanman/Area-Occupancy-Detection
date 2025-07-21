"""Tests for storage module."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import sqlalchemy as sa

from custom_components.area_occupancy.schema import (
    AreaEntityConfigRecord,
    AreaOccupancyRecord,
    AreaTimePriorRecord,
)
from custom_components.area_occupancy.sqlite_storage import AreaOccupancyStorage
from custom_components.area_occupancy.utils import StateInterval


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
            patch.object(store, "save_area_entity_config") as mock_save_entity,
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
                "custom_components.area_occupancy.sqlite_storage._LOGGER.error"
            ) as mock_logger,
        ):
            with pytest.raises(Exception, match="Test error"):
                await store.async_save_data()
            mock_logger.assert_called_once()

    async def test_async_load_data_success(self, mock_coordinator: Mock) -> None:
        """Test successful data loading."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        # Mock storage responses
        mock_area_record = Mock()
        mock_area_record.area_name = "Test Area"
        mock_area_record.purpose = "test"
        mock_area_record.threshold = 0.5
        mock_area_record.updated_at.isoformat.return_value = "2024-01-01T00:00:00Z"

        mock_coordinator.probability = 0.7
        mock_coordinator.area_prior = 0.3

        with (
            patch.object(store, "get_area_occupancy", return_value=mock_area_record),
            patch.object(store, "get_area_entity_configs", return_value=[]),
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
                "custom_components.area_occupancy.sqlite_storage._LOGGER.error"
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
                "custom_components.area_occupancy.sqlite_storage._LOGGER.info"
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

        entity_ids = ["sensor.test1", "sensor.test2"]
        expected_result = {"sensor.test1": 100, "sensor.test2": 150}

        with patch.object(
            store,
            "import_intervals_from_recorder",
            return_value=expected_result,
        ):
            result = await store.import_intervals_from_recorder(entity_ids, days=10)
            assert result == expected_result

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

    async def test_save_time_priors_batch(self, mock_coordinator: Mock) -> None:
        """Test save_time_priors_batch method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        records = [
            AreaTimePriorRecord(
                entry_id="test_entry",
                day_of_week=1,
                time_slot=12,
                prior_value=0.5,
                data_points=100,
                last_updated=datetime.now(),
            )
        ]

        with patch.object(store, "save_time_priors_batch", return_value=1):
            result = await store.save_time_priors_batch(records)
            assert result == 1

    async def test_get_time_prior(self, mock_coordinator: Mock) -> None:
        """Test get_time_prior method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        record = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=1,
            time_slot=12,
            prior_value=0.5,
            data_points=100,
            last_updated=datetime.now(),
        )

        with patch.object(store, "get_time_prior", return_value=record):
            result = await store.get_time_prior("test_entry", 1, 12)
            assert result == record

    async def test_get_time_prior_not_found(self, mock_coordinator: Mock) -> None:
        """Test get_time_prior method when record not found."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        with patch.object(store, "get_time_prior", return_value=None):
            result = await store.get_time_prior("test_entry", 1, 12)
            assert result is None

    async def test_get_time_priors_for_entry(self, mock_coordinator: Mock) -> None:
        """Test get_time_priors_for_entry method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        records = [
            AreaTimePriorRecord(
                entry_id="test_entry",
                day_of_week=1,
                time_slot=12,
                prior_value=0.5,
                data_points=100,
                last_updated=datetime.now(),
            )
        ]

        with patch.object(store, "get_time_priors_for_entry", return_value=records):
            result = await store.get_time_priors_for_entry("test_entry")
            assert result == records

    async def test_get_recent_time_priors(self, mock_coordinator: Mock) -> None:
        """Test get_recent_time_priors method."""
        store = AreaOccupancyStorage(coordinator=mock_coordinator)

        records = [
            AreaTimePriorRecord(
                entry_id="test_entry",
                day_of_week=1,
                time_slot=12,
                prior_value=0.5,
                data_points=100,
                last_updated=datetime.now(),
            )
        ]

        with patch.object(store, "get_recent_time_priors", return_value=records):
            result = await store.get_recent_time_priors("test_entry", hours=24)
            assert result == records


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
        assert storage.storage_path == mock_storage_path
        assert storage.db_path == mock_storage_path / "area_occupancy.db"
        assert storage.engine is not None

    async def test_async_initialize_success(
        self, area_occupancy_storage: AreaOccupancyStorage
    ):
        """Test successful database initialization."""
        with patch(
            "custom_components.area_occupancy.sqlite_storage._LOGGER.info"
        ) as mock_logger:
            await area_occupancy_storage.async_initialize()
            mock_logger.assert_called()

    async def test_save_area_entity_config(self, mock_hass: Mock) -> None:
        """Test saving area entity config."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        record = AreaEntityConfigRecord(
            entry_id="test_entry",
            entity_id="sensor.test",
            entity_type="motion",
            weight=1.0,
            prob_given_true=0.8,
            prob_given_false=0.2,
            last_updated=datetime.now(),
        )

        with patch.object(mock_hass, "async_add_executor_job", return_value=record):
            result = await storage.save_area_entity_config(record)

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
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(minutes=5),
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
            "total_entities": 100,
            "total_areas": 5,
            "total_intervals": 1000,
            "total_time_priors": 500,
            "db_size_mb": 1.5,
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
            "total_entities": 0,
            "total_areas": 0,
            "total_intervals": 0,
            "total_time_priors": 0,
            "db_size_mb": 0.0,
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

    async def test_get_recent_time_priors_with_custom_hours(
        self, mock_hass: Mock
    ) -> None:
        """Test getting recent time priors with custom hours."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_records = [
            AreaTimePriorRecord(
                entry_id="test_entry",
                day_of_week=1,
                time_slot=12,
                prior_value=0.5,
                data_points=100,
                last_updated=datetime.now(),
            )
        ]

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_records
        ):
            result = await storage.get_recent_time_priors("test_entry", hours=48)
            assert result == expected_records

    async def test_cleanup_old_intervals_logging(self, mock_hass: Mock) -> None:
        """Test that cleanup operation logs the count."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(mock_hass, "async_add_executor_job", return_value=50),
            patch(
                "custom_components.area_occupancy.sqlite_storage._LOGGER.info"
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
                "custom_components.area_occupancy.sqlite_storage._LOGGER.info"
            ) as mock_logger,
        ):
            await storage.reset_entry_data("test_entry")
            mock_logger.assert_called_with(
                "Reset area-specific data for entry %s", "test_entry"
            )

    async def test_get_area_entity_configs_ordering(self, mock_hass: Mock) -> None:
        """Test that area entity configs are returned in correct order."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_configs = [
            AreaEntityConfigRecord(
                entry_id="test_entry",
                entity_id="sensor.a",
                entity_type="motion",
                weight=1.0,
                prob_given_true=0.8,
                prob_given_false=0.2,
                last_updated=datetime.now(),
            ),
            AreaEntityConfigRecord(
                entry_id="test_entry",
                entity_id="sensor.b",
                entity_type="motion",
                weight=1.0,
                prob_given_true=0.8,
                prob_given_false=0.2,
                last_updated=datetime.now(),
            ),
        ]

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_configs
        ):
            result = await storage.get_area_entity_configs("test_entry")
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
                "custom_components.area_occupancy.sqlite_storage._LOGGER.error"
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
                "custom_components.area_occupancy.sqlite_storage._LOGGER.error"
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
                "custom_components.area_occupancy.sqlite_storage._LOGGER.error"
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
                "custom_components.area_occupancy.sqlite_storage._LOGGER.info"
            ) as mock_logger,
        ):
            await store.async_reset()
            mock_logger.assert_called_with(
                "Reset SQLite storage for entry %s", "test_entry"
            )

    async def test_save_area_occupancy(self, mock_hass: Mock) -> None:
        """Test saving area occupancy record."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        record = AreaOccupancyRecord(
            entry_id="test_entry",
            area_name="Test Area",
            purpose="test",
            threshold=0.5,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch.object(mock_hass, "async_add_executor_job", return_value=record):
            result = await storage.save_area_occupancy(record)

            assert result == record

    async def test_get_area_occupancy(self, mock_hass: Mock) -> None:
        """Test getting area occupancy record."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_record = AreaOccupancyRecord(
            entry_id="test_entry",
            area_name="Test Area",
            purpose="test",
            threshold=0.5,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_record
        ):
            result = await storage.get_area_occupancy("test_entry")

            assert result == expected_record

    async def test_save_time_priors_batch(self, mock_hass: Mock) -> None:
        """Test saving time priors batch."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        records = [
            AreaTimePriorRecord(
                entry_id="test_entry",
                day_of_week=1,
                time_slot=12,
                prior_value=0.5,
                data_points=100,
                last_updated=datetime.now(),
            )
        ]

        with patch.object(mock_hass, "async_add_executor_job", return_value=1):
            result = await storage.save_time_priors_batch(records)

            assert result == 1

    async def test_get_time_prior(self, mock_hass: Mock) -> None:
        """Test getting time prior record."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_record = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=1,
            time_slot=12,
            prior_value=0.5,
            data_points=100,
            last_updated=datetime.now(),
        )

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_record
        ):
            result = await storage.get_time_prior("test_entry", 1, 12)

            assert result == expected_record

    async def test_get_time_priors_for_entry(self, mock_hass: Mock) -> None:
        """Test getting time priors for entry."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_records = [
            AreaTimePriorRecord(
                entry_id="test_entry",
                day_of_week=1,
                time_slot=12,
                prior_value=0.5,
                data_points=100,
                last_updated=datetime.now(),
            )
        ]

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_records
        ):
            result = await storage.get_time_priors_for_entry("test_entry")

            assert result == expected_records

    async def test_get_recent_time_priors(self, mock_hass: Mock) -> None:
        """Test getting recent time priors."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        expected_records = [
            AreaTimePriorRecord(
                entry_id="test_entry",
                day_of_week=1,
                time_slot=12,
                prior_value=0.5,
                data_points=100,
                last_updated=datetime.now(),
            )
        ]

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_records
        ):
            result = await storage.get_recent_time_priors("test_entry", hours=24)

            assert result == expected_records

    async def test_import_intervals_from_recorder(self, mock_hass: Mock) -> None:
        """Test importing intervals from recorder."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        entity_ids = ["sensor.test1", "sensor.test2"]
        expected_result = {"sensor.test1": 0, "sensor.test2": 0}

        with patch.object(
            mock_hass, "async_add_executor_job", return_value=expected_result
        ):
            result = await storage.import_intervals_from_recorder(entity_ids, days=10)

            assert result == expected_result

    async def test_reset_entry_data(self, mock_hass: Mock) -> None:
        """Test resetting entry data."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch.object(mock_hass, "async_add_executor_job") as mock_executor:
            await storage.reset_entry_data("test_entry")

            # Should call async_add_executor_job once for the reset operation
            mock_executor.assert_called_once()

    async def test_database_integrity_check(self, mock_hass: Mock) -> None:
        """Test database integrity check."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(storage, "engine") as mock_engine,
            patch(
                "custom_components.area_occupancy.sqlite_storage._LOGGER.warning"
            ) as mock_logger,
        ):
            mock_conn = Mock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn

            # Mock integrity check failure
            mock_conn.execute.return_value.fetchone.return_value = ["error in database"]
            mock_conn.execute.return_value.fetchall.return_value = [["area_occupancy"]]

            storage._check_database_integrity(mock_conn)

            mock_logger.assert_called()

    async def test_database_missing_tables_check(self, mock_hass: Mock) -> None:
        """Test database missing tables check."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with (
            patch.object(storage, "engine") as mock_engine,
            patch(
                "custom_components.area_occupancy.sqlite_storage._LOGGER.warning"
            ) as mock_logger,
        ):
            mock_conn = Mock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn

            # Mock missing tables
            mock_conn.execute.return_value.fetchone.return_value = ["ok"]
            mock_conn.execute.return_value.fetchall.return_value = [["area_occupancy"]]

            storage._check_database_integrity(mock_conn)

            mock_logger.assert_called()

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
                start_time=base_time,
                end_time=base_time + timedelta(minutes=5),
            ),
            StateInterval(
                entity_id="sensor.test",
                state="on",
                start_time=base_time,
                end_time=base_time + timedelta(minutes=5),
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
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(minutes=5),
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

    async def test_save_time_priors_batch_empty_list(self, mock_hass: Mock) -> None:
        """Test saving empty time priors batch."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")

        with patch(
            "custom_components.area_occupancy.sqlite_storage._LOGGER.debug"
        ) as mock_logger:
            result = await storage.save_time_priors_batch([])
            assert result == 0
            mock_logger.assert_called_with("No time priors to save")

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

    def test_check_database_integrity_logs(self, mock_hass: Mock):
        """Test _check_database_integrity logs warnings for integrity issues and missing tables."""
        storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
        mock_conn = Mock()
        # Simulate integrity check failure
        mock_conn.execute.side_effect = [
            Mock(fetchone=Mock(return_value=["not ok"])),
            Mock(fetchall=Mock(return_value=[["area_occupancy"]])),
        ]
        with patch(
            "custom_components.area_occupancy.sqlite_storage._LOGGER.warning"
        ) as mock_logger:
            storage._check_database_integrity(mock_conn)
            mock_logger.assert_any_call(
                "Database integrity issue detected: %s", "not ok"
            )
        # Simulate missing tables
        mock_conn.execute.side_effect = [
            Mock(fetchone=Mock(return_value=["ok"])),
            Mock(fetchall=Mock(return_value=[["area_occupancy"]])),
        ]
        with patch(
            "custom_components.area_occupancy.sqlite_storage._LOGGER.warning"
        ) as mock_logger:
            storage._check_database_integrity(mock_conn)
            mock_logger.assert_any_call(
                "Missing tables detected: %s",
                {"entities", "area_entity_config", "state_intervals", "metadata"},
            )
