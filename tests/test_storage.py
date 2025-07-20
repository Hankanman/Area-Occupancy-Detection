"""Tests for storage module."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from custom_components.area_occupancy.sqlite_storage import AreaOccupancySQLiteStore


# ruff: noqa: SLF001
class TestAreaOccupancySQLiteStore:
    """Test AreaOccupancySQLiteStore class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test AreaOccupancySQLiteStore initialization."""
        store = AreaOccupancySQLiteStore(mock_coordinator)

        # Verify instance attributes
        assert store._coordinator == mock_coordinator
        assert store._storage is not None

    async def test_async_initialize(self, mock_coordinator: Mock) -> None:
        """Test async_initialize method."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        with patch.object(store._storage, 'async_initialize') as mock_init:
            await store.async_initialize()
            mock_init.assert_called_once()

    async def test_async_save_data_success(self, mock_coordinator: Mock) -> None:
        """Test successful data saving."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        # Mock coordinator attributes
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.config.purpose = "test"
        mock_coordinator.threshold = 0.5
        mock_coordinator.entities.entities = {}
        
        with patch.object(store._storage, 'save_area_occupancy') as mock_save_area:
            await store.async_save_data()
            mock_save_area.assert_called_once()

    async def test_async_save_data_with_entities(self, mock_coordinator: Mock) -> None:
        """Test data saving with entities."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
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
            patch.object(store._storage, 'save_area_occupancy') as mock_save_area,
            patch.object(store._storage, 'save_area_entity_config') as mock_save_entity,
        ):
            await store.async_save_data()
            mock_save_area.assert_called_once()
            mock_save_entity.assert_called_once()

    async def test_async_save_data_error(self, mock_coordinator: Mock) -> None:
        """Test data saving with error."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        # Mock coordinator attributes
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.config.purpose = "test"
        mock_coordinator.threshold = 0.5
        mock_coordinator.entities.entities = {}
        
        with (
            patch.object(store._storage, 'save_area_occupancy', side_effect=Exception("Test error")),
            patch('custom_components.area_occupancy.sqlite_storage._LOGGER.error') as mock_logger,
        ):
            with pytest.raises(Exception, match="Test error"):
                await store.async_save_data()
            mock_logger.assert_called_once()

    async def test_async_load_data_success(self, mock_coordinator: Mock) -> None:
        """Test successful data loading."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        # Mock storage responses
        mock_area_record = Mock()
        mock_area_record.area_name = "Test Area"
        mock_area_record.purpose = "test"
        mock_area_record.threshold = 0.5
        mock_area_record.updated_at.isoformat.return_value = "2024-01-01T00:00:00Z"
        
        mock_coordinator.probability = 0.7
        mock_coordinator.area_prior = 0.3
        
        with (
            patch.object(store._storage, 'get_area_occupancy', return_value=mock_area_record),
            patch.object(store._storage, 'get_area_entity_configs', return_value=[]),
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
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        with patch.object(store._storage, 'get_area_occupancy', return_value=None):
            result = await store.async_load_data()
            assert result is None

    async def test_async_load_data_error(self, mock_coordinator: Mock) -> None:
        """Test data loading with error."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        import sqlalchemy as sa
        
        with (
            patch.object(store._storage, 'get_area_occupancy', side_effect=sa.exc.SQLAlchemyError("Test error")),
            patch('custom_components.area_occupancy.sqlite_storage._LOGGER.error') as mock_logger,
        ):
            result = await store.async_load_data()
            assert result is None
            mock_logger.assert_called_once()

    async def test_async_reset(self, mock_coordinator: Mock) -> None:
        """Test async_reset method."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        mock_coordinator.entry_id = "test_entry"
        
        with (
            patch.object(store._storage, 'reset_entry_data') as mock_reset,
            patch('custom_components.area_occupancy.sqlite_storage._LOGGER.info') as mock_logger,
        ):
            await store.async_reset()
            mock_reset.assert_called_once_with("test_entry")
            mock_logger.assert_called_once()

    async def test_async_get_stats(self, mock_coordinator: Mock) -> None:
        """Test async_get_stats method."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        expected_stats = {"total_entities": 5, "total_areas": 2}
        
        with patch.object(store._storage, 'get_stats', return_value=expected_stats):
            result = await store.async_get_stats()
            assert result == expected_stats

    async def test_import_intervals_from_recorder(self, mock_coordinator: Mock) -> None:
        """Test import_intervals_from_recorder method."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        entity_ids = ["sensor.test1", "sensor.test2"]
        expected_result = {"sensor.test1": 100, "sensor.test2": 150}
        
        with patch.object(store._storage, 'import_intervals_from_recorder', return_value=expected_result):
            result = await store.import_intervals_from_recorder(entity_ids, days=10)
            assert result == expected_result

    async def test_cleanup_old_intervals(self, mock_coordinator: Mock) -> None:
        """Test cleanup_old_intervals method."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        with patch.object(store._storage, 'cleanup_old_intervals', return_value=50):
            result = await store.cleanup_old_intervals(retention_days=365)
            assert result == 50

    async def test_is_state_intervals_empty(self, mock_coordinator: Mock) -> None:
        """Test is_state_intervals_empty method."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        with patch.object(store._storage, 'is_state_intervals_empty', return_value=True):
            result = await store.is_state_intervals_empty()
            assert result is True

    async def test_get_total_intervals_count(self, mock_coordinator: Mock) -> None:
        """Test get_total_intervals_count method."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        with patch.object(store._storage, 'get_total_intervals_count', return_value=1000):
            result = await store.get_total_intervals_count()
            assert result == 1000

    async def test_get_historical_intervals(self, mock_coordinator: Mock) -> None:
        """Test get_historical_intervals method."""
        store = AreaOccupancySQLiteStore(mock_coordinator)
        
        from datetime import datetime, timedelta
        from custom_components.area_occupancy.utils import StateInterval
        
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        expected_intervals = [
            {
                "state": "on",
                "start": start_time,
                "end": end_time,
            }
        ]
        
        with patch.object(store._storage, 'get_historical_intervals', return_value=expected_intervals):
            result = await store.get_historical_intervals("sensor.test", start_time, end_time)
            assert result == expected_intervals
