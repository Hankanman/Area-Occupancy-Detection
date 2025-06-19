"""Tests for storage module."""

from unittest.mock import AsyncMock, Mock, patch

from custom_components.area_occupancy.const import CONF_VERSION, CONF_VERSION_MINOR
from custom_components.area_occupancy.storage import AreaOccupancyStore
from custom_components.area_occupancy.types import PriorState


# ruff: noqa: SLF001
class TestAreaOccupancyStore:
    """Test AreaOccupancyStore class."""

    def test_initialization(self, mock_hass: Mock) -> None:
        """Test AreaOccupancyStore initialization."""
        store = AreaOccupancyStore(mock_hass)
        
        # Verify instance attributes
        assert store.hass == mock_hass
        assert store._current_version == CONF_VERSION
        assert store._current_minor_version == CONF_VERSION_MINOR

    async def test_async_migrate_func_major_version_change(
        self, mock_hass: Mock
    ) -> None:
        """Test migration with major version change."""
        store = AreaOccupancyStore(mock_hass)
        
        old_data = {"some": "old_data"}
        result = await store._async_migrate_func(8, 0, old_data)
        
        # Should return new structure for major version change
        assert "instances" in result
        assert result["instances"] == {}

    async def test_async_migrate_func_compatible_version(
        self, mock_hass: Mock
    ) -> None:
        """Test migration with compatible version."""
        store = AreaOccupancyStore(mock_hass)
        
        old_data = {"instances": {"test": "data"}}
        result = await store._async_migrate_func(CONF_VERSION, CONF_VERSION_MINOR, old_data)
        
        # Should preserve data for compatible version
        assert result == old_data

    async def test_async_migrate_func_invalid_data(
        self, mock_hass: Mock
    ) -> None:
        """Test migration with invalid data format."""
        store = AreaOccupancyStore(mock_hass)
        
        invalid_data = {}  # Use empty dict to test invalid data format
        result = await store._async_migrate_func(
            CONF_VERSION, CONF_VERSION_MINOR, invalid_data
        )
        
        # Should return new structure for invalid data
        assert "instances" in result
        assert result["instances"] == {}

    async def test_async_save_instance_prior_state_success(
        self, mock_hass: Mock
    ) -> None:
        """Test saving instance prior state successfully."""
        store = AreaOccupancyStore(mock_hass)
        prior_state = PriorState()
        
        with patch.object(store, 'async_load', return_value={"instances": {}}), \
             patch.object(store, 'async_save') as mock_save:
            
            await store.async_save_instance_prior_state("test_entry", "Test Area", prior_state)
            
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert "instances" in saved_data
            assert "test_entry" in saved_data["instances"]

    async def test_async_load_instance_prior_state_success(
        self, mock_hass: Mock, valid_storage_data: dict
    ) -> None:
        """Test loading instance prior state successfully."""
        store = AreaOccupancyStore(mock_hass)
        
        with patch.object(store, 'async_load', return_value=valid_storage_data):
            result = await store.async_load_instance_prior_state("test_entry")
            
            assert result.name == valid_storage_data["instances"]["test_entry"]["name"]
            assert result.prior_state is not None
            assert result.last_updated == valid_storage_data["instances"]["test_entry"]["last_updated"]

    async def test_async_load_instance_prior_state_no_data(
        self, mock_hass: Mock
    ) -> None:
        """Test loading instance prior state when no data exists."""
        store = AreaOccupancyStore(mock_hass)
        
        with patch.object(store, 'async_load', return_value=None):
            result = await store.async_load_instance_prior_state("test_entry")
            
            assert result.name is None
            assert result.prior_state is None
            assert result.last_updated is None

    async def test_async_load_instance_prior_state_error(
        self, mock_hass: Mock
    ) -> None:
        """Test loading instance prior state with storage error."""
        store = AreaOccupancyStore(mock_hass)
        
        with patch.object(store, 'async_load', side_effect=OSError("Storage error")):
            try:
                await store.async_load_instance_prior_state("test_entry")
                assert False, "Should have raised StorageLoadError"
            except Exception as e:
                assert "Failed to load prior state" in str(e)

    async def test_async_remove_instance_success(
        self, mock_hass: Mock
    ) -> None:
        """Test removing instance successfully."""
        store = AreaOccupancyStore(mock_hass)
        existing_data = {"instances": {"test_entry": {"name": "test"}}}
        
        with patch.object(store, 'async_load', return_value=existing_data), \
             patch.object(store, 'async_save') as mock_save:
            
            result = await store.async_remove_instance("test_entry")
            
            assert result is True
            mock_save.assert_called_once()

    async def test_async_cleanup_orphaned_instances(
        self, mock_hass: Mock
    ) -> None:
        """Test cleanup of orphaned instances."""
        store = AreaOccupancyStore(mock_hass)
        existing_data = {
            "instances": {
                "active_entry": {"name": "active"},
                "orphaned_entry": {"name": "orphaned"}
            }
        }
        
        with patch.object(store, 'async_load', return_value=existing_data), \
             patch.object(store, 'async_save') as mock_save:
            
            result = await store.async_cleanup_orphaned_instances({"active_entry"})
            
            assert result is True
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert "active_entry" in saved_data["instances"]
            assert "orphaned_entry" not in saved_data["instances"]
