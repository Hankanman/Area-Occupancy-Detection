"""Tests for storage module."""

from unittest.mock import AsyncMock, Mock, patch

from custom_components.area_occupancy.const import CONF_VERSION, CONF_VERSION_MINOR
from custom_components.area_occupancy.storage import AreaOccupancyStore


# ruff: noqa: SLF001
class TestAreaOccupancyStore:
    """Test AreaOccupancyStore class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test AreaOccupancyStore initialization."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            store = AreaOccupancyStore(mock_coordinator)

            # Verify instance attributes
            assert store._coordinator == mock_coordinator

    async def test_async_migrate_func_major_version_change(
        self, mock_coordinator: Mock
    ) -> None:
        """Test migration with major version change."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            store = AreaOccupancyStore(mock_coordinator)

            old_data = {"some": "old_data"}
            result = await store._async_migrate_func(8, 0, old_data)

            # Should return empty storage for major version change
            assert result.get("entities") == {}
            assert set(result.keys()) == {"entities"}

    async def test_async_migrate_func_compatible_version(
        self, mock_coordinator: Mock
    ) -> None:
        """Test migration with compatible version."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            store = AreaOccupancyStore(mock_coordinator)

            old_data = {
                "name": "Test Area",
                "probability": 0.7,
                "entities": {"sensor.test": {}},
                "prior": 0.5,
                "threshold": 0.2,
                "last_updated": "2024-01-01T00:00:00Z",
            }
            result = await store._async_migrate_func(
                CONF_VERSION, CONF_VERSION_MINOR, old_data
            )

            # Should preserve existing data
            assert result.get("name") == "Test Area"
            assert result.get("probability") == 0.7
            assert result.get("entities") == {"sensor.test": {}}
            assert result.get("prior") == 0.5
            assert result.get("threshold") == 0.2
            assert result.get("last_updated") == "2024-01-01T00:00:00Z"

    async def test_async_migrate_func_invalid_data(
        self, mock_coordinator: Mock
    ) -> None:
        """Test migration with invalid data format."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            store = AreaOccupancyStore(mock_coordinator)

            invalid_data = {}  # Use empty dict to test invalid data format
            result = await store._async_migrate_func(
                CONF_VERSION, CONF_VERSION_MINOR, invalid_data
            )

            # Should return complete storage structure with defaults
            assert result.get("entities") == {}
            assert result.get("name") is None
            assert result.get("probability") is None
            assert result.get("prior") is None
            assert result.get("threshold") is None
            assert result.get("last_updated") is None

    async def test_async_save_data(
        self, mock_coordinator: Mock, mock_entity_manager: Mock
    ) -> None:
        """Test saving coordinator data."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_delay_save"
            ) as mock_delay_save,
        ):
            store = AreaOccupancyStore(mock_coordinator)

            # Mock entity manager data
            mock_entity_manager.to_dict.return_value = {
                "entities": {"sensor.test": {"entity_id": "sensor.test"}}
            }
            mock_coordinator.entities = mock_entity_manager
            mock_coordinator.config = Mock()
            mock_coordinator.config.name = "Test Area"
            # Set attributes directly on the mock instead of using PropertyMock
            mock_coordinator.probability = 0.5
            mock_coordinator.prior = 0.1
            mock_coordinator.threshold = 0.2

            await store.async_save_data()

            # Should call async_delay_save with proper data
            mock_delay_save.assert_called_once()
            call_args = mock_delay_save.call_args
            assert call_args[1]["delay"] == 30.0

    async def test_async_load_data_success(
        self, mock_coordinator: Mock, valid_storage_data: dict
    ) -> None:
        """Test loading coordinator data successfully."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=valid_storage_data,
            ),
        ):
            store = AreaOccupancyStore(mock_coordinator)

            result = await store.async_load_data()

            assert result is not None
            assert "entities" in result
            assert "last_updated" in result

    async def test_async_load_data_no_data(self, mock_coordinator: Mock) -> None:
        """Test loading coordinator data when no data exists."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch("homeassistant.helpers.storage.Store.async_load", return_value=None),
        ):
            store = AreaOccupancyStore(mock_coordinator)

            result = await store.async_load_data()

            assert result is None

    async def test_async_load_coordinator_data_error(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading coordinator data with storage error."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                side_effect=OSError("Storage error"),
            ),
        ):
            store = AreaOccupancyStore(mock_coordinator)

            result = await store.async_load_data()

            # Should return None on error
            assert result is None

    async def test_async_load_data_invalid_format(self, mock_coordinator: Mock) -> None:
        """Test compatibility check with invalid data format."""
        invalid_data = {"no_entities": "key"}  # Missing required entities key

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=invalid_data,
            ),
            patch(
                "homeassistant.helpers.storage.Store.async_remove",
                new_callable=AsyncMock,
            ) as mock_remove,
        ):
            store = AreaOccupancyStore(mock_coordinator)

            result = await store.async_load_data()

            # Should reset storage for invalid format
            assert result is None
            mock_remove.assert_called_once()

    async def test_async_load_data_storage_error(self, mock_coordinator: Mock) -> None:
        """Test compatibility check with storage error."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                side_effect=OSError("Storage error"),
            ),
        ):
            store = AreaOccupancyStore(mock_coordinator)

            result = await store.async_load_data()

            # Should return None with no reset on storage error
            assert result is None
