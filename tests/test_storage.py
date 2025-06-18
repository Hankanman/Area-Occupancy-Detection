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
            assert result.get("entity_types") == {}

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
                "entity_types": {"motion": {}},
            }
            result = await store._async_migrate_func(
                CONF_VERSION, CONF_VERSION_MINOR, old_data
            )

            # Should preserve existing data
            assert result.get("name") == "Test Area"
            assert result.get("probability") == 0.7
            assert result.get("entities") == {"sensor.test": {}}
            assert result.get("entity_types") == {"motion": {}}

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

            # Should return empty storage for invalid data
            assert result.get("entities") == {}
            assert result.get("entity_types") == {}

    def test_async_save_coordinator_data(
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

            store.async_save_coordinator_data(mock_entity_manager)

            # Should call async_delay_save with proper data
            mock_delay_save.assert_called_once()
            call_args = mock_delay_save.call_args
            assert call_args[1]["delay"] == 30.0

    async def test_async_load_coordinator_data_success(
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

            result = await store.async_load_coordinator_data()

            assert result is not None
            assert "entities" in result
            assert "entity_types" in result
            assert "last_updated" in result

    async def test_async_load_coordinator_data_no_data(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading coordinator data when no data exists."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch("homeassistant.helpers.storage.Store.async_load", return_value=None),
        ):
            store = AreaOccupancyStore(mock_coordinator)

            result = await store.async_load_coordinator_data()

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

            result = await store.async_load_coordinator_data()

            # Should return None on error
            assert result is None

    async def test_async_load_with_compatibility_check_old_version(
        self, mock_coordinator: Mock
    ) -> None:
        """Test compatibility check with old config entry version."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            store = AreaOccupancyStore(mock_coordinator)

            # Mock async_remove method
            store.async_remove = AsyncMock()

            result, was_reset = await store.async_load_with_compatibility_check(
                "test_entry", 8
            )

            # Should reset storage for old version
            assert result is None
            assert was_reset is True
            store.async_remove.assert_called_once()

    async def test_async_load_with_compatibility_check_compatible_version(
        self, mock_coordinator: Mock, valid_storage_data: dict
    ) -> None:
        """Test compatibility check with compatible version."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch.object(
                AreaOccupancyStore,
                "async_load_coordinator_data",
                return_value=valid_storage_data,
            ),
        ):
            store = AreaOccupancyStore(mock_coordinator)

            result, was_reset = await store.async_load_with_compatibility_check(
                "test_entry", 9
            )

            # Should load data successfully
            assert result == valid_storage_data
            assert was_reset is False

    async def test_async_load_with_compatibility_check_invalid_format(
        self, mock_coordinator: Mock
    ) -> None:
        """Test compatibility check with invalid data format."""
        invalid_data = {"no_entities": "key"}  # Missing required entities key

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch.object(
                AreaOccupancyStore,
                "async_load_coordinator_data",
                return_value=invalid_data,
            ),
        ):
            store = AreaOccupancyStore(mock_coordinator)
            store.async_remove = AsyncMock()

            result, was_reset = await store.async_load_with_compatibility_check(
                "test_entry", 9
            )

            # Should reset storage for invalid format
            assert result is None
            assert was_reset is True
            store.async_remove.assert_called_once()

    async def test_async_load_with_compatibility_check_storage_error(
        self, mock_coordinator: Mock
    ) -> None:
        """Test compatibility check with storage error."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch.object(
                AreaOccupancyStore,
                "async_load_coordinator_data",
                side_effect=OSError("Storage error"),
            ),
        ):
            store = AreaOccupancyStore(mock_coordinator)

            result, was_reset = await store.async_load_with_compatibility_check(
                "test_entry", 9
            )

            # Should return None with no reset on storage error
            assert result is None
            assert was_reset is False
