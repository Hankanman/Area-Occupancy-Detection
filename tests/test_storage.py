"""Tests for storage module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    CONF_VERSION,
    CONF_VERSION_MINOR,
    STORAGE_KEY,
)
from custom_components.area_occupancy.storage import StorageManager
from homeassistant.util import dt as dt_util


class TestStorageManager:
    """Test StorageManager class."""

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_initialization(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test StorageManager initialization."""
        mock_store_init.return_value = None  # Mock the Store.__init__ method

        storage = StorageManager(mock_coordinator)

        # Verify Store initialization
        mock_store_init.assert_called_once_with(
            hass=mock_coordinator.hass,
            version=CONF_VERSION,
            key=STORAGE_KEY,
            minor_version=CONF_VERSION_MINOR,
        )

        # Verify instance attributes
        assert storage._coordinator == mock_coordinator
        assert storage._initialized is False
        assert storage._pending_save_count == 0
        assert storage._last_data_hash is None
        assert storage._periodic_save_tracker is None
        assert storage._dirty is False

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_create_empty_storage(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test creating empty storage structure."""
        mock_store_init.return_value = None  # Mock the Store.__init__ method

        storage = StorageManager(mock_coordinator)

        result = storage.create_empty_storage()

        assert "data" in result
        assert "instances" in result["data"]
        assert result["data"]["instances"] == {}
        assert result["version"] == 9
        assert result["minor_version"] == 1

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    async def test_async_initialize_new_storage(
        self, mock_load: AsyncMock, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test async initialization with new storage."""
        mock_store_init.return_value = None
        mock_load.return_value = None  # New storage

        storage = StorageManager(mock_coordinator)

        with patch.object(storage, "_start_periodic_save_timer") as mock_start_timer:
            await storage.async_initialize()

            mock_start_timer.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    async def test_async_initialize_existing_storage(
        self, mock_load: AsyncMock, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test async initialization with existing storage."""
        mock_store_init.return_value = None
        existing_data = {
            "version": 9,
            "minor_version": 1,
            "data": {
                "instances": {"test_entry": {"entities": {}}},
            },
        }
        mock_load.return_value = existing_data

        storage = StorageManager(mock_coordinator)

        with patch.object(storage, "_start_periodic_save_timer") as mock_start_timer:
            await storage.async_initialize()

            mock_start_timer.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_async_shutdown(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test async shutdown."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Set up some state
        mock_tracker = Mock()
        storage._periodic_save_tracker = mock_tracker
        storage._dirty = True

        with patch.object(storage, "_force_save_now") as mock_force_save:
            await storage.async_shutdown()

            # Should cancel timer and force save
            mock_tracker.assert_called_once()
            mock_force_save.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_handle_periodic_save(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test handling periodic save."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)
        storage._dirty = True

        with patch.object(storage, "_force_save_now") as mock_force_save:
            with patch.object(
                storage, "_start_periodic_save_timer"
            ) as mock_start_timer:
                await storage._handle_periodic_save(dt_util.utcnow())

                mock_force_save.assert_called_once()
                mock_start_timer.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_async_perform_cleanup(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test performing cleanup operations."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)
        # Set the hass attribute that the method needs
        storage.hass = mock_coordinator.hass

        with patch.object(storage, "async_cleanup_orphaned_instances") as mock_cleanup:
            await storage._async_perform_cleanup()

            # Should call cleanup
            mock_cleanup.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_async_remove_instance(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test removing an instance from storage."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Setup data with target instance
        storage_data = {
            "instances": {
                "test_entry": {"entities": {}},
                "other_entry": {"entities": {}},
            },
        }
        mock_load.return_value = storage_data

        result = await storage.async_remove_instance("test_entry")

        assert result is True
        assert "test_entry" not in storage_data["instances"]
        assert "other_entry" in storage_data["instances"]
        mock_save.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_async_remove_instance_not_found(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test removing a non-existent instance from storage."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Setup data without target instance
        storage_data = {
            "instances": {
                "other_entry": {"entities": {}},
            },
        }
        mock_load.return_value = storage_data

        result = await storage.async_remove_instance("test_entry")

        assert result is False
        mock_save.assert_not_called()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_async_cleanup_orphaned_instances(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test cleaning up orphaned instances."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Setup data with orphaned instances
        storage_data = {
            "instances": {
                "active_entry": {"entities": {}},
                "orphaned_entry1": {"entities": {}},
                "orphaned_entry2": {"entities": {}},
            },
        }
        mock_load.return_value = storage_data

        active_entries = {"active_entry"}

        result = await storage.async_cleanup_orphaned_instances(active_entries)

        assert result is True
        assert "active_entry" in storage_data["instances"]
        assert "orphaned_entry1" not in storage_data["instances"]
        assert "orphaned_entry2" not in storage_data["instances"]
        mock_save.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_async_cleanup_orphaned_instances_none_found(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test cleaning up when no orphaned instances exist."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Setup data with no orphaned instances
        storage_data = {
            "instances": {
                "active_entry": {"entities": {}},
            },
        }
        mock_load.return_value = storage_data

        active_entries = {"active_entry"}

        result = await storage.async_cleanup_orphaned_instances(active_entries)

        assert result is False
        mock_save.assert_not_called()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    async def test_async_load_instance_data_success(
        self, mock_load: AsyncMock, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test loading instance data successfully."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        instance_data = {"entities": {"test": "data"}}
        storage_data = {
            "instances": {"test_entry": instance_data},
        }
        mock_load.return_value = storage_data

        result = await storage.async_load_instance_data("test_entry")

        assert result == instance_data

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    async def test_async_load_instance_data_not_found(
        self, mock_load: AsyncMock, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test loading non-existent instance data."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        storage_data = {
            "instances": {},
        }
        mock_load.return_value = storage_data

        result = await storage.async_load_instance_data("test_entry")

        assert result is None

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_calculate_data_hash(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test calculating data hash."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        data = {"test": "data", "number": 123}

        hash1 = storage._calculate_data_hash(data)
        hash2 = storage._calculate_data_hash(data)
        hash3 = storage._calculate_data_hash({"different": "data"})

        # Same data should produce same hash
        assert hash1 == hash2
        # Different data should produce different hash
        assert hash1 != hash3
        # Hash should be string
        assert isinstance(hash1, str)

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_should_save_immediately_config_update(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test should_save_immediately for config updates."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Simulate max pending saves to trigger immediate save
        storage._pending_save_count = 10  # MAX_PENDING_SAVES
        assert storage._should_save_immediately() is True

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_async_save_instance_data_immediate(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
        mock_entity_manager: Mock,
    ) -> None:
        """Test saving instance data immediately."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)
        mock_load.return_value = storage.create_empty_storage()

        with patch.object(storage, "_should_save_immediately", return_value=True):
            await storage.async_save_instance_data(
                "test_entry", mock_entity_manager, force=False
            )
            mock_save.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_async_save_instance_data_forced(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
        mock_entity_manager: Mock,
    ) -> None:
        """Test saving instance data when forced."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)
        mock_load.return_value = storage.create_empty_storage()

        await storage.async_save_instance_data(
            "test_entry", mock_entity_manager, force=True
        )
        mock_save.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_schedule_debounced_save(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test scheduling debounced save."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        instance_data = {"test": "data"}
        data_hash = "test_hash"

        with patch("asyncio.create_task") as mock_create_task:
            await storage._schedule_debounced_save(
                "test_entry", instance_data, data_hash
            )
            mock_create_task.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_debounced_save_worker(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test debounced save worker."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        instance_data = {"test": "data"}
        data_hash = "test_hash"

        with patch.object(storage, "_perform_save") as mock_perform_save:
            with patch("asyncio.sleep"):  # Skip the actual sleep
                await storage._debounced_save_worker(
                    "test_entry", instance_data, data_hash
                )

                # Should perform save
                mock_perform_save.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_perform_save_data_changed(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test performing save when data has changed."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)
        mock_load.return_value = storage.create_empty_storage()

        instance_data = {"test": "data"}
        data_hash = "new_hash"

        # Set different last hash
        storage._last_data_hash = "old_hash"

        await storage._perform_save("test_entry", instance_data, data_hash)

        # Should save
        mock_save.assert_called_once()
        assert storage._last_data_hash == data_hash

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_perform_save_data_unchanged(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test performing save when data is unchanged."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)
        mock_load.return_value = storage.create_empty_storage()

        instance_data = {"test": "data"}
        data_hash = "same_hash"

        # Set same last hash
        storage._last_data_hash = "same_hash"

        await storage._perform_save("test_entry", instance_data, data_hash)

        # Should still save (the method always saves)
        mock_save.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_async_reset(
        self, mock_save: AsyncMock, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test resetting storage."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Setup some state
        storage._last_data_hash = "hash"
        storage._dirty = True

        await storage.async_reset()

        # Should reset state
        assert storage._last_data_hash is None
        assert storage._dirty is False
        mock_save.assert_called_once_with({})

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_async_load_with_compatibility_check_compatible(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - compatible version."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Setup compatible data
        compatible_data = {"entities": {}}

        with patch.object(
            storage, "async_load_instance_data", return_value=compatible_data
        ):
            with patch.object(storage, "_validate_storage_format", return_value=True):
                (
                    result,
                    needs_migration,
                ) = await storage.async_load_with_compatibility_check("test_entry", 9)

                assert result == compatible_data
                assert needs_migration is False

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_async_load_with_compatibility_check_needs_migration(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - needs migration."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        with patch.object(storage, "async_remove_instance"):
            result, needs_migration = await storage.async_load_with_compatibility_check(
                "test_entry",
                2,  # Old config entry version
            )

            assert result is None
            assert needs_migration is True

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_async_load_with_compatibility_check_not_found(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - data not found."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        with patch.object(storage, "async_load_instance_data", return_value=None):
            result, needs_migration = await storage.async_load_with_compatibility_check(
                "test_entry", 9
            )

            assert result is None
            assert needs_migration is False

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_validate_storage_format_valid(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test validating valid storage format."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        valid_data = {
            "entities": {
                "test_entity": {
                    "entity_id": "test_entity",
                    "type": "motion",
                    "probability": 0.5,
                    "prior": {"prior": 0.3},
                    "decay": {"decay_factor": 1.0},
                }
            }
        }

        assert storage._validate_storage_format(valid_data) is True

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_validate_storage_format_invalid(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test validating invalid storage format."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        invalid_data = {
            "version": 1,
            # Missing required fields
        }

        assert storage._validate_storage_format(invalid_data) is False

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_validate_entity_format_valid(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test validating valid entity format."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        valid_entity = {
            "entity_id": "test_entity",
            "type": "motion",
            "probability": 0.5,
            "state": "on",
            "is_active": True,
            "available": True,
            "prior": {"prior": 0.3},
            "decay": {"decay_factor": 1.0},
        }

        assert storage._validate_entity_format("test_entity", valid_entity) is True

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_validate_entity_format_invalid(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test validating invalid entity format."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        invalid_entity = {
            # Missing required entity_id field
            "probability": 0.5,
        }

        assert storage._validate_entity_format("test_entity", invalid_entity) is False


class TestStorageManagerIntegration:
    """Test StorageManager integration scenarios."""

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_full_storage_lifecycle(
        self,
        mock_save: AsyncMock,
        mock_load: AsyncMock,
        mock_store_init: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test complete storage lifecycle from initialization to shutdown."""
        mock_store_init.return_value = None
        # Start with no existing data
        mock_load.return_value = None

        storage = StorageManager(mock_coordinator)
        # Set the hass attribute that the method needs
        storage.hass = mock_coordinator.hass

        # Initialize storage
        with patch.object(storage, "_start_periodic_save_timer"):
            await storage.async_initialize()

        # Save some data
        mock_load.return_value = storage.create_empty_storage()
        await storage.async_save_instance_data(
            "test_entry", mock_coordinator.entity_manager, force=True
        )

        # Verify save was called
        assert mock_save.call_count >= 1

        # Test cleanup
        active_entries = {"test_entry", "another_entry"}
        mock_load.return_value = {"instances": {"test_entry": {}, "orphaned": {}}}
        await storage.async_cleanup_orphaned_instances(active_entries)

        # Test shutdown
        await storage.async_shutdown()

        # Should have saved multiple times
        assert mock_save.call_count >= 2

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_load")
    async def test_storage_with_existing_data(
        self, mock_load: AsyncMock, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test storage initialization with existing data."""
        mock_store_init.return_value = None
        existing_data = {
            "instances": {
                "existing_entry": {
                    "entities": {
                        "sensor.test": {
                            "entity_id": "sensor.test",
                            "probability": 0.3,
                            "state": "off",
                        }
                    }
                }
            },
        }
        mock_load.return_value = existing_data

        storage = StorageManager(mock_coordinator)

        # Should be able to access existing instance data
        instance_data = await storage.async_load_instance_data("existing_entry")
        assert instance_data is not None
        assert "entities" in instance_data

    @patch("homeassistant.helpers.storage.Store.__init__")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_error_handling_during_save(
        self, mock_save: AsyncMock, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test error handling during save operations."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Mock async_save to raise an exception
        mock_save.side_effect = Exception("Save failed")

        # Should not raise exception - should handle gracefully
        with pytest.raises(Exception):  # The exception should be re-raised
            await storage.async_save_instance_data(
                "test_entry", mock_coordinator.entity_manager, force=True
            )
