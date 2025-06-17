"""Tests for storage module."""

from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.storage import StorageManager
from homeassistant.exceptions import HomeAssistantError


# ruff: noqa: SLF001
class TestStorageManager:
    """Test StorageManager class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test StorageManager initialization."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            storage = StorageManager(mock_coordinator)

            # Verify instance attributes
            assert storage._coordinator == mock_coordinator

    def test_create_empty_storage(self, mock_coordinator: Mock) -> None:
        """Test creating empty storage structure."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            storage = StorageManager(mock_coordinator)

            result = storage.create_empty_storage()

            assert "instances" in result
            assert result["instances"] == {}
            assert result["version"] == 9
            assert result["minor_version"] == 1

    async def test_async_initialize_new_storage(self, mock_coordinator: Mock) -> None:
        """Test async initialization with new storage."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch("homeassistant.helpers.storage.Store.async_load", return_value=None),
        ):
            storage = StorageManager(mock_coordinator)

            with patch.object(storage, "_async_perform_cleanup") as mock_cleanup:
                await storage.async_initialize()
                mock_cleanup.assert_called_once()

    async def test_async_initialize_existing_storage(
        self, mock_coordinator: Mock
    ) -> None:
        """Test async initialization with existing storage."""
        existing_data = {
            "version": 9,
            "minor_version": 1,
            "instances": {"test_entry": {"entities": {}}},
        }
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=existing_data,
            ),
        ):
            storage = StorageManager(mock_coordinator)

            with patch.object(storage, "_async_perform_cleanup") as mock_cleanup:
                await storage.async_initialize()
                mock_cleanup.assert_called_once()

    async def test_async_perform_cleanup(self, mock_coordinator: Mock) -> None:
        """Test performing cleanup operations."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            storage = StorageManager(mock_coordinator)
            # Set the hass attribute that the method needs
            storage.hass = mock_coordinator.hass

            # Mock config entries to return some active entries
            mock_entry = Mock()
            mock_entry.entry_id = "test_entry"
            mock_coordinator.hass.config_entries.async_entries.return_value = [
                mock_entry
            ]

            with patch.object(
                storage, "async_cleanup_orphaned_instances"
            ) as mock_cleanup:
                await storage._async_perform_cleanup()

                # Should call cleanup when there are active entries
                mock_cleanup.assert_called_once()

    async def test_async_remove_instance(self, mock_coordinator: Mock) -> None:
        """Test removing an instance from storage."""
        # Setup data with target instance
        storage_data = {
            "instances": {
                "test_entry": {"entities": {}},
                "other_entry": {"entities": {}},
            },
        }

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=storage_data,
            ),
            patch("homeassistant.helpers.storage.Store.async_save") as mock_save,
        ):
            storage = StorageManager(mock_coordinator)

            result = await storage.async_remove_instance("test_entry")

            assert result is True
            assert "test_entry" not in storage_data["instances"]
            assert "other_entry" in storage_data["instances"]
            mock_save.assert_called_once()

    async def test_async_remove_instance_not_found(
        self, mock_coordinator: Mock
    ) -> None:
        """Test removing a non-existent instance from storage."""
        # Setup data without target instance
        storage_data = {
            "instances": {
                "other_entry": {"entities": {}},
            },
        }

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=storage_data,
            ),
            patch("homeassistant.helpers.storage.Store.async_save") as mock_save,
        ):
            storage = StorageManager(mock_coordinator)

            result = await storage.async_remove_instance("test_entry")

            assert result is False
            mock_save.assert_not_called()

    async def test_async_cleanup_orphaned_instances(
        self, mock_coordinator: Mock
    ) -> None:
        """Test cleaning up orphaned instances."""
        # Setup data with orphaned instances
        storage_data = {
            "instances": {
                "active_entry": {"entities": {}},
                "orphaned_entry1": {"entities": {}},
                "orphaned_entry2": {"entities": {}},
            },
        }

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=storage_data,
            ),
            patch("homeassistant.helpers.storage.Store.async_save") as mock_save,
        ):
            storage = StorageManager(mock_coordinator)

            active_entries = {"active_entry"}

            result = await storage.async_cleanup_orphaned_instances(active_entries)

            assert result is True
            assert "active_entry" in storage_data["instances"]
            assert "orphaned_entry1" not in storage_data["instances"]
            assert "orphaned_entry2" not in storage_data["instances"]
            mock_save.assert_called_once()

    async def test_async_cleanup_orphaned_instances_none_found(
        self, mock_coordinator: Mock
    ) -> None:
        """Test cleaning up when no orphaned instances exist."""
        # Setup data with no orphaned instances
        storage_data = {
            "instances": {
                "active_entry": {"entities": {}},
            },
        }

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=storage_data,
            ),
            patch("homeassistant.helpers.storage.Store.async_save") as mock_save,
        ):
            storage = StorageManager(mock_coordinator)

            active_entries = {"active_entry"}

            result = await storage.async_cleanup_orphaned_instances(active_entries)

            assert result is False
            mock_save.assert_not_called()

    async def test_async_load_instance_data_success(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading instance data successfully."""
        instance_data = {"entities": {"test": "data"}}
        storage_data = {
            "instances": {"test_entry": instance_data},
        }

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=storage_data,
            ),
        ):
            storage = StorageManager(mock_coordinator)

            result = await storage.async_load_instance_data("test_entry")

            assert result == instance_data

    async def test_async_load_instance_data_not_found(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading non-existent instance data."""
        storage_data = {
            "instances": {},
        }

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=storage_data,
            ),
        ):
            storage = StorageManager(mock_coordinator)

            result = await storage.async_load_instance_data("test_entry")

            assert result is None

    async def test_async_save_instance_data(
        self, mock_coordinator: Mock, mock_entity_manager: Mock
    ) -> None:
        """Test saving instance data."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch("homeassistant.helpers.storage.Store.async_load") as mock_load,
            patch("homeassistant.helpers.storage.Store.async_save") as mock_save,
        ):
            storage = StorageManager(mock_coordinator)
            mock_load.return_value = storage.create_empty_storage()

            await storage.async_save_instance_data("test_entry", mock_entity_manager)
            mock_save.assert_called_once()

    async def test_async_load_with_compatibility_check_compatible(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - compatible version."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            storage = StorageManager(mock_coordinator)

            # Setup compatible data
            compatible_data = {"entities": {}}

            with patch.object(
                storage, "async_load_instance_data", return_value=compatible_data
            ):
                (
                    result,
                    needs_migration,
                ) = await storage.async_load_with_compatibility_check("test_entry", 9)

                assert result == compatible_data
                assert needs_migration is False

    async def test_async_load_with_compatibility_check_needs_migration(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - needs migration."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            storage = StorageManager(mock_coordinator)

            with patch.object(storage, "async_remove_instance"):
                (
                    result,
                    needs_migration,
                ) = await storage.async_load_with_compatibility_check(
                    "test_entry",
                    2,  # Old config entry version
                )

                assert result is None
                assert needs_migration is True

    async def test_async_load_with_compatibility_check_not_found(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - data not found."""
        with patch("homeassistant.helpers.storage.Store.__init__", return_value=None):
            storage = StorageManager(mock_coordinator)

            with patch.object(storage, "async_load_instance_data", return_value=None):
                (
                    result,
                    needs_migration,
                ) = await storage.async_load_with_compatibility_check("test_entry", 9)

                assert result is None
                assert needs_migration is False


class TestStorageManagerIntegration:
    """Test StorageManager integration scenarios."""

    async def test_full_storage_lifecycle(self, mock_coordinator: Mock) -> None:
        """Test complete storage lifecycle from initialization."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch("homeassistant.helpers.storage.Store.async_load") as mock_load,
            patch("homeassistant.helpers.storage.Store.async_save") as mock_save,
        ):
            # Start with no existing data
            mock_load.return_value = None

            storage = StorageManager(mock_coordinator)
            # Set the hass attribute that the method needs
            storage.hass = mock_coordinator.hass

            # Initialize storage
            with patch.object(storage, "_async_perform_cleanup"):
                await storage.async_initialize()

            # Save some data
            mock_load.return_value = storage.create_empty_storage()
            await storage.async_save_instance_data(
                "test_entry", mock_coordinator.entities
            )

            # Verify save was called
            assert mock_save.call_count >= 1

            # Test cleanup
            active_entries = {"test_entry", "another_entry"}
            mock_load.return_value = {
                "version": 9,
                "minor_version": 1,
                "instances": {"test_entry": {}, "orphaned": {}},
            }
            await storage.async_cleanup_orphaned_instances(active_entries)

            # Should have saved multiple times
            assert mock_save.call_count >= 2

    async def test_storage_with_existing_data(self, mock_coordinator: Mock) -> None:
        """Test storage initialization with existing data."""
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

        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value=existing_data,
            ),
        ):
            storage = StorageManager(mock_coordinator)

            # Should be able to access existing instance data
            instance_data = await storage.async_load_instance_data("existing_entry")
            assert instance_data is not None
            assert "entities" in instance_data

    async def test_error_handling_during_save(self, mock_coordinator: Mock) -> None:
        """Test error handling during save operations."""
        with (
            patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                return_value={"instances": {}},
            ),
            patch("homeassistant.helpers.storage.Store.async_save") as mock_save,
        ):
            storage = StorageManager(mock_coordinator)

            # Mock async_save to raise an exception
            mock_save.side_effect = OSError("Save failed")

            # Should raise HomeAssistantError wrapping the OSError
            with pytest.raises(HomeAssistantError):
                await storage.async_save_instance_data(
                    "test_entry", mock_coordinator.entities
                )
