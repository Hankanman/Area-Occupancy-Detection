"""Tests for storage module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.data.entity import EntityManager
from custom_components.area_occupancy.storage import StorageManager
from homeassistant.util import dt as dt_util


class TestStorageManager:
    """Test StorageManager class."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator for testing."""
        coordinator = Mock()

        # Mock hass with proper data structure
        coordinator.hass = Mock()
        coordinator.hass.data = {}
        coordinator.hass.config = Mock()
        coordinator.hass.config.path = Mock(return_value="/mock/config/path")

        coordinator.config_entry = Mock()
        coordinator.config_entry.entry_id = "test_entry_id"
        coordinator.entity_manager = Mock(spec=EntityManager)
        return coordinator

    @pytest.fixture
    def mock_entity_manager(self) -> Mock:
        """Create a mock entity manager."""
        manager = Mock(spec=EntityManager)
        manager.to_dict.return_value = {
            "entities": {
                "test_entity": {
                    "entity_id": "test_entity",
                    "probability": 0.5,
                    "state": "on",
                }
            }
        }
        return manager

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_initialization(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test StorageManager initialization."""
        mock_store_init.return_value = None  # Mock the Store.__init__ method

        storage = StorageManager(mock_coordinator)

        assert storage._coordinator == mock_coordinator
        assert storage.hass == mock_coordinator.hass
        assert storage._initialized is False
        assert storage._pending_save_count == 0
        assert storage._last_data_hash is None
        assert storage._periodic_save_tracker is None

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_create_empty_storage(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test creating empty storage structure."""
        mock_store_init.return_value = None  # Mock the Store.__init__ method

        storage = StorageManager(mock_coordinator)

        result = storage.create_empty_storage()

        assert "version" in result
        assert "minor_version" in result
        assert "instances" in result
        assert result["instances"] == {}

    @patch("homeassistant.helpers.storage.Store.async_load")
    async def test_async_initialize_new_storage(
        self, mock_load: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test async initialization with new storage."""
        mock_load.return_value = None  # New storage

        storage = StorageManager(mock_coordinator)

        with patch.object(storage, "_start_periodic_save_timer") as mock_start_timer:
            await storage.async_initialize()

            assert storage._data == storage.create_empty_storage()
            mock_start_timer.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.async_load")
    async def test_async_initialize_existing_storage(
        self, mock_load: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test async initialization with existing storage."""
        existing_data = {
            "version": 1,
            "minor_version": 0,
            "instances": {"test_entry": {"entity_manager": {"entities": {}}}},
        }
        mock_load.return_value = existing_data

        storage = StorageManager(mock_coordinator)

        with patch.object(storage, "_start_periodic_save_timer") as mock_start_timer:
            await storage.async_initialize()

            assert storage._data == existing_data
            mock_start_timer.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    async def test_async_shutdown(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test async shutdown."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        # Set up some state
        storage._periodic_save_tracker = Mock()
        storage._dirty = True

        with patch.object(storage, "_force_save_now") as mock_force_save:
            await storage.async_shutdown()

            # Should cancel timer and force save
            storage._periodic_save_tracker.assert_called_once()
            mock_force_save.assert_called_once()

    @patch("homeassistant.helpers.storage.Store.__init__")
    def test_start_periodic_save_timer(
        self, mock_store_init: Mock, mock_coordinator: Mock
    ) -> None:
        """Test starting periodic save timer."""
        mock_store_init.return_value = None
        storage = StorageManager(mock_coordinator)

        with patch(
            "homeassistant.helpers.event.async_track_point_in_time"
        ) as mock_track:
            storage._start_periodic_save_timer()

            mock_track.assert_called_once()
            assert storage._periodic_save_tracker is not None

    async def test_handle_periodic_save(self, mock_coordinator: Mock) -> None:
        """Test handling periodic save."""
        storage = StorageManager(mock_coordinator)

        with patch.object(storage, "_force_save_now") as mock_force_save:
            with patch.object(
                storage, "_start_periodic_save_timer"
            ) as mock_start_timer:
                await storage._handle_periodic_save(dt_util.utcnow())

                mock_force_save.assert_called_once()
                mock_start_timer.assert_called_once()

    async def test_async_perform_cleanup(self, mock_coordinator: Mock) -> None:
        """Test performing cleanup operations."""
        storage = StorageManager(mock_coordinator)

        # Setup data with some instances
        storage._data = {
            "version": 1,
            "minor_version": 0,
            "instances": {
                "active_entry": {"entity_manager": {"entities": {}}},
                "inactive_entry": {"entity_manager": {"entities": {}}},
            },
        }

        # Mock active entries
        active_entries = {"active_entry"}

        with patch.object(storage, "async_cleanup_orphaned_instances") as mock_cleanup:
            await storage._async_perform_cleanup()

            # Should call cleanup (though we're not passing active_entries here)
            mock_cleanup.assert_called_once()

    async def test_async_remove_instance(self, mock_coordinator: Mock) -> None:
        """Test removing instance data."""
        storage = StorageManager(mock_coordinator)

        # Setup data with instance
        storage._data = {
            "version": 1,
            "minor_version": 0,
            "instances": {
                "test_entry": {"entity_manager": {"entities": {}}},
                "other_entry": {"entity_manager": {"entities": {}}},
            },
        }

        with patch.object(storage, "async_save") as mock_save:
            result = await storage.async_remove_instance("test_entry")

            assert result is True
            assert "test_entry" not in storage._data["instances"]
            assert "other_entry" in storage._data["instances"]
            mock_save.assert_called_once()

    async def test_async_remove_instance_not_found(
        self, mock_coordinator: Mock
    ) -> None:
        """Test removing non-existent instance."""
        storage = StorageManager(mock_coordinator)

        # Setup data without target instance
        storage._data = {
            "version": 1,
            "minor_version": 0,
            "instances": {
                "other_entry": {"entity_manager": {"entities": {}}},
            },
        }

        with patch.object(storage, "async_save") as mock_save:
            result = await storage.async_remove_instance("test_entry")

            assert result is False
            mock_save.assert_not_called()

    async def test_async_cleanup_orphaned_instances(
        self, mock_coordinator: Mock
    ) -> None:
        """Test cleaning up orphaned instances."""
        storage = StorageManager(mock_coordinator)

        # Setup data with orphaned instances
        storage._data = {
            "version": 1,
            "minor_version": 0,
            "instances": {
                "active_entry": {"entity_manager": {"entities": {}}},
                "orphaned_entry1": {"entity_manager": {"entities": {}}},
                "orphaned_entry2": {"entity_manager": {"entities": {}}},
            },
        }

        active_entries = {"active_entry"}

        with patch.object(storage, "async_save") as mock_save:
            result = await storage.async_cleanup_orphaned_instances(active_entries)

            assert result is True
            assert "active_entry" in storage._data["instances"]
            assert "orphaned_entry1" not in storage._data["instances"]
            assert "orphaned_entry2" not in storage._data["instances"]
            mock_save.assert_called_once()

    async def test_async_cleanup_orphaned_instances_none_found(
        self, mock_coordinator: Mock
    ) -> None:
        """Test cleaning up when no orphaned instances exist."""
        storage = StorageManager(mock_coordinator)

        # Setup data with no orphaned instances
        storage._data = {
            "version": 1,
            "minor_version": 0,
            "instances": {
                "active_entry": {"entity_manager": {"entities": {}}},
            },
        }

        active_entries = {"active_entry"}

        with patch.object(storage, "async_save") as mock_save:
            result = await storage.async_cleanup_orphaned_instances(active_entries)

            assert result is False
            mock_save.assert_not_called()

    async def test_async_load_instance_data_success(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading instance data successfully."""
        storage = StorageManager(mock_coordinator)

        instance_data = {"entity_manager": {"entities": {"test": "data"}}}
        storage._data = {
            "version": 1,
            "minor_version": 0,
            "instances": {"test_entry": instance_data},
        }

        result = await storage.async_load_instance_data("test_entry")

        assert result == instance_data

    async def test_async_load_instance_data_not_found(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading non-existent instance data."""
        storage = StorageManager(mock_coordinator)

        storage._data = {"version": 1, "minor_version": 0, "instances": {}}

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

        # Mock config entry update
        mock_coordinator.config_entry.state = "loaded"

        # Should save immediately for critical operations
        assert storage._should_save_immediately() is True

    async def test_async_save_instance_data_immediate(
        self, mock_coordinator: Mock, mock_entity_manager: Mock
    ) -> None:
        """Test saving instance data immediately."""
        storage = StorageManager(mock_coordinator)
        storage._data = storage.create_empty_storage()

        with patch.object(storage, "_should_save_immediately", return_value=True):
            with patch.object(storage, "_perform_save") as mock_perform_save:
                await storage.async_save_instance_data(
                    "test_entry", mock_entity_manager, force=False
                )

                mock_perform_save.assert_called_once()

    async def test_async_save_instance_data_debounced(
        self, mock_coordinator: Mock, mock_entity_manager: Mock
    ) -> None:
        """Test saving instance data with debouncing."""
        storage = StorageManager(mock_coordinator)
        storage._data = storage.create_empty_storage()

        with patch.object(storage, "_should_save_immediately", return_value=False):
            with patch.object(storage, "_schedule_debounced_save") as mock_schedule:
                await storage.async_save_instance_data(
                    "test_entry", mock_entity_manager, force=False
                )

                mock_schedule.assert_called_once()

    async def test_async_save_instance_data_forced(
        self, mock_coordinator: Mock, mock_entity_manager: Mock
    ) -> None:
        """Test saving instance data when forced."""
        storage = StorageManager(mock_coordinator)
        storage._data = storage.create_empty_storage()

        with patch.object(storage, "_perform_save") as mock_perform_save:
            await storage.async_save_instance_data(
                "test_entry", mock_entity_manager, force=True
            )

            mock_perform_save.assert_called_once()

    async def test_schedule_debounced_save(self, mock_coordinator: Mock) -> None:
        """Test scheduling debounced save."""
        storage = StorageManager(mock_coordinator)

        instance_data = {"test": "data"}
        data_hash = "test_hash"

        with patch("homeassistant.helpers.event.async_call_later") as mock_call_later:
            await storage._schedule_debounced_save(
                "test_entry", instance_data, data_hash
            )

            mock_call_later.assert_called_once()

    async def test_debounced_save_worker(self, mock_coordinator: Mock) -> None:
        """Test debounced save worker."""
        storage = StorageManager(mock_coordinator)

        instance_data = {"test": "data"}
        data_hash = "test_hash"

        with patch.object(storage, "_perform_save") as mock_perform_save:
            await storage._debounced_save_worker("test_entry", instance_data, data_hash)

            # Should clean up pending saves and perform save
            assert "test_entry" not in storage._pending_saves
            mock_perform_save.assert_called_once()

    async def test_perform_save_data_changed(self, mock_coordinator: Mock) -> None:
        """Test performing save when data has changed."""
        storage = StorageManager(mock_coordinator)
        storage._data = storage.create_empty_storage()

        instance_data = {"test": "data"}
        data_hash = "new_hash"

        # Set different last hash
        storage._last_save_hash["test_entry"] = "old_hash"

        with patch.object(storage, "async_save") as mock_save:
            await storage._perform_save("test_entry", instance_data, data_hash)

            # Should update data and save
            assert storage._data["instances"]["test_entry"] == instance_data
            assert storage._last_save_hash["test_entry"] == data_hash
            mock_save.assert_called_once()

    async def test_perform_save_data_unchanged(self, mock_coordinator: Mock) -> None:
        """Test performing save when data is unchanged."""
        storage = StorageManager(mock_coordinator)
        storage._data = storage.create_empty_storage()

        instance_data = {"test": "data"}
        data_hash = "same_hash"

        # Set same last hash
        storage._last_save_hash["test_entry"] = "same_hash"

        with patch.object(storage, "async_save") as mock_save:
            await storage._perform_save("test_entry", instance_data, data_hash)

            # Should not save when data hasn't changed
            mock_save.assert_not_called()

    async def test_force_save_now(self, mock_coordinator: Mock) -> None:
        """Test forcing immediate save."""
        storage = StorageManager(mock_coordinator)

        # Setup pending saves
        storage._pending_saves = {
            "entry1": {"data": "test1", "hash": "hash1"},
            "entry2": {"data": "test2", "hash": "hash2"},
        }

        with patch.object(storage, "_perform_save") as mock_perform_save:
            await storage._force_save_now()

            # Should perform all pending saves
            assert mock_perform_save.call_count == 2
            assert storage._pending_saves == {}

    async def test_async_reset(self, mock_coordinator: Mock) -> None:
        """Test resetting storage."""
        storage = StorageManager(mock_coordinator)

        # Setup some state
        storage._data = {"old": "data"}
        storage._last_save_hash = {"entry": "hash"}
        storage._pending_saves = {"entry": "data"}

        with patch.object(storage, "async_save") as mock_save:
            await storage.async_reset()

            # Should reset to empty state
            assert storage._data == storage.create_empty_storage()
            assert storage._last_save_hash == {}
            assert storage._pending_saves == {}
            mock_save.assert_called_once()

    async def test_async_load_with_compatibility_check_compatible(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - compatible version."""
        storage = StorageManager(mock_coordinator)

        # Setup compatible data
        compatible_data = {"entity_manager": {"entities": {}}}
        storage._data = {
            "version": 1,
            "minor_version": 0,
            "instances": {"test_entry": compatible_data},
        }

        result, needs_migration = await storage.async_load_with_compatibility_check(
            "test_entry", 1
        )

        assert result == compatible_data
        assert needs_migration is False

    async def test_async_load_with_compatibility_check_needs_migration(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - needs migration."""
        storage = StorageManager(mock_coordinator)

        # Setup old version data
        old_data = {"entity_manager": {"entities": {}}}
        storage._data = {
            "version": 1,
            "minor_version": 0,
            "instances": {"test_entry": old_data},
        }

        result, needs_migration = await storage.async_load_with_compatibility_check(
            "test_entry",
            2,  # Newer config entry version
        )

        assert result == old_data
        assert needs_migration is True

    async def test_async_load_with_compatibility_check_not_found(
        self, mock_coordinator: Mock
    ) -> None:
        """Test loading with compatibility check - data not found."""
        storage = StorageManager(mock_coordinator)

        storage._data = {"version": 1, "minor_version": 0, "instances": {}}

        result, needs_migration = await storage.async_load_with_compatibility_check(
            "test_entry", 1
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
            "version": 1,
            "minor_version": 0,
            "instances": {
                "test_entry": {
                    "entity_manager": {
                        "entities": {
                            "test_entity": {
                                "entity_id": "test_entity",
                                "probability": 0.5,
                            }
                        }
                    }
                }
            },
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
            "probability": 0.5,
            "state": "on",
            "is_active": True,
            "available": True,
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

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a comprehensive mock coordinator."""
        coordinator = Mock()
        coordinator.hass = Mock()
        coordinator.config_entry = Mock()
        coordinator.config_entry.entry_id = "test_entry_id"
        coordinator.config_entry.state = "loaded"

        # Mock entity manager with comprehensive data
        entity_manager = Mock(spec=EntityManager)
        entity_manager.to_dict.return_value = {
            "entities": {
                "binary_sensor.motion1": {
                    "entity_id": "binary_sensor.motion1",
                    "type": "motion",
                    "probability": 0.7,
                    "state": "on",
                    "is_active": True,
                    "available": True,
                    "last_updated": dt_util.utcnow().isoformat(),
                    "prior": {
                        "prior": 0.35,
                        "prob_given_true": 0.8,
                        "prob_given_false": 0.1,
                        "last_updated": dt_util.utcnow().isoformat(),
                    },
                    "decay": {
                        "is_decaying": False,
                        "decay_start_time": None,
                        "decay_start_probability": 0.0,
                        "decay_window": 300,
                        "decay_enabled": True,
                        "decay_factor": 1.0,
                    },
                }
            }
        }
        coordinator.entity_manager = entity_manager

        return coordinator

    @patch("homeassistant.helpers.storage.Store.async_load")
    @patch("homeassistant.helpers.storage.Store.async_save")
    async def test_full_storage_lifecycle(
        self, mock_save: AsyncMock, mock_load: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test complete storage lifecycle from initialization to shutdown."""
        # Start with no existing data
        mock_load.return_value = None

        storage = StorageManager(mock_coordinator)

        # Initialize storage
        await storage.async_initialize()
        assert storage._data == storage.create_empty_storage()

        # Save some data
        await storage.async_save_instance_data(
            "test_entry", mock_coordinator.entity_manager, force=True
        )

        # Verify data was stored
        assert "test_entry" in storage._data["instances"]
        assert mock_save.call_count >= 1

        # Test cleanup
        active_entries = {"test_entry", "another_entry"}
        await storage.async_cleanup_orphaned_instances(active_entries)

        # Test shutdown
        await storage.async_shutdown()

        # Should have saved final state
        assert mock_save.call_count >= 2

    @patch("homeassistant.helpers.storage.Store.async_load")
    async def test_storage_with_existing_data(
        self, mock_load: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test storage initialization with existing data."""
        existing_data = {
            "version": 1,
            "minor_version": 0,
            "instances": {
                "existing_entry": {
                    "entity_manager": {
                        "entities": {
                            "sensor.test": {
                                "entity_id": "sensor.test",
                                "probability": 0.3,
                                "state": "off",
                            }
                        }
                    }
                }
            },
        }
        mock_load.return_value = existing_data

        storage = StorageManager(mock_coordinator)
        await storage.async_initialize()

        # Should load existing data
        assert storage._data == existing_data

        # Should be able to access existing instance data
        instance_data = await storage.async_load_instance_data("existing_entry")
        assert instance_data is not None
        assert "entity_manager" in instance_data

    async def test_debounced_save_optimization(self, mock_coordinator: Mock) -> None:
        """Test that debounced saves work correctly for optimization."""
        storage = StorageManager(mock_coordinator)
        storage._data = storage.create_empty_storage()

        # Mock should_save_immediately to return False for debouncing
        with patch.object(storage, "_should_save_immediately", return_value=False):
            with patch.object(storage, "_schedule_debounced_save") as mock_schedule:
                # Save same data multiple times quickly
                for i in range(5):
                    await storage.async_save_instance_data(
                        "test_entry", mock_coordinator.entity_manager, force=False
                    )

                # Should have scheduled debounced saves
                assert mock_schedule.call_count == 5

                # Verify pending saves are tracked
                assert "test_entry" in storage._pending_saves

    async def test_error_handling_during_save(self, mock_coordinator: Mock) -> None:
        """Test error handling during save operations."""
        storage = StorageManager(mock_coordinator)
        storage._data = storage.create_empty_storage()

        # Mock async_save to raise an exception
        with patch.object(storage, "async_save", side_effect=Exception("Save failed")):
            # Should not raise exception - should handle gracefully
            await storage.async_save_instance_data(
                "test_entry", mock_coordinator.entity_manager, force=True
            )

            # Data should still be updated in memory even if save failed
            assert "test_entry" in storage._data["instances"]
