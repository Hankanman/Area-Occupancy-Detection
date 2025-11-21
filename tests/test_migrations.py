"""Tests for migrations.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_MOTION_SENSORS,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
)
from custom_components.area_occupancy.migrations import (
    _cleanup_registry_devices_and_entities,
    async_migrate_entry,
    async_reset_database_if_needed,
)
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er


class TestAsyncMigrateEntry:
    """Test async_migrate_entry function."""

    @pytest.fixture
    def mock_config_entry_v1_0(self, mock_config_entry: Mock) -> Mock:
        """Create a mock config entry at version 1.0."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = mock_config_entry.entry_id
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}
        return entry

    @pytest.fixture
    def mock_config_entry_current(self, mock_config_entry: Mock) -> Mock:
        """Create a mock config entry at current version."""
        entry = Mock(spec=ConfigEntry)
        entry.version = CONF_VERSION
        entry.minor_version = CONF_VERSION_MINOR
        entry.entry_id = mock_config_entry.entry_id
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        return entry

    async def test_async_migrate_entry_v1_0_to_current(
        self, hass: HomeAssistant, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration from version 1.0 to current."""
        # Need CONF_AREA_ID for migration
        mock_config_entry_v1_0.data[CONF_AREA_ID] = "test_area"
        mock_config_entry_v1_0.title = "Test Area"
        mock_config_entry_v1_0.unique_id = "test_area"

        # Mock hass.config_entries.async_entries to return the entry
        hass.config_entries.async_entries = Mock(return_value=[mock_config_entry_v1_0])

        # Mock async_update_entry
        update_calls = []

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        result = await async_migrate_entry(hass, mock_config_entry_v1_0)
        assert result is True

        # Should have updated entry with new format
        assert len(update_calls) == 1
        assert update_calls[0][1]["version"] == CONF_VERSION
        assert CONF_AREAS in update_calls[0][1]["data"]

    async def test_async_migrate_entry_already_current(
        self, hass: HomeAssistant, mock_config_entry_current: Mock
    ) -> None:
        """Test migration when already at current version."""
        result = await async_migrate_entry(hass, mock_config_entry_current)
        assert result is True

    async def test_async_migrate_entry_future_version(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration from future version."""
        mock_entry = Mock(spec=ConfigEntry)
        mock_entry.version = CONF_VERSION + 1
        mock_entry.minor_version = 0
        mock_entry.entry_id = "test_entry_id"
        mock_entry.state = ConfigEntryState.LOADED
        mock_entry.data = {}
        mock_entry.options = {}

        result = await async_migrate_entry(hass, mock_entry)
        assert result is True


class TestAsyncResetDatabaseIfNeeded:
    """Test async_reset_database_if_needed function."""

    async def test_async_reset_database_if_needed_old_version(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database reset when version is old (< 13)."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        # Call with old version
        await async_reset_database_if_needed(hass, 12)

        # Database file should be deleted
        assert not db_path.exists()

    async def test_async_reset_database_if_needed_current_version(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database reset skipped for version >= 13."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        # Call with current version
        await async_reset_database_if_needed(hass, 13)

        # Database file should still exist (not deleted)
        assert db_path.exists()

    async def test_async_reset_database_if_needed_no_db_file(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test when database file doesn't exist."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()

        # Call with old version but no database file
        await async_reset_database_if_needed(hass, 12)

        # Should complete without error
        # No database file to delete, so nothing happens

    async def test_async_reset_database_if_needed_deletes_wal_files(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test that WAL files are also deleted."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        wal_path = storage_dir / "area_occupancy.db-wal"
        shm_path = storage_dir / "area_occupancy.db-shm"
        db_path.write_bytes(b"SQLite format 3")
        wal_path.write_bytes(b"WAL data")
        shm_path.write_bytes(b"SHM data")

        # Call with old version
        await async_reset_database_if_needed(hass, 12)

        # All files should be deleted
        assert not db_path.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()


class TestAsyncMigrateEntryAdditional:
    """Additional tests for async_migrate_entry function."""

    async def test_async_migrate_entry_database_reset(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test migration resets database for old versions."""
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_AREA_ID: "test_area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.title = "Test Area"
        entry.unique_id = "test_area"

        # Mock hass.config_entries.async_entries
        hass.config_entries.async_entries = Mock(return_value=[entry])

        # Mock async_update_entry
        def mock_update(entry, **kwargs):
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        # Database should be deleted for version < 13
        assert not db_path.exists()

    async def test_async_migrate_entry_no_database_reset_for_current_version(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test migration does not reset database for current version."""
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        entry = Mock(spec=ConfigEntry)
        entry.version = CONF_VERSION
        entry.minor_version = CONF_VERSION_MINOR
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        # Mock hass.config_entries.async_entries (entry already at current version, no migration)
        hass.config_entries.async_entries = Mock(return_value=[entry])

        # Mock async_update_entry (should not be called for current version)
        update_calls = []
        hass.config_entries.async_update_entry = Mock(
            side_effect=lambda e, **kw: update_calls.append((e, kw))
        )

        result = await async_migrate_entry(hass, entry)
        assert result is True
        # Database should still exist for version >= 13
        assert db_path.exists()
        # Should not update entry when already at current version
        assert len(update_calls) == 0


class TestMultipleEntriesMigration:
    """Test migration of multiple config entries."""

    @pytest.fixture
    def mock_entry_1(self) -> Mock:
        """Create first mock config entry with old format."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_1"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_AREA_ID: "area_1",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_THRESHOLD: 50.0,
        }
        entry.options = {}
        entry.title = "Area 1"
        entry.unique_id = "area_1"
        return entry

    @pytest.fixture
    def mock_entry_2(self) -> Mock:
        """Create second mock config entry with old format."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_2"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_AREA_ID: "area_2",
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
            CONF_THRESHOLD: 60.0,
        }
        entry.options = {}
        entry.title = "Area 2"
        entry.unique_id = "area_2"
        return entry

    async def test_migrate_multiple_entries(
        self, hass: HomeAssistant, mock_entry_1: Mock, mock_entry_2: Mock
    ) -> None:
        """Test migrating multiple old entries into one entry."""
        # Mock hass.config_entries.async_entries to return both entries
        hass.config_entries.async_entries = Mock(
            return_value=[mock_entry_1, mock_entry_2]
        )

        # Mock async_update_entry to capture updates
        update_calls = []

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            # Update the entry object
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Mock async_remove to track removals
        remove_calls = []
        hass.config_entries.async_remove = Mock(
            side_effect=lambda entry_id: remove_calls.append(entry_id)
        )

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(2, 3),  # 2 devices, 3 entities removed
        ):
            # Migrate the first entry
            result = await async_migrate_entry(hass, mock_entry_1)

            # Migration should succeed
            assert result is True

        # Should update the first entry with combined data
        assert len(update_calls) == 1
        updated_entry, update_kwargs = update_calls[0]
        assert updated_entry == mock_entry_1
        assert update_kwargs["version"] == CONF_VERSION
        assert CONF_AREAS in update_kwargs["data"]
        assert len(update_kwargs["data"][CONF_AREAS]) == 2

        # Verify area configs were preserved
        areas = update_kwargs["data"][CONF_AREAS]
        area_ids = [area[CONF_AREA_ID] for area in areas]
        assert "area_1" in area_ids
        assert "area_2" in area_ids

        # Should remove the second entry
        assert len(remove_calls) == 1
        assert remove_calls[0] == "entry_2"

    async def test_migrate_single_entry(
        self, hass: HomeAssistant, mock_entry_1: Mock
    ) -> None:
        """Test migrating a single old entry to new format."""
        # Mock hass.config_entries.async_entries to return only one entry
        hass.config_entries.async_entries = Mock(return_value=[mock_entry_1])

        # Mock async_update_entry
        update_calls = []

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(1, 2),  # 1 device, 2 entities removed
        ):
            # Migrate the entry
            result = await async_migrate_entry(hass, mock_entry_1)

            # Migration should succeed
            assert result is True

        # Should update entry with new format
        assert len(update_calls) == 1
        updated_entry, update_kwargs = update_calls[0]
        assert updated_entry == mock_entry_1
        assert update_kwargs["version"] == CONF_VERSION
        assert CONF_AREAS in update_kwargs["data"]
        assert len(update_kwargs["data"][CONF_AREAS]) == 1

        # Verify area config was preserved
        area = update_kwargs["data"][CONF_AREAS][0]
        assert area[CONF_AREA_ID] == "area_1"
        assert area[CONF_MOTION_SENSORS] == ["binary_sensor.motion1"]

    async def test_migrate_entry_with_options(self, hass: HomeAssistant) -> None:
        """Test that options are merged into data during migration."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_with_options"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_AREA_ID: "area_1",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {
            CONF_THRESHOLD: 70.0,
        }
        entry.title = "Area 1"
        entry.unique_id = "area_1"

        hass.config_entries.async_entries = Mock(return_value=[entry])

        update_calls = []

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(1, 2),  # 1 device, 2 entities removed
        ):
            result = await async_migrate_entry(hass, entry)
            assert result is True

        # Verify options were merged
        area = update_calls[0][1]["data"][CONF_AREAS][0]
        assert area[CONF_AREA_ID] == "area_1"
        assert area[CONF_MOTION_SENSORS] == ["binary_sensor.motion1"]
        assert area[CONF_THRESHOLD] == 70.0  # From options

    async def test_migrate_entry_missing_area_id_uses_title(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration when CONF_AREA_ID is missing but title exists."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_no_area_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.title = "My Area"
        entry.unique_id = "my_area"

        hass.config_entries.async_entries = Mock(return_value=[entry])

        update_calls = []

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(1, 2),  # 1 device, 2 entities removed
        ):
            result = await async_migrate_entry(hass, entry)
            assert result is True

        # Should use title as fallback
        area = update_calls[0][1]["data"][CONF_AREAS][0]
        assert area[CONF_AREA_ID] == "My Area"

    async def test_migrate_entry_missing_area_id_no_fallback(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration fails when CONF_AREA_ID is missing and no fallback."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_no_area_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.title = None
        entry.unique_id = None

        hass.config_entries.async_entries = Mock(return_value=[entry])

        result = await async_migrate_entry(hass, entry)
        assert result is False  # Should fail

    async def test_migrate_already_at_version_13(self, hass: HomeAssistant) -> None:
        """Test that migration skips entries already at version 13."""
        entry = Mock(spec=ConfigEntry)
        entry.version = CONF_VERSION
        entry.minor_version = CONF_VERSION_MINOR
        entry.entry_id = "entry_current"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_AREAS: []}
        entry.options = {}

        hass.config_entries.async_entries = Mock(return_value=[entry])

        update_calls = []
        hass.config_entries.async_update_entry = Mock(
            side_effect=lambda e, **kw: update_calls.append((e, kw))
        )

        result = await async_migrate_entry(hass, entry)
        assert result is True

        # Should not update entry
        assert len(update_calls) == 0

    async def test_migrate_multiple_entries_with_unconvertible_entry(
        self, hass: HomeAssistant, mock_entry_1: Mock
    ) -> None:
        """Test that an unconvertible entry prevents migration and leaves entries intact."""
        # Create an unconvertible entry (missing CONF_AREA_ID and no fallback)
        unconvertible_entry = Mock(spec=ConfigEntry)
        unconvertible_entry.version = 12
        unconvertible_entry.minor_version = 0
        unconvertible_entry.entry_id = "entry_unconvertible"
        unconvertible_entry.state = ConfigEntryState.LOADED
        unconvertible_entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        unconvertible_entry.options = {}
        unconvertible_entry.title = None
        unconvertible_entry.unique_id = None

        # Mock hass.config_entries.async_entries to return both entries
        hass.config_entries.async_entries = Mock(
            return_value=[mock_entry_1, unconvertible_entry]
        )

        # Track update calls
        update_calls = []
        original_data_1 = mock_entry_1.data.copy()
        original_data_unconvertible = unconvertible_entry.data.copy()

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Track remove calls
        remove_calls = []
        hass.config_entries.async_remove = Mock(
            side_effect=lambda entry_id: remove_calls.append(entry_id)
        )

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(2, 3),  # 2 devices, 3 entities removed
        ):
            # Attempt migration - should fail
            result = await async_migrate_entry(hass, mock_entry_1)

            # Migration should fail
            assert result is False

        # Verify no entries were updated
        assert len(update_calls) == 0

        # Verify no entries were removed
        assert len(remove_calls) == 0

        # Verify original entry data is still intact
        assert mock_entry_1.data == original_data_1
        assert unconvertible_entry.data == original_data_unconvertible

        # Verify entry versions are still unchanged
        assert mock_entry_1.version == 12
        assert unconvertible_entry.version == 12


class TestRegistryCleanup:
    """Test registry cleanup during migration."""

    async def test_cleanup_registry_devices_and_entities(
        self, hass: HomeAssistant
    ) -> None:
        """Test that devices and entities are removed from registries."""
        device_registry = dr.async_get(hass)
        entity_registry = er.async_get(hass)

        # Create mock entities and devices for test entries
        entry_id_1 = "entry_1"
        entry_id_2 = "entry_2"

        # Create mock entity entries using Mock objects
        entity_1 = Mock()
        entity_1.entity_id = "binary_sensor.area_1_occupancy"
        entity_1.config_entry_id = entry_id_1

        entity_2 = Mock()
        entity_2.entity_id = "binary_sensor.area_2_occupancy"
        entity_2.config_entry_id = entry_id_2

        # Add entities to registry
        entity_registry.entities = {
            entity_1.entity_id: entity_1,
            entity_2.entity_id: entity_2,
        }

        # Track remove calls
        remove_calls = []
        entity_registry.async_remove = Mock(
            side_effect=lambda eid: remove_calls.append(eid)
        )

        # Create mock device entries using Mock objects
        device_1 = Mock()
        device_1.id = "device_1"
        device_1.config_entries = {entry_id_1}

        device_2 = Mock()
        device_2.id = "device_2"
        device_2.config_entries = {entry_id_2}

        device_registry.devices = {
            device_1.id: device_1,
            device_2.id: device_2,
        }

        # Track device remove calls
        device_remove_calls = []
        device_registry.async_remove_device = Mock(
            side_effect=lambda did: device_remove_calls.append(did)
        )

        # Call cleanup
        (
            devices_removed,
            entities_removed,
        ) = await _cleanup_registry_devices_and_entities(hass, [entry_id_1, entry_id_2])

        # Verify entities were removed
        assert entities_removed == 2
        assert len(remove_calls) == 2
        assert entity_1.entity_id in remove_calls
        assert entity_2.entity_id in remove_calls

        # Verify devices were removed
        assert devices_removed == 2
        assert len(device_remove_calls) == 2
        assert device_1.id in device_remove_calls
        assert device_2.id in device_remove_calls

    async def test_cleanup_registry_handles_missing_entities(
        self, hass: HomeAssistant
    ) -> None:
        """Test that cleanup handles missing entities gracefully."""
        device_registry = dr.async_get(hass)
        entity_registry = er.async_get(hass)

        entry_id = "entry_1"

        # Mock entity registry with no entities
        entity_registry.entities = {}
        entity_registry.async_remove = Mock()

        # Mock device registry with no devices
        device_registry.devices = {}
        device_registry.async_remove_device = Mock()

        # Call cleanup - should not raise
        (
            devices_removed,
            entities_removed,
        ) = await _cleanup_registry_devices_and_entities(hass, [entry_id])

        # Verify no errors and counts are zero
        assert devices_removed == 0
        assert entities_removed == 0

    async def test_migrate_calls_registry_cleanup(self, hass: HomeAssistant) -> None:
        """Test that migration calls registry cleanup."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_1"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_AREA_ID: "area_1",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.title = "Area 1"
        entry.unique_id = "area_1"

        hass.config_entries.async_entries = Mock(return_value=[entry])

        update_calls = []

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Mock registry cleanup to verify it's called
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(2, 3),  # 2 devices, 3 entities removed
        ) as mock_cleanup:
            result = await async_migrate_entry(hass, entry)

            # Migration should succeed
            assert result is True

            # Registry cleanup should be called with all entry IDs
            mock_cleanup.assert_called_once()
            call_args = mock_cleanup.call_args
            assert call_args[0][1] == [entry.entry_id]  # Check entry_ids argument
