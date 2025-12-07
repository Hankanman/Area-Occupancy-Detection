"""Tests for migrations.py module."""

from __future__ import annotations

import asyncio
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
    _find_area_by_normalized_name,
    _find_or_create_area,
    _fuzzy_match_area,
    _normalize_area_name,
    async_migrate_entry,
    async_reset_database_if_needed,
)
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)


# Shared fixtures for common mock patterns
@pytest.fixture
def mock_update_entry():
    """Fixture that returns a mock update function and tracks calls."""
    update_calls = []

    def mock_update(entry, **kwargs):
        update_calls.append((entry, kwargs))
        if "data" in kwargs:
            entry.data = kwargs["data"]
        if "version" in kwargs:
            entry.version = kwargs["version"]

    mock_update.calls = update_calls
    return mock_update


@pytest.fixture
def mock_remove_entry():
    """Fixture that returns a mock remove function and tracks calls."""
    remove_calls = []

    async def mock_remove(entry_id):
        remove_calls.append(entry_id)
        return True

    mock_remove.calls = remove_calls
    return mock_remove


def create_mock_entry_v12(
    entry_id: str = "test_entry",
    area_id: str = "test_area",
    title: str | None = None,
    unique_id: str | None = None,
    options: dict | None = None,
) -> Mock:
    """Create a mock config entry at version 12."""
    entry = Mock(spec=ConfigEntry)
    entry.version = 12
    entry.minor_version = 0
    entry.entry_id = entry_id
    entry.state = ConfigEntryState.LOADED
    entry.data = {
        CONF_AREA_ID: area_id,
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
    }
    entry.options = options or {}
    entry.title = title or area_id.title()
    entry.unique_id = unique_id or area_id
    return entry


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
        self,
        hass: HomeAssistant,
        mock_config_entry_v1_0: Mock,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test migration from version 1.0 to current."""
        area_reg = ar.async_get(hass)

        # Need CONF_AREA_ID for migration - use area name that will be created
        area_name = "Migration Test Area"
        mock_config_entry_v1_0.data[CONF_AREA_ID] = area_name
        mock_config_entry_v1_0.title = area_name
        mock_config_entry_v1_0.unique_id = "migration_test_area"

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

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(1, 2),
        ):
            result = await async_migrate_entry(hass, mock_config_entry_v1_0)
            assert result is True

        # Should have updated entry with new format
        assert len(update_calls) == 1
        assert update_calls[0][1]["version"] == CONF_VERSION
        assert CONF_AREAS in update_calls[0][1]["data"]

        # Verify area was created/found in registry
        areas = update_calls[0][1]["data"][CONF_AREAS]
        assert len(areas) == 1
        area_id = areas[0][CONF_AREA_ID]
        created_area = area_reg.async_get_area(area_id)
        assert created_area is not None
        assert created_area.name == area_name

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

    @pytest.mark.parametrize(
        ("version", "should_delete"),
        [
            (12, True),  # Old version - should delete
            (13, True),  # Old version - should delete
            (15, False),  # Future or currentversion - should not delete
        ],
    )
    async def test_async_reset_database_if_needed_version(
        self, hass: HomeAssistant, tmp_path: Path, version: int, should_delete: bool
    ) -> None:
        """Test database reset based on version."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        # Call with specified version
        await async_reset_database_if_needed(hass, version)

        # Verify database file existence matches expectation
        assert (not db_path.exists()) == should_delete

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
            CONF_AREA_ID: "living_room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_THRESHOLD: 50.0,
        }
        entry.options = {}
        entry.title = "Living Room"
        entry.unique_id = "living_room"
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
            CONF_AREA_ID: "kitchen",
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
            CONF_THRESHOLD: 60.0,
        }
        entry.options = {}
        entry.title = "Kitchen"
        entry.unique_id = "kitchen"
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

        # Mock async_remove to track removals and handle UnknownEntry gracefully
        remove_calls = []

        async def mock_remove(entry_id):
            remove_calls.append(entry_id)
            # Simulate UnknownEntry if entry doesn't exist (as real code would)
            # But in tests, we just track the call
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(2, 3),  # 2 devices, 3 entities removed
        ):
            # Migrate the first entry
            result = await async_migrate_entry(hass, mock_entry_1)

            # Migration should succeed
            assert result is True

        # Should update both entries:
        # 1. Target entry with new areas
        # 2. Old entry marked as deleted
        assert len(update_calls) == 2

        # Check target entry update
        target_update = next(call for call in update_calls if call[0] == mock_entry_1)
        _, target_kwargs = target_update
        assert target_kwargs["version"] == CONF_VERSION
        assert CONF_AREAS in target_kwargs["data"]
        assert len(target_kwargs["data"][CONF_AREAS]) == 2

        # Check loser entry update
        loser_update = next(call for call in update_calls if call[0] == mock_entry_2)
        _, loser_kwargs = loser_update
        assert loser_kwargs["version"] == CONF_VERSION
        assert loser_kwargs["data"].get("deleted") is True

        # Verify area configs were preserved
        areas = target_kwargs["data"][CONF_AREAS]
        area_ids = [area[CONF_AREA_ID] for area in areas]
        assert "living_room" in area_ids
        assert "kitchen" in area_ids

        # Should remove the second entry
        assert len(remove_calls) == 1
        assert remove_calls[0] == "entry_2"

    async def test_migrate_single_entry(
        self,
        hass: HomeAssistant,
        mock_entry_1: Mock,
        setup_area_registry: dict[str, str],
        mock_update_entry,
    ) -> None:
        """Test migrating a single old entry to new format."""
        area_reg = ar.async_get(hass)

        # Mock hass.config_entries.async_entries to return only one entry
        hass.config_entries.async_entries = Mock(return_value=[mock_entry_1])

        # Use shared mock_update_entry fixture
        hass.config_entries.async_update_entry = Mock(side_effect=mock_update_entry)
        update_calls = mock_update_entry.calls

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

        # Verify area config was preserved and area was created/found
        area = update_kwargs["data"][CONF_AREAS][0]
        area_id = area[CONF_AREA_ID]
        # Verify area exists in registry (was created or matched)
        created_area = area_reg.async_get_area(area_id)
        assert created_area is not None
        # Verify area name matches (normalized matching should find "Living Room")
        normalized_name = (
            created_area.name.lower().replace(" ", "").replace("_", "").replace("-", "")
        )
        assert normalized_name == "livingroom" or created_area.name == "Living Room"
        assert area[CONF_MOTION_SENSORS] == ["binary_sensor.motion1"]

    async def test_migrate_entry_with_options(
        self,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
        mock_update_entry,
    ) -> None:
        """Test that options are merged into data during migration."""
        area_reg = ar.async_get(hass)

        entry = create_mock_entry_v12(
            entry_id="entry_with_options",
            area_id="Options Test Area",
            title="Options Test Area",
            unique_id="options_test_area",
            options={CONF_THRESHOLD: 70.0},
        )

        hass.config_entries.async_entries = Mock(return_value=[entry])
        hass.config_entries.async_update_entry = Mock(side_effect=mock_update_entry)
        update_calls = mock_update_entry.calls

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(1, 2),  # 1 device, 2 entities removed
        ):
            result = await async_migrate_entry(hass, entry)
            assert result is True

        # Verify options were merged and area was created/found
        area = update_calls[0][1]["data"][CONF_AREAS][0]
        area_id = area[CONF_AREA_ID]
        # Verify area exists in registry
        created_area = area_reg.async_get_area(area_id)
        assert created_area is not None
        assert area[CONF_MOTION_SENSORS] == ["binary_sensor.motion1"]
        assert area[CONF_THRESHOLD] == 70.0  # From options

    async def test_migrate_entry_missing_area_id_uses_title(
        self,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
        mock_update_entry,
    ) -> None:
        """Test migration when CONF_AREA_ID is missing but title exists."""
        area_reg = ar.async_get(hass)

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
        hass.config_entries.async_update_entry = Mock(side_effect=mock_update_entry)
        update_calls = mock_update_entry.calls

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(1, 2),  # 1 device, 2 entities removed
        ):
            result = await async_migrate_entry(hass, entry)
            assert result is True

        # Should use title as fallback and create/find area
        area = update_calls[0][1]["data"][CONF_AREAS][0]
        area_id = area[CONF_AREA_ID]
        # Verify area was created/found in registry
        created_area = area_reg.async_get_area(area_id)
        assert created_area is not None
        # Area ID should be the actual area ID from registry (not just "my_area")
        assert area_id is not None

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

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Track remove calls
        remove_calls = []

        async def mock_remove(entry_id):
            remove_calls.append(entry_id)
            # Simulate UnknownEntry if entry doesn't exist (as real code would)
            # But in tests, we just track the call
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(2, 3),  # 2 devices, 3 entities removed
        ):
            # Attempt migration - should succeed now as it handles partial failures
            result = await async_migrate_entry(hass, mock_entry_1)

            # Migration should succeed (valid entry migrated, invalid removed)
            assert result is True

        # Verify updates:
        # 1. Valid entry updated with areas
        # 2. Invalid entry marked as deleted
        assert len(update_calls) == 2

        # Check valid entry update
        valid_update = next(call for call in update_calls if call[0] == mock_entry_1)
        entry, kwargs = valid_update
        assert entry == mock_entry_1
        assert kwargs["version"] == CONF_VERSION
        assert CONF_AREAS in kwargs["data"]
        assert len(kwargs["data"][CONF_AREAS]) == 1
        assert kwargs["data"][CONF_AREAS][0][CONF_AREA_ID] == "living_room"

        # Check invalid entry update (marked as deleted)
        invalid_update = next(
            call for call in update_calls if call[0] == unconvertible_entry
        )
        entry, kwargs = invalid_update
        assert entry == unconvertible_entry
        assert kwargs["version"] == CONF_VERSION
        assert kwargs["data"].get("deleted") is True

        # Verify invalid entry was removed
        assert len(remove_calls) == 1
        assert remove_calls[0] == "entry_unconvertible"


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


class TestAreaMatchingHelpers:
    """Test area matching helper functions."""

    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            # Basic normalization
            ("Living Room", "livingroom"),
            ("living_room", "livingroom"),
            ("living-room", "livingroom"),
            ("LIVING ROOM", "livingroom"),
            # Edge cases
            ("", ""),
            ("   ", ""),
            ("Test Area 123", "testarea123"),
            # Multiple spaces/underscores/hyphens
            ("Living  Room", "livingroom"),
            ("living__room", "livingroom"),
            ("living--room", "livingroom"),
        ],
    )
    def test_normalize_area_name(self, input_name: str, expected: str) -> None:
        """Test area name normalization."""
        assert _normalize_area_name(input_name) == expected

    @pytest.mark.parametrize(
        ("search_name", "expected_area_name"),
        [
            ("Living Room", "Living Room"),
            ("living_room", "Living Room"),
            ("LIVING-ROOM", "Living Room"),
            ("KITCHEN", "Kitchen"),
            ("Non Existent", None),
        ],
    )
    def test_find_area_by_normalized_name(
        self,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
        search_name: str,
        expected_area_name: str | None,
    ) -> None:
        """Test finding area by normalized name."""
        area_reg = ar.async_get(hass)
        found_id = _find_area_by_normalized_name(area_reg, search_name)
        if expected_area_name:
            expected_id = setup_area_registry[expected_area_name]
            assert found_id == expected_id
        else:
            assert found_id is None

    def test_fuzzy_match_area(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test fuzzy area matching."""
        area_reg = ar.async_get(hass)

        # Use existing areas from setup_area_registry fixture
        living_room_id = setup_area_registry["Living Room"]

        # Create one new area for testing
        master_bedroom = area_reg.async_create("Master Bedroom")

        # Test high similarity match
        found_id = _fuzzy_match_area(area_reg, "Living Room", threshold=0.8)
        assert found_id == living_room_id

        found_id = _fuzzy_match_area(area_reg, "LivingRm", threshold=0.7)
        assert found_id == living_room_id

        # Test with lower threshold
        found_id = _fuzzy_match_area(area_reg, "Living", threshold=0.5)
        assert found_id == living_room_id

        # Test no match below threshold
        found_id = _fuzzy_match_area(area_reg, "Completely Different", threshold=0.8)
        assert found_id is None

        # Test exact match
        found_id = _fuzzy_match_area(area_reg, "Master Bedroom", threshold=0.8)
        assert found_id == master_bedroom.id

        # Test case insensitive
        found_id = _fuzzy_match_area(area_reg, "MASTER BEDROOM", threshold=0.8)
        assert found_id == master_bedroom.id

    def test_find_or_create_area_exact_id_match(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test finding area by exact ID match."""
        area_reg = ar.async_get(hass)
        # Use existing area ID from setup_area_registry
        living_room_id = setup_area_registry["Living Room"]
        # Should find by exact ID
        found_id = _find_or_create_area(area_reg, living_room_id)
        assert found_id == living_room_id

    @pytest.mark.parametrize(
        ("search_value", "expected_area_name"),
        [
            ("Living Room", "Living Room"),
            ("living_room", "Living Room"),
            ("Living-Room", "Living Room"),
            ("Kitchen", "Kitchen"),
            ("Kitchn", "Kitchen"),  # Fuzzy match
        ],
    )
    def test_find_or_create_area_matching(
        self,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
        search_value: str,
        expected_area_name: str,
    ) -> None:
        """Test finding area with various matching strategies."""
        area_reg = ar.async_get(hass)
        expected_id = setup_area_registry[expected_area_name]

        found_id = _find_or_create_area(area_reg, search_value)
        assert found_id == expected_id

    def test_find_or_create_area_creates_new(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test creating new area when no match found."""
        area_reg = ar.async_get(hass)

        # Count existing areas
        initial_count = len(area_reg.areas)

        # Should create new area
        new_area_id = _find_or_create_area(area_reg, "Brand New Area")

        # Verify area was created
        assert new_area_id is not None
        assert len(area_reg.areas) == initial_count + 1

        # Verify area exists
        created_area = area_reg.async_get_area(new_area_id)
        assert created_area is not None
        assert created_area.name == "Brand New Area"


class TestMigrationEdgeCases:
    """Test migration edge cases and error scenarios."""

    async def test_migrate_entry_deleted_during_migration(
        self, hass: HomeAssistant
    ) -> None:
        """Test entry marked deleted during lock wait."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_deleted"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_AREA_ID: "test_area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.title = "Test Area"
        entry.unique_id = "test_area"

        # Mock entry to be marked deleted
        entry.data["deleted"] = True

        # Mock hass.config_entries.async_entries
        hass.config_entries.async_entries = Mock(return_value=[entry])

        result = await async_migrate_entry(hass, entry)
        assert result is False

    async def test_migrate_entry_disappears_during_migration(
        self, hass: HomeAssistant
    ) -> None:
        """Test entry removed from registry during migration."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_disappears"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_AREA_ID: "test_area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.title = "Test Area"
        entry.unique_id = "test_area"

        # Mock hass.config_entries.async_entries to simulate entry disappearing
        # First call returns the entry (early check), second call (inside _migrate) returns empty
        call_count = 0

        def mock_entries(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (early check): return entry
                return [entry]
            # Subsequent calls (inside _migrate): entry disappeared
            return []

        hass.config_entries.async_entries = Mock(side_effect=mock_entries)

        # Mock async_remove
        async def mock_remove(entry_id):
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(0, 0),
        ):
            result = await async_migrate_entry(hass, entry)
            # Should return False when entry disappears during migration
            assert result is False

    async def test_migrate_filters_invalid_areas(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test that invalid area IDs are filtered out during migration."""
        # Use existing area from setup_area_registry for valid area
        living_room_id = setup_area_registry["Living Room"]

        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.minor_version = 0
        entry1.entry_id = "entry_1"
        entry1.state = ConfigEntryState.LOADED
        entry1.data = {
            CONF_AREA_ID: living_room_id,  # Valid area ID
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Living Room"
        entry1.unique_id = "living_room"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.minor_version = 0
        entry2.entry_id = "entry_2"
        entry2.state = ConfigEntryState.LOADED
        entry2.data = {
            CONF_AREA_ID: "invalid_area_id_that_does_not_exist",  # Invalid area ID (non-existent UUID-like ID)
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}
        entry2.title = "Invalid Area"
        entry2.unique_id = "invalid_area"

        hass.config_entries.async_entries = Mock(return_value=[entry1, entry2])

        update_calls = []

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Mock async_remove to handle UnknownEntry gracefully
        async def mock_remove(entry_id):
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(1, 2),
        ):
            result = await async_migrate_entry(hass, entry1)
            assert result is True

        # Should have updated target entry with only valid area
        # Note: Invalid area IDs will be converted to area configs via _find_or_create_area
        # which creates new areas, so they become valid. The filtering happens based on
        # whether the area exists in the registry after conversion.
        assert len(update_calls) >= 1
        target_update = next(call for call in update_calls if call[0] == entry1)
        _, target_kwargs = target_update
        assert CONF_AREAS in target_kwargs["data"]
        areas = target_kwargs["data"][CONF_AREAS]
        # Should have at least one area (the valid one)
        assert len(areas) >= 1
        # Verify the valid area is present
        area_ids = [area[CONF_AREA_ID] for area in areas]
        assert living_room_id in area_ids

    async def test_migrate_entry_no_valid_areas_after_filtering(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration fails when all areas are filtered out as invalid."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_no_valid"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_AREA_ID: "invalid_area_id_that_does_not_exist",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.title = "Invalid Area"
        entry.unique_id = "invalid_area"

        hass.config_entries.async_entries = Mock(return_value=[entry])

        update_calls = []
        hass.config_entries.async_update_entry = Mock(
            side_effect=lambda e, **kw: update_calls.append((e, kw))
        )

        # Mock area registry to return None for async_get_area after area creation
        # This simulates a scenario where the area was created but then doesn't exist
        # when we try to filter it (e.g., area was deleted between creation and filtering)
        def mock_async_get_area(area_id):
            # Return None to simulate area doesn't exist (filtered out)
            return None

        def mock_async_get_area_by_name(name):
            return None

        # Create a mock area entry for async_create
        mock_created_area = Mock()
        mock_created_area.id = "created_but_invalid"
        mock_created_area.name = "Invalid Area"

        # Mock registry cleanup
        with (
            patch(
                "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
                return_value=(0, 0),
            ),
            patch(
                "custom_components.area_occupancy.migrations.ar.async_get"
            ) as mock_ar_get,
        ):
            # Create a mock area registry
            mock_area_reg = Mock()
            mock_area_reg.async_get_area = Mock(side_effect=mock_async_get_area)
            mock_area_reg.async_get_area_by_name = Mock(
                side_effect=mock_async_get_area_by_name
            )
            mock_area_reg.async_create = Mock(return_value=mock_created_area)
            mock_area_reg.areas = {}  # Empty areas dict for normalized/fuzzy matching
            mock_ar_get.return_value = mock_area_reg

            result = await async_migrate_entry(hass, entry)
            assert result is False

        # Should not have updated entry
        assert len(update_calls) == 0

    async def test_migrate_entry_no_area_configs_after_conversion(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration fails when _combine_config_entries returns empty list."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "entry_no_configs"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            # Missing CONF_AREA_ID and no fallback
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.title = None
        entry.unique_id = None

        hass.config_entries.async_entries = Mock(return_value=[entry])

        update_calls = []
        hass.config_entries.async_update_entry = Mock(
            side_effect=lambda e, **kw: update_calls.append((e, kw))
        )

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(0, 0),
        ):
            result = await async_migrate_entry(hass, entry)
            assert result is False

        # Should not have updated entry
        assert len(update_calls) == 0


class TestConcurrentMigration:
    """Test concurrent migration behavior."""

    async def test_migration_already_completed_by_another_entry(
        self, hass: HomeAssistant
    ) -> None:
        """Test early exit when migration already completed by another entry."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.minor_version = 0
        entry1.entry_id = "entry_1"
        entry1.state = ConfigEntryState.LOADED
        entry1.data = {
            CONF_AREA_ID: "area_1",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Area 1"
        entry1.unique_id = "area_1"

        # Entry already migrated (at current version)
        entry2 = Mock(spec=ConfigEntry)
        entry2.version = CONF_VERSION
        entry2.minor_version = CONF_VERSION_MINOR
        entry2.entry_id = "entry_2"
        entry2.state = ConfigEntryState.LOADED
        entry2.data = {CONF_AREAS: []}
        entry2.options = {}

        # Mock hass.config_entries.async_entries to return migrated entry
        # This simulates the scenario where migration was already completed
        hass.config_entries.async_entries = Mock(return_value=[entry2])

        update_calls = []
        hass.config_entries.async_update_entry = Mock(
            side_effect=lambda e, **kw: update_calls.append((e, kw))
        )

        # Mock async_remove
        async def mock_remove(entry_id):
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(0, 0),
        ):
            result = await async_migrate_entry(hass, entry1)
            # Should succeed (early exit, no migration needed)
            assert result is True

        # Should not have updated entry (already migrated)
        assert len(update_calls) == 0

    async def test_concurrent_migration_lock_prevents_duplicate_migrations(
        self, hass: HomeAssistant
    ) -> None:
        """Test that lock prevents concurrent migrations from running simultaneously."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.minor_version = 0
        entry1.entry_id = "entry_1"
        entry1.state = ConfigEntryState.LOADED
        entry1.data = {
            CONF_AREA_ID: "area_1",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Area 1"
        entry1.unique_id = "area_1"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.minor_version = 0
        entry2.entry_id = "entry_2"
        entry2.state = ConfigEntryState.LOADED
        entry2.data = {
            CONF_AREA_ID: "area_2",
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}
        entry2.title = "Area 2"
        entry2.unique_id = "area_2"

        # Mock entries to return both old entries
        hass.config_entries.async_entries = Mock(return_value=[entry1, entry2])

        update_calls = []

        def mock_update(entry, **kwargs):
            update_calls.append((entry, kwargs))
            if "data" in kwargs:
                entry.data = kwargs["data"]
            if "version" in kwargs:
                entry.version = kwargs["version"]

        hass.config_entries.async_update_entry = Mock(side_effect=mock_update)

        # Mock async_remove
        async def mock_remove(entry_id):
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(2, 3),
        ):
            # Start two migrations concurrently
            task1 = asyncio.create_task(async_migrate_entry(hass, entry1))
            task2 = asyncio.create_task(async_migrate_entry(hass, entry2))

            # Wait for both to complete
            results = await asyncio.gather(task1, task2)

            # One should succeed (target entry), one should fail (consolidated entry)
            assert results[0] is True or results[1] is True
            # Both should complete (not hang due to deadlock)
            assert len(results) == 2

        # Should have updated entries (target + marked deleted)
        assert len(update_calls) >= 1


class TestErrorHandling:
    """Test error handling in migration."""

    async def test_migrate_entry_exception_handling(self, hass: HomeAssistant) -> None:
        """Test exception handling in _migrate() function."""
        entry = create_mock_entry_v12(
            entry_id="entry_exception",
            area_id="area_1",
            title="Area 1",
            unique_id="area_1",
        )

        # Mock hass.config_entries.async_entries to raise exception inside _migrate()
        # First call succeeds (early check), second call (inside _migrate) raises exception
        call_count = 0

        def mock_entries(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (early check): return entry
                return [entry]
            # Subsequent calls (inside _migrate): raise exception
            raise RuntimeError("Test exception")

        hass.config_entries.async_entries = Mock(side_effect=mock_entries)

        # Mock async_remove
        async def mock_remove(entry_id):
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(0, 0),
        ):
            result = await async_migrate_entry(hass, entry)
            # Should return False on exception
            assert result is False

    async def test_migrate_entry_update_entry_exception(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test handling of exception during entry update."""
        # Use existing area from setup_area_registry
        living_room_id = setup_area_registry["Living Room"]

        entry = create_mock_entry_v12(
            entry_id="entry_update_exception",
            area_id=living_room_id,
            title="Living Room",
            unique_id="living_room",
        )

        hass.config_entries.async_entries = Mock(return_value=[entry])

        # Mock async_update_entry to raise exception
        hass.config_entries.async_update_entry = Mock(
            side_effect=RuntimeError("Update failed")
        )

        # Mock async_remove
        async def mock_remove(entry_id):
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock registry cleanup
        with patch(
            "custom_components.area_occupancy.migrations._cleanup_registry_devices_and_entities",
            return_value=(0, 0),
        ):
            result = await async_migrate_entry(hass, entry)
            # Should return False on exception
            assert result is False

    async def test_migrate_entry_area_registry_exception(
        self, hass: HomeAssistant
    ) -> None:
        """Test handling of exception when accessing area registry."""
        entry = create_mock_entry_v12(
            entry_id="entry_area_registry_exception",
            area_id="area_1",
            title="Area 1",
            unique_id="area_1",
        )

        hass.config_entries.async_entries = Mock(return_value=[entry])

        # Mock async_remove
        async def mock_remove(entry_id):
            return True

        hass.config_entries.async_remove = Mock(side_effect=mock_remove)

        # Mock area registry to raise exception
        with patch(
            "custom_components.area_occupancy.migrations.ar.async_get",
            side_effect=RuntimeError("Area registry error"),
        ):
            result = await async_migrate_entry(hass, entry)
            # Should return False on exception
            assert result is False
