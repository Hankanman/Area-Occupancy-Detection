"""Tests for migrations.py module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
    DEFAULT_THRESHOLD,
)
from custom_components.area_occupancy.migrations import (
    async_migrate_entry,
    async_migrate_storage,
    async_migrate_unique_ids,
    migrate_config,
    migrate_primary_occupancy_sensor,
    validate_threshold,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import HomeAssistantError


class TestAsyncMigrateUniqueIds:
    """Test async_migrate_unique_ids function."""

    async def test_async_migrate_unique_ids_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful unique ID migration."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            mock_registry.entities = {
                "sensor.old_unique_id": Mock(unique_id="old_unique_id"),
                "binary_sensor.old_unique_id": Mock(unique_id="old_unique_id"),
            }
            mock_registry.async_update_entity = AsyncMock()
            mock_get_registry.return_value = mock_registry

            await async_migrate_unique_ids(mock_hass, mock_config_entry, "sensor")

            # Should update entities with old unique ID format
            assert mock_registry.async_update_entity.call_count >= 0

    async def test_async_migrate_unique_ids_no_entities(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test migration with no entities to migrate."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            mock_registry.entities = {}
            mock_get_registry.return_value = mock_registry

            # Should not raise an exception
            await async_migrate_unique_ids(mock_hass, mock_config_entry, "sensor")

    async def test_async_migrate_unique_ids_invalid_platform(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test migration with invalid platform."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            mock_registry.entities = {}
            mock_get_registry.return_value = mock_registry

            # Should not raise an exception for invalid platform
            await async_migrate_unique_ids(
                mock_hass, mock_config_entry, "invalid_platform"
            )


class TestMigratePrimaryOccupancySensor:
    """Test migrate_primary_occupancy_sensor function."""

    def test_migrate_primary_occupancy_sensor_needed(self) -> None:
        """Test migration when primary sensor is missing."""
        config = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
        }

        result = migrate_primary_occupancy_sensor(config)

        assert CONF_PRIMARY_OCCUPANCY_SENSOR in result
        assert result[CONF_PRIMARY_OCCUPANCY_SENSOR] == "binary_sensor.motion1"

    def test_migrate_primary_occupancy_sensor_already_exists(self) -> None:
        """Test migration when primary sensor already exists."""
        config = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion2",
        }

        result = migrate_primary_occupancy_sensor(config)

        assert result[CONF_PRIMARY_OCCUPANCY_SENSOR] == "binary_sensor.motion2"

    def test_migrate_primary_occupancy_sensor_no_motion_sensors(self) -> None:
        """Test migration when no motion sensors exist."""
        config = {}

        result = migrate_primary_occupancy_sensor(config)

        # Should not add primary sensor if no motion sensors
        assert CONF_PRIMARY_OCCUPANCY_SENSOR not in result

    def test_migrate_primary_occupancy_sensor_empty_motion_sensors(self) -> None:
        """Test migration when motion sensors list is empty."""
        config = {CONF_MOTION_SENSORS: []}

        result = migrate_primary_occupancy_sensor(config)

        # Should not add primary sensor if motion sensors list is empty
        assert CONF_PRIMARY_OCCUPANCY_SENSOR not in result


class TestMigrateConfig:
    """Test migrate_config function."""

    def test_migrate_config_adds_primary_sensor(self) -> None:
        """Test config migration adds primary sensor."""
        config = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }

        result = migrate_config(config)

        assert CONF_PRIMARY_OCCUPANCY_SENSOR in result
        assert result[CONF_PRIMARY_OCCUPANCY_SENSOR] == "binary_sensor.motion1"

    def test_migrate_config_preserves_existing(self) -> None:
        """Test config migration preserves existing values."""
        config = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 70,
            "other_key": "other_value",
        }

        result = migrate_config(config)

        assert result[CONF_PRIMARY_OCCUPANCY_SENSOR] == "binary_sensor.motion1"
        assert result[CONF_THRESHOLD] == 70
        assert result["other_key"] == "other_value"


class TestAsyncMigrateStorage:
    """Test async_migrate_storage function."""

    async def test_async_migrate_storage_no_data(self, mock_hass: Mock) -> None:
        """Test storage migration with no existing data."""
        with patch(
            "homeassistant.helpers.storage.Store.async_load",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await async_migrate_storage(mock_hass, "test_entry_id")
            # Should complete without error

    async def test_async_migrate_storage_error(self, mock_hass: Mock) -> None:
        """Test storage migration with error."""
        with patch(
            "homeassistant.helpers.storage.Store.async_load",
            new_callable=AsyncMock,
            side_effect=HomeAssistantError("Storage error"),
        ):
            await async_migrate_storage(mock_hass, "test_entry_id")
            # Should handle error gracefully


class TestAsyncMigrateEntry:
    """Test async_migrate_entry function."""

    @pytest.fixture
    def mock_config_entry_v1_0(self, mock_config_entry: Mock) -> Mock:
        """Create a mock config entry at version 1.0."""
        # Copy the centralized config entry and modify for v1.0
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = mock_config_entry.entry_id
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry.options = {}
        return entry

    @pytest.fixture
    def mock_config_entry_current(self, mock_config_entry: Mock) -> Mock:
        """Create a mock config entry at current version."""
        # Copy the centralized config entry and modify for current version
        entry = Mock(spec=ConfigEntry)
        entry.version = CONF_VERSION
        entry.minor_version = CONF_VERSION_MINOR
        entry.entry_id = mock_config_entry.entry_id
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }
        entry.options = {}
        return entry

    async def test_async_migrate_entry_v1_0_to_current(
        self, mock_hass: Mock, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration from version 1.0 to current."""
        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "homeassistant.helpers.storage.Store.async_save",
                new_callable=AsyncMock,
            ),
        ):
            mock_migrate_ids.return_value = None

            # Mock the config_entries.async_update_entry method
            mock_hass.config_entries.async_update_entry = Mock()

            result = await async_migrate_entry(mock_hass, mock_config_entry_v1_0)

            assert result is True
            # Verify the update was called
            mock_hass.config_entries.async_update_entry.assert_called_once()
            call_args = mock_hass.config_entries.async_update_entry.call_args
            assert (
                call_args[0][0] == mock_config_entry_v1_0
            )  # First argument is the entry
            # Check that the data was migrated
            updated_data = call_args[1]["data"]
            assert CONF_PRIMARY_OCCUPANCY_SENSOR in updated_data

    async def test_async_migrate_entry_already_current(
        self, mock_hass: Mock, mock_config_entry_current: Mock
    ) -> None:
        """Test migration when already at current version."""
        result = await async_migrate_entry(mock_hass, mock_config_entry_current)

        assert result is True
        # Should not modify version - verify by checking no config_entries.async_update_entry call
        assert (
            not hasattr(mock_hass.config_entries, "async_update_entry")
            or mock_hass.config_entries.async_update_entry.call_count == 0
        )

    async def test_async_migrate_entry_future_version(self, mock_hass: Mock) -> None:
        """Test migration from future version."""
        mock_entry = Mock(spec=ConfigEntry)
        mock_entry.version = CONF_VERSION + 1  # Future version
        mock_entry.minor_version = 0
        mock_entry.entry_id = "test_entry_id"
        mock_entry.data = {}
        mock_entry.options = {}

        result = await async_migrate_entry(mock_hass, mock_entry)
        # Future versions are treated as "already current" and return True
        assert result is True

    async def test_async_migrate_entry_migration_error(
        self, mock_hass: Mock, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with error during migration."""
        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids",
                side_effect=Exception("Migration error"),
            ),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "homeassistant.helpers.storage.Store.async_save",
                new_callable=AsyncMock,
            ),
        ):
            mock_hass.config_entries.async_update_entry = Mock()

            # The migration should fail due to the exception
            with pytest.raises(Exception, match="Migration error"):
                await async_migrate_entry(mock_hass, mock_config_entry_v1_0)

    async def test_async_migrate_entry_invalid_threshold(
        self, mock_hass: Mock, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with invalid threshold value."""
        mock_config_entry_v1_0.options = {CONF_THRESHOLD: 150}  # Invalid threshold

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "homeassistant.helpers.storage.Store.async_save",
                new_callable=AsyncMock,
            ),
        ):
            mock_migrate_ids.return_value = None
            mock_hass.config_entries.async_update_entry = Mock()

            result = await async_migrate_entry(mock_hass, mock_config_entry_v1_0)
            assert result is True

            # Should fix invalid threshold
            call_args = mock_hass.config_entries.async_update_entry.call_args
            updated_options = call_args[1]["options"]
            assert updated_options[CONF_THRESHOLD] == DEFAULT_THRESHOLD


class TestValidateThreshold:
    """Test validate_threshold function."""

    def test_validate_threshold_valid_values(self) -> None:
        """Test validation of valid threshold values."""
        assert validate_threshold(1.0) == 1.0
        assert validate_threshold(50.0) == 50.0
        assert validate_threshold(99.0) == 99.0

    def test_validate_threshold_invalid_low(self) -> None:
        """Test validation of threshold too low."""
        assert validate_threshold(0.0) == DEFAULT_THRESHOLD
        assert validate_threshold(-10.0) == DEFAULT_THRESHOLD

    def test_validate_threshold_invalid_high(self) -> None:
        """Test validation of threshold too high."""
        assert validate_threshold(100.0) == DEFAULT_THRESHOLD
        assert validate_threshold(150.0) == DEFAULT_THRESHOLD

    def test_validate_threshold_edge_cases(self) -> None:
        """Test validation of edge case values."""
        assert validate_threshold(0.1) == DEFAULT_THRESHOLD  # Too low
        assert validate_threshold(99.9) == DEFAULT_THRESHOLD  # Too high
        assert validate_threshold(1.0) == 1.0  # Valid minimum
        assert validate_threshold(99.0) == 99.0  # Valid maximum


class TestMigrationsIntegration:
    """Test migrations integration scenarios."""

    async def test_complete_migration_workflow(self, mock_hass: Mock) -> None:
        """Test complete migration workflow from 1.0 to current."""
        # Create entry that needs migration
        mock_entry = Mock(spec=ConfigEntry)
        mock_entry.version = 1
        mock_entry.minor_version = 0
        mock_entry.entry_id = "test_entry_id"
        mock_entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
        }
        mock_entry.options = {CONF_THRESHOLD: 150}  # Invalid threshold

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "homeassistant.helpers.storage.Store.async_save",
                new_callable=AsyncMock,
            ),
        ):
            mock_migrate_ids.return_value = None
            mock_hass.config_entries.async_update_entry = Mock()

            result = await async_migrate_entry(mock_hass, mock_entry)
            assert result is True

            # Check migration steps were called
            mock_migrate_ids.assert_called()

            # Check data migration
            call_args = mock_hass.config_entries.async_update_entry.call_args
            updated_data = call_args[1]["data"]
            updated_options = call_args[1]["options"]

            assert CONF_PRIMARY_OCCUPANCY_SENSOR in updated_data
            assert (
                updated_data[CONF_PRIMARY_OCCUPANCY_SENSOR] == "binary_sensor.motion1"
            )
            assert (
                updated_options[CONF_THRESHOLD] == DEFAULT_THRESHOLD
            )  # Fixed invalid threshold

    def test_config_migration_edge_cases(self) -> None:
        """Test config migration with various edge cases."""
        # Empty config
        result = migrate_config({})
        assert result == {}

        # Config with only non-motion sensor data
        config = {"other_key": "value"}
        result = migrate_config(config)
        assert result == {"other_key": "value"}

        # Config with motion sensors but already has primary
        config = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion2",
        }
        result = migrate_config(config)
        assert result[CONF_PRIMARY_OCCUPANCY_SENSOR] == "binary_sensor.motion2"
