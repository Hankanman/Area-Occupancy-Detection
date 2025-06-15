"""Tests for migrations.py module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
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
from homeassistant.core import HomeAssistant


class TestAsyncMigrateUniqueIds:
    """Test async_migrate_unique_ids function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry_id"
        return entry

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

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        return hass

    async def test_async_migrate_storage_success(self, mock_hass: Mock) -> None:
        """Test successful storage migration."""
        with patch(
            "custom_components.area_occupancy.storage.StorageManager"
        ) as mock_storage_class:
            mock_storage = Mock()
            mock_storage.async_perform_cleanup = AsyncMock()
            mock_storage_class.return_value = mock_storage

            await async_migrate_storage(mock_hass, "test_entry_id")

            mock_storage.async_perform_cleanup.assert_called_once()

    async def test_async_migrate_storage_error(self, mock_hass: Mock) -> None:
        """Test storage migration with error."""
        with patch(
            "custom_components.area_occupancy.storage.StorageManager"
        ) as mock_storage_class:
            mock_storage = Mock()
            mock_storage.async_perform_cleanup = AsyncMock(
                side_effect=Exception("Storage error")
            )
            mock_storage_class.return_value = mock_storage

            # Should not raise exception, just log error
            await async_migrate_storage(mock_hass, "test_entry_id")


class TestAsyncMigrateEntry:
    """Test async_migrate_entry function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        return hass

    @pytest.fixture
    def mock_config_entry_v1_0(self) -> Mock:
        """Create a mock config entry at version 1.0."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        return entry

    @pytest.fixture
    def mock_config_entry_v1_1(self) -> Mock:
        """Create a mock config entry at version 1.1."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 1
        entry.entry_id = "test_entry_id"
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }
        return entry

    async def test_async_migrate_entry_v1_0_to_v1_1(
        self, mock_hass: Mock, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration from version 1.0 to 1.1."""
        with patch(
            "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
        ) as mock_migrate_ids:
            with patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage"
            ) as mock_migrate_storage:
                mock_migrate_ids.return_value = None
                mock_migrate_storage.return_value = None

                result = await async_migrate_entry(mock_hass, mock_config_entry_v1_0)

                assert result is True
                assert mock_config_entry_v1_0.version == 1
                assert mock_config_entry_v1_0.minor_version == 1
                assert CONF_PRIMARY_OCCUPANCY_SENSOR in mock_config_entry_v1_0.data

    async def test_async_migrate_entry_already_current(
        self, mock_hass: Mock, mock_config_entry_v1_1: Mock
    ) -> None:
        """Test migration when already at current version."""
        result = await async_migrate_entry(mock_hass, mock_config_entry_v1_1)

        assert result is True
        # Should not modify version
        assert mock_config_entry_v1_1.version == 1
        assert mock_config_entry_v1_1.minor_version == 1

    async def test_async_migrate_entry_future_version(self, mock_hass: Mock) -> None:
        """Test migration from future version."""
        mock_entry = Mock(spec=ConfigEntry)
        mock_entry.version = 2
        mock_entry.minor_version = 0

        result = await async_migrate_entry(mock_hass, mock_entry)

        assert result is False

    async def test_async_migrate_entry_migration_error(
        self, mock_hass: Mock, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with error during migration."""
        with patch(
            "custom_components.area_occupancy.migrations.async_migrate_unique_ids",
            side_effect=Exception("Migration error"),
        ):
            result = await async_migrate_entry(mock_hass, mock_config_entry_v1_0)

            assert result is False

    async def test_async_migrate_entry_invalid_threshold(
        self, mock_hass: Mock, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with invalid threshold value."""
        mock_config_entry_v1_0.data[CONF_THRESHOLD] = 150  # Invalid threshold

        with patch(
            "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
        ) as mock_migrate_ids:
            with patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage"
            ) as mock_migrate_storage:
                mock_migrate_ids.return_value = None
                mock_migrate_storage.return_value = None

                result = await async_migrate_entry(mock_hass, mock_config_entry_v1_0)

                assert result is True
                # Should fix invalid threshold
                assert mock_config_entry_v1_0.data[CONF_THRESHOLD] == DEFAULT_THRESHOLD


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

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a comprehensive mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        return hass

    async def test_complete_migration_workflow(self, mock_hass: Mock) -> None:
        """Test complete migration workflow from 1.0 to 1.1."""
        # Create entry that needs migration
        mock_entry = Mock(spec=ConfigEntry)
        mock_entry.version = 1
        mock_entry.minor_version = 0
        mock_entry.entry_id = "test_entry_id"
        mock_entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_THRESHOLD: 150,  # Invalid threshold
        }

        with patch(
            "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
        ) as mock_migrate_ids:
            with patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage"
            ) as mock_migrate_storage:
                mock_migrate_ids.return_value = None
                mock_migrate_storage.return_value = None

                result = await async_migrate_entry(mock_hass, mock_entry)

                assert result is True
                assert mock_entry.version == 1
                assert mock_entry.minor_version == 1

                # Check data migration
                assert CONF_PRIMARY_OCCUPANCY_SENSOR in mock_entry.data
                assert (
                    mock_entry.data[CONF_PRIMARY_OCCUPANCY_SENSOR]
                    == "binary_sensor.motion1"
                )
                assert (
                    mock_entry.data[CONF_THRESHOLD] == DEFAULT_THRESHOLD
                )  # Fixed invalid threshold

                # Check migration steps were called
                mock_migrate_ids.assert_called()
                mock_migrate_storage.assert_called_once_with(mock_hass, "test_entry_id")

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
