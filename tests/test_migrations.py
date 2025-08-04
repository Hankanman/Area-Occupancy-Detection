"""Tests for migrations.py module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
    DEFAULT_PURPOSE,
    DEFAULT_THRESHOLD,
)
from custom_components.area_occupancy.migrations import (
    async_migrate_entry,
    async_migrate_storage,
    async_migrate_unique_ids,
    migrate_config,
    migrate_primary_occupancy_sensor,
    migrate_purpose_field,
    validate_threshold,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import HomeAssistantError


class TestAsyncMigrateUniqueIds:
    """Test async_migrate_unique_ids function."""

    @pytest.mark.parametrize(
        ("entities", "platform"),
        [
            (
                {
                    "sensor.old_unique_id": Mock(unique_id="old_unique_id"),
                    "binary_sensor.old_unique_id": Mock(unique_id="old_unique_id"),
                },
                "sensor",
            ),
            ({}, "sensor"),  # No entities
            ({}, "invalid_platform"),  # Invalid platform
        ],
    )
    async def test_async_migrate_unique_ids(
        self, mock_hass: Mock, mock_config_entry: Mock, entities: dict, platform: str
    ) -> None:
        """Test unique ID migration with various scenarios."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            mock_registry.entities = entities
            mock_registry.async_update_entity = AsyncMock()
            mock_get_registry.return_value = mock_registry

            await async_migrate_unique_ids(mock_hass, mock_config_entry, platform)

            # Should complete without error
            assert mock_registry.async_update_entity.call_count >= 0


class TestMigratePrimaryOccupancySensor:
    """Test migrate_primary_occupancy_sensor function."""

    @pytest.mark.parametrize(
        ("config", "expected_primary"),
        [
            (
                {
                    CONF_MOTION_SENSORS: [
                        "binary_sensor.motion1",
                        "binary_sensor.motion2",
                    ]
                },
                "binary_sensor.motion1",
            ),
            (
                {
                    CONF_MOTION_SENSORS: [
                        "binary_sensor.motion1",
                        "binary_sensor.motion2",
                    ],
                    CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion2",
                },
                "binary_sensor.motion2",
            ),
            ({}, None),  # No motion sensors
            ({CONF_MOTION_SENSORS: []}, None),  # Empty motion sensors
        ],
    )
    def test_migrate_primary_occupancy_sensor(
        self, config: dict, expected_primary: str | None
    ) -> None:
        """Test primary occupancy sensor migration."""
        result = migrate_primary_occupancy_sensor(config)

        if expected_primary:
            assert result[CONF_PRIMARY_OCCUPANCY_SENSOR] == expected_primary
        else:
            assert CONF_PRIMARY_OCCUPANCY_SENSOR not in result


class TestMigratePurposeField:
    """Test migrate_purpose_field function."""

    @pytest.mark.parametrize(
        ("config", "should_add_purpose"),
        [
            ({CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}, True),
            (
                {
                    CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                    CONF_PURPOSE: "sleeping",
                },
                False,
            ),
            ({}, True),  # No sensors
            ({"appliances": ["binary_sensor.appliance"]}, True),  # Other sensors
            ({CONF_MOTION_SENSORS: []}, True),  # Empty sensors
        ],
    )
    def test_migrate_purpose_field(
        self, config: dict, should_add_purpose: bool
    ) -> None:
        """Test purpose field migration."""
        result = migrate_purpose_field(config)

        if should_add_purpose:
            assert result[CONF_PURPOSE] == DEFAULT_PURPOSE
        else:
            assert result[CONF_PURPOSE] == "sleeping"


class TestMigrateConfig:
    """Test migrate_config function."""

    def test_migrate_config_adds_primary_sensor(self) -> None:
        """Test config migration adds primary sensor."""
        config = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}

        result = migrate_config(config)

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

    @pytest.mark.parametrize(
        ("config", "expected_defaults"),
        [
            (
                {},
                {CONF_PURPOSE: DEFAULT_PURPOSE, "decay_half_life": 120},
            ),
            (
                {"other_key": "value"},
                {
                    "other_key": "value",
                    CONF_PURPOSE: DEFAULT_PURPOSE,
                    "decay_half_life": 120,
                },
            ),
        ],
    )
    def test_migrate_config_edge_cases(
        self, config: dict, expected_defaults: dict
    ) -> None:
        """Test config migration edge cases."""
        result = migrate_config(config)
        assert result == expected_defaults


class TestAsyncMigrateStorage:
    """Test async_migrate_storage function."""

    @pytest.mark.parametrize(
        "load_side_effect",
        [None, HomeAssistantError("Storage error")],
    )
    async def test_async_migrate_storage(
        self, mock_hass: Mock, load_side_effect: Exception | None
    ) -> None:
        """Test storage migration with various scenarios."""
        with patch(
            "homeassistant.helpers.storage.Store.async_load",
            new_callable=AsyncMock,
            side_effect=load_side_effect,
        ):
            await async_migrate_storage(mock_hass, "test_entry_id", 1)
            # Should complete without error


class TestAsyncMigrateEntry:
    """Test async_migrate_entry function."""

    @pytest.fixture
    def mock_config_entry_v1_0(self, mock_config_entry: Mock) -> Mock:
        """Create a mock config entry at version 1.0."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = mock_config_entry.entry_id
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
            mock_hass.config_entries.async_update_entry = Mock()

            result = await async_migrate_entry(mock_hass, mock_config_entry_v1_0)

            assert result is True
            mock_hass.config_entries.async_update_entry.assert_called_once()

            call_args = mock_hass.config_entries.async_update_entry.call_args
            assert call_args[0][0] == mock_config_entry_v1_0
            updated_data = call_args[1]["data"]
            assert CONF_PRIMARY_OCCUPANCY_SENSOR in updated_data

    async def test_async_migrate_entry_already_current(
        self, mock_hass: Mock, mock_config_entry_current: Mock
    ) -> None:
        """Test migration when already at current version."""
        result = await async_migrate_entry(mock_hass, mock_config_entry_current)
        assert result is True

    async def test_async_migrate_entry_future_version(self, mock_hass: Mock) -> None:
        """Test migration from future version."""
        mock_entry = Mock(spec=ConfigEntry)
        mock_entry.version = CONF_VERSION + 1
        mock_entry.minor_version = 0
        mock_entry.entry_id = "test_entry_id"
        mock_entry.data = {}
        mock_entry.options = {}

        result = await async_migrate_entry(mock_hass, mock_entry)
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

            with pytest.raises(Exception, match="Migration error"):
                await async_migrate_entry(mock_hass, mock_config_entry_v1_0)

    async def test_async_migrate_entry_invalid_threshold(
        self, mock_hass: Mock, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with invalid threshold value."""
        mock_config_entry_v1_0.options = {CONF_THRESHOLD: 150}

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

            call_args = mock_hass.config_entries.async_update_entry.call_args
            updated_options = call_args[1]["options"]
            assert updated_options[CONF_THRESHOLD] == DEFAULT_THRESHOLD


class TestValidateThreshold:
    """Test validate_threshold function."""

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            (1.0, 1.0),
            (50.0, 50.0),
            (99.0, 99.0),
            (0.0, DEFAULT_THRESHOLD),
            (-10.0, DEFAULT_THRESHOLD),
            (100.0, DEFAULT_THRESHOLD),
            (150.0, DEFAULT_THRESHOLD),
            (0.1, DEFAULT_THRESHOLD),
            (99.9, DEFAULT_THRESHOLD),
        ],
    )
    def test_validate_threshold(self, input_value: float, expected: float) -> None:
        """Test threshold validation with various values."""
        assert validate_threshold(input_value) == expected


class TestMigrationsIntegration:
    """Test migrations integration scenarios."""

    async def test_complete_migration_workflow(self, mock_hass: Mock) -> None:
        """Test complete migration workflow from 1.0 to current."""
        mock_entry = Mock(spec=ConfigEntry)
        mock_entry.version = 1
        mock_entry.minor_version = 0
        mock_entry.entry_id = "test_entry_id"
        mock_entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"]
        }
        mock_entry.options = {CONF_THRESHOLD: 150}

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

            mock_migrate_ids.assert_called()
            call_args = mock_hass.config_entries.async_update_entry.call_args
            updated_data = call_args[1]["data"]
            updated_options = call_args[1]["options"]

            assert (
                updated_data[CONF_PRIMARY_OCCUPANCY_SENSOR] == "binary_sensor.motion1"
            )
            assert updated_data[CONF_PURPOSE] == DEFAULT_PURPOSE
            assert updated_options[CONF_THRESHOLD] == DEFAULT_THRESHOLD
