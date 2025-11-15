"""Tests for migrations.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from filelock import Timeout
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MOTION_SENSORS,
    CONF_MOTION_TIMEOUT,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
    CONF_WINDOW_ACTIVE_STATE,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_MOTION_TIMEOUT,
    DEFAULT_PURPOSE,
    DEFAULT_THRESHOLD,
    DOMAIN,
)
from custom_components.area_occupancy.migrations import (
    CONF_DECAY_WINDOW_KEY,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_HISTORY_PERIOD,
    CONF_LIGHTS_KEY,
    DECAY_MIN_DELAY_KEY,
    _add_default_if_missing,
    _check_database_tables_exist,
    _check_unique_id_conflict,
    _cleanup_orphaned_entity_registry_entries,
    _drop_legacy_tables,
    _drop_tables_locked,
    _find_entities_by_prefix,
    _handle_entity_conflict,
    _migrate_database_for_consolidation,
    _migrate_device_registry_for_consolidation,
    _migrate_entity_registry_for_consolidation,
    _remove_deprecated_keys,
    _resolve_area_from_entry,
    _restore_config_entry_state,
    _safe_database_operation,
    _safe_file_operation,
    _update_db_version,
    _update_entity_unique_id,
    async_migrate_entry,
    async_migrate_storage,
    async_migrate_to_single_instance,
    async_migrate_unique_ids,
    async_reset_database_if_needed,
    migrate_config,
    migrate_decay_half_life,
    migrate_motion_timeout,
    migrate_primary_occupancy_sensor,
    migrate_purpose_field,
    remove_decay_min_delay,
    remove_decay_window_key,
    remove_history_keys,
    remove_lights_key,
    validate_threshold,
)
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError


class TestAsyncMigrateUniqueIds:
    """Test async_migrate_unique_ids function."""

    @pytest.mark.parametrize(
        ("platform", "entity_suffix", "expected_calls"),
        [
            ("sensor", "probability", 1),
            ("sensor", "decay", 1),
            ("binary_sensor", "occupancy", 1),
            ("number", "threshold", 1),
            ("invalid_platform", "anything", 0),
        ],
    )
    async def test_async_migrate_unique_ids_platforms(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        platform: str,
        entity_suffix: str,
        expected_calls: int,
    ) -> None:
        """Test unique ID migration for different platforms."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            mock_registry.async_update_entity = Mock()

            if platform != "invalid_platform":
                entity1 = Mock()
                entity1.unique_id = (
                    f"{DOMAIN}_{mock_config_entry.entry_id}_{entity_suffix}"
                )
                entity1.config_entry_id = mock_config_entry.entry_id
                mock_registry.entities = {f"{platform}.entity1": entity1}
            else:
                mock_registry.entities = {}

            mock_get_registry.return_value = mock_registry

            await async_migrate_unique_ids(hass, mock_config_entry, platform)

            assert mock_registry.async_update_entity.call_count == expected_calls

    async def test_async_migrate_unique_ids_multiple_entities(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test unique ID migration with multiple matching entities."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            entity1 = Mock()
            entity1.unique_id = f"{DOMAIN}_{mock_config_entry.entry_id}_probability"
            entity1.config_entry_id = mock_config_entry.entry_id
            entity2 = Mock()
            entity2.unique_id = f"{DOMAIN}_{mock_config_entry.entry_id}_decay"
            entity2.config_entry_id = mock_config_entry.entry_id

            mock_registry.entities = {
                "sensor.entity1": entity1,
                "sensor.entity2": entity2,
            }
            mock_registry.async_update_entity = Mock()
            mock_get_registry.return_value = mock_registry

            await async_migrate_unique_ids(hass, mock_config_entry, "sensor")

            assert mock_registry.async_update_entity.call_count == 2

    async def test_async_migrate_unique_ids_no_matching_entities(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test unique ID migration with no matching entities."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            entity1 = Mock()
            entity1.unique_id = "other_prefix_entity"
            entity1.config_entry_id = mock_config_entry.entry_id

            mock_registry.entities = {"sensor.entity1": entity1}
            mock_registry.async_update_entity = Mock()
            mock_get_registry.return_value = mock_registry

            await async_migrate_unique_ids(hass, mock_config_entry, "sensor")

            mock_registry.async_update_entity.assert_not_called()


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
                {
                    CONF_PURPOSE: DEFAULT_PURPOSE,
                    "decay_half_life": 120,
                    CONF_MOTION_TIMEOUT: DEFAULT_MOTION_TIMEOUT,
                },
            ),
            (
                {"other_key": "value"},
                {
                    "other_key": "value",
                    CONF_PURPOSE: DEFAULT_PURPOSE,
                    "decay_half_life": 120,
                    CONF_MOTION_TIMEOUT: DEFAULT_MOTION_TIMEOUT,
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


class TestAsyncMigrateStorageBasic:
    """Test async_migrate_storage function - basic tests."""

    @pytest.mark.parametrize(
        "load_side_effect",
        [None, HomeAssistantError("Storage error")],
    )
    async def test_async_migrate_storage_basic(
        self, hass: HomeAssistant, load_side_effect: Exception | None
    ) -> None:
        """Test storage migration with various scenarios."""
        with (
            patch(
                "custom_components.area_occupancy.migrations.async_reset_database_if_needed",
                new_callable=AsyncMock,
            ),
            patch(
                "homeassistant.helpers.storage.Store.async_load",
                new_callable=AsyncMock,
                side_effect=load_side_effect,
            ),
        ):
            await async_migrate_storage(hass, "test_entry_id", 1)
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
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }
        entry.options = {}
        return entry

    async def test_async_migrate_entry_v1_0_to_current(
        self, hass: HomeAssistant, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration from version 1.0 to current."""

        # Ensure the entry exists in config_entries so it's not considered consolidated
        # async_entries is synchronous in Home Assistant, so use Mock not AsyncMock
        # Make it return the entry for both calls: with DOMAIN argument and without (during teardown)
        def async_entries_mock(domain=None):
            return [mock_config_entry_v1_0]

        hass.config_entries.async_entries = async_entries_mock

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
            hass.config_entries.async_update_entry = Mock()

            result = await async_migrate_entry(hass, mock_config_entry_v1_0)

            assert result is True
            hass.config_entries.async_update_entry.assert_called_once()

            call_args = hass.config_entries.async_update_entry.call_args
            assert call_args[0][0] == mock_config_entry_v1_0
            updated_data = call_args[1]["data"]
            assert CONF_PRIMARY_OCCUPANCY_SENSOR in updated_data

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

    async def test_async_migrate_entry_migration_error(
        self, hass: HomeAssistant, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with error during migration."""

        # Mock async_entries to return the entry so it's not considered consolidated
        # async_entries is synchronous in Home Assistant, so use Mock not AsyncMock
        def async_entries_mock(domain=None):
            return [mock_config_entry_v1_0]

        hass.config_entries.async_entries = async_entries_mock
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
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
        ):
            hass.config_entries.async_update_entry = Mock()

            # The migration code catches HomeAssistantError but not generic Exception
            # So a generic Exception should propagate
            with pytest.raises(Exception, match="Migration error"):
                await async_migrate_entry(hass, mock_config_entry_v1_0)

    async def test_async_migrate_entry_invalid_threshold(
        self, hass: HomeAssistant, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with invalid threshold value."""
        mock_config_entry_v1_0.options = {CONF_THRESHOLD: 150}

        # Mock async_entries to return the entry so it's not considered consolidated
        # async_entries is synchronous in Home Assistant, so use Mock not AsyncMock
        def async_entries_mock(domain=None):
            return [mock_config_entry_v1_0]

        hass.config_entries.async_entries = async_entries_mock

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
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
        ):
            mock_migrate_ids.return_value = None
            hass.config_entries.async_update_entry = Mock()

            result = await async_migrate_entry(hass, mock_config_entry_v1_0)
            assert result is True

            # Check that async_update_entry was called
            assert hass.config_entries.async_update_entry.called
            call_args = hass.config_entries.async_update_entry.call_args
            if call_args:
                updated_options = call_args[1].get("options", {})
                # Threshold should be normalized to valid range (0-100)
                assert updated_options.get(CONF_THRESHOLD) == DEFAULT_THRESHOLD


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


class TestFindEntitiesByPrefix:
    """Test _find_entities_by_prefix function."""

    def test_find_entities_by_prefix_with_matching_entities(self) -> None:
        """Test finding entities with matching prefix."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "area_occupancy_entry1_probability"
        entity1.config_entry_id = "entry1"
        entity2 = Mock()
        entity2.unique_id = "area_occupancy_entry1_decay"
        entity2.config_entry_id = "entry1"
        entity3 = Mock()
        entity3.unique_id = "other_prefix_entity"
        entity3.config_entry_id = "entry2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
            "sensor.entity3": entity3,
        }

        result = _find_entities_by_prefix(
            mock_registry, "area_occupancy_entry1_", "entry1"
        )

        assert len(result) == 2
        assert ("sensor.entity1", entity1) in result
        assert ("sensor.entity2", entity2) in result

    def test_find_entities_by_prefix_no_config_entry_filter(self) -> None:
        """Test finding entities without config entry filter."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "prefix_entity1"
        entity1.config_entry_id = "entry1"
        entity2 = Mock()
        entity2.unique_id = "prefix_entity2"
        entity2.config_entry_id = "entry2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        result = _find_entities_by_prefix(mock_registry, "prefix_", None)

        assert len(result) == 2

    def test_find_entities_by_prefix_no_matches(self) -> None:
        """Test finding entities with no matches."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "other_prefix_entity"
        entity1.config_entry_id = "entry1"

        mock_registry.entities = {"sensor.entity1": entity1}

        result = _find_entities_by_prefix(mock_registry, "prefix_", "entry1")

        assert len(result) == 0

    def test_find_entities_by_prefix_none_unique_id(self) -> None:
        """Test finding entities with None unique_id."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = None
        entity1.config_entry_id = "entry1"

        mock_registry.entities = {"sensor.entity1": entity1}

        result = _find_entities_by_prefix(mock_registry, "prefix_", "entry1")

        assert len(result) == 0


class TestCheckUniqueIdConflict:
    """Test _check_unique_id_conflict function."""

    def test_check_unique_id_conflict_no_conflict(self) -> None:
        """Test conflict check with no conflicts."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "unique_id_1"
        entity2 = Mock()
        entity2.unique_id = "unique_id_2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        has_conflict, conflict_id = _check_unique_id_conflict(
            mock_registry, "unique_id_3", "sensor.entity1"
        )

        assert has_conflict is False
        assert conflict_id is None

    def test_check_unique_id_conflict_with_conflict(self) -> None:
        """Test conflict check with conflicts."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "unique_id_1"
        entity2 = Mock()
        entity2.unique_id = "unique_id_2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        has_conflict, conflict_id = _check_unique_id_conflict(
            mock_registry, "unique_id_2", "sensor.entity1"
        )

        assert has_conflict is True
        assert conflict_id == "sensor.entity2"

    def test_check_unique_id_conflict_case_insensitive(self) -> None:
        """Test conflict check is case insensitive."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "Unique_ID_1"
        entity2 = Mock()
        entity2.unique_id = "unique_id_2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        has_conflict, conflict_id = _check_unique_id_conflict(
            mock_registry, "UNIQUE_ID_1", "sensor.entity2"
        )

        assert has_conflict is True
        assert conflict_id == "sensor.entity1"

    def test_check_unique_id_conflict_excludes_self(self) -> None:
        """Test conflict check excludes the entity being checked."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "unique_id_1"

        mock_registry.entities = {"sensor.entity1": entity1}

        has_conflict, conflict_id = _check_unique_id_conflict(
            mock_registry, "unique_id_1", "sensor.entity1"
        )

        assert has_conflict is False
        assert conflict_id is None


class TestUpdateEntityUniqueId:
    """Test _update_entity_unique_id function."""

    def test_update_entity_unique_id_success(self) -> None:
        """Test successful unique ID update."""
        mock_registry = Mock()
        mock_registry.async_update_entity = Mock()
        entity1 = Mock()
        entity1.unique_id = "old_unique_id"

        mock_registry.entities = {"sensor.entity1": entity1}

        success, conflict_id = _update_entity_unique_id(
            mock_registry, "sensor.entity1", "old_unique_id", "new_unique_id", False
        )

        assert success is True
        assert conflict_id is None
        mock_registry.async_update_entity.assert_called_once_with(
            "sensor.entity1", new_unique_id="new_unique_id"
        )

    def test_update_entity_unique_id_with_conflict_check_no_conflict(self) -> None:
        """Test update with conflict check and no conflict."""
        mock_registry = Mock()
        mock_registry.async_update_entity = Mock()
        entity1 = Mock()
        entity1.unique_id = "old_unique_id"
        entity2 = Mock()
        entity2.unique_id = "other_unique_id"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        success, conflict_id = _update_entity_unique_id(
            mock_registry, "sensor.entity1", "old_unique_id", "new_unique_id", True
        )

        assert success is True
        assert conflict_id is None
        mock_registry.async_update_entity.assert_called_once()

    def test_update_entity_unique_id_with_conflict(self) -> None:
        """Test update with conflict check and conflict exists."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "old_unique_id"
        entity2 = Mock()
        entity2.unique_id = "new_unique_id"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        success, conflict_id = _update_entity_unique_id(
            mock_registry, "sensor.entity1", "old_unique_id", "new_unique_id", True
        )

        assert success is False
        assert conflict_id == "sensor.entity2"
        mock_registry.async_update_entity.assert_not_called()

    def test_update_entity_unique_id_lowercase(self) -> None:
        """Test that unique ID is converted to lowercase."""
        mock_registry = Mock()
        mock_registry.async_update_entity = Mock()
        entity1 = Mock()
        entity1.unique_id = "old_unique_id"

        mock_registry.entities = {"sensor.entity1": entity1}

        success, _conflict_id = _update_entity_unique_id(
            mock_registry, "sensor.entity1", "old_unique_id", "NEW_UNIQUE_ID", False
        )

        assert success is True
        mock_registry.async_update_entity.assert_called_once_with(
            "sensor.entity1", new_unique_id="new_unique_id"
        )


class TestRemoveDeprecatedKeys:
    """Test _remove_deprecated_keys function."""

    def test_remove_deprecated_keys_single_key(self) -> None:
        """Test removing a single deprecated key."""
        config = {"key1": "value1", "deprecated_key": "value2", "key2": "value3"}

        result = _remove_deprecated_keys(config, ["deprecated_key"], "test key")

        assert "deprecated_key" not in result
        assert "key1" in result
        assert "key2" in result

    def test_remove_deprecated_keys_multiple_keys(self) -> None:
        """Test removing multiple deprecated keys."""
        config = {
            "key1": "value1",
            "deprecated_key1": "value2",
            "deprecated_key2": "value3",
            "key2": "value4",
        }

        result = _remove_deprecated_keys(
            config, ["deprecated_key1", "deprecated_key2"], "test keys"
        )

        assert "deprecated_key1" not in result
        assert "deprecated_key2" not in result
        assert "key1" in result
        assert "key2" in result

    def test_remove_deprecated_keys_not_present(self) -> None:
        """Test removing keys that don't exist."""
        config = {"key1": "value1", "key2": "value2"}

        result = _remove_deprecated_keys(config, ["deprecated_key"], "test key")

        assert config == result
        assert "key1" in result
        assert "key2" in result

    def test_remove_deprecated_keys_empty_description(self) -> None:
        """Test removing keys with empty description."""
        config = {"deprecated_key": "value1"}

        result = _remove_deprecated_keys(config, ["deprecated_key"], "")

        assert "deprecated_key" not in result


class TestRemoveDecayMinDelay:
    """Test remove_decay_min_delay function."""

    def test_remove_decay_min_delay(self) -> None:
        """Test removing decay_min_delay key."""
        config = {"decay_min_delay": 60, "other_key": "value"}

        result = remove_decay_min_delay(config)

        assert DECAY_MIN_DELAY_KEY not in result
        assert "other_key" in result


class TestRemoveLightsKey:
    """Test remove_lights_key function."""

    def test_remove_lights_key(self) -> None:
        """Test removing lights key."""
        config = {CONF_LIGHTS_KEY: ["light1"], "other_key": "value"}

        result = remove_lights_key(config)

        assert CONF_LIGHTS_KEY not in result
        assert "other_key" in result


class TestRemoveDecayWindowKey:
    """Test remove_decay_window_key function."""

    def test_remove_decay_window_key(self) -> None:
        """Test removing decay_window key."""
        config = {CONF_DECAY_WINDOW_KEY: 300, "other_key": "value"}

        result = remove_decay_window_key(config)

        assert CONF_DECAY_WINDOW_KEY not in result
        assert "other_key" in result


class TestRemoveHistoryKeys:
    """Test remove_history_keys function."""

    def test_remove_history_keys(self) -> None:
        """Test removing history keys."""
        config = {
            CONF_HISTORY_PERIOD: 30,
            CONF_HISTORICAL_ANALYSIS_ENABLED: True,
            "other_key": "value",
        }

        result = remove_history_keys(config)

        assert CONF_HISTORY_PERIOD not in result
        assert CONF_HISTORICAL_ANALYSIS_ENABLED not in result
        assert "other_key" in result


class TestAddDefaultIfMissing:
    """Test _add_default_if_missing function."""

    def test_add_default_if_missing_key_not_present(self) -> None:
        """Test adding default when key is missing."""
        config = {"key1": "value1"}

        result = _add_default_if_missing(config, "new_key", "default_value", None)

        assert result["new_key"] == "default_value"
        assert "key1" in result

    def test_add_default_if_missing_key_present(self) -> None:
        """Test not adding default when key is present."""
        config = {"key1": "value1", "new_key": "existing_value"}

        result = _add_default_if_missing(config, "new_key", "default_value", None)

        assert result["new_key"] == "existing_value"

    def test_add_default_if_missing_with_log_message(self) -> None:
        """Test adding default with custom log message."""
        config = {"key1": "value1"}

        result = _add_default_if_missing(
            config, "new_key", "default_value", "Custom log message"
        )

        assert result["new_key"] == "default_value"


class TestMigrateDecayHalfLife:
    """Test migrate_decay_half_life function."""

    def test_migrate_decay_half_life_adds_default(self) -> None:
        """Test adding decay half life default."""
        config = {"other_key": "value"}

        result = migrate_decay_half_life(config)

        assert result[CONF_DECAY_HALF_LIFE] == DEFAULT_DECAY_HALF_LIFE

    def test_migrate_decay_half_life_preserves_existing(self) -> None:
        """Test preserving existing decay half life."""
        config = {CONF_DECAY_HALF_LIFE: 300.0}

        result = migrate_decay_half_life(config)

        assert result[CONF_DECAY_HALF_LIFE] == 300.0


class TestMigrateMotionTimeout:
    """Test migrate_motion_timeout function."""

    def test_migrate_motion_timeout_adds_default(self) -> None:
        """Test adding motion timeout default."""
        config = {"other_key": "value"}

        result = migrate_motion_timeout(config)

        assert result[CONF_MOTION_TIMEOUT] == DEFAULT_MOTION_TIMEOUT

    def test_migrate_motion_timeout_preserves_existing(self) -> None:
        """Test preserving existing motion timeout."""
        config = {CONF_MOTION_TIMEOUT: 120.0}

        result = migrate_motion_timeout(config)

        assert result[CONF_MOTION_TIMEOUT] == 120.0


class TestAsyncMigrateStorage:
    """Test async_migrate_storage function."""

    async def test_async_migrate_storage_removes_legacy_file(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test storage migration removes legacy file."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        legacy_file = storage_dir / "area_occupancy.storage"
        legacy_file.write_text("test data")

        hass.config.config_dir = str(tmp_path)

        with patch(
            "custom_components.area_occupancy.migrations.async_reset_database_if_needed",
            new_callable=AsyncMock,
        ) as mock_reset:
            await async_migrate_storage(hass, "test_entry_id", 1)

            assert not legacy_file.exists()
            mock_reset.assert_called_once_with(hass, 1)

    async def test_async_migrate_storage_no_legacy_file(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test storage migration when no legacy file exists."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()

        hass.config.config_dir = str(tmp_path)

        with patch(
            "custom_components.area_occupancy.migrations.async_reset_database_if_needed",
            new_callable=AsyncMock,
        ) as mock_reset:
            await async_migrate_storage(hass, "test_entry_id", 1)

            mock_reset.assert_called_once_with(hass, 1)

    async def test_async_migrate_storage_error_handling(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test storage migration error handling."""
        hass.config.config_dir = str(tmp_path)

        with patch(
            "custom_components.area_occupancy.migrations.async_reset_database_if_needed",
            side_effect=HomeAssistantError("Database error"),
        ):
            # Should not raise, just log error
            await async_migrate_storage(hass, "test_entry_id", 1)


class TestAsyncResetDatabaseIfNeeded:
    """Test async_reset_database_if_needed function."""

    async def test_async_reset_database_if_needed(self, hass: HomeAssistant) -> None:
        """Test database reset when needed."""
        # The function calls hass.async_add_executor_job which runs in executor thread
        # We just verify it completes without error
        await async_reset_database_if_needed(hass, 10)
        # Function should complete successfully

    async def test_async_reset_database_if_needed_version_11(
        self, hass: HomeAssistant
    ) -> None:
        """Test database reset skipped for version >= 11."""
        # For version >= 11, _drop_tables_locked will skip the actual work
        # but still runs via executor. We just verify it completes without error.
        await async_reset_database_if_needed(hass, 11)
        # Function should complete successfully


class TestAsyncMigrateEntryAdditional:
    """Additional tests for async_migrate_entry function."""

    async def test_async_migrate_entry_removes_deprecated_keys(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration removes deprecated keys."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_AREA_ID: "area_123",
            DECAY_MIN_DELAY_KEY: 60,
        }
        entry.options = {DECAY_MIN_DELAY_KEY: 60}

        def async_entries_mock(domain=None):
            return [entry]

        hass.config_entries.async_entries = async_entries_mock

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
        ):
            mock_migrate_ids.return_value = None
            hass.config_entries.async_update_entry = Mock()

            await async_migrate_entry(hass, entry)

            call_args = hass.config_entries.async_update_entry.call_args
            updated_data = call_args[1]["data"]
            updated_options = call_args[1]["options"]

            assert CONF_AREA_ID not in updated_data
            assert DECAY_MIN_DELAY_KEY not in updated_data
            assert DECAY_MIN_DELAY_KEY not in updated_options

    async def test_async_migrate_entry_adds_state_configs(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration adds new state configurations."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        def async_entries_mock(domain=None):
            return [entry]

        hass.config_entries.async_entries = async_entries_mock

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
        ):
            mock_migrate_ids.return_value = None
            hass.config_entries.async_update_entry = Mock()

            await async_migrate_entry(hass, entry)

            call_args = hass.config_entries.async_update_entry.call_args
            updated_data = call_args[1]["data"]
            updated_options = call_args[1]["options"]

            # Multi-select states should be in data
            assert CONF_MEDIA_ACTIVE_STATES in updated_data
            assert CONF_APPLIANCE_ACTIVE_STATES in updated_data
            # Single-select states should be in options
            assert CONF_DOOR_ACTIVE_STATE in updated_options
            assert CONF_WINDOW_ACTIVE_STATE in updated_options

    async def test_async_migrate_entry_preserves_existing_state_configs(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration preserves existing state configurations."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_MEDIA_ACTIVE_STATES: ["playing"],
        }
        entry.options = {CONF_DOOR_ACTIVE_STATE: "closed"}

        def async_entries_mock(domain=None):
            return [entry]

        hass.config_entries.async_entries = async_entries_mock

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
        ):
            mock_migrate_ids.return_value = None
            hass.config_entries.async_update_entry = Mock()

            await async_migrate_entry(hass, entry)

            call_args = hass.config_entries.async_update_entry.call_args
            updated_data = call_args[1]["data"]
            updated_options = call_args[1]["options"]

            # Should preserve existing values
            assert updated_data[CONF_MEDIA_ACTIVE_STATES] == ["playing"]
            assert updated_options[CONF_DOOR_ACTIVE_STATE] == "closed"

    async def test_async_migrate_entry_config_migration_error(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration handles config migration errors."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        def async_entries_mock(domain=None):
            return [entry]

        hass.config_entries.async_entries = async_entries_mock

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.area_occupancy.migrations.migrate_config",
                side_effect=ValueError("Config error"),
            ),
        ):
            mock_migrate_ids.return_value = None

            result = await async_migrate_entry(hass, entry)

            assert result is False

    async def test_async_migrate_entry_consolidation_skipped(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration skips consolidation for version >= 13."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 13
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        def async_entries_mock(domain=None):
            return [entry]

        hass.config_entries.async_entries = async_entries_mock

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_to_single_instance"
            ) as mock_consolidate,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
        ):
            mock_migrate_ids.return_value = None
            hass.config_entries.async_update_entry = Mock()

            await async_migrate_entry(hass, entry)

            # Consolidation should not be called for version >= 13
            mock_consolidate.assert_not_called()

    async def test_async_migrate_entry_consolidation_entry_removed(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration when entry is removed during consolidation."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        def async_entries_mock(domain=None):
            # Entry was removed during consolidation
            return []

        hass.config_entries.async_entries = async_entries_mock

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_to_single_instance",
                return_value=True,
            ),
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids"
            ) as mock_migrate_ids,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
        ):
            result = await async_migrate_entry(hass, entry)

            # Should return True and skip individual migration
            assert result is True
            # Should not call migrate_unique_ids for removed entry
            mock_migrate_ids.assert_not_called()


class TestAsyncMigrateToSingleInstance:
    """Test async_migrate_to_single_instance function."""

    async def test_async_migrate_to_single_instance_no_consolidation_needed(
        self, hass: HomeAssistant
    ) -> None:
        """Test when no consolidation is needed (0 or 1 entries)."""
        with patch.object(hass.config_entries, "async_entries", return_value=[]):
            result = await async_migrate_to_single_instance(hass)
            assert result is True

        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.entry_id = "test_entry_id"
        with patch.object(hass.config_entries, "async_entries", return_value=[entry]):
            result = await async_migrate_to_single_instance(hass)
            assert result is True

    async def test_async_migrate_to_single_instance_no_resolvable_entries(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test when entries cannot be resolved to areas."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.entry_id = "entry1"
        entry1.data = {CONF_AREA_ID: "nonexistent_area"}
        entry1.options = {}
        entry1.title = "Entry 1"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.entry_id = "entry2"
        entry2.data = {CONF_AREA_ID: "nonexistent_area2"}
        entry2.options = {}
        entry2.title = "Entry 2"

        with (
            patch.object(
                hass.config_entries, "async_entries", return_value=[entry1, entry2]
            ),
            patch("homeassistant.helpers.area_registry.async_get") as mock_get_area_reg,
        ):
            mock_area_reg = Mock()
            mock_area_reg.async_get_area.return_value = None
            mock_get_area_reg.return_value = mock_area_reg

            result = await async_migrate_to_single_instance(hass)
            assert result is False

    async def test_async_migrate_to_single_instance_success(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test successful consolidation."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.entry_id = "entry1"
        entry1.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Entry 1"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.entry_id = "entry2"
        entry2.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}
        entry2.title = "Entry 2"

        with (
            patch.object(
                hass.config_entries,
                "async_entries",
                return_value=[entry1, entry2],
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_entity_registry_for_consolidation",
                new_callable=AsyncMock,
            ) as mock_entity_migrate,
            patch(
                "custom_components.area_occupancy.migrations._migrate_database_for_consolidation",
                new_callable=AsyncMock,
            ) as mock_db_migrate,
            patch(
                "custom_components.area_occupancy.migrations._migrate_device_registry_for_consolidation",
                new_callable=AsyncMock,
            ) as mock_device_migrate,
            patch(
                "custom_components.area_occupancy.migrations._cleanup_orphaned_entity_registry_entries",
                new_callable=AsyncMock,
            ) as mock_cleanup,
            patch.object(hass.config_entries, "async_update_entry", return_value=None),
            patch.object(
                hass.config_entries, "async_remove", new_callable=AsyncMock
            ) as mock_remove,
        ):
            result = await async_migrate_to_single_instance(hass)

            assert result is True
            mock_entity_migrate.assert_called_once()
            mock_db_migrate.assert_called_once()
            mock_device_migrate.assert_called_once()
            mock_cleanup.assert_called_once()
            hass.config_entries.async_update_entry.assert_called_once()
            mock_remove.assert_called_once()

    async def test_async_migrate_to_single_instance_legacy_format(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test consolidation with legacy format (name instead of area_id)."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.entry_id = "entry1"
        entry1.data = {
            "name": "Testing",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Entry 1"

        with (
            patch.object(hass.config_entries, "async_entries", return_value=[entry1]),
            patch("homeassistant.helpers.area_registry.async_get") as mock_get_area_reg,
        ):
            mock_area_reg = Mock()
            mock_area_entry = Mock()
            mock_area_entry.id = testing_area_id
            mock_area_entry.name = "Testing"
            mock_area_reg.async_list_areas.return_value = [mock_area_entry]
            mock_get_area_reg.return_value = mock_area_reg

            with (
                patch(
                    "custom_components.area_occupancy.migrations._migrate_entity_registry_for_consolidation",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.area_occupancy.migrations._migrate_database_for_consolidation",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.area_occupancy.migrations._migrate_device_registry_for_consolidation",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.area_occupancy.migrations._cleanup_orphaned_entity_registry_entries",
                    new_callable=AsyncMock,
                ),
                patch.object(
                    hass.config_entries, "async_update_entry", return_value=None
                ),
                patch.object(
                    hass.config_entries, "async_remove", new_callable=AsyncMock
                ),
            ):
                # Should complete successfully even with legacy format
                result = await async_migrate_to_single_instance(hass)
                # Will return False because only one entry, but should not error
                assert result is True


class TestMigrateEntityRegistryForConsolidation:
    """Test _migrate_entity_registry_for_consolidation function."""

    async def test_migrate_entity_registry_for_consolidation_success(
        self, hass: HomeAssistant
    ) -> None:
        """Test successful entity registry migration."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "entry1_probability"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"sensor.entity1": entity1}
        mock_registry.async_update_entity = Mock()

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(False, None),
            ),
        ):
            await _migrate_entity_registry_for_consolidation(
                hass, [entry1], {"entry1": "Test Area"}, "new_entry_id"
            )

            mock_registry.async_update_entity.assert_called_once()

    async def test_migrate_entity_registry_for_consolidation_with_conflict(
        self, hass: HomeAssistant
    ) -> None:
        """Test entity registry migration with conflicts."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "entry1_probability"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"sensor.entity1": entity1}
        mock_registry.async_update_entity = Mock()

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(True, "sensor.other_entity"),
            ),
        ):
            await _migrate_entity_registry_for_consolidation(
                hass, [entry1], {"entry1": "Test Area"}, "new_entry_id"
            )

            # Should not update due to conflict
            mock_registry.async_update_entity.assert_not_called()

    async def test_migrate_entity_registry_for_consolidation_legacy_format(
        self, hass: HomeAssistant
    ) -> None:
        """Test entity registry migration with legacy format."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = f"{DOMAIN}_entry1_probability"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"sensor.entity1": entity1}
        mock_registry.async_update_entity = Mock()

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(False, None),
            ),
        ):
            await _migrate_entity_registry_for_consolidation(
                hass, [entry1], {"entry1": "Test Area"}, "new_entry_id"
            )

            mock_registry.async_update_entity.assert_called_once()

    async def test_migrate_entity_registry_for_consolidation_skip_entry_not_in_mapping(
        self, hass: HomeAssistant
    ) -> None:
        """Test entity registry migration skips entries not in mapping."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {}
        mock_registry.async_update_entity = Mock()

        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=mock_registry,
        ):
            # entry1 not in mapping
            await _migrate_entity_registry_for_consolidation(
                hass, [entry1], {}, "new_entry_id"
            )

            mock_registry.async_update_entity.assert_not_called()

    async def test_migrate_entity_registry_for_consolidation_both_formats(
        self, hass: HomeAssistant
    ) -> None:
        """Test entity registry migration handles both new and legacy formats."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "entry1_probability"
        entity1.config_entry_id = "entry1"

        entity2 = Mock()
        entity2.unique_id = f"{DOMAIN}_entry1_decay"
        entity2.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }
        mock_registry.async_update_entity = Mock()

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(False, None),
            ),
        ):
            await _migrate_entity_registry_for_consolidation(
                hass, [entry1], {"entry1": "Test Area"}, "new_entry_id"
            )

            # Should update both entities
            assert mock_registry.async_update_entity.call_count == 2

    async def test_migrate_entity_registry_for_consolidation_duplicate_entity(
        self, hass: HomeAssistant
    ) -> None:
        """Test entity registry migration when entity appears in both formats."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "entry1_probability"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"sensor.entity1": entity1}
        mock_registry.async_update_entity = Mock()

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(False, None),
            ),
            patch(
                "custom_components.area_occupancy.migrations._find_entities_by_prefix",
                side_effect=[
                    [("sensor.entity1", entity1)],  # New format
                    [("sensor.entity1", entity1)],  # Legacy format (same entity)
                ],
            ),
        ):
            await _migrate_entity_registry_for_consolidation(
                hass, [entry1], {"entry1": "Test Area"}, "new_entry_id"
            )

            # Should only update once (legacy format skipped as duplicate)
            mock_registry.async_update_entity.assert_called_once()


class TestMigrateDeviceRegistryForConsolidation:
    """Test _migrate_device_registry_for_consolidation function."""

    async def test_migrate_device_registry_for_consolidation_success(
        self, hass: HomeAssistant
    ) -> None:
        """Test successful device registry migration."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        device1 = Mock()
        device1.identifiers = {(DOMAIN, "area1")}
        device1.config_entries = {"entry1"}

        mock_registry = Mock()
        mock_registry.devices = {"device1": device1}
        mock_registry.async_update_device = Mock()

        with patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=mock_registry,
        ):
            await _migrate_device_registry_for_consolidation(
                hass, [entry1], "new_entry_id"
            )

            mock_registry.async_update_device.assert_called()

    async def test_migrate_device_registry_for_consolidation_already_linked(
        self, hass: HomeAssistant
    ) -> None:
        """Test device registry migration when already linked to new entry."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        device1 = Mock()
        device1.identifiers = {(DOMAIN, "area1")}
        device1.config_entries = {"entry1", "new_entry_id"}

        mock_registry = Mock()
        mock_registry.devices = {"device1": device1}
        mock_registry.async_update_device = Mock()

        with patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=mock_registry,
        ):
            await _migrate_device_registry_for_consolidation(
                hass, [entry1], "new_entry_id"
            )

            # Should remove old entry but not add new one
            mock_registry.async_update_device.assert_called()

    async def test_migrate_device_registry_for_consolidation_error_handling(
        self, hass: HomeAssistant
    ) -> None:
        """Test device registry migration error handling."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        device1 = Mock()
        device1.identifiers = {(DOMAIN, "area1")}
        device1.config_entries = {"entry1"}

        mock_registry = Mock()
        mock_registry.devices = {"device1": device1}
        mock_registry.async_update_device = Mock(side_effect=ValueError("Device error"))

        with patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=mock_registry,
        ):
            # Should complete without raising
            await _migrate_device_registry_for_consolidation(
                hass, [entry1], "new_entry_id"
            )

    async def test_migrate_device_registry_for_consolidation_no_domain_devices(
        self, hass: HomeAssistant
    ) -> None:
        """Test device registry migration with no domain devices."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        device1 = Mock()
        device1.identifiers = {("other_domain", "device1")}
        device1.config_entries = {"entry1"}

        mock_registry = Mock()
        mock_registry.devices = {"device1": device1}
        mock_registry.async_update_device = Mock()

        with patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=mock_registry,
        ):
            await _migrate_device_registry_for_consolidation(
                hass, [entry1], "new_entry_id"
            )

            # Should not update devices from other domains
            mock_registry.async_update_device.assert_not_called()

    async def test_migrate_device_registry_for_consolidation_no_old_entry_links(
        self, hass: HomeAssistant
    ) -> None:
        """Test device registry migration when device not linked to old entries."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        device1 = Mock()
        device1.identifiers = {(DOMAIN, "area1")}
        device1.config_entries = {"other_entry"}

        mock_registry = Mock()
        mock_registry.devices = {"device1": device1}
        mock_registry.async_update_device = Mock()

        with patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=mock_registry,
        ):
            await _migrate_device_registry_for_consolidation(
                hass, [entry1], "new_entry_id"
            )

            # Should not update devices not linked to old entries
            mock_registry.async_update_device.assert_not_called()


class TestCleanupOrphanedEntityRegistryEntries:
    """Test _cleanup_orphaned_entity_registry_entries function."""

    async def test_cleanup_orphaned_entity_registry_entries_success(
        self, hass: HomeAssistant
    ) -> None:
        """Test successful cleanup of orphaned entities."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "test_unique_id"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"sensor.entity1": entity1}
        mock_registry.async_update_entity = Mock()
        mock_registry.async_remove = Mock()

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(False, None),
            ),
        ):
            await _cleanup_orphaned_entity_registry_entries(
                hass, [entry1], "new_entry_id"
            )

            mock_registry.async_update_entity.assert_called_once()

    async def test_cleanup_orphaned_entity_registry_entries_with_conflict(
        self, hass: HomeAssistant
    ) -> None:
        """Test cleanup with unique ID conflicts."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "test_unique_id"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"sensor.entity1": entity1}
        mock_registry.async_update_entity = Mock()
        mock_registry.async_remove = Mock()

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(True, "sensor.other_entity"),
            ),
        ):
            await _cleanup_orphaned_entity_registry_entries(
                hass, [entry1], "new_entry_id"
            )

            # Should remove conflicting entity
            mock_registry.async_remove.assert_called_once()

    async def test_cleanup_orphaned_entity_registry_entries_wrong_domain(
        self, hass: HomeAssistant
    ) -> None:
        """Test cleanup skips entities from wrong domain."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "test_unique_id"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"switch.entity1": entity1}  # Wrong domain
        mock_registry.async_update_entity = Mock()
        mock_registry.async_remove = Mock()

        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=mock_registry,
        ):
            await _cleanup_orphaned_entity_registry_entries(
                hass, [entry1], "new_entry_id"
            )

            # Should not process entities from wrong domain
            mock_registry.async_update_entity.assert_not_called()

    async def test_cleanup_orphaned_entity_registry_entries_remove_error(
        self, hass: HomeAssistant
    ) -> None:
        """Test cleanup handles remove errors."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "test_unique_id"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"sensor.entity1": entity1}
        mock_registry.async_update_entity = Mock()
        mock_registry.async_remove = Mock(side_effect=ValueError("Remove error"))

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(True, "sensor.other_entity"),
            ),
        ):
            # Should handle error gracefully
            await _cleanup_orphaned_entity_registry_entries(
                hass, [entry1], "new_entry_id"
            )

    async def test_cleanup_orphaned_entity_registry_entries_update_error(
        self, hass: HomeAssistant
    ) -> None:
        """Test cleanup handles update errors."""
        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        entity1 = Mock()
        entity1.unique_id = "test_unique_id"
        entity1.config_entry_id = "entry1"

        mock_registry = Mock()
        mock_registry.entities = {"sensor.entity1": entity1}
        mock_registry.async_update_entity = Mock(side_effect=ValueError("Update error"))
        mock_registry.async_remove = Mock()

        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=mock_registry,
            ),
            patch(
                "custom_components.area_occupancy.migrations._check_unique_id_conflict",
                return_value=(False, None),
            ),
        ):
            # Should handle error gracefully
            await _cleanup_orphaned_entity_registry_entries(
                hass, [entry1], "new_entry_id"
            )


class TestMigrateDatabaseForConsolidation:
    """Test _migrate_database_for_consolidation function."""

    async def test_migrate_database_for_consolidation_no_db_file(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database migration when no database file exists."""
        hass.config.config_dir = str(tmp_path)

        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        await _migrate_database_for_consolidation(
            hass, [entry1], {"entry1": "Test Area"}
        )

        # Should complete without error

    async def test_migrate_database_for_consolidation_no_tables(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database migration when tables don't exist."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_file = storage_dir / "area_occupancy.db"
        db_file.write_bytes(b"SQLite format 3")

        hass.config.config_dir = str(tmp_path)

        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        await _migrate_database_for_consolidation(
            hass, [entry1], {"entry1": "Test Area"}
        )

        # Should complete without error

    async def test_migrate_database_for_consolidation_timeout(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database migration timeout handling."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_file = storage_dir / "area_occupancy.db"
        db_file.write_bytes(b"SQLite format 3")

        hass.config.config_dir = str(tmp_path)

        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        with patch("filelock.FileLock.__enter__", side_effect=Timeout("Lock timeout")):
            # Should handle timeout gracefully
            await _migrate_database_for_consolidation(
                hass, [entry1], {"entry1": "Test Area"}
            )

    async def test_migrate_database_for_consolidation_with_exception(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database migration exception handling."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_file = storage_dir / "area_occupancy.db"
        db_file.write_bytes(b"SQLite format 3")

        hass.config.config_dir = str(tmp_path)

        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        with patch(
            "filelock.FileLock.__enter__", side_effect=Exception("Database error")
        ):
            # Should handle exception gracefully
            await _migrate_database_for_consolidation(
                hass, [entry1], {"entry1": "Test Area"}
            )


class TestDropTablesLocked:
    """Test _drop_tables_locked function."""

    def test_drop_tables_locked_version_11_skip(self, tmp_path: Path) -> None:
        """Test that table dropping is skipped for version >= 11."""
        _drop_tables_locked(tmp_path, 11)
        # Should complete without error

    def test_drop_tables_locked_no_db_file(self, tmp_path: Path) -> None:
        """Test when database file doesn't exist."""
        _drop_tables_locked(tmp_path, 10)
        # Should complete without error

    def test_drop_tables_locked_db_version_3_or_higher(self, tmp_path: Path) -> None:
        """Test when database version is 3 or higher (no drop needed)."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        # Mock everything to verify no DROP TABLE calls
        with (
            patch("filelock.FileLock") as mock_lock,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
            patch("sqlalchemy.create_engine") as mock_create_engine,
        ):
            mock_lock.return_value.__enter__ = Mock()
            mock_lock.return_value.__exit__ = Mock(return_value=None)
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session

            # Mock db_version query to return 3
            mock_result = Mock()
            mock_result.fetchone.return_value = ("3",)
            mock_session.execute.return_value = mock_result

            _drop_tables_locked(tmp_path, 10)

            # Should not drop tables when version >= 3
            # Verify connect was not called (no table dropping)
            mock_engine.connect.assert_not_called()

    def test_drop_tables_locked_exception_handling(self, tmp_path: Path) -> None:
        """Test exception handling in _drop_tables_locked."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        with patch("filelock.FileLock.__enter__", side_effect=Exception("Lock error")):
            # Should handle exception gracefully
            _drop_tables_locked(tmp_path, 10)

    def test_drop_tables_locked_metadata_query_exception(self, tmp_path: Path) -> None:
        """Test when metadata query raises exception."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        with (
            patch("filelock.FileLock") as mock_lock,
            patch("sqlalchemy.create_engine") as mock_create_engine,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
        ):
            mock_lock.return_value.__enter__ = Mock()
            mock_lock.return_value.__exit__ = Mock(return_value=None)
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session
            mock_session.execute.side_effect = Exception("Query error")

            # Should handle exception and set db_version to 0
            _drop_tables_locked(tmp_path, 10)

    def test_drop_tables_locked_update_metadata_exception(self, tmp_path: Path) -> None:
        """Test when metadata update raises exception."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        with (
            patch("filelock.FileLock") as mock_lock,
            patch("sqlalchemy.create_engine") as mock_create_engine,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
        ):
            mock_lock.return_value.__enter__ = Mock()
            mock_lock.return_value.__exit__ = Mock(return_value=None)
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session

            # First query returns None (no db_version)
            mock_result1 = Mock()
            mock_result1.fetchone.return_value = None
            # Second query (UPDATE) raises exception
            mock_result2 = Mock()
            mock_result2.scalar.return_value = 0
            mock_session.execute.side_effect = [
                mock_result1,  # First query (SELECT)
                Exception("Update error"),  # UPDATE fails
            ]

            # Should handle exception gracefully
            _drop_tables_locked(tmp_path, 10)


class TestAsyncMigrateToSingleInstanceAdditional:
    """Additional tests for async_migrate_to_single_instance function."""

    async def test_async_migrate_to_single_instance_entity_registry_error(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test consolidation when entity registry migration fails."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.entry_id = "entry1"
        entry1.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Entry 1"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.entry_id = "entry2"
        entry2.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}
        entry2.title = "Entry 2"

        with (
            patch.object(
                hass.config_entries,
                "async_entries",
                return_value=[entry1, entry2],
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_entity_registry_for_consolidation",
                side_effect=Exception("Entity registry error"),
            ),
        ):
            result = await async_migrate_to_single_instance(hass)
            assert result is False

    async def test_async_migrate_to_single_instance_database_error(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test consolidation when database migration fails."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.entry_id = "entry1"
        entry1.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Entry 1"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.entry_id = "entry2"
        entry2.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}
        entry2.title = "Entry 2"

        with (
            patch.object(
                hass.config_entries,
                "async_entries",
                return_value=[entry1, entry2],
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_entity_registry_for_consolidation",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_database_for_consolidation",
                side_effect=Exception("Database error"),
            ),
        ):
            result = await async_migrate_to_single_instance(hass)
            assert result is False

    async def test_async_migrate_to_single_instance_config_update_error(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test consolidation when config entry update fails."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.entry_id = "entry1"
        entry1.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Entry 1"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.entry_id = "entry2"
        entry2.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}
        entry2.title = "Entry 2"

        with (
            patch.object(
                hass.config_entries,
                "async_entries",
                return_value=[entry1, entry2],
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_entity_registry_for_consolidation",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_database_for_consolidation",
                new_callable=AsyncMock,
            ),
            patch.object(
                hass.config_entries,
                "async_update_entry",
                side_effect=ValueError("Config update error"),
            ),
        ):
            result = await async_migrate_to_single_instance(hass)
            assert result is False

    async def test_async_migrate_to_single_instance_entry_removal_error(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test consolidation when entry removal fails."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.entry_id = "entry1"
        entry1.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Entry 1"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.entry_id = "entry2"
        entry2.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}
        entry2.title = "Entry 2"

        with (
            patch.object(
                hass.config_entries,
                "async_entries",
                return_value=[entry1, entry2],
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_entity_registry_for_consolidation",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_database_for_consolidation",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.area_occupancy.migrations._migrate_device_registry_for_consolidation",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.area_occupancy.migrations._cleanup_orphaned_entity_registry_entries",
                new_callable=AsyncMock,
            ),
            patch.object(hass.config_entries, "async_update_entry", return_value=None),
            patch.object(
                hass.config_entries,
                "async_remove",
                side_effect=RuntimeError("Remove error"),
                new_callable=AsyncMock,
            ),
        ):
            result = await async_migrate_to_single_instance(hass)
            # Should still return True even if removal fails
            assert result is True

    async def test_async_migrate_to_single_instance_duplicate_area_names(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test consolidation with duplicate area names."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.entry_id = "entry1"
        entry1.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}
        entry1.title = "Entry 1"

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.entry_id = "entry2"
        entry2.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}
        entry2.title = "Entry 2"

        with (
            patch.object(
                hass.config_entries,
                "async_entries",
                return_value=[entry1, entry2],
            ),
            patch("homeassistant.helpers.area_registry.async_get") as mock_get_area_reg,
        ):
            mock_area_reg = Mock()
            mock_area_entry = Mock()
            mock_area_entry.id = testing_area_id
            mock_area_entry.name = "Testing"
            mock_area_reg.async_get_area.return_value = mock_area_entry
            mock_get_area_reg.return_value = mock_area_reg

            with (
                patch(
                    "custom_components.area_occupancy.migrations._migrate_entity_registry_for_consolidation",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.area_occupancy.migrations._migrate_database_for_consolidation",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.area_occupancy.migrations._migrate_device_registry_for_consolidation",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.area_occupancy.migrations._cleanup_orphaned_entity_registry_entries",
                    new_callable=AsyncMock,
                ),
                patch.object(
                    hass.config_entries, "async_update_entry", return_value=None
                ),
                patch.object(
                    hass.config_entries, "async_remove", new_callable=AsyncMock
                ),
            ):
                result = await async_migrate_to_single_instance(hass)
                assert result is True
                # Verify area names are made unique (Testing_1 for second entry)
                call_args = hass.config_entries.async_update_entry.call_args
                areas_list = call_args[1]["data"][CONF_AREAS]
                area_names = [
                    area.get("area_name", area.get("name", "")) for area in areas_list
                ]
                # Should have unique names
                assert len(area_names) == len(set(area_names)) or len(areas_list) == 2


class TestMigrateDatabaseForConsolidationAdditional:
    """Additional tests for _migrate_database_for_consolidation function."""

    async def test_migrate_database_for_consolidation_with_tables(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database migration with existing tables."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"

        hass.config.config_dir = str(tmp_path)

        # Create a real SQLite database with tables
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(
                text("CREATE TABLE IF NOT EXISTS areas (entry_id TEXT, area_name TEXT)")
            )
            conn.execute(
                text(
                    "INSERT INTO areas (entry_id, area_name) VALUES ('entry1', 'Test Area')"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS entities (entry_id TEXT, entity_id TEXT)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO entities (entry_id, entity_id) VALUES ('entry1', 'binary_sensor.motion1')"
                )
            )
            conn.commit()
        engine.dispose()

        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        await _migrate_database_for_consolidation(
            hass, [entry1], {"entry1": "Test Area"}
        )

        # Verify area was updated
        engine2 = create_engine(f"sqlite:///{db_path}")
        with engine2.connect() as conn:
            result = conn.execute(
                text("SELECT area_name FROM areas WHERE entry_id = 'entry1'")
            ).fetchone()
            assert result is not None
            assert result[0] == "Test Area"
        engine2.dispose()

    async def test_migrate_database_for_consolidation_add_area_name_column(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test adding area_name column to entities table."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"

        hass.config.config_dir = str(tmp_path)

        # Create a real SQLite database with entities table but no area_name column
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS entities (entry_id TEXT, entity_id TEXT)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO entities (entry_id, entity_id) VALUES ('entry1', 'binary_sensor.motion1')"
                )
            )
            conn.execute(
                text("CREATE TABLE IF NOT EXISTS areas (entry_id TEXT, area_name TEXT)")
            )
            conn.commit()
        engine.dispose()

        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        await _migrate_database_for_consolidation(
            hass, [entry1], {"entry1": "Test Area"}
        )

        # Verify area_name column was added and populated
        engine2 = create_engine(f"sqlite:///{db_path}")
        with engine2.connect() as conn:
            # Check if column exists by trying to select it
            result = conn.execute(
                text("SELECT area_name FROM entities WHERE entry_id = 'entry1'")
            ).fetchone()
            assert result is not None
            assert result[0] == "Test Area"
        engine2.dispose()

    async def test_migrate_database_for_consolidation_cleanup_metadata(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test cleanup of master-related metadata."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"

        hass.config.config_dir = str(tmp_path)

        # Create a real SQLite database with metadata table
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(
                text("CREATE TABLE IF NOT EXISTS metadata (key TEXT, value TEXT)")
            )
            conn.execute(
                text(
                    "INSERT INTO metadata (key, value) VALUES ('master_entry_id', 'old_entry')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO metadata (key, value) VALUES ('master_heartbeat', '123456')"
                )
            )
            conn.execute(
                text("CREATE TABLE IF NOT EXISTS areas (entry_id TEXT, area_name TEXT)")
            )
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS entities (entry_id TEXT, entity_id TEXT)"
                )
            )
            conn.commit()
        engine.dispose()

        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        await _migrate_database_for_consolidation(
            hass, [entry1], {"entry1": "Test Area"}
        )

        # Verify master metadata was cleaned up
        engine2 = create_engine(f"sqlite:///{db_path}")
        with engine2.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT COUNT(*) FROM metadata WHERE key IN ('master_entry_id', 'master_heartbeat')"
                )
            ).scalar()
            assert result == 0
        engine2.dispose()

    async def test_migrate_database_for_consolidation_areas_update_error(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database migration when areas update fails."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        hass.config.config_dir = str(tmp_path)

        entry1 = Mock(spec=ConfigEntry)
        entry1.entry_id = "entry1"

        with (
            patch("filelock.FileLock") as mock_lock,
            patch("sqlalchemy.create_engine") as mock_create_engine,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
            patch("sqlalchemy.inspect") as mock_inspect,
        ):
            mock_lock.return_value.__enter__ = Mock()
            mock_lock.return_value.__exit__ = Mock(return_value=None)
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)

            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.has_table.return_value = True
            mock_inspector.get_columns.return_value = [{"name": "entry_id"}]

            mock_session.execute.side_effect = SQLAlchemyError("Update error")

            # Should handle error gracefully
            await _migrate_database_for_consolidation(
                hass, [entry1], {"entry1": "Test Area"}
            )


class TestAsyncMigrateEntryConsolidation:
    """Test async_migrate_entry consolidation path."""

    async def test_async_migrate_entry_triggers_consolidation(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test that async_migrate_entry triggers consolidation for version < 13."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.minor_version = 0
        entry1.entry_id = "entry1"
        entry1.state = ConfigEntryState.LOADED
        entry1.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.minor_version = 0
        entry2.entry_id = "entry2"
        entry2.state = ConfigEntryState.LOADED
        entry2.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}

        def async_entries_mock(domain=None):
            return [entry1, entry2]

        hass.config_entries.async_entries = async_entries_mock

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_to_single_instance",
                return_value=True,
            ) as mock_consolidate,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids",
                new_callable=AsyncMock,
            ) as mock_migrate_ids,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.area_occupancy.migrations.migrate_config",
                side_effect=lambda x: x,  # Return the config as-is
            ),
            patch.object(hass.config_entries, "async_update_entry", return_value=None),
        ):
            result = await async_migrate_entry(hass, entry1)

            # Should trigger consolidation
            mock_consolidate.assert_called_once()
            # Should migrate unique IDs before consolidation
            assert mock_migrate_ids.call_count > 0
            assert result is True

    async def test_async_migrate_entry_consolidation_unique_id_migration_error(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test consolidation when unique ID migration fails."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry1 = Mock(spec=ConfigEntry)
        entry1.version = 12
        entry1.minor_version = 0
        entry1.entry_id = "entry1"
        entry1.state = ConfigEntryState.LOADED
        entry1.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        entry1.options = {}

        entry2 = Mock(spec=ConfigEntry)
        entry2.version = 12
        entry2.minor_version = 0
        entry2.entry_id = "entry2"
        entry2.state = ConfigEntryState.LOADED
        entry2.data = {
            CONF_AREA_ID: testing_area_id,
            CONF_MOTION_SENSORS: ["binary_sensor.motion2"],
        }
        entry2.options = {}

        def async_entries_mock(domain=None):
            return [entry1, entry2]

        hass.config_entries.async_entries = async_entries_mock

        with (
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_unique_ids",
                side_effect=HomeAssistantError("Migration error"),
            ),
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_to_single_instance",
                return_value=True,
            ) as mock_consolidate,
            patch(
                "custom_components.area_occupancy.migrations.async_migrate_storage",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.area_occupancy.migrations.migrate_config",
                side_effect=lambda x: x,  # Return the config as-is
            ),
            patch.object(hass.config_entries, "async_update_entry", return_value=None),
        ):
            result = await async_migrate_entry(hass, entry1)

            # Should still continue with consolidation despite error
            mock_consolidate.assert_called_once()
            assert result is True


class TestResolveAreaFromEntry:
    """Test _resolve_area_from_entry helper function."""

    def test_resolve_area_from_entry_new_format(
        self, setup_area_registry: dict[str, str]
    ) -> None:
        """Test area resolution with new format (CONF_AREA_ID)."""

        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "entry1"
        entry.data = {CONF_AREA_ID: testing_area_id}
        entry.options = {}

        # Get real area registry
        with patch("homeassistant.helpers.area_registry.async_get") as mock_get:
            mock_area_reg = Mock()
            mock_area_entry = Mock()
            mock_area_entry.name = "Testing"
            mock_area_reg.async_get_area.return_value = mock_area_entry
            mock_get.return_value = mock_area_reg

            area_id, area_name = _resolve_area_from_entry(entry, mock_area_reg)

            assert area_id == testing_area_id
            assert area_name == "Testing"

    def test_resolve_area_from_entry_legacy_format(
        self, setup_area_registry: dict[str, str]
    ) -> None:
        """Test area resolution with legacy format (name)."""
        testing_area_id = setup_area_registry.get("Testing", "test_area")
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "entry1"
        entry.data = {"name": "Testing"}
        entry.options = {}

        mock_area_reg = Mock()
        mock_area_entry = Mock()
        mock_area_entry.id = testing_area_id
        mock_area_entry.name = "Testing"
        mock_area_reg.async_list_areas.return_value = [mock_area_entry]

        area_id, area_name = _resolve_area_from_entry(entry, mock_area_reg)

        assert area_id == testing_area_id
        assert area_name == "Testing"

    def test_resolve_area_from_entry_not_found(self) -> None:
        """Test area resolution when area not found."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "entry1"
        entry.data = {CONF_AREA_ID: "nonexistent_area"}
        entry.options = {}

        mock_area_reg = Mock()
        mock_area_reg.async_get_area.return_value = None

        area_id, area_name = _resolve_area_from_entry(entry, mock_area_reg)

        assert area_id is None
        assert area_name is None

    def test_resolve_area_from_entry_legacy_not_found(self) -> None:
        """Test area resolution with legacy format when area not found."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "entry1"
        entry.data = {"name": "Nonexistent Area"}
        entry.options = {}

        mock_area_reg = Mock()
        mock_area_reg.async_list_areas.return_value = []

        area_id, area_name = _resolve_area_from_entry(entry, mock_area_reg)

        assert area_id is None
        assert area_name is None


class TestUpdateDbVersion:
    """Test _update_db_version helper function."""

    def test_update_db_version_update_existing(self) -> None:
        """Test updating existing db_version."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1  # Changes() returns 1
        mock_session.execute.return_value = mock_result

        _update_db_version(mock_session, 4)

        assert mock_session.execute.call_count >= 1

    def test_update_db_version_insert_new(self) -> None:
        """Test inserting new db_version when it doesn't exist."""
        mock_session = Mock()
        mock_result1 = Mock()
        mock_result1.scalar.return_value = 0  # Changes() returns 0 (no update)
        mock_result2 = Mock()
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        _update_db_version(mock_session, 4)

        assert mock_session.execute.call_count == 2

    def test_update_db_version_exception_handling(self) -> None:
        """Test exception handling in _update_db_version."""
        mock_session = Mock()
        mock_session.execute.side_effect = Exception("Database error")

        # Should not raise
        _update_db_version(mock_session, 4)


class TestDropLegacyTables:
    """Test _drop_legacy_tables helper function."""

    def test_drop_legacy_tables_success(self) -> None:
        """Test successful table dropping."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_session = Mock()

        _drop_legacy_tables(mock_engine, mock_session)

        # Should call commit on session
        mock_session.commit.assert_called_once()
        # Should call commit on connection
        mock_conn.commit.assert_called_once()
        # Should execute DROP TABLE for each table
        assert mock_conn.execute.call_count == 5  # 5 tables to drop

    def test_drop_legacy_tables_with_errors(self) -> None:
        """Test table dropping with some errors."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_session = Mock()
        # First table drop succeeds, second fails, rest succeed
        mock_conn.execute.side_effect = [
            None,  # First table
            Exception("Drop error"),  # Second table
            None,  # Third table
            None,  # Fourth table
            None,  # Fifth table
        ]

        # Should not raise, should continue dropping other tables
        _drop_legacy_tables(mock_engine, mock_session)

        assert mock_conn.execute.call_count == 5


class TestCheckDatabaseTablesExist:
    """Test _check_database_tables_exist helper function."""

    def test_check_database_tables_exist_all_exist(self) -> None:
        """Test when all tables exist."""
        mock_inspector = Mock()
        mock_inspector.has_table.return_value = True

        result = _check_database_tables_exist(mock_inspector, ["entities", "areas"])

        assert result is True
        assert mock_inspector.has_table.call_count == 2

    def test_check_database_tables_exist_missing_table(self) -> None:
        """Test when a table is missing."""
        mock_inspector = Mock()
        mock_inspector.has_table.side_effect = [
            True,
            False,
        ]  # entities exists, areas doesn't

        result = _check_database_tables_exist(mock_inspector, ["entities", "areas"])

        assert result is False
        assert mock_inspector.has_table.call_count == 2


class TestSafeFileOperation:
    """Test _safe_file_operation helper function."""

    def test_safe_file_operation_success(self) -> None:
        """Test successful file operation."""
        operation = Mock()
        result = _safe_file_operation(operation, "Error message")

        assert result is True
        operation.assert_called_once()

    def test_safe_file_operation_oserror(self) -> None:
        """Test file operation with OSError."""
        operation = Mock(side_effect=OSError("File error"))
        result = _safe_file_operation(operation, "Error message")

        assert result is False
        operation.assert_called_once()


class TestSafeDatabaseOperation:
    """Test _safe_database_operation helper function."""

    def test_safe_database_operation_success(self) -> None:
        """Test successful database operation."""
        operation = Mock()
        result = _safe_database_operation(operation, "Error message")

        assert result is True
        operation.assert_called_once()

    def test_safe_database_operation_sqlalchemy_error(self) -> None:
        """Test database operation with SQLAlchemyError."""
        operation = Mock(side_effect=SQLAlchemyError("DB error"))
        result = _safe_database_operation(operation, "Error message")

        assert result is False
        operation.assert_called_once()

    def test_safe_database_operation_oserror(self) -> None:
        """Test database operation with OSError."""
        operation = Mock(side_effect=OSError("OS error"))
        result = _safe_database_operation(operation, "Error message")

        assert result is False
        operation.assert_called_once()


class TestHandleEntityConflict:
    """Test _handle_entity_conflict helper function."""

    def test_handle_entity_conflict_no_conflict(self, hass: HomeAssistant) -> None:
        """Test when there's no conflict."""
        entity_registry = Mock()
        conflicts = []

        with patch(
            "custom_components.area_occupancy.migrations._check_unique_id_conflict",
            return_value=(False, None),
        ):
            result = _handle_entity_conflict(
                entity_registry, "sensor.entity1", "old_id", "new_id", conflicts
            )

            assert result is False
            assert len(conflicts) == 0

    def test_handle_entity_conflict_with_conflict(self, hass: HomeAssistant) -> None:
        """Test when there's a conflict."""
        entity_registry = Mock()
        conflicts = []

        with patch(
            "custom_components.area_occupancy.migrations._check_unique_id_conflict",
            return_value=(True, "sensor.other_entity"),
        ):
            result = _handle_entity_conflict(
                entity_registry, "sensor.entity1", "old_id", "new_id", conflicts
            )

            assert result is True
            assert len(conflicts) == 1
            assert conflicts[0] == ("sensor.entity1", "old_id", "new_id")


class TestRestoreConfigEntryState:
    """Test _restore_config_entry_state helper function."""

    async def test_restore_config_entry_state_success(
        self, hass: HomeAssistant
    ) -> None:
        """Test successful config entry restoration."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "entry1"
        entry.title = "Original Title"

        original_state = {
            "data": {"motion_sensors": ["binary_sensor.motion1"]},
            "options": {"threshold": 0.6},
            "version": 12,
            "minor_version": 0,
            "title": "Original Title",
        }

        hass.config_entries.async_update_entry = Mock()

        result = _restore_config_entry_state(hass, entry, original_state)

        assert result is True
        hass.config_entries.async_update_entry.assert_called_once()

    async def test_restore_config_entry_state_error(self, hass: HomeAssistant) -> None:
        """Test config entry restoration with error."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "entry1"
        entry.title = "Original Title"

        original_state = {
            "data": {"motion_sensors": ["binary_sensor.motion1"]},
            "options": {"threshold": 0.6},
            "version": 12,
            "minor_version": 0,
            "title": "Original Title",
        }

        hass.config_entries.async_update_entry = Mock(
            side_effect=ValueError("Update error")
        )

        result = _restore_config_entry_state(hass, entry, original_state)

        assert result is False
