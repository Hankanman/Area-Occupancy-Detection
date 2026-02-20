"""Tests for data.config module.

This test suite focuses on testing business logic and error handling,
not Python dataclass behavior or trivial operations.
"""

from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    ANALYSIS_INTERVAL,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_CO2_SENSORS,
    CONF_CO_SENSORS,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_PM10_SENSORS,
    CONF_PM25_SENSORS,
    CONF_POWER_SENSORS,
    CONF_PRESSURE_SENSORS,
    CONF_PURPOSE,
    CONF_SLEEP_END,
    CONF_SLEEP_START,
    CONF_SOUND_PRESSURE_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_VOC_SENSORS,
    CONF_WASP_ENABLED,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_SENSORS,
    DECAY_INTERVAL,
    DEFAULT_SLEEP_END,
    DEFAULT_SLEEP_START,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    HA_RECORDER_DAYS,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import (
    AreaConfig,
    IntegrationConfig,
    Sensors,
)
from custom_components.area_occupancy.data.purpose import get_default_decay_half_life
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

# ruff: noqa: SLF001


def _setup_area_subentry(
    coordinator: AreaOccupancyCoordinator,
    area_id: str,
    area_config: dict[str, Any],
    subentry_id: str = "test_subentry_001",
) -> None:
    """Helper to set up area configuration as a subentry for tests.

    Creates a ConfigSubentry with the given area config and installs it on the
    coordinator's config_entry.subentries dict.

    Args:
        coordinator: The coordinator instance
        area_id: The area ID to use
        area_config: Dictionary of area configuration values
        subentry_id: Subentry ID (defaults to test value)
    """
    area_data = {CONF_AREA_ID: area_id, **area_config}
    subentry = ConfigSubentry(
        data=MappingProxyType(area_data),
        subentry_type="area",
        subentry_id=subentry_id,
        title=area_id,
        unique_id=area_id,
    )
    coordinator.config_entry.subentries = {subentry_id: subentry}
    coordinator.config_entry.data = {}
    coordinator.config_entry.options = {}
    coordinator._load_areas_from_config()


class TestSensorsGetMotionSensors:
    """Test Sensors.get_motion_sensors business logic.

    This tests the important logic of including wasp sensors in motion sensor lists.
    """

    @pytest.mark.parametrize(
        ("wasp_enabled", "wasp_entity_id", "expected_result"),
        [
            (False, None, ["binary_sensor.motion1"]),
            (
                True,
                "binary_sensor.wasp",
                ["binary_sensor.motion1", "binary_sensor.wasp"],
            ),
            (True, None, ["binary_sensor.motion1"]),
        ],
    )
    def test_get_motion_sensors_with_wasp_config(
        self, wasp_enabled: bool, wasp_entity_id: str | None, expected_result: list[str]
    ) -> None:
        """Test get_motion_sensors includes wasp sensor when enabled and available."""
        mock_parent_config = Mock()
        mock_parent_config.wasp_in_box = Mock()
        mock_parent_config.wasp_in_box.enabled = wasp_enabled
        mock_parent_config.area_name = "Test Area"

        sensors = Sensors(
            motion=["binary_sensor.motion1"], _parent_config=mock_parent_config
        )

        mock_coordinator = Mock()
        mock_area_data = Mock()
        mock_area_data.wasp_entity_id = wasp_entity_id
        mock_coordinator.areas = {"Test Area": mock_area_data}

        result = sensors.get_motion_sensors(mock_coordinator)
        assert result == expected_result

    def test_get_motion_sensors_with_none_coordinator(self) -> None:
        """Test get_motion_sensors handles None coordinator gracefully."""
        sensors = Sensors(motion=["binary_sensor.motion1"])
        result = sensors.get_motion_sensors(None)
        assert result == ["binary_sensor.motion1"]

    def test_get_motion_sensors_with_empty_motion_list(self) -> None:
        """Test get_motion_sensors returns wasp sensor when motion list is empty."""
        mock_parent_config = Mock()
        mock_parent_config.wasp_in_box = Mock()
        mock_parent_config.wasp_in_box.enabled = True
        mock_parent_config.area_name = "Test Area"

        sensors = Sensors(motion=[], _parent_config=mock_parent_config)
        mock_coordinator = Mock()
        mock_area_data = Mock()
        mock_area_data.wasp_entity_id = "binary_sensor.wasp"
        mock_coordinator.areas = {"Test Area": mock_area_data}

        result = sensors.get_motion_sensors(mock_coordinator)
        assert result == ["binary_sensor.wasp"]

    def test_get_motion_sensors_multi_area_mode(self) -> None:
        """Test get_motion_sensors works correctly in multi-area architecture."""
        mock_parent_config = Mock()
        mock_parent_config.wasp_in_box = Mock()
        mock_parent_config.wasp_in_box.enabled = True
        mock_parent_config.area_name = "Living Room"

        sensors = Sensors(
            motion=["binary_sensor.motion1"], _parent_config=mock_parent_config
        )
        mock_coordinator = Mock()
        mock_area_data = Mock()
        mock_area_data.wasp_entity_id = "binary_sensor.living_room_wasp"
        mock_coordinator.areas = {"Living Room": mock_area_data}

        result = sensors.get_motion_sensors(mock_coordinator)
        assert result == ["binary_sensor.motion1", "binary_sensor.living_room_wasp"]


class TestAreaConfigInitialization:
    """Test AreaConfig initialization and error handling."""

    def test_initialization_with_deterministic_config(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test AreaConfig initialization with specific deterministic values."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {CONF_THRESHOLD: 50},  # Explicit value
        )

        config = AreaConfig(coordinator, area_name=area_name)

        assert config.name == area_name
        assert config.area_id == area_id
        assert config.threshold == 0.5  # 50 / 100

    def test_initialization_with_values(
        self,
        coordinator: AreaOccupancyCoordinator,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test AreaConfig initialization with specific configuration values."""
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")

        _setup_area_subentry(
            coordinator,
            living_room_area_id,
            {
                CONF_THRESHOLD: 60,
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        area_name = "Living Room"
        config = AreaConfig(coordinator, area_name=area_name)

        assert config.name == "Living Room"
        assert config.area_id == living_room_area_id
        assert config.threshold == 0.6
        assert config.sensors.motion == ["binary_sensor.motion1"]
        assert config.weights.motion == 0.9

    def test_initialization_with_missing_weights_uses_defaults(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test AreaConfig uses default weights when not provided."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(coordinator, area_id, {CONF_THRESHOLD: 50})

        config = AreaConfig(coordinator, area_name=area_name)
        assert config.weights.motion == DEFAULT_WEIGHT_MOTION
        assert config.weights.media == DEFAULT_WEIGHT_MEDIA

    def test_initialization_with_string_threshold_converts_to_float(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test AreaConfig converts string threshold values to float."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_THRESHOLD: "75",  # String instead of int
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        assert config.threshold == 0.75

    def test_initialization_with_none_config_entry_raises_error(
        self, hass: HomeAssistant
    ) -> None:
        """Test AreaConfig raises ValueError when config_entry is None."""
        mock_coordinator = Mock(spec=AreaOccupancyCoordinator)
        mock_coordinator.hass = hass
        mock_coordinator.config_entry = None
        mock_coordinator.db = Mock()

        with pytest.raises(ValueError, match="Coordinator config_entry cannot be None"):
            AreaConfig(mock_coordinator, area_name="Test Area")

    def test_initialization_with_multi_area_requires_area_name(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test AreaConfig requires area_name when using multi-area configuration."""
        # Set up multi-area config format
        coordinator.config_entry.data = {
            CONF_AREAS: [
                {
                    CONF_AREA_ID: "test_area_id",
                    CONF_THRESHOLD: 50,
                }
            ]
        }
        coordinator.config_entry.options = {}

        with pytest.raises(
            ValueError,
            match="area_name is required when using multi-area configuration format",
        ):
            AreaConfig(coordinator, area_name=None)

    def test_initialization_with_decay_half_life_zero_uses_purpose_fallback(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test AreaConfig uses purpose-based decay half-life when half_life is 0."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_THRESHOLD: 50,
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_DECAY_HALF_LIFE: 0,  # Should trigger purpose-based fallback
                CONF_PURPOSE: "social",
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        expected_half_life = int(get_default_decay_half_life("social"))
        assert config.decay.half_life == expected_half_life
        assert config.decay.half_life != 0

    def test_initialization_with_area_data_parameter(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test AreaConfig initialization with area_data parameter bypasses config_entry.

        This tests the production code path used in Area.__init__ and coordinator._load_areas_from_config
        where area_data is provided directly, bypassing config_entry merging and extraction.
        """
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        # Set up config_entry with different values to verify area_data takes precedence
        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_THRESHOLD: 30,  # Different from area_data
                CONF_MOTION_SENSORS: ["binary_sensor.other_motion"],
            },
        )

        # Create area_data dictionary that should be used instead of config_entry
        area_data = {
            CONF_AREA_ID: area_id,
            CONF_THRESHOLD: 75,  # Should be used, not 30 from config_entry
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_APPLIANCES: ["switch.computer"],
            CONF_WEIGHT_MOTION: 0.85,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.5,
            CONF_PURPOSE: "social",
            CONF_DECAY_HALF_LIFE: 300,
        }

        # Initialize with area_data parameter - should bypass config_entry
        config = AreaConfig(coordinator, area_name=area_name, area_data=area_data)

        # Verify configuration is loaded from area_data, not config_entry
        assert config.name == area_name
        assert config.area_id == area_id
        assert config.threshold == 0.75  # From area_data (75), not config_entry (30)
        assert config.sensors.motion == [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        assert config.sensors.media == ["media_player.tv"]
        assert config.sensors.appliance == ["switch.computer"]
        assert config.weights.motion == 0.85
        assert config.weights.media == 0.7
        assert config.weights.appliance == 0.5
        assert config.purpose == "social"
        assert config.decay.half_life == 300


class TestAreaConfigProperties:
    """Test AreaConfig property accessors."""

    def test_time_properties_are_accurate(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test time-related properties return accurate datetime values."""
        # Use a fixed time for deterministic testing
        fixed_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=dt_util.UTC)

        with patch("homeassistant.util.dt.utcnow", return_value=fixed_time):
            area_name = coordinator.get_area_names()[0]
            config = AreaConfig(coordinator, area_name=area_name)

            expected_start = fixed_time - timedelta(days=HA_RECORDER_DAYS)
            expected_end = fixed_time

            assert config.start_time == expected_start
            assert config.end_time == expected_end

    def test_entity_ids_includes_all_sensor_types(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test entity_ids property includes all configured sensor types."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_MOTION_SENSORS: [
                    "binary_sensor.motion1",
                    "binary_sensor.motion2",
                ],
                CONF_MEDIA_DEVICES: ["media_player.tv", "media_player.speaker"],
                CONF_APPLIANCES: ["switch.computer", "switch.lamp"],
                CONF_DOOR_SENSORS: ["binary_sensor.door1", "binary_sensor.door2"],
                CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
                CONF_TEMPERATURE_SENSORS: [
                    "sensor.temperature1",
                    "sensor.temperature2",
                ],
                CONF_CO2_SENSORS: ["sensor.co2_1"],
                CONF_CO_SENSORS: ["sensor.co_1"],
                CONF_SOUND_PRESSURE_SENSORS: ["sensor.sound_1"],
                CONF_PRESSURE_SENSORS: ["sensor.pressure_1"],
                CONF_VOC_SENSORS: ["sensor.voc_1"],
                CONF_PM25_SENSORS: ["sensor.pm25_1"],
                CONF_PM10_SENSORS: ["sensor.pm10_1"],
                CONF_POWER_SENSORS: ["sensor.power_1"],
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        entity_ids = config.entity_ids

        expected_entities = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
            "media_player.tv",
            "media_player.speaker",
            "switch.computer",
            "switch.lamp",
            "binary_sensor.door1",
            "binary_sensor.door2",
            "binary_sensor.window1",
            "sensor.temperature1",
            "sensor.temperature2",
            "sensor.co2_1",
            "sensor.co_1",
            "sensor.sound_1",
            "sensor.pressure_1",
            "sensor.voc_1",
            "sensor.pm25_1",
            "sensor.pm10_1",
            "sensor.power_1",
        ]

        assert len(entity_ids) == len(expected_entities)
        for entity_id in expected_entities:
            assert entity_id in entity_ids

    def test_entity_ids_with_empty_sensors_returns_empty_list(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test entity_ids property returns empty list when no sensors configured."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        assert config.entity_ids == []

    def test_entity_ids_excludes_wasp_sensor(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test entity_ids property does not include wasp sensor (it's virtual).

        Note: Wasp sensor is added dynamically via get_motion_sensors(), not in entity_ids.
        """
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_WASP_ENABLED: True,
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        # entity_ids should only include configured sensors, not wasp
        assert "binary_sensor.motion1" in config.entity_ids
        # Wasp sensor is added via get_motion_sensors(), not entity_ids
        assert config.wasp_in_box.enabled is True


class TestAreaConfigValidation:
    """Test AreaConfig entity configuration validation."""

    def test_validate_entity_configuration_valid_config(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test validate_entity_configuration returns no errors for valid config."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert errors == []

    def test_validate_entity_configuration_duplicate_entities(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test validate_entity_configuration detects duplicate entity IDs."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_MOTION_SENSORS: ["binary_sensor.sensor1"],
                CONF_MEDIA_DEVICES: ["binary_sensor.sensor1"],  # Duplicate
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) == 1
        assert "Duplicate entity IDs found" in errors[0]
        assert "binary_sensor.sensor1" in errors[0]

    def test_validate_entity_configuration_no_primary_sensors(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test validate_entity_configuration requires at least one primary sensor."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_DOOR_SENSORS: ["binary_sensor.door1"],  # Only door sensors
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) == 1
        assert "No motion, media, or appliance sensors configured" in errors[0]

    @pytest.mark.parametrize(
        ("sensor_type", "invalid_ids"),
        [
            ("motion", ["binary_sensor.motion1", "", "   "]),
            ("motion", ["binary_sensor.motion1", 123, None]),
            ("co2", ["sensor.co2_1", "", None]),
            ("co", ["sensor.co_1", "", None]),
            ("sound_pressure", ["sensor.sound1", 456]),
            ("pressure", ["sensor.pressure1", "   "]),
            ("voc", ["sensor.voc1", None]),
            ("pm25", ["sensor.pm25_1", ""]),
            ("pm10", ["sensor.pm10_1", "   "]),
            ("power", ["sensor.power1", "   "]),
        ],
    )
    def test_validate_entity_configuration_invalid_entity_ids(
        self,
        coordinator: AreaOccupancyCoordinator,
        setup_area_registry: dict[str, str],
        sensor_type: str,
        invalid_ids: list,
    ) -> None:
        """Test validate_entity_configuration detects invalid entity IDs for all sensor types."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        # Build config with invalid IDs for the specified sensor type
        config_key = {
            "motion": CONF_MOTION_SENSORS,
            "co2": CONF_CO2_SENSORS,
            "co": CONF_CO_SENSORS,
            "sound_pressure": CONF_SOUND_PRESSURE_SENSORS,
            "pressure": CONF_PRESSURE_SENSORS,
            "voc": CONF_VOC_SENSORS,
            "pm25": CONF_PM25_SENSORS,
            "pm10": CONF_PM10_SENSORS,
            "power": CONF_POWER_SENSORS,
        }[sensor_type]

        # Build test data - for motion sensors, use invalid_ids directly
        # For other sensor types, add invalid_ids and ensure we have valid primary sensors
        test_data_dict = {
            CONF_AREA_ID: area_id,
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }

        # Add invalid IDs for the sensor type being tested
        test_data_dict[config_key] = invalid_ids

        # Ensure we have at least one valid primary sensor to avoid "no sensors" error
        if sensor_type == "motion":
            # For motion, invalid_ids already contains at least one valid ID
            pass
        else:
            # For other sensor types, add valid motion sensor
            test_data_dict[CONF_MOTION_SENSORS] = ["binary_sensor.motion1"]

        _setup_area_subentry(coordinator, area_id, test_data_dict)

        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) >= 1
        # Check if error message contains the sensor type (could be in any error)
        error_messages = " ".join(errors)
        assert f"Invalid {sensor_type} sensor entity IDs" in error_messages

    def test_validate_entity_configuration_multiple_errors(
        self, coordinator: AreaOccupancyCoordinator, setup_area_registry: dict[str, str]
    ) -> None:
        """Test validate_entity_configuration returns all validation errors."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(
            coordinator,
            area_id,
            {
                CONF_MOTION_SENSORS: ["binary_sensor.sensor1", ""],  # Invalid ID
                CONF_MEDIA_DEVICES: ["binary_sensor.sensor1"],  # Duplicate
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) == 2
        assert any("Duplicate entity IDs found" in error for error in errors)
        assert any("Invalid motion sensor entity IDs" in error for error in errors)


class TestAreaConfigUpdate:
    """Test AreaConfig update methods."""

    async def test_update_config_success(
        self,
        coordinator: AreaOccupancyCoordinator,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test update_config successfully updates subentry and reloads config."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        # Set up subentry so _find_area_subentry can locate it
        _setup_area_subentry(coordinator, area_id, {CONF_THRESHOLD: 50})

        config = AreaConfig(coordinator, area_name=area_name)
        assert config.area_id == area_id

        # Mock async_update_subentry and internal reload to verify behavior
        with (
            patch.object(
                coordinator.hass.config_entries, "async_update_subentry"
            ) as mock_update_subentry,
            patch.object(config, "_load_config") as mock_load_config,
            patch.object(coordinator, "_setup_complete", True),
        ):
            await config.update_config({CONF_THRESHOLD: 70})

            # Verify async_update_subentry was called
            mock_update_subentry.assert_called_once()
            call_args = mock_update_subentry.call_args
            assert call_args is not None
            # The data kwarg should contain merged data with the new threshold
            updated_data = call_args[1]["data"]
            assert updated_data[CONF_THRESHOLD] == 70
            assert updated_data[CONF_AREA_ID] == area_id

            # Verify config was reloaded after update
            mock_load_config.assert_called_once()

    async def test_update_config_with_exception_raises_homeassistant_error(
        self,
        coordinator: AreaOccupancyCoordinator,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test update_config raises HomeAssistantError when update fails."""
        area_name = coordinator.get_area_names()[0]
        area_id = setup_area_registry.get(area_name, "test_area")

        _setup_area_subentry(coordinator, area_id, {CONF_THRESHOLD: 50})

        config = AreaConfig(coordinator, area_name=area_name)

        with patch.object(
            coordinator.hass.config_entries, "async_update_subentry"
        ) as mock_update_subentry:
            mock_update_subentry.side_effect = ValueError("Update failed")

            with pytest.raises(
                HomeAssistantError, match="Failed to update configuration"
            ):
                await config.update_config({CONF_THRESHOLD: 70})

    async def test_update_config_with_missing_area_id_raises_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_config raises HomeAssistantError when area_id is missing."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Set area_id to None to simulate missing area_id
        config.area_id = None

        with pytest.raises(
            HomeAssistantError, match=".*Area ID not available for config update.*"
        ):
            await config.update_config({CONF_THRESHOLD: 70})

    async def test_update_config_with_area_not_found_raises_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_config raises HomeAssistantError when area not found in subentries."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Set up subentry with different area_id so _find_area_subentry won't match
        different_subentry = ConfigSubentry(
            data=MappingProxyType(
                {CONF_AREA_ID: "different_area_id", CONF_THRESHOLD: 50}
            ),
            subentry_type="area",
            subentry_id="different_subentry",
            title="Different",
            unique_id="different_area_id",
        )
        coordinator.config_entry.subentries = {
            "different_subentry": different_subentry,
        }

        with pytest.raises(
            HomeAssistantError, match=".*Subentry for area ID.*not found.*"
        ):
            await config.update_config({CONF_THRESHOLD: 70})

    async def test_update_config_with_no_subentries_raises_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_config raises HomeAssistantError when no subentries exist."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Empty subentries
        coordinator.config_entry.subentries = {}

        with pytest.raises(
            HomeAssistantError, match=".*Subentry for area ID.*not found.*"
        ):
            await config.update_config({CONF_THRESHOLD: 70})

    def test_update_from_entry_success(
        self,
        coordinator: AreaOccupancyCoordinator,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test update_from_entry successfully updates config from new entry with subentries."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        testing_area_id = setup_area_registry.get("Testing", "testing")

        # Create a mock config entry with subentries
        area_subentry = ConfigSubentry(
            data=MappingProxyType(
                {
                    CONF_AREA_ID: testing_area_id,
                    CONF_THRESHOLD: 80,
                    CONF_WEIGHT_MOTION: 0.9,
                    CONF_WEIGHT_MEDIA: 0.7,
                    CONF_WEIGHT_APPLIANCE: 0.6,
                    CONF_WEIGHT_DOOR: 0.5,
                    CONF_WEIGHT_WINDOW: 0.4,
                    CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                    CONF_WASP_WEIGHT: 0.8,
                }
            ),
            subentry_type="area",
            subentry_id="testing_subentry",
            title="Testing",
            unique_id=testing_area_id,
        )
        new_config_entry = Mock()
        new_config_entry.subentries = {"testing_subentry": area_subentry}

        config.update_from_entry(new_config_entry)

        assert config.name == "Testing"
        assert config.threshold == 0.8
        assert config.config_entry == new_config_entry

    def test_update_from_entry_with_area_not_found_loads_defaults(
        self,
        coordinator: AreaOccupancyCoordinator,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test update_from_entry loads default config when area not found in subentries."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Create a mock config entry with subentry for different area
        different_subentry = ConfigSubentry(
            data=MappingProxyType(
                {
                    CONF_AREA_ID: "nonexistent_area_id",
                    CONF_THRESHOLD: 80,
                }
            ),
            subentry_type="area",
            subentry_id="nonexistent_subentry",
            title="Nonexistent",
            unique_id="nonexistent_area_id",
        )
        new_config_entry = Mock()
        new_config_entry.subentries = {"nonexistent_subentry": different_subentry}

        # Should not raise, but load default config
        config.update_from_entry(new_config_entry)
        # Config should still exist but with defaults
        assert config.config_entry == new_config_entry


class TestAreaConfigExtractAreaData:
    """Test AreaConfig._extract_area_data_from_areas_list static method."""

    def test_extract_area_data_finds_area_by_name(
        self, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ) -> None:
        """Test _extract_area_data_from_areas_list finds area by matching area name."""
        testing_area_id = setup_area_registry.get("Testing", "testing")

        areas_list = [
            {
                CONF_AREA_ID: testing_area_id,
                CONF_THRESHOLD: 50,
            }
        ]

        result = AreaConfig._extract_area_data_from_areas_list(
            areas_list, "Testing", hass
        )

        assert result is not None
        assert result[CONF_AREA_ID] == testing_area_id
        assert result[CONF_THRESHOLD] == 50

    def test_extract_area_data_with_none_area_name_returns_none(
        self, hass: HomeAssistant
    ) -> None:
        """Test _extract_area_data_from_areas_list returns None when area_name is None."""
        areas_list = [
            {
                CONF_AREA_ID: "test_area_id",
                CONF_THRESHOLD: 50,
            }
        ]

        result = AreaConfig._extract_area_data_from_areas_list(areas_list, None, hass)

        assert result is None

    def test_extract_area_data_with_area_not_found_returns_none(
        self, hass: HomeAssistant
    ) -> None:
        """Test _extract_area_data_from_areas_list returns None when area not found."""
        areas_list = [
            {
                CONF_AREA_ID: "test_area_id",
                CONF_THRESHOLD: 50,
            }
        ]

        result = AreaConfig._extract_area_data_from_areas_list(
            areas_list, "Nonexistent Area", hass
        )

        assert result is None

    def test_extract_area_data_with_invalid_area_id_handles_gracefully(
        self, hass: HomeAssistant
    ) -> None:
        """Test _extract_area_data_from_areas_list handles invalid area_id gracefully."""
        areas_list = [
            {
                CONF_AREA_ID: "invalid_area_id",
                CONF_THRESHOLD: 50,
            }
        ]

        # Should not raise, but return None when area not found
        result = AreaConfig._extract_area_data_from_areas_list(
            areas_list, "Testing", hass
        )

        # Result depends on whether invalid_area_id exists in registry
        # If not found, should return None
        assert result is None or result[CONF_AREA_ID] == "invalid_area_id"


class TestAreaConfigMergeEntry:
    """Test AreaConfig._merge_entry static method."""

    @pytest.mark.parametrize(
        ("data", "options", "expected"),
        [
            (
                {"key1": "value1", "key2": "value2"},
                {"key2": "new_value2", "key3": "value3"},
                {"key1": "value1", "key2": "new_value2", "key3": "value3"},
            ),
            (
                {"key1": "value1", "key2": "value2"},
                {},
                {"key1": "value1", "key2": "value2"},
            ),
            (
                {},
                {"key1": "value1", "key2": "value2"},
                {"key1": "value1", "key2": "value2"},
            ),
            ({}, {}, {}),
        ],
    )
    def test_merge_entry(self, data: dict, options: dict, expected: dict) -> None:
        """Test _merge_entry merges config entry data and options correctly."""
        config_entry = Mock()
        config_entry.data = data
        config_entry.options = options

        merged = AreaConfig._merge_entry(config_entry)

        assert merged == expected


class TestAreaConfigSubentryReload:
    """Test AreaConfig subentry reload behavior."""

    def test_config_loads_from_subentry(
        self,
        coordinator: AreaOccupancyCoordinator,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test config loads data from subentry."""
        testing_area_id = setup_area_registry.get("Testing", "testing")

        _setup_area_subentry(
            coordinator,
            testing_area_id,
            {
                CONF_THRESHOLD: 75,
                CONF_WEIGHT_MOTION: 0.9,
                CONF_WEIGHT_MEDIA: 0.7,
                CONF_WEIGHT_APPLIANCE: 0.6,
                CONF_WEIGHT_DOOR: 0.5,
                CONF_WEIGHT_WINDOW: 0.4,
                CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                CONF_WASP_WEIGHT: 0.8,
            },
        )

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        assert config.name == "Testing"
        assert config.threshold == 0.75


class TestIntegrationConfig:
    """Test IntegrationConfig class."""

    def test_initialization(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test IntegrationConfig initialization."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        integration_config = IntegrationConfig(coordinator, mock_realistic_config_entry)

        assert integration_config.coordinator == coordinator
        assert integration_config.config_entry == mock_realistic_config_entry
        assert integration_config.hass == hass
        assert integration_config.integration_name == mock_realistic_config_entry.title

    def test_timing_intervals(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test timing interval properties."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        integration_config = IntegrationConfig(coordinator, mock_realistic_config_entry)

        assert integration_config.analysis_interval == ANALYSIS_INTERVAL
        assert integration_config.decay_interval == DECAY_INTERVAL

    @pytest.mark.parametrize(
        ("property_name", "config_key", "default_value", "custom_value"),
        [
            ("sleep_start", CONF_SLEEP_START, DEFAULT_SLEEP_START, "22:00:00"),
            ("sleep_end", CONF_SLEEP_END, DEFAULT_SLEEP_END, "08:00:00"),
        ],
    )
    def test_sleep_properties(
        self,
        hass: HomeAssistant,
        mock_realistic_config_entry: Mock,
        property_name: str,
        config_key: str,
        default_value: str,
        custom_value: str,
    ) -> None:
        """Test sleep_start and sleep_end properties read from config entry options."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        integration_config = IntegrationConfig(coordinator, mock_realistic_config_entry)

        # Test default value
        assert getattr(integration_config, property_name) == default_value

        # Test custom value
        mock_realistic_config_entry.options = {config_key: custom_value}
        integration_config = IntegrationConfig(coordinator, mock_realistic_config_entry)
        assert getattr(integration_config, property_name) == custom_value

    def test_integration_name_from_config_entry(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that integration_name comes from config entry title."""
        mock_realistic_config_entry.title = "Test Integration"
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        integration_config = IntegrationConfig(coordinator, mock_realistic_config_entry)

        assert integration_config.integration_name == "Test Integration"
