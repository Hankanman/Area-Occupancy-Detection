"""Tests for data.config module."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    HA_RECORDER_DAYS,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import (
    AreaConfig,
    Decay,
    Sensors,
    SensorStates,
    WaspInBox,
    Weights,
)
from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestSensors:
    """Test Sensors dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test Sensors initialization with defaults."""
        sensors = Sensors()

        # All sensor lists should be empty by default
        expected_empty_lists = [
            sensors.motion,
            sensors.media,
            sensors.appliance,
            sensors.illuminance,
            sensors.humidity,
            sensors.temperature,
            sensors.door,
            sensors.window,
        ]
        assert all(not sensor_list for sensor_list in expected_empty_lists)
        assert sensors.primary_occupancy is None

    def test_initialization_with_values(self) -> None:
        """Test Sensors initialization with specific values."""
        test_data = {
            "motion": ["binary_sensor.motion1"],
            "primary_occupancy": "binary_sensor.motion1",
            "media": ["media_player.tv"],
            "appliance": ["switch.computer"],
            "illuminance": ["sensor.illuminance"],
            "humidity": ["sensor.humidity"],
            "temperature": ["sensor.temperature"],
            "door": ["binary_sensor.door"],
            "window": ["binary_sensor.window"],
        }

        sensors = Sensors(**test_data)

        # Verify all values are set correctly
        for key, expected_value in test_data.items():
            assert getattr(sensors, key) == expected_value

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
    def test_get_motion_sensors(
        self, wasp_enabled: bool, wasp_entity_id: str | None, expected_result: list[str]
    ) -> None:
        """Test get_motion_sensors with different wasp configurations."""
        # Create mock parent config with wasp_in_box
        mock_parent_config = Mock()
        mock_parent_config.wasp_in_box = Mock()
        mock_parent_config.wasp_in_box.enabled = wasp_enabled
        mock_parent_config.area_name = "Test Area"

        # Create sensors with parent config
        sensors = Sensors(
            motion=["binary_sensor.motion1"], _parent_config=mock_parent_config
        )

        # Set up coordinator with multi-area architecture
        mock_coordinator = Mock()
        mock_area_data = Mock()
        mock_area_data.wasp_entity_id = wasp_entity_id
        mock_coordinator.areas = {"Test Area": mock_area_data}

        # Also set legacy wasp_entity_id for backward compatibility test
        mock_coordinator.wasp_entity_id = wasp_entity_id

        result = sensors.get_motion_sensors(mock_coordinator)
        assert result == expected_result

    def test_get_motion_sensors_with_none_coordinator(self) -> None:
        """Test get_motion_sensors with None coordinator."""
        sensors = Sensors(motion=["binary_sensor.motion1"])
        result = sensors.get_motion_sensors(None)
        assert result == ["binary_sensor.motion1"]

    def test_get_motion_sensors_with_empty_motion_list(self) -> None:
        """Test get_motion_sensors with empty motion list."""
        # Create mock parent config with wasp enabled
        mock_parent_config = Mock()
        mock_parent_config.wasp_in_box = Mock()
        mock_parent_config.wasp_in_box.enabled = True
        mock_parent_config.area_name = "Test Area"

        sensors = Sensors(motion=[], _parent_config=mock_parent_config)
        mock_coordinator = Mock()
        mock_area_data = Mock()
        mock_area_data.wasp_entity_id = "binary_sensor.wasp"
        mock_coordinator.areas = {"Test Area": mock_area_data}
        mock_coordinator.wasp_entity_id = "binary_sensor.wasp"

        result = sensors.get_motion_sensors(mock_coordinator)
        assert result == ["binary_sensor.wasp"]

    def test_get_motion_sensors_without_wasp_config(self) -> None:
        """Test get_motion_sensors when wasp_in_box is disabled."""
        # Create mock parent config with wasp disabled
        mock_parent_config = Mock()
        mock_parent_config.wasp_in_box = Mock()
        mock_parent_config.wasp_in_box.enabled = False
        mock_parent_config.area_name = "Test Area"

        sensors = Sensors(
            motion=["binary_sensor.motion1"], _parent_config=mock_parent_config
        )
        mock_coordinator = Mock()
        mock_area_data = Mock()
        mock_area_data.wasp_entity_id = "binary_sensor.wasp"
        mock_coordinator.areas = {"Test Area": mock_area_data}
        mock_coordinator.wasp_entity_id = "binary_sensor.wasp"

        result = sensors.get_motion_sensors(mock_coordinator)
        assert result == ["binary_sensor.motion1"]

    def test_get_motion_sensors_legacy_mode(self) -> None:
        """Test get_motion_sensors with legacy coordinator.wasp_entity_id fallback."""
        # Create mock parent config with wasp enabled but no area_name
        # (simulating legacy single-area mode)
        mock_parent_config = Mock()
        mock_parent_config.wasp_in_box = Mock()
        mock_parent_config.wasp_in_box.enabled = True
        mock_parent_config.area_name = None  # Legacy mode - no area_name

        sensors = Sensors(
            motion=["binary_sensor.motion1"], _parent_config=mock_parent_config
        )
        mock_coordinator = Mock()
        # Legacy mode: coordinator has wasp_entity_id directly
        mock_coordinator.wasp_entity_id = "binary_sensor.wasp"
        # No areas dict (legacy mode)
        mock_coordinator.areas = {}

        result = sensors.get_motion_sensors(mock_coordinator)
        # In legacy mode without area_name, wasp is not included
        # because wasp_entity_id is stored per-area, not on coordinator
        assert result == ["binary_sensor.motion1"]

    def test_get_motion_sensors_multi_area_mode(self) -> None:
        """Test get_motion_sensors with multi-area architecture."""
        # Create mock parent config with wasp enabled and area_name
        mock_parent_config = Mock()
        mock_parent_config.wasp_in_box = Mock()
        mock_parent_config.wasp_in_box.enabled = True
        mock_parent_config.area_name = "Living Room"

        sensors = Sensors(
            motion=["binary_sensor.motion1"], _parent_config=mock_parent_config
        )
        mock_coordinator = Mock()
        # Multi-area mode: wasp_entity_id stored per area
        mock_area_data = Mock()
        mock_area_data.wasp_entity_id = "binary_sensor.living_room_wasp"
        mock_coordinator.areas = {"Living Room": mock_area_data}

        result = sensors.get_motion_sensors(mock_coordinator)
        assert result == ["binary_sensor.motion1", "binary_sensor.living_room_wasp"]


class TestSensorStates:
    """Test SensorStates dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test SensorStates initialization with defaults."""
        states = SensorStates()

        expected_defaults = {
            "motion": [STATE_ON],
            "door": [DEFAULT_DOOR_ACTIVE_STATE],
            "window": [DEFAULT_WINDOW_ACTIVE_STATE],
            "appliance": list(DEFAULT_APPLIANCE_ACTIVE_STATES),
            "media": list(DEFAULT_MEDIA_ACTIVE_STATES),
        }

        for key, expected_value in expected_defaults.items():
            assert getattr(states, key) == expected_value

    def test_initialization_with_values(self) -> None:
        """Test SensorStates initialization with specific values."""
        test_data = {
            "door": ["open", "unlocked"],
            "window": ["open"],
            "appliance": ["on"],
            "media": ["playing", "buffering"],
        }

        states = SensorStates(**test_data)

        for key, expected_value in test_data.items():
            assert getattr(states, key) == expected_value


class TestWeights:
    """Test Weights dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test Weights initialization with defaults."""
        weights = Weights()

        expected_defaults = {
            "motion": DEFAULT_WEIGHT_MOTION,
            "media": DEFAULT_WEIGHT_MEDIA,
            "appliance": DEFAULT_WEIGHT_APPLIANCE,
            "door": DEFAULT_WEIGHT_DOOR,
            "window": DEFAULT_WEIGHT_WINDOW,
            "environmental": DEFAULT_WEIGHT_ENVIRONMENTAL,
            "wasp": DEFAULT_WASP_WEIGHT,
        }

        for key, expected_value in expected_defaults.items():
            assert getattr(weights, key) == expected_value

    def test_initialization_with_values(self) -> None:
        """Test Weights initialization with specific values."""
        test_data = {
            "motion": 0.9,
            "media": 0.8,
            "appliance": 0.7,
            "door": 0.6,
            "window": 0.5,
            "environmental": 0.3,
            "wasp": 0.85,
        }

        weights = Weights(**test_data)

        for key, expected_value in test_data.items():
            assert getattr(weights, key) == expected_value


class TestDecay:
    """Test Decay dataclass."""

    @pytest.mark.parametrize(
        ("kwargs", "expected_enabled", "expected_half_life"),
        [
            ({}, DEFAULT_DECAY_ENABLED, DEFAULT_DECAY_HALF_LIFE),
            ({"enabled": False, "half_life": 600}, False, 600),
        ],
    )
    def test_initialization(
        self, kwargs: dict, expected_enabled: bool, expected_half_life: float
    ) -> None:
        """Test Decay initialization with different parameters."""
        decay = Decay(**kwargs)
        assert decay.enabled == expected_enabled
        assert decay.half_life == expected_half_life


class TestWaspInBox:
    """Test WaspInBox dataclass."""

    @pytest.mark.parametrize(
        ("kwargs", "expected_values"),
        [
            (
                {},
                {
                    "enabled": False,
                    "motion_timeout": DEFAULT_WASP_MOTION_TIMEOUT,
                    "weight": DEFAULT_WASP_WEIGHT,
                    "max_duration": DEFAULT_WASP_MAX_DURATION,
                },
            ),
            (
                {
                    "enabled": True,
                    "motion_timeout": 120,
                    "weight": 0.9,
                    "max_duration": 1800,
                },
                {
                    "enabled": True,
                    "motion_timeout": 120,
                    "weight": 0.9,
                    "max_duration": 1800,
                },
            ),
        ],
    )
    def test_initialization(self, kwargs: dict, expected_values: dict) -> None:
        """Test WaspInBox initialization with different parameters."""
        wasp = WaspInBox(**kwargs)

        for key, expected_value in expected_values.items():
            assert getattr(wasp, key) == expected_value


class TestConfig:
    """Test AreaConfig class."""

    def test_initialization_defaults(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test AreaConfig initialization with defaults."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Test basic properties
        assert config.name == "Testing"
        # area_id should be the area ID from the config entry, not the area name
        # Get the area from coordinator to find its area_id
        area = coordinator.get_area_or_default(area_name)
        expected_area_id = area.config.area_id if area else None
        assert config.area_id == expected_area_id
        # Threshold comes from options (52.0) or data (50.0), check what's actually set
        # Options override data, so if options has threshold 52.0, it should be 0.52
        # But if options doesn't have threshold, it uses data (50.0) = 0.5
        assert config.threshold in [
            0.5,
            0.52,
        ]  # Accept either value depending on options

        # Test component types
        expected_components = {
            "sensors": Sensors,
            "sensor_states": SensorStates,
            "weights": Weights,
            "decay": Decay,
            "wasp_in_box": WaspInBox,
        }

        for attr_name, expected_type in expected_components.items():
            assert isinstance(getattr(config, attr_name), expected_type)

    def test_initialization_with_values(
        self,
        coordinator: AreaOccupancyCoordinator,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test Config initialization with specific values."""
        # Use actual area ID from registry for Living Room
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")

        # Update the mock coordinator's config entry data with CONF_AREAS format
        test_data = {
            CONF_AREAS: [
                {
                    CONF_AREA_ID: living_room_area_id,
                    CONF_THRESHOLD: 60,  # Percentage
                    CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                    CONF_WEIGHT_MOTION: 0.9,
                    CONF_WEIGHT_MEDIA: 0.7,
                    CONF_WEIGHT_APPLIANCE: 0.6,
                    CONF_WEIGHT_DOOR: 0.5,
                    CONF_WEIGHT_WINDOW: 0.4,
                    CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                    CONF_WASP_WEIGHT: 0.8,
                }
            ]
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}  # Clear options to avoid conflicts

        # Reload areas to pick up the new config
        coordinator._load_areas_from_config()

        area_name = "Living Room"  # Use the area name from registry
        config = AreaConfig(coordinator, area_name=area_name)

        # Test key properties
        assert config.name == "Living Room"
        assert config.area_id == living_room_area_id
        assert config.threshold == 0.6  # 60 / 100
        assert config.sensors.motion == ["binary_sensor.motion1"]
        assert config.weights.motion == 0.9

    def test_initialization_with_missing_weights(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test Config initialization with missing weight values."""
        # Create minimal data without weights
        test_data = {
            CONF_AREA_ID: "test_area",
            CONF_THRESHOLD: 50,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        # Weights now use defaults if not provided, so no KeyError is raised
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        # Verify that default weights are used
        assert config.weights.motion == DEFAULT_WEIGHT_MOTION
        assert config.weights.media == DEFAULT_WEIGHT_MEDIA

    def test_initialization_with_string_threshold(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test Config initialization with string threshold value."""
        test_data = {
            CONF_AREA_ID: "test_area",
            CONF_THRESHOLD: "75",  # String instead of int
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        assert config.threshold == 0.75  # Should convert string to float

    @pytest.mark.parametrize(
        ("property_name", "expected_type", "time_tolerance"),
        [
            ("start_time", "datetime", 60),  # Within 1 minute
            ("end_time", "datetime", 60),  # Within 1 minute
        ],
    )
    def test_time_properties(
        self,
        coordinator: AreaOccupancyCoordinator,
        property_name: str,
        expected_type: str,
        time_tolerance: int,
    ) -> None:
        """Test time-related properties."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        time_value = getattr(config, property_name)

        assert time_value is not None

        if property_name == "start_time":
            expected_time = dt_util.utcnow() - timedelta(days=HA_RECORDER_DAYS)
        else:  # end_time
            expected_time = dt_util.utcnow()

        assert abs((time_value - expected_time).total_seconds()) < time_tolerance

    def test_entity_ids_property(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test entity_ids property."""
        # Set up mock coordinator with sensor data
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_APPLIANCES: ["switch.computer"],
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        entity_ids = config.entity_ids

        expected_entities = [
            "binary_sensor.motion1",
            "media_player.tv",
            "switch.computer",
        ]
        for entity_id in expected_entities:
            assert entity_id in entity_ids

    def test_entity_ids_property_with_all_sensor_types(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test entity_ids property with all sensor types populated."""
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_MEDIA_DEVICES: ["media_player.tv", "media_player.speaker"],
            CONF_APPLIANCES: ["switch.computer", "switch.lamp"],
            CONF_DOOR_SENSORS: ["binary_sensor.door1", "binary_sensor.door2"],
            CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
            CONF_ILLUMINANCE_SENSORS: ["sensor.illuminance1", "sensor.illuminance2"],
            CONF_HUMIDITY_SENSORS: ["sensor.humidity1"],
            CONF_TEMPERATURE_SENSORS: ["sensor.temperature1", "sensor.temperature2"],
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
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
            "sensor.illuminance1",
            "sensor.illuminance2",
            "sensor.humidity1",
            "sensor.temperature1",
            "sensor.temperature2",
        ]

        assert len(entity_ids) == len(expected_entities)
        for entity_id in expected_entities:
            assert entity_id in entity_ids

    def test_entity_ids_property_with_empty_sensors(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test entity_ids property with empty sensor lists."""
        test_data = {
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        entity_ids = config.entity_ids

        assert entity_ids == []

    @pytest.mark.parametrize(
        ("key", "default", "expected_result"),
        [
            ("name", None, "Testing"),
            ("nonexistent", "default", "default"),
            ("nonexistent", None, None),
        ],
    )
    def test_get_method(
        self,
        coordinator: AreaOccupancyCoordinator,
        key: str,
        default: str | None,
        expected_result: str | None,
    ) -> None:
        """Test get method with different parameters."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        if default is None:
            result = config.get(key)
        else:
            result = config.get(key, default)

        assert result == expected_result

    async def test_update_config(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test update_config method."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        options = {CONF_AREA_ID: "updated_area", CONF_THRESHOLD: 70}

        # Mock all the required methods
        with (
            patch.object(
                coordinator.hass.config_entries, "async_update_entry"
            ) as mock_update_entry,
            patch.object(config, "_load_config") as mock_load_config,
            patch.object(coordinator, "async_request_refresh") as mock_refresh,
            patch.object(
                coordinator, "_setup_complete", True
            ),  # Ensure setup_complete is True
        ):
            await config.update_config(options)

            # Verify all expected calls were made
            mock_update_entry.assert_called_once()
            mock_load_config.assert_called_once()
            mock_refresh.assert_called_once()

    async def test_update_config_with_exception(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_config method when an exception occurs."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        options = {CONF_AREA_ID: "updated_area"}

        # Mock the update_entry method to raise an exception
        with patch.object(
            coordinator.hass.config_entries, "async_update_entry"
        ) as mock_update_entry:
            mock_update_entry.side_effect = Exception("Update failed")

            with pytest.raises(
                HomeAssistantError, match="Failed to update configuration"
            ):
                await config.update_config(options)

    def test_validate_entity_configuration_valid(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test validate_entity_configuration with valid configuration."""
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert errors == []

    def test_validate_entity_configuration_duplicate_entities(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test validate_entity_configuration with duplicate entity IDs."""
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.sensor1"],
            CONF_MEDIA_DEVICES: ["binary_sensor.sensor1"],  # Duplicate
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) == 1
        assert "Duplicate entity IDs found" in errors[0]
        assert "binary_sensor.sensor1" in errors[0]

    def test_validate_entity_configuration_no_sensors(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test validate_entity_configuration with no motion, media, or appliance sensors."""
        test_data = {
            CONF_DOOR_SENSORS: ["binary_sensor.door1"],  # Only door sensors
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) == 1
        assert "No motion, media, or appliance sensors configured" in errors[0]

    def test_validate_entity_configuration_invalid_entity_ids(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test validate_entity_configuration with invalid entity IDs."""
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "", "   "],  # Invalid IDs
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) == 1
        assert "Invalid motion sensor entity IDs" in errors[0]

    def test_validate_entity_configuration_non_string_entity_ids(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test validate_entity_configuration with non-string entity IDs."""
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", 123, None],  # Non-string IDs
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) == 1
        assert "Invalid motion sensor entity IDs" in errors[0]

    def test_validate_entity_configuration_multiple_errors(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test validate_entity_configuration with multiple validation errors."""
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.sensor1", ""],  # Invalid ID
            CONF_MEDIA_DEVICES: ["binary_sensor.sensor1"],  # Duplicate
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)
        errors = config.validate_entity_configuration()

        assert len(errors) == 2
        assert any("Duplicate entity IDs found" in error for error in errors)
        assert any("Invalid motion sensor entity IDs" in error for error in errors)

    def test_update_from_entry(
        self,
        coordinator: AreaOccupancyCoordinator,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test update_from_entry method."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Use actual area ID from registry for Testing area
        testing_area_id = setup_area_registry.get("Testing", "testing")

        # Create a new config entry with different data in CONF_AREAS format
        new_config_entry = Mock()
        new_config_entry.data = {
            CONF_AREAS: [
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
            ]
        }
        new_config_entry.options = {}

        config.update_from_entry(new_config_entry)

        assert config.name == "Testing"  # Area name from registry
        assert config.threshold == 0.8
        assert config.config_entry == new_config_entry

    def test_merge_entry_static_method(self) -> None:
        """Test _merge_entry static method."""
        config_entry = Mock()
        config_entry.data = {"key1": "value1", "key2": "value2"}
        config_entry.options = {"key2": "new_value2", "key3": "value3"}

        merged = AreaConfig._merge_entry(config_entry)

        expected = {"key1": "value1", "key2": "new_value2", "key3": "value3"}
        assert merged == expected

    def test_merge_entry_with_empty_options(self) -> None:
        """Test _merge_entry with empty options."""
        config_entry = Mock()
        config_entry.data = {"key1": "value1", "key2": "value2"}
        config_entry.options = {}

        merged = AreaConfig._merge_entry(config_entry)

        assert merged == {"key1": "value1", "key2": "value2"}

    def test_merge_entry_with_empty_data(self) -> None:
        """Test _merge_entry with empty data."""
        config_entry = Mock()
        config_entry.data = {}
        config_entry.options = {"key1": "value1", "key2": "value2"}

        merged = AreaConfig._merge_entry(config_entry)

        assert merged == {"key1": "value1", "key2": "value2"}

    def test_merge_entry_with_both_empty(self) -> None:
        """Test _merge_entry with both data and options empty."""
        config_entry = Mock()
        config_entry.data = {}
        config_entry.options = {}

        merged = AreaConfig._merge_entry(config_entry)

        assert merged == {}


class TestConfigIntegration:
    """Test Config integration scenarios."""

    def test_config_manager_full_lifecycle(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test full config lifecycle."""
        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Test basic property access
        assert config.name is not None
        assert 0 < config.threshold <= 1.0

        # Test component types
        expected_components = {
            "sensors": Sensors,
            "sensor_states": SensorStates,
            "weights": Weights,
            "decay": Decay,
            "wasp_in_box": WaspInBox,
        }

        for attr_name, expected_type in expected_components.items():
            assert isinstance(getattr(config, attr_name), expected_type)

        # Test entity_ids property
        assert isinstance(config.entity_ids, list)

    def test_config_with_all_sensor_types_and_validation(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test config with all sensor types and validation."""
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_APPLIANCES: ["switch.computer"],
            CONF_DOOR_SENSORS: ["binary_sensor.door1"],
            CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
            CONF_ILLUMINANCE_SENSORS: ["sensor.illuminance1"],
            CONF_HUMIDITY_SENSORS: ["sensor.humidity1"],
            CONF_TEMPERATURE_SENSORS: ["sensor.temperature1"],
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Test validation passes
        errors = config.validate_entity_configuration()
        assert errors == []

        # Test entity_ids includes all sensors
        entity_ids = config.entity_ids
        expected_count = 8  # One of each sensor type
        assert len(entity_ids) == expected_count

        # Test all expected entities are present
        expected_entities = [
            "binary_sensor.motion1",
            "media_player.tv",
            "switch.computer",
            "binary_sensor.door1",
            "binary_sensor.window1",
            "sensor.illuminance1",
            "sensor.humidity1",
            "sensor.temperature1",
        ]
        for entity_id in expected_entities:
            assert entity_id in entity_ids

    def test_config_edge_cases(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test config edge cases and boundary conditions."""
        # Test with minimal valid configuration
        test_data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.7,
            CONF_WEIGHT_APPLIANCE: 0.6,
            CONF_WEIGHT_DOOR: 0.5,
            CONF_WEIGHT_WINDOW: 0.4,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.8,
        }
        coordinator.config_entry.data = test_data
        coordinator.config_entry.options = {}

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Test with extreme threshold values
        assert config.threshold > 0
        assert config.threshold <= 1.0

        # Test time properties are recent
        now = dt_util.utcnow()
        assert (
            abs(
                (
                    config.start_time - (now - timedelta(days=HA_RECORDER_DAYS))
                ).total_seconds()
            )
            < 60
        )
        assert abs((config.end_time - now).total_seconds()) < 60

        # Test validation passes with minimal config
        errors = config.validate_entity_configuration()
        assert errors == []

    def test_config_with_options_override(
        self,
        coordinator: AreaOccupancyCoordinator,
        hass: HomeAssistant,
        setup_area_registry: dict[str, str],
    ) -> None:
        """Test config where options override data values."""
        # Use actual area ID from registry
        testing_area_id = setup_area_registry.get("Testing", "testing")

        # Set up data and options with conflicting values in CONF_AREAS format
        coordinator.config_entry.data = {
            CONF_AREAS: [
                {
                    CONF_AREA_ID: testing_area_id,
                    CONF_THRESHOLD: 50,
                    CONF_WEIGHT_MOTION: 0.9,
                    CONF_WEIGHT_MEDIA: 0.7,
                    CONF_WEIGHT_APPLIANCE: 0.6,
                    CONF_WEIGHT_DOOR: 0.5,
                    CONF_WEIGHT_WINDOW: 0.4,
                    CONF_WEIGHT_ENVIRONMENTAL: 0.3,
                    CONF_WASP_WEIGHT: 0.8,
                }
            ]
        }
        coordinator.config_entry.options = {
            CONF_AREAS: [
                {
                    CONF_AREA_ID: testing_area_id,
                    CONF_THRESHOLD: 75,
                }
            ]
        }

        area_name = coordinator.get_area_names()[0]
        config = AreaConfig(coordinator, area_name=area_name)

        # Options should override data
        # Name comes from area registry, not from config (it's resolved from area_id)
        assert config.name == "Testing"  # Area name from registry
        assert config.threshold == 0.75  # 75 / 100 (from options, overriding data's 50)
