"""Tests for data.config module."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_THRESHOLD,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
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
from custom_components.area_occupancy.data.config import (
    Config,
    Decay,
    Sensors,
    SensorStates,
    WaspInBox,
    Weights,
)
from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util


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
        sensors = Sensors(motion=["binary_sensor.motion1"])
        mock_coordinator = Mock()
        mock_coordinator.config.wasp_in_box.enabled = wasp_enabled
        mock_coordinator.wasp_entity_id = wasp_entity_id

        result = sensors.get_motion_sensors(mock_coordinator)
        assert result == expected_result


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
    """Test Config class."""

    def test_initialization_defaults(self, mock_coordinator: Mock) -> None:
        """Test Config initialization with defaults."""
        config = Config(mock_coordinator)

        # Test basic properties
        assert config.name == "Testing"
        assert config.area_id is None  # Not set in the mock data
        assert config.threshold == 0.52  # 52.0 / 100.0 (from options)

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

    def test_initialization_with_values(self, mock_coordinator: Mock) -> None:
        """Test Config initialization with specific values."""
        # Update the mock coordinator's config entry data
        test_data = {
            CONF_NAME: "Living Room",
            CONF_AREA_ID: "living_room",
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
        mock_coordinator.config_entry.data = test_data
        mock_coordinator.config_entry.options = {}  # Clear options to avoid conflicts

        config = Config(mock_coordinator)

        # Test key properties
        assert config.name == "Living Room"
        assert config.area_id == "living_room"
        assert config.threshold == 0.6  # 60 / 100
        assert config.sensors.motion == ["binary_sensor.motion1"]
        assert config.weights.motion == 0.9

    @pytest.mark.parametrize(
        ("property_name", "expected_type", "time_tolerance"),
        [
            ("start_time", "datetime", 60),  # Within 1 minute
            ("end_time", "datetime", 60),  # Within 1 minute
        ],
    )
    def test_time_properties(
        self,
        mock_coordinator: Mock,
        property_name: str,
        expected_type: str,
        time_tolerance: int,
    ) -> None:
        """Test time-related properties."""
        config = Config(mock_coordinator)
        time_value = getattr(config, property_name)

        assert time_value is not None

        if property_name == "start_time":
            expected_time = dt_util.utcnow() - timedelta(days=HA_RECORDER_DAYS)
        else:  # end_time
            expected_time = dt_util.utcnow()

        assert abs((time_value - expected_time).total_seconds()) < time_tolerance

    def test_entity_ids_property(self, mock_coordinator: Mock) -> None:
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
        mock_coordinator.config_entry.data = test_data
        mock_coordinator.config_entry.options = {}

        config = Config(mock_coordinator)
        entity_ids = config.entity_ids

        expected_entities = [
            "binary_sensor.motion1",
            "media_player.tv",
            "switch.computer",
        ]
        for entity_id in expected_entities:
            assert entity_id in entity_ids

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
        mock_coordinator: Mock,
        key: str,
        default: str | None,
        expected_result: str | None,
    ) -> None:
        """Test get method with different parameters."""
        config = Config(mock_coordinator)

        if default is None:
            result = config.get(key)
        else:
            result = config.get(key, default)

        assert result == expected_result

    async def test_update_config(self, mock_coordinator: Mock) -> None:
        """Test update_config method."""
        config = Config(mock_coordinator)
        options = {CONF_NAME: "Updated Name", CONF_THRESHOLD: 70}

        # Mock all the required methods
        with (
            patch.object(
                mock_coordinator.hass.config_entries, "async_update_entry"
            ) as mock_update_entry,
            patch.object(config, "_load_config") as mock_load_config,
            patch.object(mock_coordinator, "async_request_refresh") as mock_refresh,
        ):
            await config.update_config(options)

            # Verify all expected calls were made
            mock_update_entry.assert_called_once()
            mock_load_config.assert_called_once()
            mock_refresh.assert_called_once()


class TestConfigIntegration:
    """Test Config integration scenarios."""

    def test_config_manager_full_lifecycle(self, mock_coordinator: Mock) -> None:
        """Test full config lifecycle."""
        config = Config(mock_coordinator)

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
