"""Tests for data.config module."""

from datetime import timedelta
from unittest.mock import Mock

from custom_components.area_occupancy.const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WASP_MAX_DURATION,
    CONF_WASP_MOTION_TIMEOUT,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
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
from homeassistant.util import dt as dt_util


class TestSensors:
    """Test Sensors dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test Sensors initialization with defaults."""
        sensors = Sensors()

        assert sensors.motion == []
        assert sensors.primary_occupancy is None
        assert sensors.media == []
        assert sensors.appliance == []
        assert sensors.illuminance == []
        assert sensors.humidity == []
        assert sensors.temperature == []
        assert sensors.door == []
        assert sensors.window == []

    def test_initialization_with_values(self) -> None:
        """Test Sensors initialization with specific values."""
        sensors = Sensors(
            motion=["binary_sensor.motion1", "binary_sensor.motion2"],
            primary_occupancy="binary_sensor.motion1",
            media=["media_player.tv"],
            appliance=["switch.coffee_maker"],
            illuminance=["sensor.illuminance"],
            humidity=["sensor.humidity"],
            temperature=["sensor.temperature"],
            door=["binary_sensor.door"],
            window=["binary_sensor.window"],
        )

        assert sensors.motion == ["binary_sensor.motion1", "binary_sensor.motion2"]
        assert sensors.primary_occupancy == "binary_sensor.motion1"
        assert sensors.media == ["media_player.tv"]
        assert sensors.appliance == ["switch.coffee_maker"]
        assert sensors.illuminance == ["sensor.illuminance"]
        assert sensors.humidity == ["sensor.humidity"]
        assert sensors.temperature == ["sensor.temperature"]
        assert sensors.door == ["binary_sensor.door"]
        assert sensors.window == ["binary_sensor.window"]

    def test_get_motion_sensors_without_wasp(self) -> None:
        """Test get_motion_sensors without wasp enabled."""
        sensors = Sensors(motion=["binary_sensor.motion1", "binary_sensor.motion2"])

        # Mock coordinator without wasp
        mock_coordinator = Mock()
        mock_coordinator.config.wasp_in_box.enabled = False
        mock_coordinator.wasp_entity_id = None

        result = sensors.get_motion_sensors(mock_coordinator)

        assert result == ["binary_sensor.motion1", "binary_sensor.motion2"]

    def test_get_motion_sensors_with_wasp_enabled(self) -> None:
        """Test get_motion_sensors with wasp enabled and available."""
        sensors = Sensors(motion=["binary_sensor.motion1"])

        # Mock coordinator with wasp enabled
        mock_coordinator = Mock()
        mock_coordinator.config.wasp_in_box.enabled = True
        mock_coordinator.wasp_entity_id = "binary_sensor.wasp_box"

        result = sensors.get_motion_sensors(mock_coordinator)

        assert result == ["binary_sensor.motion1", "binary_sensor.wasp_box"]

    def test_get_motion_sensors_with_wasp_enabled_no_entity(self) -> None:
        """Test get_motion_sensors with wasp enabled but no entity."""
        sensors = Sensors(motion=["binary_sensor.motion1"])

        # Mock coordinator with wasp enabled but no entity_id
        mock_coordinator = Mock()
        mock_coordinator.config.wasp_in_box.enabled = True
        mock_coordinator.wasp_entity_id = None

        result = sensors.get_motion_sensors(mock_coordinator)

        assert result == ["binary_sensor.motion1"]  # Should not include wasp


class TestSensorStates:
    """Test SensorStates dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test SensorStates initialization with defaults."""
        states = SensorStates()

        assert states.door == [DEFAULT_DOOR_ACTIVE_STATE]
        assert states.window == [DEFAULT_WINDOW_ACTIVE_STATE]
        assert states.appliance == list(DEFAULT_APPLIANCE_ACTIVE_STATES)
        assert states.media == list(DEFAULT_MEDIA_ACTIVE_STATES)

    def test_initialization_with_values(self) -> None:
        """Test SensorStates initialization with specific values."""
        states = SensorStates(
            door=["open", "unlocked"],
            window=["open"],
            appliance=["on"],
            media=["playing", "buffering"],
        )

        assert states.door == ["open", "unlocked"]
        assert states.window == ["open"]
        assert states.appliance == ["on"]
        assert states.media == ["playing", "buffering"]


class TestWeights:
    """Test Weights dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test Weights initialization with defaults."""
        weights = Weights()

        assert weights.motion == DEFAULT_WEIGHT_MOTION
        assert weights.media == DEFAULT_WEIGHT_MEDIA
        assert weights.appliance == DEFAULT_WEIGHT_APPLIANCE
        assert weights.door == DEFAULT_WEIGHT_DOOR
        assert weights.window == DEFAULT_WEIGHT_WINDOW
        assert weights.environmental == DEFAULT_WEIGHT_ENVIRONMENTAL
        assert weights.wasp == DEFAULT_WASP_WEIGHT

    def test_initialization_with_values(self) -> None:
        """Test Weights initialization with specific values."""
        weights = Weights(
            motion=0.9,
            media=0.8,
            appliance=0.7,
            door=0.6,
            window=0.5,
            environmental=0.3,
            wasp=0.85,
        )

        assert weights.motion == 0.9
        assert weights.media == 0.8
        assert weights.appliance == 0.7
        assert weights.door == 0.6
        assert weights.window == 0.5
        assert weights.environmental == 0.3
        assert weights.wasp == 0.85


class TestDecay:
    """Test Decay dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test Decay initialization with defaults."""
        decay = Decay()

        assert decay.enabled == DEFAULT_DECAY_ENABLED
        assert decay.half_life == DEFAULT_DECAY_HALF_LIFE

    def test_initialization_with_values(self) -> None:
        """Test Decay initialization with specific values."""
        decay = Decay(enabled=False, half_life=600)

        assert decay.enabled is False
        assert decay.half_life == 600


class TestConfig:
    """Test Config dataclass."""

    def test_initialization_defaults(self, mock_coordinator: Mock) -> None:
        """Test Config initialization with defaults."""
        config = Config(mock_coordinator)

        assert config.name == "Testing"
        assert config.area_id is None  # Not set in the mock data
        assert config.threshold == 0.52  # 52.0 / 100.0 (from options)
        assert isinstance(config.sensors, Sensors)
        assert isinstance(config.sensor_states, SensorStates)
        assert isinstance(config.weights, Weights)
        assert isinstance(config.decay, Decay)
        assert isinstance(config.wasp_in_box, WaspInBox)

    def test_initialization_with_values(self, mock_coordinator: Mock) -> None:
        """Test Config initialization with specific values."""
        # Since Config now loads from coordinator, we need to mock the coordinator's config
        mock_coordinator.config_entry.data = {
            CONF_NAME: "Living Room",
            CONF_AREA_ID: "living_room",
            CONF_THRESHOLD: 60,  # Percentage
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_WEIGHT_MOTION: 0.9,
        }
        # Clear options to avoid conflicts
        mock_coordinator.config_entry.options = {}

        config = Config(mock_coordinator)

        assert config.name == "Living Room"
        assert config.area_id == "living_room"
        assert config.threshold == 0.6  # Converted from percentage
        assert config.sensors.motion == ["binary_sensor.motion1"]
        assert config.weights.motion == 0.9

    def test_start_time_property(self, mock_config: Mock) -> None:
        """Test start_time property calculation."""
        start_time = mock_config.start_time
        expected_start = dt_util.utcnow() - timedelta(days=HA_RECORDER_DAYS)

        # Allow some tolerance for test execution time
        assert abs((start_time - expected_start).total_seconds()) < 5

    def test_end_time_property(self, mock_config: Mock) -> None:
        """Test end_time property calculation."""
        end_time = mock_config.end_time
        expected_end = dt_util.utcnow()

        # Allow some tolerance for test execution time
        assert abs((end_time - expected_end).total_seconds()) < 5

    def test_from_dict_minimal(self, mock_coordinator: Mock) -> None:
        """Test Config initialization with minimal data."""
        # Set up the coordinator's config_entry data
        mock_coordinator.config_entry.data = {
            CONF_NAME: "Test Area",
            CONF_THRESHOLD: 50,  # Percentage
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        # Clear options to avoid conflicts
        mock_coordinator.config_entry.options = {}

        config = Config(mock_coordinator)

        assert config.name == "Test Area"
        assert config.threshold == 0.5  # Converted from percentage
        assert config.sensors.motion == ["binary_sensor.motion1"]

    def test_from_dict_comprehensive(self, mock_coordinator: Mock) -> None:
        """Test Config initialization with comprehensive data."""
        # Set up the coordinator's config_entry data
        mock_coordinator.config_entry.data = {
            CONF_NAME: "Living Room",
            CONF_AREA_ID: "living_room",
            CONF_THRESHOLD: 60,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_APPLIANCES: ["switch.coffee_maker"],
            CONF_ILLUMINANCE_SENSORS: ["sensor.illuminance"],
            CONF_HUMIDITY_SENSORS: ["sensor.humidity"],
            CONF_TEMPERATURE_SENSORS: ["sensor.temperature"],
            CONF_DOOR_SENSORS: ["binary_sensor.door"],
            CONF_WINDOW_SENSORS: ["binary_sensor.window"],
            CONF_DOOR_ACTIVE_STATE: "open",
            CONF_WINDOW_ACTIVE_STATE: "open",
            CONF_APPLIANCE_ACTIVE_STATES: ["on"],
            CONF_MEDIA_ACTIVE_STATES: ["playing"],
            CONF_WEIGHT_MOTION: 0.9,
            CONF_WEIGHT_MEDIA: 0.8,
            CONF_WEIGHT_APPLIANCE: 0.7,
            CONF_WEIGHT_DOOR: 0.6,
            CONF_WEIGHT_WINDOW: 0.5,
            CONF_WEIGHT_ENVIRONMENTAL: 0.3,
            CONF_WASP_WEIGHT: 0.85,
            CONF_DECAY_ENABLED: False,
            CONF_DECAY_HALF_LIFE: 600,
            CONF_WASP_ENABLED: True,
            CONF_WASP_MOTION_TIMEOUT: 30,
            CONF_WASP_MAX_DURATION: 7200,
        }
        # Clear options to avoid conflicts
        mock_coordinator.config_entry.options = {}

        config = Config(mock_coordinator)

        # Test all values
        assert config.name == "Living Room"
        assert config.area_id == "living_room"
        assert config.threshold == 0.6
        assert config.sensors.motion == [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        assert config.sensors.primary_occupancy == "binary_sensor.motion1"
        assert config.sensors.media == ["media_player.tv"]
        assert config.sensors.appliance == ["switch.coffee_maker"]
        assert config.sensors.illuminance == ["sensor.illuminance"]
        assert config.sensors.humidity == ["sensor.humidity"]
        assert config.sensors.temperature == ["sensor.temperature"]
        assert config.sensors.door == ["binary_sensor.door"]
        assert config.sensors.window == ["binary_sensor.window"]
        assert config.sensor_states.door == ["open"]
        assert config.sensor_states.window == ["open"]
        assert config.sensor_states.appliance == ["on"]
        assert config.sensor_states.media == ["playing"]
        assert config.weights.motion == 0.9
        assert config.weights.media == 0.8
        assert config.weights.appliance == 0.7
        assert config.weights.door == 0.6
        assert config.weights.window == 0.5
        assert config.weights.environmental == 0.3
        assert config.weights.wasp == 0.85
        assert config.decay.enabled is False
        assert config.decay.half_life == 600
        assert config.wasp_in_box.enabled is True
        assert config.wasp_in_box.motion_timeout == 30
        assert config.wasp_in_box.max_duration == 7200

    def test_from_dict_with_invalid_weights(self, mock_coordinator: Mock) -> None:
        """Test Config initialization with invalid weight values."""
        # Set up the coordinator's config_entry data
        mock_coordinator.config_entry.data = {
            CONF_NAME: "Test Area",
            CONF_THRESHOLD: 50,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_WEIGHT_MOTION: -0.5,  # Invalid negative weight
            CONF_WEIGHT_MEDIA: 1.5,  # Large weight (allowed)
        }
        # Clear options to avoid conflicts
        mock_coordinator.config_entry.options = {}

        config = Config(mock_coordinator)

        # Only negative weights should be replaced with defaults
        assert config.weights.motion == DEFAULT_WEIGHT_MOTION
        assert config.weights.media == 1.5  # Large weights are allowed

    def test_get_method(self, mock_coordinator: Mock) -> None:
        """Test Config.get method."""
        config = Config(mock_coordinator)

        # Test getting existing attributes
        assert config.get("name") == "Testing"
        assert config.get("threshold") == 0.52

        # Test getting non-existent attributes with default
        assert config.get("non_existent", "default") == "default"
        assert config.get("another_missing") is None

    async def test_update_config(self, mock_coordinator: Mock) -> None:
        """Test update_config method."""
        config = Config(mock_coordinator)

        # Initial config
        assert config.decay.enabled == DEFAULT_DECAY_ENABLED

        # Update with new options
        new_options = {CONF_DECAY_ENABLED: False, CONF_THRESHOLD: 80}

        await config.update_config(new_options)

        # Should update the config
        assert config.decay.enabled is False
        assert config.threshold == 0.8  # 80 / 100 (converted from percentage)


class TestConfigIntegration:
    """Test Config integration scenarios."""

    def test_config_manager_full_lifecycle(self, mock_coordinator: Mock) -> None:
        """Test Config through a complete lifecycle."""
        # Create manager
        manager = Config(mock_coordinator)

        # Initial config access - Config doesn't have a 'config' attribute
        # Instead, test the actual properties that exist
        assert hasattr(manager, "name")
        assert hasattr(manager, "threshold")
        assert hasattr(manager, "sensors")
        assert hasattr(manager, "entity_ids")

        # Test that the config was properly initialized
        assert manager.coordinator == mock_coordinator
