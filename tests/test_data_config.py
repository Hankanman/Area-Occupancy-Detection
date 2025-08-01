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
    DEFAULT_THRESHOLD,
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
    ConfigManager,
    Decay,
    Sensors,
    SensorStates,
    WaspInBox,
    Weights,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestSensors:
    """Test Sensors dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test Sensors initialization with defaults."""
        sensors = Sensors()

        assert sensors.motion == []
        assert sensors.primary_occupancy is None
        assert sensors.media == []
        assert sensors.appliances == []
        assert sensors.illuminance == []
        assert sensors.humidity == []
        assert sensors.temperature == []
        assert sensors.doors == []
        assert sensors.windows == []

    def test_initialization_with_values(self) -> None:
        """Test Sensors initialization with specific values."""
        sensors = Sensors(
            motion=["binary_sensor.motion1", "binary_sensor.motion2"],
            primary_occupancy="binary_sensor.motion1",
            media=["media_player.tv"],
            appliances=["switch.coffee_maker"],
            illuminance=["sensor.illuminance"],
            humidity=["sensor.humidity"],
            temperature=["sensor.temperature"],
            doors=["binary_sensor.door"],
            windows=["binary_sensor.window"],
        )

        assert sensors.motion == ["binary_sensor.motion1", "binary_sensor.motion2"]
        assert sensors.primary_occupancy == "binary_sensor.motion1"
        assert sensors.media == ["media_player.tv"]
        assert sensors.appliances == ["switch.coffee_maker"]
        assert sensors.illuminance == ["sensor.illuminance"]
        assert sensors.humidity == ["sensor.humidity"]
        assert sensors.temperature == ["sensor.temperature"]
        assert sensors.doors == ["binary_sensor.door"]
        assert sensors.windows == ["binary_sensor.window"]

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

    def test_initialization_defaults(self) -> None:
        """Test Config initialization with defaults."""
        config = Config()

        assert config.name == "Area Occupancy"
        assert config.area_id is None
        assert config.threshold == DEFAULT_THRESHOLD
        assert isinstance(config.sensors, Sensors)
        assert isinstance(config.sensor_states, SensorStates)
        assert isinstance(config.weights, Weights)
        assert isinstance(config.decay, Decay)
        assert isinstance(config.wasp_in_box, WaspInBox)

    def test_initialization_with_values(self) -> None:
        """Test Config initialization with specific values."""
        sensors = Sensors(motion=["binary_sensor.motion1"])
        weights = Weights(motion=0.9)

        config = Config(
            name="Living Room",
            area_id="living_room",
            threshold=0.6,
            sensors=sensors,
            weights=weights,
        )

        assert config.name == "Living Room"
        assert config.area_id == "living_room"
        assert config.threshold == 0.6
        assert config.sensors == sensors
        assert config.weights == weights

    def test_start_time_property(self) -> None:
        """Test start_time property calculation."""
        config = Config()
        start_time = config.start_time
        expected_start = dt_util.utcnow() - timedelta(days=HA_RECORDER_DAYS)

        # Allow some tolerance for test execution time
        assert abs((start_time - expected_start).total_seconds()) < 5

    def test_end_time_property(self) -> None:
        """Test end_time property calculation."""
        config = Config()

        end_time = config.end_time
        expected_end = dt_util.utcnow()

        # Allow some tolerance for test execution time
        assert abs((end_time - expected_end).total_seconds()) < 5

    def test_from_dict_minimal(self) -> None:
        """Test Config.from_dict with minimal data."""
        data = {
            CONF_NAME: "Test Area",
            CONF_THRESHOLD: 50,  # Percentage
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }

        config = Config.from_dict(data)

        assert config.name == "Test Area"
        assert config.threshold == 0.5  # Converted from percentage
        assert config.sensors.motion == ["binary_sensor.motion1"]

    def test_from_dict_comprehensive(self) -> None:
        """Test Config.from_dict with comprehensive data."""
        data = {
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

        config = Config.from_dict(data)

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
        assert config.sensors.appliances == ["switch.coffee_maker"]
        assert config.sensors.illuminance == ["sensor.illuminance"]
        assert config.sensors.humidity == ["sensor.humidity"]
        assert config.sensors.temperature == ["sensor.temperature"]
        assert config.sensors.doors == ["binary_sensor.door"]
        assert config.sensors.windows == ["binary_sensor.window"]
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

    def test_from_dict_with_invalid_weights(self) -> None:
        """Test Config.from_dict with invalid weight values."""
        data = {
            CONF_NAME: "Test Area",
            CONF_THRESHOLD: 50,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_WEIGHT_MOTION: -0.5,  # Invalid negative weight
            CONF_WEIGHT_MEDIA: 1.5,  # Large weight (allowed)
        }

        config = Config.from_dict(data)

        # Only negative weights should be replaced with defaults
        assert config.weights.motion == DEFAULT_WEIGHT_MOTION
        assert config.weights.media == 1.5  # Large weights are allowed

    def test_as_dict(self) -> None:
        """Test Config.as_dict method."""
        original_data = {
            CONF_NAME: "Test Area",
            CONF_THRESHOLD: 50,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }

        config = Config.from_dict(original_data)
        result = config.as_dict()

        # Should return the original raw data
        assert result == original_data


class TestConfigManager:
    """Test ConfigManager class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test ConfigManager initialization."""
        manager = ConfigManager(mock_coordinator)

        assert manager.config_entry == mock_coordinator.config_entry
        assert manager._config is not None
        assert manager._hass == mock_coordinator.hass

    def test_hass_property(self, mock_coordinator: Mock) -> None:
        """Test hass property."""
        manager = ConfigManager(mock_coordinator)

        assert manager.hass == mock_coordinator.hass

    def test_set_hass(self, mock_coordinator: Mock) -> None:
        """Test set_hass method."""
        manager = ConfigManager(mock_coordinator)
        new_hass = Mock()

        manager.set_hass(new_hass)

        assert manager._hass == new_hass

    def test_merge_entry(self, mock_config_entry: Mock) -> None:
        """Test _merge_entry static method."""
        result = ConfigManager._merge_entry(mock_config_entry)

        # Should merge data and options
        assert CONF_NAME in result
        assert CONF_THRESHOLD in result
        assert CONF_MOTION_SENSORS in result
        assert result[CONF_NAME] == "Test Area"

    def test_config_property_first_access(self, mock_coordinator: Mock) -> None:
        """Test config property on first access."""
        manager = ConfigManager(mock_coordinator)

        # Should have created config during initialization
        config = manager.config

        assert isinstance(config, Config)
        assert config.name == "Testing"  # Using realistic fixture data
        assert manager._config is not None

    def test_config_property_cached(self, mock_coordinator: Mock) -> None:
        """Test config property returns cached value."""
        manager = ConfigManager(mock_coordinator)

        # First access
        config1 = manager.config
        # Second access
        config2 = manager.config

        # Should return the same cached instance
        assert config1 is config2

    def test_update_from_entry(self, mock_coordinator: Mock) -> None:
        """Test update_from_entry method."""
        manager = ConfigManager(mock_coordinator)

        # Create initial config
        initial_config = manager.config
        assert initial_config.name == "Testing"  # Using realistic fixture data

        # Create new config entry with updated data
        new_config_entry = Mock(spec=ConfigEntry)
        new_config_entry.data = {
            CONF_NAME: "Updated Area",
            CONF_THRESHOLD: 70,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        new_config_entry.options = {}

        # Update from entry
        manager.update_from_entry(new_config_entry)

        # Should create new config with updated data
        updated_config = manager.config
        assert updated_config.name == "Updated Area"
        assert updated_config.threshold == 0.7

    def test_get_method(self, mock_coordinator: Mock) -> None:
        """Test get method."""
        manager = ConfigManager(mock_coordinator)

        # Test getting existing value
        name = manager.get("name")
        assert name == "Testing"  # Using realistic fixture data

        # Test getting non-existent value with default
        value = manager.get("nonexistent_key", "default_value")
        assert value == "default_value"

    async def test_update_config(self, mock_coordinator: Mock) -> None:
        """Test update_config method."""
        manager = ConfigManager(mock_coordinator)

        # Initial config
        initial_config = manager.config
        assert initial_config.decay.enabled == DEFAULT_DECAY_ENABLED

        # Update with new options
        new_options = {CONF_DECAY_ENABLED: False, CONF_THRESHOLD: 80}

        await manager.update_config(new_options)

        # Should update the config
        updated_config = manager.config
        assert updated_config.decay.enabled is False
        assert updated_config.threshold == 0.8  # 80 / 100 (converted from percentage)


class TestConfigIntegration:
    """Test Config integration scenarios."""

    def test_config_with_all_sensor_types(self) -> None:
        """Test config creation with all possible sensor types."""
        data = {
            CONF_NAME: "Complete Setup",
            CONF_THRESHOLD: 55,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_MEDIA_DEVICES: ["media_player.tv", "media_player.stereo"],
            CONF_APPLIANCES: ["switch.coffee_maker", "switch.dishwasher"],
            CONF_ILLUMINANCE_SENSORS: ["sensor.illuminance1", "sensor.illuminance2"],
            CONF_HUMIDITY_SENSORS: ["sensor.humidity"],
            CONF_TEMPERATURE_SENSORS: ["sensor.temperature"],
            CONF_DOOR_SENSORS: ["binary_sensor.front_door", "binary_sensor.back_door"],
            CONF_WINDOW_SENSORS: ["binary_sensor.window1", "binary_sensor.window2"],
        }

        config = Config.from_dict(data)

        # Verify all sensor lists are properly populated
        assert len(config.sensors.motion) == 2
        assert len(config.sensors.media) == 2
        assert len(config.sensors.appliances) == 2
        assert len(config.sensors.illuminance) == 2
        assert len(config.sensors.humidity) == 1
        assert len(config.sensors.temperature) == 1
        assert len(config.sensors.doors) == 2
        assert len(config.sensors.windows) == 2

    def test_config_validation_edge_cases(self) -> None:
        """Test config validation with edge cases."""
        data = {
            CONF_NAME: "",  # Empty name
            CONF_THRESHOLD: 0,  # Minimum threshold
            CONF_MOTION_SENSORS: [],  # Empty sensor list
            CONF_WEIGHT_MOTION: 0,  # Edge case weight
        }

        config = Config.from_dict(data)

        # Should handle edge cases gracefully
        assert config.name == ""  # Empty name is allowed
        assert config.threshold == 0.0  # Minimum threshold
        assert config.sensors.motion == []  # Empty list is allowed
        assert config.weights.motion == 0.0  # Zero weight is allowed (not negative)

    def test_config_manager_full_lifecycle(self, mock_coordinator: Mock) -> None:
        """Test ConfigManager through a complete lifecycle."""
        # Create manager
        manager = ConfigManager(mock_coordinator)

        # Initial config access
        initial_config = manager.config
        assert initial_config.name == "Testing"  # Using realistic fixture data
        assert initial_config.threshold == 0.52  # 50 / 100
        assert initial_config.decay.enabled == DEFAULT_DECAY_ENABLED

        # Create new config entry with updated data
        new_config_entry = Mock(spec=ConfigEntry)
        new_config_entry.data = {
            CONF_NAME: "Updated Area",
            CONF_THRESHOLD: 70,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
        }
        new_config_entry.options = {CONF_DECAY_ENABLED: False}

        manager.update_from_entry(new_config_entry)

        # Verify updates
        updated_config = manager.config
        assert updated_config.name == "Updated Area"
        assert updated_config.threshold == 0.7
        assert len(updated_config.sensors.motion) == 2
        assert updated_config.decay.enabled is False
