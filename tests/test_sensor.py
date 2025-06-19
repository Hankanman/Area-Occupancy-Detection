"""Test sensor entities."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.area_occupancy.const import (
    CONF_HISTORY_PERIOD,
    DEFAULT_HISTORY_PERIOD,
    DOMAIN,
    NAME_DECAY_SENSOR,
    NAME_PRIORS_SENSOR,
    NAME_PROBABILITY_SENSOR,
)
from custom_components.area_occupancy.sensor import (
    AreaOccupancyDecaySensor,
    AreaOccupancyProbabilitySensor,
    AreaOccupancySensorBase,
    PriorsSensor,
    async_setup_entry,
    format_float,
)
from custom_components.area_occupancy.types import EntityType, PriorState, ProbabilityState


class TestAreaOccupancySensorBase:
    """Test AreaOccupancySensorBase class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test sensor base initialization."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry_id")
        
        assert sensor._attr_has_entity_name is True
        assert sensor._attr_should_poll is False
        assert sensor._attr_device_info == mock_coordinator.device_info
        assert sensor._attr_suggested_display_precision == 1

    def test_set_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test setting enabled default."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry_id")
        
        sensor.set_enabled_default(False)
        assert sensor._attr_entity_registry_enabled_default is False
        
        sensor.set_enabled_default(True)
        assert sensor._attr_entity_registry_enabled_default is True


class TestPriorsSensor:
    """Test PriorsSensor class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test priors sensor initialization."""
        sensor = PriorsSensor(mock_coordinator, "test_entry_id")
        
        assert sensor._attr_name == NAME_PRIORS_SENSOR
        assert sensor._attr_unique_id == f"test_entry_id_{NAME_PRIORS_SENSOR.lower().replace(' ', '_')}"
        assert sensor._attr_device_class == SensorDeviceClass.POWER_FACTOR
        assert sensor._attr_native_unit_of_measurement == PERCENTAGE
        assert sensor._attr_state_class == SensorStateClass.MEASUREMENT
        assert sensor._attr_entity_category == EntityCategory.DIAGNOSTIC

    def test_native_value_with_valid_prior_state(self, mock_coordinator: Mock) -> None:
        """Test native value with valid prior state."""
        prior_state = Mock()
        prior_state.overall_prior = 0.75
        mock_coordinator.prior_state = prior_state
        
        sensor = PriorsSensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value == 75.0

    def test_native_value_with_no_prior_state(self, mock_coordinator: Mock) -> None:
        """Test native value with no prior state."""
        mock_coordinator.prior_state = None
        
        sensor = PriorsSensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value is None

    def test_native_value_with_error(self, mock_coordinator: Mock) -> None:
        """Test native value with error."""
        mock_coordinator.prior_state = Mock()
        mock_coordinator.prior_state.overall_prior = Mock(side_effect=ValueError("Test error"))
        
        sensor = PriorsSensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value is None

    def test_extra_state_attributes_with_valid_data(self, mock_coordinator: Mock) -> None:
        """Test extra state attributes with valid data."""
        prior_state = Mock()
        prior_state.motion_prior = 0.8
        prior_state.media_prior = 0.6
        prior_state.appliance_prior = 0.4
        prior_state.door_prior = 0.0  # Should be excluded
        prior_state.window_prior = 0.0  # Should be excluded
        prior_state.light_prior = 0.2
        prior_state.analysis_period = 7
        
        mock_coordinator.prior_state = prior_state
        mock_coordinator.last_prior_update = "2024-01-01T00:00:00"
        
        sensor = PriorsSensor(mock_coordinator, "test_entry_id")
        attributes = sensor.extra_state_attributes
        
        assert EntityType.MOTION.value in attributes
        assert EntityType.MEDIA.value in attributes
        assert EntityType.APPLIANCE.value in attributes
        assert EntityType.LIGHT.value in attributes
        assert EntityType.DOOR.value not in attributes  # Zero values excluded
        assert EntityType.WINDOW.value not in attributes  # Zero values excluded
        assert attributes["last_updated"] == "2024-01-01T00:00:00"
        assert attributes["total_period"] == "7 days"

    def test_extra_state_attributes_with_no_prior_state(self, mock_coordinator: Mock) -> None:
        """Test extra state attributes with no prior state."""
        mock_coordinator.prior_state = None
        
        sensor = PriorsSensor(mock_coordinator, "test_entry_id")
        
        assert sensor.extra_state_attributes == {}

    def test_extra_state_attributes_with_error(self, mock_coordinator: Mock) -> None:
        """Test extra state attributes with error."""
        mock_coordinator.prior_state = Mock(side_effect=ValueError("Test error"))
        
        sensor = PriorsSensor(mock_coordinator, "test_entry_id")
        
        assert sensor.extra_state_attributes == {}


class TestAreaOccupancyProbabilitySensor:
    """Test AreaOccupancyProbabilitySensor class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test probability sensor initialization."""
        sensor = AreaOccupancyProbabilitySensor(mock_coordinator, "test_entry_id")
        
        assert sensor._attr_name == NAME_PROBABILITY_SENSOR
        assert sensor._attr_unique_id == f"test_entry_id_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}"
        assert sensor._attr_device_class == SensorDeviceClass.POWER_FACTOR
        assert sensor._attr_native_unit_of_measurement == PERCENTAGE
        assert sensor._attr_state_class == SensorStateClass.MEASUREMENT
        assert sensor._attr_entity_category is None

    def test_native_value_with_valid_data(self, mock_coordinator: Mock) -> None:
        """Test native value with valid data."""
        mock_coordinator.data = Mock()
        mock_coordinator.probability = 0.75
        
        sensor = AreaOccupancyProbabilitySensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value == 75.0

    def test_native_value_with_no_data(self, mock_coordinator: Mock) -> None:
        """Test native value with no data."""
        mock_coordinator.data = None
        
        sensor = AreaOccupancyProbabilitySensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value == 0.0

    def test_native_value_with_missing_probability(self, mock_coordinator: Mock) -> None:
        """Test native value with missing probability attribute."""
        mock_coordinator.data = Mock()
        # Remove probability attribute
        if hasattr(mock_coordinator, 'probability'):
            delattr(mock_coordinator, 'probability')
        
        sensor = AreaOccupancyProbabilitySensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value == 0.0

    def test_extra_state_attributes_with_valid_data(self, mock_coordinator: Mock, mock_hass: Mock) -> None:
        """Test extra state attributes with valid data."""
        # Mock coordinator data
        data = Mock()
        data.sensor_probabilities = {
            "binary_sensor.motion": {
                "weight": 0.8,
                "probability": 0.9,
                "weighted_probability": 0.72
            }
        }
        mock_coordinator.data = data
        mock_coordinator.threshold = 0.6
        
        # Mock hass states
        mock_state = Mock()
        mock_state.attributes = {"friendly_name": "Motion Sensor"}
        mock_hass.states.get.return_value = mock_state
        
        sensor = AreaOccupancyProbabilitySensor(mock_coordinator, "test_entry_id")
        sensor.hass = mock_hass
        
        attributes = sensor.extra_state_attributes
        
        assert "active_triggers" in attributes
        assert "sensor_probabilities" in attributes
        assert "threshold" in attributes
        assert attributes["threshold"] == "60.0%"
        assert "Motion Sensor" in attributes["active_triggers"]

    def test_extra_state_attributes_with_no_data(self, mock_coordinator: Mock) -> None:
        """Test extra state attributes with no data."""
        mock_coordinator.data = None
        
        sensor = AreaOccupancyProbabilitySensor(mock_coordinator, "test_entry_id")
        
        assert sensor.extra_state_attributes == {}

    def test_extra_state_attributes_with_error(self, mock_coordinator: Mock) -> None:
        """Test extra state attributes with error."""
        mock_coordinator.data = Mock(side_effect=AttributeError("Test error"))
        
        sensor = AreaOccupancyProbabilitySensor(mock_coordinator, "test_entry_id")
        
        assert sensor.extra_state_attributes == {}


class TestAreaOccupancyDecaySensor:
    """Test AreaOccupancyDecaySensor class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test decay sensor initialization."""
        sensor = AreaOccupancyDecaySensor(mock_coordinator, "test_entry_id")
        
        assert sensor._attr_name == NAME_DECAY_SENSOR
        assert sensor._attr_unique_id == f"test_entry_id_{NAME_DECAY_SENSOR.lower().replace(' ', '_')}"
        assert sensor._attr_device_class == SensorDeviceClass.POWER_FACTOR
        assert sensor._attr_native_unit_of_measurement == PERCENTAGE
        assert sensor._attr_state_class == SensorStateClass.MEASUREMENT
        assert sensor._attr_entity_category == EntityCategory.DIAGNOSTIC

    def test_native_value_with_valid_data(self, mock_coordinator: Mock) -> None:
        """Test native value with valid data."""
        data = Mock()
        data.decay_status = 25.5
        mock_coordinator.data = data
        
        sensor = AreaOccupancyDecaySensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value == 25.5

    def test_native_value_with_no_data(self, mock_coordinator: Mock) -> None:
        """Test native value with no data."""
        mock_coordinator.data = None
        
        sensor = AreaOccupancyDecaySensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value == 0.0

    def test_native_value_with_missing_decay_status(self, mock_coordinator: Mock) -> None:
        """Test native value with missing decay_status attribute."""
        mock_coordinator.data = Mock()
        # Remove decay_status attribute
        if hasattr(mock_coordinator.data, 'decay_status'):
            delattr(mock_coordinator.data, 'decay_status')
        
        sensor = AreaOccupancyDecaySensor(mock_coordinator, "test_entry_id")
        
        assert sensor.native_value == 0.0


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_setup_with_history_period(self, mock_hass: Mock) -> None:
        """Test setup with history period configured."""
        # Mock config entry
        config_entry = Mock()
        config_entry.entry_id = "test_entry_id"
        
        # Mock coordinator with history period
        mock_coordinator = Mock()
        mock_coordinator.config = {CONF_HISTORY_PERIOD: 7}
        
        # Mock hass data
        mock_hass.data = {
            DOMAIN: {
                "test_entry_id": {
                    "coordinator": mock_coordinator
                }
            }
        }
        
        # Mock async_add_entities
        async_add_entities = AsyncMock()
        
        await async_setup_entry(mock_hass, config_entry, async_add_entities)
        
        # Should add 3 sensors (probability, decay, priors)
        async_add_entities.assert_called_once()
        added_entities = async_add_entities.call_args[0][0]
        assert len(added_entities) == 3
        assert any(isinstance(e, AreaOccupancyProbabilitySensor) for e in added_entities)
        assert any(isinstance(e, AreaOccupancyDecaySensor) for e in added_entities)
        assert any(isinstance(e, PriorsSensor) for e in added_entities)

    async def test_setup_without_history_period(self, mock_hass: Mock) -> None:
        """Test setup without history period configured."""
        # Mock config entry
        config_entry = Mock()
        config_entry.entry_id = "test_entry_id"
        
        # Mock coordinator without history period
        mock_coordinator = Mock()
        mock_coordinator.config = {CONF_HISTORY_PERIOD: 0}
        
        # Mock hass data
        mock_hass.data = {
            DOMAIN: {
                "test_entry_id": {
                    "coordinator": mock_coordinator
                }
            }
        }
        
        # Mock async_add_entities
        async_add_entities = AsyncMock()
        
        await async_setup_entry(mock_hass, config_entry, async_add_entities)
        
        # Should add 2 sensors (probability, decay) - no priors sensor
        async_add_entities.assert_called_once()
        added_entities = async_add_entities.call_args[0][0]
        assert len(added_entities) == 2
        assert any(isinstance(e, AreaOccupancyProbabilitySensor) for e in added_entities)
        assert any(isinstance(e, AreaOccupancyDecaySensor) for e in added_entities)
        assert not any(isinstance(e, PriorsSensor) for e in added_entities)

    async def test_setup_with_default_history_period(self, mock_hass: Mock) -> None:
        """Test setup with default history period."""
        # Mock config entry
        config_entry = Mock()
        config_entry.entry_id = "test_entry_id"
        
        # Mock coordinator with no history period config (will use default)
        mock_coordinator = Mock()
        mock_coordinator.config = {}
        
        # Mock hass data
        mock_hass.data = {
            DOMAIN: {
                "test_entry_id": {
                    "coordinator": mock_coordinator
                }
            }
        }
        
        # Mock async_add_entities
        async_add_entities = AsyncMock()
        
        await async_setup_entry(mock_hass, config_entry, async_add_entities)
        
        # Should add 3 sensors (default history period > 0)
        async_add_entities.assert_called_once()
        added_entities = async_add_entities.call_args[0][0]
        assert len(added_entities) == 3


class TestFormatFloat:
    """Test format_float utility function."""

    def test_format_float_with_valid_float(self) -> None:
        """Test format_float with valid float."""
        assert format_float(75.12345) == 75.12
        assert format_float(0.0) == 0.0
        assert format_float(100.0) == 100.0

    def test_format_float_with_int(self) -> None:
        """Test format_float with integer."""
        assert format_float(75) == 75.0
        assert format_float(0) == 0.0

    def test_format_float_with_string_number(self) -> None:
        """Test format_float with string number."""
        assert format_float("75.12345") == 75.12
        assert format_float("0") == 0.0

    def test_format_float_with_invalid_value(self) -> None:
        """Test format_float with invalid value."""
        assert format_float("invalid") == 0.0
        assert format_float(None) == 0.0
        assert format_float([]) == 0.0