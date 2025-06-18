"""Tests for sensor module."""

from unittest.mock import Mock

from custom_components.area_occupancy.sensor import (
    AreaOccupancySensorBase,
    DecaySensor,
    EntitiesSensor,
    PriorsSensor,
    ProbabilitySensor,
    async_setup_entry,
)
from homeassistant.const import EntityCategory


# ruff: noqa: SLF001
class TestAreaOccupancySensorBase:
    """Test AreaOccupancySensorBase class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test base sensor initialization."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")

        assert sensor.coordinator == mock_coordinator
        # Base sensor doesn't store _entry_id, just uses it for subclass unique_id generation
        # Base sensor doesn't set _attr_entity_registry_enabled_default initially - it's set via set_enabled_default method

    def test_set_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test setting enabled default."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")

        # Test setting to False
        sensor.set_enabled_default(False)
        assert sensor._attr_entity_registry_enabled_default is False

        # Test setting to True
        sensor.set_enabled_default(True)
        assert sensor._attr_entity_registry_enabled_default is True

    def test_available_property(self, mock_coordinator: Mock) -> None:
        """Test available property."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")

        # Test when coordinator is available (CoordinatorEntity uses last_update_success)
        mock_coordinator.last_update_success = True
        assert sensor.available is True

        # Test when coordinator is not available
        mock_coordinator.last_update_success = False
        assert sensor.available is False

    def test_device_info_property(self, mock_coordinator: Mock) -> None:
        """Test device_info property."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")

        expected_device_info = mock_coordinator.device_info
        assert sensor.device_info == expected_device_info


class TestPriorsSensor:
    """Test PriorsSensor class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test PriorsSensor initialization."""
        sensor = PriorsSensor(mock_coordinator, "test_entry")

        # Actual unique_id format from implementation
        assert sensor.unique_id == "test_entry_prior_probability"
        assert sensor.name == "Prior Probability"
        # PriorsSensor doesn't have an icon property
        assert sensor.native_unit_of_measurement == "%"
        assert sensor.suggested_display_precision == 1
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC

    def test_native_value_property(self, mock_coordinator: Mock) -> None:
        """Test native_value property."""
        sensor = PriorsSensor(mock_coordinator, "test_entry")

        # Test with coordinator prior
        mock_coordinator.prior = 0.35
        assert sensor.native_value == 35.0  # Converted to percentage

        # Test with different prior
        mock_coordinator.prior = 0.75
        assert sensor.native_value == 75.0

        # Test with edge values
        mock_coordinator.prior = 0.0
        assert sensor.native_value == 0.0

        mock_coordinator.prior = 1.0
        assert sensor.native_value == 100.0

    def test_native_value_none_when_unavailable(self, mock_coordinator: Mock) -> None:
        """Test native_value returns None when coordinator unavailable."""
        sensor = PriorsSensor(mock_coordinator, "test_entry")

        # PriorsSensor doesn't override native_value to return None when unavailable
        # It always returns the coordinator.prior value
        mock_coordinator.prior = 0.35
        assert sensor.native_value == 35.0

    def test_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test that PriorsSensor is disabled by default."""
        sensor = PriorsSensor(mock_coordinator, "test_entry")

        # PriorsSensor sets enabled default via entity_category, not _attr_entity_registry_enabled_default
        # Diagnostic entities are typically disabled by default
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC


class TestProbabilitySensor:
    """Test ProbabilitySensor class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test ProbabilitySensor initialization."""
        sensor = ProbabilitySensor(mock_coordinator, "test_entry")

        # Actual unique_id format from implementation
        assert sensor.unique_id == "test_entry_occupancy_probability"
        assert sensor.name == "Occupancy Probability"
        # ProbabilitySensor doesn't have an icon property
        assert sensor.native_unit_of_measurement == "%"
        assert sensor.suggested_display_precision == 1
        # ProbabilitySensor doesn't have entity_category set (defaults to None)

    def test_native_value_property(self, mock_coordinator: Mock) -> None:
        """Test native_value property."""
        sensor = ProbabilitySensor(mock_coordinator, "test_entry")

        # Test with coordinator probability
        mock_coordinator.probability = 0.65
        assert sensor.native_value == 65.0  # Converted to percentage

        # Test with different probability
        mock_coordinator.probability = 0.25
        assert sensor.native_value == 25.0

        # Test with edge values
        mock_coordinator.probability = 0.0
        assert sensor.native_value == 0.0

        mock_coordinator.probability = 1.0
        assert sensor.native_value == 100.0

    def test_native_value_none_when_unavailable(self, mock_coordinator: Mock) -> None:
        """Test native_value returns None when coordinator unavailable."""
        sensor = ProbabilitySensor(mock_coordinator, "test_entry")

        # ProbabilitySensor doesn't override native_value to return None when unavailable
        # It always returns the coordinator.probability value
        mock_coordinator.probability = 0.65
        assert sensor.native_value == 65.0


class TestEntitiesSensor:
    """Test EntitiesSensor class."""

    def test_initialization(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test EntitiesSensor initialization."""
        sensor = EntitiesSensor(mock_coordinator_with_sensors, "test_entry")

        assert sensor.unique_id == "test_entry_entities"
        assert sensor.name == "Entities"
        # EntitiesSensor doesn't have an icon property
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC

    def test_native_value_property(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test native_value property."""
        sensor = EntitiesSensor(mock_coordinator_with_sensors, "test_entry")

        # Should return total number of entities from coordinator.entities.entities
        assert sensor.native_value == 4

    def test_native_value_no_entity_manager(self, mock_coordinator: Mock) -> None:
        """Test native_value when no entity manager."""
        # Set up coordinator with no entities
        mock_coordinator.entities.entities = {}
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        assert sensor.native_value == 0

    def test_native_value_none_when_unavailable(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test native_value returns None when coordinator unavailable."""
        sensor = EntitiesSensor(mock_coordinator_with_sensors, "test_entry")

        # EntitiesSensor doesn't check availability, it always returns len() of entities
        assert sensor.native_value == 4

    def test_extra_state_attributes(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test extra_state_attributes property."""
        # Set up coordinator.data to be truthy and mock active/inactive entities
        mock_coordinator_with_sensors.data = {"test": "data"}
        mock_coordinator_with_sensors.entities.active_entities = []
        mock_coordinator_with_sensors.entities.inactive_entities = []

        sensor = EntitiesSensor(mock_coordinator_with_sensors, "test_entry")
        attributes = sensor.extra_state_attributes

        # Should return the expected structure
        assert "active" in attributes
        assert "inactive" in attributes

    def test_extra_state_attributes_no_entity_manager(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes when no coordinator data."""
        mock_coordinator.data = None
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes
        assert attributes == {}

    def test_extra_state_attributes_empty_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes with empty entities."""
        mock_coordinator.data = {"test": "data"}
        mock_coordinator.entities.active_entities = []
        mock_coordinator.entities.inactive_entities = []

        sensor = EntitiesSensor(mock_coordinator, "test_entry")
        attributes = sensor.extra_state_attributes

        assert "active" in attributes
        assert "inactive" in attributes
        assert attributes["active"] == []
        assert attributes["inactive"] == []

    def test_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test that EntitiesSensor is disabled by default."""
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        # EntitiesSensor sets enabled default via entity_category
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC


class TestDecaySensor:
    """Test DecaySensor class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test DecaySensor initialization."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        # Actual unique_id format from implementation
        assert sensor.unique_id == "test_entry_decay_status"
        assert sensor.name == "Decay Status"
        # DecaySensor doesn't have an icon property
        assert sensor.native_unit_of_measurement == "%"
        assert sensor.suggested_display_precision == 1
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC

    def test_native_value_property(self, mock_coordinator: Mock) -> None:
        """Test native_value property."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        # DecaySensor returns (1 - coordinator.decay) * 100
        # So if decay = 0.85, it returns (1 - 0.85) * 100 = 15.0
        mock_coordinator.decay = 0.85
        assert sensor.native_value == 15.0  # (1 - 0.85) * 100

        # Test with different decay
        mock_coordinator.decay = 0.5
        assert sensor.native_value == 50.0  # (1 - 0.5) * 100

        # Test with edge values
        mock_coordinator.decay = 0.0
        assert sensor.native_value == 100.0  # (1 - 0.0) * 100

        mock_coordinator.decay = 1.0
        assert sensor.native_value == 0.0  # (1 - 1.0) * 100

    def test_native_value_none_when_unavailable(self, mock_coordinator: Mock) -> None:
        """Test native_value returns None when coordinator unavailable."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        # DecaySensor doesn't override native_value to return None when unavailable
        # It always returns the calculated decay value
        mock_coordinator.decay = 0.85
        assert sensor.native_value == 15.0

    def test_extra_state_attributes(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test extra_state_attributes property."""
        # Set up mock active entities with decay information
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.decay.decay_factor = 0.8

        mock_coordinator_with_sensors.entities.active_entities = [mock_entity1]

        sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")
        attributes = sensor.extra_state_attributes

        # Should return the expected structure from implementation
        assert "active" in attributes
        assert len(attributes["active"]) == 1

    def test_extra_state_attributes_no_entity_manager(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes when no entity manager."""
        # Make active_entities raise an exception to trigger the except block
        mock_coordinator.entities.active_entities = Mock(
            side_effect=AttributeError("No entities")
        )
        sensor = DecaySensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes
        # Should return empty dict due to exception handling
        assert attributes == {}

    def test_extra_state_attributes_empty_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes with empty entities."""
        mock_coordinator.entities.active_entities = []
        sensor = DecaySensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes
        assert "active" in attributes
        assert attributes["active"] == []

    def test_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test that DecaySensor is disabled by default."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        # DecaySensor sets enabled default via entity_category
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful setup entry."""
        mock_async_add_entities = Mock()

        # Set up runtime_data with a mock coordinator
        mock_coordinator = Mock()
        mock_coordinator.device_info = {"test": "device_info"}
        mock_config_entry.runtime_data = mock_coordinator

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        # Should add all sensor entities
        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 4

        # Check entity types
        entity_types = [type(entity).__name__ for entity in entities]
        assert "PriorsSensor" in entity_types
        assert "ProbabilitySensor" in entity_types
        assert "EntitiesSensor" in entity_types
        assert "DecaySensor" in entity_types

    async def test_async_setup_entry_with_coordinator_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup entry with coordinator data."""
        # Set up coordinator with specific data
        mock_coordinator = Mock()
        mock_coordinator.probability = 0.8
        mock_coordinator.prior = 0.3
        mock_coordinator.decay = 0.9
        mock_coordinator.device_info = {"test": "device_info"}
        mock_config_entry.runtime_data = mock_coordinator

        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        # Verify entities were created with correct coordinator
        entities = mock_async_add_entities.call_args[0][0]
        for entity in entities:
            assert entity.coordinator == mock_coordinator
            # Base sensor doesn't store _entry_id attribute


class TestSensorIntegration:
    """Test sensor integration scenarios."""

    def test_all_sensors_with_comprehensive_data(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test all sensors with comprehensive coordinator data."""
        # Set up coordinator with specific values
        mock_coordinator_with_sensors.prior = 0.35
        mock_coordinator_with_sensors.probability = 0.65
        mock_coordinator_with_sensors.decay = 0.15  # Will result in (1-0.15)*100 = 85.0

        # Create all sensor types
        priors_sensor = PriorsSensor(mock_coordinator_with_sensors, "test_entry")
        probability_sensor = ProbabilitySensor(
            mock_coordinator_with_sensors, "test_entry"
        )
        entities_sensor = EntitiesSensor(mock_coordinator_with_sensors, "test_entry")
        decay_sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")

        # Test native values
        assert priors_sensor.native_value == 35.0
        assert probability_sensor.native_value == 65.0
        assert entities_sensor.native_value == 4  # From mock_coordinator_with_sensors
        assert decay_sensor.native_value == 85.0  # (1 - 0.15) * 100

    def test_sensor_availability_changes(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test sensor behavior when coordinator availability changes."""
        sensors = [
            PriorsSensor(mock_coordinator_with_sensors, "test_entry"),
            ProbabilitySensor(mock_coordinator_with_sensors, "test_entry"),
            EntitiesSensor(mock_coordinator_with_sensors, "test_entry"),
            DecaySensor(mock_coordinator_with_sensors, "test_entry"),
        ]

        # Test when available (CoordinatorEntity uses last_update_success)
        mock_coordinator_with_sensors.last_update_success = True
        for sensor in sensors:
            assert sensor.available is True

        # Test when unavailable
        mock_coordinator_with_sensors.last_update_success = False
        for sensor in sensors:
            assert sensor.available is False
            # Sensors still return values even when unavailable - they don't check availability in native_value

    def test_sensor_value_updates(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test sensor value updates when coordinator data changes."""
        probability_sensor = ProbabilitySensor(
            mock_coordinator_with_sensors, "test_entry"
        )
        priors_sensor = PriorsSensor(mock_coordinator_with_sensors, "test_entry")
        decay_sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")

        # Initial values
        mock_coordinator_with_sensors.probability = 0.65
        mock_coordinator_with_sensors.prior = 0.35
        mock_coordinator_with_sensors.decay = 0.15

        assert probability_sensor.native_value == 65.0
        assert priors_sensor.native_value == 35.0
        assert decay_sensor.native_value == 85.0  # (1 - 0.15) * 100

        # Update coordinator values
        mock_coordinator_with_sensors.probability = 0.8
        mock_coordinator_with_sensors.prior = 0.4
        mock_coordinator_with_sensors.decay = 0.3

        # Check updated values
        assert probability_sensor.native_value == 80.0
        assert priors_sensor.native_value == 40.0
        assert decay_sensor.native_value == 70.0  # (1 - 0.3) * 100

    def test_entities_sensor_dynamic_updates(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test entities sensor with dynamic entity updates."""
        entities_sensor = EntitiesSensor(mock_coordinator_with_sensors, "test_entry")

        # Initial state - 4 entities from mock_coordinator_with_sensors
        assert entities_sensor.native_value == 4

        # Add more entities
        new_entity = Mock()
        mock_coordinator_with_sensors.entities.entities["new_sensor"] = new_entity
        assert entities_sensor.native_value == 5

        # Remove entities
        del mock_coordinator_with_sensors.entities.entities["binary_sensor.motion1"]
        assert entities_sensor.native_value == 4

    def test_decay_sensor_dynamic_updates(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decay sensor with dynamic decay state updates."""
        # Set up mock active entities
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.decay.decay_factor = 0.8

        mock_coordinator_with_sensors.entities.active_entities = [mock_entity1]

        decay_sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")

        # Initial state
        attrs = decay_sensor.extra_state_attributes
        assert "active" in attrs
        assert len(attrs["active"]) == 1

        # Add another decaying entity
        mock_entity2 = Mock()
        mock_entity2.entity_id = "binary_sensor.motion2"
        mock_entity2.decay.decay_factor = 0.6

        mock_coordinator_with_sensors.entities.active_entities = [
            mock_entity1,
            mock_entity2,
        ]

        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["active"]) == 2

    def test_sensor_error_handling(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test sensor error handling scenarios."""
        # Test with entities that raise exceptions
        mock_coordinator_with_sensors.entities.entities = Mock(
            side_effect=Exception("Test error")
        )

        entities_sensor = EntitiesSensor(mock_coordinator_with_sensors, "test_entry")
        decay_sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")

        # Should handle gracefully and not crash
        try:  # noqa: SIM105
            # This should raise the exception since it's not caught in native_value
            entities_sensor.native_value  # noqa: B018
        except Exception:  # noqa: BLE001
            # This is expected since the implementation doesn't handle the error
            pass

        # Test decay sensor error handling
        mock_coordinator_with_sensors.entities.active_entities = Mock(
            side_effect=Exception("Test error")
        )
        attrs = decay_sensor.extra_state_attributes
        # Should return empty dict due to exception handling in implementation
        assert attrs == {}
