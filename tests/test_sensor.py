"""Tests for sensor module."""

from datetime import timedelta
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.sensor import (
    AreaOccupancySensorBase,
    DecaySensor,
    EntitiesSensor,
    PriorsSensor,
    ProbabilitySensor,
    async_setup_entry,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util


class TestAreaOccupancySensorBase:
    """Test AreaOccupancySensorBase class."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.available = True
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }
        return coordinator

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test base sensor initialization."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")

        assert sensor.coordinator == mock_coordinator
        assert sensor._entry_id == "test_entry"
        assert sensor._attr_entity_registry_enabled_default is True

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

        # Test when coordinator is available
        mock_coordinator.available = True
        assert sensor.available is True

        # Test when coordinator is not available
        mock_coordinator.available = False
        assert sensor.available is False

    def test_device_info_property(self, mock_coordinator: Mock) -> None:
        """Test device_info property."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")

        expected_device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }

        assert sensor.device_info == expected_device_info


class TestPriorsSensor:
    """Test PriorsSensor class."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.available = True
        coordinator.prior = 0.35
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }
        return coordinator

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test PriorsSensor initialization."""
        sensor = PriorsSensor(mock_coordinator, "test_entry")

        assert sensor.unique_id == "test_entry_priors"
        assert sensor.name == "Priors"
        assert sensor.icon == "mdi:chart-histogram"
        assert sensor.native_unit_of_measurement == "%"
        assert sensor.suggested_display_precision == 1
        assert sensor.entity_category == "diagnostic"

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

        mock_coordinator.available = False
        assert sensor.native_value is None

    def test_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test that PriorsSensor is disabled by default."""
        sensor = PriorsSensor(mock_coordinator, "test_entry")

        assert sensor._attr_entity_registry_enabled_default is False


class TestProbabilitySensor:
    """Test ProbabilitySensor class."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.available = True
        coordinator.probability = 0.65
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }
        return coordinator

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test ProbabilitySensor initialization."""
        sensor = ProbabilitySensor(mock_coordinator, "test_entry")

        assert sensor.unique_id == "test_entry_probability"
        assert sensor.name == "Probability"
        assert sensor.icon == "mdi:percent"
        assert sensor.native_unit_of_measurement == "%"
        assert sensor.suggested_display_precision == 1
        assert sensor.state_class == "measurement"

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

        mock_coordinator.available = False
        assert sensor.native_value is None


class TestEntitiesSensor:
    """Test EntitiesSensor class."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.available = True
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }

        # Mock entity manager with entities
        coordinator.entity_manager = Mock()
        coordinator.entity_manager.entities = {
            "binary_sensor.motion1": Mock(available=True, is_active=True),
            "binary_sensor.motion2": Mock(available=True, is_active=False),
            "light.test_light": Mock(available=False, is_active=False),
            "media_player.tv": Mock(available=True, is_active=True),
        }

        return coordinator

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test EntitiesSensor initialization."""
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        assert sensor.unique_id == "test_entry_entities"
        assert sensor.name == "Entities"
        assert sensor.icon == "mdi:format-list-bulleted"
        assert sensor.entity_category == "diagnostic"

    def test_native_value_property(self, mock_coordinator: Mock) -> None:
        """Test native_value property."""
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        # Should return total number of entities
        assert sensor.native_value == 4

    def test_native_value_no_entity_manager(self, mock_coordinator: Mock) -> None:
        """Test native_value when no entity manager."""
        mock_coordinator.entity_manager = None
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        assert sensor.native_value == 0

    def test_native_value_none_when_unavailable(self, mock_coordinator: Mock) -> None:
        """Test native_value returns None when coordinator unavailable."""
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        mock_coordinator.available = False
        assert sensor.native_value is None

    def test_extra_state_attributes(self, mock_coordinator: Mock) -> None:
        """Test extra_state_attributes property."""
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes

        assert "total" in attributes
        assert "active" in attributes
        assert "inactive" in attributes
        assert "available" in attributes
        assert "unavailable" in attributes

        assert attributes["total"] == 4
        assert attributes["active"] == 2  # motion1 and tv are active
        assert attributes["inactive"] == 2  # motion2 and light are inactive
        assert attributes["available"] == 3  # motion1, motion2, tv are available
        assert attributes["unavailable"] == 1  # light is unavailable

    def test_extra_state_attributes_no_entity_manager(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes when no entity manager."""
        mock_coordinator.entity_manager = None
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes

        assert attributes["total"] == 0
        assert attributes["active"] == 0
        assert attributes["inactive"] == 0
        assert attributes["available"] == 0
        assert attributes["unavailable"] == 0

    def test_extra_state_attributes_empty_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes with empty entities."""
        mock_coordinator.entity_manager.entities = {}
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes

        assert attributes["total"] == 0
        assert attributes["active"] == 0
        assert attributes["inactive"] == 0
        assert attributes["available"] == 0
        assert attributes["unavailable"] == 0

    def test_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test that EntitiesSensor is disabled by default."""
        sensor = EntitiesSensor(mock_coordinator, "test_entry")

        assert sensor._attr_entity_registry_enabled_default is False


class TestDecaySensor:
    """Test DecaySensor class."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.available = True
        coordinator.decay = 0.85
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }

        # Mock entity manager with decay information
        coordinator.entity_manager = Mock()
        mock_entity1 = Mock()
        mock_entity1.decay.is_decaying = True
        mock_entity1.decay.decay_start_time = dt_util.utcnow() - timedelta(minutes=5)
        mock_entity1.decay.decay_factor = 0.8

        mock_entity2 = Mock()
        mock_entity2.decay.is_decaying = False
        mock_entity2.decay.decay_start_time = None
        mock_entity2.decay.decay_factor = 1.0

        coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity1,
            "light.test_light": mock_entity2,
        }

        return coordinator

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test DecaySensor initialization."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        assert sensor.unique_id == "test_entry_decay"
        assert sensor.name == "Decay"
        assert sensor.icon == "mdi:chart-line-variant"
        assert sensor.native_unit_of_measurement == "%"
        assert sensor.suggested_display_precision == 1
        assert sensor.entity_category == "diagnostic"

    def test_native_value_property(self, mock_coordinator: Mock) -> None:
        """Test native_value property."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        # Test with coordinator decay
        mock_coordinator.decay = 0.85
        assert sensor.native_value == 85.0  # Converted to percentage

        # Test with different decay
        mock_coordinator.decay = 0.5
        assert sensor.native_value == 50.0

        # Test with edge values
        mock_coordinator.decay = 0.0
        assert sensor.native_value == 0.0

        mock_coordinator.decay = 1.0
        assert sensor.native_value == 100.0

    def test_native_value_none_when_unavailable(self, mock_coordinator: Mock) -> None:
        """Test native_value returns None when coordinator unavailable."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        mock_coordinator.available = False
        assert sensor.native_value is None

    def test_extra_state_attributes(self, mock_coordinator: Mock) -> None:
        """Test extra_state_attributes property."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes

        assert "entities_decaying" in attributes
        assert "total_entities" in attributes
        assert "decay_details" in attributes

        assert attributes["entities_decaying"] == 1  # Only motion1 is decaying
        assert attributes["total_entities"] == 2

        # Check decay details
        decay_details = attributes["decay_details"]
        assert "binary_sensor.motion1" in decay_details
        assert decay_details["binary_sensor.motion1"]["is_decaying"] is True
        assert (
            decay_details["binary_sensor.motion1"]["decay_factor"] == 80.0
        )  # Converted to percentage

    def test_extra_state_attributes_no_entity_manager(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes when no entity manager."""
        mock_coordinator.entity_manager = None
        sensor = DecaySensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes

        assert attributes["entities_decaying"] == 0
        assert attributes["total_entities"] == 0
        assert attributes["decay_details"] == {}

    def test_extra_state_attributes_empty_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes with empty entities."""
        mock_coordinator.entity_manager.entities = {}
        sensor = DecaySensor(mock_coordinator, "test_entry")

        attributes = sensor.extra_state_attributes

        assert attributes["entities_decaying"] == 0
        assert attributes["total_entities"] == 0
        assert attributes["decay_details"] == {}

    def test_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test that DecaySensor is disabled by default."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        assert sensor._attr_entity_registry_enabled_default is False


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry"
        entry.runtime_data = Mock()  # Mock coordinator
        return entry

    async def test_async_setup_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful setup entry."""
        mock_async_add_entities = Mock()

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
        mock_config_entry.runtime_data = mock_coordinator

        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        # Verify entities were created with correct coordinator
        entities = mock_async_add_entities.call_args[0][0]
        for entity in entities:
            assert entity.coordinator == mock_coordinator
            assert entity._entry_id == "test_entry"


class TestSensorIntegration:
    """Test sensor integration scenarios."""

    @pytest.fixture
    def comprehensive_coordinator(self) -> Mock:
        """Create a comprehensive mock coordinator."""
        coordinator = Mock()
        coordinator.available = True
        coordinator.probability = 0.65
        coordinator.prior = 0.35
        coordinator.decay = 0.85
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }

        # Mock entity manager with comprehensive data
        coordinator.entity_manager = Mock()

        # Create mock entities with different states
        mock_entities = {}
        for i, (entity_id, is_active, available, is_decaying) in enumerate(
            [
                ("binary_sensor.motion1", True, True, True),
                ("binary_sensor.motion2", False, True, False),
                ("light.test_light", True, False, False),
                ("media_player.tv", True, True, True),
                ("binary_sensor.door", False, True, False),
            ]
        ):
            mock_entity = Mock()
            mock_entity.is_active = is_active
            mock_entity.available = available
            mock_entity.decay.is_decaying = is_decaying
            mock_entity.decay.decay_start_time = (
                dt_util.utcnow() - timedelta(minutes=i + 1) if is_decaying else None
            )
            mock_entity.decay.decay_factor = 0.8 - (i * 0.1) if is_decaying else 1.0
            mock_entities[entity_id] = mock_entity

        coordinator.entity_manager.entities = mock_entities
        return coordinator

    def test_all_sensors_with_comprehensive_data(
        self, comprehensive_coordinator: Mock
    ) -> None:
        """Test all sensors with comprehensive coordinator data."""
        # Create all sensor types
        priors_sensor = PriorsSensor(comprehensive_coordinator, "test_entry")
        probability_sensor = ProbabilitySensor(comprehensive_coordinator, "test_entry")
        entities_sensor = EntitiesSensor(comprehensive_coordinator, "test_entry")
        decay_sensor = DecaySensor(comprehensive_coordinator, "test_entry")

        # Test native values
        assert priors_sensor.native_value == 35.0
        assert probability_sensor.native_value == 65.0
        assert entities_sensor.native_value == 5
        assert decay_sensor.native_value == 85.0

        # Test entities sensor attributes
        entities_attrs = entities_sensor.extra_state_attributes
        assert entities_attrs["total"] == 5
        assert entities_attrs["active"] == 3  # motion1, light, tv
        assert entities_attrs["inactive"] == 2  # motion2, door
        assert entities_attrs["available"] == 4  # all except light
        assert entities_attrs["unavailable"] == 1  # light

        # Test decay sensor attributes
        decay_attrs = decay_sensor.extra_state_attributes
        assert decay_attrs["entities_decaying"] == 2  # motion1, tv
        assert decay_attrs["total_entities"] == 5
        assert len(decay_attrs["decay_details"]) == 5

    def test_sensor_availability_changes(self, comprehensive_coordinator: Mock) -> None:
        """Test sensor behavior when coordinator availability changes."""
        sensors = [
            PriorsSensor(comprehensive_coordinator, "test_entry"),
            ProbabilitySensor(comprehensive_coordinator, "test_entry"),
            EntitiesSensor(comprehensive_coordinator, "test_entry"),
            DecaySensor(comprehensive_coordinator, "test_entry"),
        ]

        # Test when available
        comprehensive_coordinator.available = True
        for sensor in sensors:
            assert sensor.available is True
            assert sensor.native_value is not None

        # Test when unavailable
        comprehensive_coordinator.available = False
        for sensor in sensors:
            assert sensor.available is False
            assert sensor.native_value is None

    def test_sensor_value_updates(self, comprehensive_coordinator: Mock) -> None:
        """Test sensor value updates when coordinator data changes."""
        probability_sensor = ProbabilitySensor(comprehensive_coordinator, "test_entry")
        priors_sensor = PriorsSensor(comprehensive_coordinator, "test_entry")
        decay_sensor = DecaySensor(comprehensive_coordinator, "test_entry")

        # Initial values
        assert probability_sensor.native_value == 65.0
        assert priors_sensor.native_value == 35.0
        assert decay_sensor.native_value == 85.0

        # Update coordinator values
        comprehensive_coordinator.probability = 0.8
        comprehensive_coordinator.prior = 0.4
        comprehensive_coordinator.decay = 0.7

        # Check updated values
        assert probability_sensor.native_value == 80.0
        assert priors_sensor.native_value == 40.0
        assert decay_sensor.native_value == 70.0

    def test_entities_sensor_dynamic_updates(
        self, comprehensive_coordinator: Mock
    ) -> None:
        """Test entities sensor with dynamic entity updates."""
        entities_sensor = EntitiesSensor(comprehensive_coordinator, "test_entry")

        # Initial state
        assert entities_sensor.native_value == 5

        # Add more entities
        new_entity = Mock()
        new_entity.is_active = True
        new_entity.available = True
        comprehensive_coordinator.entity_manager.entities["new_sensor"] = new_entity

        assert entities_sensor.native_value == 6

        # Remove entities
        del comprehensive_coordinator.entity_manager.entities["binary_sensor.motion1"]
        assert entities_sensor.native_value == 5

    def test_decay_sensor_dynamic_updates(
        self, comprehensive_coordinator: Mock
    ) -> None:
        """Test decay sensor with dynamic decay state updates."""
        decay_sensor = DecaySensor(comprehensive_coordinator, "test_entry")

        # Initial state - 2 entities decaying
        attrs = decay_sensor.extra_state_attributes
        assert attrs["entities_decaying"] == 2

        # Stop decay on one entity
        comprehensive_coordinator.entity_manager.entities[
            "binary_sensor.motion1"
        ].decay.is_decaying = False

        attrs = decay_sensor.extra_state_attributes
        assert attrs["entities_decaying"] == 1

        # Start decay on another entity
        comprehensive_coordinator.entity_manager.entities[
            "binary_sensor.motion2"
        ].decay.is_decaying = True
        comprehensive_coordinator.entity_manager.entities[
            "binary_sensor.motion2"
        ].decay.decay_factor = 0.6

        attrs = decay_sensor.extra_state_attributes
        assert attrs["entities_decaying"] == 2

    def test_sensor_error_handling(self, comprehensive_coordinator: Mock) -> None:
        """Test sensor error handling scenarios."""
        # Test with None entity manager
        comprehensive_coordinator.entity_manager = None

        entities_sensor = EntitiesSensor(comprehensive_coordinator, "test_entry")
        decay_sensor = DecaySensor(comprehensive_coordinator, "test_entry")

        # Should handle gracefully
        assert entities_sensor.native_value == 0
        assert decay_sensor.extra_state_attributes["entities_decaying"] == 0

        # Test with entity manager that raises exceptions
        comprehensive_coordinator.entity_manager = Mock()
        comprehensive_coordinator.entity_manager.entities = Mock(
            side_effect=Exception("Test error")
        )

        # Should handle gracefully and not crash
        try:
            entities_sensor.native_value
            decay_sensor.extra_state_attributes
        except Exception:
            pytest.fail("Sensors should handle entity manager errors gracefully")
