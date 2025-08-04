"""Tests for sensor module."""

from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.sensor import (
    AreaOccupancySensorBase,
    DecaySensor,
    EvidenceSensor,
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

    def test_set_enabled_default(self, mock_coordinator: Mock) -> None:
        """Test setting enabled default."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")

        sensor.set_enabled_default(False)
        assert sensor._attr_entity_registry_enabled_default is False

        sensor.set_enabled_default(True)
        assert sensor._attr_entity_registry_enabled_default is True

    def test_available_property(self, mock_coordinator: Mock) -> None:
        """Test available property."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")

        mock_coordinator.last_update_success = True
        assert sensor.available is True

        mock_coordinator.last_update_success = False
        assert sensor.available is False

    def test_device_info_property(self, mock_coordinator: Mock) -> None:
        """Test device_info property."""
        sensor = AreaOccupancySensorBase(mock_coordinator, "test_entry")
        assert sensor.device_info == mock_coordinator.device_info


class TestPercentageSensors:
    """Test sensors that convert values to percentages."""

    @pytest.mark.parametrize(
        ("sensor_class", "coordinator_attr", "expected_name", "expected_unique_id"),
        [
            (
                PriorsSensor,
                "area_prior",
                "Prior Probability",
                "test_entry_prior_probability",
            ),
            (
                ProbabilitySensor,
                "probability",
                "Occupancy Probability",
                "test_entry_occupancy_probability",
            ),
        ],
    )
    def test_initialization(
        self,
        mock_coordinator: Mock,
        sensor_class,
        coordinator_attr,
        expected_name,
        expected_unique_id,
    ) -> None:
        """Test percentage sensor initialization."""
        sensor = sensor_class(mock_coordinator, "test_entry")

        assert sensor.unique_id == expected_unique_id
        assert sensor.name == expected_name
        assert sensor.native_unit_of_measurement == "%"
        assert sensor.suggested_display_precision == 1

    @pytest.mark.parametrize(
        ("sensor_class", "coordinator_attr"),
        [
            (PriorsSensor, "area_prior"),
            (ProbabilitySensor, "probability"),
        ],
    )
    @pytest.mark.parametrize(
        ("input_value", "expected_percentage"),
        [
            (0.0, 0.0),
            (0.25, 25.0),
            (0.5, 50.0),
            (0.75, 75.0),
            (1.0, 100.0),
        ],
    )
    def test_native_value_conversion(
        self,
        mock_coordinator: Mock,
        sensor_class,
        coordinator_attr,
        input_value,
        expected_percentage,
    ) -> None:
        """Test percentage conversion for different input values."""
        sensor = sensor_class(mock_coordinator, "test_entry")
        setattr(mock_coordinator, coordinator_attr, input_value)
        assert sensor.native_value == expected_percentage


class TestEvidenceSensor:
    """Test EvidenceSensor class."""

    def test_initialization(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test EvidenceSensor initialization."""
        sensor = EvidenceSensor(mock_coordinator_with_sensors, "test_entry")

        assert sensor.unique_id == "test_entry_evidence"
        assert sensor.name == "Evidence"
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC

    def test_native_value_property(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test native_value property."""
        sensor = EvidenceSensor(mock_coordinator_with_sensors, "test_entry")
        assert sensor.native_value == 4

    def test_native_value_no_entities(self, mock_coordinator: Mock) -> None:
        """Test native_value when no entities."""
        mock_coordinator.entities.entities = {}
        sensor = EvidenceSensor(mock_coordinator, "test_entry")
        assert sensor.native_value == 0

    def test_extra_state_attributes(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test extra_state_attributes property."""
        mock_coordinator_with_sensors.data = {"test": "data"}
        mock_coordinator_with_sensors.entities.active_entities = []
        mock_coordinator_with_sensors.entities.inactive_entities = []

        sensor = EvidenceSensor(mock_coordinator_with_sensors, "test_entry")
        attributes = sensor.extra_state_attributes

        assert "evidence" in attributes
        assert "no_evidence" in attributes

    def test_extra_state_attributes_no_data(self, mock_coordinator: Mock) -> None:
        """Test extra_state_attributes when no coordinator data."""
        mock_coordinator.data = None
        sensor = EvidenceSensor(mock_coordinator, "test_entry")
        assert sensor.extra_state_attributes == {}


class TestDecaySensor:
    """Test DecaySensor class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test DecaySensor initialization."""
        sensor = DecaySensor(mock_coordinator, "test_entry")

        assert sensor.unique_id == "test_entry_decay_status"
        assert sensor.name == "Decay Status"
        assert sensor.native_unit_of_measurement == "%"
        assert sensor.suggested_display_precision == 1
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC

    @pytest.mark.parametrize(
        ("decay_value", "expected_percentage"),
        [
            (0.0, 100.0),  # (1 - 0.0) * 100
            (0.15, 85.0),  # (1 - 0.15) * 100
            (0.5, 50.0),  # (1 - 0.5) * 100
            (0.85, 15.0),  # (1 - 0.85) * 100
            (1.0, 0.0),  # (1 - 1.0) * 100
        ],
    )
    def test_native_value_property(
        self, mock_coordinator: Mock, decay_value: float, expected_percentage: float
    ) -> None:
        """Test native_value property with different decay values."""
        sensor = DecaySensor(mock_coordinator, "test_entry")
        mock_coordinator.decay = decay_value
        assert sensor.native_value == expected_percentage

    def test_extra_state_attributes_with_decaying_entities(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test extra_state_attributes with decaying entities."""
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.decay.decay_factor = 0.8

        mock_coordinator_with_sensors.entities.decaying_entities = [mock_entity]

        sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")
        attributes = sensor.extra_state_attributes

        assert "decaying" in attributes
        assert len(attributes["decaying"]) == 1

    def test_extra_state_attributes_error_handling(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes error handling."""
        mock_coordinator.entities.decaying_entities = Mock(
            side_effect=AttributeError("No entities")
        )
        sensor = DecaySensor(mock_coordinator, "test_entry")
        assert sensor.extra_state_attributes == {}

    def test_extra_state_attributes_empty_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test extra_state_attributes with empty entities."""
        mock_coordinator.entities.decaying_entities = []
        sensor = DecaySensor(mock_coordinator, "test_entry")
        assert sensor.extra_state_attributes == {"decaying": []}


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful setup entry."""
        mock_async_add_entities = Mock()
        mock_coordinator = Mock()
        mock_coordinator.device_info = {"test": "device_info"}
        mock_config_entry.runtime_data = mock_coordinator

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 4

        entity_types = [type(entity).__name__ for entity in entities]
        expected_types = [
            "PriorsSensor",
            "ProbabilitySensor",
            "EvidenceSensor",
            "DecaySensor",
        ]
        for expected_type in expected_types:
            assert expected_type in entity_types

    async def test_async_setup_entry_with_coordinator_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup entry with coordinator data."""
        mock_coordinator = Mock()
        mock_coordinator.probability = 0.8
        mock_coordinator.prior = 0.3
        mock_coordinator.decay = 0.9
        mock_coordinator.device_info = {"test": "device_info"}
        mock_config_entry.runtime_data = mock_coordinator

        mock_async_add_entities = Mock()
        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        entities = mock_async_add_entities.call_args[0][0]
        for entity in entities:
            assert entity.coordinator == mock_coordinator


class TestSensorIntegration:
    """Test sensor integration scenarios."""

    def test_all_sensors_with_comprehensive_data(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test all sensors with comprehensive coordinator data."""
        mock_coordinator_with_sensors.area_prior = 0.35
        mock_coordinator_with_sensors.probability = 0.65
        mock_coordinator_with_sensors.decay = 0.15

        sensors = [
            PriorsSensor(mock_coordinator_with_sensors, "test_entry"),
            ProbabilitySensor(mock_coordinator_with_sensors, "test_entry"),
            EvidenceSensor(mock_coordinator_with_sensors, "test_entry"),
            DecaySensor(mock_coordinator_with_sensors, "test_entry"),
        ]

        expected_values = [35.0, 65.0, 4, 85.0]  # 85.0 = (1 - 0.15) * 100
        for sensor, expected_value in zip(sensors, expected_values, strict=False):
            assert sensor.native_value == expected_value

    def test_sensor_availability_changes(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test sensor behavior when coordinator availability changes."""
        sensors = [
            PriorsSensor(mock_coordinator_with_sensors, "test_entry"),
            ProbabilitySensor(mock_coordinator_with_sensors, "test_entry"),
            EvidenceSensor(mock_coordinator_with_sensors, "test_entry"),
            DecaySensor(mock_coordinator_with_sensors, "test_entry"),
        ]

        mock_coordinator_with_sensors.last_update_success = True
        for sensor in sensors:
            assert sensor.available is True

        mock_coordinator_with_sensors.last_update_success = False
        for sensor in sensors:
            assert sensor.available is False

    def test_sensor_value_updates(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test sensor value updates when coordinator data changes."""
        probability_sensor = ProbabilitySensor(
            mock_coordinator_with_sensors, "test_entry"
        )
        priors_sensor = PriorsSensor(mock_coordinator_with_sensors, "test_entry")
        decay_sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")

        # Initial values
        mock_coordinator_with_sensors.probability = 0.65
        mock_coordinator_with_sensors.area_prior = 0.35
        mock_coordinator_with_sensors.decay = 0.15

        assert probability_sensor.native_value == 65.0
        assert priors_sensor.native_value == 35.0
        assert decay_sensor.native_value == 85.0

        # Update coordinator values
        mock_coordinator_with_sensors.probability = 0.8
        mock_coordinator_with_sensors.area_prior = 0.4
        mock_coordinator_with_sensors.decay = 0.3

        assert probability_sensor.native_value == 80.0
        assert priors_sensor.native_value == 40.0
        assert decay_sensor.native_value == 70.0

    def test_evidence_sensor_dynamic_updates(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test evidence sensor with dynamic entity updates."""
        evidence_sensor = EvidenceSensor(mock_coordinator_with_sensors, "test_entry")
        assert evidence_sensor.native_value == 4

        # Add entity
        mock_coordinator_with_sensors.entities.entities["new_sensor"] = Mock()
        assert evidence_sensor.native_value == 5

        # Remove entity
        del mock_coordinator_with_sensors.entities.entities["binary_sensor.motion1"]
        assert evidence_sensor.native_value == 4

    def test_decay_sensor_dynamic_updates(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decay sensor with dynamic decay state updates."""
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.decay.decay_factor = 0.8

        mock_coordinator_with_sensors.entities.decaying_entities = [mock_entity1]

        decay_sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")
        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["decaying"]) == 1

        # Add another decaying entity
        mock_entity2 = Mock()
        mock_entity2.entity_id = "binary_sensor.motion2"
        mock_entity2.decay.decay_factor = 0.6

        mock_coordinator_with_sensors.entities.decaying_entities = [
            mock_entity1,
            mock_entity2,
        ]
        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["decaying"]) == 2

    def test_sensor_error_handling(self, mock_coordinator_with_sensors: Mock) -> None:
        """Test sensor error handling scenarios."""
        # Test evidence sensor error handling
        mock_coordinator_with_sensors.entities.entities = Mock(
            side_effect=Exception("Test error")
        )
        evidence_sensor = EvidenceSensor(mock_coordinator_with_sensors, "test_entry")

        with pytest.raises(RuntimeError):
            _ = evidence_sensor.native_value

        # Test decay sensor error handling
        mock_coordinator_with_sensors.entities.decaying_entities = Mock(
            side_effect=Exception("Test error")
        )
        decay_sensor = DecaySensor(mock_coordinator_with_sensors, "test_entry")
        assert decay_sensor.extra_state_attributes == {}
