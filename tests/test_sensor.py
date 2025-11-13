"""Tests for sensor module."""

from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.sensor import (
    AreaOccupancySensorBase,
    DecaySensor,
    EvidenceSensor,
    PriorsSensor,
    ProbabilitySensor,
    async_setup_entry,
)
from homeassistant.const import EntityCategory


# ruff: noqa: SLF001, PLC0415
class TestAreaOccupancySensorBase:
    """Test AreaOccupancySensorBase class."""

    def test_initialization(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test base sensor initialization."""
        area_name = coordinator_with_areas.get_area_names()[0]
        sensor = AreaOccupancySensorBase(coordinator_with_areas, area_name)
        assert sensor.coordinator == coordinator_with_areas

    def test_set_enabled_default(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test setting enabled default."""
        area_name = coordinator_with_areas.get_area_names()[0]
        sensor = AreaOccupancySensorBase(coordinator_with_areas, area_name)

        sensor.set_enabled_default(False)
        assert sensor._attr_entity_registry_enabled_default is False

        sensor.set_enabled_default(True)
        assert sensor._attr_entity_registry_enabled_default is True

    def test_available_property(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test available property."""
        area_name = coordinator_with_areas.get_area_names()[0]
        sensor = AreaOccupancySensorBase(coordinator_with_areas, area_name)

        coordinator_with_areas.last_update_success = True
        assert sensor.available is True

        coordinator_with_areas.last_update_success = False
        assert sensor.available is False

    def test_device_info_property(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test device_info property."""
        area_name = coordinator_with_areas.get_area_names()[0]
        sensor = AreaOccupancySensorBase(coordinator_with_areas, area_name)
        # device_info is now a method that takes area_name
        expected_device_info = coordinator_with_areas.device_info(area_name)
        assert sensor.device_info == expected_device_info


class TestPercentageSensors:
    """Test sensors that convert values to percentages."""

    @pytest.mark.parametrize(
        ("sensor_class", "coordinator_attr", "expected_name", "expected_unique_id"),
        [
            (
                PriorsSensor,
                "area_prior",
                "Prior Probability",
                "test_area_prior_probability",
            ),
            (
                ProbabilitySensor,
                "probability",
                "Occupancy Probability",
                "test_area_occupancy_probability",
            ),
        ],
    )
    def test_initialization(
        self,
        coordinator_with_areas: AreaOccupancyCoordinator,
        sensor_class,
        coordinator_attr,
        expected_name,
        expected_unique_id,
    ) -> None:
        """Test percentage sensor initialization."""
        area_name = coordinator_with_areas.get_area_names()[0]
        sensor = sensor_class(coordinator_with_areas, area_name)

        # unique_id uses area_name directly, so it will match the actual area name
        # Check that it contains the expected suffix
        assert (
            expected_unique_id.replace("test_area", area_name) in sensor.unique_id
            or sensor.unique_id == expected_unique_id
        )
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
        coordinator_with_areas: AreaOccupancyCoordinator,
        sensor_class,
        coordinator_attr,
        input_value,
        expected_percentage,
    ) -> None:
        """Test percentage conversion for different input values."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        sensor = sensor_class(coordinator_with_areas, area_name)
        # Mock area methods
        if coordinator_attr == "area_prior":
            area.area_prior = Mock(return_value=input_value)
        elif coordinator_attr == "probability":
            area.probability = Mock(return_value=input_value)
        assert sensor.native_value == expected_percentage


class TestEvidenceSensor:
    """Test EvidenceSensor class."""

    def test_initialization(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test EvidenceSensor initialization."""
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        sensor = EvidenceSensor(coordinator_with_areas_with_sensors, area_name)

        # unique_id uses area_name directly
        assert sensor.unique_id == f"{area_name}_evidence"
        assert sensor.name == "Evidence"
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC

    def test_native_value_property(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value property."""
        # Entities are already set up by the fixture
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        sensor = EvidenceSensor(coordinator_with_areas_with_sensors, area_name)
        assert sensor.native_value == 4

    def test_native_value_no_entities(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value when no entities."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        area._entities = type("obj", (object,), {"entities": {}})()
        sensor = EvidenceSensor(coordinator_with_areas, area_name)
        assert sensor.native_value == 0

    def test_extra_state_attributes(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes property."""
        coordinator_with_areas_with_sensors.data = {"test": "data"}
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        area = coordinator_with_areas_with_sensors.get_area_or_default(area_name)
        area.entities.active_entities = []
        area.entities.inactive_entities = []

        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        sensor = EvidenceSensor(coordinator_with_areas_with_sensors, area_name)
        attributes = sensor.extra_state_attributes

        assert "evidence" in attributes
        assert "no_evidence" in attributes

    def test_extra_state_attributes_no_data(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes when no coordinator data."""
        coordinator_with_areas.data = None
        area_name = coordinator_with_areas.get_area_names()[0]
        sensor = EvidenceSensor(coordinator_with_areas, area_name)
        assert sensor.extra_state_attributes == {}


class TestDecaySensor:
    """Test DecaySensor class."""

    def test_initialization(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test DecaySensor initialization."""
        area_name = coordinator_with_areas.get_area_names()[0]
        sensor = DecaySensor(coordinator_with_areas, area_name)

        # unique_id uses area_name directly
        assert sensor.unique_id == f"{area_name}_decay_status"
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
        self,
        coordinator_with_areas: AreaOccupancyCoordinator,
        decay_value: float,
        expected_percentage: float,
    ) -> None:
        """Test native_value property with different decay values."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        sensor = DecaySensor(coordinator_with_areas, area_name)
        # Mock area.decay method
        area.decay = Mock(return_value=decay_value)
        assert sensor.native_value == expected_percentage

    def test_extra_state_attributes_with_decaying_entities(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes with decaying entities."""
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        # Use the appliance entity which is already decaying in the fixture
        # The fixture already sets up decaying_entities, so we can use them directly
        sensor = DecaySensor(coordinator_with_areas_with_sensors, area_name)
        attributes = sensor.extra_state_attributes

        assert "decaying" in attributes
        assert len(attributes["decaying"]) == 1

    def test_extra_state_attributes_error_handling(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes error handling."""

        area_name = coordinator_with_areas.get_area_names()[0]
        sensor = DecaySensor(coordinator_with_areas, area_name)
        attrs = sensor.extra_state_attributes
        assert isinstance(attrs, dict)

    def test_extra_state_attributes_empty_entities(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes with empty entities."""
        from unittest.mock import PropertyMock, patch

        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        # Patch decaying_entities property to return empty list
        with patch.object(
            type(area.entities),
            "decaying_entities",
            new_callable=PropertyMock(return_value=[]),
        ):
            sensor = DecaySensor(coordinator_with_areas, area_name)
            assert sensor.extra_state_attributes == {"decaying": []}


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful setup entry.

        Note: "All Areas" sensors are now created automatically when
        at least one area exists (changed from requiring 2+ areas).
        """
        mock_async_add_entities = Mock()
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator_with_areas

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        # Should have 4 sensors per area + 3 All Areas sensors (no EvidenceSensor for All Areas)
        # With 1 area: 4 (area) + 3 (All Areas) = 7 total
        assert len(entities) == 7

        entity_types = [type(entity).__name__ for entity in entities]
        expected_types = [
            "PriorsSensor",
            "ProbabilitySensor",
            "EvidenceSensor",
            "DecaySensor",
        ]
        # All expected types should be present (from area sensors)
        for expected_type in expected_types:
            assert expected_type in entity_types

        # Verify All Areas sensors are present (no EvidenceSensor for All Areas)
        from custom_components.area_occupancy.const import ALL_AREAS_IDENTIFIER

        all_areas_entities = [
            e
            for e in entities
            if hasattr(e, "_area_name") and e._area_name == ALL_AREAS_IDENTIFIER
        ]
        area_entities = [
            e
            for e in entities
            if hasattr(e, "_area_name") and e._area_name != ALL_AREAS_IDENTIFIER
        ]

        # Should have 3 All Areas sensors (PriorsSensor, ProbabilitySensor, DecaySensor)
        assert len(all_areas_entities) == 3, (
            f"Expected 3 All Areas sensors, got {len(all_areas_entities)}"
        )
        all_areas_types = [type(e).__name__ for e in all_areas_entities]
        assert "PriorsSensor" in all_areas_types
        assert "ProbabilitySensor" in all_areas_types
        assert "DecaySensor" in all_areas_types
        # EvidenceSensor should NOT be in All Areas
        assert "EvidenceSensor" not in all_areas_types

        # Should have 4 area sensors (one for each area)
        assert len(area_entities) == 4, (
            f"Expected 4 area sensors, got {len(area_entities)}"
        )

    async def test_async_setup_entry_with_coordinator_data(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test setup entry with coordinator data."""
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator_with_areas

        mock_async_add_entities = Mock()
        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        entities = mock_async_add_entities.call_args[0][0]
        for entity in entities:
            assert entity.coordinator == coordinator_with_areas


class TestSensorIntegration:
    """Test sensor integration scenarios."""

    def test_all_sensors_with_comprehensive_data(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test all sensors with comprehensive coordinator data."""
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        area = coordinator_with_areas_with_sensors.get_area_or_default(area_name)
        # Mock area methods
        area.area_prior = Mock(return_value=0.35)
        area.probability = Mock(return_value=0.65)
        area.decay = Mock(return_value=0.15)

        sensors = [
            PriorsSensor(coordinator_with_areas_with_sensors, area_name),
            ProbabilitySensor(coordinator_with_areas_with_sensors, area_name),
            EvidenceSensor(coordinator_with_areas_with_sensors, area_name),
            DecaySensor(coordinator_with_areas_with_sensors, area_name),
        ]

        expected_values = [35.0, 65.0, 4, 85.0]  # 85.0 = (1 - 0.15) * 100
        for sensor, expected_value in zip(sensors, expected_values, strict=False):
            assert sensor.native_value == expected_value

    def test_sensor_availability_changes(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test sensor behavior when coordinator availability changes."""
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        sensors = [
            PriorsSensor(coordinator_with_areas_with_sensors, area_name),
            ProbabilitySensor(coordinator_with_areas_with_sensors, area_name),
            EvidenceSensor(coordinator_with_areas_with_sensors, area_name),
            DecaySensor(coordinator_with_areas_with_sensors, area_name),
        ]

        coordinator_with_areas_with_sensors.last_update_success = True
        for sensor in sensors:
            assert sensor.available is True

        coordinator_with_areas_with_sensors.last_update_success = False
        for sensor in sensors:
            assert sensor.available is False

    def test_sensor_value_updates(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test sensor value updates when coordinator data changes."""
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        area = coordinator_with_areas_with_sensors.get_area_or_default(area_name)
        probability_sensor = ProbabilitySensor(
            coordinator_with_areas_with_sensors, area_name
        )
        priors_sensor = PriorsSensor(coordinator_with_areas_with_sensors, area_name)
        decay_sensor = DecaySensor(coordinator_with_areas_with_sensors, area_name)

        # Initial values - mock area methods
        area.probability = Mock(return_value=0.65)
        area.area_prior = Mock(return_value=0.35)
        area.decay = Mock(return_value=0.15)

        assert probability_sensor.native_value == 65.0
        assert priors_sensor.native_value == 35.0
        assert decay_sensor.native_value == 85.0

        # Update area values
        area.probability = Mock(return_value=0.8)
        area.area_prior = Mock(return_value=0.4)
        area.decay = Mock(return_value=0.3)

        assert probability_sensor.native_value == 80.0
        assert priors_sensor.native_value == 40.0
        assert decay_sensor.native_value == 70.0

    def test_evidence_sensor_dynamic_updates(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test evidence sensor with dynamic entity updates."""
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        area = coordinator_with_areas_with_sensors.get_area_or_default(area_name)
        # Entities are already set up by the fixture
        evidence_sensor = EvidenceSensor(coordinator_with_areas_with_sensors, area_name)
        assert evidence_sensor.native_value == 4

        # Add entity
        area.entities.entities["new_sensor"] = Mock()
        assert evidence_sensor.native_value == 5

        # Remove entity (remove one of the existing entities)
        del area.entities.entities["binary_sensor.motion"]
        assert evidence_sensor.native_value == 4

    def test_decay_sensor_dynamic_updates(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test decay sensor with dynamic decay state updates."""
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.decay.decay_factor = 0.8

        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        area = coordinator_with_areas_with_sensors.get_area_or_default(area_name)
        area.entities.decaying_entities = [mock_entity1]

        decay_sensor = DecaySensor(coordinator_with_areas_with_sensors, area_name)
        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["decaying"]) == 1

        # Add another decaying entity
        mock_entity2 = Mock()
        mock_entity2.entity_id = "binary_sensor.motion2"
        mock_entity2.decay.decay_factor = 0.6

        area.entities.decaying_entities = [
            mock_entity1,
            mock_entity2,
        ]
        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["decaying"]) == 2

    def test_sensor_error_handling(
        self, coordinator_with_areas_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test sensor error handling scenarios."""
        # Test evidence sensor error handling
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        area = coordinator_with_areas_with_sensors.get_area_or_default(area_name)
        area.entities.entities = Mock()
        area.entities.entities.__len__ = Mock(side_effect=TypeError("Test error"))
        evidence_sensor = EvidenceSensor(coordinator_with_areas_with_sensors, area_name)

        with pytest.raises(TypeError):
            _ = evidence_sensor.native_value

        # Test decay sensor error handling
        area_name = coordinator_with_areas_with_sensors.get_area_names()[0]
        area = coordinator_with_areas_with_sensors.get_area_or_default(area_name)
        area.entities.decaying_entities = Mock(side_effect=Exception("Test error"))
        decay_sensor = DecaySensor(coordinator_with_areas_with_sensors, area_name)
        assert decay_sensor.extra_state_attributes == {}
