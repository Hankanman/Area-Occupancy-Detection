"""Tests for sensor module."""

from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest

from custom_components.area_occupancy.const import ALL_AREAS_IDENTIFIER
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.sensor import (
    AreaOccupancySensorBase,
    DecaySensor,
    EvidenceSensor,
    PriorsSensor,
    ProbabilitySensor,
    async_setup_entry,
)
from custom_components.area_occupancy.utils import generate_entity_unique_id
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant


# ruff: noqa: SLF001
class TestAreaOccupancySensorBase:
    """Test AreaOccupancySensorBase class."""

    def test_initialization(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test base sensor initialization."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = AreaOccupancySensorBase(area_handle=handle)
        assert sensor.coordinator == coordinator

    def test_set_enabled_default(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test setting enabled default."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = AreaOccupancySensorBase(area_handle=handle)

        sensor.set_enabled_default(False)
        assert sensor._attr_entity_registry_enabled_default is False

        sensor.set_enabled_default(True)
        assert sensor._attr_entity_registry_enabled_default is True

    def test_available_property(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test available property."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = AreaOccupancySensorBase(area_handle=handle)

        coordinator.last_update_success = True
        assert sensor.available is True

        coordinator.last_update_success = False
        assert sensor.available is False

    def test_device_info_property(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test device_info property."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = AreaOccupancySensorBase(area_handle=handle)
        # Device info should match the parent area's device info
        area = coordinator.get_area(area_name)
        assert sensor.device_info == area.device_info()


class TestPercentageSensors:
    """Test sensors that convert values to percentages."""

    @pytest.mark.parametrize(
        ("sensor_class", "expected_name"),
        [
            (
                PriorsSensor,
                "Prior Probability",
            ),
            (
                ProbabilitySensor,
                "Occupancy Probability",
            ),
        ],
    )
    def test_initialization(
        self,
        coordinator: AreaOccupancyCoordinator,
        sensor_class,
        expected_name,
    ) -> None:
        """Test percentage sensor initialization."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = sensor_class(area_handle=handle)

        expected_unique_id_new = generate_entity_unique_id(
            coordinator.entry_id,
            sensor.device_info,
            expected_name,
        )
        assert sensor.unique_id == expected_unique_id_new
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
        coordinator: AreaOccupancyCoordinator,
        sensor_class,
        coordinator_attr,
        input_value,
        expected_percentage,
    ) -> None:
        """Test percentage conversion for different input values."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        handle = coordinator.get_area_handle(area_name)
        sensor = sensor_class(area_handle=handle)
        # Mock area methods
        if coordinator_attr == "area_prior":
            area.area_prior = Mock(return_value=input_value)
        elif coordinator_attr == "probability":
            area.probability = Mock(return_value=input_value)
        assert sensor.native_value == expected_percentage


class TestEvidenceSensor:
    """Test EvidenceSensor class."""

    def test_initialization(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test EvidenceSensor initialization."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensor = EvidenceSensor(area_handle=handle)

        # unique_id uses entry_id, device_id, and entity_name
        entry_id = coordinator_with_sensors.entry_id
        area = coordinator_with_sensors.get_area(area_name)
        device_id = next(iter(area.device_info()["identifiers"]))[1]
        expected_unique_id = f"{entry_id}_{device_id}_evidence"
        assert sensor.unique_id == expected_unique_id
        assert sensor.name == "Evidence"
        assert sensor.entity_category == EntityCategory.DIAGNOSTIC

    def test_native_value_property(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value property."""
        # Entities are already set up by the fixture
        # Note: coordinator_with_sensors uses mock_realistic_config_entry which has 15 entities
        # plus the 4 added by the fixture, totaling 19.
        area_name = coordinator_with_sensors.get_area_names()[0]
        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensor = EvidenceSensor(area_handle=handle)
        assert sensor.native_value == 19

    def test_native_value_no_entities(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value when no entities."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area._entities = type("obj", (object,), {"entities": {}})()
        handle = coordinator.get_area_handle(area_name)
        sensor = EvidenceSensor(area_handle=handle)
        assert sensor.native_value == 0

    def test_extra_state_attributes(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes property."""
        coordinator_with_sensors.data = {"test": "data"}
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)

        # Use real EntityManager but mock the entities inside
        # Clear existing entities
        area.entities._entities.clear()

        # Create a mock entity that behaves correctly
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.test"
        mock_entity.name = "Test Entity"
        mock_entity.evidence = True
        mock_entity.prob_given_true = 0.9
        mock_entity.prob_given_false = 0.1
        mock_entity.weight = 0.5
        mock_entity.state = "on"
        mock_entity.decay.is_decaying = False
        mock_entity.decay.decay_factor = 1.0
        mock_entity.type.weight = 0.5

        # 'active' is used by EntityManager to filter active_entities
        mock_entity.active = True

        area.entities._entities["binary_sensor.test"] = mock_entity

        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensor = EvidenceSensor(area_handle=handle)
        attributes = sensor.extra_state_attributes

        assert "evidence" in attributes
        assert "no_evidence" in attributes

    def test_extra_state_attributes_no_data(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes when no coordinator data."""
        coordinator.data = None
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = EvidenceSensor(area_handle=handle)
        assert sensor.extra_state_attributes == {}

    def test_native_value_nonexistent_area(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value when area doesn't exist (e.g., removed during runtime)."""
        # Create sensor for a non-existent area
        # This simulates the case where an area is removed but sensor entity hasn't been cleaned up
        handle = coordinator.get_area_handle("nonexistent_area")
        sensor = EvidenceSensor(area_handle=handle)
        # Should return None instead of raising AttributeError
        assert sensor.native_value is None


class TestDecaySensor:
    """Test DecaySensor class."""

    def test_initialization(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test DecaySensor initialization."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = DecaySensor(area_handle=handle)

        # unique_id uses entry_id, device_id, and entity_name
        entry_id = coordinator.entry_id
        area = coordinator.get_area(area_name)
        device_id = next(iter(area.device_info()["identifiers"]))[1]
        expected_unique_id = f"{entry_id}_{device_id}_decay_status"
        assert sensor.unique_id == expected_unique_id
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
        coordinator: AreaOccupancyCoordinator,
        decay_value: float,
        expected_percentage: float,
    ) -> None:
        """Test native_value property with different decay values."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        handle = coordinator.get_area_handle(area_name)
        sensor = DecaySensor(area_handle=handle)
        # Mock area.decay method
        area.decay = Mock(return_value=decay_value)
        assert sensor.native_value == expected_percentage

    def test_extra_state_attributes_with_decaying_entities(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes with decaying entities."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)

        # Mock EntityManager to control decaying_entities
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.appliance"
        mock_entity.decay.decay_factor = 0.5

        area._entities = Mock()
        area._entities.decaying_entities = [mock_entity]

        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensor = DecaySensor(area_handle=handle)
        attributes = sensor.extra_state_attributes

        assert "decaying" in attributes
        assert len(attributes["decaying"]) == 1

    def test_extra_state_attributes_error_handling(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes error handling."""

        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = DecaySensor(area_handle=handle)
        attrs = sensor.extra_state_attributes
        assert isinstance(attrs, dict)

    def test_extra_state_attributes_empty_entities(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes with empty entities."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        # Patch decaying_entities property to return empty list
        with patch.object(
            type(area.entities),
            "decaying_entities",
            new_callable=PropertyMock(return_value=[]),
        ):
            handle = coordinator.get_area_handle(area_name)
            sensor = DecaySensor(area_handle=handle)
            assert sensor.extra_state_attributes == {"decaying": []}


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful setup entry.

        Note: "All Areas" sensors are now created automatically when
        at least one area exists (changed from requiring 2+ areas).
        """
        mock_async_add_entities = Mock()
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator

        await async_setup_entry(hass, mock_config_entry, mock_async_add_entities)

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
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test setup entry with coordinator data."""
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator

        mock_async_add_entities = Mock()
        await async_setup_entry(hass, mock_config_entry, mock_async_add_entities)

        entities = mock_async_add_entities.call_args[0][0]
        for entity in entities:
            assert entity.coordinator == coordinator


class TestSensorIntegration:
    """Test sensor integration scenarios."""

    def test_all_sensors_with_comprehensive_data(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test all sensors with comprehensive coordinator data."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        # Mock area methods
        area.area_prior = Mock(return_value=0.35)
        area.probability = Mock(return_value=0.65)
        area.decay = Mock(return_value=0.15)

        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensors = [
            PriorsSensor(area_handle=handle),
            ProbabilitySensor(area_handle=handle),
            EvidenceSensor(area_handle=handle),
            DecaySensor(area_handle=handle),
        ]

        expected_values = [35.0, 65.0, 19, 85.0]  # 85.0 = (1 - 0.15) * 100
        for sensor, expected_value in zip(sensors, expected_values, strict=False):
            assert sensor.native_value == expected_value

    def test_sensor_availability_changes(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test sensor behavior when coordinator availability changes."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensors = [
            PriorsSensor(area_handle=handle),
            ProbabilitySensor(area_handle=handle),
            EvidenceSensor(area_handle=handle),
            DecaySensor(area_handle=handle),
        ]

        coordinator_with_sensors.last_update_success = True
        for sensor in sensors:
            assert sensor.available is True

        coordinator_with_sensors.last_update_success = False
        for sensor in sensors:
            assert sensor.available is False

    def test_sensor_value_updates(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test sensor value updates when coordinator data changes."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        handle = coordinator_with_sensors.get_area_handle(area_name)
        probability_sensor = ProbabilitySensor(area_handle=handle)
        priors_sensor = PriorsSensor(area_handle=handle)
        decay_sensor = DecaySensor(area_handle=handle)

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

    def test_priors_sensor_extra_attributes_include_combined_prior(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Ensure priors sensor reports raw and combined priors separately."""

        coordinator_with_sensors.data = {"ready": True}
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)

        combined_value = 0.47
        area.area_prior = Mock(return_value=combined_value)

        original_prior = area._prior
        mock_prior = SimpleNamespace(
            global_prior=0.33,
            time_prior=0.21,
            day_of_week=3,
            time_slot=5,
        )
        area._prior = mock_prior

        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensor = PriorsSensor(area_handle=handle)
        try:
            attrs = sensor.extra_state_attributes
        finally:
            area._prior = original_prior

        assert attrs["global_prior"] == 0.33
        assert attrs["combined_prior"] == combined_value
        assert attrs["time_prior"] == 0.21
        assert attrs["day_of_week"] == 3
        assert attrs["time_slot"] == 5

    def test_evidence_sensor_dynamic_updates(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test evidence sensor with dynamic entity updates."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        # Entities are already set up by the fixture
        handle = coordinator_with_sensors.get_area_handle(area_name)
        evidence_sensor = EvidenceSensor(area_handle=handle)
        assert evidence_sensor.native_value == 19

        # Add entity
        area.entities.entities["new_sensor"] = Mock()
        assert evidence_sensor.native_value == 20

        # Remove entity (remove one of the existing entities)
        del area.entities.entities["binary_sensor.motion"]
        assert evidence_sensor.native_value == 19

    def test_decay_sensor_dynamic_updates(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test decay sensor with dynamic decay state updates."""
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.decay.decay_factor = 0.8

        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        # Mock EntityManager to control decaying_entities
        area._entities = Mock()
        area._entities.decaying_entities = [mock_entity1]

        handle = coordinator_with_sensors.get_area_handle(area_name)
        decay_sensor = DecaySensor(area_handle=handle)
        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["decaying"]) == 1

        # Add another decaying entity
        mock_entity2 = Mock()
        mock_entity2.entity_id = "binary_sensor.motion2"
        mock_entity2.decay.decay_factor = 0.6

        area._entities.decaying_entities = [
            mock_entity1,
            mock_entity2,
        ]
        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["decaying"]) == 2

    def test_sensor_error_handling(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test sensor error handling scenarios."""
        # Test evidence sensor error handling
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        # Mock EntityManager to simulate error
        area._entities = Mock()
        area._entities.entities = Mock()
        area._entities.entities.__len__ = Mock(side_effect=TypeError("Test error"))
        handle = coordinator_with_sensors.get_area_handle(area_name)
        evidence_sensor = EvidenceSensor(area_handle=handle)

        with pytest.raises(TypeError):
            _ = evidence_sensor.native_value

        # Test decay sensor error handling
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        # Mock EntityManager to simulate error
        area._entities = Mock()
        area._entities.decaying_entities = Mock(side_effect=Exception("Test error"))
        handle = coordinator_with_sensors.get_area_handle(area_name)
        decay_sensor = DecaySensor(area_handle=handle)
        assert decay_sensor.extra_state_attributes == {}


class TestSensorErrorHandling:
    """Test sensor error handling scenarios."""

    async def test_async_added_to_hass_device_registry_error(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_added_to_hass with device registry error."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)
        sensor.hass = hass

        # Mock device registry to raise error when accessing
        mock_registry = Mock()
        mock_registry.async_get_device.side_effect = Exception("Registry error")
        with (
            patch(
                "custom_components.area_occupancy.sensor.dr.async_get",
                return_value=mock_registry,
            ),
            pytest.raises(Exception, match="Registry error"),
        ):
            # Should handle error gracefully - exception occurs in the if block
            # The code doesn't catch exceptions, so it will propagate
            await sensor.async_added_to_hass()

    async def test_async_added_to_hass_no_device(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_added_to_hass when device doesn't exist."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)
        sensor.hass = hass

        # Mock device registry to return None
        mock_registry = Mock()
        mock_registry.async_get_device.return_value = None
        with patch(
            "custom_components.area_occupancy.sensor.dr.async_get",
            return_value=mock_registry,
        ):
            # Should handle gracefully
            await sensor.async_added_to_hass()

    def test_extra_state_attributes_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes with error."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = PriorsSensor(area_handle=handle)

        # Mock area access to raise error
        with patch.object(
            coordinator, "get_area", side_effect=KeyError("Area not found")
        ):
            attrs = sensor.extra_state_attributes
            assert attrs == {}

    def test_native_value_none_coordinator(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value when coordinator returns None."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)

        fake_area = Mock()
        fake_area.probability.return_value = None
        # Mock area retrieval to return fake area with invalid probability
        with (
            patch.object(sensor, "_get_area", return_value=fake_area),
            pytest.raises(TypeError),
        ):
            # format_float will raise TypeError when None * 100 is passed
            # None * 100 = TypeError, so the property will raise
            _ = sensor.native_value

    def test_native_value_format_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value with format_float error."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)

        # Mock format_float to raise error
        with (
            patch(
                "custom_components.area_occupancy.sensor.format_float",
                side_effect=TypeError("Format error"),
            ),
            suppress(TypeError),
        ):
            # Should handle gracefully
            _ = sensor.native_value

    def test_handle_coordinator_update_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _handle_coordinator_update with error."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)

        # Mock parent method to raise error
        with (
            patch(
                "homeassistant.helpers.update_coordinator.CoordinatorEntity._handle_coordinator_update",
                side_effect=Exception("Update error"),
            ),
            pytest.raises(Exception, match="Update error"),
        ):
            sensor._handle_coordinator_update()
