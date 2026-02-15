"""Tests for sensor module."""

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


# ruff: noqa: SLF001, PLC0415, TID251
class TestAreaOccupancySensorBase:
    """Test AreaOccupancySensorBase class."""

    def test_initialization(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test base sensor initialization."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = AreaOccupancySensorBase(area_handle=handle)
        assert sensor.coordinator == coordinator

    def test_initialization_both_none(self) -> None:
        """Test initialization raises ValueError when both area_handle and all_areas are None."""
        with pytest.raises(
            ValueError, match="area_handle or all_areas must be provided"
        ):
            AreaOccupancySensorBase(area_handle=None, all_areas=None)

    def test_initialization_both_provided(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test initialization when both area_handle and all_areas are provided - uses area_handle."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        all_areas = coordinator.get_all_areas()

        sensor = AreaOccupancySensorBase(area_handle=handle, all_areas=all_areas)

        # Should use area_handle (area_handle or all_areas evaluates to area_handle)
        assert sensor._area_handle == handle
        assert sensor._area_name == area_name  # Not ALL_AREAS_IDENTIFIER

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
        """Test extra_state_attributes property with comprehensive validation."""
        coordinator_with_sensors.data = {"test": "data"}
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)

        # Use real EntityManager but mock the entities inside
        # Clear existing entities
        area.entities._entities.clear()

        # Create active entity with evidence
        active_entity = Mock()
        active_entity.entity_id = "binary_sensor.active"
        active_entity.name = "Active Entity"
        active_entity.evidence = True
        active_entity.prob_given_true = 0.9
        active_entity.prob_given_false = 0.1
        active_entity.weight = 0.8
        active_entity.state = "on"
        active_entity.decay.is_decaying = False
        active_entity.decay.decay_factor = 1.0
        active_entity.type.weight = 0.8
        active_entity.active = True

        # Create inactive entity without evidence
        # Note: inactive_entities requires evidence=False AND decay.is_decaying=False
        inactive_entity = Mock()
        inactive_entity.entity_id = "binary_sensor.inactive"
        inactive_entity.name = "Inactive Entity"
        inactive_entity.evidence = False
        inactive_entity.prob_given_true = 0.3
        inactive_entity.prob_given_false = 0.7
        inactive_entity.weight = 0.5
        inactive_entity.state = "off"
        inactive_entity.decay.is_decaying = False  # Must be False for inactive_entities
        inactive_entity.decay.decay_factor = 0.6
        inactive_entity.type.weight = 0.5
        inactive_entity.active = False

        area.entities._entities["binary_sensor.active"] = active_entity
        area.entities._entities["binary_sensor.inactive"] = inactive_entity

        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensor = EvidenceSensor(area_handle=handle)
        attributes = sensor.extra_state_attributes

        # Verify all expected keys exist
        assert "evidence" in attributes
        assert "no_evidence" in attributes
        assert "total" in attributes
        assert "details" in attributes

        # Verify values
        assert attributes["evidence"] == "Active Entity"
        assert attributes["no_evidence"] == "Inactive Entity"
        assert attributes["total"] == 2

        # Verify details structure and sorting (evidence first, then by weight descending)
        assert len(attributes["details"]) == 2
        # First entity should be active (has evidence)
        assert attributes["details"][0]["id"] == "binary_sensor.active"
        assert attributes["details"][0]["name"] == "Active Entity"
        assert attributes["details"][0]["evidence"] is True
        assert attributes["details"][0]["prob_given_true"] == 0.9
        assert attributes["details"][0]["prob_given_false"] == 0.1
        assert attributes["details"][0]["weight"] == 0.8
        assert attributes["details"][0]["state"] == "on"
        assert attributes["details"][0]["decaying"] is False
        assert attributes["details"][0]["decay_factor"] == 1.0

        # Second entity should be inactive (no evidence)
        assert attributes["details"][1]["id"] == "binary_sensor.inactive"
        assert attributes["details"][1]["name"] == "Inactive Entity"
        assert attributes["details"][1]["evidence"] is False
        assert (
            attributes["details"][1]["decaying"] is False
        )  # Updated to match test setup
        assert attributes["details"][1]["decay_factor"] == 0.6

    @pytest.mark.parametrize(
        ("sensor_class", "use_all_areas"),
        [
            (EvidenceSensor, False),
            (PriorsSensor, False),
            (PriorsSensor, True),  # All Areas sensor
        ],
    )
    def test_extra_state_attributes_no_data(
        self,
        coordinator: AreaOccupancyCoordinator,
        sensor_class,
        use_all_areas,
    ) -> None:
        """Test extra_state_attributes when coordinator.data is None."""
        coordinator.data = None

        if use_all_areas:
            all_areas = coordinator.get_all_areas()
            sensor = sensor_class(all_areas=all_areas)
        else:
            area_name = coordinator.get_area_names()[0]
            handle = coordinator.get_area_handle(area_name)
            sensor = sensor_class(area_handle=handle)

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
        """Test extra_state_attributes with decaying entities - verify structure."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)

        # Mock EntityManager to control decaying_entities
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.appliance1"
        mock_entity1.decay.decay_factor = 0.5
        mock_entity1.decay.half_life = 300.0

        mock_entity2 = Mock()
        mock_entity2.entity_id = "binary_sensor.appliance2"
        mock_entity2.decay.decay_factor = 0.75
        mock_entity2.decay.half_life = 600.0

        area._entities = Mock()
        area._entities.decaying_entities = [mock_entity1, mock_entity2]

        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensor = DecaySensor(area_handle=handle)
        attributes = sensor.extra_state_attributes

        assert "decaying" in attributes
        assert len(attributes["decaying"]) == 2

        # Verify structure of decaying entities
        assert attributes["decaying"][0]["id"] == "binary_sensor.appliance1"
        assert attributes["decaying"][0]["decay"] == "50.00%"  # format_percentage(0.5)
        assert attributes["decaying"][0]["half_life"] == 300.0
        assert attributes["decaying"][1]["id"] == "binary_sensor.appliance2"
        assert attributes["decaying"][1]["decay"] == "75.00%"  # format_percentage(0.75)
        assert attributes["decaying"][1]["half_life"] == 600.0

    def test_extra_state_attributes_error_handling(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes error handling with actual error scenarios."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = DecaySensor(area_handle=handle)

        # Test with AttributeError - area.entities doesn't have decaying_entities
        area = coordinator.get_area(area_name)
        original_entities = area._entities
        area._entities = Mock()
        del area._entities.decaying_entities  # Remove attribute to cause AttributeError

        attrs = sensor.extra_state_attributes
        assert attrs == {}  # Should return empty dict on error

        # Restore
        area._entities = original_entities

        # Test with TypeError - decaying_entities raises TypeError
        area._entities = Mock()
        area._entities.decaying_entities = Mock(side_effect=TypeError("Test error"))
        attrs = sensor.extra_state_attributes
        assert attrs == {}  # Should return empty dict on error

        # Restore
        area._entities = original_entities

    def test_decay_sensor_extra_attributes_structure(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test DecaySensor extra_state_attributes structure validation."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        handle = coordinator.get_area_handle(area_name)
        sensor = DecaySensor(area_handle=handle)

        # Test with empty decaying entities (using patch for consistency)
        with patch.object(
            type(area.entities),
            "decaying_entities",
            new_callable=PropertyMock(return_value=[]),
        ):
            attrs = sensor.extra_state_attributes
            assert attrs == {"decaying": []}
            assert isinstance(attrs["decaying"], list)

        # Test with multiple decaying entities
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.entity1"
        mock_entity1.decay.decay_factor = 0.3
        mock_entity1.decay.half_life = 240.0

        mock_entity2 = Mock()
        mock_entity2.entity_id = "binary_sensor.entity2"
        mock_entity2.decay.decay_factor = 0.7
        mock_entity2.decay.half_life = 480.0

        area._entities = Mock()
        area._entities.decaying_entities = [mock_entity1, mock_entity2]
        attrs = sensor.extra_state_attributes

        assert "decaying" in attrs
        assert len(attrs["decaying"]) == 2
        # Verify structure of each decaying entity
        for item in attrs["decaying"]:
            assert "id" in item
            assert "decay" in item
            assert "half_life" in item
            assert isinstance(item["id"], str)
            assert isinstance(item["decay"], str)  # format_percentage returns string
            assert item["decay"].endswith("%")
            assert isinstance(item["half_life"], float)


class TestAllAreasSensors:
    """Test 'All Areas' aggregation sensors."""

    @pytest.mark.parametrize(
        ("sensor_class", "expected_name", "has_entity_category"),
        [
            (PriorsSensor, "Prior Probability", True),
            (ProbabilitySensor, "Occupancy Probability", False),
            (DecaySensor, "Decay Status", True),
        ],
    )
    def test_all_areas_sensor_initialization(
        self,
        coordinator: AreaOccupancyCoordinator,
        sensor_class,
        expected_name,
        has_entity_category,
    ) -> None:
        """Test sensor initialization with AllAreas."""
        all_areas = coordinator.get_all_areas()
        sensor = sensor_class(all_areas=all_areas)

        assert sensor._area_name == ALL_AREAS_IDENTIFIER
        assert sensor._all_areas == all_areas
        assert sensor.coordinator == coordinator
        assert sensor.name == expected_name
        assert sensor.native_unit_of_measurement == "%"

        if has_entity_category:
            assert sensor.entity_category == EntityCategory.DIAGNOSTIC
        else:
            assert (
                not hasattr(sensor, "entity_category") or sensor.entity_category is None
            )

        # Verify device info matches AllAreas device info
        expected_device_info = all_areas.device_info()
        assert sensor.device_info == expected_device_info

    @pytest.mark.parametrize(
        (
            "sensor_class",
            "method_name",
            "input_value",
            "expected_percentage",
            "create_multiple_areas",
        ),
        [
            # PriorsSensor tests
            (PriorsSensor, "area_prior", 0.35, 35.0, False),  # Single area
            (PriorsSensor, "area_prior", 0.45, 45.0, True),  # Multiple areas
            # ProbabilitySensor tests
            (ProbabilitySensor, "probability", 0.65, 65.0, False),
            # DecaySensor tests (decay value converted: (1 - decay) * 100)
            (DecaySensor, "decay", 0.25, 75.0, False),  # (1 - 0.25) * 100
        ],
    )
    def test_all_areas_sensor_native_value(
        self,
        coordinator: AreaOccupancyCoordinator,
        sensor_class,
        method_name,
        input_value,
        expected_percentage,
        create_multiple_areas,
    ) -> None:
        """Test All Areas sensor native_value with different sensor types and area counts."""
        from tests.conftest import create_test_area

        if create_multiple_areas:
            # Create additional area for multiple areas test
            create_test_area(coordinator, area_name="Kitchen", entity_ids=[])

        all_areas = coordinator.get_all_areas()
        sensor = sensor_class(all_areas=all_areas)

        # Mock the appropriate AllAreas method
        setattr(all_areas, method_name, Mock(return_value=input_value))
        assert sensor.native_value == expected_percentage

    def test_all_areas_priors_sensor_extra_attributes_single_area(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test All Areas PriorsSensor extra_state_attributes with single area."""
        coordinator.data = {"ready": True}
        all_areas = coordinator.get_all_areas()
        sensor = PriorsSensor(all_areas=all_areas)

        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Set up area prior data
        original_prior = area._prior
        mock_prior = SimpleNamespace(
            global_prior=0.33,
            time_prior=0.21,
            day_of_week=3,
            time_slot=5,
        )
        area._prior = mock_prior
        area.area_prior = Mock(return_value=0.47)

        try:
            attrs = sensor.extra_state_attributes
        finally:
            area._prior = original_prior

        assert "areas" in attrs
        assert area_name in attrs["areas"]
        assert attrs["areas"][area_name]["global_prior"] == 0.33
        assert attrs["areas"][area_name]["combined_prior"] == 0.47
        assert attrs["areas"][area_name]["time_prior"] == 0.21
        assert attrs["areas"][area_name]["day_of_week"] == 3
        assert attrs["areas"][area_name]["time_slot"] == 5

    def test_all_areas_priors_sensor_extra_attributes_multiple_areas(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test All Areas PriorsSensor extra_state_attributes with multiple areas."""
        from tests.conftest import create_test_area

        coordinator.data = {"ready": True}
        # Create second area
        create_test_area(coordinator, area_name="Kitchen", entity_ids=[])

        all_areas = coordinator.get_all_areas()
        sensor = PriorsSensor(all_areas=all_areas)

        # Set up priors for both areas
        area_names = coordinator.get_area_names()
        for area_name in area_names:
            area = coordinator.get_area(area_name)
            mock_prior = SimpleNamespace(
                global_prior=0.3,
                time_prior=0.2,
                day_of_week=1,
                time_slot=2,
            )
            area._prior = mock_prior
            area.area_prior = Mock(return_value=0.4)

        try:
            attrs = sensor.extra_state_attributes
        finally:
            # Restore original priors
            for area_name in area_names:
                area = coordinator.get_area(area_name)
                if hasattr(area, "_prior"):
                    # Restore if we can
                    pass

        assert "areas" in attrs
        assert len(attrs["areas"]) == len(area_names)
        for area_name in area_names:
            assert area_name in attrs["areas"]
            assert "global_prior" in attrs["areas"][area_name]
            assert "combined_prior" in attrs["areas"][area_name]

    def test_all_areas_decay_sensor_extra_attributes_single_area(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test All Areas DecaySensor extra_state_attributes with single area."""
        all_areas = coordinator.get_all_areas()
        sensor = DecaySensor(all_areas=all_areas)

        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock decaying entities
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.test"
        mock_entity.decay.decay_factor = 0.5
        mock_entity.decay.half_life = 300.0

        area._entities = Mock()
        area._entities.decaying_entities = [mock_entity]

        attrs = sensor.extra_state_attributes

        assert "decaying" in attrs
        assert len(attrs["decaying"]) == 1
        assert attrs["decaying"][0]["area"] == area_name
        assert attrs["decaying"][0]["id"] == "binary_sensor.test"
        assert attrs["decaying"][0]["half_life"] == 300.0

    def test_all_areas_decay_sensor_extra_attributes_multiple_areas(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test All Areas DecaySensor extra_state_attributes with multiple areas."""
        from tests.conftest import create_test_area

        # Create second area
        create_test_area(coordinator, area_name="Kitchen", entity_ids=[])

        all_areas = coordinator.get_all_areas()
        sensor = DecaySensor(all_areas=all_areas)

        # Set up decaying entities for both areas
        area_names = coordinator.get_area_names()
        for idx, area_name in enumerate(area_names):
            area = coordinator.get_area(area_name)
            mock_entity = Mock()
            mock_entity.entity_id = f"binary_sensor.test{idx}"
            mock_entity.decay.decay_factor = 0.3 + (idx * 0.2)
            mock_entity.decay.half_life = 300.0 + (idx * 100.0)

            area._entities = Mock()
            area._entities.decaying_entities = [mock_entity]

        attrs = sensor.extra_state_attributes

        assert "decaying" in attrs
        assert len(attrs["decaying"]) == len(area_names)
        # Verify all areas are represented
        decaying_areas = {item["area"] for item in attrs["decaying"]}
        assert decaying_areas == set(area_names)
        # Verify half_life is present for each decaying entity
        for item in attrs["decaying"]:
            assert "half_life" in item
            assert isinstance(item["half_life"], float)

    def test_all_areas_sensors_with_no_areas(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test All Areas sensors when coordinator has no areas (edge case)."""
        # Clear all areas
        coordinator.areas.clear()

        all_areas = coordinator.get_all_areas()
        priors_sensor = PriorsSensor(all_areas=all_areas)
        probability_sensor = ProbabilitySensor(all_areas=all_areas)
        decay_sensor = DecaySensor(all_areas=all_areas)

        # AllAreas methods return safe defaults when no areas exist
        # area_prior and probability return MIN_PROBABILITY, decay returns 1.0
        from custom_components.area_occupancy.const import MIN_PROBABILITY

        assert priors_sensor.native_value == MIN_PROBABILITY * 100
        assert probability_sensor.native_value == MIN_PROBABILITY * 100
        assert decay_sensor.native_value == 0.0  # (1 - 1.0) * 100

    def test_all_areas_sensors_native_value_none(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test All Areas sensors when _all_areas is None."""
        all_areas = coordinator.get_all_areas()
        sensor = PriorsSensor(all_areas=all_areas)

        # Simulate _all_areas being None (shouldn't happen but test edge case)
        sensor._all_areas = None
        assert sensor.native_value is None

        attrs = sensor.extra_state_attributes
        assert attrs == {}


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
        # Should have 6 sensors per area + 5 All Areas sensors (no EvidenceSensor for All Areas)
        # With 1 area: 6 (area) + 5 (All Areas) = 11 total
        # Area sensors: ProbabilitySensor, DecaySensor, PriorsSensor, EvidenceSensor,
        #               PresenceProbabilitySensor, EnvironmentalConfidenceSensor
        # All Areas: ProbabilitySensor, DecaySensor, PriorsSensor,
        #            PresenceProbabilitySensor, EnvironmentalConfidenceSensor
        assert len(entities) == 11

        entity_types = [type(entity).__name__ for entity in entities]
        expected_types = [
            "PriorsSensor",
            "ProbabilitySensor",
            "EvidenceSensor",
            "DecaySensor",
            "PresenceProbabilitySensor",
            "EnvironmentalConfidenceSensor",
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

        # Should have 5 All Areas sensors (PriorsSensor, ProbabilitySensor, DecaySensor,
        # PresenceProbabilitySensor, EnvironmentalConfidenceSensor)
        assert len(all_areas_entities) == 5, (
            f"Expected 5 All Areas sensors, got {len(all_areas_entities)}"
        )
        all_areas_types = [type(e).__name__ for e in all_areas_entities]
        assert "PriorsSensor" in all_areas_types
        assert "ProbabilitySensor" in all_areas_types
        assert "DecaySensor" in all_areas_types
        assert "PresenceProbabilitySensor" in all_areas_types
        assert "EnvironmentalConfidenceSensor" in all_areas_types
        # EvidenceSensor should NOT be in All Areas
        assert "EvidenceSensor" not in all_areas_types

        # Should have 6 area sensors (one for each area)
        assert len(area_entities) == 6, (
            f"Expected 6 area sensors, got {len(area_entities)}"
        )

        # Verify all entities have correct coordinator assignment
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

    @pytest.mark.parametrize(
        "sensor_class",
        [
            PriorsSensor,
            ProbabilitySensor,
            EvidenceSensor,
            DecaySensor,
        ],
    )
    def test_sensor_availability_changes(
        self,
        coordinator_with_sensors: AreaOccupancyCoordinator,
        sensor_class,
    ) -> None:
        """Test sensor behavior when coordinator availability changes."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        handle = coordinator_with_sensors.get_area_handle(area_name)
        sensor = sensor_class(area_handle=handle)

        coordinator_with_sensors.last_update_success = True
        assert sensor.available is True

        coordinator_with_sensors.last_update_success = False
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
        """Ensure priors sensor reports raw and combined priors separately with full validation."""

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

        # Verify all expected keys exist
        assert "global_prior" in attrs
        assert "combined_prior" in attrs
        assert "time_prior" in attrs
        assert "day_of_week" in attrs
        assert "time_slot" in attrs

        # Verify values and types
        assert attrs["global_prior"] == 0.33
        assert isinstance(attrs["global_prior"], float)
        assert attrs["combined_prior"] == combined_value
        assert isinstance(attrs["combined_prior"], float)
        assert attrs["time_prior"] == 0.21
        assert isinstance(attrs["time_prior"], float)
        assert attrs["day_of_week"] == 3
        assert isinstance(attrs["day_of_week"], int)
        assert attrs["time_slot"] == 5
        assert isinstance(attrs["time_slot"], int)

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
        mock_entity1.decay.half_life = 300.0

        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        # Mock EntityManager to control decaying_entities
        area._entities = Mock()
        area._entities.decaying_entities = [mock_entity1]

        handle = coordinator_with_sensors.get_area_handle(area_name)
        decay_sensor = DecaySensor(area_handle=handle)
        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["decaying"]) == 1
        assert attrs["decaying"][0]["half_life"] == 300.0

        # Add another decaying entity
        mock_entity2 = Mock()
        mock_entity2.entity_id = "binary_sensor.motion2"
        mock_entity2.decay.decay_factor = 0.6
        mock_entity2.decay.half_life = 600.0

        area._entities.decaying_entities = [
            mock_entity1,
            mock_entity2,
        ]
        attrs = decay_sensor.extra_state_attributes
        assert len(attrs["decaying"]) == 2
        assert attrs["decaying"][0]["half_life"] == 300.0
        assert attrs["decaying"][1]["half_life"] == 600.0

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

    async def test_async_added_to_hass_device_registry_update(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_added_to_hass updates device area_id when configured."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)
        sensor.hass = hass

        # Configure area with area_id
        area.config.area_id = "test_area_id"

        # Mock device registry
        mock_device = Mock()
        mock_device.id = "device_id"
        mock_device.area_id = "different_area_id"  # Different from config

        mock_registry = Mock()
        mock_registry.async_get_device.return_value = mock_device
        mock_registry.async_update_device = Mock()

        with patch(
            "custom_components.area_occupancy.sensor.dr.async_get",
            return_value=mock_registry,
        ):
            await sensor.async_added_to_hass()

            # Verify device was updated with correct area_id
            mock_registry.async_update_device.assert_called_once_with(
                "device_id", area_id="test_area_id"
            )

    async def test_async_added_to_hass_all_areas_skipped(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_added_to_hass skips device registry update for 'All Areas' sensors."""
        all_areas = coordinator.get_all_areas()
        sensor = ProbabilitySensor(all_areas=all_areas)
        sensor.hass = hass

        mock_registry = Mock()
        mock_registry.async_get_device = Mock()

        with patch(
            "custom_components.area_occupancy.sensor.dr.async_get",
            return_value=mock_registry,
        ):
            await sensor.async_added_to_hass()

            # Verify device registry was not accessed (All Areas sensors skip this)
            mock_registry.async_get_device.assert_not_called()

    async def test_async_added_to_hass_no_area_id_config(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_added_to_hass skips update when area_id not configured."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)
        sensor.hass = hass

        # Ensure area_id is not configured
        area.config.area_id = None

        mock_registry = Mock()
        mock_registry.async_update_device = Mock()

        with patch(
            "custom_components.area_occupancy.sensor.dr.async_get",
            return_value=mock_registry,
        ):
            await sensor.async_added_to_hass()

            # Verify device was not updated (no area_id configured)
            mock_registry.async_update_device.assert_not_called()

    def test_get_area_returns_none_when_area_removed(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _get_area() returns None when area is removed after sensor creation."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)

        # Initially should resolve to area
        area = sensor._get_area()
        assert area is not None

        # Remove area from coordinator
        del coordinator.areas[area_name]

        # Now should return None
        area = sensor._get_area()
        assert area is None

    def test_get_area_returns_none_for_all_areas_sensor(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _get_area() returns None for 'All Areas' sensors."""
        all_areas = coordinator.get_all_areas()
        sensor = ProbabilitySensor(all_areas=all_areas)

        # _get_area() should return None for All Areas sensors
        assert sensor._get_area() is None

    def test_extra_state_attributes_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes with various error scenarios."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = PriorsSensor(area_handle=handle)
        coordinator.data = {"ready": True}

        # Test with KeyError - area access raises KeyError
        area = coordinator.get_area(area_name)
        original_prior = area._prior
        try:
            # Mock area.prior to raise AttributeError
            area._prior = Mock()
            del area._prior.global_prior  # Remove attribute to cause AttributeError

            attrs = sensor.extra_state_attributes
            assert attrs == {}  # Should return empty dict on error
        finally:
            area._prior = original_prior

        # Test with TypeError - area.area_prior() raises TypeError
        area.area_prior = Mock(side_effect=TypeError("Test error"))
        attrs = sensor.extra_state_attributes
        assert attrs == {}  # Should return empty dict on error

    @pytest.mark.parametrize(
        "sensor_class",
        [
            PriorsSensor,
            ProbabilitySensor,
            DecaySensor,
        ],
    )
    def test_native_value_get_area_returns_none(
        self, coordinator: AreaOccupancyCoordinator, sensor_class
    ) -> None:
        """Test native_value when _get_area() returns None for area-based sensors."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = sensor_class(area_handle=handle)

        # Mock _get_area to return None
        with patch.object(sensor, "_get_area", return_value=None):
            assert sensor.native_value is None

    @pytest.mark.parametrize(
        "sensor_class",
        [
            PriorsSensor,
            ProbabilitySensor,
            DecaySensor,
        ],
    )
    def test_all_areas_sensor_native_value_all_areas_none(
        self, coordinator: AreaOccupancyCoordinator, sensor_class
    ) -> None:
        """Test All Areas sensor native_value when _all_areas is None."""
        all_areas = coordinator.get_all_areas()
        sensor = sensor_class(all_areas=all_areas)

        # Simulate _all_areas being None
        sensor._all_areas = None
        assert sensor.native_value is None

    def test_extra_state_attributes_malformed_coordinator_data(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test extra_state_attributes with malformed coordinator data."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = PriorsSensor(area_handle=handle)

        # Test with empty dict (should return empty dict)
        coordinator.data = {}
        attrs = sensor.extra_state_attributes
        assert attrs == {}

        # Test with None (should return empty dict)
        coordinator.data = None
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
        """Test native_value with format_float error - errors should propagate."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)

        # Mock format_float to raise error - errors should propagate, not be suppressed
        with (
            patch(
                "custom_components.area_occupancy.sensor.format_float",
                side_effect=TypeError("Format error"),
            ),
            pytest.raises(TypeError, match="Format error"),
        ):
            # format_float errors should propagate (not handled in sensor code)
            _ = sensor.native_value

    def test_handle_coordinator_update_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _handle_coordinator_update propagates errors from parent.

        Note: ProbabilitySensor._handle_coordinator_update() simply calls super(),
        so errors from the parent CoordinatorEntity should propagate. This is expected
        behavior - the sensor doesn't catch exceptions.
        """
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        sensor = ProbabilitySensor(area_handle=handle)

        # Mock parent method to raise error - should propagate
        with (
            patch(
                "homeassistant.helpers.update_coordinator.CoordinatorEntity._handle_coordinator_update",
                side_effect=Exception("Update error"),
            ),
            pytest.raises(Exception, match="Update error"),
        ):
            sensor._handle_coordinator_update()
