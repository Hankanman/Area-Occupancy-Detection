"""Tests for binary_sensor module."""

from datetime import timedelta
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.binary_sensor import (
    Occupancy,
    WaspInBoxSensor,
    async_setup_entry,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util

# Add marker for tests that may have lingering timers due to HA internals
pytestmark = [pytest.mark.parametrize("expected_lingering_timers", [True])]


# ruff: noqa: SLF001, PLC0415
class TestOccupancy:
    """Test Occupancy binary sensor entity."""

    def test_initialization(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test Occupancy entity initialization."""
        area_name = coordinator_with_areas.get_area_names()[0]
        entity = Occupancy(coordinator_with_areas, area_name)

        assert entity.coordinator == coordinator_with_areas
        # unique_id uses area_name directly
        assert entity.unique_id == f"{area_name}_occupancy_status"
        assert entity.name == "Occupancy Status"

    async def test_async_added_to_hass(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test entity added to Home Assistant."""
        area_name = coordinator_with_areas.get_area_names()[0]
        entity = Occupancy(coordinator_with_areas, area_name)

        # Mock parent method
        with patch(
            "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass"
        ) as mock_parent:
            await entity.async_added_to_hass()
            mock_parent.assert_called_once()

        # Should set occupancy entity ID in area
        area = coordinator_with_areas.get_area_or_default(area_name)
        assert area.occupancy_entity_id == entity.entity_id

    async def test_async_will_remove_from_hass(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test entity removal from Home Assistant."""
        area_name = coordinator_with_areas.get_area_names()[0]
        entity = Occupancy(coordinator_with_areas, area_name)
        # Set entity_id first
        entity.entity_id = (
            f"binary_sensor.{area_name.lower().replace(' ', '_')}_occupancy_status"
        )
        area = coordinator_with_areas.get_area_or_default(area_name)
        area.occupancy_entity_id = entity.entity_id

        await entity.async_will_remove_from_hass()

        # Should clear occupancy entity ID in area
        assert area.occupancy_entity_id is None

    @pytest.mark.parametrize(
        ("occupied", "expected_icon", "expected_is_on"),
        [
            (True, "mdi:home-account", True),
            (False, "mdi:home-outline", False),
        ],
    )
    def test_state_properties(
        self,
        coordinator_with_areas: AreaOccupancyCoordinator,
        occupied: bool,
        expected_icon: str,
        expected_is_on: bool,
    ) -> None:
        """Test icon and is_on properties based on occupancy state."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        entity = Occupancy(coordinator_with_areas, area_name)
        # Mock area.occupied method
        area.occupied = Mock(return_value=occupied)

        assert entity.icon == expected_icon
        assert entity.is_on is expected_is_on


# Shared fixtures for WaspInBoxSensor tests
@pytest.fixture
def wasp_coordinator(
    coordinator_with_areas: AreaOccupancyCoordinator,
) -> AreaOccupancyCoordinator:
    """Create a coordinator with wasp-specific configuration."""
    from unittest.mock import Mock

    from custom_components.area_occupancy.data.config import Sensors

    # Customize the coordinator for wasp tests - use area-based access
    area_name = coordinator_with_areas.get_area_names()[0]
    area = coordinator_with_areas.get_area_or_default(area_name)
    area.config.wasp_in_box = Mock()
    area.config.wasp_in_box.enabled = True
    area.config.wasp_in_box.motion_timeout = 60
    area.config.wasp_in_box.max_duration = 3600
    area.config.wasp_in_box.weight = 0.85
    # Create a Sensors object with door and motion sensors
    area.config.sensors = Sensors(
        door=["binary_sensor.door1"],
        motion=["binary_sensor.motion1"],
        _parent_config=area.config,
    )

    # Add missing entities attribute with AsyncMock
    area.entities.async_initialize = AsyncMock()

    return coordinator_with_areas


@pytest.fixture
def wasp_config_entry(mock_config_entry: Mock) -> Mock:
    """Create a config entry with wasp-specific data."""
    mock_config_entry.data.update(
        {
            "door_sensors": ["binary_sensor.door1"],
            "motion_sensors": ["binary_sensor.motion1"],
        }
    )
    return mock_config_entry


def create_wasp_entity(
    wasp_coordinator: AreaOccupancyCoordinator, wasp_config_entry: Mock
) -> WaspInBoxSensor:
    """Create a WaspInBoxSensor with common setup."""
    # WaspInBoxSensor takes (coordinator, area_name, config_entry)
    area_name = wasp_coordinator.get_area_names()[0]
    entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)
    entity.entity_id = "binary_sensor.test_wasp_in_box"
    return entity


class TestWaspInBoxSensor:
    """Test WaspInBoxSensor binary sensor entity."""

    def test_initialization(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test WaspInBoxSensor initialization."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)

        # Set hass (normally done by HA when entity is added)
        entity.hass = mock_hass

        assert entity.hass == mock_hass
        assert entity._coordinator == wasp_coordinator
        # unique_id uses area_name directly
        assert entity.unique_id == f"{area_name}_wasp_in_box"
        assert entity.name == "Wasp in Box"
        assert entity.should_poll is False

    async def test_async_added_to_hass(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test entity added to Home Assistant."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)

        # Mock state restoration and setup methods
        with (
            patch.object(entity, "_restore_previous_state") as mock_restore,
            patch.object(entity, "_setup_entity_tracking") as mock_setup,
        ):
            await entity.async_added_to_hass()

            mock_restore.assert_called_once()
            mock_setup.assert_called_once()

        # Should set wasp entity ID in area
        area_name = wasp_coordinator.get_area_names()[0]
        area = wasp_coordinator.get_area_or_default(area_name)
        assert area.wasp_entity_id == entity.entity_id

    @pytest.mark.parametrize(
        ("has_previous_state", "expected_is_on"),
        [
            (False, False),
            (True, True),
        ],
    )
    async def test_restore_previous_state(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        has_previous_state: bool,
        expected_is_on: bool,
    ) -> None:
        """Test restoring previous state with and without stored data."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)

        if has_previous_state:
            # Mock previous state
            mock_state = Mock()
            mock_state.state = STATE_ON
            mock_state.attributes = {
                "last_occupied_time": "2023-01-01T12:00:00+00:00",
                "last_door_time": "2023-01-01T11:59:00+00:00",
                "last_motion_time": "2023-01-01T11:58:00+00:00",
            }
            mock_get_state = AsyncMock(return_value=mock_state)
            mock_timer = Mock()
        else:
            mock_get_state = AsyncMock(return_value=None)
            mock_timer = Mock()

        with (
            patch.object(entity, "async_get_last_state", mock_get_state),
            patch.object(entity, "_start_max_duration_timer", mock_timer),
        ):
            await entity._restore_previous_state()

            # Should have expected state
            assert entity._attr_is_on is expected_is_on
            if has_previous_state:
                assert entity._state == STATE_ON
                assert entity._last_occupied_time is not None
                mock_timer.assert_called_once()

    async def test_async_will_remove_from_hass(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test entity removal from Home Assistant."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)

        # Set up some state to clean up
        entity._remove_timer = Mock()
        listener_mock = Mock()
        entity._remove_state_listener = listener_mock
        area_name = wasp_coordinator.get_area_names()[0]
        area = wasp_coordinator.get_area_or_default(area_name)
        area.wasp_entity_id = entity.entity_id

        await entity.async_will_remove_from_hass()

        # Should clean up resources if listener exists
        if listener_mock:
            listener_mock.assert_called_once()
        area_name = wasp_coordinator.get_area_names()[0]
        area = wasp_coordinator.get_area_or_default(area_name)
        assert area.wasp_entity_id is None

    def test_extra_state_attributes(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test extra state attributes."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)

        # Set up some state
        entity._last_occupied_time = dt_util.utcnow()
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_ON

        attributes = entity.extra_state_attributes

        expected_attrs = [
            "door_state",
            "motion_state",
            "last_motion_time",
            "last_door_time",
            "last_occupied_time",
            "motion_timeout",
            "max_duration",
        ]
        for attr in expected_attrs:
            assert attr in attributes

    def test_weight_property(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test weight property."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)
        assert entity.weight == 0.85

    def test_get_valid_entities(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _get_valid_entities method."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)
        entity.hass = mock_hass

        # Mock hass.states.get to return valid states for configured entities
        def mock_get_state(entity_id: str) -> Mock | None:
            if entity_id in ["binary_sensor.door1", "binary_sensor.motion1"]:
                return Mock(state=STATE_OFF)
            return None

        mock_hass.states.get.side_effect = mock_get_state

        result = entity._get_valid_entities()

        assert "doors" in result
        assert "motion" in result
        assert "all" in result
        assert "binary_sensor.door1" in result["doors"]
        assert "binary_sensor.motion1" in result["motion"]

    def test_initialize_from_current_states(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _initialize_from_current_states method."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = mock_hass

        valid_entities = {
            "doors": ["binary_sensor.door1"],
            "motion": ["binary_sensor.motion1"],
        }

        # Mock current states
        mock_hass.states.get.side_effect = lambda entity_id: Mock(state=STATE_OFF)

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            entity._initialize_from_current_states(valid_entities)

        # Should initialize state tracking
        assert entity._door_state == STATE_OFF
        assert entity._motion_state == STATE_OFF

    @pytest.mark.parametrize(
        ("entity_type", "new_state", "expected_method"),
        [
            ("binary_sensor.door1", STATE_ON, "_process_door_state"),
            ("binary_sensor.motion1", STATE_ON, "_process_motion_state"),
        ],
    )
    def test_handle_state_change(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        entity_type: str,
        new_state: str,
        expected_method: str,
    ) -> None:
        """Test handling state changes for different entity types."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)

        # Set up initial state - occupied for door test
        if entity_type == "binary_sensor.door1":
            entity._attr_is_on = True

        # Mock event
        mock_event = Mock()
        mock_event.data = {
            "entity_id": entity_type,
            "new_state": Mock(state=new_state),
            "old_state": Mock(state=STATE_OFF),
        }

        with patch.object(entity, expected_method) as mock_process:
            entity._handle_state_change(mock_event)
            mock_process.assert_called_once_with(entity_type, new_state)

    @pytest.mark.parametrize(
        ("initial_occupied", "door_state", "expected_state"),
        [
            (True, STATE_ON, STATE_OFF),  # Door opens when occupied -> unoccupied
            (False, STATE_OFF, STATE_ON),  # Door closes with motion -> occupied
        ],
    )
    def test_process_door_state_scenarios(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        initial_occupied: bool,
        door_state: str,
        expected_state: str,
    ) -> None:
        """Test processing door state changes in different scenarios."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = mock_hass

        # Set up initial state
        entity._attr_is_on = initial_occupied
        entity._state = STATE_ON if initial_occupied else STATE_OFF
        entity._door_state = STATE_OFF if initial_occupied else STATE_ON

        # For the door closing scenario, add recent motion
        if not initial_occupied:
            entity._motion_state = STATE_ON
            entity._last_motion_time = dt_util.utcnow() - timedelta(seconds=30)

        # Mock hass.states.get to return appropriate door state for aggregate calculation
        mock_hass.states.get.return_value = Mock(state=door_state)

        # Mock async_write_ha_state to avoid entity registration issues
        with (
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            entity._process_door_state("binary_sensor.door1", door_state)
            mock_set_state.assert_called_once_with(expected_state)

    def test_process_motion_state_detected(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test processing motion detection."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = mock_hass

        # Mock hass.states.get to return motion as ON for aggregate calculation
        mock_hass.states.get.return_value = Mock(state=STATE_ON)

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

        # Should update motion state and time
        assert entity._motion_state == STATE_ON
        assert entity._last_motion_time is not None

    @pytest.mark.parametrize(
        ("new_state", "expected_actions"),
        [
            (STATE_ON, ["start_timer", "write_state"]),
            (STATE_OFF, ["cancel_timer", "write_state"]),
        ],
    )
    def test_set_state_scenarios(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        new_state: str,
        expected_actions: list[str],
    ) -> None:
        """Test setting state to occupied and unoccupied."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)

        # Set up initial state for unoccupied test
        if new_state == STATE_OFF:
            entity._attr_is_on = True
            entity._last_occupied_time = dt_util.utcnow()
            entity._remove_timer = Mock()

        with (
            patch.object(entity, "_start_max_duration_timer") as mock_start_timer,
            patch.object(entity, "_cancel_max_duration_timer") as mock_cancel_timer,
            patch.object(entity, "async_write_ha_state") as mock_write_state,
        ):
            entity._set_state(new_state)

            # Check state was set correctly
            assert entity._attr_is_on is (new_state == STATE_ON)

            # Check expected actions were called
            if "start_timer" in expected_actions:
                mock_start_timer.assert_called_once()
                assert entity._last_occupied_time is not None
            if "cancel_timer" in expected_actions:
                mock_cancel_timer.assert_called_once()
            if "write_state" in expected_actions:
                mock_write_state.assert_called_once()

    def test_timer_management(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test timer start, cancel, and timeout handling."""
        area_name = wasp_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(wasp_coordinator, area_name, wasp_config_entry)

        # Test starting timer
        entity._max_duration = 3600
        entity._last_occupied_time = dt_util.utcnow()

        with patch(
            "custom_components.area_occupancy.binary_sensor.async_track_point_in_time"
        ) as mock_track:
            entity._start_max_duration_timer()
            mock_track.assert_called_once()
            assert entity._remove_timer is not None

        # Test canceling timer
        timer_mock = Mock()
        entity._remove_timer = timer_mock
        entity._cancel_max_duration_timer()
        timer_mock.assert_called_once()
        assert entity._remove_timer is None

        # Test timeout handling
        entity._state = STATE_ON
        with patch.object(entity, "_reset_after_max_duration") as mock_reset:
            entity._handle_max_duration_timeout(dt_util.utcnow())
            mock_reset.assert_called_once()
            assert entity._remove_timer is None

        # Test reset after timeout
        with patch.object(entity, "_set_state") as mock_set_state:
            entity._reset_after_max_duration()
            mock_set_state.assert_called_once_with(STATE_OFF)


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @pytest.fixture
    def setup_config_entry(
        self, mock_config_entry: Mock, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> Mock:
        """Create a config entry for setup tests."""
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator_with_areas
        # Configure wasp setting on the area
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        from unittest.mock import Mock

        area.config.wasp_in_box = Mock()
        area.config.wasp_in_box.enabled = True
        return mock_config_entry

    @pytest.mark.parametrize(
        ("wasp_enabled", "expected_entity_count", "expected_types"),
        [
            (
                True,
                3,
                [Occupancy, WaspInBoxSensor, Occupancy],
            ),  # Area + Wasp + All Areas
            (False, 2, [Occupancy, Occupancy]),  # Area + All Areas
        ],
    )
    async def test_async_setup_entry(
        self,
        mock_hass: Mock,
        setup_config_entry: Mock,
        wasp_enabled: bool,
        expected_entity_count: int,
        expected_types: list,
    ) -> None:
        """Test setup entry with wasp enabled and disabled.

        Note: "All Areas" occupancy sensor is now created automatically when
        at least one area exists (changed from requiring 2+ areas).
        """
        # Configure wasp setting on the area
        coordinator = setup_config_entry.runtime_data
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area_or_default(area_name)
        area.config.wasp_in_box.enabled = wasp_enabled

        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, setup_config_entry, mock_async_add_entities)

        # Should add expected entities
        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == expected_entity_count

        # Check that we have the expected types (order may vary)
        entity_types = [type(entity).__name__ for entity in entities]
        expected_type_names = [t.__name__ for t in expected_types]
        for expected_type_name in expected_type_names:
            assert expected_type_name in entity_types, (
                f"Expected {expected_type_name} in {entity_types}"
            )

        # Verify we have exactly the right number of each type
        occupancy_count = sum(1 for e in entities if isinstance(e, Occupancy))
        wasp_count = sum(1 for e in entities if isinstance(e, WaspInBoxSensor))

        if wasp_enabled:
            assert occupancy_count == 2, (
                "Should have 2 Occupancy sensors (area + All Areas)"
            )
            assert wasp_count == 1, "Should have 1 WaspInBoxSensor"
        else:
            assert occupancy_count == 2, (
                "Should have 2 Occupancy sensors (area + All Areas)"
            )
            assert wasp_count == 0, "Should have no WaspInBoxSensor"


class TestWaspInBoxIntegration:
    """Test WaspInBoxSensor integration scenarios."""

    @pytest.fixture
    def comprehensive_wasp_sensor(
        self,
        mock_hass: Mock,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> WaspInBoxSensor:
        """Create a comprehensive wasp sensor for testing."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = mock_hass

        # Initialize with known state
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_OFF
        entity._attr_is_on = False

        return entity

    def test_complete_wasp_occupancy_cycle(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test complete wasp occupancy detection cycle."""
        entity = comprehensive_wasp_sensor

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            # Step 1: Motion detected while unoccupied
            # Mock hass.states.get to return motion as ON
            cast(Mock, entity.hass.states.get).return_value = Mock(state=STATE_ON)
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

            # Should update motion state
            assert entity._motion_state == STATE_ON

            # Step 2: Door closes with recent motion -> should trigger occupancy
            # Mock hass.states.get to return door as OFF (closed)
            cast(Mock, entity.hass.states.get).return_value = Mock(state=STATE_OFF)
            with patch.object(entity, "_start_max_duration_timer") as mock_start_timer:
                entity._process_door_state("binary_sensor.door1", STATE_OFF)

            assert entity._attr_is_on is True
            assert entity._last_occupied_time is not None
            mock_start_timer.assert_called_once()

            # Step 3: Door opens while occupied -> should end occupancy
            # Mock hass.states.get to return door as ON (open)
            cast(Mock, entity.hass.states.get).return_value = Mock(state=STATE_ON)
            with patch.object(entity, "_cancel_max_duration_timer"):
                entity._process_door_state("binary_sensor.door1", STATE_ON)

            assert entity._attr_is_on is False

    def test_wasp_timeout_scenarios(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test various timeout scenarios."""
        entity = comprehensive_wasp_sensor

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            # Test motion timeout - old motion shouldn't trigger occupancy
            old_motion_time = dt_util.utcnow() - timedelta(seconds=120)  # 2 minutes ago
            entity._last_motion_time = old_motion_time
            entity._motion_state = STATE_OFF  # Motion is not active

            entity._process_door_state("binary_sensor.door1", STATE_OFF)

            # Should not trigger occupancy due to no active motion
            assert entity._attr_is_on is False

        # Test max duration timeout
        entity._attr_is_on = True
        entity._state = STATE_ON
        entity._last_occupied_time = dt_util.utcnow()

        # Mock the _set_state method since the actual implementation calls it
        with patch.object(entity, "_set_state") as mock_set_state:
            entity._handle_max_duration_timeout(dt_util.utcnow())

        mock_set_state.assert_called_once_with(STATE_OFF)

    def test_wasp_state_persistence(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test state persistence across restarts."""
        entity = comprehensive_wasp_sensor

        # Set up occupied state
        entity._attr_is_on = True
        entity._last_occupied_time = dt_util.utcnow()
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_ON

        # Get state attributes for persistence
        attributes = entity.extra_state_attributes

        # Verify all necessary state is included
        expected_attrs = ["last_occupied_time", "door_state", "motion_state"]
        for attr in expected_attrs:
            assert attr in attributes
        assert attributes["door_state"] == STATE_OFF
        assert attributes["motion_state"] == STATE_ON

    def test_error_handling_during_state_changes(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test error handling during state changes."""
        entity = comprehensive_wasp_sensor

        # Mock an error during state writing
        with (
            patch.object(
                entity, "async_write_ha_state", side_effect=Exception("Write failed")
            ),
            pytest.raises(Exception, match="Write failed"),
        ):
            entity._set_state(STATE_ON)

        # State should still be updated internally even if write fails
        assert entity._attr_is_on is True


class TestWaspMultiSensorAggregation:
    """Test WaspInBoxSensor with multiple door and motion sensors."""

    @pytest.fixture
    def multi_sensor_coordinator(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> AreaOccupancyCoordinator:
        """Create a coordinator with multiple door and motion sensors."""
        from unittest.mock import Mock

        from custom_components.area_occupancy.data.config import Sensors

        # Use area-based access
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        area.config.wasp_in_box = Mock()
        area.config.wasp_in_box.enabled = True
        area.config.wasp_in_box.motion_timeout = 60
        area.config.wasp_in_box.max_duration = 3600
        area.config.wasp_in_box.weight = 0.85
        area.config.wasp_in_box.verification_delay = 0
        # Create a Sensors object with multiple door and motion sensors
        area.config.sensors = Sensors(
            door=["binary_sensor.door1", "binary_sensor.door2"],
            motion=["binary_sensor.motion1", "binary_sensor.motion2"],
            _parent_config=area.config,
        )
        # Use area-based access - entities are now per-area
        area.entities.async_initialize = AsyncMock()
        return coordinator_with_areas

    @pytest.fixture
    def multi_sensor_wasp(
        self,
        mock_hass: Mock,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> WaspInBoxSensor:
        """Create a wasp sensor with multiple door and motion sensors."""
        area_name = multi_sensor_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(multi_sensor_coordinator, area_name, wasp_config_entry)
        entity.hass = mock_hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Mock states for multiple sensors (all sensors default to OFF)
        mock_hass.states.get.side_effect = lambda entity_id: Mock(state=STATE_OFF)

        # Initialize state
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_OFF
        entity._attr_is_on = False
        entity._state = STATE_OFF

        return entity

    def test_aggregate_door_state_all_closed(
        self, mock_hass: Mock, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that aggregate door state is CLOSED when all doors are closed."""
        entity = multi_sensor_wasp

        # Mock all doors as closed
        def get_state(entity_id):
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state

        result = entity._get_aggregate_door_state()
        assert result == STATE_OFF  # All doors closed

    def test_aggregate_door_state_any_open(
        self, mock_hass: Mock, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that aggregate door state is OPEN if ANY door is open."""
        entity = multi_sensor_wasp

        # Mock door1 open, door2 closed
        def get_state(entity_id):
            if entity_id == "binary_sensor.door1":
                return Mock(state=STATE_ON)
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state

        result = entity._get_aggregate_door_state()
        assert result == STATE_ON  # Any door open

    def test_aggregate_motion_state_all_off(
        self, mock_hass: Mock, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that aggregate motion state is OFF when all motion sensors are off."""
        entity = multi_sensor_wasp

        # Mock all motion sensors as off
        def get_state(entity_id):
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state

        result = entity._get_aggregate_motion_state()
        assert result == STATE_OFF  # All motion off

    def test_aggregate_motion_state_any_on(
        self, mock_hass: Mock, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that aggregate motion state is ON if ANY motion sensor is active."""
        entity = multi_sensor_wasp

        # Mock motion1 active, motion2 off
        def get_state(entity_id):
            if entity_id == "binary_sensor.motion1":
                return Mock(state=STATE_ON)
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state

        result = entity._get_aggregate_motion_state()
        assert result == STATE_ON  # Any motion active

    def test_multi_door_any_opening_clears_occupancy(
        self, mock_hass: Mock, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that opening ANY door clears occupancy."""
        entity = multi_sensor_wasp

        # Set up occupied state with all doors closed
        entity._state = STATE_ON
        entity._attr_is_on = True
        entity._door_state = STATE_OFF

        # Mock door1 opening, door2 staying closed
        def get_state(entity_id):
            if entity_id == "binary_sensor.door1":
                return Mock(state=STATE_ON)  # Door 1 opens
            return Mock(state=STATE_OFF)  # Door 2 stays closed

        mock_hass.states.get.side_effect = get_state

        with (
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_cancel_verification_timer"),
        ):
            entity._process_door_state("binary_sensor.door1", STATE_ON)

        # Should be unoccupied because door1 opened
        assert entity._attr_is_on is False
        assert entity._state == STATE_OFF

    def test_multi_door_all_closed_with_motion(
        self, mock_hass: Mock, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that occupancy triggers when all doors are closed with motion."""
        entity = multi_sensor_wasp

        # Set up unoccupied state with motion active
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._motion_state = STATE_ON

        # Mock all doors as closed
        def get_state(entity_id):
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state

        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_door_state("binary_sensor.door2", STATE_OFF)

        # Should be occupied because all doors closed with motion
        assert entity._attr_is_on is True
        assert entity._state == STATE_ON

    def test_multi_motion_any_triggers_occupancy(
        self, mock_hass: Mock, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that ANY motion sensor triggers occupancy with doors closed."""
        entity = multi_sensor_wasp

        # Set up unoccupied state with doors closed
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._door_state = STATE_OFF

        # Mock motion2 activating, motion1 staying off
        def get_state(entity_id):
            if entity_id == "binary_sensor.motion2":
                return Mock(state=STATE_ON)  # Motion 2 activates
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state

        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_motion_state("binary_sensor.motion2", STATE_ON)

        # Should be occupied because motion2 detected with doors closed
        assert entity._attr_is_on is True
        assert entity._state == STATE_ON

    def test_multi_sensor_complete_cycle(
        self, mock_hass: Mock, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test complete occupancy cycle with multiple sensors."""
        entity = multi_sensor_wasp

        # Step 1: Both doors closed, motion1 triggers
        def get_state_step1(entity_id):
            if entity_id == "binary_sensor.motion1":
                return Mock(state=STATE_ON)
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state_step1

        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

        assert entity._attr_is_on is True

        # Step 2: Door1 opens (door2 still closed)
        def get_state_step2(entity_id):
            if entity_id == "binary_sensor.door1":
                return Mock(state=STATE_ON)
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state_step2

        with (
            patch.object(entity, "_cancel_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_door_state("binary_sensor.door1", STATE_ON)

        assert entity._attr_is_on is False  # Any door opening clears occupancy

        # Step 3: Door1 closes again (both doors closed)
        def get_state_step3(entity_id):
            if entity_id == "binary_sensor.motion1":
                return Mock(state=STATE_ON)  # Motion still active
            return Mock(state=STATE_OFF)

        mock_hass.states.get.side_effect = get_state_step3

        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

        assert entity._attr_is_on is True  # Occupied again with all doors closed

    def test_no_door_sensors_configured(
        self,
        mock_hass: Mock,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test aggregate door state when no door sensors are configured."""
        # Configure no door sensors - use area-based access
        area_name = multi_sensor_coordinator.get_area_names()[0]
        area = multi_sensor_coordinator.get_area_or_default(area_name)
        area.config.sensors.door = []

        area_name = multi_sensor_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(multi_sensor_coordinator, area_name, wasp_config_entry)
        entity.hass = mock_hass

        result = entity._get_aggregate_door_state()
        assert result == STATE_OFF  # Should default to closed

    def test_no_motion_sensors_configured(
        self,
        mock_hass: Mock,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test aggregate motion state when no motion sensors are configured."""
        # Configure no motion sensors
        area_name = multi_sensor_coordinator.get_area_names()[0]
        area = multi_sensor_coordinator.get_area_or_default(area_name)
        area.config.sensors.motion = []

        area_name = multi_sensor_coordinator.get_area_names()[0]
        entity = WaspInBoxSensor(multi_sensor_coordinator, area_name, wasp_config_entry)
        entity.hass = mock_hass

        result = entity._get_aggregate_motion_state()
        assert result == STATE_OFF  # Should default to off
