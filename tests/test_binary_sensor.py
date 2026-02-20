"""Tests for binary_sensor module."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.binary_sensor import (
    Occupancy,
    WaspInBoxSensor,
    async_setup_entry,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import Sensors
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import Event, HomeAssistant
from homeassistant.util import dt as dt_util

# Add marker for tests that may have lingering timers due to HA internals
pytestmark = [pytest.mark.parametrize("expected_lingering_timers", [True])]


# ruff: noqa: SLF001, PLC0415
class TestOccupancy:
    """Test Occupancy binary sensor entity."""

    def test_initialization(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test Occupancy entity initialization."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        entity = Occupancy(area_handle=handle)

        assert entity.coordinator == coordinator
        # unique_id uses entry_id, device_id, and entity_name
        entry_id = coordinator.entry_id
        area = coordinator.get_area(area_name)
        device_id = next(iter(area.device_info()["identifiers"]))[1]
        expected_unique_id = f"{entry_id}_{device_id}_occupancy_status"
        assert entity.unique_id == expected_unique_id
        assert entity.name == "Occupancy Status"

    async def test_async_added_to_hass(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test entity added to Home Assistant."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        entity = Occupancy(area_handle=handle)
        # Set hass on entity so device registry can be accessed
        entity.hass = hass

        # Mock parent method
        with patch(
            "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass"
        ) as mock_parent:
            await entity.async_added_to_hass()
            mock_parent.assert_called_once()

        # Should set occupancy entity ID in area
        area = coordinator.get_area(area_name)
        assert area.occupancy_entity_id == entity.entity_id

    async def test_async_will_remove_from_hass(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test entity removal from Home Assistant."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        entity = Occupancy(area_handle=handle)
        # Set entity_id first
        entity.entity_id = (
            f"binary_sensor.{area_name.lower().replace(' ', '_')}_occupancy_status"
        )
        area = coordinator.get_area(area_name)
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
        coordinator: AreaOccupancyCoordinator,
        occupied: bool,
        expected_icon: str,
        expected_is_on: bool,
    ) -> None:
        """Test icon and is_on properties based on occupancy state."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        handle = coordinator.get_area_handle(area_name)
        entity = Occupancy(area_handle=handle)
        # Mock area.occupied method
        area.occupied = Mock(return_value=occupied)

        assert entity.icon == expected_icon
        assert entity.is_on is expected_is_on


# Shared fixtures for WaspInBoxSensor tests
@pytest.fixture
def wasp_coordinator(
    coordinator: AreaOccupancyCoordinator,
) -> AreaOccupancyCoordinator:
    """Create a coordinator with wasp-specific configuration."""
    # Customize the coordinator for wasp tests - use area-based access
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    area.config.wasp_in_box = Mock()
    area.config.wasp_in_box.enabled = True
    area.config.wasp_in_box.motion_timeout = 60
    area.config.wasp_in_box.max_duration = 3600
    area.config.wasp_in_box.weight = 0.85
    area.config.wasp_in_box.verification_delay = 0
    # Create a Sensors object with door and motion sensors
    area.config.sensors = Sensors(
        door=["binary_sensor.door1"],
        motion=["binary_sensor.motion1"],
        _parent_config=area.config,
    )

    # Add missing entities attribute with AsyncMock
    area.entities.async_initialize = AsyncMock()

    return coordinator


@pytest.fixture
def multi_sensor_coordinator(
    coordinator: AreaOccupancyCoordinator,
) -> AreaOccupancyCoordinator:
    """Create a coordinator with multiple door and motion sensors."""
    # Use area-based access
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
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
    return coordinator


@pytest.fixture
def wasp_config_entry(mock_config_entry: Mock) -> Mock:
    """Create a config entry with wasp-specific data."""
    new_data = dict(mock_config_entry.data)
    new_data.update(
        {
            "door_sensors": ["binary_sensor.door1"],
            "motion_sensors": ["binary_sensor.motion1"],
        }
    )
    object.__setattr__(mock_config_entry, "data", new_data)
    return mock_config_entry


def create_wasp_entity(
    wasp_coordinator: AreaOccupancyCoordinator, wasp_config_entry: Mock
) -> WaspInBoxSensor:
    """Create a WaspInBoxSensor with common setup."""
    # WaspInBoxSensor takes (coordinator, area_name, config_entry)
    area_name = wasp_coordinator.get_area_names()[0]
    handle = wasp_coordinator.get_area_handle(area_name)
    entity = WaspInBoxSensor(handle, wasp_config_entry)
    entity.entity_id = "binary_sensor.test_wasp_in_box"
    return entity


class TestWaspInBoxSensor:
    """Test WaspInBoxSensor binary sensor entity."""

    def test_initialization(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test WaspInBoxSensor initialization."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

        # Set hass (normally done by HA when entity is added)
        entity.hass = hass

        assert entity.hass == hass
        assert entity._coordinator == wasp_coordinator
        # unique_id uses entry_id, device_id, and entity_name
        entry_id = wasp_coordinator.entry_id
        area = wasp_coordinator.get_area(area_name)
        device_id = next(iter(area.device_info()["identifiers"]))[1]
        expected_unique_id = f"{entry_id}_{device_id}_wasp_in_box"
        assert entity.unique_id == expected_unique_id
        assert entity.name == "Wasp in Box"
        assert entity.should_poll is False

    async def test_async_added_to_hass(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test entity added to Home Assistant."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

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
        area = wasp_coordinator.get_area(area_name)
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
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        has_previous_state: bool,
        expected_is_on: bool,
    ) -> None:
        """Test restoring previous state with and without stored data."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

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
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test entity removal from Home Assistant."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

        # Set up some state to clean up
        entity._remove_timer = Mock()
        listener_mock = Mock()
        entity._remove_state_listener = listener_mock
        area_name = wasp_coordinator.get_area_names()[0]
        area = wasp_coordinator.get_area(area_name)
        area.wasp_entity_id = entity.entity_id

        await entity.async_will_remove_from_hass()

        # Should clean up resources if listener exists
        if listener_mock:
            listener_mock.assert_called_once()
        area_name = wasp_coordinator.get_area_names()[0]
        area = wasp_coordinator.get_area(area_name)
        assert area.wasp_entity_id is None

    def test_extra_state_attributes(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test extra state attributes with actual value verification."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

        # Set up some state with known values
        now = dt_util.utcnow()
        door_time = now - timedelta(minutes=5)
        motion_time = now - timedelta(minutes=2)
        occupied_time = now - timedelta(minutes=1)
        entity._last_occupied_time = occupied_time
        entity._last_door_time = door_time
        entity._last_motion_time = motion_time
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_ON
        entity._verification_pending = True

        attributes = entity.extra_state_attributes

        # Verify all expected attributes exist
        expected_attrs = [
            "door_state",
            "motion_state",
            "last_motion_time",
            "last_door_time",
            "last_occupied_time",
            "motion_timeout",
            "max_duration",
            "verification_delay",
            "verification_pending",
        ]
        for attr in expected_attrs:
            assert attr in attributes

        # Verify actual values match expected formats
        assert attributes["door_state"] == STATE_OFF
        assert attributes["motion_state"] == STATE_ON
        assert attributes["motion_timeout"] == 60
        assert attributes["max_duration"] == 3600
        assert attributes["verification_delay"] == 0
        assert attributes["verification_pending"] is True
        # Verify datetime strings are ISO format
        assert attributes["last_occupied_time"] == occupied_time.isoformat()
        assert attributes["last_door_time"] == door_time.isoformat()
        assert attributes["last_motion_time"] == motion_time.isoformat()

        # Test with None values
        entity._last_occupied_time = None
        entity._last_door_time = None
        entity._last_motion_time = None
        attributes_none = entity.extra_state_attributes
        assert attributes_none["last_occupied_time"] is None
        assert attributes_none["last_door_time"] is None
        assert attributes_none["last_motion_time"] is None

    async def test_get_valid_entities(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _get_valid_entities method returns all configured entities."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        # Do NOT create states in hass.states
        # We want to verify that even nonexistent entities are returned

        result = entity._get_valid_entities()

        assert "doors" in result
        assert "motion" in result
        assert "all" in result
        assert "binary_sensor.door1" in result["doors"]
        assert "binary_sensor.motion1" in result["motion"]

    async def test_initialize_from_current_states(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _initialize_from_current_states method."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = hass

        valid_entities = {
            "doors": ["binary_sensor.door1"],
            "motion": ["binary_sensor.motion1"],
        }

        # Create actual states in hass.states instead of mocking
        hass.states.async_set("binary_sensor.door1", STATE_OFF)
        hass.states.async_set("binary_sensor.motion1", STATE_OFF)

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            entity._initialize_from_current_states(valid_entities)

        # Should initialize state tracking
        assert entity._door_state == STATE_OFF
        assert entity._motion_state == STATE_OFF

    @pytest.mark.parametrize(
        (
            "initial_occupied",
            "door_state",
            "motion_state",
            "motion_age_seconds",
            "expected_state",
            "should_call_set_state",
        ),
        [
            (
                True,
                STATE_ON,
                STATE_OFF,
                None,
                STATE_OFF,
                True,
            ),  # Door opens when occupied -> unoccupied
            (
                False,
                STATE_OFF,
                STATE_ON,
                30,
                STATE_ON,
                True,
            ),  # Door closes with active motion -> occupied
            (
                False,
                STATE_OFF,
                STATE_OFF,
                30,
                STATE_ON,
                True,
            ),  # Door closes with recent motion (within timeout) -> occupied
            (
                False,
                STATE_OFF,
                STATE_OFF,
                90,
                STATE_OFF,
                False,
            ),  # Door closes with expired motion -> not occupied
            (
                False,
                STATE_OFF,
                STATE_OFF,
                None,
                STATE_OFF,
                False,
            ),  # Door closes without motion -> not occupied
        ],
    )
    async def test_process_door_state_scenarios(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        initial_occupied: bool,
        door_state: str,
        motion_state: str,
        motion_age_seconds: int | None,
        expected_state: str,
        should_call_set_state: bool,
    ) -> None:
        """Test processing door state changes in different scenarios."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = hass

        # Set up initial state
        entity._attr_is_on = initial_occupied
        entity._state = STATE_ON if initial_occupied else STATE_OFF
        entity._door_state = STATE_OFF if initial_occupied else STATE_ON
        entity._motion_state = motion_state

        # Set motion time if provided
        if motion_age_seconds is not None:
            entity._last_motion_time = dt_util.utcnow() - timedelta(
                seconds=motion_age_seconds
            )
        else:
            entity._last_motion_time = None

        # Create actual state in hass.states for aggregate calculation
        hass.states.async_set("binary_sensor.door1", door_state)

        # Mock async_write_ha_state to avoid entity registration issues
        with (
            patch.object(entity, "async_write_ha_state") as mock_write,
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            entity._process_door_state("binary_sensor.door1", door_state)
            if should_call_set_state:
                mock_set_state.assert_called_once_with(expected_state)
            else:
                mock_set_state.assert_not_called()
                # Should still update state attributes
                mock_write.assert_called()

    @pytest.mark.parametrize(
        (
            "motion_state",
            "door_state",
            "initial_occupied",
            "expected_occupied",
            "should_update_time",
        ),
        [
            (
                STATE_ON,
                STATE_OFF,
                False,
                True,
                True,
            ),  # Motion ON with doors closed -> occupied
            (
                STATE_ON,
                STATE_ON,
                False,
                False,
                True,
            ),  # Motion ON with doors open -> not occupied
            (
                STATE_OFF,
                STATE_OFF,
                True,
                True,
                False,
            ),  # Motion OFF with doors closed -> maintain occupancy
            (
                STATE_OFF,
                STATE_ON,
                False,
                False,
                False,
            ),  # Motion OFF with doors open -> not occupied
        ],
    )
    async def test_process_motion_state_scenarios(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        motion_state: str,
        door_state: str,
        initial_occupied: bool,
        expected_occupied: bool,
        should_update_time: bool,
    ) -> None:
        """Test processing motion state changes in different scenarios."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = hass

        # Set up initial state
        entity._attr_is_on = initial_occupied
        entity._state = STATE_ON if initial_occupied else STATE_OFF
        entity._door_state = door_state
        entity._motion_state = STATE_OFF if motion_state == STATE_ON else STATE_ON
        old_motion_time = dt_util.utcnow() - timedelta(minutes=5)
        entity._last_motion_time = old_motion_time

        # Create actual states in hass.states for aggregate calculation
        hass.states.async_set("binary_sensor.motion1", motion_state)
        hass.states.async_set("binary_sensor.door1", door_state)

        # Mock async_write_ha_state to avoid entity registration issues
        with (
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            entity._process_motion_state("binary_sensor.motion1", motion_state)

            # Verify motion state was updated
            assert entity._motion_state == motion_state

            # Verify time was updated only when motion turns ON
            if should_update_time:
                assert entity._last_motion_time is not None
                assert entity._last_motion_time != old_motion_time
            # Time should not change when motion turns OFF
            elif motion_state == STATE_OFF:
                assert entity._last_motion_time == old_motion_time

            # Verify occupancy state
            if expected_occupied != initial_occupied:
                mock_set_state.assert_called_once_with(
                    STATE_ON if expected_occupied else STATE_OFF
                )
            else:
                # State should be maintained, just attributes updated
                mock_set_state.assert_not_called()

    @pytest.mark.parametrize(
        ("new_state", "expected_actions"),
        [
            (STATE_ON, ["start_timer", "write_state"]),
            (STATE_OFF, ["cancel_timer", "write_state"]),
        ],
    )
    def test_set_state_scenarios(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        new_state: str,
        expected_actions: list[str],
    ) -> None:
        """Test setting state to occupied and unoccupied."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

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
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test timer start, cancel, and timeout handling."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

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

    @pytest.mark.parametrize(
        ("max_duration", "should_start_timer"),
        [
            (3600, True),  # Normal case - timer should start
            (0, False),  # Disabled - timer should not start
            (None, False),  # Disabled - timer should not start
        ],
    )
    def test_max_duration_timer_disabled(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        max_duration: int | None,
        should_start_timer: bool,
    ) -> None:
        """Test max duration timer when disabled."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

        entity._max_duration = max_duration
        entity._last_occupied_time = dt_util.utcnow()

        with patch(
            "custom_components.area_occupancy.binary_sensor.async_track_point_in_time"
        ) as mock_track:
            entity._start_max_duration_timer()
            if should_start_timer:
                mock_track.assert_called_once()
                assert entity._remove_timer is not None
            else:
                mock_track.assert_not_called()
                assert entity._remove_timer is None

    def test_max_duration_timer_already_expired(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test max duration timer when already expired."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

        entity._max_duration = 60  # 1 minute
        # Set occupied time to 2 minutes ago (already expired)
        entity._last_occupied_time = dt_util.utcnow() - timedelta(seconds=120)
        entity._state = STATE_ON

        with (
            patch(
                "custom_components.area_occupancy.binary_sensor.async_track_point_in_time"
            ) as mock_track,
            patch.object(entity, "_reset_after_max_duration") as mock_reset,
        ):
            entity._start_max_duration_timer()
            # Should reset immediately, not schedule timer
            mock_reset.assert_called_once()
            mock_track.assert_not_called()
            assert entity._remove_timer is None

    def test_timer_cancellation_idempotency(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test that timer cancellation is idempotent."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

        # Cancel when timer is None should not raise
        entity._remove_timer = None
        entity._cancel_max_duration_timer()
        assert entity._remove_timer is None

        # Cancel when timer exists
        timer_mock = Mock()
        entity._remove_timer = timer_mock
        entity._cancel_max_duration_timer()
        timer_mock.assert_called_once()
        assert entity._remove_timer is None

        # Cancel again should be safe
        entity._cancel_max_duration_timer()
        assert entity._remove_timer is None


class TestVerificationTimer:
    """Test verification timer feature for WaspInBoxSensor."""

    @pytest.fixture
    def verification_coordinator(
        self, coordinator: AreaOccupancyCoordinator
    ) -> AreaOccupancyCoordinator:
        """Create a coordinator with verification delay enabled."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.wasp_in_box = Mock()
        area.config.wasp_in_box.enabled = True
        area.config.wasp_in_box.motion_timeout = 60
        area.config.wasp_in_box.max_duration = 3600
        area.config.wasp_in_box.weight = 0.85
        area.config.wasp_in_box.verification_delay = 30  # 30 seconds
        area.config.sensors = Sensors(
            door=["binary_sensor.door1"],
            motion=["binary_sensor.motion1"],
            _parent_config=area.config,
        )
        area.entities.async_initialize = AsyncMock()
        return coordinator

    @pytest.mark.parametrize(
        ("verification_delay", "should_start_timer", "expected_pending"),
        [
            (30, True, True),  # Enabled - timer should start
            (0, False, False),  # Disabled - timer should not start
        ],
    )
    def test_start_verification_timer(
        self,
        hass: HomeAssistant,
        verification_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        verification_delay: int,
        should_start_timer: bool,
        expected_pending: bool,
    ) -> None:
        """Test starting verification timer."""
        area_name = verification_coordinator.get_area_names()[0]
        handle = verification_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity._verification_delay = verification_delay

        with patch(
            "custom_components.area_occupancy.binary_sensor.async_track_point_in_time"
        ) as mock_track:
            entity._start_verification_timer()
            if should_start_timer:
                mock_track.assert_called_once()
                assert entity._remove_verification_timer is not None
            else:
                mock_track.assert_not_called()
                assert entity._remove_verification_timer is None
            assert entity._verification_pending is expected_pending

    @pytest.mark.parametrize(
        ("timer_exists", "initial_pending"),
        [
            (True, True),  # Normal case - timer exists
            (False, False),  # Idempotent case - timer already None
        ],
    )
    def test_cancel_verification_timer(
        self,
        hass: HomeAssistant,
        verification_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        timer_exists: bool,
        initial_pending: bool,
    ) -> None:
        """Test canceling verification timer."""
        area_name = verification_coordinator.get_area_names()[0]
        handle = verification_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

        if timer_exists:
            timer_mock = Mock()
            entity._remove_verification_timer = timer_mock
            entity._verification_pending = initial_pending

            entity._cancel_verification_timer()
            timer_mock.assert_called_once()
        else:
            # Cancel when timer is None should not raise (idempotent)
            entity._remove_verification_timer = None
            entity._verification_pending = initial_pending
            entity._cancel_verification_timer()

        assert entity._remove_verification_timer is None
        assert entity._verification_pending is False

    @pytest.mark.parametrize(
        (
            "initial_state",
            "motion_state",
            "has_motion_sensors",
            "expected_final_state",
            "should_call_set_state",
            "should_call_write_state",
            "use_verification_coordinator",
        ),
        [
            # Motion present - maintain occupancy
            (STATE_ON, STATE_ON, True, STATE_ON, False, True, True),
            # Motion not present - clear occupancy (false positive)
            # Note: when _set_state is mocked, async_write_ha_state won't be called
            # because it's called inside _set_state
            (STATE_ON, STATE_OFF, True, STATE_OFF, True, False, True),
            # Already unoccupied - skip verification
            (STATE_OFF, None, True, STATE_OFF, False, False, True),
            # No motion sensors - skip verification and maintain occupancy
            (STATE_ON, None, False, STATE_ON, False, True, False),
        ],
    )
    async def test_verification_check_scenarios(
        self,
        hass: HomeAssistant,
        verification_coordinator: AreaOccupancyCoordinator,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        initial_state: str,
        motion_state: str | None,
        has_motion_sensors: bool,
        expected_final_state: str,
        should_call_set_state: bool,
        should_call_write_state: bool,
        use_verification_coordinator: bool,
    ) -> None:
        """Test verification check in various scenarios."""
        coord = (
            verification_coordinator if use_verification_coordinator else coordinator
        )
        area_name = coord.get_area_names()[0]

        # Configure sensors if needed
        if not has_motion_sensors:
            area = coord.get_area(area_name)
            area.config.wasp_in_box = Mock()
            area.config.wasp_in_box.enabled = True
            area.config.wasp_in_box.verification_delay = 30
            area.config.sensors = Sensors(
                door=["binary_sensor.door1"],
                motion=[],  # No motion sensors
                _parent_config=area.config,
            )

        handle = coord.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        # Set up initial state
        entity._state = initial_state
        entity._attr_is_on = initial_state == STATE_ON

        # Set motion sensor state if sensors exist
        if has_motion_sensors and motion_state is not None:
            hass.states.async_set("binary_sensor.motion1", motion_state)

        with (
            patch.object(entity, "_set_state") as mock_set_state,
            patch.object(entity, "async_write_ha_state") as mock_write,
        ):
            entity._handle_verification_check(dt_util.utcnow())

            # Verify method calls first
            if should_call_set_state:
                mock_set_state.assert_called_once_with(STATE_OFF)
                # When _set_state is mocked, state doesn't actually change
                # So we check initial state is still there
                assert entity._state == initial_state
                assert entity._attr_is_on == (initial_state == STATE_ON)
            else:
                mock_set_state.assert_not_called()
                # Verify final state when _set_state is not called
                assert entity._state == expected_final_state
                assert entity._attr_is_on == (expected_final_state == STATE_ON)

            assert entity._verification_pending is False

            if should_call_write_state:
                mock_write.assert_called_once()
            else:
                mock_write.assert_not_called()

    def test_verification_timer_in_set_state(
        self,
        hass: HomeAssistant,
        verification_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test that verification timer is started when setting state to ON."""
        area_name = verification_coordinator.get_area_names()[0]
        handle = verification_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        with (
            patch.object(entity, "_start_max_duration_timer") as mock_max_timer,
            patch.object(
                entity, "_start_verification_timer"
            ) as mock_verification_timer,
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._set_state(STATE_ON)

            # Both timers should be started
            mock_max_timer.assert_called_once()
            mock_verification_timer.assert_called_once()

        # Verify state was set
        assert entity._attr_is_on is True
        assert entity._state == STATE_ON

    def test_verification_timer_cancelled_on_unoccupied(
        self,
        hass: HomeAssistant,
        verification_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test that verification timer is cancelled when setting state to OFF."""
        area_name = verification_coordinator.get_area_names()[0]
        handle = verification_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        # Set up occupied state with timers
        entity._attr_is_on = True
        entity._state = STATE_ON
        entity._remove_timer = Mock()
        entity._remove_verification_timer = Mock()

        with (
            patch.object(entity, "_cancel_max_duration_timer") as mock_cancel_max,
            patch.object(
                entity, "_cancel_verification_timer"
            ) as mock_cancel_verification,
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._set_state(STATE_OFF)

            # Both timers should be cancelled
            mock_cancel_max.assert_called_once()
            mock_cancel_verification.assert_called_once()

        # Verify state was set
        assert entity._attr_is_on is False
        assert entity._state == STATE_OFF


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @pytest.fixture
    def setup_config_entry(
        self, mock_config_entry: Mock, coordinator: AreaOccupancyCoordinator
    ) -> Mock:
        """Create a config entry for setup tests."""
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator
        # Configure wasp setting on the area
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
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
        hass: HomeAssistant,
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
        area = coordinator.get_area(area_name)
        area.config.wasp_in_box.enabled = wasp_enabled

        mock_async_add_entities = Mock()

        await async_setup_entry(hass, setup_config_entry, mock_async_add_entities)

        # Called once per area (with subentry_id) + once for All Areas (without)
        assert mock_async_add_entities.call_count == 2
        # Gather all entities from all calls
        entities = []
        for call_args in mock_async_add_entities.call_args_list:
            entities.extend(call_args[0][0])
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
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> WaspInBoxSensor:
        """Create a comprehensive wasp sensor for testing."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = hass

        # Initialize with known state
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_OFF
        entity._attr_is_on = False

        return entity

    async def test_complete_wasp_occupancy_cycle(
        self, hass: HomeAssistant, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test complete wasp occupancy detection cycle."""
        entity = comprehensive_wasp_sensor

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            # Step 1: Motion detected while unoccupied
            # Create actual state in hass.states
            hass.states.async_set("binary_sensor.motion1", STATE_ON)
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

            # Should update motion state
            assert entity._motion_state == STATE_ON

            # Step 2: Door closes with recent motion -> should trigger occupancy
            # Create actual state in hass.states
            hass.states.async_set("binary_sensor.door1", STATE_OFF)
            with patch.object(entity, "_start_max_duration_timer") as mock_start_timer:
                entity._process_door_state("binary_sensor.door1", STATE_OFF)

            assert entity._attr_is_on is True
            assert entity._last_occupied_time is not None
            mock_start_timer.assert_called_once()

            # Step 3: Door opens while occupied -> should end occupancy
            # Create actual state in hass.states
            hass.states.async_set("binary_sensor.door1", STATE_ON)
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

        # Verify initial state
        assert entity._attr_is_on is False
        assert entity._state == STATE_OFF
        assert entity._last_occupied_time is None

        # Mock an error during state writing
        with (
            patch.object(
                entity, "async_write_ha_state", side_effect=Exception("Write failed")
            ),
            patch.object(entity, "_start_max_duration_timer") as mock_timer,
            pytest.raises(Exception, match="Write failed"),
        ):
            entity._set_state(STATE_ON)

        # State should still be updated internally even if write fails
        assert entity._attr_is_on is True
        assert entity._state == STATE_ON
        assert entity._last_occupied_time is not None
        # Timer should have been started before the exception
        mock_timer.assert_called_once()


class TestWaspMultiSensorAggregation:
    """Test WaspInBoxSensor with multiple door and motion sensors."""

    @pytest.fixture
    def multi_sensor_wasp(
        self,
        hass: HomeAssistant,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> WaspInBoxSensor:
        """Create a wasp sensor with multiple door and motion sensors."""
        area_name = multi_sensor_coordinator.get_area_names()[0]
        handle = multi_sensor_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Mock states for multiple sensors (all sensors default to OFF)
        # Note: hass.states.get is read-only, so we'll patch it in each test that needs it

        # Initialize state
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_OFF
        entity._attr_is_on = False
        entity._state = STATE_OFF

        return entity

    @pytest.mark.parametrize(
        ("door1_state", "door2_state", "expected_result", "description"),
        [
            (STATE_OFF, STATE_OFF, STATE_OFF, "all doors closed"),
            (STATE_ON, STATE_OFF, STATE_ON, "any door open"),
            (STATE_OFF, STATE_ON, STATE_ON, "any door open"),
        ],
    )
    async def test_aggregate_door_state(
        self,
        hass: HomeAssistant,
        multi_sensor_wasp: WaspInBoxSensor,
        door1_state: str,
        door2_state: str,
        expected_result: str,
        description: str,
    ) -> None:
        """Test aggregate door state calculation."""
        entity = multi_sensor_wasp

        # Create actual states in hass.states
        hass.states.async_set("binary_sensor.door1", door1_state)
        hass.states.async_set("binary_sensor.door2", door2_state)

        result = entity._get_aggregate_door_state()
        assert result == expected_result, (
            f"Expected {expected_result} when {description}"
        )

    @pytest.mark.parametrize(
        ("motion1_state", "motion2_state", "expected_result", "description"),
        [
            (STATE_OFF, STATE_OFF, STATE_OFF, "all motion off"),
            (STATE_ON, STATE_OFF, STATE_ON, "any motion active"),
            (STATE_OFF, STATE_ON, STATE_ON, "any motion active"),
        ],
    )
    async def test_aggregate_motion_state(
        self,
        hass: HomeAssistant,
        multi_sensor_wasp: WaspInBoxSensor,
        motion1_state: str,
        motion2_state: str,
        expected_result: str,
        description: str,
    ) -> None:
        """Test aggregate motion state calculation."""
        entity = multi_sensor_wasp

        # Create actual states in hass.states
        hass.states.async_set("binary_sensor.motion1", motion1_state)
        hass.states.async_set("binary_sensor.motion2", motion2_state)

        result = entity._get_aggregate_motion_state()
        assert result == expected_result, (
            f"Expected {expected_result} when {description}"
        )

    async def test_multi_door_any_opening_clears_occupancy(
        self, hass: HomeAssistant, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that opening ANY door clears occupancy."""
        entity = multi_sensor_wasp

        # Set up occupied state with all doors closed
        entity._state = STATE_ON
        entity._attr_is_on = True
        entity._door_state = STATE_OFF

        # Create actual states in hass.states - door1 opening, door2 staying closed
        hass.states.async_set("binary_sensor.door1", STATE_ON)
        hass.states.async_set("binary_sensor.door2", STATE_OFF)

        with (
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_cancel_verification_timer"),
        ):
            entity._process_door_state("binary_sensor.door1", STATE_ON)

        # Should be unoccupied because door1 opened
        assert entity._attr_is_on is False
        assert entity._state == STATE_OFF

    async def test_multi_door_all_closed_with_motion(
        self, hass: HomeAssistant, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that occupancy triggers when all doors are closed with motion."""
        entity = multi_sensor_wasp

        # Set up unoccupied state with motion active
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._motion_state = STATE_ON

        # Create actual states in hass.states - all doors closed
        hass.states.async_set("binary_sensor.door1", STATE_OFF)
        hass.states.async_set("binary_sensor.door2", STATE_OFF)

        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_door_state("binary_sensor.door2", STATE_OFF)

        # Should be occupied because all doors closed with motion
        assert entity._attr_is_on is True
        assert entity._state == STATE_ON

    async def test_multi_motion_any_triggers_occupancy(
        self, hass: HomeAssistant, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test that ANY motion sensor triggers occupancy with doors closed."""
        entity = multi_sensor_wasp

        # Set up unoccupied state with doors closed
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._door_state = STATE_OFF

        # Create actual states in hass.states - motion2 activating, motion1 staying off
        hass.states.async_set("binary_sensor.motion1", STATE_OFF)
        hass.states.async_set("binary_sensor.motion2", STATE_ON)

        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_motion_state("binary_sensor.motion2", STATE_ON)

        # Should be occupied because motion2 detected with doors closed
        assert entity._attr_is_on is True
        assert entity._state == STATE_ON

    async def test_multi_sensor_complete_cycle(
        self, hass: HomeAssistant, multi_sensor_wasp: WaspInBoxSensor
    ) -> None:
        """Test complete occupancy cycle with multiple sensors."""
        entity = multi_sensor_wasp

        # Step 1: Both doors closed, motion1 triggers
        hass.states.async_set("binary_sensor.motion1", STATE_ON)
        hass.states.async_set("binary_sensor.door1", STATE_OFF)
        hass.states.async_set("binary_sensor.door2", STATE_OFF)
        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

        assert entity._attr_is_on is True

        # Step 2: Door1 opens (door2 still closed)
        hass.states.async_set("binary_sensor.door1", STATE_ON)
        hass.states.async_set("binary_sensor.door2", STATE_OFF)
        with (
            patch.object(entity, "_cancel_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_door_state("binary_sensor.door1", STATE_ON)

        assert entity._attr_is_on is False  # Any door opening clears occupancy

        # Step 3: Door1 closes again (both doors closed)
        hass.states.async_set("binary_sensor.motion1", STATE_ON)  # Motion still active
        hass.states.async_set("binary_sensor.door1", STATE_OFF)
        hass.states.async_set("binary_sensor.door2", STATE_OFF)
        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
        ):
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

        assert entity._attr_is_on is True  # Occupied again with all doors closed

    def test_no_door_sensors_configured(
        self,
        hass: HomeAssistant,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test aggregate door state when no door sensors are configured."""
        # Configure no door sensors - use area-based access
        area_name = multi_sensor_coordinator.get_area_names()[0]
        area = multi_sensor_coordinator.get_area(area_name)
        area.config.sensors.door = []

        area_name = multi_sensor_coordinator.get_area_names()[0]
        handle = multi_sensor_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        result = entity._get_aggregate_door_state()
        assert result == STATE_OFF  # Should default to closed

    def test_no_motion_sensors_configured(
        self,
        hass: HomeAssistant,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test aggregate motion state when no motion sensors are configured."""
        # Configure no motion sensors
        area_name = multi_sensor_coordinator.get_area_names()[0]
        area = multi_sensor_coordinator.get_area(area_name)
        area.config.sensors.motion = []

        area_name = multi_sensor_coordinator.get_area_names()[0]
        handle = multi_sensor_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        result = entity._get_aggregate_motion_state()
        assert result == STATE_OFF  # Should default to off


class TestWaspInBoxSensorErrorHandling:
    """Test WaspInBoxSensor error handling scenarios."""

    def test_setup_entity_tracking_no_entities(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _setup_entity_tracking with no entities configured."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Configure no sensors
        area.config.sensors = Sensors(
            motion=[], door=[], window=[], media=[], appliance=[]
        )

        handle = coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        # Should handle gracefully
        entity._setup_entity_tracking()
        assert entity._remove_state_listener is None

    def test_setup_entity_tracking_with_unavailable_entities(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _setup_entity_tracking tracks entities even if they don't exist yet."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Configure sensors that don't exist in hass
        area.config.sensors = Sensors(
            motion=["binary_sensor.nonexistent"],
            door=["binary_sensor.nonexistent_door"],
            window=[],
            media=[],
            appliance=[],
            _parent_config=area.config,
        )

        handle = coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        # Should track the entities anyway
        with patch(
            "custom_components.area_occupancy.binary_sensor.async_track_state_change_event"
        ) as mock_track:
            entity._setup_entity_tracking()
            # Should be called with the configured entities
            mock_track.assert_called_once()
            tracked_entities = mock_track.call_args[0][1]
            assert "binary_sensor.nonexistent" in tracked_entities
            assert "binary_sensor.nonexistent_door" in tracked_entities

            # Returns a listener
            assert entity._remove_state_listener is not None

    def test_handle_state_change_unknown_state(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _handle_state_change with unknown/unavailable state."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        area.config.sensors = Sensors(
            motion=["binary_sensor.motion1"],
            door=["binary_sensor.door1"],
            window=[],
            media=[],
            appliance=[],
        )

        handle = coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        # Mock state change event with unknown state
        event = Mock(spec=Event)
        event.data = {
            "entity_id": "binary_sensor.door1",
            "old_state": Mock(state="off"),
            "new_state": Mock(state="unknown"),
        }

        # Should handle gracefully and return early
        entity._handle_state_change(event)
        # State should not change

    def test_handle_state_change_no_new_state(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _handle_state_change with no new_state."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        area.config.sensors = Sensors(
            motion=["binary_sensor.motion1"],
            door=["binary_sensor.door1"],
            window=[],
            media=[],
            appliance=[],
        )

        handle = coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        # Mock state change event with no new_state
        event = Mock(spec=Event)
        event.data = {
            "entity_id": "binary_sensor.door1",
            "old_state": Mock(state="off"),
            "new_state": None,
        }

        # Should handle gracefully and return early
        entity._handle_state_change(event)

    def test_process_door_state_invalid_state(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _process_door_state with invalid door state."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        area.config.sensors = Sensors(
            motion=["binary_sensor.motion1"],
            door=["binary_sensor.door1"],
            window=[],
            media=[],
            appliance=[],
        )

        handle = coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Set up initial state
        entity._state = STATE_ON
        entity._motion_state = STATE_OFF

        # Process invalid door state - should handle gracefully
        with patch.object(entity, "async_write_ha_state"):
            entity._process_door_state("binary_sensor.door1", "invalid_state")

    def test_process_motion_state_invalid_state(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test _process_motion_state with invalid motion state."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        area.config.sensors = Sensors(
            motion=["binary_sensor.motion1"],
            door=["binary_sensor.door1"],
            window=[],
            media=[],
            appliance=[],
        )

        handle = coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Process invalid motion state - should handle gracefully
        with patch.object(entity, "async_write_ha_state"):
            entity._process_motion_state("binary_sensor.motion1", "invalid_state")


class TestDoorStateEdgeCases:
    """Test door state processing edge cases."""

    async def test_door_closes_without_motion_sensors(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test door closes when no motion sensors are configured."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.wasp_in_box = Mock()
        area.config.wasp_in_box.enabled = True
        area.config.wasp_in_box.motion_timeout = 60
        area.config.sensors = Sensors(
            door=["binary_sensor.door1"],
            motion=[],  # No motion sensors
            _parent_config=area.config,
        )

        handle = coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Set up unoccupied state
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._door_state = STATE_ON

        hass.states.async_set("binary_sensor.door1", STATE_OFF)

        with (
            patch.object(entity, "async_write_ha_state") as mock_write,
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

            # Should not trigger occupancy without motion sensors
            mock_set_state.assert_not_called()
            # Should still update attributes
            mock_write.assert_called()

    async def test_door_closes_motion_timeout_at_limit(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test door closes with motion timeout exactly at the limit."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = hass

        # Set up unoccupied state with motion at exact timeout limit
        entity._state = STATE_OFF
        entity._attr_is_on = False
        # Previous door state was open (STATE_ON), now closing to closed (STATE_OFF)
        entity._door_state = STATE_ON  # Previous state - door was open (DOOR_OPEN)
        entity._motion_state = STATE_OFF  # Motion is not currently active
        # Motion timeout is 60 seconds, set motion time to 59.9 seconds ago
        # Using 59.9 to account for any timing precision issues while still testing boundary
        entity._last_motion_time = dt_util.utcnow() - timedelta(seconds=59.9)

        # Set door state to closed in hass.states for aggregate calculation
        # The aggregate calculation will see door is closed (STATE_OFF = DOOR_CLOSED)
        hass.states.async_set("binary_sensor.door1", STATE_OFF)

        with (
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

            # Should trigger occupancy (motion timeout is <=, so 60 seconds is valid)
            # The implementation checks motion_age <= motion_timeout, so 60 seconds should trigger
            mock_set_state.assert_called_once_with(STATE_ON)

    async def test_door_state_no_change_when_already_in_state(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test door state change when already in that state (no-op)."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = hass

        # Set up state with door already closed
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_OFF

        hass.states.async_set("binary_sensor.door1", STATE_OFF)

        with (
            patch.object(entity, "async_write_ha_state") as mock_write,
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            # Process door state change to same state
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

            # Should update timestamp but not change occupancy
            assert entity._last_door_time is not None
            mock_set_state.assert_not_called()
            # Should still update attributes
            mock_write.assert_called()


class TestMotionStateEdgeCases:
    """Test motion state processing edge cases."""

    async def test_motion_off_with_doors_closed_maintains_occupancy(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test that motion OFF with doors closed maintains occupancy."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = hass

        # Set up occupied state
        entity._state = STATE_ON
        entity._attr_is_on = True
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_ON

        hass.states.async_set("binary_sensor.motion1", STATE_OFF)
        hass.states.async_set("binary_sensor.door1", STATE_OFF)

        with (
            patch.object(entity, "async_write_ha_state") as mock_write,
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            entity._process_motion_state("binary_sensor.motion1", STATE_OFF)

            # Should maintain occupancy
            assert entity._attr_is_on is True
            assert entity._state == STATE_ON
            mock_set_state.assert_not_called()
            # Should update attributes
            mock_write.assert_called()

    async def test_motion_on_with_doors_open_no_occupancy(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test that motion ON with doors open does not trigger occupancy."""
        entity = create_wasp_entity(wasp_coordinator, wasp_config_entry)
        entity.hass = hass

        # Set up unoccupied state
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._door_state = STATE_ON
        entity._motion_state = STATE_OFF

        hass.states.async_set("binary_sensor.motion1", STATE_ON)
        hass.states.async_set("binary_sensor.door1", STATE_ON)

        with (
            patch.object(entity, "async_write_ha_state") as mock_write,
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

            # Should not trigger occupancy when doors are open
            assert entity._attr_is_on is False
            assert entity._state == STATE_OFF
            mock_set_state.assert_not_called()
            # Should update motion state and time
            assert entity._motion_state == STATE_ON
            assert entity._last_motion_time is not None
            # Implementation only calls async_write_ha_state when motion turns OFF, not ON
            mock_write.assert_not_called()

    async def test_multiple_motion_sensors_transitioning(
        self,
        hass: HomeAssistant,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test multiple motion sensors transitioning states."""
        area_name = multi_sensor_coordinator.get_area_names()[0]
        handle = multi_sensor_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Set up unoccupied state with doors closed
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._door_state = STATE_OFF

        # Set up states: motion1 ON, motion2 OFF
        hass.states.async_set("binary_sensor.motion1", STATE_ON)
        hass.states.async_set("binary_sensor.motion2", STATE_OFF)

        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            # Process motion1 ON
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

            # Should trigger occupancy (motion1 ON with doors closed)
            mock_set_state.assert_called_once_with(STATE_ON)
            assert entity._motion_state == STATE_ON
            # Since _set_state is mocked, manually update state to reflect what would happen
            entity._state = STATE_ON
            entity._attr_is_on = True

            # Now motion1 turns OFF, but motion2 is still OFF
            hass.states.async_set("binary_sensor.motion1", STATE_OFF)
            mock_set_state.reset_mock()

            with patch.object(entity, "async_write_ha_state"):
                entity._process_motion_state("binary_sensor.motion1", STATE_OFF)

                # Should maintain occupancy (all motion OFF but doors still closed)
                mock_set_state.assert_not_called()
                assert entity._motion_state == STATE_OFF
                assert entity._attr_is_on is True


class TestAggregateStateEdgeCases:
    """Test aggregate state edge cases."""

    async def test_simultaneous_sensor_transitions(
        self,
        hass: HomeAssistant,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test simultaneous sensor state transitions."""
        area_name = multi_sensor_coordinator.get_area_names()[0]
        handle = multi_sensor_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Set up initial state
        entity._state = STATE_OFF
        entity._attr_is_on = False
        entity._door_state = STATE_ON
        entity._motion_state = STATE_OFF

        # Simulate simultaneous transitions: door1 closes, motion1 activates
        hass.states.async_set("binary_sensor.door1", STATE_OFF)
        hass.states.async_set("binary_sensor.door2", STATE_ON)  # Still one open
        hass.states.async_set("binary_sensor.motion1", STATE_ON)

        with (
            patch.object(entity, "_start_max_duration_timer"),
            patch.object(entity, "_start_verification_timer"),
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            # Process door1 closing
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

            # Should not trigger occupancy yet (door2 still open)
            mock_set_state.assert_not_called()

            # Process motion1 activating
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

            # Should not trigger occupancy (doors not all closed)
            mock_set_state.assert_not_called()

            # Now door2 closes
            hass.states.async_set("binary_sensor.door2", STATE_OFF)
            entity._process_door_state("binary_sensor.door2", STATE_OFF)

            # Now should trigger occupancy (all doors closed + motion)
            mock_set_state.assert_called_once_with(STATE_ON)

    async def test_all_sensors_unavailable(
        self,
        hass: HomeAssistant,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test aggregate state when all sensors become unavailable."""
        area_name = multi_sensor_coordinator.get_area_names()[0]
        handle = multi_sensor_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Set up initial state
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_ON

        # All sensors become unavailable
        hass.states.async_set("binary_sensor.door1", "unavailable")
        hass.states.async_set("binary_sensor.door2", "unavailable")
        hass.states.async_set("binary_sensor.motion1", "unavailable")
        hass.states.async_set("binary_sensor.motion2", "unavailable")

        # Aggregate states should handle unavailable gracefully
        door_state = entity._get_aggregate_door_state()
        motion_state = entity._get_aggregate_motion_state()

        # Should return last known good state or default
        # (implementation returns DOOR_CLOSED if all unavailable)
        assert door_state == STATE_OFF
        assert motion_state == STATE_OFF

    async def test_mixed_available_unavailable_sensors(
        self,
        hass: HomeAssistant,
        multi_sensor_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test aggregate state with mixed available/unavailable sensors."""
        area_name = multi_sensor_coordinator.get_area_names()[0]
        handle = multi_sensor_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Mixed states: door1 available and open, door2 unavailable
        hass.states.async_set("binary_sensor.door1", STATE_ON)
        hass.states.async_set("binary_sensor.door2", "unavailable")

        door_state = entity._get_aggregate_door_state()
        # Should return DOOR_OPEN (any door open)
        assert door_state == STATE_ON

        # Mixed motion: motion1 available and ON, motion2 unavailable
        hass.states.async_set("binary_sensor.motion1", STATE_ON)
        hass.states.async_set("binary_sensor.motion2", "unavailable")

        motion_state = entity._get_aggregate_motion_state()
        # Should return STATE_ON (any motion active)
        assert motion_state == STATE_ON


class TestStateRestorationEdgeCases:
    """Test state restoration edge cases."""

    @pytest.mark.parametrize(
        ("attributes", "expected_door_time", "expected_motion_time"),
        [
            # Invalid datetime strings - parse_datetime returns None
            (
                {
                    "last_occupied_time": "invalid-datetime",
                    "last_door_time": "not-a-date",
                    "last_motion_time": "bad-format",
                },
                None,
                None,
            ),
            # Missing attributes - empty dict
            ({}, None, None),
        ],
    )
    async def test_restore_state_attribute_handling(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        attributes: dict,
        expected_door_time: datetime | None,
        expected_motion_time: datetime | None,
    ) -> None:
        """Test restoring state with invalid or missing datetime attributes."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.hass = hass

        # Mock previous state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_state.attributes = attributes
        mock_get_state = AsyncMock(return_value=mock_state)

        with (
            patch.object(entity, "async_get_last_state", mock_get_state),
            patch.object(entity, "_start_max_duration_timer") as mock_timer,
        ):
            await entity._restore_previous_state()

            # Should handle gracefully - invalid/missing datetimes become None
            assert entity._state == STATE_ON
            assert entity._attr_is_on is True
            # When state is STATE_ON, _last_occupied_time is set to utcnow() if None
            assert isinstance(entity._last_occupied_time, datetime)
            assert entity._last_door_time is expected_door_time
            assert entity._last_motion_time is expected_motion_time
            mock_timer.assert_called_once()

    async def test_restore_state_max_duration_disabled(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
    ) -> None:
        """Test restoring state when max_duration is disabled."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.wasp_in_box = Mock()
        area.config.wasp_in_box.enabled = True
        area.config.wasp_in_box.motion_timeout = 60
        area.config.wasp_in_box.max_duration = 0  # Disabled
        area.config.wasp_in_box.weight = 0.85
        area.config.sensors = Sensors(
            door=["binary_sensor.door1"],
            motion=["binary_sensor.motion1"],
            _parent_config=area.config,
        )

        handle = coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)

        # Mock previous state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_state.attributes = {
            "last_occupied_time": "2023-01-01T12:00:00+00:00",
        }
        mock_get_state = AsyncMock(return_value=mock_state)

        with (
            patch.object(entity, "async_get_last_state", mock_get_state),
            patch(
                "custom_components.area_occupancy.binary_sensor.async_track_point_in_time"
            ) as mock_track,
        ):
            await entity._restore_previous_state()

            # Should restore state but not start timer (max_duration disabled)
            assert entity._state == STATE_ON
            assert entity._attr_is_on is True
            mock_track.assert_not_called()
            assert entity._remove_timer is None


class TestCleanupEdgeCases:
    """Test cleanup edge cases."""

    @pytest.mark.parametrize(
        ("timers_exist", "entity_id_matches", "expected_wasp_entity_id"),
        [
            (False, True, None),  # Timers already None, entity_id matches
            (
                True,
                False,
                "binary_sensor.different_entity",
            ),  # Timers exist, entity_id mismatch
        ],
    )
    async def test_cleanup_scenarios(
        self,
        hass: HomeAssistant,
        wasp_coordinator: AreaOccupancyCoordinator,
        wasp_config_entry: Mock,
        timers_exist: bool,
        entity_id_matches: bool,
        expected_wasp_entity_id: str | None,
    ) -> None:
        """Test cleanup in various scenarios."""
        area_name = wasp_coordinator.get_area_names()[0]
        handle = wasp_coordinator.get_area_handle(area_name)
        entity = WaspInBoxSensor(handle, wasp_config_entry)
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        area = wasp_coordinator.get_area(area_name)

        if timers_exist:
            # Set up some resources
            entity._remove_timer = Mock()
            entity._remove_state_listener = Mock()
            entity._remove_verification_timer = Mock()
        else:
            # Set all resources to None
            entity._remove_timer = None
            entity._remove_state_listener = None
            entity._remove_verification_timer = None

        if entity_id_matches:
            area.wasp_entity_id = entity.entity_id
        else:
            # Set different entity_id on area
            area.wasp_entity_id = "binary_sensor.different_entity"

        # Cleanup should not raise
        entity._cleanup_all_resources()

        # Resources should be cleaned up
        assert entity._remove_timer is None
        assert entity._remove_state_listener is None
        assert entity._remove_verification_timer is None

        # Verify area.wasp_entity_id behavior
        if entity_id_matches:
            # Should be cleared when entity_id matches
            assert area.wasp_entity_id is None
        else:
            # Should remain when entity_id doesn't match
            assert area.wasp_entity_id == expected_wasp_entity_id


class TestOccupancyEntityEdgeCases:
    """Test Occupancy entity edge cases."""

    @pytest.mark.parametrize(
        ("area_id", "expected_area_id", "should_register_config_entry"),
        [
            ("test_area_id", "test_area_id", True),  # With area_id configured
            (None, None, False),  # No area_id configured
        ],
    )
    async def test_device_registry_area_assignment(
        self,
        hass: HomeAssistant,
        device_registry,
        coordinator: AreaOccupancyCoordinator,
        area_id: str | None,
        expected_area_id: str | None,
        should_register_config_entry: bool,
    ) -> None:
        """Test device registry area assignment in async_added_to_hass."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.area_id = area_id

        handle = coordinator.get_area_handle(area_name)
        entity = Occupancy(area_handle=handle)
        entity.hass = hass

        # Register config entry in hass.config_entries so device registry can link to it
        # Only needed when area_id is configured
        if should_register_config_entry:
            # Add setup_lock attribute to mock config entry for proper teardown
            if not hasattr(coordinator.config_entry, "setup_lock"):
                from asyncio import Lock

                coordinator.config_entry.setup_lock = Lock()
            hass.config_entries._entries[coordinator.entry_id] = (
                coordinator.config_entry
            )

            # Create device in registry
            device_entry = device_registry.async_get_or_create(
                config_entry_id=coordinator.entry_id,
                identifiers=entity.device_info["identifiers"],
                name=area_name,
            )

            # Initially device has no area_id
            assert device_entry.area_id is None

        # Mock parent method
        with patch(
            "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass"
        ):
            await entity.async_added_to_hass()

        # Verify results
        if should_register_config_entry:
            # Device should now have area_id assigned
            device_entry = device_registry.async_get_device(
                identifiers=entity.device_info["identifiers"]
            )
            assert device_entry is not None
            assert device_entry.area_id == expected_area_id
        else:
            # Should not raise and should set occupancy_entity_id
            assert area.occupancy_entity_id == entity.entity_id

    def test_all_areas_occupancy_sensor(
        self,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test All Areas occupancy sensor behavior."""
        all_areas = coordinator.get_all_areas()
        entity = Occupancy(all_areas=all_areas)

        # Should have correct unique_id
        assert entity.unique_id is not None
        assert "all_areas" in entity.unique_id or "all" in entity.unique_id.lower()

        # Mock all_areas.occupied method
        all_areas.occupied = Mock(return_value=True)
        assert entity.is_on is True

        all_areas.occupied = Mock(return_value=False)
        assert entity.is_on is False

        # Test when all_areas is None
        entity._all_areas = None
        assert entity.is_on is False
