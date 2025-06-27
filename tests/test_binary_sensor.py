"""Tests for binary_sensor module."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.binary_sensor import (
    Occupancy,
    WaspInBoxSensor,
    async_setup_entry,
)
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util

# Add marker for tests that may have lingering timers due to HA internals
pytestmark = [
    pytest.mark.expected_lingering_timers(True),
    pytest.mark.asyncio,
]


# ruff: noqa: SLF001
class TestOccupancy:
    """Test Occupancy binary sensor entity."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test Occupancy entity initialization."""
        entity = Occupancy(mock_coordinator, "test_entry_id")

        assert entity.coordinator == mock_coordinator
        assert entity.unique_id == "test_entry_id_occupancy_status"
        assert entity.name == "Occupancy Status"

    async def test_async_added_to_hass(self, mock_coordinator: Mock) -> None:
        """Test entity added to Home Assistant."""
        entity = Occupancy(mock_coordinator, "test_entry_id")

        # Mock parent method
        with patch(
            "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass"
        ) as mock_parent:
            await entity.async_added_to_hass()
            mock_parent.assert_called_once()

        # Should set occupancy entity ID in coordinator
        assert mock_coordinator.occupancy_entity_id == entity.entity_id

    async def test_async_will_remove_from_hass(self, mock_coordinator: Mock) -> None:
        """Test entity removal from Home Assistant."""
        entity = Occupancy(mock_coordinator, "test_entry_id")

        await entity.async_will_remove_from_hass()

        # Should clear occupancy entity ID in coordinator
        assert mock_coordinator.occupancy_entity_id is None

    def test_icon_property(self, mock_coordinator: Mock) -> None:
        """Test icon property."""
        entity = Occupancy(mock_coordinator, "test_entry_id")

        # Test occupied state
        mock_coordinator.occupied = True
        assert entity.icon == "mdi:home-account"

        # Test unoccupied state
        mock_coordinator.occupied = False
        assert entity.icon == "mdi:home-outline"

    def test_is_on_property(self, mock_coordinator: Mock) -> None:
        """Test is_on property."""
        entity = Occupancy(mock_coordinator, "test_entry_id")

        # Test occupied state
        mock_coordinator.occupied = True
        assert entity.is_on is True

        # Test unoccupied state
        mock_coordinator.occupied = False
        assert entity.is_on is False


class TestWaspInBoxSensor:
    """Test WaspInBoxSensor binary sensor entity."""

    @pytest.fixture
    def wasp_coordinator(self, mock_coordinator: Mock) -> Mock:
        """Create a coordinator with wasp-specific configuration."""
        # Customize the coordinator for wasp tests
        mock_coordinator.config.wasp_in_box = Mock()
        mock_coordinator.config.wasp_in_box.enabled = True
        mock_coordinator.config.wasp_in_box.motion_timeout = 60
        mock_coordinator.config.wasp_in_box.max_duration = 3600
        mock_coordinator.config.wasp_in_box.weight = 0.85
        mock_coordinator.config.sensors = Mock()
        mock_coordinator.config.sensors.doors = ["binary_sensor.door1"]
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]

        # Add missing entities attribute with AsyncMock
        mock_coordinator.entities = Mock()
        mock_coordinator.entities.async_initialize = AsyncMock()

        return mock_coordinator

    @pytest.fixture
    def wasp_config_entry(self, mock_config_entry: Mock) -> Mock:
        """Create a config entry with wasp-specific data."""
        mock_config_entry.data.update(
            {
                "door_sensors": ["binary_sensor.door1"],
                "motion_sensors": ["binary_sensor.motion1"],
            }
        )
        return mock_config_entry

    def test_initialization(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test WaspInBoxSensor initialization."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        assert entity.hass == mock_hass
        assert entity._coordinator == wasp_coordinator
        assert entity.unique_id == "test_entry_id_wasp_in_box"
        assert entity.name == "Wasp in Box"
        assert entity.should_poll is False

    async def test_async_added_to_hass(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test entity added to Home Assistant."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Mock state restoration and setup methods
        with (
            patch.object(entity, "_restore_previous_state") as mock_restore,
            patch.object(entity, "_setup_entity_tracking") as mock_setup,
        ):
            await entity.async_added_to_hass()

            mock_restore.assert_called_once()
            mock_setup.assert_called_once()

        # Should set wasp entity ID in coordinator
        assert wasp_coordinator.wasp_entity_id == entity.entity_id

    async def test_restore_previous_state_no_data(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test restoring previous state with no stored data."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Mock RestoreEntity.async_get_last_state to return None
        with patch.object(entity, "async_get_last_state", return_value=None):
            await entity._restore_previous_state()

            # Should have default state
            assert entity._attr_is_on is False

    async def test_restore_previous_state_with_data(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test restoring previous state with stored data."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Mock previous state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_state.attributes = {
            "last_occupied_time": "2023-01-01T12:00:00+00:00",
            "last_door_time": "2023-01-01T11:59:00+00:00",
            "last_motion_time": "2023-01-01T11:58:00+00:00",
        }

        with (
            patch.object(entity, "async_get_last_state", return_value=mock_state),
            patch.object(entity, "_start_max_duration_timer") as mock_timer,
        ):
            await entity._restore_previous_state()

            # Should restore state and attributes
            assert entity._attr_is_on is True
            assert entity._state == STATE_ON
            assert entity._last_occupied_time is not None
            mock_timer.assert_called_once()

    async def test_async_will_remove_from_hass(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test entity removal from Home Assistant."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set up some state to clean up
        entity._remove_timer = Mock()
        listener_mock = Mock()
        entity._remove_state_listener = (
            listener_mock  # Set up the mock after entity creation
        )
        wasp_coordinator.wasp_entity_id = entity.entity_id

        await entity.async_will_remove_from_hass()

        # Should clean up resources if listener exists
        if listener_mock:
            listener_mock.assert_called_once()
        assert wasp_coordinator.wasp_entity_id is None

    def test_extra_state_attributes(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test extra state attributes."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set up some state
        entity._last_occupied_time = dt_util.utcnow()
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_ON

        attributes = entity.extra_state_attributes

        assert "door_state" in attributes
        assert "motion_state" in attributes
        assert "last_motion_time" in attributes
        assert "last_door_time" in attributes
        assert "last_occupied_time" in attributes
        assert "motion_timeout" in attributes
        assert "max_duration" in attributes

    def test_weight_property(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test weight property."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        assert entity.weight == 0.85

    def test_get_valid_entities(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test _get_valid_entities method."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Mock hass.states.get to return valid states for configured entities
        def mock_get_state(entity_id):
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
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test _initialize_from_current_states method."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set entity_id to avoid NoEntitySpecifiedError
        entity.entity_id = "binary_sensor.test_wasp_in_box"

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

    def test_handle_state_change_door_opening(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test handling door opening state change."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set up initial state - occupied
        entity._attr_is_on = True

        # Mock event for door opening
        mock_event = Mock()
        mock_event.data = {
            "entity_id": "binary_sensor.door1",
            "new_state": Mock(state=STATE_ON),
            "old_state": Mock(state=STATE_OFF),
        }

        with patch.object(entity, "_process_door_state") as mock_process:
            entity._handle_state_change(mock_event)
            mock_process.assert_called_once_with("binary_sensor.door1", STATE_ON)

    def test_handle_state_change_motion_detected(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test handling motion detection state change."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Mock event for motion detection
        mock_event = Mock()
        mock_event.data = {
            "entity_id": "binary_sensor.motion1",
            "new_state": Mock(state=STATE_ON),
            "old_state": Mock(state=STATE_OFF),
        }

        with patch.object(entity, "_process_motion_state") as mock_process:
            entity._handle_state_change(mock_event)
            mock_process.assert_called_once_with("binary_sensor.motion1", STATE_ON)

    def test_process_door_state_opening_when_occupied(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test processing door opening when currently occupied."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set entity_id to avoid NoEntitySpecifiedError
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Set up initial state - occupied with door closed
        entity._attr_is_on = True
        entity._state = STATE_ON
        entity._door_state = STATE_OFF  # Door was closed

        # Mock async_write_ha_state to avoid entity registration issues
        with (
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            # Door opens (STATE_ON) while occupied
            entity._process_door_state("binary_sensor.door1", STATE_ON)

            # Should transition to unoccupied
            mock_set_state.assert_called_once_with(STATE_OFF)

    def test_process_door_state_closing_with_recent_motion(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test processing door closing with recent motion."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set entity_id to avoid NoEntitySpecifiedError
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Set up initial state - unoccupied with motion detected
        entity._attr_is_on = False
        entity._state = STATE_OFF
        entity._motion_state = STATE_ON  # Motion is active
        entity._last_motion_time = dt_util.utcnow() - timedelta(
            seconds=30
        )  # Recent motion

        # Mock async_write_ha_state to avoid entity registration issues
        with (
            patch.object(entity, "async_write_ha_state"),
            patch.object(entity, "_set_state") as mock_set_state,
        ):
            # Door closes (STATE_OFF) with motion detected
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

            # Should transition to occupied
            mock_set_state.assert_called_once_with(STATE_ON)

    def test_process_motion_state_detected(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test processing motion detection."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set entity_id to avoid NoEntitySpecifiedError
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

        # Should update motion state and time
        assert entity._motion_state == STATE_ON
        assert entity._last_motion_time is not None

    def test_set_state_to_occupied(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test setting state to occupied."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        with (
            patch.object(entity, "_start_max_duration_timer") as mock_start_timer,
            patch.object(entity, "async_write_ha_state") as mock_write_state,
        ):
            entity._set_state(STATE_ON)

            assert entity._attr_is_on is True
            assert entity._last_occupied_time is not None
            mock_start_timer.assert_called_once()
            mock_write_state.assert_called_once()

    def test_set_state_to_unoccupied(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test setting state to unoccupied."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set up occupied state
        entity._attr_is_on = True
        entity._last_occupied_time = dt_util.utcnow()
        entity._remove_timer = Mock()

        with (
            patch.object(entity, "async_write_ha_state") as mock_write_state,
            patch.object(entity, "_cancel_max_duration_timer") as mock_cancel,
        ):
            entity._set_state(STATE_OFF)

            assert entity._attr_is_on is False
            # The implementation doesn't clear _last_occupied_time in _set_state
            # It only clears it when the state actually changes to OFF
            mock_cancel.assert_called_once()
            mock_write_state.assert_called_once()

    def test_start_max_duration_timer(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test starting max duration timer."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set up the entity with max duration enabled
        entity._max_duration = 3600  # 1 hour
        entity._last_occupied_time = dt_util.utcnow()

        # Mock the async_track_point_in_time function directly
        with patch(
            "custom_components.area_occupancy.binary_sensor.async_track_point_in_time"
        ) as mock_track:
            entity._start_max_duration_timer()

            mock_track.assert_called_once()
            assert entity._remove_timer is not None

    def test_cancel_max_duration_timer(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test canceling max duration timer."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set up timer - ensure it's a Mock object
        timer_mock = Mock()
        entity._remove_timer = timer_mock

        entity._cancel_max_duration_timer()

        timer_mock.assert_called_once()
        assert entity._remove_timer is None

    def test_handle_max_duration_timeout(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test handling max duration timeout."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        with patch.object(entity, "_reset_after_max_duration") as mock_reset:
            entity._handle_max_duration_timeout(dt_util.utcnow())

            mock_reset.assert_called_once()
            # The timer should be cleared after timeout
            assert entity._remove_timer is None

    def test_reset_after_max_duration(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> None:
        """Test resetting after max duration timeout."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set up occupied state
        entity._state = STATE_ON

        # Mock the _set_state method since the actual implementation calls it
        with patch.object(entity, "_set_state") as mock_set_state:
            entity._reset_after_max_duration()

            mock_set_state.assert_called_once_with(STATE_OFF)


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @pytest.fixture
    def setup_config_entry(self, mock_config_entry: Mock) -> Mock:
        """Create a config entry for setup tests."""
        mock_config_entry.runtime_data = Mock()  # Mock coordinator
        mock_config_entry.runtime_data.config = Mock()
        mock_config_entry.runtime_data.config.wasp_in_box = Mock()
        mock_config_entry.runtime_data.config.wasp_in_box.enabled = True
        return mock_config_entry

    async def test_async_setup_entry_with_wasp_enabled(
        self, mock_hass: Mock, setup_config_entry: Mock
    ) -> None:
        """Test setup entry with wasp enabled."""
        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, setup_config_entry, mock_async_add_entities)

        # Should add both occupancy and wasp entities
        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 2
        assert isinstance(entities[0], Occupancy)
        assert isinstance(entities[1], WaspInBoxSensor)

    async def test_async_setup_entry_with_wasp_disabled(
        self, mock_hass: Mock, setup_config_entry: Mock
    ) -> None:
        """Test setup entry with wasp disabled."""
        # Disable wasp
        setup_config_entry.runtime_data.config.wasp_in_box.enabled = False

        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, setup_config_entry, mock_async_add_entities)

        # Should add only occupancy entity
        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 1
        assert isinstance(entities[0], Occupancy)


class TestWaspInBoxIntegration:
    """Test WaspInBoxSensor integration scenarios."""

    @pytest.fixture
    def comprehensive_wasp_sensor(
        self, mock_hass: Mock, wasp_coordinator: Mock, wasp_config_entry: Mock
    ) -> WaspInBoxSensor:
        """Create a comprehensive wasp sensor for testing."""
        entity = WaspInBoxSensor(mock_hass, wasp_coordinator, wasp_config_entry)

        # Set entity_id to avoid NoEntitySpecifiedError
        entity.entity_id = "binary_sensor.test_wasp_in_box"

        # Initialize with known state
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_OFF
        entity._attr_is_on = False

        return entity

    @pytest.fixture
    def wasp_coordinator(self, mock_coordinator: Mock) -> Mock:
        """Create a coordinator with wasp-specific configuration."""
        # Customize the coordinator for wasp tests
        mock_coordinator.config.wasp_in_box = Mock()
        mock_coordinator.config.wasp_in_box.enabled = True
        mock_coordinator.config.wasp_in_box.motion_timeout = 60
        mock_coordinator.config.wasp_in_box.max_duration = 3600
        mock_coordinator.config.wasp_in_box.weight = 0.85
        mock_coordinator.config.sensors = Mock()
        mock_coordinator.config.sensors.doors = ["binary_sensor.door1"]
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]

        # Add missing entities attribute with AsyncMock
        mock_coordinator.entities = Mock()
        mock_coordinator.entities.async_initialize = AsyncMock()

        return mock_coordinator

    @pytest.fixture
    def wasp_config_entry(self, mock_config_entry: Mock) -> Mock:
        """Create a config entry with wasp-specific data."""
        mock_config_entry.data.update(
            {
                "door_sensors": ["binary_sensor.door1"],
                "motion_sensors": ["binary_sensor.motion1"],
            }
        )
        return mock_config_entry

    def test_complete_wasp_occupancy_cycle(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test complete wasp occupancy detection cycle."""
        entity = comprehensive_wasp_sensor

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            # Step 1: Motion detected while unoccupied
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

            # Should update motion state
            assert entity._motion_state == STATE_ON  # type: ignore[attr-defined]

            # Step 2: Door closes with recent motion -> should trigger occupancy
            with patch.object(entity, "_start_max_duration_timer") as mock_start_timer:
                entity._process_door_state("binary_sensor.door1", STATE_OFF)

            assert entity._attr_is_on is True
            assert entity._last_occupied_time is not None  # type: ignore[attr-defined]
            mock_start_timer.assert_called_once()

            # Step 3: Door opens while occupied -> should end occupancy
            with patch.object(entity, "_cancel_max_duration_timer"):
                entity._process_door_state("binary_sensor.door1", STATE_ON)

            assert entity._attr_is_on is False
            # The implementation doesn't clear _last_occupied_time immediately
            # It's cleared when the state actually changes to OFF

    def test_wasp_timeout_scenarios(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test various timeout scenarios."""
        entity = comprehensive_wasp_sensor

        # Mock async_write_ha_state to avoid entity registration issues
        with patch.object(entity, "async_write_ha_state"):
            # Test motion timeout - old motion shouldn't trigger occupancy
            old_motion_time = dt_util.utcnow() - timedelta(seconds=120)  # 2 minutes ago
            entity._last_motion_time = old_motion_time  # type: ignore[attr-defined]
            entity._motion_state = STATE_OFF  # Motion is not active

            entity._process_door_state("binary_sensor.door1", STATE_OFF)

            # Should not trigger occupancy due to no active motion
            assert entity._attr_is_on is False

        # Test max duration timeout
        entity._attr_is_on = True
        entity._state = STATE_ON
        entity._last_occupied_time = dt_util.utcnow()  # type: ignore[attr-defined]

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
        entity._last_occupied_time = dt_util.utcnow()  # type: ignore[attr-defined]
        entity._door_state = STATE_OFF  # type: ignore[attr-defined]
        entity._motion_state = STATE_ON  # type: ignore[attr-defined]

        # Get state attributes for persistence
        attributes = entity.extra_state_attributes

        # Verify all necessary state is included
        assert "last_occupied_time" in attributes
        assert "door_state" in attributes
        assert "motion_state" in attributes
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
