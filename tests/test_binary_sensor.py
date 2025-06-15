"""Tests for binary_sensor module."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.binary_sensor import (
    Occupancy,
    WaspInBoxSensor,
    async_setup_entry,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util


class TestOccupancy:
    """Test Occupancy binary sensor entity."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.is_occupied = True
        coordinator.probability = 0.75
        coordinator.threshold = 0.6
        coordinator.last_updated = dt_util.utcnow()
        coordinator.last_changed = dt_util.utcnow()
        coordinator.available = True
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }
        return coordinator

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test Occupancy entity initialization."""
        entity = Occupancy(mock_coordinator, "test_entry")

        assert entity.coordinator == mock_coordinator
        assert entity.unique_id == "test_entry_occupancy_status"
        assert entity.name == "Occupancy Status"

    async def test_async_added_to_hass(self, mock_coordinator: Mock) -> None:
        """Test entity added to Home Assistant."""
        entity = Occupancy(mock_coordinator, "test_entry")

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
        entity = Occupancy(mock_coordinator, "test_entry")

        await entity.async_will_remove_from_hass()

        # Should clear occupancy entity ID in coordinator
        assert mock_coordinator.occupancy_entity_id is None

    def test_icon_property(self, mock_coordinator: Mock) -> None:
        """Test icon property."""
        entity = Occupancy(mock_coordinator, "test_entry")

        # Test occupied state
        mock_coordinator.is_occupied = True
        assert entity.icon == "mdi:home-account"

        # Test unoccupied state
        mock_coordinator.is_occupied = False
        assert entity.icon == "mdi:home-outline"

    def test_is_on_property(self, mock_coordinator: Mock) -> None:
        """Test is_on property."""
        entity = Occupancy(mock_coordinator, "test_entry")

        # Test occupied state
        mock_coordinator.is_occupied = True
        assert entity.is_on is True

        # Test unoccupied state
        mock_coordinator.is_occupied = False
        assert entity.is_on is False

    def test_extra_state_attributes(self, mock_coordinator: Mock) -> None:
        """Test extra state attributes."""
        entity = Occupancy(mock_coordinator, "test_entry")

        attributes = entity.extra_state_attributes

        # The Occupancy class should have extra_state_attributes that include probability and threshold
        assert attributes is not None
        assert "probability" in attributes
        assert "threshold" in attributes
        assert attributes["probability"] == 75  # Converted to percentage
        assert attributes["threshold"] == 60  # Converted to percentage


class TestWaspInBoxSensor:
    """Test WaspInBoxSensor binary sensor entity."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.states = Mock()
        hass.states.get = Mock(return_value=None)
        hass.async_create_task = Mock()
        return hass

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.hass = Mock()
        coordinator.config = Mock()
        coordinator.config.wasp_in_box = Mock()
        coordinator.config.wasp_in_box.enabled = True
        coordinator.config.wasp_in_box.motion_timeout = 60
        coordinator.config.wasp_in_box.max_duration = 3600
        coordinator.config.wasp_in_box.weight = 0.85
        coordinator.config.sensors = Mock()
        coordinator.config.sensors.doors = ["binary_sensor.door1"]
        coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        coordinator.request_update = Mock()
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }
        return coordinator

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry"
        entry.data = {
            "name": "Test Area",
            "door_sensors": ["binary_sensor.door1"],
            "motion_sensors": ["binary_sensor.motion1"],
        }
        return entry

    def test_initialization(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test WaspInBoxSensor initialization."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        assert entity.hass == mock_hass
        assert entity._coordinator == mock_coordinator
        assert entity.unique_id == "test_entry_wasp_in_box"
        assert entity.name == "Wasp in Box"
        assert entity.should_poll is False

    async def test_async_added_to_hass(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test entity added to Home Assistant."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Mock state restoration
        with patch.object(entity, "_restore_previous_state") as mock_restore:
            with patch.object(entity, "_setup_entity_tracking") as mock_setup:
                with patch.object(
                    entity, "_initialize_from_current_states"
                ) as mock_init:
                    await entity.async_added_to_hass()

                    mock_restore.assert_called_once()
                    mock_setup.assert_called_once()
                    mock_init.assert_called_once()

        # Should set wasp entity ID in coordinator
        assert mock_coordinator.wasp_entity_id == entity.entity_id

    async def test_restore_previous_state_no_data(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test restoring previous state with no stored data."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Mock RestoreEntity.async_get_last_state to return None
        with patch.object(entity, "async_get_last_state", return_value=None):
            await entity._restore_previous_state()

            # Should have default state
            assert entity._attr_is_on is False

    async def test_restore_previous_state_with_data(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test restoring previous state with stored data."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Mock previous state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_state.attributes = {
            "occupied_since": "2023-01-01T12:00:00+00:00",
            "door_states": {"binary_sensor.door1": STATE_OFF},
            "motion_states": {"binary_sensor.motion1": STATE_OFF},
        }

        with patch.object(entity, "async_get_last_state", return_value=mock_state):
            await entity._restore_previous_state()

            # Should restore state and attributes
            assert entity._attr_is_on is True
            assert entity._last_occupied_time is not None
            assert entity._door_state == STATE_ON
            assert entity._motion_state == STATE_ON

    async def test_async_will_remove_from_hass(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test entity removal from Home Assistant."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Set up some state to clean up
        entity._remove_timer = Mock()
        entity._remove_state_listener = Mock()

        await entity.async_will_remove_from_hass()

        # Should clean up resources
        entity._remove_timer.assert_called_once()
        entity._remove_state_listener.assert_called_once()
        assert mock_coordinator.wasp_entity_id is None

    def test_extra_state_attributes(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test extra state attributes."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Set up some state
        entity._last_occupied_time = dt_util.utcnow()
        entity._door_state = STATE_OFF
        entity._motion_state = STATE_ON

        attributes = entity.extra_state_attributes

        assert "occupied_since" in attributes
        assert "door_states" in attributes
        assert "motion_states" in attributes
        assert "motion_timeout" in attributes
        assert "max_duration" in attributes

    def test_weight_property(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test weight property."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        assert entity.weight == 0.85

    def test_get_valid_entities(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _get_valid_entities method."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Mock hass.states.async_all() to return some states
        mock_hass.states.async_all.return_value = [
            Mock(entity_id="binary_sensor.door1"),
            Mock(entity_id="binary_sensor.motion1"),
            Mock(entity_id="binary_sensor.other"),
        ]

        result = entity._get_valid_entities()

        assert "doors" in result
        assert "motion" in result
        assert "binary_sensor.door1" in result["doors"]
        assert "binary_sensor.motion1" in result["motion"]
        assert "binary_sensor.other" not in result["doors"]
        assert "binary_sensor.other" not in result["motion"]

    def test_initialize_from_current_states(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _initialize_from_current_states method."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        valid_entities = {
            "doors": ["binary_sensor.door1"],
            "motion": ["binary_sensor.motion1"],
        }

        # Mock current states
        mock_hass.states.get.side_effect = lambda entity_id: Mock(state=STATE_OFF)

        entity._initialize_from_current_states(valid_entities)

        # Should initialize state dictionaries
        assert "binary_sensor.door1" in entity._door_states
        assert "binary_sensor.motion1" in entity._motion_states
        assert entity._door_states["binary_sensor.door1"] == STATE_OFF
        assert entity._motion_states["binary_sensor.motion1"] == STATE_OFF

    def test_handle_state_change_door_opening(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test handling door opening state change."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Set up initial state - occupied
        entity._attr_is_on = True
        entity._door_states = {"binary_sensor.door1": STATE_OFF}

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
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test handling motion detection state change."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Set up initial state
        entity._motion_states = {"binary_sensor.motion1": STATE_OFF}

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
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test processing door opening when currently occupied."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Set up initial state - occupied
        entity._attr_is_on = True
        entity._door_states = {"binary_sensor.door1": STATE_OFF}

        with patch.object(entity, "_set_state") as mock_set_state:
            entity._process_door_state("binary_sensor.door1", STATE_ON)

            # Should transition to unoccupied
            mock_set_state.assert_called_once_with(STATE_OFF)
            assert entity._door_states["binary_sensor.door1"] == STATE_ON

    def test_process_door_state_closing_with_recent_motion(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test processing door closing with recent motion."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Set up initial state - unoccupied with recent motion
        entity._attr_is_on = False
        entity._door_states = {"binary_sensor.door1": STATE_ON}
        entity._last_motion_time = dt_util.utcnow() - timedelta(
            seconds=30
        )  # Recent motion

        with patch.object(entity, "_set_state") as mock_set_state:
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

            # Should transition to occupied
            mock_set_state.assert_called_once_with(STATE_ON)

    def test_process_motion_state_detected(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test processing motion detection."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        entity._motion_states = {"binary_sensor.motion1": STATE_OFF}

        entity._process_motion_state("binary_sensor.motion1", STATE_ON)

        # Should update motion state and time
        assert entity._motion_states["binary_sensor.motion1"] == STATE_ON
        assert entity._last_motion_time is not None

    def test_set_state_to_occupied(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setting state to occupied."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        with patch.object(entity, "_start_max_duration_timer") as mock_start_timer:
            with patch.object(entity, "async_write_ha_state") as mock_write_state:
                entity._set_state(STATE_ON)

                assert entity._attr_is_on is True
                assert entity._occupied_since is not None
                mock_start_timer.assert_called_once()
                mock_write_state.assert_called_once()

    def test_set_state_to_unoccupied(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setting state to unoccupied."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Set up occupied state
        entity._attr_is_on = True
        entity._occupied_since = dt_util.utcnow()
        entity._max_duration_timer = Mock()

        with patch.object(entity, "async_write_ha_state") as mock_write_state:
            entity._set_state(STATE_OFF)

            assert entity._attr_is_on is False
            assert entity._occupied_since is None
            entity._max_duration_timer.assert_called_once()  # Should cancel timer
            mock_write_state.assert_called_once()

    def test_start_max_duration_timer(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test starting max duration timer."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        with patch(
            "homeassistant.helpers.event.async_track_point_in_time"
        ) as mock_track:
            entity._start_max_duration_timer()

            mock_track.assert_called_once()
            assert entity._max_duration_timer is not None

    def test_cancel_max_duration_timer(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test canceling max duration timer."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Set up timer
        entity._max_duration_timer = Mock()

        entity._cancel_max_duration_timer()

        entity._max_duration_timer.assert_called_once()
        assert entity._max_duration_timer is None

    def test_handle_max_duration_timeout(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test handling max duration timeout."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        with patch.object(entity, "_reset_after_max_duration") as mock_reset:
            entity._handle_max_duration_timeout(dt_util.utcnow())

            mock_reset.assert_called_once()

    def test_reset_after_max_duration(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> None:
        """Test resetting after max duration timeout."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        with patch.object(entity, "_set_state") as mock_set_state:
            entity._reset_after_max_duration()

            mock_set_state.assert_called_once_with(STATE_OFF)


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
        entry.runtime_data.config = Mock()
        entry.runtime_data.config.wasp_in_box = Mock()
        entry.runtime_data.config.wasp_in_box.enabled = True
        return entry

    async def test_async_setup_entry_with_wasp_enabled(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup entry with wasp enabled."""
        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        # Should add both occupancy and wasp entities
        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 2
        assert isinstance(entities[0], Occupancy)
        assert isinstance(entities[1], WaspInBoxSensor)

    async def test_async_setup_entry_with_wasp_disabled(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup entry with wasp disabled."""
        # Disable wasp
        mock_config_entry.runtime_data.config.wasp_in_box.enabled = False

        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        # Should add only occupancy entity
        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 1
        assert isinstance(entities[0], Occupancy)


class TestWaspInBoxIntegration:
    """Test WaspInBoxSensor integration scenarios."""

    @pytest.fixture
    def comprehensive_wasp_sensor(
        self, mock_hass: Mock, mock_coordinator: Mock, mock_config_entry: Mock
    ) -> WaspInBoxSensor:
        """Create a comprehensive wasp sensor for testing."""
        entity = WaspInBoxSensor(mock_hass, mock_coordinator, mock_config_entry)

        # Initialize with known state
        entity._door_states = {"binary_sensor.door1": STATE_OFF}
        entity._motion_states = {"binary_sensor.motion1": STATE_OFF}
        entity._attr_is_on = False

        return entity

    def test_complete_wasp_occupancy_cycle(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test complete wasp occupancy detection cycle."""
        entity = comprehensive_wasp_sensor

        # Step 1: Motion detected while unoccupied
        with patch.object(entity, "async_write_ha_state"):
            entity._process_motion_state("binary_sensor.motion1", STATE_ON)

        # Should update motion state but not trigger occupancy yet
        assert entity._motion_states["binary_sensor.motion1"] == STATE_ON
        assert entity._attr_is_on is False

        # Step 2: Door closes with recent motion -> should trigger occupancy
        with patch.object(entity, "_start_max_duration_timer") as mock_start_timer:
            with patch.object(entity, "async_write_ha_state"):
                entity._process_door_state("binary_sensor.door1", STATE_OFF)

        assert entity._attr_is_on is True
        assert entity._occupied_since is not None
        mock_start_timer.assert_called_once()

        # Step 3: Door opens while occupied -> should end occupancy
        with patch.object(entity, "async_write_ha_state"):
            entity._process_door_state("binary_sensor.door1", STATE_ON)

        assert entity._attr_is_on is False
        assert entity._occupied_since is None

    def test_wasp_timeout_scenarios(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test various timeout scenarios."""
        entity = comprehensive_wasp_sensor

        # Test motion timeout - old motion shouldn't trigger occupancy
        old_motion_time = dt_util.utcnow() - timedelta(seconds=120)  # 2 minutes ago
        entity._last_motion_time = old_motion_time

        with patch.object(entity, "async_write_ha_state"):
            entity._process_door_state("binary_sensor.door1", STATE_OFF)

        # Should not trigger occupancy due to old motion
        assert entity._attr_is_on is False

        # Test max duration timeout
        entity._attr_is_on = True
        entity._occupied_since = dt_util.utcnow()

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
        entity._occupied_since = dt_util.utcnow()
        entity._door_states = {"binary_sensor.door1": STATE_OFF}
        entity._motion_states = {"binary_sensor.motion1": STATE_ON}

        # Get state attributes for persistence
        attributes = entity.extra_state_attributes

        # Verify all necessary state is included
        assert "occupied_since" in attributes
        assert "door_states" in attributes
        assert "motion_states" in attributes
        assert attributes["door_states"]["binary_sensor.door1"] == STATE_OFF
        assert attributes["motion_states"]["binary_sensor.motion1"] == STATE_ON

    def test_error_handling_during_state_changes(
        self, comprehensive_wasp_sensor: WaspInBoxSensor
    ) -> None:
        """Test error handling during state changes."""
        entity = comprehensive_wasp_sensor

        # Mock an error during state writing
        with patch.object(
            entity, "async_write_ha_state", side_effect=Exception("Write failed")
        ):
            # Should handle error gracefully and not crash
            try:
                entity._set_state(STATE_ON)
            except Exception:
                pytest.fail("Should handle write state errors gracefully")

        # State should still be updated internally even if write fails
        assert entity._attr_is_on is True
