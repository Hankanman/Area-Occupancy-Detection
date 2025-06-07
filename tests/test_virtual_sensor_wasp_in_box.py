"""Tests for the Wasp in Box virtual sensor."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    ATTR_DOOR_STATE,
    ATTR_LAST_DOOR_TIME,
    ATTR_LAST_MOTION_TIME,
    ATTR_MOTION_STATE,
    ATTR_MOTION_TIMEOUT,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_WEIGHT,
    NAME_WASP_IN_BOX,
)
from custom_components.area_occupancy.types import WaspInBoxConfig
from custom_components.area_occupancy.virtual_sensor.wasp_in_box import (
    ATTR_LAST_OCCUPIED_TIME,
    WaspInBoxSensor,
    async_setup_entry,
)
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

# Note: Using fixtures from conftest.py:
# - mock_coordinator
# - mock_config_entry


@pytest.fixture
def wasp_config() -> WaspInBoxConfig:
    """Create a basic wasp sensor configuration."""
    return {
        "enabled": True,
        "motion_timeout": DEFAULT_WASP_MOTION_TIMEOUT,
        "weight": DEFAULT_WASP_WEIGHT,
        "max_duration": DEFAULT_WASP_MAX_DURATION,
    }


def create_test_sensor(
    hass: HomeAssistant, config: WaspInBoxConfig, coordinator: Mock, entry_id: str
) -> WaspInBoxSensor:
    """Create a test sensor with proper entity ID."""
    sensor = WaspInBoxSensor(hass, config, coordinator, entry_id)
    sensor.entity_id = f"binary_sensor.test_wasp_{entry_id}"
    return sensor


@pytest.fixture
async def test_sensor(hass: HomeAssistant, mock_coordinator, wasp_config):
    """Create and clean up a test sensor."""
    sensor = create_test_sensor(hass, wasp_config, mock_coordinator, "test_entry_id")
    await sensor.async_added_to_hass()
    yield sensor
    await sensor.async_will_remove_from_hass()


async def test_setup_disabled(hass: HomeAssistant, mock_coordinator):
    """Test setup when sensor is disabled."""
    config: WaspInBoxConfig = {"enabled": False}
    result = await async_setup_entry(hass, config, Mock(), mock_coordinator)
    assert result is None


async def test_setup_enabled(hass: HomeAssistant, mock_coordinator, wasp_config):
    """Test setup when sensor is enabled."""
    sensor = await async_setup_entry(hass, wasp_config, Mock(), mock_coordinator)
    try:
        assert sensor is not None
        assert isinstance(sensor, WaspInBoxSensor)
        assert sensor.name == NAME_WASP_IN_BOX
        assert sensor.device_class == BinarySensorDeviceClass.OCCUPANCY
        assert (
            sensor.unique_id
            == f"test_entry_id_{NAME_WASP_IN_BOX.lower().replace(' ', '_')}"
        )
    finally:
        if sensor:
            await sensor.async_will_remove_from_hass()


async def test_sensor_attributes(hass: HomeAssistant, test_sensor):
    """Test the sensor attributes are set correctly."""
    # Check initial attributes
    attrs = test_sensor.extra_state_attributes
    assert attrs.get(ATTR_DOOR_STATE) == STATE_OFF
    assert attrs.get(ATTR_MOTION_STATE) == STATE_OFF
    assert attrs.get(ATTR_MOTION_TIMEOUT) == DEFAULT_WASP_MOTION_TIMEOUT
    assert attrs.get("max_duration") == DEFAULT_WASP_MAX_DURATION
    assert attrs.get(ATTR_LAST_DOOR_TIME) is None
    assert attrs.get(ATTR_LAST_MOTION_TIME) is None
    assert attrs.get(ATTR_LAST_OCCUPIED_TIME) is None

    # Check other properties
    assert test_sensor.weight == DEFAULT_WASP_WEIGHT
    assert not test_sensor.should_poll
    assert test_sensor.available is True  # Should be True until added to HA


async def test_basic_door_motion_interaction(test_sensor):
    """Test basic interaction between door and motion states."""
    now = dt_util.utcnow()

    with patch("homeassistant.util.dt.utcnow", return_value=now):
        # Initial state should be unoccupied
        assert test_sensor.state == STATE_OFF

        # Simulate motion detection with door closed
        test_sensor._process_door_state("binary_sensor.test_door", STATE_OFF)
        test_sensor._process_motion_state("binary_sensor.test_motion", STATE_ON)

        # Should now be occupied
        assert test_sensor.state == STATE_ON
        assert test_sensor.is_on

        # Simulate door opening
        test_sensor._process_door_state("binary_sensor.test_door", STATE_ON)

        # Should now be unoccupied
        assert test_sensor.state == STATE_OFF
        assert not test_sensor.is_on


async def test_state_restoration(hass: HomeAssistant, mock_coordinator, wasp_config):
    """Test restoring state from previous run."""
    now = dt_util.utcnow()
    last_door_time = now - timedelta(minutes=5)
    last_motion_time = now - timedelta(minutes=3)
    last_occupied_time = now - timedelta(minutes=3)

    # Create mock last state
    mock_state = State(
        "binary_sensor.test_wasp",
        STATE_ON,
        {
            ATTR_DOOR_STATE: STATE_OFF,
            ATTR_MOTION_STATE: STATE_ON,
            ATTR_LAST_DOOR_TIME: last_door_time.isoformat(),
            ATTR_LAST_MOTION_TIME: last_motion_time.isoformat(),
            ATTR_LAST_OCCUPIED_TIME: last_occupied_time.isoformat(),
        },
    )

    with (
        patch(
            "homeassistant.helpers.restore_state.RestoreEntity.async_get_last_state",
            return_value=mock_state,
        ),
        patch("homeassistant.util.dt.utcnow", return_value=now),
    ):
        sensor = create_test_sensor(
            hass, wasp_config, mock_coordinator, "test_entry_id"
        )
        try:
            await sensor.async_added_to_hass()

            # Verify restored state
            assert sensor.state == STATE_ON
            assert sensor.is_on
            attrs = sensor.extra_state_attributes

            # Safe parsing of datetime attributes with type checking
            door_time = attrs.get(ATTR_LAST_DOOR_TIME)
            motion_time = attrs.get(ATTR_LAST_MOTION_TIME)
            occupied_time = attrs.get(ATTR_LAST_OCCUPIED_TIME)

            assert door_time and dt_util.parse_datetime(door_time) == last_door_time
            assert (
                motion_time and dt_util.parse_datetime(motion_time) == last_motion_time
            )
            assert (
                occupied_time
                and dt_util.parse_datetime(occupied_time) == last_occupied_time
            )
        finally:
            await sensor.async_will_remove_from_hass()


async def test_max_duration_timeout(test_sensor):
    """Test max duration timeout functionality."""
    now = dt_util.utcnow()
    max_duration = 300  # 5 minutes

    with patch("homeassistant.util.dt.utcnow") as mock_time:
        mock_time.return_value = now

        # Set up initial occupied state
        test_sensor._process_door_state("binary_sensor.test_door", STATE_OFF)
        test_sensor._process_motion_state("binary_sensor.test_motion", STATE_ON)
        assert test_sensor.state == STATE_ON

        # Advance time past max duration
        mock_time.return_value = now + timedelta(seconds=max_duration + 1)

        # Trigger the timeout callback
        test_sensor._handle_max_duration_timeout(mock_time.return_value)

        # Verify sensor is now unoccupied
        assert test_sensor.state == STATE_OFF
        assert not test_sensor.is_on


async def test_coordinator_state_updates(test_sensor, mock_coordinator):
    """Test that coordinator state is updated properly."""
    # Simulate occupancy detection
    test_sensor._process_door_state("binary_sensor.test_door", STATE_OFF)
    test_sensor._process_motion_state("binary_sensor.test_motion", STATE_ON)

    # Verify coordinator was updated
    assert mock_coordinator.async_request_refresh.called
    assert test_sensor.entity_id in mock_coordinator.data.current_states
    state_data = mock_coordinator.data.current_states[test_sensor.entity_id]
    assert state_data["state"] == STATE_ON
    assert "last_changed" in state_data
    assert state_data["availability"] is True


async def test_invalid_entity_handling(test_sensor):
    """Test handling of invalid or unavailable entities."""
    # Verify warning is logged when processing invalid entity
    test_sensor._process_door_state("binary_sensor.nonexistent_door", STATE_ON)

    # State should remain unchanged
    assert test_sensor.state == STATE_OFF
    assert not test_sensor.is_on
