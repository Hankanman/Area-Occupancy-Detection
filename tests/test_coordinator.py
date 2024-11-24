"""Tests for the Room Occupancy Detection coordinator."""

# pylint: disable=protected-access

from datetime import datetime, timedelta
from unittest.mock import patch
import pytest
from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, State
from custom_components.room_occupancy.coordinator import RoomOccupancyCoordinator
from custom_components.room_occupancy.const import (
    CONF_MOTION_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_TYPE,
)


async def test_coordinator_setup(hass: HomeAssistant, mock_fully_setup_entry):
    """Test coordinator initialization and setup."""
    coordinator = hass.data["room_occupancy"][mock_fully_setup_entry.entry_id][
        "coordinator"
    ]

    assert coordinator is not None
    assert coordinator.config == mock_fully_setup_entry.data
    assert coordinator.update_interval == timedelta(seconds=10)
    assert coordinator.name == "room_occupancy"


async def test_coordinator_sensor_tracking(hass: HomeAssistant):
    """Test sensor state tracking setup."""
    config = {
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_ILLUMINANCE_SENSORS: ["sensor.test_light"],
    }

    coordinator = RoomOccupancyCoordinator(
        hass,
        "test_entry_id",
        config,
    )

    # Verify sensor tracking is set up
    assert "binary_sensor.test_motion" in coordinator._sensor_states
    assert "sensor.test_light" in coordinator._sensor_states

    # Test state change tracking
    hass.states.async_set("binary_sensor.test_motion", STATE_ON)
    await hass.async_block_till_done()

    assert coordinator._sensor_states["binary_sensor.test_motion"].state == STATE_ON


async def test_coordinator_decay_calculation(hass: HomeAssistant):
    """Test decay factor calculations."""
    config = {
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_DECAY_ENABLED: True,
        CONF_DECAY_WINDOW: 600,
        CONF_DECAY_TYPE: "linear",
    }

    coordinator = RoomOccupancyCoordinator(
        hass,
        "test_entry_id",
        config,
    )

    # Test linear decay
    now = datetime.now()
    half_window = now - timedelta(seconds=300)

    with patch("homeassistant.util.dt.utcnow", return_value=now):
        decay = coordinator._calculate_decay("binary_sensor.test_motion", half_window)
        assert decay == pytest.approx(0.5)  # Half-way through decay window

    # Test exponential decay
    coordinator.config[CONF_DECAY_TYPE] = "exponential"
    with patch("homeassistant.util.dt.utcnow", return_value=now):
        decay = coordinator._calculate_decay("binary_sensor.test_motion", half_window)
        assert decay == pytest.approx(0.22313, rel=1e-3)  # exp(-3 * 0.5)


async def test_coordinator_sensor_probability(hass: HomeAssistant):
    """Test individual sensor probability calculations."""
    config = {
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_ILLUMINANCE_SENSORS: ["sensor.test_light"],
    }

    coordinator = RoomOccupancyCoordinator(
        hass,
        "test_entry_id",
        config,
    )

    # Test motion sensor probability
    motion_state = State("binary_sensor.test_motion", STATE_ON)
    prob = coordinator._get_sensor_probability(
        "binary_sensor.test_motion", motion_state
    )
    assert prob == pytest.approx(0.95)  # High probability when motion detected

    # Test illuminance sensor probability
    light_state = State("sensor.test_light", "200")
    prob = coordinator._get_sensor_probability("sensor.test_light", light_state)
    assert prob == pytest.approx(0.7)  # High light level probability


async def test_coordinator_occupancy_calculation(hass: HomeAssistant):
    """Test overall occupancy probability calculation."""
    config = {
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_ILLUMINANCE_SENSORS: ["sensor.test_light"],
    }

    coordinator = RoomOccupancyCoordinator(
        hass,
        "test_entry_id",
        config,
    )

    # Set up initial states
    hass.states.async_set("binary_sensor.test_motion", STATE_OFF)
    hass.states.async_set("sensor.test_light", "50")
    await hass.async_block_till_done()

    result = coordinator._calculate_occupancy()

    assert "probability" in result
    assert "sensor_probabilities" in result
    assert "active_triggers" in result
    assert "decay_status" in result
    assert "confidence_score" in result
    assert "sensor_availability" in result


async def test_coordinator_unavailable_sensors(hass: HomeAssistant):
    """Test handling of unavailable sensors."""
    config = {
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_ILLUMINANCE_SENSORS: ["sensor.test_light"],
    }

    coordinator = RoomOccupancyCoordinator(
        hass,
        "test_entry_id",
        config,
    )

    # Make sensors unavailable
    hass.states.async_set("binary_sensor.test_motion", STATE_UNAVAILABLE)
    hass.states.async_set("sensor.test_light", STATE_UNKNOWN)
    await hass.async_block_till_done()

    result = coordinator._calculate_occupancy()

    assert result["confidence_score"] == 0.0
    assert not any(result["sensor_availability"].values())


async def test_coordinator_update_data(hass: HomeAssistant):
    """Test coordinator data update method."""
    config = {
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
    }

    coordinator = RoomOccupancyCoordinator(
        hass,
        "test_entry_id",
        config,
    )

    # Test normal update
    data = await coordinator._async_update_data()
    assert data is not None
    assert "probability" in data

    # Test update with exception
    with patch.object(
        coordinator, "_calculate_occupancy", side_effect=Exception("Test error")
    ):
        with pytest.raises(Exception):
            await coordinator._async_update_data()


async def test_coordinator_multiple_updates(hass: HomeAssistant):
    """Test coordinator behavior with multiple rapid updates."""
    config = {
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
    }

    coordinator = RoomOccupancyCoordinator(
        hass,
        "test_entry_id",
        config,
    )

    # Simulate rapid state changes
    for _ in range(5):
        hass.states.async_set("binary_sensor.test_motion", STATE_ON)
        await hass.async_block_till_done()
        hass.states.async_set("binary_sensor.test_motion", STATE_OFF)
        await hass.async_block_till_done()

    assert coordinator.data is not None
    assert "probability" in coordinator.data


async def test_coordinator_cleanup(hass: HomeAssistant):
    """Test coordinator cleanup on unload."""
    config = {
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
    }

    coordinator = RoomOccupancyCoordinator(
        hass,
        "test_entry_id",
        config,
    )

    # Store original listeners count
    initial_listeners = len(hass.bus.async_listeners())

    # Clean up coordinator
    await coordinator.async_shutdown()

    # Verify listeners were removed
    assert len(hass.bus.async_listeners()) < initial_listeners
