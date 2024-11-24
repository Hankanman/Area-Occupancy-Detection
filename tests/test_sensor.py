"""Test the Room Occupancy Detection sensors."""

# pylint: disable=unused-argument

from datetime import timedelta
from unittest.mock import patch
from pytest_homeassistant_custom_component.common import (
    async_fire_time_changed,
    MockConfigEntry,
)
from homeassistant.const import STATE_UNAVAILABLE, STATE_ON, STATE_OFF
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.room_occupancy.const import (
    DOMAIN,
    ATTR_PROBABILITY,
    ATTR_SENSOR_PROBABILITIES,
)


async def test_probability_sensor_state(hass: HomeAssistant, mock_fully_setup_entry):
    """Test probability sensor state calculation.

    Args:
        hass: HomeAssistant test instance
        mock_fully_setup_entry: Fixture that creates and sets up a config entry with test entities.
            Required for test setup even if not directly referenced.
    """
    # Get the probability sensor state
    state = hass.states.get("sensor.test_room_occupancy_probability")
    assert state is not None
    assert float(state.state) >= 0.0
    assert float(state.state) <= 100.0

    # Verify attributes
    assert ATTR_PROBABILITY in state.attributes
    assert ATTR_SENSOR_PROBABILITIES in state.attributes


async def test_binary_sensor_state(hass: HomeAssistant, mock_fully_setup_entry):
    """Test binary sensor state based on probability threshold."""
    # Get both sensors
    prob_sensor = hass.states.get("sensor.test_room_occupancy_probability")
    bin_sensor = hass.states.get("binary_sensor.test_room_occupancy_status")

    assert prob_sensor is not None
    assert bin_sensor is not None

    # Verify binary sensor state matches probability threshold
    probability = float(prob_sensor.attributes[ATTR_PROBABILITY])
    threshold = mock_fully_setup_entry.data["threshold"]

    assert (bin_sensor.state == STATE_ON) == (probability >= threshold)


async def test_sensor_unavailable(hass: HomeAssistant, mock_fully_setup_entry):
    """Test handling of unavailable sensors."""
    # Make a sensor unavailable
    hass.states.async_set("binary_sensor.test_motion", STATE_UNAVAILABLE)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    assert state is not None
    assert "sensor_availability" in state.attributes
    assert not state.attributes["sensor_availability"]["binary_sensor.test_motion"]


async def test_probability_calculation(hass: HomeAssistant, mock_fully_setup_entry):
    """Test probability calculations with different sensor states."""
    # Set motion detected
    hass.states.async_set("binary_sensor.test_motion", STATE_ON)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    high_prob = float(state.attributes[ATTR_PROBABILITY])

    # Clear motion
    hass.states.async_set("binary_sensor.test_motion", STATE_OFF)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    low_prob = float(state.attributes[ATTR_PROBABILITY])

    assert high_prob > low_prob


async def test_sensor_decay(hass: HomeAssistant, mock_fully_setup_entry):
    """Test sensor value decay over time."""
    # Set motion detected
    hass.states.async_set("binary_sensor.test_motion", STATE_ON)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    initial_prob = float(state.attributes[ATTR_PROBABILITY])

    # Advance time by half the decay window
    future = dt_util.utcnow() + timedelta(seconds=300)
    with patch("homeassistant.util.dt.utcnow", return_value=future):
        await hass.async_block_till_done()

        state = hass.states.get("sensor.test_room_occupancy_probability")
        decayed_prob = float(state.attributes[ATTR_PROBABILITY])

        assert decayed_prob < initial_prob


async def test_multiple_motion_sensors(hass: HomeAssistant):
    """Test handling multiple motion sensors."""
    # Setup entry with multiple motion sensors
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "name": "Test Room",
            "motion_sensors": ["binary_sensor.motion_1", "binary_sensor.motion_2"],
            "threshold": 0.5,
        },
    )
    entry.add_to_hass(hass)

    # Set up mock sensors
    hass.states.async_set("binary_sensor.motion_1", STATE_OFF)
    hass.states.async_set("binary_sensor.motion_2", STATE_OFF)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Test with one motion sensor triggered
    hass.states.async_set("binary_sensor.motion_1", STATE_ON)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    single_motion_prob = float(state.attributes[ATTR_PROBABILITY])

    # Test with both motion sensors triggered
    hass.states.async_set("binary_sensor.motion_2", STATE_ON)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    double_motion_prob = float(state.attributes[ATTR_PROBABILITY])

    assert double_motion_prob > single_motion_prob


async def test_environmental_sensors(hass: HomeAssistant, mock_fully_setup_entry):
    """Test probability changes with environmental sensor changes."""
    # Get initial probability
    state = hass.states.get("sensor.test_room_occupancy_probability")
    initial_prob = float(state.attributes[ATTR_PROBABILITY])

    # Simulate environmental changes indicating occupancy
    hass.states.async_set("sensor.test_illuminance", "200")  # Higher light level
    hass.states.async_set("sensor.test_temperature", "23")  # Temperature change
    hass.states.async_set("sensor.test_humidity", "55")  # Humidity change
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    env_change_prob = float(state.attributes[ATTR_PROBABILITY])

    assert env_change_prob != initial_prob


async def test_device_state_changes(hass: HomeAssistant, mock_fully_setup_entry):
    """Test probability changes with device state changes."""
    # Get initial probability
    state = hass.states.get("sensor.test_room_occupancy_probability")
    initial_prob = float(state.attributes[ATTR_PROBABILITY])

    # Simulate device activation
    hass.states.async_set("media_player.test_tv", STATE_ON)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    device_active_prob = float(state.attributes[ATTR_PROBABILITY])

    assert device_active_prob > initial_prob


async def test_confidence_score(hass: HomeAssistant, mock_fully_setup_entry):
    """Test confidence score calculation."""
    # Start with all sensors available
    state = hass.states.get("sensor.test_room_occupancy_probability")
    initial_confidence = float(state.attributes["confidence_score"])

    # Make some sensors unavailable
    hass.states.async_set("binary_sensor.test_motion", STATE_UNAVAILABLE)
    hass.states.async_set("sensor.test_illuminance", STATE_UNAVAILABLE)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    reduced_confidence = float(state.attributes["confidence_score"])

    assert reduced_confidence < initial_confidence


async def test_sensor_weights(hass: HomeAssistant):
    """Test sensor weight configurations."""
    # Setup entry with weighted sensors
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "name": "Test Room",
            "motion_sensors": ["binary_sensor.motion"],
            "illuminance_sensors": ["sensor.illuminance"],
            "sensor_weights": {"binary_sensor.motion": 0.8, "sensor.illuminance": 0.2},
            "threshold": 0.5,
        },
    )
    entry.add_to_hass(hass)

    # Set up mock sensors
    hass.states.async_set("binary_sensor.motion", STATE_OFF)
    hass.states.async_set("sensor.illuminance", "100")

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Test weighted probability calculation
    hass.states.async_set("binary_sensor.motion", STATE_ON)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    motion_prob = float(state.attributes[ATTR_PROBABILITY])

    # Reset motion, change illuminance
    hass.states.async_set("binary_sensor.motion", STATE_OFF)
    hass.states.async_set("sensor.illuminance", "200")
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_room_occupancy_probability")
    illuminance_prob = float(state.attributes[ATTR_PROBABILITY])

    # Motion should have more impact due to higher weight
    assert abs(motion_prob - 0.5) > abs(illuminance_prob - 0.5)


async def test_coordinator_update_interval(hass: HomeAssistant, mock_fully_setup_entry):
    """Test coordinator update interval behavior."""
    with patch(
        "custom_components.room_occupancy.coordinator.RoomOccupancyCoordinator._async_update_data"
    ) as mock_update:
        # Force time to pass
        future = dt_util.utcnow() + timedelta(seconds=11)  # Just over update interval
        with patch("homeassistant.util.dt.utcnow", return_value=future):
            async_fire_time_changed(hass, future)
            await hass.async_block_till_done()

            assert len(mock_update.mock_calls) == 1
