"""Tests for room_occupancy integration."""
from unittest.mock import patch
from pytest_homeassistant_custom_component.common import MockConfigEntry
from homeassistant.const import STATE_ON, STATE_OFF
from homeassistant.core import HomeAssistant


async def test_setup(hass: HomeAssistant, init_integration: MockConfigEntry) -> None:
    """Test integration setup."""
    # Verify entities were created
    assert hass.states.get("binary_sensor.room_occupancy_status") is not None
    assert hass.states.get("sensor.room_occupancy_probability") is not None


async def test_motion_trigger(hass: HomeAssistant, init_integration: MockConfigEntry) -> None:
    """Test probability changes with motion detection."""
    # Initial state - no motion
    hass.states.async_set("binary_sensor.motion1", STATE_OFF)
    hass.states.async_set("binary_sensor.motion2", STATE_OFF)
    await hass.async_block_till_done()

    initial_state = hass.states.get("sensor.room_occupancy_probability")
    assert initial_state is not None
    initial_prob = float(initial_state.state)

    # Trigger motion sensor
    hass.states.async_set("binary_sensor.motion1", STATE_ON)
    await hass.async_block_till_done()

    # Check probability increased
    new_state = hass.states.get("sensor.room_occupancy_probability")
    assert new_state is not None
    new_prob = float(new_state.state)
    assert new_prob > initial_prob


async def test_binary_sensor_threshold(hass: HomeAssistant, init_integration: MockConfigEntry) -> None:
    """Test binary sensor state changes based on threshold."""
    # Set high probability with motion
    hass.states.async_set("binary_sensor.motion1", STATE_ON)
    hass.states.async_set("binary_sensor.motion2", STATE_ON)
    await hass.async_block_till_done()

    # Check binary sensor is on
    binary_state = hass.states.get("binary_sensor.room_occupancy_status")
    assert binary_state is not None
    assert binary_state.state == STATE_ON

    # Set low probability with no motion
    hass.states.async_set("binary_sensor.motion1", STATE_OFF)
    hass.states.async_set("binary_sensor.motion2", STATE_OFF)
    await hass.async_block_till_done()

    # Check binary sensor is off
    binary_state = hass.states.get("binary_sensor.room_occupancy_status")
    assert binary_state is not None
    assert binary_state.state == STATE_OFF


async def test_coordinator_update(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> None:
    """Test coordinator data updates."""
    mock_data = {
        "probability": 0.75,
        "prior_probability": 0.5,
        "active_triggers": ["binary_sensor.motion1"],
        "sensor_probabilities": {"binary_sensor.motion1": 0.95},
        "decay_status": {"binary_sensor.motion1": 1.0},
        "confidence_score": 0.8,
        "sensor_availability": {"binary_sensor.motion1": True},
    }

    with patch(
        "custom_components.room_occupancy.coordinator.RoomOccupancyCoordinator._async_update_data",
        return_value=mock_data,
    ):
        mock_config_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

        prob_state = hass.states.get("sensor.room_occupancy_probability")
        assert prob_state is not None
        assert float(prob_state.state) == 75.0
        assert prob_state.attributes["probability"] == 0.75
        assert prob_state.attributes["confidence_score"] == 0.8
