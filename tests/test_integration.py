"""Tests for area_occupancy integration."""

import asyncio
from unittest.mock import patch
from pytest_homeassistant_custom_component.common import MockConfigEntry
from homeassistant.const import STATE_ON, STATE_OFF
from homeassistant.core import HomeAssistant
from custom_components.area_occupancy.const import DOMAIN


async def test_setup(hass: HomeAssistant, init_integration: MockConfigEntry) -> None:
    """Test integration setup."""
    # Wait for setup to complete
    await asyncio.sleep(2)
    await hass.async_block_till_done()

    # Verify entities were created and have valid states
    state = hass.states.get("binary_sensor.test_area_occupancy_status")
    assert state is not None
    assert state.state in [STATE_ON, STATE_OFF]

    state = hass.states.get("sensor.test_area_occupancy_probability")
    assert state is not None
    assert float(state.state) >= 0

    state = hass.states.get("sensor.test_area_decay_status")
    assert state is not None
    assert float(state.state) >= 0

    state = hass.states.get("sensor.test_area_prior_probability")
    assert state is not None
    assert float(state.state) >= 0

    state = hass.states.get("number.test_area_occupancy_threshold")
    assert state is not None
    assert float(state.state) > 0


async def test_motion_trigger(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Test probability changes with motion detection."""
    # Wait for setup to complete
    await asyncio.sleep(2)
    await hass.async_block_till_done()

    # Initial state - no motion
    hass.states.async_set("binary_sensor.motion1", STATE_OFF)
    hass.states.async_set("binary_sensor.motion2", STATE_OFF)
    await hass.async_block_till_done()
    await asyncio.sleep(1)

    initial_state = hass.states.get("sensor.test_area_occupancy_probability")
    assert initial_state is not None
    initial_prob = float(initial_state.state)

    # Trigger motion sensor
    hass.states.async_set("binary_sensor.motion1", STATE_ON)
    await hass.async_block_till_done()
    await asyncio.sleep(1)

    # Check probability increased
    new_state = hass.states.get("sensor.test_area_occupancy_probability")
    assert new_state is not None
    new_prob = float(new_state.state)
    assert new_prob > initial_prob


async def test_binary_sensor_threshold(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Test binary sensor state changes based on threshold."""
    # Wait for setup to complete
    await asyncio.sleep(2)
    await hass.async_block_till_done()

    # Set high probability with motion
    hass.states.async_set("binary_sensor.motion1", STATE_ON)
    hass.states.async_set("binary_sensor.motion2", STATE_ON)
    await hass.async_block_till_done()
    await asyncio.sleep(1)

    # Check binary sensor is on when probability is above threshold
    binary_state = hass.states.get("binary_sensor.test_area_occupancy_status")
    assert binary_state is not None
    prob_state = hass.states.get("sensor.test_area_occupancy_probability")
    assert prob_state is not None
    prob = float(prob_state.state)
    threshold = float(hass.states.get("number.test_area_occupancy_threshold").state)
    expected_state = STATE_ON if prob >= threshold else STATE_OFF
    assert binary_state.state == expected_state

    # Set low probability with no motion
    hass.states.async_set("binary_sensor.motion1", STATE_OFF)
    hass.states.async_set("binary_sensor.motion2", STATE_OFF)
    await hass.async_block_till_done()
    await asyncio.sleep(1)

    # Check binary sensor is off when probability is below threshold
    binary_state = hass.states.get("binary_sensor.test_area_occupancy_status")
    assert binary_state is not None
    prob_state = hass.states.get("sensor.test_area_occupancy_probability")
    assert prob_state is not None
    prob = float(prob_state.state)
    threshold = float(hass.states.get("number.test_area_occupancy_threshold").state)
    expected_state = STATE_ON if prob >= threshold else STATE_OFF
    assert binary_state.state == expected_state


async def test_coordinator_update(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Test coordinator data updates."""
    # Wait for setup to complete
    await asyncio.sleep(2)
    await hass.async_block_till_done()

    mock_data = {
        "probability": 0.75,
        "potential_probability": 0.75,
        "prior_probability": 0.5,
        "active_triggers": ["binary_sensor.motion1"],
        "sensor_probabilities": {
            "binary_sensor.motion1": {
                "weight": 0.8,
                "probability": 0.95,
                "weighted_probability": 0.76,
            }
        },
        "decay_status": {"global_decay": 0.0},
        "sensor_availability": {"binary_sensor.motion1": True},
        "is_occupied": True,
    }

    # Set up mock state for motion sensor
    hass.states.async_set(
        "binary_sensor.motion1", STATE_ON, {"friendly_name": "Motion Sensor 1"}
    )
    await hass.async_block_till_done()
    await asyncio.sleep(1)

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data",
        return_value=mock_data,
    ):
        coordinator = hass.data[DOMAIN][init_integration.entry_id]["coordinator"]
        await coordinator.async_refresh()
        await hass.async_block_till_done()
        await asyncio.sleep(1)

        prob_state = hass.states.get("sensor.test_area_occupancy_probability")
        assert prob_state is not None
        assert float(prob_state.state) == 75.0

        # Verify attributes
        assert "active_triggers" in prob_state.attributes
        assert "sensor_probabilities" in prob_state.attributes
        assert "threshold" in prob_state.attributes

        # Verify active triggers
        assert prob_state.attributes["active_triggers"] == ["Motion Sensor 1"]

        # Verify sensor probabilities
        sensor_probs = prob_state.attributes["sensor_probabilities"]
        assert len(sensor_probs) == 1
        prob_entry = next(iter(sensor_probs))
        assert "Motion Sensor 1" in prob_entry
        assert "W: 0.8" in prob_entry
        assert "P: 0.95" in prob_entry
        assert "WP: 0.76" in prob_entry
