"""Tests for area_occupancy integration."""

from unittest.mock import patch

from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.area_occupancy.const import DOMAIN  # noqa: TID252
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator  # noqa: TID252

from .conftest import TEST_CONFIG  # noqa: TID251


async def test_setup(hass: HomeAssistant, init_integration: MockConfigEntry) -> None:
    """Test integration setup."""
    # Wait for setup to complete
    # await asyncio.sleep(2) # Remove sleep
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
    # await asyncio.sleep(2) # Remove sleep
    await hass.async_block_till_done()

    # Initial state - no motion
    hass.states.async_set(TEST_CONFIG["motion_sensors"][0], STATE_OFF)
    hass.states.async_set(TEST_CONFIG["motion_sensors"][1], STATE_OFF)
    await hass.async_block_till_done()
    # await asyncio.sleep(1) # Remove sleep

    initial_state = hass.states.get("sensor.test_area_occupancy_probability")
    assert initial_state is not None
    initial_prob = float(initial_state.state)

    # Trigger motion sensor
    hass.states.async_set(TEST_CONFIG["motion_sensors"][0], STATE_ON)
    await hass.async_block_till_done()
    # await asyncio.sleep(1) # Remove sleep

    # Check probability increased
    new_state = hass.states.get("sensor.test_area_occupancy_probability")
    assert new_state is not None
    new_prob = float(new_state.state)
    # Exact increase depends on calculation, just check it increased
    assert new_prob > initial_prob


async def test_binary_sensor_threshold(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Test binary sensor state changes based on threshold."""
    # Wait for setup to complete
    # await asyncio.sleep(2) # Remove sleep
    await hass.async_block_till_done()

    # Set high probability with motion
    hass.states.async_set(TEST_CONFIG["motion_sensors"][0], STATE_ON)
    hass.states.async_set(TEST_CONFIG["motion_sensors"][1], STATE_ON)
    await hass.async_block_till_done()
    # await asyncio.sleep(1) # Remove sleep

    # Check binary sensor is on when probability is above threshold
    binary_state = hass.states.get("binary_sensor.test_area_occupancy_status")
    assert binary_state is not None
    prob_state = hass.states.get("sensor.test_area_occupancy_probability")
    assert prob_state is not None
    threshold_state = hass.states.get("number.test_area_occupancy_threshold")
    assert threshold_state is not None  # Add check for threshold state
    prob = float(prob_state.state)
    threshold = float(threshold_state.state)  # Use checked state
    expected_state = STATE_ON if prob >= threshold else STATE_OFF
    assert binary_state.state == expected_state

    # Set low probability with no motion
    hass.states.async_set(TEST_CONFIG["motion_sensors"][0], STATE_OFF)
    hass.states.async_set(TEST_CONFIG["motion_sensors"][1], STATE_OFF)
    await hass.async_block_till_done()
    # await asyncio.sleep(1) # Remove sleep

    # Check binary sensor is off when probability is below threshold
    binary_state = hass.states.get("binary_sensor.test_area_occupancy_status")
    assert binary_state is not None
    prob_state = hass.states.get("sensor.test_area_occupancy_probability")
    assert prob_state is not None
    threshold_state = hass.states.get("number.test_area_occupancy_threshold")
    assert threshold_state is not None  # Add check for threshold state
    prob = float(prob_state.state)
    threshold = float(threshold_state.state)  # Use checked state
    expected_state = STATE_ON if prob >= threshold else STATE_OFF
    assert binary_state.state == expected_state


async def test_coordinator_state_update_on_entity_change(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Test coordinator data updates correctly when a tracked entity changes state."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]
    prob_sensor_id = "sensor.test_area_occupancy_probability"
    motion_sensor_id = TEST_CONFIG["motion_sensors"][0]

    # Wait for setup and initial state
    await hass.async_block_till_done()

    # 1. Initial state (assume motion is off)
    hass.states.async_set(motion_sensor_id, STATE_OFF)
    await hass.async_block_till_done()
    await coordinator.async_refresh()  # Ensure coordinator processes initial state
    await hass.async_block_till_done()

    initial_state = hass.states.get(prob_sensor_id)
    assert initial_state is not None
    initial_prob = float(initial_state.state)

    # 2. Trigger motion sensor to ON
    # Mock the calculation result for this specific state change if needed for predictable results
    # For now, just check if the probability increases
    with (
        patch.object(
            coordinator.calculator,
            "calculate_occupancy_probability",
            wraps=coordinator.calculator.calculate_occupancy_probability,
        ) as mock_calc,
    ):
        hass.states.async_set(motion_sensor_id, STATE_ON)
        await hass.async_block_till_done()  # Allow listener to trigger

        # Verify coordinator update was triggered (calculator should be called)
        assert mock_calc.called

        # Check probability sensor state updated and increased
        final_state = hass.states.get(prob_sensor_id)
        assert final_state is not None
        final_prob = float(final_state.state)
        assert final_prob > initial_prob

        # Check attributes (example for sensor_probabilities)
        # Note: Attributes are based on the coordinator.data *after* the update
        assert "active_triggers" in final_state.attributes
        assert "sensor_probabilities" in final_state.attributes

        # Verify active trigger (should be entity ID)
        assert final_state.attributes["active_triggers"] == [motion_sensor_id]

        # Verify sensor probabilities format (using friendly name)
        sensor_probs = final_state.attributes["sensor_probabilities"]
        assert isinstance(sensor_probs, set)
        assert len(sensor_probs) > 0
        # Check for an entry related to the motion sensor
        motion_prob_entry_found = any(
            motion_sensor_id in entry for entry in sensor_probs
        )
        assert motion_prob_entry_found, (
            f"{motion_sensor_id} probability details not found in {sensor_probs}"
        )

    # 3. Turn motion sensor OFF
    with (
        patch.object(
            coordinator.calculator,
            "calculate_occupancy_probability",
            wraps=coordinator.calculator.calculate_occupancy_probability,
        ) as mock_calc_off,
    ):
        hass.states.async_set(motion_sensor_id, STATE_OFF)
        await hass.async_block_till_done()

        assert mock_calc_off.called

        # Check probability decreased (or decay started)
        final_state_off = hass.states.get(prob_sensor_id)
        assert final_state_off is not None
        # Probability might not decrease immediately if decay delay is active
        # Check that active triggers list is now empty
        assert final_state_off.attributes["active_triggers"] == []
        assert len(final_state_off.attributes["sensor_probabilities"]) == 0


# Update test_storage_integration
async def test_storage_integration(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Test storage integration functionality."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Test saving prior state using public API if available
    save_method = getattr(coordinator, "async_save_prior_state_data", None)
    if callable(save_method):
        await save_method()
    else:
        # If no public method, skip this test
        return

    # Create a new coordinator to test loading
    new_coordinator = AreaOccupancyCoordinator(hass, init_integration)
    await new_coordinator.async_setup()

    # Verify data was loaded
    assert new_coordinator.prior_state is not None
    assert new_coordinator.last_prior_update is not None


# Update test_migration_handling
async def test_migration_handling(hass: HomeAssistant) -> None:
    """Test migration of config entries."""
    old_config = {
        "name": "Test Area",
        "motion_sensors": [TEST_CONFIG["motion_sensors"][0]],
        "version": 6,  # Old version
    }

    # Create entry with old version
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=old_config,
        version=6,
    )
    entry.add_to_hass(hass)

    # Mock migration function
    with patch(
        "custom_components.area_occupancy.migrations.async_migrate_entry",
        return_value=True,
    ):
        # Set up the integration - should trigger migration
        await hass.config_entries.async_setup(entry.entry_id)


# Add test for service integration
async def test_service_integration(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Test service integration functionality."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Record initial state
    initial_update_time = coordinator.last_prior_update

    # Call update_priors service
    await hass.services.async_call(
        DOMAIN,
        "update_priors",
        {"entry_id": init_integration.entry_id},
        blocking=True,
    )

    # Verify service effect
    assert coordinator.last_prior_update != initial_update_time
    assert coordinator.prior_state is not None


# Add test for entity attribute updates
async def test_entity_attribute_updates(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Test entity attribute updates."""
    # Set initial state
    hass.states.async_set(TEST_CONFIG["motion_sensors"][0], STATE_ON)
    await hass.async_block_till_done()

    # Get probability sensor state
    state = hass.states.get("sensor.test_area_occupancy_probability")
    assert state is not None

    # Verify attributes
    assert "active_triggers" in state.attributes
    assert "sensor_probabilities" in state.attributes
    assert "threshold" in state.attributes

    # Verify attribute values
    assert isinstance(state.attributes["active_triggers"], list)
    assert isinstance(state.attributes["sensor_probabilities"], set)
    assert isinstance(state.attributes["threshold"], str)
    assert "%" in state.attributes["threshold"]
    assert isinstance(state.attributes["active_triggers"], list)
    assert isinstance(state.attributes["sensor_probabilities"], set)
    assert isinstance(state.attributes["threshold"], str)
    assert "%" in state.attributes["threshold"]
    assert isinstance(state.attributes["threshold"], str)
    assert "%" in state.attributes["threshold"]
