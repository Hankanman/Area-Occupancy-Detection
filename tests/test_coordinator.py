"""Tests for the Area Occupancy DataUpdateCoordinator."""

import logging
from datetime import timedelta
from unittest.mock import AsyncMock, patch

from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

# Import necessary components from the custom integration
from custom_components.area_occupancy.const import (
    CONF_MOTION_SENSORS,
    CONF_NAME,
    DOMAIN,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.types import (
    OccupancyCalculationResult,
    PriorData,
    PriorState,
    ProbabilityState,
)

# Mock config data and options from conftest
from .conftest import TEST_CONFIG  # Import test config and entities setup

_LOGGER = logging.getLogger(__name__)

# --- Fixtures ---

# Use the init_integration fixture from conftest.py which provides hass and sets up the component

# --- Test Cases ---


async def test_coordinator_initialization(
    hass: HomeAssistant, init_integration: MockConfigEntry
):
    """Test successful coordinator initialization after integration setup."""
    coordinator = hass.data[DOMAIN][init_integration.entry_id]["coordinator"]
    assert isinstance(coordinator, AreaOccupancyCoordinator)
    assert coordinator.hass is hass
    assert coordinator.config_entry is init_integration
    assert coordinator.name == TEST_CONFIG[CONF_NAME]
    assert coordinator.update_interval is None  # Explicitly None now
    assert coordinator.logger is not None
    assert isinstance(coordinator.data, ProbabilityState)  # Check initial state type
    assert isinstance(
        coordinator.prior_state, PriorState
    )  # Check initial prior state type


@patch(
    "custom_components.area_occupancy.coordinator.PriorCalculator.calculate_prior",
    new_callable=AsyncMock,
)
@patch(
    "custom_components.area_occupancy.coordinator.ProbabilityCalculator.calculate_occupancy_probability"
)
async def test_coordinator_first_refresh_success(
    mock_calc_prob,
    mock_calc_priors,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test the first successful data refresh using async_refresh."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Mock prior calculation result
    primary_sensor = coordinator.inputs.primary_sensor
    mock_calc_priors.return_value = {
        "prob_given_true": 0.85,
        "prob_given_false": 0.15,
        "prior": 0.45,
    }

    # Mock probability calculation result
    mock_prob_result = OccupancyCalculationResult(
        calculated_probability=0.75,
        prior_probability=0.5,
        sensor_probabilities={
            primary_sensor: {
                "probability": 0.8,
                "weight": TEST_CONFIG["weight_motion"],
                "weighted_probability": 0.75 * TEST_CONFIG["weight_motion"],
            }
        },
    )
    mock_calc_prob.return_value = mock_prob_result

    # Mock decay handler
    with patch(
        "custom_components.area_occupancy.coordinator.DecayHandler.calculate_decay",
        return_value=(mock_prob_result.calculated_probability, 1.0, False, None, None),
    ) as mock_decay:
        # Set initial state to ON
        hass.states.async_set(primary_sensor, STATE_ON)
        await hass.async_block_till_done()

        # Trigger update
        await coordinator.async_refresh()
        await hass.async_block_till_done()

    # Assertions
    assert coordinator.last_update_success is True
    assert isinstance(coordinator.data, ProbabilityState)

    # Verify probability calculation was called
    assert mock_calc_prob.called
    last_call_args, _ = mock_calc_prob.call_args_list[-1]
    assert isinstance(last_call_args[0], dict)
    assert primary_sensor in last_call_args[0]
    assert last_call_args[0][primary_sensor]["state"] == STATE_ON
    assert isinstance(last_call_args[1], PriorState)

    # Verify decay handler was called
    assert mock_decay.called
    assert len(mock_decay.call_args_list) > 0
    # Optionally, print/log the call args for debugging
    # print(f"Decay handler call args: {mock_decay.call_args_list}")

    # Check final state
    assert coordinator.data.probability == mock_prob_result.calculated_probability
    assert coordinator.data.prior_probability == mock_prob_result.prior_probability
    assert (
        coordinator.data.sensor_probabilities == mock_prob_result.sensor_probabilities
    )
    assert coordinator.data.is_occupied == (
        coordinator.data.probability >= coordinator.data.threshold
    )
    assert coordinator.data.decaying is False


async def test_coordinator_update_failure_state_get(
    hass: HomeAssistant, init_integration: MockConfigEntry
):
    """Test coordinator update failure when primary sensor state is missing."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Simulate the primary sensor being unavailable
    primary_sensor = coordinator.inputs.primary_sensor
    hass.states.async_remove(primary_sensor)
    await hass.async_block_till_done()

    # Re-initialize states within the coordinator to reflect the removal
    await coordinator.async_initialize_states(coordinator.get_configured_sensors())
    await hass.async_block_till_done()

    # Trigger update
    # _async_update_data itself doesn't raise UpdateFailed, it returns existing data
    # The coordinator framework might raise it, but we test the internal state
    await coordinator.async_refresh()  # This calls _async_update_data
    await hass.async_block_till_done()

    # Expect coordinator.available to be False and probability low
    # Note: async_refresh won't raise UpdateFailed directly in this case, check state instead
    assert coordinator.available is False
    assert coordinator.data.probability == MIN_PROBABILITY  # Should reset or stay low
    assert (
        coordinator.last_update_success is True
    )  # Update itself might run, but result is low prob


@patch(
    "custom_components.area_occupancy.coordinator.PriorCalculator.calculate_prior",
    new_callable=AsyncMock,
)
@patch(
    "custom_components.area_occupancy.coordinator.ProbabilityCalculator.calculate_occupancy_probability"
)
async def test_coordinator_update_failure_prob_calc(
    mock_calc_prob,
    mock_calc_priors,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test coordinator update failure during composite probability calculation."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Set initial state
    hass.states.async_set(TEST_CONFIG[CONF_MOTION_SENSORS][0], STATE_ON)
    await hass.async_block_till_done()
    # Ensure coordinator sees initial state
    await coordinator.async_initialize_states(coordinator.get_configured_sensors())
    initial_prob = coordinator.data.probability

    # Make probability calculation raise an error
    mock_calc_prob.side_effect = ValueError("Prob calculation failed")

    # Trigger update
    await coordinator.async_refresh()
    await hass.async_block_till_done()

    # Update should fail gracefully, state shouldn't change drastically or become None
    assert coordinator.last_update_success is True  # Refresh itself doesn't fail
    assert coordinator.data.probability == initial_prob  # Probability remains unchanged


# Test for prior update failures can be added if needed,
# but it runs independently now.


# Add test for prior state management
async def test_coordinator_prior_state_management(
    hass: HomeAssistant, init_integration: MockConfigEntry
):
    """Test prior state management in coordinator."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Test initial prior state
    assert coordinator.prior_state is not None
    assert coordinator.prior_state.overall_prior >= MIN_PROBABILITY

    # Test prior state update
    mock_prior_data = PriorData(
        prior=0.75,
        prob_given_true=0.85,
        prob_given_false=0.15,
        last_updated=dt_util.utcnow().isoformat(),
    )

    # Update a type prior
    coordinator.prior_state.update_type_prior(
        "motion",
        prior_value=0.75,
        timestamp=dt_util.utcnow().isoformat(),
        prob_given_true=0.85,
        prob_given_false=0.15,
    )

    # Verify type prior was updated
    assert coordinator.prior_state.motion_prior == 0.75
    assert coordinator.prior_state.type_priors["motion"].prior == 0.75


# Add test for decay handler integration
async def test_coordinator_decay_handler(
    hass: HomeAssistant, init_integration: MockConfigEntry
):
    """Test decay handler integration in coordinator."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Set initial high probability
    coordinator.data.probability = 0.9
    coordinator.data.previous_probability = 0.9
    coordinator.data.decaying = False
    coordinator.data.decay_start_time = None
    coordinator.data.decay_start_probability = None

    # Trigger probability decrease
    new_prob = 0.5
    with patch.object(
        coordinator.calculator, "calculate_occupancy_probability"
    ) as mock_calc:
        mock_calc.return_value = OccupancyCalculationResult(
            calculated_probability=new_prob,
            prior_probability=0.5,
            sensor_probabilities={},
        )

        await coordinator.async_refresh()
        await hass.async_block_till_done()

    # Verify decay started
    assert coordinator.data.decaying is True
    assert coordinator.data.decay_start_time is not None
    assert coordinator.data.decay_start_probability == 0.9  # Previous probability


# Add test for historical analysis
async def test_coordinator_historical_analysis(
    hass: HomeAssistant, init_integration: MockConfigEntry
):
    """Test historical analysis functionality."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Mock history data
    mock_history_data = {
        coordinator.inputs.primary_sensor: [
            State(
                coordinator.inputs.primary_sensor,
                STATE_ON,
                last_changed=dt_util.utcnow() - timedelta(hours=1),
            ),
            State(
                coordinator.inputs.primary_sensor,
                STATE_OFF,
                last_changed=dt_util.utcnow(),
            ),
        ]
    }

    with patch(
        "homeassistant.components.recorder.history.get_significant_states",
        return_value=mock_history_data,
    ):
        await coordinator.update_learned_priors()
        await hass.async_block_till_done()

    # Verify priors were updated
    assert coordinator.prior_state.entity_priors
    assert coordinator.prior_state.type_priors
    assert coordinator.last_prior_update is not None

    # Verify priors were updated
    assert coordinator.prior_state.entity_priors
    assert coordinator.prior_state.type_priors
    assert coordinator.last_prior_update is not None
