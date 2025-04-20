"""Tests for the Area Occupancy DataUpdateCoordinator."""

import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import (
    ConfigEntryError,
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

# Import necessary components from the custom integration
from custom_components.area_occupancy.const import (  # noqa: TID252
    CONF_DECAY_ENABLED,
    CONF_HISTORY_PERIOD,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_THRESHOLD,
    DOMAIN,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator  # noqa: TID252
from custom_components.area_occupancy.exceptions import (  # noqa: TID252
    CalculationError,
    StateError,
    StorageError,
)
from custom_components.area_occupancy.types import EntityType  # noqa: TID252
from custom_components.area_occupancy.types import (
    OccupancyCalculationResult,
    PriorData,
    PriorState,
    ProbabilityState,
)

# Mock config data and options from conftest
from .conftest import TEST_CONFIG  # noqa: TID251

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
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test decay handling logic."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Test decay disabled by config
    coordinator.config[CONF_DECAY_ENABLED] = False
    coordinator._start_decay_updates()
    assert coordinator._decay_unsub is None

    # Test decay update with error
    coordinator.config[CONF_DECAY_ENABLED] = True
    coordinator.data.decaying = True

    # Mock the async_track_time_interval to capture the callback
    with patch(
        "custom_components.area_occupancy.coordinator.async_track_time_interval"
    ) as mock_track:
        # Set up a mock callback that will be called by the decay handler
        async def mock_callback(*_):
            raise Exception("Error")

        mock_track.return_value = lambda: None  # Simple cleanup function

        # Start decay updates and wait for setup
        coordinator._start_decay_updates()
        await hass.async_block_till_done()

        # Verify that async_track_time_interval was called
        assert mock_track.called
        assert coordinator._decay_unsub is not None

        # Get the callback that was registered and test error handling
        callback = mock_track.call_args[0][1]
        await callback(datetime.now())

        # Verify that decay is still enabled after error
        assert coordinator.data.decaying is True

        # Test cleanup
        coordinator._stop_decay_updates()
        assert coordinator._decay_unsub is None


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


@patch(
    "custom_components.area_occupancy.coordinator.PriorState.initialize_from_defaults"
)
async def test_coordinator_async_update_options(
    mock_initialize_priors,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test coordinator options update method."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Store initial values
    initial_threshold = coordinator.data.threshold
    initial_config = coordinator.config.copy()

    # Update options
    new_options = {CONF_THRESHOLD: 75}  # Different threshold

    # Update config entry options
    hass.config_entries.async_update_entry(
        init_integration,
        options=new_options,
    )
    await hass.async_block_till_done()

    # Call update_options directly to ensure coverage
    await coordinator.async_update_options()
    await hass.async_block_till_done()

    # Verify the coordinator was updated with new options
    assert coordinator.config[CONF_THRESHOLD] == 75
    assert coordinator.data.threshold == 75 / 100.0
    assert coordinator.config != initial_config
    assert coordinator.data.threshold != initial_threshold

    # Test error handling
    with (
        patch(
            "custom_components.area_occupancy.types.SensorInputs.from_config",
            side_effect=ValueError("Invalid config"),
        ),
        pytest.raises(ConfigEntryError),
    ):
        await coordinator.async_update_options()

    # Test HomeAssistantError handling
    with (
        patch(
            "custom_components.area_occupancy.types.SensorInputs.from_config",
            side_effect=HomeAssistantError("HA Error"),
        ),
        pytest.raises(ConfigEntryNotReady),
    ):
        await coordinator.async_update_options()


async def test_coordinator_async_update_threshold(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test threshold update method."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Test successful threshold update
    await coordinator.async_update_threshold(75.0)
    await hass.async_block_till_done()

    # Check if the config entry was updated
    assert init_integration.options[CONF_THRESHOLD] == 75.0

    # Test error handling
    with (
        patch(
            "homeassistant.config_entries.ConfigEntries.async_update_entry",
            side_effect=ValueError("Invalid threshold"),
        ),
        pytest.raises(ServiceValidationError),
    ):
        await coordinator.async_update_threshold(150.0)  # Invalid value

    # Test generic exception handling
    with (
        patch(
            "homeassistant.config_entries.ConfigEntries.async_update_entry",
            side_effect=Exception("Unknown error"),
        ),
        pytest.raises(HomeAssistantError),
    ):
        await coordinator.async_update_threshold(60.0)


@patch(
    "custom_components.area_occupancy.storage.AreaOccupancyStore.async_load_instance_prior_state"
)
async def test_coordinator_async_load_stored_data_success(
    mock_load_instance,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test successful loading of stored data."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Create mock prior state data
    mock_prior_state = PriorState()
    mock_prior_state.overall_prior = 0.75

    # Mock return value
    mock_return = MagicMock()
    mock_return.prior_state = mock_prior_state
    mock_return.last_updated = dt_util.utcnow().isoformat()
    mock_load_instance.return_value = mock_return

    # Call the load method directly
    await coordinator.async_load_stored_data()
    await hass.async_block_till_done()

    # Verify the data was loaded
    assert coordinator.prior_state.overall_prior == 0.75
    assert coordinator._last_prior_update == mock_return.last_updated  # noqa: SLF001


@patch(
    "custom_components.area_occupancy.storage.AreaOccupancyStore.async_load_instance_prior_state"
)
async def test_coordinator_async_load_stored_data_no_data(
    mock_load_instance,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test loading of stored data when none exists."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Mock return value - no data found
    mock_load_instance.return_value = None

    # Call the load method directly
    await coordinator.async_load_stored_data()
    await hass.async_block_till_done()

    # Verify default values are set
    assert coordinator._last_prior_update is None  # noqa: SLF001
    assert coordinator.prior_state is not None
    assert isinstance(coordinator.prior_state, PriorState)


@patch(
    "custom_components.area_occupancy.storage.AreaOccupancyStore.async_load_instance_prior_state"
)
async def test_coordinator_async_load_stored_data_error(
    mock_load_instance,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test error handling when loading stored data."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Mock exception
    mock_load_instance.side_effect = StorageError("Failed to load data")

    # Test that the function raises ConfigEntryNotReady
    with pytest.raises(ConfigEntryNotReady):
        await coordinator.async_load_stored_data()
    await hass.async_block_till_done()

    # Verify default values are set despite error
    assert coordinator._last_prior_update is None  # noqa: SLF001
    assert coordinator.prior_state is not None


@patch(
    "custom_components.area_occupancy.calculate_prior.PriorCalculator.calculate_prior"
)
async def test_coordinator_update_learned_priors(
    mock_calculate_prior,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test learned prior updates."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Setup mock return value for calculate_prior
    mock_prior_data = MagicMock()
    mock_prior_data.prob_given_true = 0.85
    mock_prior_data.prob_given_false = 0.15
    mock_prior_data.prior = 0.65
    mock_calculate_prior.return_value = mock_prior_data

    # Call update_learned_priors
    await coordinator.update_learned_priors(history_period=7)
    await hass.async_block_till_done()

    # Verify the calculated priors were stored
    assert mock_calculate_prior.called
    assert coordinator._last_prior_update is not None  # noqa: SLF001


@patch(
    "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._update_type_priors_from_entities"
)
@patch(
    "custom_components.area_occupancy.calculate_prior.PriorCalculator.calculate_prior"
)
async def test_coordinator_update_learned_priors_with_error(
    mock_calculate_prior,
    mock_update_type_priors,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test learned prior updates with calculation errors."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Setup mock to raise exception
    mock_calculate_prior.side_effect = CalculationError("Prior calculation failed")

    # Call update_learned_priors - should not raise exception
    await coordinator.update_learned_priors(history_period=7)
    await hass.async_block_till_done()

    # Verify method completes despite errors
    assert mock_calculate_prior.called
    # The update should NOT continue to type priors if entity priors fail completely
    assert not mock_update_type_priors.called


@patch(
    "custom_components.area_occupancy.storage.AreaOccupancyStore.async_save_instance_prior_state"
)
async def test_coordinator_save_prior_state_data(
    mock_save_state,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test saving prior state data."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Call internal save method
    await coordinator._async_save_prior_state_data()  # noqa: SLF001
    await hass.async_block_till_done()

    # Verify save was called with correct parameters
    assert mock_save_state.called
    assert mock_save_state.call_args[0][0] == init_integration.entry_id
    # The save function takes entry_id, name, and prior_state
    assert mock_save_state.call_args[0][1] == coordinator.name  # Check name is correct
    assert (
        mock_save_state.call_args[0][2] == coordinator.prior_state
    )  # Check prior state is correct


@patch("custom_components.area_occupancy.coordinator.async_track_time_interval")
async def test_coordinator_start_decay_updates(
    mock_track_interval,
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test starting decay updates."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Test the start method
    coordinator._start_decay_updates()  # noqa: SLF001

    # Verify tracking was set up
    assert mock_track_interval.called
    assert coordinator._decay_unsub is not None  # noqa: SLF001


async def test_coordinator_stop_decay_updates(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test stopping decay updates."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Setup a mock unsubscribe function
    mock_unsub = MagicMock()
    coordinator._decay_unsub = mock_unsub  # noqa: SLF001

    # Call stop method
    coordinator._stop_decay_updates()  # noqa: SLF001

    # Verify unsubscribe was called
    assert mock_unsub.called
    assert coordinator._decay_unsub is None  # noqa: SLF001
    assert mock_unsub.called
    assert coordinator._decay_unsub is None  # noqa: SLF001
    assert mock_unsub.called
    assert coordinator._decay_unsub is None  # noqa: SLF001


async def test_coordinator_property_getters(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test coordinator property getters."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Test property getters
    assert coordinator.probability == coordinator.data.probability
    assert coordinator.is_occupied == coordinator.data.is_occupied
    assert coordinator.threshold == coordinator.data.threshold
    assert coordinator.prior_update_interval == coordinator._prior_update_interval
    assert coordinator.next_prior_update == coordinator._next_prior_update
    assert coordinator.last_prior_update == coordinator._last_prior_update

    # Test availability when primary sensor is missing
    coordinator.data = ProbabilityState()
    coordinator.data.current_states = {
        "sensor.test": {
            "state": None,
            "last_changed": dt_util.utcnow().isoformat(),
            "availability": False,
        }
    }
    assert coordinator.available is False

    # Test availability with invalid primary sensor state
    primary_sensor = coordinator.inputs.primary_sensor
    coordinator.data.current_states = {
        primary_sensor: {
            "state": "unknown",
            "last_changed": dt_util.utcnow().isoformat(),
            "availability": False,
        }
    }
    assert coordinator.available is False


async def test_coordinator_setup_error_handling(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test error handling during coordinator setup."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Test storage error handling
    with (
        patch(
            "custom_components.area_occupancy.storage.AreaOccupancyStore.async_load_instance_prior_state",
            side_effect=StorageError("Storage error"),
        ),
        pytest.raises(ConfigEntryNotReady),
    ):
        await coordinator.async_setup()

    # Test state error handling
    with (
        patch.object(
            coordinator,
            "async_initialize_states",
            side_effect=StateError("State error"),
        ),
        pytest.raises(ConfigEntryNotReady),
    ):
        await coordinator.async_setup()

    # Test calculation error handling
    with patch.object(
        coordinator,
        "update_learned_priors",
        side_effect=CalculationError("Calculation error"),
    ):
        await coordinator.async_setup()
        # Should continue despite calculation error
        assert coordinator._last_prior_update is not None


async def test_coordinator_prior_update_error_paths(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test error paths in prior update logic."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Test invalid history period
    coordinator.config[CONF_HISTORY_PERIOD] = -1
    await coordinator.update_learned_priors()
    assert coordinator._last_prior_update is not None

    # Test no sensors configured
    with patch.object(coordinator, "get_configured_sensors", return_value=[]):
        await coordinator.update_learned_priors()
        assert coordinator._last_prior_update is not None

    # Test type prior update error
    with patch.object(
        coordinator,
        "_update_type_priors_from_entities",
        side_effect=Exception("Type prior error"),
    ):
        await coordinator.update_learned_priors()
        # Should continue despite error
        assert coordinator._last_prior_update is not None


async def test_coordinator_state_tracking_error_paths(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test error handling in state tracking."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Get the actual listener function - need to wait for setup to complete
    await hass.async_block_till_done()
    assert coordinator._remove_state_listener is not None

    # Create a mock state change event
    event = MagicMock()
    event.data = {
        "entity_id": "binary_sensor.motion1",
        "old_state": State("binary_sensor.motion1", "off"),
        "new_state": State("binary_sensor.motion1", "on"),
    }

    # Test with valid event data
    await coordinator._async_update_data()

    # Test with invalid event data
    event.data = {}
    await coordinator._async_update_data()

    # Store initial state for comparison
    initial_prob = coordinator.data.probability
    initial_threshold = coordinator.data.threshold
    initial_prior = coordinator.data.prior_probability
    initial_is_occupied = coordinator.data.is_occupied
    initial_decaying = coordinator.data.decaying

    # Test with exception during state update
    with patch.object(
        coordinator, "_async_update_data", side_effect=Exception("Test error")
    ) as mock_update:
        # The coordinator should handle the exception gracefully
        await coordinator.async_refresh()
        assert mock_update.called
        # Verify coordinator remains in a valid state
        assert coordinator.data is not None
        # last_update_success should be False due to the error
        assert coordinator.last_update_success is False
        # Data should remain unchanged after error
        assert coordinator.data.probability == initial_prob
        assert coordinator.data.threshold == initial_threshold
        assert coordinator.data.prior_probability == initial_prior
        assert coordinator.data.is_occupied == initial_is_occupied
        assert coordinator.data.decaying == initial_decaying

    # Test cleanup
    # Store the unsubscribe function
    unsub = coordinator._remove_state_listener
    assert callable(unsub)  # Verify it's a callable
    # Call the unsubscribe function
    unsub()
    # After calling the unsubscribe function, verify the listener is removed
    listeners = hass.bus.async_listeners().get("state_changed", [])
    assert isinstance(listeners, list)  # Ensure we have a list
    assert not listeners  # Verify the list is empty


async def test_coordinator_cleanup(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
):
    """Test coordinator cleanup and shutdown."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][
        init_integration.entry_id
    ]["coordinator"]

    # Setup decay updates and prior tracker
    coordinator._start_decay_updates()
    await coordinator._schedule_next_prior_update()

    # Test shutdown
    await coordinator.async_shutdown()

    # Verify cleanup
    assert coordinator._decay_unsub is None
    assert coordinator._prior_update_tracker is None
    assert coordinator._remove_state_listener is None
    assert coordinator.data is not None  # Should be empty ProbabilityState
    assert coordinator.prior_state is not None  # Should be empty PriorState
    assert not coordinator._entity_ids
    assert coordinator.prior_state is not None  # Should be empty PriorState
    assert not coordinator._entity_ids
    assert coordinator.prior_state is not None  # Should be empty PriorState
    assert not coordinator._entity_ids
    assert coordinator.prior_state is not None  # Should be empty PriorState
    assert not coordinator._entity_ids
    assert coordinator.prior_state is not None  # Should be empty PriorState
    assert not coordinator._entity_ids
    assert coordinator.prior_state is not None  # Should be empty PriorState
    assert not coordinator._entity_ids
    assert coordinator.prior_state is not None  # Should be empty PriorState
    assert not coordinator._entity_ids


@pytest.mark.asyncio
async def test_update_type_priors_from_entities_direct(
    monkeypatch, hass, init_integration
):
    """Directly test _update_type_priors_from_entities for correct type prior aggregation and edge cases."""
    coordinator = hass.data[DOMAIN][init_integration.entry_id]["coordinator"]

    # Patch probabilities.get_entity_type to return types based on entity_id
    entity_type_map = {
        "sensor.motion1": EntityType.MOTION,
        "sensor.media1": EntityType.MEDIA,
        "sensor.bad": EntityType.MOTION,
    }
    monkeypatch.setattr(
        coordinator.probabilities,
        "get_entity_type",
        lambda eid: entity_type_map.get(eid),  # pylint: disable=unnecessary-lambda
    )

    # Case 1: All valid entity priors for two types
    coordinator.prior_state.entity_priors = {
        "sensor.motion1": PriorData(
            prior=0.6, prob_given_true=0.8, prob_given_false=0.2, last_updated="now"
        ),
        "sensor.media1": PriorData(
            prior=0.4, prob_given_true=0.7, prob_given_false=0.3, last_updated="now"
        ),
    }
    await coordinator._update_type_priors_from_entities()
    # Should update both type_priors and simple prior attributes
    assert "motion" in coordinator.prior_state.type_priors
    assert "media" in coordinator.prior_state.type_priors
    assert abs(coordinator.prior_state.type_priors["motion"].prior - 0.6) < 1e-6
    assert abs(coordinator.prior_state.type_priors["media"].prior - 0.4) < 1e-6
    assert (
        abs(coordinator.prior_state.type_priors["motion"].prob_given_true - 0.8) < 1e-6
    )
    assert (
        abs(coordinator.prior_state.type_priors["media"].prob_given_false - 0.3) < 1e-6
    )
    assert abs(coordinator.prior_state.motion_prior - 0.6) < 1e-6
    assert abs(coordinator.prior_state.media_prior - 0.4) < 1e-6

    # Case 2: One entity prior missing prob_given_true/false (should be skipped for aggregation)
    coordinator.prior_state.entity_priors = {
        "sensor.motion1": PriorData(
            prior=0.5, prob_given_true=None, prob_given_false=None, last_updated="now"
        ),
        "sensor.media1": PriorData(
            prior=0.7, prob_given_true=0.9, prob_given_false=0.1, last_updated="now"
        ),
    }
    await coordinator._update_type_priors_from_entities()
    # Only media type should be updated
    assert "media" in coordinator.prior_state.type_priors
    assert abs(coordinator.prior_state.type_priors["media"].prior - 0.7) < 1e-6
    assert abs(coordinator.prior_state.media_prior - 0.7) < 1e-6
    # Motion type should not be updated (remains from previous or is overwritten to default)
    # If no valid priors, the type prior is not updated in this method

    # Case 3: All entity priors missing required fields (should not update any type priors)
    coordinator.prior_state.entity_priors = {
        "sensor.motion1": PriorData(
            prior=0.5, prob_given_true=None, prob_given_false=None, last_updated="now"
        ),
        "sensor.bad": PriorData(
            prior=0.2, prob_given_true=None, prob_given_false=None, last_updated="now"
        ),
    }
    # Save current type_priors for comparison
    prev_type_priors = dict(coordinator.prior_state.type_priors)
    await coordinator._update_type_priors_from_entities()
    # No new type priors should be added or changed
    assert coordinator.prior_state.type_priors == prev_type_priors

    # Case 4: entity_priors is empty (should not fail)
    coordinator.prior_state.entity_priors = {}
    await coordinator._update_type_priors_from_entities()
    # No type priors should be present or changed
    assert isinstance(coordinator.prior_state.type_priors, dict)
    # No type priors should be present or changed
    assert isinstance(coordinator.prior_state.type_priors, dict)
    # No type priors should be present or changed
    assert isinstance(coordinator.prior_state.type_priors, dict)
    # No type priors should be present or changed
    assert isinstance(coordinator.prior_state.type_priors, dict)
