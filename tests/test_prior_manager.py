"""Tests for the PriorManager class."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.area_occupancy.prior_manager import PriorManager
from custom_components.area_occupancy.types import PriorState


@pytest.fixture
def mock_storage():
    """Create a mock storage handler."""
    storage = MagicMock()
    storage.async_load_instance_prior_state = AsyncMock(return_value=None)
    storage.async_save_instance_prior_state = AsyncMock()
    return storage


@pytest.fixture
def prior_manager(hass, mock_storage, mock_probabilities, mock_state_manager):
    """Create a PriorManager instance for testing."""
    config = {
        "name": "Test Area",
        "motion_sensors": ["binary_sensor.motion1"],
    }

    mock_calculator = MagicMock()
    mock_calculator.calculate_prior = AsyncMock(return_value=None)

    return PriorManager(
        hass=hass,
        config=config,
        storage=mock_storage,
        probabilities=mock_probabilities,
        state_manager=mock_state_manager,
        prior_calculator=mock_calculator,
        config_entry_id="test_entry_id",
        disable_scheduling=True,  # Disable scheduling to prevent timer issues in tests
    )


@pytest.mark.asyncio
async def test_prior_manager_initialization(prior_manager):
    """Test that PriorManager initializes correctly."""
    assert prior_manager.hass is not None
    assert prior_manager.config is not None
    assert prior_manager.prior_state is not None
    assert isinstance(prior_manager.prior_state, PriorState)


@pytest.mark.asyncio
async def test_prior_manager_setup(prior_manager):
    """Test that PriorManager setup works correctly."""
    await prior_manager.async_setup()

    # Should have initialized prior state
    assert prior_manager.prior_state is not None
    # Scheduling is disabled in tests, so next_prior_update will be None
    assert prior_manager._next_prior_update is None


@pytest.mark.asyncio
async def test_prior_manager_load_stored_priors(prior_manager):
    """Test loading stored priors."""
    await prior_manager.load_stored_priors()

    # Should have called storage load method
    prior_manager.storage.async_load_instance_prior_state.assert_called_once()


@pytest.mark.asyncio
async def test_prior_manager_save_prior_state(prior_manager):
    """Test saving prior state."""
    await prior_manager.save_prior_state()

    # Should have called storage save method
    prior_manager.storage.async_save_instance_prior_state.assert_called_once()


@pytest.mark.asyncio
async def test_prior_manager_update_learned_priors(prior_manager):
    """Test updating learned priors."""
    # Setup prior state first
    await prior_manager.load_stored_priors()

    # Run update
    await prior_manager.update_learned_priors(history_period=1)

    # Should have attempted to get configured sensors and calculate priors
    assert prior_manager._last_prior_update is not None


@pytest.mark.asyncio
async def test_prior_manager_shutdown(prior_manager):
    """Test PriorManager shutdown."""
    # Setup first
    await prior_manager.async_setup()

    # Then shutdown
    await prior_manager.async_shutdown()

    # Should have cleared state
    assert prior_manager._prior_update_tracker is None
    assert prior_manager._next_prior_update is None


def test_prior_manager_properties(prior_manager):
    """Test PriorManager properties."""
    assert prior_manager.prior_update_interval is not None
    assert isinstance(prior_manager.prior_update_interval, timedelta)

    # Initially None until setup
    assert prior_manager.next_prior_update is None
    assert prior_manager.last_prior_update is None
