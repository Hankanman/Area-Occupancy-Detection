"""Tests for the Area Occupancy services."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.area_occupancy.const import (
    CONF_HISTORY_PERIOD,
    DEFAULT_HISTORY_PERIOD,
    DOMAIN,
)
from custom_components.area_occupancy.exceptions import CalculationError
from custom_components.area_occupancy.service import async_setup_services


@pytest.fixture
def mock_coordinator():
    """Mock coordinator fixture."""
    coordinator = MagicMock()
    coordinator.update_learned_priors = AsyncMock()
    coordinator.async_refresh = AsyncMock()
    # Configure config.get to return the default history period when called without a specific key
    # or when called with CONF_HISTORY_PERIOD
    coordinator.config = MagicMock()
    coordinator.config.get.side_effect = lambda key, default=None: (
        DEFAULT_HISTORY_PERIOD if key == CONF_HISTORY_PERIOD else default
    )
    return coordinator


async def test_setup_services(hass: HomeAssistant, mock_coordinator):  # pylint: disable=redefined-outer-name
    """Test setting up services."""
    # Setup mock data
    entry_id = "test_entry_id"
    hass.data[DOMAIN] = {entry_id: {"coordinator": mock_coordinator}}

    # Setup services
    await async_setup_services(hass)

    # Verify service was registered
    assert hass.services.has_service(DOMAIN, "update_priors")


async def test_update_priors_success(hass: HomeAssistant, mock_coordinator):  # pylint: disable=redefined-outer-name
    """Test successful update_priors service call."""
    # Setup mock data
    entry_id = "test_entry_id"
    hass.data[DOMAIN] = {entry_id: {"coordinator": mock_coordinator}}

    # Setup services
    await async_setup_services(hass)

    # Call service with default history period
    await hass.services.async_call(
        DOMAIN,
        "update_priors",
        {"entry_id": entry_id},
        blocking=True,
    )

    # Verify coordinator methods were called with default history period
    mock_coordinator.update_learned_priors.assert_called_once_with(
        DEFAULT_HISTORY_PERIOD
    )
    mock_coordinator.async_refresh.assert_called_once()


async def test_update_priors_with_custom_period(hass: HomeAssistant, mock_coordinator):  # pylint: disable=redefined-outer-name
    """Test update_priors service call with custom history period."""
    # Setup mock data
    entry_id = "test_entry_id"
    hass.data[DOMAIN] = {entry_id: {"coordinator": mock_coordinator}}

    # Setup services
    await async_setup_services(hass)

    # Call service with custom history period
    custom_period = 30
    await hass.services.async_call(
        DOMAIN,
        "update_priors",
        {
            "entry_id": entry_id,
            "history_period": custom_period,
        },
        blocking=True,
    )

    # Verify coordinator methods were called with custom period
    mock_coordinator.update_learned_priors.assert_called_once_with(custom_period)
    mock_coordinator.async_refresh.assert_called_once()


async def test_update_priors_invalid_entry_id(hass: HomeAssistant):  # pylint: disable=redefined-outer-name
    """Test update_priors service call with invalid entry_id."""
    # Setup services
    await async_setup_services(hass)

    # Setup empty domain data
    hass.data[DOMAIN] = {}

    # Call service with invalid entry_id
    with pytest.raises(HomeAssistantError) as exc_info:
        await hass.services.async_call(
            DOMAIN,
            "update_priors",
            {"entry_id": "invalid_entry_id"},
            blocking=True,
        )

    assert (
        "Failed to update priors: Integration or instance invalid_entry_id not found"
        in str(exc_info.value)
    )


async def test_update_priors_coordinator_error(hass: HomeAssistant, mock_coordinator):  # pylint: disable=redefined-outer-name
    """Test update_priors service call when coordinator update fails."""
    # Setup mock data
    entry_id = "test_entry_id"
    hass.data[DOMAIN] = {entry_id: {"coordinator": mock_coordinator}}

    # Setup services
    await async_setup_services(hass)

    # Make coordinator update raise an error
    mock_coordinator.update_learned_priors.side_effect = RuntimeError("Update failed")

    # Call service and verify error handling
    with pytest.raises(HomeAssistantError) as exc_info:
        await hass.services.async_call(
            DOMAIN,
            "update_priors",
            {"entry_id": entry_id},
            blocking=True,
        )

    assert "Failed to update priors" in str(exc_info.value)


async def test_update_priors_invalid_history_period(
    hass: HomeAssistant,
    mock_coordinator,  # pylint: disable=redefined-outer-name
):
    """Test update_priors service with invalid history period."""
    # Setup mock data
    entry_id = "test_entry_id"
    hass.data[DOMAIN] = {entry_id: {"coordinator": mock_coordinator}}

    # Setup services
    await async_setup_services(hass)

    # Test with history period too high
    with pytest.raises(vol.Invalid):
        await hass.services.async_call(
            DOMAIN,
            "update_priors",
            {
                "entry_id": entry_id,
                "history_period": 91,  # Above max allowed value
            },
            blocking=True,
        )

    # Test with history period too low
    with pytest.raises(vol.Invalid):
        await hass.services.async_call(
            DOMAIN,
            "update_priors",
            {
                "entry_id": entry_id,
                "history_period": 0,  # Below min allowed value
            },
            blocking=True,
        )

    # Verify coordinator methods were not called
    mock_coordinator.update_learned_priors.assert_not_called()
    mock_coordinator.async_refresh.assert_not_called()


async def test_update_priors_coordinator_state(
    hass: HomeAssistant,
    mock_coordinator,  # pylint: disable=redefined-outer-name
):
    """Test coordinator state updates after update_priors service call."""
    # Setup mock data
    entry_id = "test_entry_id"
    hass.data[DOMAIN] = {entry_id: {"coordinator": mock_coordinator}}

    # Setup services
    await async_setup_services(hass)

    # Mock successful prior update
    mock_coordinator.update_learned_priors.return_value = None
    mock_coordinator.async_refresh.return_value = None

    # Call service
    await hass.services.async_call(
        DOMAIN,
        "update_priors",
        {"entry_id": entry_id},
        blocking=True,
    )

    # Verify coordinator methods were called in correct order
    assert mock_coordinator.update_learned_priors.call_count == 1
    assert mock_coordinator.async_refresh.call_count == 1
    # If you want to check call order, use mock_calls or call_args_list


async def test_update_priors_coordinator_error_handling(
    hass: HomeAssistant,
    mock_coordinator,  # pylint: disable=redefined-outer-name
):
    """Test error handling during coordinator updates."""
    # Setup mock data
    entry_id = "test_entry_id"
    hass.data[DOMAIN] = {entry_id: {"coordinator": mock_coordinator}}

    # Setup services
    await async_setup_services(hass)

    # Test update_learned_priors error
    mock_coordinator.update_learned_priors.side_effect = CalculationError("Test error")

    with pytest.raises(HomeAssistantError) as exc_info:
        await hass.services.async_call(
            DOMAIN,
            "update_priors",
            {"entry_id": entry_id},
            blocking=True,
        )

    assert "Failed to update priors" in str(exc_info.value)
    mock_coordinator.async_refresh.assert_not_called()

    # Test refresh error
    mock_coordinator.update_learned_priors.side_effect = None
    mock_coordinator.async_refresh.side_effect = HomeAssistantError("Refresh error")

    with pytest.raises(HomeAssistantError) as exc_info:
        await hass.services.async_call(
            DOMAIN,
            "update_priors",
            {"entry_id": entry_id},
            blocking=True,
        )

    assert "Failed to update priors" in str(exc_info.value)
