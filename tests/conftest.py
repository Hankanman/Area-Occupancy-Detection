"""Test fixtures for room_occupancy integration."""
import os
import sys
from typing import Generator
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from custom_components.room_occupancy.const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_THRESHOLD,
)

# Make parent directory available to tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


pytest_plugins = "pytest_homeassistant_custom_component"  # pylint: disable=invalid-name

# Test data
TEST_CONFIG = {
    CONF_NAME: "Test Room",
    CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
    CONF_THRESHOLD: 0.5,
}


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations) -> Generator[None, None, None]:
    """Enable custom integrations for testing."""
    yield


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Create a mock config entry for testing."""
    return MockConfigEntry(
        domain=DOMAIN,
        data=TEST_CONFIG,
        title="Test Room",
        unique_id="uniqueid123",
    )


@pytest.fixture
async def init_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,  # pylint: disable=redefined-outer-name
) -> MockConfigEntry:
    """Set up the room occupancy integration for testing."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry
