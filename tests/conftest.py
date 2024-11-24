"""Common test fixtures for Room Occupancy Detection integration tests."""

from unittest.mock import patch
from typing import Generator
import pytest
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from custom_components.room_occupancy.const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_DEVICE_STATES,
    CONF_THRESHOLD,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_TYPE,
)


@pytest.fixture
def mock_setup_entry() -> Generator:
    """Prevent setup entry from running."""
    with patch("custom_components.room_occupancy.async_setup_entry", return_value=True):
        yield


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry with default test values."""
    return {
        CONF_NAME: "Test Room",
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_ILLUMINANCE_SENSORS: ["sensor.test_illuminance"],
        CONF_HUMIDITY_SENSORS: ["sensor.test_humidity"],
        CONF_TEMPERATURE_SENSORS: ["sensor.test_temperature"],
        CONF_DEVICE_STATES: ["media_player.test_tv"],
        CONF_THRESHOLD: 0.5,
        CONF_HISTORY_PERIOD: 7,
        CONF_DECAY_ENABLED: True,
        CONF_DECAY_WINDOW: 600,
        CONF_DECAY_TYPE: "linear",
    }


@pytest.fixture
async def mock_fully_setup_entry(hass: HomeAssistant, mock_config):
    """Create and setup a mock config entry with all test entities."""
    # Create mock entities
    hass.states.async_set("binary_sensor.test_motion", "off")
    hass.states.async_set("sensor.test_illuminance", "100")
    hass.states.async_set("sensor.test_humidity", "50")
    hass.states.async_set("sensor.test_temperature", "21")
    hass.states.async_set("media_player.test_tv", "off")

    entry = MockConfigEntry(domain=DOMAIN, data=mock_config, entry_id="test_entry_id")
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    return entry
