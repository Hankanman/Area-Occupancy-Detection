"""Test fixtures for area_occupancy integration."""

import os
import sys
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from homeassistant.const import CONF_NAME, STATE_OFF
from homeassistant.core import HomeAssistant
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.recorder import DOMAIN as RECORDER_DOMAIN, DATA_INSTANCE
from custom_components.area_occupancy.const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_THRESHOLD,
    CONF_WEIGHT_MOTION,
    DEFAULT_WEIGHT_MOTION,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_ENABLED,
)
from homeassistant.helpers import entity_registry as er

# Make parent directory available to tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


pytest_plugins = "pytest_homeassistant_custom_component"  # pylint: disable=invalid-name

# Test data
TEST_CONFIG = {
    CONF_NAME: "Test Area",
    CONF_THRESHOLD: 50,
    "motion": {
        CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
        CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
    },
    "doors": {},
    "windows": {},
    "lights": {},
    "media": {},
    "appliances": {},
    "environmental": {},
    "parameters": {
        CONF_HISTORY_PERIOD: 7,
        CONF_DECAY_ENABLED: True,
    },
}


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations,
) -> Generator[None, None, None]:
    """Enable custom integrations for testing."""
    yield


@pytest.fixture
def mock_recorder(hass: HomeAssistant):
    """Mock recorder component."""
    # Create a mock recorder instance
    mock_instance = MagicMock()

    # Create a Future for async_db_ready
    db_ready_future = asyncio.Future()
    db_ready_future.set_result(True)
    mock_instance.async_db_ready = db_ready_future

    # Mock core recorder methods
    mock_instance.async_initialize = AsyncMock(return_value=True)
    mock_instance.async_start = AsyncMock(return_value=True)
    mock_instance.async_stop = AsyncMock(return_value=True)
    mock_instance.async_shutdown = AsyncMock(return_value=True)

    # Mock database connection
    mock_instance.db_connected = True
    mock_instance.async_add_executor_job = AsyncMock(return_value={})

    # Create a mock recorder class
    mock_recorder_class = MagicMock()
    mock_recorder_class.return_value = mock_instance

    # Set up recorder in hass.data
    hass.data[RECORDER_DOMAIN] = mock_instance
    hass.data[DATA_INSTANCE] = mock_instance

    # Mock get_instance
    def get_instance(hass):
        return mock_instance

    with patch(
        "homeassistant.components.recorder.Recorder",
        mock_recorder_class,
    ), patch(
        "homeassistant.components.recorder.get_instance",
        get_instance,
    ), patch(
        "homeassistant.components.recorder.core.Recorder",
        mock_recorder_class,
    ), patch(
        "homeassistant.components.recorder.async_setup",
        AsyncMock(return_value=True),
    ):
        yield mock_instance


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Create a mock config entry for testing."""
    return MockConfigEntry(
        domain=DOMAIN,
        data=TEST_CONFIG,
        title="Test Area",
        unique_id="uniqueid123",
    )


@pytest.fixture
async def setup_test_entities(hass: HomeAssistant) -> None:
    """Set up test entities with correct device classes."""
    # Register entities in the entity registry
    registry = er.async_get(hass)
    registry.async_get_or_create(
        domain="binary_sensor",
        platform="test",
        unique_id="motion1",
        suggested_object_id="motion1",
        original_device_class=BinarySensorDeviceClass.MOTION,
    )
    registry.async_get_or_create(
        domain="binary_sensor",
        platform="test",
        unique_id="motion2",
        suggested_object_id="motion2",
        original_device_class=BinarySensorDeviceClass.MOTION,
    )
    registry.async_get_or_create(
        domain="binary_sensor",
        platform="test",
        unique_id="motion_new",
        suggested_object_id="motion_new",
        original_device_class=BinarySensorDeviceClass.MOTION,
    )

    # Set up motion sensors
    hass.states.async_set(
        "binary_sensor.motion1",
        STATE_OFF,
        {
            "friendly_name": "Motion Sensor 1",
            "device_class": BinarySensorDeviceClass.MOTION,
        },
    )
    hass.states.async_set(
        "binary_sensor.motion2",
        STATE_OFF,
        {
            "friendly_name": "Motion Sensor 2",
            "device_class": BinarySensorDeviceClass.MOTION,
        },
    )
    hass.states.async_set(
        "binary_sensor.motion_new",
        STATE_OFF,
        {
            "friendly_name": "New Motion Sensor",
            "device_class": BinarySensorDeviceClass.MOTION,
        },
    )
    await hass.async_block_till_done()


@pytest.fixture
async def init_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,  # pylint: disable=redefined-outer-name
    mock_recorder: MagicMock,  # Add mock_recorder as a dependency
    setup_test_entities,  # Add test entities fixture
) -> MockConfigEntry:
    """Set up the area occupancy integration for testing."""
    # Set up recorder component first
    await hass.async_add_executor_job(lambda: None)  # Ensure executor is running

    # Set up the integration
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    return mock_config_entry


@pytest.fixture(autouse=True)
def cleanup_debouncer():
    """Clean up any debouncer timers after each test."""
    yield

    # Get all active timers from the event loop
    loop = asyncio.get_event_loop()
    for handle in loop._scheduled:
        if isinstance(handle, asyncio.TimerHandle) and "Debouncer._on_debounce" in str(
            handle
        ):
            handle.cancel()
