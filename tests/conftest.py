"""Pytest configuration and fixtures for Area Occupancy Detection tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_registry import EntityRegistry
from homeassistant.util import dt as dt_util

from custom_components.area_occupancy.const import (
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
    CONF_VERSION,
    DEFAULT_THRESHOLD,
    DOMAIN,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import Config
from custom_components.area_occupancy.data.entity_type import EntityType, InputType


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_hass() -> Mock:
    """Create a mock Home Assistant instance."""
    hass = Mock(spec=HomeAssistant)
    hass.config = Mock()
    hass.config.path = Mock(return_value="/config")
    hass.states = Mock()
    hass.config_entries = Mock()
    hass.data = {}
    hass.bus = Mock()
    hass.services = Mock()
    hass.loop = asyncio.get_event_loop()

    # Mock async methods
    hass.async_create_task = Mock(side_effect=lambda coro: asyncio.create_task(coro))
    hass.async_add_executor_job = AsyncMock()

    return hass


@pytest.fixture
def mock_config_entry() -> Mock:
    """Create a mock config entry."""
    entry = Mock(spec=ConfigEntry)
    entry.entry_id = "test_entry_id"
    entry.version = CONF_VERSION
    entry.minor_version = 1
    entry.data = {
        CONF_NAME: "Test Area",
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.test_motion",
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_THRESHOLD: DEFAULT_THRESHOLD,
    }
    entry.options = {}
    entry.runtime_data = None
    entry.add_update_listener = Mock()
    entry.async_on_unload = Mock()
    return entry


@pytest.fixture
def sample_config_data() -> dict[str, Any]:
    """Create sample configuration data."""
    return {
        CONF_NAME: "Test Area",
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.test_motion",
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_THRESHOLD: DEFAULT_THRESHOLD,
    }


@pytest.fixture
def mock_entity_registry() -> Mock:
    """Create a mock entity registry."""
    registry = Mock(spec=EntityRegistry)
    registry.entities = {}
    registry.async_get_entity_id = Mock(return_value=None)
    registry.async_update_entity = Mock()
    return registry


@pytest.fixture
def mock_states() -> list[Mock]:
    """Create mock Home Assistant states."""
    states = []

    # Motion sensor states
    motion_state = Mock()
    motion_state.entity_id = "binary_sensor.test_motion"
    motion_state.state = STATE_ON
    motion_state.last_changed = dt_util.utcnow() - timedelta(minutes=5)
    motion_state.last_updated = dt_util.utcnow() - timedelta(minutes=5)
    motion_state.attributes = {"device_class": "motion"}
    states.append(motion_state)

    # Light sensor states
    light_state = Mock()
    light_state.entity_id = "light.test_light"
    light_state.state = STATE_ON
    light_state.last_changed = dt_util.utcnow() - timedelta(minutes=10)
    light_state.last_updated = dt_util.utcnow() - timedelta(minutes=10)
    light_state.attributes = {}
    states.append(light_state)

    return states


@pytest.fixture
async def mock_coordinator(
    mock_hass: Mock, mock_config_entry: Mock
) -> AsyncGenerator[Mock, None]:
    """Create a mock coordinator."""
    coordinator = Mock(spec=AreaOccupancyCoordinator)
    coordinator.hass = mock_hass
    coordinator.config_entry = mock_config_entry
    coordinator.entry_id = mock_config_entry.entry_id
    coordinator.available = True
    coordinator.probability = 0.5
    coordinator.is_occupied = False
    coordinator.threshold = 0.5
    coordinator.prior = 0.3
    coordinator.decay = 1.0
    coordinator.last_updated = dt_util.utcnow()
    coordinator.last_changed = dt_util.utcnow()
    coordinator.occupancy_entity_id = None
    coordinator.wasp_entity_id = None

    # Mock config
    coordinator.config = Mock(spec=Config)
    coordinator.config.name = "Test Area"
    coordinator.config.threshold = 0.5
    coordinator.config.decay.enabled = True
    coordinator.config.decay.window = 300
    coordinator.config.wasp_in_box.enabled = False

    # Mock methods
    coordinator.async_config_entry_first_refresh = AsyncMock()
    coordinator.async_shutdown = AsyncMock()
    coordinator.async_update_options = AsyncMock()
    coordinator.request_update = Mock()
    coordinator.async_add_listener = Mock(return_value=Mock())

    yield coordinator


@pytest.fixture
def mock_entity_type() -> Mock:
    """Create a mock entity type."""
    entity_type = Mock(spec=EntityType)
    entity_type.input_type = InputType.MOTION
    entity_type.weight = 0.8
    entity_type.prob_true = 0.25
    entity_type.prob_false = 0.05
    entity_type.prior = 0.35
    entity_type.active_states = [STATE_ON]
    entity_type.active_range = None
    entity_type.is_active = Mock(return_value=True)
    return entity_type


@pytest.fixture
def mock_recorder() -> Generator[Mock, None, None]:
    """Mock the recorder component."""
    with patch(
        "custom_components.area_occupancy.data.prior.get_instance"
    ) as mock_get_instance:
        mock_instance = Mock()
        mock_instance.async_add_executor_job = AsyncMock()
        mock_get_instance.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_storage() -> Generator[Mock, None, None]:
    """Mock the storage system."""
    with patch("homeassistant.helpers.storage.Store") as mock_store:
        store_instance = Mock()
        store_instance.async_load = AsyncMock(return_value=None)
        store_instance.async_save = AsyncMock()
        mock_store.return_value = store_instance
        yield store_instance


@pytest.fixture
def mock_significant_states() -> Generator[Mock, None, None]:
    """Mock significant states from recorder."""
    with patch(
        "custom_components.area_occupancy.data.prior.get_significant_states"
    ) as mock_states:
        # Create mock states for testing
        mock_state_on = Mock()
        mock_state_on.state = STATE_ON
        mock_state_on.last_changed = dt_util.utcnow() - timedelta(hours=2)

        mock_state_off = Mock()
        mock_state_off.state = STATE_OFF
        mock_state_off.last_changed = dt_util.utcnow() - timedelta(hours=1)

        mock_states.return_value = {
            "binary_sensor.test_motion": [mock_state_on, mock_state_off]
        }
        yield mock_states


class MockAsyncContextManager:
    """Mock async context manager."""

    def __init__(self, return_value: Any = None) -> None:
        self.return_value = return_value

    async def __aenter__(self) -> Any:
        return self.return_value

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


@pytest.fixture
def freeze_time() -> Generator[datetime, None, None]:
    """Fixture to freeze time for consistent testing."""
    frozen_time = dt_util.utcnow()
    with patch("homeassistant.util.dt.utcnow", return_value=frozen_time):
        yield frozen_time
