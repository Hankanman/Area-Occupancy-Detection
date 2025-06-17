"""Pytest configuration and fixtures for Area Occupancy Detection tests."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from datetime import datetime, timedelta
import tempfile
import time
import types
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

# ruff: noqa: SLF001
from custom_components.area_occupancy.const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    # Import all config constants for comprehensive config entry
    CONF_DECAY_ENABLED,
    CONF_DECAY_MIN_DELAY,
    CONF_DECAY_WINDOW,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_HISTORY_PERIOD,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
    CONF_WASP_ENABLED,
    CONF_WASP_MAX_DURATION,
    CONF_WASP_MOTION_TIMEOUT,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_MIN_DELAY,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOMAIN,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import Config
from custom_components.area_occupancy.data.entity import EntityManager
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_registry import EntityRegistry
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Configure pytest-asyncio to use function scope for event loops
pytestmark = pytest.mark.asyncio(loop_scope="function")


@pytest.fixture
def mock_hass() -> Generator[Mock]:
    """Create a comprehensive mock Home Assistant instance."""
    hass = Mock(spec=HomeAssistant)

    # Basic configuration
    hass.config = Mock()
    hass.config.path = Mock(return_value="/config")
    hass.config.config_dir = tempfile.gettempdir()

    # Add state attribute for storage operations
    hass.state = Mock()

    # States and entities
    hass.states = Mock()
    hass.states.async_all = Mock(return_value=[])
    hass.states.async_entity_ids = Mock(return_value=[])
    hass.states.get = Mock(return_value=Mock(attributes={"device_class": None}))

    # Config entries
    hass.config_entries = Mock()
    hass.config_entries.async_entries = Mock(return_value=[])
    hass.config_entries.async_forward_entry_setups = AsyncMock(return_value=True)
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    hass.config_entries.async_reload = AsyncMock(return_value=True)

    # Data storage - use proper dict to avoid "argument of type 'Mock' is not iterable" errors
    hass.data = {
        DOMAIN: {},
        "area_occupancy": {},
        "area_occupancy_coordinators": {},
        "area_occupancy_storage": {},
        "storage_manager": {},
        "entity_registry": Mock(),
        "recorder_instance": Mock(),  # Add recorder instance
    }

    # Event system
    hass.bus = Mock()
    hass.bus.async_listen = Mock()
    hass.services = Mock()
    hass.services.async_register = Mock()

    # Event loop - create new loop for each test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    hass.loop = loop

    # Async methods
    hass.async_create_task = Mock(side_effect=lambda coro: asyncio.create_task(coro))
    hass.async_add_executor_job = AsyncMock()
    hass.async_add_job = AsyncMock()
    hass.async_run_job = AsyncMock()
    hass.async_call_later = Mock(return_value=Mock())
    hass.async_track_time_interval = Mock(return_value=Mock())
    hass.async_track_point_in_time = Mock(return_value=Mock())

    # Storage system
    hass.helpers = Mock()
    hass.helpers.storage = Mock()
    hass.helpers.storage.Store = Mock()
    hass.helpers.storage.Store.async_load = AsyncMock(return_value=None)
    hass.helpers.storage.Store.async_save = AsyncMock()

    # Add storage manager mock to fix async_invalidate errors
    mock_storage_manager = Mock()
    mock_storage_manager.async_invalidate = Mock()
    hass.data["storage_manager"] = mock_storage_manager

    # Entity registry
    hass.helpers.entity_registry = Mock()
    hass.helpers.entity_registry.async_get = AsyncMock(return_value=Mock())
    hass.helpers.entity_registry.async_get_entity_id = AsyncMock(return_value=None)
    hass.helpers.entity_registry.async_update_entity = AsyncMock()

    # Event helpers
    hass.helpers.event = Mock()
    hass.helpers.event.async_track_point_in_time = Mock(return_value=Mock())
    hass.helpers.event.async_track_time_interval = Mock(return_value=Mock())

    yield hass

    # Cleanup
    try:
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()
    except (RuntimeError, AttributeError):
        pass  # Ignore loop cleanup errors


@pytest.fixture
def mock_config_entry() -> Mock:
    """Create a comprehensive mock config entry with all configuration options."""
    entry = Mock(spec=ConfigEntry)
    entry.entry_id = "test_entry_id"
    entry.domain = DOMAIN
    entry.version = CONF_VERSION
    entry.minor_version = CONF_VERSION_MINOR
    entry.source = "user"
    entry.title = "Test Area"
    entry.unique_id = "test_unique_id"

    # Comprehensive configuration data
    entry.data = {
        CONF_NAME: "Test Area",
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.test_motion",
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_THRESHOLD: DEFAULT_THRESHOLD,
        CONF_DECAY_ENABLED: DEFAULT_DECAY_ENABLED,
        CONF_DECAY_WINDOW: DEFAULT_DECAY_WINDOW,
        CONF_DECAY_MIN_DELAY: DEFAULT_DECAY_MIN_DELAY,
        CONF_HISTORICAL_ANALYSIS_ENABLED: DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
        CONF_HISTORY_PERIOD: DEFAULT_HISTORY_PERIOD,
        CONF_DOOR_SENSORS: [],
        CONF_WINDOW_SENSORS: [],
        CONF_LIGHTS: [],
        CONF_MEDIA_DEVICES: [],
        CONF_APPLIANCES: [],
        CONF_ILLUMINANCE_SENSORS: [],
        CONF_HUMIDITY_SENSORS: [],
        CONF_TEMPERATURE_SENSORS: [],
        CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
        CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
        CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
        CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
        CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
        CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
        CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        CONF_WASP_ENABLED: False,
        CONF_WASP_MOTION_TIMEOUT: DEFAULT_WASP_MOTION_TIMEOUT,
        CONF_WASP_WEIGHT: DEFAULT_WASP_WEIGHT,
        CONF_WASP_MAX_DURATION: DEFAULT_WASP_MAX_DURATION,
        CONF_DOOR_ACTIVE_STATE: DEFAULT_DOOR_ACTIVE_STATE,
        CONF_WINDOW_ACTIVE_STATE: DEFAULT_WINDOW_ACTIVE_STATE,
        CONF_MEDIA_ACTIVE_STATES: DEFAULT_MEDIA_ACTIVE_STATES,
        CONF_APPLIANCE_ACTIVE_STATES: DEFAULT_APPLIANCE_ACTIVE_STATES,
    }

    # Options and runtime data
    entry.options = {}
    entry.runtime_data = None
    entry.state = None

    # Config entry methods
    entry.add_update_listener = Mock()
    entry.async_on_unload = Mock()
    entry.async_setup = AsyncMock()
    entry.async_unload = AsyncMock()
    entry.async_remove = AsyncMock()
    entry.async_update = AsyncMock()

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

    # Add entities property that can be iterated
    class EntitiesContainer:
        def values(self):
            return []

    registry.entities = EntitiesContainer()
    return registry


@pytest.fixture
def mock_entity_manager() -> Mock:
    """Create a comprehensive mock entity manager."""
    manager = Mock(spec=EntityManager)

    # Basic attributes
    manager.entities = {}
    manager.active_entities = []
    manager.inactive_entities = []

    # Methods
    manager.get_entity = Mock(return_value=None)
    manager.add_entity = Mock()
    manager.remove_entity = Mock()
    manager.cleanup = Mock()
    manager.reset_entities = Mock()
    manager.async_initialize = AsyncMock()

    # Serialization method with comprehensive entity data
    manager.to_dict.return_value = {
        "entities": {
            "binary_sensor.test_motion": {
                "entity_id": "binary_sensor.test_motion",
                "type": "motion",
                "probability": 0.7,
                "state": "on",
                "is_active": True,
                "available": True,
                "last_updated": dt_util.utcnow().isoformat(),
                "last_changed": dt_util.utcnow().isoformat(),
                "prior": {
                    "prior": 0.35,
                    "prob_given_true": 0.8,
                    "prob_given_false": 0.1,
                    "last_updated": dt_util.utcnow().isoformat(),
                },
                "decay": {
                    "is_decaying": False,
                    "decay_start_time": None,
                    "decay_start_probability": 0.0,
                    "decay_window": 300,
                    "decay_enabled": True,
                    "decay_factor": 1.0,
                },
            }
        }
    }

    return manager


@pytest.fixture
def mock_coordinator(
    mock_hass: Mock, mock_config_entry: Mock, mock_entity_manager: Mock
) -> Mock:
    """Create a comprehensive mock coordinator."""
    coordinator = Mock(spec=AreaOccupancyCoordinator)

    # Basic attributes
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
    coordinator.last_update_success = True

    # Mock config
    coordinator.config = Mock(spec=Config)
    coordinator.config.name = "Test Area"
    coordinator.config.threshold = 0.5
    coordinator.config.decay = Mock()
    coordinator.config.decay.enabled = True
    coordinator.config.decay.window = 300
    coordinator.config.wasp_in_box = Mock()
    coordinator.config.wasp_in_box.enabled = False
    coordinator.config.motion_sensors = ["binary_sensor.test_motion"]
    coordinator.config.primary_occupancy_sensor = "binary_sensor.test_motion"

    # Add history configuration for prior tests
    coordinator.config.history = Mock()
    coordinator.config.history.enabled = True
    coordinator.config.history.period = 30
    coordinator.config.start_time = dt_util.utcnow() - timedelta(days=30)
    coordinator.config.end_time = dt_util.utcnow()

    # Add sensors configuration for prior tests
    coordinator.config.sensors = Mock()
    coordinator.config.sensors.motion = [
        "binary_sensor.motion1",
        "binary_sensor.motion2",
    ]
    coordinator.config.sensors.primary_occupancy = "binary_sensor.motion1"

    # Add weights configuration for entity_type tests
    coordinator.config.weights = Mock()
    coordinator.config.weights.motion = 0.9
    coordinator.config.weights.media = 0.7
    coordinator.config.weights.appliance = 0.6
    coordinator.config.weights.door = 0.5
    coordinator.config.weights.window = 0.4
    coordinator.config.weights.light = 0.1
    coordinator.config.weights.environmental = 0.3

    # Create a custom sensor_states object that returns None for undefined attributes
    class SensorStates:
        def __init__(self):
            self.door = [STATE_ON]

        def __getattr__(self, name):
            return None

    coordinator.config.sensor_states = SensorStates()

    # Entity manager with coordinator reference
    mock_entity_manager.coordinator = coordinator
    coordinator.entities = mock_entity_manager

    # Add mock entities for prior tests
    coordinator.entities.entities = {
        "entity1": Mock(entity_id="entity1"),
        "entity2": Mock(entity_id="entity2"),
    }

    # Entity types
    coordinator.entity_types = Mock()
    coordinator.entity_types.to_dict = Mock(return_value={"entity_types": {}})
    coordinator.entity_types.async_initialize = AsyncMock()

    # Storage
    coordinator.storage = Mock()
    coordinator.storage.async_load = AsyncMock(return_value=None)
    coordinator.storage.async_save = AsyncMock()
    coordinator.storage.async_initialize = AsyncMock()
    coordinator.storage.async_shutdown = AsyncMock()
    coordinator.storage.async_save_instance_data = AsyncMock()
    coordinator.storage.async_load_with_compatibility_check = AsyncMock(
        return_value=(None, False)
    )

    # Priors manager
    coordinator.priors = Mock()
    coordinator.priors.update_all_entity_priors = AsyncMock(return_value=0)
    coordinator.priors.calculate = AsyncMock()

    # Config manager
    coordinator.config_manager = Mock()
    coordinator.config_manager.config = coordinator.config
    coordinator.config_manager.update_config = AsyncMock()

    # Methods
    coordinator.async_config_entry_first_refresh = AsyncMock()
    coordinator.async_shutdown = AsyncMock()
    coordinator.async_update_options = AsyncMock()
    coordinator.async_refresh = AsyncMock()
    coordinator.request_update = Mock()
    coordinator.async_add_listener = Mock(return_value=Mock())
    coordinator.async_update_data = AsyncMock()
    coordinator.async_save_data = AsyncMock()
    coordinator._async_setup = AsyncMock()
    coordinator._async_update_data = AsyncMock(
        return_value={"last_updated": dt_util.utcnow().isoformat()}
    )
    coordinator._async_save_data = AsyncMock()
    coordinator._schedule_next_prior_update = AsyncMock()
    coordinator._handle_prior_update = AsyncMock()
    coordinator._async_refresh_finished = Mock()
    coordinator.async_set_updated_data = Mock()
    coordinator.update_learned_priors = AsyncMock()
    coordinator.async_load_stored_data = AsyncMock()

    # Mock the calculation method to return realistic values
    coordinator._calculate_entity_aggregates = Mock(
        return_value={
            "probability": 0.5,
            "prior": 0.3,
            "decay": 1.0,
        }
    )

    # Mock data property
    coordinator.data = {"last_updated": dt_util.utcnow().isoformat()}

    # Device info
    coordinator.device_info = {
        "identifiers": {(DOMAIN, coordinator.entry_id)},
        "name": "Test Area",
        "manufacturer": "Area Occupancy",
        "model": "Area Occupancy Detection",
        "sw_version": "1.0.0",
    }

    # Mock timers and trackers
    coordinator._prior_update_tracker = None
    coordinator._next_prior_update = None
    coordinator._last_prior_update = None

    # Mock binary_sensor_entity_ids property to return a proper dictionary
    coordinator.binary_sensor_entity_ids = {
        "occupancy": "binary_sensor.test_area_occupancy",
        "wasp": "binary_sensor.test_area_wasp",
    }

    return coordinator


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
def mock_recorder() -> Generator[Mock]:
    """Mock the recorder component."""
    with patch(
        "custom_components.area_occupancy.data.prior.get_instance"
    ) as mock_get_instance:
        mock_instance = Mock()
        mock_instance.async_add_executor_job = AsyncMock()
        mock_get_instance.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_storage() -> Generator[Mock]:
    """Mock the storage system."""
    with patch("homeassistant.helpers.storage.Store") as mock_store:
        store_instance = Mock()
        store_instance.async_load = AsyncMock(return_value=None)
        store_instance.async_save = AsyncMock()
        # Add required Store attributes
        store_instance._load_future = None
        store_instance._data = None
        store_instance.hass = None
        mock_store.return_value = store_instance
        yield store_instance


@pytest.fixture
def mock_storage_manager_patches():
    """Provide common patches for StorageManager tests."""
    return [
        patch("homeassistant.helpers.storage.Store.__init__", return_value=None),
        patch("homeassistant.helpers.storage.Store.async_load", new_callable=AsyncMock),
        patch("homeassistant.helpers.storage.Store.async_save", new_callable=AsyncMock),
        patch(
            "homeassistant.helpers.event.async_track_point_in_time", return_value=Mock()
        ),
    ]


@pytest.fixture
def mock_significant_states() -> Generator[Mock]:
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
        """Initialize the mock async context manager."""
        self.return_value = return_value

    async def __aenter__(self) -> Any:
        """Enter the context manager."""
        return self.return_value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the context manager."""


@pytest.fixture
def freeze_time() -> Generator[datetime]:
    """Fixture to freeze time for consistent testing."""
    frozen_time = dt_util.utcnow()
    with patch("homeassistant.util.dt.utcnow", return_value=frozen_time):
        yield frozen_time


@pytest.fixture
def valid_entity_data() -> dict[str, Any]:
    """Create valid entity data for testing."""
    return {
        "entity_id": "binary_sensor.test_motion",
        "probability": 0.5,
        "state": STATE_ON,
        "is_active": True,
        "available": True,
        "type": "motion",
        "prior": {"prior": 0.3},
        "decay": {"decay_factor": 1.0},
        "last_updated": dt_util.utcnow().isoformat(),
        "last_changed": dt_util.utcnow().isoformat(),
    }


@pytest.fixture
def valid_storage_data() -> dict[str, Any]:
    """Create valid storage data for testing with current format."""
    return {
        "version": CONF_VERSION,
        "minor_version": CONF_VERSION_MINOR,
        "data": {
            "instances": {
                "test_entry_id": {
                    "entities": {
                        "binary_sensor.test_motion": {
                            "entity_id": "binary_sensor.test_motion",
                            "probability": 0.5,
                            "state": STATE_ON,
                            "is_active": True,
                            "available": True,
                            "type": "motion",
                            "prior": {"prior": 0.3},
                            "decay": {"decay_factor": 1.0},
                            "last_updated": dt_util.utcnow().isoformat(),
                            "last_changed": dt_util.utcnow().isoformat(),
                        }
                    }
                }
            }
        },
    }


@pytest.fixture
def mock_frame_helper():
    """Mock the Home Assistant frame helper for config flow tests."""
    with patch("homeassistant.helpers.frame._hass") as mock_hass:
        mock_hass.hass = Mock()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        mock_hass.hass.loop = loop
        yield mock_hass
        try:
            if loop.is_running():
                loop.stop()
            if not loop.is_closed():
                loop.close()
        except (RuntimeError, AttributeError):
            pass


# Utility functions for common test patterns
def create_mock_entity_registry_with_entities(entities: list[dict]) -> Mock:
    """Create a mock entity registry with specific entities."""
    registry = Mock(spec=EntityRegistry)

    class EntitiesContainer:
        def values(self):
            return [
                Mock(
                    entity_id=entity["entity_id"],
                    domain=entity.get("domain", "binary_sensor"),
                    device_class=entity.get("device_class"),
                    original_device_class=entity.get("device_class"),
                )
                for entity in entities
            ]

    registry.entities = EntitiesContainer()
    registry.async_get_entity_id = Mock(return_value=None)
    registry.async_update_entity = Mock()
    return registry


def create_storage_data_with_entities(entry_id: str, entities: dict) -> dict[str, Any]:
    """Create storage data with specific entities."""
    return {
        "version": CONF_VERSION,
        "minor_version": CONF_VERSION_MINOR,
        "data": {"instances": {entry_id: {"entities": entities}}},
    }


# Additional centralized fixtures for common patterns across test files


@pytest.fixture
def mock_coordinator_with_threshold(mock_coordinator: Mock) -> Mock:
    """Create a coordinator mock with threshold-specific attributes."""
    mock_coordinator.threshold = 0.6
    mock_coordinator.config.threshold = 0.6
    mock_coordinator.is_occupied = False  # 0.5 < 0.6
    mock_coordinator.async_update_threshold = AsyncMock()
    return mock_coordinator


@pytest.fixture
def mock_coordinator_with_sensors(mock_coordinator: Mock) -> Mock:
    """Create a coordinator mock with sensor-specific attributes."""
    mock_coordinator.prior = 0.35
    mock_coordinator.probability = 0.65
    mock_coordinator.decay = 0.8

    # Mock entity manager with comprehensive entities
    mock_coordinator.entities.entities = {
        "binary_sensor.motion1": Mock(
            entity_id="binary_sensor.motion1",
            available=True,
            is_active=True,
            probability=0.75,
            type=Mock(input_type=InputType.MOTION, weight=0.85),
            decay=Mock(is_decaying=False, decay_factor=1.0),
            prior=Mock(prior=0.35),
        ),
        "binary_sensor.motion2": Mock(
            entity_id="binary_sensor.motion2",
            available=True,
            is_active=False,
            probability=0.25,
            type=Mock(input_type=InputType.MOTION, weight=0.85),
            decay=Mock(is_decaying=True, decay_factor=0.8),
            prior=Mock(prior=0.35),
        ),
        "light.test_light": Mock(
            entity_id="light.test_light",
            available=False,
            is_active=False,
            probability=0.15,
            type=Mock(input_type=InputType.LIGHT, weight=0.2),
            decay=Mock(is_decaying=False, decay_factor=1.0),
            prior=Mock(prior=0.1),
        ),
        "media_player.tv": Mock(
            entity_id="media_player.tv",
            available=True,
            is_active=True,
            probability=0.85,
            type=Mock(input_type=InputType.MEDIA, weight=0.7),
            decay=Mock(is_decaying=False, decay_factor=1.0),
            prior=Mock(prior=0.15),
        ),
    }

    # Update the calculation mock to return realistic aggregate values
    mock_coordinator._calculate_entity_aggregates.return_value = {
        "probability": 0.65,
        "prior": 0.35,
        "decay": 0.8,
    }

    return mock_coordinator


@pytest.fixture
def mock_prior() -> Mock:
    """Create a comprehensive mock prior."""
    from custom_components.area_occupancy.data.prior import Prior

    prior = Mock(spec=Prior)
    prior.prior = 0.35
    prior.prob_given_true = 0.8
    prior.prob_given_false = 0.1
    prior.last_updated = dt_util.utcnow()
    prior.to_dict.return_value = {
        "prior": 0.35,
        "prob_given_true": 0.8,
        "prob_given_false": 0.1,
        "last_updated": dt_util.utcnow().isoformat(),
    }
    return prior


@pytest.fixture
def mock_decay() -> Mock:
    """Create a comprehensive mock decay."""
    from custom_components.area_occupancy.data.decay import Decay

    decay = Mock(spec=Decay)
    decay.is_decaying = False
    decay.last_trigger_ts = time.time()
    decay.half_life = 60.0
    decay.decay_factor = 1.0
    decay.should_start_decay.return_value = False
    decay.should_stop_decay.return_value = False
    decay.is_decay_complete.return_value = False
    decay.to_dict.return_value = {
        "last_trigger_ts": time.time(),
        "half_life": 60.0,
        "is_decaying": False,
    }
    return decay


@pytest.fixture
def mock_service_call() -> Mock:
    """Create a mock service call with common attributes."""
    from homeassistant.core import ServiceCall

    call = Mock(spec=ServiceCall)
    call.data = {"entry_id": "test_entry_id"}
    call.return_response = True
    return call


@pytest.fixture
def mock_service_call_with_entity() -> Mock:
    """Create a mock service call with entity_id."""
    from homeassistant.core import ServiceCall

    call = Mock(spec=ServiceCall)
    call.data = {"entry_id": "test_entry_id", "entity_id": "binary_sensor.test_motion"}
    call.return_response = True
    return call


@pytest.fixture
def mock_comprehensive_entity(
    mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
) -> Mock:
    """Create a comprehensive mock entity with all components."""
    from custom_components.area_occupancy.data.entity import Entity

    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.test_motion"
    entity.type = mock_entity_type
    entity.probability = 0.5
    entity.prior = mock_prior
    entity.decay = mock_decay
    entity.state = STATE_ON
    entity.is_active = True
    entity.available = True
    entity.last_updated = dt_util.utcnow()
    entity.last_changed = dt_util.utcnow()
    entity._coordinator = None

    # Mock methods
    entity.set_coordinator = Mock()
    entity.update_probability = Mock()
    entity.start_decay_timer = Mock()
    entity.stop_decay_timer = Mock()
    entity.handle_decay_timer = AsyncMock()
    entity.stop_decay_completely = Mock()
    entity.cleanup = Mock()

    # Mock serialization
    entity.to_dict.return_value = {
        "entity_id": "binary_sensor.test_motion",
        "type": InputType.MOTION.value,
        "probability": 0.5,
        "state": STATE_ON,
        "is_active": True,
        "available": True,
        "last_updated": dt_util.utcnow().isoformat(),
        "last_changed": dt_util.utcnow().isoformat(),
        "prior": mock_prior.to_dict.return_value,
        "decay": mock_decay.to_dict.return_value,
    }

    return entity


@pytest.fixture
def mock_comprehensive_entity_manager(
    mock_coordinator: Mock, mock_comprehensive_entity: Mock
) -> Mock:
    """Create a comprehensive mock entity manager with entities."""
    manager = Mock(spec=EntityManager)

    # Basic attributes
    manager.coordinator = mock_coordinator
    manager.entities = {"binary_sensor.test_motion": mock_comprehensive_entity}
    manager.active_entities = [mock_comprehensive_entity]
    manager.inactive_entities = []

    # Properties
    manager.entity_ids = ["binary_sensor.test_motion"]

    # Methods
    manager.get_entity = Mock(return_value=mock_comprehensive_entity)
    manager.add_entity = Mock()
    manager.remove_entity = Mock()
    manager.cleanup = Mock()
    manager.reset_entities = AsyncMock()
    manager.async_initialize = AsyncMock()
    manager.setup_entity_tracking = Mock()
    manager.state_changed_listener = AsyncMock()

    # Serialization
    manager.to_dict.return_value = {
        "entities": {
            "binary_sensor.test_motion": mock_comprehensive_entity.to_dict.return_value
        }
    }

    return manager


@pytest.fixture
def mock_device_info() -> dict[str, Any]:
    """Create mock device info for entities."""
    return {
        "identifiers": {("area_occupancy", "test_entry_id")},
        "name": "Test Area",
        "manufacturer": "Area Occupancy Detection",
        "model": "Area Monitor",
        "sw_version": "1.0.0",
    }


@pytest.fixture
def mock_real_coordinator() -> Mock:
    """Create a real coordinator instance for integration tests."""
    # This fixture should be used sparingly, only for tests that need real coordinator behavior
    # Most tests should use the mock_coordinator fixture instead
    return Mock()


# Global patches for common issues
@pytest.fixture(autouse=True)
def mock_recorder_globally():
    """Automatically mock recorder for all tests."""
    with (
        patch("homeassistant.helpers.recorder.get_instance") as mock_get_instance_ha,
        patch(
            "custom_components.area_occupancy.data.prior.get_instance"
        ) as mock_get_instance_local,
    ):
        mock_instance = Mock()
        mock_instance.async_add_executor_job = AsyncMock(return_value={})
        mock_get_instance_ha.return_value = mock_instance
        mock_get_instance_local.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=True)
def mock_significant_states_globally():
    """Automatically mock significant states for all tests."""
    with patch(
        "custom_components.area_occupancy.data.prior.get_significant_states"
    ) as mock_states:
        mock_states.return_value = {}
        yield mock_states


@pytest.fixture(autouse=True)
def mock_track_point_in_time_globally():
    """Automatically mock async_track_point_in_time for all tests."""
    with patch("homeassistant.helpers.event.async_track_point_in_time") as mock_track:
        mock_track.return_value = Mock()
        yield mock_track


@pytest.fixture
def mock_entity_for_prior_tests() -> Mock:
    """Create a mock entity specifically for prior calculation tests."""
    from custom_components.area_occupancy.data.entity import Entity

    entity = Mock(spec=Entity)
    entity.entity_id = "light.test_light"

    # Mock entity type with proper numeric values (not Mock objects)
    entity.type = Mock()
    entity.type.active_states = [STATE_ON]
    entity.type.prior = 0.35  # Real float value, not Mock
    entity.type.prob_true = 0.8  # Real float value, not Mock
    entity.type.prob_false = 0.1  # Real float value, not Mock
    entity.type.input_type = Mock()
    entity.type.input_type.value = "light"

    return entity


@pytest.fixture(autouse=True)
def mock_debouncer_globally():
    """Automatically mock Debouncer for all tests."""
    with patch(
        "custom_components.area_occupancy.coordinator.Debouncer"
    ) as mock_debouncer_class:
        mock_debouncer = Mock()
        mock_debouncer.async_call = AsyncMock(
            side_effect=lambda: None
        )  # Ensure it calls the method
        mock_debouncer.async_shutdown = AsyncMock()
        mock_debouncer_class.return_value = mock_debouncer
        yield mock_debouncer


@pytest.fixture(autouse=True)
def mock_data_update_coordinator_debouncer():
    """Automatically mock DataUpdateCoordinator's debouncer for all tests."""
    original_init = DataUpdateCoordinator.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._debounced_refresh = AsyncMock()

    with patch(
        "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.__init__",
        patched_init,
    ):
        yield
