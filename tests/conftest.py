"""Pytest configuration and fixtures for Area Occupancy Detection tests."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import contextlib
from datetime import datetime, timedelta
import tempfile
import time
import types
from typing import Any
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest

# ruff: noqa: SLF001, PLC0415
from custom_components.area_occupancy.const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    # Import all config constants for comprehensive config entry
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
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
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOMAIN,
    HA_RECORDER_DAYS,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import (
    Config,
    Decay,
    Sensors,
    SensorStates,
    WaspInBox,
    Weights,
)
from custom_components.area_occupancy.data.entity import EntityManager
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.data.prior import Prior as PriorClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.entity_registry import EntityRegistry
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util


# Ensure an event loop exists for fixtures that rely on it
@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """Ensure an event loop exists and enable debug mode."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.set_debug(True)


@pytest.fixture
def mock_hass() -> Mock:
    """Create a comprehensive mock Home Assistant instance."""
    hass = Mock(spec=HomeAssistant)

    # Basic configuration
    hass.config = Mock()
    hass.config.path = Mock(return_value="/config")
    hass.config.config_dir = tempfile.gettempdir()

    # Add state attribute for database operations
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
        "area_occupancy_db": {},
        "entity_registry": Mock(),
        "recorder_instance": Mock(),  # Add recorder instance
    }

    # Event system
    hass.bus = Mock()
    hass.bus.async_listen = Mock()
    hass.services = Mock()
    hass.services.async_register = Mock()

    # Use pytest-asyncio's event loop instead of creating our own
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    hass.loop = loop

    # Async methods
    hass.async_create_task = Mock(side_effect=lambda coro: asyncio.create_task(coro))

    # Make async_add_executor_job actually execute the function
    async def async_add_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)

    hass.async_add_executor_job = async_add_executor_job

    hass.async_add_job = AsyncMock()
    hass.async_run_job = AsyncMock()

    # Create cancellable mocks for timer functions
    def create_cancellable_timer():
        mock_timer = Mock()
        mock_timer.cancel = Mock()
        mock_timer.cancelled = Mock(return_value=True)
        return mock_timer

    hass.async_call_later = Mock(
        side_effect=lambda *args, **kwargs: create_cancellable_timer()
    )
    hass.async_track_time_interval = Mock(
        side_effect=lambda *args, **kwargs: create_cancellable_timer()
    )
    hass.async_track_point_in_time = Mock(
        side_effect=lambda *args, **kwargs: create_cancellable_timer()
    )

    # Database system
    hass.helpers = Mock()
    hass.helpers.db = Mock()
    hass.helpers.db.AreaOccupancyDB = Mock()
    hass.helpers.db.AreaOccupancyDB.async_load = AsyncMock(return_value=None)
    hass.helpers.db.AreaOccupancyDB.async_save = AsyncMock()

    # Add database manager mock to fix async_invalidate errors
    mock_db_manager = Mock()
    mock_db_manager.async_invalidate = Mock()
    hass.data["area_occupancy_db"] = mock_db_manager

    # Entity registry
    hass.helpers.entity_registry = Mock()
    hass.helpers.entity_registry.async_get = AsyncMock(return_value=Mock())
    hass.helpers.entity_registry.async_get_entity_id = AsyncMock(return_value=None)
    hass.helpers.entity_registry.async_update_entity = AsyncMock()

    # Event helpers
    hass.helpers.event = Mock()
    hass.helpers.event.async_track_point_in_time = Mock(
        side_effect=lambda *args, **kwargs: create_cancellable_timer()
    )
    hass.helpers.event.async_track_time_interval = Mock(
        side_effect=lambda *args, **kwargs: create_cancellable_timer()
    )

    return hass


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
        CONF_DECAY_HALF_LIFE: DEFAULT_DECAY_HALF_LIFE,
        CONF_DOOR_SENSORS: [],
        CONF_WINDOW_SENSORS: [],
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
def mock_time_prior_data() -> dict[str, Any]:
    """Create mock time-based prior data for testing."""
    return {
        "hour": 14,
        "day_of_week": 2,  # Tuesday
        "prior_value": 0.35,
        "total_seconds": 3600,
        "last_updated": dt_util.utcnow().isoformat(),
    }


@pytest.fixture
def mock_historical_intervals() -> list[dict[str, Any]]:
    """Create mock historical intervals for testing."""
    base_time = dt_util.utcnow() - timedelta(days=1)
    return [
        {
            "entity_id": "binary_sensor.motion1",
            "state": "on",
            "start": base_time.isoformat(),
            "end": (base_time + timedelta(hours=2)).isoformat(),
            "duration_seconds": 7200,
        },
        {
            "entity_id": "binary_sensor.motion1",
            "state": "off",
            "start": (base_time + timedelta(hours=2)).isoformat(),
            "end": (base_time + timedelta(hours=4)).isoformat(),
            "duration_seconds": 7200,
        },
    ]


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
    registry.async_get_entity_id = Mock(return_value=None)
    registry.async_update_entity = Mock()

    # Add entities property that can be iterated
    class EntitiesContainer:
        def __init__(self):
            self._entities = {}

        def values(self):
            return []

        def items(self):
            return self._entities.items()

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
    manager.decaying_entities = []

    # Methods
    manager.get_entity = Mock(return_value=None)
    manager.add_entity = Mock()
    manager.remove_entity = Mock()
    manager.cleanup = AsyncMock()
    manager.reset_entities = AsyncMock()
    manager.async_initialize = AsyncMock()
    manager.update_likelihoods = AsyncMock(return_value=1)

    # Serialization method with comprehensive entity data
    manager.to_dict.return_value = {
        "entities": {
            "binary_sensor.test_motion": {
                "entity_id": "binary_sensor.test_motion",
                "type": {
                    "input_type": "motion",
                    "weight": 0.85,
                    "prob_true": 0.8,
                    "prob_false": 0.1,
                    "prior": 0.35,
                    "active_states": ["on"],
                    "active_range": None,
                },
                "prob_given_true": 0.8,
                "prob_given_false": 0.1,
                "decay": {
                    "is_decaying": False,
                    "decay_start_time": None,
                    "decay_start_probability": 0.0,
                    "half_life": 300,
                    "decay_enabled": True,
                    "decay_factor": 1.0,
                },
                "last_updated": dt_util.utcnow().isoformat(),
                "previous_evidence": False,
                "previous_probability": 0.35,
            }
        }
    }

    return manager


@pytest.fixture
def mock_coordinator(
    mock_hass: Mock,
    mock_realistic_config_entry: Mock,
    mock_config: Config,
    mock_entity_manager: Mock,
    mock_entity_type_manager: Mock,
    mock_area_prior: Mock,
) -> Mock:
    """Create a comprehensive mock coordinator using realistic fixtures."""

    coordinator = Mock(spec=AreaOccupancyCoordinator)
    coordinator.hass = mock_hass
    coordinator.config_entry = mock_realistic_config_entry
    coordinator.entry_id = mock_realistic_config_entry.entry_id
    coordinator.available = True
    coordinator.probability = 0.5
    coordinator.is_occupied = False
    coordinator.threshold = 0.5
    coordinator.area_prior = 0.3
    coordinator.decay = 1.0
    coordinator.occupancy_entity_id = None
    coordinator.wasp_entity_id = None
    coordinator.last_update_success = True

    # Use injected fixtures for config, entities, entity_types, prior
    coordinator.config = mock_config
    coordinator.entities = mock_entity_manager
    coordinator.entity_types = mock_entity_type_manager
    coordinator.prior = mock_area_prior

    # Database - use the new AreaOccupancyDB system
    coordinator.db = Mock()
    coordinator.db.async_save_data = AsyncMock()
    coordinator.db.async_load_data = AsyncMock(return_value=None)
    coordinator.db.async_save_area_data = AsyncMock()
    coordinator.db.async_save_entity_data = AsyncMock()
    coordinator.db.async_reset = AsyncMock()
    coordinator.db.async_get_stats = AsyncMock(return_value={})
    coordinator.db.get_time_prior = AsyncMock(return_value=None)
    coordinator.db.save_time_priors_batch = AsyncMock(return_value=0)
    coordinator.db.is_intervals_empty = Mock(return_value=True)
    coordinator.db.sync_states = AsyncMock()
    coordinator.db.load_data = AsyncMock()
    coordinator.db.save_data = AsyncMock()
    coordinator.db.save_area_data = AsyncMock()
    coordinator.db.save_entity_data = AsyncMock()
    coordinator.db.get_area_data = Mock(return_value=None)
    coordinator.db.ensure_area_exists = AsyncMock()
    coordinator.db.get_latest_interval = Mock(return_value=None)
    coordinator.db.get_engine = Mock(return_value=None)
    coordinator.db.init_db = Mock()
    coordinator.db.delete_db = Mock()
    coordinator.db.force_reinitialize = Mock()
    coordinator.db.get_db_version = Mock(return_value=3)
    coordinator.db.set_db_version = Mock()

    # Legacy store for backward compatibility
    coordinator.store = coordinator.db

    # Config manager
    coordinator.config.update_config = AsyncMock()

    # Only mock real public methods
    coordinator.async_shutdown = AsyncMock()
    coordinator.async_update_options = AsyncMock()
    coordinator.setup = AsyncMock()
    coordinator.update = AsyncMock()
    coordinator.track_entity_state_changes = AsyncMock()
    coordinator.async_refresh = AsyncMock()

    # Mock data property
    coordinator.data = {"last_updated": dt_util.utcnow().isoformat()}

    # Device info
    coordinator.device_info = {
        "identifiers": {(coordinator.config_entry.domain, coordinator.entry_id)},
        "name": coordinator.config.name,
        "manufacturer": "Area Occupancy",
        "model": "Area Occupancy Detection",
        "sw_version": "1.0.0",
    }

    # Mock timers and trackers
    coordinator._global_prior_timer = None
    coordinator._global_decay_timer = None
    coordinator._remove_state_listener = None

    # Mock binary_sensor_entity_ids property to return a proper dictionary
    coordinator.binary_sensor_entity_ids = {
        "occupancy": "binary_sensor.test_area_occupancy",
        "wasp": "binary_sensor.test_area_wasp",
    }

    # Mock entities manager with async methods for service tests
    coordinator.entities.cleanup = AsyncMock()
    coordinator.entities.reset_entities = AsyncMock()
    coordinator.entities.async_initialize = AsyncMock()
    coordinator.entities.initialize_states = AsyncMock()
    coordinator.entities.create_entity = AsyncMock()
    coordinator.entities.get_entity = Mock(
        return_value=mock_entity_manager.get_entity.return_value
    )
    coordinator.entities.add_entity = Mock()
    coordinator.entities.remove_entity = Mock()
    coordinator.entities.is_entity_active = Mock(return_value=True)
    coordinator.entities.entities = {}
    coordinator.entities.active_entities = []
    coordinator.entities.inactive_entities = []
    coordinator.entities.decaying_entities = []
    coordinator.entities.to_dict = Mock(return_value={"entities": {}})

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
    entity_type.to_dict.return_value = {
        "input_type": InputType.MOTION.value,
        "weight": 0.8,
        "prob_true": 0.25,
        "prob_false": 0.05,
        "prior": 0.35,
        "active_states": [STATE_ON],
        "active_range": None,
    }
    return entity_type


@pytest.fixture
def mock_entity_type_manager(mock_entity_type: Mock) -> Mock:
    """Create a mock EntityTypeManager."""
    from custom_components.area_occupancy.data.entity_type import (
        EntityTypeManager,
        InputType,
    )

    manager = Mock(spec=EntityTypeManager)
    manager.async_initialize = AsyncMock()
    manager.cleanup = Mock()
    manager.to_dict = Mock(
        return_value={
            "entity_types": {
                InputType.MOTION.value: mock_entity_type.to_dict.return_value
            }
        }
    )
    manager.get_entity_type = Mock(return_value=mock_entity_type)
    manager.learn_from_entities = Mock()
    # Property for entity_types
    type(manager).entity_types = property(
        lambda self: {InputType.MOTION: mock_entity_type}
    )
    return manager


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
def mock_db() -> Generator[Mock]:
    """Mock the database system."""
    with patch("custom_components.area_occupancy.db.AreaOccupancyDB") as mock_db_class:
        db_instance = Mock()
        db_instance.async_load_data = AsyncMock(return_value=None)
        db_instance.async_save_data = AsyncMock()
        db_instance.async_save_area_data = AsyncMock()
        db_instance.async_save_entity_data = AsyncMock()
        db_instance.async_reset = AsyncMock()
        db_instance.async_get_stats = AsyncMock(return_value={})
        db_instance.get_time_prior = AsyncMock(return_value=None)
        db_instance.save_time_priors_batch = AsyncMock(return_value=0)
        db_instance.is_intervals_empty = Mock(return_value=True)
        db_instance.sync_states = AsyncMock()
        db_instance.load_data = AsyncMock()
        db_instance.save_data = AsyncMock()
        db_instance.save_area_data = AsyncMock()
        db_instance.save_entity_data = AsyncMock()
        db_instance.get_area_data = Mock(return_value=None)
        db_instance.ensure_area_exists = AsyncMock()
        db_instance.get_latest_interval = Mock(return_value=None)
        db_instance.get_engine = Mock(return_value=None)
        db_instance.init_db = Mock()
        db_instance.delete_db = Mock()
        db_instance.force_reinitialize = Mock()
        db_instance.get_db_version = Mock(return_value=3)
        db_instance.set_db_version = Mock()
        mock_db_class.return_value = db_instance
        yield db_instance


@pytest.fixture
def mock_area_occupancy_db_patches():
    """Provide common patches for AreaOccupancyDB tests."""
    return [
        patch(
            "custom_components.area_occupancy.db.AreaOccupancyDB.__init__",
            return_value=None,
        ),
        patch(
            "custom_components.area_occupancy.db.AreaOccupancyDB.load_data",
            new_callable=AsyncMock,
        ),
        patch(
            "custom_components.area_occupancy.db.AreaOccupancyDB.save_data",
            new_callable=AsyncMock,
        ),
        patch(
            "custom_components.area_occupancy.db.AreaOccupancyDB.save_area_data",
            new_callable=AsyncMock,
        ),
        patch(
            "custom_components.area_occupancy.db.AreaOccupancyDB.save_entity_data",
            new_callable=AsyncMock,
        ),
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
        "evidence": True,
        "available": True,
        "type": "motion",
        "prior": {"prior": 0.3},
        "decay": {"decay_factor": 1.0},
        "last_updated": dt_util.utcnow().isoformat(),
        "last_changed": dt_util.utcnow().isoformat(),
    }


@pytest.fixture
def valid_db_data() -> dict[str, Any]:
    """Create valid database data for testing with new per-entry format."""
    return {
        "name": "Test Area",
        "probability": 0.5,
        "prior": 0.3,
        "threshold": 0.5,
        "last_updated": dt_util.utcnow().isoformat(),
        "entities": {
            "binary_sensor.test_motion": {
                "entity_id": "binary_sensor.test_motion",
                "probability": 0.5,
                "state": STATE_ON,
                "evidence": True,
                "available": True,
                "type": "motion",
                "prior": {"prior": 0.3},
                "decay": {"decay_factor": 1.0},
                "last_updated": dt_util.utcnow().isoformat(),
                "last_changed": dt_util.utcnow().isoformat(),
            }
        },
        "entity_types": {},
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


def create_db_data_with_entities(entry_id: str, entities: dict) -> dict[str, Any]:
    """Create database data with specific entities for new per-entry format."""
    return {
        "name": f"Test Area {entry_id}",
        "probability": 0.5,
        "prior": 0.3,
        "threshold": 0.5,
        "last_updated": dt_util.utcnow().isoformat(),
        "entities": entities,
        "entity_types": {},
    }


# Additional centralized fixtures for common patterns across test files


@pytest.fixture
def mock_active_entity(mock_coordinator, mock_entity_type, mock_decay) -> Mock:
    """Create a mock entity in active state (evidence=True, available=True)."""
    from custom_components.area_occupancy.data.entity import Entity

    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.active_entity"
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.coordinator = mock_coordinator
    entity.last_updated = dt_util.utcnow()
    entity.previous_evidence = False
    entity.previous_probability = 0.35

    # Properties
    entity.evidence = True
    entity.available = True
    entity.state = STATE_ON
    entity.probability = 0.75
    entity.active = True
    entity.active_states = [STATE_ON]
    entity.active_range = None
    entity.decay_factor = 1.0

    # Methods
    entity.has_new_evidence = Mock(return_value=True)
    entity.update_likelihood = Mock()
    entity.to_dict = Mock(
        return_value={
            "entity_id": "binary_sensor.active_entity",
            "type": mock_entity_type.to_dict.return_value,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "decay": mock_decay.to_dict.return_value,
            "last_updated": dt_util.utcnow().isoformat(),
            "previous_evidence": False,
            "previous_probability": 0.35,
        }
    )

    return entity


@pytest.fixture
def mock_inactive_entity(mock_coordinator, mock_entity_type, mock_decay) -> Mock:
    """Create a mock entity in inactive state (evidence=False, available=True)."""
    from custom_components.area_occupancy.data.entity import Entity

    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.inactive_entity"
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.coordinator = mock_coordinator
    entity.last_updated = dt_util.utcnow()
    entity.previous_evidence = True
    entity.previous_probability = 0.75

    # Properties
    entity.evidence = False
    entity.available = True
    entity.state = STATE_OFF
    entity.probability = 0.25
    entity.active = True  # Because decay is running
    entity.active_states = [STATE_ON]
    entity.active_range = None
    entity.decay_factor = 0.8

    # Methods
    entity.has_new_evidence = Mock(return_value=True)
    entity.update_likelihood = Mock()
    entity.to_dict = Mock(
        return_value={
            "entity_id": "binary_sensor.inactive_entity",
            "type": mock_entity_type.to_dict.return_value,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "decay": mock_decay.to_dict.return_value,
            "last_updated": dt_util.utcnow().isoformat(),
            "previous_evidence": True,
            "previous_probability": 0.75,
        }
    )

    return entity


@pytest.fixture
def mock_unavailable_entity(mock_coordinator, mock_entity_type, mock_decay) -> Mock:
    """Create a mock entity in unavailable state (available=False)."""
    from custom_components.area_occupancy.data.entity import Entity

    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.unavailable_entity"
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.coordinator = mock_coordinator
    entity.last_updated = None
    entity.previous_evidence = None
    entity.previous_probability = 0.15

    # Properties
    entity.evidence = None
    entity.available = False
    entity.state = None
    entity.probability = 0.15
    entity.active = False
    entity.active_states = [STATE_ON]
    entity.active_range = None
    entity.decay_factor = 1.0

    # Methods
    entity.has_new_evidence = Mock(return_value=False)
    entity.update_likelihood = Mock()
    entity.to_dict = Mock(
        return_value={
            "entity_id": "binary_sensor.unavailable_entity",
            "type": mock_entity_type.to_dict.return_value,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "decay": mock_decay.to_dict.return_value,
            "last_updated": None,
            "previous_evidence": None,
            "previous_probability": 0.15,
        }
    )

    return entity


@pytest.fixture
def mock_stale_entity(mock_coordinator, mock_entity_type, mock_decay) -> Mock:
    """Create a mock entity with stale update (> 1 hour ago)."""
    from custom_components.area_occupancy.data.entity import Entity

    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.stale_entity"
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.coordinator = mock_coordinator
    # Mock stale update (more than 1 hour ago)
    entity.last_updated = dt_util.utcnow() - timedelta(hours=2)
    entity.previous_evidence = False
    entity.previous_probability = 0.30

    # Properties
    entity.evidence = False
    entity.available = True
    entity.state = STATE_OFF
    entity.probability = 0.30
    entity.active = False
    entity.active_states = [STATE_ON]
    entity.active_range = None
    entity.decay_factor = 1.0

    # Methods
    entity.has_new_evidence = Mock(return_value=False)
    entity.update_likelihood = Mock()
    entity.to_dict = Mock(
        return_value={
            "entity_id": "binary_sensor.stale_entity",
            "type": mock_entity_type.to_dict.return_value,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "decay": mock_decay.to_dict.return_value,
            "last_updated": (dt_util.utcnow() - timedelta(hours=2)).isoformat(),
            "previous_evidence": False,
            "previous_probability": 0.30,
        }
    )

    return entity


@pytest.fixture
def mock_last_updated() -> Mock:
    """Create a mock last_updated object with isoformat method."""
    mock_timestamp = Mock()
    mock_timestamp.isoformat.return_value = "2024-01-01T00:00:00"
    return mock_timestamp


@pytest.fixture
def mock_motion_entity_type() -> Mock:
    """Create a mock motion entity type with typical motion sensor properties."""
    entity_type = Mock()
    entity_type.prior = 0.3
    entity_type.prob_true = 0.8
    entity_type.prob_false = 0.2
    entity_type.weight = 1.0
    entity_type.active_states = ["on"]
    entity_type.active_range = None
    entity_type.input_type = Mock()
    entity_type.input_type.value = "motion"
    return entity_type


@pytest.fixture
def mock_entity_manager_with_states(
    mock_active_entity: Mock,
    mock_inactive_entity: Mock,
    mock_unavailable_entity: Mock,
    mock_stale_entity: Mock,
) -> Mock:
    """Create a mock entity manager with entities in different states."""
    manager = Mock()
    manager.entities = {
        "binary_sensor.active_entity": mock_active_entity,
        "binary_sensor.inactive_entity": mock_inactive_entity,
        "binary_sensor.unavailable_entity": mock_unavailable_entity,
        "binary_sensor.stale_entity": mock_stale_entity,
    }
    manager.get_entity = Mock(return_value=mock_active_entity)
    return manager


@pytest.fixture
def mock_empty_entity_manager() -> Mock:
    """Create a mock entity manager with no entities."""
    manager = Mock()
    manager.entities = {}
    manager.get_entity = Mock(side_effect=ValueError("Entity not found"))
    return manager


@pytest.fixture
def mock_entities_container() -> Mock:
    """Create a mock entities container that can be used for coordinator.entities attribute."""
    container = Mock()
    container.entities = {}
    container.get_entity = Mock(side_effect=ValueError("Entity not found"))
    container.reset_entities = AsyncMock()
    return container


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
    mock_coordinator.area_prior = 0.35
    mock_coordinator.probability = 0.65
    mock_coordinator.decay = 0.8

    # Mock entity manager with comprehensive entities
    mock_coordinator.entities.entities = {
        "binary_sensor.motion1": Mock(
            entity_id="binary_sensor.motion1",
            available=True,
            evidence=True,
            probability=0.75,
            type=Mock(input_type=InputType.MOTION, weight=0.85),
            decay=Mock(is_decaying=False, decay_factor=1.0),
            prob_given_true=0.8,
            prob_given_false=0.1,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=False,
            previous_probability=0.35,
            active=True,
            active_states=[STATE_ON],
            active_range=None,
            decay_factor=1.0,
        ),
        "binary_sensor.motion2": Mock(
            entity_id="binary_sensor.motion2",
            available=True,
            evidence=False,
            probability=0.25,
            type=Mock(input_type=InputType.MOTION, weight=0.85),
            decay=Mock(is_decaying=True, decay_factor=0.8),
            prob_given_true=0.8,
            prob_given_false=0.1,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=True,
            previous_probability=0.75,
            active=True,
            active_states=[STATE_ON],
            active_range=None,
            decay_factor=0.8,
        ),
        "binary_sensor.appliance": Mock(
            entity_id="binary_sensor.appliance",
            available=True,
            evidence=False,
            probability=0.15,
            type=Mock(input_type=InputType.APPLIANCE, weight=0.3),
            decay=Mock(is_decaying=False, decay_factor=1.0),
            prob_given_true=0.6,
            prob_given_false=0.05,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=False,
            previous_probability=0.15,
            active=False,
            active_states=["on", "standby"],
            active_range=None,
            decay_factor=1.0,
        ),
        "media_player.tv": Mock(
            entity_id="media_player.tv",
            available=True,
            evidence=True,
            probability=0.85,
            type=Mock(input_type=InputType.MEDIA, weight=0.7),
            decay=Mock(is_decaying=False, decay_factor=1.0),
            prob_given_true=0.8,
            prob_given_false=0.1,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=False,
            previous_probability=0.35,
            active=True,
            active_states=["playing", "paused"],
            active_range=None,
            decay_factor=1.0,
        ),
    }

    # Set up decaying entities based on is_decaying status
    decaying_entities = [
        entity
        for entity in mock_coordinator.entities.entities.values()
        if entity.decay.is_decaying
    ]
    mock_coordinator.entities.decaying_entities = decaying_entities
    mock_coordinator.decaying_entities = decaying_entities  # Add both for compatibility

    return mock_coordinator


@pytest.fixture
def mock_prior() -> Mock:
    """Create a mock Prior instance for backward compatibility with tests."""
    prior = Mock()
    prior.value = 0.35
    prior._current_value = 0.35
    prior.last_updated = dt_util.utcnow()
    prior.update = AsyncMock(return_value=0.35)
    prior.calculate = AsyncMock(return_value=0.35)
    prior.to_dict.return_value = {
        "value": 0.35,
        "last_updated": prior.last_updated.isoformat(),
        "sensor_hash": None,
    }
    return prior


@pytest.fixture
def mock_area_prior() -> Mock:
    """Create a mock Prior instance for area-level prior."""
    prior = Mock(spec=PriorClass)
    prior._current_value = 0.3
    prior.value = 0.3
    prior.global_prior = 0.3
    prior.occupancy_prior = 0.25
    prior.primary_sensors_prior = 0.3
    prior.sensor_ids = ["binary_sensor.motion1", "binary_sensor.motion2"]
    prior.last_updated = dt_util.utcnow()
    prior.update = AsyncMock(return_value=0.3)
    prior.calculate = AsyncMock(return_value=0.3)
    prior.prior_intervals = []
    prior.prior_total_seconds = 0

    # Add time-based prior properties
    prior.time_prior_value = 0.25
    prior.time_prior_last_updated = dt_util.utcnow()
    prior.time_prior_intervals = []
    prior.time_prior_total_seconds = 0

    # Add methods for time-based priors
    prior.update_time_prior = AsyncMock(return_value=0.25)
    prior.calculate_time_prior = AsyncMock(return_value=0.25)
    prior.get_time_prior = AsyncMock(return_value=0.25)

    prior.to_dict.return_value = {
        "value": 0.3,
        "last_updated": prior.last_updated.isoformat(),
        "sensor_hash": None,
        "time_prior_value": 0.25,
        "time_prior_last_updated": prior.time_prior_last_updated.isoformat(),
    }
    return prior


@pytest.fixture
def mock_decay() -> Mock:
    """Create a mock Decay instance matching the real Decay class."""
    from custom_components.area_occupancy.data.decay import Decay

    decay = Mock(spec=Decay)
    decay.is_decaying = False
    decay.last_trigger_ts = time.time()
    decay.half_life = 60.0
    # Use PropertyMock for the decay_factor property
    type(decay).decay_factor = PropertyMock(return_value=1.0)

    # Add side effect for start_decay to properly simulate behavior
    def start_decay_side_effect():
        if not decay.is_decaying:
            decay.is_decaying = True
            decay.last_trigger_ts = time.time()

    decay.start_decay.side_effect = start_decay_side_effect

    decay.to_dict.return_value = {
        "last_trigger_ts": time.time(),
        "half_life": 60.0,
        "is_decaying": False,
    }
    return decay


@pytest.fixture
def mock_service_call() -> Mock:
    """Create a mock service call with common attributes."""

    call = Mock(spec=ServiceCall)
    call.data = {"entry_id": "test_entry_id"}
    call.return_response = True
    return call


@pytest.fixture
def mock_service_call_with_entity() -> Mock:
    """Create a mock service call with entity_id."""

    call = Mock(spec=ServiceCall)
    call.data = {"entry_id": "test_entry_id", "entity_id": "binary_sensor.test_motion"}
    call.return_response = True
    return call


@pytest.fixture
def mock_comprehensive_entity(
    mock_coordinator: Mock, mock_entity_type: Mock, mock_decay: Mock
) -> Mock:
    """Create a comprehensive mock entity with all components."""
    from custom_components.area_occupancy.data.entity import Entity

    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.test_motion"
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.coordinator = mock_coordinator
    entity.last_updated = dt_util.utcnow()
    entity.previous_evidence = False
    entity.previous_probability = 0.5

    # Properties
    entity.probability = 0.5
    entity.state = STATE_ON
    entity.evidence = True
    entity.available = True
    entity.active = True
    entity.active_states = [STATE_ON]
    entity.active_range = None
    entity.decay_factor = 1.0

    # Methods
    entity.has_new_evidence = Mock(return_value=True)
    entity.update_likelihood = Mock()
    entity.stop_decay_completely = Mock()
    entity.cleanup = Mock()
    entity.to_dict.return_value = {
        "entity_id": "binary_sensor.test_motion",
        "type": mock_entity_type.to_dict.return_value,
        "prob_given_true": 0.8,
        "prob_given_false": 0.1,
        "decay": mock_decay.to_dict.return_value,
        "last_updated": dt_util.utcnow().isoformat(),
        "previous_evidence": False,
        "previous_probability": 0.5,
    }
    return entity


@pytest.fixture
def mock_comprehensive_entity_manager(
    mock_coordinator: Mock, mock_comprehensive_entity: Mock
) -> Mock:
    """Create a comprehensive mock entity manager with entities."""
    from custom_components.area_occupancy.data.entity import EntityManager

    manager = Mock(spec=EntityManager)
    manager.coordinator = mock_coordinator
    manager.config = mock_coordinator.config
    manager.hass = mock_coordinator.hass
    manager._entities = {"binary_sensor.test_motion": mock_comprehensive_entity}
    manager.entities = {"binary_sensor.test_motion": mock_comprehensive_entity}
    manager.entity_ids = ["binary_sensor.test_motion"]
    manager.active_entities = [mock_comprehensive_entity]
    manager.inactive_entities = []
    manager.decaying_entities = []
    manager.to_dict = Mock(
        return_value={
            "entities": {
                "binary_sensor.test_motion": mock_comprehensive_entity.to_dict.return_value
            }
        }
    )
    manager.async_initialize = AsyncMock()
    manager.reset_entities = AsyncMock()
    manager.create_entity = AsyncMock(return_value=mock_comprehensive_entity)
    manager.get_entity = Mock(return_value=mock_comprehensive_entity)
    manager.add_entity = Mock()
    manager.remove_entity = Mock()
    manager.cleanup = AsyncMock()
    manager.is_entity_active = Mock(return_value=True)
    manager.initialize_states = AsyncMock()
    manager.update_likelihoods = AsyncMock(return_value=1)
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
    with patch("homeassistant.helpers.recorder.get_instance") as mock_get_instance_ha:
        mock_instance = Mock()
        mock_instance.async_add_executor_job = AsyncMock(return_value={})
        mock_get_instance_ha.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=True)
def mock_significant_states_globally():
    """Automatically mock significant states for all tests."""
    with patch(
        "homeassistant.components.recorder.history.get_significant_states"
    ) as mock_states:
        mock_states.return_value = {}
        yield mock_states


@pytest.fixture(autouse=True)
def mock_track_point_in_time_globally():
    """Automatically mock timer-related functions for all tests."""

    class CancellableTimerMock:
        """Mock timer that properly handles cleanup verification."""

        def __init__(self, *args, **kwargs):
            self._cancelled = True
            self._args = args
            self._callback = args[1] if len(args) > 1 else None

        def cancel(self):
            self._cancelled = True

        def cancelled(self):
            return True

        def __repr__(self):
            return f"<MockTimerHandle cancelled={self._cancelled}>"

    def create_timer_mock(*args, **kwargs):
        return CancellableTimerMock(*args, **kwargs)

    # Mock both high-level helpers and low-level event loop methods
    with (
        patch(
            "homeassistant.helpers.event.async_track_point_in_time", create_timer_mock
        ),
        patch(
            "homeassistant.helpers.event.async_track_time_interval", create_timer_mock
        ),
        patch("homeassistant.helpers.event.async_call_later", create_timer_mock),
        patch.object(asyncio.AbstractEventLoop, "call_later", create_timer_mock),
        patch.object(asyncio.AbstractEventLoop, "call_at", create_timer_mock),
    ):
        yield


@pytest.fixture
def mock_entity_for_likelihood_tests(
    mock_coordinator, mock_entity_type, mock_decay
) -> Mock:
    """Create a mock entity specifically for likelihood calculation tests."""
    from custom_components.area_occupancy.data.entity import Entity

    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.motion_sensor_1"
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.coordinator = mock_coordinator
    entity.last_updated = dt_util.utcnow()
    entity.previous_evidence = False
    entity.previous_probability = 0.35

    # Properties
    entity.evidence = True
    entity.available = True
    entity.state = STATE_ON
    entity.probability = 0.75
    entity.active = True
    entity.active_states = [STATE_ON]
    entity.active_range = None
    entity.decay_factor = 1.0

    # Methods
    entity.has_new_evidence = Mock(return_value=True)
    entity.update_likelihood = Mock()
    entity.to_dict = Mock(
        return_value={
            "entity_id": "binary_sensor.motion_sensor_1",
            "type": mock_entity_type.to_dict.return_value,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "decay": mock_decay.to_dict.return_value,
            "last_updated": dt_util.utcnow().isoformat(),
            "previous_evidence": False,
            "previous_probability": 0.35,
        }
    )

    return entity


@pytest.fixture(autouse=True)
def mock_area_occupancy_db_globally(request):
    """Automatically mock AreaOccupancyDB for all tests except database tests."""
    # Skip mocking for database tests. Checking request.cls is more reliable
    # than matching the node string only.
    cls_name = getattr(getattr(request.node, "cls", None), "__name__", "")
    if "TestAreaOccupancyDB" in str(request.node) or cls_name == "TestAreaOccupancyDB":
        yield None
        return

    with patch("custom_components.area_occupancy.db.AreaOccupancyDB") as mock_db_class:
        mock_db = Mock()
        mock_db.async_save_data = AsyncMock()
        mock_db.async_load_data = AsyncMock(return_value=None)
        mock_db.async_save_area_data = AsyncMock()
        mock_db.async_save_entity_data = AsyncMock()
        mock_db.async_reset = AsyncMock()
        mock_db.async_get_stats = AsyncMock(return_value={})
        mock_db.is_intervals_empty = Mock(return_value=True)
        mock_db.sync_states = AsyncMock()
        mock_db.load_data = AsyncMock()
        mock_db.save_data = AsyncMock()
        mock_db.save_area_data = AsyncMock()
        mock_db.save_entity_data = AsyncMock()
        mock_db.get_area_data = Mock(return_value=None)
        mock_db.ensure_area_exists = AsyncMock()
        mock_db.get_latest_interval = Mock(return_value=None)
        mock_db.get_engine = Mock(return_value=None)
        mock_db.init_db = Mock()
        mock_db.delete_db = Mock()
        mock_db.force_reinitialize = Mock()
        mock_db.get_db_version = Mock(return_value=3)
        mock_db.set_db_version = Mock()
        mock_db_class.return_value = mock_db
        yield mock_db


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


@pytest.fixture
def mock_area_occupancy_db_data():
    """Return a representative AreaOccupancyDB data dict for testing."""
    return {
        "name": "Testing",
        "probability": 0.18,
        "area_prior": 0.18,
        "threshold": 0.52,
        "last_updated": "2025-06-19T14:29:30.273647+00:00",
        "entities": {
            "binary_sensor.motion_sensor_1": {
                "entity_id": "binary_sensor.motion_sensor_1",
                "type": {
                    "input_type": "motion",
                    "weight": 0.85,
                    "prob_true": 0.25,
                    "prob_false": 0.05,
                    "prior": 0.35,
                    "active_states": ["on"],
                    "active_range": None,
                },
                "prob_given_true": 0.95,
                "prob_given_false": 0.02,
                "decay": {
                    "last_trigger_ts": 1750328374.235739,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
            "media_player.tv_player": {
                "entity_id": "media_player.tv_player",
                "type": {
                    "input_type": "media",
                    "weight": 0.7,
                    "prob_true": 0.25,
                    "prob_false": 0.02,
                    "prior": 0.3,
                    "active_states": ["playing", "paused"],
                    "active_range": None,
                },
                "prob_given_true": 0.11,
                "prob_given_false": 0.001,
                "decay": {
                    "last_trigger_ts": 1750328374.235851,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
            "binary_sensor.computer_power_sensor": {
                "entity_id": "binary_sensor.computer_power_sensor",
                "probability": 0.01,
                "type": {
                    "input_type": "appliance",
                    "weight": 0.3,
                    "prob_true": 0.2,
                    "prob_false": 0.02,
                    "prior": 0.23,
                    "active_states": ["on", "standby"],
                    "active_range": None,
                },
                "prob_given_true": 0.05,
                "prob_given_false": 0.001,
                "decay": {
                    "last_trigger_ts": 1750328374.235891,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
            "binary_sensor.door_sensor": {
                "entity_id": "binary_sensor.door_sensor",
                "probability": 0.01,
                "type": {
                    "input_type": "door",
                    "weight": 0.3,
                    "prob_true": 0.2,
                    "prob_false": 0.02,
                    "prior": 0.13,
                    "active_states": ["closed"],
                    "active_range": None,
                },
                "prob_given_true": 0.02,
                "prob_given_false": 0.001,
                "decay": {
                    "last_trigger_ts": 1750328374.235931,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
            "binary_sensor.window_sensor": {
                "entity_id": "binary_sensor.window_sensor",
                "probability": 0.01,
                "type": {
                    "input_type": "window",
                    "weight": 0.2,
                    "prob_true": 0.2,
                    "prob_false": 0.02,
                    "prior": 0.15,
                    "active_states": ["open"],
                    "active_range": None,
                },
                "prob_given_true": 0.01,
                "prob_given_false": 0.001,
                "decay": {
                    "last_trigger_ts": 1750328374.23595,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
            "sensor.illuminance_sensor_1": {
                "entity_id": "sensor.illuminance_sensor_1",
                "probability": 0.01,
                "type": {
                    "input_type": "environmental",
                    "weight": 0.1,
                    "prob_true": 0.09,
                    "prob_false": 0.01,
                    "prior": 0.07,
                    "active_states": None,
                    "active_range": [0.0, 0.2],
                },
                "prob_given_true": 0.001,
                "prob_given_false": 0.001,
                "decay": {
                    "last_trigger_ts": 1750328374.235983,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
            "sensor.humidity_sensor": {
                "entity_id": "sensor.humidity_sensor",
                "probability": 0.01,
                "type": {
                    "input_type": "environmental",
                    "weight": 0.1,
                    "prob_true": 0.09,
                    "prob_false": 0.01,
                    "prior": 0.07,
                    "active_states": None,
                    "active_range": [0.0, 0.2],
                },
                "prob_given_true": 0.001,
                "prob_given_false": 0.001,
                "decay": {
                    "last_trigger_ts": 1750328374.236049,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
            "sensor.temperature_sensor": {
                "entity_id": "sensor.temperature_sensor",
                "probability": 0.01,
                "type": {
                    "input_type": "environmental",
                    "weight": 0.1,
                    "prob_true": 0.09,
                    "prob_false": 0.01,
                    "prior": 0.07,
                    "active_states": None,
                    "active_range": [0.0, 0.2],
                },
                "prob_given_true": 0.001,
                "prob_given_false": 0.001,
                "decay": {
                    "last_trigger_ts": 1750328374.236094,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
            "binary_sensor.testing_wasp_in_box": {
                "entity_id": "binary_sensor.testing_wasp_in_box",
                "probability": 0.01,
                "type": {
                    "input_type": "motion",
                    "weight": 0.85,
                    "prob_true": 0.73,
                    "prob_false": 0.005,
                    "prior": 0.73,
                    "active_states": ["on"],
                    "active_range": None,
                },
                "prob_given_true": 0.02,
                "prob_given_false": 0.001,
                "decay": {
                    "last_trigger_ts": 1750331176.238107,
                    "half_life": 300,
                    "is_decaying": False,
                },
            },
        },
        "prior": {
            "value": 0.18,
            "last_updated": "2025-06-19T12:06:15.123456+00:00",
            "sensor_hash": 123456789,
        },
    }


@pytest.fixture
def mock_config():
    """Return a representative Config instance for testing."""
    # Create a mock config that works with the new Config class structure
    config = Mock()

    # Set basic attributes
    config.name = "Test Area"
    config.purpose = "living"
    config.area_id = "area_123"
    config.threshold = 0.5

    # Create sensor configurations
    config.sensors = Sensors(
        motion=["binary_sensor.motion1"],
        primary_occupancy="binary_sensor.motion1",
        media=["media_player.tv"],
        appliance=["switch.computer"],
        illuminance=["sensor.illuminance_sensor_1"],
        humidity=["sensor.humidity_sensor"],
        temperature=["sensor.temperature_sensor"],
        door=["binary_sensor.door_sensor"],
        window=["binary_sensor.window_sensor"],
    )

    # Create sensor states
    config.sensor_states = SensorStates(
        door=["closed"],
        window=["open"],
        appliance=["on", "standby"],
        media=["playing", "paused"],
    )

    # Create weights
    config.weights = Weights(
        motion=0.9,
        media=0.7,
        appliance=0.6,
        door=0.5,
        window=0.4,
        environmental=0.3,
        wasp=0.8,
    )

    # Create decay configuration
    config.decay = Decay(enabled=True, half_life=300)

    # Create wasp configuration
    config.wasp_in_box = WaspInBox(
        enabled=False, motion_timeout=60, weight=0.8, max_duration=600
    )

    # Add properties that might be accessed
    config.start_time = dt_util.utcnow() - timedelta(days=HA_RECORDER_DAYS)
    config.end_time = dt_util.utcnow()

    # Add methods that might be called
    config.update_config = AsyncMock()
    config.validate_entity_configuration = Mock(return_value=[])
    config.get = Mock(
        side_effect=lambda key, default=None: getattr(config, key, default)
    )

    return config


@pytest.fixture
def mock_realistic_config_entry():
    """Return a realistic ConfigEntry for Area Occupancy Detection."""
    entry = Mock(spec=ConfigEntry)
    entry.entry_id = "01JQRDH37YHVXR3X4FMDYTHQD8"
    entry.domain = "area_occupancy"
    entry.title = "Testing"
    entry.source = "user"
    entry.version = 9
    entry.minor_version = 2
    entry.unique_id = None
    entry.state = None
    entry.runtime_data = None
    entry.pref_disable_new_entities = False
    entry.pref_disable_polling = False
    entry.subentries = []
    entry.disabled_by = None
    entry.discovery_keys = {}
    entry.created_at = "2025-04-01T10:14:38.590998+00:00"
    entry.modified_at = "2025-06-19T07:10:40.167187+00:00"
    entry.data = {
        "appliance_active_states": ["on", "standby"],
        "appliances": [
            "binary_sensor.computer_power_sensor",
            "binary_sensor.game_console_power_sensor",
            "binary_sensor.tv_power_sensor",
        ],
        "decay_enabled": True,
        "decay_half_life": 600.0,
        "door_active_state": "open",
        "door_sensors": ["binary_sensor.door_sensor"],
        "humidity_sensors": ["sensor.humidity_sensor_1", "sensor.humidity_sensor_2"],
        "illuminance_sensors": [
            "sensor.illuminance_sensor_1",
            "sensor.illuminance_sensor_2",
        ],
        "media_active_states": ["playing", "paused"],
        "media_devices": ["media_player.mock_tv_player"],
        "motion_sensors": [
            "binary_sensor.motion_sensor_1",
            "binary_sensor.motion_sensor_2",
            "binary_sensor.motion_sensor_3",
        ],
        "name": "Testing",
        "primary_occupancy_sensor": "binary_sensor.motion_sensor_1",
        "temperature_sensors": [
            "sensor.temperature_sensor_1",
            "sensor.temperature_sensor_2",
        ],
        "threshold": 50.0,
        "weight_appliance": 0.3,
        "weight_door": 0.3,
        "weight_environmental": 0.1,
        "weight_media": 0.7,
        "weight_motion": 0.85,
        "weight_window": 0.2,
        "window_active_state": "open",
        "window_sensors": ["binary_sensor.window_sensor"],
    }
    entry.options = {
        "appliance_active_states": ["on", "standby"],
        "appliances": [
            "binary_sensor.computer_power_sensor",
            "binary_sensor.game_console_power_sensor",
        ],
        "decay_enabled": True,
        "decay_half_life": 300.0,
        "door_active_state": "closed",
        "door_sensors": ["binary_sensor.door_sensor"],
        "humidity_sensors": ["sensor.humidity_sensor_1", "sensor.humidity_sensor_2"],
        "illuminance_sensors": [
            "sensor.illuminance_sensor_1",
            "sensor.illuminance_sensor_2",
        ],
        "media_active_states": ["playing", "paused"],
        "media_devices": ["media_player.mock_tv_player"],
        "motion_sensors": [
            "binary_sensor.motion_sensor_1",
            "binary_sensor.motion_sensor_2",
            "binary_sensor.motion_sensor_3",
        ],
        "primary_occupancy_sensor": "binary_sensor.motion_sensor_1",
        "temperature_sensors": [
            "sensor.temperature_sensor_1",
            "sensor.temperature_sensor_2",
        ],
        "threshold": 52.0,
        "wasp_enabled": True,
        "wasp_motion_timeout": 60.0,
        "wasp_weight": 0.8,
        "weight_appliance": 0.3,
        "weight_door": 0.3,
        "weight_environmental": 0.1,
        "weight_media": 0.7,
        "weight_motion": 0.85,
        "weight_window": 0.2,
        "window_active_state": "open",
        "window_sensors": ["binary_sensor.window_sensor"],
    }
    entry.add_update_listener = Mock()
    entry.async_on_unload = Mock()
    entry.async_setup = AsyncMock()
    entry.async_unload = AsyncMock()
    entry.async_remove = AsyncMock()
    entry.async_update = AsyncMock()
    return entry


# Global patch for custom_components.area_occupancy.utils.get_instance


@pytest.fixture(autouse=True)
def auto_cancel_timers(monkeypatch):
    """Automatically track and cancel all timers created during a test."""
    loop = asyncio.get_event_loop()
    original_call_later = loop.call_later
    original_call_at = loop.call_at
    timer_handles = []

    def tracking_call_later(delay, callback, *args, **kwargs):
        handle = original_call_later(delay, callback, *args, **kwargs)
        timer_handles.append(handle)
        return handle

    def tracking_call_at(when, callback, *args, **kwargs):
        handle = original_call_at(when, callback, *args, **kwargs)
        timer_handles.append(handle)
        return handle

    monkeypatch.setattr(loop, "call_later", tracking_call_later)
    monkeypatch.setattr(loop, "call_at", tracking_call_at)

    # Patch async_track_point_in_time if used directly
    try:
        from homeassistant.helpers.event import async_track_point_in_time

        orig_async_track_point_in_time = async_track_point_in_time

        def tracking_async_track_point_in_time(hass, action, point_in_time):
            handle = orig_async_track_point_in_time(hass, action, point_in_time)
            timer_handles.append(handle)
            return handle

        monkeypatch.setattr(
            "homeassistant.helpers.event.async_track_point_in_time",
            tracking_async_track_point_in_time,
        )
    except ImportError:
        pass

    yield

    for handle in timer_handles:
        with contextlib.suppress(Exception):
            handle.cancel()


# SQLAlchemy Database Testing Fixtures
# Following best practices for in-memory SQLite testing with proper isolation


@pytest.fixture
def db_engine():
    """Create an in-memory SQLite engine for testing."""
    from sqlalchemy import create_engine

    from custom_components.area_occupancy.db import Base

    # Create in-memory SQLite engine
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    # Clean up - drop all tables
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    """Create a database session for testing with automatic rollback."""
    from sqlalchemy.orm import sessionmaker

    # Create session factory bound to the test engine
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        # Rollback any uncommitted changes and close session
        session.rollback()
        session.close()


@pytest.fixture
def transactional_db_session(db_engine):
    """Create a database session with nested transaction for maximum isolation."""
    from sqlalchemy.orm import sessionmaker

    # Create connection and start transaction
    connection = db_engine.connect()
    trans = connection.begin()

    # Create session bound to the connection
    Session = sessionmaker(bind=connection)
    session = Session()

    # Start nested SAVEPOINT for maximum isolation
    session.begin_nested()

    try:
        yield session
    finally:
        # Rollback nested transaction, then outer transaction
        session.rollback()
        trans.rollback()
        connection.close()


@pytest.fixture
def mock_area_occupancy_db(db_engine, db_session, tmp_path):
    """Create a mock AreaOccupancyDB instance using in-memory database."""
    from unittest.mock import Mock

    from custom_components.area_occupancy.db import AreaOccupancyDB

    # Create mock coordinator with config_dir using tmp_path
    mock_coordinator = Mock()
    mock_coordinator.hass = Mock()
    mock_coordinator.hass.config = Mock()
    mock_coordinator.hass.config.config_dir = str(tmp_path)
    mock_coordinator.config_entry = Mock()
    mock_coordinator.config_entry.data = {"version": 9}

    # Create the .storage directory that AreaOccupancyDB expects
    storage_dir = tmp_path / ".storage"
    storage_dir.mkdir(exist_ok=True)

    # Create AreaOccupancyDB instance but override the engine
    db = AreaOccupancyDB(coordinator=mock_coordinator)

    # Replace the engine with our test engine
    db.engine = db_engine
    db.session = db_session

    return db


@pytest.fixture
def seeded_db_session(db_session):
    """Create a database session with test data pre-loaded."""
    # Instead of creating real ORM objects, just return the session
    # The tests that use this fixture should mock the database operations
    # to avoid conflicts with the mock setup
    return db_session


@pytest.fixture
def mock_db_with_engine(mock_hass, db_engine, tmp_path):
    """Create AreaOccupancyDB instance with in-memory database."""
    from unittest.mock import Mock

    from sqlalchemy.orm import sessionmaker

    from custom_components.area_occupancy.db import AreaOccupancyDB

    # Create mock coordinator
    mock_coordinator = Mock()
    mock_coordinator.hass = mock_hass
    mock_coordinator.entry_id = "test_entry_001"
    mock_coordinator.config_entry = Mock()
    mock_coordinator.config_entry.data = {"version": 9}

    # Create real database instance
    db = AreaOccupancyDB(coordinator=mock_coordinator)

    # Override the database with our test database
    db.engine = db_engine

    # Create a fresh session for each test
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()

    # Clear any existing data
    try:
        session.query(db.Intervals).delete()
        session.query(db.Entities).delete()
        session.query(db.Areas).delete()
        session.query(db.Priors).delete()
        session.commit()
    except (ValueError, OSError):
        session.rollback()

    db.session = session

    return db


@pytest.fixture
def sample_area_data():
    """Provide sample area data for testing."""
    from datetime import datetime

    return {
        "entry_id": "test_entry_001",
        "area_name": "Test Living Room",
        "purpose": "living",
        "threshold": 0.5,
        "area_prior": 0.3,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


@pytest.fixture
def sample_entity_data():
    """Provide sample entity data for testing."""
    from datetime import datetime

    return {
        "entry_id": "test_entry_001",
        "entity_id": "binary_sensor.motion_1",
        "entity_type": "motion",
        "weight": 0.85,
        "prob_given_true": 0.8,
        "prob_given_false": 0.1,
        "last_updated": datetime.now(),
        "created_at": datetime.now(),
    }


@pytest.fixture
def sample_interval_data():
    """Provide sample interval data for testing."""
    from datetime import datetime, timedelta

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=1)

    return {
        "entity_id": "binary_sensor.motion_1",
        "state": "on",
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": 3600.0,
        "created_at": datetime.now(),
    }


@pytest.fixture
def sample_prior_data():
    """Provide sample prior data for testing."""
    from datetime import datetime

    return {
        "entry_id": "test_entry_001",
        "day_of_week": 1,  # Monday
        "time_slot": 14,  # 2 PM
        "prior_value": 0.35,
        "data_points": 10,
        "last_updated": datetime.now(),
    }
