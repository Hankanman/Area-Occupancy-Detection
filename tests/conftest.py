"""Fixtures for Area Occupancy Detection integration tests."""

import asyncio
import os
import sys
from collections.abc import Generator
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.recorder.const import DOMAIN as RECORDER_DOMAIN
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.recorder import DATA_INSTANCE
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.area_occupancy.const import (  # noqa: TID252
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
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
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
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
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
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
from custom_components.area_occupancy.probabilities import Probabilities
from custom_components.area_occupancy.types import (
    EntityType,
    PriorState,
    ProbabilityConfig,
    SensorInputs,
)

# Make parent directory available to tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest_plugins = "pytest_homeassistant_custom_component"

# Test data
TEST_CONFIG = {
    CONF_NAME: "Test Area",
    CONF_THRESHOLD: 50,
    CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
    CONF_MEDIA_DEVICES: [],
    CONF_APPLIANCES: [],
    CONF_ILLUMINANCE_SENSORS: [],
    CONF_HUMIDITY_SENSORS: [],
    CONF_TEMPERATURE_SENSORS: [],
    CONF_DOOR_SENSORS: [],
    CONF_WINDOW_SENSORS: [],
    CONF_LIGHTS: [],
    CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
    CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
    CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
    CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
    CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
    CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
    CONF_DOOR_ACTIVE_STATE: DEFAULT_DOOR_ACTIVE_STATE,
    CONF_WINDOW_ACTIVE_STATE: DEFAULT_WINDOW_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES: DEFAULT_MEDIA_ACTIVE_STATES,
    CONF_APPLIANCE_ACTIVE_STATES: DEFAULT_APPLIANCE_ACTIVE_STATES,
    CONF_HISTORY_PERIOD: DEFAULT_HISTORY_PERIOD,
    CONF_DECAY_ENABLED: DEFAULT_DECAY_ENABLED,
    CONF_DECAY_WINDOW: DEFAULT_DECAY_WINDOW,
    CONF_HISTORICAL_ANALYSIS_ENABLED: DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    CONF_DECAY_MIN_DELAY: DEFAULT_DECAY_MIN_DELAY,
}

# Common test constants
TEST_ENTRY_ID = "test_entry_id"
TEST_UNIQUE_ID = "uniqueid123"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations,
) -> Generator[None]:
    """Enable custom integrations for testing."""
    return  # type: ignore


@pytest.fixture
def mock_hass() -> Mock:
    """Return a Mock HomeAssistant instance with minimal config for testing."""
    mock = Mock(spec=HomeAssistant)
    mock.data = {DOMAIN: {}}
    mock_config = Mock()
    mock_config.config_dir = "/tmp"  # noqa: S108
    mock.config = mock_config
    return mock


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

    with (
        patch(
            "homeassistant.components.recorder.Recorder",
            mock_recorder_class,
        ),
        patch(
            "homeassistant.components.recorder.get_instance",
            get_instance,
        ),
        patch(
            "homeassistant.components.recorder.core.Recorder",
            mock_recorder_class,
        ),
        patch(
            "homeassistant.components.recorder.async_setup",
            AsyncMock(return_value=True),
        ),
    ):
        yield mock_instance


@pytest.fixture
def mock_config_entry() -> Mock:
    """Create a mock config entry for testing."""
    mock = Mock()
    mock.entry_id = TEST_ENTRY_ID
    mock.data = TEST_CONFIG
    mock.title = "Test Area"
    mock.unique_id = TEST_UNIQUE_ID
    return mock


@pytest.fixture
def mock_probabilities() -> MagicMock:
    """Create a mock probabilities provider with default/fallback values."""
    mock = MagicMock(spec=Probabilities)
    mock.get_default_prior.return_value = 0.5
    mock.is_entity_active.side_effect = lambda eid, state: state == STATE_ON
    mock.get_sensor_config.return_value = {
        "prob_given_true": DEFAULT_PROB_GIVEN_TRUE,
        "prob_given_false": DEFAULT_PROB_GIVEN_FALSE,
        "default_prior": 0.5,
        "weight": 1.0,
        "active_states": {STATE_ON},
    }
    mock.get_entity_type.return_value = EntityType.MOTION
    mock.entity_types = {
        "binary_sensor.test": EntityType.MOTION,
        "binary_sensor.test1": EntityType.MOTION,
        "binary_sensor.test2": EntityType.MOTION,
        "binary_sensor.test3": EntityType.LIGHT,
    }
    return mock


@pytest.fixture
def mock_sensor_inputs() -> MagicMock:
    """Create a mock sensor inputs with default test configuration."""
    mock = MagicMock(spec=SensorInputs)
    mock.primary_sensor = "binary_sensor.motion1"
    mock.motion_sensors = TEST_CONFIG[CONF_MOTION_SENSORS]
    mock.media_devices = TEST_CONFIG[CONF_MEDIA_DEVICES]
    mock.appliances = TEST_CONFIG[CONF_APPLIANCES]
    mock.door_sensors = TEST_CONFIG[CONF_DOOR_SENSORS]
    mock.window_sensors = TEST_CONFIG[CONF_WINDOW_SENSORS]
    mock.lights = TEST_CONFIG[CONF_LIGHTS]
    mock.is_valid_entity_id = SensorInputs.is_valid_entity_id
    return mock


@pytest.fixture
def mock_coordinator(
    mock_probabilities: MagicMock, mock_sensor_inputs: MagicMock
) -> MagicMock:
    """Create a mock coordinator with standard configuration."""
    coordinator = MagicMock()
    coordinator.config_entry = MagicMock()
    coordinator.config_entry.entry_id = TEST_ENTRY_ID
    coordinator.inputs = mock_sensor_inputs
    coordinator.probabilities = mock_probabilities
    coordinator.config = TEST_CONFIG
    coordinator.device_info = {
        "identifiers": {(DOMAIN, TEST_ENTRY_ID)},
        "name": TEST_CONFIG[CONF_NAME],
        "model": "Area Occupancy Sensor",
        "manufacturer": "Home Assistant",
    }
    coordinator.data = MagicMock()
    coordinator.data.current_states = {}
    coordinator.async_request_refresh = AsyncMock()
    coordinator.update_learned_priors = AsyncMock()
    coordinator.async_refresh = AsyncMock()
    
    # Add properties for the new test structure
    coordinator.entry_id = TEST_ENTRY_ID
    coordinator.name = "Test Area"
    coordinator.probability = 0.5
    coordinator.prior = 0.3
    coordinator.decay = 1.0
    coordinator.threshold = 0.5
    coordinator.occupied = False
    coordinator.last_updated = datetime.now()
    coordinator.last_changed = datetime.now()
    coordinator.available = True
    coordinator.last_update_success = True
    
    # Mock binary sensor entity IDs
    coordinator.binary_sensor_entity_ids = {
        "occupancy": "binary_sensor.test_occupancy",
        "wasp": "binary_sensor.test_wasp"
    }
    
    # Add additional mock methods for new tests
    coordinator.request_update = Mock()
    coordinator._async_setup = AsyncMock()
    coordinator.async_shutdown = AsyncMock()
    coordinator.async_update_options = AsyncMock()
    coordinator.async_load_stored_data = AsyncMock()
    coordinator._schedule_next_prior_update = AsyncMock()
    coordinator._handle_prior_update = AsyncMock()
    coordinator._async_refresh_finished = Mock()
    coordinator.async_set_updated_data = Mock()
    coordinator.async_add_listener = Mock(return_value=Mock())
    coordinator.setup = AsyncMock()
    coordinator.async_config_entry_first_refresh = AsyncMock()
    
    # Mock store
    coordinator.store = Mock()
    coordinator.store.async_save_data = Mock()
    
    return coordinator


@pytest.fixture
def mock_prior_state() -> PriorState:
    """Create a mock PriorState object with sample learned priors."""
    state = PriorState()
    state.update_entity_prior(
        "binary_sensor.test",
        prob_given_true=0.85,
        prob_given_false=0.15,
        prior=0.55,
        timestamp=datetime.now().isoformat(),
    )
    return state


@pytest.fixture
def default_config() -> ProbabilityConfig:
    """Return a default probability configuration for testing."""
    return {
        "prob_given_true": DEFAULT_PROB_GIVEN_TRUE,
        "prob_given_false": DEFAULT_PROB_GIVEN_FALSE,
        "default_prior": 0.5,
        "weight": 1.0,
        "active_states": {STATE_ON},
    }


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
    mock_config_entry: Mock,
    setup_test_entities: None,
    mock_recorder: MagicMock,
) -> None:
    """Set up the Area Occupancy Detection integration for testing."""
    # Create a real ConfigEntry instance instead of a Mock
    real_config_entry = MockConfigEntry(
        domain=DOMAIN,
        data=TEST_CONFIG,
        options={},
        entry_id="test_entry_id",
        title="Test Area",
        version=1,
        minor_version=1,
    )
    
    # Add the config entry to hass
    real_config_entry.add_to_hass(hass)
    
    # Now set up the entry
    await hass.config_entries.async_setup(real_config_entry.entry_id)


@pytest.fixture
def mock_entity_manager() -> Mock:
    """Create a mock entity manager."""
    mock = Mock()
    mock.entities = {}
    mock.to_dict = Mock(return_value={"entities": {}})
    return mock


@pytest.fixture 
def mock_coordinator_with_threshold() -> Mock:
    """Create a mock coordinator with specific threshold for testing."""
    coordinator = Mock()
    coordinator.threshold = 0.6
    coordinator.probability = 0.5  
    coordinator.occupied = False  # 0.5 < 0.6
    return coordinator


@pytest.fixture
def mock_coordinator_with_sensors() -> Mock:
    """Create a mock coordinator with sensor entities."""
    coordinator = Mock()
    coordinator.probability = 0.7
    coordinator.prior = 0.4
    coordinator.decay = 0.8
    return coordinator


@pytest.fixture
def valid_storage_data() -> dict:
    """Create valid storage data structure."""
    return {
        "name": "Test Area",
        "probability": 0.6,
        "prior": 0.4,
        "threshold": 0.5,
        "last_updated": "2024-01-01T00:00:00",
        "entities": {
            "binary_sensor.test": {
                "entity_id": "binary_sensor.test",
                "probability": 0.6,
                "type": {
                    "input_type": "motion",
                    "weight": 1.0,
                    "prob_true": 0.8,
                    "prob_false": 0.1,
                    "prior": 0.5,
                    "active_states": ["on"],
                    "active_range": None,
                },
                "prior": {
                    "prob_given_true": 0.8,
                    "prob_given_false": 0.1,
                    "last_updated": "2024-01-01T00:00:00",
                },
                "decay": {
                    "is_decaying": False,
                    "last_trigger_ts": 1704067200.0,
                    "half_life": 60.0,
                },
                "state": "off",
                "evidence": False,
                "available": True,
            }
        },
    }


@pytest.fixture(autouse=True)
def cleanup_debouncer():
    """Clean up any debouncer timers after each test."""
    yield

    # Get all active timers from the event loop
    try:
        loop = asyncio.get_event_loop()
        for handle in getattr(loop, "_scheduled", []):
            if isinstance(handle, asyncio.TimerHandle) and "Debouncer._on_debounce" in str(
                handle
            ):
                handle.cancel()
    except RuntimeError:
        # Event loop may not exist in some test scenarios
        pass
