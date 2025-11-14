"""Pytest configuration and fixtures for Area Occupancy Detection tests."""

from __future__ import annotations

import asyncio
from asyncio import Lock
from collections.abc import Generator
import contextlib
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
import os
import time
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import voluptuous as vol

from homeassistant.helpers import area_registry as ar

# Set environment variable for auto database initialization in tests
os.environ["AREA_OCCUPANCY_AUTO_INIT_DB"] = "1"

# ruff: noqa: SLF001
from custom_components.area_occupancy.area.area import Area
from custom_components.area_occupancy.config_flow import (
    AreaOccupancyConfigFlow,
    AreaOccupancyOptionsFlow,
)
from custom_components.area_occupancy.const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_AREAS,
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
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
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
    CONF_WEIGHT_WASP,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_PURPOSE,
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
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    HA_RECORDER_DAYS,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import (
    AreaConfig,
    Decay,
    Sensors,
    SensorStates,
    WaspInBox,
    Weights,
)
from custom_components.area_occupancy.data.decay import Decay as DecayClass
from custom_components.area_occupancy.data.entity import Entity, EntityManager
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.data.prior import Prior as PriorClass
from custom_components.area_occupancy.data.purpose import AreaPurpose, Purpose
from custom_components.area_occupancy.db import AreaOccupancyDB, Base
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.entity_registry import EntityRegistry
from homeassistant.helpers.event import async_track_point_in_time
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Note: Event loop management is handled by pytest-asyncio
# We removed the enable_event_loop_debug fixture as it was interfering
# with pytest-asyncio's event loop management and causing RuntimeError
# issues when tests run together. pytest-asyncio handles event loop
# creation and cleanup automatically.


# Ensure all config entries have state attribute for hass fixture teardown
@pytest.fixture(autouse=True)
def ensure_config_entries_have_state(hass: HomeAssistant) -> Generator[None]:
    """Ensure all config entries returned by async_entries() have state attribute.

    The hass fixture from pytest-homeassistant-custom-component checks entry.state
    during teardown. This fixture ensures all entries have state set to prevent
    AttributeError during teardown.
    """
    original_async_entries = hass.config_entries.async_entries

    def async_entries_with_state(domain=None):
        """Wrapper that ensures all returned entries have state attribute."""
        # Call original async_entries (could be real method or a mock)
        if callable(original_async_entries):
            entries = original_async_entries(domain)
        else:
            # If it's not callable, try to get entries from _entries dict
            entries = (
                list(hass.config_entries._entries.values())
                if hasattr(hass.config_entries, "_entries")
                else []
            )

        # Ensure all entries have state attribute
        for entry in entries:
            if not hasattr(entry, "state") or entry.state is None:
                entry.state = ConfigEntryState.LOADED
        return entries

    # Patch async_entries to ensure state is set
    hass.config_entries.async_entries = async_entries_with_state

    yield

    # Restore original (though it may not matter after test)
    hass.config_entries.async_entries = original_async_entries


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
        CONF_AREA_ID: "test_area",
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.test_motion",
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_PURPOSE: DEFAULT_PURPOSE,
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
        CONF_WEIGHT_WASP: DEFAULT_WASP_WEIGHT,
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
    # Set state attribute explicitly so it can be accessed during teardown
    # Use ConfigEntryState if available, otherwise use None
    try:
        entry.state = ConfigEntryState.LOADED
    except (ImportError, NameError):
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


# Removed unused fixture: sample_config_data


@pytest.fixture
def mock_entity_registry() -> Mock:
    """Create a mock entity registry."""
    registry = Mock(spec=EntityRegistry)
    registry.async_get_entity_id = Mock(return_value=None)
    registry.async_update_entity = Mock()

    # Add entities property that can be iterated
    class EntitiesContainer:
        def __init__(self) -> None:
            self._entities: dict[str, Mock] = {}

        def values(self) -> list[Mock]:
            return []

        def items(self) -> list[tuple[str, Mock]]:
            return list(self._entities.items())

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
    manager.cleanup = AsyncMock()
    manager.update_likelihoods = AsyncMock(return_value=1)

    # Remove non-existent to_dict method mock

    return manager


@pytest.fixture
def setup_area_registry(hass: HomeAssistant) -> dict[str, str]:
    """Set up Home Assistant area registry with test areas.

    This fixture ensures that test areas exist in the Home Assistant
    area registry before coordinators try to load them.

    Returns:
        Dictionary mapping area names to area IDs

    Areas created:
    - "Testing" - for default test area
    - "Living Room" - for config flow tests
    - "Kitchen" - for multi-area tests
    """
    area_reg = ar.async_get(hass)

    # Create test areas if they don't exist and collect their IDs
    test_area_names = ["Testing", "Living Room", "Kitchen"]
    area_id_map: dict[str, str] = {}

    for area_name in test_area_names:
        # Check if area already exists
        existing_area = area_reg.async_get_area_by_name(area_name)
        if existing_area:
            area_id_map[area_name] = existing_area.id
        else:
            # Create area and get its ID
            created_area = area_reg.async_create(area_name)
            area_id_map[area_name] = created_area.id

    return area_id_map


@pytest.fixture
def coordinator(
    hass: HomeAssistant,
    mock_realistic_config_entry: Mock,
) -> AreaOccupancyCoordinator:
    """Primary fixture for coordinator testing.

    Provides a real AreaOccupancyCoordinator instance with:
    - Real Home Assistant instance
    - Areas loaded from config entry
    - Real coordinator behavior
    - Proper initialization

    This is the recommended default for most coordinator tests.
    Use mocks only when you need to test error paths or control behavior.

    Example:
        def test_coordinator_method(coordinator: AreaOccupancyCoordinator):
            area_names = coordinator.get_area_names()
            assert len(area_names) > 0
    """
    coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
    coordinator._load_areas_from_config()
    return coordinator


@pytest.fixture
def coordinator_with_areas(
    coordinator: AreaOccupancyCoordinator,
) -> AreaOccupancyCoordinator:
    """Alias for coordinator fixture (backward compatibility).

    DEPRECATED: Use `coordinator` fixture instead.
    This fixture is kept for backward compatibility.

    Example:
        def test_something(coordinator_with_areas: AreaOccupancyCoordinator):
            area_names = coordinator_with_areas.get_area_names()
            assert len(area_names) > 0
    """
    return coordinator


@pytest.fixture
def coordinator_with_db(test_db: Any) -> AreaOccupancyCoordinator:
    """Create a real coordinator with real db attached.

    This is a simple wrapper around test_db that returns the coordinator.
    Use this for tests that need both coordinator and database functionality.

    Example:
        def test_db_operation(coordinator_with_db: AreaOccupancyCoordinator):
            coordinator = coordinator_with_db
            db = coordinator.db  # Real db attached to real coordinator
            area = coordinator.get_area_or_default("Test Area")
            # Use real area.prior.value instead of mocking
    """
    return test_db.coordinator


@pytest.fixture
def default_area(coordinator_with_areas: AreaOccupancyCoordinator) -> Any:
    """Get the default (first) area from coordinator.

    This fixture provides easy access to the first area from coordinator_with_areas,
    eliminating the need to call coordinator.get_area_or_default() in every test.

    Example:
        def test_something(default_area):
            assert default_area.area_name is not None
            assert default_area.config is not None
    """

    return coordinator_with_areas.get_area_or_default()


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
    # Remove non-existent to_dict method mock
    return entity_type


@pytest.fixture
def mock_entity_type_manager(mock_entity_type: Mock) -> Mock:
    """Create a mock entity type manager (simplified since EntityTypeManager doesn't exist)."""
    manager = Mock()
    manager.cleanup = Mock()
    # Remove non-existent to_dict method mock
    manager.get_entity_type = Mock(return_value=mock_entity_type)
    # Property for entity_types
    type(manager).entity_types = property(
        lambda self: {InputType.MOTION: mock_entity_type}
    )
    return manager


@pytest.fixture
def mock_purpose_manager() -> Mock:
    """Create a mock Purpose (for backward compatibility with tests)."""
    manager = Mock(spec=Purpose)
    manager.cleanup = Mock()
    manager.purpose = AreaPurpose.SOCIAL
    manager.name = "Social"
    manager.description = "Living room, family room, dining room. People linger here."
    manager.half_life = 720.0
    manager.get_purpose = Mock(return_value=manager)
    manager.get_all_purposes = Mock(return_value={})
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
        db_instance = _create_mock_db()
        mock_db_class.return_value = db_instance
        yield db_instance


@pytest.fixture
def mock_area_occupancy_db_patches() -> list[Any]:
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


# Removed unused fixtures: valid_entity_data, valid_db_data


@pytest.fixture(autouse=True)
def mock_frame_helper(hass: HomeAssistant) -> Generator[Mock]:
    """Mock the Home Assistant frame helper for all tests."""
    with (
        patch("homeassistant.helpers.frame._hass") as mock_frame_hass,
        patch("homeassistant.helpers.frame.get_integration_frame") as mock_get_frame,
        patch("homeassistant.helpers.frame.report_usage") as mock_report_usage,
        patch(
            "homeassistant.helpers.frame.report_non_thread_safe_operation"
        ) as mock_report_thread,
    ):
        mock_frame_hass.hass = hass
        mock_frame_hass.hass.loop = hass.loop

        # Mock the get_integration_frame function to return a valid frame
        mock_frame = Mock()
        mock_frame.filename = "/workspaces/Area-Occupancy-Detection/custom_components/area_occupancy/coordinator.py"
        mock_frame.lineno = 1
        mock_frame.function = "test_function"
        mock_get_frame.return_value = mock_frame

        # Mock the report functions to do nothing
        mock_report_usage.return_value = None
        mock_report_thread.return_value = None

        yield mock_frame_hass


# Utility functions for common test patterns
def create_mock_entity_registry_with_entities(entities: list[dict[str, Any]]) -> Mock:
    """Create a mock entity registry with specific entities."""
    registry = Mock(spec=EntityRegistry)

    class EntitiesContainer:
        def values(self) -> list[Mock]:
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


def _create_mock_entity(
    entity_id: str,
    coordinator: AreaOccupancyCoordinator,
    mock_entity_type: Mock,
    mock_decay: Mock,
    evidence: bool | None = True,
    available: bool = True,
    state: str | None = STATE_ON,
    probability: float = 0.75,
    active: bool = True,
    last_updated: datetime | None = None,
    previous_evidence: bool | None = False,
    previous_probability: float = 0.35,
    has_new_evidence: bool = True,
    decay_factor: float = 1.0,
) -> Mock:
    """Create mock entities with different states."""
    entity = Mock(spec=Entity)
    entity.entity_id = entity_id
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.hass = coordinator.hass
    entity.last_updated = last_updated or dt_util.utcnow()
    entity.previous_evidence = previous_evidence
    entity.previous_probability = previous_probability

    # Properties
    entity.evidence = evidence
    entity.available = available
    entity.state = state
    entity.probability = probability
    entity.active = active
    entity.active_states = [STATE_ON]
    entity.active_range = None
    entity.decay_factor = decay_factor
    entity.weight = 0.85

    # Methods
    entity.has_new_evidence = Mock(return_value=has_new_evidence)
    entity.update_likelihood = Mock()

    return entity


def _create_mock_db() -> Mock:
    """Create a mock AreaOccupancyDB instance."""
    db_instance = Mock()
    # Only mock methods that actually exist in AreaOccupancyDB
    db_instance.load_data = AsyncMock()
    db_instance.save_data = AsyncMock()
    db_instance.save_area_data = AsyncMock()
    db_instance.save_entity_data = AsyncMock()
    db_instance.is_intervals_empty = Mock(return_value=True)
    db_instance.sync_states = AsyncMock()
    db_instance.get_area_data = Mock(return_value=None)
    db_instance.ensure_area_exists = AsyncMock()
    db_instance.get_latest_interval = Mock(return_value=None)
    db_instance.get_engine = Mock(return_value=None)
    db_instance.init_db = Mock()
    db_instance.delete_db = Mock()
    db_instance.force_reinitialize = Mock()
    db_instance.get_db_version = Mock(return_value=3)
    db_instance.set_db_version = Mock()
    db_instance.get_session = Mock()
    return db_instance


@pytest.fixture
def mock_active_entity(
    coordinator: AreaOccupancyCoordinator, mock_entity_type: Mock, mock_decay: Mock
) -> Mock:
    """Create a mock entity in active state (evidence=True, available=True)."""
    return _create_mock_entity(
        entity_id="binary_sensor.active_entity",
        coordinator=coordinator,
        mock_entity_type=mock_entity_type,
        mock_decay=mock_decay,
        evidence=True,
        available=True,
        state=STATE_ON,
        probability=0.75,
        active=True,
        previous_evidence=False,
        previous_probability=0.35,
        has_new_evidence=True,
        decay_factor=1.0,
    )


@pytest.fixture
def mock_inactive_entity(
    coordinator: AreaOccupancyCoordinator, mock_entity_type: Mock, mock_decay: Mock
) -> Mock:
    """Create a mock entity in inactive state (evidence=False, available=True)."""
    return _create_mock_entity(
        entity_id="binary_sensor.inactive_entity",
        coordinator=coordinator,
        mock_entity_type=mock_entity_type,
        mock_decay=mock_decay,
        evidence=False,
        available=True,
        state=STATE_OFF,
        probability=0.25,
        active=True,  # Because decay is running
        previous_evidence=True,
        previous_probability=0.75,
        has_new_evidence=True,
        decay_factor=0.8,
    )


@pytest.fixture
def mock_unavailable_entity(
    coordinator: AreaOccupancyCoordinator, mock_entity_type: Mock, mock_decay: Mock
) -> Mock:
    """Create a mock entity in unavailable state (available=False)."""
    return _create_mock_entity(
        entity_id="binary_sensor.unavailable_entity",
        coordinator=coordinator,
        mock_entity_type=mock_entity_type,
        mock_decay=mock_decay,
        evidence=None,
        available=False,
        state=None,
        probability=0.15,
        active=False,
        last_updated=None,
        previous_evidence=None,
        previous_probability=0.15,
        has_new_evidence=False,
        decay_factor=1.0,
    )


@pytest.fixture
def mock_stale_entity(
    coordinator: AreaOccupancyCoordinator, mock_entity_type: Mock, mock_decay: Mock
) -> Mock:
    """Create a mock entity with stale update (> 1 hour ago)."""
    return _create_mock_entity(
        entity_id="binary_sensor.stale_entity",
        coordinator=coordinator,
        mock_entity_type=mock_entity_type,
        mock_decay=mock_decay,
        evidence=False,
        available=True,
        state=STATE_OFF,
        probability=0.30,
        active=False,
        last_updated=dt_util.utcnow() - timedelta(hours=2),
        previous_evidence=False,
        previous_probability=0.30,
        has_new_evidence=False,
        decay_factor=1.0,
    )


@pytest.fixture
def mock_last_updated() -> Mock:
    """Create a mock last_updated object with isoformat method."""
    mock_timestamp = Mock()
    mock_timestamp.isoformat.return_value = "2024-01-01T00:00:00"
    return mock_timestamp


# Removed unused fixture: mock_motion_entity_type (use mock_entity_type instead)


@pytest.fixture
def mock_entity_manager_with_states(
    mock_active_entity: Mock,
    mock_inactive_entity: Mock,
    mock_unavailable_entity: Mock,
    mock_stale_entity: Mock,
) -> Mock:
    """Create a mock entity manager with entities in different states."""
    entities = {
        "binary_sensor.active_entity": mock_active_entity,
        "binary_sensor.inactive_entity": mock_inactive_entity,
        "binary_sensor.unavailable_entity": mock_unavailable_entity,
        "binary_sensor.stale_entity": mock_stale_entity,
    }
    return _create_mock_entity_manager(entities)


def _create_mock_entity_manager(entities: dict[str, Mock] | None = None) -> Mock:
    """Create mock entity managers."""
    manager = Mock()
    manager.entities = entities or {}
    if entities:
        manager.get_entity = Mock(return_value=list(entities.values())[0])
    else:
        manager.get_entity = Mock(side_effect=ValueError("Entity not found"))
    return manager


@pytest.fixture
def mock_empty_entity_manager() -> Mock:
    """Create a mock entity manager with no entities."""
    return _create_mock_entity_manager()


@pytest.fixture
def mock_entities_container() -> Mock:
    """Create a mock entities container that can be used for coordinator.entities attribute."""
    return _create_mock_entity_manager()


@pytest.fixture
def coordinator_with_sensors(
    coordinator: AreaOccupancyCoordinator,
) -> AreaOccupancyCoordinator:
    """Create a real coordinator with sensors set up for testing.

    This fixture extends coordinator by setting up mock entities
    on the area for sensor testing purposes.

    Example:
        def test_sensor_entities(coordinator_with_sensors: AreaOccupancyCoordinator):
            area = coordinator_with_sensors.get_area_or_default()
            assert "binary_sensor.motion" in area.entities.entities
    """
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area_or_default(area_name)

    # Set up mock entities on the area
    mock_entities = {
        "binary_sensor.motion": Mock(
            entity_id="binary_sensor.motion",
            available=True,
            evidence=True,
            probability=0.85,
            type=Mock(input_type=InputType.MOTION, weight=0.85),
            decay=Mock(is_decaying=False, decay_factor=1.0),
            prob_given_true=0.9,
            prob_given_false=0.05,
            coordinator=coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=False,
            previous_probability=0.35,
            active=True,
            active_states=[STATE_ON],
            active_range=None,
            decay_factor=1.0,
            state=STATE_ON,
        ),
        "binary_sensor.motion2": Mock(
            entity_id="binary_sensor.motion2",
            available=True,
            evidence=True,
            probability=0.75,
            type=Mock(input_type=InputType.MOTION, weight=0.85),
            decay=Mock(is_decaying=False, decay_factor=1.0),
            prob_given_true=0.85,
            prob_given_false=0.1,
            coordinator=coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=True,
            previous_probability=0.6,
            active=True,
            active_states=[STATE_ON],
            active_range=None,
            decay_factor=1.0,
            state=STATE_ON,
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
            coordinator=coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=False,
            previous_probability=0.35,
            active=True,
            active_states=["playing", "paused"],
            active_range=None,
            decay_factor=1.0,
            state="playing",
        ),
        "binary_sensor.appliance": Mock(
            entity_id="binary_sensor.appliance",
            available=True,
            evidence=True,
            probability=0.6,
            type=Mock(input_type=InputType.APPLIANCE, weight=0.3),
            decay=Mock(is_decaying=True, decay_factor=0.8),
            prob_given_true=0.6,
            prob_given_false=0.05,
            coordinator=coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=True,
            previous_probability=0.5,
            active=True,
            active_states=["on", "standby"],
            active_range=None,
            decay_factor=0.8,
            state="on",
        ),
    }

    # Set up entities manager with mock entities
    area._entities = SimpleNamespace(
        entities=mock_entities,
        active_entities=list(mock_entities.values()),
        inactive_entities=[],
        decaying_entities=[
            entity
            for entity in mock_entities.values()
            if getattr(entity.decay, "is_decaying", False)
        ],
    )

    # Mock area methods (coordinator wrappers will delegate to these)
    area.area_prior = Mock(return_value=0.35)
    area.probability = Mock(return_value=0.65)
    area.decay = Mock(return_value=0.8)
    area.occupied = Mock(return_value=True)
    area.threshold = Mock(return_value=0.5)
    area.type_probabilities = Mock(return_value={})
    area.device_info = Mock(
        return_value={
            "identifiers": {(DOMAIN, area.area_name)},
            "name": area.config.name,
            "manufacturer": DEVICE_MANUFACTURER,
            "model": DEVICE_MODEL,
            "sw_version": DEVICE_SW_VERSION,
        }
    )

    return coordinator


@pytest.fixture
def coordinator_with_areas_with_sensors(
    coordinator_with_sensors: AreaOccupancyCoordinator,
) -> AreaOccupancyCoordinator:
    """Alias for coordinator_with_sensors fixture (backward compatibility).

    DEPRECATED: Use `coordinator_with_sensors` fixture instead.
    This fixture is kept for backward compatibility.

    Example:
        def test_sensor_entities(coordinator_with_areas_with_sensors: AreaOccupancyCoordinator):
            area = coordinator_with_areas_with_sensors.get_area_or_default()
            assert "binary_sensor.motion" in area.entities.entities
    """
    return coordinator_with_sensors


def create_test_area(
    coordinator: AreaOccupancyCoordinator,
    area_name: str = "Test Area",
    entity_ids: list[str] | None = None,
    **config_overrides: Any,
) -> Any:
    """Create a test area and add it to coordinator.

    Helper function for standardized area creation in tests. Creates an Area
    instance with the specified configuration and adds it to the coordinator's
    areas dict.

    Args:
        coordinator: The coordinator instance to add the area to
        area_name: Name for the area (default: "Test Area")
        entity_ids: Optional list of entity IDs to configure in sensors
        **config_overrides: Override any config values (e.g., threshold=0.7)

    Returns:
        Created Area instance

    Example:
        area = create_test_area(
            coordinator,
            area_name="Kitchen",
            entity_ids=["binary_sensor.motion1"],
            threshold=0.6
        )
    """
    # Create area - it will load config from coordinator.config_entry
    area = Area(coordinator, area_name=area_name)

    # Override config values if provided
    if entity_ids:
        # Set up sensors with provided entity IDs
        area.config.sensors = Sensors(
            motion=entity_ids
            if "motion" not in config_overrides
            else config_overrides.get("motion", []),
            primary_occupancy=None,
            media=[],
            appliance=[],
            illuminance=[],
            humidity=[],
            temperature=[],
            door=[],
            window=[],
            _parent_config=area.config,
        )

    # Apply any other config overrides
    for key, value in config_overrides.items():
        if hasattr(area.config, key) and key != "motion":
            setattr(area.config, key, value)

    # Add to coordinator
    coordinator.areas[area_name] = area

    # Update get_area_names and get_area_or_default mocks if they exist
    if hasattr(coordinator, "get_area_names"):
        area_names = list(coordinator.areas.keys())
        coordinator.get_area_names = Mock(return_value=area_names)
    if hasattr(coordinator, "get_area_or_default"):

        def get_area_or_default(name: str | None = None):
            if name is None:
                # When name is None, always return first area (at least one area exists)
                return next(iter(coordinator.areas.values()))
            return coordinator.areas.get(name)

        coordinator.get_area_or_default = Mock(side_effect=get_area_or_default)

    return area


@contextmanager
def patch_area_method(area_class: type, method_name: str, return_value: Any):
    """Patch an area method at class level so it works after area reloads.

    This context manager patches an area method at the class level, making
    it persist even when areas are cleared and reloaded (e.g., in async_update_options).

    Args:
        area_class: The Area class to patch
        method_name: Name of the method to patch
        return_value: Value to return from the patched method

    Example:
        with patch_area_method(Area, "async_cleanup", AsyncMock()) as mock_cleanup:
            await coordinator.async_update_options(options)
            mock_cleanup.assert_called()
    """

    with patch.object(
        area_class, method_name, return_value=return_value
    ) as mock_method:
        yield mock_method


@pytest.fixture
def mock_area(mock_config: AreaConfig, mock_entity_manager: Mock) -> Mock:
    """Create a mock area with standard attributes.

    This fixture provides a reusable mock area for tests that don't need
    a real coordinator. The area has standard attributes configured.

    Example:
        def test_something(mock_area: Mock):
            assert mock_area.area_name == "Test Area"
            assert mock_area.config is not None
    """
    area = Mock()
    area.area_name = "Test Area"
    area.config = mock_config
    area.entities = mock_entity_manager
    area.prior = Mock()
    area.purpose = Mock()
    return area


@pytest.fixture
def mock_prior() -> Mock:
    """Create a mock Prior instance for backward compatibility with tests."""
    prior = Mock()
    prior.value = 0.35
    prior._current_value = 0.35
    prior.last_updated = dt_util.utcnow()
    prior.update = AsyncMock(return_value=0.35)
    prior.calculate = AsyncMock(return_value=0.35)
    # Only mock to_dict if it exists (it does exist in Prior class)
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

    # Add methods for time-based priors (only mock methods that exist)
    prior.get_time_prior = Mock(return_value=0.25)

    # Only mock to_dict if it exists (it does exist in Prior class)
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
    decay = Mock(spec=DecayClass)
    decay.is_decaying = False
    decay.last_trigger_ts = time.time()
    decay.half_life = 60.0
    # Use PropertyMock for the decay_factor property
    type(decay).decay_factor = PropertyMock(return_value=1.0)

    # Add side effect for start_decay to properly simulate behavior
    def start_decay_side_effect() -> None:
        if not decay.is_decaying:
            decay.is_decaying = True
            decay.last_trigger_ts = time.time()

    decay.start_decay.side_effect = start_decay_side_effect
    # Remove non-existent to_dict method mock
    return decay


def _create_mock_service_call(data: dict[str, Any]) -> Mock:
    """Create mock service calls."""
    call = Mock(spec=ServiceCall)
    call.data = data
    call.return_response = True
    return call


@pytest.fixture
def mock_service_call() -> Mock:
    """Create a mock service call with common attributes."""
    return _create_mock_service_call({"entry_id": "test_entry_id"})


@pytest.fixture
def mock_service_call_with_entity() -> Mock:
    """Create a mock service call with entity_id."""
    return _create_mock_service_call(
        {"entry_id": "test_entry_id", "entity_id": "binary_sensor.test_motion"}
    )


@pytest.fixture
def mock_comprehensive_entity(
    coordinator: AreaOccupancyCoordinator, mock_entity_type: Mock, mock_decay: Mock
) -> Mock:
    """Create a comprehensive mock entity with all components."""
    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.test_motion"
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.hass = coordinator.hass
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
    entity.weight = 0.85

    # Methods
    entity.has_new_evidence = Mock(return_value=True)
    entity.update_likelihood = Mock()
    entity.cleanup = Mock()
    return entity


@pytest.fixture
def mock_comprehensive_entity_manager(
    coordinator: AreaOccupancyCoordinator, mock_comprehensive_entity: Mock
) -> Mock:
    """Create a comprehensive mock entity manager with entities."""
    manager = Mock(spec=EntityManager)
    manager.coordinator = coordinator
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area_or_default(area_name)
    manager.config = area.config
    manager.hass = coordinator.hass
    manager._entities = {"binary_sensor.test_motion": mock_comprehensive_entity}
    manager.entities = {"binary_sensor.test_motion": mock_comprehensive_entity}
    manager.entity_ids = ["binary_sensor.test_motion"]
    manager.active_entities = [mock_comprehensive_entity]
    manager.inactive_entities = []
    manager.decaying_entities = []
    # Remove non-existent to_dict method mock
    # Remove non-existent create_entity method mock
    manager.get_entity = Mock(return_value=mock_comprehensive_entity)
    manager.add_entity = Mock()
    manager.cleanup = AsyncMock()
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


# Removed unused fixture: mock_real_coordinator


# Global patches for common issues
@pytest.fixture(autouse=True)
def mock_recorder_globally() -> Generator[Mock]:
    """Automatically mock recorder for all tests."""
    with patch("homeassistant.helpers.recorder.get_instance") as mock_get_instance_ha:
        mock_instance = Mock()
        mock_instance.async_add_executor_job = AsyncMock(return_value={})
        mock_get_instance_ha.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=True)
def mock_significant_states_globally() -> Generator[Mock]:
    """Automatically mock significant states for all tests."""
    with patch(
        "homeassistant.components.recorder.history.get_significant_states"
    ) as mock_states:
        mock_states.return_value = {}
        yield mock_states


@pytest.fixture(autouse=True)
def mock_track_point_in_time_globally() -> Generator[None]:
    """Automatically mock timer-related functions for all tests."""

    class CancellableTimerMock:
        """Mock timer that properly handles cleanup verification."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._cancelled = True
            self._args = args
            self._callback = args[1] if len(args) > 1 else None

        def cancel(self) -> None:
            self._cancelled = True

        def cancelled(self) -> bool:
            return True

        def __repr__(self) -> str:
            return f"<MockTimerHandle cancelled={self._cancelled}>"

    def create_timer_mock(*args: Any, **kwargs: Any) -> CancellableTimerMock:
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
    coordinator: AreaOccupancyCoordinator, mock_entity_type: Mock, mock_decay: Mock
) -> Mock:
    """Create a mock entity specifically for likelihood calculation tests."""
    entity = Mock(spec=Entity)
    entity.entity_id = "binary_sensor.motion_sensor_1"
    entity.type = mock_entity_type
    entity.prob_given_true = 0.8
    entity.prob_given_false = 0.1
    entity.decay = mock_decay
    entity.hass = coordinator.hass
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
    entity.weight = 0.85

    # Methods
    entity.has_new_evidence = Mock(return_value=True)
    entity.update_likelihood = Mock()

    return entity


@pytest.fixture(autouse=True)
def mock_area_occupancy_db_globally(request: Any) -> Generator[Mock | None]:
    """Automatically mock AreaOccupancyDB for all tests except database tests.

    Skips mocking for:
    1. Test files named test_db.py
    2. Test classes named Test*DB* or TestDatabase*
    3. Tests using test_db or coordinator_with_db fixtures
    """
    # Get test file name
    test_file = ""
    if hasattr(request.node, "fspath"):
        test_file = str(request.node.fspath)
    elif hasattr(request, "path"):
        test_file = str(request.path)

    # Get test class name
    cls_name = getattr(getattr(request.node, "cls", None), "__name__", "")

    # Get fixture names requested by this test
    fixture_names = getattr(request, "fixturenames", [])
    if not fixture_names and hasattr(request.node, "fixturenames"):
        fixture_names = request.node.fixturenames

    # Check if this is a database test
    is_db_test = (
        "test_db.py" in test_file
        or "TestDatabase" in cls_name
        or "TestAreaOccupancyDB" in cls_name
        or "test_db" in fixture_names
        or "coordinator_with_db" in fixture_names
        or "configured_db" in fixture_names
    )

    if is_db_test:
        yield None
        return

    # Mock for all other tests
    with patch("custom_components.area_occupancy.db.AreaOccupancyDB") as mock_db_class:
        mock_db = _create_mock_db()
        mock_db_class.return_value = mock_db
        yield mock_db


@pytest.fixture(autouse=True)
def mock_data_update_coordinator_debouncer() -> Generator[None]:
    """Automatically mock DataUpdateCoordinator's debouncer for all tests."""
    original_init = DataUpdateCoordinator.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._debounced_refresh = AsyncMock()

    with patch(
        "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.__init__",
        patched_init,
    ):
        yield


@pytest.fixture
def mock_area_occupancy_db_data() -> dict[str, Any]:
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
def mock_config() -> Mock:
    """Return a representative Config instance for testing."""
    # Create a mock config that works with the new Config class structure
    config = Mock()

    # Set basic attributes
    config.name = "Test Area"
    config.purpose = AreaPurpose.SOCIAL
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

    # Add purpose manager mock (for backward compatibility)
    config.purpose_manager = Mock(spec=Purpose)
    config.purpose_manager.purpose = AreaPurpose.SOCIAL
    config.purpose_manager.name = "Social"
    config.purpose_manager.description = (
        "Living room, family room, dining room. People linger here."
    )
    config.purpose_manager.half_life = 720.0

    return config


@pytest.fixture
def mock_realistic_config_entry(
    hass: HomeAssistant, setup_area_registry: dict[str, str]
) -> Mock:
    """Return a realistic ConfigEntry for Area Occupancy Detection."""
    entry = Mock(spec=ConfigEntry)
    entry.entry_id = "01JQRDH37YHVXR3X4FMDYTHQD8"
    entry.domain = "area_occupancy"
    entry.title = "Testing"
    entry.source = "user"
    entry.version = 9
    entry.minor_version = 2
    entry.unique_id = None
    entry.state = ConfigEntryState.LOADED
    entry.runtime_data = None
    entry.pref_disable_new_entities = False
    entry.pref_disable_polling = False
    entry.subentries = []
    entry.disabled_by = None
    entry.discovery_keys = {}
    entry.created_at = "2025-04-01T10:14:38.590998+00:00"
    entry.modified_at = "2025-06-19T07:10:40.167187+00:00"
    # Use new multi-area format with CONF_AREAS
    # Get actual area ID from registry
    testing_area_id = setup_area_registry.get("Testing", "test_area_1")
    entry.data = {
        CONF_AREAS: [
            {
                CONF_AREA_ID: testing_area_id,  # Use actual area ID from registry
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
                "humidity_sensors": [
                    "sensor.humidity_sensor_1",
                    "sensor.humidity_sensor_2",
                ],
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
                "purpose": "social",
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
                "weight_wasp": 0.8,
                "weight_window": 0.2,
                "window_active_state": "open",
                "window_sensors": ["binary_sensor.window_sensor"],
            }
        ]
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
        "purpose": "social",
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
        "weight_wasp": 0.8,
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
def auto_cancel_timers(monkeypatch: Any) -> Generator[None]:
    """Automatically track and cancel all timers created during a test.

    Note: This fixture only activates if an event loop exists to avoid
    RuntimeError when event loops are closed between tests.
    """
    timer_handles: list[Any] = []
    loop = None

    # Try to get the event loop, but don't fail if it doesn't exist
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = None
        except RuntimeError:
            # No event loop available - skip timer tracking
            loop = None

    if loop is not None:
        original_call_later = loop.call_later
        original_call_at = loop.call_at

        def tracking_call_later(
            delay: float, callback: Any, *args: Any, **kwargs: Any
        ) -> Any:
            handle = original_call_later(delay, callback, *args, **kwargs)
            timer_handles.append(handle)
            return handle

        def tracking_call_at(
            when: float, callback: Any, *args: Any, **kwargs: Any
        ) -> Any:
            handle = original_call_at(when, callback, *args, **kwargs)
            timer_handles.append(handle)
            return handle

        monkeypatch.setattr(loop, "call_later", tracking_call_later)
        monkeypatch.setattr(loop, "call_at", tracking_call_at)

    # Patch async_track_point_in_time if used directly
    try:
        orig_async_track_point_in_time = async_track_point_in_time

        def tracking_async_track_point_in_time(
            hass: Any, action: Any, point_in_time: Any
        ) -> Any:
            # Don't append to timer_handles as it's not a TimerHandle
            return orig_async_track_point_in_time(hass, action, point_in_time)

        monkeypatch.setattr(
            "homeassistant.helpers.event.async_track_point_in_time",
            tracking_async_track_point_in_time,
        )
    except ImportError:
        pass

    yield

    # Clean up timers if loop is still available
    if loop is not None and not loop.is_closed():
        for handle in timer_handles:
            with contextlib.suppress(Exception):
                handle.cancel()


# SQLAlchemy Database Testing Fixtures
# Following best practices for in-memory SQLite testing with proper isolation


@pytest.fixture
def db_engine() -> Generator[Any]:
    """Create an in-memory SQLite engine for testing.

    Uses shared cache mode so data saved in one connection is visible
    to other connections in the same process (important for executor threads).
    """
    # Create in-memory SQLite engine with shared cache
    # Use StaticPool to reuse connections (required for shared cache)
    # We'll explicitly close all connections in cleanup
    engine = create_engine(
        "sqlite:///:memory:?cache=shared",
        echo=False,
        pool_pre_ping=False,  # Not needed for in-memory
        poolclass=sa.pool.StaticPool,  # Use StaticPool for shared cache
        connect_args={"check_same_thread": False},
        # Explicitly close connections when returned to pool
        pool_reset_on_return="commit",
    )

    # Create all tables
    Base.metadata.create_all(engine)

    try:
        yield engine
    finally:
        # Clean up - explicitly close all connections before disposing
        # This prevents ResourceWarnings from unclosed connections
        pool = engine.pool
        if pool and hasattr(pool, "_conn"):
            # Close all connections in StaticPool
            with suppress(Exception):
                # StaticPool stores connection in _conn
                if hasattr(pool, "_conn") and pool._conn:
                    pool._conn.close()
        # Dispose engine to close any remaining connections
        engine.dispose(close=True)
        # Drop tables
        Base.metadata.drop_all(engine)


@pytest.fixture
def test_db(
    coordinator_with_areas: AreaOccupancyCoordinator, db_engine: Any, tmp_path: Any
) -> Generator[Any]:
    """Primary fixture for database testing.

    Provides a real AreaOccupancyDB instance with:
    - In-memory SQLite database
    - Real coordinator attached
    - Proper session management
    - Automatic cleanup

    Each test gets a fresh database instance with rollback for isolation.
    The db_engine fixture (module-scoped) handles table creation/cleanup efficiently.

    Example:
        def test_db_operation(test_db: AreaOccupancyDB):
            db = test_db
            area_name = db.coordinator.get_area_names()[0]
            # Use real database operations
    """
    coordinator = coordinator_with_areas

    # Create real database instance attached to real coordinator
    db = AreaOccupancyDB(coordinator=coordinator)

    # Store the original engine so we can dispose of it immediately
    # This prevents any connections from being opened on the original engine
    original_engine = db.engine

    # Override the database with our test database BEFORE any operations
    # This prevents the original engine from opening connections
    db.engine = db_engine

    # Immediately dispose of the original engine to close any connections
    # This prevents ResourceWarnings from unclosed connections
    if original_engine is not None:
        with suppress(Exception):
            original_engine.dispose(close=True)

    # Create session factory (reused across tests in module)
    # Configure session to expire on commit to prevent connection leaks
    SessionLocal = sessionmaker(
        bind=db_engine,
        expire_on_commit=False,  # Keep objects after commit
        autoflush=False,  # Don't autoflush
        autocommit=False,  # Use transactions
    )
    db._session_maker = SessionLocal

    # Initialize database (now uses test engine)
    db.init_db()
    db.set_db_version()

    # Attach db to coordinator (it's already attached via AreaOccupancyDB.__init__)
    # but ensure it's using our test engine
    coordinator.db = db

    try:
        yield db
    finally:
        # No need to dispose original_engine here - we already did it above
        # The test engine (db_engine) is disposed by the db_engine fixture
        pass


@pytest.fixture
def db_test_session(test_db: Any) -> Generator[Any]:
    """Provide a fresh database session for each test with automatic rollback.

    This fixture provides per-test session isolation. Each test gets a fresh
    session that is automatically rolled back after the test completes.

    Example:
        def test_with_session(test_db: AreaOccupancyDB, db_test_session):
            session = db_test_session
            # Use session for direct database operations
            session.add(...)
            session.commit()
            # Changes are automatically rolled back after test
    """
    session = test_db._session_maker()

    try:
        yield session
    finally:
        # Rollback any uncommitted changes and close session
        # Ensure connection is properly closed to prevent ResourceWarnings
        with suppress(Exception):
            session.rollback()
        # Expunge all objects before closing to ensure cleanup
        if hasattr(session, "expunge_all"):
            session.expunge_all()
        # Close session - this should close the underlying connection
        session.close()
        # Explicitly close any bound connection
        if hasattr(session, "bind") and hasattr(session.bind, "invalidate"):
            with suppress(Exception):
                session.bind.invalidate()


@pytest.fixture
def db_session(db_engine: Any) -> Generator[Any]:
    """Create a database session for testing with automatic rollback."""
    # Create session factory bound to the test engine
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        # Rollback any uncommitted changes and close session
        # Also close any bound connections to prevent ResourceWarnings
        with suppress(Exception):
            session.rollback()
        # Expunge all objects before closing to ensure cleanup
        if hasattr(session, "expunge_all"):
            session.expunge_all()
        # Close session - this should close the underlying connection
        session.close()
        # Explicitly invalidate any bound connection
        if hasattr(session, "bind") and hasattr(session.bind, "invalidate"):
            with suppress(Exception):
                session.bind.invalidate()


@pytest.fixture
def transactional_db_session(db_engine: Any) -> Generator[Any]:
    """Create a database session with nested transaction for maximum isolation."""
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
        # Close session first, then connection
        session.rollback()
        session.close()
        trans.rollback()
        connection.close()


# Removed redundant fixture: seeded_db_session (use db_session directly)
# Removed deprecated fixtures: mock_area_occupancy_db, db_with_engine, mock_db_with_engine
# Use test_db fixture instead for all database testing needs


def _create_sample_data() -> dict[str, Any]:
    """Create sample data for testing."""
    now = dt_util.utcnow()
    start_time = now
    end_time = start_time + timedelta(hours=1)

    return {
        "area": {
            "entry_id": "test_entry_001",
            "area_name": "Test Living Room",
            "area_id": "area_123",
            "purpose": "social",
            "threshold": 0.5,
            "area_prior": 0.3,
            "created_at": now,
            "updated_at": now,
        },
        "entity": {
            "entry_id": "test_entry_001",
            "entity_id": "binary_sensor.motion_1",
            "entity_type": "motion",
            "weight": 0.85,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "is_decaying": False,
            "decay_start": None,
            "evidence": False,
            "last_updated": now,
            "created_at": now,
        },
        "interval": {
            "entity_id": "binary_sensor.motion_1",
            "state": "on",
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": 3600.0,
            "created_at": now,
        },
        "prior": {
            "entry_id": "test_entry_001",
            "day_of_week": 1,  # Monday
            "time_slot": 14,  # 2 PM
            "prior_value": 0.35,
            "data_points": 10,
            "last_updated": now,
        },
    }


@pytest.fixture
def sample_area_data() -> dict[str, Any]:
    """Provide sample area data for testing."""
    data = _create_sample_data()["area"]
    return dict(data) if isinstance(data, dict) else {}


@pytest.fixture
def sample_entity_data() -> dict[str, Any]:
    """Provide sample entity data for testing."""
    data = _create_sample_data()["entity"]
    return dict(data) if isinstance(data, dict) else {}


@pytest.fixture
def sample_interval_data() -> dict[str, Any]:
    """Provide sample interval data for testing."""
    data = _create_sample_data()["interval"]
    return dict(data) if isinstance(data, dict) else {}


@pytest.fixture
def sample_prior_data() -> dict[str, Any]:
    """Provide sample prior data for testing."""
    data = _create_sample_data()["prior"]
    return dict(data) if isinstance(data, dict) else {}


# Config Flow Test Fixtures
# These fixtures are shared across config flow tests to reduce duplication


@pytest.fixture
def config_flow_flow(hass: HomeAssistant) -> Any:
    """Create an AreaOccupancyConfigFlow instance for testing."""
    flow = AreaOccupancyConfigFlow()
    flow.hass = hass
    return flow


@pytest.fixture
def config_flow_options_flow(
    hass: HomeAssistant, config_flow_mock_config_entry_with_areas: Mock
) -> Any:
    """Create an AreaOccupancyOptionsFlow instance for testing."""
    flow = AreaOccupancyOptionsFlow()
    flow.hass = hass

    # Patch frame reporting to avoid issues with Mock config entries
    # The patch needs to stay active, so we patch it at module level
    with (
        patch("homeassistant.helpers.frame.report_usage", return_value=None),
        patch(
            "homeassistant.helpers.frame.async_suggest_report_issue", return_value=None
        ),
        patch("homeassistant.loader.async_get_issue_tracker", return_value=None),
    ):
        flow.config_entry = config_flow_mock_config_entry_with_areas
        yield flow


@pytest.fixture
def config_flow_base_config(
    hass: HomeAssistant, setup_area_registry: dict[str, str]
) -> dict[str, Any]:
    """Create a base valid configuration for testing."""
    # Use actual area ID from registry (Testing area)
    testing_area_id = setup_area_registry.get("Testing", "testing")
    return {
        CONF_AREA_ID: testing_area_id,
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
        CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
        CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
        CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
        CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
        CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
    }


@pytest.fixture
def config_flow_sample_area(
    hass: HomeAssistant, setup_area_registry: dict[str, str]
) -> dict[str, Any]:
    """Create a minimal sample area configuration."""
    # Use actual area ID from registry (Living Room area)
    living_room_area_id = setup_area_registry.get("Living Room", "living_room")
    return {
        CONF_AREA_ID: living_room_area_id,
        CONF_PURPOSE: "social",
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
    }


@pytest.fixture
def config_flow_sample_area_full(
    hass: HomeAssistant, setup_area_registry: dict[str, str]
) -> dict[str, Any]:
    """Create a sample area configuration with all fields."""
    # Use actual area ID from registry (Living Room area)
    living_room_area_id = setup_area_registry.get("Living Room", "living_room")
    return {
        CONF_AREA_ID: living_room_area_id,
        CONF_PURPOSE: "social",
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        CONF_MEDIA_DEVICES: ["media_player.tv"],
        CONF_DOOR_SENSORS: ["binary_sensor.door1"],
        CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
        CONF_APPLIANCES: ["switch.light"],
        CONF_THRESHOLD: 60.0,
    }


@pytest.fixture
def config_flow_valid_user_input(
    hass: HomeAssistant, setup_area_registry: dict[str, str]
) -> dict[str, Any]:
    """Create valid user input for testing."""
    # Use actual area ID from registry (Living Room area)
    living_room_area_id = setup_area_registry.get("Living Room", "living_room")
    return {
        CONF_AREA_ID: living_room_area_id,
        "motion": {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        },
        "purpose": {},
        "doors": {},
        "windows": {},
        "media": {},
        "appliances": {},
        "environmental": {},
        "wasp_in_box": {},
        "parameters": {CONF_THRESHOLD: 60},
    }


@pytest.fixture
def config_flow_mock_config_entry_with_areas(
    setup_area_registry: dict[str, str],
) -> Mock:
    """Create a mock config entry with multi-area format."""
    entry = Mock(spec=ConfigEntry)
    entry.entry_id = "test_entry_id"
    entry.unique_id = "test_unique_id"
    entry.domain = DOMAIN
    entry.state = ConfigEntryState.LOADED
    entry.disabled_by = None
    entry.setup_lock = Lock()
    # Use actual area ID from registry
    living_room_area_id = setup_area_registry.get("Living Room", "living_room")
    entry.data = {
        CONF_AREAS: [
            {
                CONF_AREA_ID: living_room_area_id,
                CONF_PURPOSE: "social",
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
                CONF_THRESHOLD: 60.0,
            }
        ]
    }
    entry.options = {}
    return entry


@pytest.fixture
def config_flow_mock_config_entry_legacy() -> Mock:
    """Create a mock config entry with legacy single-area format."""
    entry = Mock(spec=ConfigEntry)
    entry.entry_id = "test_entry_id"
    entry.state = ConfigEntryState.LOADED
    entry.data = {
        CONF_AREA_ID: "legacy_area",
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
    }
    entry.options = {}
    return entry


@pytest.fixture
def config_flow_mock_config_entry_legacy_no_name() -> Mock:
    """Create a mock config entry with legacy format but no name."""
    entry = Mock(spec=ConfigEntry)
    entry.entry_id = "test_entry_id"
    entry.state = ConfigEntryState.LOADED
    entry.data = {
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
    }
    entry.options = {}
    return entry


@pytest.fixture
def config_flow_mock_hass_for_schema(hass: HomeAssistant) -> HomeAssistant:
    """Set up hass for schema tests - no longer needed as hass fixture provides real states."""
    # Real hass fixture already has states, no mocking needed
    return hass


@pytest.fixture
def config_flow_mock_entity_registry_for_schema() -> Mock:
    """Create mock entity registry for schema tests."""
    mock_registry = Mock(spec=EntityRegistry)
    mock_registry.entities = {}
    return mock_registry


# Config Flow Helper Functions
# These helper functions reduce code duplication in config flow tests


@contextmanager
def patch_create_schema_context(return_value: dict[str, Any] | None = None):
    """Context manager to patch create_schema for tests."""
    with patch(
        "custom_components.area_occupancy.config_flow.create_schema",
        return_value=return_value or {"test": vol.Required("test")},
    ):
        yield


@contextmanager
def patch_validate_methods_context(
    flow: Any,
    validate_config: Any | None = None,
    validate_duplicate: Any | None = None,
):
    """Context manager to patch validation methods.

    Args:
        flow: The flow instance to patch methods on
        validate_config: If None, patches _validate_config. If False, doesn't patch.
            Otherwise, uses as side_effect.
        validate_duplicate: If None, patches _validate_duplicate_name_internal.
            If False, doesn't patch. Otherwise, uses as side_effect.
    """

    patches = []
    if validate_config is not None:
        if validate_config is False:
            pass  # Don't patch
        else:
            patches.append(
                patch.object(flow, "_validate_config", side_effect=validate_config)
            )
    else:
        patches.append(patch.object(flow, "_validate_config"))

    if validate_duplicate is not None:
        if validate_duplicate is False:
            pass  # Don't patch
        else:
            patches.append(
                patch.object(
                    flow,
                    "_validate_duplicate_name_internal",
                    side_effect=validate_duplicate,
                )
            )
    else:
        patches.append(patch.object(flow, "_validate_duplicate_name_internal"))

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def create_area_config(name: str = "Test Area", **overrides: Any) -> dict[str, Any]:
    """Create area config dict with sensible defaults.

    Args:
        name: Area name (will be converted to area_id)
        **overrides: Any config keys to override

    Returns:
        Area configuration dictionary
    """
    # Convert name to area_id (lowercase, replace spaces with underscores)
    area_id = name.lower().replace(" ", "_")
    config = {
        CONF_AREA_ID: area_id,
        CONF_PURPOSE: "social",
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
    }
    config.update(overrides)
    return config


def create_user_input(name: str = "Test Area", **overrides: Any) -> dict[str, Any]:
    """Create user input dict with sensible defaults.

    Args:
        name: Area name (will be converted to area_id)
        **overrides: Any input keys to override

    Returns:
        User input dictionary
    """
    # Convert name to area_id (lowercase, replace spaces with underscores)
    area_id = name.lower().replace(" ", "_")
    input_dict = {
        CONF_AREA_ID: area_id,
        "motion": {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        },
        "purpose": {},
        "doors": {},
        "windows": {},
        "media": {},
        "appliances": {},
        "environmental": {},
        "wasp_in_box": {},
        "parameters": {CONF_THRESHOLD: 60},
    }
    input_dict.update(overrides)
    return input_dict
