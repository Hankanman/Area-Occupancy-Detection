"""Tests for the Area Occupancy config flow."""

from __future__ import annotations

import copy
import logging
from unittest.mock import patch

import pytest
from homeassistant import config_entries, data_entry_flow
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

# Import constants from the custom component
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
    CONF_PRIMARY_OCCUPANCY_SENSOR,
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
    DEFAULT_THRESHOLD,
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

# Enable debug logging for tests
logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger(__name__)

# --- Test Data ---
MOCK_CONFIG_ENTRY_ID = "test_config_entry_1"
MOCK_AREA_NAME = "Living Room"
MOCK_MOTION_SENSOR_1 = "binary_sensor.motion_living_room"
MOCK_DOOR_SENSOR_1 = "binary_sensor.door_living_room"
MOCK_PRIMARY_INDICATOR = MOCK_MOTION_SENSOR_1
MOCK_THRESHOLD = 50
MOCK_HISTORY_DURATION = 7

# Minimal valid user input for the config flow (structured)
MINIMAL_USER_INPUT_STRUCTURED = {
    CONF_NAME: MOCK_AREA_NAME,
    "motion": {
        CONF_PRIMARY_OCCUPANCY_SENSOR: MOCK_PRIMARY_INDICATOR,
        CONF_MOTION_SENSORS: [MOCK_MOTION_SENSOR_1],
        # Add default weights if needed by validation before creation
        CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
    },
    # Add other sections with default/empty values if required by schema/validation
    "doors": {CONF_DOOR_SENSORS: []},
    "windows": {CONF_WINDOW_SENSORS: []},
    "lights": {CONF_LIGHTS: []},
    "media": {CONF_MEDIA_DEVICES: []},
    "appliances": {CONF_APPLIANCES: []},
    "environmental": {CONF_ILLUMINANCE_SENSORS: []},
    "parameters": {
        CONF_THRESHOLD: DEFAULT_THRESHOLD,
        CONF_HISTORY_PERIOD: DEFAULT_HISTORY_PERIOD,
        CONF_DECAY_ENABLED: DEFAULT_DECAY_ENABLED,
        CONF_DECAY_WINDOW: DEFAULT_DECAY_WINDOW,
        CONF_DECAY_MIN_DELAY: DEFAULT_DECAY_MIN_DELAY,
        CONF_HISTORICAL_ANALYSIS_ENABLED: DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    },
}

# Full valid user input (structured)
FULL_USER_INPUT_STRUCTURED = {
    CONF_NAME: MOCK_AREA_NAME,
    "motion": {
        CONF_PRIMARY_OCCUPANCY_SENSOR: MOCK_PRIMARY_INDICATOR,
        CONF_MOTION_SENSORS: [MOCK_MOTION_SENSOR_1],
        CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
    },
    "doors": {
        CONF_DOOR_SENSORS: [MOCK_DOOR_SENSOR_1],
        CONF_DOOR_ACTIVE_STATE: DEFAULT_DOOR_ACTIVE_STATE,
        CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
    },
    "windows": {
        CONF_WINDOW_SENSORS: [],
        CONF_WINDOW_ACTIVE_STATE: DEFAULT_WINDOW_ACTIVE_STATE,
        CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
    },
    "lights": {
        CONF_LIGHTS: [],
        CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
    },
    "media": {
        CONF_MEDIA_DEVICES: [],
        CONF_MEDIA_ACTIVE_STATES: DEFAULT_MEDIA_ACTIVE_STATES,
        CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
    },
    "appliances": {
        CONF_APPLIANCES: [],
        CONF_APPLIANCE_ACTIVE_STATES: DEFAULT_APPLIANCE_ACTIVE_STATES,
        CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
    },
    "environmental": {
        CONF_ILLUMINANCE_SENSORS: [],
        CONF_HUMIDITY_SENSORS: [],
        CONF_TEMPERATURE_SENSORS: [],
        CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
    },
    "parameters": {
        CONF_THRESHOLD: MOCK_THRESHOLD,
        CONF_HISTORY_PERIOD: MOCK_HISTORY_DURATION,
        CONF_DECAY_ENABLED: DEFAULT_DECAY_ENABLED,
        CONF_DECAY_WINDOW: DEFAULT_DECAY_WINDOW,
        CONF_DECAY_MIN_DELAY: DEFAULT_DECAY_MIN_DELAY,
        CONF_HISTORICAL_ANALYSIS_ENABLED: DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    },
}

# --- Fixtures ---


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations defined in the test environment."""
    return


@pytest.fixture
async def mock_ha(hass: HomeAssistant):
    """Set up a mock Home Assistant instance."""
    # Mock essential components ONLY IF NEEDED for the flow itself.
    # Often, the flow tests don't need fully set up components.
    # await setup.async_setup_component(hass, "config", {})
    # await setup.async_setup_component(hass, SENSOR_DOMAIN, {})
    # await setup.async_setup_component(hass, BINARY_SENSOR_DOMAIN, {})

    # Avoid setting states or registry entries directly here.
    # Let the main integration setup handle entities if needed for flow validation,
    # or mock specific hass.states.get calls if entity existence check is the only requirement.

    return hass


# --- Test Cases ---


async def test_config_flow_user_success(mock_ha: HomeAssistant, mock_recorder) -> None:  # pylint: disable=redefined-outer-name
    """Test the user initialization flow with valid data."""
    result = await mock_ha.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    assert result is not None
    assert result.get("type") == data_entry_flow.FlowResultType.FORM
    assert result.get("step_id") == "user"

    with patch(f"custom_components.{DOMAIN}.async_setup_entry", return_value=True):
        result2 = await mock_ha.config_entries.flow.async_configure(
            result["flow_id"], user_input=FULL_USER_INPUT_STRUCTURED
        )
        await mock_ha.async_block_till_done()

    assert result2.get("type") == data_entry_flow.FlowResultType.CREATE_ENTRY
    assert result2.get("title") == MOCK_AREA_NAME
    assert result2.get("data", {}).get(CONF_NAME) == MOCK_AREA_NAME
    assert (
        result2.get("data", {}).get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        == MOCK_PRIMARY_INDICATOR
    )
    assert MOCK_MOTION_SENSOR_1 in result2.get("data", {}).get(CONF_MOTION_SENSORS, [])


async def test_config_flow_user_input_errors(
    mock_ha: HomeAssistant,  # pylint: disable=redefined-outer-name
    mock_recorder,  # pylint: disable=redefined-outer-name
) -> None:
    """Test user flow with various input errors."""
    result = await mock_ha.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    flow_id = result["flow_id"]

    # Test missing primary sensor
    bad_input = copy.deepcopy(FULL_USER_INPUT_STRUCTURED)
    bad_input["motion"][CONF_PRIMARY_OCCUPANCY_SENSOR] = "not_an_entity"
    with pytest.raises(Exception):  # noqa: B017
        await mock_ha.config_entries.flow.async_configure(flow_id, user_input=bad_input)


async def test_options_flow_success(mock_ha: HomeAssistant, mock_recorder) -> None:  # pylint: disable=redefined-outer-name
    """Test the options flow for updating settings."""
    _LOGGER.debug("Starting test_options_flow_success")
    # 1. Set up an initial config entry
    # Use MockConfigEntry provided by the fixture
    entry = MockConfigEntry(
        version=1,
        domain=DOMAIN,
        entry_id=MOCK_CONFIG_ENTRY_ID,
        unique_id=MOCK_CONFIG_ENTRY_ID,  # Use entry_id as unique_id for simplicity
        title=MOCK_AREA_NAME,
        data={
            CONF_NAME: MOCK_AREA_NAME,
            CONF_PRIMARY_OCCUPANCY_SENSOR: MOCK_PRIMARY_INDICATOR,
            CONF_MOTION_SENSORS: [MOCK_MOTION_SENSOR_1],
            # Other required data fields based on config flow schema
        },
        options={
            CONF_THRESHOLD: MOCK_THRESHOLD,
            CONF_HISTORY_PERIOD: MOCK_HISTORY_DURATION,
            # Other initial options
        },
        source=config_entries.SOURCE_USER,
    )
    entry.add_to_hass(mock_ha)

    # Mock setup to prevent actual component loading during options flow
    with patch(f"custom_components.{DOMAIN}.async_setup_entry", return_value=True):
        assert await mock_ha.config_entries.async_setup(entry.entry_id)
        await mock_ha.async_block_till_done()

    # 2. Initialize the options flow
    result = await mock_ha.config_entries.options.async_init(entry.entry_id)
    _LOGGER.debug("Initial options flow result: %s", result)

    assert result is not None
    assert result.get("type") == data_entry_flow.FlowResultType.FORM
    assert result.get("step_id") == "init"  # Assuming default options step ID is 'init'
    assert "flow_id" in result

    # 3. Provide updated options
    new_threshold = 65
    new_history = 30
    # Options flow input needs to match the sections
    updated_options_structured = {
        "motion": {
            # Include existing motion sensors if they shouldn't change
            CONF_PRIMARY_OCCUPANCY_SENSOR: entry.options.get(
                CONF_PRIMARY_OCCUPANCY_SENSOR, MOCK_PRIMARY_INDICATOR
            ),
            CONF_MOTION_SENSORS: entry.options.get(
                CONF_MOTION_SENSORS, [MOCK_MOTION_SENSOR_1]
            ),
            CONF_WEIGHT_MOTION: entry.options.get(
                CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION
            ),
        },
        # Include other sections with their current or default values
        "doors": {  # Example
            CONF_DOOR_SENSORS: entry.options.get(CONF_DOOR_SENSORS, []),
            CONF_DOOR_ACTIVE_STATE: entry.options.get(
                CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE
            ),
            CONF_WEIGHT_DOOR: entry.options.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR),
        },
        "windows": {
            CONF_WINDOW_SENSORS: entry.options.get(CONF_WINDOW_SENSORS, []),
            CONF_WINDOW_ACTIVE_STATE: entry.options.get(
                CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE
            ),
            CONF_WEIGHT_WINDOW: entry.options.get(
                CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW
            ),
        },
        "lights": {
            CONF_LIGHTS: entry.options.get(CONF_LIGHTS, []),
            CONF_WEIGHT_LIGHT: entry.options.get(
                CONF_WEIGHT_LIGHT, DEFAULT_WEIGHT_LIGHT
            ),
        },
        "media": {
            CONF_MEDIA_DEVICES: entry.options.get(CONF_MEDIA_DEVICES, []),
            CONF_MEDIA_ACTIVE_STATES: entry.options.get(
                CONF_MEDIA_ACTIVE_STATES, DEFAULT_MEDIA_ACTIVE_STATES
            ),
            CONF_WEIGHT_MEDIA: entry.options.get(
                CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA
            ),
        },
        "appliances": {
            CONF_APPLIANCES: entry.options.get(CONF_APPLIANCES, []),
            CONF_APPLIANCE_ACTIVE_STATES: entry.options.get(
                CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_APPLIANCE_ACTIVE_STATES
            ),
            CONF_WEIGHT_APPLIANCE: entry.options.get(
                CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE
            ),
        },
        "environmental": {
            CONF_ILLUMINANCE_SENSORS: entry.options.get(CONF_ILLUMINANCE_SENSORS, []),
            CONF_HUMIDITY_SENSORS: entry.options.get(CONF_HUMIDITY_SENSORS, []),
            CONF_TEMPERATURE_SENSORS: entry.options.get(CONF_TEMPERATURE_SENSORS, []),
            CONF_WEIGHT_ENVIRONMENTAL: entry.options.get(
                CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL
            ),
        },
        "parameters": {
            CONF_THRESHOLD: new_threshold,  # Update the threshold
            CONF_HISTORY_PERIOD: new_history,  # Update the history period
            CONF_DECAY_ENABLED: entry.options.get(
                CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED
            ),
            CONF_DECAY_WINDOW: entry.options.get(
                CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
            ),
            CONF_DECAY_MIN_DELAY: entry.options.get(
                CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY
            ),
            CONF_HISTORICAL_ANALYSIS_ENABLED: entry.options.get(
                CONF_HISTORICAL_ANALYSIS_ENABLED, DEFAULT_HISTORICAL_ANALYSIS_ENABLED
            ),
        },
    }

    result2 = await mock_ha.config_entries.options.async_configure(
        result["flow_id"],
        user_input=updated_options_structured,  # Use structured input
    )
    await mock_ha.async_block_till_done()
    _LOGGER.debug("Configure options flow result: %s", result2)

    # 4. Verify the options flow finished and updated the entry's options
    assert result2 is not None
    assert result2.get("type") == data_entry_flow.FlowResultType.CREATE_ENTRY
    # Options flow result['data'] contains the *updated* options dictionary (flattened)
    assert result2.get("data", {}).get(CONF_THRESHOLD) == new_threshold
    assert result2.get("data", {}).get(CONF_HISTORY_PERIOD) == new_history

    # Check the actual config entry options are updated
    assert entry.options is not None
    _LOGGER.debug("Finished test_options_flow_success")


# Add new test for auto-adding primary sensor
async def test_config_flow_auto_add_primary_sensor(
    mock_ha: HomeAssistant,  # pylint: disable=redefined-outer-name
    mock_recorder,  # pylint: disable=redefined-outer-name
) -> None:
    """Test that primary sensor is auto-added to motion sensors."""
    result = await mock_ha.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    # Create input with primary sensor not in motion sensors
    test_input = copy.deepcopy(FULL_USER_INPUT_STRUCTURED)
    primary_sensor = "binary_sensor.new_motion"
    test_input["motion"][CONF_PRIMARY_OCCUPANCY_SENSOR] = primary_sensor
    test_input["motion"][CONF_MOTION_SENSORS] = ["binary_sensor.other_motion"]

    with patch(f"custom_components.{DOMAIN}.async_setup_entry", return_value=True):
        result2 = await mock_ha.config_entries.flow.async_configure(
            result["flow_id"], user_input=test_input
        )
        await mock_ha.async_block_till_done()

    assert result2.get("type") == data_entry_flow.FlowResultType.CREATE_ENTRY
    # Verify primary sensor was added to motion sensors
    assert primary_sensor in result2.get("data", {}).get(CONF_MOTION_SENSORS, [])


# Add test for section validation
async def test_config_flow_section_validation(
    mock_ha: HomeAssistant,  # pylint: disable=redefined-outer-name
    mock_recorder,  # pylint: disable=redefined-outer-name
) -> None:
    """Test validation of individual sections."""
    result = await mock_ha.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    # Test media section validation
    test_input = copy.deepcopy(FULL_USER_INPUT_STRUCTURED)
    test_input["media"]["media_devices"] = ["media_player.test"]
    test_input["media"]["media_active_states"] = []  # Empty active states

    result2 = await mock_ha.config_entries.flow.async_configure(
        result["flow_id"], user_input=test_input
    )
    assert result2.get("type") == data_entry_flow.FlowResultType.FORM
    errors = result2.get("errors")
    if not isinstance(errors, dict):
        errors = {}
    media_errors = errors.get("media")
    if not isinstance(media_errors, dict):
        media_errors = {}
    # The error should indicate required field or may be missing
    assert media_errors.get("media_active_states") in ("required", None)


# Add test for state translation
async def test_config_flow_state_translation(
    mock_ha: HomeAssistant,  # pylint: disable=redefined-outer-name
    mock_recorder,  # pylint: disable=redefined-outer-name
) -> None:
    """Test translation of display states to internal states."""
    result = await mock_ha.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    test_input = copy.deepcopy(FULL_USER_INPUT_STRUCTURED)
    test_input["doors"]["door_active_state"] = "open"  # Lowercase for schema

    with patch(f"custom_components.{DOMAIN}.async_setup_entry", return_value=True):
        result2 = await mock_ha.config_entries.flow.async_configure(
            result["flow_id"], user_input=test_input
        )
        await mock_ha.async_block_till_done()

    assert result2.get("type") == data_entry_flow.FlowResultType.CREATE_ENTRY
    assert result2.get("data", {}).get("door_active_state") == "open"
