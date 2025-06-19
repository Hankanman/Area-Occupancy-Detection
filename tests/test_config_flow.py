"""Test config flow."""

from __future__ import annotations

import copy
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant import config_entries, data_entry_flow
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

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
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
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
from custom_components.area_occupancy.config_flow import (
    AreaOccupancyConfigFlow,
    AreaOccupancyOptionsFlow,
)

# Enable debug logging for tests
logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger(__name__)

# --- Test Data ---
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
        CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
    },
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
    "wasp_in_box": {
        CONF_WASP_ENABLED: True,
        CONF_WASP_MOTION_TIMEOUT: DEFAULT_WASP_MOTION_TIMEOUT,
        CONF_WASP_WEIGHT: DEFAULT_WASP_WEIGHT,
        CONF_WASP_MAX_DURATION: DEFAULT_WASP_MAX_DURATION,
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

# Note: Using fixtures from conftest.py:
# - mock_recorder
# - mock_config_entry
# - setup_test_entities
# - init_integration


class TestAreaOccupancyConfigFlow:
    """Test the Area Occupancy config flow."""

    @pytest.fixture
    def flow(self, hass: HomeAssistant) -> AreaOccupancyConfigFlow:
        """Create a config flow instance."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = hass
        return flow

    async def test_form_user_step(self, flow: AreaOccupancyConfigFlow) -> None:
        """Test the user step shows the form."""
        result = await flow.async_step_user()
        
        assert result["type"] == "form"
        assert result["step_id"] == "user"
        assert result["errors"] == {}

    async def test_form_user_step_with_valid_input(
        self, flow: AreaOccupancyConfigFlow
    ) -> None:
        """Test user step with valid input."""
        user_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion_1", "binary_sensor.motion_2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion_1",
            CONF_THRESHOLD: 60,
        }

        with patch.object(flow, "async_set_unique_id"), \
             patch.object(flow, "_abort_if_unique_id_configured"):
            
            result = await flow.async_step_user(user_input)
            
            assert result["type"] == "create_entry"
            assert result["title"] == "Living Room"
            assert result["data"] == user_input

    async def test_form_user_step_with_invalid_input(
        self, flow: AreaOccupancyConfigFlow
    ) -> None:
        """Test user step with invalid input."""
        user_input = {
            CONF_NAME: "",  # Empty name should cause error
            CONF_MOTION_SENSORS: [],  # Empty sensors should cause error
        }

        result = await flow.async_step_user(user_input)
        
        assert result["type"] == "form"
        assert result["errors"] is not None

    async def test_form_user_step_duplicate_entry(
        self, flow: AreaOccupancyConfigFlow
    ) -> None:
        """Test user step with duplicate entry."""
        user_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion_1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion_1",
        }

        with patch.object(flow, "async_set_unique_id"), \
             patch.object(flow, "_abort_if_unique_id_configured", side_effect=config_entries.FlowAborted("already_configured")):
            
            with pytest.raises(config_entries.FlowAborted):
                await flow.async_step_user(user_input)

    async def test_form_with_all_sensor_types(
        self, flow: AreaOccupancyConfigFlow
    ) -> None:
        """Test form with all sensor types filled."""
        user_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion_1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion_1",
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_APPLIANCES: ["switch.dishwasher"],
            CONF_ILLUMINANCE_SENSORS: ["sensor.light_level"],
            CONF_HUMIDITY_SENSORS: ["sensor.humidity"],
            CONF_TEMPERATURE_SENSORS: ["sensor.temperature"],
            CONF_DOOR_SENSORS: ["binary_sensor.door"],
            CONF_WINDOW_SENSORS: ["binary_sensor.window"],
            CONF_LIGHTS: ["light.living_room"],
            CONF_THRESHOLD: 75,
            CONF_HISTORY_PERIOD: 14,
            CONF_DECAY_ENABLED: True,
            CONF_DECAY_WINDOW: 30,
        }

        with patch.object(flow, "async_set_unique_id"), \
             patch.object(flow, "_abort_if_unique_id_configured"):
            
            result = await flow.async_step_user(user_input)
            
            assert result["type"] == "create_entry"
            assert result["data"] == user_input

    async def test_async_get_options_flow(
        self, hass: HomeAssistant
    ) -> None:
        """Test get options flow."""
        config_entry = MockConfigEntry(
            domain=DOMAIN,
            data={
                CONF_NAME: "Test Area",
                CONF_MOTION_SENSORS: ["binary_sensor.motion"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion",
            },
            options={CONF_THRESHOLD: 60},
        )
        config_entry.add_to_hass(hass)

        flow = AreaOccupancyConfigFlow()
        flow.hass = hass
        
        options_flow = flow.async_get_options_flow(config_entry)
        assert isinstance(options_flow, AreaOccupancyOptionsFlow)


class TestAreaOccupancyOptionsFlow:
    """Test the options flow handler."""

    @pytest.fixture
    def config_entry(self) -> MockConfigEntry:
        """Create a mock config entry."""
        return MockConfigEntry(
            domain=DOMAIN,
            data={
                CONF_NAME: "Test Area",
                CONF_MOTION_SENSORS: ["binary_sensor.motion"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion",
            },
            options={
                CONF_THRESHOLD: 60,
                CONF_HISTORY_PERIOD: 7,
                CONF_DECAY_ENABLED: False,
            },
        )

    @pytest.fixture
    def options_flow(
        self, hass: HomeAssistant, config_entry: MockConfigEntry
    ) -> AreaOccupancyOptionsFlow:
        """Create an options flow instance."""
        config_entry.add_to_hass(hass)
        flow = AreaOccupancyOptionsFlow(config_entry)
        flow.hass = hass
        return flow

    async def test_options_init_step(
        self, options_flow: AreaOccupancyOptionsFlow
    ) -> None:
        """Test the init step of options flow."""
        result = await options_flow.async_step_init()
        
        assert result["type"] == "form"
        assert result["step_id"] == "init"
        assert result["errors"] == {}
        
        # Check that current values are pre-filled
        assert result["data_schema"]

    async def test_options_init_with_valid_input(
        self, options_flow: AreaOccupancyOptionsFlow
    ) -> None:
        """Test options init with valid input."""
        user_input = {
            CONF_THRESHOLD: 75,
            CONF_HISTORY_PERIOD: 14,
            CONF_DECAY_ENABLED: True,
            CONF_DECAY_WINDOW: 45,
        }

        result = await options_flow.async_step_init(user_input)
        
        assert result["type"] == "create_entry"
        assert result["data"] == user_input

    async def test_options_init_with_invalid_threshold(
        self, options_flow: AreaOccupancyOptionsFlow
    ) -> None:
        """Test options init with invalid threshold."""
        user_input = {
            CONF_THRESHOLD: 150,  # Invalid, should be 1-99
            CONF_HISTORY_PERIOD: 7,
            CONF_DECAY_ENABLED: False,
        }

        result = await options_flow.async_step_init(user_input)
        
        assert result["type"] == "form"
        assert CONF_THRESHOLD in result["errors"]

    async def test_options_init_with_invalid_history_period(
        self, options_flow: AreaOccupancyOptionsFlow
    ) -> None:
        """Test options init with invalid history period."""
        user_input = {
            CONF_THRESHOLD: 60,
            CONF_HISTORY_PERIOD: 0,  # Invalid, should be >= 1
            CONF_DECAY_ENABLED: False,
        }

        result = await options_flow.async_step_init(user_input)
        
        assert result["type"] == "form"
        assert CONF_HISTORY_PERIOD in result["errors"]

    async def test_options_init_with_boundary_values(
        self, options_flow: AreaOccupancyOptionsFlow
    ) -> None:
        """Test options init with boundary values."""
        user_input = {
            CONF_THRESHOLD: 1,  # Minimum value
            CONF_HISTORY_PERIOD: 1,  # Minimum value
            CONF_DECAY_ENABLED: True,
            CONF_DECAY_WINDOW: 1,  # Minimum value
        }

        result = await options_flow.async_step_init(user_input)
        
        assert result["type"] == "create_entry"
        assert result["data"] == user_input

    async def test_options_init_with_maximum_values(
        self, options_flow: AreaOccupancyOptionsFlow
    ) -> None:
        """Test options init with maximum boundary values."""
        user_input = {
            CONF_THRESHOLD: 99,  # Maximum value
            CONF_HISTORY_PERIOD: 30,  # Maximum value
            CONF_DECAY_ENABLED: True,
            CONF_DECAY_WINDOW: 180,  # Maximum value
        }

        result = await options_flow.async_step_init(user_input)
        
        assert result["type"] == "create_entry"
        assert result["data"] == user_input

    async def test_options_flow_with_defaults(
        self, hass: HomeAssistant
    ) -> None:
        """Test options flow uses defaults when no options set."""
        config_entry = MockConfigEntry(
            domain=DOMAIN,
            data={
                CONF_NAME: "Test Area",
                CONF_MOTION_SENSORS: ["binary_sensor.motion"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion",
            },
            options={},  # No options set
        )
        config_entry.add_to_hass(hass)

        flow = AreaOccupancyOptionsFlow(config_entry)
        flow.hass = hass

        result = await flow.async_step_init()
        
        assert result["type"] == "form"
        # Should use defaults when no options are configured

    async def test_options_flow_preserves_unchanged_values(
        self, options_flow: AreaOccupancyOptionsFlow
    ) -> None:
        """Test options flow preserves unchanged values."""
        # Only change threshold, leave others unchanged
        user_input = {
            CONF_THRESHOLD: 80,
            CONF_HISTORY_PERIOD: 7,  # Same as current
            CONF_DECAY_ENABLED: False,  # Same as current
        }

        result = await options_flow.async_step_init(user_input)
        
        assert result["type"] == "create_entry"
        assert result["data"][CONF_THRESHOLD] == 80
        assert result["data"][CONF_HISTORY_PERIOD] == 7
        assert result["data"][CONF_DECAY_ENABLED] is False

    async def test_schema_validation_errors(
        self, options_flow: AreaOccupancyOptionsFlow
    ) -> None:
        """Test schema validation catches various error types."""
        # Test with completely invalid data types
        invalid_inputs = [
            {CONF_THRESHOLD: "not_a_number"},
            {CONF_HISTORY_PERIOD: -5},
            {CONF_DECAY_WINDOW: "invalid"},
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises((vol.Invalid, ValueError, TypeError)):
                # This should trigger validation errors
                await options_flow.async_step_init(invalid_input)


# Integration test
async def test_options_flow_success(hass: HomeAssistant) -> None:
    """Test successful options flow integration."""
    # Create and add config entry
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion",
        },
        options={CONF_THRESHOLD: 50},
    )
    config_entry.add_to_hass(hass)

    # Start options flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    
    assert result["type"] == "form"
    assert result["step_id"] == "init"

    # Submit new options
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_THRESHOLD: 70,
            CONF_HISTORY_PERIOD: 14,
            CONF_DECAY_ENABLED: True,
            CONF_DECAY_WINDOW: 30,
        },
    )

    assert result["type"] == "create_entry"
    assert result["data"][CONF_THRESHOLD] == 70
    assert result["data"][CONF_HISTORY_PERIOD] == 14
    assert result["data"][CONF_DECAY_ENABLED] is True
    assert result["data"][CONF_DECAY_WINDOW] == 30
