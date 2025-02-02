"""Test the Area Occupancy Detection config and options flow."""

from __future__ import annotations

from unittest.mock import patch
from homeassistant import config_entries, data_entry_flow
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from custom_components.area_occupancy.const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_WEIGHT_MOTION,
    CONF_THRESHOLD,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_ENABLED,
    CONF_DOOR_SENSORS,
    CONF_DOOR_ACTIVE_STATE,
    CONF_WEIGHT_DOOR,
    CONF_WINDOW_SENSORS,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WEIGHT_WINDOW,
    CONF_LIGHTS,
    CONF_WEIGHT_LIGHT,
    CONF_MEDIA_DEVICES,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_WEIGHT_MEDIA,
    CONF_APPLIANCES,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_WEIGHT_APPLIANCE,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
)


async def test_form(hass: HomeAssistant, mock_recorder, setup_test_entities) -> None:
    """Test we get the form."""
    # setup_test_entities is a fixture that has already run, no need to await it

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == "form"
    assert result["errors"] == {}

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data",
        return_value={},
    ), patch(
        "custom_components.area_occupancy.async_setup_entry",
        return_value=True,
    ):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Test Area",
                "motion": {
                    CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                    CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
                },
                "doors": {
                    CONF_DOOR_SENSORS: [],
                    CONF_DOOR_ACTIVE_STATE: "closed",
                    CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
                },
                "windows": {
                    CONF_WINDOW_SENSORS: [],
                    CONF_WINDOW_ACTIVE_STATE: "closed",
                    CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
                },
                "lights": {
                    CONF_LIGHTS: [],
                    CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
                },
                "media": {
                    CONF_MEDIA_DEVICES: [],
                    CONF_MEDIA_ACTIVE_STATES: ["playing", "paused"],
                    CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
                },
                "appliances": {
                    CONF_APPLIANCES: [],
                    CONF_APPLIANCE_ACTIVE_STATES: ["on"],
                    CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
                },
                "environmental": {
                    CONF_ILLUMINANCE_SENSORS: [],
                    CONF_HUMIDITY_SENSORS: [],
                    CONF_TEMPERATURE_SENSORS: [],
                    CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
                },
                "parameters": {
                    CONF_THRESHOLD: 50,
                    CONF_HISTORY_PERIOD: 7,
                    CONF_DECAY_ENABLED: True,
                },
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] == "create_entry"
    assert result2["title"] == "Test Area"
    assert result2["data"][CONF_NAME] == "Test Area"
    assert result2["data"]["motion"][CONF_MOTION_SENSORS] == ["binary_sensor.motion1"]
    assert result2["data"]["motion"][CONF_WEIGHT_MOTION] == DEFAULT_WEIGHT_MOTION
    assert result2["data"]["parameters"][CONF_THRESHOLD] == 50


async def test_options_flow_init(hass: HomeAssistant, init_integration) -> None:
    """Test config flow options."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    assert result["type"] == data_entry_flow.FlowResultType.FORM
    assert result["step_id"] == "init"


async def test_options_flow_update(hass: HomeAssistant, init_integration) -> None:
    """Test updating options."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    new_options = {
        "motion": {
            CONF_MOTION_SENSORS: ["binary_sensor.motion_new"],
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
        },
        "doors": {
            CONF_DOOR_SENSORS: [],
            CONF_DOOR_ACTIVE_STATE: "closed",
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
        },
        "windows": {
            CONF_WINDOW_SENSORS: [],
            CONF_WINDOW_ACTIVE_STATE: "closed",
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
        },
        "lights": {
            CONF_LIGHTS: [],
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
        },
        "media": {
            CONF_MEDIA_DEVICES: [],
            CONF_MEDIA_ACTIVE_STATES: ["playing", "paused"],
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
        },
        "appliances": {
            CONF_APPLIANCES: [],
            CONF_APPLIANCE_ACTIVE_STATES: ["on"],
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
        },
        "environmental": {
            CONF_ILLUMINANCE_SENSORS: [],
            CONF_HUMIDITY_SENSORS: [],
            CONF_TEMPERATURE_SENSORS: [],
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        },
        "parameters": {
            CONF_THRESHOLD: 70,
            CONF_HISTORY_PERIOD: 14,
            CONF_DECAY_ENABLED: False,
        },
    }

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data",
        return_value={},
    ):
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input=new_options
        )
        await hass.async_block_till_done()

    assert result["type"] == data_entry_flow.FlowResultType.CREATE_ENTRY

    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    assert config_entry.data["motion"][CONF_MOTION_SENSORS] == [
        "binary_sensor.motion_new"
    ]
    assert config_entry.data["parameters"][CONF_THRESHOLD] == 70
    assert config_entry.data["parameters"][CONF_DECAY_ENABLED] is False


async def test_options_flow_validation(hass: HomeAssistant, init_integration) -> None:
    """Test options flow validation."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    invalid_options = {
        "motion": {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
        },
        "doors": {
            CONF_DOOR_SENSORS: [],
            CONF_DOOR_ACTIVE_STATE: "closed",
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
        },
        "windows": {
            CONF_WINDOW_SENSORS: [],
            CONF_WINDOW_ACTIVE_STATE: "closed",
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
        },
        "lights": {
            CONF_LIGHTS: [],
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
        },
        "media": {
            CONF_MEDIA_DEVICES: [],
            CONF_MEDIA_ACTIVE_STATES: ["playing", "paused"],
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
        },
        "appliances": {
            CONF_APPLIANCES: [],
            CONF_APPLIANCE_ACTIVE_STATES: ["on"],
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
        },
        "environmental": {
            CONF_ILLUMINANCE_SENSORS: [],
            CONF_HUMIDITY_SENSORS: [],
            CONF_TEMPERATURE_SENSORS: [],
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        },
        "parameters": {
            CONF_THRESHOLD: 150,  # Invalid threshold (above 100)
            CONF_HISTORY_PERIOD: 7,
            CONF_DECAY_ENABLED: True,
        },
    }

    try:
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input=invalid_options
        )
        await hass.async_block_till_done()
    except data_entry_flow.InvalidData as err:
        assert "threshold" in str(err)  # Verify the error is about the threshold
        return

    assert False, "Expected InvalidData error was not raised"


async def test_options_flow_preserve_name(
    hass: HomeAssistant, init_integration
) -> None:
    """Test that area name is preserved when updating options."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    original_name = config_entry.data[CONF_NAME]

    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    new_options = {
        "motion": {
            CONF_MOTION_SENSORS: ["binary_sensor.motion_new"],
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
        },
        "doors": {
            CONF_DOOR_SENSORS: [],
            CONF_DOOR_ACTIVE_STATE: "closed",
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
        },
        "windows": {
            CONF_WINDOW_SENSORS: [],
            CONF_WINDOW_ACTIVE_STATE: "closed",
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
        },
        "lights": {
            CONF_LIGHTS: [],
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
        },
        "media": {
            CONF_MEDIA_DEVICES: [],
            CONF_MEDIA_ACTIVE_STATES: ["playing", "paused"],
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
        },
        "appliances": {
            CONF_APPLIANCES: [],
            CONF_APPLIANCE_ACTIVE_STATES: ["on"],
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
        },
        "environmental": {
            CONF_ILLUMINANCE_SENSORS: [],
            CONF_HUMIDITY_SENSORS: [],
            CONF_TEMPERATURE_SENSORS: [],
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        },
        "parameters": {
            CONF_THRESHOLD: 60,
            CONF_HISTORY_PERIOD: 7,
            CONF_DECAY_ENABLED: True,
        },
    }

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data",
        return_value={},
    ):
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input=new_options
        )
        await hass.async_block_till_done()

    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    assert config_entry.data[CONF_NAME] == original_name
    assert config_entry.data["motion"][CONF_MOTION_SENSORS] == [
        "binary_sensor.motion_new"
    ]
    assert config_entry.data["parameters"][CONF_THRESHOLD] == 60


async def test_config_entry_update_listener(
    hass: HomeAssistant, init_integration
) -> None:
    """Test the update listener."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]

    # Update config entry with new options
    new_options = {
        CONF_NAME: config_entry.data[CONF_NAME],
        "motion": {
            CONF_MOTION_SENSORS: ["binary_sensor.motion_new"],
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
        },
        "doors": {
            CONF_DOOR_SENSORS: [],
            CONF_DOOR_ACTIVE_STATE: "closed",
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
        },
        "windows": {
            CONF_WINDOW_SENSORS: [],
            CONF_WINDOW_ACTIVE_STATE: "closed",
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
        },
        "lights": {
            CONF_LIGHTS: [],
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
        },
        "media": {
            CONF_MEDIA_DEVICES: [],
            CONF_MEDIA_ACTIVE_STATES: ["playing", "paused"],
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
        },
        "appliances": {
            CONF_APPLIANCES: [],
            CONF_APPLIANCE_ACTIVE_STATES: ["on"],
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
        },
        "environmental": {
            CONF_ILLUMINANCE_SENSORS: [],
            CONF_HUMIDITY_SENSORS: [],
            CONF_TEMPERATURE_SENSORS: [],
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        },
        "parameters": {
            CONF_THRESHOLD: 80,
            CONF_HISTORY_PERIOD: 7,
            CONF_DECAY_ENABLED: True,
        },
    }

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data",
        return_value={},
    ), patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator.async_initialize_states",
        return_value=None,
    ):
        hass.config_entries.async_update_entry(config_entry, data=new_options)
        await hass.async_block_till_done()

    # Verify coordinator was updated with new settings
    coordinator = hass.data[DOMAIN][config_entry.entry_id]["coordinator"]
    assert coordinator.config["motion"][CONF_MOTION_SENSORS] == [
        "binary_sensor.motion_new"
    ]
    assert coordinator.config["parameters"][CONF_THRESHOLD] == 80
