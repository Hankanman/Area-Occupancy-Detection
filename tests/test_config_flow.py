"""Test the Area Occupancy Detection config and options flow."""

from unittest.mock import patch
from homeassistant import config_entries, data_entry_flow
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from custom_components.area_occupancy.const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_THRESHOLD,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_ENABLED,
    DEFAULT_THRESHOLD,
)


async def test_form(hass: HomeAssistant) -> None:
    """Test we get the form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == "form"
    assert result["errors"] == {}

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data",
        return_value={},
    ):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Test Area",
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_THRESHOLD: 50,
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] == "create_entry"
    assert result2["title"] == "Test Area"
    assert result2["data"] == {
        CONF_NAME: "Test Area",
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        CONF_THRESHOLD: 50,
    }


async def test_options_flow_init(hass: HomeAssistant, init_integration) -> None:
    """Test config flow options."""
    # Get the config entry that was set up
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]

    # Start the options flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result["type"] == data_entry_flow.FlowResultType.FORM
    assert result["step_id"] == "init"

    # Verify form shows current values
    assert (
        result["data_schema"].schema[CONF_MOTION_SENSORS].default
        == config_entry.data[CONF_MOTION_SENSORS]
    )
    assert result["data_schema"].schema[
        CONF_THRESHOLD
    ].default == config_entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)


async def test_options_flow_update(hass: HomeAssistant, init_integration) -> None:
    """Test updating options."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]

    # Start options flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    # Update with new values
    new_options = {
        CONF_MOTION_SENSORS: ["binary_sensor.motion_new"],
        CONF_ILLUMINANCE_SENSORS: ["sensor.illuminance_new"],
        CONF_THRESHOLD: 0.7,
        CONF_HISTORY_PERIOD: 14,
        CONF_DECAY_ENABLED: False,
    }

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data"
    ):
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input=new_options
        )
        await hass.async_block_till_done()

    assert result["type"] == data_entry_flow.FlowResultType.CREATE_ENTRY

    # Verify config entry was updated
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    assert config_entry.data[CONF_MOTION_SENSORS] == ["binary_sensor.motion_new"]
    assert config_entry.data[CONF_THRESHOLD] == 0.7
    assert config_entry.data[CONF_DECAY_ENABLED] is False


async def test_options_flow_validation(hass: HomeAssistant, init_integration) -> None:
    """Test options flow validation."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]

    # Start options flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    # Test invalid threshold
    invalid_options = {
        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        CONF_THRESHOLD: 1.5,  # Invalid threshold
    }

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input=invalid_options
    )

    assert result["type"] == data_entry_flow.FlowResultType.FORM
    assert result["errors"] == {"threshold": "invalid_threshold"}


async def test_config_entry_update_listener(
    hass: HomeAssistant, init_integration
) -> None:
    """Test the update listener."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]

    # Update config entry with new options
    new_options = {
        CONF_MOTION_SENSORS: ["binary_sensor.motion_new"],
        CONF_THRESHOLD: 0.8,
    }

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data"
    ):
        hass.config_entries.async_update_entry(
            config_entry, data={CONF_NAME: config_entry.data[CONF_NAME], **new_options}
        )
        await hass.async_block_till_done()

    # Verify coordinator was updated with new settings
    coordinator = hass.data[DOMAIN][config_entry.entry_id]["coordinator"]
    assert coordinator.config[CONF_MOTION_SENSORS] == ["binary_sensor.motion_new"]
    assert coordinator.config[CONF_THRESHOLD] == 0.8


async def test_options_flow_preserve_name(
    hass: HomeAssistant, init_integration
) -> None:
    """Test that area name is preserved when updating options."""
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    original_name = config_entry.data[CONF_NAME]

    # Start options flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    # Update options
    new_options = {
        CONF_MOTION_SENSORS: ["binary_sensor.motion_new"],
        CONF_THRESHOLD: 0.6,
    }

    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data"
    ):
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input=new_options
        )
        await hass.async_block_till_done()

    # Verify name was preserved
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    assert config_entry.data[CONF_NAME] == original_name
