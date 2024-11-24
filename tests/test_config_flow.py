"""Test the Room Occupancy Detection config flow."""

from unittest.mock import patch
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_NAME
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.room_occupancy.const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_THRESHOLD,
)


async def test_form(hass: HomeAssistant):
    """Test we get the form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == "form"
    assert result["errors"] == {}

    # Test form submission with minimal required fields
    with patch(
        "custom_components.room_occupancy.async_setup_entry",
        return_value=True,
    ) as mock_setup_entry:
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Test Room",
                CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
                CONF_THRESHOLD: 0.5,
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] == "create_entry"
    assert result2["title"] == "Test Room"
    assert result2["data"] == {
        CONF_NAME: "Test Room",
        CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
        CONF_THRESHOLD: 0.5,
    }
    assert len(mock_setup_entry.mock_calls) == 1


async def test_form_invalid_sensor(hass: HomeAssistant):
    """Test we handle invalid sensor."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Test Room",
            CONF_MOTION_SENSORS: ["binary_sensor.nonexistent"],
            CONF_THRESHOLD: 0.5,
        },
    )

    assert result2["type"] == "form"
    assert result2["errors"] == {"base": "invalid_sensors"}


async def test_form_duplicate_room(hass: HomeAssistant):
    """Test we handle duplicate room names."""
    # Setup existing entry
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: "Test Room"},
        title="Test Room",
    )
    entry.add_to_hass(hass)

    # Try to add same room again
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Test Room",
            CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
            CONF_THRESHOLD: 0.5,
        },
    )

    assert result2["type"] == "form"
    assert result2["errors"] == {"base": "already_configured"}


async def test_form_invalid_threshold(hass: HomeAssistant):
    """Test we handle invalid threshold values."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Test Room",
            CONF_MOTION_SENSORS: ["binary_sensor.test_motion"],
            CONF_THRESHOLD: 1.5,  # Invalid threshold > 1
        },
    )

    assert result2["type"] == "form"
    assert result2["errors"] == {"base": "invalid_threshold"}
