"""Configuration schemas for virtual sensors."""

from __future__ import annotations

import voluptuous as vol
from homeassistant.const import CONF_ENABLED, CONF_NAME, CONF_TYPE, CONF_UNIQUE_ID
from homeassistant.helpers import config_validation as cv

from .const import (
    CONF_VIRTUAL_SENSOR_OPTIONS,
    CONF_VIRTUAL_SENSOR_UPDATE_INTERVAL,
    CONF_VIRTUAL_SENSOR_WEIGHT,
    DEFAULT_VIRTUAL_SENSOR_ENABLED,
    DEFAULT_VIRTUAL_SENSOR_UPDATE_INTERVAL,
    DEFAULT_VIRTUAL_SENSOR_WEIGHT,
    VIRTUAL_SENSOR_TYPE_WASP_IN_BOX,
)

# Base schema for all virtual sensors
VIRTUAL_SENSOR_BASE_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_TYPE): cv.string,
        vol.Required(CONF_NAME): cv.string,
        vol.Optional(CONF_UNIQUE_ID): cv.string,
        vol.Optional(
            CONF_VIRTUAL_SENSOR_UPDATE_INTERVAL,
            default=DEFAULT_VIRTUAL_SENSOR_UPDATE_INTERVAL,
        ): cv.positive_int,
        vol.Optional(
            CONF_VIRTUAL_SENSOR_WEIGHT, default=DEFAULT_VIRTUAL_SENSOR_WEIGHT
        ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
        vol.Optional(CONF_ENABLED, default=DEFAULT_VIRTUAL_SENSOR_ENABLED): cv.boolean,
        vol.Optional(CONF_VIRTUAL_SENSOR_OPTIONS, default={}): dict,
    }
)

# Schema for Wasp in Box sensor
WASP_IN_BOX_SCHEMA = VIRTUAL_SENSOR_BASE_SCHEMA.extend(
    {
        vol.Required(CONF_TYPE): vol.Equal(VIRTUAL_SENSOR_TYPE_WASP_IN_BOX),
        vol.Required("door_entity_id"): cv.entity_id,
        vol.Required("motion_entity_id"): cv.entity_id,
        vol.Optional("motion_timeout", default=300): cv.positive_int,
    }
)

# Map of sensor types to their schemas
VIRTUAL_SENSOR_SCHEMAS = {
    VIRTUAL_SENSOR_TYPE_WASP_IN_BOX: WASP_IN_BOX_SCHEMA,
}


def validate_virtual_sensor_config(config: dict) -> dict:
    """Validate a virtual sensor configuration.

    Args:
        config: The configuration dictionary to validate.

    Returns:
        The validated configuration dictionary.

    Raises:
        vol.Invalid: If the configuration is invalid.

    """
    # First validate base schema
    config = VIRTUAL_SENSOR_BASE_SCHEMA(config)

    # Then validate type-specific schema
    sensor_type = config[CONF_TYPE]
    if sensor_type in VIRTUAL_SENSOR_SCHEMAS:
        config = VIRTUAL_SENSOR_SCHEMAS[sensor_type](config)
    else:
        raise vol.Invalid(f"Unsupported virtual sensor type: {sensor_type}")

    return config
