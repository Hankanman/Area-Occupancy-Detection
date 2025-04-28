"""Virtual sensor framework for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Type

from homeassistant.core import HomeAssistant

from .base import VirtualSensor
from .const import CONF_VIRTUAL_SENSOR_TYPE
from .exceptions import VirtualSensorTypeError
from .types import VirtualSensorConfig
from .wasp_in_box import WaspInBoxSensor

_LOGGER = logging.getLogger(__name__)

# Map of virtual sensor types to their implementation classes
VIRTUAL_SENSOR_TYPES: Dict[str, Type[VirtualSensor]] = {
    "wasp_in_box": WaspInBoxSensor,
}

__all__ = [
    "VirtualSensor",
    "WaspInBoxSensor",
    "VIRTUAL_SENSOR_TYPES",
    "CONF_VIRTUAL_SENSOR_TYPE",
]


def create_virtual_sensor(
    hass: HomeAssistant,
    config: VirtualSensorConfig,
    coordinator: Any,
) -> VirtualSensor:
    """Create a virtual sensor instance based on the configuration.

    Args:
        hass: The Home Assistant instance.
        config: The virtual sensor configuration.
        coordinator: The coordinator instance.

    Returns:
        A virtual sensor instance.

    Raises:
        VirtualSensorTypeError: If the sensor type is not supported.

    """
    sensor_type = config.get(CONF_VIRTUAL_SENSOR_TYPE)
    if sensor_type not in VIRTUAL_SENSOR_TYPES:
        raise VirtualSensorTypeError(f"Unsupported virtual sensor type: {sensor_type}")

    sensor_class = VIRTUAL_SENSOR_TYPES[sensor_type]
    return sensor_class(hass, config, coordinator)


async def async_setup_virtual_sensor(
    hass: HomeAssistant,
    config: VirtualSensorConfig,
    coordinator: Any,
) -> VirtualSensor | None:
    """Set up a virtual sensor based on the configuration.

    Args:
        hass: The Home Assistant instance.
        config: The virtual sensor configuration.
        coordinator: The coordinator instance.

    Returns:
        The initialized virtual sensor or None if setup fails.

    """
    try:
        sensor_type = config.get("type")
        if sensor_type == "wasp_in_box":
            return WaspInBoxSensor(hass, config, coordinator)
        else:
            _LOGGER.error("Unknown virtual sensor type: %s", sensor_type)
            return None
    except Exception as ex:
        _LOGGER.error("Failed to set up virtual sensor: %s", ex, exc_info=True)
        return None
