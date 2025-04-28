"""Virtual sensor framework for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Optional, Type, cast

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv

from .base import VirtualSensor
from .config import validate_virtual_sensor_config
from .const import CONF_VIRTUAL_SENSOR_TYPE, DEFAULT_VIRTUAL_SENSOR_UPDATE_INTERVAL
from .coordinator import VirtualSensorCoordinator
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
    "async_setup_virtual_sensors",
]

# Schema for virtual sensors configuration
VIRTUAL_SENSORS_CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required("virtual_sensors"): vol.All(
            cv.ensure_list,
            [validate_virtual_sensor_config],
        ),
    }
)


async def async_setup_virtual_sensors(
    hass: HomeAssistant,
    config: VirtualSensorConfig,
    update_interval: Optional[timedelta] = None,
) -> Optional[VirtualSensorCoordinator]:
    """Set up virtual sensors from configuration.

    Args:
        hass: The Home Assistant instance.
        config: The configuration dictionary.
        update_interval: Optional update interval for the coordinator.

    Returns:
        The virtual sensor coordinator if setup succeeds, None otherwise.

    """
    try:
        # Validate configuration
        try:
            config = cast(VirtualSensorConfig, VIRTUAL_SENSORS_CONFIG_SCHEMA(config))
        except vol.Invalid as err:
            _LOGGER.error("Invalid virtual sensors configuration: %s", err)
            return None

        # Create coordinator
        coordinator = VirtualSensorCoordinator(
            hass,
            "Virtual Sensor Coordinator",
            update_interval
            or timedelta(seconds=DEFAULT_VIRTUAL_SENSOR_UPDATE_INTERVAL),
        )

        # Get virtual sensor configurations
        sensor_configs = config.get("virtual_sensors", [])
        if not sensor_configs:
            _LOGGER.warning("No virtual sensors configured")
            return coordinator

        # Set up each virtual sensor
        for sensor_config in sensor_configs:
            try:
                sensor_type = sensor_config.get(CONF_VIRTUAL_SENSOR_TYPE)
                if sensor_type not in VIRTUAL_SENSOR_TYPES:
                    raise VirtualSensorTypeError(
                        f"Unsupported virtual sensor type: {sensor_type}"
                    )

                sensor_class = VIRTUAL_SENSOR_TYPES[sensor_type]
                sensor = sensor_class(hass, sensor_config, coordinator)

                # Add sensor to coordinator
                await coordinator.async_add_sensor(
                    sensor_config.get("name", f"virtual_sensor_{sensor_type}"),
                    sensor,
                )

            except Exception as err:
                _LOGGER.error(
                    "Failed to set up virtual sensor %s: %s",
                    sensor_config.get("name", "unknown"),
                    err,
                    exc_info=True,
                )
                continue

        # Start coordinator
        await coordinator.async_config_entry_first_refresh()
        return coordinator

    except Exception as err:
        _LOGGER.error("Failed to set up virtual sensors: %s", err, exc_info=True)
        return None
