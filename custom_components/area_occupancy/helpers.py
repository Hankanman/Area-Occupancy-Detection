"""Helper functions for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any, Final

from homeassistant.core import HomeAssistant

from .coordinator import AreaOccupancyCoordinator
from .const import DOMAIN, DEVICE_MANUFACTURER, DEVICE_MODEL, DEVICE_SW_VERSION
from .types import ProbabilityResult

_LOGGER = logging.getLogger(__name__)
ROUNDING_PRECISION: Final = 2


def format_float(value: float) -> float:
    """Format float to consistently show 2 decimal places."""
    try:
        return round(float(value), ROUNDING_PRECISION)
    except (ValueError, TypeError):
        return 0.0


def get_friendly_names(hass: HomeAssistant, entity_ids: list[str]) -> list[str]:
    """Convert entity IDs to friendly names."""
    return [
        hass.states.get(entity_id).attributes.get("friendly_name", entity_id)
        for entity_id in entity_ids
        if hass.states.get(entity_id)
    ]


def get_sensor_attributes(
    hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
) -> dict[str, Any]:
    """Get common sensor attributes."""
    if not coordinator.data:
        return {}

    try:
        data: ProbabilityResult = coordinator.data

        attributes = {
            "active_triggers": get_friendly_names(
                hass, data.get("active_triggers", [])
            ),
        }

        # Add configured sensors info
        options_config = coordinator.options_config
        core_config = coordinator.core_config

        configured_sensors = {
            "Motion": core_config.get("motion_sensors", []),
            "Media": options_config.get("media_devices", []),
            "Appliances": options_config.get("appliances", []),
            "Illuminance": options_config.get("illuminance_sensors", []),
            "Humidity": options_config.get("humidity_sensors", []),
            "Temperature": options_config.get("temperature_sensors", []),
        }

        # Flatten all sensors to count how many have learned priors
        all_sensors = []
        for sensor_list in configured_sensors.values():
            all_sensors.extend(sensor_list)

        learned_count = sum(
            1 for sensor in all_sensors if sensor in coordinator.learned_priors
        )

        attributes["configured_sensors"] = {
            cat: get_friendly_names(hass, slist)
            for cat, slist in configured_sensors.items()
        }

        # Show how many sensors have learned priors
        attributes["learned_prior_sensors_count"] = learned_count
        attributes["total_sensors_count"] = len(all_sensors)

        return attributes

    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.error("Error getting entity attributes: %s", err)
        return {}


def get_device_info(entry_id: str, area_name: str) -> dict[str, Any]:
    """Get common device info dictionary."""
    return {
        "identifiers": {(DOMAIN, entry_id)},
        "name": area_name,
        "manufacturer": DEVICE_MANUFACTURER,
        "model": DEVICE_MODEL,
        "sw_version": DEVICE_SW_VERSION,
    }
