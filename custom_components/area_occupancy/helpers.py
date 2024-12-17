"""Helper functions for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any, Final

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.config_entries import ConfigEntry

from .coordinator import AreaOccupancyCoordinator
from .const import (
    DOMAIN,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    CONF_AREA_ID,
    NAME_PROBABILITY_SENSOR,
    NAME_DECAY_SENSOR,
    NAME_MOTION_PRIOR_SENSOR,
    NAME_MEDIA_PRIOR_SENSOR,
    NAME_APPLIANCE_PRIOR_SENSOR,
    NAME_DOOR_PRIOR_SENSOR,
    NAME_WINDOW_PRIOR_SENSOR,
    NAME_LIGHT_PRIOR_SENSOR,
    NAME_OCCUPANCY_PRIOR_SENSOR,
    NAME_BINARY_SENSOR,
    NAME_THRESHOLD_NUMBER,
)
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
        config = coordinator.config

        configured_sensors = {
            "Motion": config.get("motion_sensors", []),
            "Media": config.get("media_devices", []),
            "Appliances": config.get("appliances", []),
            "Illuminance": config.get("illuminance_sensors", []),
            "Humidity": config.get("humidity_sensors", []),
            "Temperature": config.get("temperature_sensors", []),
            "Door": config.get("door_sensors", []),
            "Window": config.get("window_sensors", []),
            "Lights": config.get("lights", []),
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


def generate_migration_map(
    area_id: str, entry_id: str, platform: str
) -> dict[str, str]:
    """Generate migration map for unique IDs based on platform."""
    if platform == "sensor":
        return {
            f"{DOMAIN}_{area_id}_probability": f"{DOMAIN}_{entry_id}_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_decay": f"{DOMAIN}_{entry_id}_{NAME_DECAY_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_motion_prior": f"{DOMAIN}_{entry_id}_{NAME_MOTION_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_{NAME_MOTION_PRIOR_SENSOR}": f"{DOMAIN}_{entry_id}_{NAME_MOTION_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_media_prior": f"{DOMAIN}_{entry_id}_{NAME_MEDIA_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_{NAME_MEDIA_PRIOR_SENSOR}": f"{DOMAIN}_{entry_id}_{NAME_MEDIA_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_appliance_prior": f"{DOMAIN}_{entry_id}_{NAME_APPLIANCE_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_{NAME_APPLIANCE_PRIOR_SENSOR}": f"{DOMAIN}_{entry_id}_{NAME_APPLIANCE_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_door_prior": f"{DOMAIN}_{entry_id}_{NAME_DOOR_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_{NAME_DOOR_PRIOR_SENSOR}": f"{DOMAIN}_{entry_id}_{NAME_DOOR_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_window_prior": f"{DOMAIN}_{entry_id}_{NAME_WINDOW_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_{NAME_WINDOW_PRIOR_SENSOR}": f"{DOMAIN}_{entry_id}_{NAME_WINDOW_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_light_prior": f"{DOMAIN}_{entry_id}_{NAME_LIGHT_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_{NAME_LIGHT_PRIOR_SENSOR}": f"{DOMAIN}_{entry_id}_{NAME_LIGHT_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_occupancy_prior": f"{DOMAIN}_{entry_id}_{NAME_OCCUPANCY_PRIOR_SENSOR.lower().replace(' ', '_')}",
            f"{DOMAIN}_{area_id}_{NAME_OCCUPANCY_PRIOR_SENSOR}": f"{DOMAIN}_{entry_id}_{NAME_OCCUPANCY_PRIOR_SENSOR.lower().replace(' ', '_')}",
        }
    elif platform == "binary_sensor":
        return {
            f"{DOMAIN}_{area_id}_occupancy": f"{DOMAIN}_{entry_id}_{NAME_BINARY_SENSOR.lower().replace(' ', '_')}",
        }
    elif platform == "number":
        return {
            f"{DOMAIN}_{area_id}_threshold": f"{DOMAIN}_{entry_id}_{NAME_THRESHOLD_NUMBER.lower().replace(' ', '_')}",
        }
    return {}


async def async_migrate_unique_ids(
    hass, config_entry: ConfigEntry, platform: str
) -> None:
    """Migrate unique IDs of entities in the entity registry."""
    entity_registry = async_get_entity_registry(hass)
    updated_entries = 0

    # Get area_id from config entry data
    area_id = config_entry.data.get(CONF_AREA_ID)
    entry_id = config_entry.entry_id

    # Generate the migration map for this specific entry
    migration_map = generate_migration_map(area_id, entry_id, platform)

    for entity_id, entity_entry in entity_registry.entities.items():
        old_unique_id = entity_entry.unique_id
        if old_unique_id in migration_map:
            new_unique_id = migration_map[old_unique_id]

            # Update the unique ID in the registry
            _LOGGER.info(
                "Migrating unique ID for %s: %s -> %s",
                entity_id,
                old_unique_id,
                new_unique_id,
            )
            entity_registry.async_update_entity(entity_id, new_unique_id=new_unique_id)
            updated_entries += 1

    if updated_entries > 0:
        _LOGGER.info(
            "Completed migrating %s unique IDs for area %s", updated_entries, area_id
        )
