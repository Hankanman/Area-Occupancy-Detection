"""Helper functions for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any, Final

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_registry import async_get
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    STORAGE_VERSION,
    PLATFORMS,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    CONF_AREA_ID,
    NAME_PROBABILITY_SENSOR,
    NAME_DECAY_SENSOR,
    NAME_BINARY_SENSOR,
    NAME_THRESHOLD_NUMBER,
    NAME_PRIORS_SENSOR,
)

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
        # Add migration for old prior sensors to new priors sensor
        old_prior_sensors = [
            "motion_prior",
            "media_prior",
            "appliance_prior",
            "door_prior",
            "window_prior",
            "light_prior",
            "occupancy_prior",
        ]

        migration_map = {
            f"{DOMAIN}_{area_id}_probability": (
                f"{DOMAIN}_{entry_id}_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}"
            ),
            f"{DOMAIN}_{area_id}_decay": (
                f"{DOMAIN}_{entry_id}_{NAME_DECAY_SENSOR.lower().replace(' ', '_')}"
            ),
        }

        # Add migrations for all old prior sensors to the new priors sensor
        for old_prior in old_prior_sensors:
            migration_map[f"{DOMAIN}_{area_id}_{old_prior}"] = (
                f"{DOMAIN}_{entry_id}_{NAME_PRIORS_SENSOR.lower()}"
            )

        return migration_map
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
    entity_registry = async_get(hass)
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


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to the new version."""
    _LOGGER.info("Migrating Area Occupancy entry from version %s", config_entry.version)

    # Get existing data
    data = {**config_entry.data}
    options = {**config_entry.options}

    # Get the entity registry
    entity_registry = async_get(hass)

    # List of old prior sensor suffixes to remove
    old_prior_sensors = [
        "motion_prior",
        "media_prior",
        "appliance_prior",
        "door_prior",
        "window_prior",
        "light_prior",
        "occupancy_prior",
    ]

    try:
        # Remove old prior sensors from registry
        for old_prior in old_prior_sensors:
            unique_id = f"{DOMAIN}_{config_entry.entry_id}_{old_prior}"
            if entity_entry := entity_registry.async_get_entity_id(
                Platform.SENSOR, DOMAIN, unique_id
            ):
                _LOGGER.info(
                    "Found and removing prior sensor with unique_id: %s", unique_id
                )
                entity_registry.async_remove(entity_entry)
    except HomeAssistantError as err:
        _LOGGER.error("Error accessing entity registry: %s", err)

    try:
        # Run the unique ID migrations
        for platform in PLATFORMS:
            await async_migrate_unique_ids(hass, config_entry, platform)
    except HomeAssistantError as err:
        _LOGGER.error("Error during unique ID migration: %s", err)

    if CONF_AREA_ID in data:
        data.pop(CONF_AREA_ID)

    try:
        # Update the config entry without the area_id
        hass.config_entries.async_update_entry(
            config_entry, data=data, options=options, version=STORAGE_VERSION
        )
    except ValueError as err:
        _LOGGER.error("Error updating config entry: %s", err)
        return False

    _LOGGER.info(
        "Successfully migrated Area Occupancy entry %s to version %s",
        config_entry.entry_id,
        STORAGE_VERSION,
    )
    return True


def is_entity_active(
    entity_id: str,
    state: str,
    entity_types: dict[str, str],
    type_configs: dict[str, dict[str, Any]],
) -> bool:
    """Check if an entity is in an active state.

    Args:
        entity_id: The entity ID to check
        state: The current state of the entity
        entity_types: Dictionary mapping entity IDs to their sensor types

    Returns:
        bool: True if the entity is considered active, False otherwise
    """
    sensor_type = entity_types.get(entity_id)
    if not sensor_type:
        return False

    sensor_config = type_configs.get(sensor_type, {})
    if not sensor_config:
        return False

    return state in sensor_config["active_states"]
