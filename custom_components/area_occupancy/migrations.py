"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

from .const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_AREA_ID,
    CONF_DOOR_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_WINDOW_ACTIVE_STATE,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOMAIN,
    NAME_BINARY_SENSOR,
    NAME_DECAY_SENSOR,
    NAME_PRIORS_SENSOR,
    NAME_PROBABILITY_SENSOR,
    NAME_THRESHOLD_NUMBER,
    PLATFORMS,
    CONF_VERSION,
)

_LOGGER = logging.getLogger(__name__)


async def async_migrate_unique_ids(
    hass: HomeAssistant, config_entry: ConfigEntry, platform: str
) -> None:
    """Migrate unique IDs of entities in the entity registry."""
    entity_registry = er.async_get(hass)
    updated_entries = 0
    entry_id = config_entry.entry_id

    # Define which entity types to look for based on platform
    entity_types = {
        "sensor": [NAME_PROBABILITY_SENSOR, NAME_DECAY_SENSOR, NAME_PRIORS_SENSOR],
        "binary_sensor": [NAME_BINARY_SENSOR],
        "number": [NAME_THRESHOLD_NUMBER],
    }

    if platform not in entity_types:
        return

    # Get the old format prefix to look for
    old_prefix = f"{DOMAIN}_{entry_id}_"

    for entity_id, entity_entry in entity_registry.entities.items():
        old_unique_id = entity_entry.unique_id
        # Check if this is one of our entities that needs migration
        if old_unique_id.startswith(old_prefix):
            # Simply remove the domain prefix to get the new ID
            new_unique_id = old_unique_id.replace(old_prefix, f"{entry_id}_")

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
        _LOGGER.info("Completed migrating %s unique IDs", updated_entries)


def migrate_primary_occupancy_sensor(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add primary occupancy sensor.

    This migration:
    1. Takes the first motion sensor as the primary occupancy sensor if none is set
    2. Preserves any existing primary occupancy sensor setting
    3. Logs the migration for debugging

    Args:
        config: The configuration to migrate

    Returns:
        The migrated configuration

    """
    if CONF_PRIMARY_OCCUPANCY_SENSOR not in config:
        motion_sensors = config.get(CONF_MOTION_SENSORS, [])
        if motion_sensors:
            config[CONF_PRIMARY_OCCUPANCY_SENSOR] = motion_sensors[0]
            _LOGGER.debug(
                "Migrated primary occupancy sensor to first motion sensor: %s",
                motion_sensors[0],
            )
        else:
            _LOGGER.warning(
                "No motion sensors found for primary occupancy sensor migration"
            )

    return config


def migrate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to latest version.

    Args:
        config: The configuration to migrate

    Returns:
        The migrated configuration

    """
    # Apply migrations in order
    return migrate_primary_occupancy_sensor(config)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to the new version."""
    _LOGGER.info("Migrating Area Occupancy entry from version %s", config_entry.version)

    # Get existing data
    data = {**config_entry.data}
    options = {**config_entry.options}

    try:
        # Run the unique ID migrations
        for platform in PLATFORMS:
            await async_migrate_unique_ids(hass, config_entry, platform)
    except HomeAssistantError as err:
        _LOGGER.error("Error during unique ID migration: %s", err)

    # Remove deprecated fields
    if CONF_AREA_ID in data:
        data.pop(CONF_AREA_ID)

    # Ensure new state configuration values are present with defaults
    new_configs = {
        CONF_DOOR_ACTIVE_STATE: DEFAULT_DOOR_ACTIVE_STATE,
        CONF_WINDOW_ACTIVE_STATE: DEFAULT_WINDOW_ACTIVE_STATE,
        CONF_MEDIA_ACTIVE_STATES: DEFAULT_MEDIA_ACTIVE_STATES,
        CONF_APPLIANCE_ACTIVE_STATES: DEFAULT_APPLIANCE_ACTIVE_STATES,
    }

    # Update data with new state configurations if not present
    for key, default_value in new_configs.items():
        if key not in data and key not in options:
            _LOGGER.info("Adding new configuration %s with default value", key)
            # For multi-select states, add to data
            if isinstance(default_value, list):
                data[key] = default_value
            # For single-select states, add to options
            else:
                options[key] = default_value

    try:
        # Apply configuration migrations
        data = migrate_config(data)
        options = migrate_config(options)

        # Update the config entry with new data and options
        hass.config_entries.async_update_entry(
            config_entry,
            data=data,
            options=options,
            version=CONF_VERSION,
        )
        _LOGGER.info("Successfully migrated config entry")
    except (ValueError, KeyError, HomeAssistantError) as err:
        _LOGGER.error("Error during config migration: %s", err)
        return False
    else:
        return True
