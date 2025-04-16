"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

import logging
from pathlib import Path
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
    CONF_VERSION,
    CONF_VERSION_MINOR,
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
)
from .storage import AreaOccupancyStore

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


async def async_migrate_storage(hass: HomeAssistant, entry_id: str) -> None:
    """Migrate storage data file format."""
    store = AreaOccupancyStore(hass)
    try:
        _LOGGER.debug("Starting storage file format migration for %s", entry_id)
        # Load data with old version to trigger migration
        data = await store.async_load()
        if data is None:
            _LOGGER.debug("No existing storage data found, skipping format migration")
            return

        # Save with current version to ensure migration
        await store.async_save(data)

        # Clean up old instance-specific storage file
        old_file = Path(hass.config.path(".storage", f"{DOMAIN}.{entry_id}.storage"))
        if old_file.exists():
            try:
                _LOGGER.debug("Removing old storage file: %s", old_file)
                old_file.unlink()
                _LOGGER.info("Successfully removed old storage file: %s", old_file)
            except OSError as err:
                _LOGGER.warning("Error removing old storage file %s: %s", old_file, err)

        _LOGGER.debug("Storage file format migration complete for %s", entry_id)
    except (HomeAssistantError, OSError, ValueError) as err:
        _LOGGER.error("Error during storage migration for %s: %s", entry_id, err)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to the new version."""
    current_major = CONF_VERSION
    current_minor = CONF_VERSION_MINOR
    entry_major = config_entry.version
    entry_minor = getattr(
        config_entry, "minor_version", 0
    )  # Use 0 if minor_version doesn't exist

    if entry_major > current_major or (
        entry_major == current_major and entry_minor >= current_minor
    ):
        # Stored version is same or newer, no migration needed
        _LOGGER.debug(
            "Skipping migration for %s: Stored version (%s.%s) >= Current version (%s.%s)",
            config_entry.entry_id,
            entry_major,
            entry_minor,
            current_major,
            current_minor,
        )
        return True  # Indicate successful (skipped) migration

    _LOGGER.info(
        "Migrating Area Occupancy entry %s from version %s.%s to %s.%s",
        config_entry.entry_id,
        entry_major,
        entry_minor,
        current_major,
        current_minor,
    )

    # --- Run Storage File Migration First ---
    await async_migrate_storage(hass, config_entry.entry_id)
    # --------------------------------------

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
            minor_version=CONF_VERSION_MINOR,
        )
        _LOGGER.info("Successfully migrated config entry %s", config_entry.entry_id)
    except (ValueError, KeyError, HomeAssistantError) as err:
        _LOGGER.error("Error during config migration: %s", err)
        return False
    else:
        return True
