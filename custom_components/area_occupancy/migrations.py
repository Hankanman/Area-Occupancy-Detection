"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

from .binary_sensor import NAME_BINARY_SENSOR
from .const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_AREA_ID,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
    CONF_WINDOW_ACTIVE_STATE,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_PURPOSE,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOMAIN,
    PLATFORMS,
)
from .number import NAME_THRESHOLD_NUMBER
from .sensor import NAME_DECAY_SENSOR, NAME_PRIORS_SENSOR, NAME_PROBABILITY_SENSOR

_LOGGER = logging.getLogger(__name__)


async def async_migrate_unique_ids(
    hass: HomeAssistant, config_entry: ConfigEntry, platform: str
) -> None:
    """Migrate unique IDs of entities in the entity registry."""
    _LOGGER.debug("Starting unique ID migration for platform %s", platform)
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
        _LOGGER.debug("Platform %s not in entity_types, skipping", platform)
        return

    # Get the old format prefix to look for
    old_prefix = f"{DOMAIN}_{entry_id}_"
    _LOGGER.debug("Looking for entities with old prefix: %s", old_prefix)

    for entity_id, entity_entry in entity_registry.entities.items():
        old_unique_id = entity_entry.unique_id
        # Convert to string to avoid AttributeError
        if old_unique_id is not None and str(old_unique_id).startswith(old_prefix):
            # Simply remove the domain prefix to get the new ID
            new_unique_id = str(old_unique_id).replace(old_prefix, f"{entry_id}_")

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
            "Completed migrating %d unique IDs for platform %s",
            updated_entries,
            platform,
        )
    else:
        _LOGGER.debug("No unique IDs to migrate for platform %s", platform)


DECAY_MIN_DELAY_KEY = "decay_min_delay"


def remove_decay_min_delay(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated decay delay option from config."""
    if DECAY_MIN_DELAY_KEY in config:
        config.pop(DECAY_MIN_DELAY_KEY)
        _LOGGER.debug("Removed deprecated decay_min_delay from config")
    return config


CONF_DECAY_WINDOW_KEY = "decay_window"


def migrate_decay_half_life(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add decay half life."""
    if CONF_DECAY_HALF_LIFE not in config:
        if CONF_DECAY_WINDOW_KEY in config:
            config[CONF_DECAY_HALF_LIFE] = config[CONF_DECAY_WINDOW_KEY]
            config.pop(CONF_DECAY_WINDOW_KEY)
            _LOGGER.debug(
                "Migrated decay window to decay half life: %s",
                config[CONF_DECAY_WINDOW_KEY],
            )
        else:
            config[CONF_DECAY_HALF_LIFE] = DEFAULT_DECAY_HALF_LIFE
            _LOGGER.debug(
                "Migrated decay half life to default value: %s", DEFAULT_DECAY_HALF_LIFE
            )
    return config


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


def migrate_purpose_field(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add purpose field with default value.

    This migration:
    1. Adds the purpose field with default value if it doesn't exist
    2. Preserves any existing purpose setting
    3. Logs the migration for debugging

    Args:
        config: The configuration to migrate

    Returns:
        The migrated configuration

    """

    if CONF_PURPOSE not in config:
        config[CONF_PURPOSE] = DEFAULT_PURPOSE
        _LOGGER.debug(
            "Migrated purpose field to default value: %s",
            DEFAULT_PURPOSE,
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
    config = remove_decay_min_delay(config)
    config = migrate_primary_occupancy_sensor(config)
    config = migrate_decay_half_life(config)
    return migrate_purpose_field(config)


LEGACY_STORAGE_KEY = "area_occupancy.storage"


async def async_migrate_storage(hass: HomeAssistant, entry_id: str) -> None:
    """Migrate legacy multi-instance storage to per-entry storage format."""
    try:
        _LOGGER.debug("Starting storage migration for entry %s", entry_id)

        # Check for and clean up legacy multi-instance storage using direct file operations
        storage_dir = Path(hass.config.config_dir) / ".storage"
        legacy_file = storage_dir / LEGACY_STORAGE_KEY

        if legacy_file.exists():
            _LOGGER.info(
                "Found legacy storage file %s, removing it for fresh start",
                legacy_file.name,
            )
            try:
                legacy_file.unlink()
                _LOGGER.info("Successfully removed legacy storage file")
            except OSError as err:
                _LOGGER.warning(
                    "Error removing legacy storage file %s: %s", legacy_file, err
                )

        _LOGGER.debug("Storage migration completed for entry %s", entry_id)
    except (HomeAssistantError, OSError, ValueError) as err:
        _LOGGER.error("Error during storage migration for entry %s: %s", entry_id, err)


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
    _LOGGER.debug("Starting storage migration for %s", config_entry.entry_id)
    await async_migrate_storage(hass, config_entry.entry_id)
    _LOGGER.debug("Storage migration completed for %s", config_entry.entry_id)
    # --------------------------------------

    # Get existing data
    _LOGGER.debug("Getting existing config data for %s", config_entry.entry_id)
    data = {**config_entry.data}
    options = {**config_entry.options}

    try:
        # Run the unique ID migrations
        _LOGGER.debug("Starting unique ID migrations for %s", config_entry.entry_id)
        for platform in PLATFORMS:
            _LOGGER.debug("Migrating unique IDs for platform %s", platform)
            await async_migrate_unique_ids(hass, config_entry, platform)
        _LOGGER.debug("Unique ID migrations completed for %s", config_entry.entry_id)
    except HomeAssistantError as err:
        _LOGGER.error("Error during unique ID migration: %s", err)

    # Remove deprecated fields
    _LOGGER.debug("Removing deprecated fields for %s", config_entry.entry_id)
    if CONF_AREA_ID in data:
        data.pop(CONF_AREA_ID)
        _LOGGER.debug("Removed deprecated CONF_AREA_ID")

    if DECAY_MIN_DELAY_KEY in data:
        data.pop(DECAY_MIN_DELAY_KEY)
        _LOGGER.debug("Removed deprecated decay_min_delay from data")
    if DECAY_MIN_DELAY_KEY in options:
        options.pop(DECAY_MIN_DELAY_KEY)
        _LOGGER.debug("Removed deprecated decay_min_delay from options")

    # Ensure new state configuration values are present with defaults
    _LOGGER.debug("Adding new state configurations for %s", config_entry.entry_id)
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
        _LOGGER.debug("Applying configuration migrations for %s", config_entry.entry_id)
        data = migrate_config(data)
        options = migrate_config(options)

        # Handle threshold value with default if not present
        threshold = options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        options[CONF_THRESHOLD] = validate_threshold(threshold)

        # Update the config entry with new data and options
        _LOGGER.debug("Updating config entry for %s", config_entry.entry_id)
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


def validate_threshold(threshold: float) -> float:
    """Validate the threshold value.

    Args:
        threshold: The threshold value to validate

    Returns:
        The validated threshold value

    """
    if threshold < 1.0 or threshold > 99.0:
        return DEFAULT_THRESHOLD
    return round(threshold, 0)
