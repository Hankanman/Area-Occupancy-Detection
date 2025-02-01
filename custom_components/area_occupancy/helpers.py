"""Helper functions for Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_registry import async_get
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    STORAGE_VERSION,
    PLATFORMS,
    CONF_AREA_ID,
    NAME_PROBABILITY_SENSOR,
    NAME_DECAY_SENSOR,
    NAME_BINARY_SENSOR,
    NAME_THRESHOLD_NUMBER,
    NAME_PRIORS_SENSOR,
    ROUNDING_PRECISION,
    CONF_DOOR_ACTIVE_STATE,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
)

_LOGGER = logging.getLogger(__name__)


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
        # Update the config entry with new data and options
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
