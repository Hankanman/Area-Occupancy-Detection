"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers import config_validation as cv

from .const import (
    DOMAIN,
    STORAGE_KEY_HISTORY,
    STORAGE_VERSION,
    CONF_AREA_ID,
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_APPLIANCES,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_WINDOW,
    CONF_DECAY_ENABLED,
    CONF_DECAY_TYPE,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_MINIMUM_CONFIDENCE,
    DEFAULT_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_TYPE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_MINIMUM_CONFIDENCE,
)
from .types import StorageData, CoreConfig, OptionsConfig
from .coordinator import AreaOccupancyCoordinator
from .service import async_setup_services
from .storage import StorageEncoder, AreaOccupancyStorage

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR, Platform.NUMBER]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


def validate_config(data: dict[str, Any], validate_core: bool = True) -> None:
    """Validate configuration data."""
    if validate_core:
        if not data.get(CONF_NAME):
            raise HomeAssistantError("Name is required")
        if not data.get(CONF_MOTION_SENSORS):
            raise HomeAssistantError("At least one motion sensor is required")
        if not data.get(CONF_AREA_ID):
            raise HomeAssistantError("Area ID is required")

    # Validate numeric bounds
    bounds = {
        CONF_THRESHOLD: (0, 1),
        CONF_HISTORY_PERIOD: (1, 30),
        CONF_DECAY_WINDOW: (60, 3600),
        CONF_MINIMUM_CONFIDENCE: (0, 1),
    }

    for key, (min_val, max_val) in bounds.items():
        if key in data and not min_val <= data[key] <= max_val:
            raise HomeAssistantError(f"{key} must be between {min_val} and {max_val}")


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Area Occupancy Detection integration."""
    hass.data.setdefault(DOMAIN, {})

    # Set up services
    await async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry."""
    try:
        hass.data.setdefault(DOMAIN, {})

        # Validate configurations
        validate_config(entry.data, validate_core=True)
        validate_config(dict(entry.options), validate_core=False)

        store = Store[StorageData](
            hass,
            STORAGE_VERSION,
            f"{STORAGE_KEY_HISTORY}_{entry.entry_id}",
            atomic_writes=True,
            encoder=StorageEncoder(),
        )

        coordinator = AreaOccupancyCoordinator(
            hass=hass,
            entry_id=entry.entry_id,
            core_config=CoreConfig(**entry.data),
            options_config=entry.options,
        )

        # Load stored data
        stored_data = await store.async_load()
        if stored_data:
            await coordinator.async_restore_state(stored_data)

        hass.async_create_task(coordinator.async_setup())

        hass.data[DOMAIN][entry.entry_id] = {
            "coordinator": coordinator,
            "store": store,
        }

        entry.async_on_unload(entry.add_update_listener(async_update_options))
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        return True

    except Exception as err:
        _LOGGER.error("Failed to set up Area Occupancy integration: %s", err)
        raise ConfigEntryNotReady(str(err)) from err


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    try:
        unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
        if unload_ok:
            storage = hass.data[DOMAIN][entry.entry_id].get("storage")
            if storage:
                await storage.async_remove()
            hass.data[DOMAIN].pop(entry.entry_id)
        return unload_ok

    except (IOError, HomeAssistantError) as err:
        _LOGGER.error("Error unloading entry: %s", err)
        return False


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options for existing area occupancy entry."""
    try:
        validate_config(dict(entry.options), validate_core=False)

        coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
        coordinator.update_options(entry.options)

        await hass.config_entries.async_reload(entry.entry_id)

    except Exception as err:
        _LOGGER.error("Error updating options: %s", err)
        raise HomeAssistantError(str(err)) from err


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
    _LOGGER.debug(
        "Migrating Area Occupancy entry from version %s", config_entry.version
    )

    try:
        if config_entry.version == 1:
            # Get existing data and options
            data = {**config_entry.data}
            options = {**config_entry.options}

            # Convert threshold values from float to percentage
            if CONF_THRESHOLD in data:
                data[CONF_THRESHOLD] = data[CONF_THRESHOLD] * 100.0
            if CONF_THRESHOLD in options:
                options[CONF_THRESHOLD] = options[CONF_THRESHOLD] * 100.0

            # Convert minimum confidence from float to percentage if present
            if CONF_MINIMUM_CONFIDENCE in data:
                data[CONF_MINIMUM_CONFIDENCE] = data[CONF_MINIMUM_CONFIDENCE] * 100.0
            if CONF_MINIMUM_CONFIDENCE in options:
                options[CONF_MINIMUM_CONFIDENCE] = (
                    options[CONF_MINIMUM_CONFIDENCE] * 100.0
                )

            # Migrate to version 2
            core_config, options_data = migrate_legacy_config(data)

            # Clean up old storage data
            old_store = Store(hass, 1, f"{DOMAIN}.{config_entry.entry_id}.storage")
            await old_store.async_remove()

            # Create new storage with migrated data
            new_store = AreaOccupancyStorage(hass, config_entry.entry_id)
            await new_store.async_save(core_config)

            # Update config entry with both percentage conversions and storage migration
            hass.config_entries.async_update_entry(
                config_entry,
                data=core_config,
                options=options_data,
                version=2,
                unique_id=core_config[CONF_AREA_ID],
            )

            _LOGGER.info(
                "Successfully migrated Area Occupancy entry %s to version 2",
                config_entry.entry_id,
            )
            return True

        return False

    except (IOError, ValueError, KeyError, HomeAssistantError) as err:
        _LOGGER.error("Error migrating Area Occupancy configuration: %s", err)
        return False


def migrate_legacy_config(config: dict[str, Any]) -> tuple[CoreConfig, OptionsConfig]:
    """Migrate legacy configuration to new format."""
    area_id = config.get(CONF_AREA_ID, str(uuid.uuid4()))

    core_config = CoreConfig(
        name=config[CONF_NAME],
        area_id=area_id,
        motion_sensors=config[CONF_MOTION_SENSORS],
    )

    sensor_lists = [
        CONF_MEDIA_DEVICES,
        CONF_APPLIANCES,
        CONF_ILLUMINANCE_SENSORS,
        CONF_HUMIDITY_SENSORS,
        CONF_TEMPERATURE_SENSORS,
    ]

    options_data = {
        CONF_THRESHOLD: config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
        CONF_HISTORY_PERIOD: config.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD),
        CONF_DECAY_ENABLED: config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
        CONF_DECAY_WINDOW: config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
        CONF_DECAY_TYPE: config.get(CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE),
        CONF_HISTORICAL_ANALYSIS_ENABLED: config.get(
            CONF_HISTORICAL_ANALYSIS_ENABLED, DEFAULT_HISTORICAL_ANALYSIS_ENABLED
        ),
        CONF_MINIMUM_CONFIDENCE: config.get(
            CONF_MINIMUM_CONFIDENCE, DEFAULT_MINIMUM_CONFIDENCE
        ),
    }

    # Add sensor lists with empty list defaults
    for sensor_list in sensor_lists:
        options_data[sensor_list] = config.get(sensor_list, [])

    validate_config(options_data, validate_core=False)

    return core_config, options_data
