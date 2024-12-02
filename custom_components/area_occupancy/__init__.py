"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging
import os.path

import yaml
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError

from .coordinator import AreaOccupancyCoordinator

from .const import (
    DOMAIN,
    PROBABILITY_CONFIG_FILE,
    HISTORY_STORAGE_FILE,
    AreaOccupancyConfig,
    CONF_DEVICE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_APPLIANCES,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry."""
    try:
        # Load base probability configuration
        config_path = os.path.join(os.path.dirname(__file__), PROBABILITY_CONFIG_FILE)
        if not os.path.exists(config_path):
            raise HomeAssistantError(
                f"Base probability configuration file not found: {config_path}"
            )

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                base_config = yaml.safe_load(file)
        except Exception as err:
            raise HomeAssistantError(
                "Failed to load base probability configuration"
            ) from err

        # Initialize or validate history storage
        history_path = hass.config.path(HISTORY_STORAGE_FILE)
        if not os.path.exists(history_path):
            os.makedirs(os.path.dirname(history_path), exist_ok=True)

        hass.data.setdefault(DOMAIN, {})

        config: AreaOccupancyConfig = dict(entry.data)
        # Add base_config to the main configuration
        config["base_config"] = base_config

        # Handle migration of legacy device_states to new categories
        if CONF_DEVICE_STATES in config:
            media_devices = []
            appliances = []
            for entity_id in config[CONF_DEVICE_STATES]:
                if entity_id.startswith("media_player."):
                    media_devices.append(entity_id)
                else:
                    appliances.append(entity_id)

            config[CONF_MEDIA_DEVICES] = (
                config.get(CONF_MEDIA_DEVICES, []) + media_devices
            )
            config[CONF_APPLIANCES] = config.get(CONF_APPLIANCES, []) + appliances
            del config[CONF_DEVICE_STATES]

            # Update the config entry with migrated data
            hass.config_entries.async_update_entry(entry, data=config)

        coordinator = AreaOccupancyCoordinator(
            hass,
            entry.entry_id,
            config,  # Now contains base_config
        )

        await coordinator.async_config_entry_first_refresh()

        hass.data[DOMAIN][entry.entry_id] = {
            "coordinator": coordinator,
            "config": config,
        }

        entry.async_on_unload(entry.add_update_listener(async_update_options))
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        return True

    except Exception as err:
        _LOGGER.exception("Failed to setup area occupancy integration")
        raise ConfigEntryNotReady from err


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
            "coordinator"
        ]
        coordinator.unsubscribe()
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options for existing area occupancy entry."""
    data: AreaOccupancyConfig = {
        CONF_NAME: entry.data[CONF_NAME],
        **entry.options,
    }
    hass.config_entries.async_update_entry(entry, data=data)
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)

    if config_entry.version == 1:
        new = {**config_entry.data}

        # Migrate device_states to new categories if present
        if CONF_DEVICE_STATES in new:
            media_devices = []
            appliances = []
            for entity_id in new[CONF_DEVICE_STATES]:
                if entity_id.startswith("media_player."):
                    media_devices.append(entity_id)
                else:
                    appliances.append(entity_id)

            new[CONF_MEDIA_DEVICES] = media_devices
            new[CONF_APPLIANCES] = appliances
            del new[CONF_DEVICE_STATES]

        config_entry.version = 2
        hass.config_entries.async_update_entry(config_entry, data=new)

        return True

    _LOGGER.error(
        "Failed to migrate area occupancy configuration from version %s",
        config_entry.version,
    )
    return False
