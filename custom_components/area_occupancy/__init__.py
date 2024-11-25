"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN, AreaOccupancyConfig
from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry."""
    try:
        hass.data.setdefault(DOMAIN, {})

        config: AreaOccupancyConfig = dict(entry.data)
        coordinator = AreaOccupancyCoordinator(hass, entry.entry_id, config)

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
        # Current version, nothing to do
        return True

    # In the future, if we need to migrate from version 1 to 2:
    # if config_entry.version == 1:
    #     new = {**config_entry.data}
    #     # Perform migration
    #     config_entry.version = 2
    #     hass.config_entries.async_update_entry(config_entry, data=new)

    _LOGGER.error(
        "Failed to migrate area occupancy configuration from version %s",
        config_entry.version,
    )
    return False
