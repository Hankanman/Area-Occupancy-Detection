"""The Room Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, CONF_NAME
from homeassistant.core import HomeAssistant, callback

from .const import DOMAIN
from .coordinator import RoomOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Room Occupancy Detection from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Initialize the coordinator
    coordinator = RoomOccupancyCoordinator(
        hass,
        entry.entry_id,
        entry.data,
    )

    # Fetch initial data
    await coordinator.async_config_entry_first_refresh()

    # Store coordinator in hass.data
    hass.data[DOMAIN][entry.entry_id] = {"coordinator": coordinator}

    # Set up update listener for config entry changes
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options for existing room occupancy entry."""
    # Update entry data with new options, preserving the room name
    data = {
        CONF_NAME: entry.data[CONF_NAME],  # Preserve the original room name
        **entry.options,  # Add all options
    }

    # Update the config entry with new data
    hass.config_entries.async_update_entry(entry, data=data)

    # Reload the config entry to apply changes
    await hass.config_entries.async_reload(entry.entry_id)


@callback
def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)

    if config_entry.version == 1:
        # No migrations implemented yet
        new_data = {**config_entry.data}

        config_entry.version = 1
        hass.config_entries.async_update_entry(config_entry, data=new_data)

    return True
