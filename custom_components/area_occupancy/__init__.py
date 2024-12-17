"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import (
    DOMAIN,
    STORAGE_VERSION,
    PLATFORMS,
)
from .coordinator import AreaOccupancyCoordinator
from .service import async_setup_services

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Area Occupancy Detection integration."""
    hass.data.setdefault(DOMAIN, {})

    # Set up services
    await async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry."""
    try:
        hass.data.setdefault(DOMAIN, {})

        # Initialize the coordinator with the unified configuration
        coordinator = AreaOccupancyCoordinator(
            hass,
            entry.entry_id,
            config=entry.data,  # Unified configuration
        )

        # Load stored data and initialize states
        await coordinator.async_load_stored_data()
        await coordinator.async_initialize_states()

        # Trigger an initial refresh
        await coordinator.async_refresh()

        # Store the coordinator for future use
        hass.data[DOMAIN][entry.entry_id] = {"coordinator": coordinator}

        # Setup platforms
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        # Add an update listener to handle configuration updates
        entry.async_on_unload(entry.add_update_listener(async_reload_entry))

        return True

    except Exception as err:
        _LOGGER.error(
            "Failed to set up Area Occupancy integration for entry %s: %s",
            entry.entry_id,
            err,
            exc_info=True,
        )
        raise ConfigEntryNotReady(str(err)) from err


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload entry when configuration is changed."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to the new version."""
    _LOGGER.info("Migrating Area Occupancy entry from version %s", config_entry.version)

    if config_entry.version < STORAGE_VERSION:
        new_version = STORAGE_VERSION
        config_entry.version = new_version

        _LOGGER.info(
            "Successfully migrated Area Occupancy entry %s to version %s",
            config_entry.entry_id,
            new_version,
        )
        return True

    return True
