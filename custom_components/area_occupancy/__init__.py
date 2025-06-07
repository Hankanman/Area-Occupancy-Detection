"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import CONF_VERSION, DOMAIN, PLATFORMS
from .coordinator import AreaOccupancyCoordinator
from .migrations import async_migrate_entry
from .service import async_setup_services
from .storage import StorageManager

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry."""
    try:
        hass.data.setdefault(DOMAIN, {})
        _LOGGER.debug("Checking entry version")

        # Check if config entry needs migration
        if entry.version != CONF_VERSION or not entry.version:
            _LOGGER.debug(
                "Migrating entry from version %s to %s", entry.version, CONF_VERSION
            )
            if not await async_migrate_entry(hass, entry):
                _LOGGER.error("Migration failed for entry %s", entry.entry_id)
                return False

        _LOGGER.debug("Creating coordinator for entry %s", entry.entry_id)
        # Initialize the coordinator with the unified configuration
        coordinator = AreaOccupancyCoordinator(hass, entry)

        # Load stored data and initialize states
        try:
            await coordinator.async_setup()
        except Exception as err:
            _LOGGER.error("Failed to load stored data: %s", err)
            raise ConfigEntryNotReady("Failed to load stored data") from err

        # Store the coordinator for future use
        hass.data[DOMAIN][entry.entry_id] = {"coordinator": coordinator}

        # Set up services
        await async_setup_services(hass)

        # Setup platforms
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        # Add an update listener to handle configuration updates
        entry.async_on_unload(entry.add_update_listener(_async_entry_updated))

    except Exception as err:
        _LOGGER.exception(
            "Failed to set up Area Occupancy integration for entry %s",
            entry.entry_id,
        )
        raise ConfigEntryNotReady(str(err)) from err
    else:
        _LOGGER.debug("Setup complete for entry %s", entry.entry_id)
        return True

async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Area Occupancy Detection integration."""
    _LOGGER.debug("Starting async_setup for %s", DOMAIN)
    hass.data.setdefault(DOMAIN, {})

    # --- Storage Cleanup Logic Start ---
    try:
        _LOGGER.debug("Checking storage for orphaned instances")
        active_entry_ids = {
            entry.entry_id for entry in hass.config_entries.async_entries(DOMAIN)
        }
        _LOGGER.debug("Active entry IDs: %s", active_entry_ids)

        store = StorageManager(hass)
        await store.async_cleanup_orphaned_instances(active_entry_ids)

    except Exception:
        # Log error but don't prevent setup from continuing
        _LOGGER.exception("Error during storage cleanup: %s")
    # --- Storage Cleanup Logic End ---

    # Set up services
    await async_setup_services(hass)
    return True


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of a config entry and clean up storage."""
    entry_id = entry.entry_id
    _LOGGER.info("Removing Area Occupancy config entry: %s", entry_id)

    try:
        store = StorageManager(hass)
        removed = await store.async_remove_instance(entry_id)
        if removed:
            _LOGGER.info(
                "Instance %s data removed from storage via store method", entry_id
            )
        else:
            _LOGGER.debug(
                "Instance %s data removal via store method reported no change",
                entry_id,
            )

    except Exception:
        # Log error but don't prevent removal flow
        _LOGGER.exception(
            "Error removing instance %s data from storage",
            entry_id,
        )




async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload entry when configuration is changed."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug("Unloading Area Occupancy config entry")

    # Unload all platforms
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        # Clean up
        coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
        await coordinator.async_shutdown()
        hass.data[DOMAIN].pop(entry.entry_id)
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)

    return unload_ok


async def _async_entry_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle config entry update."""
    _LOGGER.debug("Config entry updated, updating coordinator")
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    await coordinator.async_update_options()
    await coordinator.async_refresh()
