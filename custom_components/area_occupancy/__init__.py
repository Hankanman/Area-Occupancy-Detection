"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.typing import ConfigType

from .const import CONF_VERSION, DOMAIN, PLATFORMS
from .coordinator import AreaOccupancyCoordinator
from .migrations import async_migrate_entry
from .service import async_setup_services

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry."""

    # Migration check
    if entry.version != CONF_VERSION or not entry.version:
        _LOGGER.debug(
            "Migrating entry from version %s to %s", entry.version, CONF_VERSION
        )
        try:
            migration_result = await async_migrate_entry(hass, entry)
            if not migration_result:
                _LOGGER.error("Migration failed for entry %s", entry.entry_id)
            _LOGGER.info(
                "Migration completed successfully for entry %s", entry.entry_id
            )
        except Exception as err:
            _LOGGER.error(
                "Migration threw exception for entry %s: %s", entry.entry_id, err
            )
            raise ConfigEntryNotReady(
                f"Migration failed with exception: {err}"
            ) from err

    # Create and setup coordinator
    _LOGGER.debug("Creating coordinator for entry %s", entry.entry_id)
    coordinator = AreaOccupancyCoordinator(hass, entry)

    # Use modern coordinator setup pattern
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as err:
        _LOGGER.error("Failed to setup coordinator: %s", err)
        raise ConfigEntryNotReady(f"Failed to setup coordinator: {err}") from err

    # Store coordinator using modern pattern
    entry.runtime_data = coordinator

    # Setup platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Setup services
    await async_setup_services(hass)

    # Add update listener
    entry.async_on_unload(entry.add_update_listener(_async_entry_updated))

    _LOGGER.debug("Setup complete for entry %s", entry.entry_id)
    return True


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Area Occupancy Detection integration."""
    _LOGGER.debug("Starting async_setup for %s", DOMAIN)
    return True


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of a config entry and clean up storage."""
    entry_id = entry.entry_id
    _LOGGER.info("Removing Area Occupancy config entry: %s", entry_id)

    try:
        # Check if runtime_data exists and has a coordinator
        if hasattr(entry, "runtime_data") and entry.runtime_data:
            # Use the existing coordinator's sqlite store
            sqlite_store = entry.runtime_data.sqlite_store
        else:
            # Create a temporary coordinator just for cleanup
            temp_coordinator = AreaOccupancyCoordinator(hass, entry)
            sqlite_store = temp_coordinator.sqlite_store

        # Remove the per-entry storage file
        await sqlite_store.async_reset()
        _LOGGER.info("Per-entry storage removed for instance %s", entry_id)

    except Exception:
        # Log error but don't prevent removal flow
        _LOGGER.exception("Error removing instance %s data from storage", entry_id)


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload entry when configuration is changed."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug("Unloading Area Occupancy config entry")

    # Unload all platforms
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        # Clean up coordinator
        coordinator = entry.runtime_data
        await coordinator.async_shutdown()

    return unload_ok


async def _async_entry_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle config entry update."""
    _LOGGER.debug("Config entry updated, updating coordinator")

    coordinator = entry.runtime_data
    await coordinator.async_update_options(entry.options)
    await coordinator.async_refresh()
