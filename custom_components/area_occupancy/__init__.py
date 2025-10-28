"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import CONF_VERSION, DOMAIN, PLATFORMS
from .coordinator import AreaOccupancyCoordinator
from .migrations import async_migrate_entry
from .service import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry (fast startup mode).

    NOTE: Heavy database operations (integrity checks, historical analysis) are
    deferred to background tasks to ensure HA startup completes quickly.
    """

    # Migration check
    if entry.version != CONF_VERSION or not entry.version:
        _LOGGER.info(
            "Migrating Area Occupancy entry from version %s to %s",
            entry.version,
            CONF_VERSION,
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

    # Create and setup coordinator (fast path - no blocking operations)
    _LOGGER.info("Initializing Area Occupancy coordinator for entry %s", entry.entry_id)
    try:
        coordinator = AreaOccupancyCoordinator(hass, entry)
    except Exception as err:
        _LOGGER.error("Failed to create coordinator: %s", err)
        raise ConfigEntryNotReady(f"Failed to create coordinator: {err}") from err

    # Initialize database asynchronously (fast validation only, no integrity checks)
    try:
        _LOGGER.debug("Initializing database (quick validation mode)")
        await coordinator.async_init_database()
        _LOGGER.info("Database initialization completed")
    except Exception as err:
        _LOGGER.error("Failed to initialize database: %s", err)
        raise ConfigEntryNotReady(f"Failed to initialize database: {err}") from err

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

    # Log role-specific info
    role = "MASTER" if coordinator.is_master else "non-master"
    position = await hass.async_add_executor_job(
        coordinator.db.get_instance_position, entry.entry_id
    )
    _LOGGER.info(
        "Area Occupancy setup complete for entry %s as %s (position %d)",
        entry.entry_id,
        role,
        position,
    )
    _LOGGER.info("Analysis runs staggered with %d minute intervals", 2)
    return True


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Area Occupancy Detection integration."""
    _LOGGER.debug("Starting async_setup for %s", DOMAIN)
    return True


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of a config entry and clean up storage."""


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
        if coordinator is not None:
            await coordinator.async_shutdown()

    return unload_ok


async def _async_entry_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle config entry update."""
    _LOGGER.debug("Config entry updated, updating coordinator")

    coordinator = entry.runtime_data
    await coordinator.async_update_options(entry.options)
    await coordinator.async_refresh()
