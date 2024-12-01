# custom_components/area_occupancy/__init__.py

"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .core.sensor_manager import AreaSensorManager
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.BINARY_SENSOR, Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry."""
    _LOGGER.debug("Starting async_setup_entry for entry ID: %s", entry.entry_id)
    try:
        _LOGGER.debug("Initializing AreaSensorManager...")
        manager = AreaSensorManager(hass, entry)
        try:
            await manager.async_setup()
        except Exception as setup_err:
            _LOGGER.exception("Error during AreaSensorManager setup: %s", setup_err)
            raise ConfigEntryNotReady from setup_err

        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = manager
        _LOGGER.debug("Forwarding entry setups for platforms: %s", PLATFORMS)
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
        _LOGGER.info("Setup of entry %s completed successfully", entry.entry_id)
        return True

    except ConfigEntryNotReady as err:
        _LOGGER.warning("Integration setup deferred: %s", err)
        raise

    except ValueError as err:
        _LOGGER.error("Configuration error: %s", err)
        return False

    except (OSError, RuntimeError) as err:
        _LOGGER.exception("Unexpected error during setup: %s", err)
        return False


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug("Attempting to unload entry ID: %s", entry.entry_id)
    if entry.entry_id not in hass.data.get(DOMAIN, {}):
        _LOGGER.warning("Attempted to unload non-existent entry: %s", entry.entry_id)
        return False

    manager = hass.data[DOMAIN].pop(entry.entry_id)
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        _LOGGER.debug(
            "Successfully unloaded platforms for entry ID: %s", entry.entry_id
        )
        await manager.async_unload()
    else:
        _LOGGER.error("Failed to unload entry: %s", entry.entry_id)

    return unload_ok


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug(
        "Migrating entry ID %s from version %s",
        config_entry.entry_id,
        config_entry.version,
    )

    if config_entry.version == 1:
        new = {**config_entry.data}

        # Add new fields with defaults
        new.setdefault("decay_enabled", True)
        new.setdefault("decay_window", 600)
        new.setdefault("decay_type", "linear")
        new.setdefault("history_period", 7)

        config_entry.version = 2
        hass.config_entries.async_update_entry(config_entry, data=new)
        _LOGGER.debug(
            "Migration of entry ID %s to version 2 completed", config_entry.entry_id
        )

    return True
