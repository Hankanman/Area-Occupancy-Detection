"""The Area Occupancy Detection integration."""

from __future__ import annotations

import os
import logging
from typing import Any

import yaml
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType

from .config_management import ConfigManager
from .const import (
    DOMAIN,
    STORAGE_KEY_HISTORY,
    STORAGE_VERSION,
    StorageData,
)
from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]

# Global to store base config after loading
_CACHED_BASE_CONFIG: dict[str, Any] | None = None


async def _load_base_config(hass: HomeAssistant) -> dict[str, Any]:
    """Load base configuration from YAML."""
    global _CACHED_BASE_CONFIG

    if _CACHED_BASE_CONFIG is not None:
        return _CACHED_BASE_CONFIG

    try:
        # Get the directory of this file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, "default_probabilities.yaml")

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not config or "base_probabilities" not in config:
            raise ValueError("Invalid base configuration format")

        _CACHED_BASE_CONFIG = config
        return config

    except Exception as err:
        _LOGGER.error("Error loading base configuration: %s", err)
        raise HomeAssistantError(f"Failed to load base configuration: {err}") from err


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Area Occupancy Detection integration."""
    try:
        # Load base configuration on integration setup
        await _load_base_config(hass)
        hass.data.setdefault(DOMAIN, {})
        return True
    except HomeAssistantError as err:
        _LOGGER.error("Failed to setup integration: %s", err)
        return False


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry."""
    try:
        # Load base configuration
        base_config = await _load_base_config(hass)

        # Initialize storage
        store = Store[StorageData](
            hass,
            STORAGE_VERSION,
            f"{STORAGE_KEY_HISTORY}_{entry.entry_id}",
            atomic_writes=True,
        )

        # Load stored data
        stored_data = await store.async_load() or {
            "version": STORAGE_VERSION,
            "last_updated": "",
            "areas": {},
        }

        # Combine core config and options
        config = {
            **entry.data,  # Core configuration
            **entry.options,  # Optional settings
        }

        # Initialize coordinator with base and user config
        coordinator = AreaOccupancyCoordinator(
            hass,
            entry.entry_id,
            config,
            base_config,
            store,
        )

        # Perform setup that requires async operations
        await coordinator.async_setup()

        # Perform first update
        await coordinator.async_config_entry_first_refresh()

        # Store initialized components
        hass.data[DOMAIN][entry.entry_id] = {
            "coordinator": coordinator,
            "store": store,
        }

        # Set up entry update listener
        entry.async_on_unload(entry.add_update_listener(async_update_options))

        # Set up platforms
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        return True

    except Exception as err:
        _LOGGER.error("Failed to set up area occupancy integration: %s", err)
        raise ConfigEntryNotReady from err


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    try:
        # Unload platforms
        unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

        if unload_ok:
            coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
                "coordinator"
            ]
            store: Store = hass.data[DOMAIN][entry.entry_id]["store"]

            # Clean up coordinator and save final state
            coordinator.unsubscribe()
            await store.async_save(coordinator.get_storage_data())

            # Remove entry data
            hass.data[DOMAIN].pop(entry.entry_id)

        return unload_ok

    except Exception as err:
        _LOGGER.error("Error unloading entry: %s", err)
        return False


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options for existing area occupancy entry."""
    try:
        # Validate the combined configuration
        config = {
            **entry.data,  # Core configuration
            **entry.options,  # Current options
        }

        # Reload entry
        await hass.config_entries.async_reload(entry.entry_id)

    except Exception as err:
        _LOGGER.error("Error updating options: %s", err)
        raise


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
    _LOGGER.debug(
        "Migrating Area Occupancy entry from version %s", config_entry.version
    )

    try:
        if config_entry.version == 1:
            # Migrate to version 2 using ConfigManager
            core_config, options_config = ConfigManager.migrate_legacy_config(
                config_entry.data
            )

            # Update config entry with new format
            hass.config_entries.async_update_entry(
                config_entry, data=core_config, options=options_config, version=2
            )

            _LOGGER.info(
                "Successfully migrated entry %s to version 2", config_entry.entry_id
            )
            return True

        return False

    except Exception as err:
        _LOGGER.error("Error migrating Area Occupancy configuration: %s", err)
        return False
