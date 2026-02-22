"""The Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv, device_registry as dr
from homeassistant.helpers.typing import ConfigType

from .const import CONF_AREA_ID, CONF_AREAS, CONF_VERSION, DOMAIN, PLATFORMS
from .coordinator import AreaOccupancyCoordinator
from .migrations import async_migrate_entry
from .service import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


def _validate_migration_result(migration_result: bool, entry_id: str) -> None:
    """Validate migration result and raise if migration failed.

    Args:
        migration_result: Result of the migration operation
        entry_id: Config entry ID for error logging

    Raises:
        ConfigEntryNotReady: If migration failed
    """
    if not migration_result:
        _LOGGER.error("Migration failed for entry %s", entry_id)
        raise ConfigEntryNotReady("Migration failed")


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Area Occupancy Detection from a config entry (fast startup mode).

    NOTE: Heavy database operations (integrity checks, historical analysis) are
    deferred to background tasks to ensure HA startup completes quickly.

    With single-instance architecture, there should only be one config entry.
    A single global coordinator manages all areas.
    """
    _LOGGER.debug("Starting async_setup_entry for entry %s", entry.entry_id)

    # Check if entry is marked for deletion (handled by concurrent migration)
    if entry.data.get("deleted"):
        _LOGGER.info("Entry %s is marked for deletion, skipping setup", entry.entry_id)
        return False

    # Migration check - only migrate if version is explicitly set and less than current
    if entry.version is not None and entry.version != CONF_VERSION:
        if entry.version < CONF_VERSION:
            # This is a real old entry that needs migration
            _LOGGER.info(
                "Migrating Area Occupancy entry from version %s to %s",
                entry.version,
                CONF_VERSION,
            )
            try:
                migration_result = await async_migrate_entry(hass, entry)

                # If migration returned False, check if it's because the entry was deleted
                # (consolidated into another entry). In that case, we stop setup cleanly.
                if not migration_result:
                    if entry.data.get("deleted"):
                        _LOGGER.info(
                            "Entry %s was consolidated during migration, stopping setup",
                            entry.entry_id,
                        )
                        return False

                    # Real failure
                    _validate_migration_result(migration_result, entry.entry_id)

                # Update entry version after successful migration
                hass.config_entries.async_update_entry(entry, version=CONF_VERSION)
                _LOGGER.info(
                    "Migration completed successfully for entry %s", entry.entry_id
                )
            except ConfigEntryNotReady:
                raise
            except Exception as err:
                _LOGGER.error(
                    "Migration threw exception for entry %s: %s", entry.entry_id, err
                )
                raise ConfigEntryNotReady(
                    f"Migration failed with exception: {err}"
                ) from err
    elif entry.version is None or entry.version == 0:
        # Fresh entry - just set the version without migration
        _LOGGER.debug(
            "Setting version %d for fresh entry %s", CONF_VERSION, entry.entry_id
        )
        hass.config_entries.async_update_entry(entry, version=CONF_VERSION)

    # Get or create global coordinator (single-instance architecture)
    # Check if coordinator already exists (shouldn't happen in normal operation,
    # but supports migration scenarios where multiple entries might temporarily exist)
    if DOMAIN not in hass.data:
        # Create and setup coordinator (fast path - no blocking operations)
        _LOGGER.info(
            "Creating global Area Occupancy coordinator for entry %s", entry.entry_id
        )
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

        # Store global coordinator
        hass.data[DOMAIN] = coordinator
    else:
        # Coordinator already exists - reuse it (migration scenario)
        coordinator = hass.data[DOMAIN]
        _LOGGER.info("Reusing existing global coordinator for entry %s", entry.entry_id)

    # Store reference in entry for platform entities to access
    entry.runtime_data = coordinator

    # Setup platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Setup services (idempotent - only needs to run once)
    if DOMAIN not in hass.data.get("_services_setup", {}):
        await async_setup_services(hass)
        if "_services_setup" not in hass.data:
            hass.data["_services_setup"] = {}
        hass.data["_services_setup"][DOMAIN] = True

    # Add update listener
    entry.async_on_unload(entry.add_update_listener(_async_entry_updated))

    # Log setup completion
    area_count = len(coordinator.get_area_names())
    _LOGGER.info(
        "Area Occupancy setup complete for entry %s with %d area(s)",
        entry.entry_id,
        area_count,
    )
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
    _LOGGER.debug("Unloading Area Occupancy config entry %s", entry.entry_id)

    # Unload all platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        # Clean up global coordinator if this is the last/only entry
        # Check if any other entries exist for this domain
        other_entries = [
            e
            for e in hass.config_entries.async_entries(DOMAIN)
            if e.entry_id != entry.entry_id
        ]

        if not other_entries:
            # This is the last entry - clean up global coordinator
            coordinator = hass.data.get(DOMAIN)
            if coordinator is not None:
                _LOGGER.debug("Shutting down global coordinator (last entry)")
                await coordinator.async_shutdown()
                del hass.data[DOMAIN]

            # Clean up services flag
            if (
                "_services_setup" in hass.data
                and DOMAIN in hass.data["_services_setup"]
            ):
                del hass.data["_services_setup"][DOMAIN]
        else:
            _LOGGER.debug(
                "Keeping global coordinator active (other entries exist: %d)",
                len(other_entries),
            )

        # Clear runtime data
        entry.runtime_data = None

    return unload_ok


async def _async_entry_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle config entry update.

    Detects whether the area structure changed (add/remove) or just settings
    changed (threshold, weights).  Structural changes require a full reload
    to create/destroy entity platform entries; setting changes are handled
    with a lightweight in-place update.
    """
    coordinator = hass.data.get(DOMAIN) or entry.runtime_data
    if coordinator is None:
        _LOGGER.warning("Coordinator not found when updating entry %s", entry.entry_id)
        return

    # Determine configured area IDs from merged data+options
    merged = dict(entry.data)
    merged.update(entry.options)
    config_area_ids = {a.get(CONF_AREA_ID) for a in merged.get(CONF_AREAS, [])}

    # Determine currently loaded area IDs
    current_area_ids = {area.config.area_id for area in coordinator.areas.values()}

    if config_area_ids != current_area_ids:
        # Area structure changed — full reload needed for entity platform setup
        _LOGGER.info(
            "Area structure changed (configured=%s, loaded=%s), reloading integration",
            config_area_ids,
            current_area_ids,
        )

        # Remove devices for deleted areas before reload
        removed_area_ids = current_area_ids - config_area_ids
        if removed_area_ids:
            dev_reg = dr.async_get(hass)
            for area_id in removed_area_ids:
                device = dev_reg.async_get_device(identifiers={(DOMAIN, area_id)})
                if device:
                    _LOGGER.info("Removing device for deleted area: %s", area_id)
                    dev_reg.async_remove_device(device.id)

        await hass.config_entries.async_reload(entry.entry_id)
    else:
        # Settings-only change (threshold, weights, etc.) — lightweight update.
        _LOGGER.debug("Config entry settings updated, refreshing area configs")
        for area in coordinator.areas.values():
            area.config.update_from_entry(entry)
        await coordinator.async_request_refresh()
