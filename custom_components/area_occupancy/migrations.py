"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from filelock import FileLock

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er

from .const import CONF_AREA_ID, CONF_AREAS, CONF_VERSION, DOMAIN
from .db import DB_NAME

_LOGGER = logging.getLogger(__name__)


# ============================================================================
# Database Migrations
# ============================================================================


async def async_reset_database_if_needed(hass: HomeAssistant, version: int) -> None:
    """Delete database file for schema migration if needed.

    Args:
        hass: Home Assistant instance
        version: Version number from the config entry
    """
    storage_dir = Path(hass.config.config_dir) / ".storage"
    db_path = storage_dir / DB_NAME

    def _delete_database_file() -> None:
        """Blocking helper: delete database file if entry version requires migration."""
        # Check if entry version requires database reset
        # Version 13 introduced breaking changes requiring DB reset
        if version >= 13:
            _LOGGER.debug(
                "Config entry version %d is current, no database migration needed",
                version,
            )
            return

        # Version is older than 13, migration is needed
        if not db_path.exists():
            _LOGGER.debug(
                "Database file does not exist for version %d migration, "
                "will be created with new schema",
                version,
            )
            return

        lock_path = storage_dir / (DB_NAME + ".lock")
        try:
            with FileLock(lock_path):
                _LOGGER.info(
                    "Config entry version %d is older than current version %d. "
                    "Deleting database file to allow recreation with new schema.",
                    version,
                    CONF_VERSION,
                )

                # Delete database file and associated WAL files
                try:
                    db_path.unlink()
                    _LOGGER.debug("Deleted database file: %s", db_path)
                except Exception as e:  # noqa: BLE001
                    _LOGGER.warning("Failed to delete database file %s: %s", db_path, e)

                # Delete WAL and shared memory files if they exist
                wal_path = storage_dir / (DB_NAME + "-wal")
                shm_path = storage_dir / (DB_NAME + "-shm")
                for path in [wal_path, shm_path]:
                    if path.exists():
                        try:
                            path.unlink()
                            _LOGGER.debug("Deleted database file: %s", path)
                        except Exception as e:  # noqa: BLE001
                            _LOGGER.debug("Failed to delete %s: %s", path, e)
        finally:
            try:
                if lock_path.exists():
                    lock_path.unlink()
                    _LOGGER.debug("Removed leftover lock file: %s", lock_path)
            except Exception as cleanup_err:  # noqa: BLE001
                _LOGGER.debug("Error during lock cleanup: %s", cleanup_err)

    await hass.async_add_executor_job(_delete_database_file)


# ============================================================================
# Registry Cleanup
# ============================================================================


async def _cleanup_registry_devices_and_entities(
    hass: HomeAssistant, entry_ids: list[str]
) -> tuple[int, int]:
    """Remove all devices and entities from registries for given config entries.

    This function removes all devices and entities associated with the given
    config entry IDs. This is needed during migration because unique IDs have
    changed, and old devices/entities would become orphaned.

    Args:
        hass: Home Assistant instance
        entry_ids: List of config entry IDs to clean up

    Returns:
        Tuple of (devices_removed, entities_removed) counts
    """
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)

    devices_removed = 0
    entities_removed = 0

    # Remove all entities for each config entry
    for entry_id in entry_ids:
        # Find and remove all entities with matching config_entry_id
        entities_to_remove = []
        for entity_id, entity_entry in entity_registry.entities.items():
            if entity_entry.config_entry_id == entry_id:
                entities_to_remove.append(entity_id)

        for entity_id in entities_to_remove:
            try:
                entity_registry.async_remove(entity_id)
                entities_removed += 1
                _LOGGER.debug(
                    "Removed entity %s from registry (config_entry: %s)",
                    entity_id,
                    entry_id,
                )
            except (ValueError, KeyError, AttributeError) as err:
                _LOGGER.warning(
                    "Failed to remove entity %s from registry: %s", entity_id, err
                )

        # Find and remove all devices with matching config_entry_id
        devices_to_remove = [
            device.id
            for device in device_registry.devices.values()
            if entry_id in device.config_entries
        ]

        for device_id in devices_to_remove:
            try:
                device_registry.async_remove_device(device_id)
                devices_removed += 1
                _LOGGER.debug(
                    "Removed device %s from registry (config_entry: %s)",
                    device_id,
                    entry_id,
                )
            except (ValueError, KeyError, AttributeError) as err:
                _LOGGER.warning(
                    "Failed to remove device %s from registry: %s", device_id, err
                )

    if devices_removed > 0 or entities_removed > 0:
        _LOGGER.info(
            "Registry cleanup completed: removed %d device(s) and %d entity(ies) "
            "for %d config entry(ies)",
            devices_removed,
            entities_removed,
            len(entry_ids),
        )

    return devices_removed, entities_removed


# ============================================================================
# Entry Migration (Main Entry Point)
# ============================================================================


def _convert_entry_to_area_config(entry: ConfigEntry) -> dict[str, Any]:
    """Convert old single-area config entry to new area config dict format.

    Args:
        entry: Config entry with old format (single area config in data)

    Returns:
        Dictionary representing an area config in the new format
    """
    # Merge data and options (Home Assistant pattern)
    merged = dict(entry.data)
    if entry.options:
        merged.update(entry.options)

    # Ensure CONF_AREA_ID exists (required field)
    if CONF_AREA_ID not in merged:
        # Try to use entry title or unique_id as fallback
        area_id = getattr(entry, "title", None) or getattr(entry, "unique_id", None)
        if area_id:
            _LOGGER.warning(
                "Entry %s missing CONF_AREA_ID, using title/unique_id: %s",
                entry.entry_id,
                area_id,
            )
            merged[CONF_AREA_ID] = area_id
        else:
            _LOGGER.error(
                "Entry %s missing CONF_AREA_ID and no fallback available",
                entry.entry_id,
            )
            raise ValueError(f"Entry {entry.entry_id} missing required CONF_AREA_ID")

    # Return merged dict as area config (preserve all keys)
    return merged


def _combine_config_entries(entries: list[ConfigEntry]) -> list[dict[str, Any]]:
    """Combine multiple old config entries into list of area config dicts.

    Args:
        entries: List of config entries with old format (version < 13)

    Returns:
        List of area config dictionaries in new format

    Raises:
        ValueError: If any entry fails to convert (e.g., missing required CONF_AREA_ID)
        AttributeError: If entry is missing required attributes
        KeyError: If entry data is invalid
    """
    area_configs: list[dict[str, Any]] = []

    for entry in entries:
        area_config = _convert_entry_to_area_config(entry)
        area_configs.append(area_config)

    return area_configs


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to the new version.

    This migration combines multiple old config entries (each representing one area)
    into a single config entry with CONF_AREAS list format.

    Args:
        hass: Home Assistant instance
        config_entry: The config entry being migrated

    Returns:
        True if migration succeeded, False otherwise
    """
    # Handle v13 breaking change: reset database if needed for older versions
    await async_reset_database_if_needed(hass, config_entry.version)

    # If entry is already at version 13 or higher, no migration needed
    if config_entry.version >= CONF_VERSION:
        _LOGGER.debug(
            "Entry %s already at version %d, no migration needed",
            config_entry.entry_id,
            config_entry.version,
        )
        return True

    # Get all config entries for this domain
    all_entries = hass.config_entries.async_entries(DOMAIN)

    # Filter entries that need migration (version < 13)
    old_entries = [entry for entry in all_entries if entry.version < CONF_VERSION]

    if not old_entries:
        _LOGGER.debug(
            "No entries with version < %d found, migration not needed",
            CONF_VERSION,
        )
        return True

    _LOGGER.info(
        "Found %d config entry(ies) to migrate from version < %d",
        len(old_entries),
        CONF_VERSION,
    )

    # Clean up all devices and entities for all domain entries before migration
    # This removes orphaned devices/entities with old unique IDs
    # Setup will recreate them with new unique IDs after migration
    _LOGGER.info(
        "Cleaning up devices and entities from registries for %d config entry(ies)",
        len(all_entries),
    )
    all_entry_ids = [entry.entry_id for entry in all_entries]
    devices_removed, entities_removed = await _cleanup_registry_devices_and_entities(
        hass, all_entry_ids
    )
    if devices_removed > 0 or entities_removed > 0:
        _LOGGER.info(
            "Registry cleanup: removed %d device(s) and %d entity(ies). "
            "They will be recreated with new unique IDs during setup.",
            devices_removed,
            entities_removed,
        )

    # If there's only one old entry, convert it to new format
    if len(old_entries) == 1:
        _LOGGER.info(
            "Migrating single config entry %s to multi-area format",
            config_entry.entry_id,
        )
        try:
            area_config = _convert_entry_to_area_config(config_entry)
            new_data = {CONF_AREAS: [area_config]}

            # Update entry with new format
            hass.config_entries.async_update_entry(
                config_entry,
                data=new_data,
                version=CONF_VERSION,
            )
            _LOGGER.info(
                "Successfully migrated single entry %s to multi-area format",
                config_entry.entry_id,
            )
        except (ValueError, AttributeError, KeyError, OSError) as err:
            _LOGGER.error(
                "Failed to migrate single entry %s: %s", config_entry.entry_id, err
            )
            return False
        else:
            return True

    # Multiple entries need to be combined
    _LOGGER.info(
        "Combining %d config entries into single multi-area entry",
        len(old_entries),
    )

    try:
        # Convert all old entries to area config dicts
        area_configs = _combine_config_entries(old_entries)

        if not area_configs:
            _LOGGER.error(
                "Failed to convert any entries to area configs. Migration aborted."
            )
            return False

        # Check for duplicate area IDs
        area_ids = [config.get(CONF_AREA_ID) for config in area_configs]
        if len(area_ids) != len(set(area_ids)):
            _LOGGER.warning(
                "Duplicate area IDs found in entries to migrate. "
                "This may cause issues. Area IDs: %s",
                area_ids,
            )

        # Use the first entry as the target for the combined configuration
        target_entry = old_entries[0]

        # Create new data structure with CONF_AREAS list
        new_data = {CONF_AREAS: area_configs}

        # Update the target entry with combined data
        hass.config_entries.async_update_entry(
            target_entry,
            data=new_data,
            version=CONF_VERSION,
        )

        _LOGGER.info(
            "Successfully updated entry %s with %d area(s)",
            target_entry.entry_id,
            len(area_configs),
        )

        # Delete other old entries
        entries_to_remove = [entry for entry in old_entries if entry != target_entry]
        for entry in entries_to_remove:
            _LOGGER.info("Removing old config entry %s", entry.entry_id)
            try:
                hass.config_entries.async_remove(entry.entry_id)
            except (OSError, KeyError, ValueError) as err:
                _LOGGER.error("Failed to remove old entry %s: %s", entry.entry_id, err)
                # Continue removing other entries even if one fails

        _LOGGER.info(
            "Migration completed: %d entry(ies) combined into entry %s",
            len(old_entries),
            target_entry.entry_id,
        )
    except (ValueError, AttributeError, KeyError, OSError):
        _LOGGER.exception("Failed to combine config entries during migration")
        return False
    else:
        return True
