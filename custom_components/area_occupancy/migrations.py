"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import logging
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

from .const import CONF_AREA_ID, CONF_AREAS, CONF_VERSION, DOMAIN
from .db import DB_NAME

_LOGGER = logging.getLogger(__name__)

# Module-level lock to prevent concurrent migrations
_migration_lock = asyncio.Lock()


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

        # Re-check existence to handle race conditions
        # The asyncio.Lock (_migration_lock) already serializes access within the process
        if not db_path.exists():
            _LOGGER.debug(
                "Database file already deleted by another process, skipping deletion"
            )
            return

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
        except FileNotFoundError:
            # Handled by exists check, but safe to catch just in case
            pass
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
# Area Matching Helpers
# ============================================================================


def _normalize_area_name(name: str) -> str:
    """Normalize area name for comparison.

    Converts to lowercase and removes spaces, underscores, hyphens.

    Args:
        name: Area name to normalize

    Returns:
        Normalized area name
    """
    if not name:
        return ""
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_area_by_normalized_name(
    area_reg: ar.AreaRegistry, target_name: str
) -> str | None:
    """Find area ID by normalized name matching.

    Args:
        area_reg: Home Assistant area registry
        target_name: Area name to find

    Returns:
        Area ID if found, None otherwise
    """
    normalized_target = _normalize_area_name(target_name)
    for area_id, area in area_reg.areas.items():
        if _normalize_area_name(area.name) == normalized_target:
            return area_id
    return None


def _fuzzy_match_area(
    area_reg: ar.AreaRegistry, target_name: str, threshold: float = 0.8
) -> str | None:
    """Find area using fuzzy string matching.

    Uses simple ratio-based matching with difflib.SequenceMatcher.
    Returns area ID if similarity >= threshold, None otherwise.

    Args:
        area_reg: Home Assistant area registry
        target_name: Area name to find
        threshold: Minimum similarity ratio (0.0-1.0) to consider a match

    Returns:
        Area ID if match found above threshold, None otherwise
    """
    normalized_target = _normalize_area_name(target_name)
    best_match = None
    best_ratio = 0.0

    for area_id, area in area_reg.areas.items():
        normalized_area = _normalize_area_name(area.name)
        ratio = SequenceMatcher(None, normalized_target, normalized_area).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = area_id

    if best_ratio >= threshold:
        return best_match
    return None


def _find_or_create_area(area_reg: ar.AreaRegistry, area_name: str) -> str:
    """Find existing area or create new one.

    Tries matching methods in order:
    1. Exact match by ID
    2. Exact match by name
    3. Normalized name matching
    4. Fuzzy matching
    5. Create new area

    Args:
        area_reg: Home Assistant area registry
        area_name: Area name to find or create

    Returns:
        Area ID (existing or newly created)
    """
    # Try exact ID match first
    area_entry = area_reg.async_get_area(area_name)
    if area_entry:
        _LOGGER.debug(
            "Found area '%s' using exact ID match for '%s'",
            area_entry.name,
            area_name,
        )
        return area_entry.id

    # Try exact name match
    area_entry = area_reg.async_get_area_by_name(area_name)
    if area_entry:
        _LOGGER.info(
            "Found area '%s' using exact name match for '%s'",
            area_entry.name,
            area_name,
        )
        return area_entry.id

    # Try normalized matching
    area_id = _find_area_by_normalized_name(area_reg, area_name)
    if area_id:
        matched_area = area_reg.async_get_area(area_id)
        _LOGGER.info(
            "Found area '%s' using normalized matching for '%s'",
            matched_area.name if matched_area else area_id,
            area_name,
        )
        return area_id

    # Try fuzzy matching
    area_id = _fuzzy_match_area(area_reg, area_name)
    if area_id:
        matched_area = area_reg.async_get_area(area_id)
        matched_name = matched_area.name if matched_area else area_id
        _LOGGER.info(
            "Found area '%s' using fuzzy matching for '%s'",
            matched_name,
            area_name,
        )
        return area_id

    # Create new area
    _LOGGER.info(
        "No matching area found for '%s', creating new area in Home Assistant",
        area_name,
    )
    created_area = area_reg.async_create(area_name)
    return created_area.id


# ============================================================================
# Entry Migration (Main Entry Point)
# ============================================================================


def _convert_entry_to_area_config(
    entry: ConfigEntry, hass: HomeAssistant
) -> dict[str, Any] | None:
    """Convert old single-area config entry to new area config dict format.

    Args:
        entry: Config entry with old format (single area config in data)
        hass: Home Assistant instance for area registry validation

    Returns:
        Dictionary representing an area config in the new format, or None if area ID is invalid
    """
    # Merge data and options (Home Assistant pattern)
    merged = dict(entry.data)
    if entry.options:
        merged.update(entry.options)

    # Get area ID (with fallback to title/unique_id)
    area_id = merged.get(CONF_AREA_ID)
    if not area_id:
        # Try to use entry title or unique_id as fallback
        area_id = getattr(entry, "title", None) or getattr(entry, "unique_id", None)
        if area_id:
            _LOGGER.warning(
                "Entry %s missing CONF_AREA_ID, using title/unique_id: %s",
                entry.entry_id,
                area_id,
            )
            merged[CONF_AREA_ID] = area_id

    if not area_id:
        _LOGGER.error(
            "Entry %s missing CONF_AREA_ID and no fallback available",
            entry.entry_id,
        )
        return None

    # LOG: Starting area processing
    _LOGGER.info(
        "Processing area '%s' from entry %s",
        area_id,
        entry.entry_id,
    )

    # Find or create area in Home Assistant
    area_reg = ar.async_get(hass)
    matched_area_id = _find_or_create_area(area_reg, area_id)

    # LOG: Result
    matched_area = area_reg.async_get_area(matched_area_id)
    if matched_area:
        _LOGGER.info(
            "Area '%s' from entry %s -> matched/created area '%s' (ID: %s)",
            area_id,
            entry.entry_id,
            matched_area.name,
            matched_area_id,
        )

    # Update merged config with matched/created area ID
    merged[CONF_AREA_ID] = matched_area_id

    # Return merged dict as area config (preserve all keys)
    return merged


def _combine_config_entries(
    entries: list[ConfigEntry], hass: HomeAssistant
) -> list[dict[str, Any]]:
    """Combine multiple old config entries into list of area config dicts.

    Args:
        entries: List of config entries with old format (version < 13)
        hass: Home Assistant instance for area registry validation

    Returns:
        List of area config dictionaries in new format (invalid areas are filtered out)
    """
    area_configs: list[dict[str, Any]] = []

    for entry in entries:
        area_config = _convert_entry_to_area_config(entry, hass)
        if area_config is not None:
            area_configs.append(area_config)

    return area_configs


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:  # noqa: C901
    """Migrate old entry to the new version.

    This migration combines multiple old config entries (each representing one area)
    into a single config entry with CONF_AREAS list format.

    Uses asyncio.Lock to prevent concurrent migrations within the single process.

    Args:
        hass: Home Assistant instance
        config_entry: The config entry being migrated

    Returns:
        True if migration succeeded, False otherwise
    """
    # Acquire async lock to prevent concurrent migration attempts
    async with _migration_lock:
        # Handle v13 breaking change: reset database if needed for older versions
        # Moved inside lock to serialize DB deletion across concurrent setup attempts
        await async_reset_database_if_needed(hass, config_entry.version)

        # If entry is already at version 13 or higher, no migration needed
        if config_entry.version >= CONF_VERSION:
            # Check if we were deleted while waiting for lock
            if config_entry.data.get("deleted"):
                _LOGGER.info(
                    "Entry %s was marked deleted while waiting for lock, stopping setup",
                    config_entry.entry_id,
                )
                return False

            _LOGGER.debug(
                "Entry %s already at version %d, no migration needed",
                config_entry.entry_id,
                config_entry.version,
            )
            return True
        # EARLY CHECK: Before even starting executor, check if migration is done
        # This prevents multiple executors from starting concurrently
        all_entries = hass.config_entries.async_entries(DOMAIN)
        old_entries = [entry for entry in all_entries if entry.version < CONF_VERSION]

        if not old_entries:
            _LOGGER.debug(
                "Migration already completed by another entry, skipping entry %s",
                config_entry.entry_id,
            )
            return True

        # Check if this specific entry was already migrated
        current_entry = next(
            (e for e in all_entries if e.entry_id == config_entry.entry_id), None
        )
        if current_entry and current_entry.version >= CONF_VERSION:
            _LOGGER.debug("Entry %s already migrated, skipping", config_entry.entry_id)
            return True

        # Track entry IDs that need cleanup (set by migration function)
        cleanup_entry_ids: list[str] = []
        entries_to_remove: list[str] = []

        async def _migrate() -> bool:
            """Perform migration logic (async)."""
            try:
                # Refresh registry entries
                all_entries = hass.config_entries.async_entries(DOMAIN)

                # Check status of CURRENT entry
                current_entry = next(
                    (e for e in all_entries if e.entry_id == config_entry.entry_id),
                    None,
                )

                # If entry is gone or marked deleted, stop immediately
                if not current_entry:
                    _LOGGER.info(
                        "Entry %s not found in registry, stopping setup",
                        config_entry.entry_id,
                    )
                    return False

                if current_entry.data.get("deleted"):
                    _LOGGER.info(
                        "Entry %s is marked deleted, stopping setup",
                        config_entry.entry_id,
                    )
                    return False

                # Identify all entries needing migration
                old_entries = [
                    entry for entry in all_entries if entry.version < CONF_VERSION
                ]

                if not old_entries:
                    # Check for any "deleted" entries that need removal
                    # (This happens if another process did the migration but couldn't remove them)
                    for entry in all_entries:
                        if entry.data.get("deleted"):
                            _LOGGER.debug(
                                "Found pending deleted entry %s, adding to cleanup",
                                entry.entry_id,
                            )
                            entries_to_remove.append(entry.entry_id)
                            cleanup_entry_ids.append(entry.entry_id)

                    # No migration needed. Since we passed the "deleted" check above,
                    # we are a valid, surviving entry (or already migrated).
                    _LOGGER.debug("No entries requiring migration found")
                    return True

                # Deterministic target selection: Sort by entry_id and pick first
                old_entries.sort(key=lambda e: e.entry_id)
                target_entry = old_entries[0]

                _LOGGER.info(
                    "Migrating %d entries. Target entry: %s",
                    len(old_entries),
                    target_entry.entry_id,
                )

                # Consolidate all areas into one list
                area_configs = _combine_config_entries(old_entries, hass)

                if not area_configs:
                    _LOGGER.error(
                        "Failed to convert any entries to area configs. Migration aborted."
                    )
                    return False

                # Filter invalid areas
                area_reg = ar.async_get(hass)
                valid_areas = []
                for area_config in area_configs:
                    area_id = area_config.get(CONF_AREA_ID)
                    if area_id and area_reg.async_get_area(area_id):
                        valid_areas.append(area_config)
                    else:
                        _LOGGER.warning("Removing invalid area ID '%s'", area_id)

                if not valid_areas:
                    _LOGGER.error("No valid areas found after migration. Aborting.")
                    return False

                # Update TARGET entry with consolidated areas
                new_data = {CONF_AREAS: valid_areas}
                hass.config_entries.async_update_entry(
                    target_entry,
                    data=new_data,
                    version=CONF_VERSION,
                )

                _LOGGER.info(
                    "Successfully updated target entry %s with %d areas",
                    target_entry.entry_id,
                    len(valid_areas),
                )

                # Handle entries to remove (everyone except target)
                for entry in old_entries:
                    if entry.entry_id != target_entry.entry_id:
                        # Update to v13 + deleted to prevent them from running migration again
                        hass.config_entries.async_update_entry(
                            entry,
                            data={"deleted": True},
                            version=CONF_VERSION,
                        )
                        entries_to_remove.append(entry.entry_id)
                        cleanup_entry_ids.append(entry.entry_id)

                    # Also add to cleanup to ensure device registry is cleaned for all old entries
                    # (Target entry also needs its old device cleaned up/replaced)
                    if entry.entry_id == target_entry.entry_id:
                        cleanup_entry_ids.append(entry.entry_id)

                # If WE are the target, return True (continue setup).
                # If we are one of the removed ones, return False (stop setup).
                if config_entry.entry_id == target_entry.entry_id:
                    return True

                _LOGGER.info(
                    "Current entry %s was consolidated into %s. Stopping setup.",
                    config_entry.entry_id,
                    target_entry.entry_id,
                )
                return False  # noqa: TRY300

            except Exception:
                _LOGGER.exception("Unexpected error during migration")
                return False

        # Perform migration (in async loop, protected by lock)
        migration_result = await _migrate()

        # After successful migration, perform cleanup (safe in async loop)
        if cleanup_entry_ids:
            # Use set to remove duplicates if any
            unique_cleanup_ids = list(set(cleanup_entry_ids))
            _LOGGER.info(
                "Cleaning up devices and entities from registries for %d config entry(ies)",
                len(unique_cleanup_ids),
            )
            (
                devices_removed,
                entities_removed,
            ) = await _cleanup_registry_devices_and_entities(hass, unique_cleanup_ids)
            if devices_removed > 0 or entities_removed > 0:
                _LOGGER.info(
                    "Registry cleanup: removed %d device(s) and %d entity(ies). "
                    "They will be recreated with new unique IDs during setup.",
                    devices_removed,
                    entities_removed,
                )

    # REMOVAL MUST BE OUTSIDE LOCK TO PREVENT DEADLOCK
    # We run removals as background tasks to avoid blocking the surviving entry's setup
    # if the old entries are currently busy/locked.

    async def _remove_entry_safe(entry_id: str) -> None:
        """Remove entry safely in background."""
        _LOGGER.info("Removing old config entry %s (background task)", entry_id)
        try:
            await hass.config_entries.async_remove(entry_id)
        except (OSError, KeyError, ValueError) as err:
            _LOGGER.error("Failed to remove old entry %s: %s", entry_id, err)

    # Ensure we remove the current entry LAST
    entries_to_remove.sort(key=lambda eid: eid == config_entry.entry_id)

    for entry_id in entries_to_remove:
        hass.async_create_task(_remove_entry_safe(entry_id))

    if entries_to_remove:
        _LOGGER.info(
            "Migration completed: %d entry(ies) scheduled for removal",
            len(entries_to_remove),
        )

    return migration_result
