"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

from .binary_sensor import NAME_BINARY_SENSOR
from .const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MOTION_SENSORS,
    CONF_MOTION_TIMEOUT,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
    CONF_WINDOW_ACTIVE_STATE,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_MOTION_TIMEOUT,
    DEFAULT_NAME,
    DEFAULT_PURPOSE,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOMAIN,
    PLATFORMS,
)
from .db import DB_NAME, DB_VERSION
from .number import NAME_THRESHOLD_NUMBER
from .sensor import NAME_DECAY_SENSOR, NAME_PRIORS_SENSOR, NAME_PROBABILITY_SENSOR

_LOGGER = logging.getLogger(__name__)


# ============================================================================
# Entity Registry Migrations
# ============================================================================


async def async_migrate_unique_ids(
    hass: HomeAssistant, config_entry: ConfigEntry, platform: str
) -> None:
    """Migrate unique IDs of entities in the entity registry."""
    _LOGGER.debug("Starting unique ID migration for platform %s", platform)
    entity_registry = er.async_get(hass)
    entry_id = config_entry.entry_id

    # Define which entity types to look for based on platform
    entity_types = {
        "sensor": [NAME_PROBABILITY_SENSOR, NAME_DECAY_SENSOR, NAME_PRIORS_SENSOR],
        "binary_sensor": [NAME_BINARY_SENSOR],
        "number": [NAME_THRESHOLD_NUMBER],
    }

    if platform not in entity_types:
        _LOGGER.debug("Platform %s not in entity_types, skipping", platform)
        return

    # Get the old format prefix to look for
    old_prefix = f"{DOMAIN}_{entry_id}_"
    _LOGGER.debug("Looking for entities with old prefix: %s", old_prefix)

    # Find entities matching the old prefix
    matching_entities = _find_entities_by_prefix(
        entity_registry, old_prefix, config_entry.entry_id
    )
    updated_entries = 0

    for entity_id, entity_entry in matching_entities:
        old_unique_id = entity_entry.unique_id
        # Simply remove the domain prefix to get the new ID
        new_unique_id = str(old_unique_id).replace(old_prefix, f"{entry_id}_").lower()

        # Update the unique ID in the registry (no conflict checking needed here)
        _LOGGER.info(
            "Migrating unique ID for %s: %s -> %s",
            entity_id,
            old_unique_id,
            new_unique_id,
        )
        _update_entity_unique_id(
            entity_registry,
            entity_id,
            str(old_unique_id),
            new_unique_id,
            check_conflicts=False,
        )
        updated_entries += 1

    if updated_entries > 0:
        _LOGGER.info(
            "Completed migrating %d unique IDs for platform %s",
            updated_entries,
            platform,
        )
    else:
        _LOGGER.debug("No unique IDs to migrate for platform %s", platform)


# Entity Registry Helper Functions
# ==========================================


def _find_entities_by_prefix(
    entity_registry: er.EntityRegistry,
    prefix: str,
    config_entry_id: str | None = None,
) -> list[tuple[str, er.RegistryEntry]]:
    """Find entities matching a prefix pattern.

    Args:
        entity_registry: The entity registry to search
        prefix: The prefix to match against unique IDs
        config_entry_id: Optional config entry ID to filter by

    Returns:
        List of (entity_id, entity_entry) tuples matching the prefix
    """
    matches = []
    for entity_id, entity_entry in entity_registry.entities.items():
        # Filter by config_entry_id if provided
        if config_entry_id and entity_entry.config_entry_id != config_entry_id:
            continue

        old_unique_id = entity_entry.unique_id
        if old_unique_id is not None and str(old_unique_id).startswith(prefix):
            matches.append((entity_id, entity_entry))
    return matches


def _check_unique_id_conflict(
    entity_registry: er.EntityRegistry,
    new_unique_id: str,
    exclude_entity_id: str,
) -> tuple[bool, str | None]:
    """Check if a unique ID already exists in the registry.

    Args:
        entity_registry: The entity registry to check
        new_unique_id: The unique ID to check for conflicts
        exclude_entity_id: Entity ID to exclude from conflict check (the entity being migrated)

    Returns:
        Tuple of (has_conflict: bool, conflicting_entity_id: str | None)
    """
    new_unique_id_lower = str(new_unique_id).lower()
    for other_entity_id, other_entity_entry in entity_registry.entities.items():
        if (
            other_entity_id != exclude_entity_id
            and str(other_entity_entry.unique_id).lower() == new_unique_id_lower
        ):
            return True, other_entity_id
    return False, None


def _update_entity_unique_id(
    entity_registry: er.EntityRegistry,
    entity_id: str,
    old_unique_id: str,
    new_unique_id: str,
    check_conflicts: bool = True,
) -> tuple[bool, str | None]:
    """Update entity unique ID with optional conflict checking.

    Args:
        entity_registry: The entity registry to update
        entity_id: The entity ID to update
        old_unique_id: The current unique ID (for logging)
        new_unique_id: The new unique ID to set
        check_conflicts: Whether to check for conflicts before updating

    Returns:
        Tuple of (success: bool, conflict_entity_id: str | None)
        If check_conflicts is True and a conflict exists, returns (False, conflict_entity_id)
        Otherwise returns (True, None)
    """
    # Ensure unique_id is lowercase
    new_unique_id = str(new_unique_id).lower()

    if check_conflicts:
        has_conflict, conflict_entity_id = _check_unique_id_conflict(
            entity_registry, new_unique_id, entity_id
        )
        if has_conflict:
            return False, conflict_entity_id

    entity_registry.async_update_entity(entity_id, new_unique_id=new_unique_id)
    return True, None


# ============================================================================
# Configuration Migration Constants and Helpers
# ============================================================================

DECAY_MIN_DELAY_KEY = "decay_min_delay"
CONF_LIGHTS_KEY = "lights"
CONF_DECAY_WINDOW_KEY = "decay_window"
CONF_HISTORICAL_ANALYSIS_ENABLED = "historical_analysis_enabled"
CONF_HISTORY_PERIOD = "history_period"


# Configuration Migration Helper Functions
# ==========================================


def _safe_file_operation(operation: Callable[[], Any], error_message: str) -> bool:
    """Safely execute a file operation with error handling.

    Args:
        operation: Callable that performs the file operation
        error_message: Error message to log if operation fails

    Returns:
        True if operation succeeded, False otherwise
    """
    try:
        operation()
    except OSError as err:
        _LOGGER.warning("%s: %s", error_message, err)
        return False
    else:
        return True


def _safe_database_operation(operation: Callable[[], Any], error_message: str) -> bool:
    """Safely execute a database operation with error handling.

    Args:
        operation: Callable that performs the database operation
        error_message: Error message to log if operation fails

    Returns:
        True if operation succeeded, False otherwise
    """
    try:
        operation()
    except (SQLAlchemyError, OSError) as err:
        _LOGGER.warning("%s: %s", error_message, err)
        return False
    else:
        return True


def _remove_deprecated_keys(
    config: dict[str, Any], keys: list[str], description: str = ""
) -> dict[str, Any]:
    """Remove deprecated keys from config.

    Args:
        config: The configuration dictionary to modify
        keys: List of keys to remove
        description: Optional description for logging (if empty, uses key names)

    Returns:
        The modified configuration dictionary
    """
    removed_keys = []
    for key in keys:
        if key in config:
            config.pop(key)
            removed_keys.append(key)
            log_description = description or f"deprecated {key}"
            _LOGGER.debug("Removed %s from config", log_description)
    return config


# Configuration Migration Functions
# ==========================================


def remove_decay_min_delay(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated decay delay option from config."""
    return _remove_deprecated_keys(
        config, [DECAY_MIN_DELAY_KEY], "deprecated decay_min_delay"
    )


def remove_lights_key(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated lights key from config."""
    return _remove_deprecated_keys(config, [CONF_LIGHTS_KEY], "deprecated lights key")


def remove_decay_window_key(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated decay window key from config."""
    return _remove_deprecated_keys(
        config, [CONF_DECAY_WINDOW_KEY], "deprecated decay window key"
    )


def remove_history_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated history period key from config."""
    return _remove_deprecated_keys(
        config,
        [CONF_HISTORY_PERIOD, CONF_HISTORICAL_ANALYSIS_ENABLED],
        "deprecated history keys",
    )


def _add_default_if_missing(
    config: dict[str, Any],
    key: str,
    default_value: Any,
    log_message: str | None = None,
) -> dict[str, Any]:
    """Add a default value to config if key is missing.

    Args:
        config: The configuration dictionary to modify
        key: The configuration key to check/add
        default_value: The default value to use if key is missing
        log_message: Optional custom log message (if None, uses default format)

    Returns:
        The modified configuration dictionary
    """
    if key not in config:
        config[key] = default_value
        if log_message:
            _LOGGER.debug(log_message)
        else:
            _LOGGER.debug(
                "Added %s to config with default value: %s", key, default_value
            )
    return config


def migrate_decay_half_life(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add decay half life."""
    return _add_default_if_missing(
        config,
        CONF_DECAY_HALF_LIFE,
        DEFAULT_DECAY_HALF_LIFE,
        "Added decay half life to config",
    )


def migrate_primary_occupancy_sensor(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add primary occupancy sensor.

    This migration:
    1. Takes the first motion sensor as the primary occupancy sensor if none is set
    2. Preserves any existing primary occupancy sensor setting
    3. Logs the migration for debugging

    Args:
        config: The configuration to migrate

    Returns:
        The migrated configuration

    """
    if CONF_PRIMARY_OCCUPANCY_SENSOR not in config:
        motion_sensors = config.get(CONF_MOTION_SENSORS, [])
        if motion_sensors:
            config[CONF_PRIMARY_OCCUPANCY_SENSOR] = motion_sensors[0]
            _LOGGER.debug(
                "Migrated primary occupancy sensor to first motion sensor: %s",
                motion_sensors[0],
            )
        else:
            _LOGGER.debug(
                "No motion sensors found for primary occupancy sensor migration"
            )

    return config


def migrate_purpose_field(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add purpose field with default value.

    This migration:
    1. Adds the purpose field with default value if it doesn't exist
    2. Preserves any existing purpose setting
    3. Logs the migration for debugging

    Args:
        config: The configuration to migrate

    Returns:
        The migrated configuration

    """
    return _add_default_if_missing(
        config,
        CONF_PURPOSE,
        DEFAULT_PURPOSE,
        f"Migrated purpose field to default value: {DEFAULT_PURPOSE}",
    )


def migrate_motion_timeout(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add motion timeout."""
    return _add_default_if_missing(
        config,
        CONF_MOTION_TIMEOUT,
        DEFAULT_MOTION_TIMEOUT,
        f"Added motion timeout to config: {DEFAULT_MOTION_TIMEOUT}",
    )


# Configuration Migration Orchestration
# ==========================================


def migrate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to latest version.

    Args:
        config: The configuration to migrate

    Returns:
        The migrated configuration

    """
    # Apply migrations in order
    config = remove_decay_min_delay(config)
    config = migrate_primary_occupancy_sensor(config)
    config = migrate_decay_half_life(config)
    config = remove_decay_window_key(config)
    config = remove_lights_key(config)
    config = remove_history_keys(config)
    config = migrate_purpose_field(config)
    return migrate_motion_timeout(config)


# ============================================================================
# Storage Migrations
# ============================================================================

LEGACY_STORAGE_KEY = "area_occupancy.storage"


async def async_migrate_storage(
    hass: HomeAssistant, entry_id: str, entry_major: int
) -> None:
    """Migrate legacy multi-instance storage to per-entry storage format."""
    try:
        _LOGGER.debug("Starting storage migration for entry %s", entry_id)

        # Check for and clean up legacy multi-instance storage using direct file operations
        storage_dir = Path(hass.config.config_dir) / ".storage"
        legacy_file = storage_dir / LEGACY_STORAGE_KEY

        if legacy_file.exists():
            _LOGGER.info(
                "Found legacy storage file %s, removing it for fresh start",
                legacy_file.name,
            )
            if _safe_file_operation(
                lambda: legacy_file.unlink(),
                f"Error removing legacy storage file {legacy_file}",
            ):
                _LOGGER.info("Successfully removed legacy storage file")

        # Reset database for version < 11
        await async_reset_database_if_needed(hass, entry_major)

        _LOGGER.debug("Storage migration completed for entry %s", entry_id)
    except (HomeAssistantError, OSError, ValueError) as err:
        _LOGGER.error("Error during storage migration for entry %s: %s", entry_id, err)


# ============================================================================
# Database Migrations
# ============================================================================


def _update_db_version(session: Any, version: int) -> None:
    """Update database version in metadata table.

    Args:
        session: SQLAlchemy session
        version: Version number to set
    """
    try:
        session.execute(
            text("UPDATE metadata SET value = :version WHERE key = 'db_version'"),
            {"version": str(version)},
        )
        if session.execute(text("SELECT changes()")).scalar() == 0:
            session.execute(
                text(
                    "INSERT INTO metadata (key, value) VALUES ('db_version', :version)"
                ),
                {"version": str(version)},
            )
    except Exception:  # noqa: BLE001
        pass


def _drop_legacy_tables(engine: Any, session: Any) -> None:
    """Drop legacy database tables.

    Args:
        engine: SQLAlchemy engine
        session: SQLAlchemy session

    Note: Deprecated - use _drop_all_tables instead for version 5+
    """
    _LOGGER.info("Dropping legacy tables for schema migration")
    _update_db_version(session, DB_VERSION)
    session.commit()

    with engine.connect() as conn:
        tables_to_drop = [
            "intervals",
            "priors",
            "entities",
            "areas",
            "metadata",
        ]
        for table_name in tables_to_drop:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                _LOGGER.debug("Dropped table: %s", table_name)
            except Exception as e:  # noqa: BLE001
                _LOGGER.debug("Error dropping table %s: %s", table_name, e)
        conn.commit()
        _LOGGER.info("All legacy tables dropped successfully")


def _drop_all_tables(engine: Any, session: Any) -> None:
    """Drop all database tables for complete schema reset.

    Args:
        engine: SQLAlchemy engine
        session: SQLAlchemy session
    """
    _LOGGER.info("Dropping all tables for complete database reset")
    _update_db_version(session, DB_VERSION)
    session.commit()

    with engine.connect() as conn:
        # Drop all tables including new ones
        tables_to_drop = [
            "cross_area_stats",
            "area_relationships",
            "entity_statistics",
            "numeric_correlations",
            "numeric_aggregates",
            "numeric_samples",
            "global_priors",
            "occupied_intervals_cache",
            "interval_aggregates",
            "intervals",
            "priors",
            "entities",
            "areas",
            "metadata",
        ]
        for table_name in tables_to_drop:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                _LOGGER.debug("Dropped table: %s", table_name)
            except Exception as e:  # noqa: BLE001
                _LOGGER.debug("Error dropping table %s: %s", table_name, e)
        conn.commit()
        _LOGGER.info(
            "All tables dropped successfully - database will be recreated with new schema"
        )


def _drop_tables_locked(
    storage_dir: Path,
    entry_major: int,
    engine_factory: Any | None = None,
    session_factory: Any | None = None,
) -> None:
    """Blocking helper: perform locked drop of tables if DB version doesn't match.

    Args:
        storage_dir: Directory containing the database file
        entry_major: Major version number of the entry (for compatibility, not used for DB version check)
        engine_factory: Optional factory function to create engine (for testing)
        session_factory: Optional factory function to create session (for testing)
    """
    db_path = storage_dir / DB_NAME
    if not db_path.exists():
        _LOGGER.debug("Database file does not exist, no migration needed")
        return

    lock_path = storage_dir / (DB_NAME + ".lock")
    try:
        with FileLock(lock_path):
            if engine_factory is not None:
                engine = engine_factory()
            else:
                engine = create_engine(f"sqlite:///{db_path}")

            if session_factory is not None:
                session = session_factory()
            else:
                session = sessionmaker(bind=engine)()

            db_version = 0
            try:
                result = session.execute(
                    text("SELECT value FROM metadata WHERE key = 'db_version'")
                )
                row = result.fetchone()
                if row:
                    db_version = int(row[0])
            except Exception:  # noqa: BLE001
                db_version = 0

            # For DB_VERSION 5 (new schema), delete and recreate database if version doesn't match
            if db_version != DB_VERSION:
                _LOGGER.info(
                    "Database version mismatch (found %d, expected %d). "
                    "Deleting and recreating database with new schema.",
                    db_version,
                    DB_VERSION,
                )
                _drop_all_tables(engine, session)

            session.close()
            engine.dispose()
            _LOGGER.debug("Database engine disposed")
            _LOGGER.info("Tables dropped successfully")
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
                _LOGGER.debug("Removed leftover lock file: %s", lock_path)
        except Exception as cleanup_err:  # noqa: BLE001
            _LOGGER.debug("Error during lock cleanup: %s", cleanup_err)


async def async_reset_database_if_needed(hass: HomeAssistant, entry_major: int) -> None:
    """Drop tables for schema migration if needed in an async-friendly manner."""
    storage_dir = Path(hass.config.config_dir) / ".storage"
    await hass.async_add_executor_job(_drop_tables_locked, storage_dir, entry_major)


# ============================================================================
# Single Instance Migration
# ============================================================================


async def async_migrate_to_single_instance(hass: HomeAssistant) -> bool:
    """Migrate multiple config entries to single-instance architecture.

    This function consolidates multiple Area Occupancy config entries into
    a single entry with multiple areas in CONF_AREAS list format.

    Returns:
        bool: True if migration was successful or not needed, False on error
    """
    _LOGGER.info("Checking for single-instance migration...")

    # Find all existing entries for this domain
    entries = [
        entry
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.version < 13  # Only migrate entries before version 13
    ]

    if len(entries) <= 1:
        _LOGGER.debug("No consolidation needed: %d entries found", len(entries))
        return True

    _LOGGER.info(
        "Found %d Area Occupancy entries, consolidating into single-instance architecture",
        len(entries),
    )

    # Store original entry states for potential rollback
    original_entry_states: dict[str, dict[str, Any]] = {}
    for entry in entries:
        original_entry_states[entry.entry_id] = {
            "data": dict(entry.data),
            "options": dict(entry.options),
            "version": entry.version,
            "minor_version": getattr(entry, "minor_version", 0),
            "title": entry.title,
        }

    # Track which steps completed successfully
    migration_steps_completed = {
        "entity_registry": False,
        "device_registry": False,
        "database": False,
        "config_update": False,
        "entry_removal": [],
    }

    try:
        # Step 1: Collect all areas from existing entries
        areas_list: list[dict[str, Any]] = []
        entry_id_to_area_name: dict[str, str] = {}  # Map for entity registry migration

        area_reg = ar.async_get(hass)

        for entry in entries:
            # Resolve area from entry
            area_id, area_name = _resolve_area_from_entry(entry, area_reg)

            if area_id is None or area_name is None:
                # Resolution failed, skip this entry
                continue

            # Ensure unique area names
            original_area_name = area_name
            counter = 1
            while area_name in entry_id_to_area_name.values():
                area_name = f"{original_area_name}_{counter}"
                counter += 1

            entry_id_to_area_name[entry.entry_id] = area_name

            # Create area config (merge data and options, ensure CONF_AREA_ID is set)
            merged = dict(entry.data)
            merged.update(entry.options)
            area_config = {**merged}
            area_config[CONF_AREA_ID] = area_id
            # Remove legacy CONF_NAME if present
            area_config.pop("name", None)
            areas_list.append(area_config)

            _LOGGER.debug(
                "Prepared area '%s' from entry %s for consolidation",
                area_name,
                entry.entry_id,
            )

        # Build filtered list of entries that were successfully resolved
        # Only these entries should be processed in subsequent migration steps
        filtered_entries = [
            entry for entry in entries if entry.entry_id in entry_id_to_area_name
        ]

        if not filtered_entries:
            _LOGGER.error(
                "No entries could be resolved for consolidation. Migration aborted."
            )
            return False

        _LOGGER.info(
            "Successfully resolved %d of %d entries for consolidation",
            len(filtered_entries),
            len(entries),
        )

        # Determine base entry (first entry becomes the consolidated entry)
        base_entry = filtered_entries[0]

        # Step 2: Migrate entity registry entries
        try:
            await _migrate_entity_registry_for_consolidation(
                hass, filtered_entries, entry_id_to_area_name, base_entry.entry_id
            )
            migration_steps_completed["entity_registry"] = True
            _LOGGER.debug("Step 2 completed: Entity registry migration")
        except Exception as err:
            _LOGGER.error("Step 2 failed: Entity registry migration - %s", err)
            raise

        # Step 3: Migrate database entries
        try:
            await _migrate_database_for_consolidation(
                hass, filtered_entries, entry_id_to_area_name
            )
            migration_steps_completed["database"] = True
            _LOGGER.debug("Step 3 completed: Database migration")
        except Exception as err:
            _LOGGER.error("Step 3 failed: Database migration - %s", err)
            # Entity registry migration cannot be easily rolled back, so we continue
            # and log the issue for manual intervention
            _LOGGER.warning(
                "Database migration failed but entity registry was already migrated. "
                "Manual intervention may be required."
            )
            raise

        # Step 4: Create new consolidated config entry
        consolidated_data = {CONF_AREAS: areas_list}

        _LOGGER.info(
            "Creating consolidated config entry with %d areas", len(areas_list)
        )

        try:
            # Update the first entry to the new format
            hass.config_entries.async_update_entry(
                base_entry,
                data=consolidated_data,
                options={},  # Options are now in each area's config
                version=CONF_VERSION,
                minor_version=CONF_VERSION_MINOR,
                title="Area Occupancy Detection",
            )
            migration_steps_completed["config_update"] = True
            _LOGGER.debug("Step 4 completed: Config entry update")
        except Exception as err:
            _LOGGER.error("Step 4 failed: Config entry update - %s", err)
            # Attempt to restore original state
            original_state = original_entry_states.get(base_entry.entry_id)
            if original_state:
                _restore_config_entry_state(hass, base_entry, original_state)
            raise

        # Step 5: Migrate device registry entries
        # Must be done after consolidated config entry is updated, but before removing old entries
        try:
            await _migrate_device_registry_for_consolidation(
                hass, filtered_entries, base_entry.entry_id
            )
            migration_steps_completed["device_registry"] = True
            _LOGGER.debug("Step 5 completed: Device registry migration")
        except (ValueError, KeyError, RuntimeError, HomeAssistantError) as err:
            _LOGGER.error("Step 5 failed: Device registry migration - %s", err)
            # Device registry migration failure is not critical, continue
            _LOGGER.warning(
                "Device registry migration failed, but continuing with migration. "
                "Devices may need manual cleanup."
            )

        # Step 5a: Cleanup orphaned entity registry entries
        # Must be done after device registry migration, but before removing old entries
        # This catches any entities that were created during consolidation or missed by earlier steps
        try:
            await _cleanup_orphaned_entity_registry_entries(
                hass, filtered_entries, base_entry.entry_id
            )
            migration_steps_completed["orphaned_entities"] = True
            _LOGGER.debug("Step 5a completed: Orphaned entity registry cleanup")
        except (ValueError, KeyError, RuntimeError, HomeAssistantError) as err:
            _LOGGER.error("Step 5a failed: Orphaned entity cleanup - %s", err)
            # Orphaned entity cleanup failure is not critical, continue
            _LOGGER.warning(
                "Orphaned entity cleanup failed, but continuing with migration. "
                "Some entities may need manual cleanup."
            )

        # Step 6: Remove other entries (only those that were successfully resolved)
        removal_errors: list[tuple[str, Exception]] = []
        for entry in filtered_entries[1:]:
            try:
                _LOGGER.info("Removing old config entry: %s", entry.entry_id)
                await hass.config_entries.async_remove(entry.entry_id)
                migration_steps_completed["entry_removal"].append(entry.entry_id)
                _LOGGER.debug("Removed entry: %s", entry.entry_id)
            except (ValueError, KeyError, RuntimeError) as err:
                _LOGGER.error("Failed to remove entry %s: %s", entry.entry_id, err)
                removal_errors.append((entry.entry_id, err))
                # Continue removing other entries even if one fails

        if removal_errors:
            _LOGGER.warning(
                "Some entries could not be removed: %s. These may need manual cleanup.",
                [entry_id for entry_id, _ in removal_errors],
            )
            # Partial failure - migration mostly succeeded but some entries remain
            # This is acceptable as the main consolidation is complete

        _LOGGER.info(
            "Successfully consolidated %d entries into single-instance architecture",
            len(filtered_entries),
        )

    except Exception:
        _LOGGER.exception("Error during single-instance migration")
        _LOGGER.error(
            "Migration failed. Completed steps: %s. "
            "Original entry states have been logged for recovery.",
            migration_steps_completed,
        )
        _LOGGER.error(
            "Original entry states for recovery: %s",
            {
                entry_id: {
                    "data": state["data"],
                    "version": state["version"],
                }
                for entry_id, state in original_entry_states.items()
            },
        )
        return False
    else:
        return True


# Single Instance Migration Helper Functions
# ==========================================


def _restore_config_entry_state(
    hass: HomeAssistant, entry: ConfigEntry, original_state: dict[str, Any]
) -> bool:
    """Restore config entry to its original state.

    Args:
        hass: Home Assistant instance
        entry: The config entry to restore
        original_state: Dictionary containing original state data

    Returns:
        True if restoration succeeded, False otherwise
    """
    try:
        hass.config_entries.async_update_entry(
            entry,
            data=original_state["data"],
            options=original_state["options"],
            version=original_state["version"],
            minor_version=original_state.get("minor_version", 0),
            title=original_state.get("title", entry.title),
        )
        _LOGGER.info("Restored original config entry state for %s", entry.entry_id)
    except (ValueError, KeyError, AttributeError, RuntimeError) as restore_err:
        _LOGGER.error("Failed to restore config entry state: %s", restore_err)
        return False
    else:
        return True


def _handle_entity_conflict(
    entity_registry: er.EntityRegistry,
    entity_id: str,
    old_unique_id: str,
    new_unique_id: str,
    conflicts: list[tuple[str, str, str]],
) -> bool:
    """Handle entity unique ID conflict during migration.

    Args:
        entity_registry: The entity registry
        entity_id: The entity ID being migrated
        old_unique_id: The old unique ID
        new_unique_id: The new unique ID
        conflicts: List to append conflict information to

    Returns:
        True if conflict was handled (entity skipped), False if no conflict
    """
    has_conflict, conflict_entity_id = _check_unique_id_conflict(
        entity_registry, new_unique_id, entity_id
    )

    if has_conflict:
        conflicts.append((entity_id, old_unique_id, new_unique_id))
        _LOGGER.warning(
            "Unique ID conflict: %s already exists for entity %s. "
            "Skipping migration for %s (old unique_id: %s)",
            new_unique_id,
            conflict_entity_id,
            entity_id,
            old_unique_id,
        )
        return True
    return False


def _resolve_area_from_entry(
    entry: ConfigEntry, area_reg: ar.AreaRegistry
) -> tuple[str | None, str | None]:
    """Resolve area ID and name from a config entry.

    Handles both new format (CONF_AREA_ID) and legacy format (name).

    Args:
        entry: The config entry to resolve area from
        area_reg: The area registry to query

    Returns:
        Tuple of (area_id, area_name) or (None, None) if resolution fails
    """
    merged = dict(entry.data)
    merged.update(entry.options)

    # Try to get area_id first (new format), then fall back to name (legacy)
    area_id = merged.get(CONF_AREA_ID)
    area_name = None

    if area_id:
        # New format - resolve area name from ID
        area_entry = area_reg.async_get_area(area_id)
        if area_entry:
            area_name = area_entry.name
            return area_id, area_name
        _LOGGER.warning(
            "Area ID '%s' for entry %s not found in registry. Skipping.",
            area_id,
            entry.entry_id,
        )
        return None, None
    # Legacy format - try to resolve name to ID
    legacy_name = merged.get("name", DEFAULT_NAME)

    # Try to find area by name
    for area_entry in area_reg.async_list_areas():
        if area_entry.name == legacy_name:
            area_id = area_entry.id
            area_name = legacy_name
            return area_id, area_name

    # Could not resolve
    _LOGGER.warning(
        "Could not resolve area name '%s' for entry %s to area ID. Skipping.",
        legacy_name,
        entry.entry_id,
    )
    return None, None


async def _migrate_entity_registry_for_consolidation(
    hass: HomeAssistant,
    entries: list[ConfigEntry],
    entry_id_to_area_name: dict[str, str],
    new_entry_id: str,
) -> None:
    """Migrate entity registry entries for consolidation.

    Updates unique IDs from {entry_id}_{entity_type} or {DOMAIN}_{entry_id}_{entity_type}
    to {area_name}_{entity_type}. Also updates config_entry_id to point to the new
    consolidated config entry. Handles both new format (after unique ID migration)
    and legacy format (in case unique ID migration didn't run).

    Args:
        hass: Home Assistant instance
        entries: List of old config entries being consolidated
        entry_id_to_area_name: Mapping of old entry IDs to area names
        new_entry_id: The entry ID of the new consolidated config entry
    """
    _LOGGER.info("Migrating entity registry entries for consolidation")
    entity_registry = er.async_get(hass)
    updated_count = 0
    skipped_count = 0
    conflicts: list[
        tuple[str, str, str]
    ] = []  # (entity_id, old_unique_id, new_unique_id)

    for entry in entries:
        # Skip entries that weren't successfully resolved
        if entry.entry_id not in entry_id_to_area_name:
            _LOGGER.debug(
                "Skipping entry %s - not in entry_id_to_area_name mapping",
                entry.entry_id,
            )
            continue

        area_name = entry_id_to_area_name[entry.entry_id]
        # Handle both new format (f"{entry.entry_id}_") and legacy format (f"{DOMAIN}_{entry.entry_id}_")
        new_prefix = f"{entry.entry_id}_"
        legacy_prefix = f"{DOMAIN}_{entry.entry_id}_"

        # Find entities matching either prefix format
        new_format_entities = _find_entities_by_prefix(
            entity_registry, new_prefix, entry.entry_id
        )
        legacy_format_entities = _find_entities_by_prefix(
            entity_registry, legacy_prefix, entry.entry_id
        )

        # Process entities found with new format
        for entity_id, entity_entry in new_format_entities:
            old_unique_id_str = str(entity_entry.unique_id)
            entity_suffix = old_unique_id_str[len(new_prefix) :]
            new_unique_id = f"{area_name}_{entity_suffix}".lower()

            # Check for conflicts before updating
            if _handle_entity_conflict(
                entity_registry, entity_id, old_unique_id_str, new_unique_id, conflicts
            ):
                skipped_count += 1
                continue

            # Update both unique_id and config_entry_id in a single call
            # This ensures entities are linked to the correct config entry
            old_config_entry_id = entity_entry.config_entry_id
            entity_registry.async_update_entity(
                entity_id,
                new_unique_id=new_unique_id,
                config_entry_id=new_entry_id,
            )
            _LOGGER.info(
                "Migrating entity unique_id: %s -> %s, config_entry_id: %s -> %s (entity: %s)",
                old_unique_id_str,
                new_unique_id,
                old_config_entry_id,
                new_entry_id,
                entity_id,
            )
            updated_count += 1

        # Process entities found with legacy format
        for entity_id, entity_entry in legacy_format_entities:
            # Skip if already processed as new format
            if any(eid == entity_id for eid, _ in new_format_entities):
                continue

            old_unique_id_str = str(entity_entry.unique_id)
            _LOGGER.debug(
                "Found entity with legacy unique ID format: %s (entry: %s). "
                "This should have been migrated earlier, but handling it now.",
                old_unique_id_str,
                entry.entry_id,
            )
            entity_suffix = old_unique_id_str[len(legacy_prefix) :]
            new_unique_id = f"{area_name}_{entity_suffix}".lower()

            # Check for conflicts before updating
            if _handle_entity_conflict(
                entity_registry, entity_id, old_unique_id_str, new_unique_id, conflicts
            ):
                skipped_count += 1
                continue

            # Update both unique_id and config_entry_id in a single call
            # This ensures entities are linked to the correct config entry
            old_config_entry_id = entity_entry.config_entry_id
            entity_registry.async_update_entity(
                entity_id,
                new_unique_id=new_unique_id,
                config_entry_id=new_entry_id,
            )
            _LOGGER.info(
                "Migrating entity unique_id: %s -> %s, config_entry_id: %s -> %s (entity: %s)",
                old_unique_id_str,
                new_unique_id,
                old_config_entry_id,
                new_entry_id,
                entity_id,
            )
            updated_count += 1

    _LOGGER.info(
        "Migrated %d entity registry entries, skipped %d due to conflicts",
        updated_count,
        skipped_count,
    )
    if conflicts:
        _LOGGER.warning(
            "The following entities could not be migrated due to unique ID conflicts: %s",
            conflicts,
        )


async def _migrate_device_registry_for_consolidation(
    hass: HomeAssistant,
    entries: list[ConfigEntry],
    new_entry_id: str,
) -> None:
    """Migrate device registry entries for consolidation.

    Updates device registry entries to link to the new consolidated config entry
    instead of the old entries. This ensures that when entities are created after
    consolidation, devices are properly linked and don't cause errors.

    Args:
        hass: Home Assistant instance
        entries: List of old config entries that are being consolidated
        new_entry_id: The entry ID of the new consolidated config entry
    """
    _LOGGER.info("Migrating device registry entries for consolidation")
    device_registry = dr.async_get(hass)
    old_entry_ids = {entry.entry_id for entry in entries}
    updated_count = 0
    skipped_count = 0

    # Find all devices linked to old entry IDs for this domain
    # Devices are identified by (DOMAIN, area_id or area_name)
    for device_id, device_entry in device_registry.devices.items():
        # Check if device belongs to this domain by checking identifiers
        # Devices for this integration use identifiers like (DOMAIN, area_id)
        device_identifiers = device_entry.identifiers
        is_domain_device = any(
            identifier[0] == DOMAIN for identifier in device_identifiers
        )

        if not is_domain_device:
            # Device doesn't belong to this domain, skip it
            continue

        # Check if device is linked to any of the old entry IDs
        device_config_entries = device_entry.config_entries
        old_entry_ids_in_device = device_config_entries & old_entry_ids

        if not old_entry_ids_in_device:
            # Device is not linked to any old entries, skip it
            continue

        # Check if device is already linked to the new entry
        if new_entry_id in device_config_entries:
            # Device is already linked to new entry, just remove old entry links
            for old_entry_id in old_entry_ids_in_device:
                try:
                    device_registry.async_update_device(
                        device_id, remove_config_entry_id=old_entry_id
                    )
                    _LOGGER.debug(
                        "Removed old entry %s from device %s (already linked to new entry)",
                        old_entry_id,
                        device_id,
                    )
                except (ValueError, KeyError, RuntimeError, HomeAssistantError) as err:
                    _LOGGER.warning(
                        "Failed to remove old entry %s from device %s: %s",
                        old_entry_id,
                        device_id,
                        err,
                    )
            updated_count += 1
        else:
            # Device is only linked to old entries, need to migrate it
            try:
                # First, add the new entry link (while old entries still exist)
                device_registry.async_update_device(
                    device_id, add_config_entry_id=new_entry_id
                )
                _LOGGER.debug(
                    "Added new entry %s to device %s", new_entry_id, device_id
                )
                # Then, remove old entry links
                for old_entry_id in old_entry_ids_in_device:
                    try:
                        device_registry.async_update_device(
                            device_id, remove_config_entry_id=old_entry_id
                        )
                        _LOGGER.debug(
                            "Removed old entry %s from device %s",
                            old_entry_id,
                            device_id,
                        )
                    except (
                        ValueError,
                        KeyError,
                        RuntimeError,
                        HomeAssistantError,
                    ) as err:
                        _LOGGER.warning(
                            "Failed to remove old entry %s from device %s: %s",
                            old_entry_id,
                            device_id,
                            err,
                        )
                _LOGGER.info(
                    "Migrated device %s from entries %s to entry %s",
                    device_id,
                    old_entry_ids_in_device,
                    new_entry_id,
                )
                updated_count += 1
            except (ValueError, KeyError, RuntimeError, HomeAssistantError) as err:
                _LOGGER.warning(
                    "Failed to migrate device %s: %s. Device may need manual cleanup.",
                    device_id,
                    err,
                )
                skipped_count += 1

    _LOGGER.info(
        "Migrated %d device registry entries for domain %s, skipped %d due to errors",
        updated_count,
        DOMAIN,
        skipped_count,
    )


async def _cleanup_orphaned_entity_registry_entries(
    hass: HomeAssistant,
    entries: list[ConfigEntry],
    new_entry_id: str,
) -> None:
    """Clean up orphaned entity registry entries after consolidation.

    Finds all entity registry entries for this domain that reference removed config
    entry IDs and updates them to reference the consolidated entry ID. This ensures
    that entities created during or after consolidation are properly linked.

    Args:
        hass: Home Assistant instance
        entries: List of old config entries that are being consolidated
        new_entry_id: The entry ID of the new consolidated config entry
    """
    _LOGGER.info("Cleaning up orphaned entity registry entries for consolidation")
    entity_registry = er.async_get(hass)
    old_entry_ids = {entry.entry_id for entry in entries}
    updated_count = 0
    skipped_count = 0
    removed_count = 0

    # Find all entity registry entries for this domain that reference old entry IDs
    # Check entities from our platforms: binary_sensor, sensor, number
    for entity_id, entity_entry in entity_registry.entities.items():
        # Only process entries for this domain (check entity_id domain part)
        entity_domain = entity_id.split(".", 1)[0] if "." in entity_id else None
        if entity_domain not in ["binary_sensor", "sensor", "number"]:
            # Only process entities from our platforms
            continue

        # Check if entity references an old entry ID that will be removed
        if entity_entry.config_entry_id not in old_entry_ids:
            # Entity doesn't reference an old entry, skip it
            continue

        # Entity references an old entry ID - update it to the new consolidated entry
        old_config_entry_id = entity_entry.config_entry_id

        # Check if there would be a unique ID conflict
        # (entity with same unique_id already exists for new entry)
        # Note: We check against new_entry_id to see if an entity with the same
        # unique_id already exists linked to the consolidated entry
        has_conflict, conflict_entity_id = _check_unique_id_conflict(
            entity_registry,
            entity_entry.unique_id,
            entity_id,  # exclude_entity_id - the entity we're checking
        )

        if has_conflict:
            _LOGGER.warning(
                "Unique ID conflict: %s already exists for entity %s (new entry %s). "
                "Removing orphaned entity %s (old entry: %s)",
                entity_entry.unique_id,
                conflict_entity_id,
                new_entry_id,
                entity_id,
                old_config_entry_id,
            )
            try:
                entity_registry.async_remove(entity_id)
                removed_count += 1
            except (ValueError, KeyError, RuntimeError, HomeAssistantError) as err:
                _LOGGER.warning(
                    "Failed to remove orphaned entity %s: %s", entity_id, err
                )
                skipped_count += 1
            continue

        # Update entity to reference the new consolidated entry
        try:
            entity_registry.async_update_entity(
                entity_id,
                config_entry_id=new_entry_id,
            )
            _LOGGER.info(
                "Updated orphaned entity %s: config_entry_id %s -> %s (unique_id: %s)",
                entity_id,
                old_config_entry_id,
                new_entry_id,
                entity_entry.unique_id,
            )
            updated_count += 1
        except (ValueError, KeyError, RuntimeError, HomeAssistantError) as err:
            _LOGGER.warning(
                "Failed to update orphaned entity %s: %s. Entity may need manual cleanup.",
                entity_id,
                err,
            )
            skipped_count += 1

    _LOGGER.info(
        "Cleaned up orphaned entity registry entries: %d updated, %d removed, %d skipped (errors) for domain %s",
        updated_count,
        removed_count,
        skipped_count,
        DOMAIN,
    )


def _check_database_tables_exist(
    inspector: sa.Inspector, required_tables: list[str]
) -> bool:
    """Check if all required database tables exist.

    Args:
        inspector: SQLAlchemy inspector
        required_tables: List of table names to check

    Returns:
        True if all tables exist, False otherwise
    """
    for table_name in required_tables:
        if not inspector.has_table(table_name):
            _LOGGER.debug(
                "%s table does not exist, skipping database migration", table_name
            )
            return False
    return True


async def _migrate_database_for_consolidation(
    hass: HomeAssistant,
    entries: list[ConfigEntry],
    entry_id_to_area_name: dict[str, str],
    engine_factory: Any | None = None,
    session_factory: Any | None = None,
) -> None:
    """Migrate database entries for consolidation.

    Updates database to use area_name instead of entry_id as keys.

    Args:
        hass: Home Assistant instance
        entries: List of config entries being consolidated
        entry_id_to_area_name: Mapping of entry IDs to area names
        engine_factory: Optional factory function to create engine (for testing)
        session_factory: Optional factory function to create session (for testing)
    """
    _LOGGER.info("Migrating database entries for consolidation")

    # Get storage directory
    storage_dir = Path(hass.config.config_dir) / ".storage"
    db_path = storage_dir / DB_NAME

    if not db_path.exists():
        _LOGGER.debug("Database file not found, skipping database migration")
        return

    # Use file lock for safe migration
    lock_file = storage_dir / f"{DB_NAME}.lock"

    try:
        with FileLock(lock_file, timeout=30):
            # Open database connection
            if engine_factory is not None:
                engine = engine_factory()
            else:
                engine = create_engine(
                    f"sqlite:///{db_path}",
                    echo=False,
                    connect_args={"check_same_thread": False, "timeout": 30},
                )

            if session_factory is not None:
                session_maker = session_factory
            else:
                session_maker = sessionmaker(bind=engine)

            with session_maker() as session:
                # Update areas table: change entry_id to area_name
                try:
                    # Check if tables exist before attempting operations
                    inspector = sa.inspect(engine)
                    if not _check_database_tables_exist(
                        inspector, ["entities", "areas"]
                    ):
                        return

                    # Get all areas with old entry_id
                    result = session.execute(
                        text("SELECT entry_id, area_name FROM areas")
                    ).fetchall()

                    for row in result:
                        old_entry_id = row[0]
                        if old_entry_id in entry_id_to_area_name:
                            new_area_name = entry_id_to_area_name[old_entry_id]
                            # Update the area_name field (it might already be set correctly)
                            session.execute(
                                text(
                                    "UPDATE areas SET area_name = :area_name WHERE entry_id = :entry_id"
                                ),
                                {"area_name": new_area_name, "entry_id": old_entry_id},
                            )
                            _LOGGER.debug(
                                "Updated area in database: entry_id=%s -> area_name=%s",
                                old_entry_id,
                                new_area_name,
                            )
                except (SQLAlchemyError, OSError) as e:
                    _LOGGER.warning("Error updating areas table: %s", e)

                # Add area_name column to entities table if it doesn't exist
                try:
                    # Check if area_name column exists
                    inspector = sa.inspect(engine)
                    columns = [col["name"] for col in inspector.get_columns("entities")]

                    if "area_name" not in columns:
                        _LOGGER.info("Adding area_name column to entities table")
                        # Add the column (nullable initially so existing rows don't fail)
                        session.execute(
                            text("ALTER TABLE entities ADD COLUMN area_name TEXT")
                        )

                        # Populate area_name for existing entities based on entry_id -> area_name mapping
                        for (
                            old_entry_id,
                            new_area_name,
                        ) in entry_id_to_area_name.items():
                            session.execute(
                                text(
                                    "UPDATE entities SET area_name = :area_name WHERE entry_id = :entry_id"
                                ),
                                {"area_name": new_area_name, "entry_id": old_entry_id},
                            )
                            _LOGGER.debug(
                                "Updated entities for entry_id %s with area_name %s",
                                old_entry_id,
                                new_area_name,
                            )

                        # Make the column NOT NULL now that it's populated
                        # SQLite doesn't support MODIFY COLUMN directly, so we'll leave it nullable
                        # for now. New inserts will always have area_name.
                        _LOGGER.info(
                            "Added and populated area_name column in entities table"
                        )
                except (SQLAlchemyError, OSError) as e:
                    _LOGGER.warning("Error adding area_name column to entities: %s", e)

                # Clean up master-related metadata
                try:
                    session.execute(
                        text(
                            "DELETE FROM metadata WHERE key IN ('master_entry_id', 'master_heartbeat')"
                        )
                    )
                    _LOGGER.debug("Cleaned up master-related metadata")
                except (SQLAlchemyError, OSError) as e:
                    _LOGGER.debug("No master metadata to clean up: %s", e)

                session.commit()

        _LOGGER.info("Database migration completed")

    except Timeout:
        _LOGGER.error("Timeout acquiring database lock for migration")
    except Exception:
        _LOGGER.exception("Error during database migration")


# ============================================================================
# Entry Migration (Main Entry Point)
# ============================================================================


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to the new version."""
    current_major = CONF_VERSION
    current_minor = CONF_VERSION_MINOR
    entry_major = config_entry.version
    entry_minor = getattr(
        config_entry, "minor_version", 0
    )  # Use 0 if minor_version doesn't exist

    if entry_major > current_major or (
        entry_major == current_major and entry_minor >= current_minor
    ):
        # Stored version is same or newer, no migration needed
        _LOGGER.debug(
            "Skipping migration for %s: Stored version (%s.%s) >= Current version (%s.%s)",
            config_entry.entry_id,
            entry_major,
            entry_minor,
            current_major,
            current_minor,
        )
        return True  # Indicate successful (skipped) migration

    _LOGGER.info(
        "Migrating Area Occupancy entry %s from version %s.%s to %s.%s",
        config_entry.entry_id,
        entry_major,
        entry_minor,
        current_major,
        current_minor,
    )

    # Check if we need to consolidate multiple entries (for version < 13)
    if entry_major < 13:
        _LOGGER.info("Checking for multiple entries that need consolidation...")
        # Find all entries that would be consolidated
        entries_to_consolidate = [
            entry
            for entry in hass.config_entries.async_entries(DOMAIN)
            if entry.version < 13
        ]

        # Migrate unique IDs for all entries that will be consolidated BEFORE consolidation
        # This ensures consolidation migration can find entities with the expected prefix format
        if len(entries_to_consolidate) > 1:
            _LOGGER.info(
                "Migrating unique IDs for %d entries before consolidation",
                len(entries_to_consolidate),
            )
            for entry_to_migrate in entries_to_consolidate:
                try:
                    _LOGGER.debug(
                        "Migrating unique IDs for entry %s before consolidation",
                        entry_to_migrate.entry_id,
                    )
                    for platform in PLATFORMS:
                        await async_migrate_unique_ids(hass, entry_to_migrate, platform)
                except HomeAssistantError as err:
                    _LOGGER.warning(
                        "Error migrating unique IDs for entry %s before consolidation: %s",
                        entry_to_migrate.entry_id,
                        err,
                    )

        consolidation_result = await async_migrate_to_single_instance(hass)
        if not consolidation_result:
            _LOGGER.error("Single-instance consolidation failed")
            # Continue with per-entry migration anyway
        # If consolidation happened, the entry might have been removed
        # Check if this entry still exists
        if config_entry.entry_id not in [
            e.entry_id for e in hass.config_entries.async_entries(DOMAIN)
        ]:
            _LOGGER.info("Entry was consolidated, skipping individual migration")
            return True

    # --- Run Storage File Migration First ---
    _LOGGER.debug("Starting storage migration for %s", config_entry.entry_id)
    await async_migrate_storage(hass, config_entry.entry_id, entry_major)
    _LOGGER.debug("Storage migration completed for %s", config_entry.entry_id)
    # --------------------------------------

    # Get existing data
    _LOGGER.debug("Getting existing config data for %s", config_entry.entry_id)
    data = {**config_entry.data}
    options = {**config_entry.options}

    try:
        # Run the unique ID migrations (for entries that weren't consolidated)
        _LOGGER.debug("Starting unique ID migrations for %s", config_entry.entry_id)
        for platform in PLATFORMS:
            _LOGGER.debug("Migrating unique IDs for platform %s", platform)
            await async_migrate_unique_ids(hass, config_entry, platform)
        _LOGGER.debug("Unique ID migrations completed for %s", config_entry.entry_id)
    except HomeAssistantError as err:
        _LOGGER.error("Error during unique ID migration: %s", err)

    # Remove deprecated fields
    _LOGGER.debug("Removing deprecated fields for %s", config_entry.entry_id)
    if CONF_AREA_ID in data:
        data.pop(CONF_AREA_ID)
        _LOGGER.debug("Removed deprecated CONF_AREA_ID")

    if DECAY_MIN_DELAY_KEY in data:
        data.pop(DECAY_MIN_DELAY_KEY)
        _LOGGER.debug("Removed deprecated decay_min_delay from data")
    if DECAY_MIN_DELAY_KEY in options:
        options.pop(DECAY_MIN_DELAY_KEY)
        _LOGGER.debug("Removed deprecated decay_min_delay from options")

    # Ensure new state configuration values are present with defaults
    _LOGGER.debug("Adding new state configurations for %s", config_entry.entry_id)
    new_configs = {
        CONF_DOOR_ACTIVE_STATE: DEFAULT_DOOR_ACTIVE_STATE,
        CONF_WINDOW_ACTIVE_STATE: DEFAULT_WINDOW_ACTIVE_STATE,
        CONF_MEDIA_ACTIVE_STATES: DEFAULT_MEDIA_ACTIVE_STATES,
        CONF_APPLIANCE_ACTIVE_STATES: DEFAULT_APPLIANCE_ACTIVE_STATES,
    }

    # Update data with new state configurations if not present
    for key, default_value in new_configs.items():
        if key not in data and key not in options:
            _LOGGER.info("Adding new configuration %s with default value", key)
            # For multi-select states, add to data
            if isinstance(default_value, list):
                data[key] = default_value
            # For single-select states, add to options
            else:
                options[key] = default_value

    try:
        # Apply configuration migrations
        _LOGGER.debug("Applying configuration migrations for %s", config_entry.entry_id)
        data = migrate_config(data)
        options = migrate_config(options)

        # Handle threshold value with default if not present
        threshold = options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        options[CONF_THRESHOLD] = validate_threshold(threshold)

        # Update the config entry with new data and options
        _LOGGER.debug("Updating config entry for %s", config_entry.entry_id)
        hass.config_entries.async_update_entry(
            config_entry,
            data=data,
            options=options,
            version=CONF_VERSION,
            minor_version=CONF_VERSION_MINOR,
        )
        _LOGGER.info("Successfully migrated config entry %s", config_entry.entry_id)
    except (ValueError, KeyError, HomeAssistantError) as err:
        _LOGGER.error("Error during config migration: %s", err)
        return False
    else:
        return True


# ============================================================================
# Validation Functions
# ============================================================================


def validate_threshold(threshold: float) -> float:
    """Validate the threshold value.

    Args:
        threshold: The threshold value to validate

    Returns:
        The validated threshold value

    """
    if threshold < 1.0 or threshold > 99.0:
        return DEFAULT_THRESHOLD
    return round(threshold, 0)
