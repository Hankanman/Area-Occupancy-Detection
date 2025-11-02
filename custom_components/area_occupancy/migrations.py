"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

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
from homeassistant.helpers import entity_registry as er

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
    CONF_NAME,
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
    validate_and_sanitize_area_name,
)
from .db import DB_NAME, DB_VERSION
from .number import NAME_THRESHOLD_NUMBER
from .sensor import NAME_DECAY_SENSOR, NAME_PRIORS_SENSOR, NAME_PROBABILITY_SENSOR

_LOGGER = logging.getLogger(__name__)


async def async_migrate_unique_ids(
    hass: HomeAssistant, config_entry: ConfigEntry, platform: str
) -> None:
    """Migrate unique IDs of entities in the entity registry."""
    _LOGGER.debug("Starting unique ID migration for platform %s", platform)
    entity_registry = er.async_get(hass)
    updated_entries = 0
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

    for entity_id, entity_entry in entity_registry.entities.items():
        old_unique_id = entity_entry.unique_id
        # Convert to string to avoid AttributeError
        if old_unique_id is not None and str(old_unique_id).startswith(old_prefix):
            # Simply remove the domain prefix to get the new ID
            new_unique_id = str(old_unique_id).replace(old_prefix, f"{entry_id}_")

            # Update the unique ID in the registry
            _LOGGER.info(
                "Migrating unique ID for %s: %s -> %s",
                entity_id,
                old_unique_id,
                new_unique_id,
            )
            entity_registry.async_update_entity(entity_id, new_unique_id=new_unique_id)
            updated_entries += 1

    if updated_entries > 0:
        _LOGGER.info(
            "Completed migrating %d unique IDs for platform %s",
            updated_entries,
            platform,
        )
    else:
        _LOGGER.debug("No unique IDs to migrate for platform %s", platform)


DECAY_MIN_DELAY_KEY = "decay_min_delay"


def remove_decay_min_delay(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated decay delay option from config."""
    if DECAY_MIN_DELAY_KEY in config:
        config.pop(DECAY_MIN_DELAY_KEY)
        _LOGGER.debug("Removed deprecated decay_min_delay from config")
    return config


CONF_LIGHTS_KEY = "lights"


def remove_lights_key(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated lights key from config."""
    if CONF_LIGHTS_KEY in config:
        config.pop(CONF_LIGHTS_KEY)
        _LOGGER.debug("Removed deprecated lights key from config")
    return config


CONF_DECAY_WINDOW_KEY = "decay_window"


def remove_decay_window_key(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated decay window key from config."""
    if CONF_DECAY_WINDOW_KEY in config:
        config.pop(CONF_DECAY_WINDOW_KEY)
        _LOGGER.debug("Removed deprecated decay window key from config")
    return config


CONF_HISTORICAL_ANALYSIS_ENABLED = "historical_analysis_enabled"
CONF_HISTORY_PERIOD = "history_period"


def remove_history_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Remove deprecated history period key from config."""
    if CONF_HISTORY_PERIOD in config:
        config.pop(CONF_HISTORY_PERIOD)
        _LOGGER.debug("Removed deprecated history period key from config")
    if CONF_HISTORICAL_ANALYSIS_ENABLED in config:
        config.pop(CONF_HISTORICAL_ANALYSIS_ENABLED)
        _LOGGER.debug("Removed deprecated historical analysis enabled key from config")
    return config


def migrate_decay_half_life(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add decay half life."""
    if CONF_DECAY_HALF_LIFE not in config:
        config[CONF_DECAY_HALF_LIFE] = DEFAULT_DECAY_HALF_LIFE
        _LOGGER.debug("Added decay half life to config")

    return config


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

    if CONF_PURPOSE not in config:
        config[CONF_PURPOSE] = DEFAULT_PURPOSE
        _LOGGER.debug("Migrated purpose field to default value: %s", DEFAULT_PURPOSE)

    return config


def migrate_motion_timeout(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate configuration to add motion timeout."""
    if CONF_MOTION_TIMEOUT not in config:
        config[CONF_MOTION_TIMEOUT] = DEFAULT_MOTION_TIMEOUT
        _LOGGER.debug("Added motion timeout to config: %s", DEFAULT_MOTION_TIMEOUT)

    return config


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
            try:
                legacy_file.unlink()
                _LOGGER.info("Successfully removed legacy storage file")
            except OSError as err:
                _LOGGER.warning(
                    "Error removing legacy storage file %s: %s", legacy_file, err
                )

        # Reset database for version < 11
        await async_reset_database_if_needed(hass, entry_major)

        _LOGGER.debug("Storage migration completed for entry %s", entry_id)
    except (HomeAssistantError, OSError, ValueError) as err:
        _LOGGER.error("Error during storage migration for entry %s: %s", entry_id, err)


def _drop_tables_locked(storage_dir: Path, entry_major: int) -> None:
    """Blocking helper: perform locked drop of legacy tables if needed."""
    if entry_major >= 11:
        _LOGGER.debug("Skipping table dropping for version %s", entry_major)
        return

    _LOGGER.info("Dropping tables for schema migration")
    db_path = storage_dir / DB_NAME
    if not db_path.exists():
        return

    lock_path = storage_dir / (DB_NAME + ".lock")
    try:
        with FileLock(lock_path):
            engine = create_engine(f"sqlite:///{db_path}")
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

            if db_version < 3:
                _LOGGER.info("Dropping tables for schema migration")
                try:
                    session.execute(
                        text(
                            "UPDATE metadata SET value = :version WHERE key = 'db_version'"
                        ),
                        {"version": str(DB_VERSION)},
                    )
                    if session.execute(text("SELECT changes()")).scalar() == 0:
                        session.execute(
                            text(
                                "INSERT INTO metadata (key, value) VALUES ('db_version', :version)"
                            ),
                            {"version": str(DB_VERSION)},
                        )
                except Exception:  # noqa: BLE001
                    pass
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
                    _LOGGER.info("All tables dropped successfully")

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
        "database": False,
        "config_update": False,
        "entry_removal": [],
    }

    try:
        # Step 1: Collect all areas from existing entries
        areas_list: list[dict[str, Any]] = []
        entry_id_to_area_name: dict[str, str] = {}  # Map for entity registry migration

        for entry in entries:
            # Get area name from config
            merged = dict(entry.data)
            merged.update(entry.options)
            area_name = merged.get(CONF_NAME, DEFAULT_NAME)

            # Validate and sanitize area name
            try:
                area_name = validate_and_sanitize_area_name(area_name)
            except ValueError as err:
                _LOGGER.warning(
                    "Invalid area name '%s' for entry %s: %s. Using default name.",
                    area_name,
                    entry.entry_id,
                    err,
                )
                area_name = DEFAULT_NAME

            # Ensure unique area names
            original_area_name = area_name
            counter = 1
            while area_name in entry_id_to_area_name.values():
                area_name = f"{original_area_name}_{counter}"
                counter += 1

            entry_id_to_area_name[entry.entry_id] = area_name

            # Create area config (merge data and options, ensure CONF_NAME is set)
            area_config = {**merged}
            area_config[CONF_NAME] = area_name
            areas_list.append(area_config)

            _LOGGER.debug(
                "Prepared area '%s' from entry %s for consolidation",
                area_name,
                entry.entry_id,
            )

        # Step 2: Migrate entity registry entries
        try:
            await _migrate_entity_registry_for_consolidation(
                hass, entries, entry_id_to_area_name
            )
            migration_steps_completed["entity_registry"] = True
            _LOGGER.debug("Step 2 completed: Entity registry migration")
        except Exception as err:
            _LOGGER.error("Step 2 failed: Entity registry migration - %s", err)
            raise

        # Step 3: Migrate database entries
        try:
            await _migrate_database_for_consolidation(
                hass, entries, entry_id_to_area_name
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

        # Use the first entry as the base for the consolidated entry
        base_entry = entries[0]

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
            try:
                original_state = original_entry_states[base_entry.entry_id]
                hass.config_entries.async_update_entry(
                    base_entry,
                    data=original_state["data"],
                    options=original_state["options"],
                    version=original_state["version"],
                    minor_version=original_state["minor_version"],
                    title=original_state["title"],
                )
                _LOGGER.info(
                    "Restored original config entry state for %s", base_entry.entry_id
                )
            except (ValueError, KeyError, AttributeError, RuntimeError) as restore_err:
                _LOGGER.error("Failed to restore config entry state: %s", restore_err)
            raise

        # Step 5: Remove other entries
        removal_errors: list[tuple[str, Exception]] = []
        for entry in entries[1:]:
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
            len(entries),
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


async def _migrate_area_name_in_entity_registry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    old_area_name: str,
    new_area_name: str,
) -> None:
    """Migrate entity registry unique IDs when an area name changes.

    Updates unique IDs from {old_area_name}_{entity_type} to {new_area_name}_{entity_type}

    Args:
        hass: Home Assistant instance
        config_entry: The config entry
        old_area_name: The old area name
        new_area_name: The new area name
    """
    if old_area_name == new_area_name:
        _LOGGER.debug("Area name unchanged, skipping entity registry migration")
        return

    _LOGGER.info(
        "Migrating entity registry unique IDs for area rename: %s -> %s",
        old_area_name,
        new_area_name,
    )

    entity_registry = er.async_get(hass)
    updated_count = 0
    skipped_count = 0
    conflicts: list[tuple[str, str, str]] = []

    old_prefix = f"{old_area_name}_"
    new_prefix = f"{new_area_name}_"

    # Find all entities for this config entry
    for entity_id, entity_entry in entity_registry.entities.items():
        if entity_entry.config_entry_id != config_entry.entry_id:
            continue

        old_unique_id = entity_entry.unique_id
        if not old_unique_id or not str(old_unique_id).startswith(old_prefix):
            continue

        # Extract entity type suffix
        entity_suffix = str(old_unique_id)[len(old_prefix) :]
        new_unique_id = f"{new_prefix}{entity_suffix}"

        # Check for conflicts
        conflict_found = False
        for other_entity_id, other_entity_entry in entity_registry.entities.items():
            if (
                other_entity_id != entity_id
                and other_entity_entry.unique_id == new_unique_id
            ):
                conflict_found = True
                conflicts.append((entity_id, str(old_unique_id), new_unique_id))
                _LOGGER.warning(
                    "Unique ID conflict: %s already exists for entity %s. "
                    "Skipping migration for %s",
                    new_unique_id,
                    other_entity_id,
                    entity_id,
                )
                skipped_count += 1
                break

        if conflict_found:
            continue

        _LOGGER.info(
            "Migrating entity unique_id: %s -> %s (entity: %s)",
            old_unique_id,
            new_unique_id,
            entity_id,
        )

        entity_registry.async_update_entity(
            entity_id,
            new_unique_id=new_unique_id,
        )
        updated_count += 1

    _LOGGER.info(
        "Migrated %d entity registry entries for area rename, skipped %d due to conflicts",
        updated_count,
        skipped_count,
    )
    if conflicts:
        _LOGGER.warning(
            "The following entities could not be migrated due to unique ID conflicts: %s",
            conflicts,
        )


async def _migrate_entity_registry_for_consolidation(
    hass: HomeAssistant,
    entries: list[ConfigEntry],
    entry_id_to_area_name: dict[str, str],
) -> None:
    """Migrate entity registry entries for consolidation.

    Updates unique IDs from {entry_id}_{entity_type} to {area_name}_{entity_type}
    """
    _LOGGER.info("Migrating entity registry entries for consolidation")
    entity_registry = er.async_get(hass)
    updated_count = 0
    skipped_count = 0
    conflicts: list[
        tuple[str, str, str]
    ] = []  # (entity_id, old_unique_id, new_unique_id)

    for entry in entries:
        area_name = entry_id_to_area_name[entry.entry_id]
        old_prefix = f"{entry.entry_id}_"

        # Find all entities for this entry
        for entity_id, entity_entry in entity_registry.entities.items():
            if entity_entry.config_entry_id != entry.entry_id:
                continue

            old_unique_id = entity_entry.unique_id
            if not old_unique_id or not str(old_unique_id).startswith(old_prefix):
                continue

            # Extract entity type suffix
            entity_suffix = str(old_unique_id)[len(old_prefix) :]

            # Update to new format: {area_name}_{entity_type}
            new_unique_id = f"{area_name}_{entity_suffix}"

            # Check for conflicts: see if another entity already has this unique_id
            conflict_found = False
            for other_entity_id, other_entity_entry in entity_registry.entities.items():
                if (
                    other_entity_id != entity_id
                    and other_entity_entry.unique_id == new_unique_id
                ):
                    conflict_found = True
                    conflicts.append((entity_id, str(old_unique_id), new_unique_id))
                    _LOGGER.warning(
                        "Unique ID conflict: %s already exists for entity %s. "
                        "Skipping migration for %s (old unique_id: %s)",
                        new_unique_id,
                        other_entity_id,
                        entity_id,
                        old_unique_id,
                    )
                    skipped_count += 1
                    break

            if conflict_found:
                continue

            _LOGGER.info(
                "Migrating entity unique_id: %s -> %s (entity: %s)",
                old_unique_id,
                new_unique_id,
                entity_id,
            )

            entity_registry.async_update_entity(
                entity_id,
                new_unique_id=new_unique_id,
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


async def _migrate_database_for_consolidation(
    hass: HomeAssistant,
    entries: list[ConfigEntry],
    entry_id_to_area_name: dict[str, str],
) -> None:
    """Migrate database entries for consolidation.

    Updates database to use area_name instead of entry_id as keys.
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
            engine = create_engine(
                f"sqlite:///{db_path}",
                echo=False,
                connect_args={"check_same_thread": False, "timeout": 30},
            )
            session_maker = sessionmaker(bind=engine)

            with session_maker() as session:
                # Update areas table: change entry_id to area_name
                try:
                    # Check if tables exist before attempting operations
                    inspector = sa.inspect(engine)
                    if not inspector.has_table("entities"):
                        _LOGGER.debug(
                            "Entities table does not exist, skipping database migration"
                        )
                        return
                    if not inspector.has_table("areas"):
                        _LOGGER.debug(
                            "Areas table does not exist, skipping database migration"
                        )
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
        # Run the unique ID migrations
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
