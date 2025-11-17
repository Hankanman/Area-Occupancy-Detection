"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any

from filelock import FileLock
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .binary_sensor import NAME_BINARY_SENSOR
from .const import DEFAULT_THRESHOLD, DOMAIN
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

            # If version doesn't match current DB_VERSION, delete and recreate database
            if db_version != DB_VERSION:
                _LOGGER.info(
                    "Database version %d doesn't match current version %d. "
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
# Entry Migration (Main Entry Point)
# ============================================================================


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to the new version."""
    # No migration needed - all legacy support has been removed
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
