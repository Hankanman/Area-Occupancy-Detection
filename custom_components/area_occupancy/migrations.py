"""Migration handlers for Area Occupancy Detection."""

from __future__ import annotations

import logging
from pathlib import Path

from filelock import FileLock

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_VERSION
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
# Entry Migration (Main Entry Point)
# ============================================================================


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to the new version."""
    # Handle v13 breaking change: reset database if needed for older versions
    await async_reset_database_if_needed(hass, config_entry.version)
    return True
