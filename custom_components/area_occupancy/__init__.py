"""The Area Occupancy Detection integration."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker as create_sessionmaker

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import (
    area_registry as ar,
    config_validation as cv,
    device_registry as dr,
)
from homeassistant.helpers.typing import ConfigType

from .const import CONF_AREA_ID, CONF_AREAS, CONF_VERSION, DB_NAME, DOMAIN, PLATFORMS
from .coordinator import AreaOccupancyCoordinator
from .db.operations import delete_area_data as _delete_area_data
from .db.schema import (
    AreaRelationships,
    Areas as AreasTable,
    Correlations,
    CrossAreaStats,
    Entities,
    EntityStatistics,
    GlobalPriors,
    IntervalAggregates,
    Intervals,
    NumericAggregates,
    NumericSamples,
    OccupiedIntervalsCache,
    Priors,
)
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


def _resolve_db_path(hass: HomeAssistant) -> Path | None:
    """Return the absolute path to the Area Occupancy SQLite database, if any."""
    config_dir = getattr(hass.config, "config_dir", None)
    if not config_dir:
        return None
    return Path(config_dir) / ".storage" / DB_NAME


def _read_area_names_by_area_id(db_path: Path) -> dict[str, list[str]]:
    """Read every (area_id -> [area_name, ...]) mapping from the Areas table.

    Multiple ``area_name`` rows can exist for a single ``area_id`` because
    ``area_name`` is the Areas PK and renames leave historical rows behind.
    All names are returned so removal can clean up historical rows too.

    Runs synchronously and is expected to be invoked via
    ``hass.async_add_executor_job`` — it performs blocking SQLAlchemy I/O.
    """
    if not db_path.exists():
        return {}
    by_area_id: dict[str, list[str]] = {}
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        pool_pre_ping=True,
        poolclass=sa.pool.NullPool,
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                sa.select(AreasTable.area_id, AreasTable.area_name)
            ).all()
        for area_id, area_name in rows:
            if not area_id or not area_name:
                continue
            by_area_id.setdefault(area_id, []).append(area_name)
    finally:
        engine.dispose()
    return by_area_id


def _purge_entry_database_data(
    db_path: Path, area_names: list[str], entry_id: str
) -> dict[str, int]:
    """Delete all database rows for the given areas using a temporary offline session.

    Runs in an executor because it performs blocking SQLAlchemy I/O.
    """
    result = {"areas_attempted": len(area_names), "areas_deleted": 0, "rows_deleted": 0}
    if not area_names or not db_path.exists():
        return result

    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        pool_pre_ping=True,
        poolclass=sa.pool.NullPool,
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    try:
        session_maker = create_sessionmaker(bind=engine)

        class _OfflineDB:
            """Lightweight DB adapter matching delete_area_data's interface."""

            def __init__(self) -> None:
                self.Areas = AreasTable
                self.Entities = Entities
                self.Intervals = Intervals
                self.Priors = Priors
                self.GlobalPriors = GlobalPriors
                self.OccupiedIntervalsCache = OccupiedIntervalsCache
                self.IntervalAggregates = IntervalAggregates
                self.NumericSamples = NumericSamples
                self.NumericAggregates = NumericAggregates
                self.Correlations = Correlations
                self.EntityStatistics = EntityStatistics
                self.AreaRelationships = AreaRelationships
                self.CrossAreaStats = CrossAreaStats

            @contextmanager
            def get_session(self) -> Any:
                session = session_maker()
                try:
                    yield session
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()

        offline_db = _OfflineDB()
        for area_name in area_names:
            try:
                deleted = _delete_area_data(offline_db, area_name)
            except Exception:
                _LOGGER.exception(
                    "Failed to purge database data for area '%s' during entry removal",
                    area_name,
                )
                continue
            if deleted:
                result["areas_deleted"] += 1
                result["rows_deleted"] += deleted
            _LOGGER.info(
                "Purged database rows for area '%s' during entry removal (entry_id=%s, rows=%d)",
                area_name,
                entry_id,
                deleted,
            )
    finally:
        engine.dispose()
    return result


def _delete_db_file(db_path: Path) -> bool:
    """Delete the SQLite database file and its WAL/SHM siblings if present."""
    deleted_any = False
    for suffix in ("", "-wal", "-shm", "-journal"):
        target = db_path.with_name(db_path.name + suffix) if suffix else db_path
        try:
            if target.exists():
                target.unlink()
                deleted_any = True
                _LOGGER.info("Removed Area Occupancy database file: %s", target)
        except OSError:
            _LOGGER.exception("Failed to remove database file %s", target)
    return deleted_any


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of a config entry and clean up learned history.

    Deletes all database rows for every area in the removed entry. When this
    is the last config entry for the domain, the SQLite database file itself
    is removed so a fresh install starts with a clean slate.

    Idempotent and never raises: failures are logged but must not prevent
    Home Assistant from completing the entry removal.
    """
    _LOGGER.info(
        "Removing Area Occupancy config entry %s (cleaning up learned history)",
        entry.entry_id,
    )

    try:
        merged: dict[str, Any] = dict(entry.data)
        merged.update(entry.options)
        area_configs = merged.get(CONF_AREAS, []) or []

        db_path = _resolve_db_path(hass)

        # Resolve area_name(s) per area. Prefer names stored in the DB by
        # area_id (survives renames and captures historical rows from prior
        # renames); otherwise fall back to the HA area registry. Blocking DB
        # I/O runs in an executor.
        by_area_id: dict[str, list[str]] = {}
        if db_path is not None and db_path.exists():
            try:
                by_area_id = await hass.async_add_executor_job(
                    _read_area_names_by_area_id, db_path
                )
            except Exception:
                _LOGGER.exception(
                    "Failed to read area names from database during removal"
                )
                by_area_id = {}

        area_names: list[str] = []
        seen: set[str] = set()
        for area_data in area_configs:
            area_id = area_data.get(CONF_AREA_ID)
            if not area_id:
                continue
            candidates: list[str] = list(by_area_id.get(area_id, []))
            try:
                area_entry = ar.async_get(hass).async_get_area(area_id)
            except Exception:  # noqa: BLE001
                area_entry = None
            if area_entry is not None and area_entry.name:
                candidates.append(area_entry.name)
            for name in candidates:
                if name and name not in seen:
                    seen.add(name)
                    area_names.append(name)

        other_entries = [
            e
            for e in hass.config_entries.async_entries(DOMAIN)
            if e.entry_id != entry.entry_id
        ]

        if db_path is not None and db_path.exists() and area_names:
            try:
                stats = await hass.async_add_executor_job(
                    _purge_entry_database_data, db_path, area_names, entry.entry_id
                )
                _LOGGER.info(
                    "Cleaned learned history for %d/%d area(s) (rows deleted=%d) during "
                    "entry removal %s",
                    stats["areas_deleted"],
                    stats["areas_attempted"],
                    stats["rows_deleted"],
                    entry.entry_id,
                )
            except Exception:
                _LOGGER.exception(
                    "Failed to purge learned history during entry removal %s",
                    entry.entry_id,
                )

        # When no other entries remain, drop the whole database so a re-install
        # starts clean.
        if not other_entries and db_path is not None:
            try:
                await hass.async_add_executor_job(_delete_db_file, db_path)
            except Exception:
                _LOGGER.exception(
                    "Failed to remove database file during entry removal %s",
                    entry.entry_id,
                )
    except Exception:
        # Final safety net — entry removal must never raise.
        _LOGGER.exception(
            "Unexpected error during async_remove_entry for %s", entry.entry_id
        )


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

    # Determine configured area IDs from merged data+options.
    merged = dict(entry.data)
    merged.update(entry.options)
    config_area_ids = {
        a.get(CONF_AREA_ID) for a in merged.get(CONF_AREAS, []) if a.get(CONF_AREA_ID)
    }

    # Determine currently loaded area IDs.
    current_area_ids = {
        area.config.area_id
        for area in coordinator.areas.values()
        if area.config.area_id
    }

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
        for area_name, area in coordinator.areas.items():
            try:
                area.config.update_from_entry(entry)
                await area.entities.cleanup()
            except Exception:
                _LOGGER.exception("Failed to update config for area %s", area_name)
        await coordinator.async_request_refresh()
