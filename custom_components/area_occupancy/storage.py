"""Storage handling for Area Occupancy Detection."""

from datetime import datetime, timedelta
import logging
from typing import Optional
import asyncio

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from homeassistant.helpers.storage import Store
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.json import JSONEncoder
from homeassistant.helpers.debounce import Debouncer

from .const import (
    DOMAIN,
    STORAGE_VERSION,
    STORAGE_VERSION_MINOR,
    STORAGE_CLEANUP_INTERVAL,
    STORAGE_MAX_CACHE_AGE,
    STORAGE_SAVE_DELAY,
)
from .types import StorageData, validate_storage_data

_LOGGER = logging.getLogger(__name__)


class StorageEncoder(JSONEncoder):
    """Encoder for storage data."""

    def default(self, o):
        """Convert special objects."""
        if isinstance(o, (datetime, timedelta)):
            return o.isoformat()
        return super().default(o)


class AreaOccupancyStorage:
    """Handle storage operations."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize storage."""
        self.hass = hass
        self.entry_id = entry_id
        self._store = Store[StorageData](
            hass,
            STORAGE_VERSION,
            f"{DOMAIN}.{entry_id}.storage",
            atomic_writes=True,
            encoder=StorageEncoder,
        )
        self._save_lock = asyncio.Lock()
        self._last_save = dt_util.utcnow()
        self._save_interval = timedelta(seconds=STORAGE_SAVE_DELAY)

        # Create debounced save method
        self._pending_data = None
        self.async_save_debounced = Debouncer(
            self.hass,
            _LOGGER,
            cooldown=STORAGE_SAVE_DELAY,
            immediate=False,
            function=self._async_save_pending_data,
        )

        self._last_cleanup: Optional[datetime] = None

    async def async_load(self) -> StorageData:
        """Load storage data with migration handling."""
        try:
            stored_data = await self._store.async_load()
            if not stored_data:
                return self._create_empty_storage()

            # Check if migration is needed
            if self._needs_migration(stored_data):
                stored_data = await self._migrate_data(stored_data)

            # Validate data structure
            validated_data = validate_storage_data(stored_data)

            # Schedule cleanup if needed
            await self._maybe_cleanup_storage(validated_data)

            return validated_data

        except (IOError, ValueError, HomeAssistantError) as err:
            _LOGGER.error("Error loading storage data: %s", err)
            return self._create_empty_storage()

    def _create_empty_storage(self) -> StorageData:
        """Create empty storage structure."""
        now = dt_util.utcnow().isoformat()
        return StorageData(
            version=STORAGE_VERSION,
            version_minor=STORAGE_VERSION_MINOR,
            last_updated=now,
            data={},
            cache={},
            metadata={
                "created": now,
                "last_cleaned": None,
                "migrations": [],
                "schema_version": 1,
            },
        )

    def _needs_migration(self, data: dict) -> bool:
        """Check if data needs migration."""
        if not isinstance(data, dict):
            return True

        current_version = data.get("version", 0)
        current_minor = data.get("version_minor", 0)

        return current_version < STORAGE_VERSION or (
            current_version == STORAGE_VERSION and current_minor < STORAGE_VERSION_MINOR
        )

    async def _migrate_data(self, data: dict) -> StorageData:
        """Migrate data to current version."""
        try:
            # Version 0 to 1 migration
            if data.get("version", 0) < 1:
                data = await self._migrate_v0_to_v1(data)

            # Version 1 to 2 migration
            if data.get("version", 0) < 2:
                data = await self._migrate_v1_to_v2(data)

            # Update version and migration history
            data["version"] = STORAGE_VERSION
            data["version_minor"] = STORAGE_VERSION_MINOR
            data.setdefault("metadata", {})["migrations"] = ["v0_to_v1", "v1_to_v2"]

            return validate_storage_data(data)

        except (ValueError, TypeError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error migrating data: %s", err)
            return self._create_empty_storage()

    async def _migrate_v0_to_v1(self, data: dict) -> dict:
        """Migrate from version 0 to 1."""
        # Add your migration logic here
        return data

    async def _migrate_v1_to_v2(self, data: dict) -> dict:
        """Migrate from version 1 to 2."""
        # Add your migration logic here
        return data

    async def async_save(self, data: dict) -> None:
        """Save data with debouncing and validation."""
        try:
            stored_data = await self._store.async_load() or {}
            storage_data = validate_storage_data(
                {
                    "version": STORAGE_VERSION,
                    "version_minor": STORAGE_VERSION_MINOR,
                    "last_updated": dt_util.utcnow().isoformat(),
                    "data": data,
                    "cache": stored_data.get("cache", {}),
                    "metadata": stored_data.get("metadata", {}),
                }
            )

            self._pending_data = storage_data
            await self.async_save_debounced.async_call()

        except (ValueError, TypeError, HomeAssistantError) as err:
            _LOGGER.error("Error preparing storage data: %s", err)
            raise HomeAssistantError(f"Failed to save storage data: {err}") from err

    async def _async_save_data(self, data: StorageData) -> None:
        """Actually perform the save operation with rate limiting."""
        async with self._save_lock:
            now = dt_util.utcnow()
            if now - self._last_save < self._save_interval:
                return

            try:
                await self._store.async_save(data)
                self._last_save = now
                _LOGGER.debug("Storage data saved successfully")
            except (IOError, HomeAssistantError) as err:
                _LOGGER.error("Error saving storage data: %s", err)
                raise HomeAssistantError(f"Failed to save storage data: {err}") from err

    async def async_remove(self) -> None:
        """Remove storage file and clean up any misnamed files."""
        try:
            # Remove the main storage file
            await self._store.async_save({})
            await self._store.async_remove()

            # Clean up any misnamed storage files
            storage_dir = self.hass.config.path(".storage")
            pattern = f"*{DOMAIN}*"
            correct_name = f"{DOMAIN}.{self.entry_id}.storage"

            for file in storage_dir.glob(pattern):
                if file.name != correct_name:
                    try:
                        file.unlink()
                        _LOGGER.debug("Removed misnamed storage file: %s", file.name)
                    except OSError as err:
                        _LOGGER.error(
                            "Error removing misnamed storage file %s: %s",
                            file.name,
                            err,
                        )

            _LOGGER.debug("Storage files cleaned up successfully")
        except IOError as err:
            _LOGGER.error("Error removing storage files: %s", err)

    async def _maybe_cleanup_storage(self, data: StorageData) -> None:
        """Perform periodic cleanup of old cache data."""
        try:
            now = dt_util.utcnow()
            last_cleaned = None

            if data["metadata"].get("last_cleaned"):
                last_cleaned = dt_util.parse_datetime(data["metadata"]["last_cleaned"])

            if not last_cleaned or now - last_cleaned > STORAGE_CLEANUP_INTERVAL:

                await self._cleanup_storage(data)

        except (ValueError, TypeError) as err:
            _LOGGER.error("Error during storage cleanup check: %s", err)

    async def _cleanup_storage(self, data: StorageData) -> None:
        """Clean up old cache data."""
        try:
            now = dt_util.utcnow()
            cleaned_cache = {}

            # Clean up old cache entries
            for key, entry in data["cache"].items():
                if isinstance(entry, dict) and "timestamp" in entry:
                    timestamp = dt_util.parse_datetime(entry["timestamp"])
                    if timestamp and now - timestamp <= STORAGE_MAX_CACHE_AGE:
                        cleaned_cache[key] = entry

            # Update storage with cleaned data
            data["cache"] = cleaned_cache
            data["metadata"]["last_cleaned"] = now.isoformat()

            await self._store.async_save(data)
            _LOGGER.debug("Storage cleanup completed")

        except (IOError, ValueError, HomeAssistantError) as err:
            _LOGGER.error("Error cleaning up storage: %s", err)

    async def _async_save_pending_data(self) -> None:
        """Save pending data."""
        if self._pending_data:
            await self._async_save_data(self._pending_data)
