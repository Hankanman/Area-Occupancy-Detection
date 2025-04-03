"""Storage handling for Area Occupancy Detection."""

import logging
from pathlib import Path
from typing import Any, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_VERSION, CONF_VERSION_MINOR, DOMAIN
from .exceptions import StorageError
from .types import PriorState

_LOGGER = logging.getLogger(__name__)

# Storage configuration
STORAGE_KEY = f"{DOMAIN}.storage"
STORAGE_SAVE_DELAY_SECONDS = 120  # Buffer writes every 2 minutes

# Version control
OLD_VERSION = 6
OLD_VERSION_MINOR = 1

# File patterns
STORAGE_FILE_PATTERN = f"{DOMAIN}.*.storage"


class StorageLoadError(StorageError):
    """Error raised when loading storage fails."""


class StorageSaveError(StorageError):
    """Error raised when saving storage fails."""


class StorageMigrationError(StorageError):
    """Error raised when storage migration fails."""


class InstanceData(TypedDict):
    """TypedDict for stored instance data."""

    name: str | None
    prior_state: dict[str, Any] | None
    last_updated: str


class StoredData(TypedDict):
    """TypedDict for stored data structure."""

    instances: dict[str, InstanceData]


class AreaOccupancyStorageStore(Store[StoredData]):
    """Store class for area occupancy data."""

    def __init__(
        self,
        hass: HomeAssistant,
    ) -> None:
        """Initialize the store."""
        # Initialize with current version - migration will handle version changes
        super().__init__(
            hass,
            CONF_VERSION,  # Start with current version
            STORAGE_KEY,
            minor_version=CONF_VERSION_MINOR,
            atomic_writes=True,
            private=True,  # Mark as private since it contains state data
        )
        self.hass = hass
        self._current_version = CONF_VERSION
        self._current_minor_version = CONF_VERSION_MINOR
        self._old_version = OLD_VERSION
        self._old_minor_version = OLD_VERSION_MINOR

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
    ) -> StoredData:
        """Migrate to the new version by deleting old files and starting fresh."""
        _LOGGER.debug(
            "Starting storage migration from version %s.%s to %s.%s",
            old_major_version,
            old_minor_version,
            self._current_version,
            self._current_minor_version,
        )

        try:
            # Clean up old files first
            storage_dir = Path(self.hass.config.path(".storage"))
            _LOGGER.debug(
                "Scanning %s for old storage files matching %s",
                storage_dir,
                STORAGE_FILE_PATTERN,
            )
            found_files = list(storage_dir.glob(STORAGE_FILE_PATTERN))
            _LOGGER.debug(
                "Found %d storage files: %s",
                len(found_files),
                [f.name for f in found_files],
            )

            # Delete all old files
            for file in found_files:
                # Skip the new consolidated file
                if file.name == f"{STORAGE_KEY}":
                    _LOGGER.debug(
                        "Skipping new consolidated storage file: %s", file.name
                    )
                    continue

                try:
                    _LOGGER.debug("Removing old storage file: %s", file)
                    file.unlink()
                    _LOGGER.info("Successfully removed old storage file: %s", file)
                except OSError as err:
                    _LOGGER.warning("Error removing old storage file %s: %s", file, err)

            # Create fresh storage
            _LOGGER.info("Creating fresh storage file")
            return self.create_empty_storage()

        except Exception as err:
            raise StorageMigrationError(f"Failed to migrate storage: {err}") from err

    async def async_migrate_storage(self) -> None:
        """Force migration of storage data."""
        _LOGGER.debug("Forcing storage migration")
        # Load data with old version to trigger migration
        data = await self.async_load()
        if data is None:
            data = self.create_empty_storage()

        # Save with current version
        await self.async_save(data)
        _LOGGER.debug("Storage migration complete")

    def create_empty_storage(self) -> StoredData:
        """Create default storage structure."""
        return StoredData(instances={})


class AreaOccupancyStorage:
    """Handle storage of area occupancy data."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize storage."""
        self.hass = hass
        self.entry_id = entry_id
        self.store = AreaOccupancyStorageStore(hass)
        self._data: StoredData | None = None

    async def async_migrate_storage(self) -> None:
        """Migrate storage data."""
        try:
            _LOGGER.debug("Starting storage migration")
            # Load data with old version to trigger migration
            data = await self.store.async_load()
            if data is None:
                data = self.store.create_empty_storage()

            # Save with current version to ensure migration
            await self.store.async_save(data)

            # Clean up old instance-specific storage file
            old_file = Path(
                self.hass.config.path(".storage", f"{DOMAIN}.{self.entry_id}.storage")
            )
            if old_file.exists():
                try:
                    _LOGGER.debug("Removing old storage file: %s", old_file)
                    old_file.unlink()
                    _LOGGER.info("Successfully removed old storage file: %s", old_file)
                except OSError as err:
                    _LOGGER.warning(
                        "Error removing old storage file %s: %s", old_file, err
                    )

            _LOGGER.debug("Storage migration complete")
        except Exception as err:
            _LOGGER.error("Error during storage migration: %s", err)
            raise StorageError(f"Failed to migrate storage: {err}") from err

    async def async_load(self) -> StoredData:
        """Load data from storage.

        Returns:
            The loaded storage data

        Raises:
            StorageLoadError: If loading fails

        """
        try:
            data = await self.store.async_load()
            if data is None:
                _LOGGER.warning("No stored data found, creating empty storage")
                data = self.store.create_empty_storage()
                await self.store.async_save(data)
            else:
                _LOGGER.debug(
                    "Loaded storage data with %d instances. Current instance: %s",
                    len(data["instances"]),
                    self.entry_id,
                )
            self._data = data
        except FileNotFoundError:
            _LOGGER.warning("Storage file not found, creating empty storage")
            data = self.store.create_empty_storage()
            await self.store.async_save(data)
            self._data = data
            return data
        except Exception as err:
            _LOGGER.exception("Error loading stored data")
            raise StorageLoadError(f"Failed to load stored data: {err}") from err
        else:
            return data

    async def async_save(self, data: StoredData) -> None:
        """Save data to storage.

        Args:
            data: The data to save

        """
        try:
            # Load existing data first
            existing_data = await self.async_load()
            if "instances" not in existing_data:
                existing_data["instances"] = {}

            # Create instance data
            instance_data = InstanceData(
                name=data.get("name"),
                prior_state=data.get("prior_state"),
                last_updated=dt_util.utcnow().isoformat(),
            )

            # Update only the current instance's data
            existing_data["instances"][self.entry_id] = instance_data

            _LOGGER.debug(
                "Saving storage data with %d instances. Current instance: %s",
                len(existing_data["instances"]),
                self.entry_id,
            )
            await self.store.async_save(existing_data)
            self._data = existing_data
        except StorageError as err:
            _LOGGER.warning("Failed to save storage data: %s", err)

    def _data_to_save(self) -> StoredData:
        """Return data to save.

        Returns:
            The current data to save, or empty storage if no data exists

        """
        if not self._data or "instances" not in self._data:
            _LOGGER.debug("No valid data to save, creating empty storage")
            return self.store.create_empty_storage()

        _LOGGER.debug(
            "Saving storage data with %d instances. Current instance: %s",
            len(self._data["instances"]),
            self.entry_id,
        )
        return self._data

    async def async_save_prior_state(
        self, name: str, prior_state: PriorState, immediate: bool = False
    ) -> None:
        """Save prior state data to storage.

        Args:
            name: The name of the area
            prior_state: The prior state to save
            immediate: If True, save immediately instead of using debounced save

        """
        try:
            # Load existing data first
            existing_data = await self.async_load()
            if "instances" not in existing_data:
                existing_data["instances"] = {}

            # Create instance data for this instance
            instance_data = InstanceData(
                name=name,
                prior_state=prior_state.to_dict(),
                last_updated=dt_util.utcnow().isoformat(),
            )

            # Update only this instance's data while preserving others
            existing_data["instances"][self.entry_id] = instance_data

            _LOGGER.debug(
                "Saving prior state for instance %s (total instances: %d)",
                self.entry_id,
                len(existing_data["instances"]),
            )

            if immediate:
                await self.store.async_save(existing_data)
                _LOGGER.debug(
                    "Immediately saved prior state for instance %s (total instances: %d)",
                    self.entry_id,
                    len(existing_data["instances"]),
                )
            else:
                self._data = existing_data
                self.store.async_delay_save(
                    self._data_to_save, STORAGE_SAVE_DELAY_SECONDS
                )
                _LOGGER.debug(
                    "Scheduled save of prior state for instance %s (total instances: %d)",
                    self.entry_id,
                    len(existing_data["instances"]),
                )
        except StorageError as err:
            _LOGGER.warning("Failed to save prior state: %s", err)

    async def async_load_prior_state(self) -> tuple[str, PriorState | None]:
        """Load prior state data from storage.

        Returns:
            Tuple of (name, prior_state) where prior_state may be None if not found

        Raises:
            StorageLoadError: If loading fails

        """
        try:
            data = await self.async_load()
            if not data or "instances" not in data:
                return "", None

            instance_data = data["instances"].get(self.entry_id)
            if not instance_data:
                return "", None

            name = instance_data.get("name", "")
            stored_prior_state = instance_data.get("prior_state")
            if not stored_prior_state:
                return name, None

            return name, PriorState.from_dict(stored_prior_state)
        except Exception as err:
            _LOGGER.exception("Error loading prior state")
            raise StorageLoadError(f"Failed to load prior state: {err}") from err
