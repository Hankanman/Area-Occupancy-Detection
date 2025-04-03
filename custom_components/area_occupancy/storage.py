"""Storage handling for Area Occupancy Detection."""

import logging
from pathlib import Path
from typing import Any, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_VERSION, CONF_VERSION_MINOR, DOMAIN
from .exceptions import StorageError
from .types import PriorState

_LOGGER = logging.getLogger(__name__)

# Storage configuration
STORAGE_KEY = f"{DOMAIN}.storage"

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
            self._data = data.copy()
            await self.store.async_save(self._data)

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

            # Always ensure instances dict exists
            if "instances" not in data:
                data["instances"] = {}
                await self.store.async_save(data)

            _LOGGER.debug(
                "Loaded storage data with %d instances. Current instance: %s",
                len(data["instances"]),
                self.entry_id,
            )

            # Log all instance IDs for debugging
            instance_ids = list(data["instances"].keys())
            _LOGGER.debug("Current instances in storage: %s", instance_ids)

            self._data = data.copy()  # Make a copy to prevent race conditions
        except FileNotFoundError:
            _LOGGER.warning("Storage file not found, creating empty storage")
            data = self.store.create_empty_storage()
            data["instances"] = {}
            await self.store.async_save(data)
            self._data = data.copy()
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

            # Create instance data
            instance_data = InstanceData(
                name=data.get("name"),
                prior_state=data.get("prior_state"),
                last_updated=dt_util.utcnow().isoformat(),
            )

            # Update only the current instance's data
            existing_data["instances"][self.entry_id] = instance_data

            # Log the state of instances before saving
            _LOGGER.debug(
                "Saving storage data with %d instances. Current instance: %s. All instances: %s",
                len(existing_data["instances"]),
                self.entry_id,
                list(existing_data["instances"].keys()),
            )

            # Update local cache and save
            self._data = existing_data.copy()
            await self.store.async_save(self._data)

        except StorageError as err:
            _LOGGER.warning("Failed to save storage data: %s", err)

    async def async_save_prior_state(
        self, name: str, prior_state: PriorState, immediate: bool = False
    ) -> None:
        """Save prior state data to storage.

        Args:
            name: The name of the area
            prior_state: The prior state to save
            immediate: Deprecated parameter, all saves are immediate

        """
        try:
            # Load existing data first
            existing_data = await self.async_load()

            # Create instance data for this instance
            instance_data = InstanceData(
                name=name,
                prior_state=prior_state.to_dict(),
                last_updated=dt_util.utcnow().isoformat(),
            )

            # Update only this instance's data while preserving others
            existing_data["instances"][self.entry_id] = instance_data

            # Log all instances being saved
            _LOGGER.debug(
                "Saving prior state for instance %s (total instances: %d). All instances: %s",
                self.entry_id,
                len(existing_data["instances"]),
                list(existing_data["instances"].keys()),
            )

            # Update local cache and save
            self._data = existing_data.copy()
            await self.store.async_save(self._data)
            _LOGGER.debug(
                "Saved prior state for instance %s (total instances: %d)",
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
