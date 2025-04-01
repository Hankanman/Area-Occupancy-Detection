"""Storage handling for Area Occupancy Detection."""

import logging
from typing import Any, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_NAME, DOMAIN, STORAGE_VERSION, STORAGE_VERSION_MINOR
from .exceptions import StorageError
from .types import PriorState

_LOGGER = logging.getLogger(__name__)

# Buffer writes every 2 minutes (plus guaranteed to be written at shutdown)
STORAGE_SAVE_DELAY_SECONDS = 120


class StoredData(TypedDict):
    """TypedDict for stored data structure."""

    version: int
    version_minor: int
    last_updated: str
    data: dict[str, Any]
    name: str | None
    prior_state: dict[str, Any] | None


class AreaOccupancyStorageStore(Store[StoredData]):
    """Store class for area occupancy data."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
    ) -> None:
        """Initialize the store."""
        super().__init__(
            hass,
            STORAGE_VERSION,
            f"{DOMAIN}.{entry_id}.storage",
            minor_version=STORAGE_VERSION_MINOR,
            atomic_writes=True,
            private=True,  # Mark as private since it contains state data
        )

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
        old_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Migrate to the new version."""
        _LOGGER.debug(
            "Migrating storage from version %s.%s to %s.%s",
            old_major_version,
            old_minor_version,
            STORAGE_VERSION,
            STORAGE_VERSION_MINOR,
        )

        if not old_data:
            return self.create_empty_storage()

        data = dict(old_data)
        data["version"] = STORAGE_VERSION
        data["version_minor"] = STORAGE_VERSION_MINOR
        return data

    def create_empty_storage(self) -> StoredData:
        """Create default storage structure."""
        now = dt_util.utcnow().isoformat()
        return StoredData(
            version=STORAGE_VERSION,
            version_minor=STORAGE_VERSION_MINOR,
            last_updated=now,
            data={},
            name=None,
            prior_state=None,
        )


class AreaOccupancyStorage:
    """Handle storage of area occupancy data."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize storage."""
        self.hass = hass
        self.entry_id = entry_id
        self.store = AreaOccupancyStorageStore(hass, entry_id)
        self._data: StoredData | None = None

    async def async_load(self) -> dict[str, Any]:
        """Load data from storage."""
        try:
            data = await self.store.async_load()
            if data is None:
                data = self.store.create_empty_storage()
                await self.async_save(data)
            self._data = data
        except Exception as err:
            _LOGGER.exception("Error loading stored data")
            raise StorageError(f"Failed to load stored data: {err}") from err
        else:
            return data

    async def async_save(self, data: dict[str, Any]) -> None:
        """Save data to storage."""
        try:
            self._data = data
            self.store.async_delay_save(self._data_to_save, STORAGE_SAVE_DELAY_SECONDS)
            _LOGGER.debug("Successfully scheduled data save")
        except Exception as err:
            _LOGGER.exception("Error saving data")
            raise StorageError(f"Failed to save data: {err}") from err

    def _data_to_save(self) -> StoredData:
        """Return data to save."""
        if not self._data:
            return self.store.create_empty_storage()
        return self._data

    async def async_save_prior_state(self, name: str, prior_state: PriorState) -> None:
        """Save prior state data to storage.

        Args:
            name: The name of the area
            prior_state: The prior state to save

        """
        try:
            data = self._data or self.store.create_empty_storage()
            data["name"] = name
            data["prior_state"] = prior_state.to_dict()
            data["last_updated"] = dt_util.utcnow().isoformat()
            await self.async_save(data)
        except Exception as err:
            _LOGGER.exception("Error saving prior state")
            raise StorageError(f"Failed to save prior state: {err}") from err

    async def async_load_prior_state(self) -> tuple[str, PriorState | None]:
        """Load prior state data from storage.

        Returns:
            Tuple of (name, prior_state) where prior_state may be None if not found

        """
        try:
            data = await self.async_load()
            if not data:
                return "", None

            name = data.get(CONF_NAME, "")
            stored_prior_state = data.get("prior_state")
            if not stored_prior_state:
                return name, None

            return name, PriorState.from_dict(stored_prior_state)
        except Exception as err:
            _LOGGER.exception("Error loading prior state")
            raise StorageError(f"Failed to load prior state: {err}") from err
