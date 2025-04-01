"""Storage handling for Area Occupancy Detection."""

import logging
from typing import Any, Dict
from datetime import timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    STORAGE_VERSION,
    STORAGE_VERSION_MINOR,
    CONF_NAME,
)
from .types import PriorState
from .exceptions import StorageError

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyStorage:
    """Handle storage of area occupancy data."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize storage."""
        self.hass = hass
        self.entry_id = entry_id
        self.store = Store(
            hass,
            STORAGE_VERSION,
            f"{DOMAIN}.{entry_id}.storage",
            minor_version=STORAGE_VERSION_MINOR,
            atomic_writes=True,
        )
        # Assign the migration function
        self.store._async_migrate_func = self._async_migrate
        self._last_save = dt_util.utcnow()
        self._save_interval = timedelta(seconds=10)

    async def async_load(self) -> Dict[str, Any]:
        """Load data from storage."""
        try:
            data = await self.store.async_load()
            if data is None:
                data = self._create_empty_storage()
                await self.async_save(data)
            return data
        except Exception as err:
            _LOGGER.error("Error loading stored data: %s", err, exc_info=True)
            raise StorageError(f"Failed to load stored data: {err}") from err

    async def async_save(self, data: Dict[str, Any]) -> None:
        """Save data to storage."""
        try:
            now = dt_util.utcnow()
            if now - self._last_save < self._save_interval:
                return

            await self.store.async_save(data)
            self._last_save = now
            _LOGGER.debug("Successfully saved data")
        except Exception as err:
            _LOGGER.error("Error saving data: %s", err, exc_info=True)
            raise StorageError(f"Failed to save data: {err}") from err

    async def async_save_prior_state(self, name: str, prior_state: PriorState) -> None:
        """Save prior state data to storage.

        Args:
            name: The name of the area
            prior_state: The prior state to save
        """
        try:
            data = {
                "name": name,
                "prior_state": prior_state.to_dict(),
            }
            await self.async_save(data)
        except Exception as err:
            _LOGGER.error("Error saving prior state: %s", err, exc_info=True)
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
            _LOGGER.error("Error loading prior state: %s", err, exc_info=True)
            raise StorageError(f"Failed to load prior state: {err}") from err

    async def _async_migrate(
        self, old_version: int, old_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Migrate old data to the new version if needed."""
        _LOGGER.debug(
            "Migrating storage from version %s to %s.%s",
            old_version,
            STORAGE_VERSION,
            STORAGE_VERSION_MINOR,
        )

        if not old_data:
            return self._create_empty_storage()

        data = dict(old_data)
        data["version"] = STORAGE_VERSION
        data["version_minor"] = STORAGE_VERSION_MINOR
        return data

    def _create_empty_storage(self) -> Dict[str, Any]:
        """Create default storage structure."""
        now = dt_util.utcnow().isoformat()
        return {
            "version": STORAGE_VERSION,
            "version_minor": STORAGE_VERSION_MINOR,
            "last_updated": now,
            "data": {},
        }
