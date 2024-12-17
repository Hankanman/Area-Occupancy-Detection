"""Storage handling for Area Occupancy Detection."""

import logging
from typing import Any, Dict

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN, STORAGE_VERSION, STORAGE_VERSION_MINOR

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

    async def async_load(self) -> Dict[str, Any]:
        """Load data from storage."""
        data = await self.store.async_load()
        if data is None:
            data = self._create_empty_storage()
            await self.async_save(data)
        return data

    async def async_save(self, data: Dict[str, Any]) -> None:
        """Save data to storage."""
        await self.store.async_save(data)

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
