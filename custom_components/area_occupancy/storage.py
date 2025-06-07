"""Storage manager for Area Occupancy Detection."""

import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .exceptions import StorageError
from .models.entity import EntityManager

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.storage"


class StorageManager(Store[dict[str, Any]]):
    """Manage storage for Area Occupancy Detection."""

    def __init__(
        self,
        hass: HomeAssistant,
    ) -> None:
        """Initialize the storage manager."""
        super().__init__(
            hass,
            STORAGE_VERSION,
            STORAGE_KEY,
        )
        self._initialized = False

    def create_empty_storage(self) -> dict[str, Any]:
        """Create empty storage."""
        return {"instances": {}}

    async def async_initialize(self) -> None:
        """Initialize the storage manager and perform cleanup."""
        if self._initialized:
            return

        await self._async_perform_cleanup()
        self._initialized = True

    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: dict
    ) -> dict:
        """Migrate data to new version."""
        if old_major_version < 1:
            # Migrate from version 0 to 1
            new_data: dict[str, Any] = {"instances": {}}
            for entry_id in old_data.get("instances", {}):
                new_data["instances"][entry_id] = {
                    "entities": {},
                    "last_updated": dt_util.utcnow().isoformat(),
                }
            return new_data
        return old_data

    async def _async_perform_cleanup(self) -> None:
        """Perform storage cleanup by removing orphaned instances."""
        try:
            _LOGGER.debug("Checking storage for orphaned instances")
            active_entry_ids = {
                entry.entry_id
                for entry in self.hass.config_entries.async_entries(DOMAIN)
            }
            _LOGGER.debug("Active entry IDs: %s", active_entry_ids)

            await self.async_cleanup_orphaned_instances(active_entry_ids)

        except Exception:
            # Log error but don't prevent setup from continuing
            _LOGGER.exception("Error during storage cleanup")

    async def async_remove_instance(self, entry_id: str) -> bool:
        """Remove an instance from storage."""
        try:
            data = await self.async_load()
            if not data:
                return False

            if entry_id in data.get("instances", {}):
                del data["instances"][entry_id]
                await self.async_save(data)
                return True

        except (StorageError, HomeAssistantError) as err:
            _LOGGER.error("Error removing instance %s: %s", entry_id, err)
            return False
        else:
            return False

    async def async_cleanup_orphaned_instances(
        self, active_entry_ids: set[str]
    ) -> bool:
        """Remove orphaned instances from storage."""
        try:
            data = await self.async_load()
            if not data:
                return False

            instances = data.get("instances", {})
            if not instances:
                return False

            # Find orphaned instances
            orphaned = set(instances.keys()) - active_entry_ids
            if not orphaned:
                return False

            # Remove orphaned instances
            for entry_id in orphaned:
                del instances[entry_id]

            await self.async_save(data)

        except (StorageError, HomeAssistantError) as err:
            _LOGGER.error("Error cleaning up orphaned instances: %s", err)
            return False
        else:
            return True

    async def async_load_instance_data(self, entry_id: str) -> dict[str, Any] | None:
        """Load instance data from storage."""
        try:
            data = await self.async_load()
            if not data:
                return None

            return data.get("instances", {}).get(entry_id)

        except (StorageError, HomeAssistantError) as err:
            _LOGGER.error("Error loading instance data for %s: %s", entry_id, err)
            return None

    async def async_save_instance_data(
        self, entry_id: str, entity_manager: EntityManager
    ) -> None:
        """Save instance data to storage."""
        try:
            data = await self.async_load()
            if not data:
                data = self.create_empty_storage()

            # Convert entity manager to dict
            instance_data = entity_manager.to_dict()

            # Update storage
            data["instances"][entry_id] = instance_data
            await self.async_save(data)

        except (StorageError, HomeAssistantError) as err:
            _LOGGER.error("Error saving instance data for %s: %s", entry_id, err)
            raise
