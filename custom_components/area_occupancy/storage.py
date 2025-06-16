"""Storage manager for Area Occupancy Detection."""

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

from .const import CONF_VERSION, CONF_VERSION_MINOR, DOMAIN, STORAGE_KEY
from .data.entity import EntityManager

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


class StorageManager(Store[dict[str, Any]]):
    """Manage storage for Area Occupancy Detection."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
    ) -> None:
        """Initialize the storage manager."""
        super().__init__(
            hass=coordinator.hass,
            version=CONF_VERSION,
            key=STORAGE_KEY,
            minor_version=CONF_VERSION_MINOR,
        )
        self._coordinator = coordinator

    def create_empty_storage(self) -> dict[str, Any]:
        """Create empty storage."""
        return {
            "version": CONF_VERSION,
            "minor_version": CONF_VERSION_MINOR,
            "instances": {},
        }

    async def async_initialize(self) -> None:
        """Initialize the storage manager."""
        await self._async_perform_cleanup()

    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: dict
    ) -> dict:
        """Migrate data to new version."""
        _LOGGER.info(
            "Migrating storage from version %d.%d to %d.%d",
            old_major_version,
            old_minor_version,
            CONF_VERSION,
            CONF_VERSION_MINOR,
        )

        # For any major version difference, reset to empty storage
        if old_major_version < CONF_VERSION:
            _LOGGER.info("Major version change, resetting storage")
            return self.create_empty_storage()

        # Validate structure
        if not isinstance(old_data, dict) or "instances" not in old_data:
            _LOGGER.info("Invalid storage format, resetting")
            return self.create_empty_storage()

        # Return data with proper structure
        migrated_data = self.create_empty_storage()
        migrated_data["instances"] = old_data.get("instances", {})
        return migrated_data

    async def _async_perform_cleanup(self) -> None:
        """Perform storage cleanup by removing orphaned instances."""
        try:
            active_entry_ids = {
                entry.entry_id
                for entry in self._coordinator.hass.config_entries.async_entries(DOMAIN)
            }

            if active_entry_ids:
                await self.async_cleanup_orphaned_instances(active_entry_ids)

        except Exception as err:
            _LOGGER.warning("Error during storage cleanup: %s", err)

    async def async_remove_instance(self, entry_id: str) -> bool:
        """Remove instance data by entry ID."""
        try:
            data = await self.async_load() or self.create_empty_storage()

            if "instances" not in data:
                data["instances"] = {}

            if entry_id in data["instances"]:
                del data["instances"][entry_id]
                await self.async_save(data)
                _LOGGER.info("Removed instance data for entry %s", entry_id)
                return True

            return False

        except Exception as err:
            _LOGGER.error("Error removing instance %s: %s", entry_id, err)
            raise HomeAssistantError(f"Failed to remove instance: {err}") from err

    async def async_cleanup_orphaned_instances(
        self, active_entry_ids: set[str]
    ) -> bool:
        """Remove orphaned instances from storage."""
        try:
            data = await self.async_load() or self.create_empty_storage()

            if "instances" not in data:
                data["instances"] = {}

            instances = data["instances"]
            orphaned = set(instances.keys()) - active_entry_ids

            if not orphaned:
                return False

            _LOGGER.info("Removing %d orphaned instances: %s", len(orphaned), orphaned)

            for entry_id in orphaned:
                del instances[entry_id]

            await self.async_save(data)
            return True

        except Exception as err:
            _LOGGER.error("Error cleaning up orphaned instances: %s", err)
            return False

    async def async_load_instance_data(self, entry_id: str) -> dict[str, Any] | None:
        """Load instance data for a specific entry."""
        try:
            data = await self.async_load() or {}

            instances = data.get("instances", {})
            instance_data = instances.get(entry_id)

            if instance_data is None:
                _LOGGER.debug("No data found for entry %s", entry_id)
                return None

            _LOGGER.debug("Loaded instance data for entry %s", entry_id)
            return instance_data

        except Exception as err:
            _LOGGER.error("Error loading instance data for %s: %s", entry_id, err)
            return None

    async def async_save_instance_data(
        self,
        entry_id: str,
        entity_manager: EntityManager,
    ) -> None:
        """Save instance data to storage."""
        try:
            # Prepare the data for saving
            entity_data = entity_manager.to_dict()
            entity_types = entity_manager.coordinator.entity_types.to_dict()

            instance_data = {
                "name": self._coordinator.config.name,
                "probability": self._coordinator.probability,
                "prior": self._coordinator.prior,
                "threshold": self._coordinator.threshold,
                "entities": entity_data.get("entities", {}),
                "entity_types": entity_types.get("entity_types", {}),
            }

            # Load existing data
            data = await self.async_load() or self.create_empty_storage()

            if "instances" not in data:
                data["instances"] = {}

            # Update storage with new data
            data["instances"][entry_id] = instance_data
            await self.async_save(data)

            entity_count = len(instance_data.get("entities", {}))
            _LOGGER.debug(
                "Storage save completed for instance %s with %d entities",
                entry_id,
                entity_count,
            )

        except Exception as err:
            _LOGGER.error("Error saving instance data for %s: %s", entry_id, err)
            raise HomeAssistantError(f"Failed to save instance data: {err}") from err

    async def async_load_with_compatibility_check(
        self, entry_id: str, config_entry_version: int
    ) -> tuple[dict[str, Any] | None, bool]:
        """Load instance data with compatibility checking."""
        was_reset = False

        # Check for old config entry version
        if config_entry_version < 9:
            _LOGGER.info(
                "Config entry version %s is older than 9, resetting storage",
                config_entry_version,
            )
            await self.async_remove_instance(entry_id)
            return None, True

        try:
            loaded_data = await self.async_load_instance_data(entry_id)

            if loaded_data is None:
                _LOGGER.info(
                    "No stored data found for instance %s, will initialize with defaults",
                    entry_id,
                )
                return None, False

            # Basic format validation
            if not isinstance(loaded_data, dict) or "entities" not in loaded_data:
                _LOGGER.warning(
                    "Invalid storage format for instance %s, resetting",
                    entry_id,
                )
                await self.async_remove_instance(entry_id)
                return None, True

            _LOGGER.debug(
                "Successfully loaded storage data for instance %s",
                entry_id,
            )
            return loaded_data, False

        except Exception as err:
            _LOGGER.warning(
                "Storage error for instance %s, initializing with defaults: %s",
                entry_id,
                err,
            )
            return None, False
