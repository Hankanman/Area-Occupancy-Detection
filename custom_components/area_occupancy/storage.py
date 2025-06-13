"""Storage manager for Area Occupancy Detection."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .data.entity import EntityManager

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.storage"


class StorageManager(Store[dict[str, Any]]):
    """Manage storage for Area Occupancy Detection."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
    ) -> None:
        """Initialize the storage manager."""
        super().__init__(
            coordinator.hass,
            STORAGE_VERSION,
            STORAGE_KEY,
        )
        self._coordinator = coordinator
        self._initialized = False
        self._lock = asyncio.Lock()  # Prevent concurrent access issues

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
        async with self._lock:  # Prevent concurrent modifications
            try:
                data = await self.async_load()
                if not data:
                    return False

                instances = data.get("instances", {})
                if not isinstance(instances, dict):
                    return False

                if entry_id in instances:
                    del instances[entry_id]
                    await self.async_save(data)
                    return True

            except HomeAssistantError as err:
                _LOGGER.error("Error removing instance %s: %s", entry_id, err)
                raise HomeAssistantError(f"Failed to remove instance: {err}") from err

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

        except HomeAssistantError as err:
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

            # Validate storage structure
            if not isinstance(data, dict) or "instances" not in data:
                _LOGGER.warning("Invalid storage structure, creating empty storage")
                return None

            instances = data["instances"]
            if not isinstance(instances, dict):
                _LOGGER.warning("Invalid instances structure in storage")
                return None

            return instances.get(entry_id)

        except HomeAssistantError as err:
            _LOGGER.error("Error loading instance data for %s: %s", entry_id, err)
            return None

    async def async_save_instance_data(
        self,
        entry_id: str,
        entity_manager: EntityManager,
    ) -> None:
        """Save instance data to storage."""
        async with self._lock:  # Prevent concurrent modifications
            try:
                # Check if we're in dev mode for enhanced logging
                dev_mode = self._coordinator.dev_mode

                if dev_mode:
                    _LOGGER.debug("Starting storage save for instance %s", entry_id)

                data = await self.async_load()
                if not data:
                    data = self.create_empty_storage()

                # Validate storage structure
                if not isinstance(data, dict):
                    data = self.create_empty_storage()
                if "instances" not in data:
                    data["instances"] = {}

                # Convert entity manager to dict and flatten structure
                entity_data = entity_manager.to_dict()

                if dev_mode:
                    entity_count = len(entity_data.get("entities", {}))
                    _LOGGER.debug("Saving %d entities to storage", entity_count)

                # Update storage with flattened structure
                data["instances"][entry_id] = {
                    "name": self._coordinator.config.name,
                    "probability": self._coordinator.probability,
                    "prior": self._coordinator.prior,
                    "threshold": self._coordinator.threshold,
                    "entities": entity_data.get("entities", {}),
                }
                await self.async_save(data)

                if dev_mode:
                    _LOGGER.debug("Storage save completed for instance %s", entry_id)

            except HomeAssistantError as err:
                _LOGGER.error("Error saving instance data for %s: %s", entry_id, err)
                raise HomeAssistantError(
                    f"Failed to save instance data: {err}"
                ) from err

    async def async_reset(self) -> None:
        """Reset storage data."""
        try:
            await self.async_save({})
            _LOGGER.info("Storage data reset successfully")
        except HomeAssistantError as err:
            _LOGGER.error("Failed to reset storage data: %s", err)
            raise

    async def async_load_with_compatibility_check(
        self, entry_id: str, config_entry_version: int
    ) -> tuple[dict[str, Any] | None, bool]:
        """Load instance data with automatic compatibility checking and migration.

        This method handles:
        1. Version-based storage reset for incompatible versions
        2. Format detection and reset for corrupted/incompatible data
        3. Automatic fallback to defaults when needed

        Args:
            entry_id: The config entry ID to load data for
            config_entry_version: The version of the config entry

        Returns:
            Tuple of (loaded_data, was_reset):
            - loaded_data: The loaded data dict or None if no data/reset occurred
            - was_reset: True if storage was reset due to compatibility issues

        Raises:
            StorageError: If there's a critical storage error

        """
        was_reset = False

        # Check if we have an old config entry version that needs storage reset
        if config_entry_version < 9:
            _LOGGER.info(
                "Config entry version %s is older than 9, resetting storage for compatibility",
                config_entry_version,
            )
            await self.async_remove_instance(entry_id)
            _LOGGER.info("Storage reset complete for version compatibility")
            return None, True

        # Try to load instance data
        try:
            loaded_data = await self.async_load_instance_data(entry_id)

            if loaded_data is None:
                _LOGGER.info(
                    "No stored data found for instance %s, will initialize with defaults",
                    entry_id,
                )
                return None, False

            # Validate the data format
            if not self._validate_storage_format(loaded_data):
                _LOGGER.warning(
                    "Detected incompatible storage format for instance %s. "
                    "Resetting storage to use new format.",
                    entry_id,
                )
                await self.async_remove_instance(entry_id)
                _LOGGER.info(
                    "Storage reset complete for instance %s, will initialize with defaults",
                    entry_id,
                )
                return None, True

            # Data is valid, return it
            _LOGGER.debug(
                "Successfully loaded compatible storage data for instance %s",
                entry_id,
            )

        except HomeAssistantError as err:
            _LOGGER.warning(
                "Storage error for instance %s, initializing with defaults: %s",
                entry_id,
                err,
            )
            # Don't reset on storage errors, just return None
            return None, False

        except (ValueError, AttributeError, TypeError, KeyError) as err:
            _LOGGER.warning(
                "Data format error for instance %s, resetting storage: %s",
                entry_id,
                err,
            )
            # Reset storage on data format errors
            try:
                await self.async_remove_instance(entry_id)
                _LOGGER.info(
                    "Storage reset due to data format error for instance %s", entry_id
                )
                was_reset = True
            except HomeAssistantError:
                _LOGGER.warning("Failed to reset storage for instance %s", entry_id)

            return None, was_reset

        else:
            return loaded_data, False

    def _validate_storage_format(self, data: dict[str, Any]) -> bool:
        """Validate that the storage data has the expected format.

        Args:
            data: The loaded storage data to validate

        Returns:
            True if the format is valid, False if it needs to be reset

        """
        # Check for required top-level structure
        if not isinstance(data, dict):
            _LOGGER.debug("Storage data is not a dictionary")
            return False

        # Check if it has the new entity-based format
        if "entities" not in data:
            _LOGGER.debug("Storage data missing 'entities' key - old format detected")
            return False

        # Validate entities structure
        entities_data = data["entities"]
        if not isinstance(entities_data, dict):
            _LOGGER.debug("Storage 'entities' data is not a dictionary")
            return False

        # Sample validate a few entities to ensure format compatibility
        for entity_id, entity_data in list(entities_data.items())[:3]:  # Check first 3
            if not self._validate_entity_format(entity_id, entity_data):
                return False

        return True

    def _validate_entity_format(
        self, entity_id: str, entity_data: dict[str, Any]
    ) -> bool:
        """Validate that an individual entity has the expected format.

        Args:
            entity_id: The entity ID
            entity_data: The entity data to validate

        Returns:
            True if the format is valid, False otherwise

        """
        required_keys = {"entity_id", "type", "probability", "prior", "decay"}

        if not isinstance(entity_data, dict):
            _LOGGER.debug("Entity %s data is not a dictionary", entity_id)
            return False

        missing_keys = required_keys - set(entity_data.keys())
        if missing_keys:
            _LOGGER.debug(
                "Entity %s missing required keys: %s", entity_id, missing_keys
            )
            return False

        return True
