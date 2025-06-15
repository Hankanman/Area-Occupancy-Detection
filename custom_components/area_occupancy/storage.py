"""Storage manager for Area Occupancy Detection."""

import asyncio
import contextlib
from datetime import datetime, timedelta
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import CALLBACK_TYPE
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_point_in_time
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_VERSION, CONF_VERSION_MINOR, DOMAIN, STORAGE_KEY
from .data.entity import EntityManager

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Storage optimization constants
DEBOUNCE_DELAY = 30  # seconds - minimum time between writes
PERIODIC_SAVE_INTERVAL = 300  # seconds - force save interval (5 minutes)
MAX_PENDING_SAVES = (
    10  # maximum number of pending save requests before forcing immediate save
)


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
        self._initialized = False
        self._lock = asyncio.Lock()  # Prevent concurrent access issues
        self._data = self.create_empty_storage()  # Initialize data

        # Storage optimization attributes
        self._last_save_time: datetime | None = None
        self._last_data_hash: str | None = None
        self._pending_save_task: asyncio.Task | None = None
        self._periodic_save_tracker: CALLBACK_TYPE | None = None
        self._pending_save_count = 0
        self._dirty = False

    def create_empty_storage(self) -> dict[str, Any]:
        """Create empty storage."""
        return {
            "version": CONF_VERSION,
            "minor_version": CONF_VERSION_MINOR,
            "data": {"instances": {}},
        }

    async def async_initialize(self) -> None:
        """Initialize the storage manager and perform cleanup."""
        if self._initialized:
            return

        await self._async_perform_cleanup()
        self._start_periodic_save_timer()
        self._initialized = True

    async def async_shutdown(self) -> None:
        """Shutdown the storage manager and clean up resources."""
        # Cancel pending save task
        if self._pending_save_task and not self._pending_save_task.done():
            self._pending_save_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_save_task

        # Cancel periodic save timer
        if self._periodic_save_tracker:
            self._periodic_save_tracker()
            self._periodic_save_tracker = None

        # Perform final save if there's dirty data
        if self._dirty:
            _LOGGER.debug("Performing final save during shutdown")
            await self._force_save_now()

    def _start_periodic_save_timer(self) -> None:
        """Start the periodic save timer."""
        if self._periodic_save_tracker:
            return

        next_save = dt_util.utcnow() + timedelta(seconds=PERIODIC_SAVE_INTERVAL)
        self._periodic_save_tracker = async_track_point_in_time(
            self.hass, self._handle_periodic_save, next_save
        )
        _LOGGER.debug(
            "Started periodic save timer, next save at %s", next_save.isoformat()
        )

    async def _handle_periodic_save(self, _now: datetime) -> None:
        """Handle periodic save timer callback."""
        self._periodic_save_tracker = None

        if self._dirty:
            _LOGGER.debug("Periodic save triggered - saving dirty data")
            await self._force_save_now()

        # Restart the timer
        self._start_periodic_save_timer()

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

            instance_data = instances.get(entry_id)
            if instance_data and "entity_types" in instance_data:
                # Restore entity_types into the coordinator
                from .data.entity_type import EntityTypeManager

                self._coordinator.entity_types = EntityTypeManager.from_dict(
                    {"entity_types": instance_data["entity_types"]}, self._coordinator
                )

        except HomeAssistantError as err:
            _LOGGER.error("Error loading instance data for %s: %s", entry_id, err)
            return None

        else:
            return instance_data

    def _calculate_data_hash(self, data: dict[str, Any]) -> str:
        """Calculate a hash of the data to detect changes."""
        # Create a stable string representation of the data
        # Sort keys to ensure consistent ordering
        data_str = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _should_save_immediately(self) -> bool:
        """Determine if we should save immediately regardless of debouncing."""
        # Save immediately if we have too many pending save requests
        if self._pending_save_count >= MAX_PENDING_SAVES:
            return True

        # Save immediately if too much time has passed since last save
        if self._last_save_time:
            time_since_last_save = dt_util.utcnow() - self._last_save_time
            if time_since_last_save.total_seconds() > PERIODIC_SAVE_INTERVAL:
                return True

        return False

    async def async_save_instance_data(
        self,
        entry_id: str,
        entity_manager: EntityManager,
        force: bool = False,
    ) -> None:
        """Save instance data to storage with optimized write frequency.

        Args:
            entry_id: The config entry ID
            entity_manager: The entity manager to save data from
            force: If True, bypass debouncing and save immediately

        """
        try:
            # Prepare the data for saving
            entity_data = entity_manager.to_dict()
            entity_types = entity_manager.coordinator.entity_types.to_dict()

            new_instance_data = {
                "name": self._coordinator.config.name,
                "probability": self._coordinator.probability,
                "prior": self._coordinator.prior,
                "threshold": self._coordinator.threshold,
                "entities": entity_data.get("entities", {}),
                "entity_types": entity_types.get("entity_types", {}),
            }

            # Calculate data hash to detect changes
            data_hash = self._calculate_data_hash(new_instance_data)

            # Skip save if data hasn't changed
            if self._last_data_hash == data_hash and not force:
                _LOGGER.debug("Data unchanged, skipping save for instance %s", entry_id)
                return

            # Mark as dirty
            self._dirty = True
            self._pending_save_count += 1

            # Check if we should save immediately
            should_save_now = (
                force
                or self._should_save_immediately()
                or self._last_save_time is None  # First save
            )

            if should_save_now:
                _LOGGER.debug("Saving immediately for instance %s", entry_id)
                await self._perform_save(entry_id, new_instance_data, data_hash)
            else:
                _LOGGER.debug("Scheduling debounced save for instance %s", entry_id)
                await self._schedule_debounced_save(
                    entry_id, new_instance_data, data_hash
                )

        except HomeAssistantError as err:
            _LOGGER.error("Error preparing save for instance %s: %s", entry_id, err)
            raise

    async def _schedule_debounced_save(
        self, entry_id: str, instance_data: dict[str, Any], data_hash: str
    ) -> None:
        """Schedule a debounced save operation."""
        # Cancel existing pending save task
        if self._pending_save_task and not self._pending_save_task.done():
            self._pending_save_task.cancel()

        # Schedule new save task
        self._pending_save_task = asyncio.create_task(
            self._debounced_save_worker(entry_id, instance_data, data_hash)
        )

    async def _debounced_save_worker(
        self, entry_id: str, instance_data: dict[str, Any], data_hash: str
    ) -> None:
        """Worker task that performs debounced save after delay."""
        try:
            # Wait for debounce delay
            await asyncio.sleep(DEBOUNCE_DELAY)

            # Perform the actual save
            await self._perform_save(entry_id, instance_data, data_hash)

        except asyncio.CancelledError:
            _LOGGER.debug("Debounced save cancelled for instance %s", entry_id)
            raise
        except (HomeAssistantError, OSError, ValueError, TypeError) as err:
            _LOGGER.error("Error in debounced save worker: %s", err)

    async def _perform_save(
        self, entry_id: str, instance_data: dict[str, Any], data_hash: str
    ) -> None:
        """Perform the actual save operation."""
        async with self._lock:
            try:
                _LOGGER.debug("Performing storage save for instance %s", entry_id)

                data = await self.async_load()
                if not data:
                    data = self.create_empty_storage()

                # Validate storage structure
                if not isinstance(data, dict):
                    data = self.create_empty_storage()
                if "instances" not in data:
                    data["instances"] = {}

                # Update storage with new data
                data["instances"][entry_id] = instance_data
                await self.async_save(data)

                # Update tracking variables
                self._last_save_time = dt_util.utcnow()
                self._last_data_hash = data_hash
                self._dirty = False
                self._pending_save_count = 0

                entity_count = len(instance_data.get("entities", {}))
                _LOGGER.debug(
                    "Storage save completed for instance %s with %d entities",
                    entry_id,
                    entity_count,
                )

            except HomeAssistantError as err:
                _LOGGER.error("Error saving instance data for %s: %s", entry_id, err)
                raise HomeAssistantError(
                    f"Failed to save instance data: {err}"
                ) from err

    async def _force_save_now(self) -> None:
        """Force an immediate save of any pending data."""
        if self._pending_save_task and not self._pending_save_task.done():
            # Cancel debounced save and let it complete immediately
            self._pending_save_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_save_task

        # If still dirty, we need to trigger another save
        # This should only happen in edge cases
        if self._dirty:
            _LOGGER.warning("Forcing save with dirty data remaining")

    async def async_reset(self) -> None:
        """Reset storage data."""
        try:
            await self.async_save({})
            self._last_data_hash = None
            self._dirty = False
            self._pending_save_count = 0
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

        # Validate entity_types structure if present
        if "entity_types" in data:
            entity_types_data = data["entity_types"]
            if not isinstance(entity_types_data, dict):
                _LOGGER.debug("Storage 'entity_types' data is not a dictionary")
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
