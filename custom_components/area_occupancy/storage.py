"""Storage handling for Area Occupancy Detection."""

import logging
from typing import Any, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_VERSION, CONF_VERSION_MINOR, STORAGE_KEY
from .exceptions import StorageLoadError, StorageSaveError
from .types import PriorState

_LOGGER = logging.getLogger(__name__)


class InstanceData(TypedDict):
    """TypedDict for stored instance data."""

    name: str | None
    prior_state: dict[str, Any] | None
    last_updated: str


class StoredData(TypedDict):
    """TypedDict for stored data structure."""

    instances: dict[str, InstanceData]


class AreaOccupancyStore(Store[StoredData]):
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

    async def async_remove_instance(self, entry_id: str) -> bool:
        """Remove data for a specific instance ID from storage.

        Args:
            entry_id: The config entry ID of the instance to remove.

        Returns:
            True if data was removed, False otherwise.

        """
        try:
            stored_data = await self.async_load()
            if (
                stored_data
                and "instances" in stored_data
                and entry_id in stored_data["instances"]
            ):
                _LOGGER.debug("Removing instance %s data from storage", entry_id)
                # Create a copy to modify
                modified_data = stored_data.copy()
                modified_data["instances"] = modified_data["instances"].copy()

                del modified_data["instances"][entry_id]
                await self.async_save(modified_data)
                _LOGGER.info("Successfully removed instance %s from storage", entry_id)
                return True
            _LOGGER.debug(
                "Instance %s not found in storage, skipping removal", entry_id
            )
        except Exception:
            _LOGGER.exception(
                "Error removing instance %s from storage",
                entry_id,
            )
            return False  # Don't re-raise, allow flow to continue
        else:
            return False

    async def async_cleanup_orphaned_instances(
        self, active_entry_ids: set[str]
    ) -> bool:
        """Remove data for instances not present in the active_entry_ids set.

        Args:
            active_entry_ids: A set of currently active config entry IDs.

        Returns:
            True if any orphaned data was removed, False otherwise.

        """
        removed_any = False
        try:
            stored_data = await self.async_load()
            if stored_data and "instances" in stored_data:
                stored_entry_ids = set(stored_data["instances"].keys())
                orphaned_ids = stored_entry_ids - active_entry_ids

                if orphaned_ids:
                    _LOGGER.info(
                        "Found orphaned instance(s) in storage: %s", orphaned_ids
                    )
                    # Create a copy to modify
                    modified_data = stored_data.copy()
                    modified_data["instances"] = modified_data["instances"].copy()

                    for entry_id in orphaned_ids:
                        if entry_id in modified_data["instances"]:
                            del modified_data["instances"][entry_id]
                            _LOGGER.debug(
                                "Removed orphaned instance %s from storage data",
                                entry_id,
                            )
                            removed_any = True

                    if removed_any:
                        await self.async_save(modified_data)
                        _LOGGER.info(
                            "Saved cleaned storage data after removing %d orphan(s)",
                            len(orphaned_ids),
                        )
                else:
                    _LOGGER.debug("No orphaned instances found in storage")
            else:
                _LOGGER.debug(
                    "No storage data found or 'instances' key missing, skipping cleanup"
                )

        except Exception:
            _LOGGER.exception("Error during storage cleanup")
            # Don't re-raise, allow setup to continue

        return removed_any

    async def async_load_instance_prior_state(
        self, entry_id: str
    ) -> tuple[str | None, PriorState | None, str | None]:
        """Load prior state data for a specific instance from storage.

        Args:
            entry_id: The config entry ID of the instance to load.

        Returns:
            Tuple of (name, prior_state, last_updated) where prior_state and last_updated
            may be None if not found or data is invalid.

        Raises:
            StorageLoadError: If loading fails

        """
        try:
            data = await self.async_load()  # Use the store's own load method
            if not data or "instances" not in data:
                _LOGGER.debug("No stored data or instances found for loading priors")
                return None, None, None

            instance_data = data["instances"].get(entry_id)
            if not instance_data:
                _LOGGER.debug("No instance data found for %s", entry_id)
                return None, None, None

            name = instance_data.get("name")
            stored_prior_state_dict = instance_data.get("prior_state")
            last_updated = instance_data.get("last_updated")

            if not stored_prior_state_dict:
                _LOGGER.debug("No prior_state dict found for %s", entry_id)
                return name, None, last_updated

            prior_state = PriorState.from_dict(stored_prior_state_dict)
            _LOGGER.debug(
                "Loaded prior state for %s: name=%s, last_updated=%s",
                entry_id,
                name,
                last_updated,
            )
            return name, prior_state, last_updated

        except Exception as err:
            _LOGGER.exception("Error loading prior state for %s", entry_id)
            raise StorageLoadError(f"Failed to load prior state: {err}") from err

    async def async_save_instance_prior_state(
        self, entry_id: str, name: str, prior_state: PriorState
    ) -> None:
        """Save prior state data for a specific instance to storage.

        Args:
            entry_id: The config entry ID of the instance to save.
            name: The name of the area.
            prior_state: The prior state to save.

        """
        try:
            # Load existing data first using the store's own load method
            existing_data = await self.async_load()
            if (
                not existing_data
            ):  # Should not happen if load creates empty, but defensive check
                existing_data = self.create_empty_storage()
            if "instances" not in existing_data:  # Ensure instances dict exists
                existing_data["instances"] = {}

            # Create instance data for this instance
            instance_data = InstanceData(
                name=name,
                prior_state=prior_state.to_dict(),
                last_updated=dt_util.utcnow().isoformat(),
            )

            # Update only this instance's data while preserving others
            existing_data["instances"][entry_id] = instance_data

            # Log all instances being saved
            _LOGGER.debug(
                "Saving prior state for instance %s (total instances: %d). All instances: %s",
                entry_id,
                len(existing_data["instances"]),
                list(existing_data["instances"].keys()),
            )

            # Save the modified data using the store's save method
            await self.async_save(existing_data)
            _LOGGER.debug(
                "Saved prior state for instance %s (total instances: %d)",
                entry_id,
                len(existing_data["instances"]),
            )

        except Exception as err:  # Catch broader exceptions during save
            _LOGGER.exception("Error saving prior state for %s", entry_id)
            raise StorageSaveError(f"Failed to save prior state: {err}") from err
