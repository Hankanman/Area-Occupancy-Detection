"""Storage manager for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_VERSION, CONF_VERSION_MINOR, DOMAIN
from .data.entity import EntityManager

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyStorageData(TypedDict, total=False):
    """Typed data structure for area occupancy storage."""

    name: str | None
    probability: float | None
    prior: float | None
    threshold: float | None
    last_updated: str | None
    entities: dict[str, Any]
    entity_types: dict[str, Any]


class AreaOccupancyStore(Store[AreaOccupancyStorageData]):
    """Per-config-entry storage for Area Occupancy Detection."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Initialize the per-entry storage."""
        super().__init__(
            hass=coordinator.hass,
            version=CONF_VERSION,
            key=f"{DOMAIN}.{coordinator.entry_id}",
            atomic_writes=True,  # Enable safe writes
            minor_version=CONF_VERSION_MINOR,
        )
        self._coordinator = coordinator

    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: dict
    ) -> AreaOccupancyStorageData:
        """Migrate storage data to new format."""
        _LOGGER.info(
            "Migrating storage for entry %s from version %d.%d to %d.%d",
            self._coordinator.entry_id,
            old_major_version,
            old_minor_version,
            CONF_VERSION,
            CONF_VERSION_MINOR,
        )

        # For major version changes, start fresh
        if old_major_version < CONF_VERSION:
            _LOGGER.info(
                "Major version change for entry %s, starting with empty storage",
                self._coordinator.entry_id,
            )
            return AreaOccupancyStorageData(
                entities={},
                entity_types={},
            )

        # Handle migration from various data formats
        if isinstance(old_data, dict):
            return AreaOccupancyStorageData(
                name=old_data.get("name"),
                probability=old_data.get("probability"),
                prior=old_data.get("prior"),
                threshold=old_data.get("threshold"),
                last_updated=old_data.get("last_updated"),
                entities=old_data.get("entities", {}),
                entity_types=old_data.get("entity_types", {}),
            )

        # Fallback for unexpected data format
        return AreaOccupancyStorageData(
            entities={},
            entity_types={},
        )

    def async_save_coordinator_data(
        self,
        entity_manager: EntityManager,
    ) -> None:
        """Save coordinator data using debounced storage."""

        def get_storage_data() -> AreaOccupancyStorageData:
            entity_data = entity_manager.to_dict()
            entity_types = entity_manager.coordinator.entity_types.to_dict()

            return AreaOccupancyStorageData(
                name=self._coordinator.config.name,
                probability=self._coordinator.probability,
                prior=self._coordinator.prior,
                threshold=self._coordinator.threshold,
                last_updated=dt_util.utcnow().isoformat(),
                entities=entity_data.get("entities", {}),
                entity_types=entity_types.get("entity_types", {}),
            )

        # Use native Store debounced saving
        self.async_delay_save(get_storage_data, delay=30.0)

        entity_count = len(entity_manager.entities)
        _LOGGER.debug(
            "Scheduled storage save for entry %s with %d entities",
            self._coordinator.entry_id,
            entity_count,
        )

    async def async_load_coordinator_data(self) -> dict[str, Any] | None:
        """Load coordinator data from storage."""
        try:
            data = await self.async_load()
            if data is None:
                _LOGGER.info(
                    "No stored data found for entry %s, will initialize with defaults",
                    self._coordinator.entry_id,
                )
                return None

            # Convert back to dict format for EntityManager.from_dict
            result = {
                "last_updated": data.get("last_updated"),
                "entities": data.get("entities", {}),
                "entity_types": data.get("entity_types", {}),
            }

            _LOGGER.debug(
                "Successfully loaded storage data for entry %s",
                self._coordinator.entry_id,
            )

        except (HomeAssistantError, OSError, ValueError) as err:
            _LOGGER.warning(
                "Storage error for entry %s, initializing with defaults: %s",
                self._coordinator.entry_id,
                err,
            )
            return None
        else:
            return result

    async def async_load_with_compatibility_check(
        self, entry_id: str, config_entry_version: int
    ) -> tuple[dict[str, Any] | None, bool]:
        """Load instance data with compatibility checking."""
        # Check for old config entry version
        if config_entry_version < 9:
            _LOGGER.info(
                "Config entry version %s is older than 9, resetting storage",
                config_entry_version,
            )
            # Remove per-entry storage for old config versions
            await self.async_remove()
            return None, True

        # Load from per-entry storage
        try:
            loaded_data = await self.async_load_coordinator_data()
            if loaded_data is None:
                return None, False

            # Basic format validation
            if not isinstance(loaded_data, dict) or "entities" not in loaded_data:
                _LOGGER.warning(
                    "Invalid storage format for entry %s, resetting",
                    entry_id,
                )
                await self.async_remove()
                return None, True

        except (HomeAssistantError, OSError, ValueError) as err:
            _LOGGER.warning(
                "Storage error for entry %s, initializing with defaults: %s",
                entry_id,
                err,
            )
            return None, False
        else:
            return loaded_data, False
