"""Storage manager for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_VERSION, CONF_VERSION_MINOR, DOMAIN

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyStorageData(TypedDict, total=False):
    """Typed data structure for area occupancy storage."""

    name: str | None
    purpose: str | None
    probability: float | None
    prior: float | None
    threshold: float | None
    last_updated: str | None
    entities: dict[str, Any]


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
            )

        # Handle migration from various data formats
        if isinstance(old_data, dict):
            return AreaOccupancyStorageData(
                name=old_data.get("name"),
                purpose=old_data.get("purpose"),
                probability=old_data.get("probability"),
                prior=old_data.get("prior"),
                threshold=old_data.get("threshold"),
                last_updated=old_data.get("last_updated"),
                entities=old_data.get("entities", {}),
            )

        # Fallback for unexpected data format
        return AreaOccupancyStorageData(
            entities={},
            entity_types={},
        )

    async def async_save_data(
        self,
        force: bool = False,
    ) -> None:
        """Save coordinator data using debounced storage."""

        entity_manager = self._coordinator.entities

        def get_storage_data() -> AreaOccupancyStorageData:
            entity_data = entity_manager.to_dict()

            return AreaOccupancyStorageData(
                name=self._coordinator.config.name,
                purpose=self._coordinator.config.purpose,
                probability=self._coordinator.probability,
                prior=self._coordinator.prior,
                threshold=self._coordinator.threshold,
                last_updated=dt_util.utcnow().isoformat(),
                entities=entity_data.get("entities", {}),
            )

        # Use native Store debounced saving
        data = get_storage_data()
        if force:
            await self.async_save(data)
        else:
            self.async_delay_save(get_storage_data, delay=30.0)

    async def async_load_data(self) -> AreaOccupancyStorageData | None:
        """Load coordinator data from storage with compatibility checking.

        Returns the full data dict if valid, or None if reset/invalid.
        """
        coordinator = self._coordinator
        try:
            data = await super().async_load()
            if data is None:
                _LOGGER.info(
                    "No stored data found for entry %s, will initialize with defaults",
                    coordinator.entry_id,
                )
                return None

            # Basic format validation
            if not isinstance(data, dict) or "entities" not in data:
                _LOGGER.warning(
                    "Invalid storage format for entry %s, resetting",
                    coordinator.entry_id,
                )
                await self.async_remove()
                return None

            _LOGGER.debug(
                "Successfully loaded storage data for entry %s",
                coordinator.entry_id,
            )

        except (HomeAssistantError, OSError, ValueError) as err:
            _LOGGER.warning(
                "Storage error for entry %s, initializing with defaults: %s",
                coordinator.entry_id,
                err,
            )
            return None
        else:
            return AreaOccupancyStorageData(
                name=data.get("name"),
                purpose=data.get("purpose"),
                probability=data.get("probability"),
                prior=data.get("prior"),
                threshold=data.get("threshold"),
                last_updated=data.get("last_updated"),
                entities=data.get("entities", {}),
            )
