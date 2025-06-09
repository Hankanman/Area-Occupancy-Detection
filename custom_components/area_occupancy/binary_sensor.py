"""Binary sensor entities for Area Occupancy Detection."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .coordinator import AreaOccupancyCoordinator

NAME_BINARY_SENSOR = "Occupancy Status"


class Occupancy(CoordinatorEntity[AreaOccupancyCoordinator], BinarySensorEntity):
    """Binary sensor for the occupancy status."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_has_entity_name = True
        self._attr_unique_id = (
            f"{entry_id}_{NAME_BINARY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_name = NAME_BINARY_SENSOR
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._attr_device_info: DeviceInfo | None = coordinator.device_info

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend."""
        return "mdi:home-account" if self.is_on else "mdi:home-outline"

    @property
    def is_on(self) -> bool:
        """Return true if the area is occupied.

        Returns:
            bool: True if the area is currently occupied based on coordinator data,
                 False if no data is available or area is unoccupied.

        """
        return self.coordinator.is_occupied


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Area Occupancy Detection binary sensors."""
    coordinator: AreaOccupancyCoordinator = config_entry.runtime_data

    # 1. Create the main sensor instance
    entities = [
        Occupancy(
            coordinator=coordinator,
            entry_id=config_entry.entry_id,
        )
    ]

    async_add_entities(entities, update_before_add=True)
