"""Binary sensor entities for Area Occupancy Detection."""

from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    NAME_BINARY_SENSOR,
)
from .coordinator import AreaOccupancyCoordinator


class AreaOccupancyBinarySensor(
    CoordinatorEntity[AreaOccupancyCoordinator], BinarySensorEntity
):
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
        self._attr_device_class = "occupancy"
        self._attr_device_info = coordinator.device_info

    @property
    def is_on(self) -> bool:
        """Return the state of the sensor."""
        if not self.coordinator.data:
            return False
        return self.coordinator.data.is_occupied


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Area Occupancy Detection binary sensors."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][config_entry.entry_id][
        "coordinator"
    ]
    async_add_entities(
        [
            AreaOccupancyBinarySensor(
                coordinator=coordinator,
                entry_id=config_entry.entry_id,
            ),
        ],
        update_before_add=True,
    )
