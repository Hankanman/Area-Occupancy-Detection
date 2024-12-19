"""Binary sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    CONF_NAME,
    NAME_BINARY_SENSOR,
)
from .coordinator import AreaOccupancyCoordinator


class AreaOccupancyBinarySensor(
    CoordinatorEntity[AreaOccupancyCoordinator], BinarySensorEntity
):
    """Binary sensor indicating occupancy status."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator)
        self.coordinator = coordinator
        self.entry_id = entry_id
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_BINARY_SENSOR.lower().replace(' ', '_')}"
        self._attr_name = f"{coordinator.config[CONF_NAME]} {NAME_BINARY_SENSOR}"
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._attr_device_info = coordinator.device_info

    @property
    def is_on(self) -> bool:
        """Return True if the area is currently occupied."""
        threshold = self.coordinator.threshold
        probability = self.coordinator.data.get("probability", 0.0)
        return probability >= threshold


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Area Occupancy binary sensor based on a config entry."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]

    # Create a new binary sensor entity
    binary_sensor = AreaOccupancyBinarySensor(
        coordinator=coordinator,
        entry_id=entry.entry_id,
    )

    async_add_entities([binary_sensor], update_before_add=True)
