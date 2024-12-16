"""Binary sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

from typing import Any, Final

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
    CONF_AREA_ID,
    CONF_NAME,
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
    NAME_BINARY_SENSOR,
)
from .coordinator import AreaOccupancyCoordinator
from .helpers import get_device_info, get_sensor_attributes

ROUNDING_PRECISION: Final = 2


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
        self._attr_unique_id = (
            f"{DOMAIN}_{coordinator.core_config[CONF_AREA_ID]}_occupancy"
        )
        self._attr_name = f"{coordinator.core_config[CONF_NAME]} {NAME_BINARY_SENSOR}"
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._attr_device_info = get_device_info(
            entry_id, coordinator.core_config[CONF_NAME]
        )

    @property
    def is_on(self) -> bool:
        """Return True if the area is currently occupied."""
        threshold = (
            self.coordinator.options_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
            / 100.0
        )
        probability = self.coordinator.data.get("probability", 0.0)
        return probability >= threshold

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        return get_sensor_attributes(self.hass, self.coordinator)


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
