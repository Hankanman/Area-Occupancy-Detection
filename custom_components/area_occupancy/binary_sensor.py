"""Binary sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any, Final

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.const import PERCENTAGE
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    NAME_BINARY_SENSOR,
    CONF_AREA_ID,
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
)
from .coordinator import AreaOccupancyCoordinator
from .helpers import get_device_info, get_sensor_attributes

_LOGGER = logging.getLogger(__name__)
ROUNDING_PRECISION: Final = 2


class AreaOccupancyBinarySensor(
    CoordinatorEntity[AreaOccupancyCoordinator], BinarySensorEntity
):
    """Binary sensor for area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        threshold: float,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator)

        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._attr_name = NAME_BINARY_SENSOR
        self._attr_unique_id = (
            f"{DOMAIN}_{coordinator.core_config[CONF_AREA_ID]}_occupancy"
        )
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._threshold = threshold
        self._area_name = coordinator.core_config["name"]
        self._attr_entity_category = None
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_device_info = get_device_info(entry_id, self._area_name)

    @property
    def is_on(self) -> bool:
        """Return true if the area is occupied."""
        try:
            if not self.coordinator.data:
                return False
            return self.coordinator.data.get("is_occupied", False)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error determining occupancy state: %s", err)
            return False

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
    threshold = coordinator.options_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)

    async_add_entities(
        [
            AreaOccupancyBinarySensor(
                coordinator=coordinator,
                entry_id=entry.entry_id,
                threshold=threshold,
            )
        ],
        False,
    )
