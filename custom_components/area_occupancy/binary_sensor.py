"""Binary sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .base import AreaOccupancyBinarySensor
from .const import CONF_THRESHOLD, DEFAULT_THRESHOLD, DOMAIN, NAME_BINARY_SENSOR

_LOGGER = logging.getLogger(__name__)


@dataclass
class AreaOccupancyBinaryEntityDescription(BinarySensorEntityDescription):
    """Class describing Area Occupancy binary sensor entities."""

    def __init__(self, area_name: str) -> None:
        """Initialize the description."""
        super().__init__(
            key="occupancy_status",
            name=f"{area_name} {NAME_BINARY_SENSOR}",
            device_class=BinarySensorDeviceClass.OCCUPANCY,
        )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Area Occupancy binary sensor based on a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    threshold = entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)

    async_add_entities(
        [
            AreaOccupancyBinarySensor(
                coordinator,
                entry.entry_id,
                threshold,
            )
        ]
    )
