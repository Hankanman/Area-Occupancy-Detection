"""Sensor platform for Room Occupancy Detection integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .base import RoomOccupancyProbabilitySensor
from .const import DOMAIN, NAME_PROBABILITY_SENSOR

_LOGGER = logging.getLogger(__name__)


@dataclass
class RoomOccupancyEntityDescription(SensorEntityDescription):
    """Class describing Room Occupancy sensor entities."""

    def __init__(self, room_name: str) -> None:
        """Initialize the description."""
        super().__init__(
            key="occupancy_probability",
            name=f"{room_name} {NAME_PROBABILITY_SENSOR}",
            device_class=SensorDeviceClass.POWER_FACTOR,
            native_unit_of_measurement=PERCENTAGE,
        )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Room Occupancy sensor based on a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]

    async_add_entities(
        [
            RoomOccupancyProbabilitySensor(
                coordinator,
                entry.entry_id,
            )
        ]
    )
