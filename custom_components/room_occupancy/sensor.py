"""Sensor platform for Room Occupancy Detection integration."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .const import (
    DOMAIN,
    NAME_PROBABILITY_SENSOR,
    ATTR_PROBABILITY,
    ATTR_PRIOR_PROBABILITY,
    ATTR_ACTIVE_TRIGGERS,
    ATTR_SENSOR_PROBABILITIES,
    ATTR_DECAY_STATUS,
    ATTR_CONFIDENCE_SCORE,
    ATTR_SENSOR_AVAILABILITY,
)
from .coordinator import RoomOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

@dataclass
class RoomOccupancyEntityDescription(SensorEntityDescription):
    """Class describing Room Occupancy sensor entities."""

class RoomOccupancyProbabilitySensor(CoordinatorEntity, SensorEntity):
    """Representation of a Room Occupancy Probability sensor."""

    entity_description: RoomOccupancyEntityDescription
    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_device_class = SensorDeviceClass.POWER_FACTOR

    def __init__(
        self,
        coordinator: RoomOccupancyCoordinator,
        entry_id: str,
        description: RoomOccupancyEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry_id}_probability"
        self.entity_description = description
        self._attr_name = NAME_PROBABILITY_SENSOR

    @property
    def native_value(self) -> StateType:
        """Return the probability value."""
        if self.coordinator.data is None:
            return None
        return round(self.coordinator.data.get("probability", 0.0) * 100, 1)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional sensor state attributes."""
        if self.coordinator.data is None:
            return {}

        return {
            ATTR_PROBABILITY: self.coordinator.data.get("probability", 0.0),
            ATTR_PRIOR_PROBABILITY: self.coordinator.data.get("prior_probability", 0.0),
            ATTR_ACTIVE_TRIGGERS: self.coordinator.data.get("active_triggers", []),
            ATTR_SENSOR_PROBABILITIES: self.coordinator.data.get("sensor_probabilities", {}),
            ATTR_DECAY_STATUS: self.coordinator.data.get("decay_status", {}),
            ATTR_CONFIDENCE_SCORE: self.coordinator.data.get("confidence_score", 0.0),
            ATTR_SENSOR_AVAILABILITY: self.coordinator.data.get("sensor_availability", {}),
        }

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
                RoomOccupancyEntityDescription(
                    key="room_occupancy_probability",
                    name=NAME_PROBABILITY_SENSOR,
                ),
            )
        ]
    )
