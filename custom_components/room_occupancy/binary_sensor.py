"""Binary sensor platform for Room Occupancy Detection integration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    NAME_BINARY_SENSOR,
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
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
class RoomOccupancyBinaryEntityDescription(BinarySensorEntityDescription):
    """Class describing Room Occupancy binary sensor entities."""


class RoomOccupancyBinarySensor(CoordinatorEntity, BinarySensorEntity):
    """Representation of a Room Occupancy binary sensor."""

    entity_description: RoomOccupancyBinaryEntityDescription
    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY

    def __init__(
        self,
        coordinator: RoomOccupancyCoordinator,
        entry_id: str,
        description: RoomOccupancyBinaryEntityDescription,
        threshold: float,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry_id}_occupancy"
        self.entity_description = description
        self._attr_name = NAME_BINARY_SENSOR
        self._threshold = threshold

    @property
    def is_on(self) -> bool | None:
        """Return true if the room is occupied."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get("probability", 0.0) >= self._threshold

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional sensor state attributes."""
        if self.coordinator.data is None:
            return {}

        return {
            ATTR_PROBABILITY: self.coordinator.data.get("probability", 0.0),
            ATTR_PRIOR_PROBABILITY: self.coordinator.data.get("prior_probability", 0.0),
            ATTR_ACTIVE_TRIGGERS: self.coordinator.data.get("active_triggers", []),
            ATTR_SENSOR_PROBABILITIES: self.coordinator.data.get(
                "sensor_probabilities", {}
            ),
            ATTR_DECAY_STATUS: self.coordinator.data.get("decay_status", {}),
            ATTR_CONFIDENCE_SCORE: self.coordinator.data.get("confidence_score", 0.0),
            ATTR_SENSOR_AVAILABILITY: self.coordinator.data.get(
                "sensor_availability", {}
            ),
        }

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return (
            self.coordinator.last_update_success and self.coordinator.data is not None
        )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Room Occupancy binary sensor based on a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    threshold = entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)

    async_add_entities(
        [
            RoomOccupancyBinarySensor(
                coordinator,
                entry.entry_id,
                RoomOccupancyBinaryEntityDescription(),
                threshold,
            )
        ]
    )
