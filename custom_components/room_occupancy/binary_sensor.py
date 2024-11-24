"""Binary sensor platform for Room Occupancy Detection integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import RoomOccupancyCoordinator
from .const import (
    ATTR_ACTIVE_TRIGGERS,
    ATTR_CONFIDENCE_SCORE,
    ATTR_DECAY_STATUS,
    ATTR_PRIOR_PROBABILITY,
    ATTR_PROBABILITY,
    ATTR_SENSOR_AVAILABILITY,
    ATTR_SENSOR_PROBABILITIES,
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
    DOMAIN,
    NAME_BINARY_SENSOR,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class RoomOccupancyBinaryEntityDescription(BinarySensorEntityDescription):
    """Class describing Room Occupancy binary sensor entities."""

    def __init__(self, room_name: str) -> None:
        """Initialize the description."""
        super().__init__(
            key="occupancy_status",
            name=f"{room_name} {NAME_BINARY_SENSOR}",
            device_class=BinarySensorDeviceClass.OCCUPANCY,
        )


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

        def round_percentage(value: float) -> float:
            """Round percentage values to 2 decimal places."""
            return round(value * 100, 2)

        def round_decay_status(decay_dict: dict) -> dict:
            """Round decay status values to 2 decimal places."""
            return {k: round(v, 2) for k, v in decay_dict.items()}

        def round_sensor_probabilities(prob_dict: dict) -> dict:
            """Round sensor probability values to 2 decimal places."""
            return {k: round_percentage(v) for k, v in prob_dict.items()}

        return {
            ATTR_PROBABILITY: round_percentage(
                self.coordinator.data.get("probability", 0.0)
            ),
            ATTR_PRIOR_PROBABILITY: round_percentage(
                self.coordinator.data.get("prior_probability", 0.0)
            ),
            ATTR_ACTIVE_TRIGGERS: self.coordinator.data.get("active_triggers", []),
            ATTR_SENSOR_PROBABILITIES: round_sensor_probabilities(
                self.coordinator.data.get("sensor_probabilities", {})
            ),
            ATTR_DECAY_STATUS: round_decay_status(
                self.coordinator.data.get("decay_status", {})
            ),
            ATTR_CONFIDENCE_SCORE: round_percentage(
                self.coordinator.data.get("confidence_score", 0.0)
            ),
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
    room_name = entry.data[CONF_NAME]

    async_add_entities(
        [
            RoomOccupancyBinarySensor(
                coordinator,
                entry.entry_id,
                RoomOccupancyBinaryEntityDescription(room_name),
                threshold,
            )
        ]
    )
