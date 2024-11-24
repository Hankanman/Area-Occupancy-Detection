"""Base classes for Room Occupancy sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.components.sensor import SensorEntity
from homeassistant.const import PERCENTAGE
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTR_ACTIVE_TRIGGERS,
    ATTR_CONFIDENCE_SCORE,
    ATTR_DECAY_STATUS,
    ATTR_PRIOR_PROBABILITY,
    ATTR_PROBABILITY,
    ATTR_SENSOR_AVAILABILITY,
    ATTR_SENSOR_PROBABILITIES,
    ProbabilityResult,
)
from .coordinator import RoomOccupancyCoordinator


class RoomOccupancySensorBase(CoordinatorEntity, ABC):
    """Base class for room occupancy sensors."""

    def __init__(
        self,
        coordinator: RoomOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the base sensor."""
        super().__init__(coordinator)
        self._attr_has_entity_name = True
        self._entry_id = entry_id

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return common attributes for both sensor types."""
        if self.coordinator.data is None:
            return {}

        data: ProbabilityResult = self.coordinator.data

        def round_percentage(value: float) -> float:
            """Round percentage values to 2 decimal places."""
            return round(value * 100, 2)

        return {
            ATTR_PROBABILITY: round_percentage(data.get("probability", 0.0)),
            ATTR_PRIOR_PROBABILITY: round_percentage(
                data.get("prior_probability", 0.0)
            ),
            ATTR_ACTIVE_TRIGGERS: data.get("active_triggers", []),
            ATTR_SENSOR_PROBABILITIES: {
                k: round_percentage(v)
                for k, v in data.get("sensor_probabilities", {}).items()
            },
            ATTR_DECAY_STATUS: {
                k: round(v, 2) for k, v in data.get("decay_status", {}).items()
            },
            ATTR_CONFIDENCE_SCORE: round_percentage(data.get("confidence_score", 0.0)),
            ATTR_SENSOR_AVAILABILITY: data.get("sensor_availability", {}),
        }

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return (
            self.coordinator.last_update_success and self.coordinator.data is not None
        )

    @abstractmethod
    def get_value(self) -> StateType:
        """Get the current value of the sensor."""


class RoomOccupancyBinarySensor(RoomOccupancySensorBase, BinarySensorEntity):
    """Binary sensor for room occupancy."""

    def __init__(
        self,
        coordinator: RoomOccupancyCoordinator,
        entry_id: str,
        threshold: float,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_unique_id = f"{entry_id}_occupancy"
        self._threshold = threshold

    @property
    def is_on(self) -> bool | None:
        """Return true if the room is occupied."""
        return self.get_value()

    def get_value(self) -> bool | None:
        """Get the current value of the sensor."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get("probability", 0.0) >= self._threshold


class RoomOccupancyProbabilitySensor(RoomOccupancySensorBase, SensorEntity):
    """Probability sensor for room occupancy."""

    def __init__(
        self,
        coordinator: RoomOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the probability sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_unique_id = f"{entry_id}_probability"
        self._attr_native_unit_of_measurement = PERCENTAGE

    @property
    def native_value(self) -> StateType:
        """Return the native value of the sensor."""
        return self.get_value()

    def get_value(self) -> StateType:
        """Get the current value of the sensor."""
        if self.coordinator.data is None:
            return None
        return round(self.coordinator.data.get("probability", 0.0) * 100, 2)
