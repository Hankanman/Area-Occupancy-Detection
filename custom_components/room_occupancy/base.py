"""Base classes for Room Occupancy sensors with shared attribute logic and enhanced features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorDeviceClass,
)
from homeassistant.components.sensor import (
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import CONF_NAME, PERCENTAGE
from homeassistant.helpers.typing import StateType
from homeassistant.core import callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTR_ACTIVE_TRIGGERS,
    ATTR_CONFIDENCE_SCORE,
    ATTR_DECAY_STATUS,
    ATTR_PRIOR_PROBABILITY,
    ATTR_PROBABILITY,
    ATTR_SENSOR_AVAILABILITY,
    ATTR_SENSOR_PROBABILITIES,
    ATTR_LAST_OCCUPIED,
    ATTR_STATE_DURATION,
    ATTR_OCCUPANCY_RATE,
    ATTR_MOVING_AVERAGE,
    ATTR_RATE_OF_CHANGE,
    ATTR_MIN_PROBABILITY,
    ATTR_MAX_PROBABILITY,
    ATTR_THRESHOLD,
    ATTR_WINDOW_SIZE,
    NAME_BINARY_SENSOR,
    NAME_PROBABILITY_SENSOR,
    ProbabilityResult,
)
from .coordinator import RoomOccupancyCoordinator


class RoomOccupancySensorBase(CoordinatorEntity[ProbabilityResult], ABC):
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
        self._room_name = coordinator.config[CONF_NAME]

    @staticmethod
    def _format_float(value: float) -> float:
        """Format float to consistently show 2 decimal places."""
        return round(value, 2)

    @callback
    def _format_unique_id(self, sensor_type: str) -> str:
        """Format the unique id consistently."""
        room_id = self._room_name.lower().replace(" ", "_")
        return f"{room_id}_{sensor_type}"

    @property
    def name(self) -> str:
        """Return the display name of the sensor."""
        sensor_name = (
            NAME_BINARY_SENSOR
            if isinstance(self, RoomOccupancyBinarySensor)
            else NAME_PROBABILITY_SENSOR
        )
        return f"{self._room_name} {sensor_name}"

    @property
    def _shared_attributes(self) -> dict[str, Any]:
        """Return attributes common to all room occupancy sensors."""
        if self.coordinator.data is None:
            return {}

        data: ProbabilityResult = self.coordinator.data

        def format_percentage(value: float) -> float:
            """Format percentage values consistently."""
            return self._format_float(value * 100)

        return {
            ATTR_PROBABILITY: format_percentage(data.get("probability", 0.0)),
            ATTR_PRIOR_PROBABILITY: format_percentage(
                data.get("prior_probability", 0.0)
            ),
            ATTR_ACTIVE_TRIGGERS: data.get("active_triggers", []),
            ATTR_SENSOR_PROBABILITIES: {
                k: format_percentage(v)
                for k, v in data.get("sensor_probabilities", {}).items()
            },
            ATTR_DECAY_STATUS: {
                k: self._format_float(v)
                for k, v in data.get("decay_status", {}).items()
            },
            ATTR_CONFIDENCE_SCORE: format_percentage(data.get("confidence_score", 0.0)),
            ATTR_SENSOR_AVAILABILITY: data.get("sensor_availability", {}),
        }

    @abstractmethod
    def _sensor_specific_attributes(self) -> dict[str, Any]:
        """Return attributes specific to this sensor type."""

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        attributes = self._shared_attributes
        specific_attributes = self._sensor_specific_attributes()
        if specific_attributes:
            attributes.update(specific_attributes)
        return attributes

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
        self._attr_unique_id = self._format_unique_id("occupancy")
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._threshold = threshold

    def _sensor_specific_attributes(self) -> dict[str, Any]:
        """Return attributes specific to binary occupancy sensor."""
        data = self.coordinator.data
        if not data:
            return {}

        return {
            ATTR_THRESHOLD: self._format_float(self._threshold),
            ATTR_LAST_OCCUPIED: data.get("last_occupied"),
            ATTR_STATE_DURATION: self._format_float(
                data.get("state_duration", 0.0) / 60
            ),  # Convert to minutes
            ATTR_OCCUPANCY_RATE: self._format_float(
                data.get("occupancy_rate", 0.0) * 100
            ),  # Convert to percentage
        }

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
        self._attr_unique_id = self._format_unique_id("probability")
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    def _sensor_specific_attributes(self) -> dict[str, Any]:
        """Return attributes specific to probability sensor."""
        data = self.coordinator.data
        if not data:
            return {}

        return {
            ATTR_MOVING_AVERAGE: self._format_float(
                data.get("moving_average", 0.0) * 100
            ),
            ATTR_RATE_OF_CHANGE: self._format_float(
                data.get("rate_of_change", 0.0) * 100
            ),
            ATTR_MIN_PROBABILITY: self._format_float(
                data.get("min_probability", 0.0) * 100
            ),
            ATTR_MAX_PROBABILITY: self._format_float(
                data.get("max_probability", 0.0) * 100
            ),
            ATTR_WINDOW_SIZE: "1 hour",  # Current window size for moving average
        }

    @property
    def native_value(self) -> StateType:
        """Return the native value of the sensor."""
        return self.get_value()

    def get_value(self) -> StateType:
        """Get the current value of the sensor."""
        if self.coordinator.data is None:
            return None
        return self._format_float(self.coordinator.data.get("probability", 0.0) * 100)
