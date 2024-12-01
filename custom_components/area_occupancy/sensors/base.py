# custom_components/area_occupancy/sensors/base.py

"""Base sensor class for Area Occupancy Detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.core import callback

from ..core.coordinator import AreaOccupancyCoordinator
from ..core import ProbabilityResult


class AreaOccupancyBaseSensor(CoordinatorEntity[ProbabilityResult], ABC):
    """Base class for area occupancy sensors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the base sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._attr_unique_id = self._generate_unique_id()

    @property
    @abstractmethod
    def name_suffix(self) -> str:
        """Return the suffix for the sensor name."""

    @property
    def name(self) -> str:
        """Return the display name of this sensor."""
        return f"{self.coordinator.config.name} {self.name_suffix}"

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return (
            self.coordinator.last_update_success and self.coordinator.data is not None
        )

    @abstractmethod
    def _get_native_value(self) -> Any:
        """Get the native value for this sensor."""

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return common state attributes."""
        if not self.coordinator.data:
            return {}

        data = self.coordinator.data
        return {
            "probability": self._format_percentage(data["probability"]),
            "prior_probability": self._format_percentage(data["prior_probability"]),
            "active_triggers": data["active_triggers"],
            "sensor_probabilities": {
                k: self._format_percentage(v)
                for k, v in data["sensor_probabilities"].items()
            },
            "confidence_score": self._format_percentage(data["confidence_score"]),
            "sensor_availability": data["sensor_availability"],
            "device_states": data["device_states"],
        }

    def _generate_unique_id(self) -> str:
        """Generate a unique ID for this sensor."""
        area_id = self.coordinator.config.name.lower().replace(" ", "_")
        return f"{area_id}_{self.name_suffix.lower().replace(' ', '_')}"

    @staticmethod
    def _format_percentage(value: float) -> float:
        """Format a decimal probability as a percentage."""
        return round(value * 100, 2)

    @staticmethod
    def _format_duration(seconds: float) -> float:
        """Format duration in minutes."""
        return round(seconds / 60, 2)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()
