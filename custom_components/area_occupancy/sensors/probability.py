# custom_components/area_occupancy/sensors/probability.py

"""Probability sensor for Area Occupancy Detection."""

from __future__ import annotations
from typing import Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE
from homeassistant.helpers.restore_state import RestoreEntity

from .base import AreaOccupancyBaseSensor
from ..core import ProbabilityResult


class AreaOccupancyProbabilitySensor(
    AreaOccupancyBaseSensor, SensorEntity, RestoreEntity
):
    """Sensor for area occupancy probability."""

    _attr_device_class = SensorDeviceClass.POWER_FACTOR
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name_suffix(self) -> str:
        """Return the name suffix for this sensor."""
        return "Occupancy Probability"

    @property
    def native_value(self) -> float | None:
        """Return the current probability as a percentage."""
        return self._get_native_value()

    def _get_native_value(self) -> float | None:
        """Get the native value for this sensor."""
        if not self.coordinator.data:
            return None
        return self._format_percentage(self.coordinator.data["probability"])

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return state attributes."""
        attributes = super().extra_state_attributes

        if not self.coordinator.data:
            return attributes

        data: ProbabilityResult = self.coordinator.data

        # Add probability sensor specific attributes
        attributes.update(
            {
                "moving_average": self._format_percentage(
                    data.get("moving_average", 0)
                ),
                "rate_of_change": self._format_percentage(
                    data.get("rate_of_change", 0)
                ),
                "min_probability": self._format_percentage(
                    data.get("min_probability", 0)
                ),
                "max_probability": self._format_percentage(
                    data.get("max_probability", 0)
                ),
                "pattern_details": data.get("pattern_data", {}),
            }
        )

        return attributes

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()

        # Restore previous state if available
        last_state = await self.async_get_last_state()
        if last_state and last_state.state.replace(".", "", 1).isdigit():
            self._attr_native_value = float(last_state.state)
