# custom_components/area_occupancy/sensors/binary.py

"""Binary sensor for Area Occupancy Detection."""

from __future__ import annotations
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorDeviceClass,
)
from homeassistant.helpers.restore_state import RestoreEntity

from ..core.coordinator import AreaOccupancyCoordinator

from .base import AreaOccupancyBaseSensor
from ..core import ProbabilityResult


class AreaOccupancyBinarySensor(
    AreaOccupancyBaseSensor, BinarySensorEntity, RestoreEntity
):
    """Binary sensor indicating area occupancy status."""

    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        threshold: float,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator, entry_id)
        self._threshold = threshold

    @property
    def name_suffix(self) -> str:
        """Return the name suffix for this sensor."""
        return "Occupancy Status"

    @property
    def is_on(self) -> bool | None:
        """Return true if area is occupied."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data["probability"] >= self._threshold

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return state attributes."""
        attributes = super().extra_state_attributes

        if not self.coordinator.data:
            return attributes

        data: ProbabilityResult = self.coordinator.data

        # Add binary sensor specific attributes
        attributes.update(
            {
                "threshold": self._format_percentage(self._threshold),
                "last_occupied": data.get("last_occupied"),
                "state_duration": self._format_duration(data.get("state_duration", 0)),
                "occupancy_rate": self._format_percentage(
                    data.get("occupancy_rate", 0)
                ),
            }
        )

        return attributes

    def update_threshold(self, threshold: float) -> None:
        """Update the occupancy threshold."""
        self._threshold = threshold
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()

        # Restore previous state if available
        last_state = await self.async_get_last_state()
        if last_state:
            self._attr_is_on = last_state.state == "on"
