"""Sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    NAME_DECAY_SENSOR,
    NAME_PRIORS_SENSOR,
    NAME_PROBABILITY_SENSOR,
    ROUNDING_PRECISION,
)
from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


class AreaOccupancySensorBase(
    CoordinatorEntity[AreaOccupancyCoordinator], SensorEntity
):
    """Base class for area occupancy sensors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._attr_device_info = coordinator.device_info
        self._attr_suggested_display_precision = 1
        self._sensor_option_display_precision = 1

    def set_enabled_default(self, enabled: bool) -> None:
        """Set whether the entity should be enabled by default."""
        self._attr_entity_registry_enabled_default = enabled


class PriorsSensor(AreaOccupancySensorBase):
    """Combined sensor for all priors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the priors sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_PRIORS_SENSOR
        self._attr_unique_id = (
            f"{entry_id}_{NAME_PRIORS_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the overall occupancy prior as the state."""
        return self.coordinator.prior


class AreaOccupancyProbabilitySensor(AreaOccupancySensorBase):
    """Probability sensor for current area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the probability sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_PROBABILITY_SENSOR
        self._attr_unique_id = (
            f"{entry_id}_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = None

    @property
    def native_value(self) -> float | None:
        """Return the current occupancy probability as a percentage."""

        return format_float(self.coordinator.probability * 100)


    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            entities = self.coordinator.entities.entities
            return {
                "active": [
                    f"{entity.entity_id.split('.')[1]} | {entity.state} | {format_float(entity.decay.decay_factor)} | {format_float(entity.probability)}"
                    for entity in entities.values()
                    if entity.is_active
                ],
                "inactive": [
                    f"{entity.entity_id.split('.')[1]} | {entity.state} | {format_float(entity.decay.decay_factor)} | {format_float(entity.probability)}"
                    for entity in entities.values()
                    if not entity.is_active
                ],
                "updated": self.coordinator.last_updated
                if self.coordinator.last_updated
                else "Never",
            }
        except (TypeError, AttributeError, KeyError):
            _LOGGER.exception("Error getting probability attributes: %s")
            return {}


class AreaOccupancyDecaySensor(AreaOccupancySensorBase):
    """Decay status sensor for area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the decay sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_DECAY_SENSOR
        self._attr_unique_id = (
            f"{entry_id}_{NAME_DECAY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the decay status as a percentage."""
        if not self.coordinator.data:
            return 0.0

        try:
            # decay_status is already stored as 0.0 to 100.0
            return format_float(self.coordinator.prior)
        except AttributeError:
            _LOGGER.error("Coordinator data missing decay_status attribute")
            return 0.0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Area Occupancy sensors based on a config entry."""
    coordinator: AreaOccupancyCoordinator = entry.runtime_data

    sensors = [
        AreaOccupancyProbabilitySensor(coordinator, entry.entry_id),
        AreaOccupancyDecaySensor(coordinator, entry.entry_id),
    ]

    # Create priors sensor if history period is configured and greater than 0
    history_period = coordinator.config.decay.history_period
    if history_period > 0:
        sensors.append(PriorsSensor(coordinator, entry.entry_id))

    async_add_entities(sensors, update_before_add=True)


def format_float(value: float) -> float:
    """Format float to consistently show 2 decimal places."""
    try:
        return round(float(value), ROUNDING_PRECISION)
    except (ValueError, TypeError):
        return 0.0
