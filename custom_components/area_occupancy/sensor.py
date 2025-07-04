"""Sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .coordinator import AreaOccupancyCoordinator
from .utils import format_float, format_percentage

NAME_PRIORS_SENSOR = "Prior Probability"
NAME_DECAY_SENSOR = "Decay Status"
NAME_PROBABILITY_SENSOR = "Occupancy Probability"
NAME_EVIDENCE_SENSOR = "Evidence"


class AreaOccupancySensorBase(
    CoordinatorEntity[AreaOccupancyCoordinator], SensorEntity
):
    """Base class for area occupancy sensors."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
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

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
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
        return format_float(self.coordinator.area_prior * 100)


class ProbabilitySensor(AreaOccupancySensorBase):
    """Probability sensor for current area occupancy."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize the probability sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_PROBABILITY_SENSOR
        self._attr_unique_id = (
            f"{entry_id}_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> float | None:
        """Return the current occupancy probability as a percentage."""

        return format_float(self.coordinator.probability * 100)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        super()._handle_coordinator_update()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            return {
                "complementary_probability": format_float(
                    self.coordinator.probability * 100
                ),
                "conditional_probability": format_float(
                    self.coordinator.conditional_probability * 100
                ),
                "conditional_sorted_probability": format_float(
                    self.coordinator.conditional_sorted_probability * 100
                ),
            }
        except (TypeError, AttributeError, KeyError):
            return {}


class EvidenceSensor(AreaOccupancySensorBase):
    """Sensor for all evidence."""

    _unrecorded_attributes = frozenset({"evidence", "no_evidence", "total", "details"})

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize the entities sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_EVIDENCE_SENSOR
        self._attr_unique_id = (
            f"{entry_id}_{NAME_EVIDENCE_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> int | None:
        """Return the entities as a percentage."""
        return len(self.coordinator.entities.entities)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            active_entity_names = ", ".join(
                [
                    entity.name
                    for entity in self.coordinator.entities.active_entities
                    if entity.name
                ]
            )
            inactive_entity_names = ", ".join(
                [
                    entity.name
                    for entity in self.coordinator.entities.inactive_entities
                    if entity.name
                ]
            )
            return {
                "evidence": active_entity_names,
                "no_evidence": inactive_entity_names,
                "total": len(self.coordinator.entities.entities),
                "details": [
                    {
                        "id": entity.entity_id,
                        "evidence": entity.evidence,
                        "probability": entity.probability,
                        "weight": entity.type.weight,
                        "name": entity.name,
                        "state": entity.state,
                        "decaying": entity.decay.is_decaying,
                    }
                    for entity in sorted(
                        self.coordinator.entities.entities.values(),
                        key=lambda x: (not x.evidence, -x.type.weight),
                    )
                ],
            }
        except (TypeError, AttributeError, KeyError):
            return {}


class DecaySensor(AreaOccupancySensorBase):
    """Decay status sensor for area occupancy."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
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

        return format_float((1 - self.coordinator.decay) * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        try:
            return {
                "decaying": [
                    {
                        "id": {entity.entity_id},
                        "decay": {format_percentage(entity.decay.decay_factor)},
                    }
                    for entity in self.coordinator.entities.decaying_entities
                ]
            }
        except (TypeError, AttributeError, KeyError):
            return {}


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: Any
) -> None:
    """Set up the Area Occupancy sensors based on a config entry."""
    coordinator: AreaOccupancyCoordinator = entry.runtime_data

    entities = [
        ProbabilitySensor(coordinator, entry.entry_id),
        DecaySensor(coordinator, entry.entry_id),
        PriorsSensor(coordinator, entry.entry_id),
        EvidenceSensor(coordinator, entry.entry_id),
    ]

    async_add_entities(entities, update_before_add=True)
