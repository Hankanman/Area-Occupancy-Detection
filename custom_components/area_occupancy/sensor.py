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
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import ALL_AREAS_IDENTIFIER
from .coordinator import AreaOccupancyCoordinator
from .utils import format_float, format_percentage

_LOGGER = logging.getLogger(__name__)

NAME_PRIORS_SENSOR = "Prior Probability"
NAME_DECAY_SENSOR = "Decay Status"
NAME_PROBABILITY_SENSOR = "Occupancy Probability"
NAME_EVIDENCE_SENSOR = "Evidence"


class AreaOccupancySensorBase(
    CoordinatorEntity[AreaOccupancyCoordinator], SensorEntity
):
    """Base class for area occupancy sensors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
    ) -> None:
        """Initialize the sensor.

        Args:
            coordinator: The coordinator instance
            area_name: Name of the area this sensor represents (or ALL_AREAS_IDENTIFIER for "All Areas")
        """
        super().__init__(coordinator)
        self._area_name = area_name
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._attr_device_info = coordinator.device_info(area_name=area_name)
        self._attr_suggested_display_precision = 1
        self._sensor_option_display_precision = 1

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        # Assign device to Home Assistant area if area_id is configured
        # Only for specific areas, not "All Areas"
        if (
            self._area_name != ALL_AREAS_IDENTIFIER
            and self._area_name in self.coordinator.areas
        ):
            area = self.coordinator.areas[self._area_name]
            if area.config.area_id and self.device_info:
                device_registry = dr.async_get(self.hass)
                identifiers = self.device_info.get("identifiers", set())
                device = device_registry.async_get_device(identifiers=identifiers)
                if device and device.area_id != area.config.area_id:
                    device_registry.async_update_device(
                        device.id, area_id=area.config.area_id
                    )

    def set_enabled_default(self, enabled: bool) -> None:
        """Set whether the entity should be enabled by default."""
        self._attr_entity_registry_enabled_default = enabled


class PriorsSensor(AreaOccupancySensorBase):
    """Combined sensor for all priors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
    ) -> None:
        """Initialize the priors sensor."""
        super().__init__(coordinator, area_name)
        self._attr_name = NAME_PRIORS_SENSOR
        self._attr_unique_id = (
            f"{area_name}_{NAME_PRIORS_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the overall occupancy prior as the state."""
        return format_float(self.coordinator.area_prior(self._area_name) * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            # For "All Areas", return aggregated priors from all areas
            if self._area_name == ALL_AREAS_IDENTIFIER:
                area_names = self.coordinator.get_area_names()
                attrs = {
                    "areas": {
                        area_name: {
                            "global_prior": self.coordinator.areas[
                                area_name
                            ].prior.global_prior,
                            "combined_prior": self.coordinator.areas[
                                area_name
                            ].area_prior(),
                            "time_prior": self.coordinator.areas[
                                area_name
                            ].prior.time_prior,
                            "day_of_week": self.coordinator.areas[
                                area_name
                            ].prior.day_of_week,
                            "time_slot": self.coordinator.areas[
                                area_name
                            ].prior.time_slot,
                        }
                        for area_name in area_names
                    }
                }
            else:
                area = self.coordinator.get_area_or_default(self._area_name)
                combined_prior = area.area_prior() if area else None
                attrs = {
                    "global_prior": area.prior.global_prior if area else None,
                    "combined_prior": combined_prior,
                    "time_prior": area.prior.time_prior if area else None,
                    "day_of_week": area.prior.day_of_week if area else None,
                    "time_slot": area.prior.time_slot if area else None,
                }
        except (TypeError, AttributeError, KeyError):
            return {}
        return attrs


class ProbabilitySensor(AreaOccupancySensorBase):
    """Probability sensor for current area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
    ) -> None:
        """Initialize the probability sensor."""
        super().__init__(coordinator, area_name)
        self._attr_name = NAME_PROBABILITY_SENSOR
        self._attr_unique_id = (
            f"{area_name}_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> float | None:
        """Return the current occupancy probability as a percentage."""
        return format_float(self.coordinator.probability(self._area_name) * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        # "All Areas" does not support type_probabilities aggregation
        if self._area_name == ALL_AREAS_IDENTIFIER:
            return {}
        return self.coordinator.type_probabilities(self._area_name)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        super()._handle_coordinator_update()


class EvidenceSensor(AreaOccupancySensorBase):
    """Sensor for all evidence."""

    _unrecorded_attributes = frozenset({"evidence", "no_evidence", "total", "details"})

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
    ) -> None:
        """Initialize the entities sensor."""
        super().__init__(coordinator, area_name)
        self._attr_name = NAME_EVIDENCE_SENSOR
        self._attr_unique_id = (
            f"{area_name}_{NAME_EVIDENCE_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> int | None:
        """Return the number of entities."""
        area = self.coordinator.get_area_or_default(self._area_name)
        if area is None:
            return None
        return len(area.entities.entities)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            area = self.coordinator.get_area_or_default(self._area_name)
            active_entity_names = ", ".join(
                [entity.name for entity in area.entities.active_entities if entity.name]
            )
            inactive_entity_names = ", ".join(
                [
                    entity.name
                    for entity in area.entities.inactive_entities
                    if entity.name
                ]
            )
            return {
                "evidence": active_entity_names,
                "no_evidence": inactive_entity_names,
                "total": len(area.entities.entities),
                "details": [
                    {
                        "id": entity.entity_id,
                        "name": entity.name,
                        "evidence": entity.evidence,
                        "prob_given_true": entity.prob_given_true,
                        "prob_given_false": entity.prob_given_false,
                        "weight": entity.weight,
                        "state": entity.state,
                        "decaying": entity.decay.is_decaying,
                        "decay_factor": entity.decay.decay_factor,
                    }
                    for entity in sorted(
                        area.entities.entities.values(),
                        key=lambda x: (not x.evidence, -x.type.weight),
                    )
                ],
            }
        except (TypeError, AttributeError, KeyError):
            return {}


class DecaySensor(AreaOccupancySensorBase):
    """Decay status sensor for area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
    ) -> None:
        """Initialize the decay sensor."""
        super().__init__(coordinator, area_name)
        self._attr_name = NAME_DECAY_SENSOR
        self._attr_unique_id = (
            f"{area_name}_{NAME_DECAY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the decay status as a percentage."""
        return format_float((1 - self.coordinator.decay(self._area_name)) * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        try:
            # For "All Areas", aggregate decaying entities from all areas
            if self._area_name == ALL_AREAS_IDENTIFIER:
                area_names = self.coordinator.get_area_names()
                all_decaying = []
                for area_name in area_names:
                    area = self.coordinator.get_area_or_default(area_name)
                    all_decaying.extend(
                        [
                            {
                                "area": area_name,
                                "id": entity.entity_id,
                                "decay": format_percentage(entity.decay.decay_factor),
                            }
                            for entity in area.entities.decaying_entities
                        ]
                    )
                return {"decaying": all_decaying}
            area = self.coordinator.get_area_or_default(self._area_name)
            return {
                "decaying": [
                    {
                        "id": entity.entity_id,
                        "decay": format_percentage(entity.decay.decay_factor),
                    }
                    for entity in area.entities.decaying_entities
                ]
            }
        except (TypeError, AttributeError, KeyError):
            return {}


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: Any
) -> None:
    """Set up the Area Occupancy sensors based on a config entry."""
    coordinator: AreaOccupancyCoordinator = entry.runtime_data

    entities: list[SensorEntity] = []

    # Create sensors for each area
    for area_name in coordinator.get_area_names():
        _LOGGER.debug("Creating sensors for area: %s", area_name)
        entities.extend(
            [
                ProbabilitySensor(coordinator, area_name),
                DecaySensor(coordinator, area_name),
                PriorsSensor(coordinator, area_name),
                EvidenceSensor(coordinator, area_name),
            ]
        )

    # Create "All Areas" aggregation sensors when areas exist
    # Note: EvidenceSensor is NOT created for "All Areas"
    if len(coordinator.get_area_names()) >= 1:
        _LOGGER.debug("Creating All Areas aggregation sensors")
        entities.extend(
            [
                ProbabilitySensor(coordinator, ALL_AREAS_IDENTIFIER),
                DecaySensor(coordinator, ALL_AREAS_IDENTIFIER),
                PriorsSensor(coordinator, ALL_AREAS_IDENTIFIER),
            ]
        )

    async_add_entities(entities, update_before_add=False)
