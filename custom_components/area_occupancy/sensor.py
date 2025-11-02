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
        is_all_areas: bool = False,
    ) -> None:
        """Initialize the sensor.

        Args:
            coordinator: The coordinator instance
            area_name: Name of the area this sensor represents
            is_all_areas: True if this is the "All Areas" aggregation sensor
        """
        super().__init__(coordinator)
        self._area_name = area_name
        self._is_all_areas = is_all_areas
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._attr_device_info = coordinator.device_info(
            area_name=ALL_AREAS_IDENTIFIER if is_all_areas else area_name
        )
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
        area_name: str,
        is_all_areas: bool = False,
    ) -> None:
        """Initialize the priors sensor."""
        super().__init__(coordinator, area_name, is_all_areas)
        self._attr_name = NAME_PRIORS_SENSOR
        unique_id_area = ALL_AREAS_IDENTIFIER if is_all_areas else area_name
        self._attr_unique_id = (
            f"{unique_id_area}_{NAME_PRIORS_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the overall occupancy prior as the state."""
        if self._is_all_areas:
            # For "All Areas": average of all area priors
            area_names = self.coordinator.get_area_names()
            if not area_names:
                return None
            priors = [
                self.coordinator.area_prior(area_name) for area_name in area_names
            ]
            avg_prior = sum(priors) / len(priors) if priors else 0.0
            return format_float(avg_prior * 100)
        return format_float(self.coordinator.area_prior(self._area_name) * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            if self._is_all_areas:
                # For "All Areas": aggregate priors from all areas
                area_names = self.coordinator.get_area_names()
                return {
                    "areas": {
                        area_name: {
                            "global_prior": self.coordinator.area_prior(area_name),
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
            area = self.coordinator.get_area_or_default(self._area_name)
            if area is None:
                return {}
            return {
                "global_prior": self.coordinator.area_prior(self._area_name),
                "time_prior": area.prior.time_prior,
                "day_of_week": area.prior.day_of_week,
                "time_slot": area.prior.time_slot,
            }
        except (TypeError, AttributeError, KeyError):
            return {}


class ProbabilitySensor(AreaOccupancySensorBase):
    """Probability sensor for current area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
        is_all_areas: bool = False,
    ) -> None:
        """Initialize the probability sensor."""
        super().__init__(coordinator, area_name, is_all_areas)
        self._attr_name = NAME_PROBABILITY_SENSOR
        unique_id_area = ALL_AREAS_IDENTIFIER if is_all_areas else area_name
        self._attr_unique_id = (
            f"{unique_id_area}_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> float | None:
        """Return the current occupancy probability as a percentage."""
        if self._is_all_areas:
            # For "All Areas": average of all area probabilities
            area_names = self.coordinator.get_area_names()
            if not area_names:
                return None
            probabilities = [
                self.coordinator.probability(area_name) for area_name in area_names
            ]
            avg_prob = sum(probabilities) / len(probabilities) if probabilities else 0.0
            return format_float(avg_prob * 100)
        return format_float(self.coordinator.probability(self._area_name) * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        if self._is_all_areas:
            # For "All Areas": aggregate type probabilities from all areas
            area_names = self.coordinator.get_area_names()
            return {
                "areas": {
                    area_name: self.coordinator.type_probabilities(area_name)
                    for area_name in area_names
                }
            }
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
        is_all_areas: bool = False,
    ) -> None:
        """Initialize the entities sensor."""
        super().__init__(coordinator, area_name, is_all_areas)
        self._attr_name = NAME_EVIDENCE_SENSOR
        unique_id_area = ALL_AREAS_IDENTIFIER if is_all_areas else area_name
        self._attr_unique_id = (
            f"{unique_id_area}_{NAME_EVIDENCE_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> int | None:
        """Return the number of entities."""
        if self._is_all_areas:
            # For "All Areas": sum of all entities across all areas
            return sum(
                len(self.coordinator.areas[area_name].entities.entities)
                for area_name in self.coordinator.get_area_names()
            )
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
            if self._is_all_areas:
                # For "All Areas": aggregate evidence from all areas
                area_names = self.coordinator.get_area_names()
                all_details = []
                for area_name in area_names:
                    area = self.coordinator.get_area_or_default(area_name)
                    if area is None:
                        continue
                    for entity in sorted(
                        area.entities.entities.values(),
                        key=lambda x: (not x.evidence, -x.type.weight),
                    ):
                        all_details.append(
                            {
                                "area": area_name,
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
                        )
                return {
                    "total": self.native_value,
                    "details": all_details,
                }
            area = self.coordinator.get_area_or_default(self._area_name)
            if area is None:
                return {}
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
        is_all_areas: bool = False,
    ) -> None:
        """Initialize the decay sensor."""
        super().__init__(coordinator, area_name, is_all_areas)
        self._attr_name = NAME_DECAY_SENSOR
        unique_id_area = ALL_AREAS_IDENTIFIER if is_all_areas else area_name
        self._attr_unique_id = (
            f"{unique_id_area}_{NAME_DECAY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the decay status as a percentage."""
        if self._is_all_areas:
            # For "All Areas": average of all area decays
            area_names = self.coordinator.get_area_names()
            if not area_names:
                return None
            decays = [self.coordinator.decay(area_name) for area_name in area_names]
            avg_decay = sum(decays) / len(decays) if decays else 1.0
            return format_float((1 - avg_decay) * 100)
        return format_float((1 - self.coordinator.decay(self._area_name)) * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        try:
            if self._is_all_areas:
                # For "All Areas": aggregate decaying entities from all areas
                area_names = self.coordinator.get_area_names()
                all_decaying = []
                for area_name in area_names:
                    area = self.coordinator.get_area_or_default(area_name)
                    if area is None:
                        continue
                    for entity in area.entities.decaying_entities:
                        all_decaying.append(
                            {
                                "area": area_name,
                                "id": entity.entity_id,
                                "decay": format_percentage(entity.decay.decay_factor),
                            }
                        )
                return {"decaying": all_decaying}
            area = self.coordinator.get_area_or_default(self._area_name)
            if area is None:
                return {}
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

    # Create "All Areas" aggregation sensors if multiple areas exist
    if len(coordinator.get_area_names()) > 1:
        _LOGGER.debug("Creating All Areas aggregation sensors")
        entities.extend(
            [
                ProbabilitySensor(coordinator, ALL_AREAS_IDENTIFIER, is_all_areas=True),
                DecaySensor(coordinator, ALL_AREAS_IDENTIFIER, is_all_areas=True),
                PriorsSensor(coordinator, ALL_AREAS_IDENTIFIER, is_all_areas=True),
                EvidenceSensor(coordinator, ALL_AREAS_IDENTIFIER, is_all_areas=True),
            ]
        )

    async_add_entities(entities, update_before_add=False)
