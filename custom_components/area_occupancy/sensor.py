"""Sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

from .area import AllAreas, AreaDeviceHandle
from .const import ALL_AREAS_IDENTIFIER
from .utils import format_float, format_percentage, generate_entity_unique_id

if TYPE_CHECKING:
    from .area import Area
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

NAME_PRIORS_SENSOR = "Prior Probability"
NAME_DECAY_SENSOR = "Decay Status"
NAME_PROBABILITY_SENSOR = "Occupancy Probability"
NAME_EVIDENCE_SENSOR = "Evidence"


class AreaOccupancySensorBase(CoordinatorEntity, SensorEntity):
    """Base class for area occupancy sensors."""

    def __init__(
        self,
        area_handle: AreaDeviceHandle | None = None,
        all_areas: AllAreas | None = None,
    ) -> None:
        """Initialize the sensor."""
        source = area_handle or all_areas
        if source is None:
            raise ValueError("area_handle or all_areas must be provided")
        super().__init__(source.coordinator)
        self._area_handle = area_handle
        self._all_areas = all_areas
        self._area_name = area_handle.area_name if area_handle else ALL_AREAS_IDENTIFIER
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        device_info = (
            area_handle.device_info()
            if area_handle is not None
            else all_areas.device_info()
        )
        self._attr_device_info = device_info
        self._entry_id = source.coordinator.entry_id
        self._attr_suggested_display_precision = 1
        self._sensor_option_display_precision = 1

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        # Assign device to Home Assistant area if area_id is configured
        # Only for specific areas, not "All Areas"
        if (
            self._area_name != ALL_AREAS_IDENTIFIER
            and (area := self._get_area()) is not None
        ):
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

    def _get_area(self) -> Area | None:
        """Resolve the current Area instance for this entity."""
        if self._area_handle is None:
            return None
        return self._area_handle.resolve()


class PriorsSensor(AreaOccupancySensorBase):
    """Combined sensor for all priors."""

    def __init__(
        self,
        area_handle: AreaDeviceHandle | None = None,
        all_areas: AllAreas | None = None,
    ) -> None:
        """Initialize the priors sensor."""
        super().__init__(area_handle, all_areas)
        self._attr_name = NAME_PRIORS_SENSOR
        # Unique ID: use entry_id, device_id, and entity_name
        self._attr_unique_id = generate_entity_unique_id(
            self._entry_id,
            self.device_info,
            NAME_PRIORS_SENSOR,
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the overall occupancy prior as the state."""
        if self._area_name == ALL_AREAS_IDENTIFIER:
            if self._all_areas is None:
                return None
            return format_float(self._all_areas.area_prior() * 100)
        area = self._get_area()
        if area is None:
            return None
        return format_float(area.area_prior() * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            # For "All Areas", return aggregated priors from all areas
            if self._area_name == ALL_AREAS_IDENTIFIER:
                if self._all_areas is None:
                    return {}
                area_names = self._all_areas.coordinator.get_area_names()
                area_attrs = {}
                for area_name in area_names:
                    area = self._all_areas.coordinator.get_area(area_name)
                    if area is None:
                        continue
                    area_attrs[area_name] = {
                        "global_prior": area.prior.global_prior,
                        "combined_prior": area.area_prior(),
                        "time_prior": area.prior.time_prior,
                        "day_of_week": area.prior.day_of_week,
                        "time_slot": area.prior.time_slot,
                    }
                attrs = {"areas": area_attrs}
            else:
                area = self._get_area()
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
        area_handle: AreaDeviceHandle | None = None,
        all_areas: AllAreas | None = None,
    ) -> None:
        """Initialize the probability sensor."""
        super().__init__(area_handle, all_areas)
        self._attr_name = NAME_PROBABILITY_SENSOR
        # Unique ID: use entry_id, device_id, and entity_name
        self._attr_unique_id = generate_entity_unique_id(
            self._entry_id,
            self.device_info,
            NAME_PROBABILITY_SENSOR,
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> float | None:
        """Return the current occupancy probability as a percentage."""
        if self._area_name == ALL_AREAS_IDENTIFIER:
            if self._all_areas is None:
                return None
            return format_float(self._all_areas.probability() * 100)
        area = self._get_area()
        if area is None:
            return None
        return format_float(area.probability() * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        return {}

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        super()._handle_coordinator_update()


class EvidenceSensor(AreaOccupancySensorBase):
    """Sensor for all evidence."""

    _unrecorded_attributes = frozenset({"evidence", "no_evidence", "total", "details"})

    def __init__(
        self,
        area_handle: AreaDeviceHandle | None = None,
        all_areas: AllAreas | None = None,
    ) -> None:
        """Initialize the entities sensor."""
        super().__init__(area_handle, all_areas)
        self._attr_name = NAME_EVIDENCE_SENSOR
        # Unique ID: use entry_id, device_id, and entity_name
        self._attr_unique_id = generate_entity_unique_id(
            self._entry_id,
            self.device_info,
            NAME_EVIDENCE_SENSOR,
        )
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> int | None:
        """Return the number of entities."""
        area = self._get_area()
        if area is None:
            return None
        return len(area.entities.entities)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            area = self._get_area()
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
        area_handle: AreaDeviceHandle | None = None,
        all_areas: AllAreas | None = None,
    ) -> None:
        """Initialize the decay sensor."""
        super().__init__(area_handle, all_areas)
        self._attr_name = NAME_DECAY_SENSOR
        # Unique ID: use entry_id, device_id, and entity_name
        self._attr_unique_id = generate_entity_unique_id(
            self._entry_id,
            self.device_info,
            NAME_DECAY_SENSOR,
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the decay status as a percentage."""
        if self._area_name == ALL_AREAS_IDENTIFIER:
            if self._all_areas is None:
                return None
            decay_value = self._all_areas.decay()
        else:
            area = self._get_area()
            if area is None:
                return None
            decay_value = area.decay()
        return format_float((1 - decay_value) * 100)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        try:
            # For "All Areas", aggregate decaying entities from all areas
            if self._area_name == ALL_AREAS_IDENTIFIER:
                if self._all_areas is None:
                    return {}
                area_names = self._all_areas.coordinator.get_area_names()
                all_decaying = []
                for area_name in area_names:
                    area = self._all_areas.coordinator.get_area(area_name)
                    if area is None:
                        continue
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
            area = self._get_area()
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
        area_handle = coordinator.get_area_handle(area_name)
        _LOGGER.debug("Creating sensors for area: %s", area_name)
        entities.extend(
            [
                ProbabilitySensor(area_handle=area_handle),
                DecaySensor(area_handle=area_handle),
                PriorsSensor(area_handle=area_handle),
                EvidenceSensor(area_handle=area_handle),
            ]
        )

    # Create "All Areas" aggregation sensors when areas exist
    # Note: EvidenceSensor is NOT created for "All Areas"
    if len(coordinator.get_area_names()) >= 1:
        _LOGGER.debug("Creating All Areas aggregation sensors")
        all_areas = coordinator.get_all_areas()
        entities.extend(
            [
                ProbabilitySensor(all_areas=all_areas),
                DecaySensor(all_areas=all_areas),
                PriorsSensor(all_areas=all_areas),
            ]
        )

    async_add_entities(entities, update_before_add=False)
