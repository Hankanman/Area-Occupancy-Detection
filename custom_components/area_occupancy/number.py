"""Number platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.components.sensor import SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import CONF_THRESHOLD
from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

NAME_THRESHOLD_NUMBER = "Occupancy Threshold"


class Threshold(CoordinatorEntity[AreaOccupancyCoordinator], NumberEntity):
    """Number entity for adjusting occupancy threshold."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
    ) -> None:
        """Initialize the threshold entity.

        Args:
            coordinator: The coordinator instance
            area_name: Name of the area this entity represents
        """
        super().__init__(coordinator)
        self._area_name = area_name
        self._attr_has_entity_name = True
        self._attr_name = "Threshold"
        self._attr_unique_id = (
            f"{area_name}_{NAME_THRESHOLD_NUMBER.lower().replace(' ', '_')}"
        )
        self._attr_native_min_value = 1.0
        self._attr_native_max_value = 99.0
        self._attr_native_step = 1.0
        self._attr_mode = NumberMode.BOX
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_entity_category = EntityCategory.CONFIG
        area = coordinator.get_area_or_default(area_name)
        self._attr_device_info = area.device_info()
        self._attr_state_class = SensorStateClass.MEASUREMENT

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        # Assign device to Home Assistant area if area_id is configured
        area = self.coordinator.get_area_or_default(self._area_name)
        if area and area.config.area_id and self.device_info:
            device_registry = dr.async_get(self.hass)
            identifiers = self.device_info.get("identifiers", set())
            device = device_registry.async_get_device(identifiers=identifiers)
            if device and device.area_id != area.config.area_id:
                device_registry.async_update_device(
                    device.id, area_id=area.config.area_id
                )

    @property
    def native_value(self) -> float:
        """Return the current threshold value as a percentage."""
        # Use the area method for threshold (0.0-1.0) and convert to percentage
        area = self.coordinator.get_area_or_default(self._area_name)
        return area.threshold() * 100.0

    async def async_set_native_value(self, value: float) -> None:
        """Set new threshold value (already in percentage)."""
        if value < self._attr_native_min_value or value > self._attr_native_max_value:
            raise ServiceValidationError(
                f"Threshold value must be between {self._attr_native_min_value} and {self._attr_native_max_value}"
            )
        # Update the area's config threshold
        area = self.coordinator.get_area_or_default(self._area_name)
        await area.config.update_config({CONF_THRESHOLD: value})


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Area Occupancy threshold number based on a config entry."""
    coordinator: AreaOccupancyCoordinator = entry.runtime_data

    # Create threshold number entities for each area
    entities: list[NumberEntity] = []
    for area_name in coordinator.get_area_names():
        _LOGGER.debug("Creating threshold number entity for area: %s", area_name)
        entities.append(Threshold(coordinator=coordinator, area_name=area_name))

    async_add_entities(entities, update_before_add=False)
