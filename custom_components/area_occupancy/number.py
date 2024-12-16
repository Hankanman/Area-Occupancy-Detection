"""Number platform for Area Occupancy Detection integration."""

from __future__ import annotations

from homeassistant.components.number import (
    NumberEntity,
    NumberMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    NAME_THRESHOLD_NUMBER,
    CONF_THRESHOLD,
    CONF_AREA_ID,
    DEFAULT_THRESHOLD,
)
from .coordinator import AreaOccupancyCoordinator
from .helpers import get_device_info


class AreaOccupancyThreshold(
    CoordinatorEntity[AreaOccupancyCoordinator], NumberEntity
):  # pylint: disable=abstract-method
    """Number entity for adjusting occupancy threshold."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the threshold entity."""
        super().__init__(coordinator)

        self._attr_has_entity_name = True
        self._attr_name = NAME_THRESHOLD_NUMBER
        self._attr_unique_id = (
            f"{DOMAIN}_{coordinator.core_config[CONF_AREA_ID]}_threshold"
        )
        self._attr_native_min_value = 1.0
        self._attr_native_max_value = 99.0
        self._attr_native_step = 5.0
        self._attr_mode = NumberMode.BOX
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_entity_category = EntityCategory.CONFIG
        self._area_name = coordinator.core_config[CONF_NAME]
        self._attr_device_info = get_device_info(entry_id, self._area_name)

    @property
    def native_value(self) -> float:
        """Return the current threshold value as a percentage."""
        return self.coordinator.options_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)

    async def async_set_native_value(self, value: float) -> None:
        """Set new threshold value (already in percentage)."""
        await self.coordinator.async_update_threshold(value)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Area Occupancy threshold number based on a config entry."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]

    # Create a new number entity for the threshold
    number_entity = AreaOccupancyThreshold(
        coordinator=coordinator,
        entry_id=entry.entry_id,
    )

    async_add_entities([number_entity], update_before_add=True)
