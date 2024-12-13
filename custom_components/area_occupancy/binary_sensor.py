"""Binary sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .base import AreaOccupancyBinarySensor
from .const import (
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
    DOMAIN,
    NAME_BINARY_SENSOR,
)
from .coordinator import AreaOccupancyCoordinator


def create_binary_entity_description(area_name: str) -> BinarySensorEntityDescription:
    """Create binary sensor entity description."""
    return BinarySensorEntityDescription(
        key="occupancy_status",
        name=f"{area_name} {NAME_BINARY_SENSOR}",
        device_class=BinarySensorDeviceClass.OCCUPANCY,
        has_entity_name=True,
    )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Area Occupancy binary sensor based on a config entry."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    threshold = coordinator.options_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)

    async_add_entities(
        [
            AreaOccupancyBinarySensor(
                coordinator=coordinator,
                entry_id=entry.entry_id,
                threshold=threshold,
            )
        ],
        False,
    )
