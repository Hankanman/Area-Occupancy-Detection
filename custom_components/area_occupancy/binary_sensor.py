"""Binary sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .base import AreaOccupancyBinarySensor
from .const import (
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
    DOMAIN,
    NAME_BINARY_SENSOR,
)
from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass
class AreaOccupancyBinaryEntityDescription(BinarySensorEntityDescription):
    """Class describing Area Occupancy binary sensor entities."""

    def __init__(self, area_name: str) -> None:
        """Initialize the description."""
        super().__init__(
            key="occupancy_status",
            name=f"{area_name} {NAME_BINARY_SENSOR}",
            device_class=BinarySensorDeviceClass.OCCUPANCY,
            entity_category=None,
            has_entity_name=True,
        )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Area Occupancy binary sensor based on a config entry."""
    try:
        coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
            "coordinator"
        ]
        # Get threshold from options_config
        threshold = coordinator.options_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)

        async_add_entities(
            [
                AreaOccupancyBinarySensor(
                    coordinator=coordinator,
                    entry_id=entry.entry_id,
                    threshold=threshold,
                )
            ]
        )

    except Exception as err:
        _LOGGER.error("Error setting up binary sensor: %s", err)
        raise HomeAssistantError(
            f"Failed to set up Area Occupancy binary sensor: {err}"
        ) from err
