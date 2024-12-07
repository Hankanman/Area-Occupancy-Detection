"""Sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .base import AreaOccupancyProbabilitySensor
from .const import (
    DOMAIN,
    NAME_PROBABILITY_SENSOR,
)
from .coordinator import AreaOccupancyCoordinator
from .prior_sensors import (
    MotionPriorSensor,
    EnvironmentalPriorSensor,
    MediaPriorSensor,
    AppliancePriorSensor,
    OccupancyPriorSensor,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class AreaOccupancyEntityDescription(SensorEntityDescription):
    """Class describing Area Occupancy sensor entities."""

    def __init__(self, area_name: str) -> None:
        """Initialize the description."""
        super().__init__(
            key="occupancy_probability",
            name=f"{area_name} {NAME_PROBABILITY_SENSOR}",
            device_class=SensorDeviceClass.POWER_FACTOR,
            native_unit_of_measurement=PERCENTAGE,
            entity_category=None,
            has_entity_name=True,
        )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Area Occupancy sensor based on a config entry."""
    try:
        coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
            "coordinator"
        ]

        # Create main probability sensor
        entities = [
            AreaOccupancyProbabilitySensor(
                coordinator=coordinator,
                entry_id=entry.entry_id,
            )
        ]

        # Create prior sensors if history period is configured
        if coordinator.options_config.get("history_period"):
            prior_sensors = [
                MotionPriorSensor(
                    coordinator=coordinator,
                    entry_id=entry.entry_id,
                ),
                EnvironmentalPriorSensor(
                    coordinator=coordinator,
                    entry_id=entry.entry_id,
                ),
                MediaPriorSensor(
                    coordinator=coordinator,
                    entry_id=entry.entry_id,
                ),
                AppliancePriorSensor(
                    coordinator=coordinator,
                    entry_id=entry.entry_id,
                ),
                OccupancyPriorSensor(
                    coordinator=coordinator,
                    entry_id=entry.entry_id,
                ),
            ]
            entities.extend(prior_sensors)

        async_add_entities(entities)

    except Exception as err:
        _LOGGER.error("Error setting up sensors: %s", err)
        raise HomeAssistantError(
            f"Failed to set up Area Occupancy sensors: {err}"
        ) from err
