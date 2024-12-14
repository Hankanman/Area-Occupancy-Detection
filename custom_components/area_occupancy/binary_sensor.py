"""Binary sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any, Final

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    NAME_BINARY_SENSOR,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    ATTR_ACTIVE_TRIGGERS,
    CONF_AREA_ID,
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
)
from .coordinator import AreaOccupancyCoordinator
from .types import ProbabilityResult

_LOGGER = logging.getLogger(__name__)
ROUNDING_PRECISION: Final = 2


class AreaOccupancyBinarySensor(
    CoordinatorEntity[AreaOccupancyCoordinator], BinarySensorEntity
):
    """Binary sensor that indicates occupancy status based on computed probability."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        threshold: float,
    ) -> None:
        super().__init__(coordinator)
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._attr_name = NAME_BINARY_SENSOR
        self._attr_unique_id = (
            f"{DOMAIN}_{coordinator.core_config[CONF_AREA_ID]}_occupancy"
        )
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._threshold = threshold
        self._area_name = coordinator.core_config["name"]
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry_id)},
            "name": self._area_name,
            "manufacturer": DEVICE_MANUFACTURER,
            "model": DEVICE_MODEL,
            "sw_version": DEVICE_SW_VERSION,
        }

    @staticmethod
    def _format_float(value: float) -> float:
        try:
            return round(float(value), ROUNDING_PRECISION)
        except (ValueError, TypeError):
            return 0.0

    @property
    def is_on(self) -> bool | None:
        """Return true if the area is occupied."""
        if not self.coordinator.data:
            return None
        try:
            return self.coordinator.data.get("is_occupied", False)
        except (AttributeError, KeyError) as err:
            _LOGGER.error("Error determining occupancy state: %s", err)
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            data: ProbabilityResult = self.coordinator.data

            def get_friendly_names(entity_ids: list[str]) -> list[str]:
                """Convert entity IDs to friendly names."""
                return [
                    self.hass.states.get(entity_id).attributes.get(
                        "friendly_name", entity_id
                    )
                    for entity_id in entity_ids
                    if self.hass.states.get(entity_id)
                ]

            attributes = {
                ATTR_ACTIVE_TRIGGERS: [
                    self.hass.states.get(entity_id).attributes.get(
                        "friendly_name", entity_id
                    )
                    for entity_id in data.get("active_triggers", [])
                    if self.hass.states.get(entity_id)
                ],
            }

            options_config = self.coordinator.options_config
            core_config = self.coordinator.core_config

            if any(
                core_config.get(key) or options_config.get(key)
                for key in [
                    "motion_sensors",
                    "media_devices",
                    "appliances",
                    "illuminance_sensors",
                    "humidity_sensors",
                    "temperature_sensors",
                ]
            ):
                attributes["configured_sensors"] = {
                    "Motion": get_friendly_names(core_config.get("motion_sensors", [])),
                    "Media": get_friendly_names(
                        options_config.get("media_devices", [])
                    ),
                    "Appliances": get_friendly_names(
                        options_config.get("appliances", [])
                    ),
                    "Illuminance": get_friendly_names(
                        options_config.get("illuminance_sensors", [])
                    ),
                    "Humidity": get_friendly_names(
                        options_config.get("humidity_sensors", [])
                    ),
                    "Temperature": get_friendly_names(
                        options_config.get("temperature_sensors", [])
                    ),
                }

            return attributes

        except (AttributeError, KeyError, TypeError) as err:
            _LOGGER.error("Error getting entity attributes: %s", err)
            return {}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
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
