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
from homeassistant.const import PERCENTAGE
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
    """Binary sensor for area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        threshold: float,
    ) -> None:
        """Initialize the binary sensor."""
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
        self._attr_entity_category = None
        self._attr_native_unit_of_measurement = PERCENTAGE

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry_id)},
            "name": self._area_name,
            "manufacturer": DEVICE_MANUFACTURER,
            "model": DEVICE_MODEL,
            "sw_version": DEVICE_SW_VERSION,
        }

    @staticmethod
    def _format_float(value: float) -> float:
        """Format float to consistently show 2 decimal places."""
        try:
            return round(float(value), ROUNDING_PRECISION)
        except (ValueError, TypeError):
            return 0.0

    @property
    def is_on(self) -> bool:
        """Return true if the area is occupied."""
        try:
            if not self.coordinator.data:
                return False
            return self.coordinator.data.get("is_occupied", False)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error determining occupancy state: %s", err)
            return False

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
                ATTR_ACTIVE_TRIGGERS: get_friendly_names(
                    data.get("active_triggers", [])
                ),
            }

            # Add configured sensors info
            options_config = self.coordinator.options_config
            core_config = self.coordinator.core_config

            configured_sensors = {
                "Motion": core_config.get("motion_sensors", []),
                "Media": options_config.get("media_devices", []),
                "Appliances": options_config.get("appliances", []),
                "Illuminance": options_config.get("illuminance_sensors", []),
                "Humidity": options_config.get("humidity_sensors", []),
                "Temperature": options_config.get("temperature_sensors", []),
            }

            # Flatten all sensors to count how many have learned priors
            all_sensors = []
            for sensor_list in configured_sensors.values():
                all_sensors.extend(sensor_list)

            learned_count = 0
            for sensor in all_sensors:
                if sensor in self.coordinator.learned_priors:
                    learned_count += 1

            attributes["configured_sensors"] = {
                cat: get_friendly_names(slist)
                for cat, slist in configured_sensors.items()
            }

            # Show how many sensors have learned priors
            attributes["learned_prior_sensors_count"] = learned_count
            attributes["total_sensors_count"] = len(all_sensors)

            return attributes

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error getting entity attributes: %s", err)
            return {}


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
