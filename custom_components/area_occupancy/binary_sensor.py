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
    ATTR_ACTIVE_TRIGGERS,
    ATTR_CONFIDENCE_SCORE,
    ATTR_DECAY_STATUS,
    ATTR_LAST_OCCUPIED,
    ATTR_OCCUPANCY_RATE,
    ATTR_PRIOR_PROBABILITY,
    ATTR_PROBABILITY,
    ATTR_SENSOR_AVAILABILITY,
    ATTR_SENSOR_PROBABILITIES,
    ATTR_STATE_DURATION,
    ATTR_THRESHOLD,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    NAME_BINARY_SENSOR,
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

        # Entity attributes
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._attr_name = NAME_BINARY_SENSOR
        self._attr_unique_id = (
            f"{DOMAIN}_{coordinator.core_config[CONF_AREA_ID]}_occupancy"
        )
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._threshold = threshold
        self._area_name = coordinator.core_config["name"]

        # Device info
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
    def is_on(self) -> bool | None:
        """Return true if the area is occupied."""
        try:
            if not self.coordinator.data:
                return None
            return self.coordinator.data.get("probability", 0.0) >= self._threshold
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error determining occupancy state: %s", err)
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        if not self.coordinator.data:
            return {}

        try:
            data: ProbabilityResult = self.coordinator.data

            def format_percentage(value: float) -> float:
                """Format percentage values consistently."""
                return self._format_float(value * 100)

            attributes = {
                ATTR_PROBABILITY: format_percentage(data.get("probability", 0.0)),
                ATTR_PRIOR_PROBABILITY: format_percentage(
                    data.get("prior_probability", 0.0)
                ),
                ATTR_ACTIVE_TRIGGERS: data.get("active_triggers", []),
                ATTR_SENSOR_PROBABILITIES: {
                    k: format_percentage(v)
                    for k, v in data.get("sensor_probabilities", {}).items()
                },
                ATTR_DECAY_STATUS: {
                    k: self._format_float(v)
                    for k, v in data.get("decay_status", {}).items()
                },
                ATTR_CONFIDENCE_SCORE: format_percentage(
                    data.get("confidence_score", 0.0)
                ),
                ATTR_SENSOR_AVAILABILITY: data.get("sensor_availability", {}),
                ATTR_THRESHOLD: self._format_float(self._threshold),
                ATTR_LAST_OCCUPIED: data.get("last_occupied"),
                ATTR_STATE_DURATION: self._format_float(
                    data.get("state_duration", 0.0) / 60  # Convert to minutes
                ),
                ATTR_OCCUPANCY_RATE: self._format_float(
                    data.get("occupancy_rate", 0.0) * 100
                ),
            }

            # Add configuration info
            options_config = self.coordinator.options_config
            core_config = self.coordinator.core_config
            attributes.update(
                {
                    "configured_motion_sensors": core_config.get("motion_sensors", []),
                    "configured_media_devices": options_config.get("media_devices", []),
                    "configured_appliances": options_config.get("appliances", []),
                    "configured_illuminance_sensors": options_config.get(
                        "illuminance_sensors", []
                    ),
                    "configured_humidity_sensors": options_config.get(
                        "humidity_sensors", []
                    ),
                    "configured_temperature_sensors": options_config.get(
                        "temperature_sensors", []
                    ),
                }
            )
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
