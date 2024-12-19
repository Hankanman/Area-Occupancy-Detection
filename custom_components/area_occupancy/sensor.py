"""Sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from datetime import timedelta

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PERCENTAGE,
    EntityCategory,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    NAME_PROBABILITY_SENSOR,
    NAME_PRIORS_SENSOR,
    CONF_HISTORY_PERIOD,
    DEFAULT_HISTORY_PERIOD,
    NAME_DECAY_SENSOR,
    CONF_NAME,
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
)
from .coordinator import AreaOccupancyCoordinator
from .types import (
    ProbabilityResult,
    ProbabilityAttributes,
    PriorsAttributes,
)
from .helpers import format_float

_LOGGER = logging.getLogger(__name__)


class AreaOccupancySensorBase(
    CoordinatorEntity[AreaOccupancyCoordinator], SensorEntity
):
    """Base class for area occupancy sensors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        super().__init__(coordinator)
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._area_name = coordinator.config[CONF_NAME]
        self._attr_device_info = coordinator.device_info
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
        entry_id: str,
    ) -> None:
        """Initialize the priors sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_PRIORS_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_PRIORS_SENSOR.lower().replace(' ', '_')}"
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the overall occupancy prior as the state."""
        try:
            type_priors = self.coordinator.type_priors
            if not type_priors:
                return None

            # Calculate average of all type priors
            priors = [prior["prior"] for prior in type_priors.values()]
            return sum(priors) / len(priors) * 100

        except (TypeError, ValueError, AttributeError, KeyError) as err:
            _LOGGER.error("Error calculating priors: %s", err)
            return None

    @property
    def extra_state_attributes(self) -> PriorsAttributes:
        """Return all prior probabilities as attributes."""
        try:
            type_priors = self.coordinator.type_priors

            attributes = {
                f"{sensor_type}": (
                    f"T: {round(format_float(prior['prob_given_true']) * 100, 1)}% | "
                    f"F: {round(format_float(prior['prob_given_false']) * 100, 1)}%"
                )
                for sensor_type, prior in type_priors.items()
            }

            attributes.update(
                {
                    "last_updated": dt_util.utcnow().isoformat(),
                    "total_period": f"{str(
                        timedelta(
                            days=self.coordinator.config.get(
                                CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                            )
                        ).days
                    )} days",
                }
            )

            return attributes

        except (TypeError, ValueError, AttributeError, KeyError) as err:
            _LOGGER.error("Error getting prior attributes: %s", err)
            return {}


class AreaOccupancyProbabilitySensor(AreaOccupancySensorBase):
    """Probability sensor for current area occupancy."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_PROBABILITY_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}"
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = None

    @property
    def native_value(self) -> float | None:
        try:
            if not self.coordinator.data:
                return None
            probability = self.coordinator.data.get("probability", 0.0)
            return format_float(probability * 100)
        except (TypeError, ValueError, AttributeError) as err:
            _LOGGER.error("Error getting probability value: %s", err)
            return None

    @property
    def extra_state_attributes(self) -> ProbabilityAttributes:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            data: ProbabilityResult = self.coordinator.data

            # Create formatted probability entries with details
            sensor_probabilities = set()  # Use a set instead of dict
            for entity_id, prob_details in data.get("sensor_probabilities", {}).items():
                friendly_name = (
                    self.hass.states.get(entity_id).attributes.get(
                        "friendly_name", entity_id
                    )
                    if self.hass.states.get(entity_id)
                    else entity_id
                )

                formatted_entry = (
                    f"{friendly_name} | "
                    f"W: {format_float(prob_details['weight'])} | "
                    f"P: {format_float(prob_details['probability'])} | "
                    f"WP: {format_float(prob_details['weighted_probability'])}"
                )
                sensor_probabilities.add(formatted_entry)

            return {
                "active_triggers": [
                    self.hass.states.get(entity_id).attributes.get(
                        "friendly_name", entity_id
                    )
                    for entity_id in data.get("active_triggers", [])
                    if self.hass.states.get(entity_id)
                ],
                "sensor_probabilities": sensor_probabilities,
                "threshold": f"{self.coordinator.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)}%",
            }
        except (TypeError, AttributeError, KeyError) as err:
            _LOGGER.error("Error getting probability attributes: %s", err)
            return {}


class AreaOccupancyDecaySensor(AreaOccupancySensorBase):
    """Decay status sensor for area occupancy."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_DECAY_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_DECAY_SENSOR.lower().replace(' ', '_')}"
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        try:
            if not self.coordinator.data or "decay_status" not in self.coordinator.data:
                return 0.0

            decay_values = [
                v
                for v in self.coordinator.data["decay_status"].values()
                if v is not None
            ]

            if not decay_values:
                return 0.0

            average_decay = sum(decay_values) / len(decay_values)
            return format_float(average_decay * 100)

        except (TypeError, KeyError, ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error calculating decay value: %s", err)
            return 0.0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Area Occupancy sensors based on a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]

    sensors = [
        AreaOccupancyProbabilitySensor(coordinator, entry.entry_id),
        AreaOccupancyDecaySensor(coordinator, entry.entry_id),
    ]

    # Create priors sensor if history period is configured and greater than 0
    history_period = coordinator.config.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD)
    if history_period > 0:
        sensors.append(PriorsSensor(coordinator, entry.entry_id))

    async_add_entities(sensors, update_before_add=True)
