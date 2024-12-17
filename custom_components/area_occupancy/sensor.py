"""Sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Final

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
    NAME_MOTION_PRIOR_SENSOR,
    NAME_MEDIA_PRIOR_SENSOR,
    NAME_APPLIANCE_PRIOR_SENSOR,
    NAME_OCCUPANCY_PRIOR_SENSOR,
    NAME_DOOR_PRIOR_SENSOR,
    NAME_WINDOW_PRIOR_SENSOR,
    NAME_LIGHT_PRIOR_SENSOR,
    ATTR_TOTAL_PERIOD,
    ATTR_PROB_GIVEN_TRUE,
    ATTR_PROB_GIVEN_FALSE,
    ATTR_LAST_UPDATED,
    ATTR_ACTIVE_TRIGGERS,
    ATTR_SENSOR_PROBABILITIES,
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_APPLIANCES,
    CONF_DOOR_SENSORS,
    CONF_WINDOW_SENSORS,
    CONF_LIGHTS,
    CONF_HISTORY_PERIOD,
    DEFAULT_HISTORY_PERIOD,
    NAME_DECAY_SENSOR,
    CONF_NAME,
)
from .coordinator import AreaOccupancyCoordinator
from .probabilities import (
    DOOR_PROB_GIVEN_TRUE,
    DOOR_PROB_GIVEN_FALSE,
    WINDOW_PROB_GIVEN_TRUE,
    WINDOW_PROB_GIVEN_FALSE,
    LIGHT_PROB_GIVEN_TRUE,
    LIGHT_PROB_GIVEN_FALSE,
    MOTION_PROB_GIVEN_TRUE,
    MOTION_PROB_GIVEN_FALSE,
    MEDIA_PROB_GIVEN_TRUE,
    MEDIA_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
    APPLIANCE_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    DEFAULT_PROB_GIVEN_FALSE,
)
from .types import ProbabilityResult
from .helpers import format_float, get_device_info

_LOGGER = logging.getLogger(__name__)
ROUNDING_PRECISION: Final = 2


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
        self._attr_device_info = get_device_info(entry_id, self._area_name)

    @staticmethod
    def _format_float(value: float) -> float:
        """Format float to consistently show 2 decimal places."""
        return format_float(value)


class PriorProbabilitySensorBase(AreaOccupancySensorBase, SensorEntity):
    """Base class for prior probability sensors that aggregate learned priors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        default_p_true: float = DEFAULT_PROB_GIVEN_TRUE,
        default_p_false: float = DEFAULT_PROB_GIVEN_FALSE,
    ) -> None:
        super().__init__(coordinator, entry_id)
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

        # Store default probabilities for fallback
        self._default_p_true = default_p_true
        self._default_p_false = default_p_false
        self._last_calculation = None

    def _get_sensor_list(self) -> list[str]:
        """Return a list of sensors for this category. Implemented by subclasses."""
        raise NotImplementedError

    def _get_aggregated_learned_priors(self) -> tuple[float, float]:
        """Compute aggregated learned priors for all sensors in this category."""
        sensor_list = self._get_sensor_list()
        learned = self.coordinator.learned_priors

        p_true_values = []
        p_false_values = []

        for entity_id in sensor_list:
            priors = learned.get(entity_id)
            if priors:
                p_true_values.append(priors["prob_given_true"])
                p_false_values.append(priors["prob_given_false"])

        if p_true_values and p_false_values:
            avg_p_true = sum(p_true_values) / len(p_true_values)
            avg_p_false = sum(p_false_values) / len(p_false_values)
            return avg_p_true, avg_p_false

        # If no learned priors found, use defaults
        return self._default_p_true, self._default_p_false

    @property
    def native_value(self) -> float | None:
        """Return the prior probability as a percentage."""
        p_true, _ = self._get_aggregated_learned_priors()
        return round(p_true * 100, 4)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        p_true, p_false = self._get_aggregated_learned_priors()
        return {
            ATTR_TOTAL_PERIOD: str(
                timedelta(
                    days=self.coordinator.config.get(
                        CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                    )
                )
            ),
            ATTR_PROB_GIVEN_TRUE: p_true,
            ATTR_PROB_GIVEN_FALSE: p_false,
            ATTR_LAST_UPDATED: (dt_util.utcnow().isoformat()),
        }


class MotionPriorSensor(PriorProbabilitySensorBase):
    """Sensor for aggregated motion prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(
            coordinator,
            entry_id,
            default_p_true=MOTION_PROB_GIVEN_TRUE,
            default_p_false=MOTION_PROB_GIVEN_FALSE,
        )
        self._attr_name = NAME_MOTION_PRIOR_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_MOTION_PRIOR_SENSOR.lower().replace(' ', '_')}"

    def _get_sensor_list(self) -> list[str]:
        return self.coordinator.config.get(CONF_MOTION_SENSORS, [])


class MediaPriorSensor(PriorProbabilitySensorBase):
    """Sensor for aggregated media device prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(
            coordinator,
            entry_id,
            default_p_true=MEDIA_PROB_GIVEN_TRUE,
            default_p_false=MEDIA_PROB_GIVEN_FALSE,
        )
        self._attr_name = NAME_MEDIA_PRIOR_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_MEDIA_PRIOR_SENSOR.lower().replace(' ', '_')}"

    def _get_sensor_list(self) -> list[str]:
        return self.coordinator.config.get(CONF_MEDIA_DEVICES, [])


class AppliancePriorSensor(PriorProbabilitySensorBase):
    """Sensor for aggregated appliance prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(
            coordinator,
            entry_id,
            default_p_true=APPLIANCE_PROB_GIVEN_TRUE,
            default_p_false=APPLIANCE_PROB_GIVEN_FALSE,
        )
        self._attr_name = NAME_APPLIANCE_PRIOR_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_APPLIANCE_PRIOR_SENSOR.lower().replace(' ', '_')}"

    def _get_sensor_list(self) -> list[str]:
        return self.coordinator.config.get(CONF_APPLIANCES, [])


class DoorPriorSensor(PriorProbabilitySensorBase):
    """Sensor for aggregated door prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(
            coordinator,
            entry_id,
            default_p_true=DOOR_PROB_GIVEN_TRUE,
            default_p_false=DOOR_PROB_GIVEN_FALSE,
        )
        self._attr_name = NAME_DOOR_PRIOR_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_DOOR_PRIOR_SENSOR.lower().replace(' ', '_')}"

    def _get_sensor_list(self) -> list[str]:
        return self.coordinator.config.get(CONF_DOOR_SENSORS, [])


class WindowPriorSensor(PriorProbabilitySensorBase):
    """Sensor for aggregated window prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(
            coordinator,
            entry_id,
            default_p_true=WINDOW_PROB_GIVEN_TRUE,
            default_p_false=WINDOW_PROB_GIVEN_FALSE,
        )
        self._attr_name = NAME_WINDOW_PRIOR_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_WINDOW_PRIOR_SENSOR.lower().replace(' ', '_')}"

    def _get_sensor_list(self) -> list[str]:
        return self.coordinator.config.get(CONF_WINDOW_SENSORS, [])


class LightPriorSensor(PriorProbabilitySensorBase):
    """Sensor for aggregated light prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(
            coordinator,
            entry_id,
            default_p_true=LIGHT_PROB_GIVEN_TRUE,
            default_p_false=LIGHT_PROB_GIVEN_FALSE,
        )
        self._attr_name = NAME_LIGHT_PRIOR_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_LIGHT_PRIOR_SENSOR.lower().replace(' ', '_')}"

    def _get_sensor_list(self) -> list[str]:
        return self.coordinator.config.get(CONF_LIGHTS, [])


class OccupancyPriorSensor(PriorProbabilitySensorBase):
    """Sensor for aggregated occupancy prior probability across all configured sensors."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        super().__init__(
            coordinator,
            entry_id,
            default_p_true=DEFAULT_PROB_GIVEN_TRUE,
            default_p_false=DEFAULT_PROB_GIVEN_FALSE,
        )
        self._attr_name = NAME_OCCUPANCY_PRIOR_SENSOR
        self._attr_unique_id = f"{DOMAIN}_{coordinator.entry_id}_{NAME_OCCUPANCY_PRIOR_SENSOR.lower().replace(' ', '_')}"

    def _get_sensor_list(self) -> list[str]:
        # All configured sensors
        return self.coordinator.get_configured_sensors()


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
            return self._format_float(probability * 100)
        except (TypeError, ValueError, AttributeError) as err:
            _LOGGER.error("Error getting probability value: %s", err)
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            data: ProbabilityResult = self.coordinator.data

            def format_percentage(value: float) -> float:
                return self._format_float(value * 100)

            return {
                ATTR_ACTIVE_TRIGGERS: [
                    self.hass.states.get(entity_id).attributes.get(
                        "friendly_name", entity_id
                    )
                    for entity_id in data.get("active_triggers", [])
                    if self.hass.states.get(entity_id)
                ],
                ATTR_SENSOR_PROBABILITIES: {
                    (
                        self.hass.states.get(k).attributes.get("friendly_name", k)
                        if self.hass.states.get(k)
                        else k
                    ): format_percentage(v)
                    for k, v in data.get("sensor_probabilities", {}).items()
                },
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
            return self._format_float(average_decay * 100)

        except (TypeError, KeyError, ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error calculating decay value: %s", err)
            return 0.0

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        try:
            if not self.coordinator.data or "decay_status" not in self.coordinator.data:
                return {}

            decay_status = self.coordinator.data["decay_status"]
            formatted_decays = {}

            for entity_id, decay_value in decay_status.items():
                state = self.hass.states.get(entity_id)
                if state and decay_value is not None:
                    friendly_name = state.attributes.get("friendly_name", entity_id)
                    formatted_decays[friendly_name] = self._format_float(
                        decay_value * 100
                    )

            return {"individual_decays": formatted_decays} if formatted_decays else {}

        except (TypeError, KeyError, AttributeError) as err:
            _LOGGER.error("Error getting decay attributes: %s", err)
            return {}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Area Occupancy sensors based on a config entry."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]

    sensors = []

    # Always create the core sensors
    sensors.append(AreaOccupancyProbabilitySensor(coordinator, entry.entry_id))
    sensors.append(AreaOccupancyDecaySensor(coordinator, entry.entry_id))

    # Create prior sensors if history period is configured and greater than 0
    history_period = coordinator.config.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD)
    if history_period > 0:
        prior_sensor_classes = []

        if coordinator.config.get(CONF_MOTION_SENSORS):
            prior_sensor_classes.append(MotionPriorSensor)
        if coordinator.config.get(CONF_MEDIA_DEVICES):
            prior_sensor_classes.append(MediaPriorSensor)
        if coordinator.config.get(CONF_APPLIANCES):
            prior_sensor_classes.append(AppliancePriorSensor)
        if coordinator.config.get(CONF_DOOR_SENSORS):
            prior_sensor_classes.append(DoorPriorSensor)
        if coordinator.config.get(CONF_WINDOW_SENSORS):
            prior_sensor_classes.append(WindowPriorSensor)
        if coordinator.config.get(CONF_LIGHTS):
            prior_sensor_classes.append(LightPriorSensor)

        # If any prior sensors are being added, include the OccupancyPriorSensor
        if prior_sensor_classes:
            prior_sensor_classes.append(OccupancyPriorSensor)

        sensors.extend(
            cls(coordinator, entry_id=entry.entry_id) for cls in prior_sensor_classes
        )

    async_add_entities(sensors, update_before_add=True)
