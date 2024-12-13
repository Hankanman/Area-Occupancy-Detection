"""Sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Final

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.core import callback

from .const import (
    DOMAIN,
    NAME_PROBABILITY_SENSOR,
    NAME_MOTION_PRIOR_SENSOR,
    NAME_MEDIA_PRIOR_SENSOR,
    NAME_APPLIANCE_PRIOR_SENSOR,
    NAME_OCCUPANCY_PRIOR_SENSOR,
    ATTR_TOTAL_PERIOD,
    ATTR_PROB_GIVEN_TRUE,
    ATTR_PROB_GIVEN_FALSE,
    ATTR_LAST_UPDATED,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    ATTR_ACTIVE_TRIGGERS,
    ATTR_SENSOR_PROBABILITIES,
    CONF_AREA_ID,
    NAME_DECAY_SENSOR,
)
from .coordinator import AreaOccupancyCoordinator
from .historical_analysis import HistoricalAnalysis
from .probabilities import (
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
        name: str,
    ) -> None:
        """Initialize the base sensor."""
        super().__init__(coordinator)

        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._attr_name = name
        self._attr_unique_id = (
            f"{DOMAIN}_{coordinator.core_config[CONF_AREA_ID]}_{name}"
        )
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

    @callback
    def _format_unique_id(self, sensor_type: str) -> str:
        """Format the unique id consistently."""
        return f"{DOMAIN}_{self.coordinator.core_config[CONF_AREA_ID]}_{sensor_type}"


class PriorProbabilitySensorBase(AreaOccupancySensorBase, SensorEntity):
    """Base class for prior probability sensors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        name: str,
    ) -> None:
        """Initialize the base prior probability sensor."""
        super().__init__(coordinator, entry_id, name)

        self._attr_has_entity_name = True
        self._attr_name = name
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

        # Initialize with non-zero default probabilities
        self._prob_given_true = DEFAULT_PROB_GIVEN_TRUE
        self._prob_given_false = DEFAULT_PROB_GIVEN_FALSE
        self._last_calculation: datetime | None = None
        self._analyzer = HistoricalAnalysis(coordinator.hass)

    @property
    def native_value(self) -> float | None:
        """Return the prior probability as a percentage."""
        return round(self._prob_given_true * 100, 4)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        attributes = {
            ATTR_TOTAL_PERIOD: str(
                timedelta(days=self.coordinator.options_config["history_period"])
            ),
            ATTR_PROB_GIVEN_TRUE: self._prob_given_true,
            ATTR_PROB_GIVEN_FALSE: self._prob_given_false,
            ATTR_LAST_UPDATED: (
                self._last_calculation.isoformat() if self._last_calculation else None
            ),
        }
        return attributes

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        _LOGGER.debug("%s added to hass: %s", self.__class__.__name__, self.entity_id)

        # Schedule calculation in background
        self.hass.async_create_task(self._calculate_prior())
        self.async_write_ha_state()

    async def _calculate_prior(self) -> None:
        """Calculate prior probability."""
        if (
            self._last_calculation
            and dt_util.utcnow() - self._last_calculation < timedelta(hours=6)
        ):
            _LOGGER.debug("Prior calculation skipped, last calculation was recent.")
            return

        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(
            days=self.coordinator.options_config["history_period"]
        )

        self._prob_given_true, self._prob_given_false = (
            await self._analyzer.calculate_prior(self.entity_id, start_time, end_time)
        )
        self._last_calculation = dt_util.utcnow()


class MotionPriorSensor(PriorProbabilitySensorBase):
    """Sensor for motion prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize motion prior sensor."""
        super().__init__(coordinator, entry_id, NAME_MOTION_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("motion_prior")
        # Use motion-specific defaults
        self._prob_given_true = MOTION_PROB_GIVEN_TRUE
        self._prob_given_false = MOTION_PROB_GIVEN_FALSE


class MediaPriorSensor(PriorProbabilitySensorBase):
    """Sensor for media device prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize media prior sensor."""
        super().__init__(coordinator, entry_id, NAME_MEDIA_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("media_prior")
        # Use media-specific defaults
        self._prob_given_true = MEDIA_PROB_GIVEN_TRUE
        self._prob_given_false = MEDIA_PROB_GIVEN_FALSE


class AppliancePriorSensor(PriorProbabilitySensorBase):
    """Sensor for appliance prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize appliance prior sensor."""
        super().__init__(coordinator, entry_id, NAME_APPLIANCE_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("appliance_prior")
        # Use appliance-specific defaults
        self._prob_given_true = APPLIANCE_PROB_GIVEN_TRUE
        self._prob_given_false = APPLIANCE_PROB_GIVEN_FALSE


class OccupancyPriorSensor(PriorProbabilitySensorBase):
    """Sensor for occupancy prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize occupancy prior sensor."""
        super().__init__(coordinator, entry_id, NAME_OCCUPANCY_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("occupancy_prior")


class AreaOccupancyProbabilitySensor(AreaOccupancySensorBase):
    """Probability sensor for area occupancy."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize the probability sensor."""
        super().__init__(coordinator, entry_id, NAME_PROBABILITY_SENSOR)
        self._attr_unique_id = self._format_unique_id("probability")
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> float | None:
        """Return the native value of the sensor."""
        try:
            if not self.coordinator.data:
                return None
            probability = self.coordinator.data.get("probability", 0.0)
            return self._format_float(probability * 100)
        except Exception as err:  # pylint: disable=broad-except
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
                """Format percentage values consistently."""
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
                    self.hass.states.get(k).attributes.get(
                        "friendly_name", k
                    ): format_percentage(v)
                    for k, v in data.get("sensor_probabilities", {}).items()
                    if self.hass.states.get(k)
                },
            }
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error getting probability attributes: %s", err)
            return {}


class AreaOccupancyDecaySensor(AreaOccupancySensorBase):
    """Decay status sensor for area occupancy."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize the decay sensor."""
        super().__init__(coordinator, entry_id, NAME_DECAY_SENSOR)
        self._attr_unique_id = self._format_unique_id("decay")
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> float | None:
        """Return the average decay value."""
        try:
            if not self.coordinator.data or "decay_status" not in self.coordinator.data:
                _LOGGER.debug("No decay data available")
                return 0.0

            decay_values = [
                v
                for v in self.coordinator.data["decay_status"].values()
                if v is not None
            ]

            if not decay_values:
                _LOGGER.debug("No valid decay values found")
                return 0.0

            average_decay = sum(decay_values) / len(decay_values)
            return self._format_float(average_decay * 100)

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error calculating decay value: %s", err)
            return 0.0

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
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

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error getting decay attributes: %s", err)
            return {}


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

        entities = [
            AreaOccupancyProbabilitySensor(coordinator, entry.entry_id),
            AreaOccupancyDecaySensor(coordinator, entry.entry_id),
        ]

        # Create prior sensors if history period is configured
        if coordinator.options_config.get("history_period"):
            prior_sensor_classes = [
                MotionPriorSensor,
                MediaPriorSensor,
                AppliancePriorSensor,
                OccupancyPriorSensor,
            ]
            entities.extend(
                cls(coordinator, entry.entry_id) for cls in prior_sensor_classes
            )

        async_add_entities(entities, False)

    except Exception as err:
        _LOGGER.error("Error setting up sensors: %s", err)
        raise HomeAssistantError(
            f"Failed to set up Area Occupancy sensors: {err}"
        ) from err
