"""Sensor platform for Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    CONF_HISTORY_PERIOD,
    DEFAULT_HISTORY_PERIOD,
    DOMAIN,
    NAME_DECAY_SENSOR,
    NAME_PRIORS_SENSOR,
    NAME_PROBABILITY_SENSOR,
    ROUNDING_PRECISION,
)
from .coordinator import AreaOccupancyCoordinator
from .types import EntityType, PriorsAttributes, ProbabilityAttributes, ProbabilityState

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
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_has_entity_name = True
        self._attr_should_poll = False
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
        self._attr_unique_id = (
            f"{entry_id}_{NAME_PRIORS_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the overall occupancy prior as the state."""
        try:
            if not self.coordinator.prior_state:
                return None

            # Return the overall prior directly from prior_state
            return self.coordinator.prior_state.overall_prior * 100

        except (TypeError, ValueError, AttributeError, KeyError) as err:
            _LOGGER.error("Error calculating priors: %s", err)
            return None

    @property
    def extra_state_attributes(self) -> PriorsAttributes:
        """Return all prior probabilities as attributes."""
        try:
            if not self.coordinator.prior_state:
                return {}

            prior_state = self.coordinator.prior_state

            # Initialize PriorsAttributes with empty dict
            attributes: PriorsAttributes = {}

            # Map sensor types to corresponding prior values from prior_state
            type_prior_map = {
                EntityType.MOTION: prior_state.motion_prior,
                EntityType.MEDIA: prior_state.media_prior,
                EntityType.APPLIANCE: prior_state.appliance_prior,
                EntityType.DOOR: prior_state.door_prior,
                EntityType.WINDOW: prior_state.window_prior,
                EntityType.LIGHT: prior_state.light_prior,
            }

            # Add priors that have non-zero values
            for attr_name, prior_value in type_prior_map.items():
                if prior_value > 0:
                    # Use the explicit string value of the StrEnum member as the key
                    attributes[attr_name.value] = (
                        f"Prior: {round(prior_value * 100, 1)}%"
                    )

            # Add metadata attributes
            last_updated_ts = self.coordinator.last_prior_update
            attributes["last_updated"] = last_updated_ts if last_updated_ts else "Never"
            attributes["total_period"] = f"{prior_state.analysis_period} days"

        except (TypeError, ValueError, AttributeError, KeyError) as err:
            _LOGGER.error("Error getting prior attributes: %s", err)
            return {}
        else:
            return attributes


class AreaOccupancyProbabilitySensor(AreaOccupancySensorBase):
    """Probability sensor for current area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the probability sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_PROBABILITY_SENSOR
        self._attr_unique_id = (
            f"{entry_id}_{NAME_PROBABILITY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = None

    @property
    def native_value(self) -> float | None:
        """Return the current occupancy probability as a percentage."""
        if not self.coordinator.data:
            return 0.0

        try:
            # Use the new coordinator property
            return format_float(self.coordinator.probability * 100)
        except AttributeError:
            _LOGGER.error("Coordinator missing probability attribute")
            return 0.0

    @property
    def extra_state_attributes(self) -> ProbabilityAttributes:
        """Return entity specific state attributes."""
        if not self.coordinator.data:
            return {}
        try:
            data: ProbabilityState = self.coordinator.data

            # Create formatted probability entries with details
            sensor_probabilities = set()  # Use a set instead of dict
            active_triggers = []

            for entity_id, prob_details in data.sensor_probabilities.items():
                friendly_name = (
                    state.attributes.get("friendly_name", entity_id)
                    if (state := self.hass.states.get(entity_id))
                    else entity_id
                )

                formatted_entry = (
                    f"{friendly_name} | "
                    f"W: {format_float(prob_details['weight'])} | "
                    f"P: {format_float(prob_details['probability'])} | "
                    f"WP: {format_float(prob_details['weighted_probability'])}"
                )
                sensor_probabilities.add(formatted_entry)
                active_triggers.append(friendly_name)

            return {
                "active_triggers": active_triggers,
                "sensor_probabilities": sensor_probabilities,
                "threshold": f"{self.coordinator.threshold * 100}%",
            }
        except (TypeError, AttributeError, KeyError):
            _LOGGER.exception("Error getting probability attributes: %s")
            return {}


class AreaOccupancyDecaySensor(AreaOccupancySensorBase):
    """Decay status sensor for area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the decay sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_name = NAME_DECAY_SENSOR
        self._attr_unique_id = (
            f"{entry_id}_{NAME_DECAY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return the decay status as a percentage."""
        if not self.coordinator.data:
            return 0.0

        try:
            # decay_status is already stored as 0.0 to 100.0
            return format_float(self.coordinator.data.decay_status)
        except AttributeError:
            _LOGGER.error("Coordinator data missing decay_status attribute")
            return 0.0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Area Occupancy sensors based on a config entry."""
    coordinator: AreaOccupancyCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]

    sensors = [
        AreaOccupancyProbabilitySensor(coordinator, entry.entry_id),
        AreaOccupancyDecaySensor(coordinator, entry.entry_id),
    ]

    # Create priors sensor if history period is configured and greater than 0
    history_period = coordinator.config.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD)
    if history_period > 0:
        sensors.append(PriorsSensor(coordinator, entry.entry_id))

    async_add_entities(sensors, update_before_add=True)


def format_float(value: float) -> float:
    """Format float to consistently show 2 decimal places."""
    try:
        return round(float(value), ROUNDING_PRECISION)
    except (ValueError, TypeError):
        return 0.0
