"""Base classes for Area Occupancy sensors with shared attribute logic."""

from __future__ import annotations

from typing import Any, Final
import logging

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import (
    PERCENTAGE,
)
from homeassistant.core import callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.helpers.entity import Entity

from .const import (
    ATTR_ACTIVE_TRIGGERS,
    ATTR_CONFIDENCE_SCORE,
    ATTR_DECAY_STATUS,
    ATTR_PRIOR_PROBABILITY,
    ATTR_PROBABILITY,
    ATTR_SENSOR_AVAILABILITY,
    ATTR_SENSOR_PROBABILITIES,
    ATTR_LAST_OCCUPIED,
    ATTR_STATE_DURATION,
    ATTR_OCCUPANCY_RATE,
    ATTR_MOVING_AVERAGE,
    ATTR_RATE_OF_CHANGE,
    ATTR_MIN_PROBABILITY,
    ATTR_MAX_PROBABILITY,
    ATTR_THRESHOLD,
    ATTR_WINDOW_SIZE,
    ATTR_MEDIA_STATES,
    ATTR_APPLIANCE_STATES,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    NAME_BINARY_SENSOR,
    NAME_PROBABILITY_SENSOR,
    CONF_AREA_ID,
)
from .types import ProbabilityResult
from .coordinator import AreaOccupancyCoordinator


_LOGGER = logging.getLogger(__name__)

ROUNDING_PRECISION: Final = 2


class AreaOccupancyEntity(CoordinatorEntity[AreaOccupancyCoordinator], Entity):
    """Base entity for Area Occupancy."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, name: str) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self._attr_name = name
        self._attr_unique_id = f"{coordinator.entry_id}_{name}"
        self._attr_device_info = coordinator.device_info
        self._attr_extra_state_attributes = {}  # Initialize empty attributes dict

    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        try:
            self._update_attributes()
            self.async_write_ha_state()
        except Exception as err:  # pylint: disable=broad-except
            self.coordinator.logger.error("Error handling coordinator update: %s", err)

    def _update_attributes(self) -> None:
        """Update the entity attributes with coordinator data."""
        raise NotImplementedError("Subclasses must implement _update_attributes")


class AreaOccupancySensorBase(AreaOccupancyEntity):
    """Base class for area occupancy sensors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the base sensor."""
        super().__init__(coordinator, coordinator.core_config["name"])

        # Entity attributes
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._entry_id = entry_id
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

    @property
    def name(self) -> str | None:
        """Return the display name of the sensor."""
        # Only set default name for main sensors
        if isinstance(
            self, (AreaOccupancyBinarySensor, AreaOccupancyProbabilitySensor)
        ):
            return (
                NAME_BINARY_SENSOR
                if isinstance(self, AreaOccupancyBinarySensor)
                else NAME_PROBABILITY_SENSOR
            )
        # Let prior sensors use their own names
        return self._attr_name

    @property
    def _shared_attributes(self) -> dict[str, Any]:
        """Return attributes common to all area occupancy sensors."""
        if not self.coordinator.data:
            return {}

        try:
            data: ProbabilityResult = self.coordinator.data

            def format_percentage(value: float) -> float:
                """Format percentage values consistently."""
                return self._format_float(value * 100)

            return {
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
                ATTR_MEDIA_STATES: data.get("device_states", {}).get(
                    "media_states", {}
                ),
                ATTR_APPLIANCE_STATES: data.get("device_states", {}).get(
                    "appliance_states", {}
                ),
            }

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error formatting shared attributes: %s", err)
            return {}

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        # Get motion sensors configuration
        motion_sensors = self.coordinator.core_config.get("motion_sensors", [])

        # Check if we have motion sensors configured
        if not motion_sensors:
            _LOGGER.debug("%s: No motion sensors configured", self.entity_id)
            return False

        # Check overall coordinator health
        if not self.coordinator.last_update_success:
            _LOGGER.debug("%s: Coordinator update failed", self.entity_id)
            return False

        # Check if we have sensor data
        if not self.coordinator.data:
            _LOGGER.debug("%s: No data available from coordinator", self.entity_id)
            return False

        # Check motion sensor states directly from coordinator's sensor states
        motion_states = {
            sensor_id: self.coordinator.get_sensor_states().get(sensor_id, {})
            for sensor_id in motion_sensors
        }

        is_available = any(
            state.get("availability", False) for state in motion_states.values()
        )

        return is_available

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        try:
            attributes = self._shared_attributes
            specific_attributes = self._sensor_specific_attributes()
            if specific_attributes:
                attributes.update(specific_attributes)

            # Add configuration info to attributes using options_config
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

    def _sensor_specific_attributes(self) -> dict[str, Any]:
        """Return attributes specific to this sensor type.

        Must be implemented by child classes.
        """
        raise NotImplementedError

    def _update_entity_state(self, result: ProbabilityResult) -> None:
        """Update entity state from coordinator data."""
        raise NotImplementedError


class AreaOccupancyBinarySensor(AreaOccupancySensorBase, BinarySensorEntity):
    """Binary sensor for area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        threshold: float,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator, entry_id)

        self._attr_unique_id = self._format_unique_id("occupancy")
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._threshold = threshold

    def _sensor_specific_attributes(self) -> dict[str, Any]:
        """Return attributes specific to binary occupancy sensor."""
        if not self.coordinator.data:
            return {}

        data = self.coordinator.data
        return {
            ATTR_THRESHOLD: self._format_float(self._threshold),
            ATTR_LAST_OCCUPIED: data.get("last_occupied"),
            ATTR_STATE_DURATION: self._format_float(
                data.get("state_duration", 0.0) / 60  # Convert to minutes
            ),
            ATTR_OCCUPANCY_RATE: self._format_float(
                data.get("occupancy_rate", 0.0) * 100
            ),
        }

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

    def _update_attributes(self) -> None:
        """Update the entity attributes with coordinator data."""
        if not self.coordinator.data:
            return
        result = self.coordinator.data
        self._attr_is_on = result["probability"] >= self._threshold
        self._attr_extra_state_attributes.update(self._shared_attributes)
        self._attr_extra_state_attributes.update(self._sensor_specific_attributes())


class AreaOccupancyProbabilitySensor(AreaOccupancySensorBase, SensorEntity):
    """Probability sensor for area occupancy."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the probability sensor."""
        super().__init__(coordinator, entry_id)

        self._attr_unique_id = self._format_unique_id("probability")
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    def _sensor_specific_attributes(self) -> dict[str, Any]:
        """Return attributes specific to probability sensor."""
        if not self.coordinator.data:
            return {}

        data = self.coordinator.data
        return {
            ATTR_MOVING_AVERAGE: self._format_float(
                data.get("moving_average", 0.0) * 100
            ),
            ATTR_RATE_OF_CHANGE: self._format_float(
                data.get("rate_of_change", 0.0) * 100
            ),
            ATTR_MIN_PROBABILITY: self._format_float(
                data.get("min_probability", 0.0) * 100
            ),
            ATTR_MAX_PROBABILITY: self._format_float(
                data.get("max_probability", 0.0) * 100
            ),
            ATTR_WINDOW_SIZE: "1 hour",  # Current window size for moving average
        }

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

    def _update_attributes(self) -> None:
        """Update the entity attributes with coordinator data."""
        if not self.coordinator.data:
            return
        result = self.coordinator.data
        self._attr_native_value = self._format_float(result["probability"] * 100)
        self._attr_extra_state_attributes.update(self._shared_attributes)
        self._attr_extra_state_attributes.update(self._sensor_specific_attributes())
