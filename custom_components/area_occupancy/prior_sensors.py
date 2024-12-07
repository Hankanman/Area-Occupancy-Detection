"""Prior probability sensors for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import (
    PERCENTAGE,
    STATE_ON,
    STATE_OFF,
    STATE_PLAYING,
    STATE_PAUSED,
)
from homeassistant.core import callback
from homeassistant.util import dt as dt_util

from .base import AreaOccupancySensorBase
from .const import (
    NAME_MOTION_PRIOR_SENSOR,
    NAME_ENVIRONMENTAL_PRIOR_SENSOR,
    NAME_MEDIA_PRIOR_SENSOR,
    NAME_APPLIANCE_PRIOR_SENSOR,
    NAME_OCCUPANCY_PRIOR_SENSOR,
    ATTR_TOTAL_SAMPLES,
    ATTR_ACTIVE_SAMPLES,
    ATTR_SAMPLING_PERIOD,
)
from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


class PriorProbabilitySensorBase(AreaOccupancySensorBase, SensorEntity):
    """Base class for prior probability sensors."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        name: str,
    ) -> None:
        """Initialize the base prior probability sensor."""
        super().__init__(coordinator, entry_id)

        self._attr_has_entity_name = True
        self._attr_name = name
        self._attr_device_class = SensorDeviceClass.POWER_FACTOR
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._total_samples = 0
        self._active_samples = 0
        self._sampling_period = None

    def _format_percentage(self, value: float) -> float:
        """Format percentage to 4 decimal places."""
        return round(float(value * 100), 4)

    @property
    def native_value(self) -> float | None:
        """Return the prior probability as a percentage."""
        try:
            if self._total_samples == 0:
                return 0.0
            return self._format_percentage(self._active_samples / self._total_samples)
        except Exception as err:
            _LOGGER.error("Error calculating prior probability: %s", err)
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        return {
            ATTR_TOTAL_SAMPLES: self._total_samples,
            ATTR_ACTIVE_SAMPLES: self._active_samples,
            ATTR_SAMPLING_PERIOD: self._sampling_period,
        }

    async def _async_update_historical_data(
        self, entity_ids: list[str], process_state_func
    ) -> None:
        """Update historical data using recorder."""
        if not entity_ids:
            return

        try:
            start_time = dt_util.utcnow() - timedelta(
                days=self.coordinator.options_config["history_period"]
            )

            def get_history():
                return get_significant_states(
                    self.hass,
                    start_time,
                    dt_util.utcnow(),
                    entity_ids,
                )

            states = await get_instance(self.hass).async_add_executor_job(get_history)

            self._total_samples = 0
            self._active_samples = 0

            if states:
                for entity_id, entity_states in states.items():
                    for state in entity_states:
                        if state:  # Ensure state is not None
                            self._total_samples += 1
                            if process_state_func(state):
                                self._active_samples += 1

            self._sampling_period = (
                f"{self.coordinator.options_config['history_period']} days"
            )

        except Exception as err:
            _LOGGER.error("Error updating historical data: %s", err)


class MotionPriorSensor(PriorProbabilitySensorBase):
    """Sensor for motion prior probability."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize motion prior sensor."""
        super().__init__(coordinator, entry_id, NAME_MOTION_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("motion_prior")

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        await self._async_update_historical_data(
            self.coordinator.core_config["motion_sensors"],
            lambda state: state.state == STATE_ON,
        )


class EnvironmentalPriorSensor(PriorProbabilitySensorBase):
    """Sensor for environmental correlation prior probability."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize environmental prior sensor."""
        super().__init__(coordinator, entry_id, NAME_ENVIRONMENTAL_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("environmental_prior")

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        env_sensors = (
            self.coordinator.options_config.get("illuminance_sensors", [])
            + self.coordinator.options_config.get("humidity_sensors", [])
            + self.coordinator.options_config.get("temperature_sensors", [])
        )

        if env_sensors and self.coordinator.core_config.get("motion_sensors"):
            await self._async_analyze_environmental_correlation(
                self.coordinator.core_config["motion_sensors"][
                    0
                ],  # Use first motion sensor
                env_sensors,
            )

    async def _async_analyze_environmental_correlation(
        self, motion_sensor: str, env_sensors: list[str]
    ) -> None:
        """Analyze correlation between motion and environmental changes."""
        try:
            start_time = dt_util.utcnow() - timedelta(
                days=self.coordinator.options_config["history_period"]
            )

            def get_history():
                return get_significant_states(
                    self.hass,
                    start_time,
                    dt_util.utcnow(),
                    [motion_sensor] + env_sensors,
                )

            states = await get_instance(self.hass).async_add_executor_job(get_history)

            self._total_samples = 0
            self._active_samples = 0

            if not states:
                return

            motion_states = states.get(motion_sensor, [])
            window = timedelta(minutes=5)

            for motion_state in motion_states:
                if motion_state.state == STATE_ON:
                    self._total_samples += 1
                    motion_time = motion_state.last_changed

                    # Check environmental sensors for changes
                    for env_id in env_sensors:
                        env_states = states.get(env_id, [])
                        for i in range(1, len(env_states)):
                            try:
                                prev_value = float(env_states[i - 1].state)
                                curr_value = float(env_states[i].state)
                                if (
                                    abs(curr_value - prev_value) > 0.1 * prev_value
                                ):  # 10% change
                                    if (
                                        abs(motion_time - env_states[i].last_changed)
                                        < window
                                    ):
                                        self._active_samples += 1
                                        break
                            except (ValueError, TypeError):
                                continue

            self._sampling_period = (
                f"{self.coordinator.options_config['history_period']} days"
            )

        except Exception as err:
            _LOGGER.error("Error analyzing environmental correlation: %s", err)


class MediaPriorSensor(PriorProbabilitySensorBase):
    """Sensor for media device prior probability."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize media prior sensor."""
        super().__init__(coordinator, entry_id, NAME_MEDIA_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("media_prior")

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        await self._async_update_historical_data(
            self.coordinator.options_config.get("media_devices", []),
            lambda state: state.state in [STATE_PLAYING, STATE_PAUSED],
        )


class AppliancePriorSensor(PriorProbabilitySensorBase):
    """Sensor for appliance prior probability."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize appliance prior sensor."""
        super().__init__(coordinator, entry_id, NAME_APPLIANCE_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("appliance_prior")

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        await self._async_update_historical_data(
            self.coordinator.options_config.get("appliances", []),
            lambda state: state.state == STATE_ON,
        )


class OccupancyPriorSensor(PriorProbabilitySensorBase):
    """Sensor for occupancy prior probability."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize occupancy prior sensor."""
        super().__init__(coordinator, entry_id, NAME_OCCUPANCY_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("occupancy_prior")

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        area_name = self.coordinator.core_config["name"].lower().replace(" ", "_")
        await self._async_update_historical_data(
            [f"binary_sensor.{area_name}_occupancy_status"],
            lambda state: state.state == STATE_ON,
        )
