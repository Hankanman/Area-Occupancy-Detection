"""Prior probability sensors for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE
from homeassistant.util import dt as dt_util

from .base import AreaOccupancySensorBase
from .const import (
    NAME_MOTION_PRIOR_SENSOR,
    NAME_MEDIA_PRIOR_SENSOR,
    NAME_APPLIANCE_PRIOR_SENSOR,
    NAME_OCCUPANCY_PRIOR_SENSOR,
    ATTR_TOTAL_PERIOD,
    ATTR_PROB_GIVEN_TRUE,
    ATTR_PROB_GIVEN_FALSE,
    ATTR_LAST_UPDATED,
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


class PriorProbabilitySensorBase(AreaOccupancySensorBase, SensorEntity):
    """Base class for prior probability sensors."""

    def _sensor_specific_attributes(self) -> dict[str, Any]:
        """Return sensor-specific attributes."""
        return {
            ATTR_PROB_GIVEN_TRUE: self._prob_given_true,
            ATTR_PROB_GIVEN_FALSE: self._prob_given_false,
        }

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

        # Initialize with non-zero default probabilities
        self._prob_given_true = DEFAULT_PROB_GIVEN_TRUE
        self._prob_given_false = DEFAULT_PROB_GIVEN_FALSE
        self._last_calculation: datetime | None = None
        self._analyzer = HistoricalAnalysis(coordinator.hass)

    async def _calculate_prior(self) -> None:
        """Calculate prior probability."""
        if (
            self._last_calculation
            and dt_util.utcnow() - self._last_calculation < timedelta(hours=6)
        ):
            _LOGGER.debug("Prior calculation skipped, last calculation was recent.")
            return

        _LOGGER.debug("Starting prior probability calculation for %s", self.entity_id)
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(
            days=self.coordinator.options_config["history_period"]
        )

        self._prob_given_true, self._prob_given_false = (
            await self._analyzer.calculate_prior(self.entity_id, start_time, end_time)
        )
        self._last_calculation = dt_util.utcnow()
        _LOGGER.debug(
            "Prior probability calculation completed for %s: prob_given_true=%f, prob_given_false=%f",
            self.entity_id,
            self._prob_given_true,
            self._prob_given_false,
        )

    @property
    def native_value(self) -> float | None:
        """Return the prior probability as a percentage."""
        return round(self._prob_given_true * 100, 4)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        return {
            ATTR_TOTAL_PERIOD: str(
                timedelta(days=self.coordinator.options_config["history_period"])
            ),
            ATTR_PROB_GIVEN_TRUE: self._prob_given_true,
            ATTR_PROB_GIVEN_FALSE: self._prob_given_false,
            ATTR_LAST_UPDATED: (
                self._last_calculation.isoformat() if self._last_calculation else None
            ),
        }

    def _update_attributes(self) -> None:
        """Update the entity attributes with coordinator data."""
        if not self.coordinator.data:
            return
        self._attr_native_value = round(self._prob_given_true * 100, 4)
        self._attr_extra_state_attributes.update(self._shared_attributes)
        self._attr_extra_state_attributes.update(self._sensor_specific_attributes())
        self._attr_extra_state_attributes.update(
            {
                ATTR_TOTAL_PERIOD: str(
                    timedelta(days=self.coordinator.options_config["history_period"])
                ),
                ATTR_PROB_GIVEN_TRUE: self._prob_given_true,
                ATTR_PROB_GIVEN_FALSE: self._prob_given_false,
                ATTR_LAST_UPDATED: (
                    self._last_calculation.isoformat()
                    if self._last_calculation
                    else None
                ),
            }
        )

    def _update_entity_state(self, result: ProbabilityResult) -> None:
        """Update entity state from coordinator data."""
        self._attr_native_value = round(self._prob_given_true * 100, 4)
        self._attr_extra_state_attributes.update(self._shared_attributes)
        self._attr_extra_state_attributes.update(self._sensor_specific_attributes())


class MotionPriorSensor(PriorProbabilitySensorBase):
    """Sensor for motion prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize motion prior sensor."""
        super().__init__(coordinator, entry_id, NAME_MOTION_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("motion_prior")
        # Use motion-specific defaults
        self._prob_given_true = MOTION_PROB_GIVEN_TRUE
        self._prob_given_false = MOTION_PROB_GIVEN_FALSE

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        _LOGGER.debug("MotionPriorSensor added to hass: %s", self.entity_id)
        await self._calculate_prior()


class MediaPriorSensor(PriorProbabilitySensorBase):
    """Sensor for media device prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize media prior sensor."""
        super().__init__(coordinator, entry_id, NAME_MEDIA_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("media_prior")
        # Use media-specific defaults
        self._prob_given_true = MEDIA_PROB_GIVEN_TRUE
        self._prob_given_false = MEDIA_PROB_GIVEN_FALSE

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        _LOGGER.debug("MediaPriorSensor added to hass: %s", self.entity_id)
        await self._calculate_prior()


class AppliancePriorSensor(PriorProbabilitySensorBase):
    """Sensor for appliance prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize appliance prior sensor."""
        super().__init__(coordinator, entry_id, NAME_APPLIANCE_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("appliance_prior")
        # Use appliance-specific defaults
        self._prob_given_true = APPLIANCE_PROB_GIVEN_TRUE
        self._prob_given_false = APPLIANCE_PROB_GIVEN_FALSE

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        _LOGGER.debug("AppliancePriorSensor added to hass: %s", self.entity_id)
        await self._calculate_prior()


class OccupancyPriorSensor(PriorProbabilitySensorBase):
    """Sensor for occupancy prior probability."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize occupancy prior sensor."""
        super().__init__(coordinator, entry_id, NAME_OCCUPANCY_PRIOR_SENSOR)
        self._attr_unique_id = self._format_unique_id("occupancy_prior")

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        _LOGGER.debug("OccupancyPriorSensor added to hass: %s", self.entity_id)
        await self._calculate_prior()
