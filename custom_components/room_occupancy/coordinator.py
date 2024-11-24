"""Coordinator for Room Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .calculations import DecayConfig, ProbabilityCalculator
from .const import (
    CONF_DECAY_ENABLED,
    CONF_DECAY_TYPE,
    CONF_DECAY_WINDOW,
    CONF_DEVICE_STATES,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MOTION_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_TYPE,
    DEFAULT_DECAY_WINDOW,
    DOMAIN,
    ProbabilityResult,
    SensorId,
    SensorStates,
)

_LOGGER = logging.getLogger(__name__)


class RoomOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Class to manage fetching room occupancy data."""

    def __init__(
        self, hass: HomeAssistant, entry_id: str, config: dict[str, Any]
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),
        )
        self.entry_id = entry_id
        self.config = config
        self._motion_timestamps: dict[SensorId, datetime] = {}
        self._sensor_states: SensorStates = {}
        self._unsubscribe_handlers: list[callable] = []
        self._calculator = self._create_calculator()

        # Initialize states
        for entity_id in self._get_all_configured_sensors():
            state = hass.states.get(entity_id)
            if state and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                self._sensor_states[entity_id] = {
                    "state": state.state,
                    "last_changed": state.last_changed.isoformat(),
                    "availability": True,
                }

        self._setup_state_listeners()

    def _create_calculator(self) -> ProbabilityCalculator:
        """Create probability calculator with current configuration."""
        return ProbabilityCalculator(
            motion_sensors=self.config.get(CONF_MOTION_SENSORS, []),
            illuminance_sensors=self.config.get(CONF_ILLUMINANCE_SENSORS, []),
            humidity_sensors=self.config.get(CONF_HUMIDITY_SENSORS, []),
            temperature_sensors=self.config.get(CONF_TEMPERATURE_SENSORS, []),
            device_states=self.config.get(CONF_DEVICE_STATES, []),
            decay_config=DecayConfig(
                enabled=self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
                window=self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
                type=self.config.get(CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE),
            ),
        )

    def _get_all_configured_sensors(self) -> list[str]:
        """Get list of all configured sensor entity IDs."""
        sensors = []
        sensors.extend(self.config.get(CONF_MOTION_SENSORS, []))
        sensors.extend(self.config.get(CONF_ILLUMINANCE_SENSORS, []))
        sensors.extend(self.config.get(CONF_HUMIDITY_SENSORS, []))
        sensors.extend(self.config.get(CONF_TEMPERATURE_SENSORS, []))
        sensors.extend(self.config.get(CONF_DEVICE_STATES, []))
        return sensors

    def unsubscribe(self) -> None:
        """Unsubscribe from all registered events."""
        while self._unsubscribe_handlers:
            self._unsubscribe_handlers.pop()()

    def _setup_state_listeners(self) -> None:
        """Set up state change listeners for all configured sensors."""

        @callback
        def async_state_changed(event) -> None:
            """Handle sensor state changes."""
            entity_id: str = event.data["entity_id"]
            new_state = event.data["new_state"]

            if not new_state or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                if entity_id in self._sensor_states:
                    self._sensor_states[entity_id]["availability"] = False
                return

            try:
                # For numeric sensors, validate the value
                if any(
                    entity_id in self.config.get(sensor_type, [])
                    for sensor_type in [
                        CONF_ILLUMINANCE_SENSORS,
                        CONF_HUMIDITY_SENSORS,
                        CONF_TEMPERATURE_SENSORS,
                    ]
                ):
                    float(new_state.state)

                # Update state if validation passed
                self._sensor_states[entity_id] = {
                    "state": new_state.state,
                    "last_changed": new_state.last_changed.isoformat(),
                    "availability": True,
                }

                # Update motion timestamps if needed
                if (
                    entity_id in self.config.get(CONF_MOTION_SENSORS, [])
                    and new_state.state == STATE_ON
                ):
                    self._motion_timestamps[entity_id] = dt_util.utcnow()

                # Trigger update without full state refresh
                self.async_set_updated_data(self._get_calculated_data())

            except ValueError:
                _LOGGER.debug(
                    "Sensor %s provided invalid numeric value: %s",
                    entity_id,
                    new_state.state,
                )

        self.unsubscribe()
        # Track all configured sensors
        self._unsubscribe_handlers.append(
            async_track_state_change_event(
                self.hass,
                self._get_all_configured_sensors(),
                async_state_changed,
            )
        )

    async def _async_update_data(self) -> ProbabilityResult:
        """Periodic update - used as fallback and for decay updates."""
        return self._get_calculated_data()

    def _get_calculated_data(self) -> ProbabilityResult:
        """Calculate current occupancy data using the probability calculator."""
        return self._calculator.calculate(
            self._sensor_states,
            self._motion_timestamps,
        )
