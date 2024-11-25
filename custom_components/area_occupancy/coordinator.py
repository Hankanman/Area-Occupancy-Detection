"""Coordinator for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from collections import deque

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
    CONF_THRESHOLD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_TYPE,
    DEFAULT_DECAY_WINDOW,
    DOMAIN,
    ProbabilityResult,
    SensorId,
    SensorStates,
)

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Class to manage fetching area occupancy data."""

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

        # New tracking attributes
        self._probability_history = deque(maxlen=12)  # 1 hour of 5-minute readings
        self._last_occupied: datetime | None = None
        self._last_state_change: datetime | None = None
        self._occupancy_history = deque(
            [False] * 288, maxlen=288
        )  # 24 hours of 5-minute readings

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

    def _update_historical_data(self, probability: float, is_occupied: bool) -> None:
        """Update historical tracking data."""
        now = dt_util.utcnow()

        # Update probability history
        self._probability_history.append(probability)

        # Update occupancy tracking
        self._occupancy_history.append(is_occupied)

        # Update last occupied time
        if is_occupied:
            self._last_occupied = now

        # Update state change time if state changed
        current_state = is_occupied
        if not self._last_state_change or (
            bool(self._occupancy_history[-2]) != current_state
        ):
            self._last_state_change = now

    def _get_historical_metrics(self) -> dict[str, float]:
        """Calculate historical metrics."""
        now = dt_util.utcnow()

        # Calculate moving average
        moving_avg = (
            sum(self._probability_history) / len(self._probability_history)
            if self._probability_history
            else 0.0
        )

        # Calculate rate of change (per hour)
        rate_of_change = 0.0
        if len(self._probability_history) >= 2:
            change = self._probability_history[-1] - self._probability_history[0]
            time_window = len(self._probability_history) * 5  # 5 minutes per reading
            rate_of_change = (change / time_window) * 60  # Convert to per hour

        # Calculate occupancy rate
        occupancy_rate = (
            sum(1 for x in self._occupancy_history if x) / len(self._occupancy_history)
            if self._occupancy_history
            else 0.0
        )

        # Calculate state duration
        state_duration = 0.0
        if self._last_state_change:
            state_duration = (now - self._last_state_change).total_seconds()

        return {
            "moving_average": moving_avg,
            "rate_of_change": rate_of_change,
            "occupancy_rate": occupancy_rate,
            "state_duration": state_duration,
            "min_probability": (
                min(self._probability_history) if self._probability_history else 0.0
            ),
            "max_probability": (
                max(self._probability_history) if self._probability_history else 0.0
            ),
            "last_occupied": (
                self._last_occupied.isoformat() if self._last_occupied else None
            ),
        }

    def _get_calculated_data(self) -> ProbabilityResult:
        """Calculate current occupancy data using the probability calculator."""
        base_result = self._calculator.calculate(
            self._sensor_states,
            self._motion_timestamps,
        )

        # Update historical data
        self._update_historical_data(
            base_result["probability"],
            base_result["probability"] >= self.config.get(CONF_THRESHOLD, 0.5),
        )

        # Add historical metrics to result
        historical_metrics = self._get_historical_metrics()
        base_result.update(historical_metrics)

        return base_result
