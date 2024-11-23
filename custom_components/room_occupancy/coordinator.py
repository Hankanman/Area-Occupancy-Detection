"""Data coordinator for Room Occupancy Detection integration."""
from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
import numpy as np

from .const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_DEVICE_STATES,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_TYPE,
)
from .probability import BayesianProbability

_LOGGER = logging.getLogger(__name__)

class RoomOccupancyCoordinator(DataUpdateCoordinator):
    """Class to manage fetching Room Occupancy data."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        config: dict[str, Any],
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=10),
        )
        self.entry_id = entry_id
        self.config = config
        self.bayesian = BayesianProbability()
        self._sensor_states: dict[str, State] = {}
        self._last_trigger_times: dict[str, datetime] = {}
        self._setup_sensor_tracking()

    def _setup_sensor_tracking(self) -> None:
        """Set up state tracking for all configured sensors."""
        sensors_to_track = []
        
        # Add all configured sensors to tracking
        for sensor_type in [
            CONF_MOTION_SENSORS,
            CONF_ILLUMINANCE_SENSORS,
            CONF_HUMIDITY_SENSORS,
            CONF_TEMPERATURE_SENSORS,
            CONF_DEVICE_STATES,
        ]:
            if sensor_type in self.config:
                sensors_to_track.extend(self.config[sensor_type])

        # Set up state tracking for each sensor
        @callback
        def async_state_changed_listener(event) -> None:
            """Handle sensor state changes."""
            self._sensor_states[event.data["entity_id"]] = event.data["new_state"]
            self._last_trigger_times[event.data["entity_id"]] = datetime.now()
            self.async_set_updated_data(self._calculate_occupancy())

        async_track_state_change_event(
            self.hass,
            sensors_to_track,
            async_state_changed_listener,
        )

        # Initialize current states
        for entity_id in sensors_to_track:
            state = self.hass.states.get(entity_id)
            if state is not None:
                self._sensor_states[entity_id] = state
                self._last_trigger_times[entity_id] = datetime.now()

    def _calculate_decay(self, sensor_id: str, last_trigger: datetime) -> float:
        """Calculate the decay factor for a sensor based on time since last trigger."""
        if not self.config.get(CONF_DECAY_ENABLED, True):
            return 1.0

        time_diff = (datetime.now() - last_trigger).total_seconds()
        decay_window = self.config.get(CONF_DECAY_WINDOW, 600)  # Default 10 minutes

        if time_diff >= decay_window:
            return 0.0

        if self.config.get(CONF_DECAY_TYPE, "linear") == "linear":
            return 1.0 - (time_diff / decay_window)
        else:  # exponential
            return np.exp(-3.0 * time_diff / decay_window)

    def _get_sensor_probability(self, sensor_id: str, state: State) -> float:
        """Calculate the probability contribution from a single sensor."""
        if state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return 0.0

        # Apply different probability calculations based on sensor type
        if sensor_id in self.config.get(CONF_MOTION_SENSORS, []):
            prob = 0.95 if state.state == STATE_ON else 0.05
        elif sensor_id in self.config.get(CONF_ILLUMINANCE_SENSORS, []):
            try:
                value = float(state.state)
                # Simple threshold-based probability for illuminance
                prob = 0.7 if value > 10 else 0.3
            except ValueError:
                return 0.0
        elif sensor_id in self.config.get(CONF_HUMIDITY_SENSORS, []):
            try:
                # Compare against historical average if available
                value = float(state.state)
                prob = 0.6  # Default probability for now
            except ValueError:
                return 0.0
        elif sensor_id in self.config.get(CONF_TEMPERATURE_SENSORS, []):
            try:
                # Compare against historical average if available
                value = float(state.state)
                prob = 0.6  # Default probability for now
            except ValueError:
                return 0.0
        elif sensor_id in self.config.get(CONF_DEVICE_STATES, []):
            prob = 0.8 if state.state == STATE_ON else 0.2
        else:
            return 0.0

        # Apply decay factor
        decay = self._calculate_decay(sensor_id, self._last_trigger_times[sensor_id])
        return prob * decay

    def _calculate_occupancy(self) -> dict[str, Any]:
        """Calculate room occupancy probability using Bayesian inference."""
        sensor_probabilities = {}
        active_triggers = []
        sensor_availability = {}

        # Calculate individual sensor probabilities
        for sensor_id, state in self._sensor_states.items():
            probability = self._get_sensor_probability(sensor_id, state)
            sensor_probabilities[sensor_id] = probability
            
            if probability > 0.5:  # Consider as active trigger
                active_triggers.append(sensor_id)
            
            sensor_availability[sensor_id] = state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            )

        # Calculate final probability using Bayesian inference
        if not sensor_probabilities:
            final_probability = 0.0
        else:
            final_probability = self.bayesian.calculate_probability(
                list(sensor_probabilities.values())
            )

        # Calculate confidence score based on sensor availability and diversity
        available_sensors = sum(sensor_availability.values())
        total_sensors = len(sensor_availability)
        confidence_score = available_sensors / total_sensors if total_sensors > 0 else 0.0

        return {
            "probability": final_probability,
            "sensor_probabilities": sensor_probabilities,
            "active_triggers": active_triggers,
            "decay_status": {
                sensor_id: self._calculate_decay(
                    sensor_id, self._last_trigger_times[sensor_id]
                )
                for sensor_id in self._sensor_states
            },
            "confidence_score": confidence_score,
            "sensor_availability": sensor_availability,
        }

    async def _async_update_data(self) -> dict[str, Any]:
        """Update data via library."""
        return self._calculate_occupancy()
