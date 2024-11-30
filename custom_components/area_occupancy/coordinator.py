"""Coordinator for Area Occupancy Detection."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any
from collections import deque

import yaml
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError

from .calculations import DecayConfig, ProbabilityCalculator
from .const import (
    CONF_DECAY_ENABLED,
    CONF_DECAY_TYPE,
    CONF_DECAY_WINDOW,
    CONF_APPLIANCES,
    CONF_MEDIA_DEVICES,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MOTION_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_TYPE,
    DEFAULT_DECAY_WINDOW,
    DOMAIN,
    PROBABILITY_CONFIG_FILE,
    HISTORY_STORAGE_FILE,
    STORAGE_VERSION,
    ProbabilityResult,
    SensorId,
    SensorStates,
    HistoryStorage,
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
        self._history_data = self._load_history()

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
        self._setup_history_update()

    def _load_base_config(self) -> dict[str, Any]:
        """Load the base probability configuration."""
        config_path = os.path.join(os.path.dirname(__file__), PROBABILITY_CONFIG_FILE)
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except (yaml.YAMLError, OSError) as err:
            _LOGGER.error("Failed to load base probability config: %s", err)
            raise HomeAssistantError(
                "Failed to load probability configuration"
            ) from err

    def _load_history(self) -> HistoryStorage:
        """Load or create the history storage file."""
        storage_path = self.hass.config.path(HISTORY_STORAGE_FILE)

        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r", encoding="utf-8") as file:
                    data = yaml.safe_load(file)
                    if data and isinstance(data, dict):
                        return data
            except (yaml.YAMLError, OSError) as err:
                _LOGGER.error("Failed to load history storage: %s", err)

        # Create new storage if file doesn't exist or is invalid
        return {
            "version": STORAGE_VERSION,
            "last_updated": dt_util.utcnow().isoformat(),
            "global_data": {
                "priors": {},
                "patterns": {},
            },
            "areas": {},
        }

    def _save_history(self) -> None:
        """Save the history storage file."""
        storage_path = self.hass.config.path(HISTORY_STORAGE_FILE)
        try:
            # Update last_updated timestamp
            self._history_data["last_updated"] = dt_util.utcnow().isoformat()

            # Ensure directory exists
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)

            # Write data atomically using temporary file
            temp_path = storage_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as file:
                yaml.dump(self._history_data, file, default_flow_style=False)
            os.replace(temp_path, storage_path)
        except (yaml.YAMLError, OSError) as err:
            _LOGGER.error("Failed to save history storage: %s", err)

    def _update_history(self) -> None:
        """Update historical data with current readings."""
        area_id = self.config["name"].lower().replace(" ", "_")
        current_data = self.data if self.data is not None else {}

        if not current_data:
            return

        # Initialize area history if needed
        if area_id not in self._history_data["areas"]:
            self._history_data["areas"][area_id] = {
                "priors": {},
                "patterns": {},
                "device_states": {},
                "environmental_baselines": {},
            }

        area_history = self._history_data["areas"][area_id]

        # Update priors
        for sensor_type, probability in current_data.get(
            "sensor_probabilities", {}
        ).items():
            if sensor_type not in area_history["priors"]:
                area_history["priors"][sensor_type] = []
            area_history["priors"][sensor_type].append(probability)
            # Keep only last 100 values
            area_history["priors"][sensor_type] = area_history["priors"][sensor_type][
                -100:
            ]

        # Update device states
        for device_type, states in current_data.get("device_states", {}).items():
            if device_type not in area_history["device_states"]:
                area_history["device_states"][device_type] = []

            timestamp = dt_util.utcnow().isoformat()
            for device_id, state in states.items():
                area_history["device_states"][device_type].append(
                    {
                        "device_id": device_id,
                        "state": state,
                        "timestamp": timestamp,
                    }
                )
            # Keep only last 24 hours of states
            cutoff = dt_util.utcnow() - timedelta(hours=24)
            area_history["device_states"][device_type] = [
                entry
                for entry in area_history["device_states"][device_type]
                if dt_util.parse_datetime(entry["timestamp"]) > cutoff
            ]

        # Update environmental baselines
        if "environmental_probability" in current_data.get("sensor_probabilities", {}):
            current_env = current_data["sensor_probabilities"][
                "environmental_probability"
            ]
            for sensor_type in ["temperature", "humidity", "illuminance"]:
                if sensor_type in area_history["environmental_baselines"]:
                    # Running average
                    prev = area_history["environmental_baselines"][sensor_type]
                    area_history["environmental_baselines"][sensor_type] = (
                        prev * 0.9 + current_env * 0.1
                    )
                else:
                    area_history["environmental_baselines"][sensor_type] = current_env

        # Save updates
        self._save_history()

    def _setup_history_update(self) -> None:
        """Set up periodic history updates."""
        update_interval = timedelta(minutes=15)  # Update every 15 minutes

        @callback
        def history_update(_: datetime) -> None:
            """Update history data periodically."""
            self._update_history()

        self._unsubscribe_handlers.append(
            async_track_time_interval(self.hass, history_update, update_interval)
        )

    def _create_calculator(self) -> ProbabilityCalculator:
        """Create probability calculator with current configuration."""
        return ProbabilityCalculator(
            base_config=self.config.get(
                "base_config", {}
            ),  # Get base_config from main config
            motion_sensors=self.config.get(CONF_MOTION_SENSORS, []),
            media_devices=self.config.get(CONF_MEDIA_DEVICES, []),
            appliances=self.config.get(CONF_APPLIANCES, []),
            illuminance_sensors=self.config.get(CONF_ILLUMINANCE_SENSORS, []),
            humidity_sensors=self.config.get(CONF_HUMIDITY_SENSORS, []),
            temperature_sensors=self.config.get(CONF_TEMPERATURE_SENSORS, []),
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
        sensors.extend(self.config.get(CONF_MEDIA_DEVICES, []))
        sensors.extend(self.config.get(CONF_APPLIANCES, []))
        sensors.extend(self.config.get(CONF_ILLUMINANCE_SENSORS, []))
        sensors.extend(self.config.get(CONF_HUMIDITY_SENSORS, []))
        sensors.extend(self.config.get(CONF_TEMPERATURE_SENSORS, []))
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
                _LOGGER.warning(
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
        data = self._get_calculated_data()
        self._update_history()
        return data

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
            if len(self._occupancy_history) > 1
            else True
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

        # Apply any learned patterns from history
        area_id = self.config["name"].lower().replace(" ", "_")
        if area_id in self._history_data.get("areas", {}):
            area_history = self._history_data["areas"][area_id]

            # Adjust probabilities based on historical patterns
            for sensor_type, priors in area_history.get("priors", {}).items():
                if priors:
                    historical_weight = 0.2  # 20% weight to historical data
                    current_prob = base_result["sensor_probabilities"].get(
                        sensor_type, 0.0
                    )
                    historical_avg = sum(priors) / len(priors)
                    adjusted_prob = (
                        current_prob * (1 - historical_weight)
                        + historical_avg * historical_weight
                    )
                    base_result["sensor_probabilities"][sensor_type] = adjusted_prob

            # Adjust environmental baselines
            env_baselines = area_history.get("environmental_baselines", {})
            if (
                env_baselines
                and "environmental_probability" in base_result["sensor_probabilities"]
            ):
                baseline_weight = 0.1  # 10% weight to baseline adjustments
                current_env_prob = base_result["sensor_probabilities"][
                    "environmental_probability"
                ]
                baseline_avg = sum(env_baselines.values()) / len(env_baselines)
                base_result["sensor_probabilities"]["environmental_probability"] = (
                    current_env_prob * (1 - baseline_weight)
                    + baseline_avg * baseline_weight
                )

        return base_result

    def get_historical_patterns(self) -> dict[str, Any]:
        """Get historical patterns for the current area."""
        area_id = self.config["name"].lower().replace(" ", "_")
        if area_id in self._history_data.get("areas", {}):
            return self._history_data["areas"][area_id]
        return {}

    async def async_reset_history(self) -> None:
        """Reset historical data for this area."""
        area_id = self.config["name"].lower().replace(" ", "_")
        if area_id in self._history_data.get("areas", {}):
            self._history_data["areas"][area_id] = {
                "priors": {},
                "patterns": {},
                "device_states": {},
                "environmental_baselines": {},
            }
            self._save_history()
            await self.async_refresh()
