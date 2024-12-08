"""Coordinator for Area Occupancy Detection."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import deque

from homeassistant.const import (
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    CONF_AREA_ID,
)
from .types import (
    ProbabilityResult,
    SensorId,
    SensorStates,
    SensorState,
    StorageData,
    CoreConfig,
    OptionsConfig,
    DecayConfig,
    TimeslotData,
)
from .calculations import ProbabilityCalculator
from .historical_analysis import HistoricalAnalysis

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Class to manage fetching area occupancy data."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        core_config: CoreConfig,
        options_config: OptionsConfig,
        store: Store[StorageData],
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),
        )

        if not core_config.get("motion_sensors"):
            raise HomeAssistantError("No motion sensors configured")

        self.entry_id = entry_id
        self.store = store
        self.core_config = core_config
        self.options_config = options_config

        # Initialize state tracking
        self._state_lock = asyncio.Lock()
        self._last_probability: Optional[float] = None
        self._max_change_rate = 0.3
        self._sensor_states: SensorStates = {}
        self._motion_timestamps: dict[SensorId, datetime] = {}
        self._unsubscribe_handlers: list[callable] = []

        # Initialize history tracking
        self._probability_history = deque(maxlen=12)  # 1 hour of 5-minute readings
        self._last_occupied: datetime | None = None
        self._last_state_change: datetime | None = None
        self._occupancy_history = deque([False] * 288, maxlen=288)  # 24 hours

        # Initialize timeslot tracking with default empty structure
        self._timeslot_data: TimeslotData = {
            "slots": {},
            "last_updated": dt_util.utcnow(),
        }
        self._last_timeslot_update: datetime | None = None

        # Initialize components
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize coordinator components."""
        self._historical_analysis = HistoricalAnalysis(self.hass)
        self._calculator = self._create_calculator()

    def _create_calculator(self) -> ProbabilityCalculator:
        """Create probability calculator with current configuration."""
        return ProbabilityCalculator(
            motion_sensors=self.core_config["motion_sensors"],
            media_devices=self.options_config.get("media_devices", []),
            appliances=self.options_config.get("appliances", []),
            illuminance_sensors=self.options_config.get("illuminance_sensors", []),
            humidity_sensors=self.options_config.get("humidity_sensors", []),
            temperature_sensors=self.options_config.get("temperature_sensors", []),
            decay_config=DecayConfig(
                enabled=self.options_config.get("decay_enabled", True),
                window=self.options_config.get("decay_window", 600),
                type=self.options_config.get("decay_type", "linear"),
            ),
        )

    async def async_setup(self) -> None:
        """Perform setup tasks that require async operations."""
        try:
            # Initialize historical analysis
            if self.options_config.get("historical_analysis_enabled", True):
                try:
                    self._timeslot_data = (
                        await self._historical_analysis.calculate_timeslots(
                            self._get_all_configured_sensors(),
                            self.options_config["history_period"],
                        )
                    )
                    self._last_timeslot_update = dt_util.utcnow()
                except Exception as err:  # pylint: disable=broad-except
                    _LOGGER.warning("Failed to initialize historical analysis: %s", err)
                    # Ensure default structure exists even if analysis fails
                    self._timeslot_data = {
                        "slots": {},
                        "last_updated": dt_util.utcnow(),
                    }

            # Set up state change listeners
            self._setup_state_listeners()

            # Set up update intervals
            self._setup_update_intervals()

            # Initialize states
            self._initialize_states()

        except Exception as err:
            _LOGGER.error("Failed to setup coordinator: %s", err)
            raise HomeAssistantError(f"Coordinator setup failed: {err}") from err

    @callback
    def _initialize_states(self) -> None:
        """Initialize states for all configured sensors."""
        try:
            for entity_id in self._get_all_configured_sensors():
                # Get state object
                state = self.hass.states.get(entity_id)

                # Handle missing or unavailable states
                if not state:
                    self._sensor_states[entity_id] = {
                        "state": None,
                        "last_changed": dt_util.utcnow().isoformat(),
                        "availability": False,
                    }
                    _LOGGER.debug("No state available yet for %s", entity_id)
                    continue

                # Handle valid states
                if state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    try:
                        self._sensor_states[entity_id] = {
                            "state": state.state,
                            "last_changed": state.last_changed.isoformat(),
                            "availability": True,
                        }
                    except AttributeError as err:
                        _LOGGER.warning(
                            "Could not access state attributes for %s: %s",
                            entity_id,
                            err,
                        )
                        self._sensor_states[entity_id] = {
                            "state": None,
                            "last_changed": dt_util.utcnow().isoformat(),
                            "availability": False,
                        }
                else:
                    self._sensor_states[entity_id] = {
                        "state": None,
                        "last_changed": dt_util.utcnow().isoformat(),
                        "availability": False,
                    }

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error initializing states: %s", err)
            # Don't raise the error - allow the coordinator to continue with empty states
            self._sensor_states = {}

    async def _async_handle_state_change(
        self, entity_id: str, new_state: SensorState | None
    ) -> None:
        """Handle state changes for sensors."""
        async with self._state_lock:
            if not new_state or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                if entity_id in self._sensor_states:
                    self._sensor_states[entity_id] = {
                        "state": None,
                        "last_changed": dt_util.utcnow().isoformat(),
                        "availability": False,
                    }
                return

            # Handle motion sensor states
            if (
                entity_id in self.core_config["motion_sensors"]
                and new_state.state == STATE_ON
            ):
                self._motion_timestamps[entity_id] = dt_util.utcnow()

            # Update sensor state
            self._sensor_states[entity_id] = {
                "state": new_state.state,
                "last_changed": new_state.last_changed.isoformat(),
                "availability": True,
            }

            # Trigger update
            await self.async_refresh()

    def _setup_state_listeners(self) -> None:
        """Set up state change listeners."""
        self.unsubscribe()
        self._unsubscribe_handlers.append(
            async_track_state_change_event(
                self.hass,
                self._get_all_configured_sensors(),
                self._async_handle_state_change,
            )
        )

    def _setup_update_intervals(self) -> None:
        """Set up periodic update intervals."""
        # Update timeslot data every 6 hours
        self._unsubscribe_handlers.append(
            async_track_time_interval(
                self.hass,
                self._async_update_timeslots,
                timedelta(hours=6),
            )
        )

    async def _async_update_timeslots(self, _: datetime) -> None:
        """Update timeslot data periodically."""
        try:
            if not self.options_config.get("historical_analysis_enabled", True):
                return

            new_timeslot_data = await self._historical_analysis.calculate_timeslots(
                self._get_all_configured_sensors(),
                self.options_config["history_period"],
            )

            # Validate returned data structure
            if isinstance(new_timeslot_data, dict) and "slots" in new_timeslot_data:
                self._timeslot_data = new_timeslot_data
                self._last_timeslot_update = dt_util.utcnow()
            else:
                _LOGGER.warning(
                    "Invalid timeslot data structure received, maintaining current data"
                )

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error updating timeslots: %s", err)
            # Keep existing data on error

    async def _async_update_data(self) -> ProbabilityResult:
        """Update data from all sources."""
        try:
            # Get current timeslot data
            current_time = dt_util.utcnow()
            slot_key = f"{current_time.hour:02d}:{(current_time.minute // 30) * 30:02d}"

            # Safely get slot data with fallback to empty dict
            current_slot = {}
            if isinstance(self._timeslot_data, dict):
                slots = self._timeslot_data.get("slots", {})
                if isinstance(slots, dict):
                    current_slot = slots.get(slot_key, {})

            # Calculate probability
            result = await self._async_calculate_probability(current_slot)

            # Update historical tracking
            self._update_historical_tracking(result)

            # Store results
            await self._async_store_result(result)

            return result

        except Exception as err:
            _LOGGER.error("Error updating data: %s", err)
            raise

    def _update_historical_tracking(self, result: ProbabilityResult) -> None:
        """Update historical tracking data."""
        now = dt_util.utcnow()

        # Update probability history
        self._probability_history.append(result["probability"])

        # Update occupancy history
        is_occupied = result["probability"] >= self.options_config["threshold"]
        self._occupancy_history.append(is_occupied)

        if is_occupied:
            self._last_occupied = now

        if not self._last_state_change or (
            bool(self._occupancy_history[-2]) != is_occupied
            if len(self._occupancy_history) > 1
            else True
        ):
            self._last_state_change = now

    async def _async_calculate_probability(
        self, timeslot_data: dict
    ) -> ProbabilityResult:
        """Calculate current probability using historical context."""
        # Get base calculation
        base_result = await self._calculator.calculate(
            self._sensor_states,
            self._motion_timestamps,
        )

        # Adjust using historical probabilities if available
        if timeslot_data and "prob_given_true" in timeslot_data:
            historical_weight = 0.3
            current_weight = 0.7

            historical_prob = timeslot_data["prob_given_true"]
            current_prob = base_result["probability"]

            adjusted_prob = (current_prob * current_weight) + (
                historical_prob * historical_weight
            )

            base_result.update(
                {
                    "probability": min(1.0, max(0.0, adjusted_prob)),
                    "historical_probability": historical_prob,
                    "historical_confidence": timeslot_data.get("confidence", 0.0),
                }
            )

        # Add historical metrics
        base_result.update(self._get_historical_metrics())

        return base_result

    def _get_historical_metrics(self) -> dict[str, Any]:
        """Calculate historical metrics."""
        now = dt_util.utcnow()

        # Calculate moving average
        moving_avg = (
            sum(self._probability_history) / len(self._probability_history)
            if self._probability_history
            else 0.0
        )

        # Calculate rate of change
        rate_of_change = 0.0
        if len(self._probability_history) >= 2:
            change = self._probability_history[-1] - self._probability_history[0]
            time_window = len(self._probability_history) * 5
            rate_of_change = (change / time_window) * 60

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

    async def _async_store_result(self, result: ProbabilityResult) -> None:
        """Store calculation result."""
        try:
            stored_data = await self.store.async_load() or {}
            area_id = self.core_config[CONF_AREA_ID]

            # Initialize if needed
            stored_data.setdefault("areas", {})
            stored_data["areas"].setdefault(area_id, {})

            # Update stored data
            stored_data["areas"][area_id].update(
                {
                    "last_updated": dt_util.utcnow().isoformat(),
                    "last_probability": result["probability"],
                    "last_timeslot_update": (
                        self._last_timeslot_update.isoformat()
                        if self._last_timeslot_update
                        else None
                    ),
                    "historical_metrics": self._get_historical_metrics(),
                    "configuration": {
                        "motion_sensors": self.core_config["motion_sensors"],
                        "media_devices": self.options_config.get("media_devices", []),
                        "appliances": self.options_config.get("appliances", []),
                        "illuminance_sensors": self.options_config.get(
                            "illuminance_sensors", []
                        ),
                        "humidity_sensors": self.options_config.get(
                            "humidity_sensors", []
                        ),
                        "temperature_sensors": self.options_config.get(
                            "temperature_sensors", []
                        ),
                    },
                }
            )

            await self.store.async_save(stored_data)

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error storing result: %s", err)

    def get_configured_sensors(self) -> list[str]:
        """Get list of all configured sensor entity IDs.

        Returns:
            list[str]: List of all configured sensor entity IDs.
        """
        return self._get_all_configured_sensors()

    def _get_all_configured_sensors(self) -> list[str]:
        """Internal method to get list of all configured sensor entity IDs."""
        return [
            *self.core_config["motion_sensors"],
            *self.options_config.get("media_devices", []),
            *self.options_config.get("appliances", []),
            *self.options_config.get("illuminance_sensors", []),
            *self.options_config.get("humidity_sensors", []),
            *self.options_config.get("temperature_sensors", []),
        ]

    def unsubscribe(self) -> None:
        """Unsubscribe from all registered events."""
        while self._unsubscribe_handlers:
            self._unsubscribe_handlers.pop()()

    def update_options(self, options_config: OptionsConfig) -> None:
        """Update coordinator with new options."""
        try:
            self.options_config = options_config
            self._setup_components()

            # Clear invalid sensor states
            current_sensors = set(self._get_all_configured_sensors())
            self._sensor_states = {
                entity_id: state
                for entity_id, state in self._sensor_states.items()
                if entity_id in current_sensors
            }
            self._motion_timestamps = {
                entity_id: timestamp
                for entity_id, timestamp in self._motion_timestamps.items()
                if entity_id in self.core_config["motion_sensors"]
            }

            _LOGGER.debug(
                "Updated coordinator options for %s: %s",
                self.core_config["name"],
                options_config,
            )
        except Exception as err:
            _LOGGER.error("Error updating coordinator options: %s", err)
            raise HomeAssistantError(
                f"Failed to update coordinator options: {err}"
            ) from err

    async def async_reset_history(self) -> None:
        """Reset historical data for this area."""
        try:
            # Reset storage
            stored_data = await self.store.async_load() or {}
            area_id = self.core_config[CONF_AREA_ID]

            if "areas" in stored_data and area_id in stored_data["areas"]:
                stored_data["areas"][area_id] = {
                    "last_updated": dt_util.utcnow().isoformat(),
                    "last_probability": 0.0,
                    "configuration": stored_data["areas"][area_id].get(
                        "configuration", {}
                    ),
                }
                await self.store.async_save(stored_data)

            # Reset local tracking
            self._probability_history.clear()
            self._occupancy_history.extend([False] * self._occupancy_history.maxlen)
            self._last_occupied = None
            self._last_state_change = None
            self._last_probability = None
            self._timeslot_data = {"slots": {}, "last_updated": dt_util.utcnow()}
            self._last_timeslot_update = None

            # Force refresh
            await self.async_refresh()

        except Exception as err:
            _LOGGER.error("Failed to reset history: %s", err)
            raise HomeAssistantError("Failed to reset historical data") from err

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        return {
            "core_config": self.core_config,
            "options_config": self.options_config,
            "sensor_states": self._sensor_states,
            "historical_data": {
                "probability_history": list(self._probability_history),
                "occupancy_history": list(self._occupancy_history),
                "last_occupied": (
                    self._last_occupied.isoformat() if self._last_occupied else None
                ),
                "last_state_change": (
                    self._last_state_change.isoformat()
                    if self._last_state_change
                    else None
                ),
            },
            "timeslot_data": {
                "last_update": (
                    self._last_timeslot_update.isoformat()
                    if self._last_timeslot_update
                    else None
                ),
                "slot_count": len(self._timeslot_data["slots"]),
            },
        }
