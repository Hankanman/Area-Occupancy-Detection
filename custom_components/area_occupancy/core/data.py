# custom_components/area_occupancy/core/data.py

"""Data provider for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant, State
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.components.recorder import get_instance, history
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError

from . import AreaConfig, SensorState
from .storage import StorageProvider

_LOGGER = logging.getLogger(__name__)


class AreaDataProvider:
    """Manages sensor data retrieval and state tracking."""

    def __init__(
        self,
        hass: HomeAssistant,
        config: AreaConfig,
        storage: StorageProvider,
        history_window: timedelta = timedelta(days=7),
    ) -> None:
        """Initialize the data provider."""
        self.hass = hass
        self.config = config
        self._storage = storage
        self._history_window = history_window
        self._sensor_states: dict[str, SensorState] = {}
        self._last_history_fetch: datetime | None = None
        self._history_data: dict[str, Any] = {}
        self._min_history_interval = timedelta(hours=1)

    @property
    def configured_sensors(self) -> list[str]:
        """Get list of all configured sensor entity IDs."""
        return (
            self.config.motion_sensors
            + self.config.media_devices
            + self.config.appliances
            + self.config.illuminance_sensors
            + self.config.humidity_sensors
            + self.config.temperature_sensors
        )

    async def async_setup(self) -> None:
        """Set up the data provider."""
        try:
            # Load stored data
            stored_data = await self._storage.async_load()
            if stored_data:
                self._history_data = stored_data

            # Initialize sensor states
            await self._initialize_sensor_states()

            # Perform initial history fetch
            await self._fetch_historical_data()

        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error("Failed to setup data provider: %s", err)
            raise HomeAssistantError("Failed to initialize data provider") from err

    async def get_sensor_states(self) -> dict[str, SensorState]:
        """Get current states of all configured sensors."""
        try:
            updated_states: dict[str, SensorState] = {}

            for entity_id in self.configured_sensors:
                state = self.hass.states.get(entity_id)
                if state is None:
                    # Handle missing state
                    updated_states[entity_id] = SensorState(
                        entity_id=entity_id,
                        state="",
                        last_changed=dt_util.utcnow(),
                        available=False,
                    )
                    continue

                updated_states[entity_id] = self._convert_ha_state(state)

            # Update internal state cache
            self._sensor_states.update(updated_states)

            return self._sensor_states.copy()

        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error("Error getting sensor states: %s", err)
            raise HomeAssistantError("Failed to get sensor states") from err

    async def get_historical_data(self) -> dict[str, Any]:
        """Get historical data for analysis."""
        try:
            # Check if we need to fetch new historical data
            now = dt_util.utcnow()
            if (
                self._last_history_fetch is None
                or now - self._last_history_fetch >= self._min_history_interval
            ):
                await self._fetch_historical_data()

            return self._history_data.copy()

        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error("Error getting historical data: %s", err)
            return {}

    async def _initialize_sensor_states(self) -> None:
        """Initialize sensor states from current Home Assistant states."""
        for entity_id in self.configured_sensors:
            state = self.hass.states.get(entity_id)
            if state is not None:
                self._sensor_states[entity_id] = self._convert_ha_state(state)
            else:
                _LOGGER.warning("No state found for entity: %s", entity_id)
                self._sensor_states[entity_id] = SensorState(
                    entity_id=entity_id,
                    state="",
                    last_changed=dt_util.utcnow(),
                    available=False,
                )

    async def _fetch_historical_data(self) -> None:
        """Fetch historical state data from recorder."""
        try:
            start_time = dt_util.utcnow() - self._history_window

            # Get history from recorder
            recorder = get_instance(self.hass)
            history_data = await recorder.async_add_executor_job(
                history.get_significant_states,
                self.hass,
                start_time,
                dt_util.utcnow(),
                self.configured_sensors,
            )

            if not history_data:
                _LOGGER.warning("No historical data found for %s", self.config.name)
                return

            # Process historical data
            processed_data = await self._process_historical_data(history_data)

            # Update internal storage
            self._history_data = processed_data
            self._last_history_fetch = dt_util.utcnow()

            # Save to persistent storage
            await self._storage.async_save(processed_data)

        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error("Error fetching historical data: %s", err)
            raise HomeAssistantError("Failed to fetch historical data") from err

    async def _process_historical_data(
        self, history_data: dict[str, list[State]]
    ) -> dict[str, Any]:
        """Process raw historical data into analyzable format."""
        processed = {
            "states": {},
            "patterns": {
                "time_slots": {},
                "days": {},
            },
            "metadata": {
                "start_time": dt_util.utcnow() - self._history_window,
                "end_time": dt_util.utcnow(),
                "total_samples": 0,
                "valid_samples": 0,
            },
        }

        for entity_id, states in history_data.items():
            if not states:
                continue

            entity_data = []
            for state in states:
                if state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    entity_data.append(
                        {
                            "state": state.state,
                            "timestamp": state.last_changed.isoformat(),
                            "available": True,
                        }
                    )
                    processed["metadata"]["valid_samples"] += 1

                processed["metadata"]["total_samples"] += 1

            processed["states"][entity_id] = entity_data

            # Process time-based patterns
            if entity_id in self.config.motion_sensors:
                self._process_motion_patterns(states, processed["patterns"])

        return processed

    def _process_motion_patterns(
        self, states: list[State], patterns: dict[str, dict]
    ) -> None:
        """Process motion sensor states into time-based patterns."""
        for state in states:
            if state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                timestamp = dt_util.as_local(state.last_changed)
                time_slot = self._get_time_slot(timestamp)
                day_name = timestamp.strftime("%A").lower()

                # Update time slot stats
                if time_slot not in patterns["time_slots"]:
                    patterns["time_slots"][time_slot] = {"active": 0, "total": 0}
                patterns["time_slots"][time_slot]["total"] += 1
                if state.state == "on":
                    patterns["time_slots"][time_slot]["active"] += 1

                # Update day stats
                if day_name not in patterns["days"]:
                    patterns["days"][day_name] = {"active": 0, "total": 0}
                patterns["days"][day_name]["total"] += 1
                if state.state == "on":
                    patterns["days"][day_name]["active"] += 1

    @staticmethod
    def _convert_ha_state(state: State) -> SensorState:
        """Convert Home Assistant state to internal SensorState."""
        return SensorState(
            entity_id=state.entity_id,
            state=state.state,
            last_changed=state.last_changed,
            available=state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN),
        )

    @staticmethod
    def _get_time_slot(timestamp: datetime) -> str:
        """Convert timestamp to time slot string."""
        minutes = timestamp.hour * 60 + timestamp.minute
        slot = (minutes // 30) * 30  # 30-minute slots
        return f"{slot // 60:02d}:{slot % 60:02d}"

    async def async_stop(self) -> None:
        """Stop the data provider and clean up."""
        try:
            # Save final state to storage
            if self._history_data:
                await self._storage.async_save(self._history_data)
        except HomeAssistantError as err:
            _LOGGER.error("HomeAssistantError stopping data provider: %s", err)
        except OSError as err:
            _LOGGER.error("OSError stopping data provider: %s", err)
        except RuntimeError as err:
            _LOGGER.error("RuntimeError stopping data provider: %s", err)
