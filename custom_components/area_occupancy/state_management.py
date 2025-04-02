"""State management for the Area Occupancy Detection integration."""

import asyncio

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util


class OccupancyStateManager:
    """Manage sensor states for Area Occupancy Detection integration.

    This class maintains state information for all sensors,
    and tracks the currently active sensors separately.
    """

    def __init__(self) -> None:
        """Initialize the OccupancyStateManager."""
        self._all_sensor_states: dict[str, dict] = {}
        self._active_sensors: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def initialize_states(
        self, hass: HomeAssistant, sensor_ids: list[str]
    ) -> None:
        """Initialize sensor states from Home Assistant for the given sensor IDs."""
        async with self._lock:
            for entity_id in sensor_ids:
                state_obj = hass.states.get(entity_id)
                is_available = bool(
                    state_obj
                    and state_obj.state not in ["unknown", "unavailable", None, ""]
                )
                sensor_info = {
                    "state": state_obj.state if is_available else None,
                    "last_changed": (
                        state_obj.last_changed.isoformat()
                        if state_obj and state_obj.last_changed
                        else dt_util.utcnow().isoformat()
                    ),
                    "availability": is_available,
                }
                self._all_sensor_states[entity_id] = sensor_info
                # Also mark sensor as active if available
                if is_available:
                    self._active_sensors[entity_id] = sensor_info

    async def update_sensor(self, entity_id: str, new_state_obj) -> None:
        """Update a sensor's state based on a new state object."""
        async with self._lock:
            is_available = new_state_obj.state not in [
                "unknown",
                "unavailable",
                None,
                "",
            ]
            sensor_info = {
                "state": new_state_obj.state if is_available else None,
                "last_changed": (
                    new_state_obj.last_changed.isoformat()
                    if new_state_obj.last_changed
                    else dt_util.utcnow().isoformat()
                ),
                "availability": is_available,
            }
            self._all_sensor_states[entity_id] = sensor_info
            # Check if sensor should be active using external logic; here we simply mark available as active
            if is_available:
                self._active_sensors[entity_id] = sensor_info
            else:
                # If not available, remove it from active sensors if present
                self._active_sensors.pop(entity_id, None)

    async def remove_sensor(self, entity_id: str) -> None:
        """Remove sensor from active sensors."""
        async with self._lock:
            self._active_sensors.pop(entity_id, None)

    async def get_all_sensor_states(self) -> dict[str, dict]:
        """Return a copy of all sensor states."""
        async with self._lock:
            return self._all_sensor_states.copy()

    async def get_active_sensors(self) -> dict[str, dict]:
        """Return a copy of currently active sensors."""
        async with self._lock:
            return self._active_sensors.copy()

    async def mark_sensor_inactive(self, entity_id: str) -> None:
        """Mark a sensor as inactive (remove from active sensors) but keep its state in _all_sensor_states."""
        async with self._lock:
            self._active_sensors.pop(entity_id, None)
