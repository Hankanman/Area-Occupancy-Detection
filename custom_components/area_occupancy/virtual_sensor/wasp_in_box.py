"""Wasp in Box virtual sensor implementation."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_state_change

from .base import VirtualSensor
from .types import VirtualSensorConfig, VirtualSensorStateEnum

_LOGGER = logging.getLogger(__name__)


class WaspInBoxSensor(VirtualSensor):
    """Wasp in Box virtual sensor implementation."""

    def __init__(
        self,
        hass: HomeAssistant,
        config: VirtualSensorConfig,
        coordinator: Any,
    ) -> None:
        """Initialize the Wasp in Box sensor."""
        super().__init__(hass, config, coordinator)
        self._door_entity_id = config.get("door_entity_id")
        self._motion_entity_id = config.get("motion_entity_id")
        self._motion_timeout = config.get("motion_timeout", 300)  # Default 5 minutes
        self._door_state = VirtualSensorStateEnum.UNKNOWN
        self._motion_state = VirtualSensorStateEnum.UNKNOWN
        self._last_motion_time: Optional[datetime] = None
        self._last_door_time: Optional[datetime] = None
        self._last_updated: Optional[datetime] = None

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        if self._door_entity_id:
            self.async_on_remove(
                async_track_state_change(
                    self._hass,
                    self._door_entity_id,
                    self._async_door_state_changed,
                )
            )
        if self._motion_entity_id:
            self.async_on_remove(
                async_track_state_change(
                    self._hass,
                    self._motion_entity_id,
                    self._async_motion_state_changed,
                )
            )

    async def _async_door_state_changed(
        self, entity_id: str, old_state: Any, new_state: Any
    ) -> None:
        """Handle door state changes."""
        if new_state is None:
            return

        self._door_state = (
            VirtualSensorStateEnum.OCCUPIED
            if new_state.state == "on"
            else VirtualSensorStateEnum.UNOCCUPIED
        )
        self._last_door_time = new_state.last_updated
        await self._update_state()

    async def _async_motion_state_changed(
        self, entity_id: str, old_state: Any, new_state: Any
    ) -> None:
        """Handle motion state changes."""
        if new_state is None:
            return

        self._motion_state = (
            VirtualSensorStateEnum.OCCUPIED
            if new_state.state == "on"
            else VirtualSensorStateEnum.UNOCCUPIED
        )
        self._last_motion_time = new_state.last_updated
        await self._update_state()

    async def _update_state(self) -> None:
        """Update the sensor state."""
        now = datetime.now()
        motion_timeout = timedelta(seconds=self._motion_timeout)

        # Check if motion has timed out
        if (
            self._last_motion_time is not None
            and now - self._last_motion_time > motion_timeout
        ):
            self._motion_state = VirtualSensorStateEnum.UNOCCUPIED

        # Determine occupancy state
        if self._door_state == VirtualSensorStateEnum.OCCUPIED:
            self._state = VirtualSensorStateEnum.OCCUPIED
        elif self._motion_state == VirtualSensorStateEnum.OCCUPIED:
            self._state = VirtualSensorStateEnum.OCCUPIED
        else:
            self._state = VirtualSensorStateEnum.UNOCCUPIED

        self._available = True
        self._last_updated = now

        self._attributes = {
            "door_entity_id": self._door_entity_id,
            "motion_entity_id": self._motion_entity_id,
            "door_state": str(self._door_state),
            "motion_state": str(self._motion_state),
            "last_motion_time": self._last_motion_time,
            "last_door_time": self._last_door_time,
            "motion_timeout": self._motion_timeout,
        }
