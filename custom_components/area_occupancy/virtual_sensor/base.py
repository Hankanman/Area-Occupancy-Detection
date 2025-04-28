"""Base class for virtual sensors."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity

from .types import VirtualSensorConfig, VirtualSensorStateEnum

_LOGGER = logging.getLogger(__name__)


class VirtualSensor(Entity):
    """Base class for virtual sensors."""

    def __init__(
        self,
        hass: HomeAssistant,
        config: VirtualSensorConfig,
        coordinator: Any,
    ) -> None:
        """Initialize the virtual sensor."""
        self._hass = hass
        self._config = config
        self._coordinator = coordinator
        self._state = VirtualSensorStateEnum.UNKNOWN
        self._attributes: Dict[str, Any] = {}
        self._available = True
        self._last_updated: Optional[datetime] = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._config.get("name", "Virtual Sensor")

    @property
    def state(self) -> str:
        """Return the state of the sensor."""
        return str(self._state)

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return self._attributes

    @property
    def should_poll(self) -> bool:
        """Return True if entity should be polled."""
        return False

    async def async_update(self) -> None:
        """Update the sensor state."""
        try:
            await self._update_state()
        except Exception as ex:
            _LOGGER.error("Error updating virtual sensor: %s", ex, exc_info=True)
            self._state = VirtualSensorStateEnum.ERROR
            self._available = False

    async def _update_state(self) -> None:
        """Update the sensor state.

        This method should be implemented by subclasses to update the sensor state
        based on the current conditions.
        """
        raise NotImplementedError
