"""Base class for virtual sensors."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.restore_state import RestoreEntity

from .coordinator import VirtualSensorCoordinator
from .types import VirtualSensorConfig, VirtualSensorStateEnum

_LOGGER = logging.getLogger(__name__)


class VirtualSensor(RestoreEntity, Entity):
    """Base class for virtual sensors."""

    def __init__(
        self,
        hass: HomeAssistant,
        config: VirtualSensorConfig,
        coordinator: VirtualSensorCoordinator,
    ) -> None:
        """Initialize the virtual sensor."""
        super().__init__()
        self._hass = hass
        self._config = config
        self._coordinator = coordinator
        self._state = VirtualSensorStateEnum.UNKNOWN
        self._attributes: Dict[str, Any] = {}
        self._available = True
        self._last_updated: Optional[datetime] = None
        self._unique_id = f"virtual_sensor_{config.get('name', 'unknown')}"

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()

        # Restore state if available
        if (last_state := await self.async_get_last_state()) is not None:
            self._state = VirtualSensorStateEnum(last_state.state)
            self._attributes = dict(last_state.attributes)
            self._last_updated = last_state.last_updated
            _LOGGER.debug("Restored state for %s: %s", self.entity_id, self._state)

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

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._config.get("name", "Virtual Sensor")

    @property
    def unique_id(self) -> str:
        """Return the unique ID of the sensor."""
        return self._unique_id

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    @property
    def state(self) -> str:
        """Return the state of the sensor."""
        return str(self._state)

    @property
    def should_poll(self) -> bool:
        """Return True if entity should be polled."""
        return False

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return self._attributes
