"""Coordinator for virtual sensors."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Union

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .base import VirtualSensor
from .const import DEFAULT_VIRTUAL_SENSOR_UPDATE_INTERVAL, ERROR_UPDATE_FAILED
from .exceptions import VirtualSensorUpdateError
from .types import VirtualSensorState, VirtualSensorUpdateResult

_LOGGER = logging.getLogger(__name__)


class VirtualSensorCoordinator(
    DataUpdateCoordinator[Dict[str, VirtualSensorUpdateResult]]
):
    """Class to manage fetching data from virtual sensors."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        update_interval: Optional[timedelta] = None,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=name,
            update_interval=update_interval
            or timedelta(seconds=DEFAULT_VIRTUAL_SENSOR_UPDATE_INTERVAL),
        )
        self._sensors: Dict[str, VirtualSensor] = {}
        self._update_tasks: List[asyncio.Task[VirtualSensorUpdateResult]] = []

    @property
    def sensors(self) -> Dict[str, VirtualSensor]:
        """Return the registered sensors."""
        return self._sensors

    async def _async_update_data(self) -> Dict[str, VirtualSensorUpdateResult]:
        """Fetch data from all registered sensors."""
        results: Dict[str, VirtualSensorUpdateResult] = {}

        # Cancel any existing update tasks
        for task in self._update_tasks:
            if not task.done():
                task.cancel()
        self._update_tasks.clear()

        # Create update tasks for all sensors
        for sensor_id, sensor in self._sensors.items():
            task = asyncio.create_task(self._update_sensor(sensor_id, sensor))
            self._update_tasks.append(task)

        try:
            # Wait for all updates to complete
            update_results: Sequence[
                Union[VirtualSensorUpdateResult, BaseException]
            ] = await asyncio.gather(*self._update_tasks, return_exceptions=True)

            # Process results
            for sensor_id, result in zip(self._sensors.keys(), update_results):
                if isinstance(result, BaseException):
                    results[sensor_id] = VirtualSensorUpdateResult(
                        success=False,
                        state=None,
                        error=str(result),
                    )
                else:
                    results[sensor_id] = result

        except Exception as err:
            raise UpdateFailed(f"{ERROR_UPDATE_FAILED}: {err}") from err

        return results

    async def _update_sensor(
        self, sensor_id: str, sensor: VirtualSensor
    ) -> VirtualSensorUpdateResult:
        """Update a single sensor."""
        try:
            await sensor.async_update()
            state = VirtualSensorState(
                state=sensor.state,
                attributes=sensor.extra_state_attributes,
                last_updated=sensor._last_updated.isoformat()
                if sensor._last_updated
                else None,
            )
            return VirtualSensorUpdateResult(
                success=True,
                state=state,
                error=None,
            )
        except Exception as err:
            _LOGGER.error("Error updating sensor %s: %s", sensor_id, err, exc_info=True)
            return VirtualSensorUpdateResult(
                success=False,
                state=None,
                error=str(err),
            )

    async def async_add_sensor(self, sensor_id: str, sensor: VirtualSensor) -> None:
        """Add a sensor to the coordinator."""
        self._sensors[sensor_id] = sensor
        _LOGGER.debug("Added sensor %s to coordinator", sensor_id)

    async def async_remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor from the coordinator."""
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
            _LOGGER.debug("Removed sensor %s from coordinator", sensor_id)

    def get_sensor(self, sensor_id: str) -> Optional[VirtualSensor]:
        """Get a sensor by ID."""
        return self._sensors.get(sensor_id)

    async def async_refresh_sensor(self, sensor_id: str) -> None:
        """Refresh a specific sensor."""
        if sensor := self._sensors.get(sensor_id):
            try:
                await sensor.async_update()
                await self.async_request_refresh()
            except Exception as err:
                raise VirtualSensorUpdateError(str(err), sensor_id) from err
