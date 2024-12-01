# custom_components/area_occupancy/core/sensor_manager.py

"""Entity management for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.const import CONF_NAME

from . import AreaConfig
from .coordinator import AreaOccupancyCoordinator
from .data import AreaDataProvider
from .storage import AreaStorageProvider
from ..sensors.binary import AreaOccupancyBinarySensor
from ..sensors.probability import AreaOccupancyProbabilitySensor

_LOGGER = logging.getLogger(__name__)


class AreaSensorManager:
    """Manages area occupancy sensors."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the sensor manager."""
        self.hass = hass
        self.entry = entry
        self._config = self._create_config(entry.data)
        self._storage = AreaStorageProvider(
            hass, self._config.name.lower().replace(" ", "_")
        )
        self._data_provider = AreaDataProvider(hass, self._config, self._storage)
        self._coordinator = AreaOccupancyCoordinator(
            hass, entry.entry_id, self._config, self._data_provider
        )
        self._sensors: list[Any] = []

    @staticmethod
    def _create_config(config_data: dict[str, Any]) -> AreaConfig:
        """Create area configuration from config entry data."""
        return AreaConfig(
            name=config_data[CONF_NAME],
            motion_sensors=config_data.get("motion_sensors", []),
            media_devices=config_data.get("media_devices", []),
            appliances=config_data.get("appliances", []),
            illuminance_sensors=config_data.get("illuminance_sensors", []),
            humidity_sensors=config_data.get("humidity_sensors", []),
            temperature_sensors=config_data.get("temperature_sensors", []),
            threshold=config_data.get("threshold", 0.5),
            decay_enabled=config_data.get("decay_enabled", True),
            decay_window=config_data.get("decay_window", 600),
            decay_type=config_data.get("decay_type", "linear"),
            history_period=config_data.get("history_period", 7),
        )

    async def async_setup(self) -> None:
        """Set up the sensor manager."""
        try:
            # Initialize storage and data provider
            await self._storage.async_load()
            await self._data_provider.async_setup()

            # Initialize coordinator
            await self._coordinator.async_setup()

        except (IOError, ValueError) as err:
            _LOGGER.error("Failed to setup sensor manager: %s", err)
            raise

    async def async_setup_entities(
        self, async_add_entities: AddEntitiesCallback
    ) -> None:
        """Set up the sensor entities."""
        try:
            # Create sensors
            binary_sensor = AreaOccupancyBinarySensor(
                self._coordinator,
                self.entry.entry_id,
                self._config.threshold,
            )

            probability_sensor = AreaOccupancyProbabilitySensor(
                self._coordinator,
                self.entry.entry_id,
            )

            self._sensors = [binary_sensor, probability_sensor]

            # Add entities
            async_add_entities(self._sensors)

        except (IOError, ValueError) as err:
            _LOGGER.error("Failed to setup sensor entities: %s", err)
            raise

    async def async_unload(self) -> bool:
        """Unload the sensor manager."""
        try:
            # Stop data provider
            await self._data_provider.async_stop()

            # Remove entities
            for sensor in self._sensors:
                await self.hass.config_entries.async_remove_entity(
                    self.entry.entry_id,
                    sensor.entity_id,
                )

            return True

        except (IOError, ValueError) as err:
            _LOGGER.error("Error unloading sensor manager: %s", err)
            return False

    async def async_update_config(self, config_data: dict[str, Any]) -> None:
        """Update configuration."""
        try:
            # Create new config
            new_config = self._create_config(config_data)

            # Update components
            self._config = new_config
            self._data_provider = AreaDataProvider(
                self.hass, self._config, self._storage
            )
            await self._data_provider.async_setup()

            # Reinitialize coordinator
            await self._coordinator.async_setup()

            # Update sensors
            for sensor in self._sensors:
                if isinstance(sensor, AreaOccupancyBinarySensor):
                    sensor.update_threshold(self._config.threshold)

        except (IOError, ValueError) as err:
            _LOGGER.error("Error updating sensor manager config: %s", err)
            raise
