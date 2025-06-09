"""Virtual sensor package for Area Occupancy Detection."""

from __future__ import annotations

import logging

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from ..coordinator import AreaOccupancyCoordinator
from .wasp_in_box import WaspInBoxConfig, async_setup_entry as setup_wasp_in_box

_LOGGER = logging.getLogger(__name__)


async def async_setup_virtual_sensors(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
    coordinator: AreaOccupancyCoordinator,
) -> list[BinarySensorEntity]:
    """Set up virtual sensors for the area occupancy detection integration.

    Returns:
        A list of created virtual sensor entities.

    """
    _LOGGER.debug(
        "Setting up virtual sensors for area %s",
        coordinator.config.name,
    )

    created_sensors: list[BinarySensorEntity] = []

    # Check for flattened wasp settings - the config flow flattens section dictionaries
    config = coordinator.config
    wasp_enabled = config.wasp_in_box.enabled

    # Set up the Wasp in Box sensor if enabled
    if wasp_enabled:
        _LOGGER.debug("Found wasp enabled in config: %s", wasp_enabled)
        try:
            # Construct the dict
            wasp_config_dict: WaspInBoxConfig = {
                "enabled": True,
                "motion_timeout": config.wasp_in_box.motion_timeout,
                "weight": config.wasp_in_box.weight,
                "max_duration": config.wasp_in_box.max_duration,
            }

            _LOGGER.debug("Created wasp config dict: %s", wasp_config_dict)

            # Get the sensor instance and pass the coordinator
            wasp_sensor = await setup_wasp_in_box(
                hass,
                wasp_config_dict,
                async_add_entities,  # Keep this for compatibility with function signature
                coordinator,
            )

            if wasp_sensor:
                created_sensors.append(wasp_sensor)
                _LOGGER.debug(
                    "Successfully created Wasp in Box sensor: %s",
                    getattr(wasp_sensor, "unique_id", "Unknown"),
                )
            else:
                _LOGGER.warning(
                    "Wasp in Box sensor setup did not return a sensor instance"
                )

        except Exception:
            _LOGGER.exception("Failed to set up Wasp in Box sensor")

    # Add other virtual sensors here as they are implemented
    _LOGGER.debug("Created %d virtual sensors", len(created_sensors))
    return created_sensors
