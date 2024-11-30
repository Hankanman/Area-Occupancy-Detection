"""Config flow with options for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.components.media_player import DOMAIN as MEDIA_PLAYER_DOMAIN
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN

from .const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_APPLIANCES,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_TYPE,
    DEFAULT_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_TYPE,
)

_LOGGER = logging.getLogger(__name__)


def get_config_schema(
    defaults: dict[str, Any] | None = None,
) -> dict[vol.Required | vol.Optional, Any]:
    """Get the config schema with optional defaults."""
    defaults = defaults or {}

    return {
        vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, "")): str,
        # Required sensors section
        vol.Required(
            CONF_MOTION_SENSORS, default=defaults.get(CONF_MOTION_SENSORS, [])
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain=BINARY_SENSOR_DOMAIN,
                device_class="motion",
                multiple=True,
            ),
        ),
        # Media devices section
        vol.Optional(
            CONF_MEDIA_DEVICES,
            default=defaults.get(CONF_MEDIA_DEVICES, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain=[MEDIA_PLAYER_DOMAIN],
                multiple=True,
            ),
        ),
        # Appliances section
        vol.Optional(
            CONF_APPLIANCES,
            default=defaults.get(CONF_APPLIANCES, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain=[BINARY_SENSOR_DOMAIN, SWITCH_DOMAIN],
                device_class=["power", "plug", "outlet"],
                multiple=True,
            ),
        ),
        # Environmental sensors section
        vol.Optional(
            CONF_ILLUMINANCE_SENSORS,
            default=defaults.get(CONF_ILLUMINANCE_SENSORS, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain=SENSOR_DOMAIN,
                device_class="illuminance",
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_HUMIDITY_SENSORS,
            default=defaults.get(CONF_HUMIDITY_SENSORS, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain=SENSOR_DOMAIN,
                device_class="humidity",
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_TEMPERATURE_SENSORS,
            default=defaults.get(CONF_TEMPERATURE_SENSORS, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain=SENSOR_DOMAIN,
                device_class="temperature",
                multiple=True,
            ),
        ),
        # Configuration options section
        vol.Optional(
            CONF_THRESHOLD,
            default=defaults.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0.0,
                max=1.0,
                step=0.05,
                mode="slider",
            ),
        ),
        vol.Optional(
            CONF_HISTORY_PERIOD,
            default=defaults.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=30,
                step=1,
                mode="slider",
                unit_of_measurement="days",
            ),
        ),
        # Decay configuration section
        vol.Optional(
            CONF_DECAY_ENABLED,
            default=defaults.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
        ): selector.BooleanSelector(),
        vol.Optional(
            CONF_DECAY_WINDOW,
            default=defaults.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=60,
                max=3600,
                step=60,
                mode="slider",
                unit_of_measurement="seconds",
            ),
        ),
        vol.Optional(
            CONF_DECAY_TYPE,
            default=defaults.get(CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE),
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=["linear", "exponential"],
                mode="dropdown",
            ),
        ),
    }


class AreaOccupancyConfigFlow(
    config_entries.ConfigFlow, domain=DOMAIN
):  # pylint: disable=abstract-method
    """Handle a config flow for Area Occupancy Detection."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self.config_data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Validate the configuration
                if not user_input.get(CONF_MOTION_SENSORS):
                    errors["base"] = "no_motion_sensors"
                else:
                    # Check for duplicate entries
                    await self.async_set_unique_id(user_input[CONF_NAME])
                    self._abort_if_unique_id_configured()

                    # Convert any legacy device_states to new categories
                    if "device_states" in user_input:
                        # Attempt to categorize legacy device states
                        media_devices = []
                        appliances = []
                        for entity_id in user_input["device_states"]:
                            if entity_id.startswith(MEDIA_PLAYER_DOMAIN + "."):
                                media_devices.append(entity_id)
                            else:
                                appliances.append(entity_id)

                        # Update configuration with new categories
                        user_input[CONF_MEDIA_DEVICES] = (
                            user_input.get(CONF_MEDIA_DEVICES, []) + media_devices
                        )
                        user_input[CONF_APPLIANCES] = (
                            user_input.get(CONF_APPLIANCES, []) + appliances
                        )
                        del user_input["device_states"]

                    self.config_data = user_input
                    return self.async_create_entry(
                        title=user_input[CONF_NAME],
                        data=user_input,
                    )

            except Exception as error:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected error occurred: %s", error)
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(get_config_schema(self.config_data)),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> AreaOccupancyOptionsFlow:
        """Get the options flow for this handler."""
        return AreaOccupancyOptionsFlow(config_entry)


class AreaOccupancyOptionsFlow(config_entries.OptionsFlow):
    """Handle Area Occupancy options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage Area Occupancy options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Validate configurations
                if not user_input.get(CONF_MOTION_SENSORS):
                    errors["base"] = "no_motion_sensors"
                elif not 0 <= user_input.get(CONF_THRESHOLD, DEFAULT_THRESHOLD) <= 1:
                    errors[CONF_THRESHOLD] = "invalid_threshold"
                else:
                    # Handle conversion of legacy configurations
                    if "device_states" in user_input:
                        media_devices = []
                        appliances = []
                        for entity_id in user_input["device_states"]:
                            if entity_id.startswith(MEDIA_PLAYER_DOMAIN + "."):
                                media_devices.append(entity_id)
                            else:
                                appliances.append(entity_id)

                        user_input[CONF_MEDIA_DEVICES] = (
                            user_input.get(CONF_MEDIA_DEVICES, []) + media_devices
                        )
                        user_input[CONF_APPLIANCES] = (
                            user_input.get(CONF_APPLIANCES, []) + appliances
                        )
                        del user_input["device_states"]

                    return self.async_create_entry(
                        title="",
                        data={
                            CONF_NAME: self.config_entry.data[CONF_NAME],
                            **user_input,
                        },
                    )

            except Exception as error:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected error occurred: %s", error)
                errors["base"] = "unknown"

        # Remove name from schema as it shouldn't be changed
        options_schema = get_config_schema(self.config_entry.data)
        if CONF_NAME in options_schema:
            del options_schema[CONF_NAME]

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(options_schema),
            errors=errors,
        )
