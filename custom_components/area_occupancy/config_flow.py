"""Config flow for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, OptionsFlow
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError

from .const import DOMAIN, CONF_MOTION_SENSORS
from .config_management import ConfigManager, CoreConfig, OptionsConfig

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._core_config: CoreConfig | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial setup step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Split configuration
                core_data = {CONF_NAME: user_input[CONF_NAME]}
                options_data = {k: v for k, v in user_input.items() if k != CONF_NAME}

                # Validate both parts
                core_config = ConfigManager.validate_core_config(core_data)
                options_config = ConfigManager.validate_options(options_data)

                # Check for duplicate entries
                await self.async_set_unique_id(core_config["name"])
                self._abort_if_unique_id_configured()

                # Store validated configuration
                return self.async_create_entry(
                    title=core_config["name"],
                    data=core_config,
                    options=options_config,
                )

            except vol.Invalid as err:
                if CONF_MOTION_SENSORS in str(err):
                    errors["base"] = "no_motion_sensors"
                else:
                    errors["base"] = "unknown"
            except Exception as err:
                _LOGGER.exception("Unexpected error occurred: %s", err)
                errors["base"] = "unknown"

        # Show configuration form with both core and options schema
        schema = {
            **ConfigManager.get_core_schema(self._core_config or {}).schema,
            **ConfigManager.get_options_schema({}).schema,
        }

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> AreaOccupancyOptionsFlow:
        """Get the options flow for this handler."""
        return AreaOccupancyOptionsFlow(config_entry)


class AreaOccupancyOptionsFlow(OptionsFlow):
    """Handle Area Occupancy options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage Area Occupancy options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Validate options including motion sensors
                options = ConfigManager.validate_options(user_input)

                return self.async_create_entry(
                    title="",
                    data=options,
                )

            except vol.Invalid as err:
                if CONF_MOTION_SENSORS in str(err):
                    errors["base"] = "no_motion_sensors"
                else:
                    errors["base"] = "invalid_options"
            except Exception as err:
                _LOGGER.exception("Unexpected error occurred: %s", err)
                errors["base"] = "unknown"

        # Show options form with current values including motion sensors
        return self.async_show_form(
            step_id="init",
            data_schema=ConfigManager.get_options_schema(self.config_entry.options),
            errors=errors,
        )
