"""Config flow with options for Room Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.schema_config_entry_flow import (
    SchemaOptionsFlowHandler,
)

from .const import (
    CONF_DECAY_ENABLED,
    CONF_DECAY_TYPE,
    CONF_DECAY_WINDOW,
    CONF_DEVICE_STATES,
    CONF_HISTORY_PERIOD,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MOTION_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_TYPE,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_THRESHOLD,
    DOMAIN,
    RoomOccupancyConfig,
)

_LOGGER = logging.getLogger(__name__)


def get_config_schema(
    defaults: dict[str, Any] | None = None,
) -> dict[vol.Required | vol.Optional, Any]:
    """Get the config schema with optional defaults."""
    defaults = defaults or {}

    return {
        vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, "")): str,
        vol.Required(
            CONF_MOTION_SENSORS, default=defaults.get(CONF_MOTION_SENSORS, [])
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="binary_sensor",
                device_class="motion",
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_ILLUMINANCE_SENSORS,
            default=defaults.get(CONF_ILLUMINANCE_SENSORS, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="illuminance",
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_HUMIDITY_SENSORS,
            default=defaults.get(CONF_HUMIDITY_SENSORS, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="humidity",
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_TEMPERATURE_SENSORS,
            default=defaults.get(CONF_TEMPERATURE_SENSORS, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="temperature",
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_DEVICE_STATES,
            default=defaults.get(CONF_DEVICE_STATES, []),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(multiple=True),
        ),
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


class RoomOccupancyConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Room Occupancy Detection."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self.config_data: RoomOccupancyConfig = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Check for duplicate entries
                await self.async_set_unique_id(user_input[CONF_NAME])
                self._abort_if_unique_id_configured()

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
            data_schema=vol.Schema(get_config_schema()),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> RoomOccupancyOptionsFlow:
        """Get the options flow for this handler."""
        return RoomOccupancyOptionsFlow(config_entry)


class RoomOccupancyOptionsFlow(SchemaOptionsFlowHandler):
    """Handle Room Occupancy options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        # Initialize with config entry and empty options flow schema
        super().__init__(config_entry, {})
        self.options = get_config_schema(config_entry.data)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage Room Occupancy options."""
        if user_input is not None:
            # Preserve the room name from the original config
            user_input[CONF_NAME] = self.config_entry.data[CONF_NAME]
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(self.options),
        )
