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

from .const import (
    CONF_MOTION_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_DEVICE_STATES,
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
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class RoomOccupancyConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Room Occupancy Detection."""

    VERSION = 1

    def is_matching(self, match_config: dict[str, Any]) -> bool:
        """Return True if an existing config entry matches the match_config."""
        if not match_config.get(CONF_NAME):
            return False

        existing_entries = self._async_current_entries()

        for entry in existing_entries:
            if entry.data.get(CONF_NAME) == match_config[CONF_NAME]:
                return True

        return False

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> RoomOccupancyOptionsFlow:
        """Get the options flow for this handler."""
        return RoomOccupancyOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            try:
                # Check if a config entry with this name already exists
                if self.is_matching(user_input):
                    return self.async_abort(reason="already_configured")

                await self.async_set_unique_id(user_input[CONF_NAME])
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=user_input[CONF_NAME],
                    data=user_input,
                )
            except Exception as error:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected error occurred: %s", error)
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME): str,
                    vol.Required(CONF_MOTION_SENSORS): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="binary_sensor",
                            device_class="motion",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(CONF_ILLUMINANCE_SENSORS): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="illuminance",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(CONF_HUMIDITY_SENSORS): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="humidity",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(CONF_TEMPERATURE_SENSORS): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="temperature",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(CONF_DEVICE_STATES): selector.EntitySelector(
                        selector.EntitySelectorConfig(multiple=True),
                    ),
                    vol.Optional(
                        CONF_THRESHOLD, default=DEFAULT_THRESHOLD
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            mode="slider",
                        ),
                    ),
                    vol.Optional(
                        CONF_HISTORY_PERIOD, default=DEFAULT_HISTORY_PERIOD
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
                        CONF_DECAY_ENABLED, default=DEFAULT_DECAY_ENABLED
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_DECAY_WINDOW, default=DEFAULT_DECAY_WINDOW
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
                        CONF_DECAY_TYPE, default=DEFAULT_DECAY_TYPE
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=["linear", "exponential"],
                            mode="dropdown",
                        ),
                    ),
                }
            ),
            errors=errors,
        )


class RoomOccupancyOptionsFlow(config_entries.OptionsFlow):
    """Handle Room Occupancy options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage Room Occupancy options."""
        errors = {}

        if user_input is not None:
            # Update the config entry
            return self.async_create_entry(title="", data=user_input)

        # Pre-fill form with current values or defaults
        options = {
            vol.Optional(
                CONF_MOTION_SENSORS,
                default=self.config_entry.data.get(CONF_MOTION_SENSORS, []),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="binary_sensor",
                    device_class="motion",
                    multiple=True,
                ),
            ),
            vol.Optional(
                CONF_ILLUMINANCE_SENSORS,
                default=self.config_entry.data.get(CONF_ILLUMINANCE_SENSORS, []),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="illuminance",
                    multiple=True,
                ),
            ),
            vol.Optional(
                CONF_HUMIDITY_SENSORS,
                default=self.config_entry.data.get(CONF_HUMIDITY_SENSORS, []),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="humidity",
                    multiple=True,
                ),
            ),
            vol.Optional(
                CONF_TEMPERATURE_SENSORS,
                default=self.config_entry.data.get(CONF_TEMPERATURE_SENSORS, []),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="temperature",
                    multiple=True,
                ),
            ),
            vol.Optional(
                CONF_DEVICE_STATES,
                default=self.config_entry.data.get(CONF_DEVICE_STATES, []),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(multiple=True),
            ),
            vol.Optional(
                CONF_THRESHOLD,
                default=self.config_entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
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
                default=self.config_entry.data.get(
                    CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                ),
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
                default=self.config_entry.data.get(
                    CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED
                ),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_DECAY_WINDOW,
                default=self.config_entry.data.get(
                    CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
                ),
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
                default=self.config_entry.data.get(CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=["linear", "exponential"],
                    mode="dropdown",
                ),
            ),
        }

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(options),
            errors=errors,
        )
