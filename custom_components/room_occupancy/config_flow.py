"""Config flow for Room Occupancy Detection integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
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

    def is_matching(self, other_flow: dict[str, Any]) -> bool:
        """Check if the user input matches the criteria for this config flow."""
        # Implement the matching logic here
        return True

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            # Validate the input
            try:
                # Check if entry already exists
                await self.async_set_unique_id(user_input[CONF_NAME])
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=user_input[CONF_NAME],
                    data=user_input,
                )
            except (ValueError, KeyError) as error:
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
                    ): bool,
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
