# custom_components/area_occupancy/config_flow.py

"""Config flow for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_APPLIANCES,
    CONF_DEVICE_STATES,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_TYPE,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_MINIMUM_CONFIDENCE,
    DEFAULT_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_TYPE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_MINIMUM_CONFIDENCE,
)
from .validation_helpers import (
    validate_threshold,
    validate_decay_window,
    validate_required_sensors,
)

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> "AreaOccupancyOptionsFlow":
        """Get the options flow for this handler."""
        return AreaOccupancyOptionsFlow(config_entry)

    async def async_step_user(self, user_input: dict | None = None) -> FlowResult:
        """Handle the initial step."""
        _LOGGER.debug("Starting config flow: Step 'user'")
        errors = {}

        if user_input is not None:
            _LOGGER.debug("User input received: %s", user_input)
            try:
                # Perform any necessary validation
                if not user_input.get(CONF_NAME):
                    errors["name"] = "Name is required."
                    _LOGGER.warning("Validation failed: Missing name.")
                else:
                    _LOGGER.info("Validation successful. Creating entry.")
                    return self.async_create_entry(
                        title=user_input[CONF_NAME], data=user_input
                    )
            except Exception as e:
                _LOGGER.exception("Error during config flow: %s", e)
                errors["base"] = "unknown_error"

        _LOGGER.debug("Rendering form for user input with errors: %s", errors)
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME): str,
                    vol.Required(CONF_MOTION_SENSORS): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="binary_sensor", device_class="motion", multiple=True
                        )
                    ),
                    vol.Optional(CONF_MEDIA_DEVICES): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="media_player", multiple=True
                        )
                    ),
                    vol.Optional(CONF_APPLIANCES): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["switch", "binary_sensor"], multiple=True
                        )
                    ),
                    vol.Optional(CONF_DEVICE_STATES): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["switch", "binary_sensor"], multiple=True
                        )
                    ),
                    vol.Optional(CONF_ILLUMINANCE_SENSORS): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor", device_class="illuminance", multiple=True
                        )
                    ),
                    vol.Optional(CONF_HUMIDITY_SENSORS): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor", device_class="humidity", multiple=True
                        )
                    ),
                    vol.Optional(CONF_TEMPERATURE_SENSORS): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor", device_class="temperature", multiple=True
                        )
                    ),
                    vol.Optional(
                        CONF_THRESHOLD, default=DEFAULT_THRESHOLD
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05)
                    ),
                    vol.Optional(
                        CONF_HISTORY_PERIOD, default=DEFAULT_HISTORY_PERIOD
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1, max=30, step=1, unit_of_measurement="days"
                        )
                    ),
                    vol.Optional(
                        CONF_DECAY_ENABLED, default=DEFAULT_DECAY_ENABLED
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_DECAY_WINDOW, default=DEFAULT_DECAY_WINDOW
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=60, max=3600, step=60)
                    ),
                    vol.Optional(
                        CONF_DECAY_TYPE, default=DEFAULT_DECAY_TYPE
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=["linear", "exponential"], mode="dropdown"
                        )
                    ),
                    vol.Optional(
                        CONF_HISTORICAL_ANALYSIS_ENABLED,
                        default=DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_MINIMUM_CONFIDENCE, default=DEFAULT_MINIMUM_CONFIDENCE
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05)
                    ),
                }
            ),
            errors=errors,
        )


class AreaOccupancyOptionsFlow(config_entries.OptionsFlow):
    """Handle Area Occupancy options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        _LOGGER.debug(
            "Entered async_step_init for entry ID: %s", self.config_entry.entry_id
        )
        errors: dict[str, str] = {}

        if user_input is not None:
            _LOGGER.debug("User input received in options step: %s", user_input)
            try:
                # Validate inputs
                errors["threshold"] = validate_threshold(
                    user_input.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
                )
                _LOGGER.debug(
                    "Threshold validation complete for entry ID: %s",
                    self.config_entry.entry_id,
                )

                errors["decay_window"] = validate_decay_window(
                    user_input.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)
                )
                _LOGGER.debug(
                    "Decay window validation complete for entry ID: %s",
                    self.config_entry.entry_id,
                )

                errors["motion_sensors"] = validate_required_sensors(
                    user_input.get(CONF_MOTION_SENSORS, [])
                )
                _LOGGER.debug(
                    "Motion sensors validation complete for entry ID: %s",
                    self.config_entry.entry_id,
                )

                # Filter out None values from errors
                errors = {
                    key: value for key, value in errors.items() if value is not None
                }
                if errors:
                    _LOGGER.warning("Validation errors found: %s", errors)
                else:
                    _LOGGER.info(
                        "Validation successful for options step. Creating entry for entry ID: %s",
                        self.config_entry.entry_id,
                    )
                    return self.async_create_entry(title="", data=user_input)
            except Exception as e:
                _LOGGER.exception(
                    "Error occurred during options step for entry ID: %s: %s",
                    self.config_entry.entry_id,
                    e,
                )
                errors["base"] = "unknown_error"

        _LOGGER.debug(
            "Rendering options form for entry ID: %s with errors: %s",
            self.config_entry.entry_id,
            errors,
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MOTION_SENSORS,
                        default=self.config_entry.data.get(CONF_MOTION_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="binary_sensor", device_class="motion", multiple=True
                        )
                    ),
                    vol.Optional(
                        CONF_MEDIA_DEVICES,
                        default=self.config_entry.data.get(CONF_MEDIA_DEVICES, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="media_player", multiple=True
                        )
                    ),
                    vol.Optional(
                        CONF_APPLIANCES,
                        default=self.config_entry.data.get(CONF_APPLIANCES, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["switch", "binary_sensor"], multiple=True
                        )
                    ),
                    vol.Optional(
                        CONF_DEVICE_STATES,
                        default=self.config_entry.data.get(CONF_DEVICE_STATES, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["switch", "binary_sensor"], multiple=True
                        )
                    ),
                    vol.Optional(
                        CONF_ILLUMINANCE_SENSORS,
                        default=self.config_entry.data.get(
                            CONF_ILLUMINANCE_SENSORS, []
                        ),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor", device_class="illuminance", multiple=True
                        )
                    ),
                    vol.Optional(
                        CONF_HUMIDITY_SENSORS,
                        default=self.config_entry.data.get(CONF_HUMIDITY_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor", device_class="humidity", multiple=True
                        )
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE_SENSORS,
                        default=self.config_entry.data.get(
                            CONF_TEMPERATURE_SENSORS, []
                        ),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor", device_class="temperature", multiple=True
                        )
                    ),
                    vol.Optional(
                        CONF_THRESHOLD,
                        default=self.config_entry.data.get(
                            CONF_THRESHOLD, DEFAULT_THRESHOLD
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05)
                    ),
                    vol.Optional(
                        CONF_HISTORY_PERIOD,
                        default=self.config_entry.data.get(
                            CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1, max=30, step=1, unit_of_measurement="days"
                        )
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
                        selector.NumberSelectorConfig(min=60, max=3600, step=60)
                    ),
                    vol.Optional(
                        CONF_DECAY_TYPE,
                        default=self.config_entry.data.get(
                            CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE
                        ),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=["linear", "exponential"], mode="dropdown"
                        )
                    ),
                    vol.Optional(
                        CONF_HISTORICAL_ANALYSIS_ENABLED,
                        default=self.config_entry.data.get(
                            CONF_HISTORICAL_ANALYSIS_ENABLED,
                            DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                        ),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_MINIMUM_CONFIDENCE,
                        default=self.config_entry.data.get(
                            CONF_MINIMUM_CONFIDENCE, DEFAULT_MINIMUM_CONFIDENCE
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05)
                    ),
                }
            ),
            errors=errors,
        )
