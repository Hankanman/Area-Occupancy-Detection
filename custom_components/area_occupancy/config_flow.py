"""Area Occupancy Detection config and options flow."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    OptionsFlow,
    OptionsFlowWithConfigEntry,
)
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

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

_LOGGER = logging.getLogger(__name__)

# Step IDs
STEP_USER = "user"
STEP_MOTION = "motion_sensors"
STEP_OPTIONAL = "optional_sensors"


class AreaOccupancyConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize config flow."""
        self._data: dict[str, Any] = {}
        self._title: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._title = user_input[CONF_NAME]
            self._data = {CONF_NAME: user_input[CONF_NAME]}
            return await self.async_step_motion()

        return self.async_show_form(
            step_id=STEP_USER,
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME): str,
                }
            ),
            errors=errors,
        )

    async def async_step_motion(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure required motion sensors."""
        errors: dict[str, str] = {}

        if user_input is not None:
            if not user_input.get(CONF_MOTION_SENSORS):
                errors["base"] = "no_motion_sensors"
            else:
                self._data.update(
                    {CONF_MOTION_SENSORS: user_input[CONF_MOTION_SENSORS]}
                )
                return await self.async_step_optional()

        return self.async_show_form(
            step_id=STEP_MOTION,
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MOTION_SENSORS,
                        description={
                            "suggested_value": self._data.get(CONF_MOTION_SENSORS, [])
                        },
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="binary_sensor",
                            device_class="motion",
                            multiple=True,
                        ),
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_optional(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure optional sensors and settings."""
        if user_input is not None:
            # Create entry with only core data
            return self.async_create_entry(
                title=self._title or "",
                data=self._data,
                options=user_input,
            )

        return self.async_show_form(
            step_id=STEP_OPTIONAL,
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MEDIA_DEVICES, default=[]
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="media_player",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(CONF_APPLIANCES, default=[]): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["binary_sensor", "switch"],
                            device_class=["power", "plug", "outlet"],
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_ILLUMINANCE_SENSORS, default=[]
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="illuminance",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_HUMIDITY_SENSORS, default=[]
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="humidity",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE_SENSORS, default=[]
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="temperature",
                            multiple=True,
                        ),
                    ),
                }
            ),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> AreaOccupancyOptionsFlow:
        """Get the options flow for this handler."""
        return AreaOccupancyOptionsFlow(config_entry)


class AreaOccupancyOptionsFlow(OptionsFlowWithConfigEntry):
    """Handle Area Occupancy options."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate motion sensors are still configured
            if not user_input.get(CONF_MOTION_SENSORS):
                errors["base"] = "no_motion_sensors"
            else:
                return self.async_create_entry(
                    title="",
                    data=user_input,
                )

        # Prepare default values from current options
        options = {
            **self.options,
        }

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MOTION_SENSORS,
                        default=options.get(CONF_MOTION_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="binary_sensor",
                            device_class="motion",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_MEDIA_DEVICES,
                        default=options.get(CONF_MEDIA_DEVICES, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="media_player",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_APPLIANCES,
                        default=options.get(CONF_APPLIANCES, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["binary_sensor", "switch"],
                            device_class=["power", "plug", "outlet"],
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_ILLUMINANCE_SENSORS,
                        default=options.get(CONF_ILLUMINANCE_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="illuminance",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_HUMIDITY_SENSORS,
                        default=options.get(CONF_HUMIDITY_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="humidity",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE_SENSORS,
                        default=options.get(CONF_TEMPERATURE_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="temperature",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_THRESHOLD,
                        default=options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
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
                        default=options.get(
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
                        default=options.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_DECAY_WINDOW,
                        default=options.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
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
                        default=options.get(CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=["linear", "exponential"],
                            mode="dropdown",
                            translation_key="decay_type",
                        ),
                    ),
                    vol.Optional(
                        CONF_HISTORICAL_ANALYSIS_ENABLED,
                        default=options.get(
                            CONF_HISTORICAL_ANALYSIS_ENABLED,
                            DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                        ),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_MINIMUM_CONFIDENCE,
                        default=options.get(
                            CONF_MINIMUM_CONFIDENCE, DEFAULT_MINIMUM_CONFIDENCE
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            mode="slider",
                        ),
                    ),
                }
            ),
            errors=errors,
        )
