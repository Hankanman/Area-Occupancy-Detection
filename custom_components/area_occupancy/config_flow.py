"""Config flow for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import voluptuous as vol

from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    OptionsFlowWithConfigEntry,
)
from homeassistant.const import (
    CONF_NAME,
    Platform,
)
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectSelector,
    SelectSelectorConfig,
)

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
    CONF_AREA_ID,
    DEFAULT_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_TYPE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_MINIMUM_CONFIDENCE,
)

_LOGGER = logging.getLogger(__name__)


def create_form_schema(defaults: dict[str, Any] | None = None) -> dict:
    """Create a form schema with optional default values."""
    if defaults is None:
        defaults = {}

    return {
        vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, "")): str,
        vol.Required(
            CONF_MOTION_SENSORS, default=defaults.get(CONF_MOTION_SENSORS, [])
        ): EntitySelector(
            EntitySelectorConfig(
                domain=Platform.BINARY_SENSOR,
                device_class=BinarySensorDeviceClass.MOTION,
                multiple=True,
            ),
        ),
    }


def create_device_schema(defaults: dict[str, Any] | None = None) -> dict:
    """Create device configuration schema with optional default values."""
    if defaults is None:
        defaults = {}

    return {
        vol.Optional(
            CONF_MEDIA_DEVICES, default=defaults.get(CONF_MEDIA_DEVICES, [])
        ): EntitySelector(
            EntitySelectorConfig(
                domain=Platform.MEDIA_PLAYER,
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_APPLIANCES, default=defaults.get(CONF_APPLIANCES, [])
        ): EntitySelector(
            EntitySelectorConfig(
                domain=[Platform.BINARY_SENSOR, Platform.SWITCH],
                device_class=["power", "plug", "outlet"],
                multiple=True,
            ),
        ),
    }


def create_environmental_schema(defaults: dict[str, Any] | None = None) -> dict:
    """Create environmental sensor schema with optional default values."""
    if defaults is None:
        defaults = {}

    return {
        vol.Optional(
            CONF_ILLUMINANCE_SENSORS, default=defaults.get(CONF_ILLUMINANCE_SENSORS, [])
        ): EntitySelector(
            EntitySelectorConfig(
                domain=Platform.SENSOR,
                device_class=SensorDeviceClass.ILLUMINANCE,
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_HUMIDITY_SENSORS, default=defaults.get(CONF_HUMIDITY_SENSORS, [])
        ): EntitySelector(
            EntitySelectorConfig(
                domain=Platform.SENSOR,
                device_class=SensorDeviceClass.HUMIDITY,
                multiple=True,
            ),
        ),
        vol.Optional(
            CONF_TEMPERATURE_SENSORS, default=defaults.get(CONF_TEMPERATURE_SENSORS, [])
        ): EntitySelector(
            EntitySelectorConfig(
                domain=Platform.SENSOR,
                device_class=SensorDeviceClass.TEMPERATURE,
                multiple=True,
            ),
        ),
    }


def create_parameters_schema(defaults: dict[str, Any] | None = None) -> dict:
    """Create parameters schema with optional default values."""
    if defaults is None:
        defaults = {}

    return {
        vol.Optional(
            CONF_THRESHOLD, default=defaults.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        ): NumberSelector(
            NumberSelectorConfig(
                min=0.0,
                max=1.0,
                step=0.05,
                mode="slider",
            ),
        ),
        vol.Optional(
            CONF_HISTORY_PERIOD,
            default=defaults.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD),
        ): NumberSelector(
            NumberSelectorConfig(
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
        ): BooleanSelector(),
        vol.Optional(
            CONF_DECAY_WINDOW,
            default=defaults.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
        ): NumberSelector(
            NumberSelectorConfig(
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
        ): SelectSelector(
            SelectSelectorConfig(
                options=["linear", "exponential"],
                mode="dropdown",
                translation_key="decay_type",
            ),
        ),
        vol.Optional(
            CONF_HISTORICAL_ANALYSIS_ENABLED,
            default=defaults.get(
                CONF_HISTORICAL_ANALYSIS_ENABLED, DEFAULT_HISTORICAL_ANALYSIS_ENABLED
            ),
        ): BooleanSelector(),
        vol.Optional(
            CONF_MINIMUM_CONFIDENCE,
            default=defaults.get(CONF_MINIMUM_CONFIDENCE, DEFAULT_MINIMUM_CONFIDENCE),
        ): NumberSelector(
            NumberSelectorConfig(
                min=0.0,
                max=1.0,
                step=0.05,
                mode="slider",
            ),
        ),
    }


class AreaOccupancyConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize config flow."""
        self._core_data: dict[str, Any] = {}
        self._options_data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Generate unique ID
                area_id = str(uuid.uuid4())
                await self.async_set_unique_id(area_id)
                self._abort_if_unique_id_configured()

                # Store core configuration
                self._core_data = {
                    CONF_NAME: user_input[CONF_NAME],
                    CONF_AREA_ID: area_id,
                    CONF_MOTION_SENSORS: user_input[CONF_MOTION_SENSORS],
                }

                # Proceed to devices step
                return await self.async_step_devices()

            except Exception as err:
                _LOGGER.error("Error in user step: %s", err)
                errors["base"] = "unknown"

        schema = create_form_schema(self._core_data)
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    async def async_step_devices(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle device configuration."""
        if user_input is not None:
            self._options_data.update(user_input)
            return await self.async_step_environmental()

        schema = create_device_schema(self._options_data)
        return self.async_show_form(
            step_id="devices",
            data_schema=vol.Schema(schema),
            description_placeholders={"name": self._core_data[CONF_NAME]},
        )

    async def async_step_environmental(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle environmental sensor configuration."""
        if user_input is not None:
            self._options_data.update(user_input)
            return await self.async_step_parameters()

        schema = create_environmental_schema(self._options_data)
        return self.async_show_form(
            step_id="environmental",
            data_schema=vol.Schema(schema),
            description_placeholders={"name": self._core_data[CONF_NAME]},
        )

    async def async_step_parameters(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle parameter configuration."""
        if user_input is not None:
            self._options_data.update(user_input)
            return self.async_create_entry(
                title=self._core_data[CONF_NAME],
                data=self._core_data,
                options=self._options_data,
            )

        schema = create_parameters_schema(self._options_data)
        return self.async_show_form(
            step_id="parameters",
            data_schema=vol.Schema(schema),
            description_placeholders={"name": self._core_data[CONF_NAME]},
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> AreaOccupancyOptionsFlow:
        """Get the options flow for this handler."""
        return AreaOccupancyOptionsFlow(config_entry)


class AreaOccupancyOptionsFlow(OptionsFlowWithConfigEntry):
    """Handle Area Occupancy options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        super().__init__(config_entry)
        self._core_data = dict(config_entry.data)
        self._options_data = dict(config_entry.options)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return await self.async_step_motion()

    async def async_step_motion(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle motion sensor options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                if not user_input.get(CONF_MOTION_SENSORS):
                    errors["base"] = "no_motion_sensors"
                else:
                    self._core_data[CONF_MOTION_SENSORS] = user_input[
                        CONF_MOTION_SENSORS
                    ]
                    return await self.async_step_devices()

            except Exception as err:
                _LOGGER.error("Error in motion step: %s", err)
                errors["base"] = "unknown"

        schema = {
            vol.Required(
                CONF_MOTION_SENSORS,
                default=self._core_data.get(CONF_MOTION_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.BINARY_SENSOR,
                    device_class=BinarySensorDeviceClass.MOTION,
                    multiple=True,
                ),
            ),
        }

        return self.async_show_form(
            step_id="motion",
            data_schema=vol.Schema(schema),
            errors=errors,
            description_placeholders={"name": self._core_data[CONF_NAME]},
        )

    async def async_step_devices(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle device options."""
        if user_input is not None:
            self._options_data.update(user_input)
            return await self.async_step_environmental()

        schema = create_device_schema(self._options_data)
        return self.async_show_form(
            step_id="devices",
            data_schema=vol.Schema(schema),
            description_placeholders={"name": self._core_data[CONF_NAME]},
        )

    async def async_step_environmental(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle environmental sensor options."""
        if user_input is not None:
            self._options_data.update(user_input)
            return await self.async_step_parameters()

        schema = create_environmental_schema(self._options_data)
        return self.async_show_form(
            step_id="environmental",
            data_schema=vol.Schema(schema),
            description_placeholders={"name": self._core_data[CONF_NAME]},
        )

    async def async_step_parameters(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle parameter options."""
        if user_input is not None:
            self._options_data.update(user_input)

            # Update config entry
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=self._core_data,
                options=self._options_data,
            )

            return self.async_create_entry(title="", data=self._options_data)

        schema = create_parameters_schema(self._options_data)
        return self.async_show_form(
            step_id="parameters",
            data_schema=vol.Schema(schema),
            description_placeholders={"name": self._core_data[CONF_NAME]},
        )
