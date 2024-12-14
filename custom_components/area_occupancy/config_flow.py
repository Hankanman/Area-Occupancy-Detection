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
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
)
from homeassistant.exceptions import HomeAssistantError

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
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_AREA_ID,
    DEFAULT_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
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
                device_class=[
                    BinarySensorDeviceClass.MOTION,
                    BinarySensorDeviceClass.OCCUPANCY,
                ],
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
                min=0,
                max=100,
                step=1,
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
            CONF_HISTORICAL_ANALYSIS_ENABLED,
            default=defaults.get(
                CONF_HISTORICAL_ANALYSIS_ENABLED, DEFAULT_HISTORICAL_ANALYSIS_ENABLED
            ),
        ): BooleanSelector(),
    }


class BaseOccupancyFlow:
    """Base class for config and options flow."""

    def _validate_config(self, data: dict[str, Any]) -> None:
        """Validate configuration data."""
        # Core validation
        if CONF_NAME in data and not data.get(CONF_NAME):
            raise HomeAssistantError("Name is required")

        if CONF_MOTION_SENSORS in data and not data.get(CONF_MOTION_SENSORS):
            raise HomeAssistantError("At least one motion sensor is required")

        # Numeric bounds validation
        bounds = {
            CONF_THRESHOLD: (0, 100),
            CONF_HISTORY_PERIOD: (1, 30),
            CONF_DECAY_WINDOW: (60, 3600),
        }

        for field, (min_val, max_val) in bounds.items():
            if field in data and not min_val <= data[field] <= max_val:
                raise HomeAssistantError(
                    f"{field.replace('_', ' ').title()} must be between {min_val} and {max_val}"
                )

    async def _handle_step(
        self,
        step_id: str,
        schema_func: callable,
        next_step: str | None = None,
        validate: bool = False,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Generic step handler."""
        # Add type hint to help IDE and type checking
        self: ConfigFlow | OptionsFlowWithConfigEntry  # type: ignore[assignment]

        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                if validate:
                    self._validate_config(user_input)

                if step_id == "user":
                    # Special handling for initial step
                    area_id = str(uuid.uuid4())
                    await self.async_set_unique_id(area_id)
                    self._abort_if_unique_id_configured()
                    self._core_data = {
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_AREA_ID: area_id,
                        CONF_MOTION_SENSORS: user_input[CONF_MOTION_SENSORS],
                    }
                elif step_id == "parameters":
                    # Special handling for final step
                    self._options_data.update(user_input)
                    if isinstance(self, OptionsFlowWithConfigEntry):
                        return self.async_create_entry(
                            title="", data=self._options_data
                        )
                    return self.async_create_entry(
                        title=self._core_data.get(CONF_NAME, ""),
                        data=self._core_data,
                        options=self._options_data,
                    )
                else:
                    self._options_data.update(user_input)

                if next_step:
                    return await getattr(self, f"async_step_{next_step}")()

            except HomeAssistantError as err:
                _LOGGER.error("Validation error: %s", err)
                errors["base"] = str(err)
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error("Unexpected error: %s", err)
                errors["base"] = "unknown"

        schema = schema_func(
            self._options_data if hasattr(self, "_options_data") else None
        )
        return self.async_show_form(
            step_id=step_id,
            data_schema=vol.Schema(schema),
            errors=errors,
            description_placeholders=(
                {"name": self._core_data.get(CONF_NAME)}
                if hasattr(self, "_core_data")
                else None
            ),
        )


class AreaOccupancyConfigFlow(ConfigFlow, BaseOccupancyFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize config flow."""
        self._core_data: dict[str, Any] = {}
        self._options_data: dict[str, Any] = {}

    def is_matching(self, other_flow: ConfigEntry) -> bool:
        """Check if the entry matches the current flow."""
        return other_flow.data.get(CONF_AREA_ID) == getattr(self, "_core_data", {}).get(
            CONF_AREA_ID
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        return await self._handle_step(
            "user", create_form_schema, "devices", True, user_input
        )

    async def async_step_devices(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle device configuration."""
        return await self._handle_step(
            "devices", create_device_schema, "environmental", False, user_input
        )

    async def async_step_environmental(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle environmental sensor configuration."""
        return await self._handle_step(
            "environmental",
            create_environmental_schema,
            "parameters",
            False,
            user_input,
        )

    async def async_step_parameters(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle parameter configuration."""
        return await self._handle_step(
            "parameters", create_parameters_schema, None, True, user_input
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> "AreaOccupancyOptionsFlow":
        """Get the options flow."""
        return AreaOccupancyOptionsFlow(config_entry)


class AreaOccupancyOptionsFlow(OptionsFlowWithConfigEntry, BaseOccupancyFlow):
    """Handle options flow."""

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
        schema = {
            vol.Required(
                CONF_MOTION_SENSORS,
                default=self._core_data.get(CONF_MOTION_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.BINARY_SENSOR,
                    device_class=[
                        BinarySensorDeviceClass.MOTION,
                        BinarySensorDeviceClass.OCCUPANCY,
                    ],
                    multiple=True,
                )
            )
        }
        return await self._handle_step(
            "motion", lambda x: schema, "devices", True, user_input
        )

    # Reuse the same step methods from config flow
    async def async_step_devices(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle device options."""
        return await self._handle_step(
            "devices", create_device_schema, "environmental", False, user_input
        )

    async def async_step_environmental(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle environmental sensor options."""
        return await self._handle_step(
            "environmental",
            create_environmental_schema,
            "parameters",
            False,
            user_input,
        )

    async def async_step_parameters(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle parameter options."""
        return await self._handle_step(
            "parameters", create_parameters_schema, None, True, user_input
        )
