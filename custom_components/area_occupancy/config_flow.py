"""Area Occupancy Detection config and options flow."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
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
                name = user_input[CONF_NAME]
                motion_sensors = user_input[CONF_MOTION_SENSORS]

                # Validate motion sensors are provided
                if not motion_sensors:
                    errors["base"] = "no_motion_sensors"
                else:
                    # Generate unique ID for the area
                    area_id = str(uuid.uuid4())

                    # Set unique ID based on generated ID
                    await self.async_set_unique_id(area_id)
                    self._abort_if_unique_id_configured()

                    # Store core data with area ID
                    self._core_data = {
                        CONF_NAME: name,
                        CONF_MOTION_SENSORS: motion_sensors,
                        CONF_AREA_ID: area_id,
                    }

                    return await self.async_step_devices()

            except Exception:
                errors["base"] = "unknown"

        # Show initial form
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
                }
            ),
            errors=errors,
        )

    async def async_step_devices(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle device configuration step."""
        if user_input is not None:
            # Store device configuration in options
            self._options_data.update(
                {
                    CONF_MEDIA_DEVICES: user_input.get(CONF_MEDIA_DEVICES, []),
                    CONF_APPLIANCES: user_input.get(CONF_APPLIANCES, []),
                }
            )
            return await self.async_step_environmental()

        return self.async_show_form(
            step_id="devices",
            description_placeholders={"name": self._core_data[CONF_NAME]},
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
                }
            ),
        )

    async def async_step_environmental(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle environmental sensor configuration step."""
        if user_input is not None:
            # Store environmental sensor configuration in options
            self._options_data.update(
                {
                    CONF_ILLUMINANCE_SENSORS: user_input.get(
                        CONF_ILLUMINANCE_SENSORS, []
                    ),
                    CONF_HUMIDITY_SENSORS: user_input.get(CONF_HUMIDITY_SENSORS, []),
                    CONF_TEMPERATURE_SENSORS: user_input.get(
                        CONF_TEMPERATURE_SENSORS, []
                    ),
                }
            )
            return await self.async_step_parameters()

        return self.async_show_form(
            step_id="environmental",
            description_placeholders={"name": self._core_data[CONF_NAME]},
            data_schema=vol.Schema(
                {
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

    async def async_step_parameters(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle parameter configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Validate parameters
                if not 0 <= user_input[CONF_THRESHOLD] <= 1:
                    errors["threshold"] = "invalid_threshold"
                elif not 1 <= user_input[CONF_HISTORY_PERIOD] <= 30:
                    errors["history_period"] = "invalid_history"
                elif not 60 <= user_input[CONF_DECAY_WINDOW] <= 3600:
                    errors["decay_window"] = "invalid_decay"
                elif not 0 <= user_input[CONF_MINIMUM_CONFIDENCE] <= 1:
                    errors["minimum_confidence"] = "invalid_confidence"
                else:
                    # Store parameter configuration in options
                    self._options_data.update(user_input)

                    # Create entry with separated core data and options
                    return self.async_create_entry(
                        title=self._core_data[CONF_NAME],
                        data=self._core_data,
                        options=self._options_data,
                    )

            except Exception:
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="parameters",
            description_placeholders={"name": self._core_data[CONF_NAME]},
            data_schema=vol.Schema(
                {
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
                            translation_key="decay_type",
                        ),
                    ),
                    vol.Optional(
                        CONF_HISTORICAL_ANALYSIS_ENABLED,
                        default=DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_MINIMUM_CONFIDENCE, default=DEFAULT_MINIMUM_CONFIDENCE
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

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> AreaOccupancyOptionsFlow:
        """Get the options flow for this handler."""
        return AreaOccupancyOptionsFlow(config_entry)


class AreaOccupancyOptionsFlow(OptionsFlowWithConfigEntry):
    """Handle Area Occupancy options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        super().__init__(config_entry)
        self.current_options = dict(config_entry.options)
        self._temp_options: dict[str, Any] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options starting step."""
        return await self.async_step_motion()

    async def async_step_motion(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle motion sensor configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            if not user_input.get(CONF_MOTION_SENSORS):
                errors["base"] = "no_motion_sensors"
            else:
                # Update core config with new motion sensors
                updated_data = dict(self.config_entry.data)
                updated_data[CONF_MOTION_SENSORS] = user_input[CONF_MOTION_SENSORS]

                # Update the config entry's core data
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data=updated_data,
                )

                # Continue to next step without storing in options
                return await self.async_step_devices()

        return self.async_show_form(
            step_id="motion",
            description_placeholders={"name": self.config_entry.title},
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MOTION_SENSORS,
                        default=self.config_entry.data.get(CONF_MOTION_SENSORS, []),
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

    async def async_step_devices(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle device configuration step."""
        if user_input is not None:
            self._temp_options.update(
                {
                    CONF_MEDIA_DEVICES: user_input.get(CONF_MEDIA_DEVICES, []),
                    CONF_APPLIANCES: user_input.get(CONF_APPLIANCES, []),
                }
            )
            return await self.async_step_environmental()

        return self.async_show_form(
            step_id="devices",
            description_placeholders={"name": self.config_entry.title},
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MEDIA_DEVICES,
                        default=self.current_options.get(CONF_MEDIA_DEVICES, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="media_player",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_APPLIANCES,
                        default=self.current_options.get(CONF_APPLIANCES, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["binary_sensor", "switch"],
                            device_class=["power", "plug", "outlet"],
                            multiple=True,
                        ),
                    ),
                }
            ),
        )

    async def async_step_environmental(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle environmental sensor configuration step."""
        if user_input is not None:
            self._temp_options.update(
                {
                    CONF_ILLUMINANCE_SENSORS: user_input.get(
                        CONF_ILLUMINANCE_SENSORS, []
                    ),
                    CONF_HUMIDITY_SENSORS: user_input.get(CONF_HUMIDITY_SENSORS, []),
                    CONF_TEMPERATURE_SENSORS: user_input.get(
                        CONF_TEMPERATURE_SENSORS, []
                    ),
                }
            )
            return await self.async_step_parameters()

        return self.async_show_form(
            step_id="environmental",
            description_placeholders={"name": self.config_entry.title},
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_ILLUMINANCE_SENSORS,
                        default=self.current_options.get(CONF_ILLUMINANCE_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="illuminance",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_HUMIDITY_SENSORS,
                        default=self.current_options.get(CONF_HUMIDITY_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="humidity",
                            multiple=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE_SENSORS,
                        default=self.current_options.get(CONF_TEMPERATURE_SENSORS, []),
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

    async def async_step_parameters(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle parameter configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Validate parameters
                if not 0 <= user_input[CONF_THRESHOLD] <= 1:
                    errors["threshold"] = "invalid_threshold"
                elif not 1 <= user_input[CONF_HISTORY_PERIOD] <= 30:
                    errors["history_period"] = "invalid_history"
                elif not 60 <= user_input[CONF_DECAY_WINDOW] <= 3600:
                    errors["decay_window"] = "invalid_decay"
                elif not 0 <= user_input[CONF_MINIMUM_CONFIDENCE] <= 1:
                    errors["minimum_confidence"] = "invalid_confidence"
                else:
                    self._temp_options.update(user_input)
                    return self.async_create_entry(title="", data=self._temp_options)

            except Exception:
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="parameters",
            description_placeholders={"name": self.config_entry.title},
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_THRESHOLD,
                        default=self.current_options.get(
                            CONF_THRESHOLD, DEFAULT_THRESHOLD
                        ),
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
                        default=self.current_options.get(
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
                        default=self.current_options.get(
                            CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED
                        ),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_DECAY_WINDOW,
                        default=self.current_options.get(
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
                        default=self.current_options.get(
                            CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE
                        ),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=["linear", "exponential"],
                            mode="dropdown",
                            translation_key="decay_type",
                        ),
                    ),
                    vol.Optional(
                        CONF_HISTORICAL_ANALYSIS_ENABLED,
                        default=self.current_options.get(
                            CONF_HISTORICAL_ANALYSIS_ENABLED,
                            DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                        ),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_MINIMUM_CONFIDENCE,
                        default=self.current_options.get(
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
