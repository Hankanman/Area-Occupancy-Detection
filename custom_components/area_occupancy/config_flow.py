"""Config flow for Area Occupancy Detection integration."""

from __future__ import annotations

import logging
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
from homeassistant.data_entry_flow import FlowResult, section
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectSelector,
    SelectSelectorConfig,
)
from homeassistant.helpers import entity_registry
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_DOOR_SENSORS,
    CONF_DOOR_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    CONF_LIGHTS,
    CONF_THRESHOLD,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_MIN_DELAY,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_VERSION,
    DEFAULT_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_MIN_DELAY,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_WINDOW,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_WINDOW_ACTIVE_STATE,
)

from .state_mapping import (
    get_state_options,
    get_default_state,
)

_LOGGER = logging.getLogger(__name__)


def create_schema(
    hass, defaults: dict[str, Any] | None = None, is_options: bool = False
) -> dict:
    """Create a schema with optional default values."""
    if defaults is None:
        defaults = {}

    registry = entity_registry.async_get(hass)

    # Get state options
    door_states = get_state_options("door")
    door_state_options = [
        {"value": option.value, "label": option.name}
        for option in door_states["options"]
    ]

    media_states = get_state_options("media")
    media_state_options = [
        {"value": option.value, "label": option.name}
        for option in media_states["options"]
    ]

    window_states = get_state_options("window")
    window_state_options = [
        {"value": option.value, "label": option.name}
        for option in window_states["options"]
    ]

    appliance_states = get_state_options("appliance")
    appliance_state_options = [
        {"value": option.value, "label": option.name}
        for option in appliance_states["options"]
    ]

    # Get entity lists
    appliance_excluded_classes = [
        BinarySensorDeviceClass.MOTION,
        BinarySensorDeviceClass.OCCUPANCY,
        BinarySensorDeviceClass.PRESENCE,
        BinarySensorDeviceClass.WINDOW,
        BinarySensorDeviceClass.DOOR,
        BinarySensorDeviceClass.GARAGE_DOOR,
        BinarySensorDeviceClass.OPENING,
        BinarySensorDeviceClass.LIGHT,
    ]

    domains_to_check = ["binary_sensor", "switch", "fan"]
    entity_ids = []
    for domain in domains_to_check:
        entity_ids.extend(hass.states.async_entity_ids(domain))

    include_appliance_entities = []
    for eid in entity_ids:
        state = hass.states.get(eid)
        if state:
            device_class = state.attributes.get("device_class")
            if device_class not in appliance_excluded_classes:
                include_appliance_entities.append(eid)

    include_window_entities = []
    include_door_entities = []
    for entry in registry.entities.values():
        if entry.domain == Platform.BINARY_SENSOR:
            if entry.device_class == BinarySensorDeviceClass.WINDOW or (
                "window" in entry.entity_id.lower()
                and (
                    entry.device_class
                    in [
                        BinarySensorDeviceClass.DOOR,
                        BinarySensorDeviceClass.GARAGE_DOOR,
                        BinarySensorDeviceClass.OPENING,
                        BinarySensorDeviceClass.WINDOW,
                    ]
                    or entry.original_device_class
                    in [
                        BinarySensorDeviceClass.DOOR,
                        BinarySensorDeviceClass.GARAGE_DOOR,
                        BinarySensorDeviceClass.OPENING,
                        BinarySensorDeviceClass.WINDOW,
                    ]
                )
            ):
                include_window_entities.append(entry.entity_id)
            elif entry.device_class == BinarySensorDeviceClass.DOOR or (
                "window" not in entry.entity_id.lower()
                and (
                    entry.device_class
                    in [
                        BinarySensorDeviceClass.DOOR,
                        BinarySensorDeviceClass.GARAGE_DOOR,
                        BinarySensorDeviceClass.OPENING,
                    ]
                    or entry.original_device_class
                    in [
                        BinarySensorDeviceClass.DOOR,
                        BinarySensorDeviceClass.GARAGE_DOOR,
                        BinarySensorDeviceClass.OPENING,
                    ]
                )
            ):
                include_door_entities.append(entry.entity_id)

    schema = {}

    if not is_options:
        schema[vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, ""))] = str

    schema.update(
        {
            vol.Required("motion"): section(
                vol.Schema(
                    {
                        vol.Required(
                            CONF_MOTION_SENSORS,
                            default=defaults.get(CONF_MOTION_SENSORS, []),
                        ): EntitySelector(
                            EntitySelectorConfig(
                                domain=Platform.BINARY_SENSOR,
                                device_class=[
                                    BinarySensorDeviceClass.MOTION,
                                    BinarySensorDeviceClass.OCCUPANCY,
                                    BinarySensorDeviceClass.PRESENCE,
                                ],
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_WEIGHT_MOTION,
                            default=defaults.get(
                                CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION
                            ),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                min=0,
                                max=1,
                                step=0.05,
                                mode="slider",
                            ),
                        ),
                    }
                ),
                {"collapsed": True},
            ),
            vol.Required("doors"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_DOOR_SENSORS,
                            default=defaults.get(CONF_DOOR_SENSORS, []),
                        ): EntitySelector(
                            EntitySelectorConfig(
                                include_entities=include_door_entities,
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_DOOR_ACTIVE_STATE,
                            default=defaults.get(
                                CONF_DOOR_ACTIVE_STATE, get_default_state("door")
                            ),
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=door_state_options,
                                mode="dropdown",
                            )
                        ),
                        vol.Optional(
                            CONF_WEIGHT_DOOR,
                            default=defaults.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                min=0,
                                max=1,
                                step=0.05,
                                mode="slider",
                            ),
                        ),
                    }
                ),
                {"collapsed": True},
            ),
            vol.Required("windows"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_WINDOW_SENSORS,
                            default=defaults.get(CONF_WINDOW_SENSORS, []),
                        ): EntitySelector(
                            EntitySelectorConfig(
                                include_entities=include_window_entities,
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_WINDOW_ACTIVE_STATE,
                            default=defaults.get(
                                CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE
                            ),
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=window_state_options,
                                mode="dropdown",
                            )
                        ),
                        vol.Optional(
                            CONF_WEIGHT_WINDOW,
                            default=defaults.get(
                                CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW
                            ),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                min=0,
                                max=1,
                                step=0.05,
                                mode="slider",
                            ),
                        ),
                    }
                ),
                {"collapsed": True},
            ),
            vol.Required("lights"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_LIGHTS, default=defaults.get(CONF_LIGHTS, [])
                        ): EntitySelector(
                            EntitySelectorConfig(
                                domain=[Platform.LIGHT, Platform.SWITCH],
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_WEIGHT_LIGHT,
                            default=defaults.get(
                                CONF_WEIGHT_LIGHT, DEFAULT_WEIGHT_LIGHT
                            ),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                min=0,
                                max=1,
                                step=0.05,
                                mode="slider",
                            ),
                        ),
                    }
                ),
                {"collapsed": True},
            ),
            vol.Required("media"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_MEDIA_DEVICES,
                            default=defaults.get(CONF_MEDIA_DEVICES, []),
                        ): EntitySelector(
                            EntitySelectorConfig(
                                domain=Platform.MEDIA_PLAYER,
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_MEDIA_ACTIVE_STATES,
                            default=defaults.get(
                                CONF_MEDIA_ACTIVE_STATES, DEFAULT_MEDIA_ACTIVE_STATES
                            ),
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=media_state_options,
                                multiple=True,
                                mode="dropdown",
                            )
                        ),
                        vol.Optional(
                            CONF_WEIGHT_MEDIA,
                            default=defaults.get(
                                CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA
                            ),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                min=0,
                                max=1,
                                step=0.05,
                                mode="slider",
                            ),
                        ),
                    }
                ),
                {"collapsed": True},
            ),
            vol.Required("appliances"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_APPLIANCES, default=defaults.get(CONF_APPLIANCES, [])
                        ): EntitySelector(
                            EntitySelectorConfig(
                                include_entities=include_appliance_entities,
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_APPLIANCE_ACTIVE_STATES,
                            default=defaults.get(
                                CONF_APPLIANCE_ACTIVE_STATES,
                                DEFAULT_APPLIANCE_ACTIVE_STATES,
                            ),
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=appliance_state_options,
                                multiple=True,
                                mode="dropdown",
                            )
                        ),
                        vol.Optional(
                            CONF_WEIGHT_APPLIANCE,
                            default=defaults.get(
                                CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE
                            ),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                min=0,
                                max=1,
                                step=0.05,
                                mode="slider",
                            ),
                        ),
                    }
                ),
                {"collapsed": True},
            ),
            vol.Required("environmental"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_ILLUMINANCE_SENSORS,
                            default=defaults.get(CONF_ILLUMINANCE_SENSORS, []),
                        ): EntitySelector(
                            EntitySelectorConfig(
                                domain=Platform.SENSOR,
                                device_class=SensorDeviceClass.ILLUMINANCE,
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_HUMIDITY_SENSORS,
                            default=defaults.get(CONF_HUMIDITY_SENSORS, []),
                        ): EntitySelector(
                            EntitySelectorConfig(
                                domain=Platform.SENSOR,
                                device_class=SensorDeviceClass.HUMIDITY,
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_TEMPERATURE_SENSORS,
                            default=defaults.get(CONF_TEMPERATURE_SENSORS, []),
                        ): EntitySelector(
                            EntitySelectorConfig(
                                domain=Platform.SENSOR,
                                device_class=SensorDeviceClass.TEMPERATURE,
                                multiple=True,
                            ),
                        ),
                        vol.Optional(
                            CONF_WEIGHT_ENVIRONMENTAL,
                            default=defaults.get(
                                CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL
                            ),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                min=0,
                                max=1,
                                step=0.05,
                                mode="slider",
                            ),
                        ),
                    }
                ),
                {"collapsed": True},
            ),
            vol.Required("parameters"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_THRESHOLD,
                            default=defaults.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
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
                            default=defaults.get(
                                CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                            ),
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
                            default=defaults.get(
                                CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED
                            ),
                        ): BooleanSelector(),
                        vol.Optional(
                            CONF_DECAY_WINDOW,
                            default=defaults.get(
                                CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
                            ),
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
                            CONF_DECAY_MIN_DELAY,
                            default=defaults.get(
                                CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY
                            ),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                min=0,
                                max=3600,
                                step=10,
                                mode="box",
                                unit_of_measurement="seconds",
                            )
                        ),
                        vol.Optional(
                            CONF_HISTORICAL_ANALYSIS_ENABLED,
                            default=defaults.get(
                                CONF_HISTORICAL_ANALYSIS_ENABLED,
                                DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                            ),
                        ): BooleanSelector(),
                    }
                ),
                {"collapsed": True},
            ),
        }
    )

    return schema


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


class AreaOccupancyConfigFlow(ConfigFlow, BaseOccupancyFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection."""

    VERSION = CONF_VERSION

    def __init__(self) -> None:
        """Initialize config flow."""
        self._data: dict[str, Any] = {}

    def is_matching(self, other_flow: ConfigEntry) -> bool:
        """Check if the entry matches the current flow."""
        return other_flow.entry_id == getattr(self, "entry_id", None)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Flatten sectioned data
                flattened_input = {}
                for key, value in user_input.items():
                    if isinstance(value, dict):
                        # If the value is a dictionary (section), merge its contents
                        flattened_input.update(value)
                    else:
                        # If the value is not a dictionary, add it directly
                        flattened_input[key] = value

                self._validate_config(flattened_input)
                return self.async_create_entry(
                    title=flattened_input.get(CONF_NAME, ""),
                    data=flattened_input,
                )

            except HomeAssistantError as err:
                _LOGGER.error("Validation error: %s", err)
                errors["base"] = str(err)
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error("Unexpected error: %s", err)
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(create_schema(self.hass, self._data)),
            errors=errors,
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
        self._data = dict(config_entry.options)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Flatten sectioned data
                flattened_input = {}
                for key, value in user_input.items():
                    if isinstance(value, dict):
                        # If the value is a dictionary (section), merge its contents
                        flattened_input.update(value)
                    else:
                        # If the value is not a dictionary, add it directly
                        flattened_input[key] = value

                self._validate_config(flattened_input)
                return self.async_create_entry(title="", data=flattened_input)

            except HomeAssistantError as err:
                _LOGGER.error("Validation error: %s", err)
                errors["base"] = str(err)
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error("Unexpected error: %s", err)
                errors["base"] = "unknown"

        defaults = {
            **self.config_entry.data,
            **self.config_entry.options,
            **self._data,
        }

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(create_schema(self.hass, defaults, True)),
            errors=errors,
        )
