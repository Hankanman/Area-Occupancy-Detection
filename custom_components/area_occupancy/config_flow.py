"""Config flow for Area Occupancy Detection integration.

This module handles the configuration flow for the Area Occupancy Detection integration.
It provides both initial configuration and options update capabilities, with comprehensive
validation of all inputs to ensure a valid configuration.
"""

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
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult, section
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    SelectSelector,
    SelectSelectorConfig,
)

from .const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_DECAY_ENABLED,
    CONF_DECAY_MIN_DELAY,
    CONF_DECAY_WINDOW,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_HISTORY_PERIOD,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_MIN_DELAY,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOMAIN,
)
from .state_mapping import get_default_state, get_state_options

_LOGGER = logging.getLogger(__name__)

# UI Configuration Constants
WEIGHT_STEP = 0.05
WEIGHT_MIN = 0
WEIGHT_MAX = 1

THRESHOLD_STEP = 1
THRESHOLD_MIN = 0
THRESHOLD_MAX = 100

HISTORY_PERIOD_STEP = 1
HISTORY_PERIOD_MIN = 1
HISTORY_PERIOD_MAX = 30

DECAY_WINDOW_STEP = 60
DECAY_WINDOW_MIN = 60
DECAY_WINDOW_MAX = 3600

DECAY_MIN_DELAY_STEP = 10
DECAY_MIN_DELAY_MIN = 0
DECAY_MIN_DELAY_MAX = 3600


def _get_state_select_options(state_type: str) -> list[dict[str, str]]:
    """Get state options for SelectSelector."""
    states = get_state_options(state_type)
    return [
        {"value": option.value, "label": option.name} for option in states["options"]
    ]


def _get_include_entities(hass: HomeAssistant) -> dict[str, list[str]]:
    """Get lists of entities to include for specific selectors."""
    registry = er.async_get(hass)
    include_appliance_entities = []
    include_window_entities = []
    include_door_entities = []

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

    # Check binary_sensor, switch, fan for potential appliances
    domains_to_check = [Platform.BINARY_SENSOR, Platform.SWITCH, Platform.FAN]
    entity_ids = []
    for domain in domains_to_check:
        entity_ids.extend(hass.states.async_entity_ids(domain))

    for eid in entity_ids:
        state = hass.states.get(eid)
        if state:
            device_class = state.attributes.get("device_class")
            if device_class not in appliance_excluded_classes:
                include_appliance_entities.append(eid)

    # Check registry for specific door/window classes
    for entry in registry.entities.values():
        if entry.domain == Platform.BINARY_SENSOR:
            is_window_candidate = (
                entry.device_class == BinarySensorDeviceClass.WINDOW
                or (
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
                )
            )
            is_door_candidate = entry.device_class == BinarySensorDeviceClass.DOOR or (
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
            )

            if is_window_candidate:
                include_window_entities.append(entry.entity_id)
            elif is_door_candidate:
                include_door_entities.append(entry.entity_id)

    return {
        "appliance": include_appliance_entities,
        "window": include_window_entities,
        "door": include_door_entities,
    }


def _create_motion_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the motion section."""
    return vol.Schema(
        {
            vol.Required(
                CONF_PRIMARY_OCCUPANCY_SENSOR,
                default=defaults.get(CONF_PRIMARY_OCCUPANCY_SENSOR, ""),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.BINARY_SENSOR,
                    device_class=[
                        BinarySensorDeviceClass.MOTION,
                        BinarySensorDeviceClass.OCCUPANCY,
                        BinarySensorDeviceClass.PRESENCE,
                    ],
                    multiple=False,
                ),
            ),
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
                default=defaults.get(CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=WEIGHT_MIN,
                    max=WEIGHT_MAX,
                    step=WEIGHT_STEP,
                    mode="slider",
                ),
            ),
        }
    )


def _create_doors_section_schema(
    defaults: dict[str, Any],
    include_entities: list[str],
    state_options: list[dict[str, str]],
) -> vol.Schema:
    """Create schema for the doors section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_DOOR_SENSORS,
                default=defaults.get(CONF_DOOR_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=include_entities,
                    multiple=True,
                ),
            ),
            vol.Optional(
                CONF_DOOR_ACTIVE_STATE,
                default=defaults.get(CONF_DOOR_ACTIVE_STATE, get_default_state("door")),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=state_options,
                    mode="dropdown",
                )
            ),
            vol.Optional(
                CONF_WEIGHT_DOOR,
                default=defaults.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=WEIGHT_MIN,
                    max=WEIGHT_MAX,
                    step=WEIGHT_STEP,
                    mode="slider",
                ),
            ),
        }
    )


def _create_windows_section_schema(
    defaults: dict[str, Any],
    include_entities: list[str],
    state_options: list[dict[str, str]],
) -> vol.Schema:
    """Create schema for the windows section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_WINDOW_SENSORS,
                default=defaults.get(CONF_WINDOW_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=include_entities,
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
                    options=state_options,
                    mode="dropdown",
                )
            ),
            vol.Optional(
                CONF_WEIGHT_WINDOW,
                default=defaults.get(CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=WEIGHT_MIN,
                    max=WEIGHT_MAX,
                    step=WEIGHT_STEP,
                    mode="slider",
                ),
            ),
        }
    )


def _create_lights_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the lights section."""
    return vol.Schema(
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
                default=defaults.get(CONF_WEIGHT_LIGHT, DEFAULT_WEIGHT_LIGHT),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=WEIGHT_MIN,
                    max=WEIGHT_MAX,
                    step=WEIGHT_STEP,
                    mode="slider",
                ),
            ),
        }
    )


def _create_media_section_schema(
    defaults: dict[str, Any], state_options: list[dict[str, str]]
) -> vol.Schema:
    """Create schema for the media section."""
    return vol.Schema(
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
                    options=state_options,
                    multiple=True,
                    mode="dropdown",
                )
            ),
            vol.Optional(
                CONF_WEIGHT_MEDIA,
                default=defaults.get(CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=WEIGHT_MIN,
                    max=WEIGHT_MAX,
                    step=WEIGHT_STEP,
                    mode="slider",
                ),
            ),
        }
    )


def _create_appliances_section_schema(
    defaults: dict[str, Any],
    include_entities: list[str],
    state_options: list[dict[str, str]],
) -> vol.Schema:
    """Create schema for the appliances section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_APPLIANCES, default=defaults.get(CONF_APPLIANCES, [])
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=include_entities,
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
                    options=state_options,
                    multiple=True,
                    mode="dropdown",
                )
            ),
            vol.Optional(
                CONF_WEIGHT_APPLIANCE,
                default=defaults.get(CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=WEIGHT_MIN,
                    max=WEIGHT_MAX,
                    step=WEIGHT_STEP,
                    mode="slider",
                ),
            ),
        }
    )


def _create_environmental_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the environmental section."""
    return vol.Schema(
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
                    min=WEIGHT_MIN,
                    max=WEIGHT_MAX,
                    step=WEIGHT_STEP,
                    mode="slider",
                ),
            ),
        }
    )


def _create_parameters_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the parameters section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_THRESHOLD,
                default=defaults.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=THRESHOLD_MIN,
                    max=THRESHOLD_MAX,
                    step=THRESHOLD_STEP,
                    mode="slider",
                ),
            ),
            vol.Optional(
                CONF_HISTORY_PERIOD,
                default=defaults.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=HISTORY_PERIOD_MIN,
                    max=HISTORY_PERIOD_MAX,
                    step=HISTORY_PERIOD_STEP,
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
                    min=DECAY_WINDOW_MIN,
                    max=DECAY_WINDOW_MAX,
                    step=DECAY_WINDOW_STEP,
                    mode="slider",
                    unit_of_measurement="seconds",
                ),
            ),
            vol.Optional(
                CONF_DECAY_MIN_DELAY,
                default=defaults.get(CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=DECAY_MIN_DELAY_MIN,
                    max=DECAY_MIN_DELAY_MAX,
                    step=DECAY_MIN_DELAY_STEP,
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
    )


def create_schema(
    hass: HomeAssistant,
    defaults: dict[str, Any] | None = None,
    is_options: bool = False,
) -> dict:
    """Create a schema with optional default values, using helper functions."""
    if defaults is None:
        defaults = {}

    # Pre-calculate expensive lookups
    include_entities = _get_include_entities(hass)
    door_state_options = _get_state_select_options("door")
    media_state_options = _get_state_select_options("media")
    window_state_options = _get_state_select_options("window")
    appliance_state_options = _get_state_select_options("appliance")

    schema = {}
    if not is_options:
        schema[vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, ""))] = str

    schema.update(
        {
            vol.Required("motion"): section(
                _create_motion_section_schema(defaults),
                {"collapsed": True},
            ),
            vol.Required("doors"): section(
                _create_doors_section_schema(
                    defaults,
                    include_entities["door"],
                    door_state_options,
                ),
                {"collapsed": True},
            ),
            vol.Required("windows"): section(
                _create_windows_section_schema(
                    defaults,
                    include_entities["window"],
                    window_state_options,
                ),
                {"collapsed": True},
            ),
            vol.Required("lights"): section(
                _create_lights_section_schema(defaults),
                {"collapsed": True},
            ),
            vol.Required("media"): section(
                _create_media_section_schema(defaults, media_state_options),
                {"collapsed": True},
            ),
            vol.Required("appliances"): section(
                _create_appliances_section_schema(
                    defaults,
                    include_entities["appliance"],
                    appliance_state_options,
                ),
                {"collapsed": True},
            ),
            vol.Required("environmental"): section(
                _create_environmental_section_schema(defaults),
                {"collapsed": True},
            ),
            vol.Required("parameters"): section(
                _create_parameters_section_schema(defaults),
                {"collapsed": True},
            ),
        }
    )

    return schema


class BaseOccupancyFlow:
    """Base class for config and options flow.

    This class provides shared validation logic used by both the config flow
    and options flow. It ensures consistent validation across both flows.
    """

    def _validate_config(self, data: dict[str, Any]) -> None:
        """Validate the configuration.

        Performs comprehensive validation of all configuration fields including:
        - Required sensors and their relationships
        - State configurations for different device types
        - Weight values and their ranges

        Args:
            data: Dictionary containing the configuration to validate

        Raises:
            ValueError: If any validation check fails

        """
        motion_sensors = data.get(CONF_MOTION_SENSORS, [])
        if not motion_sensors:
            raise ValueError("At least one motion sensor is required")

        primary_sensor = data.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor:
            raise ValueError("A primary occupancy sensor must be selected")
        if primary_sensor not in motion_sensors:
            raise ValueError(
                "Primary occupancy sensor must be selected from the motion sensors"
            )

        # Validate media devices
        media_devices = data.get(CONF_MEDIA_DEVICES, [])
        media_states = data.get(CONF_MEDIA_ACTIVE_STATES, DEFAULT_MEDIA_ACTIVE_STATES)
        if media_devices and not media_states:
            raise ValueError(
                "Media active states are required when media devices are configured"
            )

        # Validate appliances
        appliances = data.get(CONF_APPLIANCES, [])
        appliance_states = data.get(
            CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_APPLIANCE_ACTIVE_STATES
        )
        if appliances and not appliance_states:
            raise ValueError(
                "Appliance active states are required when appliances are configured"
            )

        # Validate doors
        door_sensors = data.get(CONF_DOOR_SENSORS, [])
        door_state = data.get(CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE)
        if door_sensors and not door_state:
            raise ValueError(
                "Door active state is required when door sensors are configured"
            )

        # Validate windows
        window_sensors = data.get(CONF_WINDOW_SENSORS, [])
        window_state = data.get(CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE)
        if window_sensors and not window_state:
            raise ValueError(
                "Window active state is required when window sensors are configured"
            )

        # Validate weights
        weights = [
            (CONF_WEIGHT_MOTION, data.get(CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION)),
            (CONF_WEIGHT_MEDIA, data.get(CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA)),
            (
                CONF_WEIGHT_APPLIANCE,
                data.get(CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE),
            ),
            (CONF_WEIGHT_DOOR, data.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR)),
            (CONF_WEIGHT_WINDOW, data.get(CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW)),
            (CONF_WEIGHT_LIGHT, data.get(CONF_WEIGHT_LIGHT, DEFAULT_WEIGHT_LIGHT)),
            (
                CONF_WEIGHT_ENVIRONMENTAL,
                data.get(CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL),
            ),
        ]
        for name, weight in weights:
            if not WEIGHT_MIN <= weight <= WEIGHT_MAX:
                raise ValueError(
                    f"{name} must be between {WEIGHT_MIN} and {WEIGHT_MAX}"
                )


class AreaOccupancyConfigFlow(ConfigFlow, BaseOccupancyFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection.

    This class handles the initial configuration flow when the integration is first set up.
    It provides a multi-step configuration process with comprehensive validation.
    """

    def __init__(self) -> None:
        """Initialize config flow.

        Sets up the initial empty data dictionary that will store configuration
        as it is built through the flow.
        """
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
                # --- Auto-add primary sensor to motion sensors --- >
                motion_section = user_input.get("motion", {})
                primary_sensor = motion_section.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
                motion_sensors = motion_section.get(CONF_MOTION_SENSORS, [])

                if primary_sensor and primary_sensor not in motion_sensors:
                    _LOGGER.debug(
                        "Auto-adding primary sensor %s to motion sensors list",
                        primary_sensor,
                    )
                    motion_sensors.append(primary_sensor)
                    # Update the motion section in the original user_input
                    user_input["motion"][CONF_MOTION_SENSORS] = motion_sensors
                # < --- End Auto-add ---

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
            except (ValueError, KeyError, TypeError) as err:
                _LOGGER.error("Unexpected error: %s", err)
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(create_schema(self.hass, self._data)),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> AreaOccupancyOptionsFlow:
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
                # --- Auto-add primary sensor to motion sensors --- >
                motion_section = user_input.get("motion", {})
                primary_sensor = motion_section.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
                motion_sensors = motion_section.get(CONF_MOTION_SENSORS, [])

                if primary_sensor and primary_sensor not in motion_sensors:
                    _LOGGER.debug(
                        "Auto-adding primary sensor %s to motion sensors list (Options Flow)",
                        primary_sensor,
                    )
                    motion_sensors.append(primary_sensor)
                    # Update the motion section in the original user_input
                    user_input["motion"][CONF_MOTION_SENSORS] = motion_sensors
                # < --- End Auto-add ---

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
            except (ValueError, KeyError, TypeError) as err:
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
