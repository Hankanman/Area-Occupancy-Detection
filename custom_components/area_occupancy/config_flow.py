"""Config flow for Area Occupancy Detection integration.

This module handles the configuration flow for the Area Occupancy Detection integration.
It provides both initial configuration and options update capabilities, with comprehensive
validation of all inputs to ensure a valid configuration.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import voluptuous as vol

from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import section
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREAS,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MIN_PRIOR_OVERRIDE,
    CONF_MOTION_SENSORS,
    CONF_MOTION_TIMEOUT,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WASP_MAX_DURATION,
    CONF_WASP_MOTION_TIMEOUT,
    CONF_WASP_VERIFICATION_DELAY,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_MIN_PRIOR_OVERRIDE,
    DEFAULT_MOTION_TIMEOUT,
    DEFAULT_PURPOSE,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_VERIFICATION_DELAY,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOMAIN,
    validate_and_sanitize_area_name,
)
from .data.purpose import PURPOSE_DEFINITIONS, AreaPurpose, get_purpose_options
from .migrations import _migrate_area_name_in_entity_registry
from .state_mapping import get_default_state, get_state_options

_LOGGER = logging.getLogger(__name__)

# UI Configuration Constants
WEIGHT_STEP = 0.05
WEIGHT_MIN = 0
WEIGHT_MAX = 1

THRESHOLD_STEP = 1
THRESHOLD_MIN = 1
THRESHOLD_MAX = 100


def _get_state_select_options(state_type: str) -> list[dict[str, str]]:
    """Get state options for SelectSelector."""
    states = get_state_options(state_type)
    return [
        {"value": option.value, "label": option.name} for option in states["options"]
    ]


def _get_default_decay_half_life(purpose: str | None = None) -> float:
    """Get the default decay half-life based on the selected purpose."""
    if purpose is not None:
        try:
            purpose_enum = AreaPurpose(purpose)
            return PURPOSE_DEFINITIONS[purpose_enum].half_life
        except (ValueError, KeyError):
            pass
    # Fallback to default purpose half-life
    return PURPOSE_DEFINITIONS[AreaPurpose(DEFAULT_PURPOSE)].half_life


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
    ]

    # Check binary_sensor, switch, fan, light for potential appliances
    domains_to_check = [
        Platform.BINARY_SENSOR,
        Platform.SWITCH,
        Platform.FAN,
        Platform.LIGHT,
    ]
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
                )
            ),
            vol.Required(
                CONF_MOTION_SENSORS, default=defaults.get(CONF_MOTION_SENSORS, [])
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.BINARY_SENSOR,
                    device_class=[
                        BinarySensorDeviceClass.MOTION,
                        BinarySensorDeviceClass.OCCUPANCY,
                        BinarySensorDeviceClass.PRESENCE,
                    ],
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_WEIGHT_MOTION,
                default=defaults.get(CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=WEIGHT_MIN,
                    max=WEIGHT_MAX,
                    step=WEIGHT_STEP,
                    mode=NumberSelectorMode.SLIDER,
                )
            ),
            vol.Optional(
                CONF_MOTION_TIMEOUT,
                default=defaults.get(CONF_MOTION_TIMEOUT, DEFAULT_MOTION_TIMEOUT),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=3600,
                    step=5,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="seconds",
                )
            ),
        }
    )


def _create_doors_section_schema(
    defaults: dict[str, Any],
    include_entities: list[str],
    state_options: list[SelectOptionDict],
) -> vol.Schema:
    """Create schema for the doors section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_DOOR_SENSORS, default=defaults.get(CONF_DOOR_SENSORS, [])
            ): EntitySelector(
                EntitySelectorConfig(include_entities=include_entities, multiple=True)
            ),
            vol.Optional(
                CONF_DOOR_ACTIVE_STATE,
                default=defaults.get(CONF_DOOR_ACTIVE_STATE, get_default_state("door")),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=state_options, mode=SelectSelectorMode.DROPDOWN
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
                    mode=NumberSelectorMode.SLIDER,
                )
            ),
        }
    )


def _create_windows_section_schema(
    defaults: dict[str, Any],
    include_entities: list[str],
    state_options: list[SelectOptionDict],
) -> vol.Schema:
    """Create schema for the windows section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_WINDOW_SENSORS, default=defaults.get(CONF_WINDOW_SENSORS, [])
            ): EntitySelector(
                EntitySelectorConfig(include_entities=include_entities, multiple=True)
            ),
            vol.Optional(
                CONF_WINDOW_ACTIVE_STATE,
                default=defaults.get(
                    CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE
                ),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=state_options, mode=SelectSelectorMode.DROPDOWN
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
                    mode=NumberSelectorMode.SLIDER,
                )
            ),
        }
    )


def _create_media_section_schema(
    defaults: dict[str, Any], state_options: list[SelectOptionDict]
) -> vol.Schema:
    """Create schema for the media section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_MEDIA_DEVICES, default=defaults.get(CONF_MEDIA_DEVICES, [])
            ): EntitySelector(
                EntitySelectorConfig(domain=Platform.MEDIA_PLAYER, multiple=True)
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
                    mode=SelectSelectorMode.DROPDOWN,
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
                    mode=NumberSelectorMode.SLIDER,
                )
            ),
        }
    )


def _create_appliances_section_schema(
    defaults: dict[str, Any],
    include_entities: list[str],
    state_options: list[SelectOptionDict],
) -> vol.Schema:
    """Create schema for the appliances section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_APPLIANCES, default=defaults.get(CONF_APPLIANCES, [])
            ): EntitySelector(
                EntitySelectorConfig(include_entities=include_entities, multiple=True)
            ),
            vol.Optional(
                CONF_APPLIANCE_ACTIVE_STATES,
                default=defaults.get(
                    CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_APPLIANCE_ACTIVE_STATES
                ),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=state_options,
                    multiple=True,
                    mode=SelectSelectorMode.DROPDOWN,
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
                    mode=NumberSelectorMode.SLIDER,
                )
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
                )
            ),
            vol.Optional(
                CONF_HUMIDITY_SENSORS, default=defaults.get(CONF_HUMIDITY_SENSORS, [])
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.SENSOR,
                    device_class=SensorDeviceClass.HUMIDITY,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE_SENSORS,
                default=defaults.get(CONF_TEMPERATURE_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.SENSOR,
                    device_class=SensorDeviceClass.TEMPERATURE,
                    multiple=True,
                )
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
                    mode=NumberSelectorMode.SLIDER,
                )
            ),
        }
    )


def _create_purpose_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the purpose section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_PURPOSE, default=defaults.get(CONF_PURPOSE, DEFAULT_PURPOSE)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=cast("list[SelectOptionDict]", get_purpose_options()),
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
        }
    )


def _create_parameters_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the parameters section."""
    # Get the purpose-based default for decay half-life
    purpose = defaults.get(CONF_PURPOSE, DEFAULT_PURPOSE)
    purpose_based_default = _get_default_decay_half_life(purpose)

    # Use the purpose-based default if no explicit value is already set
    decay_half_life_default = defaults.get(CONF_DECAY_HALF_LIFE, purpose_based_default)

    return vol.Schema(
        {
            vol.Optional(
                CONF_THRESHOLD, default=defaults.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
            ): NumberSelector(
                NumberSelectorConfig(
                    min=THRESHOLD_MIN,
                    max=THRESHOLD_MAX,
                    step=THRESHOLD_STEP,
                    mode=NumberSelectorMode.SLIDER,
                )
            ),
            vol.Optional(
                CONF_DECAY_ENABLED,
                default=defaults.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
            ): BooleanSelector(),
            vol.Optional(
                CONF_DECAY_HALF_LIFE, default=decay_half_life_default
            ): NumberSelector(
                NumberSelectorConfig(
                    min=10,
                    max=3600,
                    step=1,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="seconds",
                )
            ),
            vol.Optional(
                CONF_MIN_PRIOR_OVERRIDE,
                default=defaults.get(
                    CONF_MIN_PRIOR_OVERRIDE, DEFAULT_MIN_PRIOR_OVERRIDE
                ),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    mode=NumberSelectorMode.SLIDER,
                    unit_of_measurement="probability",
                )
            ),
        }
    )


def _create_wasp_in_box_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the wasp in box section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_WASP_ENABLED, default=defaults.get(CONF_WASP_ENABLED, False)
            ): BooleanSelector(),
            vol.Optional(
                CONF_WASP_MOTION_TIMEOUT,
                default=defaults.get(
                    CONF_WASP_MOTION_TIMEOUT, DEFAULT_WASP_MOTION_TIMEOUT
                ),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=3600,
                    step=60,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="seconds",
                )
            ),
            vol.Optional(
                CONF_WASP_WEIGHT,
                default=defaults.get(CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_WASP_MAX_DURATION,
                default=defaults.get(CONF_WASP_MAX_DURATION, DEFAULT_WASP_MAX_DURATION),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=86400,  # 24 hours in seconds
                    step=300,  # 5-minute increments
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="seconds",
                )
            ),
            vol.Optional(
                CONF_WASP_VERIFICATION_DELAY,
                default=defaults.get(
                    CONF_WASP_VERIFICATION_DELAY, DEFAULT_WASP_VERIFICATION_DELAY
                ),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=120,  # 2 minutes max
                    step=5,  # 5-second increments
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="seconds",
                )
            ),
        }
    )


def create_schema(
    hass: HomeAssistant,
    defaults: dict[str, Any] | None = None,
    is_options: bool = False,
) -> dict:
    """Create a schema with optional default values, using helper functions."""
    # Ensure defaults is a dictionary
    defaults = defaults if defaults is not None else {}

    # Pre-calculate expensive lookups
    include_entities = _get_include_entities(hass)
    door_state_options = _get_state_select_options("door")
    media_state_options = _get_state_select_options("media")
    window_state_options = _get_state_select_options("window")
    appliance_state_options = _get_state_select_options("appliance")

    # Initialize the dictionary for the schema
    schema_dict: dict[vol.Marker, Any] = {}

    if not is_options:
        # Add the name field only for the initial config flow
        schema_dict[vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, ""))] = str
        # Add purpose section right after name in initial config flow
        schema_dict[vol.Required("purpose")] = section(
            _create_purpose_section_schema(defaults), {"collapsed": False}
        )
    else:
        # Add purpose section at the top for options flow
        schema_dict[vol.Required("purpose")] = section(
            _create_purpose_section_schema(defaults), {"collapsed": False}
        )

    # Add sections by assigning keys directly to the dictionary
    schema_dict[vol.Required("motion")] = section(
        _create_motion_section_schema(defaults), {"collapsed": True}
    )
    schema_dict[vol.Required("doors")] = section(
        _create_doors_section_schema(
            defaults,
            include_entities["door"],
            cast("list[SelectOptionDict]", door_state_options),
        ),
        {"collapsed": True},
    )
    schema_dict[vol.Required("windows")] = section(
        _create_windows_section_schema(
            defaults,
            include_entities["window"],
            cast("list[SelectOptionDict]", window_state_options),
        ),
        {"collapsed": True},
    )
    schema_dict[vol.Required("media")] = section(
        _create_media_section_schema(
            defaults, cast("list[SelectOptionDict]", media_state_options)
        ),
        {"collapsed": True},
    )
    schema_dict[vol.Required("appliances")] = section(
        _create_appliances_section_schema(
            defaults,
            include_entities["appliance"],
            cast("list[SelectOptionDict]", appliance_state_options),
        ),
        {"collapsed": True},
    )
    schema_dict[vol.Required("environmental")] = section(
        _create_environmental_section_schema(defaults), {"collapsed": True}
    )
    schema_dict[vol.Required("wasp_in_box")] = section(
        _create_wasp_in_box_section_schema(defaults), {"collapsed": True}
    )
    schema_dict[vol.Required("parameters")] = section(
        _create_parameters_section_schema(defaults), {"collapsed": True}
    )

    # Pass the correctly structured dictionary to vol.Schema
    return schema_dict


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
        # Validate name
        name = data.get(CONF_NAME, "")
        if not name:
            raise vol.Invalid("Name is required")

        # Validate and sanitize area name
        try:
            sanitized_name = validate_and_sanitize_area_name(name)
            # Update data with sanitized name if it changed
            if sanitized_name != name:
                data[CONF_NAME] = sanitized_name
        except ValueError as err:
            raise vol.Invalid(str(err)) from err

        # Validate purpose
        purpose = data.get(CONF_PURPOSE, DEFAULT_PURPOSE)
        if not purpose:
            raise vol.Invalid("Purpose is required")

        # Validate motion sensors
        motion_sensors = data.get(CONF_MOTION_SENSORS, [])
        if not motion_sensors:
            raise vol.Invalid("At least one motion sensor is required")

        primary_sensor = data.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor:
            raise vol.Invalid("A primary occupancy sensor must be selected")
        if primary_sensor not in motion_sensors:
            raise vol.Invalid(
                "Primary occupancy sensor must be one of the selected motion sensors"
            )

        # Validate threshold
        threshold = data.get(CONF_THRESHOLD)
        if threshold is not None:
            if (
                not isinstance(threshold, (int, float))
                or threshold < 1
                or threshold > 100
            ):
                raise vol.Invalid("Threshold must be between 1 and 100")

        # Validate media devices
        media_devices = data.get(CONF_MEDIA_DEVICES, [])
        media_states = data.get(CONF_MEDIA_ACTIVE_STATES, DEFAULT_MEDIA_ACTIVE_STATES)
        if media_devices and not media_states:
            raise vol.Invalid(
                "Media active states are required when media devices are configured"
            )

        # Validate appliances
        appliances = data.get(CONF_APPLIANCES, [])
        appliance_states = data.get(
            CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_APPLIANCE_ACTIVE_STATES
        )
        if appliances and not appliance_states:
            raise vol.Invalid(
                "Appliance active states are required when appliances are configured"
            )

        # Validate doors
        door_sensors = data.get(CONF_DOOR_SENSORS, [])
        door_state = data.get(CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE)
        if door_sensors and not door_state:
            raise vol.Invalid(
                "Door active state is required when door sensors are configured"
            )

        # Validate windows
        window_sensors = data.get(CONF_WINDOW_SENSORS, [])
        window_state = data.get(CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE)
        if window_sensors and not window_state:
            raise vol.Invalid(
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
            (
                CONF_WEIGHT_ENVIRONMENTAL,
                data.get(CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL),
            ),
        ]
        for name, weight in weights:
            if not WEIGHT_MIN <= weight <= WEIGHT_MAX:
                raise vol.Invalid(
                    f"{name} must be between {WEIGHT_MIN} and {WEIGHT_MAX}"
                )

        # Validate decay settings
        decay_enabled = data.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        if decay_enabled:
            decay_window = data.get(CONF_DECAY_HALF_LIFE, DEFAULT_DECAY_HALF_LIFE)
            if (
                not isinstance(decay_window, (int, float))
                or decay_window < 10
                or decay_window > 3600
            ):
                raise vol.Invalid("Decay half life must be between 10 and 3600 seconds")


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

    def _validate_duplicate_name_internal(
        self, flattened_input: dict[str, Any], areas: list[dict[str, Any]]
    ) -> None:
        """Validate that area name is not a duplicate.

        Args:
            flattened_input: The flattened input configuration
            areas: List of existing area configurations

        Raises:
            vol.Invalid: If area name is a duplicate
        """
        area_name = flattened_input.get(CONF_NAME, "")
        if area_name:
            for area in areas:
                if area.get(CONF_NAME) == area_name and (
                    not self._area_being_edited
                    or area.get(CONF_NAME) != self._area_being_edited
                ):
                    msg = f"An area named '{area_name}' already exists"
                    raise vol.Invalid(msg)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
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
                    # Check if the key corresponds to a section dictionary
                    if isinstance(value, dict):
                        # Check if it's the wasp_in_box section specifically
                        if key == "wasp_in_box":
                            # Flatten wasp settings using const keys
                            flattened_input[CONF_WASP_ENABLED] = value.get(
                                CONF_WASP_ENABLED, False
                            )
                            flattened_input[CONF_WASP_MOTION_TIMEOUT] = value.get(
                                CONF_WASP_MOTION_TIMEOUT, DEFAULT_WASP_MOTION_TIMEOUT
                            )
                            flattened_input[CONF_WASP_WEIGHT] = value.get(
                                CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT
                            )
                            flattened_input[CONF_WASP_MAX_DURATION] = value.get(
                                CONF_WASP_MAX_DURATION, DEFAULT_WASP_MAX_DURATION
                            )
                            flattened_input[CONF_WASP_VERIFICATION_DELAY] = value.get(
                                CONF_WASP_VERIFICATION_DELAY,
                                DEFAULT_WASP_VERIFICATION_DELAY,
                            )
                        elif key == "purpose":
                            # Flatten purpose settings
                            flattened_input[CONF_PURPOSE] = value.get(
                                CONF_PURPOSE, DEFAULT_PURPOSE
                            )
                        else:
                            # Flatten other sections as before
                            flattened_input.update(value)
                    else:
                        # Handle top-level keys like CONF_NAME
                        flattened_input[key] = value

                # Auto-set decay half-life based on purpose selection
                selected_purpose = flattened_input.get(CONF_PURPOSE)
                if selected_purpose:
                    # Check if user has explicitly set a decay half-life
                    user_set_decay = flattened_input.get(CONF_DECAY_HALF_LIFE)
                    purpose_default = _get_default_decay_half_life(selected_purpose)

                    # Get all purpose-based half-life values to check if current value was auto-set
                    purpose_half_lives = {
                        purpose_def.half_life
                        for purpose_def in PURPOSE_DEFINITIONS.values()
                    }
                    purpose_half_lives.add(DEFAULT_DECAY_HALF_LIFE)

                    # If no explicit decay half-life, or it matches any purpose default (indicating it was auto-set)
                    if (
                        user_set_decay is None
                        or user_set_decay == DEFAULT_DECAY_HALF_LIFE
                        or user_set_decay in purpose_half_lives
                    ):
                        flattened_input[CONF_DECAY_HALF_LIFE] = purpose_default
                        _LOGGER.debug(
                            "Auto-setting decay half-life to %s seconds for purpose %s",
                            purpose_default,
                            selected_purpose,
                        )

                self._validate_config(flattened_input)

                # Store in new multi-area format
                # Use a fixed title for the integration entry
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()

                # Store the first area in CONF_AREAS list format
                config_data = {CONF_AREAS: [flattened_input]}
                return self.async_create_entry(
                    title="Area Occupancy Detection", data=config_data
                )

            except HomeAssistantError as err:
                _LOGGER.error("Validation error: %s", err)
                errors["base"] = str(err)
            except vol.Invalid as err:
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
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> AreaOccupancyOptionsFlow:
        """Get the options flow."""
        return AreaOccupancyOptionsFlow()

    @staticmethod
    @callback
    def async_get_device_options_flow(
        config_entry: ConfigEntry, device_id: str
    ) -> AreaOccupancyOptionsFlow:
        """Get device-specific options flow for individual device reconfiguration.

        Args:
            config_entry: The config entry
            device_id: The device ID from the device registry

        Returns:
            AreaOccupancyOptionsFlow configured for the specific device
        """
        flow = AreaOccupancyOptionsFlow()
        # Store device_id to resolve in async_step_init when we have access to hass
        flow._device_id = device_id  # noqa: SLF001
        return flow


class AreaOccupancyOptionsFlow(OptionsFlow, BaseOccupancyFlow):
    """Handle options flow with multi-area management."""

    def __init__(self) -> None:
        """Initialize options flow."""
        super().__init__()
        self._area_being_edited: str | None = None
        self._area_to_remove: str | None = None
        self._device_id: str | None = (
            None  # Store device_id for device-specific configuration
        )

    def _get_areas_from_config(self) -> list[dict[str, Any]]:
        """Get areas list from config entry, handling both legacy and new formats."""
        merged = dict(self.config_entry.data)
        merged.update(self.config_entry.options)

        # Check if we have the new multi-area format
        if CONF_AREAS in merged and isinstance(merged[CONF_AREAS], list):
            return merged[CONF_AREAS]

        # Legacy format: create area from existing config
        legacy_area = {**merged}
        if CONF_NAME not in legacy_area:
            legacy_area[CONF_NAME] = "Area"
        return [legacy_area]

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show area management menu."""
        # If called from device registry, resolve device_id and set area being edited
        if self._device_id:
            device_registry = dr.async_get(self.hass)
            if device := device_registry.async_get(self._device_id):
                # Find the device identifier that matches our domain
                for identifier in device.identifiers:
                    if identifier[0] == DOMAIN:
                        self._area_being_edited = identifier[1]
                        break
            # Clear device_id after resolving
            self._device_id = None

        # If called from device registry, skip to area config
        if self._area_being_edited:
            return await self.async_step_area_config()

        # Get current areas
        areas = self._get_areas_from_config()

        if user_input is not None:
            action = user_input.get("action")
            if action == "add_area":
                # User wants to add a new area
                self._area_being_edited = None
                return await self.async_step_area_config()
            if action and action.startswith("edit_"):
                # User wants to edit an existing area
                area_name = action.replace("edit_", "", 1)
                self._area_being_edited = area_name
                return await self.async_step_area_config()
            if action and action.startswith("remove_"):
                # User wants to remove an area
                area_name = action.replace("remove_", "", 1)
                self._area_to_remove = area_name
                return await self.async_step_remove_area()

        # If only one area, show simplified view (backward compatibility)
        if len(areas) == 1:
            # Single area - redirect to edit that area directly
            self._area_being_edited = areas[0].get(CONF_NAME, "Area")
            return await self.async_step_area_config()

        # Multiple areas - show menu
        options_list = [
            cast(SelectOptionDict, {"value": "add_area", "label": "Add New Area"})
        ]
        options_list.extend(
            [
                cast(
                    SelectOptionDict,
                    {
                        "value": f"edit_{area.get(CONF_NAME, 'Unknown')}",
                        "label": f"Edit {area.get(CONF_NAME, 'Unknown')}",
                    },
                )
                for area in areas
            ]
        )
        options_list.extend(
            [
                cast(
                    SelectOptionDict,
                    {
                        "value": f"remove_{area.get(CONF_NAME, 'Unknown')}",
                        "label": f"Remove {area.get(CONF_NAME, 'Unknown')}",
                    },
                )
                for area in areas
            ]
        )

        schema = vol.Schema(
            {
                vol.Required("action"): SelectSelector(
                    SelectSelectorConfig(
                        options=options_list,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                )
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)

    async def async_step_area_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure a single area (add or edit)."""
        errors: dict[str, str] = {}
        areas = self._get_areas_from_config()

        # Get defaults for editing
        defaults: dict[str, Any] = {}
        if self._area_being_edited:
            # Find the area being edited
            for area in areas:
                if area.get(CONF_NAME) == self._area_being_edited:
                    defaults = area.copy()
                    break

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
                    user_input["motion"][CONF_MOTION_SENSORS] = motion_sensors
                # < --- End Auto-add ---

                # Flatten sectioned data
                flattened_input = {}
                for key, value in user_input.items():
                    if isinstance(value, dict):
                        if key == "wasp_in_box":
                            flattened_input[CONF_WASP_ENABLED] = value.get(
                                CONF_WASP_ENABLED, False
                            )
                            flattened_input[CONF_WASP_MOTION_TIMEOUT] = value.get(
                                CONF_WASP_MOTION_TIMEOUT, DEFAULT_WASP_MOTION_TIMEOUT
                            )
                            flattened_input[CONF_WASP_WEIGHT] = value.get(
                                CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT
                            )
                            flattened_input[CONF_WASP_MAX_DURATION] = value.get(
                                CONF_WASP_MAX_DURATION, DEFAULT_WASP_MAX_DURATION
                            )
                            flattened_input[CONF_WASP_VERIFICATION_DELAY] = value.get(
                                CONF_WASP_VERIFICATION_DELAY,
                                DEFAULT_WASP_VERIFICATION_DELAY,
                            )
                        elif key == "purpose":
                            flattened_input[CONF_PURPOSE] = value.get(
                                CONF_PURPOSE, DEFAULT_PURPOSE
                            )
                        else:
                            flattened_input.update(value)
                    else:
                        flattened_input[key] = value

                # Auto-set decay half-life based on purpose
                selected_purpose = flattened_input.get(CONF_PURPOSE)
                if selected_purpose:
                    user_set_decay = flattened_input.get(CONF_DECAY_HALF_LIFE)
                    purpose_default = _get_default_decay_half_life(selected_purpose)
                    purpose_half_lives = {
                        purpose_def.half_life
                        for purpose_def in PURPOSE_DEFINITIONS.values()
                    }
                    purpose_half_lives.add(DEFAULT_DECAY_HALF_LIFE)
                    if (
                        user_set_decay is None
                        or user_set_decay == DEFAULT_DECAY_HALF_LIFE
                        or user_set_decay in purpose_half_lives
                    ):
                        flattened_input[CONF_DECAY_HALF_LIFE] = purpose_default

                # Validate the area configuration
                self._validate_config(flattened_input)

                # Check for duplicate names (if adding or renaming)
                self._validate_duplicate_name_internal(flattened_input, areas)

                # Handle area name changes - migrate entity registry if needed
                if self._area_being_edited:
                    old_area = next(
                        (
                            a
                            for a in areas
                            if a.get(CONF_NAME) == self._area_being_edited
                        ),
                        None,
                    )
                    new_area_name = flattened_input.get(CONF_NAME, "")
                    if (
                        old_area
                        and new_area_name
                        and new_area_name != self._area_being_edited
                    ):
                        # Area name changed - migrate entity registry
                        _LOGGER.info(
                            "Area name changed from '%s' to '%s', migrating entity registry",
                            self._area_being_edited,
                            new_area_name,
                        )
                        try:
                            await _migrate_area_name_in_entity_registry(
                                self.hass,
                                self.config_entry,
                                self._area_being_edited,
                                new_area_name,
                            )
                        except (
                            HomeAssistantError,
                            ValueError,
                            KeyError,
                        ) as migrate_err:
                            _LOGGER.error(
                                "Failed to migrate entity registry for area rename: %s",
                                migrate_err,
                            )
                            # Log warning but don't fail - entities will be recreated on reload
                            # We add a warning message but don't prevent the config update
                            _LOGGER.warning(
                                "Entity registry migration failed for area rename. "
                                "Entities will be recreated on reload."
                            )

                # Update or add area
                updated_areas = []
                area_updated = False
                for area in areas:
                    if (
                        self._area_being_edited
                        and area.get(CONF_NAME) == self._area_being_edited
                    ):
                        # Update existing area
                        updated_areas.append(flattened_input)
                        area_updated = True
                    else:
                        # Keep other areas
                        updated_areas.append(area)

                if not area_updated:
                    # Add new area
                    updated_areas.append(flattened_input)

                # Save updated configuration
                config_data = {CONF_AREAS: updated_areas}
                return self.async_create_entry(title="", data=config_data)

            except HomeAssistantError as err:
                _LOGGER.error("Validation error: %s", err)
                errors["base"] = str(err)
            except vol.Invalid as err:
                _LOGGER.error("Validation error: %s", err)
                errors["base"] = str(err)
            except (ValueError, KeyError, TypeError) as err:
                _LOGGER.error("Unexpected error: %s", err)
                errors["base"] = "unknown"

        # Ensure purpose field has a default
        if CONF_PURPOSE not in defaults:
            defaults[CONF_PURPOSE] = DEFAULT_PURPOSE

        # Create schema with name field for new areas
        schema_dict = create_schema(self.hass, defaults, True)
        # Add name field if adding new area (not editing)
        if not self._area_being_edited:
            schema_dict[
                vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, ""))
            ] = str

        return self.async_show_form(
            step_id="area_config",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
        )

    async def async_step_remove_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm removal of an area."""
        # Get area_name from instance variable
        area_name = self._area_to_remove
        if not area_name:
            return await self.async_step_init()

        areas = self._get_areas_from_config()

        if user_input is not None:
            if user_input.get("confirm"):
                # Remove the area
                updated_areas = [
                    area for area in areas if area.get(CONF_NAME) != area_name
                ]

                if not updated_areas:
                    return self.async_show_form(
                        step_id="remove_area",
                        data_schema=vol.Schema({}),
                        errors={"base": "Cannot remove the last area"},
                    )

                config_data = {CONF_AREAS: updated_areas}
                return self.async_create_entry(title="", data=config_data)

            # User cancelled
            return await self.async_step_init()

        # Show confirmation
        schema = vol.Schema(
            {
                vol.Required("confirm", default=False): BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="remove_area",
            data_schema=schema,
            description_placeholders={"area_name": area_name},
        )
