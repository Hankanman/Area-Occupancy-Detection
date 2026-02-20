"""Config flow for Area Occupancy Detection integration.

This module handles the configuration flow for the Area Occupancy Detection integration.
It provides both initial configuration and options update capabilities, with comprehensive
validation of all inputs to ensure a valid configuration.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, cast

import voluptuous as vol

from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import AbortFlow, section
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar, entity_registry as er
from homeassistant.helpers.selector import (
    AreaSelector,
    AreaSelectorConfig,
    BooleanSelector,
    DurationSelector,
    DurationSelectorConfig,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TimeSelector,
)

from .const import (
    CONF_ACTION_ADD_AREA,
    CONF_ACTION_CANCEL,
    CONF_ACTION_EDIT,
    CONF_ACTION_GLOBAL_SETTINGS,
    CONF_ACTION_MANAGE_PEOPLE,
    CONF_ACTION_REMOVE,
    CONF_AIR_QUALITY_SENSORS,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_CO2_SENSORS,
    CONF_CO_SENSORS,
    CONF_COVER_ACTIVE_STATES,
    CONF_COVER_SENSORS,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MIN_PRIOR_OVERRIDE,
    CONF_MOTION_PROB_GIVEN_FALSE,
    CONF_MOTION_PROB_GIVEN_TRUE,
    CONF_MOTION_SENSORS,
    CONF_MOTION_TIMEOUT,
    CONF_OPTION_PREFIX_AREA,
    CONF_PEOPLE,
    CONF_PERSON_CONFIDENCE_THRESHOLD,
    CONF_PERSON_DEVICE_TRACKER,
    CONF_PERSON_ENTITY,
    CONF_PERSON_SLEEP_AREA,
    CONF_PERSON_SLEEP_SENSOR,
    CONF_PM10_SENSORS,
    CONF_PM25_SENSORS,
    CONF_POWER_SENSORS,
    CONF_PRESSURE_SENSORS,
    CONF_PURPOSE,
    CONF_SLEEP_END,
    CONF_SLEEP_START,
    CONF_SOUND_PRESSURE_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VOC_SENSORS,
    CONF_WASP_ENABLED,
    CONF_WASP_MAX_DURATION,
    CONF_WASP_MOTION_TIMEOUT,
    CONF_WASP_VERIFICATION_DELAY,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_COVER,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_POWER,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_COVER_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_MIN_PRIOR_OVERRIDE,
    DEFAULT_MOTION_PROB_GIVEN_FALSE,
    DEFAULT_MOTION_PROB_GIVEN_TRUE,
    DEFAULT_MOTION_TIMEOUT,
    DEFAULT_PURPOSE,
    DEFAULT_SLEEP_CONFIDENCE_THRESHOLD,
    DEFAULT_SLEEP_END,
    DEFAULT_SLEEP_START,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_VERIFICATION_DELAY,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_COVER,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_POWER,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOMAIN,
    DURATION_FIELDS,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    get_default_state,
    get_state_options,
)
from .data.purpose import Purpose, get_purpose_options

_LOGGER = logging.getLogger(__name__)

# UI Configuration Constants
WEIGHT_STEP = 0.05
WEIGHT_MIN = 0
WEIGHT_MAX = 1

THRESHOLD_STEP = 1
THRESHOLD_MIN = 1
THRESHOLD_MAX = 100


def _seconds_to_duration(seconds: float) -> dict[str, int]:
    """Convert seconds to duration dict for DurationSelector.

    Args:
        seconds: Duration in seconds

    Returns:
        Dictionary with days, hours, minutes, seconds keys.
    """
    total = int(seconds)
    return {
        "days": total // 86400,
        "hours": (total % 86400) // 3600,
        "minutes": (total % 3600) // 60,
        "seconds": total % 60,
    }


def _duration_to_seconds(duration: dict[str, int] | float) -> int:
    """Convert duration dict or raw number to seconds.

    Args:
        duration: Duration dict with days/hours/minutes/seconds keys, or raw seconds.

    Returns:
        Total seconds as integer
    """
    if isinstance(duration, (int, float)):
        return int(duration)
    return (
        duration.get("days", 0) * 86400
        + duration.get("hours", 0) * 3600
        + duration.get("minutes", 0) * 60
        + duration.get("seconds", 0)
    )


def _get_state_select_options(state_type: str) -> list[dict[str, str]]:
    """Get state options for SelectSelector."""
    states = get_state_options(state_type)
    return [
        {"value": option.value, "label": option.name} for option in states["options"]
    ]


def _entity_contains_keyword(hass: HomeAssistant, entity_id: str, keyword: str) -> bool:
    """Check if entity ID or friendly name contains a keyword.

    Args:
        hass: Home Assistant instance
        entity_id: Entity ID to check
        keyword: Keyword to search for (case-insensitive)

    Returns:
        True if keyword is found in entity_id or friendly name
    """
    # Convert keyword to lowercase for case-insensitive comparison
    keyword_lower = keyword.lower()

    # Check entity ID
    if keyword_lower in entity_id.lower():
        return True

    # Check friendly name from state
    state = hass.states.get(entity_id)
    if state and state.name:
        if keyword_lower in state.name.lower():
            return True

    return False


def _is_weather_entity(entity_id: str, platform: str | None) -> bool:
    """Check if an entity is from a weather integration.

    Weather entities measure outdoor conditions and are not suitable for
    room occupancy detection.

    Args:
        entity_id: Entity ID to check
        platform: Platform/integration that created the entity

    Returns:
        True if entity is from a weather integration
    """
    # List of weather integration platforms to exclude
    weather_platforms = {
        "weather",
        "met",
        "openweathermap",
        "accuweather",
        "weatherflow",
        "pirateweather",
        "darksky",
        "buienradar",
        "bom",
        "weatherkit",
        "metoffice",
        "nws",
        "dwd",  # Deutscher Wetterdienst (German Weather Service) - official integration
        "dwd_weather",  # DWD Weather by FL550 - HACS custom integration
    }

    # Check if platform is a known weather integration
    if platform and platform.lower() in weather_platforms:
        return True

    # Check if entity_id contains weather-related keywords
    # (as a fallback for entities without platform info)
    # Note: "outdoor" intentionally excluded - too generic, could catch legitimate sensors
    entity_lower = entity_id.lower()
    weather_keywords = ["weather", "forecast"]
    return any(keyword in entity_lower for keyword in weather_keywords)


def _get_include_entities(hass: HomeAssistant) -> dict[str, list[str]]:
    """Get lists of entities to include for specific selectors."""
    registry = er.async_get(hass)
    include_appliance_entities = []
    include_window_entities = []
    include_door_entities = []
    include_temperature_entities = []
    include_humidity_entities = []
    include_pressure_entities = []
    include_air_quality_entities = []
    include_pm25_entities = []
    include_pm10_entities = []
    include_motion_entities = []

    door_window_classes = (
        BinarySensorDeviceClass.DOOR,
        BinarySensorDeviceClass.GARAGE_DOOR,
        BinarySensorDeviceClass.OPENING,
        BinarySensorDeviceClass.WINDOW,
    )
    door_classes = (
        BinarySensorDeviceClass.DOOR,
        BinarySensorDeviceClass.GARAGE_DOOR,
    )
    door_keyword_classes = (
        BinarySensorDeviceClass.DOOR,
        BinarySensorDeviceClass.GARAGE_DOOR,
        BinarySensorDeviceClass.OPENING,
    )

    appliance_excluded_classes = [
        BinarySensorDeviceClass.MOTION,
        BinarySensorDeviceClass.OCCUPANCY,
        BinarySensorDeviceClass.PRESENCE,
        *door_window_classes,
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
            device_class = entry.device_class
            original_device_class = entry.original_device_class

            # Check if entity contains "window" or "door" keyword in entity_id or friendly name
            has_window_keyword = _entity_contains_keyword(
                hass, entry.entity_id, "window"
            )
            has_door_keyword = _entity_contains_keyword(hass, entry.entity_id, "door")

            window_class = (BinarySensorDeviceClass.WINDOW,)
            is_window_candidate = (
                device_class in window_class
                or original_device_class in window_class
                or (
                    has_window_keyword
                    and not has_door_keyword
                    and (
                        device_class in door_window_classes
                        or original_device_class in door_window_classes
                    )
                )
            )
            is_door_candidate = (
                device_class in door_classes
                or original_device_class in door_classes
                or (
                    has_door_keyword
                    and (
                        device_class in door_keyword_classes
                        or original_device_class in door_keyword_classes
                    )
                )
                or (
                    not has_window_keyword
                    and (
                        BinarySensorDeviceClass.OPENING
                        in (device_class, original_device_class)
                    )
                )
            )

            if is_window_candidate:
                include_window_entities.append(entry.entity_id)
            elif is_door_candidate:
                include_door_entities.append(entry.entity_id)

            # Exclude our own integration's sensors from motion selection
            # to prevent circular dependencies
            if entry.platform != DOMAIN:
                motion_classes = (
                    BinarySensorDeviceClass.MOTION,
                    BinarySensorDeviceClass.OCCUPANCY,
                    BinarySensorDeviceClass.PRESENCE,
                )
                if (
                    entry.device_class in motion_classes
                    or entry.original_device_class in motion_classes
                ):
                    include_motion_entities.append(entry.entity_id)

        # Filter environmental sensors to exclude weather entities
        elif entry.domain == Platform.SENSOR:
            # Skip weather entities
            if _is_weather_entity(entry.entity_id, entry.platform):
                continue

            device_class = entry.device_class
            original_device_class = entry.original_device_class

            # Include temperature sensors (excluding weather)
            temp_class = (SensorDeviceClass.TEMPERATURE,)
            if device_class in temp_class or original_device_class in temp_class:
                include_temperature_entities.append(entry.entity_id)

            # Include humidity sensors (excluding weather)
            humidity_classes = (SensorDeviceClass.HUMIDITY, SensorDeviceClass.MOISTURE)
            if (
                device_class in humidity_classes
                or original_device_class in humidity_classes
            ):
                include_humidity_entities.append(entry.entity_id)

            # Include pressure sensors (excluding weather)
            pressure_classes = (
                SensorDeviceClass.PRESSURE,
                SensorDeviceClass.ATMOSPHERIC_PRESSURE,
            )
            if (
                device_class in pressure_classes
                or original_device_class in pressure_classes
            ):
                include_pressure_entities.append(entry.entity_id)

            # Include air quality sensors (excluding weather)
            aqi_class = (SensorDeviceClass.AQI,)
            if device_class in aqi_class or original_device_class in aqi_class:
                include_air_quality_entities.append(entry.entity_id)

            # Include PM2.5 sensors (excluding weather)
            pm25_class = (SensorDeviceClass.PM25,)
            if device_class in pm25_class or original_device_class in pm25_class:
                include_pm25_entities.append(entry.entity_id)

            # Include PM10 sensors (excluding weather)
            pm10_class = (SensorDeviceClass.PM10,)
            if device_class in pm10_class or original_device_class in pm10_class:
                include_pm10_entities.append(entry.entity_id)

    # Collect all cover entities (blinds, shades, garage doors, shutters, etc.)
    include_cover_entities = [
        entry.entity_id
        for entry in registry.entities.values()
        if entry.entity_id.startswith("cover.") and not entry.disabled
    ]

    return {
        "appliance": include_appliance_entities,
        "window": include_window_entities,
        "door": include_door_entities,
        "cover": include_cover_entities,
        "temperature": include_temperature_entities,
        "humidity": include_humidity_entities,
        "pressure": include_pressure_entities,
        "air_quality": include_air_quality_entities,
        "pm25": include_pm25_entities,
        "pm10": include_pm10_entities,
        "motion": include_motion_entities,
    }


def _create_motion_section_schema(
    defaults: dict[str, Any],
    motion_entities: list[str],
    *,
    show_advanced: bool = False,
) -> vol.Schema:
    """Create schema for the motion section."""
    fields: dict[vol.Marker, Any] = {
        vol.Required(
            CONF_MOTION_SENSORS, default=defaults.get(CONF_MOTION_SENSORS, [])
        ): EntitySelector(
            EntitySelectorConfig(
                include_entities=motion_entities,
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
            default=_seconds_to_duration(
                defaults.get(CONF_MOTION_TIMEOUT, DEFAULT_MOTION_TIMEOUT)
            ),
        ): DurationSelector(DurationSelectorConfig(enable_day=False)),
    }

    if show_advanced:
        fields[
            vol.Optional(
                CONF_MOTION_PROB_GIVEN_TRUE,
                default=defaults.get(
                    CONF_MOTION_PROB_GIVEN_TRUE, DEFAULT_MOTION_PROB_GIVEN_TRUE
                ),
            )
        ] = NumberSelector(
            NumberSelectorConfig(
                min=MIN_PROBABILITY,
                max=MAX_PROBABILITY,
                step=0.01,
                mode=NumberSelectorMode.BOX,
            )
        )
        fields[
            vol.Optional(
                CONF_MOTION_PROB_GIVEN_FALSE,
                default=defaults.get(
                    CONF_MOTION_PROB_GIVEN_FALSE, DEFAULT_MOTION_PROB_GIVEN_FALSE
                ),
            )
        ] = NumberSelector(
            NumberSelectorConfig(
                min=0.001,
                max=MAX_PROBABILITY,
                step=0.001,
                mode=NumberSelectorMode.BOX,
            )
        )

    return vol.Schema(fields)


def _create_windows_and_doors_section_schema(
    defaults: dict[str, Any],
    door_entities: list[str],
    window_entities: list[str],
    cover_entities: list[str],
    door_state_options: list[SelectOptionDict],
    window_state_options: list[SelectOptionDict],
    cover_state_options: list[SelectOptionDict],
) -> vol.Schema:
    """Create schema for the combined windows, doors, and covers section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_DOOR_SENSORS, default=defaults.get(CONF_DOOR_SENSORS, [])
            ): EntitySelector(
                EntitySelectorConfig(include_entities=door_entities, multiple=True)
            ),
            vol.Optional(
                CONF_DOOR_ACTIVE_STATE,
                default=defaults.get(CONF_DOOR_ACTIVE_STATE, get_default_state("door")),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=door_state_options, mode=SelectSelectorMode.DROPDOWN
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
            vol.Optional(
                CONF_WINDOW_SENSORS, default=defaults.get(CONF_WINDOW_SENSORS, [])
            ): EntitySelector(
                EntitySelectorConfig(include_entities=window_entities, multiple=True)
            ),
            vol.Optional(
                CONF_WINDOW_ACTIVE_STATE,
                default=defaults.get(
                    CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE
                ),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=window_state_options, mode=SelectSelectorMode.DROPDOWN
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
            vol.Optional(
                CONF_COVER_SENSORS, default=defaults.get(CONF_COVER_SENSORS, [])
            ): EntitySelector(
                EntitySelectorConfig(include_entities=cover_entities, multiple=True)
            ),
            vol.Optional(
                CONF_COVER_ACTIVE_STATES,
                default=defaults.get(
                    CONF_COVER_ACTIVE_STATES, list(DEFAULT_COVER_ACTIVE_STATES)
                ),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=cover_state_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_WEIGHT_COVER,
                default=defaults.get(CONF_WEIGHT_COVER, DEFAULT_WEIGHT_COVER),
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


def _create_environmental_section_schema(
    defaults: dict[str, Any],
    temperature_entities: list[str],
    humidity_entities: list[str],
    pressure_entities: list[str],
    air_quality_entities: list[str],
    pm25_entities: list[str],
    pm10_entities: list[str],
) -> vol.Schema:
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
                    include_entities=humidity_entities,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE_SENSORS,
                default=defaults.get(CONF_TEMPERATURE_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=temperature_entities,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_CO2_SENSORS,
                default=defaults.get(CONF_CO2_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.SENSOR,
                    device_class=SensorDeviceClass.CO2,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_CO_SENSORS,
                default=defaults.get(CONF_CO_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.SENSOR,
                    device_class=SensorDeviceClass.CO,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_SOUND_PRESSURE_SENSORS,
                default=defaults.get(CONF_SOUND_PRESSURE_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.SENSOR,
                    device_class=SensorDeviceClass.SOUND_PRESSURE,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_PRESSURE_SENSORS,
                default=defaults.get(CONF_PRESSURE_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=pressure_entities,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_AIR_QUALITY_SENSORS,
                default=defaults.get(CONF_AIR_QUALITY_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=air_quality_entities,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_VOC_SENSORS,
                default=defaults.get(CONF_VOC_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.SENSOR,
                    device_class=[
                        SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
                        SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS_PARTS,
                    ],
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_PM25_SENSORS,
                default=defaults.get(CONF_PM25_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=pm25_entities,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_PM10_SENSORS,
                default=defaults.get(CONF_PM10_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=pm10_entities,
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


def _create_power_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the power section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_POWER_SENSORS,
                default=defaults.get(CONF_POWER_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain=Platform.SENSOR,
                    device_class=[
                        SensorDeviceClass.POWER,
                        SensorDeviceClass.CURRENT,
                    ],
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_WEIGHT_POWER,
                default=defaults.get(CONF_WEIGHT_POWER, DEFAULT_WEIGHT_POWER),
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


def _create_parameters_section_schema(
    defaults: dict[str, Any],
    *,
    show_advanced: bool = False,
) -> vol.Schema:
    """Create schema for the parameters section."""
    # Default decay half-life to 0 (use purpose value)
    decay_half_life_default = defaults.get(
        CONF_DECAY_HALF_LIFE, DEFAULT_DECAY_HALF_LIFE
    )

    fields: dict[vol.Marker, Any] = {
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
    }

    if show_advanced:
        fields[
            vol.Optional(
                CONF_DECAY_HALF_LIFE,
                default=_seconds_to_duration(decay_half_life_default),
            )
        ] = DurationSelector(DurationSelectorConfig(enable_day=False))
        fields[
            vol.Optional(
                CONF_MIN_PRIOR_OVERRIDE,
                default=defaults.get(
                    CONF_MIN_PRIOR_OVERRIDE, DEFAULT_MIN_PRIOR_OVERRIDE
                ),
            )
        ] = NumberSelector(
            NumberSelectorConfig(
                min=0.0,
                max=1.0,
                step=0.01,
                mode=NumberSelectorMode.SLIDER,
                unit_of_measurement="probability",
            )
        )

    return vol.Schema(fields)


def _create_wasp_in_box_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the wasp in box section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_WASP_ENABLED, default=defaults.get(CONF_WASP_ENABLED, False)
            ): BooleanSelector(),
            vol.Optional(
                CONF_WASP_MOTION_TIMEOUT,
                default=_seconds_to_duration(
                    defaults.get(CONF_WASP_MOTION_TIMEOUT, DEFAULT_WASP_MOTION_TIMEOUT)
                ),
            ): DurationSelector(DurationSelectorConfig(enable_day=False)),
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
                default=_seconds_to_duration(
                    defaults.get(CONF_WASP_MAX_DURATION, DEFAULT_WASP_MAX_DURATION)
                ),
            ): DurationSelector(DurationSelectorConfig(enable_day=True)),
            vol.Optional(
                CONF_WASP_VERIFICATION_DELAY,
                default=_seconds_to_duration(
                    defaults.get(
                        CONF_WASP_VERIFICATION_DELAY, DEFAULT_WASP_VERIFICATION_DELAY
                    )
                ),
            ): DurationSelector(DurationSelectorConfig(enable_day=False)),
        }
    )


def create_schema(
    hass: HomeAssistant,
    defaults: dict[str, Any] | None = None,
    is_options: bool = False,
    include_entities: dict[str, list[str]] | None = None,
    *,
    show_advanced: bool = False,
) -> dict:
    """Create a schema with optional default values, using helper functions.

    Args:
        hass: Home Assistant instance
        defaults: Optional default values for form fields
        is_options: Whether this is for options flow (vs initial config flow)
        include_entities: Optional pre-computed entity lists. If not provided,
            will be computed from hass.
        show_advanced: Whether to show advanced options (motion probs, decay half-life, min prior)

    Returns:
        Schema dictionary for form
    """
    # Ensure defaults is a dictionary
    defaults = defaults if defaults is not None else {}

    # Pre-calculate expensive lookups (or use provided)
    if include_entities is None:
        include_entities = _get_include_entities(hass)
    door_state_options = _get_state_select_options("door")
    media_state_options = _get_state_select_options("media")
    window_state_options = _get_state_select_options("window")
    cover_state_options = _get_state_select_options("cover")
    appliance_state_options = _get_state_select_options("appliance")

    # Initialize the dictionary for the schema
    schema_dict: dict[vol.Marker, Any] = {}

    # Get default area ID from defaults (for editing existing areas)
    default_area_id = defaults.get(CONF_AREA_ID, "")

    # Add area selector (same for both initial and options flow)
    schema_dict[vol.Required(CONF_AREA_ID, default=default_area_id)] = AreaSelector(
        AreaSelectorConfig()
    )
    # Add purpose field at root level (not in a section)
    schema_dict[
        vol.Optional(CONF_PURPOSE, default=defaults.get(CONF_PURPOSE, DEFAULT_PURPOSE))
    ] = SelectSelector(
        SelectSelectorConfig(
            options=cast("list[SelectOptionDict]", get_purpose_options()),
            mode=SelectSelectorMode.DROPDOWN,
        )
    )

    # Add sections by assigning keys directly to the dictionary
    schema_dict[vol.Required("motion")] = section(
        _create_motion_section_schema(
            defaults, include_entities["motion"], show_advanced=show_advanced
        ),
        {"collapsed": True},
    )
    schema_dict[vol.Required("windows_and_doors")] = section(
        _create_windows_and_doors_section_schema(
            defaults,
            include_entities["door"],
            include_entities["window"],
            include_entities["cover"],
            cast("list[SelectOptionDict]", door_state_options),
            cast("list[SelectOptionDict]", window_state_options),
            cast("list[SelectOptionDict]", cover_state_options),
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
        _create_environmental_section_schema(
            defaults,
            include_entities["temperature"],
            include_entities["humidity"],
            include_entities["pressure"],
            include_entities["air_quality"],
            include_entities["pm25"],
            include_entities["pm10"],
        ),
        {"collapsed": True},
    )
    schema_dict[vol.Required("power")] = section(
        _create_power_section_schema(defaults), {"collapsed": True}
    )
    schema_dict[vol.Required("wasp_in_box")] = section(
        _create_wasp_in_box_section_schema(defaults), {"collapsed": True}
    )
    schema_dict[vol.Required("parameters")] = section(
        _create_parameters_section_schema(defaults, show_advanced=show_advanced),
        {"collapsed": True},
    )

    # Pass the correctly structured dictionary to vol.Schema
    return schema_dict


# ── Wizard step schemas ──────────────────────────────────────────────


def _create_basics_step_schema() -> dict[vol.Marker, Any]:
    """Create schema for wizard step 1: area selection and purpose."""
    return {
        vol.Required(CONF_AREA_ID): AreaSelector(AreaSelectorConfig()),
        vol.Optional(CONF_PURPOSE, default=DEFAULT_PURPOSE): SelectSelector(
            SelectSelectorConfig(
                options=cast("list[SelectOptionDict]", get_purpose_options()),
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }


def _create_motion_step_schema(
    hass: HomeAssistant,
    include_entities: dict[str, list[str]] | None = None,
    *,
    show_advanced: bool = False,
) -> dict[vol.Marker, Any]:
    """Create schema for wizard step 2: motion sensor configuration."""
    if include_entities is None:
        include_entities = _get_include_entities(hass)

    fields: dict[vol.Marker, Any] = {
        vol.Required(CONF_MOTION_SENSORS, default=[]): EntitySelector(
            EntitySelectorConfig(
                include_entities=include_entities["motion"],
                multiple=True,
            )
        ),
        vol.Optional(CONF_WEIGHT_MOTION, default=DEFAULT_WEIGHT_MOTION): NumberSelector(
            NumberSelectorConfig(
                min=WEIGHT_MIN,
                max=WEIGHT_MAX,
                step=WEIGHT_STEP,
                mode=NumberSelectorMode.SLIDER,
            )
        ),
        vol.Optional(
            CONF_MOTION_TIMEOUT,
            default=_seconds_to_duration(DEFAULT_MOTION_TIMEOUT),
        ): DurationSelector(DurationSelectorConfig(enable_day=False)),
    }

    if show_advanced:
        fields[
            vol.Optional(
                CONF_MOTION_PROB_GIVEN_TRUE,
                default=DEFAULT_MOTION_PROB_GIVEN_TRUE,
            )
        ] = NumberSelector(
            NumberSelectorConfig(
                min=MIN_PROBABILITY,
                max=MAX_PROBABILITY,
                step=0.01,
                mode=NumberSelectorMode.BOX,
            )
        )
        fields[
            vol.Optional(
                CONF_MOTION_PROB_GIVEN_FALSE,
                default=DEFAULT_MOTION_PROB_GIVEN_FALSE,
            )
        ] = NumberSelector(
            NumberSelectorConfig(
                min=0.001,
                max=MAX_PROBABILITY,
                step=0.001,
                mode=NumberSelectorMode.BOX,
            )
        )

    return fields


def _create_sensors_step_schema(
    hass: HomeAssistant,
    include_entities: dict[str, list[str]] | None = None,
) -> dict[vol.Marker, Any]:
    """Create schema for wizard step 3: additional sensors with sections."""
    if include_entities is None:
        include_entities = _get_include_entities(hass)

    defaults: dict[str, Any] = {}
    door_state_options = _get_state_select_options("door")
    media_state_options = _get_state_select_options("media")
    window_state_options = _get_state_select_options("window")
    cover_state_options = _get_state_select_options("cover")
    appliance_state_options = _get_state_select_options("appliance")

    return {
        vol.Required("windows_and_doors"): section(
            _create_windows_and_doors_section_schema(
                defaults,
                include_entities["door"],
                include_entities["window"],
                include_entities["cover"],
                cast("list[SelectOptionDict]", door_state_options),
                cast("list[SelectOptionDict]", window_state_options),
                cast("list[SelectOptionDict]", cover_state_options),
            ),
            {"collapsed": True},
        ),
        vol.Required("media"): section(
            _create_media_section_schema(
                defaults, cast("list[SelectOptionDict]", media_state_options)
            ),
            {"collapsed": True},
        ),
        vol.Required("appliances"): section(
            _create_appliances_section_schema(
                defaults,
                include_entities["appliance"],
                cast("list[SelectOptionDict]", appliance_state_options),
            ),
            {"collapsed": True},
        ),
        vol.Required("environmental"): section(
            _create_environmental_section_schema(
                defaults,
                include_entities["temperature"],
                include_entities["humidity"],
                include_entities["pressure"],
                include_entities["air_quality"],
                include_entities["pm25"],
                include_entities["pm10"],
            ),
            {"collapsed": True},
        ),
        vol.Required("power"): section(
            _create_power_section_schema(defaults), {"collapsed": True}
        ),
    }


def _create_behavior_step_schema(
    *,
    show_advanced: bool = False,
) -> dict[vol.Marker, Any]:
    """Create schema for wizard step 4: thresholds, decay, and wasp-in-box."""
    fields: dict[vol.Marker, Any] = {
        vol.Optional(CONF_THRESHOLD, default=DEFAULT_THRESHOLD): NumberSelector(
            NumberSelectorConfig(
                min=THRESHOLD_MIN,
                max=THRESHOLD_MAX,
                step=THRESHOLD_STEP,
                mode=NumberSelectorMode.SLIDER,
            )
        ),
        vol.Optional(
            CONF_DECAY_ENABLED, default=DEFAULT_DECAY_ENABLED
        ): BooleanSelector(),
    }

    if show_advanced:
        fields[
            vol.Optional(
                CONF_DECAY_HALF_LIFE,
                default=_seconds_to_duration(DEFAULT_DECAY_HALF_LIFE),
            )
        ] = DurationSelector(DurationSelectorConfig(enable_day=False))
        fields[
            vol.Optional(
                CONF_MIN_PRIOR_OVERRIDE,
                default=DEFAULT_MIN_PRIOR_OVERRIDE,
            )
        ] = NumberSelector(
            NumberSelectorConfig(
                min=0.0,
                max=1.0,
                step=0.01,
                mode=NumberSelectorMode.SLIDER,
                unit_of_measurement="probability",
            )
        )

    # Wasp-in-box fields at top level
    fields[vol.Optional(CONF_WASP_ENABLED, default=False)] = BooleanSelector()
    fields[
        vol.Optional(
            CONF_WASP_MOTION_TIMEOUT,
            default=_seconds_to_duration(DEFAULT_WASP_MOTION_TIMEOUT),
        )
    ] = DurationSelector(DurationSelectorConfig(enable_day=False))
    fields[vol.Optional(CONF_WASP_WEIGHT, default=DEFAULT_WASP_WEIGHT)] = (
        NumberSelector(
            NumberSelectorConfig(
                min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER
            )
        )
    )
    fields[
        vol.Optional(
            CONF_WASP_MAX_DURATION,
            default=_seconds_to_duration(DEFAULT_WASP_MAX_DURATION),
        )
    ] = DurationSelector(DurationSelectorConfig(enable_day=True))
    fields[
        vol.Optional(
            CONF_WASP_VERIFICATION_DELAY,
            default=_seconds_to_duration(DEFAULT_WASP_VERIFICATION_DELAY),
        )
    ] = DurationSelector(DurationSelectorConfig(enable_day=False))

    return fields


def _nest_config_for_sections(flat_config: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Restructure flat area config into section-nested format for suggested values.

    Args:
        flat_config: Flat area configuration dictionary

    Returns:
        Nested dictionary matching the sectioned schema structure
    """
    nested: dict[str, Any] = {}

    # Root-level fields
    if CONF_AREA_ID in flat_config:
        nested[CONF_AREA_ID] = flat_config[CONF_AREA_ID]
    if CONF_PURPOSE in flat_config:
        nested[CONF_PURPOSE] = flat_config[CONF_PURPOSE]

    # Motion section
    motion: dict[str, Any] = {}
    for key in (
        CONF_MOTION_SENSORS,
        CONF_WEIGHT_MOTION,
        CONF_MOTION_TIMEOUT,
        CONF_MOTION_PROB_GIVEN_TRUE,
        CONF_MOTION_PROB_GIVEN_FALSE,
    ):
        if key in flat_config:
            val = flat_config[key]
            # Convert seconds to duration for DurationSelector fields
            if key in DURATION_FIELDS:
                val = _seconds_to_duration(val)
            motion[key] = val
    if motion:
        nested["motion"] = motion

    # Windows and doors section
    windows_and_doors: dict[str, Any] = {}
    for key in (
        CONF_DOOR_SENSORS,
        CONF_DOOR_ACTIVE_STATE,
        CONF_WEIGHT_DOOR,
        CONF_WINDOW_SENSORS,
        CONF_WINDOW_ACTIVE_STATE,
        CONF_WEIGHT_WINDOW,
        CONF_COVER_SENSORS,
        CONF_COVER_ACTIVE_STATES,
        CONF_WEIGHT_COVER,
    ):
        if key in flat_config:
            windows_and_doors[key] = flat_config[key]
    if windows_and_doors:
        nested["windows_and_doors"] = windows_and_doors

    # Media section
    media: dict[str, Any] = {}
    for key in (CONF_MEDIA_DEVICES, CONF_MEDIA_ACTIVE_STATES, CONF_WEIGHT_MEDIA):
        if key in flat_config:
            media[key] = flat_config[key]
    if media:
        nested["media"] = media

    # Appliances section
    appliances: dict[str, Any] = {}
    for key in (CONF_APPLIANCES, CONF_APPLIANCE_ACTIVE_STATES, CONF_WEIGHT_APPLIANCE):
        if key in flat_config:
            appliances[key] = flat_config[key]
    if appliances:
        nested["appliances"] = appliances

    # Environmental section
    environmental: dict[str, Any] = {}
    for key in (
        CONF_ILLUMINANCE_SENSORS,
        CONF_HUMIDITY_SENSORS,
        CONF_TEMPERATURE_SENSORS,
        CONF_CO2_SENSORS,
        CONF_CO_SENSORS,
        CONF_SOUND_PRESSURE_SENSORS,
        CONF_PRESSURE_SENSORS,
        CONF_AIR_QUALITY_SENSORS,
        CONF_VOC_SENSORS,
        CONF_PM25_SENSORS,
        CONF_PM10_SENSORS,
        CONF_WEIGHT_ENVIRONMENTAL,
    ):
        if key in flat_config:
            environmental[key] = flat_config[key]
    if environmental:
        nested["environmental"] = environmental

    # Power section
    power: dict[str, Any] = {}
    for key in (CONF_POWER_SENSORS, CONF_WEIGHT_POWER):
        if key in flat_config:
            power[key] = flat_config[key]
    if power:
        nested["power"] = power

    # Wasp in box section
    wasp: dict[str, Any] = {}
    for key in (
        CONF_WASP_ENABLED,
        CONF_WASP_MOTION_TIMEOUT,
        CONF_WASP_WEIGHT,
        CONF_WASP_MAX_DURATION,
        CONF_WASP_VERIFICATION_DELAY,
    ):
        if key in flat_config:
            val = flat_config[key]
            if key in DURATION_FIELDS:
                val = _seconds_to_duration(val)
            wasp[key] = val
    if wasp:
        nested["wasp_in_box"] = wasp

    # Parameters section
    parameters: dict[str, Any] = {}
    for key in (
        CONF_THRESHOLD,
        CONF_DECAY_ENABLED,
        CONF_DECAY_HALF_LIFE,
        CONF_MIN_PRIOR_OVERRIDE,
    ):
        if key in flat_config:
            val = flat_config[key]
            if key in DURATION_FIELDS:
                val = _seconds_to_duration(val)
            parameters[key] = val
    if parameters:
        nested["parameters"] = parameters

    return nested


def _draft_to_suggested(draft: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    """Extract suggested values from draft, converting duration fields.

    Args:
        draft: Flat area configuration draft
        keys: Set of field keys to extract

    Returns:
        Dictionary of suggested values with durations converted for display
    """
    suggested: dict[str, Any] = {}
    for key in keys:
        if key in draft:
            val = draft[key]
            if key in DURATION_FIELDS:
                val = _seconds_to_duration(val)
            suggested[key] = val
    return suggested


def _resolve_area_id_to_name(hass: HomeAssistant, area_id: str) -> str:
    """Resolve area ID to area name for display.

    Args:
        hass: Home Assistant instance
        area_id: Home Assistant area ID

    Returns:
        Area name from Home Assistant registry

    Raises:
        ValueError: If area ID doesn't exist in registry
    """
    registry = ar.async_get(hass)
    area_entry = registry.async_get_area(area_id)
    if not area_entry:
        raise ValueError(
            f"Area ID '{area_id}' not found in Home Assistant area registry"
        )
    return area_entry.name


def _get_purpose_display_name(purpose: str) -> str:
    """Get display name for a purpose value.

    Args:
        purpose: Purpose enum value string

    Returns:
        Human-readable purpose name
    """
    return Purpose.display_name(purpose)


def _find_area_by_sanitized_id(
    areas: list[dict[str, Any]], sanitized_id: str
) -> dict[str, Any] | None:
    """Find an area by matching sanitized area ID.

    Args:
        areas: List of area configurations
        sanitized_id: Sanitized area ID to find

    Returns:
        Area configuration dict if found, None otherwise
    """
    for area in areas:
        area_id = area.get(CONF_AREA_ID)
        if not area_id:
            continue
        area_sanitized = area_id.replace(" ", "_").replace("/", "_")
        if area_sanitized == sanitized_id:
            return area
    return None


def _build_area_description_placeholders(
    area_config: dict[str, Any], area_id: str, hass: HomeAssistant | None = None
) -> dict[str, str]:
    """Build description placeholders for area action form.

    Args:
        area_config: Area configuration dictionary
        area_id: Area ID
        hass: Home Assistant instance (optional, for resolving area name)

    Returns:
        Dictionary of placeholders for form description
    """
    # Resolve area name from ID
    area_name = area_id
    if hass:
        try:
            area_name = _resolve_area_id_to_name(hass, area_id)
        except ValueError:
            area_name = area_id

    purpose = area_config.get(CONF_PURPOSE, DEFAULT_PURPOSE)
    purpose_name = _get_purpose_display_name(purpose)

    return {
        "area_name": area_name,
        "purpose": purpose_name,
        "motion_count": str(len(area_config.get(CONF_MOTION_SENSORS, []))),
        "media_count": str(len(area_config.get(CONF_MEDIA_DEVICES, []))),
        "door_count": str(len(area_config.get(CONF_DOOR_SENSORS, []))),
        "window_count": str(len(area_config.get(CONF_WINDOW_SENSORS, []))),
        "appliance_count": str(len(area_config.get(CONF_APPLIANCES, []))),
        "threshold": str(area_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)),
    }


def _count_area_sensors(area: dict[str, Any]) -> int:
    """Count total configured input sensors for an area.

    Args:
        area: Area configuration dictionary

    Returns:
        Total number of configured input sensors
    """
    sensor_keys = [
        CONF_AIR_QUALITY_SENSORS,
        CONF_APPLIANCES,
        CONF_CO2_SENSORS,
        CONF_CO_SENSORS,
        CONF_COVER_SENSORS,
        CONF_DOOR_SENSORS,
        CONF_HUMIDITY_SENSORS,
        CONF_ILLUMINANCE_SENSORS,
        CONF_MEDIA_DEVICES,
        CONF_MOTION_SENSORS,
        CONF_PM10_SENSORS,
        CONF_PM25_SENSORS,
        CONF_POWER_SENSORS,
        CONF_PRESSURE_SENSORS,
        CONF_SOUND_PRESSURE_SENSORS,
        CONF_TEMPERATURE_SENSORS,
        CONF_VOC_SENSORS,
        CONF_WINDOW_SENSORS,
    ]
    return sum(len(area.get(key, [])) for key in sensor_keys)


def _get_area_summary_info(area: dict[str, Any]) -> str:
    """Get formatted summary information for an area.

    Args:
        area: Area configuration dictionary

    Returns:
        Formatted string with purpose, sensor count, and threshold
    """
    purpose = area.get(CONF_PURPOSE, DEFAULT_PURPOSE)
    purpose_name = _get_purpose_display_name(purpose)

    # Count sensors
    motion_count = len(area.get(CONF_MOTION_SENSORS, []))
    media_count = len(area.get(CONF_MEDIA_DEVICES, []))
    door_count = len(area.get(CONF_DOOR_SENSORS, []))
    window_count = len(area.get(CONF_WINDOW_SENSORS, []))
    appliance_count = len(area.get(CONF_APPLIANCES, []))
    total_sensors = (
        motion_count + media_count + door_count + window_count + appliance_count
    )

    threshold = area.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)

    return (
        f"Purpose: {purpose_name} • {total_sensors} sensors • Threshold: {threshold}%"
    )


def _apply_purpose_based_decay_default(
    flattened_input: dict[str, Any], purpose: str | None
) -> None:
    """Apply purpose-based default for decay half-life.

    If decay half-life is not set or matches a default value, set it to 0
    (which means "use purpose value"). Modifies flattened_input in place.

    Args:
        flattened_input: Flattened configuration dictionary
        purpose: Selected purpose value
    """
    if not purpose:
        return

    user_set_decay = flattened_input.get(CONF_DECAY_HALF_LIFE)
    if user_set_decay is None or Purpose.is_purpose_half_life(user_set_decay):
        # Set to 0 to indicate "use purpose value"
        flattened_input[CONF_DECAY_HALF_LIFE] = 0


def _flatten_sectioned_input(user_input: dict[str, Any]) -> dict[str, Any]:
    """Flatten sectioned user input into flat configuration dictionary.

    Converts nested section structure (motion, doors, windows, etc.) into
    a flat dictionary suitable for validation and storage.

    Args:
        user_input: Sectioned user input dictionary

    Returns:
        Flattened configuration dictionary
    """
    flattened_input = {}
    for key, value in user_input.items():
        if isinstance(value, dict) and key not in DURATION_FIELDS:
            # All sections (motion, doors, windows, wasp_in_box, etc.) are flattened the same way
            flattened_input.update(value)
        else:
            flattened_input[key] = value

    # Convert duration fields from DurationSelector format back to seconds
    for field in DURATION_FIELDS:
        if field in flattened_input:
            flattened_input[field] = _duration_to_seconds(flattened_input[field])

    return flattened_input


def _find_area_by_id(
    areas: list[dict[str, Any]], area_id: str
) -> dict[str, Any] | None:
    """Find an area by ID in a list of areas.

    Args:
        areas: List of area configuration dictionaries
        area_id: Area ID to find

    Returns:
        Area configuration dictionary if found, None otherwise
    """
    for area in areas:
        if area.get(CONF_AREA_ID) == area_id:
            return area
    return None


def _update_area_in_list(
    areas: list[dict[str, Any]],
    updated_area: dict[str, Any],
    area_id: str | None,
) -> list[dict[str, Any]]:
    """Update or add an area in a list of areas.

    Args:
        areas: List of area configuration dictionaries
        updated_area: Updated area configuration
        area_id: Area ID being updated (None for new area)

    Returns:
        Updated list of areas
    """
    updated_areas = []
    area_updated = False
    for area in areas:
        if area_id and area.get(CONF_AREA_ID) == area_id:
            # Update existing area
            updated_areas.append(updated_area)
            area_updated = True
        else:
            # Keep other areas
            updated_areas.append(area)

    if not area_updated:
        # Add new area
        updated_areas.append(updated_area)

    return updated_areas


def _remove_area_from_list(
    areas: list[dict[str, Any]], area_id: str
) -> list[dict[str, Any]]:
    """Remove an area from a list of areas.

    Args:
        areas: List of area configuration dictionaries
        area_id: Area ID to remove

    Returns:
        Updated list of areas with specified area removed
    """
    return [area for area in areas if area.get(CONF_AREA_ID) != area_id]


def _validate_person_input(user_input: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize person configuration input.

    Args:
        user_input: Raw user input from person config form

    Returns:
        Validated person data dict

    Raises:
        vol.Invalid: If required fields are missing or empty
    """
    person_entity = user_input.get(CONF_PERSON_ENTITY, "")
    sleep_sensor = user_input.get(CONF_PERSON_SLEEP_SENSOR, "")
    sleep_area = user_input.get(CONF_PERSON_SLEEP_AREA, "")

    if not person_entity:
        raise vol.Invalid("person_entity_required")
    if not sleep_sensor:
        raise vol.Invalid("sleep_sensor_required")
    if not sleep_area:
        raise vol.Invalid("sleep_area_required")

    raw_threshold = user_input.get(
        CONF_PERSON_CONFIDENCE_THRESHOLD, DEFAULT_SLEEP_CONFIDENCE_THRESHOLD
    )
    try:
        threshold = int(raw_threshold)
    except (ValueError, TypeError) as err:
        raise vol.Invalid("confidence_not_number") from err
    threshold = max(1, min(100, threshold))

    result = {
        CONF_PERSON_ENTITY: person_entity,
        CONF_PERSON_SLEEP_SENSOR: sleep_sensor,
        CONF_PERSON_SLEEP_AREA: sleep_area,
        CONF_PERSON_CONFIDENCE_THRESHOLD: threshold,
    }

    device_tracker = user_input.get(CONF_PERSON_DEVICE_TRACKER, "")
    if device_tracker:
        result[CONF_PERSON_DEVICE_TRACKER] = device_tracker

    return result


def _handle_step_error(err: Exception) -> str:
    """Handle step errors and convert to user-friendly error message.

    Args:
        err: Exception that occurred during step processing

    Returns:
        Error message string for display to user
    """
    if isinstance(err, HomeAssistantError):
        _LOGGER.error("Validation error: %s", err)
        return str(err)
    if isinstance(err, vol.Invalid):
        _LOGGER.error("Validation error: %s", err)
        return str(err)
    # ValueError, KeyError, TypeError
    _LOGGER.error("Unexpected error: %s", err)
    return "unknown"


def _create_area_selector_schema(
    areas: list[dict[str, Any]], hass: HomeAssistant | None = None
) -> vol.Schema:
    """Create schema for area selection step.

    Args:
        areas: List of configured areas
        hass: Home Assistant instance (optional, for resolving area names)

    Returns:
        Schema with SelectSelector in LIST mode (radio buttons) for area selection
    """
    # Ensure areas is a list
    if not isinstance(areas, list):
        areas = []

    options: list[SelectOptionDict] = []

    # Add each area as an option
    for area in areas:
        if not isinstance(area, dict):
            _LOGGER.warning("Skipping invalid area config (not a dict): %s", area)
            continue
        area_id = area.get(CONF_AREA_ID)
        if not area_id:
            _LOGGER.warning(
                "Area config missing area_id, skipping: %s",
                area,
            )
            continue

        # Resolve area name from ID
        area_name = "Unknown"
        if hass:
            try:
                area_name = _resolve_area_id_to_name(hass, area_id)
            except ValueError as err:
                # Area was deleted, log and skip it
                _LOGGER.warning(
                    "Area ID '%s' not found in registry (may have been deleted), skipping: %s",
                    area_id,
                    err,
                )
                continue
        else:
            # Fallback to area_id if we can't resolve
            area_name = area_id

        summary = _get_area_summary_info(area)
        # Use area_id for option value (sanitized)
        sanitized_id = area_id.replace(" ", "_").replace("/", "_")
        # Include summary in label for better UX
        options.append(
            {
                "value": f"{CONF_OPTION_PREFIX_AREA}{sanitized_id}",
                "label": f"{area_name} - {summary}",
            }
        )

    return vol.Schema(
        {
            vol.Required("selected_option"): SelectSelector(
                SelectSelectorConfig(
                    options=options,
                    mode=SelectSelectorMode.LIST,
                )
            )
        }
    )


def _create_action_selection_schema() -> vol.Schema:
    """Create schema for action selection step.

    Returns:
        Schema with SelectSelector in LIST mode (radio buttons) for action selection
    """
    return vol.Schema(
        {
            vol.Required("action"): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": CONF_ACTION_EDIT, "label": "Edit"},
                        {"value": CONF_ACTION_REMOVE, "label": "Remove"},
                        {"value": CONF_ACTION_CANCEL, "label": "Cancel"},
                    ],
                    mode=SelectSelectorMode.LIST,
                )
            )
        }
    )


def _create_global_settings_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for global settings."""
    return vol.Schema(
        {
            vol.Required(
                CONF_SLEEP_START,
                default=defaults.get(CONF_SLEEP_START, DEFAULT_SLEEP_START),
            ): TimeSelector(),
            vol.Required(
                CONF_SLEEP_END,
                default=defaults.get(CONF_SLEEP_END, DEFAULT_SLEEP_END),
            ): TimeSelector(),
        }
    )


class BaseOccupancyFlow:
    """Base class for config and options flow.

    This class provides shared validation logic used by both the config flow
    and options flow. It ensures consistent validation across both flows.
    """

    def _validate_duplicate_area_id(
        self,
        flattened_input: dict[str, Any],
        areas: list[dict[str, Any]],
        area_id_being_edited: str | None = None,
        hass: HomeAssistant | None = None,
    ) -> dict[str, str]:
        """Validate that area ID is not a duplicate.

        Args:
            flattened_input: The flattened input configuration
            areas: List of existing area configurations
            area_id_being_edited: Optional area ID being edited (to exclude from duplicate check)
            hass: Home Assistant instance (for resolving area names in error messages)

        Returns:
            Dictionary mapping field keys to error translation keys
        """
        errors: dict[str, str] = {}
        area_id = flattened_input.get(CONF_AREA_ID, "")
        if area_id:
            for area in areas:
                existing_area_id = area.get(CONF_AREA_ID)
                if existing_area_id == area_id and (
                    not area_id_being_edited or existing_area_id != area_id_being_edited
                ):
                    errors[CONF_AREA_ID] = "area_already_configured"
                    break
        return errors

    def _validate_config(
        self, data: dict[str, Any], hass: HomeAssistant | None = None
    ) -> dict[str, str]:
        """Validate the configuration and return per-field errors.

        Performs comprehensive validation of all configuration fields including:
        - Required area ID and validation against Home Assistant registry
        - Required sensors and their relationships
        - State configurations for different device types
        - Weight values and their ranges

        Args:
            data: Dictionary containing the configuration to validate
            hass: Home Assistant instance (for validating area ID)

        Returns:
            Dictionary mapping field keys to error translation keys.
            Empty dict means validation passed.
        """
        errors: dict[str, str] = {}

        # Validate area ID
        area_id = data.get(CONF_AREA_ID, "")
        if not area_id:
            errors[CONF_AREA_ID] = "area_required"
        elif hass:
            try:
                _resolve_area_id_to_name(hass, area_id)
            except ValueError:
                errors[CONF_AREA_ID] = "area_not_found"

        # Validate purpose
        purpose = data.get(CONF_PURPOSE, DEFAULT_PURPOSE)
        if not purpose:
            errors[CONF_PURPOSE] = "purpose_required"

        # Validate motion sensors (section field → base fallback)
        motion_sensors = data.get(CONF_MOTION_SENSORS, [])
        if not motion_sensors:
            errors.setdefault("base", "motion_required")

        # Validate motion sensor likelihoods
        motion_prob_given_true = data.get(
            CONF_MOTION_PROB_GIVEN_TRUE, DEFAULT_MOTION_PROB_GIVEN_TRUE
        )
        motion_prob_given_false = data.get(
            CONF_MOTION_PROB_GIVEN_FALSE, DEFAULT_MOTION_PROB_GIVEN_FALSE
        )
        if motion_prob_given_true <= motion_prob_given_false:
            errors.setdefault("base", "prob_true_must_exceed_false")

        # Validate threshold
        threshold = data.get(CONF_THRESHOLD)
        if threshold is not None and (
            not isinstance(threshold, (int, float)) or threshold < 1 or threshold > 100
        ):
            errors[CONF_THRESHOLD] = "invalid_threshold"

        # Validate media devices
        media_devices = data.get(CONF_MEDIA_DEVICES, [])
        media_states = data.get(CONF_MEDIA_ACTIVE_STATES, DEFAULT_MEDIA_ACTIVE_STATES)
        if media_devices and not media_states:
            errors[CONF_MEDIA_DEVICES] = "media_states_required"

        # Validate appliances
        appliances = data.get(CONF_APPLIANCES, [])
        appliance_states = data.get(
            CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_APPLIANCE_ACTIVE_STATES
        )
        if appliances and not appliance_states:
            errors[CONF_APPLIANCES] = "appliance_states_required"

        # Validate doors
        door_sensors = data.get(CONF_DOOR_SENSORS, [])
        door_state = data.get(CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE)
        if door_sensors and not door_state:
            errors[CONF_DOOR_SENSORS] = "door_state_required"

        # Validate windows
        window_sensors = data.get(CONF_WINDOW_SENSORS, [])
        window_state = data.get(CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE)
        if window_sensors and not window_state:
            errors[CONF_WINDOW_SENSORS] = "window_state_required"

        # Validate covers
        cover_sensors = data.get(CONF_COVER_SENSORS, [])
        cover_states = data.get(CONF_COVER_ACTIVE_STATES, DEFAULT_COVER_ACTIVE_STATES)
        if cover_sensors and not cover_states:
            errors[CONF_COVER_SENSORS] = "cover_states_required"

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
            (CONF_WEIGHT_COVER, data.get(CONF_WEIGHT_COVER, DEFAULT_WEIGHT_COVER)),
            (
                CONF_WEIGHT_ENVIRONMENTAL,
                data.get(CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL),
            ),
            (
                CONF_WEIGHT_POWER,
                data.get(CONF_WEIGHT_POWER, DEFAULT_WEIGHT_POWER),
            ),
        ]
        for name, weight in weights:
            if not WEIGHT_MIN <= weight <= WEIGHT_MAX:
                errors[name] = "invalid_weight"
                break

        # Validate decay settings
        decay_enabled = data.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        if decay_enabled:
            decay_window = data.get(CONF_DECAY_HALF_LIFE, DEFAULT_DECAY_HALF_LIFE)
            # Allow 0 (use purpose value) or values between 10 and 3600
            if not isinstance(decay_window, (int, float)) or (
                decay_window != 0 and (decay_window < 10 or decay_window > 3600)
            ):
                errors[CONF_DECAY_HALF_LIFE] = "invalid_decay_half_life"

        return errors

    def _route_area_action(self, action: str, area_id: str) -> None:
        """Route area action to appropriate step.

        Args:
            action: Action selected by user
            area_id: Area ID being acted upon
        """
        if action == CONF_ACTION_EDIT:
            # User wants to edit the area - clear any stale removal state
            self._area_to_remove = None
        elif action == CONF_ACTION_REMOVE:
            # User wants to remove the area
            self._area_to_remove = area_id
            self._area_being_edited = None
        elif action == CONF_ACTION_CANCEL:
            # User cancelled - clear any stale removal state
            self._area_to_remove = None
            self._area_being_edited = None

    # ── Multi-step area config wizard ────────────────────────────────

    def _get_wizard_areas(self) -> list[dict[str, Any]]:
        """Get the list of areas for duplicate checking. Overridden by subclasses."""
        raise NotImplementedError

    async def _on_area_config_complete(
        self, config: dict[str, Any]
    ) -> ConfigFlowResult:
        """Handle completion of the area config wizard. Overridden by subclasses."""
        raise NotImplementedError

    def _init_area_wizard(self) -> None:
        """Initialize the area config wizard draft."""
        if self._area_being_edited:
            areas = self._get_wizard_areas()
            area = _find_area_by_id(areas, self._area_being_edited)
            self._area_config_draft = area.copy() if area else {}
        else:
            self._area_config_draft = {}

    async def async_step_area_basics(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Wizard step 1: Area selection and purpose."""
        errors: dict[str, str] = {}

        if user_input is not None:
            area_id = user_input.get(CONF_AREA_ID, "")
            if not area_id:
                errors[CONF_AREA_ID] = "area_required"
            elif self.hass:
                try:
                    _resolve_area_id_to_name(self.hass, area_id)
                except ValueError:
                    errors[CONF_AREA_ID] = "area_not_found"

            if not errors:
                areas = self._get_wizard_areas()
                errors.update(
                    self._validate_duplicate_area_id(
                        user_input, areas, self._area_being_edited, self.hass
                    )
                )

            if not errors:
                self._area_config_draft.update(user_input)
                return await self.async_step_area_motion()

        schema_dict = _create_basics_step_schema()
        base_schema = vol.Schema(schema_dict)

        # Apply suggested values from draft or user_input (for error re-display)
        suggested = (
            user_input
            if user_input is not None
            else _draft_to_suggested(
                self._area_config_draft, {CONF_AREA_ID, CONF_PURPOSE}
            )
        )
        if suggested:
            data_schema = self.add_suggested_values_to_schema(base_schema, suggested)
        else:
            data_schema = base_schema

        # Build description placeholders
        if self._area_being_edited and self._area_config_draft.get(CONF_AREA_ID):
            area_name = self._area_config_draft.get(CONF_AREA_ID, "")
            with contextlib.suppress(ValueError):
                area_name = _resolve_area_id_to_name(
                    self.hass, self._area_config_draft[CONF_AREA_ID]
                )
            placeholders = {"mode": "Editing", "area_name": area_name}
        else:
            placeholders = {"mode": "Adding", "area_name": "New Area"}

        return self.async_show_form(
            step_id="area_basics",
            data_schema=data_schema,
            errors=errors,
            description_placeholders=placeholders,
            last_step=False,
        )

    async def async_step_area_motion(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Wizard step 2: Motion sensor configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            flattened = _flatten_sectioned_input(user_input)

            if not flattened.get(CONF_MOTION_SENSORS):
                errors["base"] = "motion_required"

            prob_true = flattened.get(
                CONF_MOTION_PROB_GIVEN_TRUE, DEFAULT_MOTION_PROB_GIVEN_TRUE
            )
            prob_false = flattened.get(
                CONF_MOTION_PROB_GIVEN_FALSE, DEFAULT_MOTION_PROB_GIVEN_FALSE
            )
            if prob_true <= prob_false:
                errors["base"] = "prob_true_must_exceed_false"

            if not errors:
                self._area_config_draft.update(flattened)
                return await self.async_step_area_sensors()

        schema_dict = _create_motion_step_schema(
            self.hass, show_advanced=self.show_advanced_options
        )
        base_schema = vol.Schema(schema_dict)

        # Suggested values
        motion_keys = {
            CONF_MOTION_SENSORS,
            CONF_WEIGHT_MOTION,
            CONF_MOTION_TIMEOUT,
            CONF_MOTION_PROB_GIVEN_TRUE,
            CONF_MOTION_PROB_GIVEN_FALSE,
        }
        if user_input is not None:
            data_schema = self.add_suggested_values_to_schema(base_schema, user_input)
        else:
            suggested = _draft_to_suggested(self._area_config_draft, motion_keys)
            if suggested:
                data_schema = self.add_suggested_values_to_schema(
                    base_schema, suggested
                )
            else:
                data_schema = base_schema

        return self.async_show_form(
            step_id="area_motion",
            data_schema=data_schema,
            errors=errors,
            last_step=False,
        )

    async def async_step_area_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Wizard step 3: Additional sensor configuration (sections)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            flattened = _flatten_sectioned_input(user_input)

            # Validate sensor-state combinations
            if flattened.get(CONF_MEDIA_DEVICES, []) and not flattened.get(
                CONF_MEDIA_ACTIVE_STATES, DEFAULT_MEDIA_ACTIVE_STATES
            ):
                errors["base"] = "media_states_required"
            if flattened.get(CONF_APPLIANCES, []) and not flattened.get(
                CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_APPLIANCE_ACTIVE_STATES
            ):
                errors["base"] = "appliance_states_required"
            if flattened.get(CONF_DOOR_SENSORS, []) and not flattened.get(
                CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE
            ):
                errors["base"] = "door_state_required"
            if flattened.get(CONF_WINDOW_SENSORS, []) and not flattened.get(
                CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE
            ):
                errors["base"] = "window_state_required"
            if flattened.get(CONF_COVER_SENSORS, []) and not flattened.get(
                CONF_COVER_ACTIVE_STATES, DEFAULT_COVER_ACTIVE_STATES
            ):
                errors["base"] = "cover_states_required"

            if not errors:
                self._area_config_draft.update(flattened)
                return await self.async_step_area_behavior()

        schema_dict = _create_sensors_step_schema(self.hass)
        base_schema = vol.Schema(schema_dict)

        # For edit mode with sections, need nested suggested values
        if user_input is not None:
            data_schema = self.add_suggested_values_to_schema(base_schema, user_input)
        elif self._area_config_draft:
            nested = _nest_config_for_sections(self._area_config_draft)
            sensor_sections = {
                "windows_and_doors",
                "media",
                "appliances",
                "environmental",
                "power",
            }
            suggested = {k: v for k, v in nested.items() if k in sensor_sections}
            if suggested:
                data_schema = self.add_suggested_values_to_schema(
                    base_schema, suggested
                )
            else:
                data_schema = base_schema
        else:
            data_schema = base_schema

        return self.async_show_form(
            step_id="area_sensors",
            data_schema=data_schema,
            errors=errors,
            last_step=False,
        )

    async def async_step_area_behavior(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Wizard step 4: Behavior parameters (threshold, decay, wasp)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            flattened = _flatten_sectioned_input(user_input)
            self._area_config_draft.update(flattened)

            # Auto-set decay half-life based on purpose
            selected_purpose = self._area_config_draft.get(CONF_PURPOSE)
            _apply_purpose_based_decay_default(
                self._area_config_draft, selected_purpose
            )

            # Run full validation on the complete draft
            validation_errors = self._validate_config(
                self._area_config_draft, self.hass
            )

            if not validation_errors:
                return await self._on_area_config_complete(self._area_config_draft)

            errors.update(validation_errors)

        schema_dict = _create_behavior_step_schema(
            show_advanced=self.show_advanced_options
        )
        base_schema = vol.Schema(schema_dict)

        # Suggested values
        behavior_keys = {
            CONF_THRESHOLD,
            CONF_DECAY_ENABLED,
            CONF_DECAY_HALF_LIFE,
            CONF_MIN_PRIOR_OVERRIDE,
            CONF_WASP_ENABLED,
            CONF_WASP_MOTION_TIMEOUT,
            CONF_WASP_WEIGHT,
            CONF_WASP_MAX_DURATION,
            CONF_WASP_VERIFICATION_DELAY,
        }
        if user_input is not None:
            data_schema = self.add_suggested_values_to_schema(base_schema, user_input)
        else:
            suggested = _draft_to_suggested(self._area_config_draft, behavior_keys)
            if suggested:
                data_schema = self.add_suggested_values_to_schema(
                    base_schema, suggested
                )
            else:
                data_schema = base_schema

        return self.async_show_form(
            step_id="area_behavior",
            data_schema=data_schema,
            errors=errors,
            last_step=True,
        )


class AreaSubentryFlowHandler(ConfigSubentryFlow, BaseOccupancyFlow):
    """Handle subentry flow for adding/editing areas.

    HA provides built-in add/edit/delete UI for subentries. This handler
    implements the multi-step wizard for area configuration.
    """

    def __init__(self) -> None:
        """Initialize subentry flow."""
        super().__init__()
        self._area_being_edited: str | None = None
        self._area_to_remove: str | None = None
        self._area_config_draft: dict[str, Any] = {}
        self._is_reconfigure: bool = False

    def _get_wizard_areas(self) -> list[dict[str, Any]]:
        """Get areas list for duplicate checking from existing subentries."""
        entry = self._get_entry()
        return [
            dict(sub.data)
            for sub in entry.subentries.values()
            if sub.subentry_type == "area"
        ]

    async def _on_area_config_complete(
        self, config: dict[str, Any]
    ) -> SubentryFlowResult:
        """Handle wizard completion: create or update area subentry."""
        area_id = config.get(CONF_AREA_ID, "")
        area_reg = ar.async_get(self.hass)
        area_entry = area_reg.async_get_area(area_id)
        title = area_entry.name if area_entry else area_id

        if self._is_reconfigure:
            return self.async_update_and_abort(
                data=config,
                title=title,
                unique_id=area_id,
            )

        return self.async_create_entry(
            title=title,
            data=config,
            unique_id=area_id,
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a new area — start the multi-step wizard."""
        self._is_reconfigure = False
        self._area_being_edited = None
        self._area_config_draft = {}
        return await self.async_step_area_basics(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an existing area — pre-populate draft from subentry data."""
        self._is_reconfigure = True
        subentry = self._get_reconfigure_subentry()
        self._area_being_edited = subentry.data.get(CONF_AREA_ID)
        self._area_config_draft = dict(subentry.data)
        return await self.async_step_area_basics(user_input)


class AreaOccupancyConfigFlow(ConfigFlow, BaseOccupancyFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection.

    This class handles the initial configuration flow when the integration is first set up.
    It provides a multi-step configuration process with comprehensive validation.
    """

    VERSION = CONF_VERSION

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentry types supported by this integration."""
        return {"area": AreaSubentryFlowHandler}

    def __init__(self) -> None:
        """Initialize config flow.

        Sets up the initial empty data dictionary that will store configuration
        as it is built through the flow.
        """
        self._data: dict[str, Any] = {}
        self._areas: list[
            dict[str, Any]
        ] = []  # Store areas being configured during initial setup
        self._area_being_edited: str | None = None  # Store area ID (not name)
        self._area_to_remove: str | None = None  # Store area ID (not name)
        self._area_config_draft: dict[str, Any] = {}

    def _get_wizard_areas(self) -> list[dict[str, Any]]:
        """Get areas list for duplicate checking."""
        return self._areas

    async def _on_area_config_complete(
        self, config: dict[str, Any]
    ) -> ConfigFlowResult:
        """Handle wizard completion: update areas list and return to menu."""
        self._areas = _update_area_in_list(self._areas, config, self._area_being_edited)
        self._area_being_edited = None
        self._area_config_draft = {}
        return await self.async_step_user()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step - show area selection form or auto-start first area."""
        # Check if a config entry already exists (e.g., user clicked "Add device" button)
        # In single-instance architecture, only one config entry should exist
        # Users should use Options Flow to add more areas
        existing_entries = [
            entry
            for entry in self.hass.config_entries.async_entries(DOMAIN)
            if entry.source != "ignore"
        ]
        if existing_entries and user_input is None:
            # Config entry already exists - guide user to Options Flow
            return self.async_abort(
                reason="already_configured",
                description_placeholders={
                    "title": "Area Occupancy Detection",
                    "hint": "To add more areas, please go to Settings > Devices & Services > Integrations > Area Occupancy Detection, then click the cog icon (⚙️) to open the config menu.",
                },
            )

        # If no areas exist yet, automatically start configuring the first area
        # This provides a smoother user experience - users don't need to click "Add New Area" first
        if not self._areas and user_input is None:
            self._area_being_edited = None
            return await self.async_step_area_config()

        # Hybrid approach: Static menu for main actions if areas exist
        # "Finish Setup" maps to async_step_finish_setup
        menu_options = [CONF_ACTION_ADD_AREA]
        if self._areas:
            menu_options.append("manage_areas")
            menu_options.append("finish_setup")

        return self.async_show_menu(
            step_id="user",
            menu_options=menu_options,
        )

    async def async_step_manage_areas(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show list of areas to manage during initial setup."""
        errors: dict[str, str] = {}

        if user_input is not None:
            selected_option = user_input.get("selected_option", "")
            if selected_option.startswith(CONF_OPTION_PREFIX_AREA):
                # User selected an area - extract area ID and go to action step
                sanitized_id = selected_option.replace(CONF_OPTION_PREFIX_AREA, "", 1)
                # Find the actual area by matching sanitized IDs
                area = _find_area_by_sanitized_id(self._areas, sanitized_id)
                if area:
                    self._area_being_edited = area.get(CONF_AREA_ID)
                    return await self.async_step_area_action()
                # If we couldn't find the area, show error
                errors["base"] = "Selected area could not be found"

        return self.async_show_form(
            step_id="manage_areas",
            data_schema=_create_area_selector_schema(self._areas, hass=self.hass),
            errors=errors,
        )

    async def async_step_add_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add a new area."""
        self._area_being_edited = None
        return await self.async_step_area_config(user_input)

    async def async_step_finish_setup(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Finish setup and create the config entry."""
        errors: dict[str, str] = {}

        if not self._areas:
            errors["base"] = (
                "At least one area must be configured before finishing setup"
            )
            return await self.async_step_user()

        try:
            # Validate all areas before creating entry
            for area in self._areas:
                area_errors = self._validate_config(area, self.hass)
                if area_errors:
                    # Surface the first error found
                    errors.update(area_errors)
                    return await self.async_step_user()

            # Store in new multi-area format
            # Use a fixed title for the integration entry
            await self.async_set_unique_id(DOMAIN)
            try:
                self._abort_if_unique_id_configured()
            except AbortFlow as err:
                if err.reason == "already_configured":
                    # Guide user to use Options Flow instead
                    raise AbortFlow(
                        "already_configured",
                        description_placeholders={
                            "title": "Area Occupancy Detection",
                            "hint": "To add more areas, please use the Options Flow from Settings > Devices & Services > Area Occupancy Detection > Configure.",
                        },
                    ) from err
                raise

            # Build subentries from collected areas
            area_reg = ar.async_get(self.hass)
            subentries = []
            for area_config in self._areas:
                area_id = area_config.get(CONF_AREA_ID, "")
                area_entry = area_reg.async_get_area(area_id)
                title = area_entry.name if area_entry else area_id
                subentries.append(
                    {
                        "subentry_type": "area",
                        "data": area_config,
                        "title": title,
                        "unique_id": area_id,
                    }
                )

            # Global config (no CONF_AREAS — areas live in subentries)
            config_data: dict[str, Any] = {}
            return self.async_create_entry(
                title="Area Occupancy Detection",
                data=config_data,
                subentries=subentries,
            )
        except AbortFlow:
            raise
        except HomeAssistantError as err:
            _LOGGER.error("Validation error: %s", err)
            errors["base"] = str(err)
        except vol.Invalid as err:
            _LOGGER.error("Validation error: %s", err)
            errors["base"] = str(err)
        except Exception:
            _LOGGER.exception("Unexpected error creating entry")
            errors["base"] = "unknown"

        return await self.async_step_user()

    async def async_step_area_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Initialize the area config wizard."""
        self._init_area_wizard()
        return await self.async_step_area_basics()

    async def async_step_area_action(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle action selection for a specific area."""
        # Get area_id from instance variable
        area_id = self._area_being_edited
        if not area_id:
            return await self.async_step_user()

        # Find the area being managed
        area_config = _find_area_by_id(self._areas, area_id)

        if not area_config:
            return await self.async_step_user()

        if user_input is not None:
            action = user_input.get("action", "")
            self._route_area_action(action, area_id)
            if action == CONF_ACTION_EDIT:
                # User wants to edit the area
                return await self.async_step_area_config()
            if action == CONF_ACTION_REMOVE:
                # User wants to remove the area
                return await self.async_step_remove_area()
            if action == CONF_ACTION_CANCEL:
                # User cancelled
                return await self.async_step_user()

        # Show action selection form with area details
        schema = _create_action_selection_schema()
        description_placeholders = _build_area_description_placeholders(
            area_config, area_id, self.hass
        )

        return self.async_show_form(
            step_id="area_action",
            data_schema=schema,
            description_placeholders=description_placeholders,
        )

    async def async_step_remove_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm removal of an area during initial setup."""
        # Get area_id from instance variable
        area_id = self._area_to_remove
        if not area_id:
            return await self.async_step_user()

        # Resolve area name for display
        area_name = area_id
        with contextlib.suppress(ValueError):
            area_name = _resolve_area_id_to_name(self.hass, area_id)

        # Find area config for sensor counts
        area_config = _find_area_by_id(self._areas, area_id) or {}
        sensor_count = _count_area_sensors(area_config)
        # Base: 1 binary_sensor (occupancy) + 8 sensors + 1 number (threshold) = 10
        entity_count = 10
        if area_config.get(CONF_WASP_ENABLED):
            entity_count += 1

        if user_input is not None:
            if user_input.get("confirm"):
                # Remove the area
                updated_areas = _remove_area_from_list(self._areas, area_id)

                if not updated_areas:
                    return self.async_show_form(
                        step_id="remove_area",
                        data_schema=vol.Schema({}),
                        errors={"base": "Cannot remove the last area"},
                    )

                self._areas = updated_areas
                self._area_to_remove = None

                # Return to user step to show updated menu
                return await self.async_step_user()

            # User cancelled
            self._area_to_remove = None
            return await self.async_step_user()

        # Show confirmation
        schema = vol.Schema(
            {
                vol.Required("confirm", default=False): BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="remove_area",
            data_schema=schema,
            description_placeholders={
                "area_name": area_name,
                "sensor_count": str(sensor_count),
                "entity_count": str(entity_count),
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> AreaOccupancyOptionsFlow:
        """Get the options flow."""
        return AreaOccupancyOptionsFlow()


class AreaOccupancyOptionsFlow(OptionsFlow, BaseOccupancyFlow):
    """Handle options flow for global settings and people management.

    Area management is handled via AreaSubentryFlowHandler — HA provides
    built-in add/edit/delete UI for subentries on the integration page.
    """

    def __init__(self) -> None:
        """Initialize options flow."""
        super().__init__()
        self._person_being_edited: int | None = None  # Index into people list
        self._person_to_remove: int | None = None  # Index into people list for removal

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show options menu (global settings and people management).

        Area management is handled via subentry flows — HA provides built-in
        add/edit/delete UI for subentries on the integration page.
        """
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                CONF_ACTION_GLOBAL_SETTINGS,
                CONF_ACTION_MANAGE_PEOPLE,
            ],
        )

    async def async_step_global_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage global settings."""
        if user_input is not None:
            # Update the config entry options directly
            new_options = dict(self.config_entry.options)
            new_options.update(user_input)

            return self.async_create_entry(title="", data=new_options)

        # Get current values
        defaults = {
            CONF_SLEEP_START: self.config_entry.options.get(
                CONF_SLEEP_START, DEFAULT_SLEEP_START
            ),
            CONF_SLEEP_END: self.config_entry.options.get(
                CONF_SLEEP_END, DEFAULT_SLEEP_END
            ),
        }

        return self.async_show_form(
            step_id="global_settings",
            data_schema=_create_global_settings_schema(defaults),
        )

    async def async_step_manage_people(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage configured people for sleep tracking."""
        errors: dict[str, str] = {}
        people: list[dict[str, Any]] = list(
            self.config_entry.options.get(CONF_PEOPLE, [])
        )

        if user_input is not None:
            selected = user_input.get("selected_option", "")
            if selected == "add_person":
                self._person_being_edited = None
                return await self.async_step_person_config()
            if selected.startswith("edit_"):
                try:
                    idx = int(selected.replace("edit_", ""))
                except (ValueError, TypeError):
                    idx = -1
                if 0 <= idx < len(people):
                    self._person_being_edited = idx
                    return await self.async_step_person_config()
                errors["base"] = "invalid_selection"
            elif selected.startswith("remove_"):
                try:
                    idx = int(selected.replace("remove_", ""))
                except (ValueError, TypeError):
                    idx = -1
                if 0 <= idx < len(people):
                    self._person_to_remove = idx
                    return await self.async_step_remove_person()
                errors["base"] = "invalid_selection"

        # Build options list
        options: list[SelectOptionDict] = []
        for i, person in enumerate(people):
            person_entity = person.get(CONF_PERSON_ENTITY, "unknown")
            sleep_area = person.get(CONF_PERSON_SLEEP_AREA, "unknown")

            # Resolve names for display
            area_name = sleep_area
            with contextlib.suppress(ValueError):
                area_name = _resolve_area_id_to_name(self.hass, sleep_area)

            # Get person friendly name from state
            person_state = self.hass.states.get(person_entity)
            person_name = (
                person_state.attributes.get("friendly_name", person_entity)
                if person_state
                else person_entity
            )

            threshold = person.get(
                CONF_PERSON_CONFIDENCE_THRESHOLD, DEFAULT_SLEEP_CONFIDENCE_THRESHOLD
            )
            options.append(
                {
                    "value": f"edit_{i}",
                    "label": f"{person_name} → {area_name} (threshold: {threshold}%)",
                }
            )
            options.append(
                {
                    "value": f"remove_{i}",
                    "label": f"Remove {person_name}",
                }
            )

        options.append({"value": "add_person", "label": "Add Person"})

        schema = vol.Schema(
            {
                vol.Required("selected_option"): SelectSelector(
                    SelectSelectorConfig(
                        options=options,
                        mode=SelectSelectorMode.LIST,
                    )
                )
            }
        )

        return self.async_show_form(
            step_id="manage_people",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_remove_person(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm removal of a person."""
        idx = self._person_to_remove
        people: list[dict[str, Any]] = list(
            self.config_entry.options.get(CONF_PEOPLE, [])
        )
        if idx is None or not (0 <= idx < len(people)):
            return await self.async_step_init()

        person = people[idx]
        person_entity = person.get(CONF_PERSON_ENTITY, "unknown")

        # Get person friendly name from state
        person_state = self.hass.states.get(person_entity)
        person_name = (
            person_state.attributes.get("friendly_name", person_entity)
            if person_state
            else person_entity
        )

        if user_input is not None:
            if user_input.get("confirm"):
                updated_people = [p for i, p in enumerate(people) if i != idx]
                config_data = dict(self.config_entry.options)
                config_data[CONF_PEOPLE] = updated_people
                result = self.async_create_entry(title="", data=config_data)
                self.hass.async_create_task(
                    self.hass.config_entries.async_reload(self.config_entry.entry_id)
                )
                self._person_to_remove = None
                return result

            # User cancelled
            self._person_to_remove = None
            return await self.async_step_init()

        schema = vol.Schema(
            {
                vol.Required("confirm", default=False): BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="remove_person",
            data_schema=schema,
            description_placeholders={"person_name": person_name},
        )

    async def async_step_person_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure a person for sleep tracking."""
        errors: dict[str, str] = {}
        people: list[dict[str, Any]] = list(
            self.config_entry.options.get(CONF_PEOPLE, [])
        )

        # Get defaults for editing
        defaults: dict[str, Any] = {}
        idx = getattr(self, "_person_being_edited", None)
        if idx is not None and 0 <= idx < len(people):
            defaults = dict(people[idx])

        if user_input is not None:
            try:
                person_data = _validate_person_input(user_input)

                # Update or add person
                updated_people = list(people)
                if idx is not None and 0 <= idx < len(updated_people):
                    updated_people[idx] = person_data
                else:
                    updated_people.append(person_data)

                config_data = dict(self.config_entry.options)
                config_data[CONF_PEOPLE] = updated_people
                result = self.async_create_entry(title="", data=config_data)
                # Trigger integration reload to update sleep presence sensors
                self.hass.async_create_task(
                    self.hass.config_entries.async_reload(self.config_entry.entry_id)
                )

            except (vol.Invalid, ValueError, TypeError) as err:
                errors["base"] = _handle_step_error(err)
            else:
                return result

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_PERSON_ENTITY,
                    default=defaults.get(CONF_PERSON_ENTITY, ""),
                ): EntitySelector(EntitySelectorConfig(domain="person")),
                vol.Required(
                    CONF_PERSON_SLEEP_SENSOR,
                    default=defaults.get(CONF_PERSON_SLEEP_SENSOR, ""),
                ): EntitySelector(EntitySelectorConfig(domain="sensor")),
                vol.Required(
                    CONF_PERSON_SLEEP_AREA,
                    default=defaults.get(CONF_PERSON_SLEEP_AREA, ""),
                ): AreaSelector(AreaSelectorConfig()),
                vol.Optional(
                    CONF_PERSON_CONFIDENCE_THRESHOLD,
                    default=defaults.get(
                        CONF_PERSON_CONFIDENCE_THRESHOLD,
                        DEFAULT_SLEEP_CONFIDENCE_THRESHOLD,
                    ),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=1,
                        max=100,
                        step=5,
                        mode=NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Optional(
                    CONF_PERSON_DEVICE_TRACKER,
                    description={
                        "suggested_value": defaults.get(CONF_PERSON_DEVICE_TRACKER, "")
                    },
                ): EntitySelector(EntitySelectorConfig(domain="device_tracker")),
            }
        )

        return self.async_show_form(
            step_id="person_config",
            data_schema=schema,
            errors=errors,
        )
