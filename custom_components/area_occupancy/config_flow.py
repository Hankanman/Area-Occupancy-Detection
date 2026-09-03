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
    OptionsFlow,
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

from .config_helpers import (
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    WEIGHT_MAX,
    WEIGHT_MIN,
    WEIGHT_STEP,
    apply_purpose_based_decay_default,
    duration_to_seconds,
    find_area_by_id,
    remove_area_from_list,
    seconds_to_duration,
    update_area_in_list,
    validate_area_config,
    validate_person_input,
)
from .const import (
    CONF_ACTION_ADD_AREA,
    CONF_ACTION_GLOBAL_SETTINGS,
    CONF_ACTION_MANAGE_PEOPLE,
    CONF_ADJACENT_AREAS,
    CONF_AIR_QUALITY_SENSORS,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_CO2_SENSORS,
    CONF_CO_SENSORS,
    CONF_COVER_ACTIVE_STATES,
    CONF_COVER_SENSORS,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_EXCLUDE_FROM_ALL_AREAS,
    CONF_HEALTH_ENABLED,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LOCK_ACTIVE_STATE,
    CONF_LOCK_SENSORS,
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
    CONF_PERSON_SLEEP_SENSORS,
    CONF_PM10_SENSORS,
    CONF_PM25_SENSORS,
    CONF_POWER_SENSORS,
    CONF_PRESSURE_SENSORS,
    CONF_PURPOSE,
    CONF_SENSOR_PRECISION,
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
    CONF_WEIGHT_LOCK,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_POWER,
    CONF_WEIGHT_WIFI_CLIENTS,
    CONF_WEIGHT_WINDOW,
    CONF_WIFI_CLIENTS_SENSORS,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_COVER_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_EXCLUDE_FROM_ALL_AREAS,
    DEFAULT_HEALTH_ENABLED,
    DEFAULT_LOCK_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_MIN_PRIOR_OVERRIDE,
    DEFAULT_MOTION_PROB_GIVEN_FALSE,
    DEFAULT_MOTION_PROB_GIVEN_TRUE,
    DEFAULT_MOTION_TIMEOUT,
    DEFAULT_PURPOSE,
    DEFAULT_SENSOR_PRECISION,
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
    DEFAULT_WEIGHT_LOCK,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_POWER,
    DEFAULT_WEIGHT_WIFI_CLIENTS,
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
    return bool(state and state.name and keyword_lower in state.name.lower())


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
    include_wifi_clients_entities = []

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
            if is_door_candidate:
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

            # Wi-Fi client-count sensors have no reliable device_class to
            # filter by, so offer every sensor-domain entity except this
            # integration's own output sensors (probability, priors, decay,
            # etc.) — selecting one of those would create a feedback loop.
            if entry.platform != DOMAIN:
                include_wifi_clients_entities.append(entry.entity_id)

    # Collect all cover entities (blinds, shades, garage doors, shutters, etc.)
    include_cover_entities = [
        entry.entity_id
        for entry in registry.entities.values()
        if entry.entity_id.startswith("cover.") and not entry.disabled
    ]

    # Collect all lock entities (smart locks, e.g. door locks). Locks don't
    # have meaningfully distinct device_class variants the way binary_sensor
    # door/window entities do, so just include the whole domain. Discover
    # via hass.states rather than the entity registry: entities without a
    # unique_id (some MQTT-configured locks, e.g.) have no registry entry
    # and would otherwise be silently excluded from selection. The registry
    # is still consulted, but only to filter out disabled entities.
    disabled_lock_entities = {
        entry.entity_id
        for entry in registry.entities.values()
        if entry.entity_id.startswith("lock.") and entry.disabled
    }
    include_lock_entities = [
        entity_id
        for entity_id in hass.states.async_entity_ids("lock")
        if entity_id not in disabled_lock_entities
    ]

    return {
        "appliance": include_appliance_entities,
        "window": include_window_entities,
        "door": include_door_entities,
        "lock": include_lock_entities,
        "cover": include_cover_entities,
        "temperature": include_temperature_entities,
        "humidity": include_humidity_entities,
        "pressure": include_pressure_entities,
        "air_quality": include_air_quality_entities,
        "pm25": include_pm25_entities,
        "pm10": include_pm10_entities,
        "motion": include_motion_entities,
        "wifi_clients": include_wifi_clients_entities,
    }


def _create_windows_and_doors_section_schema(
    defaults: dict[str, Any],
    door_entities: list[str],
    window_entities: list[str],
    cover_entities: list[str],
    lock_entities: list[str],
    door_state_options: list[SelectOptionDict],
    window_state_options: list[SelectOptionDict],
    cover_state_options: list[SelectOptionDict],
    lock_state_options: list[SelectOptionDict],
) -> vol.Schema:
    """Create schema for the combined windows, doors, locks, and covers section."""
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
                    options=door_state_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
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
                CONF_LOCK_SENSORS, default=defaults.get(CONF_LOCK_SENSORS, [])
            ): EntitySelector(
                EntitySelectorConfig(include_entities=lock_entities, multiple=True)
            ),
            vol.Optional(
                CONF_LOCK_ACTIVE_STATE,
                default=defaults.get(CONF_LOCK_ACTIVE_STATE, get_default_state("lock")),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=lock_state_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_WEIGHT_LOCK,
                default=defaults.get(CONF_WEIGHT_LOCK, DEFAULT_WEIGHT_LOCK),
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
                    options=window_state_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
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
                    custom_value=True,
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
                    custom_value=True,
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
                    custom_value=True,
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


def _create_wifi_clients_section_schema(
    defaults: dict[str, Any], wifi_clients_entities: list[str]
) -> vol.Schema:
    """Create schema for the Wi-Fi client-count section.

    Unlike power sensors, Wi-Fi client-count sensors (e.g. from the UniFi
    Network integration) have no reliable SensorDeviceClass to auto-filter
    by, so this offers every sensor-domain entity except this integration's
    own output sensors (see ``_get_include_entities``'s ``wifi_clients`` key)
    rather than a device_class-scanned include list.
    """
    return vol.Schema(
        {
            vol.Optional(
                CONF_WIFI_CLIENTS_SENSORS,
                default=defaults.get(CONF_WIFI_CLIENTS_SENSORS, []),
            ): EntitySelector(
                EntitySelectorConfig(
                    include_entities=wifi_clients_entities,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_WEIGHT_WIFI_CLIENTS,
                default=defaults.get(
                    CONF_WEIGHT_WIFI_CLIENTS, DEFAULT_WEIGHT_WIFI_CLIENTS
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


def _create_wasp_in_box_section_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Create schema for the wasp in box section."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_WASP_ENABLED, default=defaults.get(CONF_WASP_ENABLED, False)
            ): BooleanSelector(),
            vol.Optional(
                CONF_WASP_MOTION_TIMEOUT,
                default=seconds_to_duration(
                    defaults.get(CONF_WASP_MOTION_TIMEOUT, DEFAULT_WASP_MOTION_TIMEOUT)
                ),
            ): DurationSelector(DurationSelectorConfig(enable_day=False)),
            vol.Optional(
                CONF_WASP_WEIGHT,
                default=defaults.get(CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    mode=NumberSelectorMode.SLIDER,
                    unit_of_measurement="weight",
                )
            ),
            vol.Optional(
                CONF_WASP_MAX_DURATION,
                default=seconds_to_duration(
                    defaults.get(CONF_WASP_MAX_DURATION, DEFAULT_WASP_MAX_DURATION)
                ),
            ): DurationSelector(DurationSelectorConfig(enable_day=True)),
            vol.Optional(
                CONF_WASP_VERIFICATION_DELAY,
                default=seconds_to_duration(
                    defaults.get(
                        CONF_WASP_VERIFICATION_DELAY, DEFAULT_WASP_VERIFICATION_DELAY
                    )
                ),
            ): DurationSelector(DurationSelectorConfig(enable_day=False)),
        }
    )


def _create_basics_step_schema(
    *,
    is_editing: bool = False,
    adjacent_options: list[SelectOptionDict] | None = None,
) -> dict[vol.Marker, Any]:
    """Create schema for wizard step 1: area selection and purpose.

    Args:
        is_editing: True if editing an existing area (skips area selector).
        adjacent_options: Other areas in this entry available as adjacency
            choices. Each option is `{"value": area_id, "label": area_name}`.
            When None or empty, the adjacency field is omitted (e.g. the
            first area being added has no neighbours to pick from).
    """
    fields: dict[vol.Marker, Any] = {}
    if not is_editing:
        fields[vol.Required(CONF_AREA_ID)] = AreaSelector()
    fields[vol.Optional(CONF_PURPOSE, default=DEFAULT_PURPOSE)] = SelectSelector(
        SelectSelectorConfig(
            options=cast("list[SelectOptionDict]", get_purpose_options()),
            mode=SelectSelectorMode.DROPDOWN,
        )
    )
    if adjacent_options:
        fields[vol.Optional(CONF_ADJACENT_AREAS, default=[])] = SelectSelector(
            SelectSelectorConfig(
                options=adjacent_options,
                multiple=True,
                mode=SelectSelectorMode.DROPDOWN,
            )
        )
    return fields


def _create_motion_step_schema(
    hass: HomeAssistant,
    include_entities: dict[str, list[str]] | None = None,
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
                unit_of_measurement="weight",
            )
        ),
        vol.Optional(
            CONF_MOTION_TIMEOUT,
            default=seconds_to_duration(DEFAULT_MOTION_TIMEOUT),
        ): DurationSelector(DurationSelectorConfig(enable_day=False)),
        vol.Optional(
            CONF_MOTION_PROB_GIVEN_TRUE,
            default=DEFAULT_MOTION_PROB_GIVEN_TRUE,
        ): NumberSelector(
            NumberSelectorConfig(
                min=MIN_PROBABILITY,
                max=MAX_PROBABILITY,
                step=0.01,
                mode=NumberSelectorMode.BOX,
            )
        ),
        vol.Optional(
            CONF_MOTION_PROB_GIVEN_FALSE,
            default=DEFAULT_MOTION_PROB_GIVEN_FALSE,
        ): NumberSelector(
            NumberSelectorConfig(
                min=0.001,
                max=MAX_PROBABILITY,
                step=0.001,
                mode=NumberSelectorMode.BOX,
            )
        ),
    }

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
    lock_state_options = _get_state_select_options("lock")
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
                include_entities["lock"],
                cast("list[SelectOptionDict]", door_state_options),
                cast("list[SelectOptionDict]", window_state_options),
                cast("list[SelectOptionDict]", cover_state_options),
                cast("list[SelectOptionDict]", lock_state_options),
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
        vol.Required("wifi_clients"): section(
            _create_wifi_clients_section_schema(
                defaults, include_entities["wifi_clients"]
            ),
            {"collapsed": True},
        ),
    }


def _create_behavior_step_schema(
    defaults: dict[str, Any] | None = None,
) -> dict[vol.Marker, Any]:
    """Create schema for wizard step 4: thresholds, decay, and wasp-in-box."""
    defaults = defaults or {}

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
        vol.Optional(
            CONF_EXCLUDE_FROM_ALL_AREAS, default=DEFAULT_EXCLUDE_FROM_ALL_AREAS
        ): BooleanSelector(),
        vol.Optional(
            CONF_DECAY_HALF_LIFE,
            default=seconds_to_duration(DEFAULT_DECAY_HALF_LIFE),
        ): DurationSelector(DurationSelectorConfig(enable_day=False)),
        vol.Optional(
            CONF_MIN_PRIOR_OVERRIDE,
            default=DEFAULT_MIN_PRIOR_OVERRIDE,
        ): NumberSelector(
            NumberSelectorConfig(
                min=0.0,
                max=1.0,
                step=0.01,
                mode=NumberSelectorMode.SLIDER,
                unit_of_measurement="probability",
            )
        ),
        vol.Required("wasp_in_box"): section(
            _create_wasp_in_box_section_schema(defaults),
            {"collapsed": True},
        ),
    }

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
                val = seconds_to_duration(val)
            motion[key] = val
    if motion:
        nested["motion"] = motion

    # Windows and doors section
    windows_and_doors: dict[str, Any] = {}
    for key in (
        CONF_DOOR_SENSORS,
        CONF_DOOR_ACTIVE_STATE,
        CONF_WEIGHT_DOOR,
        CONF_LOCK_SENSORS,
        CONF_LOCK_ACTIVE_STATE,
        CONF_WEIGHT_LOCK,
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

    # Wi-Fi clients section
    wifi_clients: dict[str, Any] = {}
    for key in (CONF_WIFI_CLIENTS_SENSORS, CONF_WEIGHT_WIFI_CLIENTS):
        if key in flat_config:
            wifi_clients[key] = flat_config[key]
    if wifi_clients:
        nested["wifi_clients"] = wifi_clients

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
                val = seconds_to_duration(val)
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
                val = seconds_to_duration(val)
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
                val = seconds_to_duration(val)
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
        "lock_count": str(len(area_config.get(CONF_LOCK_SENSORS, []))),
        "window_count": str(len(area_config.get(CONF_WINDOW_SENSORS, []))),
        "appliance_count": str(len(area_config.get(CONF_APPLIANCES, []))),
        "threshold": str(area_config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)),
    }


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
    lock_count = len(area.get(CONF_LOCK_SENSORS, []))
    window_count = len(area.get(CONF_WINDOW_SENSORS, []))
    appliance_count = len(area.get(CONF_APPLIANCES, []))
    total_sensors = (
        motion_count
        + media_count
        + door_count
        + lock_count
        + window_count
        + appliance_count
    )

    threshold = area.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)

    return (
        f"Purpose: {purpose_name} • {total_sensors} sensors • Threshold: {threshold}%"
    )


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
            flattened_input[field] = duration_to_seconds(flattened_input[field])

    return flattened_input


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
            vol.Required(
                CONF_HEALTH_ENABLED,
                default=defaults.get(CONF_HEALTH_ENABLED, DEFAULT_HEALTH_ENABLED),
            ): BooleanSelector(),
            vol.Required(
                CONF_SENSOR_PRECISION,
                default=defaults.get(CONF_SENSOR_PRECISION, DEFAULT_SENSOR_PRECISION),
            ): vol.All(
                NumberSelector(
                    NumberSelectorConfig(
                        min=0,
                        max=2,
                        step=1,
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                vol.Coerce(int),
            ),
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
        """Validate a flat area configuration and return per-field errors.

        Every hass-free rule lives in ``config_helpers.validate_area_config``
        so other writers (entities, services) validate identically. This
        wrapper adds the one check that needs the registry: the selected
        Home Assistant area must exist.

        Args:
            data: Flat (un-sectioned) area configuration
            hass: Home Assistant instance (for validating the area id)

        Returns:
            Mapping of field key (or ``"base"``) to ``strings.json`` error key.
            Empty when validation passed.
        """
        errors = validate_area_config(data)
        area_id = data.get(CONF_AREA_ID, "")
        if area_id and hass and CONF_AREA_ID not in errors:
            try:
                _resolve_area_id_to_name(hass, area_id)
            except ValueError:
                errors[CONF_AREA_ID] = "area_not_found"
        return errors

    def _prepare_area_action_edit(self) -> None:
        """Prepare state for editing an area."""
        self._area_to_remove = None

    def _prepare_area_action_remove(self) -> None:
        """Prepare state for removing an area."""
        area_id = self._area_being_edited
        self._area_to_remove = area_id
        self._area_being_edited = None

    def _prepare_area_action_cancel(self) -> None:
        """Prepare state for cancelling area action."""
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

    def _build_adjacent_area_options(self) -> list[SelectOptionDict]:
        """Build the multi-select options for the adjacency field.

        Returns options for every area configured in this entry except the
        one currently being edited (you can't be adjacent to yourself).
        Labels resolve to the HA area registry name when possible, falling
        back to the raw area_id.
        """
        options: list[SelectOptionDict] = []
        seen: set[str] = set()
        editing_area_id = self._area_being_edited
        for area in self._get_wizard_areas():
            other_area_id = area.get(CONF_AREA_ID)
            # Defensive: malformed persisted data could leave a non-string
            # value here. Comparison still works but downstream
            # ``_resolve_area_id_to_name`` and ``SelectOptionDict`` both
            # expect str — skip rather than crash later.
            if not isinstance(other_area_id, str) or not other_area_id:
                continue
            if other_area_id == editing_area_id or other_area_id in seen:
                continue
            seen.add(other_area_id)
            label = other_area_id
            if self.hass:
                with contextlib.suppress(ValueError):
                    label = _resolve_area_id_to_name(self.hass, other_area_id)
            options.append(SelectOptionDict(value=other_area_id, label=label))
        options.sort(key=lambda opt: opt["label"].lower())
        return options

    def _init_area_wizard(self) -> None:
        """Initialize the area config wizard draft."""
        if self._area_being_edited:
            areas = self._get_wizard_areas()
            area = find_area_by_id(areas, self._area_being_edited)
            self._area_config_draft = area.copy() if area else {}
        else:
            self._area_config_draft = {}

    def _get_wizard_placeholders(self) -> dict[str, str]:
        """Get description placeholders showing the area being configured."""
        area_id = self._area_config_draft.get(CONF_AREA_ID, "")
        area_name = "New Area"
        if area_id:
            with contextlib.suppress(ValueError):
                area_name = _resolve_area_id_to_name(self.hass, area_id)
        return {"area_name": area_name}

    async def async_step_area_basics(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Wizard step 1: Area selection and purpose."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Read-only area selector doesn't submit a value when editing,
            # so fill it in from the draft
            if self._area_being_edited and CONF_AREA_ID not in user_input:
                user_input[CONF_AREA_ID] = self._area_config_draft.get(
                    CONF_AREA_ID, self._area_being_edited
                )
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

        adjacent_options = self._build_adjacent_area_options()
        schema_dict = _create_basics_step_schema(
            is_editing=self._area_being_edited is not None,
            adjacent_options=adjacent_options,
        )
        base_schema = vol.Schema(schema_dict)

        # Apply suggested values from draft or user_input (for error re-display)
        suggested = (
            user_input
            if user_input is not None
            else _draft_to_suggested(
                self._area_config_draft,
                {CONF_AREA_ID, CONF_PURPOSE, CONF_ADJACENT_AREAS},
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

        schema_dict = _create_motion_step_schema(self.hass)
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
            description_placeholders=self._get_wizard_placeholders(),
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
            if flattened.get(CONF_LOCK_SENSORS, []) and not flattened.get(
                CONF_LOCK_ACTIVE_STATE, DEFAULT_LOCK_ACTIVE_STATE
            ):
                errors["base"] = "lock_state_required"
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
                "wifi_clients",
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
            description_placeholders=self._get_wizard_placeholders(),
            last_step=False,
        )

    async def async_step_area_behavior(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Wizard step 4: Behavior parameters (threshold, decay, wasp)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            flattened = _flatten_sectioned_input(user_input)

            # Validate on a copy to avoid corrupting the draft on failure
            candidate = {**self._area_config_draft, **flattened}

            # Auto-set decay half-life based on purpose
            selected_purpose = candidate.get(CONF_PURPOSE)
            apply_purpose_based_decay_default(candidate, selected_purpose)

            # Run full validation on the complete candidate
            validation_errors = self._validate_config(candidate, self.hass)

            if not validation_errors:
                self._area_config_draft.update(candidate)
                return await self._on_area_config_complete(self._area_config_draft)

            errors.update(validation_errors)

        schema_dict = _create_behavior_step_schema(self._area_config_draft)
        base_schema = vol.Schema(schema_dict)

        # Suggested values - top-level behavior + nested wasp section
        behavior_keys = {
            CONF_THRESHOLD,
            CONF_DECAY_ENABLED,
            CONF_DECAY_HALF_LIFE,
            CONF_MIN_PRIOR_OVERRIDE,
            CONF_EXCLUDE_FROM_ALL_AREAS,
        }
        if user_input is not None:
            data_schema = self.add_suggested_values_to_schema(base_schema, user_input)
        elif self._area_config_draft:
            suggested = _draft_to_suggested(self._area_config_draft, behavior_keys)
            # Build nested wasp section suggested values
            nested = _nest_config_for_sections(self._area_config_draft)
            if "wasp_in_box" in nested:
                suggested["wasp_in_box"] = nested["wasp_in_box"]
            if suggested:
                data_schema = self.add_suggested_values_to_schema(
                    base_schema, suggested
                )
            else:
                data_schema = base_schema
        else:
            data_schema = base_schema

        return self.async_show_form(
            step_id="area_behavior",
            data_schema=data_schema,
            errors=errors,
            description_placeholders=self._get_wizard_placeholders(),
            last_step=True,
        )


class AreaOccupancyConfigFlow(ConfigFlow, BaseOccupancyFlow, domain=DOMAIN):
    """Handle a config flow for Area Occupancy Detection.

    This class handles the initial configuration flow when the integration is first set up.
    It provides a multi-step configuration process with comprehensive validation.
    """

    VERSION = CONF_VERSION

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
        self._areas = update_area_in_list(self._areas, config, self._area_being_edited)
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

            # Store areas in CONF_AREAS list
            config_data: dict[str, Any] = {CONF_AREAS: self._areas}
            return self.async_create_entry(
                title="Area Occupancy Detection",
                data=config_data,
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
        """Handle action selection for a specific area via menu."""
        area_id = self._area_being_edited
        if not area_id:
            return await self.async_step_user()

        area_config = find_area_by_id(self._areas, area_id)
        if not area_config:
            return await self.async_step_user()

        description_placeholders = _build_area_description_placeholders(
            area_config, area_id, self.hass
        )

        return self.async_show_menu(
            step_id="area_action",
            menu_options=["edit_area", "remove_area_confirm", "cancel_area_action"],
            description_placeholders=description_placeholders,
        )

    async def async_step_edit_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit the selected area."""
        self._prepare_area_action_edit()
        return await self.async_step_area_config()

    async def async_step_remove_area_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Initiate area removal."""
        self._prepare_area_action_remove()
        return await self.async_step_remove_area()

    async def async_step_cancel_area_action(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Cancel area action and return to main menu."""
        self._prepare_area_action_cancel()
        return await self.async_step_user()

    async def async_step_remove_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm removal of an area during initial setup via menu."""
        area_id = self._area_to_remove
        if not area_id:
            return await self.async_step_user()

        return self.async_show_menu(
            step_id="remove_area",
            menu_options=["confirm_remove_area", "cancel_remove_area"],
        )

    async def async_step_confirm_remove_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Execute area removal."""
        area_id = self._area_to_remove
        if not area_id:
            return await self.async_step_user()

        updated_areas = remove_area_from_list(self._areas, area_id)
        if not updated_areas:
            return self.async_abort(reason="cannot_remove_last_area")

        self._areas = updated_areas
        self._area_to_remove = None
        return await self.async_step_user()

    async def async_step_cancel_remove_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Cancel area removal."""
        self._area_to_remove = None
        return await self.async_step_user()

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> AreaOccupancyOptionsFlow:
        """Get the options flow."""
        return AreaOccupancyOptionsFlow()


class AreaOccupancyOptionsFlow(OptionsFlow, BaseOccupancyFlow):
    """Handle options flow for global settings, area management, and people management."""

    def __init__(self) -> None:
        """Initialize options flow."""
        super().__init__()
        self._area_being_edited: str | None = None
        self._area_to_remove: str | None = None
        self._area_to_reset: str | None = None  # area_id pending learning reset
        self._area_config_draft: dict[str, Any] = {}
        self._person_being_edited: int | None = None  # Index into people list
        self._person_to_remove: int | None = None  # Index into people list for removal

    def _get_areas_from_config(self) -> list[dict[str, Any]]:
        """Get areas list from merged config entry data+options."""
        merged = dict(self.config_entry.data)
        merged.update(self.config_entry.options)
        areas = merged.get(CONF_AREAS, [])
        if not isinstance(areas, list):
            _LOGGER.warning(
                "CONF_AREAS has unexpected type %s, using empty list",
                type(areas).__name__,
            )
            return []
        valid_areas: list[dict[str, Any]] = []
        for i, item in enumerate(areas):
            if isinstance(item, dict):
                valid_areas.append(item)
            else:
                _LOGGER.warning(
                    "CONF_AREAS[%d] has unexpected type %s, skipping",
                    i,
                    type(item).__name__,
                )
        return valid_areas

    def _get_wizard_areas(self) -> list[dict[str, Any]]:
        """Get areas list for duplicate checking."""
        return self._get_areas_from_config()

    async def _on_area_config_complete(
        self, config: dict[str, Any]
    ) -> ConfigFlowResult:
        """Handle wizard completion: update CONF_AREAS list in config entry.

        Stores the updated area list in entry.options via async_create_entry.
        The _async_entry_updated listener detects the structural change and
        triggers a full reload to create/destroy entity platform entries.
        """
        areas = self._get_areas_from_config()
        areas = update_area_in_list(areas, config, self._area_being_edited)

        self._area_being_edited = None
        self._area_config_draft = {}

        # Store updated areas in options; the update listener handles the reload
        config_data = dict(self.config_entry.options)
        config_data[CONF_AREAS] = areas
        return self.async_create_entry(title="", data=config_data)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show options menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                CONF_ACTION_ADD_AREA,
                "manage_areas",
                CONF_ACTION_GLOBAL_SETTINGS,
                CONF_ACTION_MANAGE_PEOPLE,
            ],
        )

    async def async_step_add_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add a new area via the wizard."""
        self._area_being_edited = None
        self._area_config_draft = {}
        self._init_area_wizard()
        return await self.async_step_area_basics()

    async def async_step_manage_areas(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show list of areas to manage."""
        errors: dict[str, str] = {}
        areas = self._get_areas_from_config()

        if user_input is not None:
            selected_option = user_input.get("selected_option", "")
            if selected_option.startswith(CONF_OPTION_PREFIX_AREA):
                sanitized_id = selected_option.replace(CONF_OPTION_PREFIX_AREA, "", 1)
                area = _find_area_by_sanitized_id(areas, sanitized_id)
                if area:
                    self._area_being_edited = area.get(CONF_AREA_ID)
                    return await self.async_step_area_action()
                errors["base"] = "Selected area could not be found"

        return self.async_show_form(
            step_id="manage_areas",
            data_schema=_create_area_selector_schema(areas, hass=self.hass),
            errors=errors,
        )

    async def async_step_area_action(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle action selection for a specific area via menu."""
        area_id = self._area_being_edited
        if not area_id:
            return await self.async_step_init()

        areas = self._get_areas_from_config()
        area_config = find_area_by_id(areas, area_id)
        if not area_config:
            return await self.async_step_init()

        description_placeholders = _build_area_description_placeholders(
            area_config, area_id, self.hass
        )

        return self.async_show_menu(
            step_id="area_action",
            menu_options=[
                "edit_area",
                "reset_learning_confirm",
                "remove_area_confirm",
                "cancel_area_action",
            ],
            description_placeholders=description_placeholders,
        )

    async def async_step_edit_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit the selected area."""
        self._prepare_area_action_edit()
        self._init_area_wizard()
        return await self.async_step_area_basics()

    async def async_step_remove_area_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Initiate area removal."""
        self._prepare_area_action_remove()
        return await self.async_step_remove_area()

    async def async_step_reset_learning_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Initiate learning reset for the area being edited."""
        self._area_to_reset = self._area_being_edited
        return await self.async_step_reset_learning()

    async def async_step_cancel_area_action(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Cancel area action and return to main menu."""
        self._prepare_area_action_cancel()
        return await self.async_step_init()

    async def async_step_remove_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm removal of an area via menu."""
        area_id = self._area_to_remove
        if not area_id:
            return await self.async_step_init()

        return self.async_show_menu(
            step_id="remove_area",
            menu_options=["confirm_remove_area", "cancel_remove_area"],
        )

    async def async_step_confirm_remove_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Execute area removal."""
        area_id = self._area_to_remove
        if not area_id:
            return await self.async_step_init()

        areas = self._get_areas_from_config()
        updated_areas = remove_area_from_list(areas, area_id)
        if not updated_areas:
            return self.async_abort(reason="cannot_remove_last_area")

        self._area_to_remove = None
        config_data = dict(self.config_entry.options)
        config_data[CONF_AREAS] = updated_areas
        return self.async_create_entry(title="", data=config_data)

    async def async_step_cancel_remove_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Cancel area removal."""
        self._area_to_remove = None
        return await self.async_step_init()

    async def async_step_reset_learning(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Yes/no menu confirming learning reset for the selected area."""
        if not self._area_to_reset:
            return await self.async_step_init()

        return self.async_show_menu(
            step_id="reset_learning",
            menu_options=["confirm_reset_learning", "cancel_reset_learning"],
        )

    async def async_step_confirm_reset_learning(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Execute the per-area learning reset.

        Wipes the area's learned history (intervals, priors, correlations,
        cache) without removing the area from the configuration. Returns to
        the area_action menu so the user can continue editing.
        """
        # Lazy import to avoid pulling service.py into the config-flow load path.
        from .service import (  # noqa: PLC0415
            _find_area_by_area_id,
            async_purge_area_data,
        )

        area_id = self._area_to_reset
        if not area_id:
            return await self.async_step_init()

        coordinator = getattr(self.config_entry, "runtime_data", None)
        if coordinator is None:
            _LOGGER.warning(
                "Reset learning aborted for area_id '%s': coordinator not loaded",
                area_id,
            )
            self._area_to_reset = None
            return self.async_abort(reason="reset_learning_failed")

        area_name, area = _find_area_by_area_id(coordinator, area_id)
        if area_name is None or area is None:
            _LOGGER.warning(
                "Reset learning aborted: no configured area found for area_id '%s'",
                area_id,
            )
            self._area_to_reset = None
            return self.async_abort(reason="reset_learning_failed")

        _LOGGER.info(
            "Resetting learned history for area '%s' (area_id=%s) via options flow",
            area_name,
            area_id,
        )

        try:
            await async_purge_area_data(self.hass, coordinator, area_name, area)
        except HomeAssistantError:
            _LOGGER.exception(
                "Reset learning failed for area '%s' (area_id=%s)", area_name, area_id
            )
            self._area_to_reset = None
            return self.async_abort(reason="reset_learning_failed")

        self._area_to_reset = None
        # Stay scoped to the same area so the user lands back in its menu.
        self._area_being_edited = area_id
        return await self.async_step_area_action()

    async def async_step_cancel_reset_learning(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Cancel the learning reset and return to the area_action menu."""
        area_id = self._area_to_reset
        self._area_to_reset = None
        if area_id:
            self._area_being_edited = area_id
            return await self.async_step_area_action()
        return await self.async_step_init()

    async def async_step_global_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage global settings."""
        if user_input is not None:
            # Validate and coerce using the schema
            schema = _create_global_settings_schema(self.config_entry.options)
            user_input = schema(user_input)

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
            CONF_HEALTH_ENABLED: self.config_entry.options.get(
                CONF_HEALTH_ENABLED, DEFAULT_HEALTH_ENABLED
            ),
            CONF_SENSOR_PRECISION: self.config_entry.options.get(
                CONF_SENSOR_PRECISION, DEFAULT_SENSOR_PRECISION
            ),
        }

        return self.async_show_form(
            step_id="global_settings",
            data_schema=_create_global_settings_schema(defaults),
        )

    def _get_person_display_name(self, person_entity: str) -> str:
        """Get friendly display name for a person entity."""
        person_state = self.hass.states.get(person_entity)
        return (
            person_state.attributes.get("friendly_name", person_entity)
            if person_state
            else person_entity
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
            if selected.startswith("person_"):
                try:
                    idx = int(selected.replace("person_", ""))
                except (ValueError, TypeError):
                    idx = -1
                if 0 <= idx < len(people):
                    self._person_being_edited = idx
                    return await self.async_step_person_action()
                errors["base"] = "invalid_selection"

        # Build options list - one entry per person
        options: list[SelectOptionDict] = []
        for i, person in enumerate(people):
            person_entity = person.get(CONF_PERSON_ENTITY, "unknown")
            sleep_area = person.get(CONF_PERSON_SLEEP_AREA, "unknown")

            area_name = sleep_area
            with contextlib.suppress(ValueError):
                area_name = _resolve_area_id_to_name(self.hass, sleep_area)

            person_name = self._get_person_display_name(person_entity)
            threshold = person.get(
                CONF_PERSON_CONFIDENCE_THRESHOLD, DEFAULT_SLEEP_CONFIDENCE_THRESHOLD
            )
            options.append(
                {
                    "value": f"person_{i}",
                    "label": f"{person_name} → {area_name} (threshold: {threshold}%)",
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

    async def async_step_person_action(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show action menu for a selected person."""
        idx = self._person_being_edited
        people: list[dict[str, Any]] = list(
            self.config_entry.options.get(CONF_PEOPLE, [])
        )
        if idx is None or not (0 <= idx < len(people)):
            return await self.async_step_init()

        person = people[idx]
        person_name = self._get_person_display_name(
            person.get(CONF_PERSON_ENTITY, "unknown")
        )

        return self.async_show_menu(
            step_id="person_action",
            menu_options=[
                "edit_person",
                "remove_person_confirm",
                "cancel_person_action",
            ],
            description_placeholders={"person_name": person_name},
        )

    async def async_step_edit_person(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit the selected person."""
        return await self.async_step_person_config()

    async def async_step_remove_person_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Initiate person removal."""
        self._person_to_remove = self._person_being_edited
        self._person_being_edited = None
        return await self.async_step_remove_person()

    async def async_step_cancel_person_action(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Cancel person action and return to main menu."""
        self._person_being_edited = None
        return await self.async_step_init()

    async def async_step_remove_person(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm removal of a person via menu."""
        idx = self._person_to_remove
        people: list[dict[str, Any]] = list(
            self.config_entry.options.get(CONF_PEOPLE, [])
        )
        if idx is None or not (0 <= idx < len(people)):
            return await self.async_step_init()

        person = people[idx]
        person_name = self._get_person_display_name(
            person.get(CONF_PERSON_ENTITY, "unknown")
        )

        return self.async_show_menu(
            step_id="remove_person",
            menu_options=["confirm_remove_person", "cancel_remove_person"],
            description_placeholders={"person_name": person_name},
        )

    async def async_step_confirm_remove_person(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Execute person removal."""
        idx = self._person_to_remove
        people: list[dict[str, Any]] = list(
            self.config_entry.options.get(CONF_PEOPLE, [])
        )
        if idx is None or not (0 <= idx < len(people)):
            return await self.async_step_init()

        updated_people = [p for i, p in enumerate(people) if i != idx]
        config_data = dict(self.config_entry.options)
        config_data[CONF_PEOPLE] = updated_people
        result = self.async_create_entry(title="", data=config_data)
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(self.config_entry.entry_id)
        )
        self._person_to_remove = None
        return result

    async def async_step_cancel_remove_person(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Cancel person removal."""
        self._person_to_remove = None
        return await self.async_step_init()

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
            # Migrate old single-sensor key to list for form population
            if (
                CONF_PERSON_SLEEP_SENSORS not in defaults
                and CONF_PERSON_SLEEP_SENSOR in defaults
            ):
                old_val = defaults.pop(CONF_PERSON_SLEEP_SENSOR)
                defaults[CONF_PERSON_SLEEP_SENSORS] = [old_val] if old_val else []

        if user_input is not None:
            # Check for duplicate person entity before validation
            new_entity = user_input.get(CONF_PERSON_ENTITY, "")
            duplicate = any(
                existing.get(CONF_PERSON_ENTITY) == new_entity
                for i, existing in enumerate(people)
                if i != idx
            )
            if duplicate:
                errors["base"] = "person_already_configured"

            if not errors:
                try:
                    person_data = validate_person_input(user_input)

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
                        self.hass.config_entries.async_reload(
                            self.config_entry.entry_id
                        )
                    )

                except (vol.Invalid, ValueError, TypeError) as err:
                    errors["base"] = _handle_step_error(err)
                else:
                    return result

        base_schema = vol.Schema(
            {
                vol.Required(CONF_PERSON_ENTITY): EntitySelector(
                    EntitySelectorConfig(domain="person")
                ),
                vol.Required(CONF_PERSON_SLEEP_SENSORS): EntitySelector(
                    EntitySelectorConfig(
                        domain=["sensor", "binary_sensor"], multiple=True
                    )
                ),
                vol.Required(CONF_PERSON_SLEEP_AREA): AreaSelector(
                    AreaSelectorConfig()
                ),
                vol.Optional(
                    CONF_PERSON_CONFIDENCE_THRESHOLD,
                    default=DEFAULT_SLEEP_CONFIDENCE_THRESHOLD,
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=1,
                        max=100,
                        step=5,
                        mode=NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Optional(CONF_PERSON_DEVICE_TRACKER): EntitySelector(
                    EntitySelectorConfig(domain="device_tracker")
                ),
            }
        )

        # Use suggested values for edit mode
        suggested = user_input if user_input is not None else defaults
        if suggested:
            data_schema = self.add_suggested_values_to_schema(base_schema, suggested)
        else:
            data_schema = base_schema

        return self.async_show_form(
            step_id="person_config",
            data_schema=data_schema,
            errors=errors,
        )
