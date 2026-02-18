"""Constants and types for the Area Occupancy Detection integration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Final, TypedDict

from homeassistant.const import (
    STATE_BUFFERING,
    STATE_CLOSED,
    STATE_CLOSING,
    STATE_IDLE,
    STATE_OFF,
    STATE_ON,
    STATE_OPEN,
    STATE_OPENING,
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STANDBY,
    Platform,
)

_LOGGER = logging.getLogger(__name__)

DOMAIN: Final = "area_occupancy"
PLATFORMS = [Platform.BINARY_SENSOR, Platform.NUMBER, Platform.SENSOR]

# Device information
DEVICE_MANUFACTURER: Final = "Hankanman"
DEVICE_MODEL: Final = "Area Occupancy Detector"
DEVICE_SW_VERSION: Final = "2026.2.1"
CONF_VERSION: Final = 16  # Incremented for timezone normalization + local bucketing
CONF_VERSION_MINOR: Final = 0
HA_RECORDER_DAYS: Final = 10  # days

# Multi-area architecture constants
CONF_AREAS: Final = "areas"  # Key for storing list of area configurations
ALL_AREAS_IDENTIFIER: Final = (
    "all_areas"  # Identifier for "All Areas" aggregation device
)

# Config flow action constants
CONF_ACTION_ADD_AREA: Final = "add_area"
CONF_ACTION_FINISH_SETUP: Final = "finish_setup"
CONF_ACTION_EDIT: Final = "edit"
CONF_ACTION_REMOVE: Final = "remove"
CONF_ACTION_CANCEL: Final = "cancel"
CONF_ACTION_GLOBAL_SETTINGS: Final = "global_settings"
CONF_ACTION_MANAGE_PEOPLE: Final = "manage_people"
CONF_OPTION_PREFIX_AREA: Final = "area_"

# Configuration constants
# CONF_NAME removed - use CONF_AREA_ID instead
CONF_AREA_ID: Final = "area_id"
CONF_PURPOSE: Final = "purpose"
CONF_MOTION_SENSORS: Final = "motion_sensors"
CONF_MOTION_PROB_GIVEN_TRUE: Final = "motion_prob_given_true"
CONF_MOTION_PROB_GIVEN_FALSE: Final = "motion_prob_given_false"
CONF_MEDIA_DEVICES: Final = "media_devices"
CONF_APPLIANCES: Final = "appliances"
CONF_ILLUMINANCE_SENSORS: Final = "illuminance_sensors"
CONF_HUMIDITY_SENSORS: Final = "humidity_sensors"
CONF_TEMPERATURE_SENSORS: Final = "temperature_sensors"
CONF_CO2_SENSORS: Final = "co2_sensors"
CONF_CO_SENSORS: Final = "co_sensors"
CONF_SOUND_PRESSURE_SENSORS: Final = "sound_pressure_sensors"
CONF_PRESSURE_SENSORS: Final = "pressure_sensors"
CONF_AIR_QUALITY_SENSORS: Final = "air_quality_sensors"
CONF_VOC_SENSORS: Final = "voc_sensors"
CONF_PM25_SENSORS: Final = "pm25_sensors"
CONF_PM10_SENSORS: Final = "pm10_sensors"
CONF_POWER_SENSORS: Final = "power_sensors"
CONF_DOOR_SENSORS: Final = "door_sensors"
CONF_DOOR_ACTIVE_STATE: Final = "door_active_state"
CONF_WINDOW_SENSORS: Final = "window_sensors"
CONF_WINDOW_ACTIVE_STATE: Final = "window_active_state"
CONF_COVER_SENSORS: Final = "cover_sensors"
CONF_COVER_ACTIVE_STATES: Final = "cover_active_states"
CONF_APPLIANCE_ACTIVE_STATES: Final = "appliance_active_states"
CONF_THRESHOLD: Final = "threshold"
CONF_DECAY_ENABLED: Final = "decay_enabled"
CONF_DECAY_HALF_LIFE: Final = "decay_half_life"
CONF_DEVICE_STATES: Final = "device_states"
CONF_MEDIA_ACTIVE_STATES: Final = "media_active_states"
CONF_SENSORS: Final = "sensors"
CONF_ENTITY_ID: Final = "entity_id"
CONF_MOTION_TIMEOUT: Final = "motion_timeout"
CONF_MIN_PRIOR_OVERRIDE: Final = "min_prior_override"
CONF_SLEEP_START: Final = "sleep_start"
CONF_SLEEP_END: Final = "sleep_end"

# People configuration constants
CONF_PEOPLE: Final = "people"
CONF_PERSON_ENTITY: Final = "person_entity"
CONF_PERSON_SLEEP_SENSOR: Final = "sleep_confidence_sensor"
CONF_PERSON_SLEEP_AREA: Final = "sleep_area_id"
CONF_PERSON_CONFIDENCE_THRESHOLD: Final = "confidence_threshold"
CONF_PERSON_DEVICE_TRACKER: Final = "device_tracker"


# Configured Weights
CONF_WEIGHT_SLEEP: Final = "weight_sleep"
CONF_WEIGHT_MOTION: Final = "weight_motion"
CONF_WEIGHT_MEDIA: Final = "weight_media"
CONF_WEIGHT_APPLIANCE: Final = "weight_appliance"
CONF_WEIGHT_DOOR: Final = "weight_door"
CONF_WEIGHT_WINDOW: Final = "weight_window"
CONF_WEIGHT_COVER: Final = "weight_cover"
CONF_WEIGHT_ENVIRONMENTAL: Final = "weight_environmental"
CONF_WEIGHT_POWER: Final = "weight_power"
CONF_WEIGHT_WASP: Final = "weight_wasp"

# Default values
DEFAULT_THRESHOLD: Final = 50.0
DEFAULT_PURPOSE: Final = "social"  # Default area purpose
DEFAULT_DECAY_ENABLED: Final = True
DEFAULT_DECAY_HALF_LIFE: Final = 0  # 0 means "use purpose value"
DEFAULT_DOOR_ACTIVE_STATE: Final = STATE_CLOSED
DEFAULT_WINDOW_ACTIVE_STATE: Final = STATE_OPEN
DEFAULT_MEDIA_ACTIVE_STATES: Final[list[str]] = [STATE_PLAYING, STATE_PAUSED]
DEFAULT_APPLIANCE_ACTIVE_STATES: Final[list[str]] = [STATE_ON, STATE_STANDBY]
DEFAULT_COVER_ACTIVE_STATES: Final[list[str]] = [STATE_OPENING, STATE_CLOSING]
DEFAULT_NAME: Final = "Area Occupancy"
DEFAULT_PRIOR_UPDATE_INTERVAL: Final = 1  # hours
DEFAULT_MOTION_TIMEOUT: Final = 300  # 5 minutes in seconds
DEFAULT_MOTION_PROB_GIVEN_TRUE: Final = 0.95  # Matches DEFAULT_TYPES[InputType.MOTION]
DEFAULT_MOTION_PROB_GIVEN_FALSE: Final = (
    0.005  # Matches DEFAULT_TYPES[InputType.MOTION]
)
DEFAULT_MIN_PRIOR_OVERRIDE: Final = 0.0  # 0.0 = disabled by default
DEFAULT_SLEEP_START: Final = "23:00:00"
DEFAULT_SLEEP_END: Final = "07:00:00"
DEFAULT_SLEEP_CONFIDENCE_THRESHOLD: Final = 50
DEFAULT_SLEEP_WEIGHT: Final = 0.9
SLEEP_PRESENCE_HALF_LIFE: Final = (
    7200  # 2 hour half-life for sleep (persistent presence)
)

# Database recovery defaults
DEFAULT_ENABLE_AUTO_RECOVERY: Final = True
DEFAULT_MAX_RECOVERY_ATTEMPTS: Final = 3
DEFAULT_ENABLE_PERIODIC_BACKUPS: Final = True
DEFAULT_BACKUP_INTERVAL_HOURS: Final = 24

# Default weights
DEFAULT_WEIGHT_MOTION: Final = 1.0  # Full weight for ground truth sensors
DEFAULT_WEIGHT_MEDIA: Final = 0.7
DEFAULT_WEIGHT_APPLIANCE: Final = 0.4
DEFAULT_WEIGHT_DOOR: Final = 0.3
DEFAULT_WEIGHT_WINDOW: Final = 0.2
DEFAULT_WEIGHT_COVER: Final = (
    0.5  # Covers (blinds/shades) being operated is strong activity signal
)
DEFAULT_WEIGHT_ENVIRONMENTAL: Final = 0.1
DEFAULT_WEIGHT_POWER: Final = 0.3

# Safety bounds
MIN_PROBABILITY: Final = 0.01
MAX_PROBABILITY: Final = 0.99
MIN_PRIOR: Final[float] = 0.01
MAX_PRIOR: Final[float] = 0.99
MIN_WEIGHT: Final[float] = 0.01
MAX_WEIGHT: Final[float] = 0.99

# Time Prior Bounds
TIME_PRIOR_MIN_BOUND: Final[float] = 0.03
TIME_PRIOR_MAX_BOUND: Final[float] = 0.9

# Default prior probabilities
DEFAULT_PROB_GIVEN_TRUE: Final[float] = 0.5
DEFAULT_PROB_GIVEN_FALSE: Final[float] = 0.1
DEFAULT_TIME_PRIOR: Final[float] = 0.5

# Door sensor defaults
DOOR_PROB_GIVEN_TRUE: Final[float] = 0.2
DOOR_PROB_GIVEN_FALSE: Final[float] = 0.02
DOOR_DEFAULT_PRIOR: Final[float] = 0.1356

# Window sensor defaults
WINDOW_PROB_GIVEN_TRUE: Final[float] = 0.2
WINDOW_PROB_GIVEN_FALSE: Final[float] = 0.02
WINDOW_DEFAULT_PRIOR: Final[float] = 0.1569

# Media device defaults
MEDIA_PROB_GIVEN_TRUE: Final[float] = 0.25
MEDIA_PROB_GIVEN_FALSE: Final[float] = 0.02
MEDIA_DEFAULT_PRIOR: Final[float] = 0.30

# Appliance defaults
APPLIANCE_PROB_GIVEN_TRUE: Final[float] = 0.2
APPLIANCE_PROB_GIVEN_FALSE: Final[float] = 0.02
APPLIANCE_DEFAULT_PRIOR: Final[float] = 0.2356

# Cover defaults (blinds, shades, garage doors being operated)
COVER_PROB_GIVEN_TRUE: Final[float] = 0.35
COVER_PROB_GIVEN_FALSE: Final[float] = 0.02
COVER_DEFAULT_PRIOR: Final[float] = 0.25

# Environmental defaults
ENVIRONMENTAL_PROB_GIVEN_TRUE: Final[float] = 0.09
ENVIRONMENTAL_PROB_GIVEN_FALSE: Final[float] = 0.01
ENVIRONMENTAL_DEFAULT_PRIOR: Final[float] = 0.0769

# Wasp in Box defaults (High confidence when active)
WASP_PROB_GIVEN_TRUE: Final[float] = 0.95
WASP_PROB_GIVEN_FALSE: Final[float] = 0.05
WASP_DEFAULT_PRIOR: Final[float] = 0.60

# Sleep presence defaults (High confidence when active)
SLEEP_PROB_GIVEN_TRUE: Final[float] = 0.95
SLEEP_PROB_GIVEN_FALSE: Final[float] = 0.02

# Helper constants
ROUNDING_PRECISION: Final = 2

# Performance optimization constants
DEFAULT_LOOKBACK_DAYS: Final = 60  # Days of interval data to load for analysis
DEFAULT_CACHE_TTL_SECONDS: Final = 3600  # Cache TTL for occupied intervals (1 hour)
RETENTION_DAYS: Final = 365  # Days to retain interval data before pruning

# Database interval filtering
MIN_INTERVAL_SECONDS: Final = 5  # Exclude intervals shorter than 5 seconds
MAX_INTERVAL_SECONDS: Final = (
    46800  # Exclude intervals longer than 13 hours (13 * 3600)
)

# Database retention and aggregation constants
# Raw data retention (before aggregation)
RETENTION_RAW_INTERVALS_DAYS: Final = 28  # Days to keep raw intervals
RETENTION_RAW_NUMERIC_SAMPLES_DAYS: Final = 7  # Days to keep raw numeric samples

# Aggregation retention periods
RETENTION_DAILY_AGGREGATES_DAYS: Final = 28  # Days to keep daily aggregates
RETENTION_WEEKLY_AGGREGATES_DAYS: Final = 56  # Days to keep weekly aggregates
RETENTION_MONTHLY_AGGREGATES_YEARS: Final = (
    5  # Years to keep monthly aggregates (indefinite for trends)
)
RETENTION_HOURLY_NUMERIC_DAYS: Final = 30  # Days to keep hourly numeric aggregates
RETENTION_WEEKLY_NUMERIC_YEARS: Final = (
    3  # Years to keep weekly numeric aggregates for seasonal analysis
)

# ────────────────────────────────────── Database Constants ───────────────────────────

# Database filename
DB_NAME: Final = "area_occupancy.db"

# States to exclude from intervals
INVALID_STATES: Final = {"unknown", "unavailable", None, "", "NaN"}

# Database default values
DEFAULT_AREA_PRIOR: Final = 0.15
DEFAULT_ENTITY_WEIGHT: Final = 0.85
DEFAULT_ENTITY_PROB_GIVEN_TRUE: Final = 0.8
DEFAULT_ENTITY_PROB_GIVEN_FALSE: Final = 0.05

# Aggregation period types
AGGREGATION_PERIOD_HOURLY: Final = "hourly"
AGGREGATION_PERIOD_DAILY: Final = "daily"
AGGREGATION_PERIOD_WEEKLY: Final = "weekly"
AGGREGATION_PERIOD_MONTHLY: Final = "monthly"
AGGREGATION_PERIOD_YEARLY: Final = "yearly"
AGGREGATION_LEVEL_RAW: Final = "raw"

# Correlation analysis constants
# Minimum samples needed for reliable correlation
MIN_CORRELATION_SAMPLES: Final = 50
# Minimum confidence for correlation to be considered significant
CORRELATION_CONFIDENCE_THRESHOLD: Final = 0.7
# Strong correlation threshold (absolute value)
CORRELATION_STRONG_THRESHOLD: Final = 0.7
# Moderate correlation threshold (absolute value)
CORRELATION_MODERATE_THRESHOLD: Final = 0.4
# Weak correlation threshold (absolute value)
CORRELATION_WEAK_THRESHOLD: Final = 0.15

# Global prior retention
# Number of historical global prior calculations to keep
GLOBAL_PRIOR_HISTORY_COUNT: Final = 15

# Correlation retention
# Number of months of correlation history to keep per sensor
CORRELATION_MONTHS_TO_KEEP: Final = 24

# Coordinator timer intervals
DECAY_INTERVAL: Final = 10  # seconds
ANALYSIS_INTERVAL: Final = 3600  # seconds (1 hour)
SAVE_INTERVAL: Final = 600  # seconds (10 minutes) - periodic database save interval


########################################################
# Virtual sensor constants
########################################################

# Entity naming
NAME_WASP_IN_BOX: Final = "Wasp in Box"
NAME_SLEEP_PRESENCE: Final = "Sleeping"

# Configuration keys
CONF_WASP_ENABLED: Final = "wasp_enabled"
CONF_WASP_MOTION_TIMEOUT: Final = "wasp_motion_timeout"
CONF_WASP_WEIGHT: Final = "wasp_weight"
CONF_WASP_MAX_DURATION: Final = "wasp_max_duration"
CONF_WASP_VERIFICATION_DELAY: Final = "wasp_verification_delay"

# Default values
DEFAULT_WASP_MOTION_TIMEOUT: Final = 300  # 5 minutes in seconds
DEFAULT_WASP_WEIGHT: Final = 0.8
DEFAULT_WASP_MAX_DURATION: Final = 3600  # 1 hour in seconds
DEFAULT_WASP_VERIFICATION_DELAY: Final = 0  # Disabled by default (0 = no verification)

# Attributes
ATTR_DOOR_STATE: Final = "door_state"
ATTR_MOTION_STATE: Final = "motion_state"
ATTR_LAST_MOTION_TIME: Final = "last_motion_time"
ATTR_LAST_DOOR_TIME: Final = "last_door_time"
ATTR_MOTION_TIMEOUT: Final = "motion_timeout"
ATTR_WASP_MAX_DURATION: Final = "wasp_max_duration"
ATTR_LAST_OCCUPIED_TIME: Final = "last_occupied_time"
ATTR_MAX_DURATION: Final = "max_duration"
ATTR_VERIFICATION_DELAY: Final = "verification_delay"
ATTR_VERIFICATION_PENDING: Final = "verification_pending"

# Sleep Presence attributes
ATTR_SLEEP_CONFIDENCE: Final = "sleep_confidence"
ATTR_PERSON_STATE: Final = "person_state"
ATTR_SLEEP_THRESHOLD: Final = "sleep_threshold"
ATTR_PEOPLE_SLEEPING: Final = "people_sleeping"
ATTR_PEOPLE_DETAILS: Final = "people"
ATTR_PERSON_NAME: Final = "name"
ATTR_PERSON_SLEEPING: Final = "sleeping"


# ────────────────────────────────────── State Mapping ───────────────────────────


@dataclass
class StateOption:
    """Represents a state option with its value and friendly name."""

    value: str
    name: str
    icon: str | None = None


class PlatformStates(TypedDict):
    """Type for platform states configuration."""

    options: list[StateOption]
    default: str


# Door states configuration
# Includes transitional states for garage doors and motorized covers
DOOR_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_OPEN, "Open", "mdi:door-open"),
        StateOption(STATE_OPENING, "Opening", "mdi:door-open"),
        StateOption(STATE_CLOSED, "Closed", "mdi:door"),
        StateOption(STATE_CLOSING, "Closing", "mdi:door"),
    ],
    "default": STATE_CLOSED,
}

# Window states configuration
# Includes transitional states for motorized blinds/shades
WINDOW_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_OPEN, "Open", "mdi:window-open"),
        StateOption(STATE_OPENING, "Opening", "mdi:window-open"),
        StateOption(STATE_CLOSED, "Closed", "mdi:window-closed"),
        StateOption(STATE_CLOSING, "Closing", "mdi:window-closed"),
    ],
    "default": STATE_OPEN,
}

# Media player states configuration
# All states from homeassistant.components.media_player.MediaPlayerState
MEDIA_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_PLAYING, "Playing", "mdi:play"),
        StateOption(STATE_PAUSED, "Paused", "mdi:pause"),
        StateOption(STATE_BUFFERING, "Buffering", "mdi:timer-sand"),
        StateOption(STATE_IDLE, "Idle", "mdi:sleep"),
        StateOption(STATE_STANDBY, "Standby", "mdi:power-sleep"),
        StateOption(STATE_ON, "On", "mdi:power"),
        StateOption(STATE_OFF, "Off", "mdi:power-off"),
    ],
    "default": STATE_PLAYING,
}

# Appliance states configuration
APPLIANCE_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_ON, "On", "mdi:power"),
        StateOption(STATE_OFF, "Off", "mdi:power-off"),
        StateOption(STATE_STANDBY, "Standby", "mdi:power-sleep"),
    ],
    "default": STATE_ON,
}

# Cover states configuration (blinds, shades, garage doors, shutters)
# All states from homeassistant.components.cover.CoverState
COVER_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_OPENING, "Opening", "mdi:blinds-open"),
        StateOption(STATE_CLOSING, "Closing", "mdi:blinds"),
        StateOption(STATE_OPEN, "Open", "mdi:blinds-open"),
        StateOption(STATE_CLOSED, "Closed", "mdi:blinds"),
    ],
    "default": STATE_OPENING,
}

# Motion sensor states configuration
MOTION_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_ON, "Active", "mdi:motion-sensor"),
        StateOption(STATE_OFF, "Inactive", "mdi:motion-sensor-off"),
    ],
    "default": STATE_ON,
}


def get_state_options(platform_type: str) -> PlatformStates:
    """Get state options for a given platform type."""
    platform_map = {
        "door": DOOR_STATES,
        "window": WINDOW_STATES,
        "cover": COVER_STATES,
        "media": MEDIA_STATES,
        "appliance": APPLIANCE_STATES,
        "motion": MOTION_STATES,
    }
    return platform_map.get(platform_type, MOTION_STATES)


def get_friendly_state_name(platform_type: str, state: str) -> str:
    """Get the friendly name for a state in a given platform type."""
    states = get_state_options(platform_type)
    for option in states["options"]:
        if option.value == state:
            return option.name
    return state


def get_state_icon(platform_type: str, state: str) -> str | None:
    """Get the icon for a state in a given platform type."""
    states = get_state_options(platform_type)
    for option in states["options"]:
        if option.value == state:
            return option.icon
    return None


def get_default_state(platform_type: str) -> str:
    """Get the default state for a given platform type."""
    states = get_state_options(platform_type)
    return states["default"]


# Sensor type string -> InputType mapping (alphabetized).
# Lazy-loaded to avoid circular imports with data.entity_type.
_SENSOR_TYPE_MAPPING: dict[str, Any] | None = None


def get_sensor_type_mapping() -> dict[str, Any]:
    """Get sensor type mapping, building lazily to avoid circular imports."""
    global _SENSOR_TYPE_MAPPING  # noqa: PLW0603
    if _SENSOR_TYPE_MAPPING is None:
        from .data.entity_type import InputType  # noqa: PLC0415

        _SENSOR_TYPE_MAPPING = {
            "air_quality": InputType.AIR_QUALITY,
            "appliance": InputType.APPLIANCE,
            "co": InputType.CO,
            "co2": InputType.CO2,
            "cover": InputType.COVER,
            "door": InputType.DOOR,
            "humidity": InputType.HUMIDITY,
            "illuminance": InputType.ILLUMINANCE,
            "media": InputType.MEDIA,
            "motion": InputType.MOTION,
            "pm10": InputType.PM10,
            "pm25": InputType.PM25,
            "power": InputType.POWER,
            "pressure": InputType.PRESSURE,
            "sleep": InputType.SLEEP,
            "sound_pressure": InputType.SOUND_PRESSURE,
            "temperature": InputType.TEMPERATURE,
            "voc": InputType.VOC,
            "window": InputType.WINDOW,
        }
    return _SENSOR_TYPE_MAPPING
