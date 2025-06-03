"""Constants and types for the Area Occupancy Detection integration."""

from __future__ import annotations

from datetime import timedelta
from typing import Final

from homeassistant.const import (
    STATE_CLOSED,
    STATE_ON,
    STATE_OPEN,
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STANDBY,
    Platform,
)

DOMAIN: Final = "area_occupancy"
PLATFORMS = [Platform.BINARY_SENSOR, Platform.NUMBER, Platform.SENSOR]

# Device information
DEVICE_MANUFACTURER: Final = "Hankanman"
DEVICE_MODEL: Final = "Area Occupancy Detector"
DEVICE_SW_VERSION: Final = "2025.5.1"
CONF_VERSION: Final = 8
CONF_VERSION_MINOR: Final = 1

# Configuration constants
CONF_NAME: Final = "name"
CONF_MOTION_SENSORS: Final = "motion_sensors"
CONF_PRIMARY_OCCUPANCY_SENSOR: Final = "primary_occupancy_sensor"
CONF_MEDIA_DEVICES: Final = "media_devices"
CONF_APPLIANCES: Final = "appliances"
CONF_ILLUMINANCE_SENSORS: Final = "illuminance_sensors"
CONF_HUMIDITY_SENSORS: Final = "humidity_sensors"
CONF_TEMPERATURE_SENSORS: Final = "temperature_sensors"
CONF_DOOR_SENSORS: Final = "door_sensors"
CONF_DOOR_ACTIVE_STATE: Final = "door_active_state"
CONF_WINDOW_SENSORS: Final = "window_sensors"
CONF_WINDOW_ACTIVE_STATE: Final = "window_active_state"
CONF_APPLIANCE_ACTIVE_STATES: Final = "appliance_active_states"
CONF_LIGHTS: Final = "lights"
CONF_THRESHOLD: Final = "threshold"
CONF_HISTORY_PERIOD: Final = "history_period"
CONF_DECAY_ENABLED: Final = "decay_enabled"
CONF_DECAY_WINDOW: Final = "decay_window"
CONF_HISTORICAL_ANALYSIS_ENABLED: Final = "historical_analysis_enabled"
CONF_DEVICE_STATES: Final = "device_states"
CONF_AREA_ID: Final = "area_id"
CONF_DECAY_MIN_DELAY: Final = "decay_min_delay"
CONF_MEDIA_ACTIVE_STATES: Final = "media_active_states"

# Configured Weights
CONF_WEIGHT_MOTION: Final = "weight_motion"
CONF_WEIGHT_MEDIA: Final = "weight_media"
CONF_WEIGHT_APPLIANCE: Final = "weight_appliance"
CONF_WEIGHT_DOOR: Final = "weight_door"
CONF_WEIGHT_WINDOW: Final = "weight_window"
CONF_WEIGHT_LIGHT: Final = "weight_light"
CONF_WEIGHT_ENVIRONMENTAL: Final = "weight_environmental"
CONF_WEIGHT_WASP: Final = "weight_wasp"

# File paths and configuration
CACHE_DURATION: Final = timedelta(hours=6)

# Default values
DEFAULT_THRESHOLD: Final = 50.0
DEFAULT_HISTORY_PERIOD: Final = 7  # days
DEFAULT_DECAY_ENABLED: Final = True
DEFAULT_DECAY_WINDOW: Final = 300  # seconds (5 minutes)
DEFAULT_HISTORICAL_ANALYSIS_ENABLED: Final = True
DEFAULT_DECAY_MIN_DELAY: Final = 60  # 1 minute
DEFAULT_DOOR_ACTIVE_STATE: Final = STATE_CLOSED
DEFAULT_WINDOW_ACTIVE_STATE: Final = STATE_OPEN
DEFAULT_MEDIA_ACTIVE_STATES: Final[list[str]] = [STATE_PLAYING, STATE_PAUSED]
DEFAULT_APPLIANCE_ACTIVE_STATES: Final[list[str]] = [STATE_ON, STATE_STANDBY]

# Default weights
DEFAULT_WEIGHT_MOTION: Final = 0.85
DEFAULT_WEIGHT_MEDIA: Final = 0.7
DEFAULT_WEIGHT_APPLIANCE: Final = 0.4
DEFAULT_WEIGHT_DOOR: Final = 0.3
DEFAULT_WEIGHT_WINDOW: Final = 0.2
DEFAULT_WEIGHT_LIGHT: Final = 0.2
DEFAULT_WEIGHT_ENVIRONMENTAL: Final = 0.1

# Entity naming
NAME_PROBABILITY_SENSOR: Final = "Occupancy Probability"
NAME_BINARY_SENSOR: Final = "Occupancy Status"
NAME_PRIORS_SENSOR: Final = "Prior Probability"
NAME_DECAY_SENSOR = "Decay Status"
NAME_THRESHOLD_NUMBER: Final = "Occupancy Threshold"

# Decay lambda - A higher value results in faster decay.
# Original value (0.866) resulted in ~65% remaining at half decay_window.
# Doubled value (1.733) results in ~42% remaining at half decay_window.
DECAY_LAMBDA = 1.732867952

# Safety bounds
MIN_PROBABILITY: Final[float] = 0.01
MAX_PROBABILITY: Final[float] = 0.99
MIN_PRIOR: Final[float] = 0.0001
MAX_PRIOR: Final[float] = 0.9999
MIN_WEIGHT: Final[float] = 0.01
MAX_WEIGHT: Final[float] = 0.99

# Default prior probabilities
DEFAULT_PRIOR: Final[float] = 0.1713
DEFAULT_PROB_GIVEN_TRUE: Final[float] = 0.5
DEFAULT_PROB_GIVEN_FALSE: Final[float] = 0.1

# Motion sensor defaults
MOTION_PROB_GIVEN_TRUE: Final[float] = 0.25
MOTION_PROB_GIVEN_FALSE: Final[float] = 0.05
MOTION_DEFAULT_PRIOR: Final[float] = 0.35

# Door sensor defaults
DOOR_PROB_GIVEN_TRUE: Final[float] = 0.2
DOOR_PROB_GIVEN_FALSE: Final[float] = 0.02
DOOR_DEFAULT_PRIOR: Final[float] = 0.1356

# Window sensor defaults
WINDOW_PROB_GIVEN_TRUE: Final[float] = 0.2
WINDOW_PROB_GIVEN_FALSE: Final[float] = 0.02
WINDOW_DEFAULT_PRIOR: Final[float] = 0.1569

# Light sensor defaults
LIGHT_PROB_GIVEN_TRUE: Final[float] = 0.2
LIGHT_PROB_GIVEN_FALSE: Final[float] = 0.02
LIGHT_DEFAULT_PRIOR: Final[float] = 0.3846

# Media device defaults
MEDIA_PROB_GIVEN_TRUE: Final[float] = 0.25
MEDIA_PROB_GIVEN_FALSE: Final[float] = 0.02
MEDIA_DEFAULT_PRIOR: Final[float] = 0.30

# Appliance defaults
APPLIANCE_PROB_GIVEN_TRUE: Final[float] = 0.2
APPLIANCE_PROB_GIVEN_FALSE: Final[float] = 0.02
APPLIANCE_DEFAULT_PRIOR: Final[float] = 0.2356

# Environmental defaults
ENVIRONMENTAL_PROB_GIVEN_TRUE: Final[float] = 0.09
ENVIRONMENTAL_PROB_GIVEN_FALSE: Final[float] = 0.01
ENVIRONMENTAL_DEFAULT_PRIOR: Final[float] = 0.0769

# Wasp in Box defaults (High confidence when active)
WASP_PROB_GIVEN_TRUE: Final[float] = 0.95
WASP_PROB_GIVEN_FALSE: Final[float] = 0.05
WASP_DEFAULT_PRIOR: Final[float] = 0.60

# Helper constants
ROUNDING_PRECISION: Final = 2

# Storage constants
STORAGE_KEY: Final = f"{DOMAIN}.storage"

########################################################
# Virtual sensor constants
########################################################

# Entity naming
NAME_WASP_IN_BOX: Final = "Wasp in Box"

# Configuration keys
CONF_WASP_ENABLED: Final = "wasp_enabled"
CONF_WASP_MOTION_TIMEOUT: Final = "wasp_motion_timeout"
CONF_WASP_WEIGHT: Final = "wasp_weight"
CONF_WASP_MAX_DURATION: Final = "wasp_max_duration"

# Default values
DEFAULT_WASP_MOTION_TIMEOUT: Final = 300  # 5 minutes in seconds
DEFAULT_WASP_WEIGHT: Final = 0.8
DEFAULT_WASP_MAX_DURATION: Final = 3600  # 1 hour in seconds

# Attributes
ATTR_DOOR_STATE: Final = "door_state"
ATTR_MOTION_STATE: Final = "motion_state"
ATTR_LAST_MOTION_TIME: Final = "last_motion_time"
ATTR_LAST_DOOR_TIME: Final = "last_door_time"
ATTR_MOTION_TIMEOUT: Final = "motion_timeout"
ATTR_WASP_MAX_DURATION: Final = "wasp_max_duration"
