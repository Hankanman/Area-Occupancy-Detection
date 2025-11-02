"""Constants and types for the Area Occupancy Detection integration."""

from __future__ import annotations

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
DEVICE_SW_VERSION: Final = "2025.11.1"
CONF_VERSION: Final = 12
CONF_VERSION_MINOR: Final = 1
HA_RECORDER_DAYS: Final = 10  # days

# Configuration constants
CONF_NAME: Final = "name"
CONF_PURPOSE: Final = "purpose"
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
CONF_THRESHOLD: Final = "threshold"
CONF_DECAY_ENABLED: Final = "decay_enabled"
CONF_DECAY_HALF_LIFE: Final = "decay_half_life"
CONF_DEVICE_STATES: Final = "device_states"
CONF_AREA_ID: Final = "area_id"
CONF_MEDIA_ACTIVE_STATES: Final = "media_active_states"
CONF_SENSORS: Final = "sensors"
CONF_ENTITY_ID: Final = "entity_id"
CONF_MOTION_TIMEOUT: Final = "motion_timeout"


# Configured Weights
CONF_WEIGHT_MOTION: Final = "weight_motion"
CONF_WEIGHT_MEDIA: Final = "weight_media"
CONF_WEIGHT_APPLIANCE: Final = "weight_appliance"
CONF_WEIGHT_DOOR: Final = "weight_door"
CONF_WEIGHT_WINDOW: Final = "weight_window"
CONF_WEIGHT_ENVIRONMENTAL: Final = "weight_environmental"
CONF_WEIGHT_WASP: Final = "weight_wasp"

# Default values
DEFAULT_THRESHOLD: Final = 50.0
DEFAULT_PURPOSE: Final = "social"  # Default area purpose
DEFAULT_DECAY_ENABLED: Final = True
DEFAULT_DECAY_HALF_LIFE: Final = 120  # seconds (2 minutes)
DEFAULT_DOOR_ACTIVE_STATE: Final = STATE_CLOSED
DEFAULT_WINDOW_ACTIVE_STATE: Final = STATE_OPEN
DEFAULT_MEDIA_ACTIVE_STATES: Final[list[str]] = [STATE_PLAYING, STATE_PAUSED]
DEFAULT_APPLIANCE_ACTIVE_STATES: Final[list[str]] = [STATE_ON, STATE_STANDBY]
DEFAULT_NAME: Final = "Area Occupancy"
DEFAULT_PRIOR_UPDATE_INTERVAL: Final = 1  # hours
DEFAULT_MOTION_TIMEOUT: Final = 300  # 5 minutes in seconds

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
DEFAULT_WEIGHT_ENVIRONMENTAL: Final = 0.1

# Safety bounds
MIN_PROBABILITY: Final = 0.01
MAX_PROBABILITY: Final = 0.99
MIN_PRIOR: Final[float] = 0.1
MAX_PRIOR: Final[float] = 0.99
MIN_WEIGHT: Final[float] = 0.01
MAX_WEIGHT: Final[float] = 0.99

# Default prior probabilities
DEFAULT_PROB_GIVEN_TRUE: Final[float] = 0.5
DEFAULT_PROB_GIVEN_FALSE: Final[float] = 0.1

# Motion sensor defaults (deprecated - use DEFAULT_TYPES in entity_type.py)
# MOTION_PROB_GIVEN_TRUE: Final[float] = 0.25  # DEPRECATED
# MOTION_PROB_GIVEN_FALSE: Final[float] = 0.05  # DEPRECATED
# MOTION_DEFAULT_PRIOR: Final[float] = 0.35  # DEPRECATED

# Primary occupancy sensor defaults (optimized for ground truth reliability)
PRIMARY_PROB_GIVEN_TRUE: Final[float] = (
    0.95  # Very high confidence when area is occupied
)
PRIMARY_PROB_GIVEN_FALSE: Final[float] = 0.02  # Very low false positive rate

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

# Performance optimization constants
DEFAULT_LOOKBACK_DAYS: Final = 90  # Days of interval data to load for analysis
DEFAULT_CACHE_TTL_SECONDS: Final = 3600  # Cache TTL for occupied intervals (1 hour)
RETENTION_DAYS: Final = 365  # Days to retain interval data before pruning

# Database interval filtering
MIN_INTERVAL_SECONDS: Final = 5  # Exclude intervals shorter than 5 seconds
MAX_INTERVAL_SECONDS: Final = (
    46800  # Exclude intervals longer than 13 hours (13 * 3600)
)

# Coordinator timer intervals
DECAY_INTERVAL: Final = 10  # seconds
ANALYSIS_INTERVAL: Final = 3600  # seconds (1 hour)

# Database save debounce
SAVE_DEBOUNCE_SECONDS: Final = 5  # Delay before saving after state changes

# Master coordination constants
ANALYSIS_STAGGER_MINUTES: Final = 2  # minutes between instance analysis runs
MASTER_HEARTBEAT_INTERVAL: Final = 15  # seconds between master heartbeats
MASTER_HEALTH_TIMEOUT: Final = 30  # seconds before master considered dead
MASTER_HEALTH_CHECK_INTERVAL: Final = 10  # seconds between health checks

# Dispatcher signal names (replace event bus)
SIGNAL_STATE_SAVE_REQUEST: Final = f"{DOMAIN}_save_request"
SIGNAL_MASTER_HEARTBEAT: Final = f"{DOMAIN}_heartbeat"

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
