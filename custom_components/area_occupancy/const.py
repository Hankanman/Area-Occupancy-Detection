"""Constants and types for the Area Occupancy Detection integration."""

from __future__ import annotations

from typing import Final
from datetime import timedelta

DOMAIN: Final = "area_occupancy"

# Device information
DEVICE_MANUFACTURER: Final = "Hankanman"
DEVICE_MODEL: Final = "Area Occupancy Detector"
DEVICE_SW_VERSION: Final = "2024.12.1"

# Configuration constants
CONF_MOTION_SENSORS: Final = "motion_sensors"
CONF_MEDIA_DEVICES: Final = "media_devices"
CONF_APPLIANCES: Final = "appliances"
CONF_ILLUMINANCE_SENSORS: Final = "illuminance_sensors"
CONF_HUMIDITY_SENSORS: Final = "humidity_sensors"
CONF_TEMPERATURE_SENSORS: Final = "temperature_sensors"
CONF_DOOR_SENSORS: Final = "door_sensors"
CONF_WINDOW_SENSORS: Final = "window_sensors"
CONF_LIGHTS: Final = "lights"
CONF_THRESHOLD: Final = "threshold"
CONF_HISTORY_PERIOD: Final = "history_period"
CONF_DECAY_ENABLED: Final = "decay_enabled"
CONF_DECAY_WINDOW: Final = "decay_window"
CONF_HISTORICAL_ANALYSIS_ENABLED: Final = "historical_analysis_enabled"
CONF_DEVICE_STATES: Final = "device_states"
CONF_AREA_ID: Final = "area_id"
CONF_DECAY_MIN_DELAY: Final = "decay_min_delay"

# File paths and configuration
STORAGE_KEY_HISTORY: Final = "area_occupancy_history"
STORAGE_KEY_PATTERNS: Final = "area_occupancy_patterns"
STORAGE_VERSION: Final = 2
STORAGE_VERSION_MINOR: Final = 1
STORAGE_CLEANUP_INTERVAL: Final = timedelta(days=7)
STORAGE_MAX_CACHE_AGE: Final = timedelta(days=30)
STORAGE_SAVE_DELAY: Final = 60  # seconds

# Default values
DEFAULT_THRESHOLD: Final = 50.0
DEFAULT_HISTORY_PERIOD: Final = 7  # days
DEFAULT_DECAY_ENABLED: Final = True
DEFAULT_DECAY_WINDOW: Final = 600  # seconds (10 minutes)
DEFAULT_HISTORICAL_ANALYSIS_ENABLED: Final = True
DEFAULT_CACHE_TTL: Final = 3600  # seconds (1 hour)
DEFAULT_DECAY_MIN_DELAY: Final = 60  # 1 minute

# Entity naming
NAME_PROBABILITY_SENSOR: Final = "Occupancy Probability"
NAME_BINARY_SENSOR: Final = "Occupancy Status"
NAME_MOTION_PRIOR_SENSOR: Final = "Motion Prior"
NAME_ENVIRONMENTAL_PRIOR_SENSOR: Final = "Environmental Prior"
NAME_MEDIA_PRIOR_SENSOR: Final = "Media Prior"
NAME_APPLIANCE_PRIOR_SENSOR: Final = "Appliance Prior"
NAME_DOOR_PRIOR_SENSOR: Final = "Door Prior"
NAME_WINDOW_PRIOR_SENSOR: Final = "Window Prior"
NAME_LIGHT_PRIOR_SENSOR: Final = "Light Prior"
NAME_OCCUPANCY_PRIOR_SENSOR: Final = "Occupancy Prior"
NAME_DECAY_SENSOR = "Decay Status"
NAME_THRESHOLD_NUMBER: Final = "Occupancy Threshold"

# Attribute keys
ATTR_ACTIVE_TRIGGERS: Final = "active_triggers"
ATTR_SENSOR_PROBABILITIES: Final = "sensor_probabilities"
ATTR_PROB_GIVEN_TRUE: Final = "prob_given_true"
ATTR_PROB_GIVEN_FALSE: Final = "prob_given_false"
ATTR_LAST_UPDATED: Final = "last_updated"
ATTR_TOTAL_PERIOD: Final = "total_period"
ATTR_START_TIME: Final = "start_time"
ATTR_END_TIME: Final = "end_time"
ATTR_OUTPUT_FILE: Final = "output_file"
