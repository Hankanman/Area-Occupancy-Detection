"""Constants and types for the Area Occupancy Detection integration."""

from __future__ import annotations

from typing import Final
from datetime import timedelta
from homeassistant.const import Platform

DOMAIN: Final = "area_occupancy"
PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR, Platform.NUMBER]

# Device information
DEVICE_MANUFACTURER: Final = "Hankanman"
DEVICE_MODEL: Final = "Area Occupancy Detector"
DEVICE_SW_VERSION: Final = "2024.12.2"

# Configuration constants
CONF_NAME: Final = "name"
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
CONF_VERSION: Final = 4
STORAGE_VERSION: Final = 4
STORAGE_VERSION_MINOR: Final = 1
CACHE_DURATION: Final = timedelta(hours=6)

# Default values
DEFAULT_THRESHOLD: Final = 50.0
DEFAULT_HISTORY_PERIOD: Final = 7  # days
DEFAULT_DECAY_ENABLED: Final = True
DEFAULT_DECAY_WINDOW: Final = 600  # seconds (10 minutes)
DEFAULT_HISTORICAL_ANALYSIS_ENABLED: Final = True
DEFAULT_DECAY_MIN_DELAY: Final = 60  # 1 minute

# Entity naming
NAME_PROBABILITY_SENSOR: Final = "Occupancy Probability"
NAME_BINARY_SENSOR: Final = "Occupancy Status"
NAME_PRIORS_SENSOR: Final = "Prior Probability"
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
ATTR_MOTION_PRIOR: Final = "motion_prior"
ATTR_MEDIA_PRIOR: Final = "media_prior"
ATTR_APPLIANCE_PRIOR: Final = "appliance_prior"
ATTR_DOOR_PRIOR: Final = "door_prior"
ATTR_WINDOW_PRIOR: Final = "window_prior"
ATTR_LIGHT_PRIOR: Final = "light_prior"
ATTR_OCCUPANCY_PRIOR: Final = "occupancy_prior"
