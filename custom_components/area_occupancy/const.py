# custom_components/area_occupancy/const.py

"""Constants for Area Occupancy Detection."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "area_occupancy"

# Configuration keys
CONF_MOTION_SENSORS: Final = "motion_sensors"
CONF_MEDIA_DEVICES: Final = "media_devices"
CONF_APPLIANCES: Final = "appliances"
CONF_DEVICE_STATES: Final = "device_states"  # Legacy support
CONF_ILLUMINANCE_SENSORS: Final = "illuminance_sensors"
CONF_HUMIDITY_SENSORS: Final = "humidity_sensors"
CONF_TEMPERATURE_SENSORS: Final = "temperature_sensors"
CONF_THRESHOLD: Final = "threshold"
CONF_HISTORY_PERIOD: Final = "history_period"
CONF_DECAY_ENABLED: Final = "decay_enabled"
CONF_DECAY_WINDOW: Final = "decay_window"
CONF_DECAY_TYPE: Final = "decay_type"
CONF_HISTORICAL_ANALYSIS_ENABLED: Final = "historical_analysis_enabled"
CONF_MINIMUM_CONFIDENCE: Final = "minimum_confidence"

# Default values
DEFAULT_THRESHOLD: Final = 0.5
DEFAULT_HISTORY_PERIOD: Final = 7  # days
DEFAULT_DECAY_ENABLED: Final = True
DEFAULT_DECAY_WINDOW: Final = 600  # seconds
DEFAULT_DECAY_TYPE: Final = "linear"
DEFAULT_HISTORICAL_ANALYSIS_ENABLED: Final = True
DEFAULT_MINIMUM_CONFIDENCE: Final = 0.3

# Entity names
NAME_BINARY_SENSOR: Final = "Occupancy Status"
NAME_PROBABILITY_SENSOR: Final = "Occupancy Probability"

# Attributes
ATTR_PROBABILITY: Final = "probability"
ATTR_PRIOR_PROBABILITY: Final = "prior_probability"
ATTR_ACTIVE_TRIGGERS: Final = "active_triggers"
ATTR_SENSOR_PROBABILITIES: Final = "sensor_probabilities"
ATTR_DECAY_STATUS: Final = "decay_status"
ATTR_CONFIDENCE_SCORE: Final = "confidence_score"
ATTR_SENSOR_AVAILABILITY: Final = "sensor_availability"
ATTR_LAST_OCCUPIED: Final = "last_occupied"
ATTR_STATE_DURATION: Final = "state_duration"
ATTR_OCCUPANCY_RATE: Final = "occupancy_rate"
ATTR_MOVING_AVERAGE: Final = "moving_average"
ATTR_RATE_OF_CHANGE: Final = "rate_of_change"
ATTR_MIN_PROBABILITY: Final = "min_probability"
ATTR_MAX_PROBABILITY: Final = "max_probability"
ATTR_THRESHOLD: Final = "threshold"
ATTR_PATTERN_DATA: Final = "pattern_data"
ATTR_DEVICE_STATES: Final = "device_states"
