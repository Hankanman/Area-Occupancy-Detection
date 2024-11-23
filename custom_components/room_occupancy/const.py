"""Constants for the Room Occupancy Detection integration."""
from typing import Final

DOMAIN: Final = "room_occupancy"

# Configuration constants
CONF_MOTION_SENSORS = "motion_sensors"
CONF_ILLUMINANCE_SENSORS = "illuminance_sensors"
CONF_HUMIDITY_SENSORS = "humidity_sensors"
CONF_TEMPERATURE_SENSORS = "temperature_sensors"
CONF_DEVICE_STATES = "device_states"
CONF_THRESHOLD = "threshold"
CONF_HISTORY_PERIOD = "history_period"
CONF_DECAY_ENABLED = "decay_enabled"
CONF_DECAY_WINDOW = "decay_window"
CONF_DECAY_TYPE = "decay_type"

# Default values
DEFAULT_THRESHOLD = 0.5
DEFAULT_HISTORY_PERIOD = 7  # days
DEFAULT_DECAY_ENABLED = True
DEFAULT_DECAY_WINDOW = 600  # seconds (10 minutes)
DEFAULT_DECAY_TYPE = "linear"

# Attributes
ATTR_PROBABILITY = "probability"
ATTR_PRIOR_PROBABILITY = "prior_probability"
ATTR_ACTIVE_TRIGGERS = "active_triggers"
ATTR_SENSOR_PROBABILITIES = "sensor_probabilities"
ATTR_DECAY_STATUS = "decay_status"
ATTR_CONFIDENCE_SCORE = "confidence_score"
ATTR_SENSOR_AVAILABILITY = "sensor_availability"

# Entity naming
NAME_PROBABILITY_SENSOR = "Room Occupancy Probability"
NAME_BINARY_SENSOR = "Room Occupancy Status"
