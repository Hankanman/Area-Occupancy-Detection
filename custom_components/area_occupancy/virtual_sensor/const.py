"""Constants for the virtual sensor framework."""

from __future__ import annotations

from typing import Final

# Configuration keys
CONF_VIRTUAL_SENSOR_TYPE: Final = "type"
CONF_VIRTUAL_SENSOR_NAME: Final = "name"
CONF_VIRTUAL_SENSOR_UPDATE_INTERVAL: Final = "update_interval"
CONF_VIRTUAL_SENSOR_WEIGHT: Final = "weight"
CONF_VIRTUAL_SENSOR_ENABLED: Final = "enabled"
CONF_VIRTUAL_SENSOR_OPTIONS: Final = "options"

# Default values
DEFAULT_VIRTUAL_SENSOR_UPDATE_INTERVAL: Final = 60  # seconds
DEFAULT_VIRTUAL_SENSOR_WEIGHT: Final = 1.0
DEFAULT_VIRTUAL_SENSOR_ENABLED: Final = True

# Sensor types
VIRTUAL_SENSOR_TYPE_WASP_IN_BOX: Final = "wasp_in_box"

# States
VIRTUAL_SENSOR_STATE_UNKNOWN: Final = "unknown"
VIRTUAL_SENSOR_STATE_OCCUPIED: Final = "occupied"
VIRTUAL_SENSOR_STATE_UNOCCUPIED: Final = "unoccupied"
VIRTUAL_SENSOR_STATE_ERROR: Final = "error"

# Events
EVENT_VIRTUAL_SENSOR_STATE_CHANGED: Final = "virtual_sensor_state_changed"
EVENT_VIRTUAL_SENSOR_ERROR: Final = "virtual_sensor_error"
EVENT_VIRTUAL_SENSOR_CONFIG_UPDATED: Final = "virtual_sensor_config_updated"

# Attributes
ATTR_VIRTUAL_SENSOR_TYPE: Final = "virtual_sensor_type"
ATTR_VIRTUAL_SENSOR_WEIGHT: Final = "weight"
ATTR_VIRTUAL_SENSOR_LAST_UPDATE: Final = "last_update"
ATTR_VIRTUAL_SENSOR_ERROR: Final = "error"
ATTR_VIRTUAL_SENSOR_OPTIONS: Final = "options"

# Platform name
PLATFORM_VIRTUAL_SENSOR: Final = "virtual_sensor"

# Configuration schema keys
CONF_SCHEMA_VIRTUAL_SENSOR: Final = "virtual_sensor"
CONF_SCHEMA_VIRTUAL_SENSOR_TYPE: Final = "type"
CONF_SCHEMA_VIRTUAL_SENSOR_OPTIONS: Final = "options"

# Error messages
ERROR_INVALID_SENSOR_TYPE: Final = "invalid_sensor_type"
ERROR_INVALID_CONFIG: Final = "invalid_config"
ERROR_SENSOR_NOT_FOUND: Final = "sensor_not_found"
ERROR_UPDATE_FAILED: Final = "update_failed"

# Logging
LOG_LEVEL_DEBUG: Final = "debug"
LOG_LEVEL_INFO: Final = "info"
LOG_LEVEL_WARNING: Final = "warning"
LOG_LEVEL_ERROR: Final = "error"

# Update intervals (in seconds)
UPDATE_INTERVAL_FAST: Final = 5
UPDATE_INTERVAL_NORMAL: Final = 60
UPDATE_INTERVAL_SLOW: Final = 300

# Weight ranges
WEIGHT_MIN: Final = 0.0
WEIGHT_MAX: Final = 1.0
WEIGHT_DEFAULT: Final = 0.5
