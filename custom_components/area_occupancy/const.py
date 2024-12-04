"""Constants and types for the Area Occupancy Detection integration."""

from __future__ import annotations

from typing import Final, TypedDict, NotRequired, Literal, Any, Dict

# Type aliases
SensorId = str
SensorStates = Dict[SensorId, "SensorState"]

DOMAIN: Final = "area_occupancy"

# Configuration constants
CONF_MOTION_SENSORS: Final = "motion_sensors"
CONF_MEDIA_DEVICES: Final = "media_devices"
CONF_APPLIANCES: Final = "appliances"
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
CONF_DEVICE_STATES: Final = "device_states"
CONF_AREA_ID: Final = "area_id"

# File paths and configuration
STORAGE_KEY_HISTORY: Final = "area_occupancy_history"
STORAGE_KEY_PATTERNS: Final = "area_occupancy_patterns"
STORAGE_VERSION: Final = 1

# Default values
DEFAULT_THRESHOLD: Final = 0.5
DEFAULT_HISTORY_PERIOD: Final = 7  # days
DEFAULT_DECAY_ENABLED: Final = True
DEFAULT_DECAY_WINDOW: Final = 600  # seconds (10 minutes)
DEFAULT_DECAY_TYPE: Final = "linear"
DEFAULT_HISTORICAL_ANALYSIS_ENABLED: Final = True
DEFAULT_MINIMUM_CONFIDENCE: Final = 0.3

# Entity naming
NAME_PROBABILITY_SENSOR: Final = "Occupancy Probability"
NAME_BINARY_SENSOR: Final = "Occupancy Status"

# Attribute keys
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
ATTR_WINDOW_SIZE: Final = "window_size"
ATTR_MEDIA_STATES: Final = "media_states"
ATTR_APPLIANCE_STATES: Final = "appliance_states"
ATTR_HISTORICAL_PATTERNS: Final = "historical_patterns"
ATTR_TYPICAL_OCCUPANCY: Final = "typical_occupancy_rate"
ATTR_DAY_OCCUPANCY: Final = "day_occupancy_rate"
ATTR_SENSOR_CORRELATIONS: Final = "sensor_correlations"


# Type definitions
class SensorState(TypedDict):
    """Type for sensor state data."""

    state: str | float | None
    last_changed: str
    availability: bool


class EnvironmentalData(TypedDict):
    """Type for environmental sensor data."""

    current_value: float
    baseline: float
    threshold: float
    weight: float


class DeviceState(TypedDict):
    """Type for device state data."""

    entity_id: str
    state: str
    timestamp: str
    confidence: float


class TimeSlotStats(TypedDict):
    """Type for time slot statistics."""

    occupied_ratio: float
    samples: int
    confidence: float


class HistoricalPatterns(TypedDict):
    """Type for historical pattern data."""

    time_slots: dict[str, TimeSlotStats]
    day_patterns: dict[str, TimeSlotStats]
    sensor_correlations: dict[str, dict[str, float]]
    typical_occupancy_rate: float
    day_occupancy_rate: float


class SensorProbabilities(TypedDict):
    """Type for sensor probability data."""

    motion_probability: float
    media_probability: float
    appliance_probability: float
    environmental_probability: float


class ProbabilityResult(TypedDict):
    """Type for probability calculation results."""

    probability: float
    prior_probability: float
    active_triggers: list[str]
    sensor_probabilities: SensorProbabilities
    device_states: dict[str, dict[str, str]]
    decay_status: dict[str, float]
    confidence_score: float
    sensor_availability: dict[str, bool]
    last_occupied: NotRequired[str]
    state_duration: NotRequired[float]
    occupancy_rate: NotRequired[float]
    moving_average: NotRequired[float]
    rate_of_change: NotRequired[float]
    min_probability: NotRequired[float]
    max_probability: NotRequired[float]
    historical_patterns: NotRequired[HistoricalPatterns]


class AreaConfig(TypedDict):
    """Type for area configuration data."""

    name: str
    motion_sensors: list[str]
    media_devices: NotRequired[list[str]]
    appliances: NotRequired[list[str]]
    illuminance_sensors: NotRequired[list[str]]
    humidity_sensors: NotRequired[list[str]]
    temperature_sensors: NotRequired[list[str]]
    threshold: NotRequired[float]
    history_period: NotRequired[int]
    decay_enabled: NotRequired[bool]
    decay_window: NotRequired[int]
    decay_type: NotRequired[Literal["linear", "exponential"]]
    historical_analysis_enabled: NotRequired[bool]
    minimum_confidence: NotRequired[float]


class StorageData(TypedDict):
    """Type for persistent storage data."""

    version: int
    last_updated: str
    areas: dict[str, dict[str, Any]]
