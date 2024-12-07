"""Constants and types for the Area Occupancy Detection integration."""

from __future__ import annotations
from dataclasses import dataclass

from datetime import datetime
from typing import NamedTuple, TypedDict, NotRequired, Literal, Any, Dict, List

# Type aliases
SensorId = str
SensorStates = Dict[SensorId, "SensorState"]


# Type definitions
class CoreConfig(TypedDict):
    """Core configuration that cannot be changed after setup."""

    name: str
    area_id: str
    motion_sensors: list[str]


class OptionsConfig(TypedDict, total=False):
    """Optional configuration that can be updated."""

    media_devices: list[str]
    appliances: list[str]
    illuminance_sensors: list[str]
    humidity_sensors: list[str]
    temperature_sensors: list[str]
    threshold: float
    history_period: int
    decay_enabled: bool
    decay_window: int
    decay_type: str
    historical_analysis_enabled: bool
    minimum_confidence: float


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


class TimeSlot(TypedDict):
    """Time slot statistics."""

    total_samples: int
    active_samples: int
    average_duration: float
    confidence: float
    last_updated: datetime


class DayPattern(TypedDict):
    """Day pattern statistics."""

    occupied_ratio: float
    samples: int
    confidence: float


class SensorHistory(TypedDict):
    """Historical data for a sensor."""

    total_activations: int
    average_duration: float
    time_slots: Dict[str, TimeSlot]
    day_patterns: Dict[str, DayPattern]
    correlated_sensors: Dict[str, float]


class PatternSummary(NamedTuple):
    """Summary of occupancy pattern data."""

    peak_times: list[tuple[str, float]]
    pattern_stability: float
    total_analyzed_days: int
    significant_changes: int


@dataclass
class DecayConfig:
    """Configuration for sensor decay calculations."""

    enabled: bool = True
    window: int = 600  # seconds
    type: Literal["linear", "exponential"] = "linear"


class CalculationResult(TypedDict):
    """Result from probability calculation steps."""

    probability: float
    triggers: List[str]
    result_type: str
    decay_status: dict[str, float]
    states: dict[str, str]


class EnvironmentalReading(TypedDict):
    """Environmental sensor reading with context."""

    value: float
    timestamp: datetime
    baseline: float
    noise_level: float
    gradient: float  # Rate of change
    is_significant: bool


class EnvironmentalAnalysis(NamedTuple):
    """Analysis results for environmental sensors."""

    baseline: float
    noise_level: float
    change_threshold: float
    gradient_threshold: float
    is_significant: bool


class BayesianProbability(TypedDict):
    """Bayesian probability components."""

    prob_given_true: float  # P(Evidence|Occupied)
    prob_given_false: float  # P(Evidence|Unoccupied)
    confidence: float
    last_updated: datetime


@dataclass
class SensorTransition:
    """Sensor state transition details."""

    entity_id: str
    from_state: Any
    to_state: Any
    timestamp: datetime
    duration: float  # Duration in previous state
    is_valid: bool  # Whether transition is reliable


class StoredPattern(TypedDict):
    """Pattern data for storage."""

    timestamp: datetime
    occupied: bool
    sensor_states: Dict[str, str]
    environmental_readings: Dict[str, EnvironmentalReading]
    transitions: List[SensorTransition]


class HistoricalData(TypedDict):
    """Complete historical analysis data."""

    version: int
    last_updated: datetime
    patterns: List[StoredPattern]
    day_patterns: Dict[str, DayPattern]
    probabilities: SensorProbabilities
    environmental_baselines: Dict[str, float]
    environmental_thresholds: Dict[str, float]


class SensorCorrelation(TypedDict):
    """Correlation between sensors."""

    correlation: float
    confidence: float
    last_updated: datetime
    sample_count: int
