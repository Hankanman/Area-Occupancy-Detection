"""Type definitions for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import TypedDict, NotRequired, Literal, Any, Dict, List

# Type aliases
SensorId = str
SensorStates = Dict[SensorId, "SensorState"]


# Core configuration types
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


# State and sensor types
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


# Historical analysis types
class TimeslotEntity(TypedDict):
    """Entity data within a timeslot."""

    id: str
    prob_given_true: float
    prob_given_false: float


class Timeslot(TypedDict):
    """Data structure for a single timeslot."""

    entities: list[TimeslotEntity]
    prob_given_true: float
    prob_given_false: float
    day_of_week: str
    confidence: float


class TimeslotData(TypedDict):
    """Complete timeslot data structure."""

    slots: dict[str, Timeslot]  # HH:MM format keys
    last_updated: datetime


class HistoricalResult(TypedDict):
    """Results from historical analysis."""

    prob_given_true: float
    prob_given_false: float
    confidence: float
    sample_count: int
    time_analyzed: datetime


# Probability calculation types
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
    historical_probability: NotRequired[float]
    historical_confidence: NotRequired[float]
    last_occupied: NotRequired[str]
    state_duration: NotRequired[float]
    occupancy_rate: NotRequired[float]
    moving_average: NotRequired[float]
    rate_of_change: NotRequired[float]
    min_probability: NotRequired[float]
    max_probability: NotRequired[float]
    historical_patterns: NotRequired[dict[str, Any]]


# Storage types
class StorageData(TypedDict):
    """Type for persistent storage data."""

    version: int
    last_updated: str
    areas: dict[str, dict[str, Any]]


# Configuration types
@dataclass
class DecayConfig:
    """Configuration for sensor decay calculations."""

    enabled: bool = True
    window: int = 600  # seconds
    type: Literal["linear", "exponential"] = "linear"


# Calculation result types
class CalculationResult(TypedDict):
    """Result from probability calculation steps."""

    probability: float
    triggers: List[str]
    result_type: str
    decay_status: dict[str, float]
    states: dict[str, str]


# Environmental data types
class EnvironmentalReading(TypedDict):
    """Environmental sensor reading with context."""

    value: float
    timestamp: datetime
    baseline: float
    noise_level: float
    gradient: float
    is_significant: bool


class EnvironmentalAnalysis(TypedDict):
    """Analysis results for environmental sensors."""

    baseline: float
    noise_level: float
    change_threshold: float
    gradient_threshold: float
    is_significant: bool


# Pattern and correlation types
class SensorCorrelation(TypedDict):
    """Correlation between sensors."""

    correlation: float
    confidence: float
    last_updated: datetime
    sample_count: int


class StoredPattern(TypedDict):
    """Pattern data for storage."""

    timestamp: str
    occupied: bool
    sensor_states: Dict[str, str]
    environmental_readings: Dict[str, EnvironmentalReading]
    correlations: Dict[str, SensorCorrelation]
