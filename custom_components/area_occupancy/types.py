"""Type definitions for Area Occupancy Detection."""

from __future__ import annotations

from typing import TypedDict, NotRequired, Any


# Unified configuration type
class Config(TypedDict, total=False):
    """Unified configuration for the integration."""

    # Required fields
    name: str
    area_id: str
    motion_sensors: list[str]

    # Optional fields
    media_devices: list[str]
    appliances: list[str]
    illuminance_sensors: list[str]
    humidity_sensors: list[str]
    temperature_sensors: list[str]
    door_sensors: list[str]
    window_sensors: list[str]
    lights: list[str]
    threshold: float
    history_period: int
    decay_enabled: bool
    decay_window: int
    decay_min_delay: int
    historical_analysis_enabled: bool


# Probability calculation types
class SensorProbabilities(TypedDict):
    """Type for sensor probability data."""

    motion_probability: float
    door_probability: float
    window_probability: float
    light_probability: float
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
    sensor_availability: dict[str, bool]
    is_occupied: bool
    historical_probability: NotRequired[float]
    last_occupied: NotRequired[str]
    state_duration: NotRequired[float]
    occupancy_rate: NotRequired[float]
    moving_average: NotRequired[float]
    rate_of_change: NotRequired[float]
    min_probability: NotRequired[float]
    max_probability: NotRequired[float]
    historical_patterns: NotRequired[dict[str, Any]]
