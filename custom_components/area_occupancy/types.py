"""Type definitions for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict, Any, Literal, Sequence, Set
from homeassistant.core import State

MotionState = Literal["on", "off"]

StateList = list[State]

StateSequence = Sequence[State]

EntityType = Literal[
    "motion",
    "media",
    "appliance",
    "door",
    "window",
    "light",
    "environmental",
]


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


class DeviceInfo(TypedDict):
    """Type for device information."""

    identifiers: dict[str, str]
    name: str
    manufacturer: str
    model: str
    sw_version: str


# Probability calculation types
class SensorProbability(TypedDict):
    """Type for sensor probability details."""

    probability: float
    weight: float
    weighted_probability: float


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


class SensorState(TypedDict):
    """Type for sensor state data."""

    state: str | None
    last_changed: str
    availability: bool


class SensorStates(TypedDict):
    """Type for collection of sensor states."""

    states: dict[str, SensorState]


class SensorConfig(TypedDict):
    """Type for sensor configuration."""

    prob_given_true: float
    prob_given_false: float
    default_prior: float
    weight: float


class DecayStatus(TypedDict):
    """Type for decay status."""

    global_decay: float


class StateInterval(TypedDict):
    """Type for state interval data."""

    start: datetime
    end: datetime
    state: Any


class MotionInterval(TypedDict):
    """Type for motion interval data."""

    start: datetime
    end: datetime


class StateDurations(TypedDict):
    """Type for state durations."""

    total_motion_active_time: float
    total_motion_inactive_time: float
    total_motion_time: float


class ConditionalProbability(TypedDict):
    """Type for conditional probability calculation results."""

    prob_given_true: float
    prob_given_false: float
    prior: float


class CalculationResult(TypedDict):
    """Type for single sensor calculation result."""

    weighted_prob: float
    is_active: bool
    prob_details: SensorProbability


class ProbabilityAttributes(TypedDict, total=False):
    """Type for probability sensor attributes."""

    active_triggers: list[str]
    sensor_probabilities: Set[str]
    threshold: str


class PriorsAttributes(TypedDict, total=False):
    """Type for priors sensor attributes."""

    motion_prior: float
    media_prior: float
    appliance_prior: float
    door_prior: float
    window_prior: float
    light_prior: float
    last_updated: str
    total_period: str


class LearnedPrior(TypedDict):
    """Type for learned prior data."""

    prob_given_true: float
    prob_given_false: float
    prior: float
    last_updated: str
