"""Type definitions for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict, Any, Literal, Sequence, Set, Dict, List, Optional, Union
from dataclasses import dataclass, field
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

@dataclass
class SensorCalculation:
    """Data class to hold sensor calculation results."""

    weighted_probability: float
    is_active: bool
    details: Dict[str, float]

    @classmethod
    def empty(cls) -> "SensorCalculation":
        """Create an empty sensor calculation with zero values.

        Returns:
            A SensorCalculation instance with zero values
        """
        return cls(
            weighted_probability=0.0,
            is_active=False,
            details={"probability": 0.0, "weight": 0.0, "weighted_probability": 0.0},
        )

@dataclass
class ProbabilityState:
    """State of probability calculations."""

    probability: float = field(default=0.0)
    previous_probability: float = field(default=0.0)
    threshold: float = field(default=0.5)
    prior_probability: float = field(default=0.0)
    active_triggers: List[str] = field(default_factory=list)
    sensor_probabilities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decay_status: float = field(default=0.0)
    device_states: Dict[str, Dict[str, str]] = field(default_factory=dict)
    sensor_availability: Dict[str, bool] = field(default_factory=dict)
    is_occupied: bool = field(default=False)
    decaying: bool = field(default=False)

    def __post_init__(self) -> None:
        """Validate probability values."""
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
        if not 0 <= self.previous_probability <= 1:
            raise ValueError("Previous probability must be between 0 and 1")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if not 0 <= self.prior_probability <= 1:
            raise ValueError("Prior probability must be between 0 and 1")
        if not 0 <= self.decay_status <= 1:
            raise ValueError("Decay status must be between 0 and 1")

    def update(
        self,
        probability: float | None = None,
        previous_probability: float | None = None,
        threshold: float | None = None,
        prior_probability: float | None = None,
        active_triggers: List[str] | None = None,
        sensor_probabilities: Dict[str, Dict[str, float]] | None = None,
        decay_status: float | None = None,
        device_states: Dict[str, Dict[str, str]] | None = None,
        sensor_availability: Dict[str, bool] | None = None,
        is_occupied: bool | None = None,
        decaying: bool | None = None,
    ) -> None:
        """Update the state with new values while maintaining the same instance."""
        if probability is not None:
            self.probability = max(0.0, min(probability, 1.0))
        if previous_probability is not None:
            self.previous_probability = max(0.0, min(previous_probability, 1.0))
        if threshold is not None:
            self.threshold = max(0.0, min(threshold, 1.0))
        if prior_probability is not None:
            self.prior_probability = max(0.0, min(prior_probability, 1.0))
        if active_triggers is not None:
            self.active_triggers = active_triggers
        if sensor_probabilities is not None:
            self.sensor_probabilities = sensor_probabilities
        if decay_status is not None:
            self.decay_status = max(0.0, min(decay_status, 1.0))
        if device_states is not None:
            self.device_states = device_states
        if sensor_availability is not None:
            self.sensor_availability = sensor_availability
        if is_occupied is not None:
            self.is_occupied = is_occupied
        if decaying is not None:
            self.decaying = decaying

    def to_dict(
        self,
    ) -> Dict[
        str,
        Union[
            float,
            List[str],
            Dict[str, Union[float, Dict[str, str], Dict[str, bool], bool]],
        ],
    ]:
        """Convert the dataclass to a dictionary."""
        return {
            "probability": self.probability,
            "previous_probability": self.previous_probability,
            "threshold": self.threshold,
            "prior_probability": self.prior_probability,
            "active_triggers": self.active_triggers,
            "sensor_probabilities": self.sensor_probabilities,
            "decay_status": self.decay_status,
            "device_states": self.device_states,
            "sensor_availability": self.sensor_availability,
            "is_occupied": self.is_occupied,
            "decaying": self.decaying,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[
            str,
            Union[
                float,
                List[str],
                Dict[str, Union[float, Dict[str, str], Dict[str, bool], bool]],
            ],
        ],
    ) -> ProbabilityState:
        """Create a ProbabilityState from a dictionary."""
        return cls(
            probability=float(data["probability"]),
            previous_probability=float(data["previous_probability"]),
            threshold=float(data["threshold"]),
            prior_probability=float(data["prior_probability"]),
            active_triggers=list(data["active_triggers"]),
            sensor_probabilities=dict(data["sensor_probabilities"]),
            decay_status=float(data["decay_status"]),
            device_states=dict(data["device_states"]),
            sensor_availability=dict(data["sensor_availability"]),
            is_occupied=bool(data["is_occupied"]),
            decaying=bool(data.get("decaying", False)),
        )


class SensorState(TypedDict):
    """State of a sensor."""

    state: str
    attributes: Dict[str, Union[str, float, bool]]
    last_updated: Optional[str]
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
    """Result of a probability calculation."""

    probability: float
    is_active: bool
    details: Dict[str, float]


class ProbabilityAttributes(TypedDict, total=False):
    """Type for probability sensor attributes."""

    active_triggers: list[str]
    sensor_probabilities: Set[str]
    threshold: str


class PriorsAttributes(TypedDict, total=False):
    """Type for priors sensor attributes."""

    motion_prior: str
    media_prior: str
    appliance_prior: str
    door_prior: str
    window_prior: str
    light_prior: str
    last_updated: str
    total_period: str


class LearnedPrior(TypedDict):
    """Type for learned prior data."""

    prob_given_true: float
    prob_given_false: float
    prior: float
    last_updated: str
