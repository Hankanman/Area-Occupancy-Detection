"""Type definitions for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict, Any, Literal, Sequence, Set, Dict, List, Optional, Union
from dataclasses import dataclass, field
from homeassistant.core import State
from homeassistant.util import dt as dt_util

from .const import MIN_PROBABILITY

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
    sensor_probabilities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decay_status: float = field(default=0.0)
    current_states: Dict[str, Dict[str, Union[str, bool]]] = field(default_factory=dict)
    previous_states: Dict[str, Dict[str, Union[str, bool]]] = field(
        default_factory=dict
    )
    is_occupied: bool = field(default=False)
    decaying: bool = field(default=False)

    @property
    def active_triggers(self) -> List[str]:
        """Get list of active triggers from sensor probabilities."""
        return list(self.sensor_probabilities.keys())

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
        sensor_probabilities: Dict[str, Dict[str, float]] | None = None,
        decay_status: float | None = None,
        current_states: Dict[str, Dict[str, Union[str, bool]]] | None = None,
        previous_states: Dict[str, Dict[str, Union[str, bool]]] | None = None,
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
        if sensor_probabilities is not None:
            self.sensor_probabilities = sensor_probabilities
        if decay_status is not None:
            self.decay_status = max(0.0, min(decay_status, 1.0))
        if current_states is not None:
            self.current_states = current_states
        if previous_states is not None:
            self.previous_states = previous_states
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
            Dict[str, Union[float, Dict[str, Union[str, bool]], bool]],
        ],
    ]:
        """Convert the dataclass to a dictionary."""
        return {
            "probability": self.probability,
            "previous_probability": self.previous_probability,
            "threshold": self.threshold,
            "prior_probability": self.prior_probability,
            "sensor_probabilities": self.sensor_probabilities,
            "decay_status": self.decay_status,
            "current_states": self.current_states,
            "previous_states": self.previous_states,
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
                Dict[str, Union[float, Dict[str, Union[str, bool]], bool]],
            ],
        ],
    ) -> ProbabilityState:
        """Create a ProbabilityState from a dictionary."""
        return cls(
            probability=float(data["probability"]),
            previous_probability=float(data["previous_probability"]),
            threshold=float(data["threshold"]),
            prior_probability=float(data["prior_probability"]),
            sensor_probabilities=dict(data["sensor_probabilities"]),
            decay_status=float(data["decay_status"]),
            current_states=dict(data["current_states"]),
            previous_states=dict(data.get("previous_states", {})),
            is_occupied=bool(data["is_occupied"]),
            decaying=bool(data.get("decaying", False)),
        )


@dataclass
class PriorState:
    """State of prior probability calculations."""

    # Overall prior probability for the area
    overall_prior: float = field(default=0.0)

    # Prior probabilities by sensor type
    motion_prior: float = field(default=0.0)
    media_prior: float = field(default=0.0)
    appliance_prior: float = field(default=0.0)
    door_prior: float = field(default=0.0)
    window_prior: float = field(default=0.0)
    light_prior: float = field(default=0.0)
    environmental_prior: float = field(default=0.0)

    # Individual entity priors
    entity_priors: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Last updated timestamps
    last_updated: Dict[str, str] = field(default_factory=dict)

    # Analysis period in days
    analysis_period: int = field(default=7)

    def __post_init__(self) -> None:
        """Validate prior values."""
        for attr_name in [
            "overall_prior",
            "motion_prior",
            "media_prior",
            "appliance_prior",
            "door_prior",
            "window_prior",
            "light_prior",
            "environmental_prior",
        ]:
            value = getattr(self, attr_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr_name} must be between 0 and 1")

    def update(
        self,
        overall_prior: float | None = None,
        motion_prior: float | None = None,
        media_prior: float | None = None,
        appliance_prior: float | None = None,
        door_prior: float | None = None,
        window_prior: float | None = None,
        light_prior: float | None = None,
        environmental_prior: float | None = None,
        entity_priors: Dict[str, Dict[str, float]] | None = None,
        last_updated: Dict[str, str] | None = None,
        analysis_period: int | None = None,
    ) -> None:
        """Update the state with new values while maintaining the same instance."""
        if overall_prior is not None:
            self.overall_prior = max(0.0, min(overall_prior, 1.0))
        if motion_prior is not None:
            self.motion_prior = max(0.0, min(motion_prior, 1.0))
        if media_prior is not None:
            self.media_prior = max(0.0, min(media_prior, 1.0))
        if appliance_prior is not None:
            self.appliance_prior = max(0.0, min(appliance_prior, 1.0))
        if door_prior is not None:
            self.door_prior = max(0.0, min(door_prior, 1.0))
        if window_prior is not None:
            self.window_prior = max(0.0, min(window_prior, 1.0))
        if light_prior is not None:
            self.light_prior = max(0.0, min(light_prior, 1.0))
        if environmental_prior is not None:
            self.environmental_prior = max(0.0, min(environmental_prior, 1.0))
        if entity_priors is not None:
            self.entity_priors = entity_priors
        if last_updated is not None:
            self.last_updated = last_updated
        if analysis_period is not None:
            self.analysis_period = analysis_period

    def update_entity_prior(
        self,
        entity_id: str,
        prob_given_true: float,
        prob_given_false: float,
        prior: float,
        timestamp: str,
    ) -> None:
        """Update prior for a specific entity."""
        self.entity_priors[entity_id] = {
            "prob_given_true": max(0.0, min(prob_given_true, 1.0)),
            "prob_given_false": max(0.0, min(prob_given_false, 1.0)),
            "prior": max(0.0, min(prior, 1.0)),
        }
        self.last_updated[entity_id] = timestamp

    def update_type_prior(
        self,
        sensor_type: str,
        prior_value: float,
        timestamp: str,
    ) -> None:
        """Update prior for a specific sensor type."""
        if sensor_type == "motion":
            self.motion_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "media":
            self.media_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "appliance":
            self.appliance_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "door":
            self.door_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "window":
            self.window_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "light":
            self.light_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "environmental":
            self.environmental_prior = max(0.0, min(prior_value, 1.0))
        self.last_updated[sensor_type] = timestamp

    def calculate_overall_prior(self) -> float:
        """Calculate the overall prior from all sensor type priors."""
        type_priors = [
            self.motion_prior,
            self.media_prior,
            self.appliance_prior,
            self.door_prior,
            self.window_prior,
            self.light_prior,
            self.environmental_prior,
        ]
        valid_priors = [p for p in type_priors if p > 0]
        if not valid_priors:
            return 0.0
        return sum(valid_priors) / len(valid_priors)

    def initialize_from_defaults(self, probabilities) -> None:
        """Initialize the state from default type priors.

        Args:
            probabilities: The probabilities configuration instance
        """
        timestamp = dt_util.utcnow().isoformat()

        # Initialize from default type priors
        for sensor_type, prior_data in probabilities.get_initial_type_priors().items():
            self.update_type_prior(
                sensor_type, prior_data.get("prior", MIN_PROBABILITY), timestamp
            )

        # Calculate and set the overall prior
        overall_prior = self.calculate_overall_prior()
        self.update(overall_prior=overall_prior)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return {
            "overall_prior": self.overall_prior,
            "motion_prior": self.motion_prior,
            "media_prior": self.media_prior,
            "appliance_prior": self.appliance_prior,
            "door_prior": self.door_prior,
            "window_prior": self.window_prior,
            "light_prior": self.light_prior,
            "environmental_prior": self.environmental_prior,
            "entity_priors": self.entity_priors,
            "last_updated": self.last_updated,
            "analysis_period": self.analysis_period,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorState":
        """Create a PriorState from a dictionary."""
        return cls(
            overall_prior=float(data.get("overall_prior", 0.0)),
            motion_prior=float(data.get("motion_prior", 0.0)),
            media_prior=float(data.get("media_prior", 0.0)),
            appliance_prior=float(data.get("appliance_prior", 0.0)),
            door_prior=float(data.get("door_prior", 0.0)),
            window_prior=float(data.get("window_prior", 0.0)),
            light_prior=float(data.get("light_prior", 0.0)),
            environmental_prior=float(data.get("environmental_prior", 0.0)),
            entity_priors=dict(data.get("entity_priors", {})),
            last_updated=dict(data.get("last_updated", {})),
            analysis_period=int(data.get("analysis_period", 7)),
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
