"""State containers for Area Occupancy Detection."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from homeassistant.util import dt as dt_util

from ..const import MAX_PROBABILITY, MIN_PRIOR, MIN_PROBABILITY
from ..models.feature import Feature
from ..models.prior import Prior


@dataclass
class ProbabilityState:
    """State of probability calculations."""

    probability: float = field(default=0.0)
    previous_probability: float = field(default=0.0)
    threshold: float = field(default=0.5)
    prior_probability: float = field(default=0.0)
    sensor_probabilities: dict[str, Feature] = field(default_factory=dict)
    decay_status: float = field(default=0.0)
    current_states: dict[str, Feature] = field(default_factory=dict)
    previous_states: dict[str, Feature] = field(default_factory=dict)
    is_occupied: bool = field(default=False)
    decaying: bool = field(default=False)
    decay_start_time: datetime | None = field(default=None)
    decay_start_probability: float | None = field(default=None)

    @property
    def active_triggers(self) -> list[str]:
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
        # Decay status is 0-100
        if not 0 <= self.decay_status <= 100:
            raise ValueError("Decay status must be between 0 and 100")

    def update(
        self,
        probability: float | None = None,
        previous_probability: float | None = None,
        threshold: float | None = None,
        prior_probability: float | None = None,
        sensor_probabilities: dict[str, Feature] | None = None,
        decay_status: float | None = None,
        current_states: dict[str, Feature] | None = None,
        previous_states: dict[str, Feature] | None = None,
        is_occupied: bool | None = None,
        decaying: bool | None = None,
        decay_start_time: datetime | None = None,
        decay_start_probability: float | None = None,
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
            self.sensor_probabilities = dict(sensor_probabilities)
        if decay_status is not None:
            self.decay_status = max(0.0, min(decay_status, 100.0))
        if current_states is not None:
            self.current_states = dict(current_states)
        if previous_states is not None:
            self.previous_states = dict(previous_states)
        if is_occupied is not None:
            self.is_occupied = is_occupied
        if decaying is not None:
            self.decaying = decaying
        if decay_start_time is not None:
            self.decay_start_time = decay_start_time
        if decay_start_probability is not None:
            self.decay_start_probability = decay_start_probability

    def to_dict(self) -> dict[str, Any]:
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
            "decay_start_time": self.decay_start_time.isoformat()
            if self.decay_start_time
            else None,
            "decay_start_probability": self.decay_start_probability,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbabilityState":
        """Create a ProbabilityState from a dictionary."""
        decay_start_time_str = data.get("decay_start_time")
        decay_start_time = (
            dt_util.parse_datetime(str(decay_start_time_str))
            if decay_start_time_str
            else None
        )
        return cls(
            probability=float(data.get("probability", 0.0)),
            previous_probability=float(data.get("previous_probability", 0.0)),
            threshold=float(data.get("threshold", 0.5)),
            prior_probability=float(data.get("prior_probability", 0.0)),
            sensor_probabilities=data.get("sensor_probabilities", {}),
            decay_status=float(data.get("decay_status", 0.0)),
            current_states=data.get("current_states", {}),
            previous_states=data.get("previous_states", {}),
            is_occupied=bool(data.get("is_occupied", False)),
            decaying=bool(data.get("decaying", False)),
            decay_start_time=decay_start_time,
            decay_start_probability=data.get("decay_start_probability"),
        )


@dataclass
class PriorState:
    """State of prior probability calculations."""

    # Overall prior probability for the area
    overall_prior: float = field(default=MIN_PROBABILITY)

    # Prior probabilities by sensor type (simple float for easy access)
    motion_prior: float = field(default=MIN_PROBABILITY)
    media_prior: float = field(default=MIN_PROBABILITY)
    appliance_prior: float = field(default=MIN_PROBABILITY)
    door_prior: float = field(default=MIN_PROBABILITY)
    window_prior: float = field(default=MIN_PROBABILITY)
    light_prior: float = field(default=MIN_PROBABILITY)
    environmental_prior: float = field(default=MIN_PROBABILITY)
    wasp_in_box_prior: float = field(default=MIN_PROBABILITY)

    # Individual entity priors using PriorData
    entity_priors: dict[str, Prior] = field(default_factory=dict)

    # Type-level priors with full probability data using PriorData
    type_priors: dict[str, Prior] = field(default_factory=dict)

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
            "wasp_in_box_prior",
        ]:
            value = getattr(self, attr_name)
            if not MIN_PROBABILITY <= value <= MAX_PROBABILITY:
                raise ValueError(
                    f"{attr_name} must be between {MIN_PROBABILITY} and {MAX_PROBABILITY}"
                )

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
        wasp_in_box_prior: float | None = None,
        entity_priors: dict[str, Prior] | None = None,
        type_priors: dict[str, Prior] | None = None,
        analysis_period: int | None = None,
    ) -> None:
        """Update the state with new values while maintaining the same instance."""
        if overall_prior is not None:
            self.overall_prior = max(
                MIN_PROBABILITY, min(overall_prior, MAX_PROBABILITY)
            )
        if motion_prior is not None:
            self.motion_prior = max(MIN_PROBABILITY, min(motion_prior, MAX_PROBABILITY))
        if media_prior is not None:
            self.media_prior = max(MIN_PROBABILITY, min(media_prior, MAX_PROBABILITY))
        if appliance_prior is not None:
            self.appliance_prior = max(
                MIN_PROBABILITY, min(appliance_prior, MAX_PROBABILITY)
            )
        if door_prior is not None:
            self.door_prior = max(MIN_PROBABILITY, min(door_prior, MAX_PROBABILITY))
        if window_prior is not None:
            self.window_prior = max(MIN_PROBABILITY, min(window_prior, MAX_PROBABILITY))
        if light_prior is not None:
            self.light_prior = max(MIN_PROBABILITY, min(light_prior, MAX_PROBABILITY))
        if environmental_prior is not None:
            self.environmental_prior = max(
                MIN_PROBABILITY, min(environmental_prior, MAX_PROBABILITY)
            )
        if wasp_in_box_prior is not None:
            self.wasp_in_box_prior = max(
                MIN_PROBABILITY, min(wasp_in_box_prior, MAX_PROBABILITY)
            )
        if entity_priors is not None:
            self.entity_priors = entity_priors
        if type_priors is not None:
            self.type_priors = type_priors
        if analysis_period is not None:
            self.analysis_period = analysis_period

    def to_dict(self) -> dict[str, Any]:
        """Convert the PriorState dataclass to a dictionary for storage."""
        return {
            "overall_prior": self.overall_prior,
            "motion_prior": self.motion_prior,
            "media_prior": self.media_prior,
            "appliance_prior": self.appliance_prior,
            "door_prior": self.door_prior,
            "window_prior": self.window_prior,
            "light_prior": self.light_prior,
            "environmental_prior": self.environmental_prior,
            "wasp_in_box_prior": self.wasp_in_box_prior,
            # Serialize PriorData objects
            "entity_priors": {
                k: v.to_dict()
                for k, v in self.entity_priors.items()
                if v and v.prior >= MIN_PRIOR
            },
            "type_priors": {
                k: v.to_dict()
                for k, v in self.type_priors.items()
                if v and v.prior >= MIN_PRIOR
            },
            "analysis_period": self.analysis_period,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriorState":
        """Create a PriorState from a dictionary."""
        return cls(
            overall_prior=float(data.get("overall_prior", MIN_PROBABILITY)),
            motion_prior=float(data.get("motion_prior", MIN_PROBABILITY)),
            media_prior=float(data.get("media_prior", MIN_PROBABILITY)),
            appliance_prior=float(data.get("appliance_prior", MIN_PROBABILITY)),
            door_prior=float(data.get("door_prior", MIN_PROBABILITY)),
            window_prior=float(data.get("window_prior", MIN_PROBABILITY)),
            light_prior=float(data.get("light_prior", MIN_PROBABILITY)),
            environmental_prior=float(data.get("environmental_prior", MIN_PROBABILITY)),
            wasp_in_box_prior=float(data.get("wasp_in_box_prior", MIN_PROBABILITY)),
            # Deserialize PriorData objects
            entity_priors={
                k: pd
                for k, v in data.get("entity_priors", {}).items()
                if isinstance(v, dict) and (pd := Prior.from_dict(v)) is not None
            },
            type_priors={
                k: pd
                for k, v in data.get("type_priors", {}).items()
                if isinstance(v, dict) and (pd := Prior.from_dict(v)) is not None
            },
            analysis_period=int(data.get("analysis_period", 7)),
        )
