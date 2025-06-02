"""Type definitions for Area Occupancy Detection."""

from datetime import datetime
from typing import TypedDict


class Input(TypedDict):
    """Type for sensor state information."""

    entity_id: str | None
    type: str
    weight: float
    state: str | None
    last_changed: str
    availability: bool


class SensorProbability(TypedDict):
    """Probability details for a single sensor.

    Required fields:
        probability: Raw probability value (0-1)
        weight: Weight factor for this sensor (0-1)
        weighted_probability: Probability after applying weight
    """

    is_active: bool
    probability: float
    weighted_probability: float
    input: Input

class OverallProbability(TypedDict):
    """Holds the results of the occupancy probability calculation."""

    calculated_probability: float
    prior_probability: float
    sensor_probabilities: dict[str, SensorProbability]

class TimeInterval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime
    state: str
