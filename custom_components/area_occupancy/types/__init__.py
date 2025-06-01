"""Type definitions for Area Occupancy Detection."""

from datetime import datetime
from typing import TypedDict


class SensorInfo(TypedDict):
    """Type for sensor state information."""

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

    probability: float
    weight: float
    weighted_probability: float


class TimeInterval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime
    state: str
