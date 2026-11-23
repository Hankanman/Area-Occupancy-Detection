"""Shared data types for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class GaussianParams:
    """Learned Gaussian distribution parameters for numeric sensor likelihoods."""

    mean_occupied: float
    std_occupied: float
    mean_unoccupied: float
    std_unoccupied: float


@dataclass(frozen=True)
class EnvironmentalData:
    """Captured environmental sensor readings at a point in time."""

    timestamp: datetime
    temperature: float | None = None
    humidity: float | None = None
    pressure: float | None = None
    co2: float | None = None
    voc: float | None = None


@dataclass(frozen=True)
class EnvironmentalAnalysisResult:
    """Result of environmental analysis for occupancy detection."""

    occupancy_probability: float
    env_data: EnvironmentalData
    learned_params: GaussianParams | None = None
    anomalies: dict[str, float] | None = None


@dataclass(frozen=True)
class ZonePriors:
    """Empirical priors for an aggregate zone ("All Areas" or a floor) (#557).

    Measured from the union of the member rooms' occupied intervals, so they
    answer "how often was anyone in this zone", which an average of the
    rooms' own priors understates.

    Attributes:
        global_prior: Fraction of the observation window in which any member
            room was occupied, bounded like a room's global prior.
        time_priors: ``(day_of_week, hour)`` to the same fraction per weekly
            slot, bounded like a room's time priors.
        data_points: ``(day_of_week, hour)`` to the weeks observed.
        computed_at: When these were computed.
    """

    global_prior: float
    time_priors: dict[tuple[int, int], float]
    data_points: dict[tuple[int, int], int]
    computed_at: datetime
