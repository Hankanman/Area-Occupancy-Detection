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
