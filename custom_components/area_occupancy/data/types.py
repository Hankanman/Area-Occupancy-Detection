"""Shared data types for Area Occupancy Detection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GaussianParams:
    """Learned Gaussian distribution parameters for numeric sensor likelihoods."""

    mean_occupied: float
    std_occupied: float
    mean_unoccupied: float
    std_unoccupied: float
