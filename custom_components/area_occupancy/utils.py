"""Utility functions for the area occupancy component."""

from datetime import datetime

from homeassistant.util import dt as dt_util

from .const import (
    MAX_PRIOR,
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PRIOR,
    MIN_PROBABILITY,
    MIN_WEIGHT,
)

#################################
# Validation Methods
#################################


def validate_prob(value: float) -> float:
    """Validate a probability value."""
    if not MIN_PROBABILITY <= value <= MAX_PROBABILITY:
        raise ValueError(
            f"Probability must be between {MIN_PROBABILITY} and {MAX_PROBABILITY} got: {value}"
        )
    return max(MIN_PROBABILITY, min(value, MAX_PROBABILITY))


def validate_prior(value: float) -> float:
    """Validate a prior value."""
    if not MIN_PRIOR <= value <= MAX_PRIOR:
        raise ValueError(
            f"Prior must be between {MIN_PRIOR} and {MAX_PRIOR} got: {value}"
        )
    return max(MIN_PRIOR, min(value, MAX_PRIOR))


def validate_datetime(value: datetime | None) -> datetime:
    """Validate a datetime value."""
    if not isinstance(value, datetime):
        return dt_util.utcnow()
    return value


def validate_weight(value: float) -> float:
    """Validate a weight value."""
    if not MIN_WEIGHT <= value <= MAX_WEIGHT:
        raise ValueError(
            f"Weight must be between {MIN_WEIGHT} and {MAX_WEIGHT} got: {value}"
        )
    return max(MIN_WEIGHT, min(value, MAX_WEIGHT))
