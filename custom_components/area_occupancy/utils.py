"""Utility functions for the area occupancy component."""

from datetime import datetime
import math

from homeassistant.util import dt as dt_util

from .const import (
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PROBABILITY,
    MIN_WEIGHT,
    ROUNDING_PRECISION,
)

#################################
# Validation Methods
#################################


def validate_prob(value: float) -> float:
    """Validate a probability value."""
    return max(0.0001, min(value, 1.0))


def validate_prior(value: float) -> float:
    """Validate a prior value."""
    return max(0.0001, min(value, 1.0))


def validate_datetime(value: datetime | None) -> datetime:
    """Validate a datetime value."""
    if not isinstance(value, datetime):
        return dt_util.utcnow()
    return value


def validate_weight(value: float) -> float:
    """Validate a weight value."""
    return max(MIN_WEIGHT, min(value, MAX_WEIGHT))


def validate_decay_factor(value: float) -> float:
    """Validate a decay factor value."""
    return max(MIN_PROBABILITY, min(value, MAX_PROBABILITY))


def format_float(value: float) -> float:
    """Format float to consistently show 2 decimal places."""
    try:
        return round(float(value), ROUNDING_PRECISION)
    except (ValueError, TypeError):
        return 0.0


EPS = 1e-12


def bayesian_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    is_active: bool,
) -> float:
    """One-step Bayesian update.

    Args:
        prior: Prior probability P(H) in [0,1].
        prob_given_true:     P(Evidence present | Hypothesis true)
        prob_given_false: P(Evidence present | Hypothesis false)
        is_active:        Evidence.PRESENT or Evidence.ABSENT

    Returns:
        Posterior probability P(H | evidence)

    """
    # Sanity checks
    for name, p in {
        "prior": prior,
        "p_e_given_h": prob_given_true,
        "p_e_given_not_h": prob_given_false,
    }.items():
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"{name}={p} is not a probability in [0,1].")

    if is_active is True:
        num = prob_given_true * prior
        den = prob_given_true * prior + prob_given_false * (1 - prior)
    else:  # Evidence.ABSENT
        num = (1 - prob_given_true) * prior
        den = (1 - prob_given_true) * prior + (1 - prob_given_false) * (1 - prior)

    # Guard against divide-by-zero
    if math.isclose(den, 0.0, abs_tol=EPS):
        return 0.0
    return num / den
