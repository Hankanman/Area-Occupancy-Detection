"""Utility functions for the area occupancy component."""

from datetime import datetime

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
    return max(0.001, min(value, 1.0))


def validate_prior(value: float) -> float:
    """Validate a prior value."""
    return max(0.001, min(value, 1.0))


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


def bayesian_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    is_active: bool,
) -> float:
    """Perform a Bayesian update using Bayes' theorem.

    Args:
        prior: The prior probability (0.0-1.0)
        prob_given_true: Probability of evidence given true state (0.0-1.0)
        prob_given_false: Probability of evidence given false state (0.0-1.0)
        is_active: Whether the sensor is in an active state (default: True)

    Returns:
        The updated probability (0.0-1.0)

    """
    # Validate input probabilities
    probability = 0.0
    if is_active:
        # P(occupied | active) = P(active | occupied) * P(occupied) / P(active)
        numerator = prob_given_true * prior
        denominator = (prob_given_true * prior) + (prob_given_false * (1 - prior))
    else:
        # P(occupied | inactive) = P(inactive | occupied) * P(occupied) / P(inactive)
        # For inactive state, we use prob_given_false directly since it represents
        # the probability of the sensor being inactive when the area is occupied
        numerator = prob_given_false * prior
        denominator = (prob_given_false * prior) + (
            (1 - prob_given_false) * (1 - prior)
        )

    if denominator == 0:
        probability = MIN_PROBABILITY
    else:
        probability = numerator / denominator

    return validate_prob(probability)
