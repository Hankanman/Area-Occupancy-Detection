"""Utility functions for the area occupancy component."""

from datetime import datetime
import logging

from homeassistant.util import dt as dt_util

from .const import (
    MAX_PRIOR,
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PRIOR,
    MIN_PROBABILITY,
    MIN_WEIGHT,
)

_LOGGER = logging.getLogger(__name__)

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


def bayesian_update(
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
    prior = validate_prob(prior)
    prob_given_true = validate_prob(prob_given_true)
    prob_given_false = validate_prob(prob_given_false)

    # Calculate using Bayes' theorem: P(H|E) = P(E|H)P(H) / P(E)
    # where P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)
    # Use prob_given_true if active, prob_given_false if inactive
    evidence_prob = prob_given_true if is_active else prob_given_false
    numerator = evidence_prob * prior
    denominator = numerator + (prob_given_false if is_active else prob_given_true) * (
        1 - prior
    )

    if denominator == 0:
        _LOGGER.warning(
            "Denominator is zero in Bayesian update, returning MIN_PROBABILITY"
        )
        return MIN_PROBABILITY

    result = numerator / denominator
    return validate_prob(result)
