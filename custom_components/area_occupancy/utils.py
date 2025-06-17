"""Utility functions for the area occupancy component."""

from datetime import datetime
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from .const import (
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PROBABILITY,
    MIN_WEIGHT,
    ROUNDING_PRECISION,
)

if TYPE_CHECKING:
    from .data.entity import Entity

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
    return round(float(value), ROUNDING_PRECISION)


EPS = 1e-12


def bayesian_probability(
    *,
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    is_active: bool | None,
    weight: float,
    decay_factor: float,
) -> float:
    """Weighted, time-decaying single-step Bayesian update (fractional Bayes).

    This function implements a Bayesian update that:
    1. Weights the evidence based on sensor reliability
    2. Applies time decay to reduce evidence strength over time
    3. Handles both positive (ON) and negative (OFF) evidence

    Args:
        prior: Current probability estimate (0-1)
        prob_given_true: P(sensor=ON | room=occupied)
        prob_given_false: P(sensor=ON | room=empty)
        is_active: Current sensor state (True=ON, False=OFF, None=no change)
        weight: Sensor reliability weight (0-1)
        decay_factor: Time decay factor (0-1)

    Returns:
        Updated probability estimate (0-1)

    """
    # Return prior unchanged if no new evidence or zero weight/decay
    if is_active is None or weight == 0.0 or decay_factor == 0.0:
        return prior

    # Calculate likelihood ratio based on sensor state
    # For ON state: P(sensor=ON|occupied) / P(sensor=ON|empty)
    # For OFF state: P(sensor=OFF|occupied) / P(sensor=OFF|empty)
    likelihood_ratio = (
        (prob_given_true + EPS) / (prob_given_false + EPS)
        if is_active
        else (1.0 - prob_given_true + EPS) / (1.0 - prob_given_false + EPS)
    )

    # Apply sensor weight and time decay to likelihood ratio
    # This reduces evidence strength for less reliable sensors
    # and for evidence that is older
    likelihood_ratio **= weight * decay_factor

    # Convert probability to odds ratio
    prior_odds = prior / (1.0 - prior + EPS)

    # Update odds ratio using likelihood ratio
    post_odds = prior_odds * likelihood_ratio

    # Convert back to probability
    return post_odds / (1.0 + post_odds)


def overall_probability(entities: dict[str, "Entity"], prior: float) -> float:
    """Calculate new probability using Bayesian inference based on current observations.

    This method implements Bayes' theorem to update probabilities based on observations.
    For each observation, it:
    1. Checks if the observation is True/False/None
    2. Updates the prior probability using the observation's conditional probabilities
    3. Handles None cases by skipping the update

    The calculation uses:
    - Prior probability (initial belief)
    - P(Data|Hypothesis) - probability of observation given hypothesis is true
    - P(Data|~Hypothesis) - probability of observation given hypothesis is false

    Args:
        entities: Dictionary of entities with observations
        prior: Prior probability

    Returns:
        float: The updated probability (posterior) after considering all observations

    """
    posterior = prior

    for e in entities.values():
        # Observation state --------------------------------------------------
        observed = (
            True
            if e.is_active or e.decay.is_decaying
            else False
            if not e.is_active and not e.decay.is_decaying
            else None
        )

        # Effective parameters ----------------------------------------------
        w = e.type.weight  # static sensor importance
        df = e.decay.decay_factor if e.decay.is_decaying else 1.0

        # Bayesian fusion ----------------------------------------------------
        posterior = bayesian_probability(
            prior=posterior,
            prob_given_true=e.prior.prob_given_true,
            prob_given_false=e.prior.prob_given_false,
            is_active=observed,
            weight=w,
            decay_factor=df,
        )

    return posterior
