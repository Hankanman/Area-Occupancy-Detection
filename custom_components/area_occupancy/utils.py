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
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    evidence: bool | None,
    weight: float = 1.0,
    decay_factor: float = 1.0,
) -> float:
    """Calculate Bayesian posterior probability.

    This function computes the posterior probability using Bayes' theorem with
    weighted evidence and decay factors.

    Args:
        prior: Prior probability [0,1]
        prob_given_true: P(sensor=ON | area=occupied) [0,1]
        prob_given_false: P(sensor=ON | area=unoccupied) [0,1]
        evidence: Current sensor state (True=ON, False=OFF, None=no change)
        weight: Evidence weight multiplier [0,1] (default: 1.0)
        decay_factor: Time-based decay factor [0,1] (default: 1.0)

    Returns:
        Posterior probability [0,1]

    """
    # Validate all inputs to ensure they're in valid ranges
    prior = max(MIN_PROBABILITY, min(prior, MAX_PROBABILITY))
    prob_given_true = max(MIN_PROBABILITY, min(prob_given_true, MAX_PROBABILITY))
    prob_given_false = max(MIN_PROBABILITY, min(prob_given_false, MAX_PROBABILITY))
    weight = max(0.0, min(weight, 1.0))
    decay_factor = max(0.0, min(decay_factor, 1.0))

    # No update if no evidence, zero weight, or zero decay
    if evidence is None or weight == 0.0 or decay_factor == 0.0:
        return prior

    # Apply weight and decay to the evidence strength
    effective_weight = weight * decay_factor

    # Choose likelihood based on evidence
    if evidence:
        # P(sensor=ON | area=occupied)
        likelihood = prob_given_true
        # P(sensor=ON | area=unoccupied)
        likelihood_complement = prob_given_false
    else:
        # P(sensor=OFF | area=occupied) = 1 - P(sensor=ON | area=occupied)
        likelihood = 1.0 - prob_given_true
        # P(sensor=OFF | area=unoccupied) = 1 - P(sensor=ON | area=unoccupied)
        likelihood_complement = 1.0 - prob_given_false

    # Calculate marginal probability P(evidence)
    evidence_prob = likelihood * prior + likelihood_complement * (1.0 - prior)

    # Avoid division by zero
    if evidence_prob <= EPS:
        return prior

    # Standard Bayesian update: P(occupied | evidence) = P(evidence | occupied) * P(occupied) / P(evidence)
    posterior_full = (likelihood * prior) / evidence_prob

    # Apply weighting: interpolate between prior and full Bayesian update
    if effective_weight < 1.0:
        posterior = prior + effective_weight * (posterior_full - prior)
    else:
        posterior = posterior_full

    # Ensure result is in valid probability range
    return max(MIN_PROBABILITY, min(posterior, MAX_PROBABILITY))


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
            if e.evidence or e.decay.is_decaying
            else False
            if not e.evidence and not e.decay.is_decaying
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
            evidence=observed,
            weight=w,
            decay_factor=df,
        )

    return posterior
