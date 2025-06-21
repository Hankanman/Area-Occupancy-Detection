"""Utility functions for the Area Occupancy component."""

from __future__ import annotations

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


# ───────────────────────────────────────── Validation ────────────────────────
def validate_prob(value: float) -> float:
    """Validate probability value."""
    return max(0.001, min(value, 1.0))


def validate_prior(value: float) -> float:
    """Validate prior probability value."""
    return max(0.001, min(value, 1.0))


def validate_datetime(value: datetime | None) -> datetime:
    """Validate datetime value."""
    return value if isinstance(value, datetime) else dt_util.utcnow()


def validate_weight(value: float) -> float:
    """Validate weight value."""
    return max(MIN_WEIGHT, min(value, MAX_WEIGHT))


def validate_decay_factor(value: float) -> float:
    """Validate decay factor value."""
    return max(MIN_PROBABILITY, min(value, MAX_PROBABILITY))


def format_float(value: float) -> float:
    """Format float value."""
    return round(float(value), ROUNDING_PRECISION)


def format_percentage(value: float) -> str:
    """Format float value as percentage."""
    return f"{value * 100:.2f}%"


EPS = 1e-12


# ────────────────────────────────────── Core Bayes ───────────────────────────
def bayesian_probability(
    *,  # keyword-only → prevents accidental positional mix-ups
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    evidence: bool | None,
    weight: float,
    decay_factor: float,
) -> float:
    """Fractional-power odds-space Bayesian update.

    Args:
        prior: Prior probability
        prob_given_true: Probability of evidence given true
        prob_given_false: Probability of evidence given false
        evidence: Evidence
        weight: Weight
        decay_factor: Decay factor

    Returns:
        Posterior probability

    """
    if evidence is None or weight == 0 or decay_factor == 0:
        return prior

    # Calculate Bayes factor
    bayes_factor = (
        (prob_given_true + EPS) / (prob_given_false + EPS)
        if evidence
        else (1 - prob_given_true + EPS) / (1 - prob_given_false + EPS)
    )

    # Apply weight and decay factor
    bayes_factor **= weight * decay_factor

    # Calculate posterior odds
    odds = prior / (1.0 - prior + EPS)
    posterior_odds = odds * bayes_factor

    # Return posterior probability
    return posterior_odds / (1.0 + posterior_odds)


# ─────────────────────────────── Area-level fusion ───────────────────────────
def overall_probability(
    entities: dict[str, Entity],
    prior: float,
) -> float:
    """Combine current beliefs of all entities into one room posterior.

    Args:
        entities: Dictionary of entities
        prior: Prior probability

    Returns:
        Posterior probability

    """
    posterior = prior
    for e in entities.values():
        # Calculate observed evidence
        observed = (
            True
            if e.evidence or e.decay.is_decaying
            else False
            if not e.evidence and not e.decay.is_decaying
            else None
        )

        # Update posterior probability
        posterior = bayesian_probability(
            prior=posterior,
            prob_given_true=e.prior.prob_given_true,
            prob_given_false=e.prior.prob_given_false,
            evidence=observed,
            weight=e.type.weight,
            decay_factor=e.decay.decay_factor if e.decay.is_decaying else 1.0,
        )
    return validate_prob(posterior)
