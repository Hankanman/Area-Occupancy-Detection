"""Utility functions for the Area Occupancy component."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
import os
from pathlib import Path
import time
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


# ─────────────────────────────────────── File Lock ───────────────────────────
class FileLock:
    """Simple file-based lock using context manager with atomic file creation."""

    def __init__(self, lock_path: Path, timeout: int = 60):
        """Initialize the lock."""
        self.lock_path = lock_path
        self.timeout = timeout
        self._lock_fd = None

    def __enter__(self):
        """Enter the context manager."""
        start_time = time.time()

        while True:
            try:
                # Atomic file creation with O_EXCL flag
                # This ensures only one process can create the file
                self._lock_fd = os.open(
                    self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode=0o644
                )
                # Write PID to lock file for debugging
                os.write(self._lock_fd, str(os.getpid()).encode())
                os.fsync(self._lock_fd)  # Ensure data is written to disk
            except FileExistsError:
                # Lock file already exists, check timeout
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(
                        f"Timeout waiting for lock: {self.lock_path}"
                    ) from None
                time.sleep(0.1)
            else:
                return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        # Remove lock file
        with suppress(FileNotFoundError):
            self.lock_path.unlink()


# ───────────────────────────────────────── Validation ────────────────────────
def validate_prob(value: complex) -> float:
    """Validate probability value, handling complex numbers."""
    # Handle complex numbers by taking the real part
    if isinstance(value, complex):
        value = value.real

    # Ensure it's a valid float
    if not isinstance(value, (int, float)) or not (-1e10 < value < 1e10):
        return 0.5

    return max(0.001, min(float(value), 1.0))


def validate_prior(value: float) -> float:
    """Validate prior probability value."""
    return max(0.000001, min(value, 0.999999))


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
    decay_factor: float,  # Remove weight parameter
) -> float:
    """Simplified Bayesian update - weight is already applied in likelihood values.

    Args:
        prior: Prior probability
        prob_given_true: Weighted probability of evidence given true
        prob_given_false: Weighted probability of evidence given false
        evidence: Evidence
        decay_factor: Decay factor

    Returns:
        Posterior probability

    """
    if evidence is None or decay_factor == 0:
        return prior

    # Validate inputs first
    prob_given_true = max(0.001, min(prob_given_true, 0.999))
    prob_given_false = max(0.001, min(prob_given_false, 0.999))
    prior = max(0.001, min(prior, 0.999))

    # Calculate Bayes factor
    bayes_factor = (
        (prob_given_true + EPS) / (prob_given_false + EPS)
        if evidence
        else (1 - prob_given_true + EPS) / (1 - prob_given_false + EPS)
    )

    # Ensure bayes_factor is positive to avoid complex numbers
    bayes_factor = max(EPS, bayes_factor)

    # Apply only decay factor (weight already applied in likelihood)
    if decay_factor != 1.0:
        bayes_factor = bayes_factor**decay_factor

    # Calculate posterior odds
    odds = prior / (1.0 - prior + EPS)
    posterior_odds = odds * bayes_factor

    # Return posterior probability
    return posterior_odds / (1.0 + posterior_odds)


# ─────────────────────────────── Area-level fusion ───────────────────────────
# Not used
def complementary_probability(entities: dict[str, Entity], prior: float) -> float:
    """Calculate the complementary probability.

    This function computes the probability that at least ONE entity provides
    evidence for occupancy, using the complement rule:
    P(at least one) = 1 - product(P(not each)). For each contributing entity,
    a Bayesian update is performed assuming evidence is True (or decaying),
    and the complement of the posterior is multiplied across all such entities.
    Is not affected by the order of the entities.
    Does not consider negative evidence.

    Args:
        entities: Dictionary of Entity objects to consider.
        prior: The prior probability of occupancy.

    Returns:
        The combined probability that at least one contributing entity
        indicates occupancy, after Bayesian updates and decay are applied.

    """

    contributing_entities = [
        e for e in entities.values() if e.evidence or e.decay.is_decaying
    ]

    product = 1.0
    for e in contributing_entities:
        posterior = bayesian_probability(
            prior=prior,
            prob_given_true=e.prob_given_true,
            prob_given_false=e.prob_given_false,
            evidence=True,
            decay_factor=e.decay_factor,
        )
        weighted_posterior = posterior * e.type.weight
        product *= 1 - weighted_posterior

    return 1 - product


# Not used
def conditional_probability(entities: dict[str, Entity], prior: float) -> float:
    """Return conditional probability, accounting for entity weights.

    Sequentially update the prior probability by applying Bayes' theorem for each entity,
    using the entity's evidence and likelihoods. The posterior from each step becomes the
    prior for the next entity. Each entity's weight is used to interpolate between the
    previous posterior and the new posterior, so that higher-weight entities have more
    influence on the result.

    Args:
        entities: Dictionary of Entity objects to process.
        prior: Initial prior probability.

    Returns:
        The final posterior probability after all updates.

    """

    posterior = prior
    for e in entities.values():
        # Use effective evidence: True if evidence is True OR if decaying
        effective_evidence = e.evidence or e.decay.is_decaying
        entity_posterior = bayesian_probability(
            prior=posterior,
            prob_given_true=e.prob_given_true,
            prob_given_false=e.prob_given_false,
            evidence=effective_evidence,
            decay_factor=e.decay_factor,
        )
        # Interpolate between previous posterior and entity_posterior using entity weight
        weight = e.type.weight
        posterior = posterior * (1 - weight) + entity_posterior * weight

    return posterior


def conditional_sorted_probability(entities: dict[str, Entity], prior: float) -> float:
    """Return conditional sorted probability.

    Sequentially update the prior probability by applying Bayes' theorem for each entity,
    using the entity's evidence and likelihoods. The posterior from each step becomes the
    prior for the next entity. This method reflects the effect of each entity's evidence
    (and decay, if applicable) on the overall probability. The entities are sorted by
    evidence status (active first) and then by weight (highest weight first) to ensure
    that the most relevant entities are considered first.

    Args:
        entities: Dictionary of Entity objects to process.
        prior: Initial prior probability.

    Returns:
        The final posterior probability after all updates.

    """

    sorted_entities = sorted(
        entities.values(),
        key=lambda x: (not (x.evidence or x.decay.is_decaying), -x.type.weight),
    )
    posterior = prior
    for e in sorted_entities:
        # Use effective evidence: True if evidence is True OR if decaying
        effective_evidence = e.evidence or e.decay.is_decaying
        entity_posterior = bayesian_probability(
            prior=posterior,
            prob_given_true=e.prob_given_true,
            prob_given_false=e.prob_given_false,
            evidence=effective_evidence,
            decay_factor=e.decay_factor,
        )
        # Interpolate between previous posterior and entity_posterior using entity weight
        weight = e.type.weight
        posterior = posterior * (1 - weight) + entity_posterior * weight

    return posterior
