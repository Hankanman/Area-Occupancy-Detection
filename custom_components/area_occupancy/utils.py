"""Utility functions for the Area Occupancy component."""

from __future__ import annotations

from contextlib import suppress
import math
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

from .const import MAX_PROBABILITY, MIN_PROBABILITY, ROUNDING_PRECISION

if TYPE_CHECKING:
    from .data.entity import Entity


# ─────────────────────────────────────── Utility Functions ───────────────────


def format_float(value: float) -> float:
    """Format float value."""
    return round(float(value), ROUNDING_PRECISION)


def format_percentage(value: float) -> str:
    """Format float value as percentage."""
    return f"{value * 100:.2f}%"


def clamp_probability(value: float) -> float:
    """Clamp probability to valid range."""
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, value))


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


# ────────────────────────────────────── Core Bayes ───────────────────────────
def bayesian_probability(
    entities: dict[str, Entity], area_prior: float = 0.5, time_prior: float = 0.5
):
    """Compute posterior probability of occupancy given current features, area prior, and time prior.

    Args:
        entities: Dict mapping entity_id to Entity objects containing evidence and likelihood
        area_prior: Base prior probability of occupancy for this area (default: 0.5)
        time_prior: Time-based modifier for the prior (default: 0.5)

    """
    # Handle edge cases first
    if not entities:
        # No entities provided - return combined prior
        return combine_priors(area_prior, time_prior)

    # Check for entities with zero weights (they contribute nothing)
    active_entities = {k: v for k, v in entities.items() if v.weight > 0.0}

    if not active_entities:
        # All entities have zero weight - return combined prior
        return combine_priors(area_prior, time_prior)

    # Check for entities with invalid likelihoods
    entities_to_remove = []
    for entity_id, entity in active_entities.items():
        if (
            entity.prob_given_true <= 0.0
            or entity.prob_given_true >= 1.0
            or entity.prob_given_false <= 0.0
            or entity.prob_given_false >= 1.0
        ):
            # Mark entities with invalid likelihoods for removal
            entities_to_remove.append(entity_id)

    # Remove invalid entities after iteration
    for entity_id in entities_to_remove:
        active_entities.pop(entity_id, None)

    if not active_entities:
        # All entities had invalid likelihoods - return combined prior
        return combine_priors(area_prior, time_prior)

    # Check for extreme decay factors
    for entity in active_entities.values():
        if entity.decay.decay_factor < 0.0 or entity.decay.decay_factor > 1.0:
            # Clamp decay factor to valid range
            entity.decay.decay_factor = max(0.0, min(1.0, entity.decay.decay_factor))

    # Combine area prior with time prior using the helper function
    combined_prior = combine_priors(area_prior, time_prior)

    # Clamp combined prior
    combined_prior = clamp_probability(combined_prior)

    # log-space for numerical stability
    log_true = math.log(combined_prior)
    log_false = math.log(1 - combined_prior)

    for entity in entities.values():
        value = entity.evidence
        decay_factor = entity.decay.decay_factor
        is_decaying = entity.decay.is_decaying

        # Determine effective evidence: True if evidence is True OR if decaying
        effective_evidence = value or is_decaying

        if effective_evidence:
            # Evidence is present (either current or decaying) - use likelihoods with decay applied
            p_t = entity.prob_given_true
            p_f = entity.prob_given_false

            # Apply decay factor to reduce the strength of the evidence
            if is_decaying and decay_factor < 1.0:
                # When decaying, interpolate between neutral (0.5) and full evidence based on decay factor
                neutral_prob = 0.5
                p_t = neutral_prob + (p_t - neutral_prob) * decay_factor
                p_f = neutral_prob + (p_f - neutral_prob) * decay_factor
        else:
            # No evidence present - use neutral probabilities
            p_t = 0.5
            p_f = 0.5

        # Clamp probabilities to avoid log(0) or log(1)
        p_t = clamp_probability(p_t)
        p_f = clamp_probability(p_f)

        log_true += math.log(p_t) * entity.weight
        log_false += math.log(p_f) * entity.weight

    # convert back
    max_log = max(log_true, log_false)
    true_prob = math.exp(log_true - max_log)
    false_prob = math.exp(log_false - max_log)

    # Handle numerical overflow/underflow edge case
    total_prob = true_prob + false_prob
    if total_prob == 0.0:
        # Both probabilities are zero - return combined prior as fallback
        return combined_prior

    return true_prob / total_prob


def combine_priors(
    area_prior: float, time_prior: float, time_weight: float = 0.2
) -> float:
    """Combine area prior and time prior using weighted averaging in logit space.

    Args:
        area_prior: Base prior probability of occupancy for this area
        time_prior: Time-based modifier for the prior
        time_weight: Weight given to time_prior (0.0 to 1.0, default: 0.2)

    Returns:
        float: Combined prior probability

    """
    # Handle edge cases first
    if time_weight == 0.0:
        # No time influence, return area_prior
        return clamp_probability(area_prior)

    if time_weight == 1.0:
        # Full time influence, return time_prior (with clamping)
        return clamp_probability(time_prior)

    if time_prior == 0.0:
        # Time slot has never been occupied - this is strong evidence
        # Use a very small probability but not zero
        time_prior = MIN_PROBABILITY
    elif time_prior == 1.0:
        # Time slot has always been occupied - this is strong evidence
        time_prior = MAX_PROBABILITY

    # Handle area_prior edge cases
    if area_prior == 0.0:
        # Area has never been occupied - this is strong evidence
        area_prior = MIN_PROBABILITY
    elif area_prior == 1.0:
        # Area has always been occupied - this is strong evidence
        area_prior = MAX_PROBABILITY

    # Handle identical priors case
    if abs(area_prior - time_prior) < 1e-10:
        # Priors are essentially identical, return the common value
        return area_prior

    # Clamp other inputs to valid ranges
    area_prior = clamp_probability(area_prior)
    time_weight = max(0.0, min(1.0, time_weight))

    area_weight = 1.0 - time_weight

    # Convert to logit space for better interpolation
    def prob_to_logit(p):
        return math.log(p / (1 - p))

    def logit_to_prob(logit):
        return 1 / (1 + math.exp(-logit))

    # Interpolate in logit space for more principled combination
    area_logit = prob_to_logit(area_prior)
    time_logit = prob_to_logit(time_prior)

    # Weighted combination in logit space
    combined_logit = area_weight * area_logit + time_weight * time_logit
    combined_prior = logit_to_prob(combined_logit)

    return clamp_probability(combined_prior)
