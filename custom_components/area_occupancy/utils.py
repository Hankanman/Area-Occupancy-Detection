"""Utility functions for the Area Occupancy component."""

from __future__ import annotations

from contextlib import suppress
import math
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

from .const import ROUNDING_PRECISION

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


def format_float(value: float) -> float:
    """Format float value."""
    return round(float(value), ROUNDING_PRECISION)


def format_percentage(value: float) -> str:
    """Format float value as percentage."""
    return f"{value * 100:.2f}%"


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
    # Clamp priors to avoid log(0) or log(1)
    area_prior = max(0.001, min(0.999, area_prior))
    time_prior = max(0.001, min(0.999, time_prior))

    # Combine area prior with time prior modifier
    # Use time_prior as a multiplier on the area_prior
    combined_prior = area_prior * time_prior / 0.5  # Normalize by default time prior

    # Clamp combined prior
    combined_prior = max(0.001, min(0.999, combined_prior))

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
        p_t = max(0.001, min(0.999, p_t))
        p_f = max(0.001, min(0.999, p_f))

        log_true += math.log(p_t) * entity.weight
        log_false += math.log(p_f) * entity.weight

    # convert back
    max_log = max(log_true, log_false)
    true_prob = math.exp(log_true - max_log)
    false_prob = math.exp(log_false - max_log)
    return true_prob / (true_prob + false_prob)
