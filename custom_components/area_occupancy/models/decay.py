"""Decay model for Area Occupancy Detection."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from homeassistant.util import dt as dt_util

from ..const import DECAY_LAMBDA, MAX_PROBABILITY, MIN_PROBABILITY
from ..utils import validate_datetime, validate_prob

_LOGGER = logging.getLogger(__name__)

# Validation constants
MIN_DECAY_WINDOW: Final[int] = 1  # Minimum decay window in seconds
MAX_DECAY_WINDOW: Final[int] = 3600  # Maximum decay window in seconds


@dataclass
class Decay:
    """Represents decay state for an entity."""

    is_decaying: bool = False
    decay_start_time: datetime | None = None
    decay_start_probability: float | None = None
    decay_window: int = 300  # Default 5 minutes
    decay_enabled: bool = True
    decay_factor: float = 1.0

    def __post_init__(self):
        """Validate properties after initialization."""
        if self.decay_start_probability is not None:
            self.decay_start_probability = validate_prob(self.decay_start_probability)
        self.decay_start_time = validate_datetime(self.decay_start_time)
        self.decay_factor = validate_prob(self.decay_factor)

        # Validate decay window
        if not isinstance(self.decay_window, (int, float)):
            raise TypeError(
                f"Decay window must be a number, got {type(self.decay_window)}"
            )
        if not MIN_DECAY_WINDOW <= self.decay_window <= MAX_DECAY_WINDOW:
            raise ValueError(
                f"Decay window must be between {MIN_DECAY_WINDOW} and {MAX_DECAY_WINDOW} seconds"
            )

    def update_decay(
        self,
        current_probability: float,
        previous_probability: float,
    ) -> tuple[float, float]:
        """Calculate decay factor and apply it to probability.

        Args:
            current_probability: The current probability value (used to detect increase)
            previous_probability: The probability value from the previous step

        Returns:
            Tuple of (decayed_probability, decay_factor)

        Raises:
            ValueError: If probabilities are invalid

        """
        # Validate inputs
        if not self._validate_probabilities(current_probability, previous_probability):
            raise ValueError("Probabilities must be between 0 and 1")

        if not self.decay_enabled:
            _LOGGER.debug("Decay disabled, returning current probability")
            self.decay_factor = 1.0
            return current_probability, self.decay_factor

        # --- Determine new decay state ---
        # Condition 1: Probability increased or stayed the same
        if current_probability >= previous_probability:
            if self.is_decaying:  # Log only if decay was previously active
                _LOGGER.debug(
                    "Probability increased/stable: %.3f -> %.3f. Stopping decay",
                    previous_probability,
                    current_probability,
                )
            self.decay_start_time = None
            self.is_decaying = False
            self.decay_start_probability = None
            self.decay_factor = 1.0
            return current_probability, self.decay_factor

        # Condition 2: Probability decreased, decay wasn't active before
        if not self.is_decaying:
            _LOGGER.debug(
                "Probability decreased: %.3f -> %.3f. Starting decay",
                previous_probability,
                current_probability,
            )
            self.decay_start_time = dt_util.utcnow()
            self.is_decaying = True
            self.decay_start_probability = previous_probability
            self.decay_factor = 1.0
            return previous_probability, self.decay_factor

        # Condition 3: Probability decreased, decay was already active
        if not self.decay_start_time or self.decay_start_probability is None:
            _LOGGER.error(
                "Inconsistent state: is_decaying=True but decay_start_time (%s) or decay_start_probability (%s) is None. Resetting decay",
                self.decay_start_time,
                self.decay_start_probability,
            )
            self.decay_start_time = dt_util.utcnow()
            self.is_decaying = True
            self.decay_start_probability = previous_probability
            self.decay_factor = 1.0
            return previous_probability, self.decay_factor

        # --- Calculate decay based on existing start time and start probability ---
        try:
            elapsed = (dt_util.utcnow() - self.decay_start_time).total_seconds()
            self.decay_factor = self._calculate_decay_factor(elapsed)

            # Calculate the potential decayed value based on the probability when decay *started*
            potential_decayed_prob = self._apply_decay_factor(
                self.decay_start_probability, self.decay_factor
            )

            # The final decayed probability should not be lower than what current sensors suggest
            decayed_probability = max(potential_decayed_prob, current_probability)

            # Check if decay reached minimum threshold
            if decayed_probability <= MIN_PROBABILITY:
                _LOGGER.debug("Decay complete: reached minimum probability")
                self.decay_start_time = None
                self.is_decaying = False
                self.decay_start_probability = None
                return MIN_PROBABILITY, self.decay_factor

            # Decay continues
            _LOGGER.debug(
                "Decay active: t=%.1fs factor=%.3f p_start=%.3f -> p_decayed=%.3f (floor=%.3f)",
                elapsed,
                self.decay_factor,
                self.decay_start_probability,
                decayed_probability,
                current_probability,
            )

        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error in decay calculation: %s", err)
            self.is_decaying = False
            self.decay_start_time = None
            self.decay_start_probability = None
            self.decay_factor = 1.0
            return current_probability, self.decay_factor
        else:
            return decayed_probability, self.decay_factor

    def reset(self) -> None:
        """Reset decay state to defaults."""
        self.is_decaying = False
        self.decay_start_time = None
        self.decay_start_probability = None
        self.decay_window = 300  # Default 5 minutes
        self.decay_enabled = True
        self.decay_factor = 1.0

    def _validate_probabilities(
        self, current_probability: float, previous_probability: float
    ) -> bool:
        """Validate probability values."""
        return all(0 <= p <= 1 for p in (current_probability, previous_probability))

    def _calculate_decay_factor(self, elapsed_seconds: float) -> float:
        """Calculate the decay factor based on elapsed time."""
        return math.exp(-DECAY_LAMBDA * (elapsed_seconds / self.decay_window))

    def _apply_decay_factor(self, probability: float, decay_factor: float) -> float:
        """Apply decay factor to probability with bounds checking."""
        return max(
            MIN_PROBABILITY,
            min(probability * decay_factor, MAX_PROBABILITY),
        )
