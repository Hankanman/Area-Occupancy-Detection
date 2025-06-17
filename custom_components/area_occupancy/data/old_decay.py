"""Decay model for Area Occupancy Detection."""

from dataclasses import dataclass
from datetime import datetime
import logging
import math
from typing import Any, Final

from homeassistant.util import dt as dt_util

from ..const import MIN_PROBABILITY
from ..utils import validate_decay_factor, validate_prob

_LOGGER = logging.getLogger(__name__)

DECAY_LAMBDA = 1.732867952
# Validation constants
DECAY_INTERVAL: Final[int] = 1  # Decay interval in seconds

# Decay completion constants
DECAY_COMPLETION_THRESHOLD: Final[float] = (
    0.02  # Complete decay when probability drops to 2%
)
DECAY_FACTOR_THRESHOLD: Final[float] = 0.02  # Complete decay when factor drops to 1%


@dataclass
class Decay:
    """Decay model for Area Occupancy Detection."""

    is_decaying: bool
    decay_start_time: datetime | None
    decay_start_probability: float
    decay_enabled: bool
    decay_factor: float

    def __post_init__(self) -> None:
        """Validate properties after initialization."""
        self.decay_start_probability = validate_prob(self.decay_start_probability)
        self.decay_factor = validate_decay_factor(self.decay_factor)

    def to_dict(self) -> dict[str, Any]:
        """Convert decay to dictionary for storage."""
        return {
            "is_decaying": self.is_decaying,
            "decay_start_time": self.decay_start_time.isoformat()
            if self.decay_start_time
            else None,
            "decay_start_probability": self.decay_start_probability,
            "decay_enabled": self.decay_enabled,
            "decay_factor": self.decay_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decay":
        """Create decay from dictionary."""
        return cls(
            is_decaying=data["is_decaying"],
            decay_start_time=dt_util.parse_datetime(data["decay_start_time"])
            if data["decay_start_time"]
            else None,
            decay_start_probability=data["decay_start_probability"],
            decay_enabled=data["decay_enabled"],
            decay_factor=data["decay_factor"],
        )

    def update_decay(
        self,
        current_probability: float,
        previous_probability: float,
        decay_window: int,
    ) -> tuple[float, float]:
        """Calculate decay factor and apply it to probability.

        This implements the decay mechanism as described in the documentation:
        1. Decay starts ONLY when probability decreases from previous calculation
        2. When decay starts, we record the time and the probability BEFORE the decrease
        3. We then exponentially decay from that starting probability over time
        4. The final probability can never go below what current sensors suggest (floor value)
        5. Decay stops when probability increases/stays same or reaches minimum

        Args:
            current_probability: The probability calculated from current sensor states
            previous_probability: The probability from the previous calculation cycle
            decay_window: Decay window in seconds

        Returns:
            Tuple of (final_probability, decay_factor)

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

        # --- CONDITION 1: Probability increased or stayed the same ---
        # When this happens, we stop any active decay and return to normal operation
        if current_probability >= previous_probability:
            if self.is_decaying:  # Log only if decay was previously active
                _LOGGER.debug(
                    "Probability increased/stable: %.3f -> %.3f. Stopping decay",
                    previous_probability,
                    current_probability,
                )
            # Reset decay state completely
            self._reset_decay_state()
            self.decay_factor = 1.0
            return current_probability, self.decay_factor

        # --- CONDITION 2: Probability decreased, decay wasn't active before ---
        # This is when we START decay - record the starting point and begin the decay process
        if not self.is_decaying:
            _LOGGER.debug(
                "Probability decreased: %.3f -> %.3f. Starting decay",
                previous_probability,
                current_probability,
            )
            # Initialize decay with the probability BEFORE the decrease as starting point
            self.decay_start_time = dt_util.utcnow()
            self.is_decaying = True
            self.decay_start_probability = previous_probability
            self.decay_factor = 1.0

            # For the first decay cycle, we maintain the previous probability
            # and begin decaying from there in subsequent cycles
            return previous_probability, self.decay_factor

        # --- CONDITION 3: Probability decreased, decay was already active ---
        # Continue the existing decay process
        if not self.decay_start_time or self.decay_start_probability is None:
            _LOGGER.error(
                "Inconsistent decay state: is_decaying=True but missing start time or probability. Resetting"
            )
            # Recover by starting fresh decay
            self.decay_start_time = dt_util.utcnow()
            self.is_decaying = True
            self.decay_start_probability = previous_probability
            self.decay_factor = 1.0
            return previous_probability, self.decay_factor

        # --- Calculate and apply exponential decay ---
        try:
            elapsed = (dt_util.utcnow() - self.decay_start_time).total_seconds()
            self.decay_factor = self._calculate_decay_factor(elapsed, decay_window)

            # Apply decay to the STARTING probability (when decay began)
            potential_decayed_prob = self._apply_decay_factor(
                self.decay_start_probability, self.decay_factor
            )

            # FLOOR VALUE: Never go below what current sensors suggest
            # This ensures that if sensors become more active during decay,
            # we respect that increased activity
            final_probability = max(potential_decayed_prob, current_probability)

            # Enhanced decay completion detection
            if self.is_decay_complete(final_probability):
                self._reset_decay_state()
                return MIN_PROBABILITY, self.decay_factor

            # Log decay progress for debugging
            _LOGGER.debug(
                "Decay active: elapsed=%.1fs, factor=%.3f, start=%.3f -> decayed=%.3f (floor=%.3f)",
                elapsed,
                self.decay_factor,
                self.decay_start_probability,
                final_probability,
                current_probability,
            )

        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error in decay calculation: %s", err)
            # Reset to safe state on calculation error
            self._reset_decay_state()
            self.decay_factor = 1.0
            return current_probability, self.decay_factor

        else:
            return final_probability, self.decay_factor

    def _reset_decay_state(self) -> None:
        """Reset decay state to inactive."""
        self.is_decaying = False
        self.decay_start_time = None
        self.decay_start_probability = MIN_PROBABILITY

    def reset(self) -> None:
        """Reset decay state to defaults."""
        self._reset_decay_state()
        self.decay_enabled = True
        self.decay_factor = 1.0

    def _validate_probabilities(
        self, current_probability: float, previous_probability: float
    ) -> bool:
        """Validate probability values.

        Args:
            current_probability: The current probability value
            previous_probability: The previous probability value

        Returns:
            True if both probabilities are valid (between 0 and 1)

        """
        return all(
            isinstance(p, (int, float)) and 0 <= p <= 1
            for p in (current_probability, previous_probability)
        )

    def _calculate_decay_factor(
        self, elapsed_seconds: float, decay_window: int
    ) -> float:
        """Calculate the decay factor based on elapsed time.

        Args:
            elapsed_seconds: Time elapsed since decay started
            decay_window: Decay window in seconds

        Returns:
            Decay factor between 0 and 1

        Raises:
            ValueError: If elapsed_seconds is negative

        """
        if elapsed_seconds < 0:
            raise ValueError(f"Elapsed time cannot be negative: {elapsed_seconds}")

        if elapsed_seconds == 0:
            return 1.0

        exponent = -DECAY_LAMBDA * (elapsed_seconds / decay_window)
        return math.exp(exponent)

    def _apply_decay_factor(self, probability: float, decay_factor: float) -> float:
        """Apply decay factor to probability with bounds checking.

        Args:
            probability: Original probability value
            decay_factor: Decay factor to apply (0-1)

        Returns:
            Decayed probability value

        Raises:
            ValueError: If inputs are invalid

        """
        if not (0 <= probability <= 1):
            raise ValueError(f"Probability must be between 0 and 1: {probability}")
        if not (0 <= decay_factor <= 1):
            raise ValueError(f"Decay factor must be between 0 and 1: {decay_factor}")

        result = probability * decay_factor
        return validate_prob(result)

    def should_start_decay(
        self, previous_is_active: bool, current_is_active: bool
    ) -> bool:
        """Determine if decay should start based on state transition.

        Args:
            previous_is_active: Previous is_active state
            current_is_active: Current is_active state

        Returns:
            True if decay should start

        """
        return (
            previous_is_active is True
            and current_is_active is False
            and not self.is_decaying
            and self.decay_enabled
        )

    def should_stop_decay(
        self, previous_is_active: bool, current_is_active: bool
    ) -> bool:
        """Determine if decay should stop based on state transition.

        Args:
            previous_is_active: Previous is_active state
            current_is_active: Current is_active state

        Returns:
            True if decay should stop

        """
        return (
            self.is_decaying
            and previous_is_active is False
            and current_is_active is True
        )

    def start_decay(self, starting_probability: float) -> None:
        """Start decay process with given starting probability.

        Args:
            starting_probability: Probability to start decay from

        """
        if not self.decay_enabled:
            return

        self.is_decaying = True
        self.decay_start_time = dt_util.utcnow()
        self.decay_start_probability = validate_prob(starting_probability)
        self.decay_factor = 1.0

    def stop_decay(self) -> None:
        """Stop decay process and reset state."""
        self.is_decaying = False
        self.decay_start_time = None
        self.decay_start_probability = MIN_PROBABILITY
        self.decay_factor = 1.0

    def is_decay_complete(self, current_probability: float) -> bool:
        """Check if decay has completed.

        Args:
            current_probability: Current probability value

        Returns:
            True if decay is complete

        """
        if not self.is_decaying:
            return True

        if not self.decay_start_time:
            return True

        # 1. Reached absolute minimum
        if current_probability <= MIN_PROBABILITY:
            return True

        # 2. Reached practical completion threshold (2%)
        if current_probability <= DECAY_COMPLETION_THRESHOLD:
            return True

        # 3. Decay factor has become negligible (1%)
        if self.decay_factor <= DECAY_FACTOR_THRESHOLD:
            return True

        return False
