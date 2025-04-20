"""Handle decay calculations for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime
import logging
import math
from typing import Final

from .const import (
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    DECAY_LAMBDA,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from .exceptions import ConfigurationError

_LOGGER = logging.getLogger(__name__)

# Validation constants
MIN_DECAY_WINDOW: Final[int] = 1  # Minimum decay window in seconds
MAX_DECAY_WINDOW: Final[int] = 3600  # Maximum decay window in seconds


class DecayHandler:
    """Handle decay calculations for probability values.

    This class manages the exponential decay of probability values over time.
    It implements a decay model where the probability decreases exponentially
    based on elapsed time since the last probability increase.

    Attributes:
        decay_enabled: Whether decay is enabled
        decay_window: Time window in seconds over which decay occurs
        config: Configuration dictionary containing decay settings

    """

    def __init__(self, config: dict) -> None:
        """Initialize the decay handler.

        Args:
            config: Configuration dictionary containing decay settings

        Raises:
            ConfigurationError: If decay window is invalid

        """
        self.config = config
        self.decay_enabled = self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        self.decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)

        # Validate decay window
        if not isinstance(self.decay_window, (int, float)):
            raise ConfigurationError(
                f"Decay window must be a number, got {type(self.decay_window)}"
            )
        if not MIN_DECAY_WINDOW <= self.decay_window <= MAX_DECAY_WINDOW:
            raise ConfigurationError(
                f"Decay window must be between {MIN_DECAY_WINDOW} and {MAX_DECAY_WINDOW} seconds"
            )

        _LOGGER.debug(
            "Initialized: enabled=%s window=%ds",
            self.decay_enabled,
            self.decay_window,
        )

    def calculate_decay(
        self,
        current_probability: float,
        previous_probability: float,
        is_decaying: bool,
        decay_start_time: datetime | None,
        decay_start_probability: float | None,
    ) -> tuple[float, float, bool, datetime | None, float | None]:
        """Calculate decay factor and apply it to probability.

        Args:
            current_probability: The current probability value (used to detect increase)
            previous_probability: The probability value from the previous step
            is_decaying: Whether decay is currently active
            decay_start_time: Timestamp when decay started, if active
            decay_start_probability: Probability when decay started, if active

        Returns:
            Tuple of (
                decayed_probability: The new probability after applying decay
                decay_factor: The calculated decay factor (1.0 if no decay applied)
                new_is_decaying: The updated decaying status (bool)
                new_decay_start_time: The updated decay start time (datetime | None)
                new_decay_start_probability: The updated probability when decay started (float | None)
            )

        Raises:
            ValueError: If probabilities are invalid
            ConfigurationError: If decay parameters are invalid

        """
        # Validate inputs (only probabilities that are used for calculation)
        if not self._validate_probabilities(current_probability, previous_probability):
            raise ValueError("Probabilities must be between 0 and 1")

        if not self.decay_enabled:
            _LOGGER.debug("Decay disabled")
            # No decay, return current state
            return current_probability, 1.0, False, None, None

        # --- Determine new decay state ---
        new_decay_start_time = decay_start_time
        new_is_decaying = is_decaying
        new_decay_start_probability = decay_start_probability

        # Condition 1: Probability increased or stayed the same
        if current_probability >= previous_probability:
            if is_decaying:  # Log only if decay was previously active
                _LOGGER.debug(
                    "Probability increased/stable: %.3f -> %.3f. Stopping decay",
                    previous_probability,
                    current_probability,
                )
            new_decay_start_time = None
            new_is_decaying = False
            new_decay_start_probability = None  # Clear start probability
            # Return the *current* probability as decay stopped/didn't start
            return (
                current_probability,
                1.0,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )

        # Condition 2: Probability decreased, decay wasn't active before
        if not is_decaying:
            _LOGGER.debug(
                "Probability decreased: %.3f -> %.3f. Starting decay",
                previous_probability,
                current_probability,
            )
            new_decay_start_time = datetime.now()
            new_is_decaying = True
            # Store the probability from *before* the decrease as the start point
            new_decay_start_probability = previous_probability
            # Return the *previous* probability as decay starts now (no decay applied yet)
            return (
                previous_probability,
                1.0,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )

        # Condition 3: Probability decreased, decay was already active
        # decay_start_time and decay_start_probability must exist if is_decaying is True
        if not decay_start_time or decay_start_probability is None:
            # Should not happen if is_decaying is True, but handle defensively
            _LOGGER.error(
                "Inconsistent state: is_decaying=True but decay_start_time (%s) or decay_start_probability (%s) is None. Resetting decay",
                decay_start_time,
                decay_start_probability,
            )
            new_decay_start_time = datetime.now()
            new_is_decaying = True
            new_decay_start_probability = (
                previous_probability  # Use previous as fallback
            )
            return (
                previous_probability,
                1.0,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )

        # --- Calculate decay based on existing start time and start probability ---
        try:
            elapsed = (datetime.now() - decay_start_time).total_seconds()
            decay_factor = self._calculate_decay_factor(elapsed)

            # Calculate the potential decayed value based on the probability when decay *started*
            potential_decayed_prob = self._apply_decay_factor(
                decay_start_probability, decay_factor
            )

            # The final decayed probability should not be lower than what current sensors suggest
            decayed_probability = max(potential_decayed_prob, current_probability)

            # Check if decay reached minimum threshold
            if decayed_probability <= MIN_PROBABILITY:
                _LOGGER.debug("Decay complete: reached minimum probability")
                new_decay_start_time = None
                new_is_decaying = False
                new_decay_start_probability = None  # Clear start probability
                # Return MIN_PROBABILITY, the calculated factor, and reset decay state
                return (
                    MIN_PROBABILITY,
                    decay_factor,
                    new_is_decaying,
                    new_decay_start_time,
                    new_decay_start_probability,
                )

            # Decay continues
            _LOGGER.debug(
                "Decay active: t=%.1fs factor=%.3f p_start=%.3f -> p_decayed=%.3f (floor=%.3f)",
                elapsed,
                decay_factor,
                decay_start_probability,  # Log the start probability
                decayed_probability,
                current_probability,  # Log the floor
            )
            # new_is_decaying remains True
            # new_decay_start_time remains the original start time
            # new_decay_start_probability remains the original start probability

        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error in decay calculation: %s", err)
            new_is_decaying = False
            new_decay_start_time = None
            new_decay_start_probability = None
            # Return the original current probability on error, factor 1.0, reset decay
            return (
                current_probability,
                1.0,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )
        else:
            # Return decayed probability, factor, and *current* decay state (still active)
            return (
                decayed_probability,
                decay_factor,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,  # Return the existing start probability
            )

    def _validate_probabilities(
        self, current_probability: float, previous_probability: float
    ) -> bool:
        """Validate probability values.

        Args:
            current_probability: The current probability value
            previous_probability: The previous probability value

        Returns:
            True if all probabilities are valid, False otherwise

        """
        return all(0 <= p <= 1 for p in (current_probability, previous_probability))

    def _calculate_decay_factor(self, elapsed_seconds: float) -> float:
        """Calculate the decay factor based on elapsed time.

        Args:
            elapsed_seconds: Number of seconds elapsed since decay start

        Returns:
            Calculated decay factor

        """
        return math.exp(-DECAY_LAMBDA * (elapsed_seconds / self.decay_window))

    def _apply_decay_factor(self, probability: float, decay_factor: float) -> float:
        """Apply decay factor to probability with bounds checking.

        Args:
            probability: The probability to decay
            decay_factor: The decay factor to apply

        Returns:
            The decayed probability, bounded between MIN_PROBABILITY and MAX_PROBABILITY

        """
        return max(
            MIN_PROBABILITY,
            min(probability * decay_factor, MAX_PROBABILITY),
        )
