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
from .types import ProbabilityState

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
        self, probability_state: ProbabilityState
    ) -> tuple[float, float]:
        """Calculate decay factor and apply it to probability.

        Args:
            probability_state: The current probability state

        Returns:
            Tuple of (decayed_probability, decay_factor)

        Raises:
            ValueError: If probabilities are invalid or threshold is invalid
            ConfigurationError: If decay parameters are invalid

        """
        # Validate inputs
        if not self._validate_probabilities(probability_state):
            raise ValueError("Probabilities and threshold must be between 0 and 1")

        if not self.decay_enabled:
            _LOGGER.debug("Decay disabled")
            probability_state.decaying = False
            probability_state.decay_start_time = None
            return probability_state.probability, 1.0

        if probability_state.probability >= probability_state.previous_probability:
            if probability_state.decay_start_time:
                _LOGGER.debug(
                    "Probability increased: %.3f -> %.3f",
                    probability_state.previous_probability,
                    probability_state.probability,
                )
            probability_state.decay_start_time = None
            probability_state.decaying = False
            return probability_state.probability, 1.0

        if not probability_state.decay_start_time:
            _LOGGER.debug(
                "Starting decay: %.3f -> %.3f",
                probability_state.previous_probability,
                probability_state.probability,
            )
            probability_state.decay_start_time = datetime.now()
            probability_state.decaying = True
            return probability_state.previous_probability, 1.0

        try:
            elapsed = (
                datetime.now() - probability_state.decay_start_time
            ).total_seconds()
            # Calculate exponential decay factor based on elapsed time and window
            decay_factor = self._calculate_decay_factor(elapsed)

            decayed_probability = self._apply_decay_factor(
                probability_state.previous_probability, decay_factor
            )

            # Continue decay even if below threshold, only stop at MIN_PROBABILITY
            if decayed_probability <= MIN_PROBABILITY:
                _LOGGER.debug("Decay complete: reached minimum probability")
                probability_state.decay_start_time = None
                probability_state.decaying = False
                return MIN_PROBABILITY, decay_factor

            _LOGGER.debug(
                "Decay: t=%.1fs f=%.3f p=%.3f",
                elapsed,
                decay_factor,
                decayed_probability,
            )

            probability_state.decaying = True

        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error in decay calculation: %s", err)
            probability_state.decaying = False
            probability_state.decay_start_time = None
            return probability_state.probability, 1.0
        else:
            return decayed_probability, decay_factor

    def _validate_probabilities(self, probability_state: ProbabilityState) -> bool:
        """Validate probability values.

        Args:
            probability_state: The probability state to validate

        Returns:
            True if all probabilities are valid, False otherwise

        """
        return all(
            0 <= p <= 1
            for p in (
                probability_state.probability,
                probability_state.previous_probability,
                probability_state.threshold,
            )
        )

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

    def reset(self) -> None:
        """Reset the decay handler configuration to defaults."""
        self.decay_enabled = DEFAULT_DECAY_ENABLED
        self.decay_window = DEFAULT_DECAY_WINDOW
