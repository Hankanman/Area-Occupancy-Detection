"""Handle decay calculations for Area Occupancy Detection."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Tuple

from .const import (
    DECAY_LAMBDA,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
)
from .exceptions import ConfigurationError

from .types import ProbabilityState

_LOGGER = logging.getLogger(__name__)


class DecayHandler:
    """Handle decay calculations for probability values.

    This class manages the exponential decay of probability values over time.
    It implements a decay model where the probability decreases exponentially
    based on elapsed time since the last probability increase.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the decay handler.

        Args:
            config: Configuration dictionary containing decay settings
        """
        self.config = config
        self.decay_enabled = self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        self.decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)
        self.decay_start_time: datetime | None = None

        if self.decay_window <= 0:
            raise ConfigurationError(
                f"Decay window must be positive, got {self.decay_window}"
            )

        _LOGGER.debug(
            "DecayHandler initialized with enabled=%s, window=%d seconds",
            self.decay_enabled,
            self.decay_window,
        )

    def calculate_decay(
        self, probability_state: ProbabilityState
    ) -> Tuple[float, float]:
        """Calculate decay factor and apply it to probability.

        Args:
            probability_state: The current probability state

        Returns:
            Tuple of (decayed_probability, decay_factor)

        Raises:
            ValueError: If probabilities are invalid or threshold is invalid
        """
        # Validate inputs
        if not all(
            0 <= p <= 1
            for p in (
                probability_state.probability,
                probability_state.previous_probability,
                probability_state.threshold,
            )
        ):
            raise ValueError("Probabilities and threshold must be between 0 and 1")

        if not self.decay_enabled:
            _LOGGER.debug("Decay disabled, returning current probability")
            probability_state.decaying = False
            return probability_state.probability, 1.0

        if probability_state.probability >= probability_state.previous_probability:
            if self.decay_start_time:
                _LOGGER.debug(
                    "Probability increased (%.3f >= %.3f), resetting decay",
                    probability_state.probability,
                    probability_state.previous_probability,
                )
            self.decay_start_time = None
            probability_state.decaying = False
            return probability_state.probability, 1.0

        if not self.decay_start_time:
            _LOGGER.debug(
                "Starting decay from %.3f to %.3f",
                probability_state.previous_probability,
                probability_state.probability,
            )
            self.decay_start_time = datetime.now()
            probability_state.decaying = True
            return probability_state.previous_probability, 1.0

        try:
            elapsed = (datetime.now() - self.decay_start_time).total_seconds()
            # Calculate exponential decay factor based on elapsed time and window
            decay_factor = math.exp(-DECAY_LAMBDA * (elapsed / self.decay_window))

            decayed_probability = max(
                MIN_PROBABILITY,
                min(
                    probability_state.previous_probability * decay_factor,
                    MAX_PROBABILITY,
                ),
            )

            if decayed_probability < probability_state.threshold:
                _LOGGER.debug(
                    "Decayed probability (%.3f) fell below threshold (%.3f), resetting decay",
                    decayed_probability,
                    probability_state.threshold,
                )
                self.decay_start_time = None
                probability_state.decaying = False
                return probability_state.probability, decay_factor

            _LOGGER.debug(
                "Decay progress: elapsed=%.1fs, factor=%.3f, probability=%.3f",
                elapsed,
                decay_factor,
                decayed_probability,
            )

            probability_state.decaying = True
            return decayed_probability, decay_factor

        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error in decay calculation: %s", err, exc_info=True)
            probability_state.decaying = False
            return probability_state.probability, 1.0

    def reset(self) -> None:
        """Reset the decay handler state."""
        self.decay_start_time = None
