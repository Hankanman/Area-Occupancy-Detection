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

_LOGGER = logging.getLogger(__name__)


class DecayHandler:
    """Handle decay calculations for probability values."""

    def __init__(self, config: dict) -> None:
        """Initialize the decay handler."""
        self.config = config
        self.decay_enabled = self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        self.decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)
        self.decay_start_time: datetime | None = None

        if self.decay_window <= 0:
            raise ConfigurationError(f"Decay window must be positive, got {self.decay_window}")

        _LOGGER.debug(
            "DecayHandler initialized: enabled=%s, window=%d seconds",
            self.decay_enabled,
            self.decay_window
        )

    def calculate_decay(
        self,
        current_probability: float,
        previous_probability: float,
        threshold: float,
        now: datetime,
    ) -> Tuple[float, float]:
        """Calculate decay factor and apply it to probability.

        Args:
            current_probability: The current calculated probability
            previous_probability: The previous probability value
            threshold: The threshold for occupancy
            now: Current timestamp

        Returns:
            Tuple of (decayed_probability, decay_factor)

        Raises:
            ValueError: If probabilities are invalid or threshold is invalid
        """
        # Validate inputs
        if not all(0 <= p <= 1 for p in (current_probability, previous_probability, threshold)):
            raise ValueError("Probabilities and threshold must be between 0 and 1")

        if not self.decay_enabled:
            _LOGGER.debug("Decay disabled, returning current probability")
            return current_probability, 1.0

        if current_probability >= previous_probability:
            if self.decay_start_time:
                _LOGGER.debug(
                    "Probability increased (%.3f >= %.3f), resetting decay",
                    current_probability,
                    previous_probability
                )
            self.decay_start_time = None
            return current_probability, 1.0

        if not self.decay_start_time:
            _LOGGER.debug(
                "Starting decay from %.3f to %.3f",
                previous_probability,
                current_probability
            )
            self.decay_start_time = now
            return previous_probability, 1.0

        try:
            elapsed = (now - self.decay_start_time).total_seconds()
            decay_factor = math.exp(-DECAY_LAMBDA * (elapsed / self.decay_window))

            decayed_probability = max(
                MIN_PROBABILITY, min(previous_probability * decay_factor, MAX_PROBABILITY)
            )

            if decayed_probability < threshold:
                _LOGGER.debug(
                    "Decayed probability %.3f below threshold %.3f, resetting decay",
                    decayed_probability,
                    threshold
                )
                self.decay_start_time = None
                return current_probability, decay_factor

            _LOGGER.debug(
                "Decay progress: elapsed=%.1fs, factor=%.3f, prob=%.3f",
                elapsed,
                decay_factor,
                decayed_probability
            )

            return decayed_probability, decay_factor

        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error in decay calculation: %s", err, exc_info=True)
            return current_probability, 1.0

    def get_decay_status(self, decay_factor: float) -> dict:
        """Get the current decay status.

        Args:
            decay_factor: The current decay factor

        Returns:
            Dictionary containing decay status information

        Raises:
            ValueError: If decay_factor is invalid
        """
        if not 0 <= decay_factor <= 1:
            raise ValueError("Decay factor must be between 0 and 1")

        if not self.decay_enabled or decay_factor == 1.0:
            return {"global_decay": 0.0}
        return {"global_decay": round(1.0 - decay_factor, 4)}

    def update_config(self, config: dict) -> None:
        """Update configuration values.

        Args:
            config: New configuration dictionary

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        new_decay_enabled = self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        new_decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)

        if new_decay_window <= 0:
            raise ConfigurationError(f"Decay window must be positive, got {new_decay_window}")

        self.decay_enabled = new_decay_enabled
        self.decay_window = new_decay_window
        self.decay_start_time = None

        _LOGGER.debug(
            "DecayHandler config updated: enabled=%s, window=%d seconds",
            self.decay_enabled,
            self.decay_window
        )
