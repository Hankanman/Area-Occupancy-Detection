"""Handle decay calculations for Area Occupancy Detection."""

from __future__ import annotations

import logging
import math
from datetime import datetime

from .const import (
    DECAY_LAMBDA,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
)

_LOGGER = logging.getLogger(__name__)


class DecayHandler:
    """Handle decay calculations for probability values."""

    def __init__(self, config: dict) -> None:
        """Initialize the decay handler."""
        self.config = config
        self.decay_enabled = self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        self.decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)
        self.decay_start_time: datetime | None = None

    def calculate_decay(
        self,
        current_probability: float,
        previous_probability: float,
        threshold: float,
        now: datetime,
    ) -> tuple[float, float]:
        """Calculate decay factor and apply it to probability.

        Args:
            current_probability: The current calculated probability
            previous_probability: The previous probability value
            threshold: The threshold for occupancy
            now: Current timestamp

        Returns:
            Tuple of (decayed_probability, decay_factor)
        """
        if not self.decay_enabled:
            return current_probability, 1.0

        if current_probability >= previous_probability:
            self.decay_start_time = None
            return current_probability, 1.0

        if not self.decay_start_time:
            self.decay_start_time = now
            return previous_probability, 1.0

        elapsed = (now - self.decay_start_time).total_seconds()
        decay_factor = math.exp(-DECAY_LAMBDA * (elapsed / self.decay_window))

        decayed_probability = max(
            MIN_PROBABILITY, min(previous_probability * decay_factor, MAX_PROBABILITY)
        )

        if decayed_probability < threshold:
            _LOGGER.debug("Decaying probability below threshold; resetting decay")
            self.decay_start_time = None
            return current_probability, decay_factor

        _LOGGER.debug("Decaying probability above threshold; continuing decay")

        return decayed_probability, decay_factor

    def get_decay_status(self, decay_factor: float) -> dict:
        """Get the current decay status.

        Args:
            decay_factor: The current decay factor

        Returns:
            Dictionary containing decay status information
        """
        if not self.decay_enabled or decay_factor == 1.0:
            return {"global_decay": 0.0}
        return {"global_decay": round(1.0 - decay_factor, 4)}

    def update_config(self, config: dict) -> None:
        """Update configuration values."""
        self.config = config
        self.decay_enabled = self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)
        self.decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)
