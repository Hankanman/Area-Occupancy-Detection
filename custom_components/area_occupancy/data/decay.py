"""Decay model for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime
import logging

from homeassistant.util import dt as dt_util

from ..time_utils import to_local, to_utc
from .purpose import Purpose

_LOGGER = logging.getLogger(__name__)


class Decay:
    """Decay model for Area Occupancy Detection."""

    def __init__(
        self,
        half_life: float,
        is_decaying: bool = False,
        decay_start: datetime | None = None,
        purpose: str | None = None,
        sleep_start: str | None = None,
        sleep_end: str | None = None,
    ) -> None:
        """Initialize the decay model.

        Args:
            half_life: Purpose-based half-life in seconds.
            is_decaying: Whether decay is currently active.
            decay_start: When decay began. Defaults to current time if None.
            purpose: Area purpose string.
            sleep_start: Sleep start time string (HH:MM:SS).
            sleep_end: Sleep end time string (HH:MM:SS).
        """
        # Ensure decay_start is timezone-aware
        if decay_start is not None:
            self.decay_start = to_utc(decay_start)
        else:
            self.decay_start = dt_util.utcnow()

        self._base_half_life = half_life
        self.is_decaying = is_decaying
        self._purpose = Purpose(purpose) if purpose is not None else None
        self.sleep_start = sleep_start
        self.sleep_end = sleep_end

    @property
    def purpose(self) -> Purpose | None:
        """Return the resolved Purpose instance, or None."""
        return self._purpose

    @property
    def half_life(self) -> float:
        """Return the effective half-life based on purpose and time of day."""
        # If no purpose or purpose has no awake_half_life, use base half-life
        if self._purpose is None or self._purpose.awake_half_life is None:
            return self._base_half_life

        # If sleep times are not configured, use base half-life
        if not self.sleep_start or not self.sleep_end:
            return self._base_half_life

        try:
            # Parse sleep times
            now = to_local(dt_util.utcnow())
            start_time = datetime.strptime(self.sleep_start, "%H:%M:%S").time()
            end_time = datetime.strptime(self.sleep_end, "%H:%M:%S").time()

            current_time = now.time()

            # Check if current time is within sleep window
            is_sleeping = False
            if start_time <= end_time:
                # Same day window (e.g., 13:00 to 15:00)
                is_sleeping = start_time <= current_time <= end_time
            else:
                # Overnight window (e.g., 23:00 to 07:00)
                is_sleeping = current_time >= start_time or current_time <= end_time

            if is_sleeping:
                # Use the configured half-life (should be high for sleeping)
                return self._base_half_life
        except (ValueError, TypeError):
            _LOGGER.exception("Error calculating half-life for sleeping purpose")
            return self._base_half_life

        # Outside sleep window, use the purpose's awake half-life
        return self._purpose.awake_half_life

    @property
    def decay_factor(self) -> float:
        """Freshness of last motion edge ∈[0,1].

        This is a pure read-only property that calculates the decay factor
        without modifying state. Use tick() to update state based on the
        decay factor.

        Returns:
            1.0 if not decaying or decay_start is in the future
            0.0 if half_life is invalid (zero or negative)
            Calculated factor otherwise (0.0 to 1.0)
        """
        if not self.is_decaying:
            return 1.0

        age = (dt_util.utcnow() - self.decay_start).total_seconds()

        # Handle negative age (decay_start in future) - no decay has occurred yet
        if age < 0:
            return 1.0

        # Handle zero or negative half_life - prevent division by zero
        if self.half_life <= 0:
            _LOGGER.warning(
                "Invalid half_life value %s detected, treating as immediate decay",
                self.half_life,
            )
            return 0.0

        factor = float(0.5 ** (age / self.half_life))
        # Return 0.0 when factor drops below practical threshold
        if factor < 0.05:
            return 0.0
        return factor

    def tick(self) -> float:
        """Update decay state and return current decay factor.

        This method should be called periodically (e.g., by the decay timer)
        to update the decay state. It stops decay when the factor drops below
        the practical threshold (5%) or when half_life is invalid.

        Returns:
            The current decay factor (0.0 to 1.0)
        """
        factor = self.decay_factor

        # Stop decay if factor has reached practical zero or half_life is invalid
        if self.is_decaying and factor <= 0.0:
            self.is_decaying = False

        return factor

    def start_decay(self) -> None:
        """Begin decay **only if not already running**."""
        if not self.is_decaying:
            self.is_decaying = True
            self.decay_start = dt_util.utcnow()

    def stop_decay(self) -> None:
        """Stop decay **only if already running**."""
        if self.is_decaying:
            self.is_decaying = False
