"""Decay model for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime

from homeassistant.util import dt as dt_util

from ..utils import ensure_timezone_aware
from .purpose import PURPOSE_DEFINITIONS, AreaPurpose


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
            self.decay_start = ensure_timezone_aware(decay_start)
        else:
            self.decay_start = dt_util.utcnow()

        self._base_half_life = half_life
        self.is_decaying = is_decaying
        self.purpose = purpose
        self.sleep_start = sleep_start
        self.sleep_end = sleep_end

    @property
    def half_life(self) -> float:
        """Return the effective half-life based on purpose and time of day."""
        # If purpose is not sleeping, use base half-life
        if self.purpose != AreaPurpose.SLEEPING:
            return self._base_half_life

        # If sleep times are not configured, use base half-life
        if not self.sleep_start or not self.sleep_end:
            return self._base_half_life

        try:
            # Parse sleep times
            now = dt_util.utcnow()
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

            # Outside sleep window, behave like RELAXING
            return PURPOSE_DEFINITIONS[AreaPurpose.RELAXING].half_life

        except (ValueError, TypeError):
            # Fallback on error
            return self._base_half_life

    @property
    def decay_factor(self) -> float:
        """Freshness of last motion edge ∈[0,1]; auto-stops below 5 %."""
        if not self.is_decaying:
            return 1.0

        age = (dt_util.utcnow() - self.decay_start).total_seconds()
        factor = float(0.5 ** (age / self.half_life))
        if factor < 0.05:  # practical zero
            self.is_decaying = False
            return 0.0
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
