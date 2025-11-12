"""Decay model for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime

from homeassistant.util import dt as dt_util

from ..utils import ensure_timezone_aware


class Decay:
    """Decay model for Area Occupancy Detection."""

    def __init__(
        self,
        half_life: float,
        decay_start: datetime | None = None,
        is_decaying: bool = False,
    ) -> None:
        """Initialize the decay model.

        Args:
            decay_start: When decay began. Defaults to current time if None.
            half_life: Purpose-based half-life in seconds.
            is_decaying: Whether decay is currently active.
        """
        # Ensure decay_start is timezone-aware
        if decay_start is not None:
            self.decay_start = ensure_timezone_aware(decay_start)
        else:
            self.decay_start = dt_util.utcnow()

        self.half_life = half_life
        self.is_decaying = is_decaying

    @property
    def decay_factor(self) -> float:
        """Freshness of last motion edge ∈[0,1]; auto-stops below 5 %."""
        if not self.is_decaying:
            return 1.0

        # Ensure decay_start is timezone-aware to avoid subtraction errors
        decay_start_aware = ensure_timezone_aware(self.decay_start)
        age = (dt_util.utcnow() - decay_start_aware).total_seconds()
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
