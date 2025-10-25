"""Decay model for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from homeassistant.util import dt as dt_util

from ..utils import ensure_timezone_aware

# Note: Actual half-life is purpose-driven via PurposeManager.
# This fallback is used only when no purpose is configured.
DEFAULT_HALF_LIFE = 30.0  # seconds


@dataclass
class Decay:
    """Decay model for Area Occupancy Detection."""

    decay_start: datetime = field(default_factory=dt_util.utcnow)  # when decay began
    half_life: float = DEFAULT_HALF_LIFE  # purpose-based half-life
    is_decaying: bool = False

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

    @classmethod
    def create(
        cls,
        decay_start: datetime | None = None,
        half_life: float | None = None,
        is_decaying: bool | None = None,
    ) -> Decay:
        """Create a Decay instance with optional parameters."""
        # Ensure decay_start is timezone-aware if provided
        if decay_start is not None:
            decay_start = ensure_timezone_aware(decay_start)

        return cls(
            decay_start=decay_start if decay_start is not None else dt_util.utcnow(),
            half_life=half_life if half_life is not None else DEFAULT_HALF_LIFE,
            is_decaying=is_decaying if is_decaying is not None else False,
        )
