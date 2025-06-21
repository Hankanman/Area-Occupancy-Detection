"""Decay model for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

DEFAULT_HALF_LIFE = 30.0  # seconds - default to social area (12 minutes)


@dataclass
class Decay:
    """Decay model for Area Occupancy Detection."""

    last_trigger_ts: float = field(default_factory=time.time)  # UNIX epoch seconds
    half_life: float = DEFAULT_HALF_LIFE  # purpose-based half-life
    is_decaying: bool = False

    @property
    def decay_factor(self) -> float:
        """Freshness of last motion edge ∈[0,1]; auto-stops below 5 %."""
        if not self.is_decaying:
            return 1.0
        age = time.time() - self.last_trigger_ts
        factor = 0.5 ** (age / self.half_life)
        if factor < 0.05:  # practical zero
            self.is_decaying = False
            return 0.0
        return factor

    def start_decay(self) -> None:
        """Begin decay **only if not already running**."""
        if not self.is_decaying:
            self.is_decaying = True
            self.last_trigger_ts = time.time()

    def stop_decay(self) -> None:
        """Stop decay **only if already running**."""
        if self.is_decaying:
            self.is_decaying = False

    def to_dict(self) -> dict[str, Any]:
        """Convert decay to dictionary for storage."""
        return {
            "last_trigger_ts": self.last_trigger_ts,
            "half_life": self.half_life,
            "is_decaying": self.is_decaying,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decay:
        """Create decay from dictionary."""
        return Decay(
            last_trigger_ts=data.get("last_trigger_ts", time.time()),
            half_life=data.get("half_life", DEFAULT_HALF_LIFE),
            is_decaying=data.get("is_decaying", False),
        )
