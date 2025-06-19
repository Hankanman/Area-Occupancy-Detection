"""Decay model for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

DEFAULT_HALF_LIFE = 720.0  # seconds - default to social area (12 minutes)


@dataclass
class Decay:
    """Decay model for Area Occupancy Detection."""

    last_trigger_ts: float = field(default_factory=time.time)  # UNIX epoch seconds
    half_life: float = DEFAULT_HALF_LIFE  # purpose-based half-life
    is_decaying: bool = False

    def update_half_life(self, new_half_life: float) -> None:
        """Update the half-life value."""
        self.half_life = new_half_life

    @property
    def decay_factor(self) -> float:
        """Freshness of the last motion event in [0,1]."""
        if not self.is_decaying:
            return 1.0
        age = time.time() - self.last_trigger_ts
        if age <= 0:
            return 1.0
        factor = math.pow(0.5, age / self.half_life)
        # Auto-stop decay when factor becomes negligible
        if factor < 0.05:
            self.is_decaying = False
            return 0.0
        return factor

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

    def should_start_decay(
        self, previous_evidence: bool, current_evidence: bool
    ) -> bool:
        """Determine if decay should start based on state transition.

        Args:
            previous_evidence: Previous evidence state
            current_evidence: Current evidence state

        Returns:
            True if decay should start

        """
        return (
            previous_evidence is True
            and current_evidence is False
            and not self.is_decaying
        )

    def should_stop_decay(
        self, previous_evidence: bool, current_evidence: bool
    ) -> bool:
        """Determine if decay should stop based on state transition.

        Args:
            previous_evidence: Previous evidence state
            current_evidence: Current evidence state

        Returns:
            True if decay should stop

        """
        return (
            self.is_decaying and previous_evidence is False and current_evidence is True
        )

    def is_decay_complete(self, current_probability: float) -> bool:
        """Check if decay has completed.

        Args:
            current_probability: Current probability value

        Returns:
            True if decay is complete

        """
        if not self.is_decaying:
            return True

        if not self.last_trigger_ts:
            return True

        # 1. Reached absolute minimum
        if current_probability <= 0.0:
            return True

        # 2. Reached practical completion threshold (2%)
        if current_probability <= 0.02:
            return True

        # 3. Decay factor has become negligible (1%)
        if self.decay_factor <= 0.01:
            return True

        return False
