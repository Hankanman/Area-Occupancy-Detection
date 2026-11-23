"""Shadow-mode evidence for away mode lowering occupancy (#584).

While the away-mode entity (#573) is on, the plan is for every room's
prior to drop to ``AWAY_PRIOR`` so a TV on a schedule or a drifting sensor
can't hold an empty house occupied. Before that is allowed to change any
reading, this module records what it *would* do, per area:

* how long the household was away, and how much of that time the area
  was really occupied (motion, media or sleep): their ratio is the
  learned away prior, the number the fixed ``AWAY_PRIOR`` is checked
  against;
* how long the away-adjusted decision would have differed from the live
  one, and how much of that time someone was really there. That last
  number is the one that matters: time away mode would have hidden a
  real person.

Shadow-mode contract: nothing here is read by the probability path.
Like ``online_prior.py`` it accumulates per tick, crediting the time
since the previous tick to that tick's state, ignores gaps longer than
``MAX_TICK_GAP_SECONDS`` (restarts, stalls), and persists via the HA
storage helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..const import AWAY_PRIOR
from ..time_utils import ensure_utc_datetime
from .online_prior import MAX_TICK_GAP_SECONDS

# Away time needed before the learned away prior is reported: an hour of
# being out is one errand, not a pattern.
AWAY_SHADOW_MIN_AWAY_SECONDS = 3600.0


@dataclass
class AwayShadowState:
    """Serializable accumulators for one area."""

    away_seconds: float = 0.0
    occupied_away_seconds: float = 0.0
    differ_seconds: float = 0.0
    differ_present_seconds: float = 0.0
    last_tick: datetime | None = None
    last_away: bool = False
    last_present: bool = False
    last_differ: bool = False

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "away_seconds": self.away_seconds,
            "occupied_away_seconds": self.occupied_away_seconds,
            "differ_seconds": self.differ_seconds,
            "differ_present_seconds": self.differ_present_seconds,
            "last_tick": self.last_tick.isoformat() if self.last_tick else None,
            "last_away": self.last_away,
            "last_present": self.last_present,
            "last_differ": self.last_differ,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AwayShadowState:
        """Restore from storage; malformed payloads fall back to empty."""
        try:
            return cls(
                away_seconds=float(data.get("away_seconds", 0.0)),
                occupied_away_seconds=float(data.get("occupied_away_seconds", 0.0)),
                differ_seconds=float(data.get("differ_seconds", 0.0)),
                differ_present_seconds=float(data.get("differ_present_seconds", 0.0)),
                last_tick=(
                    ensure_utc_datetime(datetime.fromisoformat(raw))
                    if (raw := data.get("last_tick"))
                    else None
                ),
                last_away=bool(data.get("last_away", False)),
                last_present=bool(data.get("last_present", False)),
                last_differ=bool(data.get("last_differ", False)),
            )
        except (AttributeError, TypeError, ValueError):
            return cls()


class AwayShadow:
    """Accumulates the away-mode shadow evidence for one area."""

    def __init__(self, state: AwayShadowState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state or AwayShadowState()

    def observe(
        self,
        *,
        now: datetime,
        away: bool,
        present: bool,
        live_occupied: bool,
        away_occupied: bool,
    ) -> None:
        """Record one tick.

        Args:
            now: The tick time.
            away: Whether the away-mode entity is on.
            present: Whether a ground-truth sensor (motion, media, sleep)
                is active: someone is really there.
            live_occupied: The live occupancy decision.
            away_occupied: The decision with the away prior instead.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        if state.last_tick is not None:
            gap = (now - state.last_tick).total_seconds()
            if 0 < gap <= MAX_TICK_GAP_SECONDS and state.last_away:
                state.away_seconds += gap
                if state.last_present:
                    state.occupied_away_seconds += gap
                if state.last_differ:
                    state.differ_seconds += gap
                    if state.last_present:
                        state.differ_present_seconds += gap
        state.last_tick = now
        state.last_away = away
        state.last_present = present
        state.last_differ = away and live_occupied != away_occupied

    def learned_away_prior(self) -> float | None:
        """Share of away time the area was really occupied, once measurable."""
        if self.state.away_seconds < AWAY_SHADOW_MIN_AWAY_SECONDS:
            return None
        return self.state.occupied_away_seconds / self.state.away_seconds

    def snapshot(self) -> dict:
        """The JSON-safe diagnostics block."""
        state = self.state
        learned = self.learned_away_prior()
        return {
            "shadow_mode": True,
            "engaged": False,
            "away_prior": AWAY_PRIOR,
            "away_hours": round(state.away_seconds / 3600, 2),
            "learned_away_prior": round(learned, 4) if learned is not None else None,
            "would_differ_hours": round(state.differ_seconds / 3600, 2),
            # Time away mode would have read empty while someone was there:
            # the number that must stay near zero before it goes live.
            "would_hide_presence_hours": round(state.differ_present_seconds / 3600, 2),
        }
