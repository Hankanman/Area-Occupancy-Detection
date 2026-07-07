"""Shadow-mode online global-prior estimator (DB-retirement epic, #500).

Production computes the global prior by replaying motion intervals from
the sidecar SQLite database every hour:

    global_prior = clamp(occupied_seconds / (now - first_interval_start))

This module maintains the same ratio as **sufficient statistics updated
incrementally** — an occupied-seconds accumulator fed from live motion
evidence at the coordinator's tick cadence, and the first-observation
timestamp as the period anchor. If the online value tracks the
DB-computed value on real homes, the raw-interval replay (and eventually
the database itself) is unnecessary for prior learning.

Shadow-mode contract: the online value is computed, persisted (HA
storage helper), diffed against the DB value each analysis cycle, and
exported in diagnostics. It is **never read by the probability path**.
Promotion routes through #500's 30-day shadow-diff gate.

Known, accepted approximations vs the DB path (they bound the expected
diff; see #500 step 2 for how the diff is judged):

* **Sampling**: occupied time accrues in tick-sized quanta (~10s)
  from live motion evidence, vs the DB's exact interval boundaries
  replayed from the recorder.
* **No motion-timeout extension**: the DB path extends each motion
  interval by the area's motion timeout during merging; the online
  numerator does not, so it reads slightly low on areas with sparse
  motion.
* **Retention**: the DB period re-anchors as old intervals are pruned
  (365-day retention); the online period anchor is fixed at first
  observation. Irrelevant inside the 30-day shadow window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..const import MAX_PRIOR, MIN_PRIOR
from ..time_utils import ensure_utc_datetime

# A tick gap larger than this is treated as downtime for the NUMERATOR:
# we can't know whether motion continued while HA was stopped, so no
# occupied time accrues across the gap. The DENOMINATOR intentionally
# keeps growing across downtime — the DB path behaves the same way
# (recorder gaps contribute period but no intervals).
MAX_TICK_GAP_SECONDS = 60.0


@dataclass
class OnlinePriorState:
    """Serializable sufficient statistics for one area."""

    occupied_seconds: float = 0.0
    first_observation: datetime | None = None
    last_tick: datetime | None = None
    last_motion_active: bool = False

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "occupied_seconds": self.occupied_seconds,
            "first_observation": self.first_observation.isoformat()
            if self.first_observation
            else None,
            "last_tick": self.last_tick.isoformat() if self.last_tick else None,
            "last_motion_active": self.last_motion_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OnlinePriorState:
        """Restore from storage; malformed fields fall back to empty state."""
        try:
            return cls(
                occupied_seconds=float(data.get("occupied_seconds", 0.0)),
                first_observation=ensure_utc_datetime(
                    datetime.fromisoformat(data["first_observation"])
                )
                if data.get("first_observation")
                else None,
                last_tick=ensure_utc_datetime(datetime.fromisoformat(data["last_tick"]))
                if data.get("last_tick")
                else None,
                last_motion_active=bool(data.get("last_motion_active", False)),
            )
        except (KeyError, TypeError, ValueError):
            return cls()


class OnlinePriorEstimator:
    """Incremental global-prior estimator for one area."""

    def __init__(self, state: OnlinePriorState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state or OnlinePriorState()

    def observe(self, *, motion_active: bool, now: datetime) -> None:
        """Record one coordinator tick.

        Occupied time accrues for the elapsed span since the previous
        tick when motion evidence was active at the START of the span
        (piecewise-constant assumption at tick granularity). Spans
        longer than ``MAX_TICK_GAP_SECONDS`` contribute nothing to the
        numerator — see module docstring on downtime.
        """
        now = ensure_utc_datetime(now)
        if self.state.first_observation is None:
            self.state.first_observation = now
        if self.state.last_tick is not None:
            elapsed = (now - self.state.last_tick).total_seconds()
            if 0.0 < elapsed <= MAX_TICK_GAP_SECONDS and self.state.last_motion_active:
                self.state.occupied_seconds += elapsed
        self.state.last_tick = now
        self.state.last_motion_active = motion_active

    def prior(self, now: datetime) -> float | None:
        """Return the current online prior, or None before any observation."""
        if self.state.first_observation is None:
            return None
        period = (
            ensure_utc_datetime(now) - self.state.first_observation
        ).total_seconds()
        if period <= 0:
            return None
        raw = self.state.occupied_seconds / period
        return max(MIN_PRIOR, min(MAX_PRIOR, raw))

    def observed_days(self, now: datetime) -> float:
        """Return how long this estimator has been observing, in days."""
        if self.state.first_observation is None:
            return 0.0
        return (
            ensure_utc_datetime(now) - self.state.first_observation
        ).total_seconds() / 86400.0
