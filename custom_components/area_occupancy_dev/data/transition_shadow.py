"""Shadow-mode area transitions learned live, without the database (#603).

Adjacency learns which room people go to next from ``db.transitions``,
which replays each area's occupied intervals out of SQLite every hour.
The same detection can run as it happens: each area's live ground-truth
label (:mod:`.ground_truth`) going occupied is a "start" and going empty
an "end", and :class:`TransitionShadow` applies exactly the database's
rules to that stream: the same transition and trajectory windows, the same
adjacency check, 1-hop ``X -> Y`` and 2-hop ``W -> X -> Y`` chains bucketed
by hour of week, and the same recency half-life.

Shadow-mode contract: counted, persisted and compared daily with the
database's counts (diagnostics ``transition_shadow``); the adjacency boost
keeps reading the database snapshot until #603's switch-over.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

from ..const import (
    ADJACENCY_RECENCY_HALF_LIFE_DAYS,
    ADJACENCY_TRAJECTORY_WINDOW_S,
    ADJACENCY_TRANSITION_WINDOW_S,
)
from ..time_utils import ensure_utc_datetime, to_local

# A conditional next-room share needs this much decayed count behind it
# before it is compared (a single walk-through is not a distribution).
TRANSITION_MIN_COUNT = 5.0

Counts = dict[str, dict[int, dict[str, float]]]  # "from|mid" -> hour -> to -> n


def _hour_of_week(ts: datetime) -> int:
    local = to_local(ts)
    return local.weekday() * 24 + local.hour


@dataclass
class TransitionShadowState:
    """Live transition counts and the detector's short memory (persisted)."""

    counts: Counts = field(default_factory=dict)
    last_decay: datetime | None = None
    # Areas currently labelled occupied, and recent "end" events.
    occupied: dict[str, bool] = field(default_factory=dict)
    recent_ends: list[tuple[str, datetime]] = field(default_factory=list)
    # Daily worst gap to the database's next-room shares, oldest first.
    diff_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "counts": {
                chain: {str(h): dict(t) for h, t in by_hour.items()}
                for chain, by_hour in self.counts.items()
            },
            "last_decay": self.last_decay.isoformat() if self.last_decay else None,
            "occupied": dict(self.occupied),
            "recent_ends": [(a, t.isoformat()) for a, t in self.recent_ends],
            "diff_history": list(self.diff_history),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TransitionShadowState:
        """Restore from storage; malformed payloads fall back to empty."""
        try:
            return cls(
                counts={
                    str(chain): {
                        int(h): {str(t): float(n) for t, n in tos.items()}
                        for h, tos in by_hour.items()
                    }
                    for chain, by_hour in (data.get("counts") or {}).items()
                },
                last_decay=(
                    ensure_utc_datetime(datetime.fromisoformat(raw))
                    if (raw := data.get("last_decay"))
                    else None
                ),
                occupied={
                    str(k): bool(v) for k, v in (data.get("occupied") or {}).items()
                },
                recent_ends=[
                    (str(a), ensure_utc_datetime(datetime.fromisoformat(t)))
                    for a, t in (data.get("recent_ends") or [])
                ],
                diff_history=[
                    dict(e)
                    for e in (data.get("diff_history") or [])
                    if isinstance(e, dict)
                ],
            )
        except (AttributeError, TypeError, ValueError):
            return cls()


class TransitionShadow:
    """Detects area transitions from live labels, as ``db.transitions`` does."""

    def __init__(self, state: TransitionShadowState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state or TransitionShadowState()

    def _decay(self, now: datetime) -> None:
        state = self.state
        if state.last_decay is not None:
            hours = (now - state.last_decay).total_seconds() / 3600
            if hours > 0:
                factor = 0.5 ** (hours / (24 * ADJACENCY_RECENCY_HALF_LIFE_DAYS))
                for by_hour in state.counts.values():
                    for tos in by_hour.values():
                        for to_area in tos:
                            tos[to_area] *= factor
        state.last_decay = now

    def _count(self, from_area: str, mid_area: str, to_area: str, hour: int) -> None:
        tos = self.state.counts.setdefault(f"{from_area}|{mid_area}", {}).setdefault(
            hour, {}
        )
        tos[to_area] = tos.get(to_area, 0.0) + 1.0

    def observe(
        self,
        *,
        now: datetime,
        labels: dict[str, bool],
        adjacency: dict[str, set[str]],
    ) -> None:
        """Record one tick of every area's label.

        Args:
            now: The tick time.
            labels: area name -> live ground-truth label.
            adjacency: area name -> its configured adjacent areas.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        self._decay(now)
        transition_window = timedelta(seconds=ADJACENCY_TRANSITION_WINDOW_S)
        trajectory_window = timedelta(seconds=ADJACENCY_TRAJECTORY_WINDOW_S)
        ends: deque[tuple[str, datetime]] = deque(state.recent_ends, maxlen=8)

        # Ends first, so an area emptying in the same tick another fills
        # is a candidate "from" for it (the database orders by timestamp).
        for area_name, label in labels.items():
            if state.occupied.get(area_name) and not label:
                if ends and ends[-1][0] == area_name:
                    ends[-1] = (area_name, now)
                else:
                    ends.append((area_name, now))
        for area_name, label in labels.items():
            if label and not state.occupied.get(area_name, False):
                while ends and now - ends[0][1] > trajectory_window:
                    ends.popleft()
                if ends:
                    x_area, x_end = ends[-1]
                    if (
                        x_area != area_name
                        and now - x_end <= transition_window
                        and area_name in adjacency.get(x_area, set())
                    ):
                        hour = _hour_of_week(now)
                        self._count(x_area, "", area_name, hour)
                        if len(ends) >= 2:
                            w_area, w_end = ends[-2]
                            if x_end - w_end <= trajectory_window and x_area in (
                                adjacency.get(w_area, set())
                            ):
                                self._count(w_area, x_area, area_name, hour)
        state.occupied = dict(labels)
        state.recent_ends = list(ends)

    @staticmethod
    def _shares(counts: Counts) -> dict[str, dict[str, float]]:
        """Next-room shares per chain, over the whole week."""
        shares: dict[str, dict[str, float]] = {}
        for chain, by_hour in counts.items():
            totals: dict[str, float] = {}
            for tos in by_hour.values():
                for to_area, n in tos.items():
                    totals[to_area] = totals.get(to_area, 0.0) + n
            total = sum(totals.values())
            if total >= TRANSITION_MIN_COUNT:
                shares[chain] = {t: n / total for t, n in totals.items()}
        return shares

    def compare(self, db_counts: Counts) -> dict:
        """Compare next-room shares with the database's.

        Returns:
            Per chain both share maps, and ``max_diff``: the largest gap in
            any next room's share where both sides have enough data.
        """
        live = self._shares(self.state.counts)
        db = self._shares(db_counts)
        chains: dict[str, dict] = {}
        diffs: list[float] = []
        for chain in sorted(set(live) | set(db)):
            row: dict = {"live": live.get(chain), "db": db.get(chain)}
            if chain in live and chain in db:
                gap = max(
                    abs(live[chain].get(t, 0.0) - db[chain].get(t, 0.0))
                    for t in set(live[chain]) | set(db[chain])
                )
                row["diff"] = round(gap, 4)
                diffs.append(gap)
            chains[chain] = row
        return {"chains": chains, "max_diff": max(diffs) if diffs else None}

    def record_divergence(self, day: str, max_diff: float | None) -> None:
        """Fold today's worst gap into the daily history (kept 90 days)."""
        if max_diff is None:
            return
        history = self.state.diff_history
        if history and history[-1].get("date") == day:
            history[-1]["max_diff"] = max(history[-1]["max_diff"], max_diff)
        else:
            history.append({"date": day, "max_diff": max_diff})
        del history[:-90]

    def days_within_tolerance(self, tolerance: float) -> int:
        """Consecutive most-recent days whose worst gap stayed in tolerance."""
        streak = 0
        for entry in reversed(self.state.diff_history):
            if entry.get("max_diff", math.inf) > tolerance:
                break
            streak += 1
        return streak
