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
from typing import Any



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



def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    """Return a finite, non-negative float, or the supplied default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number >= 0 else default



@dataclass
class TransitionShadowState:
    """Live transition counts and the detector's short memory (persisted)."""



    counts: Counts = field(default_factory=dict)
    last_decay: datetime | None = None
    # Areas currently labelled occupied, and recent "end" events.
    occupied: dict[str, bool] = field(default_factory=dict)
    recent_ends: list[tuple[str, datetime]] = field(default_factory=list)
    # Daily worst gap to the database's next-room shares, oldest first.
    diff_history: list[dict[str, Any]] = field(default_factory=list)



    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> TransitionShadowState:
        """Restore from storage; malformed payloads fall back to empty."""
        try:
            raw_counts = data.get("counts") or {}
            counts: Counts = {}
            for chain, by_hour in raw_counts.items():
                if not isinstance(by_hour, dict):
                    continue
                valid_hours: dict[int, dict[str, float]] = {}
                for raw_hour, tos in by_hour.items():
                    try:
                        hour = int(raw_hour)
                    except (TypeError, ValueError):
                        continue
                    if not 0 <= hour < 168 or not isinstance(tos, dict):
                        continue
                    valid_tos = {
                        str(to_area): number
                        for to_area, raw_number in tos.items()
                        if (number := _finite_nonnegative(raw_number, -1.0)) >= 0
                    }
                    if valid_tos:
                        valid_hours[hour] = valid_tos
                if valid_hours:
                    counts[str(chain)] = valid_hours

            raw_last_decay = data.get("last_decay")
            last_decay = (
                ensure_utc_datetime(datetime.fromisoformat(raw_last_decay))
                if isinstance(raw_last_decay, str) and raw_last_decay
                else None
            )

            occupied_raw = data.get("occupied") or {}
            occupied = (
                {str(k): bool(v) for k, v in occupied_raw.items()}
                if isinstance(occupied_raw, dict)
                else {}
            )

            recent_ends: list[tuple[str, datetime]] = []
            for item in data.get("recent_ends") or []:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                area_name, raw_time = item
                if not isinstance(raw_time, str):
                    continue
                try:
                    event_time = ensure_utc_datetime(datetime.fromisoformat(raw_time))
                except (TypeError, ValueError):
                    continue
                recent_ends.append((str(area_name), event_time))

            history: list[dict[str, Any]] = []
            for item in data.get("diff_history") or []:
                if not isinstance(item, dict):
                    continue
                date_value = item.get("date")
                diff_value = _finite_nonnegative(item.get("max_diff"), -1.0)
                if isinstance(date_value, str) and diff_value >= 0:
                    history.append({"date": date_value, "max_diff": diff_value})

            return cls(
                counts=counts,
                last_decay=last_decay,
                occupied=occupied,
                recent_ends=recent_ends[-8:],
                diff_history=history[-90:],
            )
        except (AttributeError, TypeError, ValueError):
            return cls()



class TransitionShadow:
    """Detects area transitions from live labels, as ``db.transitions`` does."""



    def __init__(self, state: TransitionShadowState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state if state is not None else TransitionShadowState()



    def _decay(self, now: datetime) -> None:
        state = self.state
        if state.last_decay is not None:
            elapsed_hours = (now - state.last_decay).total_seconds() / 3600
            half_life_days = _finite_nonnegative(
                ADJACENCY_RECENCY_HALF_LIFE_DAYS, 0.0
            )
            if elapsed_hours > 0 and half_life_days > 0:
                factor = 0.5 ** (elapsed_hours / (24 * half_life_days))
                for by_hour in state.counts.values():
                    for tos in by_hour.values():
                        for to_area, count in list(tos.items()):
                            tos[to_area] = _finite_nonnegative(count) * factor
        # Do not move the decay anchor backwards if an older tick arrives.
        if state.last_decay is None or now > state.last_decay:
            state.last_decay = now



    def _count(self, from_area: str, mid_area: str, to_area: str, hour: int) -> None:
        tos = self.state.counts.setdefault(f"{from_area}|{mid_area}", {}).setdefault(
            hour, {}
        )
        tos[to_area] = _finite_nonnegative(tos.get(to_area, 0.0)) + 1.0



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
        transition_window = timedelta(
            seconds=_finite_nonnegative(ADJACENCY_TRANSITION_WINDOW_S)
        )
        trajectory_window = timedelta(
            seconds=_finite_nonnegative(ADJACENCY_TRAJECTORY_WINDOW_S)
        )
        ends: deque[tuple[str, datetime]] = deque(state.recent_ends, maxlen=8)

        # Ends first, so an area emptying in the same tick another fills
        # is a candidate "from" for it (the database orders by timestamp).
        for area_name, label in labels.items():
            if state.occupied.get(area_name, False) and not label:
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
                            if (
                                w_area != x_area
                                and w_area != area_name
                                and x_end - w_end <= trajectory_window
                                and x_area in adjacency.get(w_area, set())
                            ):
                                self._count(w_area, x_area, area_name, hour)

        # Persist all supplied labels; labels are expected to be a full snapshot.
        state.occupied = {str(area): bool(label) for area, label in labels.items()}
        state.recent_ends = list(ends)



    @staticmethod
    def _shares(counts: Counts) -> dict[str, dict[str, float]]:
        """Next-room shares per chain, over the whole week."""
        shares: dict[str, dict[str, float]] = {}
        for chain, by_hour in counts.items():
            totals: dict[str, float] = {}
            for tos in by_hour.values():
                for to_area, raw_count in tos.items():
                    count = _finite_nonnegative(raw_count)
                    totals[to_area] = totals.get(to_area, 0.0) + count
            total = sum(totals.values())
            if math.isfinite(total) and total >= TRANSITION_MIN_COUNT:
                shares[chain] = {t: n / total for t, n in totals.items()}
        return shares



    def compare(self, db_counts: Counts) -> dict[str, Any]:
        """Compare next-room shares with the database's.



        Returns:
            Per chain both share maps, and ``max_diff``: the largest gap in
            any next room's share where both sides have enough data.
        """
        live = self._shares(self.state.counts)
        db = self._shares(db_counts)
        chains: dict[str, dict[str, Any]] = {}
        diffs: list[float] = []
        for chain in sorted(set(live) | set(db)):
            row: dict[str, Any] = {"live": live.get(chain), "db": db.get(chain)}
            if chain in live and chain in db:
                targets = set(live[chain]) | set(db[chain])
                if targets:
                    gap = max(
                        abs(live[chain].get(target, 0.0) - db[chain].get(target, 0.0))
                        for target in targets
                    )
                    row["diff"] = round(gap, 4)
                    diffs.append(gap)
            chains[chain] = row
        return {"chains": chains, "max_diff": max(diffs) if diffs else None}



    def record_divergence(self, day: str, max_diff: float | None) -> None:
        """Fold today's worst gap into the daily history (kept 90 days)."""
        if max_diff is None or not math.isfinite(max_diff) or max_diff < 0:
            return
        history = self.state.diff_history
        if history and history[-1].get("date") == day:
            previous = _finite_nonnegative(history[-1].get("max_diff"))
            history[-1]["max_diff"] = max(previous, max_diff)
        else:
            history.append({"date": day, "max_diff": max_diff})
        del history[:-90]



    def days_within_tolerance(self, tolerance: float) -> int:
        """Consecutive most-recent days whose worst gap stayed in tolerance."""
        tolerance = _finite_nonnegative(tolerance)
        streak = 0
        for entry in reversed(self.state.diff_history):
            diff = _finite_nonnegative(entry.get("max_diff"), math.inf)
            if diff > tolerance:
                break
            streak += 1
        return streak
