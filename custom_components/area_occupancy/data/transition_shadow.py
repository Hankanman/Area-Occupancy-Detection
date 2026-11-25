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
from datetime import date, datetime, timedelta
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
# Decayed counts below this are dropped so dead chains do not pile up in
# storage (far below TRANSITION_MIN_COUNT, so shares are unaffected in practice).
MIN_KEPT_COUNT = 1e-4
MAX_RECENT_ENDS = 8
MAX_DIFF_HISTORY_DAYS = 90
HOURS_PER_WEEK = 7 * 24

Counts = dict[str, dict[int, dict[str, float]]]  # "from|mid" -> hour -> to -> n


def _hour_of_week(ts: datetime) -> int:
    local = to_local(ts)
    return local.weekday() * 24 + local.hour


def _as_count(value: Any) -> float | None:
    """Return ``value`` as a finite, non-negative float, else ``None``."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _parse_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return ensure_utc_datetime(datetime.fromisoformat(raw))
    except ValueError:
        return None


def _parse_day(raw: Any) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _parse_counts(raw: Any) -> Counts:
    if not isinstance(raw, dict):
        return {}
    counts: Counts = {}
    for chain, by_hour in raw.items():
        if not isinstance(chain, str) or not isinstance(by_hour, dict):
            continue
        hours: dict[int, dict[str, float]] = {}
        for raw_hour, tos in by_hour.items():
            if isinstance(raw_hour, bool) or not isinstance(tos, dict):
                continue
            try:
                hour = int(raw_hour)
            except (TypeError, ValueError):
                continue
            if not 0 <= hour < HOURS_PER_WEEK:
                continue
            valid = {
                to: n
                for to, raw_n in tos.items()
                if isinstance(to, str) and (n := _as_count(raw_n)) is not None
            }
            if valid:
                hours[hour] = valid
        if hours:
            counts[chain] = hours
    return counts


def _parse_recent_ends(raw: Any) -> list[tuple[str, datetime]]:
    if not isinstance(raw, list):
        return []
    ends: list[tuple[str, datetime]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        area, raw_time = item
        if isinstance(area, str) and (ts := _parse_datetime(raw_time)) is not None:
            ends.append((area, ts))
    return ends[-MAX_RECENT_ENDS:]


def _parse_diff_history(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    history: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        day = item.get("date")
        diff = _as_count(item.get("max_diff"))
        if _parse_day(day) is not None and diff is not None:
            history.append({"date": day, "max_diff": diff})
    return history[-MAX_DIFF_HISTORY_DAYS:]


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
            "diff_history": [dict(e) for e in self.diff_history],
        }

    @classmethod
    def from_dict(cls, data: Any) -> TransitionShadowState:
        """Restore from storage; malformed parts are dropped individually."""
        if not isinstance(data, dict):
            return cls()
        occupied = data.get("occupied")
        return cls(
            counts=_parse_counts(data.get("counts")),
            last_decay=_parse_datetime(data.get("last_decay")),
            occupied=(
                {k: bool(v) for k, v in occupied.items() if isinstance(k, str)}
                if isinstance(occupied, dict)
                else {}
            ),
            recent_ends=_parse_recent_ends(data.get("recent_ends")),
            diff_history=_parse_diff_history(data.get("diff_history")),
        )


class TransitionShadow:
    """Detects area transitions from live labels, as ``db.transitions`` does."""

    def __init__(self, state: TransitionShadowState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state if state is not None else TransitionShadowState()

    def _decay(self, now: datetime) -> None:
        state = self.state
        if state.last_decay is not None and now <= state.last_decay:
            # Never move the decay anchor backwards on an out-of-order tick.
            return
        if state.last_decay is not None:
            hours = (now - state.last_decay).total_seconds() / 3600
            factor = 0.5 ** (hours / (24 * ADJACENCY_RECENCY_HALF_LIFE_DAYS))
            for chain in list(state.counts):
                by_hour = state.counts[chain]
                for hour in list(by_hour):
                    tos = by_hour[hour]
                    for to_area in list(tos):
                        tos[to_area] *= factor
                        if tos[to_area] < MIN_KEPT_COUNT:
                            del tos[to_area]
                    if not tos:
                        del by_hour[hour]
                if not by_hour:
                    del state.counts[chain]
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
            labels: area name -> live ground-truth label (full snapshot).
            adjacency: area name -> its configured adjacent areas.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        self._decay(now)

        transition_window = timedelta(seconds=ADJACENCY_TRANSITION_WINDOW_S)
        trajectory_window = timedelta(seconds=ADJACENCY_TRAJECTORY_WINDOW_S)
        ends: deque[tuple[str, datetime]] = deque(
            state.recent_ends, maxlen=MAX_RECENT_ENDS
        )

        # Ends first, so an area emptying in the same tick another fills
        # is a candidate "from" for it (the database orders by timestamp).
        for area_name, label in labels.items():
            if state.occupied.get(area_name, False) and not label:
                if ends and ends[-1][0] == area_name:
                    ends[-1] = (area_name, now)
                else:
                    ends.append((area_name, now))

        for area_name, label in labels.items():
            if not label or state.occupied.get(area_name, False):
                continue
            while ends and now - ends[0][1] > trajectory_window:
                ends.popleft()
            if not ends:
                continue
            x_area, x_end = ends[-1]
            if (
                x_area == area_name
                or now - x_end > transition_window
                or area_name not in adjacency.get(x_area, set())
            ):
                continue
            hour = _hour_of_week(now)
            self._count(x_area, "", area_name, hour)
            if len(ends) >= 2:
                w_area, w_end = ends[-2]
                if x_end - w_end <= trajectory_window and x_area in adjacency.get(
                    w_area, set()
                ):
                    self._count(w_area, x_area, area_name, hour)

        state.occupied = dict(labels)
        state.recent_ends = list(ends)

    @staticmethod
    def _shares(counts: Counts) -> dict[str, dict[str, float]]:
        """Next-room shares per chain, pooled over all hours of the week."""
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
                gap = max(
                    (
                        abs(live[chain].get(t, 0.0) - db[chain].get(t, 0.0))
                        for t in set(live[chain]) | set(db[chain])
                    ),
                    default=0.0,
                )
                row["diff"] = round(gap, 4)
                diffs.append(gap)
            chains[chain] = row
        return {"chains": chains, "max_diff": max(diffs, default=None)}

    def record_divergence(self, day: str, max_diff: float | None) -> None:
        """Fold a day's worst gap into the daily history (last 90 entries)."""
        diff = _as_count(max_diff)
        new_day = _parse_day(day)
        if diff is None or new_day is None:
            return
        history = self.state.diff_history
        last_day = _parse_day(history[-1]["date"]) if history else None
        if last_day == new_day:
            history[-1]["max_diff"] = max(history[-1]["max_diff"], diff)
            return
        if last_day is not None and new_day < last_day:
            return  # Out-of-order day; keep the history sorted.
        history.append({"date": day, "max_diff": diff})
        del history[:-MAX_DIFF_HISTORY_DAYS]

    def days_within_tolerance(self, tolerance: float) -> int:
        """Consecutive most-recent days whose worst gap stayed in tolerance.

        A missing day (no comparison recorded) ends the streak.
        """
        streak = 0
        expected: date | None = None
        for entry in reversed(self.state.diff_history):
            day = _parse_day(entry.get("date"))
            if day is None or (expected is not None and day != expected):
                break
            if entry.get("max_diff", math.inf) > tolerance:
                break
            streak += 1
            expected = day - timedelta(days=1)
        return streak
