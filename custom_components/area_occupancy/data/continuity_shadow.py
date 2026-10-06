"""Shadow-mode presence continuity for open-plan rooms (#558).

Motion loses people who sit still, and today an open-plan room then decays
to empty on its half-life (stretched up to x1.75 while its usual next rooms
stay quiet). Continuity asks for more: don't clear a room because motion
lost you, clear it when you're seen leaving. While no exit is observed the
room would decay on a longer half-life, ``1 + 2 * silence`` times its own
(capped at ``CONTINUITY_MAX_HALF_LIFE``), and on an exit it would go back to
normal at once. An exit is a door of the room opening, or a neighbouring
room lighting up within ``ADJACENCY_TRANSITION_WINDOW_S`` of the last
evidence.

This is the part of #558 most exposed to flaky sensors, so it runs in
shadow only. Each time a room's ground-truth sensors go quiet an episode
starts, and it ends with one of three outcomes:

* **resumed**: the same room's sensors fire again with no exit seen, so
  the person never left. Today's model *missed a hold* if it had already
  cleared the room; continuity missed one if it would have cleared too.
* **exit**: an exit was seen. Both models are right to clear.
* **unexplained**: neither within ``EPISODE_MAX_SECONDS``: they left by a
  way nobody watches, and any time continuity held the room beyond today's
  model was a *false hold*.

Continuity's clear time is estimated from the live one: a room clears when
its decaying evidence crosses the threshold, at a time proportional to the
half-life, so it scales by the ratio of the two half-lives at the start of
the episode. Promotion needs, per area, 14 days and 50 episodes with a
false-hold rate of at most 5%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math

from ..const import ADJACENCY_TRANSITION_WINDOW_S
from ..time_utils import ensure_utc_datetime, to_local

# Continuity's half-life multiplier at full silence (today's is 1.75).
CONTINUITY_MULTIPLIER = 3.0
# No room's continuity decay outlasts this half-life.
CONTINUITY_MAX_HALF_LIFE = 1800.0
# An episode with neither a resume nor an exit by now is unexplained.
EPISODE_MAX_SECONDS = 7200.0
# Days of counters kept, and how many of the latest are pooled.
HISTORY_DAYS = 45
WINDOW_DAYS = 30

_COUNTERS = (
    "episodes",
    "resumed",
    "exits",
    "unexplained",
    "live_missed",
    "continuity_missed",
    "false_holds",
    "false_hold_seconds",
    "recovered_seconds",
)


def continuity_half_life(base: float, live: float, silence: float) -> float:
    """The half-life continuity would decay on while no exit is seen.

    Args:
        base: The room's own half-life.
        live: Today's effective half-life (with the adjacency stretch).
        silence: The adjacency silence score in [0, 1].
    """
    stretched = base * (1.0 + (CONTINUITY_MULTIPLIER - 1.0) * silence)
    return max(live, min(stretched, CONTINUITY_MAX_HALF_LIFE))


@dataclass
class Episode:
    """One open episode: the room's sensors went quiet at ``start``."""

    start: datetime
    half_life_ratio: float
    # Neighbours that were empty at the start (an exit is one lighting up).
    quiet_neighbours: list[str]
    live_cleared: float | None = None

    def to_dict(self) -> dict:
        """Serialize (JSON-safe)."""
        return {
            "start": self.start.isoformat(),
            "half_life_ratio": self.half_life_ratio,
            "quiet_neighbours": list(self.quiet_neighbours),
            "live_cleared": self.live_cleared,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Episode:
        """Restore; raises on malformed data."""
        cleared = data.get("live_cleared")
        return cls(
            start=ensure_utc_datetime(datetime.fromisoformat(data["start"])),
            half_life_ratio=float(data["half_life_ratio"]),
            quiet_neighbours=[str(n) for n in data.get("quiet_neighbours") or []],
            live_cleared=None if cleared is None else float(cleared),
        )


@dataclass
class ContinuityState:
    """One area's continuity shadow (persisted)."""

    present: bool | None = None
    doors_open: bool | None = None
    episode: Episode | None = None
    # Local date (of the episode start) -> counters.
    days: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "present": self.present,
            "doors_open": self.doors_open,
            "episode": self.episode.to_dict() if self.episode else None,
            "days": {day: dict(c) for day, c in self.days.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> ContinuityState:
        """Restore from storage; malformed payloads fall back to empty."""
        try:
            present = data.get("present")
            doors_open = data.get("doors_open")
            episode = data.get("episode")
            return cls(
                present=None if present is None else bool(present),
                doors_open=None if doors_open is None else bool(doors_open),
                episode=Episode.from_dict(episode) if episode else None,
                days={
                    str(day): {k: float(counters.get(k, 0.0)) for k in _COUNTERS}
                    for day, counters in (data.get("days") or {}).items()
                },
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return cls()


class ContinuityShadow:
    """Scores continuity against today's decay for one area."""

    def __init__(self, state: ContinuityState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state or ContinuityState()

    def observe(
        self,
        now: datetime,
        *,
        present: bool,
        occupied: bool,
        doors_open: bool,
        neighbours: dict[str, bool],
        half_life_ratio: float,
    ) -> None:
        """Record one tick.

        Args:
            now: The tick time.
            present: Whether any of the room's ground-truth sensors is active.
            occupied: Today's published decision for the room.
            doors_open: Whether one of the room's doors is open (an exit is
                one opening, so a door left open says nothing).
            neighbours: Adjacent room -> its live ground-truth label.
            half_life_ratio: Continuity's half-life over today's, now.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        was_present = state.present
        state.present = present
        door_opened = doors_open and state.doors_open is False
        state.doors_open = doors_open
        episode = state.episode
        if episode is not None:
            elapsed = (now - episode.start).total_seconds()
            if present:
                self._close(episode, "resumed", elapsed)
            elif door_opened or (
                elapsed <= ADJACENCY_TRANSITION_WINDOW_S
                and any(neighbours.get(n) for n in episode.quiet_neighbours)
            ):
                self._close(episode, "exits", elapsed)
            elif elapsed >= EPISODE_MAX_SECONDS:
                self._close(episode, "unexplained", elapsed)
            elif episode.live_cleared is None and not occupied:
                episode.live_cleared = elapsed
        # A room that is held occupied when its sensors go quiet starts one.
        if state.episode is None and was_present is True and not present and occupied:
            state.episode = Episode(
                start=now,
                half_life_ratio=half_life_ratio,
                quiet_neighbours=sorted(n for n, on in neighbours.items() if not on),
            )

    def _close(self, episode: Episode, outcome: str, elapsed: float) -> None:
        """Score a finished episode into its day's counters."""
        day = to_local(episode.start).date().isoformat()
        counters = self.state.days.setdefault(day, dict.fromkeys(_COUNTERS, 0.0))
        counters["episodes"] += 1
        counters[outcome] += 1
        live = episode.live_cleared
        continuity = None if live is None else live * episode.half_life_ratio
        if outcome == "unexplained":
            if continuity is not None and continuity > live:
                counters["false_holds"] += 1
                counters["false_hold_seconds"] += continuity - live
        elif live is not None:
            # Time continuity kept the room occupied past today's clear,
            # up to the resume or exit, while the person was still there.
            counters["recovered_seconds"] += max(min(continuity, elapsed) - live, 0.0)
            if outcome == "resumed":
                counters["live_missed"] += 1
                if continuity < elapsed:
                    counters["continuity_missed"] += 1
        self.state.episode = None
        for old in sorted(self.state.days)[:-HISTORY_DAYS]:
            del self.state.days[old]

    def summary(self) -> dict | None:
        """Outcomes pooled over the latest days (diagnostics)."""
        days = sorted(self.state.days)[-WINDOW_DAYS:]
        if not days:
            return None
        total = dict.fromkeys(_COUNTERS, 0.0)
        for day in days:
            for key in _COUNTERS:
                total[key] += self.state.days[day].get(key, 0.0)

        def rate(numerator: str, denominator: str) -> float | None:
            if not total[denominator]:
                return None
            return round(total[numerator] / total[denominator], 4)

        return {
            "shadow_mode": True,
            "days": len(days),
            "episodes": int(total["episodes"]),
            "resumed": int(total["resumed"]),
            "exits": int(total["exits"]),
            "unexplained": int(total["unexplained"]),
            # Of the episodes where the person never left, the share each
            # model had already cleared the room for.
            "missed_hold_rate_live": rate("live_missed", "resumed"),
            "missed_hold_rate_continuity": rate("continuity_missed", "resumed"),
            "false_hold_rate": rate("false_holds", "episodes"),
            "false_hold_minutes": round(total["false_hold_seconds"] / 60, 1),
            "recovered_minutes": round(total["recovered_seconds"] / 60, 1),
            "open_episode": self.state.episode is not None,
        }


def half_life_ratio(base: float, live: float, silence: float) -> float:
    """Continuity's half-life over today's (1 when there's nothing to stretch)."""
    if live <= 0 or not math.isfinite(live):
        return 1.0
    return continuity_half_life(base, live, silence) / live
