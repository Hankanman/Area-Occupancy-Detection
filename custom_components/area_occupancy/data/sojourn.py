"""Shadow-mode sojourn-time survival model — a learned decay curve.

Today's decay is a fixed exponential: after the last evidence, the
decaying entities fade on ``half_life`` seconds regardless of how long
people actually stay in the room. ``data/forecast.py`` already names the
principled successor — learning the persistence from the room's own
occupied-run history — and this module is that successor, estimated
online from live ticks instead of replayed intervals.

Why not learn from ``Intervals.duration_seconds`` directly: an occupied
interval *ends when the evidence ends* — it measures sensor-active time,
not dwell time, and says nothing about the silence that follows. The
question decay actually has to answer is different:

    P(the person is still in the room | the sensors have been silent
      for s seconds)

That is a survival-analysis problem with competing risks, and the
coordinator already produces the needed episode stream (the same
machinery ``continuity_shadow`` uses):

* **resumed** — evidence returned with no exit seen: the person was
  still there. The episode is *censored* at its silence length (it
  leaves the risk set; what happens after the resume is a new episode).
* **exit** — a door opened or a neighbour lit up within the transition
  window: observed departure, the event the survival curve counts.
* **unexplained** — neither within ``SOJOURN_EPISODE_MAX_SECONDS``:
  censored, like a study participant lost to follow-up. Counting it as
  an exit would bias every room toward "people always leave", and
  counting it as a resume would bias toward "never leave"; censoring is
  the honest third answer.
* **censored** — a tick gap larger than ``MAX_TICK_GAP_SECONDS`` means
  the silence duration is unknown, not merely long.

The life table is actuarial: each episode contributes to every silence
bin below its end, events land in the bin containing the exit, and
censored episodes count at half weight in their end bin. The result is
``survival_at(s)`` — the learned probability that a room silent for
``s`` seconds still holds someone — which can be compared directly
against the exponential ``2**(-s / half_life)`` the live decay implies.

Shadow-mode contract: computed, persisted (HA storage helper) and
exported in diagnostics. Never read by the probability path. Each
closed episode is also scored *prequentially* — the curve is evaluated
against episodes it has not trained on — as a Brier score beside the
exponential baseline's, pooled per day, so the promotion gate has a
number instead of a hunch.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, cast

from ..const import ADJACENCY_TRANSITION_WINDOW_S
from ..time_utils import ensure_utc_datetime, to_local
from .online_prior import MAX_TICK_GAP_SECONDS


# Silence-duration bins (upper edges, seconds), geometric from just above
# the tick cadence to beyond the episode cap. Log spacing matches the
# shape of dwell distributions: resolution where exits actually happen
# (seconds to minutes), coarser in the tail where episodes are rare.
def _build_edges() -> tuple[float, ...]:
    edges: list[float] = []
    edge = 8.0
    while edge < 24000.0:
        edges.append(edge)
        edge *= 1.7
    edges.append(24000.0)
    return tuple(edges)


SOJOURN_BIN_EDGES = _build_edges()

# An episode with neither a resume nor an exit by now is censored: they
# left by a way nobody watches, or the room genuinely stays occupied
# (overnight stillness). Same role as continuity's EPISODE_MAX_SECONDS.
SOJOURN_EPISODE_MAX_SECONDS = 6 * 3600.0
# Closed episodes retained for the life table, oldest dropped first.
SOJOURN_EPISODES_KEPT = 4000
# Days of per-day outcome/Brier tallies retained.
SOJOURN_SCORE_DAYS = 45
# Window pooled into the diagnostics summary.
SOJOURN_SUMMARY_DAYS = 30

_OUTCOMES = ("resumed", "exits", "unexplained", "censored")
_SCORE_KEYS = ("episodes", "scored", "brier_model", "brier_baseline", *_OUTCOMES)


@dataclass
class SojournEpisode:
    """One closed silence episode (persisted)."""

    start: datetime
    silence_seconds: float
    outcome: str
    # The area's effective decay half-life when the episode began — the
    # exponential baseline the Brier comparison scores against.
    half_life_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize (JSON-safe)."""
        return {
            "start": self.start.isoformat(),
            "silence_seconds": self.silence_seconds,
            "outcome": self.outcome,
            "half_life_seconds": self.half_life_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SojournEpisode:
        """Restore; raises on malformed data."""
        return cls(
            start=ensure_utc_datetime(datetime.fromisoformat(data["start"])),
            silence_seconds=float(data["silence_seconds"]),
            outcome=str(data["outcome"]),
            half_life_seconds=float(data.get("half_life_seconds", 0.0)),
        )


@dataclass
class SojournState:
    """One area's sojourn shadow (persisted)."""

    # Sensors went quiet at ``episode.start`` while the room was held
    # occupied; closed when evidence returns, an exit is seen, or the
    # episode cap / a tick gap censors it.
    episode_start: datetime | None = None
    quiet_neighbours: list[str] = field(default_factory=list)
    episode_half_life: float = 0.0
    present: bool | None = None
    doors_open: bool | None = None
    last_tick: datetime | None = None
    # Closed episodes, oldest first — the life table's raw material.
    episodes: list[SojournEpisode] = field(default_factory=list)
    # Local date -> outcome counts + prequential Brier sums.
    score_days: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "episode_start": self.episode_start.isoformat()
            if self.episode_start
            else None,
            "quiet_neighbours": list(self.quiet_neighbours),
            "episode_half_life": self.episode_half_life,
            "present": self.present,
            "doors_open": self.doors_open,
            "last_tick": self.last_tick.isoformat() if self.last_tick else None,
            "episodes": [e.to_dict() for e in self.episodes],
            "score_days": {day: dict(c) for day, c in self.score_days.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SojournState:
        """Restore from storage; malformed payloads fall back to empty."""
        try:
            start = data.get("episode_start")
            last_tick = data.get("last_tick")
            present = data.get("present")
            doors_open = data.get("doors_open")
            return cls(
                episode_start=ensure_utc_datetime(datetime.fromisoformat(start))
                if start
                else None,
                quiet_neighbours=[
                    str(n) for n in cast(list[Any], data.get("quiet_neighbours") or [])
                ],
                episode_half_life=float(data.get("episode_half_life", 0.0)),
                present=None if present is None else bool(present),
                doors_open=None if doors_open is None else bool(doors_open),
                last_tick=ensure_utc_datetime(datetime.fromisoformat(last_tick))
                if last_tick
                else None,
                episodes=[
                    SojournEpisode.from_dict(cast(dict[str, Any], e))
                    for e in cast(list[Any], data.get("episodes") or [])
                    if isinstance(e, dict)
                ],
                score_days={
                    str(day): {
                        str(k): float(v)
                        for k, v in cast(dict[str, Any], counters).items()
                    }
                    for day, counters in cast(
                        dict[Any, Any], data.get("score_days") or {}
                    ).items()
                    if isinstance(counters, dict)
                },
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return cls()


def _life_table(
    episodes: list[SojournEpisode],
) -> tuple[list[int], list[int], list[int]]:
    """Actuarial life table over the silence bins.

    Returns ``(entered, events, censored)`` per bin. An episode is at
    risk in every bin below the one containing its end silence; exits
    are events there, every other outcome censors at half weight.
    """
    n = len(SOJOURN_BIN_EDGES) + 1  # bins + overflow tail
    entered = [0] * n
    events = [0] * n
    censored = [0] * n
    for episode in episodes:
        if episode.outcome not in ("resumed", "exits", "unexplained", "censored"):
            continue
        j = bisect_left(SOJOURN_BIN_EDGES, episode.silence_seconds)
        for i in range(j):
            entered[i] += 1
        entered[j] += 1
        if episode.outcome == "exits":
            events[j] += 1
        else:
            censored[j] += 1
    return entered, events, censored


def _survival_curve(
    episodes: list[SojournEpisode],
) -> tuple[list[float], list[float]]:
    """``(edges, S)`` — survival probability at each bin's upper edge."""
    entered, events, censored = _life_table(episodes)
    survival = 1.0
    edges: list[float] = []
    values: list[float] = []
    for i, edge in enumerate(SOJOURN_BIN_EDGES):
        risk = entered[i] - 0.5 * censored[i]
        if risk > 0:
            survival *= max(0.0, 1.0 - events[i] / risk)
        edges.append(edge)
        values.append(survival)
    return edges, values


class SojournShadow:
    """Learns P(still occupied | silent for s) for one area."""

    def __init__(self, state: SojournState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state: SojournState = state or SojournState()

    def observe(
        self,
        now: datetime,
        *,
        present: bool,
        occupied: bool,
        doors_open: bool,
        neighbours: dict[str, bool],
        half_life_seconds: float,
    ) -> None:
        """Record one tick.

        Args:
            now: The tick time.
            present: Whether any of the room's ground-truth sensors is
                active (unbridged — the same input continuity uses).
            occupied: Today's published decision for the room.
            doors_open: Whether one of the room's doors is open.
            neighbours: Adjacent room -> its live ground-truth label.
            half_life_seconds: The area's effective decay half-life this
                tick — the exponential baseline stored with each episode.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        # A tick gap means the silence duration is unknown, not merely
        # long: censor the open episode at the last observed tick and
        # restart the present/door tracking from what we can see now.
        if (
            state.last_tick is not None
            and (now - state.last_tick).total_seconds() > MAX_TICK_GAP_SECONDS
        ):
            if state.episode_start is not None:
                self._close(
                    state.last_tick,
                    (state.last_tick - state.episode_start).total_seconds(),
                    "censored",
                )
            state.present = present
            state.doors_open = doors_open
            state.last_tick = now
            return
        was_present = state.present
        state.present = present
        door_opened = doors_open and state.doors_open is False
        state.doors_open = doors_open
        if state.episode_start is not None:
            silence = (now - state.episode_start).total_seconds()
            if present:
                self._close(now, silence, "resumed")
            elif door_opened or (
                silence <= ADJACENCY_TRANSITION_WINDOW_S
                and any(neighbours.get(n) for n in state.quiet_neighbours)
            ):
                self._close(now, silence, "exits")
            elif silence >= SOJOURN_EPISODE_MAX_SECONDS:
                self._close(now, silence, "unexplained")
        # A room held occupied when its sensors go quiet starts an episode.
        if (
            state.episode_start is None
            and was_present is True
            and not present
            and occupied
        ):
            state.episode_start = now
            state.episode_half_life = half_life_seconds
            state.quiet_neighbours = sorted(n for n, on in neighbours.items() if not on)
        state.last_tick = now

    def _close(self, end: datetime, silence: float, outcome: str) -> None:
        """Record a finished episode, then score it prequentially."""
        state = self.state
        assert state.episode_start is not None
        episode = SojournEpisode(
            start=state.episode_start,
            silence_seconds=max(silence, 0.0),
            outcome=outcome,
            half_life_seconds=state.episode_half_life,
        )
        state.episode_start = None
        state.quiet_neighbours = []
        day = to_local(end).date().isoformat()
        self._score(episode, day)
        state.episodes.append(episode)
        del state.episodes[:-SOJOURN_EPISODES_KEPT]
        counters = state.score_days.setdefault(day, dict.fromkeys(_SCORE_KEYS, 0.0))
        counters["episodes"] += 1
        counters[outcome] += 1
        for old in sorted(state.score_days)[:-SOJOURN_SCORE_DAYS]:
            del state.score_days[old]

    def _score(self, episode: SojournEpisode, day: str) -> None:
        """Prequential Brier score vs the exponential baseline.

        Scored before the episode joins ``state.episodes``, so the curve
        never predicts data it trained on. Only decisive outcomes count:
        ``resumed`` means the person was still there (y=1), ``exits``
        means they were gone (y=0); censored outcomes carry no label.
        """
        if episode.outcome not in ("resumed", "exits"):
            return
        model_p = self.survival_at(episode.silence_seconds)
        half_life = episode.half_life_seconds
        baseline_p = (
            0.5 ** (episode.silence_seconds / half_life) if half_life > 0 else 0.5
        )
        y = 1.0 if episode.outcome == "resumed" else 0.0
        counters = self.state.score_days.setdefault(
            day, dict.fromkeys(_SCORE_KEYS, 0.0)
        )
        counters["scored"] += 1
        counters["brier_model"] += (model_p - y) ** 2
        counters["brier_baseline"] += (baseline_p - y) ** 2

    def survival_at(self, silence_seconds: float) -> float:
        """P(still occupied | sensors silent for ``silence_seconds``).

        Log-linear interpolation between learned survival points;
        ``1.0`` at zero silence and flat beyond the last bin edge.
        """
        if silence_seconds <= 0.0:
            return 1.0
        if not self.state.episodes:
            # No closed episodes yet: fall back to the neutral prior so a
            # cold start reports 0.5 rather than a misleading 1.0.
            return 0.5
        edges, values = _survival_curve(self.state.episodes)
        j = bisect_left(edges, silence_seconds)
        if j >= len(edges):
            return values[-1]
        if j == 0:
            # Between 0 and the first edge: log-interpolate from S(0)=1.
            s1 = values[0]
            if s1 <= 0.0:
                return 0.0
            frac = silence_seconds / edges[0]
            return math.exp(math.log(s1) * frac)
        s_prev = values[j - 1]
        s_next = values[j]
        if s_prev <= 0.0 or s_next <= 0.0:
            return 0.0 if s_next <= 0.0 else s_prev
        frac = (silence_seconds - edges[j - 1]) / (edges[j] - edges[j - 1])
        return math.exp(math.log(s_prev) + frac * (math.log(s_next) - math.log(s_prev)))

    def summary(self) -> dict[str, Any] | None:
        """Learned curve + pooled Brier comparison (diagnostics)."""
        days = sorted(self.state.score_days)[-SOJOURN_SUMMARY_DAYS:]
        if not days and not self.state.episodes:
            return None
        total: dict[str, float] = dict.fromkeys(_SCORE_KEYS, 0.0)
        for day in days:
            for key in _SCORE_KEYS:
                total[key] += self.state.score_days[day].get(key, 0.0)
        scored = total["scored"]
        brier_model = total["brier_model"] / scored if scored else None
        brier_baseline = total["brier_baseline"] / scored if scored else None
        # Share of the baseline's Brier the model removes; positive means
        # the learned curve predicts presence during silence better than
        # the exponential the live decay assumes.
        skill = (
            1.0 - brier_model / brier_baseline
            if brier_model is not None and brier_baseline
            else None
        )
        # Silence at which the learned curve crosses 50% — the room's
        # empirical "typical" silence survival, comparable to half-life.
        median_silence = None
        edges, values = _survival_curve(self.state.episodes)
        prev_s, prev_v = 0.0, 1.0
        for edge, value in zip(edges, values, strict=True):
            if prev_v >= 0.5 > value:
                # Log-linear crossing, same interpolation as survival_at.
                frac = math.log(0.5 / prev_v) / math.log(value / prev_v)
                median_silence = prev_s + frac * (edge - prev_s)
                break
            prev_s, prev_v = edge, value
        return {
            "shadow_mode": True,
            "engaged": False,
            "episodes": len(self.state.episodes),
            "days": len(days),
            "outcomes": {o: int(total[o]) for o in _OUTCOMES},
            "scored": int(scored),
            "brier_model": round(brier_model, 4) if brier_model is not None else None,
            "brier_baseline": (
                round(brier_baseline, 4) if brier_baseline is not None else None
            ),
            "brier_skill": round(skill, 4) if skill is not None else None,
            "median_silence_seconds": (
                round(median_silence, 1) if median_silence is not None else None
            ),
            "survival": {
                "60s": round(self.survival_at(60.0), 4),
                "300s": round(self.survival_at(300.0), 4),
                "1800s": round(self.survival_at(1800.0), 4),
            },
            "open_episode": self.state.episode_start is not None,
        }
