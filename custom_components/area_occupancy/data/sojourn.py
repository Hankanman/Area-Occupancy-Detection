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

    P(the room still holds someone | the sensors have been silent
      for s seconds)

That is a survival-analysis problem with competing risks, and the
coordinator already produces the needed episode stream (the same
machinery ``continuity_shadow`` uses):

* **resumed** — evidence returned with no confirmed exit: someone was
  still there. The episode is *censored* at its silence length (it
  leaves the risk set; what happens after the resume is a new episode).
* **exits** — a departure signal was seen *and held*: a door opened, or
  a previously-quiet neighbour lit up. The observed departure the
  survival curve counts.
* **unexplained** — neither within ``SOJOURN_EPISODE_MAX_SECONDS``:
  censored, like a study participant lost to follow-up. Counting it as
  an exit would bias every room toward "people always leave", and
  counting it as a resume would bias toward "never leave"; censoring is
  the honest third answer.
* **censored** — a tick gap larger than ``MAX_TICK_GAP_SECONDS`` means
  the silence duration is unknown, not merely long.

### Exit candidates and the grace period

A door opening or a neighbour lighting up is only a *candidate* exit:
in multi-person homes one person can leave while another stays, and a
door can open for a delivery. Candidates therefore sit in a grace
period (``SOJOURN_EXIT_GRACE_SECONDS``): if ground-truth evidence
returns meanwhile, the episode closes as ``resumed`` — the room never
actually emptied. Only a candidate that survives the grace period
becomes an ``exits`` event, scored at the silence length where the
signal first appeared (the best estimate of when the room emptied).

Neighbour signals get two strengths: within
``ADJACENCY_TRANSITION_WINDOW_S`` of the silence start they are a
strong candidate (weight 1.0); up to ``SOJOURN_NEIGHBOR_EXIT_WINDOW_S``
they are weak (``SOJOURN_WEAK_EXIT_WEIGHT``), because a neighbour
lighting up minutes later may be unrelated activity. Weak exits still
feed the life table at fractional weight but are not Brier-scored —
their label is genuinely uncertain.

### The learned curve, its priors, and its fallback

The life table is actuarial with two shrinkages:

* **Hazard prior** (``SOJOURN_HAZARD_PRIOR`` pseudo-survivors per bin):
  a bin where every observed episode exited no longer drives the
  survival estimate to a hard 0.0 on a handful of samples.
* **Recency weighting**: each episode counts ``0.5 ** (age /
  SOJOURN_FORGET_HALF_LIFE_DAYS)`` — room layouts, sensors and habits
  drift, and a 2024 pattern should not outweigh this month's.

``survival_at`` blends the learned curve with the exponential baseline
``2**(-s / half_life)`` in proportion to the evidence behind it — and
the evidence that matters is *observed exits*, not episodes. A room
with no door sensor and no informative neighbour produces episodes but
zero exit events; there the learned curve is indistinguishable from
"people never leave", so the estimate stays on the exponential until
real departures accumulate (``SOJOURN_EXIT_EVIDENCE_PRIOR`` weighted
exits). This is also the honest answer for households where exits are
rare: the model cannot beat the baseline on data it cannot see.

Episodes are split by context — sleep window vs waking hours, weekend
vs weekday (``sojourn_context``) — because a bedroom's 3 a.m. silence
and its 3 p.m. nap are different distributions. A context with too
little weight falls back to the pooled curve
(``SOJOURN_MIN_CONTEXT_WEIGHT``).

Shadow-mode contract: computed, persisted (HA storage helper) and
exported in diagnostics. Never read by the probability path. Each
scored episode is evaluated *prequentially* — the curve predicts only
episodes it has not trained on — as a Brier score beside the
exponential baseline's, pooled per day. ``summary()`` reports the
promotion gate explicitly: minimum scored episodes, decisive exits and
observed days before ``brier_skill > 0`` means anything.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
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

# An episode with neither a resume nor a confirmed exit by now is
# censored: they left by a way nobody watches, or the room genuinely
# stays occupied (overnight stillness).
SOJOURN_EPISODE_MAX_SECONDS = 6 * 3600.0
# How long an exit candidate may sit unconfirmed before it counts.
# Long enough for a resident to walk back in after opening the door;
# short enough that a real departure isn't held for minutes.
SOJOURN_EXIT_GRACE_SECONDS = 90.0
# A neighbour lighting up within the transition window is a strong exit
# candidate; within this longer window it is a weak one (weight below) —
# delayed arrivals happen, but so does unrelated neighbour activity.
SOJOURN_NEIGHBOR_EXIT_WINDOW_S = 180.0
SOJOURN_WEAK_EXIT_WEIGHT = 0.5
# Pseudo-survivors added to each bin's risk set: shrinks the hazard
# toward "no exits observed" so a few unlucky bins can't zero the curve.
SOJOURN_HAZARD_PRIOR = 4.0
# Episodes count 0.5**(age / this) in the life table — old patterns fade.
SOJOURN_FORGET_HALF_LIFE_DAYS = 90.0
# Blend weight of the learned curve vs the exponential baseline. The
# learned share also requires observed exit evidence (below), so rooms
# without exit sensors stay on the baseline indefinitely.
SOJOURN_BLEND_PRIOR = 30.0
SOJOURN_EXIT_EVIDENCE_PRIOR = 8.0
# A context's episodes must carry at least this much recency weight
# before its own curve is used; below that the pooled curve answers.
SOJOURN_MIN_CONTEXT_WEIGHT = 10.0
# Promotion gate (first-pass values, tune from real diagnostics):
# minimum decisive Brier-scored episodes, decisive scored exits, and
# days of score history before brier_skill > 0 can promote anything.
SOJOURN_GATE_MIN_SCORED = 100
SOJOURN_GATE_MIN_EXITS = 20
SOJOURN_GATE_MIN_DAYS = 14
# The room counts as "held" for episode start when the live decision
# says occupied OR the probability is at least this fraction of the
# threshold — catches borderline cases the binary decision already
# gave up on, while still excluding clearly-empty rooms.
SOJOURN_START_MIN_THRESHOLD_FRAC = 0.5
# Closed episodes retained for the life table, oldest dropped first.
SOJOURN_EPISODES_KEPT = 4000
# Days of per-day outcome/Brier tallies retained.
SOJOURN_SCORE_DAYS = 45
# Window pooled into the diagnostics summary.
SOJOURN_SUMMARY_DAYS = 30

# Recency reference for states with no tick history: far enough in the
# past that every episode weighs 1.0 — no forgetting without a clock.
# ``timezone(timedelta(0))`` instead of ``timezone.utc``: identical, and
# keeps ruff's UP017 and basedpyright's ``datetime.UTC`` lookup happy.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone(timedelta(0)))

_OUTCOMES = ("resumed", "exits", "unexplained", "censored")
_SCORE_KEYS = (
    "episodes",
    "scored",
    "scored_exits",
    "brier_model",
    "brier_baseline",
    *_OUTCOMES,
)


def _parse_hhmmss(value: str | None) -> time | None:
    """Parse an ``HH:MM:SS`` config string into a ``time``, or None."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%H:%M:%S").time()
    except (TypeError, ValueError):
        return None


def _sleep_window(
    sleep_start: str | None, sleep_end: str | None
) -> tuple[time, time] | None:
    """The configured sleep window as ``(start, end)``, or None.

    Overnight windows (start > end, e.g. 22:00–08:00) wrap midnight.
    Missing, unparseable or empty (start == end) bounds mean "no sleep
    window".
    """
    start = _parse_hhmmss(sleep_start)
    end = _parse_hhmmss(sleep_end)
    if start is None or end is None or start == end:
        return None
    return start, end


def _within_window(current: time, start: time, end: time) -> bool:
    """Whether ``current`` falls inside ``[start, end)``, wrapping midnight."""
    if start < end:
        return start <= current < end
    return current >= start or current < end


def sojourn_context(
    now: datetime, sleep_start: str | None, sleep_end: str | None
) -> str:
    """The episode's context key: sleep/wake × weekend/weekday.

    Returns e.g. ``"night"``, ``"day_weekend"``, ``"any_weekend"``.
    ``"any"`` is used when no valid sleep window is configured — the
    day/night distinction would be invented, not learned.
    """
    local = to_local(ensure_utc_datetime(now))
    window = _sleep_window(sleep_start, sleep_end)
    if window is None:
        base = "any"
    elif _within_window(local.time(), *window):
        base = "night"
    else:
        base = "day"
    return f"{base}_weekend" if local.weekday() >= 5 else base


@dataclass
class SojournEpisode:
    """One closed silence episode (persisted)."""

    start: datetime
    # When the episode's outcome was observed (close time).
    end: datetime
    silence_seconds: float
    outcome: str
    # 1.0 for decisive exits (door, neighbour inside the transition
    # window), SOJOURN_WEAK_EXIT_WEIGHT for delayed neighbour signals,
    # 0.0 for non-exit outcomes.
    exit_weight: float
    # Context key from ``sojourn_context`` at episode start.
    context: str
    # The area's effective decay half-life when the episode began — the
    # exponential baseline the Brier comparison scores against.
    half_life_seconds: float

    def weight(self, now: datetime) -> float:
        """Recency weight: 0.5**(age / SOJOURN_FORGET_HALF_LIFE_DAYS)."""
        age_days = (
            max((ensure_utc_datetime(now) - self.end).total_seconds(), 0.0) / 86400.0
        )
        return 0.5 ** (age_days / SOJOURN_FORGET_HALF_LIFE_DAYS)

    @property
    def decisive_exit(self) -> bool:
        """Whether this episode is a confidently-labelled departure."""
        return self.outcome == "exits" and self.exit_weight >= 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize (JSON-safe)."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "silence_seconds": self.silence_seconds,
            "outcome": self.outcome,
            "exit_weight": self.exit_weight,
            "context": self.context,
            "half_life_seconds": self.half_life_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SojournEpisode:
        """Restore; raises ValueError on malformed data.

        Validates what the math depends on: finite non-negative silence,
        a known outcome, a finite non-negative half-life. ``end`` is
        optional (v1 payloads) and reconstructed from start + silence.
        """
        silence = float(data["silence_seconds"])
        half_life = float(data.get("half_life_seconds", 0.0))
        outcome = str(data["outcome"])
        if (
            not math.isfinite(silence)
            or silence < 0
            or outcome not in _OUTCOMES
            or not math.isfinite(half_life)
            or half_life < 0
        ):
            raise ValueError(f"malformed sojourn episode: {data!r}")
        start = ensure_utc_datetime(datetime.fromisoformat(data["start"]))
        end_raw = data.get("end")
        end = (
            ensure_utc_datetime(datetime.fromisoformat(end_raw))
            if end_raw
            else start + timedelta(seconds=silence)
        )
        exit_weight = float(data.get("exit_weight", 1.0 if outcome == "exits" else 0.0))
        return cls(
            start=start,
            end=end,
            silence_seconds=silence,
            outcome=outcome,
            exit_weight=max(0.0, min(exit_weight, 1.0)),
            context=str(data.get("context") or ""),
            half_life_seconds=half_life,
        )


@dataclass
class SojournState:
    """One area's sojourn shadow (persisted)."""

    # Sensors went quiet at ``episode_start`` while the room was held
    # occupied; resolved by resume, confirmed exit, cap or tick gap.
    episode_start: datetime | None = None
    quiet_neighbours: list[str] = field(default_factory=list)
    episode_half_life: float = 0.0
    episode_context: str = ""
    # A pending exit candidate: ``(signal_time, silence_at_signal, weight)``.
    exit_candidate: tuple[datetime, float, float] | None = None
    present: bool | None = None
    doors_open: bool | None = None
    last_tick: datetime | None = None
    # Most recent effective half-life seen — the baseline survival_at
    # blends toward when no per-episode value is supplied.
    last_half_life: float = 0.0
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
            "episode_context": self.episode_context,
            "exit_candidate": (
                [
                    self.exit_candidate[0].isoformat(),
                    self.exit_candidate[1],
                    self.exit_candidate[2],
                ]
                if self.exit_candidate
                else None
            ),
            "present": self.present,
            "doors_open": self.doors_open,
            "last_tick": self.last_tick.isoformat() if self.last_tick else None,
            "last_half_life": self.last_half_life,
            "episodes": [e.to_dict() for e in self.episodes],
            "score_days": {day: dict(c) for day, c in self.score_days.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SojournState:
        """Restore from storage; malformed payloads fall back to empty.

        Tolerant per field: a malformed episode is skipped rather than
        discarding the whole area's history.
        """
        try:
            start = data.get("episode_start")
            last_tick = data.get("last_tick")
            present = data.get("present")
            doors_open = data.get("doors_open")
            candidate = data.get("exit_candidate")
            episodes: list[SojournEpisode] = []
            for raw in cast(list[Any], data.get("episodes") or []):
                if not isinstance(raw, dict):
                    continue
                try:
                    episodes.append(SojournEpisode.from_dict(cast(dict[str, Any], raw)))
                except (KeyError, TypeError, ValueError):
                    continue
            exit_candidate = None
            cand = cast(Any, candidate)
            seq = cast(Any, cand) if isinstance(cand, (list, tuple)) else None
            if seq is not None and len(seq) == 3:
                try:
                    exit_candidate = (
                        ensure_utc_datetime(datetime.fromisoformat(str(seq[0]))),
                        float(seq[1]),
                        float(seq[2]),
                    )
                except (TypeError, ValueError):
                    exit_candidate = None
            return cls(
                episode_start=ensure_utc_datetime(datetime.fromisoformat(start))
                if start
                else None,
                quiet_neighbours=[
                    str(n) for n in cast(list[Any], data.get("quiet_neighbours") or [])
                ],
                episode_half_life=float(data.get("episode_half_life", 0.0)),
                episode_context=str(data.get("episode_context") or ""),
                exit_candidate=exit_candidate,
                present=None if present is None else bool(present),
                doors_open=None if doors_open is None else bool(doors_open),
                last_tick=ensure_utc_datetime(datetime.fromisoformat(last_tick))
                if last_tick
                else None,
                last_half_life=float(data.get("last_half_life", 0.0)),
                episodes=episodes,
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
    episodes: list[SojournEpisode], now: datetime
) -> tuple[list[float], list[float], list[float]]:
    """Weighted actuarial life table over the silence bins.

    Returns ``(entered, events, censored)`` per bin as floats — each
    episode contributes its recency weight. An episode is at risk in
    every bin below the one containing its end silence; decisive exits
    are events there (weak exits count at ``exit_weight``), every other
    outcome censors at half weight.
    """
    n = len(SOJOURN_BIN_EDGES) + 1  # bins + overflow tail
    entered = [0.0] * n
    events = [0.0] * n
    censored = [0.0] * n
    for episode in episodes:
        if episode.outcome not in _OUTCOMES:
            continue
        weight = episode.weight(now)
        j = bisect_left(SOJOURN_BIN_EDGES, episode.silence_seconds)
        for i in range(j):
            entered[i] += weight
        entered[j] += weight
        if episode.outcome == "exits":
            events[j] += weight * episode.exit_weight
        else:
            censored[j] += weight
    return entered, events, censored


def _survival_points(
    episodes: list[SojournEpisode], now: datetime
) -> tuple[list[float], list[float]]:
    """``(edges, S)`` — learned survival at each bin's upper edge."""
    entered, events, censored = _life_table(episodes, now)
    survival = 1.0
    edges: list[float] = []
    values: list[float] = []
    for i, edge in enumerate(SOJOURN_BIN_EDGES):
        # Hazard prior: pretend SOJOURN_HAZARD_PRIOR extra episodes
        # survived the bin without exiting — small samples can't zero it.
        risk = entered[i] - 0.5 * censored[i] + SOJOURN_HAZARD_PRIOR
        if risk > 0:
            survival *= max(0.0, 1.0 - events[i] / risk)
        edges.append(edge)
        values.append(survival)
    return edges, values


def _interp_survival(
    edges: list[float], values: list[float], silence_seconds: float
) -> float:
    """Log-linear interpolation of the learned curve at ``silence``."""
    if silence_seconds <= 0.0:
        return 1.0
    j = bisect_left(edges, silence_seconds)
    if j >= len(edges):
        return values[-1]
    if j == 0:
        s1 = values[0]
        if s1 <= 0.0:
            return 0.0
        return math.exp(math.log(s1) * (silence_seconds / edges[0]))
    s_prev = values[j - 1]
    s_next = values[j]
    if s_prev <= 0.0 or s_next <= 0.0:
        return 0.0 if s_next <= 0.0 else s_prev
    frac = (silence_seconds - edges[j - 1]) / (edges[j] - edges[j - 1])
    return math.exp(math.log(s_prev) + frac * (math.log(s_next) - math.log(s_prev)))


class SojournShadow:
    """Learns P(still occupied | silent for s) for one area."""

    def __init__(self, state: SojournState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state: SojournState = state or SojournState()

    def observe(
        self,
        now: datetime,
        *,
        room_held: bool,
        present: bool,
        doors_open: bool,
        neighbours: dict[str, bool],
        half_life_seconds: float,
        context: str = "",
    ) -> None:
        """Record one tick.

        Args:
            now: The tick time.
            room_held: The room counts as occupied for episode purposes —
                the coordinator passes the live decision OR a
                near-threshold probability (see
                ``SOJOURN_START_MIN_THRESHOLD_FRAC``).
            present: Whether any of the room's ground-truth sensors is
                active (unbridged — the same input continuity uses).
            doors_open: Whether one of the room's doors is open.
            neighbours: Adjacent room -> its live ground-truth label.
            half_life_seconds: The area's effective decay half-life this
                tick — the exponential baseline stored with each episode
                and used for blending.
            context: ``sojourn_context`` key for this tick.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        state.last_half_life = half_life_seconds
        # A tick gap means the silence duration is unknown, not merely
        # long: censor the open episode (and any pending exit candidate)
        # at the last observed tick, then restart tracking from now.
        if (
            state.last_tick is not None
            and (now - state.last_tick).total_seconds() > MAX_TICK_GAP_SECONDS
        ):
            if state.episode_start is not None:
                self._close(
                    state.last_tick,
                    (state.last_tick - state.episode_start).total_seconds(),
                    "censored",
                    0.0,
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
                self._close(now, silence, "resumed", 0.0)
            else:
                self._update_exit(now, silence, door_opened, neighbours)
                candidate = state.exit_candidate
                if (
                    candidate is not None
                    and (now - candidate[0]).total_seconds()
                    >= SOJOURN_EXIT_GRACE_SECONDS
                ):
                    # Confirmed: the room emptied at the signal time.
                    self._close(candidate[0], candidate[1], "exits", candidate[2])
                elif (now - state.episode_start).total_seconds() >= (
                    SOJOURN_EPISODE_MAX_SECONDS
                ):
                    if candidate is not None:
                        self._close(candidate[0], candidate[1], "exits", candidate[2])
                    else:
                        self._close(
                            now,
                            (now - state.episode_start).total_seconds(),
                            "unexplained",
                            0.0,
                        )
        # A room held occupied when its sensors go quiet starts an episode.
        if (
            state.episode_start is None
            and was_present is True
            and not present
            and room_held
        ):
            state.episode_start = now
            state.episode_half_life = half_life_seconds
            state.episode_context = context
            state.quiet_neighbours = sorted(n for n, on in neighbours.items() if not on)
        state.last_tick = now

    def _update_exit(
        self,
        now: datetime,
        silence: float,
        door_opened: bool,
        neighbours: dict[str, bool],
    ) -> None:
        """Record or upgrade an exit candidate for the open episode.

        The earliest signal wins: once a candidate exists, later signals
        only upgrade its weight (weak -> strong), never its timestamp —
        the room most plausibly emptied when the FIRST departure signal
        appeared.
        """
        state = self.state
        assert state.episode_start is not None
        candidate = state.exit_candidate
        if door_opened:
            signal = (now, silence, 1.0)
        else:
            # Neighbour rising edge: strong inside the transition window,
            # weak out to SOJOURN_NEIGHBOR_EXIT_WINDOW_S.
            weight = 0.0
            if silence <= SOJOURN_NEIGHBOR_EXIT_WINDOW_S:
                for n in state.quiet_neighbours:
                    if neighbours.get(n):
                        weight = (
                            1.0
                            if silence <= ADJACENCY_TRANSITION_WINDOW_S
                            else SOJOURN_WEAK_EXIT_WEIGHT
                        )
                        break
            if weight == 0.0:
                return
            signal = (now, silence, weight)
        if candidate is None or signal[1] < candidate[1]:
            state.exit_candidate = signal
        elif signal[2] > candidate[2]:
            # Same or later signal but stronger: keep the earlier time,
            # take the stronger weight.
            state.exit_candidate = (candidate[0], candidate[1], signal[2])

    def _close(
        self,
        end: datetime,
        silence: float,
        outcome: str,
        exit_weight: float,
    ) -> None:
        """Record a finished episode, then score it prequentially.

        ``end`` is when the outcome happened — for a confirmed exit that
        is the signal time, not the grace-expiry tick — and buckets the
        daily tallies and the recency weight.
        """
        state = self.state
        assert state.episode_start is not None
        episode = SojournEpisode(
            start=state.episode_start,
            end=end,
            silence_seconds=max(silence, 0.0),
            outcome=outcome,
            exit_weight=exit_weight,
            context=state.episode_context,
            half_life_seconds=state.episode_half_life,
        )
        state.episode_start = None
        state.quiet_neighbours = []
        state.exit_candidate = None
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
        never predicts data it trained on. Only confidently-labelled
        outcomes count: ``resumed`` means someone was still there (y=1),
        a decisive ``exits`` means the room emptied (y=0). Weak exits,
        unexplained and censored episodes carry uncertain labels and are
        not scored.
        """
        if episode.outcome == "resumed":
            y = 1.0
        elif episode.decisive_exit:
            y = 0.0
        else:
            return
        model_p = self.survival_at(
            episode.silence_seconds,
            context=episode.context,
            baseline_half_life=episode.half_life_seconds,
        )
        half_life = episode.half_life_seconds
        baseline_p = (
            0.5 ** (episode.silence_seconds / half_life) if half_life > 0 else 0.5
        )
        counters = self.state.score_days.setdefault(
            day, dict.fromkeys(_SCORE_KEYS, 0.0)
        )
        counters["scored"] += 1
        if episode.decisive_exit:
            counters["scored_exits"] += 1
        counters["brier_model"] += (model_p - y) ** 2
        counters["brier_baseline"] += (baseline_p - y) ** 2

    def _reference_now(self) -> datetime:
        """The clock the recency weights are measured against.

        ``last_tick`` while the shadow is live; for a restored or
        hand-built state without ticks, the newest episode end — freshly
        loaded history isn't "infinitely old". With neither, ``_EPOCH``
        makes every weight 1.0: no forgetting without a clock.
        """
        if self.state.last_tick is not None:
            return self.state.last_tick
        if self.state.episodes:
            return max(e.end for e in self.state.episodes)
        return _EPOCH

    def _episodes_for(self, context: str | None) -> list[SojournEpisode]:
        """Episodes for ``context``, or the pooled set when it's thin."""
        if not context:
            return self.state.episodes
        matching = [e for e in self.state.episodes if e.context == context]
        now = self._reference_now()
        if (
            sum(e.weight(now) for e in matching) >= SOJOURN_MIN_CONTEXT_WEIGHT
            or not self.state.episodes
        ):
            return matching
        return self.state.episodes

    def survival_at(
        self,
        silence_seconds: float,
        *,
        context: str | None = None,
        baseline_half_life: float | None = None,
    ) -> float:
        """P(still occupied | sensors silent for ``silence_seconds``).

        Blends the learned survival curve with the exponential baseline
        ``2**(-s / half_life)``: the learned share grows with episode
        weight AND requires observed exit evidence — a room whose exits
        are invisible to the sensors stays on the baseline rather than
        learning "people never leave".
        """
        if silence_seconds <= 0.0:
            return 1.0
        episodes = self._episodes_for(context)
        now = self._reference_now()
        total_weight = sum(e.weight(now) for e in episodes)
        exit_evidence = sum(
            e.weight(now) * e.exit_weight for e in episodes if e.outcome == "exits"
        )
        half_life = (
            baseline_half_life
            if baseline_half_life is not None
            else self.state.last_half_life
        )
        if total_weight <= 0.0:
            # Nothing learned at all: the neutral prior.
            return 0.5
        learned_p = self._learned_only(episodes, now, silence_seconds)
        if half_life <= 0.0:
            # Episodes exist but no baseline half-life is known.
            return learned_p
        baseline_p = 0.5 ** (silence_seconds / half_life)
        # Evidence shares: episodes say how much the curve is trusted at
        # all; observed exits say whether it learned departures or just
        # "people keep resuming".
        n_share = total_weight / (total_weight + SOJOURN_BLEND_PRIOR)
        e_share = exit_evidence / (exit_evidence + SOJOURN_EXIT_EVIDENCE_PRIOR)
        alpha = n_share * e_share
        return alpha * learned_p + (1.0 - alpha) * baseline_p

    def _learned_only(
        self, episodes: list[SojournEpisode], now: datetime, silence: float
    ) -> float:
        """The raw actuarial estimate, before baseline blending."""
        if not episodes:
            return 0.5
        edges, values = _survival_points(episodes, now)
        return _interp_survival(edges, values, silence)

    def survival_curve(
        self, context: str | None = None
    ) -> tuple[list[float], list[float]]:
        """``(edges, S)`` learned points for diagnostics/exports."""
        now = self._reference_now()
        return _survival_points(self._episodes_for(context), now)

    def summary(self) -> dict[str, Any] | None:
        """Learned curve + pooled Brier comparison + promotion gate."""
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
        skill = (
            1.0 - brier_model / brier_baseline
            if brier_model is not None and brier_baseline
            else None
        )
        scored_days = len(
            {day for day in days if self.state.score_days[day].get("scored", 0.0) > 0}
        )
        # Silence at which the pooled learned curve crosses 50% — the
        # room's empirical "typical" silence survival.
        median_silence = self._median_silence()
        # Per-context curves: only contexts with real weight are listed.
        contexts: dict[str, Any] = {}
        now = self._reference_now()
        seen = {e.context for e in self.state.episodes}
        for ctx in sorted(seen):
            eps = [e for e in self.state.episodes if e.context == ctx]
            weight = sum(e.weight(now) for e in eps)
            if weight < SOJOURN_MIN_CONTEXT_WEIGHT:
                continue
            ctx_median = self._median_silence(eps, now)
            contexts[ctx] = {
                "episodes": len(eps),
                "weight": round(weight, 1),
                "exits": sum(1 for e in eps if e.outcome == "exits"),
                "median_silence_seconds": (
                    round(ctx_median, 1) if ctx_median is not None else None
                ),
            }
        decisive_exits = sum(1 for e in self.state.episodes if e.decisive_exit)
        gate = {
            "scored": int(scored),
            "scored_exits": int(total["scored_exits"]),
            "days": scored_days,
            "min_scored": SOJOURN_GATE_MIN_SCORED,
            "min_exits": SOJOURN_GATE_MIN_EXITS,
            "min_days": SOJOURN_GATE_MIN_DAYS,
            "brier_skill": round(skill, 4) if skill is not None else None,
            "ready": bool(
                scored >= SOJOURN_GATE_MIN_SCORED
                and total["scored_exits"] >= SOJOURN_GATE_MIN_EXITS
                and scored_days >= SOJOURN_GATE_MIN_DAYS
                and skill is not None
                and skill > 0
            ),
        }
        return {
            "shadow_mode": True,
            "engaged": False,
            "episodes": len(self.state.episodes),
            "decisive_exits": decisive_exits,
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
            "exit_blind": decisive_exits == 0 and len(self.state.episodes) >= 20,
            "contexts": contexts,
            "gate": gate,
            "survival": {
                "60s": round(self.survival_at(60.0), 4),
                "300s": round(self.survival_at(300.0), 4),
                "1800s": round(self.survival_at(1800.0), 4),
            },
            "open_episode": self.state.episode_start is not None,
        }

    def _median_silence(
        self,
        episodes: list[SojournEpisode] | None = None,
        now: datetime | None = None,
    ) -> float | None:
        """Silence where the learned curve crosses 50%, if it does."""
        eps = self.state.episodes if episodes is None else episodes
        if not eps:
            return None
        ref = now or self._reference_now()
        edges, values = _survival_points(eps, ref)
        prev_s, prev_v = 0.0, 1.0
        for edge, value in zip(edges, values, strict=True):
            if prev_v >= 0.5 > value:
                # value can be 0.0: clamp for the log interpolation.
                v_next = max(value, 1e-12)
                frac = math.log(0.5 / prev_v) / math.log(v_next / prev_v)
                return prev_s + frac * (edge - prev_s)
            prev_s, prev_v = edge, value
        return None
