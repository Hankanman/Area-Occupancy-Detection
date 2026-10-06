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

* **Sampling**: occupied time accrues in tick-sized quanta from live
  presence evidence (piecewise-constant between ticks), vs the DB's
  exact interval boundaries replayed from the recorder. Ticks are not
  perfectly uniform — see ``coordinator._record_shadow_tick`` and
  ``_handle_decay_timer`` for how a steady cadence is maintained even
  when decay is disabled for every area.
* **No motion-timeout extension**: the DB path extends each motion
  interval by the area's motion timeout during merging; the online
  numerator does not, so it reads slightly low on areas with sparse
  motion.
* **Retention**: the DB period re-anchors as old intervals are pruned
  (365-day retention); the online period anchor is fixed at first
  observation. Irrelevant inside the 30-day shadow window.

The numerator's presence definition mirrors
``db.queries.get_occupied_intervals``'s ground truth — motion ∪ media ∪
sleep evidence, not motion alone (see the ``observe`` caller in
``coordinator.py``) — so the two paths are measuring the same thing
modulo the two approximations above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..const import (
    MAX_PRIOR,
    MIN_PRIOR,
    ONLINE_PRIOR_DIFF_HISTORY_DAYS,
    TIME_PRIOR_MAX_BOUND,
    TIME_PRIOR_MIN_BOUND,
)
from ..time_utils import ensure_utc_datetime, to_local

# A tick gap larger than this is treated as downtime for the NUMERATOR:
# we can't know whether motion continued while HA was stopped, so no
# occupied time accrues across the gap. The DENOMINATOR intentionally
# keeps growing across downtime — the DB path behaves the same way
# (recorder gaps contribute period but no intervals).
MAX_TICK_GAP_SECONDS = 60.0

# A weekly slot reports no online time prior until it has accumulated at
# least this much observed time. One hour of observation in a given
# (weekday, hour) slot means roughly one week of uptime covering it once —
# below that the ratio is a coin toss dressed up as a probability.
MIN_SLOT_OBSERVATION_SECONDS = 3600.0

# 168 hour-of-week buckets: to_local(t).weekday() * 24 + to_local(t).hour,
# the same convention the DB time priors and AreaTransitions use.
HOURS_PER_WEEK = 168


@dataclass
class OnlinePriorState:
    """Serializable sufficient statistics for one area.

    The v2 additions (weekly slot accumulators, divergence history) are
    optional in storage: a v1 payload restores with empty buckets and no
    history, preserving the scalar accumulators — see the Store migration
    in ``coordinator.py``.
    """

    occupied_seconds: float = 0.0
    first_observation: datetime | None = None
    last_tick: datetime | None = None
    last_motion_active: bool = False
    # Weekly time-prior accumulators, keyed by hour_of_week (0-167).
    # Sparse: a slot absent from both maps has never been observed.
    slot_occupied_seconds: dict[int, float] = field(default_factory=dict)
    slot_total_seconds: dict[int, float] = field(default_factory=dict)
    # Daily-collapsed shadow-diff summaries, oldest first, capped at
    # ONLINE_PRIOR_DIFF_HISTORY_DAYS entries. Each entry:
    # {"date": "YYYY-MM-DD", "scalar_diff": float,
    #  "bucket_diff": float | None, "buckets": int}
    # where the diffs are the day's MAX absolute divergence (worst case,
    # so a briefly-good hourly sample can't mask a bad day).
    diff_history: list[dict] = field(default_factory=list)
    # Fingerprint of the area's ground-truth sensors and their active
    # states when these statistics started. Statistics gathered under a
    # different set (sensors swapped, "paused" no longer active, another
    # house) don't describe the area any more; see ``is_stale``.
    config_signature: str | None = None

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "occupied_seconds": self.occupied_seconds,
            "first_observation": self.first_observation.isoformat()
            if self.first_observation
            else None,
            "last_tick": self.last_tick.isoformat() if self.last_tick else None,
            "last_motion_active": self.last_motion_active,
            # JSON object keys are strings; parsed back to int in from_dict.
            "slot_occupied_seconds": {
                str(k): v for k, v in self.slot_occupied_seconds.items()
            },
            "slot_total_seconds": {
                str(k): v for k, v in self.slot_total_seconds.items()
            },
            "diff_history": list(self.diff_history),
            "config_signature": self.config_signature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OnlinePriorState:
        """Restore from storage; malformed fields fall back to empty state.

        Tolerant of a v1 payload: the slot maps and diff history simply
        default to empty, so a schema extension never zeroes the scalar
        accumulators users have been building since 2026.7.1.
        """
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
                slot_occupied_seconds={
                    int(k): float(v)
                    for k, v in (data.get("slot_occupied_seconds") or {}).items()
                },
                slot_total_seconds={
                    int(k): float(v)
                    for k, v in (data.get("slot_total_seconds") or {}).items()
                },
                diff_history=[
                    dict(entry)
                    for entry in (data.get("diff_history") or [])
                    if isinstance(entry, dict)
                ],
                config_signature=(
                    str(data["config_signature"])
                    if data.get("config_signature")
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            return cls()


def is_stale(
    state: OnlinePriorState,
    signature: str,
    db_first_seen: datetime | None,
) -> bool:
    """Whether stored statistics no longer describe the area.

    True when they were gathered under a different ground-truth sensor
    set. Statistics saved before the fingerprint existed are judged by
    age instead: starting more than a day before anything the database
    holds means they outlived a database reset or a sensor swap, and
    their history would never line up with the database's again.

    Args:
        state: The stored statistics.
        signature: The area's current ground-truth fingerprint.
        db_first_seen: The area's earliest ground-truth row in the DB.

    Returns:
        True if the statistics should be discarded.
    """
    if state.config_signature is not None:
        return state.config_signature != signature
    if state.first_observation is None or db_first_seen is None:
        return False
    return state.first_observation < ensure_utc_datetime(db_first_seen) - timedelta(
        days=1
    )


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

        The weekly slot accumulators bucket the same span by the
        hour-of-week of the span's START (local wall clock, matching the
        DB time priors' bucketing). A span crossing an hour boundary is
        attributed wholly to the starting bucket — at the tick cadence a
        span is at most ``MAX_TICK_GAP_SECONDS``, so the misattribution
        is bounded at one minute per boundary crossing. Downtime gaps
        contribute to neither slot map: unlike the scalar denominator
        (anchored at first observation), the slot denominators only grow
        while ticking, since we cannot know WHICH weekly slots an
        arbitrary downtime span covered without replaying it hour by
        hour — the very bookkeeping this estimator exists to avoid.
        """
        now = ensure_utc_datetime(now)
        if self.state.first_observation is None:
            self.state.first_observation = now
        if self.state.last_tick is not None:
            elapsed = (now - self.state.last_tick).total_seconds()
            if 0.0 < elapsed <= MAX_TICK_GAP_SECONDS:
                span_start_local = to_local(self.state.last_tick)
                slot = span_start_local.weekday() * 24 + span_start_local.hour
                self.state.slot_total_seconds[slot] = (
                    self.state.slot_total_seconds.get(slot, 0.0) + elapsed
                )
                if self.state.last_motion_active:
                    self.state.occupied_seconds += elapsed
                    self.state.slot_occupied_seconds[slot] = (
                        self.state.slot_occupied_seconds.get(slot, 0.0) + elapsed
                    )
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

    def time_prior(self, hour_of_week: int) -> float | None:
        """Return the online time prior for one weekly slot, or None.

        Mirrors the DB path's clamp bounds. ``None`` means the slot has
        not yet accumulated ``MIN_SLOT_OBSERVATION_SECONDS`` of observed
        time — a not-enough-data answer, deliberately distinct from a
        low probability (the same learned-vs-unobserved distinction the
        DB path keeps via its ``data_points`` map).

        Args:
            hour_of_week: ``weekday() * 24 + hour`` in local wall clock,
                0 (Monday 00:xx) through 167 (Sunday 23:xx).
        """
        total = self.state.slot_total_seconds.get(hour_of_week, 0.0)
        if total < MIN_SLOT_OBSERVATION_SECONDS:
            return None
        occupied = self.state.slot_occupied_seconds.get(hour_of_week, 0.0)
        raw = occupied / total
        return max(TIME_PRIOR_MIN_BOUND, min(TIME_PRIOR_MAX_BOUND, raw))

    def observed_slot_count(self) -> int:
        """Return how many weekly slots have met the observation floor."""
        return sum(
            1
            for total in self.state.slot_total_seconds.values()
            if total >= MIN_SLOT_OBSERVATION_SECONDS
        )

    def record_divergence(
        self,
        *,
        now: datetime,
        scalar_diff: float,
        bucket_diff: float | None,
        buckets_compared: int,
    ) -> None:
        """Fold one analysis cycle's shadow diff into the daily history.

        Hourly samples collapse to one entry per (UTC) day holding the
        day's MAX absolute divergence — a worst-case record, so one good
        sample cannot mask a bad day. History is capped at
        ``ONLINE_PRIOR_DIFF_HISTORY_DAYS`` entries, oldest dropped.
        This is what makes #500's "within tolerance for 30 days"
        promotion gate actually measurable instead of a debug-log
        archaeology exercise.
        """
        day = ensure_utc_datetime(now).date().isoformat()
        history = self.state.diff_history
        scalar_abs = abs(scalar_diff)
        bucket_abs = abs(bucket_diff) if bucket_diff is not None else None
        if history and history[-1].get("date") == day:
            entry = history[-1]
            entry["scalar_diff"] = max(float(entry.get("scalar_diff", 0.0)), scalar_abs)
            prior_bucket = entry.get("bucket_diff")
            if bucket_abs is not None:
                entry["bucket_diff"] = (
                    bucket_abs
                    if prior_bucket is None
                    else max(float(prior_bucket), bucket_abs)
                )
            entry["buckets"] = max(int(entry.get("buckets", 0)), buckets_compared)
        else:
            history.append(
                {
                    "date": day,
                    "scalar_diff": scalar_abs,
                    "bucket_diff": bucket_abs,
                    "buckets": buckets_compared,
                }
            )
            del history[:-ONLINE_PRIOR_DIFF_HISTORY_DAYS]

    def days_within_tolerance(self, tolerance: float) -> int:
        """Return the current streak of trailing days within tolerance.

        Counts consecutive history entries from the newest backwards
        whose worst scalar diff stayed at or under ``tolerance`` — the
        number #500's promotion gate compares against 30. A streak, not
        a total: one bad day resets the clock, which is the point.
        """
        streak = 0
        for entry in reversed(self.state.diff_history):
            try:
                if float(entry.get("scalar_diff", 1.0)) <= tolerance:
                    streak += 1
                else:
                    break
            except (TypeError, ValueError):
                break
        return streak
