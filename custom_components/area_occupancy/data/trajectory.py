"""Household-wide trajectory tracker for the adjacent-areas Phase 4 wiring.

The boost (``compute_adjacency_boost``) and decay modifier
(``compute_decay_modifier``) need a ``Trajectory`` describing the two
areas the household most recently left, other than the target. This
module owns that rolling state and exposes:

* :meth:`TrajectoryTracker.observe` — call once per area per coordinator
  refresh with whether the area's ground-truth presence evidence (motion,
  media, sleep) was active on the previous tick and is active now. End
  edges (``was_present`` → ``is_present=False``) are departures and push
  onto an internal deque keyed by ``end_time``.
* :meth:`TrajectoryTracker.trajectory_for` — given a target area and the
  current time, returns a :class:`~.adjacency.Trajectory` whose
  ``prev_area`` / ``prev_prev_area`` are the two most recent departures
  in the deque, excluding the target itself, and within the configured
  trajectory window.

The deque shape mirrors what ``db.transitions._detect_transitions``
walks at write time, and the coordinator feeds it the same kind of
event: the learner's area ends are the ends of ground-truth occupied
intervals, i.e. the moment an area's motion, media and sleep sensors go
quiet. An area's probability crossing below its threshold is not a
departure; it happens minutes later, while the decay tail runs out.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..const import ADJACENCY_TRAJECTORY_WINDOW_S
from .adjacency import Trajectory


@dataclass(frozen=True)
class _RecentEnd:
    """One area-end event captured for trajectory purposes."""

    area_name: str
    end_time: datetime


# Deque depth: with 2-hop lookups we only ever look at the last two
# non-target entries, but a small safety margin lets us tolerate runs
# of self-edges (same area ending repeatedly across ticks) without
# collapsing the trajectory.
_DEQUE_MAX = 8


class TrajectoryTracker:
    """Per-coordinator rolling window of recent departures.

    The single instance lives on :class:`AreaOccupancyCoordinator`. The
    refresh path observes each area's presence edge each tick; the
    boost/decay paths read the trajectory snapshot back out.
    """

    def __init__(self, window_seconds: int = ADJACENCY_TRAJECTORY_WINDOW_S) -> None:
        """Initialize an empty tracker with the configured window."""
        self._window = timedelta(seconds=window_seconds)
        self._recent: deque[_RecentEnd] = deque(maxlen=_DEQUE_MAX)

    def observe(
        self,
        area_name: str,
        *,
        was_present: bool,
        is_present: bool,
        now: datetime,
    ) -> None:
        """Record this tick's presence edge for ``area_name``.

        Only end edges (``was_present=True, is_present=False``) push
        onto the deque; other transitions only trigger a window prune
        so stale entries are evicted in step with wall-clock time.

        An end for the area that is already newest in the deque refreshes
        that entry's ``end_time`` instead of adding another, as
        ``_detect_transitions`` does: a sensor cycling on and off as
        someone moves around a room doesn't bloat the deque, and the
        entry keeps the latest time the area was left.
        """
        if was_present and not is_present:
            entry = _RecentEnd(area_name=area_name, end_time=now)
            if self._recent and self._recent[-1].area_name == area_name:
                self._recent[-1] = entry
            else:
                self._recent.append(entry)
        self._prune(now)

    def _prune(self, now: datetime) -> None:
        """Drop entries older than the trajectory window."""
        while self._recent and now - self._recent[0].end_time > self._window:
            self._recent.popleft()

    def trajectory_for(
        self, target_area: str, *, hour_of_week: int, now: datetime
    ) -> Trajectory:
        """Return the (prev, prev_prev) trajectory for predicting target.

        Walks the deque newest-to-oldest within the trajectory window,
        skipping ``target_area`` itself, and picks at most the two most
        recent distinct-area entries. Returns ``Trajectory(None, None)``
        when no relevant ends exist.
        """
        prev: _RecentEnd | None = None
        prev_prev: str | None = None
        for entry in reversed(self._recent):
            if now - entry.end_time > self._window:
                break
            if entry.area_name == target_area:
                continue
            if prev is None:
                prev = entry
                continue
            if entry.area_name == prev.area_name:
                continue
            prev_prev = entry.area_name
            break
        return Trajectory(
            prev_area=prev.area_name if prev is not None else None,
            prev_prev_area=prev_prev,
            hour_of_week=hour_of_week,
            prev_end_time=prev.end_time if prev is not None else None,
        )

    def snapshot(self) -> list[tuple[str, datetime]]:
        """Return the current deque contents as a list (for diagnostics)."""
        return [(e.area_name, e.end_time) for e in self._recent]
