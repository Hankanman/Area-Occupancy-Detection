"""Household-wide trajectory tracker for the adjacent-areas Phase 4 wiring.

The boost (``compute_adjacency_boost``) and decay modifier
(``compute_decay_modifier``) need a ``Trajectory`` describing the two
most recently-occupied OTHER areas in the household. This module owns
that rolling state and exposes:

* :meth:`TrajectoryTracker.observe` — call once per area per coordinator
  refresh with the area's previous-tick and current-tick occupancy. End
  edges (``was_occupied`` → ``is_occupied=False``) push onto an internal
  deque keyed by ``end_time``.
* :meth:`TrajectoryTracker.trajectory_for` — given a target area and the
  current time, returns a :class:`~.adjacency.Trajectory` whose
  ``prev_area`` / ``prev_prev_area`` are the two most recent end events
  in the deque, excluding the target itself, and within the configured
  trajectory window.

The deque shape mirrors what ``db.transitions._detect_transitions``
walks at write time, so the runtime trajectory and the learned
transition rows describe the same kind of event.
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
    """Per-coordinator rolling window of recent area-end events.

    The single instance lives on :class:`AreaOccupancyCoordinator`. The
    refresh path observes each area's occupancy edge each tick; the
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
        was_occupied: bool,
        is_occupied: bool,
        now: datetime,
    ) -> None:
        """Record this tick's occupancy edge for ``area_name``.

        Only end edges (``was_occupied=True, is_occupied=False``) push
        onto the deque; other transitions only trigger a window prune
        so stale entries are evicted in step with wall-clock time.

        ``area_name`` is suppressed if it would create a consecutive
        same-area entry — keeps the deque from collapsing when a flaky
        sensor flips occupancy on/off rapidly.
        """
        if (
            was_occupied
            and not is_occupied
            and (not self._recent or self._recent[-1].area_name != area_name)
        ):
            self._recent.append(_RecentEnd(area_name=area_name, end_time=now))
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
        prev: str | None = None
        prev_prev: str | None = None
        for entry in reversed(self._recent):
            if now - entry.end_time > self._window:
                break
            if entry.area_name == target_area:
                continue
            if prev is None:
                prev = entry.area_name
                continue
            if entry.area_name == prev:
                continue
            prev_prev = entry.area_name
            break
        return Trajectory(
            prev_area=prev,
            prev_prev_area=prev_prev,
            hour_of_week=hour_of_week,
        )

    def snapshot(self) -> list[tuple[str, datetime]]:
        """Return the current deque contents as a list (for diagnostics)."""
        return [(e.area_name, e.end_time) for e in self._recent]
