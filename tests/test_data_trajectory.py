"""Tests for the household trajectory tracker (Phase 4b)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from custom_components.area_occupancy.data.adjacency import Trajectory
from custom_components.area_occupancy.data.trajectory import TrajectoryTracker


def _t(seconds: int = 0) -> datetime:
    """Helper: build a deterministic UTC timestamp at a given offset."""
    return datetime(2026, 5, 4, 10, 0, 0, tzinfo=UTC) + timedelta(seconds=seconds)


class TestObserve:
    def test_end_edge_pushes_onto_deque(self):
        t = TrajectoryTracker()
        t.observe("hall", was_present=True, is_present=False, now=_t(0))
        assert t.snapshot() == [("hall", _t(0))]

    def test_non_end_edges_do_not_push(self):
        t = TrajectoryTracker()
        # off → off, off → on, on → on: none push
        t.observe("hall", was_present=False, is_present=False, now=_t(0))
        t.observe("hall", was_present=False, is_present=True, now=_t(1))
        t.observe("hall", was_present=True, is_present=True, now=_t(2))
        assert t.snapshot() == []

    def test_consecutive_same_area_ends_collapse_to_latest(self):
        """A sensor cycling on and off keeps one entry, at the latest end.

        Mirrors ``db.transitions._detect_transitions``, which refreshes the
        end time rather than keeping the first one.
        """
        t = TrajectoryTracker()
        t.observe("hall", was_present=True, is_present=False, now=_t(0))
        t.observe("hall", was_present=True, is_present=False, now=_t(1))
        t.observe("hall", was_present=True, is_present=False, now=_t(2))
        assert t.snapshot() == [("hall", _t(2))]

    def test_alternating_areas_are_kept_distinct(self):
        t = TrajectoryTracker()
        t.observe("hall", was_present=True, is_present=False, now=_t(0))
        t.observe("study", was_present=True, is_present=False, now=_t(10))
        t.observe("hall", was_present=True, is_present=False, now=_t(20))
        assert [e[0] for e in t.snapshot()] == ["hall", "study", "hall"]

    def test_window_prune_drops_stale_entries(self):
        t = TrajectoryTracker(window_seconds=300)
        t.observe("hall", was_present=True, is_present=False, now=_t(0))
        t.observe("study", was_present=True, is_present=False, now=_t(60))
        t.observe("kitchen", was_present=True, is_present=False, now=_t(400))
        # Only kitchen survives — hall and study are >300s old.
        assert [e[0] for e in t.snapshot()] == ["kitchen"]


class TestTrajectoryFor:
    def test_no_recents_returns_empty_trajectory(self):
        t = TrajectoryTracker()
        traj = t.trajectory_for("bedroom", hour_of_week=10, now=_t(0))
        assert traj == Trajectory(prev_area=None, prev_prev_area=None, hour_of_week=10)

    def test_target_area_is_skipped(self):
        """The target's own ends never appear in its trajectory."""
        t = TrajectoryTracker()
        t.observe("bedroom", was_present=True, is_present=False, now=_t(0))
        t.observe("hall", was_present=True, is_present=False, now=_t(10))
        t.observe("bedroom", was_present=True, is_present=False, now=_t(20))
        traj = t.trajectory_for("bedroom", hour_of_week=10, now=_t(30))
        assert traj.prev_area == "hall"
        assert traj.prev_prev_area is None

    def test_two_distinct_neighbours_yield_two_hop_trajectory(self):
        t = TrajectoryTracker()
        t.observe("study", was_present=True, is_present=False, now=_t(0))
        t.observe("hall", was_present=True, is_present=False, now=_t(60))
        traj = t.trajectory_for("bathroom", hour_of_week=42, now=_t(70))
        assert traj.prev_area == "hall"
        assert traj.prev_prev_area == "study"
        assert traj.hour_of_week == 42

    def test_window_excluded_entries_dropped_from_trajectory(self):
        t = TrajectoryTracker(window_seconds=300)
        t.observe("study", was_present=True, is_present=False, now=_t(0))
        t.observe("hall", was_present=True, is_present=False, now=_t(60))
        # Querying 350s after study's end: study is outside the window,
        # only hall remains — falls back to 1-hop trajectory.
        traj = t.trajectory_for("bathroom", hour_of_week=10, now=_t(350))
        assert traj.prev_area == "hall"
        assert traj.prev_prev_area is None

    def test_repeated_neighbour_does_not_double_up(self):
        """If only one distinct other area is in window, prev_prev stays None."""
        t = TrajectoryTracker()
        t.observe("hall", was_present=True, is_present=False, now=_t(0))
        t.observe("study", was_present=True, is_present=False, now=_t(10))
        t.observe("hall", was_present=True, is_present=False, now=_t(20))
        traj = t.trajectory_for("bathroom", hour_of_week=10, now=_t(30))
        # Newest-first: hall, study, hall → prev=hall, prev_prev=study.
        assert traj.prev_area == "hall"
        assert traj.prev_prev_area == "study"

    def test_prev_end_time_is_when_prev_area_was_left(self):
        """The trajectory carries the departure time of ``prev_area``."""
        t = TrajectoryTracker()
        t.observe("study", was_present=True, is_present=False, now=_t(0))
        t.observe("hall", was_present=True, is_present=False, now=_t(60))
        traj = t.trajectory_for("bathroom", hour_of_week=10, now=_t(70))
        assert traj.prev_area == "hall"
        assert traj.prev_end_time == _t(60)

    def test_prev_end_time_is_none_without_trajectory(self):
        t = TrajectoryTracker()
        traj = t.trajectory_for("bathroom", hour_of_week=10, now=_t(0))
        assert traj.prev_end_time is None
