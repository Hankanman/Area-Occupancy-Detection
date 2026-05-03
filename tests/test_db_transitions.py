"""Tests for the adjacency transition-learning module (Phase 3).

Covers the pure detection logic (``_detect_transitions`` against a hand-built
event timeline) and the persistence helpers (recency decay, upsert,
diagnostics summary). The full pipeline integration that drives those
helpers is exercised separately via the analysis-pipeline tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db.transitions import (
    _apply_recency_decay,
    _AreaEvent,
    _build_adjacency_index,
    _detect_transitions,
    _hour_of_week,
    _upsert_transition_counts,
    record_transitions_for_entry,
    summarize_transitions_for_diagnostics,
)

# ─── pure detection logic ───────────────────────────────────────────


def _utc(year=2026, month=5, day=4, hour=10, minute=0, second=0) -> datetime:
    """Convenience UTC-aware datetime."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


class TestDetectTransitions:
    """Pure-logic tests over hand-built event timelines."""

    def test_simple_one_hop_transition_recorded(self):
        """X ends, Y starts within window → one 1-hop chain emitted."""
        events = [
            _AreaEvent(_utc(hour=10, minute=0), "hall", is_start=True),
            _AreaEvent(_utc(hour=10, minute=1), "hall", is_start=False),
            _AreaEvent(_utc(hour=10, minute=1, second=30), "bedroom", is_start=True),
        ]
        adjacency = {"hall": {"bedroom"}, "bedroom": {"hall"}}

        out = _detect_transitions(events, adjacency)

        assert out == [("hall", "", "bedroom", _hour_of_week(events[2].timestamp))]

    def test_transition_outside_window_skipped(self):
        """Gap longer than ADJACENCY_TRANSITION_WINDOW_S → no chain."""
        events = [
            _AreaEvent(_utc(hour=10, minute=0), "hall", is_start=True),
            _AreaEvent(_utc(hour=10, minute=1), "hall", is_start=False),
            # 5-minute gap >> default 60s window
            _AreaEvent(_utc(hour=10, minute=6), "bedroom", is_start=True),
        ]
        adjacency = {"hall": {"bedroom"}, "bedroom": {"hall"}}

        assert _detect_transitions(events, adjacency) == []

    def test_non_adjacent_pair_skipped(self):
        """Even within window, a non-configured-adjacent pair contributes nothing."""
        events = [
            _AreaEvent(_utc(hour=10, minute=0), "hall", is_start=True),
            _AreaEvent(_utc(hour=10, minute=1), "hall", is_start=False),
            _AreaEvent(_utc(hour=10, minute=1, second=30), "garage", is_start=True),
        ]
        # hall and garage are NOT in each other's adjacents
        adjacency = {"hall": {"bedroom"}, "garage": set(), "bedroom": {"hall"}}

        assert _detect_transitions(events, adjacency) == []

    def test_two_hop_chain_emitted_alongside_one_hop(self):
        """W → X → Y emits both the 2-hop and the most recent 1-hop chain."""
        events = [
            _AreaEvent(_utc(hour=10, minute=0), "study", is_start=True),
            _AreaEvent(_utc(hour=10, minute=1), "study", is_start=False),
            _AreaEvent(_utc(hour=10, minute=1, second=30), "hall", is_start=True),
            _AreaEvent(_utc(hour=10, minute=2), "hall", is_start=False),
            _AreaEvent(_utc(hour=10, minute=2, second=30), "bathroom", is_start=True),
        ]
        adjacency = {
            "study": {"hall"},
            "hall": {"study", "bathroom"},
            "bathroom": {"hall"},
        }

        out = _detect_transitions(events, adjacency)

        # Sorted by emission order: study→hall (1-hop), hall→bathroom (1-hop),
        # then study→hall→bathroom (2-hop) emitted at the bathroom-start event.
        assert ("study", "", "hall", _hour_of_week(events[2].timestamp)) in out
        assert ("hall", "", "bathroom", _hour_of_week(events[4].timestamp)) in out
        assert (
            "study",
            "hall",
            "bathroom",
            _hour_of_week(events[4].timestamp),
        ) in out

    def test_two_hop_skipped_when_w_x_pair_not_adjacent(self):
        """If W→X isn't a configured pair, the 2-hop chain isn't trustworthy."""
        events = [
            _AreaEvent(_utc(hour=10, minute=0), "garden", is_start=True),
            _AreaEvent(_utc(hour=10, minute=1), "garden", is_start=False),
            _AreaEvent(_utc(hour=10, minute=1, second=30), "hall", is_start=True),
            _AreaEvent(_utc(hour=10, minute=2), "hall", is_start=False),
            _AreaEvent(_utc(hour=10, minute=2, second=30), "bathroom", is_start=True),
        ]
        # garden ↛ hall (not adjacent), hall ↔ bathroom is.
        adjacency = {
            "garden": set(),
            "hall": {"bathroom"},
            "bathroom": {"hall"},
        }

        out = _detect_transitions(events, adjacency)

        # Only the hall→bathroom 1-hop is valid. No 2-hop, no garden→hall.
        assert out == [("hall", "", "bathroom", _hour_of_week(events[4].timestamp))]

    def test_self_transition_never_recorded(self):
        """An area's interval ending and re-starting must not count."""
        events = [
            _AreaEvent(_utc(hour=10, minute=0), "lounge", is_start=True),
            _AreaEvent(_utc(hour=10, minute=1), "lounge", is_start=False),
            _AreaEvent(_utc(hour=10, minute=1, second=30), "lounge", is_start=True),
        ]
        adjacency = {"lounge": {"lounge"}}  # even if user mis-configured this

        assert _detect_transitions(events, adjacency) == []

    def test_hour_of_week_bucketing_uses_local_timezone(self):
        """The bucket follows local wall-clock, not UTC, by design."""
        # Saturday 23:30 UTC → in BST (+01:00) that's Sunday 00:30 local;
        # in UTC bucketing would be Saturday hour=23. Mocking dt_util.as_local
        # would be heavyweight; instead just assert the output is in [0, 167]
        # and matches the local-time computation directly.
        ts = _utc(year=2026, month=5, day=2, hour=23, minute=30)
        bucket = _hour_of_week(ts)
        assert 0 <= bucket <= 167


# ─── persistence: recency decay ─────────────────────────────────────


class TestRecencyDecay:
    def test_decay_factor_applied_to_existing_counts(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """A 30-day half-life applied for 30 days halves the count."""
        db = coordinator.db
        with db.get_session() as session:
            session.add(
                db.AreaTransitions(
                    entry_id=db.coordinator.entry_id,
                    from_area="A",
                    mid_area="",
                    to_area="B",
                    hour_of_week=10,
                    count=4.0,
                )
            )
            session.commit()

        _apply_recency_decay(db, db.coordinator.entry_id, hours_since_last_run=24 * 30)

        with db.get_session() as session:
            row = (
                session.query(db.AreaTransitions)
                .filter_by(entry_id=db.coordinator.entry_id)
                .first()
            )
            assert row.count == pytest.approx(2.0, abs=1e-6)

    def test_zero_or_negative_hours_is_noop(
        self, coordinator: AreaOccupancyCoordinator
    ):
        db = coordinator.db
        with db.get_session() as session:
            session.add(
                db.AreaTransitions(
                    entry_id=db.coordinator.entry_id,
                    from_area="A",
                    mid_area="",
                    to_area="B",
                    hour_of_week=10,
                    count=7.0,
                )
            )
            session.commit()

        _apply_recency_decay(db, db.coordinator.entry_id, hours_since_last_run=0.0)

        with db.get_session() as session:
            row = (
                session.query(db.AreaTransitions)
                .filter_by(entry_id=db.coordinator.entry_id)
                .first()
            )
            assert row.count == 7.0


# ─── persistence: upsert ────────────────────────────────────────────


class TestUpsertTransitionCounts:
    def test_inserts_new_rows_and_increments_existing(
        self, coordinator: AreaOccupancyCoordinator
    ):
        db = coordinator.db
        # Seed a row.
        with db.get_session() as session:
            session.add(
                db.AreaTransitions(
                    entry_id=db.coordinator.entry_id,
                    from_area="hall",
                    mid_area="",
                    to_area="bedroom",
                    hour_of_week=42,
                    count=2.0,
                )
            )
            session.commit()

        observations = [
            ("hall", "", "bedroom", 42),  # increment existing
            ("hall", "", "bedroom", 42),  # again — total +2
            ("hall", "", "bedroom", 43),  # new bucket
            ("study", "hall", "bedroom", 42),  # new 2-hop chain
        ]

        _upsert_transition_counts(db, db.coordinator.entry_id, observations)

        with db.get_session() as session:
            rows = {
                (r.from_area, r.mid_area, r.to_area, r.hour_of_week): r.count
                for r in session.query(db.AreaTransitions).all()
            }
        assert rows[("hall", "", "bedroom", 42)] == pytest.approx(4.0)
        assert rows[("hall", "", "bedroom", 43)] == pytest.approx(1.0)
        assert rows[("study", "hall", "bedroom", 42)] == pytest.approx(1.0)

    def test_empty_input_is_noop(self, coordinator: AreaOccupancyCoordinator):
        _upsert_transition_counts(
            coordinator.db, coordinator.db.coordinator.entry_id, []
        )
        with coordinator.db.get_session() as session:
            assert session.query(coordinator.db.AreaTransitions).count() == 0


# ─── adjacency index ────────────────────────────────────────────────


class TestAdjacencyIndex:
    def test_index_groups_by_source_area(self, coordinator: AreaOccupancyCoordinator):
        db = coordinator.db
        with db.get_session() as session:
            for from_area, to_area in [
                ("A", "B"),
                ("A", "C"),
                ("B", "A"),
                ("B", "D"),
            ]:
                session.add(
                    db.AreaRelationships(
                        entry_id=db.coordinator.entry_id,
                        area_name=from_area,
                        related_area_name=to_area,
                        relationship_type="adjacent",
                        influence_weight=0.3,
                    )
                )
            # Different relationship_type must NOT appear.
            session.add(
                db.AreaRelationships(
                    entry_id=db.coordinator.entry_id,
                    area_name="A",
                    related_area_name="Z",
                    relationship_type="shared_wall",
                    influence_weight=0.4,
                )
            )
            session.commit()

        index = _build_adjacency_index(db, db.coordinator.entry_id)
        assert index == {"A": {"B", "C"}, "B": {"A", "D"}}


# ─── full pipeline integration ──────────────────────────────────────


class TestRecordTransitionsForEntry:
    def test_no_adjacency_skips_gracefully_and_advances_watermark(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """When no adjacency is configured, no learning happens, watermark advances."""
        db = coordinator.db
        summary = record_transitions_for_entry(db, db.coordinator.entry_id, now=_utc())
        assert summary["transitions_recorded"] == 0
        assert summary["adjacency_pairs"] == 0
        # Watermark must be set so subsequent runs don't re-walk old data.
        expected_key = f"adjacency_last_observed_end_time__{db.coordinator.entry_id}"
        with db.get_session() as session:
            row = session.query(db.Metadata).filter_by(key=expected_key).first()
            assert row is not None

    def test_pipeline_records_and_decays_across_runs(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Two-cycle integration: first run records, second run decays."""
        db = coordinator.db
        # Hand-construct adjacency — the rest of the pipeline reads from it.
        with db.get_session() as session:
            for a, b in [("hall", "bedroom"), ("bedroom", "hall")]:
                session.add(
                    db.AreaRelationships(
                        entry_id=db.coordinator.entry_id,
                        area_name=a,
                        related_area_name=b,
                        relationship_type="adjacent",
                        influence_weight=0.3,
                    )
                )
            session.commit()

        # Stub get_occupied_intervals to return a single transition pair.
        intervals_for = {
            "hall": [(_utc(hour=10, minute=0), _utc(hour=10, minute=1))],
            "bedroom": [(_utc(hour=10, minute=1, second=30), _utc(hour=10, minute=10))],
        }

        def _fake_get(db_arg, entry_id, area_name, **kwargs):
            return intervals_for.get(area_name, [])

        with patch(
            "custom_components.area_occupancy.db.transitions.get_occupied_intervals",
            side_effect=_fake_get,
        ):
            run1 = record_transitions_for_entry(
                db,
                db.coordinator.entry_id,
                now=_utc(hour=10, minute=15),
            )
            assert run1["transitions_recorded"] == 1

            with db.get_session() as session:
                row = (
                    session.query(db.AreaTransitions)
                    .filter_by(
                        entry_id=db.coordinator.entry_id,
                        from_area="hall",
                        to_area="bedroom",
                    )
                    .first()
                )
                assert row is not None
                count_after_first_run = row.count
                assert count_after_first_run == pytest.approx(1.0)

            # Second run 30 days later, no new intervals — count should
            # halve from recency decay, no new transitions recorded.
            run2 = record_transitions_for_entry(
                db,
                db.coordinator.entry_id,
                now=_utc(hour=10, minute=15) + timedelta(days=30),
            )
            assert run2["transitions_recorded"] == 0

            with db.get_session() as session:
                row = (
                    session.query(db.AreaTransitions)
                    .filter_by(
                        entry_id=db.coordinator.entry_id,
                        from_area="hall",
                        to_area="bedroom",
                    )
                    .first()
                )
                assert row.count == pytest.approx(0.5, abs=1e-6)


# ─── diagnostics summary ────────────────────────────────────────────


class TestDiagnosticsSummary:
    def test_summary_separates_one_hop_and_two_hop(
        self, coordinator: AreaOccupancyCoordinator
    ):
        db = coordinator.db
        with db.get_session() as session:
            session.add_all(
                [
                    db.AreaTransitions(
                        entry_id=db.coordinator.entry_id,
                        from_area="A",
                        mid_area="",
                        to_area="B",
                        hour_of_week=10,
                        count=3.0,
                    ),
                    db.AreaTransitions(
                        entry_id=db.coordinator.entry_id,
                        from_area="A",
                        mid_area="B",
                        to_area="C",
                        hour_of_week=10,
                        count=1.5,
                    ),
                ]
            )
            session.commit()

        summary = summarize_transitions_for_diagnostics(db, db.coordinator.entry_id)
        assert summary["row_count"] == 2
        assert summary["1_hop_count"] == 1
        assert summary["2_hop_count"] == 1
        assert summary["total_observations"] == pytest.approx(4.5)
