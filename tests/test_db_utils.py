"""Tests for database utility functions."""

from contextlib import contextmanager
from datetime import UTC, timedelta, timezone

from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db.utils import (
    is_intervals_empty,
    is_timestamp_in_prepared_intervals,
    is_timestamp_occupied,
    is_valid_state,
    prepare_occupied_intervals,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


class TestIsValidState:
    """Test is_valid_state function."""

    def test_valid_states(self):
        """Test that valid states return True."""
        assert is_valid_state("on") is True
        assert is_valid_state("off") is True
        assert is_valid_state("playing") is True
        assert is_valid_state("idle") is True
        assert is_valid_state(0) is True
        assert is_valid_state(1) is True
        assert is_valid_state(25.5) is True

    def test_invalid_states(self):
        """Test that invalid states return False."""
        assert is_valid_state("unknown") is False
        assert is_valid_state("unavailable") is False
        assert is_valid_state(None) is False
        assert is_valid_state("") is False
        assert is_valid_state("NaN") is False


class TestIsIntervalsEmpty:
    """Test is_intervals_empty function."""

    def test_empty_intervals(self, coordinator: AreaOccupancyCoordinator):
        """Test is_intervals_empty with empty intervals table."""
        db = coordinator.db
        result = is_intervals_empty(db)
        assert result is True

    def test_non_empty_intervals(self, coordinator: AreaOccupancyCoordinator):
        """Test is_intervals_empty with non-empty intervals table."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        end = dt_util.utcnow()
        start = end - timedelta(seconds=60)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_session() as session:
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                state="on",
                start_time=start,
                end_time=end,
                duration_seconds=60,
                aggregation_level="raw",
            )
            session.add(interval)
            session.commit()

        result = is_intervals_empty(db)
        assert result is False

    def test_no_such_table_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test is_intervals_empty when table doesn't exist."""
        db = coordinator.db

        @contextmanager
        def mock_session():
            class MockSession:
                def query(self, *args):
                    raise SQLAlchemyError("no such table: intervals")

                def close(self):
                    pass

            yield MockSession()

        monkeypatch.setattr(db, "get_session", mock_session)
        result = is_intervals_empty(db)
        assert result is True  # Should return True when table doesn't exist

    def test_other_sqlalchemy_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test is_intervals_empty with other SQLAlchemy error."""
        db = coordinator.db

        @contextmanager
        def mock_session():
            class MockSession:
                def query(self, *args):
                    raise SQLAlchemyError("Connection error")

                def close(self):
                    pass

            yield MockSession()

        monkeypatch.setattr(db, "get_session", mock_session)
        result = is_intervals_empty(db)
        assert result is True  # Should return True as fallback

    def test_home_assistant_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test is_intervals_empty with HomeAssistantError."""
        db = coordinator.db

        @contextmanager
        def mock_session():
            raise HomeAssistantError("Database error")

        monkeypatch.setattr(db, "get_session", mock_session)
        result = is_intervals_empty(db)
        assert result is True  # Should return True as fallback


class TestPreparedIntervalLookup:
    """Tests for ``prepare_occupied_intervals`` + ``is_timestamp_in_prepared_intervals``.

    Behaviour parity with the legacy O(N) ``is_timestamp_occupied`` helper
    is the core invariant — the bisect-based lookup is only useful if it
    answers the same questions for the inputs the analysis pipeline
    produces (sorted, non-overlapping intervals).
    """

    def test_empty_inputs(self) -> None:
        starts, ends = prepare_occupied_intervals([])
        assert starts == [] and ends == []
        assert (
            is_timestamp_in_prepared_intervals(dt_util.utcnow(), starts, ends) is False
        )

    def test_membership_matches_legacy_helper(self) -> None:
        """Every probe lands the same answer through both code paths.

        Covers the boundary cases the legacy tests already enforce
        (start inclusive, end exclusive, gap, before, after) plus a
        deliberately-shuffled input order so the sort step inside
        ``prepare_occupied_intervals`` is exercised.
        """
        now = dt_util.utcnow()
        intervals = [
            (now + timedelta(hours=2), now + timedelta(hours=3)),  # second
            (now, now + timedelta(hours=1)),  # first
            (now + timedelta(hours=5), now + timedelta(hours=6)),  # third
        ]
        starts, ends = prepare_occupied_intervals(intervals)
        # Sort guarantee — bisect relies on it.
        assert starts == sorted(starts)

        probes = [
            now - timedelta(seconds=1),
            now,  # at start (inclusive)
            now + timedelta(minutes=30),  # inside first
            now + timedelta(hours=1),  # at end (exclusive)
            now + timedelta(hours=1, minutes=30),  # in gap
            now + timedelta(hours=2, minutes=30),  # inside second
            now + timedelta(hours=4),  # in gap
            now + timedelta(hours=5, minutes=30),  # inside third
            now + timedelta(hours=10),  # past end
        ]
        for probe in probes:
            assert is_timestamp_in_prepared_intervals(
                probe, starts, ends
            ) == is_timestamp_occupied(probe, intervals), probe

    def test_naive_inputs_normalised_to_utc(self) -> None:
        """Naive datetimes are interpreted as UTC, matching ``to_utc``.

        The hot loop callers in ``analyze_correlation`` mostly pass
        already-aware datetimes (``from_db_utc`` re-attaches UTC), but
        this guards against a regression that breaks parity with the
        legacy helper for ``tzinfo=None`` inputs.
        """
        naive_now = dt_util.utcnow().replace(tzinfo=None)
        intervals = [(naive_now, naive_now + timedelta(hours=1))]
        starts, ends = prepare_occupied_intervals(intervals)
        # Endpoints are stored UTC-aware after normalisation.
        assert starts[0].tzinfo is not None
        assert ends[0].tzinfo is not None
        # Probe with a non-UTC aware timestamp that converts to a UTC
        # instant inside the interval — the helper must convert the
        # probe to UTC before comparison, otherwise a Sydney clock at
        # 12:30 wouldn't match a UTC interval covering ~04:30 UTC.
        plus_two = timezone(timedelta(hours=2))
        # naive_now + 30min, interpreted as UTC, then re-expressed as +02:00
        probe = (
            (naive_now + timedelta(minutes=30)).replace(tzinfo=UTC).astimezone(plus_two)
        )
        assert is_timestamp_in_prepared_intervals(probe, starts, ends) is True
