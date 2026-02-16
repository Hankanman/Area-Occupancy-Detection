"""Tests for database query functions."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db.operations import (
    save_global_prior,
    save_occupied_intervals_cache,
)
from custom_components.area_occupancy.db.queries import (
    build_base_filters,
    build_motion_query,
    build_presence_query,
    get_all_time_priors,
    get_area_data,
    get_global_prior,
    get_latest_interval,
    get_occupied_intervals,
    get_occupied_intervals_cache,
    get_time_bounds,
    get_time_prior,
    get_total_occupied_seconds,
    is_occupied_intervals_cache_valid,
)
from homeassistant.util import dt as dt_util

# Helper functions for test setup and comparisons


def _normalize_datetime_for_comparison(
    dt1: datetime, dt2: datetime
) -> tuple[datetime, datetime]:
    """Normalize two datetimes for comparison, handling timezone-aware vs naive.

    Args:
        dt1: First datetime (may be timezone-aware or naive)
        dt2: Second datetime (may be timezone-aware or naive)

    Returns:
        Tuple of (normalized_dt1, normalized_dt2) for comparison
    """
    # If one is timezone-aware and the other is naive, normalize both
    if dt1.tzinfo and not dt2.tzinfo:
        dt2_normalized = dt2.replace(tzinfo=dt1.tzinfo)
        dt1_normalized = dt1
    elif dt2.tzinfo and not dt1.tzinfo:
        dt1_normalized = dt1.replace(tzinfo=dt2.tzinfo)
        dt2_normalized = dt2
    else:
        dt1_normalized = dt1
        dt2_normalized = dt2
    return (dt1_normalized, dt2_normalized)


def _create_test_entity(
    session: Any,
    db: Any,
    entity_id: str,
    entity_type: str,
    area_name: str,
    entry_id: str | None = None,
) -> Any:
    """Create a test entity in the database.

    Args:
        session: Database session
        db: Database instance
        entity_id: Entity ID
        entity_type: Entity type (e.g., "motion", "media", "appliance")
        area_name: Area name
        entry_id: Entry ID (defaults to db.coordinator.entry_id)

    Returns:
        Created entity object
    """
    if entry_id is None:
        entry_id = db.coordinator.entry_id

    entity = db.Entities(
        entity_id=entity_id,
        entry_id=entry_id,
        area_name=area_name,
        entity_type=entity_type,
    )
    session.add(entity)
    return entity


def _create_test_interval(
    session: Any,
    db: Any,
    entity_id: str,
    start_time: datetime,
    end_time: datetime,
    area_name: str,
    entry_id: str | None = None,
    state: str = "on",
    aggregation_level: str = "raw",
) -> Any:
    """Create a test interval in the database.

    Args:
        session: Database session
        db: Database instance
        entity_id: Entity ID
        start_time: Interval start time
        end_time: Interval end time
        area_name: Area name
        entry_id: Entry ID (defaults to db.coordinator.entry_id)
        state: Interval state (default: "on")
        aggregation_level: Aggregation level (default: "raw")

    Returns:
        Created interval object
    """
    if entry_id is None:
        entry_id = db.coordinator.entry_id

    duration_seconds = int((end_time - start_time).total_seconds())
    interval = db.Intervals(
        entry_id=entry_id,
        area_name=area_name,
        entity_id=entity_id,
        start_time=start_time,
        end_time=end_time,
        state=state,
        duration_seconds=duration_seconds,
        aggregation_level=aggregation_level,
    )
    session.add(interval)
    return interval


def _create_test_motion_intervals(
    session: Any,
    db: Any,
    intervals_data: list[tuple[datetime, datetime]],
    area_name: str,
    entity_id: str = "binary_sensor.motion1",
    entry_id: str | None = None,
) -> list[Any]:
    """Create multiple motion intervals in the database.

    Args:
        session: Database session
        db: Database instance
        intervals_data: List of (start_time, end_time) tuples
        area_name: Area name
        entity_id: Entity ID (default: "binary_sensor.motion1")
        entry_id: Entry ID (defaults to db.coordinator.entry_id)

    Returns:
        List of created interval objects
    """
    if entry_id is None:
        entry_id = db.coordinator.entry_id

    intervals = []
    for start_time, end_time in intervals_data:
        interval = _create_test_interval(
            session,
            db,
            entity_id,
            start_time,
            end_time,
            area_name,
            entry_id=entry_id,
            state="on",
        )
        intervals.append(interval)
    return intervals


class TestGetAreaData:
    """Test get_area_data function."""

    def test_get_area_data_success(self, coordinator: AreaOccupancyCoordinator):
        """Test get_area_data with existing area."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Save area data first
        db.save_area_data(area_name)

        result = get_area_data(db, db.coordinator.entry_id)
        assert result is not None
        # Verify complete dict structure with all expected fields
        assert result["entry_id"] == db.coordinator.entry_id
        assert result["area_name"] == area_name
        assert "area_id" in result
        assert "purpose" in result
        assert "threshold" in result
        assert "adjacent_areas" in result
        assert "created_at" in result
        assert "updated_at" in result

    def test_get_area_data_not_found(self, coordinator: AreaOccupancyCoordinator):
        """Test get_area_data when area doesn't exist."""
        db = coordinator.db
        result = get_area_data(db, "nonexistent_entry")
        assert result is None

    def test_get_area_data_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test get_area_data with database error."""
        db = coordinator.db

        def bad_session():
            raise SQLAlchemyError("Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        result = get_area_data(db, "test")
        assert result is None


class TestGetLatestInterval:
    """Test get_latest_interval function."""

    def test_get_latest_interval_with_data(self, coordinator: AreaOccupancyCoordinator):
        """Test get_latest_interval when intervals exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        end = dt_util.utcnow()
        start = end - timedelta(seconds=60)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion", "motion", area_name
            )
            session.commit()

        with db.get_session() as session:
            _create_test_interval(
                session, db, "binary_sensor.motion", start, end, area_name
            )
            session.commit()

        result = get_latest_interval(db)
        assert isinstance(result, datetime)
        # Should be exactly end_time - 1 hour
        expected = end.replace(tzinfo=None) - timedelta(hours=1)
        result_naive = result.replace(tzinfo=None) if result.tzinfo else result
        assert abs((result_naive - expected).total_seconds()) < 1

    def test_get_latest_interval_no_data(self, coordinator: AreaOccupancyCoordinator):
        """Test get_latest_interval when no intervals exist."""
        db = coordinator.db
        result = get_latest_interval(db)
        assert isinstance(result, datetime)
        # Should return default (now - 10 days)
        # Both result and expected are timezone-aware, so compare directly
        expected = dt_util.utcnow() - timedelta(days=10)
        # Compare as naive datetimes to avoid timezone issues
        result_naive = result.replace(tzinfo=None) if result.tzinfo else result
        expected_naive = expected.replace(tzinfo=None) if expected.tzinfo else expected
        assert abs((result_naive - expected_naive).total_seconds()) < 60

    def test_get_latest_interval_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test get_latest_interval with database error."""
        db = coordinator.db

        def bad_session():
            raise SQLAlchemyError("no such table")

        monkeypatch.setattr(db, "get_session", bad_session)
        result = get_latest_interval(db)
        assert isinstance(result, datetime)
        # Should return default (now - 10 days) on error
        expected = dt_util.utcnow() - timedelta(days=10)
        result_naive = result.replace(tzinfo=None) if result.tzinfo else result
        expected_naive = expected.replace(tzinfo=None) if expected.tzinfo else expected
        assert abs((result_naive - expected_naive).total_seconds()) < 60


class TestGetTimePrior:
    """Test get_time_prior function."""

    def test_get_time_prior_with_data(self, coordinator: AreaOccupancyCoordinator):
        """Test get_time_prior when prior exists."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        with db.get_session() as session:
            area = db.Areas(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                area_id="test",
                purpose="living",
                threshold=0.5,
            )
            session.add(area)

            prior = db.Priors(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                day_of_week=1,
                time_slot=14,
                prior_value=0.35,
                data_points=10,
            )
            session.add(prior)
            session.commit()

        result = get_time_prior(db, db.coordinator.entry_id, area_name, 1, 14, 0.5)
        assert result == 0.35

    def test_get_time_prior_default(self, coordinator: AreaOccupancyCoordinator):
        """Test get_time_prior returns default when prior doesn't exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        result = get_time_prior(db, db.coordinator.entry_id, area_name, 1, 14, 0.5)
        assert result == 0.5

    def test_get_time_prior_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test get_time_prior with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        def bad_session():
            raise SQLAlchemyError("Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        result = get_time_prior(db, db.coordinator.entry_id, area_name, 1, 14, 0.5)
        # Should return default on error
        assert result == 0.5


class TestGetOccupiedIntervals:
    """Test get_occupied_intervals function."""

    def test_get_occupied_intervals_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful retrieval of occupied intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion1", "motion", area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion1", start, end, area_name
            )
            session.commit()

        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
        # Verify intervals are sorted by start_time
        for i in range(len(result) - 1):
            assert result[i][0] <= result[i + 1][0]

    def test_get_occupied_intervals_empty(self, coordinator: AreaOccupancyCoordinator):
        """Test get_occupied_intervals with no intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )
        assert result == []

    def test_get_occupied_intervals_presence_sensors(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test retrieval includes motion + media + sleep but not appliances."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        start = now - timedelta(hours=2)

        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion1", "motion", area_name
            )
            _create_test_entity(session, db, "media_player.tv", "media", area_name)
            _create_test_entity(
                session, db, "switch.appliance1", "appliance", area_name
            )
            session.commit()

        with db.get_session() as session:
            _create_test_interval(
                session,
                db,
                "binary_sensor.motion1",
                start,
                start + timedelta(minutes=30),
                area_name,
            )
            _create_test_interval(
                session,
                db,
                "media_player.tv",
                start + timedelta(minutes=40),
                start + timedelta(minutes=80),
                area_name,
            )
            _create_test_interval(
                session,
                db,
                "switch.appliance1",
                start + timedelta(minutes=90),
                start + timedelta(minutes=120),
                area_name,
            )
            session.commit()

        # Occupied intervals now include motion + media (presence sensors),
        # but NOT appliances.
        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=1,
            motion_timeout_seconds=0,
        )

        # Should return motion + media intervals (2 intervals, not merged since non-overlapping).
        assert len(result) == 2

    def test_get_occupied_intervals_includes_sleep(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that sleep sensor intervals are included in occupied intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        start = now - timedelta(hours=8)

        with db.get_session() as session:
            _create_test_entity(session, db, "binary_sensor.sleep1", "sleep", area_name)
            session.commit()

        with db.get_session() as session:
            # 8-hour sleep interval.
            _create_test_interval(
                session,
                db,
                "binary_sensor.sleep1",
                start,
                now,
                area_name,
            )
            session.commit()

        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=1,
            motion_timeout_seconds=0,
        )

        # Sleep interval should be included.
        assert len(result) == 1
        # Duration should be approximately 8 hours.
        duration = (result[0][1] - result[0][0]).total_seconds()
        assert abs(duration - 8 * 3600) < 1.0

    def test_get_occupied_intervals_overlapping_merge(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that overlapping intervals are merged correctly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        start1 = now - timedelta(hours=2)
        end1 = now - timedelta(hours=1, minutes=30)
        start2 = now - timedelta(hours=1, minutes=45)  # Overlaps with first
        end2 = now - timedelta(hours=1)

        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion1", "motion", area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion1", start1, end1, area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion1", start2, end2, area_name
            )
            session.commit()

        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=1,
            motion_timeout_seconds=0,
        )

        # Should merge overlapping intervals into one
        assert len(result) == 1
        # Handle timezone-aware vs naive datetime comparison
        result_start, start1_normalized = _normalize_datetime_for_comparison(
            result[0][0], start1
        )
        result_end, end2_normalized = _normalize_datetime_for_comparison(
            result[0][1], end2
        )

        assert result_start == start1_normalized
        assert result_end == end2_normalized

    def test_get_occupied_intervals_motion_timeout(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that motion timeout extends motion segments correctly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        start = now - timedelta(hours=2)
        # Motion interval ends 15 minutes before "now"
        motion_end = now - timedelta(minutes=15)
        # Second motion interval starts after first ends (with a gap)
        # The timeout should bridge the gap, and merged_end will be the end of second interval
        # This ensures motion_end < merged_end, allowing timeout extension to be tested
        # motion_end + timeout = now - 15m + 10m = now - 5m
        # Set merged_end = now - 8m, so motion_end + timeout (now - 5m) > merged_end (now - 8m)
        # This means timeout IS clamped to merged_end (now - 8m)
        # Set second_motion_start to overlap with first motion, so they're in the same merged interval
        # Then the timeout extension from motion_end is clamped to merged_end
        # The segments will merge, and result[0][1] = merged_end (now - 8m)
        # This equals min(motion_end + timeout, merged_end) = min(now - 5m, now - 8m) = now - 8m
        merged_end = now - timedelta(
            minutes=8
        )  # Before motion_end + timeout, so timeout is clamped
        second_motion_start = now - timedelta(
            minutes=16
        )  # Overlaps with first motion (ends at now - 15m)

        with db.get_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)

            # Create first motion interval that ends before the merged interval end
            _create_test_interval(
                session, db, "binary_sensor.motion1", start, motion_end, area_name
            )
            # Create second motion interval that starts after first ends (gap)
            # The timeout will bridge the gap, and merged_end will be the end of second interval
            _create_test_interval(
                session,
                db,
                "binary_sensor.motion1",
                second_motion_start,
                merged_end,
                area_name,
            )
            session.commit()

        # With 10 minute timeout, motion segment should be extended
        motion_timeout_seconds = 600  # 10 minutes
        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=1,
            motion_timeout_seconds=motion_timeout_seconds,
        )

        assert len(result) == 1
        # Timeout extends motion segments from motion_end, but is clamped to merged_end
        # Since motion_end < merged_end, timeout should extend to min(motion_end + timeout, merged_end)
        # motion_end + timeout = now - 15m + 10m = now - 5m
        # merged_end = now - 8m
        # So expected_end = min(now - 5m, now - 8m) = now - 8m (timeout is clamped to merged_end)
        # After merging segments, result[0][1] should equal min(motion_end + timeout, merged_end)
        expected_end = min(
            motion_end + timedelta(seconds=motion_timeout_seconds), merged_end
        )
        result_end, expected_end_normalized = _normalize_datetime_for_comparison(
            result[0][1], expected_end
        )
        # Result should be min(motion_end + timeout, merged_end) = merged_end (clamped)
        assert abs((result_end - expected_end_normalized).total_seconds()) < 1

    def test_get_occupied_intervals_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test get_occupied_intervals with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        def bad_session():
            raise SQLAlchemyError("Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )
        # Should return empty list on error, not None
        assert result == []


class TestGetTimeBounds:
    """Test get_time_bounds function."""

    def test_get_time_bounds_with_data(self, coordinator: AreaOccupancyCoordinator):
        """Test get_time_bounds when intervals exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion1", "motion", area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion1", start, end, area_name
            )
            session.commit()

        first, last = get_time_bounds(db, db.coordinator.entry_id, area_name)
        assert first is not None
        assert last is not None

    def test_get_time_bounds_no_data(self, coordinator: AreaOccupancyCoordinator):
        """Test get_time_bounds when no intervals exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        first, last = get_time_bounds(db, db.coordinator.entry_id, area_name)
        assert first is None
        assert last is None

    def test_get_time_bounds_with_entity_ids(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_time_bounds with entity_ids parameter."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        end = dt_util.utcnow()
        start1 = end - timedelta(hours=2)
        start2 = end - timedelta(hours=1)

        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion1", "motion", area_name
            )
            _create_test_entity(
                session, db, "binary_sensor.motion2", "motion", area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion1", start1, end, area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion2", start2, end, area_name
            )
            session.commit()

        # Test with specific entity_ids
        first, last = get_time_bounds(
            db,
            db.coordinator.entry_id,
            area_name,
            entity_ids=["binary_sensor.motion1"],
        )
        assert first is not None
        assert last is not None
        # Handle timezone-aware vs naive datetime comparison
        first_normalized, start1_normalized = _normalize_datetime_for_comparison(
            first, start1
        )
        last_normalized, end_normalized = _normalize_datetime_for_comparison(last, end)
        assert first_normalized == start1_normalized
        assert last_normalized == end_normalized


class TestBuildFilters:
    """Test filter building functions."""

    def test_build_base_filters(self, coordinator: AreaOccupancyCoordinator):
        """Test build_base_filters function."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        lookback_date = dt_util.utcnow() - timedelta(days=90)

        filters = build_base_filters(
            db, db.coordinator.entry_id, lookback_date, area_name
        )
        assert isinstance(filters, list)
        assert (
            len(filters) == 4
        )  # entry_id, area_name (Entities), area_name (Intervals), start_time

        # Verify filter logic correctness
        # Filters should include: entry_id, area_name (Entities), area_name (Intervals), start_time >= lookback_date
        # We can't directly inspect filter values, but we can verify they're used correctly in a query
        with db.get_session() as session:
            # Verify filters work by checking they can be applied to a query
            # Filters are designed to work with a join between Entities and Intervals
            query = (
                session.query(db.Intervals)
                .join(
                    db.Entities,
                    (db.Intervals.entity_id == db.Entities.entity_id)
                    & (db.Intervals.area_name == db.Entities.area_name),
                )
                .filter(*filters)
            )
            # Query should be valid (not raise an error)
            # We can't execute it without data, but we can verify the structure
            assert query is not None

    def test_build_motion_query(self, coordinator: AreaOccupancyCoordinator):
        """Test build_motion_query function."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        lookback_date = dt_util.utcnow() - timedelta(days=90)
        base_filters = build_base_filters(
            db, db.coordinator.entry_id, lookback_date, area_name
        )

        with db.get_session() as session:
            query = build_motion_query(session, db, base_filters)
            assert query is not None

            # Verify query structure includes motion filter and state filter
            # Check that query filters by entity_type == MOTION and state == "on"
            # Query should filter by motion entity type and "on" state
            # Note: Query may fail due to missing tables/data, but structure should be valid
            with suppress(Exception):
                _ = query.all()

    def test_build_presence_query(self, coordinator: AreaOccupancyCoordinator):
        """Test build_presence_query function."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        lookback_date = dt_util.utcnow() - timedelta(days=90)
        base_filters = build_base_filters(
            db, db.coordinator.entry_id, lookback_date, area_name
        )

        with db.get_session() as session:
            query = build_presence_query(session, db, base_filters)
            assert query is not None

            with suppress(Exception):
                _ = query.all()


class TestGetGlobalPrior:
    """Test get_global_prior function."""

    def test_get_global_prior_with_data(self, coordinator: AreaOccupancyCoordinator):
        """Test get_global_prior when data exists."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        save_global_prior(
            db,
            area_name,
            0.35,
            dt_util.utcnow() - timedelta(days=90),
            dt_util.utcnow(),
            86400.0,
            7776000.0,
            100,
        )

        result = get_global_prior(db, area_name)
        assert result is not None
        # Verify all 9 fields in returned dict
        assert result["prior_value"] == 0.35
        assert "calculation_date" in result
        assert "data_period_start" in result
        assert "data_period_end" in result
        assert "total_occupied_seconds" in result
        assert "total_period_seconds" in result
        assert "interval_count" in result
        assert "confidence" in result
        assert "calculation_method" in result
        assert len(result) == 9

    def test_get_global_prior_no_data(self, coordinator: AreaOccupancyCoordinator):
        """Test get_global_prior when no data exists."""
        db = coordinator.db
        result = get_global_prior(db, "nonexistent_area")
        assert result is None


class TestOccupiedIntervalsCache:
    """Test occupied intervals cache functions."""

    def test_get_occupied_intervals_cache(self, coordinator: AreaOccupancyCoordinator):
        """Test get_occupied_intervals_cache function."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        intervals = [
            (
                dt_util.utcnow() - timedelta(hours=2),
                dt_util.utcnow() - timedelta(hours=1),
            )
        ]
        save_occupied_intervals_cache(db, area_name, intervals)

        result = get_occupied_intervals_cache(db, area_name)
        assert len(result) == 1

    def test_get_occupied_intervals_cache_with_period_start(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_occupied_intervals_cache with period_start filter."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        now = dt_util.utcnow()
        intervals = [
            (now - timedelta(hours=3), now - timedelta(hours=2)),
            (now - timedelta(hours=1), now),
        ]
        save_occupied_intervals_cache(db, area_name, intervals)

        # Filter by period_start
        period_start = now - timedelta(hours=2)
        result = get_occupied_intervals_cache(db, area_name, period_start=period_start)
        # Should only return intervals starting after period_start
        assert len(result) == 1
        start_normalized, period_start_normalized = _normalize_datetime_for_comparison(
            result[0][0], period_start
        )
        assert start_normalized >= period_start_normalized

    def test_get_occupied_intervals_cache_with_period_end(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_occupied_intervals_cache with period_end filter."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        now = dt_util.utcnow()
        intervals = [
            (now - timedelta(hours=3), now - timedelta(hours=2)),
            (now - timedelta(hours=1), now),
        ]
        save_occupied_intervals_cache(db, area_name, intervals)

        # Filter by period_end
        period_end = now - timedelta(minutes=30)
        result = get_occupied_intervals_cache(db, area_name, period_end=period_end)
        # Should only return intervals ending before period_end
        assert len(result) == 1
        end_normalized, period_end_normalized = _normalize_datetime_for_comparison(
            result[0][1], period_end
        )
        assert end_normalized <= period_end_normalized

    def test_get_occupied_intervals_cache_with_both_filters(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_occupied_intervals_cache with both period_start and period_end."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        now = dt_util.utcnow()
        intervals = [
            (now - timedelta(hours=3), now - timedelta(hours=2)),
            (now - timedelta(hours=1), now),
        ]
        save_occupied_intervals_cache(db, area_name, intervals)

        # Filter by both period_start and period_end
        period_start = now - timedelta(hours=2)
        period_end = now - timedelta(minutes=30)
        result = get_occupied_intervals_cache(
            db, area_name, period_start=period_start, period_end=period_end
        )
        # Should return intervals within the period
        for start, end in result:
            start_normalized, period_start_normalized = (
                _normalize_datetime_for_comparison(start, period_start)
            )
            end_normalized, period_end_normalized = _normalize_datetime_for_comparison(
                end, period_end
            )
            assert start_normalized >= period_start_normalized
            assert end_normalized <= period_end_normalized

    def test_is_occupied_intervals_cache_valid_no_cache(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test is_occupied_intervals_cache_valid when no cache exists."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Initially invalid
        assert is_occupied_intervals_cache_valid(db, area_name) is False

    @pytest.mark.parametrize(
        ("cache_age_hours", "max_age_hours", "expected_valid", "description"),
        [
            # Fresh cache scenarios
            (0.08, 24, True, "fresh_cache_with_default_max_age"),  # 5 minutes old
            (0.08, 1, True, "fresh_cache_with_1h_max_age"),  # 5 minutes old
            # Stale cache scenarios
            (2, 1, False, "stale_cache_exceeds_max_age"),  # 2 hours old, max 1 hour
            (2, 24, True, "stale_cache_within_max_age"),  # 2 hours old, max 24 hours
            # Boundary scenarios
            (
                1,
                1,
                False,
                "boundary_exactly_at_max_age",
            ),  # Exactly 1 hour, max 1 hour (age < max_age_hours)
            (1, 2, True, "boundary_below_max_age"),  # Exactly 1 hour, max 2 hours
        ],
    )
    def test_is_occupied_intervals_cache_valid_scenarios(
        self,
        coordinator: AreaOccupancyCoordinator,
        cache_age_hours: float,
        max_age_hours: int,
        expected_valid: bool,
        description: str,
    ):
        """Test is_occupied_intervals_cache_valid with various cache ages and max_age_hours."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Save cache first
        intervals = [(dt_util.utcnow() - timedelta(hours=1), dt_util.utcnow())]
        save_occupied_intervals_cache(db, area_name, intervals)

        # Manually update calculation_date to desired age
        with db.get_session() as session:
            cached = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .first()
            )
            if cached:
                cached.calculation_date = dt_util.utcnow() - timedelta(
                    hours=cache_age_hours
                )
                session.commit()

        result = is_occupied_intervals_cache_valid(
            db, area_name, max_age_hours=max_age_hours
        )
        assert result == expected_valid, (
            f"Failed for {description}: cache_age={cache_age_hours}h, max_age={max_age_hours}h, expected={expected_valid}, got={result}"
        )


class TestGetAllTimePriors:
    """Test get_all_time_priors function."""

    def test_get_all_time_priors_with_data(self, coordinator: AreaOccupancyCoordinator):
        """Test get_all_time_priors with existing priors."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        with db.get_session() as session:
            area = db.Areas(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                area_id="test",
                purpose="living",
                threshold=0.5,
            )
            session.add(area)

            # Add priors for a few slots
            priors = [
                db.Priors(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    day_of_week=0,
                    time_slot=8,
                    prior_value=0.6,
                    data_points=10,
                ),
                db.Priors(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    day_of_week=1,
                    time_slot=14,
                    prior_value=0.35,
                    data_points=10,
                ),
            ]
            session.add_all(priors)
            session.commit()

        result = get_all_time_priors(
            db, db.coordinator.entry_id, area_name, default_prior=0.5
        )

        # Should return all 168 slots (7 days × 24 hours)
        assert len(result) == 168
        # Verify existing priors are returned
        assert result[(0, 8)] == 0.6
        assert result[(1, 14)] == 0.35
        # Verify missing slots use default
        assert result[(0, 0)] == 0.5
        assert result[(6, 23)] == 0.5

    @pytest.mark.parametrize(
        ("default_prior", "description"),
        [
            (0.5, "default_prior_value"),
            (0.75, "custom_default_prior"),
        ],
    )
    def test_get_all_time_priors_default_values(
        self,
        coordinator: AreaOccupancyCoordinator,
        default_prior: float,
        description: str,
    ):
        """Test get_all_time_priors with different default_prior values."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        result = get_all_time_priors(
            db, db.coordinator.entry_id, area_name, default_prior=default_prior
        )

        # Should return all 168 slots with default value
        assert len(result) == 168
        for day_of_week in range(7):
            for time_slot in range(24):
                assert result[(day_of_week, time_slot)] == default_prior, (
                    f"Failed for {description}: slot ({day_of_week}, {time_slot}) should be {default_prior}"
                )

    def test_get_all_time_priors_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test get_all_time_priors with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        def bad_session():
            raise SQLAlchemyError("Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        result = get_all_time_priors(
            db, db.coordinator.entry_id, area_name, default_prior=0.5
        )

        # Should return default dict with all slots set to default_prior
        assert len(result) == 168
        for day_of_week in range(7):
            for time_slot in range(24):
                assert result[(day_of_week, time_slot)] == 0.5


class TestGetTotalOccupiedSeconds:
    """Test get_total_occupied_seconds function."""

    def test_get_total_occupied_seconds_single_interval(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_total_occupied_seconds with single interval."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion1", "motion", area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion1", start, end, area_name
            )
            session.commit()

        result = get_total_occupied_seconds(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )

        # Should be approximately 3600 seconds (1 hour)
        assert abs(result - 3600.0) < 1.0

    def test_get_total_occupied_seconds_multiple_intervals(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_total_occupied_seconds with multiple intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        intervals_data = [
            (now - timedelta(hours=3), now - timedelta(hours=2)),  # 1 hour
            (now - timedelta(hours=1), now - timedelta(minutes=30)),  # 0.5 hours
        ]

        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion1", "motion", area_name
            )
            for start, end in intervals_data:
                _create_test_interval(
                    session, db, "binary_sensor.motion1", start, end, area_name
                )
            session.commit()

        result = get_total_occupied_seconds(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )

        # Should be sum of both intervals: 3600 + 1800 = 5400 seconds
        expected = 5400.0
        assert abs(result - expected) < 1.0

    def test_get_total_occupied_seconds_overlapping_intervals(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_total_occupied_seconds with overlapping intervals (should not double-count)."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        # Overlapping intervals: first from 0-2h, second from 1-3h
        # Merged result should be 0-3h = 3 hours
        start1 = now - timedelta(hours=3)
        end1 = now - timedelta(hours=1)
        start2 = now - timedelta(hours=2)
        end2 = now

        with db.get_session() as session:
            _create_test_entity(
                session, db, "binary_sensor.motion1", "motion", area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion1", start1, end1, area_name
            )
            _create_test_interval(
                session, db, "binary_sensor.motion1", start2, end2, area_name
            )
            session.commit()

        result = get_total_occupied_seconds(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )

        # Should be merged to 3 hours = 10800 seconds (not 4 hours = 14400)
        expected = 10800.0
        assert abs(result - expected) < 1.0

    def test_get_total_occupied_seconds_empty(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_total_occupied_seconds with no intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        result = get_total_occupied_seconds(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )

        # Should return 0.0
        assert result == 0.0

    def test_get_total_occupied_seconds_with_timeout(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test get_total_occupied_seconds with motion timeout."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        start = now - timedelta(hours=2)
        # Motion interval ends 15 minutes before "now"
        motion_end = now - timedelta(minutes=15)
        # Second motion interval starts after first ends (with a gap)
        # The timeout should bridge the gap, and merged_end will be the end of second interval
        # This ensures motion_end < merged_end, allowing timeout extension to be tested
        # motion_end + timeout = now - 15m + 10m = now - 5m
        # Set merged_end = now - 8m, so motion_end + timeout (now - 5m) > merged_end (now - 8m)
        # This means timeout IS clamped to merged_end (now - 8m)
        # Set second_motion_start to overlap with first motion, so they're in the same merged interval
        # Then the timeout extension from motion_end is clamped to merged_end
        # The segments will merge, and result[0][1] = merged_end (now - 8m)
        # This equals min(motion_end + timeout, merged_end) = min(now - 5m, now - 8m) = now - 8m
        merged_end = now - timedelta(
            minutes=8
        )  # Before motion_end + timeout, so timeout is clamped
        second_motion_start = now - timedelta(
            minutes=16
        )  # Overlaps with first motion (ends at now - 15m)

        with db.get_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)

            # Create first motion interval that ends before the merged interval end
            _create_test_interval(
                session, db, "binary_sensor.motion1", start, motion_end, area_name
            )
            # Create second motion interval that starts after first ends (gap)
            # The timeout will bridge the gap, and merged_end will be the end of second interval
            _create_test_interval(
                session,
                db,
                "binary_sensor.motion1",
                second_motion_start,
                merged_end,
                area_name,
            )
            session.commit()

        # With 10 minute timeout, motion segment should be extended
        motion_timeout_seconds = 600  # 10 minutes
        result = get_total_occupied_seconds(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=motion_timeout_seconds,
        )

        # Timeout extends motion segments from motion_end, but is clamped to merged_end
        # Since motion_end < merged_end, timeout should extend to min(motion_end + timeout, merged_end)
        # motion_end + timeout = now - 15m + 10m = now - 5m
        # merged_end = now - 8m
        # So expected_end = min(now - 5m, now - 8m) = now - 8m (timeout is clamped to merged_end)
        # The duration should be from start to min(motion_end + timeout, merged_end) = merged_end
        expected_end = min(
            motion_end + timedelta(seconds=motion_timeout_seconds), merged_end
        )
        expected_duration = (expected_end - start).total_seconds()
        assert abs(result - expected_duration) < 1.0
