"""Tests for data analysis module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.analysis import PriorAnalyzer
from custom_components.area_occupancy.db.utils import (
    apply_motion_timeout,
    find_overlapping_motion_intervals,
    is_timestamp_occupied,
    merge_overlapping_intervals,
    segment_interval_with_motion,
)
from homeassistant.util import dt as dt_util


class TestPriorAnalyzerParameterValidation:
    """Test PriorAnalyzer parameter validation."""

    def test_prior_analyzer_init(self, coordinator: Mock) -> None:
        """Test PriorAnalyzer initialization."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)
        assert analyzer.coordinator == coordinator
        assert analyzer.db == coordinator.db
        assert analyzer.area_name == area_name
        assert analyzer.config == coordinator.areas[area_name].config

    def test_prior_analyzer_init_invalid_area(self, coordinator: Mock) -> None:
        """Test PriorAnalyzer initialization with invalid area."""
        with pytest.raises(ValueError, match="Area 'Invalid Area' not found"):
            PriorAnalyzer(coordinator, "Invalid Area")


class TestPriorAnalyzerWithRealDB:
    """Integration tests for PriorAnalyzer with real database."""

    def test_get_occupied_intervals_with_real_data(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_occupied_intervals with real database data."""
        db = coordinator.db
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        # Create test entity and interval
        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with db.get_session() as session:
            entity = db.Entities(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_session() as session:
            interval = db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                state="on",
                start_time=start,
                end_time=end,
                duration_seconds=3600.0,
                aggregation_level="raw",
            )
            session.add(interval)
            session.commit()

        # Get occupied intervals
        # Note: Using raw calculation path since cache might not be populated
        intervals = analyzer.get_occupied_intervals()
        assert len(intervals) > 0


class TestMergeOverlappingIntervals:
    """Test merge_overlapping_intervals function."""

    @pytest.mark.parametrize(
        ("intervals", "expected"),
        [
            ([], []),
            ([(0, 1)], [(0, 1)]),
            ([(0, 1), (2, 3), (4, 5)], [(0, 1), (2, 3), (4, 5)]),
            ([(0, 2), (1, 3)], [(0, 3)]),
            ([(0, 1), (1, 2)], [(0, 2)]),
            ([(2, 3), (0, 1), (1, 2)], [(0, 3)]),
            ([(0, 1), (0.5, 2), (1.5, 3), (2.5, 4)], [(0, 4)]),
            ([(0, 1), (0.5, 2), (3, 4), (3.5, 5)], [(0, 2), (3, 5)]),
            ([(0, 0), (0, 0), (1, 1)], [(0, 0), (1, 1)]),
            ([(0, 3), (1, 2)], [(0, 3)]),
            ([(0, 1), (0, 1), (0, 1)], [(0, 1)]),
            ([(0, 1), (100, 101), (200, 201)], [(0, 1), (100, 101), (200, 201)]),
            ([(0, 0.000001), (0.000001, 0.000002)], [(0, 0.000002)]),
        ],
    )
    def test_merge_overlapping_intervals(
        self, freeze_time: datetime, intervals: list, expected: list
    ) -> None:
        """Test merge_overlapping_intervals with various scenarios."""
        now = freeze_time
        # Convert relative intervals to absolute timestamps
        input_intervals = [
            (now + timedelta(hours=start), now + timedelta(hours=end))
            for start, end in intervals
        ]
        expected_intervals = [
            (now + timedelta(hours=start), now + timedelta(hours=end))
            for start, end in expected
        ]
        result = merge_overlapping_intervals(input_intervals)
        assert result == expected_intervals


class TestFindOverlappingMotionIntervals:
    """Test find_overlapping_motion_intervals function."""

    def test_no_overlapping_intervals(self, freeze_time: datetime) -> None:
        """Test find_overlapping_motion_intervals with no overlapping intervals."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=1))
        motion_intervals = [
            (now + timedelta(hours=2), now + timedelta(hours=3)),
            (now + timedelta(hours=4), now + timedelta(hours=5)),
        ]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        assert result == []

    def test_single_overlapping_interval(self, freeze_time: datetime) -> None:
        """Test find_overlapping_motion_intervals with single overlapping interval."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=3)),
            (now + timedelta(hours=4), now + timedelta(hours=5)),
        ]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        assert len(result) == 1
        assert result[0] == (now + timedelta(hours=1), now + timedelta(hours=3))

    def test_multiple_overlapping_intervals(self, freeze_time: datetime) -> None:
        """Test find_overlapping_motion_intervals with multiple overlapping intervals."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=3))
        motion_intervals = [
            (now - timedelta(hours=1), now + timedelta(hours=1)),  # Starts before
            (now + timedelta(hours=1), now + timedelta(hours=2)),  # Within
            (now + timedelta(hours=2), now + timedelta(hours=4)),  # Extends past
            (now + timedelta(hours=5), now + timedelta(hours=6)),  # After
        ]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        assert len(result) == 3
        assert (now - timedelta(hours=1), now + timedelta(hours=1)) in result
        assert (now + timedelta(hours=1), now + timedelta(hours=2)) in result
        assert (now + timedelta(hours=2), now + timedelta(hours=4)) in result

    def test_partially_overlapping_intervals(self, freeze_time: datetime) -> None:
        """Test find_overlapping_motion_intervals with partially overlapping intervals."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now + timedelta(minutes=30), now + timedelta(hours=1, minutes=30)),
            (
                now + timedelta(hours=1, minutes=45),
                now + timedelta(hours=2, minutes=30),
            ),
        ]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        assert len(result) == 2

    def test_intervals_at_boundaries(self, freeze_time: datetime) -> None:
        """Test find_overlapping_motion_intervals with intervals at boundaries."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now, now + timedelta(hours=1)),  # Starts at merged start
            (now + timedelta(hours=1), now + timedelta(hours=2)),  # Ends at merged end
            (
                now + timedelta(hours=2),
                now + timedelta(hours=3),
            ),  # Starts at merged end (touching)
        ]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        # Current implementation treats touching intervals as overlapping
        assert len(result) == 3
        assert (now, now + timedelta(hours=1)) in result
        assert (now + timedelta(hours=1), now + timedelta(hours=2)) in result
        assert (now + timedelta(hours=2), now + timedelta(hours=3)) in result

    def test_interval_completely_within_merged(self, freeze_time: datetime) -> None:
        """Test find_overlapping_motion_intervals with interval completely within merged."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=3))
        motion_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=2)),
        ]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        assert len(result) == 1
        assert result[0] == (now + timedelta(hours=1), now + timedelta(hours=2))

    def test_empty_motion_intervals(self, freeze_time: datetime) -> None:
        """Test find_overlapping_motion_intervals with empty motion_intervals."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=1))
        result = find_overlapping_motion_intervals(merged_interval, [])
        assert result == []

    def test_motion_interval_exactly_same_as_merged(
        self, freeze_time: datetime
    ) -> None:
        """Test find_overlapping_motion_intervals with motion interval exactly matching merged."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [(now, now + timedelta(hours=2))]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        assert len(result) == 1
        assert result[0] == (now, now + timedelta(hours=2))

    def test_motion_interval_starts_before_ends_after(
        self, freeze_time: datetime
    ) -> None:
        """Test find_overlapping_motion_intervals with motion extending beyond merged."""
        now = freeze_time
        merged_interval = (now + timedelta(hours=1), now + timedelta(hours=2))
        motion_intervals = [(now, now + timedelta(hours=3))]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        assert len(result) == 1
        assert result[0] == (now, now + timedelta(hours=3))

    def test_motion_intervals_touching_at_start(self, freeze_time: datetime) -> None:
        """Test find_overlapping_motion_intervals with motion touching merged start."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [(now - timedelta(hours=1), now)]
        result = find_overlapping_motion_intervals(merged_interval, motion_intervals)
        # Current implementation treats touching as overlapping
        assert len(result) == 1


class TestSegmentIntervalWithMotion:
    """Test segment_interval_with_motion function."""

    def test_no_motion_overlap(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with no motion overlap."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now + timedelta(hours=3), now + timedelta(hours=4)),
        ]
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds=300
        )
        assert result == [merged_interval]

    def test_motion_covers_entire_interval(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with motion covering entire interval."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now, now + timedelta(hours=2)),
        ]
        timeout_seconds = 600
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        # Should be single segment with timeout applied (or merged after)
        assert len(result) >= 1
        assert result[0][0] == now
        assert result[0][1] >= now + timedelta(hours=2)

    def test_motion_at_start(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with motion at start."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now, now + timedelta(hours=1)),
        ]
        timeout_seconds = 300
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        # Should have: motion segment (with timeout) + after segment
        assert len(result) == 2
        assert result[0] == (now, now + timedelta(hours=1, seconds=timeout_seconds))
        assert result[1] == (
            now + timedelta(hours=1, seconds=timeout_seconds),
            now + timedelta(hours=2),
        )

    def test_motion_at_end(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with motion at end."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=2)),
        ]
        timeout_seconds = 300
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        # Should have: before segment + motion segment (with timeout)
        assert len(result) >= 2
        assert result[0][0] == now
        assert result[0][1] == now + timedelta(hours=1)
        assert result[1][0] == now + timedelta(hours=1)
        assert result[1][1] >= now + timedelta(hours=2)

    def test_motion_in_middle(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with motion in middle."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=3))
        motion_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=2)),
        ]
        timeout_seconds = 300
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        # Should have: before + motion (with timeout) + after
        assert len(result) == 3
        assert result[0] == (now, now + timedelta(hours=1))
        assert result[1] == (
            now + timedelta(hours=1),
            now + timedelta(hours=2, seconds=timeout_seconds),
        )
        assert result[2] == (
            now + timedelta(hours=2, seconds=timeout_seconds),
            now + timedelta(hours=3),
        )

    def test_multiple_motion_intervals(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with multiple motion intervals."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=3))
        motion_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=1, minutes=30)),
            (now + timedelta(hours=1, minutes=45), now + timedelta(hours=2)),
        ]
        timeout_seconds = 300
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        # Should have: before + motion1 (with timeout) + gap + motion2 (with timeout) + after
        assert len(result) == 5
        assert result[0] == (now, now + timedelta(hours=1))  # Before
        assert result[1] == (
            now + timedelta(hours=1),
            now + timedelta(hours=1, minutes=30, seconds=timeout_seconds),
        )  # Motion1 with timeout
        assert result[2] == (
            now + timedelta(hours=1, minutes=30, seconds=timeout_seconds),
            now + timedelta(hours=1, minutes=45),
        )  # Gap between motions
        assert result[3] == (
            now + timedelta(hours=1, minutes=45),
            now + timedelta(hours=2, seconds=timeout_seconds),
        )  # Motion2 with timeout
        assert result[4] == (
            now + timedelta(hours=2, seconds=timeout_seconds),
            now + timedelta(hours=3),
        )  # After

    def test_timeout_application_only_to_motion(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion applies timeout only to motion segment."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=1, minutes=30)),
        ]
        timeout_seconds = 600
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        # Before segment should have no timeout
        assert result[0] == (now, now + timedelta(hours=1))
        # Motion segment should have timeout
        assert result[1] == (
            now + timedelta(hours=1),
            now + timedelta(hours=1, minutes=30, seconds=timeout_seconds),
        )
        # After segment should have no timeout applied to it
        assert result[2] == (
            now + timedelta(hours=1, minutes=30, seconds=timeout_seconds),
            now + timedelta(hours=2),
        )

    def test_motion_exactly_at_merged_start(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with motion starting exactly at merged start."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [(now, now + timedelta(hours=1))]
        timeout_seconds = 300
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        assert len(result) == 2
        assert result[0][0] == now
        assert result[1][0] == now + timedelta(hours=1, seconds=timeout_seconds)

    def test_motion_exactly_at_merged_end(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with motion ending exactly at merged end."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [(now + timedelta(hours=1), now + timedelta(hours=2))]
        timeout_seconds = 300
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        assert len(result) == 2
        assert result[0][1] == now + timedelta(hours=1)
        assert result[1][0] == now + timedelta(hours=1)

    def test_zero_timeout(self, freeze_time: datetime) -> None:
        """Test segment_interval_with_motion with zero timeout."""
        now = freeze_time
        merged_interval = (now, now + timedelta(hours=2))
        motion_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=1, minutes=30))
        ]
        timeout_seconds = 0
        result = segment_interval_with_motion(
            merged_interval, motion_intervals, timeout_seconds
        )
        # Should still segment correctly, just no timeout extension
        assert len(result) == 3
        assert result[1][1] == now + timedelta(hours=1, minutes=30)


class TestApplyMotionTimeout:
    """Test apply_motion_timeout function."""

    def test_empty_merged_intervals(self, freeze_time: datetime) -> None:
        """Test apply_motion_timeout with empty merged intervals."""
        result = apply_motion_timeout([], [], timeout_seconds=300)
        assert result == []

    def test_single_merged_interval_with_no_motion(self, freeze_time: datetime) -> None:
        """Test apply_motion_timeout with single merged interval and no motion."""
        now = freeze_time
        merged_intervals = [(now, now + timedelta(hours=1))]
        motion_intervals = []
        result = apply_motion_timeout(
            merged_intervals, motion_intervals, timeout_seconds=300
        )
        # Should return interval as-is
        assert result == merged_intervals

    def test_multiple_merged_intervals(self, freeze_time: datetime) -> None:
        """Test apply_motion_timeout with multiple merged intervals."""
        now = freeze_time
        merged_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=2), now + timedelta(hours=3)),
        ]
        motion_intervals = [
            (now + timedelta(minutes=30), now + timedelta(minutes=45)),
        ]
        timeout_seconds = 600
        result = apply_motion_timeout(
            merged_intervals, motion_intervals, timeout_seconds
        )
        # First interval should be segmented, second should be unchanged
        assert len(result) >= 2

    def test_complex_merging_after_timeout(self, freeze_time: datetime) -> None:
        """Test apply_motion_timeout with complex merging after timeout extension."""
        now = freeze_time
        merged_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=1, minutes=30), now + timedelta(hours=2)),
        ]
        motion_intervals = [
            (now + timedelta(minutes=50), now + timedelta(hours=1, minutes=10)),
        ]
        timeout_seconds = 1800  # 30 minutes
        result = apply_motion_timeout(
            merged_intervals, motion_intervals, timeout_seconds
        )
        # Timeout extension might cause intervals to merge
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_timeout_creates_overlap(self, freeze_time: datetime) -> None:
        """Test apply_motion_timeout where timeout creates overlap that merges."""
        now = freeze_time
        merged_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=1, minutes=5), now + timedelta(hours=2)),
        ]
        motion_intervals = [
            (now + timedelta(minutes=55), now + timedelta(hours=1, minutes=5)),
        ]
        timeout_seconds = 600  # 10 minutes - should extend past second interval start
        result = apply_motion_timeout(
            merged_intervals, motion_intervals, timeout_seconds
        )
        # Should merge because timeout extension overlaps with second interval
        assert len(result) >= 1
        assert result[0][0] == now

    def test_zero_timeout(self, freeze_time: datetime) -> None:
        """Test apply_motion_timeout with zero timeout."""
        now = freeze_time
        merged_intervals = [(now, now + timedelta(hours=1))]
        motion_intervals = [(now + timedelta(minutes=30), now + timedelta(minutes=45))]
        result = apply_motion_timeout(
            merged_intervals, motion_intervals, timeout_seconds=0
        )
        # Should still segment but without extension
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_very_large_timeout(self, freeze_time: datetime) -> None:
        """Test apply_motion_timeout with very large timeout causing massive overlap."""
        now = freeze_time
        merged_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=2), now + timedelta(hours=3)),
        ]
        motion_intervals = [
            (now + timedelta(minutes=55), now + timedelta(hours=1, minutes=5)),
        ]
        timeout_seconds = 86400  # 24 hours - should merge both intervals
        result = apply_motion_timeout(
            merged_intervals, motion_intervals, timeout_seconds
        )
        # Should merge due to large timeout extension
        assert len(result) >= 1

    def test_multiple_motion_intervals_per_merged(self, freeze_time: datetime) -> None:
        """Test apply_motion_timeout with multiple motion intervals per merged interval."""
        now = freeze_time
        merged_intervals = [(now, now + timedelta(hours=3))]
        motion_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=1, minutes=30)),
            (now + timedelta(hours=2), now + timedelta(hours=2, minutes=30)),
        ]
        timeout_seconds = 600
        result = apply_motion_timeout(
            merged_intervals, motion_intervals, timeout_seconds
        )
        # Should segment and potentially merge if timeouts overlap
        assert isinstance(result, list)
        assert len(result) >= 1


class TestIsTimestampOccupied:
    """Test is_timestamp_occupied function."""

    def test_empty_occupied_intervals(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with empty occupied_intervals."""
        now = freeze_time
        result = is_timestamp_occupied(now, [])
        assert result is False

    def test_timestamp_before_all_intervals(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp before all intervals."""
        now = freeze_time
        occupied_intervals = [
            (now + timedelta(hours=1), now + timedelta(hours=2)),
            (now + timedelta(hours=3), now + timedelta(hours=4)),
        ]
        result = is_timestamp_occupied(now, occupied_intervals)
        assert result is False

    def test_timestamp_after_all_intervals(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp after all intervals."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=2), now + timedelta(hours=3)),
        ]
        result = is_timestamp_occupied(now + timedelta(hours=4), occupied_intervals)
        assert result is False

    def test_timestamp_within_interval_middle(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp in middle of interval."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=2), now + timedelta(hours=3)),
        ]
        result = is_timestamp_occupied(now + timedelta(minutes=30), occupied_intervals)
        assert result is True

    def test_timestamp_at_interval_start(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp exactly at interval start."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
        ]
        result = is_timestamp_occupied(now, occupied_intervals)
        # start <= timestamp < end, so timestamp == start should be True
        assert result is True

    def test_timestamp_just_before_start(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp just before interval start."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
        ]
        result = is_timestamp_occupied(
            now - timedelta(microseconds=1), occupied_intervals
        )
        assert result is False

    def test_timestamp_just_after_end(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp just after interval end."""
        now = freeze_time
        end_time = now + timedelta(hours=1)
        occupied_intervals = [
            (now, end_time),
        ]
        result = is_timestamp_occupied(
            end_time + timedelta(microseconds=1), occupied_intervals
        )
        assert result is False

    def test_single_point_interval(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with single point interval."""
        now = freeze_time
        occupied_intervals = [
            (now, now),  # Zero duration interval
        ]
        # start <= timestamp < end: now <= now < now is False (can't be < self)
        result = is_timestamp_occupied(now, occupied_intervals)
        assert result is False

    def test_timestamp_in_gap_between_intervals(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp in gap between intervals."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=2), now + timedelta(hours=3)),
        ]
        # Timestamp in the gap
        result = is_timestamp_occupied(
            now + timedelta(hours=1, minutes=30), occupied_intervals
        )
        assert result is False

    def test_multiple_identical_intervals(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with multiple identical intervals."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
            (now, now + timedelta(hours=1)),
            (now, now + timedelta(hours=1)),
        ]
        result = is_timestamp_occupied(now + timedelta(minutes=30), occupied_intervals)
        assert result is True

    def test_very_large_interval(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with very large interval."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(days=365)),
        ]
        result = is_timestamp_occupied(now + timedelta(days=100), occupied_intervals)
        assert result is True

    def test_timestamp_far_future(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp far in future."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
        ]
        result = is_timestamp_occupied(now + timedelta(days=1000), occupied_intervals)
        assert result is False

    def test_timestamp_far_past(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp far in past."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
        ]
        result = is_timestamp_occupied(now - timedelta(days=1000), occupied_intervals)
        assert result is False
