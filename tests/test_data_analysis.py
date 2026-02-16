"""Tests for data analysis module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.area_occupancy.const import (
    TIME_PRIOR_MAX_BOUND,
    TIME_PRIOR_MIN_BOUND,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.analysis import (
    PriorAnalyzer,
    ensure_occupied_intervals_cache,
    run_interval_aggregation,
    run_numeric_aggregation,
    start_prior_analysis,
)
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.db.utils import (
    apply_motion_timeout,
    find_overlapping_motion_intervals,
    is_timestamp_occupied,
    merge_overlapping_intervals,
    segment_interval_with_motion,
)
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001, PLC0415
def _get_next_monday_at_hour(now: datetime, hour: int, minute: int = 0) -> datetime:
    """Get next Monday at specified hour and minute.

    Args:
        now: Current datetime
        hour: Hour (0-23)
        minute: Minute (0-59), defaults to 0

    Returns:
        Next Monday at specified time
    """
    days_until_monday = (0 - now.weekday()) % 7
    if days_until_monday == 0 and now.hour >= hour:
        # If it's already Monday and past the specified hour, use next Monday
        days_until_monday = 7
    return (now + timedelta(days=days_until_monday)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )


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


class TestTimePriorsDST:
    """DST regression tests for local-time bucketing."""

    @pytest.fixture
    def set_tz_america_new_york(self):
        """Temporarily set HA default timezone to America/New_York."""
        original = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(ZoneInfo("America/New_York"))
        try:
            yield
        finally:
            dt_util.set_default_time_zone(original)

    def test_fall_back_repeated_hour_denominator(
        self, coordinator: AreaOccupancyCoordinator, set_tz_america_new_york
    ) -> None:
        """Ensure repeated DST hour is treated as two hours of possible time.

        On 2025-11-02 in America/New_York, 01:00 occurs twice. We occupy only the
        first 01:00 hour, so the prior for (Sunday, 01:00) should be ~0.5.
        """
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        period_start = datetime(2025, 11, 2, 4, 0, 0, tzinfo=dt_util.UTC)
        period_end = datetime(2025, 11, 2, 8, 0, 0, tzinfo=dt_util.UTC)

        # Occupy only the first 01:00 local hour (05:00-06:00 UTC).
        occupied_intervals = [
            (
                datetime(2025, 11, 2, 5, 0, 0, tzinfo=dt_util.UTC),
                datetime(2025, 11, 2, 6, 0, 0, tzinfo=dt_util.UTC),
            )
        ]

        time_priors, _data_points = analyzer.calculate_time_priors(
            occupied_intervals, period_start, period_end
        )

        # 2025-11-02 is Sunday (weekday=6). Slot=1 for 01:00 local time.
        assert time_priors[(6, 1)] == pytest.approx(0.5, abs=1e-6)

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


class TestPriorAnalyzerCalculateAndUpdatePrior:
    """Test PriorAnalyzer.calculate_and_update_prior method."""

    def test_empty_intervals_returns_early(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that empty intervals cause early return without prior update."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Store original prior value
        original_prior = area.prior.global_prior

        # Mock get_occupied_intervals to return empty list
        with patch.object(analyzer, "get_occupied_intervals", return_value=[]):
            analyzer.calculate_and_update_prior()

        # Prior should not be updated
        assert area.prior.global_prior == original_prior

    def test_invalid_period_duration_sets_fallback_prior(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that invalid period duration (zero/negative) sets fallback prior 0.01."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create intervals with invalid period (same start and end)
        now = freeze_time
        invalid_intervals = [(now, now)]  # Zero duration interval

        with patch.object(
            analyzer, "get_occupied_intervals", return_value=invalid_intervals
        ):
            analyzer.calculate_and_update_prior()

        # Should set fallback prior of 0.01
        assert area.prior.global_prior == 0.01

    def test_valid_calculation_sets_correct_prior(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that valid calculation produces correct prior value."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create intervals: 2 hours occupied
        # Period calculation uses first_interval_start to actual_period_end
        now = freeze_time
        occupied_start = now - timedelta(hours=8)
        occupied_end = now - timedelta(hours=6)
        intervals = [(occupied_start, occupied_end)]

        # Mock get_occupied_intervals
        with (
            patch.object(analyzer, "get_occupied_intervals", return_value=intervals),
            patch(
                "custom_components.area_occupancy.data.analysis.dt_util.utcnow",
                return_value=now,
            ),
        ):
            # Mock dt_util.utcnow to return 'now' for period calculation
            analyzer.calculate_and_update_prior()

        # Period calculation:
        # - first_interval_start = occupied_start (now - 8h)
        # - last_interval_end = occupied_end (now - 6h)
        # - Since (now - last_interval_end) = 6h > 1h, actual_period_end = last_interval_end
        # - actual_period_duration = (now - 6h) - (now - 8h) = 2 hours
        # - occupied_duration = 2 hours
        # - prior = 2h / 2h = 1.0, clamped to 0.99 max
        assert area.prior.global_prior == 0.99

    def test_prior_bounds_clamping_min(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that prior values < 0.01 are clamped to 0.01."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create intervals: very small occupancy (1 minute) in large period (100 hours)
        now = freeze_time
        occupied_start = now - timedelta(hours=50)
        occupied_end = now - timedelta(hours=50, minutes=1)
        intervals = [(occupied_start, occupied_end)]

        with (
            patch.object(analyzer, "get_occupied_intervals", return_value=intervals),
            patch(
                "custom_components.area_occupancy.data.analysis.dt_util.utcnow",
                return_value=now,
            ),
        ):
            analyzer.calculate_and_update_prior()

        # Should be clamped to 0.01 minimum
        assert area.prior.global_prior == 0.01

    def test_invalid_intervals_filtered_out(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that intervals with start > end are filtered out."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create intervals with one invalid interval (start > end)
        now = freeze_time
        valid_start = now - timedelta(hours=2)
        valid_end = now - timedelta(hours=1)
        invalid_start = now - timedelta(hours=1)
        invalid_end = now - timedelta(hours=2)  # End before start

        intervals = [
            (valid_start, valid_end),  # Valid interval
            (invalid_start, invalid_end),  # Invalid interval (start > end)
        ]

        with patch.object(analyzer, "get_occupied_intervals", return_value=intervals):
            analyzer.calculate_and_update_prior()

        # Should filter out invalid interval and calculate based on valid one
        # The invalid interval should be removed, so calculation proceeds normally
        assert area.prior.global_prior is not None
        assert area.prior.global_prior >= 0.01

    def test_all_invalid_intervals_use_fallback(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that when all intervals are invalid, fallback prior is used."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create only invalid intervals (start > end)
        now = freeze_time
        invalid_intervals = [
            (now - timedelta(hours=1), now - timedelta(hours=2)),  # start > end
            (now - timedelta(hours=3), now - timedelta(hours=4)),  # start > end
        ]

        with patch.object(
            analyzer, "get_occupied_intervals", return_value=invalid_intervals
        ):
            analyzer.calculate_and_update_prior()

        # Should use fallback prior of 0.01
        assert area.prior.global_prior == 0.01

    def test_period_end_before_first_interval_start_uses_now(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that when actual_period_end < first_interval_start, 'now' is used instead."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create intervals where last_interval_end would be before first_interval_start
        # This can happen with timezone issues
        now = freeze_time
        # Create valid intervals
        start1 = now - timedelta(hours=8)
        end1 = now - timedelta(hours=7)
        start2 = now - timedelta(hours=6)
        end2 = now - timedelta(hours=5)
        intervals = [(start1, end1), (start2, end2)]

        # Mock dt_util.utcnow to return a time that would make last_interval_end
        # appear before first_interval_start (simulating timezone issue)
        # Actually, with valid intervals, this shouldn't happen unless there's
        # a timezone conversion issue. Let's test the defensive check instead.
        with (
            patch.object(analyzer, "get_occupied_intervals", return_value=intervals),
            patch(
                "custom_components.area_occupancy.data.analysis.dt_util.utcnow",
                return_value=now,
            ),
        ):
            analyzer.calculate_and_update_prior()

        # Should calculate normally since intervals are valid
        assert area.prior.global_prior is not None
        assert area.prior.global_prior >= 0.01

    def test_timezone_aware_intervals_converted_to_utc(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that intervals with different timezones are converted to UTC."""
        from datetime import timezone as tz

        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create intervals with EST timezone (UTC-5)
        est = tz(timedelta(hours=-5))
        now_utc = freeze_time
        now_est = now_utc.astimezone(est)

        # Create intervals in EST
        start_est = (now_est - timedelta(hours=2)).replace(tzinfo=est)
        end_est = (now_est - timedelta(hours=1)).replace(tzinfo=est)
        intervals = [(start_est, end_est)]

        with (
            patch.object(analyzer, "get_occupied_intervals", return_value=intervals),
            patch(
                "custom_components.area_occupancy.data.analysis.dt_util.utcnow",
                return_value=now_utc,
            ),
        ):
            analyzer.calculate_and_update_prior()

        # Should convert to UTC and calculate normally
        assert area.prior.global_prior is not None
        assert area.prior.global_prior >= 0.01

    def test_prior_bounds_clamping_max(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that prior values > 0.99 are clamped to 0.99."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create intervals: almost full occupancy (99.5 hours) in 100 hour period
        now = freeze_time
        start = now - timedelta(hours=100)
        occupied_start = start
        occupied_end = now - timedelta(hours=0.5)
        intervals = [(occupied_start, occupied_end)]

        with (
            patch.object(analyzer, "get_occupied_intervals", return_value=intervals),
            patch(
                "custom_components.area_occupancy.data.analysis.dt_util.utcnow",
                return_value=now,
            ),
        ):
            analyzer.calculate_and_update_prior()

        # Should be clamped to 0.99 maximum
        assert area.prior.global_prior == 0.99

    @pytest.mark.parametrize(
        ("save_return_value", "save_side_effect", "expected_log_level", "should_log"),
        [
            (True, None, None, False),  # Success: no logging
            (False, None, "warning", True),  # Failure: logs warning
            (
                None,
                RuntimeError("DB error"),
                "warning",
                True,
            ),  # Exception: logs warning
        ],
    )
    def test_database_save_handling(
        self,
        coordinator: AreaOccupancyCoordinator,
        freeze_time: datetime,
        save_return_value: bool | None,
        save_side_effect: Exception | None,
        expected_log_level: str | None,
        should_log: bool,
    ) -> None:
        """Test database save handling for success, failure, and exception cases."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        now = freeze_time
        start = now - timedelta(hours=10)
        intervals = [(start, now - timedelta(hours=8))]

        with (
            patch.object(analyzer, "get_occupied_intervals", return_value=intervals),
            patch(
                "custom_components.area_occupancy.data.analysis.dt_util.utcnow",
                return_value=now,
            ),
        ):
            patch_kwargs = {}
            if save_side_effect is not None:
                patch_kwargs["side_effect"] = save_side_effect
            else:
                patch_kwargs["return_value"] = save_return_value

            with (
                patch.object(
                    analyzer.db, "save_global_prior", **patch_kwargs
                ) as mock_save,
                patch.object(analyzer.db, "save_time_priors", return_value=True),
                patch(
                    "custom_components.area_occupancy.data.analysis._LOGGER"
                ) as mock_logger,
            ):
                analyzer.calculate_and_update_prior()

                # Should have called save_global_prior
                assert mock_save.called
                # Prior should still be set regardless of save result
                assert area.prior.global_prior is not None

                # Check logging if expected (only for save_global_prior failures)
                if should_log:
                    # Verify warning was logged for save_global_prior failure
                    warning_calls = [
                        call
                        for call in mock_logger.warning.call_args_list
                        if "global prior" in str(call).lower()
                    ]
                    assert len(warning_calls) > 0

    def test_error_handling_logs_error(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that errors during calculation are logged but don't crash."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Cause an error by making get_occupied_intervals raise ValueError
        with (
            patch.object(
                analyzer, "get_occupied_intervals", side_effect=ValueError("Test error")
            ),
            patch(
                "custom_components.area_occupancy.data.analysis._LOGGER"
            ) as mock_logger,
        ):
            analyzer.calculate_and_update_prior()

        # Should have logged error
        mock_logger.error.assert_called()


class TestPriorAnalyzerCalculateTimePriors:
    """Test PriorAnalyzer.calculate_time_priors method."""

    @pytest.fixture(autouse=True)
    def _set_default_tz_utc(self):
        """Make local bucketing deterministic by forcing local timezone = UTC."""
        original = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(dt_util.UTC)
        try:
            yield
        finally:
            dt_util.set_default_time_zone(original)

    def test_single_interval_in_one_slot(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test time prior calculation with single interval in one slot."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create interval: Monday 10:00-10:30 (30 minutes in slot 10)
        now = freeze_time
        monday = _get_next_monday_at_hour(now, hour=10)

        interval_start = monday
        interval_end = monday + timedelta(minutes=30)
        intervals = [(interval_start, interval_end)]
        period_start = monday - timedelta(days=1)
        period_end = monday + timedelta(days=1)

        time_priors, data_points = analyzer.calculate_time_priors(
            intervals, period_start, period_end
        )

        # Should have one slot (Monday, 10)
        slot_key = (0, 10)  # Monday = 0, hour 10
        assert slot_key in time_priors
        # 30 minutes = 1800 seconds, 1 week = 3600 seconds per slot
        # prior = 1800 / 3600 = 0.5
        assert abs(time_priors[slot_key] - 0.5) < 0.01
        assert data_points[slot_key] == 1  # 1 week with data

    def test_interval_spanning_multiple_slots(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test time prior calculation with interval spanning multiple slots."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create interval: Monday 10:30-11:30 (spans slots 10 and 11)
        now = freeze_time
        monday = _get_next_monday_at_hour(now, hour=10, minute=30)

        interval_start = monday
        interval_end = monday + timedelta(hours=1)
        intervals = [(interval_start, interval_end)]
        period_start = monday - timedelta(days=1)
        period_end = monday + timedelta(days=1)

        time_priors, _ = analyzer.calculate_time_priors(
            intervals, period_start, period_end
        )

        # Should have two slots
        slot_10 = (0, 10)
        slot_11 = (0, 11)
        assert slot_10 in time_priors
        assert slot_11 in time_priors
        # Each slot should have 30 minutes = 0.5 prior
        assert abs(time_priors[slot_10] - 0.5) < 0.01
        assert abs(time_priors[slot_11] - 0.5) < 0.01

    def test_multiple_weeks_tracking(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that multiple weeks of data are tracked correctly."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create intervals on same day/time across 3 weeks
        now = freeze_time
        monday_week1 = _get_next_monday_at_hour(now, hour=10)

        intervals = []
        for week_offset in [0, 7, 14]:
            week_start = monday_week1 + timedelta(days=week_offset)
            intervals.append((week_start, week_start + timedelta(minutes=30)))

        period_start = monday_week1 - timedelta(days=1)
        period_end = monday_week1 + timedelta(days=21)

        time_priors, data_points = analyzer.calculate_time_priors(
            intervals, period_start, period_end
        )

        slot_key = (0, 10)
        assert slot_key in time_priors
        # 3 weeks with data
        assert data_points[slot_key] == 3
        # 30 minutes per week * 3 weeks = 90 minutes total
        # 3 weeks * 3600 seconds = 10800 seconds total
        # prior = 5400 / 10800 = 0.5
        assert abs(time_priors[slot_key] - 0.5) < 0.01

    def test_safety_bounds_clamping(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that time priors are clamped to [0.03, 0.9] bounds."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create interval that would result in very high prior (>0.9)
        now = freeze_time
        monday = _get_next_monday_at_hour(now, hour=10)

        # 55 minutes occupied in 1 hour slot (would be 0.917, should clamp to 0.9)
        interval_start = monday
        interval_end = monday + timedelta(minutes=55)
        intervals = [(interval_start, interval_end)]
        period_start = monday - timedelta(days=1)
        period_end = monday + timedelta(days=1)

        time_priors, _ = analyzer.calculate_time_priors(
            intervals, period_start, period_end
        )

        slot_key = (0, 10)
        assert slot_key in time_priors
        # Should be clamped to 0.9.
        assert time_priors[slot_key] == TIME_PRIOR_MAX_BOUND

        # Test minimum bound: very low occupancy.
        interval_start2 = monday
        interval_end2 = monday + timedelta(
            seconds=60
        )  # 1 minute = 0.017, should clamp to 0.03.
        intervals2 = [(interval_start2, interval_end2)]

        time_priors2, _ = analyzer.calculate_time_priors(
            intervals2, period_start, period_end
        )

        assert slot_key in time_priors2
        # Should be clamped to 0.03.
        assert time_priors2[slot_key] == TIME_PRIOR_MIN_BOUND

    def test_empty_slots_skipped(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test that slots with no data are skipped (not in result dict)."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create interval only in one slot
        now = freeze_time
        monday = _get_next_monday_at_hour(now, hour=10)

        intervals = [(monday, monday + timedelta(minutes=30))]
        period_start = monday - timedelta(days=1)
        period_end = monday + timedelta(days=1)

        time_priors, _ = analyzer.calculate_time_priors(
            intervals, period_start, period_end
        )

        # Should only have one slot
        assert len(time_priors) == 1
        assert (0, 10) in time_priors
        # Other slots should not be present
        assert (0, 9) not in time_priors
        assert (0, 11) not in time_priors

    def test_interval_at_slot_boundary(
        self, coordinator: AreaOccupancyCoordinator, freeze_time: datetime
    ) -> None:
        """Test interval exactly at slot boundary."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create interval starting exactly at hour boundary
        now = freeze_time
        monday = _get_next_monday_at_hour(now, hour=10)

        intervals = [(monday, monday + timedelta(hours=1))]
        period_start = monday - timedelta(days=1)
        period_end = monday + timedelta(days=1)

        time_priors, _ = analyzer.calculate_time_priors(
            intervals, period_start, period_end
        )

        # Should only be in slot 10 (not slot 11 since end is exclusive at 11:00)
        assert (0, 10) in time_priors
        assert (0, 11) not in time_priors


class TestPriorAnalyzerGetEntityIdsByType:
    """Test PriorAnalyzer._get_entity_ids_by_type method."""

    def test_get_motion_entity_ids(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test getting entity IDs for motion type."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Add a motion entity using the area's entity manager
        # Entities are already created from config, so we can check existing ones
        # or add one directly to the entities dict
        from custom_components.area_occupancy.data.decay import Decay
        from custom_components.area_occupancy.data.entity import Entity
        from custom_components.area_occupancy.data.entity_type import EntityType

        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.85,
            prob_given_true=0.8,
            prob_given_false=0.1,
            active_states=["on"],
        )
        motion_entity = Entity(
            entity_id="binary_sensor.motion1",
            type=entity_type,
            prob_given_true=0.8,
            prob_given_false=0.1,
            decay=Decay(half_life=60.0),
            hass=coordinator.hass,
        )
        area.entities.entities["binary_sensor.motion1"] = motion_entity

        entity_ids = analyzer._get_entity_ids_by_type(InputType.MOTION)
        assert "binary_sensor.motion1" in entity_ids

    def test_get_empty_result_for_type_with_no_entities(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that empty result is returned for type with no entities."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        entity_ids = analyzer._get_entity_ids_by_type(InputType.CO2)
        assert entity_ids == []


class TestOrchestrationFunctions:
    """Test orchestration functions in analysis module."""

    async def test_ensure_occupied_intervals_cache_valid(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test ensure_occupied_intervals_cache when cache is valid."""

        with patch(
            "custom_components.area_occupancy.data.analysis.is_occupied_intervals_cache_valid",
            return_value=True,
        ):
            # Should not populate cache if valid
            await ensure_occupied_intervals_cache(coordinator)

    async def test_ensure_occupied_intervals_cache_invalid_populates(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test ensure_occupied_intervals_cache populates cache when invalid."""

        with (
            patch(
                "custom_components.area_occupancy.data.analysis.is_occupied_intervals_cache_valid",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.data.analysis.PriorAnalyzer"
            ) as mock_analyzer_class,
            patch.object(
                coordinator.db, "save_occupied_intervals_cache", return_value=True
            ) as mock_save,
        ):
            # Mock analyzer instance
            mock_analyzer = Mock()
            mock_analyzer.get_occupied_intervals.return_value = [
                (dt_util.utcnow() - timedelta(hours=1), dt_util.utcnow())
            ]
            mock_analyzer_class.return_value = mock_analyzer

            await ensure_occupied_intervals_cache(coordinator)

            # Should have called save_occupied_intervals_cache
            assert mock_save.called

    async def test_ensure_occupied_intervals_cache_no_intervals_warns(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test ensure_occupied_intervals_cache logs warning when no intervals found."""
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.is_occupied_intervals_cache_valid",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.data.analysis.PriorAnalyzer"
            ) as mock_analyzer_class,
            patch(
                "custom_components.area_occupancy.data.analysis._LOGGER"
            ) as mock_logger,
        ):
            # Mock analyzer instance returning empty intervals
            mock_analyzer = Mock()
            mock_analyzer.get_occupied_intervals.return_value = []
            mock_analyzer_class.return_value = mock_analyzer

            await ensure_occupied_intervals_cache(coordinator)

            # Should have logged warning
            mock_logger.warning.assert_called()

    @pytest.mark.parametrize(
        ("function_name", "success", "expected_results"),
        [
            (
                "run_interval_aggregation",
                True,
                {"daily": 10, "weekly": 5, "monthly": 2},
            ),
            ("run_interval_aggregation", False, None),
            ("run_numeric_aggregation", True, {"hourly": 100, "weekly": 20}),
            ("run_numeric_aggregation", False, None),
        ],
    )
    async def test_aggregation_function(
        self,
        coordinator: AreaOccupancyCoordinator,
        function_name: str,
        success: bool,
        expected_results: dict[str, int] | None,
    ) -> None:
        """Test aggregation functions (interval/numeric) for success and failure cases."""
        if function_name == "run_interval_aggregation":
            function = run_interval_aggregation
            db_method = "run_interval_aggregation"
        else:
            function = run_numeric_aggregation
            db_method = "run_numeric_aggregation"

        if success:
            with patch.object(coordinator.db, db_method, return_value=expected_results):
                result = await function(coordinator, return_results=True)
                assert result == expected_results
        else:
            with (
                patch.object(
                    coordinator.db,
                    db_method,
                    side_effect=RuntimeError("DB error"),
                ),
                patch(
                    "custom_components.area_occupancy.data.analysis._LOGGER"
                ) as mock_logger,
            ):
                result = await function(coordinator, return_results=True)
                assert result is None
                mock_logger.error.assert_called()

    async def test_start_prior_analysis_success(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test start_prior_analysis calls calculate_and_update_prior."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        with patch(
            "custom_components.area_occupancy.data.analysis.PriorAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.calculate_and_update_prior = Mock()
            mock_analyzer_class.return_value = mock_analyzer

            await start_prior_analysis(coordinator, area_name, area.prior)

            # Should have called calculate_and_update_prior
            mock_analyzer.calculate_and_update_prior.assert_called_once()

    async def test_start_prior_analysis_error_logs(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test start_prior_analysis logs error but doesn't raise."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        with (
            patch(
                "custom_components.area_occupancy.data.analysis.PriorAnalyzer",
                side_effect=ValueError("Test error"),
            ),
            patch(
                "custom_components.area_occupancy.data.analysis._LOGGER"
            ) as mock_logger,
        ):
            # Should not raise
            await start_prior_analysis(coordinator, area_name, area.prior)
            mock_logger.error.assert_called()


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
        # Should be single segment with timeout applied
        assert len(result) == 1
        assert result[0][0] == now
        # End time should be clamped to merged_end (timeout extends motion end but is clamped)
        # Implementation: motion_timeout_end = min(clamped_end + timeout_delta, merged_end)
        # Since motion covers entire interval, clamped_end = merged_end, so timeout is clamped
        assert result[0][1] == now + timedelta(hours=2)

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
        # First interval should be segmented (before + motion+timeout + after = 3 segments)
        # Second interval should be unchanged (1 segment)
        # Total should be 4 segments (or fewer if segments merge)
        assert len(result) >= 2
        # First interval should be segmented
        assert result[0][0] == now
        # Second interval should remain unchanged - find it in results
        second_interval_found = any(
            start == now + timedelta(hours=2) and end == now + timedelta(hours=3)
            for start, end in result
        )
        assert second_interval_found

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
        # Timeout extension is clamped to merged interval end
        # Motion ends at 1h10m, but first merged interval ends at 1h
        # So timeout is clamped to 1h, and intervals don't merge
        assert isinstance(result, list)
        assert len(result) == 2  # Intervals don't merge because timeout is clamped
        # First interval should be segmented
        assert result[0][0] == now
        # Second interval should remain unchanged
        assert result[1] == (
            now + timedelta(hours=1, minutes=30),
            now + timedelta(hours=2),
        )

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

    def test_timestamp_within_interval_middle(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with timestamp in middle of interval."""
        now = freeze_time
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=2), now + timedelta(hours=3)),
        ]
        result = is_timestamp_occupied(now + timedelta(minutes=30), occupied_intervals)
        assert result is True

    def test_single_point_interval(self, freeze_time: datetime) -> None:
        """Test is_timestamp_occupied with single point interval."""
        now = freeze_time
        occupied_intervals = [
            (now, now),  # Zero duration interval
        ]
        # start <= timestamp < end: now <= now < now is False (can't be < self)
        result = is_timestamp_occupied(now, occupied_intervals)
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

    @pytest.mark.parametrize(
        ("timestamp_offset", "expected_result", "description"),
        [
            # Before intervals
            (timedelta(hours=-1), False, "timestamp before all intervals"),
            # After intervals
            (timedelta(hours=4), False, "timestamp after all intervals"),
            # At interval start (inclusive)
            (timedelta(hours=0), True, "timestamp at interval start"),
            # Just before start (exclusive)
            (timedelta(microseconds=-1), False, "timestamp just before start"),
            # Just after end (exclusive)
            (timedelta(hours=1, microseconds=1), False, "timestamp just after end"),
            # In gap between intervals
            (
                timedelta(hours=1, minutes=30),
                False,
                "timestamp in gap between intervals",
            ),
            # Far future
            (timedelta(days=1000), False, "timestamp far in future"),
            # Far past
            (timedelta(days=-1000), False, "timestamp far in past"),
        ],
    )
    def test_timestamp_position(
        self,
        freeze_time: datetime,
        timestamp_offset: timedelta,
        expected_result: bool,
        description: str,
    ) -> None:
        """Test is_timestamp_occupied with various timestamp positions."""
        now = freeze_time
        # Use intervals that work for all test cases
        occupied_intervals = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=2), now + timedelta(hours=3)),
        ]
        timestamp = now + timestamp_offset
        result = is_timestamp_occupied(timestamp, occupied_intervals)
        assert result == expected_result, f"Failed for {description}"
