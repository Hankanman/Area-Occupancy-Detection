"""Tests for database aggregation functions."""

from datetime import datetime, timedelta

import pytest

from custom_components.area_occupancy.const import (
    RETENTION_DAILY_AGGREGATES_DAYS,
    RETENTION_HOURLY_NUMERIC_DAYS,
    RETENTION_RAW_INTERVALS_DAYS,
    RETENTION_RAW_NUMERIC_SAMPLES_DAYS,
    RETENTION_WEEKLY_AGGREGATES_DAYS,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db.aggregation import (
    aggregate_daily_to_weekly,
    aggregate_hourly_to_weekly,
    aggregate_numeric_samples_to_hourly,
    aggregate_raw_to_daily,
    aggregate_weekly_to_monthly,
    prune_old_aggregates,
    prune_old_numeric_aggregates,
    prune_old_numeric_samples,
    run_interval_aggregation,
    run_numeric_aggregation,
)
from homeassistant.util import dt as dt_util

# Helper functions for common test patterns


def _create_test_entity(
    session, db, area_name: str, entity_id: str, entity_type: str
) -> None:
    """Create a test database entity.

    Args:
        session: Database session
        db: Database instance
        area_name: Area name for the entity
        entity_id: Entity ID
        entity_type: Entity type (e.g., "motion", "temperature")
    """
    entity = db.Entities(
        entry_id=db.coordinator.entry_id,
        area_name=area_name,
        entity_id=entity_id,
        entity_type=entity_type,
    )
    session.add(entity)
    session.commit()


def _get_old_date(retention_days: int, offset_days: int = 1) -> datetime:
    """Calculate an old date based on retention period.

    Args:
        retention_days: Retention period in days
        offset_days: Additional days to subtract (default: 1)

    Returns:
        Datetime that is retention_days + offset_days ago
    """
    return dt_util.utcnow() - timedelta(days=retention_days + offset_days)


def _get_monday_start(old_date: datetime) -> datetime:
    """Calculate Monday start for week-based tests.

    Args:
        old_date: Reference date

    Returns:
        Datetime representing the start of the week (Monday 00:00:00)
    """
    days_since_monday = old_date.weekday()
    monday_start = old_date - timedelta(days=days_since_monday)
    return monday_start.replace(hour=0, minute=0, second=0, microsecond=0)


def _get_month_start(old_date: datetime) -> datetime:
    """Calculate month start for month-based tests.

    Args:
        old_date: Reference date

    Returns:
        Datetime representing the start of the month (day 1, 00:00:00)
    """
    return old_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _setup_area_and_entity(
    db, area_name: str, entity_id: str, entity_type: str
) -> None:
    """Set up area and entity for testing.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Entity ID
        entity_type: Entity type
    """
    db.save_area_data(area_name)
    with db.get_session() as session:
        _create_test_entity(session, db, area_name, entity_id, entity_type)


class TestAggregateRawToDaily:
    """Test aggregate_raw_to_daily function."""

    def test_aggregate_raw_to_daily_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful aggregation from raw to daily."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Use date older than retention period
        old_date = _get_old_date(RETENTION_RAW_INTERVALS_DAYS)

        # Ensure area and entity exist first (foreign key requirements)
        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        # Create intervals with known values for validation
        interval_durations = [
            1800.0,
            3600.0,
            2700.0,
            3600.0,
            4500.0,
        ]  # Different durations
        total_duration = sum(interval_durations)
        expected_count = 5
        expected_min_duration = min(interval_durations)
        expected_max_duration = max(interval_durations)
        expected_avg_duration = total_duration / expected_count

        with db.get_session() as session:
            intervals = []
            for i, duration in enumerate(interval_durations):
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    state="on",
                    start_time=old_date + timedelta(hours=i),
                    end_time=old_date
                    + timedelta(hours=i)
                    + timedelta(seconds=duration),
                    duration_seconds=duration,
                    aggregation_level="raw",
                )
                intervals.append(interval)
                session.add(interval)
            session.commit()
            interval_ids = [interval.id for interval in intervals]

        # Verify raw intervals exist before aggregation
        with db.get_session() as session:
            raw_count_before = (
                session.query(db.Intervals)
                .filter(db.Intervals.id.in_(interval_ids))
                .count()
            )
            assert raw_count_before == expected_count

        result_count, result_ids = aggregate_raw_to_daily(db, area_name)
        assert result_count > 0
        assert isinstance(result_ids, list)
        assert len(result_ids) == result_count

        # Verify daily aggregates were created with correct values
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="daily")
                .all()
            )
            assert len(aggregates) > 0

            # Verify calculation correctness
            daily_agg = aggregates[0]
            assert daily_agg.interval_count == expected_count
            assert abs(daily_agg.total_duration_seconds - total_duration) < 0.01
            assert abs(daily_agg.avg_duration_seconds - expected_avg_duration) < 0.01
            assert abs(daily_agg.min_duration_seconds - expected_min_duration) < 0.01
            assert abs(daily_agg.max_duration_seconds - expected_max_duration) < 0.01
            assert daily_agg.state == "on"
            assert daily_agg.entity_id == "binary_sensor.motion1"

        # Verify raw intervals were deleted after aggregation
        with db.get_session() as session:
            raw_count_after = (
                session.query(db.Intervals)
                .filter(db.Intervals.id.in_(interval_ids))
                .count()
            )
            assert raw_count_after == 0, (
                "Raw intervals should be deleted after aggregation"
            )

    @pytest.mark.parametrize(
        "aggregate_func",
        [
            aggregate_raw_to_daily,
            aggregate_numeric_samples_to_hourly,
        ],
    )
    def test_aggregate_no_data(
        self, aggregate_func, coordinator: AreaOccupancyCoordinator
    ):
        """Test aggregation with no source data."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        result_count, result_ids = aggregate_func(db, area_name)
        assert result_count == 0
        assert result_ids == []

    def test_aggregate_raw_to_daily_multiple_days(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test aggregation with intervals spanning multiple days."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Start further back to ensure all 3 days are old enough for aggregation
        # Need to be older than RETENTION_RAW_INTERVALS_DAYS, so start at RETENTION_RAW_INTERVALS_DAYS + 3
        old_date = _get_old_date(RETENTION_RAW_INTERVALS_DAYS, offset_days=3)
        # Start at beginning of a day
        day_start = old_date.replace(hour=0, minute=0, second=0, microsecond=0)

        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        # Create intervals spanning 3 days
        intervals_per_day = 3
        with db.get_session() as session:
            for day_offset in range(3):
                for hour_offset in range(intervals_per_day):
                    interval = db.Intervals(
                        entry_id=db.coordinator.entry_id,
                        area_name=area_name,
                        entity_id="binary_sensor.motion1",
                        state="on",
                        start_time=day_start
                        + timedelta(days=day_offset, hours=hour_offset),
                        end_time=day_start
                        + timedelta(days=day_offset, hours=hour_offset + 1),
                        duration_seconds=3600.0,
                        aggregation_level="raw",
                    )
                    session.add(interval)
            session.commit()

        result_count, _result_ids = aggregate_raw_to_daily(db, area_name)
        assert result_count == 3, "Should create one daily aggregate per day"

        # Verify multiple daily aggregates were created (one per day)
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="daily")
                .all()
            )
            assert len(aggregates) == 3, "Should have 3 daily aggregates"

            # Verify each aggregate has correct interval count
            for agg in aggregates:
                assert agg.interval_count == intervals_per_day

    def test_aggregate_raw_to_daily_duplicate_aggregate(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that duplicate aggregates are not created."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        old_date = _get_old_date(RETENTION_RAW_INTERVALS_DAYS)
        day_start = old_date.replace(hour=0, minute=0, second=0, microsecond=0)

        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        # Create intervals all on the same day
        with db.get_session() as session:
            for i in range(5):
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    state="on",
                    start_time=day_start + timedelta(hours=i),
                    end_time=day_start + timedelta(hours=i + 1),
                    duration_seconds=3600.0,
                    aggregation_level="raw",
                )
                session.add(interval)
            session.commit()

        # Run aggregation twice - should not create duplicates
        result_count1, _ = aggregate_raw_to_daily(db, area_name)
        result_count2, _ = aggregate_raw_to_daily(db, area_name)

        assert result_count1 > 0
        assert result_count2 == 0, "Second aggregation should not create duplicates"

        # Verify only one aggregate exists
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="daily")
                .all()
            )
            assert len(aggregates) == 1, "Should have exactly one daily aggregate"

    def test_aggregate_raw_to_daily_all_areas(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test aggregation with area_name=None processes all areas."""
        db = coordinator.db
        area_names = db.coordinator.get_area_names()
        area_name1 = area_names[0]
        old_date = dt_util.utcnow() - timedelta(days=RETENTION_RAW_INTERVALS_DAYS + 1)

        db.save_area_data(area_name1)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name1,
                entity_id="binary_sensor.motion1",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_session() as session:
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name1,
                entity_id="binary_sensor.motion1",
                state="on",
                start_time=old_date,
                end_time=old_date + timedelta(hours=1),
                duration_seconds=3600.0,
                aggregation_level="raw",
            )
            session.add(interval)
            session.commit()
            interval_id = interval.id

        # Verify interval exists before aggregation
        with db.get_session() as session:
            interval_count_before = (
                session.query(db.Intervals).filter_by(id=interval_id).count()
            )
            assert interval_count_before == 1

        # Test with area_name=None (all areas)
        result_count, result_ids = aggregate_raw_to_daily(db, area_name=None)
        assert result_count > 0, "Should aggregate for all areas when area_name=None"
        assert len(result_ids) == result_count

        # Verify daily aggregate was created
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name1, aggregation_period="daily")
                .all()
            )
            assert len(aggregates) > 0, "Daily aggregate should be created for area"

        # Verify raw interval was deleted
        with db.get_session() as session:
            interval_count_after = (
                session.query(db.Intervals).filter_by(id=interval_id).count()
            )
            assert interval_count_after == 0, (
                "Raw interval should be deleted after aggregation"
            )


class TestAggregateDailyToWeekly:
    """Test aggregate_daily_to_weekly function."""

    def test_aggregate_daily_to_weekly_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful aggregation from daily to weekly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area and entity exist first (foreign key requirements)
        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        # Create daily aggregates with known values
        # Ensure all aggregates fall in the same week by starting on a Monday
        # Start further back to ensure the entire week (including Sunday) is old enough
        old_date_for_week = _get_old_date(
            RETENTION_DAILY_AGGREGATES_DAYS, offset_days=7
        )
        monday_start = _get_monday_start(old_date_for_week)

        daily_counts = [10, 15, 12, 8, 20, 18, 14]
        daily_durations = [
            36000.0,
            54000.0,
            43200.0,
            28800.0,
            72000.0,
            64800.0,
            50400.0,
        ]
        # Calculate min/max duration per interval (not total duration)
        daily_min_durations = [
            duration / count
            for duration, count in zip(daily_durations, daily_counts, strict=True)
        ]
        daily_max_durations = [
            duration / count
            for duration, count in zip(daily_durations, daily_counts, strict=True)
        ]
        expected_total_count = sum(daily_counts)
        expected_total_duration = sum(daily_durations)
        expected_min_duration = min(daily_min_durations)
        expected_max_duration = max(daily_max_durations)

        with db.get_session() as session:
            daily_aggregates = []
            for i, (count, duration) in enumerate(
                zip(daily_counts, daily_durations, strict=True)
            ):
                aggregate = db.IntervalAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    aggregation_period="daily",
                    period_start=monday_start + timedelta(days=i),
                    period_end=monday_start + timedelta(days=i + 1),
                    state="on",
                    interval_count=count,
                    total_duration_seconds=duration,
                    min_duration_seconds=duration / count,
                    max_duration_seconds=duration / count,
                    avg_duration_seconds=duration / count,
                    first_occurrence=monday_start + timedelta(days=i),
                    last_occurrence=monday_start + timedelta(days=i + 1),
                )
                daily_aggregates.append(aggregate)
                session.add(aggregate)
            session.commit()
            daily_ids = [agg.id for agg in daily_aggregates]

        # Verify daily aggregates exist before aggregation
        with db.get_session() as session:
            daily_count_before = (
                session.query(db.IntervalAggregates)
                .filter(db.IntervalAggregates.id.in_(daily_ids))
                .count()
            )
            assert daily_count_before == len(daily_counts)

        result_count, result_ids = aggregate_daily_to_weekly(db, area_name)
        assert result_count > 0
        assert isinstance(result_ids, list)
        assert len(result_ids) == result_count

        # Verify weekly aggregates were created with correct values
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="weekly")
                .order_by(db.IntervalAggregates.period_start)
                .all()
            )
            assert len(aggregates) > 0

            # All 7 daily aggregates should be in one weekly aggregate (same week)
            # Sum all weekly aggregates to verify totals match
            total_interval_count = sum(agg.interval_count for agg in aggregates)
            total_duration = sum(agg.total_duration_seconds for agg in aggregates)
            min_duration = min(
                agg.min_duration_seconds
                for agg in aggregates
                if agg.min_duration_seconds is not None
            )
            max_duration = max(
                agg.max_duration_seconds
                for agg in aggregates
                if agg.max_duration_seconds is not None
            )

            assert total_interval_count == expected_total_count, (
                f"Expected {expected_total_count} intervals, got {total_interval_count}"
            )
            assert abs(total_duration - expected_total_duration) < 0.01
            assert abs(min_duration - expected_min_duration) < 0.01
            assert abs(max_duration - expected_max_duration) < 0.01

            # Verify first aggregate has correct properties
            weekly_agg = aggregates[0]
            assert weekly_agg.state == "on"
            assert weekly_agg.entity_id == "binary_sensor.motion1"

        # Verify daily aggregates were deleted after aggregation
        with db.get_session() as session:
            daily_count_after = (
                session.query(db.IntervalAggregates)
                .filter(db.IntervalAggregates.id.in_(daily_ids))
                .count()
            )
            assert daily_count_after == 0, (
                "Daily aggregates should be deleted after aggregation"
            )

    def test_aggregate_daily_to_weekly_multiple_weeks(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test aggregation across multiple weeks."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Start further back to ensure all 14 days are old enough for aggregation
        # Need to be older than RETENTION_DAILY_AGGREGATES_DAYS, so start at RETENTION_DAILY_AGGREGATES_DAYS + 15
        old_date = _get_old_date(RETENTION_DAILY_AGGREGATES_DAYS, offset_days=15)
        # Start on a Monday to ensure proper week grouping
        monday_start = _get_monday_start(old_date)

        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        # Create daily aggregates spanning 2 weeks (14 days)
        with db.get_session() as session:
            for i in range(14):
                aggregate = db.IntervalAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    aggregation_period="daily",
                    period_start=monday_start + timedelta(days=i),
                    period_end=monday_start + timedelta(days=i + 1),
                    state="on",
                    interval_count=10,
                    total_duration_seconds=36000.0,
                    min_duration_seconds=3600.0,
                    max_duration_seconds=3600.0,
                    avg_duration_seconds=3600.0,
                    first_occurrence=monday_start + timedelta(days=i),
                    last_occurrence=monday_start + timedelta(days=i + 1),
                )
                session.add(aggregate)
            session.commit()

        result_count, _result_ids = aggregate_daily_to_weekly(db, area_name)
        assert result_count >= 2, (
            "Should create at least 2 weekly aggregates for 2 weeks"
        )

        # Verify weekly aggregates were created
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="weekly")
                .all()
            )
            assert len(aggregates) >= 2, "Should have at least 2 weekly aggregates"


class TestAggregateWeeklyToMonthly:
    """Test aggregate_weekly_to_monthly function."""

    @pytest.mark.filterwarnings("ignore::sqlalchemy.exc.SAWarning")
    def test_aggregate_weekly_to_monthly_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful aggregation from weekly to monthly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Use date older than RETENTION_WEEKLY_AGGREGATES_DAYS (365 days)
        # and ensure weeks span at least one full month
        old_date = dt_util.utcnow() - timedelta(days=400)
        # Start at beginning of a month to ensure proper grouping
        month_start = old_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Ensure area and entity exist first (foreign key requirements)
        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        with db.get_session() as session:
            # Create weekly aggregates spanning at least one full month (4-5 weeks)
            for i in range(5):  # 5 weeks to ensure we span a full month
                week_start = month_start + timedelta(weeks=i)
                aggregate = db.IntervalAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    aggregation_period="weekly",
                    period_start=week_start,
                    period_end=week_start + timedelta(weeks=1),
                    state="on",
                    interval_count=70,
                    total_duration_seconds=360000.0,
                )
                session.add(aggregate)
            session.commit()

        result = aggregate_weekly_to_monthly(db, area_name)
        # Should create at least one monthly aggregate if weeks span a full month
        assert result > 0, (
            "Should create monthly aggregates when weeks span a full month"
        )

        # Verify monthly aggregates were created
        with db.get_session() as session:
            monthly_aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="monthly")
                .all()
            )
            # Should have at least one monthly aggregate
            assert len(monthly_aggregates) > 0, "Monthly aggregates should be created"

            # Verify calculation correctness
            monthly_agg = monthly_aggregates[0]
            assert monthly_agg.interval_count > 0
            assert monthly_agg.total_duration_seconds > 0
            assert monthly_agg.state == "on"
            assert monthly_agg.entity_id == "binary_sensor.motion1"

    def test_aggregate_weekly_to_monthly_multiple_months(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test aggregation across multiple months."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Start further back to ensure all 9 weeks are old enough for aggregation
        # Need to be older than RETENTION_WEEKLY_AGGREGATES_DAYS, so start at RETENTION_WEEKLY_AGGREGATES_DAYS + 70 days (10 weeks)
        old_date = _get_old_date(RETENTION_WEEKLY_AGGREGATES_DAYS, offset_days=70)
        # Start at beginning of a month
        month_start = _get_month_start(old_date)

        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        # Create weekly aggregates spanning 2 months (9 weeks)
        with db.get_session() as session:
            for i in range(9):
                week_start = month_start + timedelta(weeks=i)
                aggregate = db.IntervalAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    aggregation_period="weekly",
                    period_start=week_start,
                    period_end=week_start + timedelta(weeks=1),
                    state="on",
                    interval_count=70,
                    total_duration_seconds=360000.0,
                    min_duration_seconds=3600.0,
                    max_duration_seconds=3600.0,
                    avg_duration_seconds=3600.0,
                    first_occurrence=week_start,
                    last_occurrence=week_start + timedelta(weeks=1),
                )
                session.add(aggregate)
            session.commit()

        result = aggregate_weekly_to_monthly(db, area_name)
        assert result >= 2, "Should create at least 2 monthly aggregates for 2 months"

        # Verify monthly aggregates were created
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="monthly")
                .all()
            )
            assert len(aggregates) >= 2, "Should have at least 2 monthly aggregates"


class TestRunIntervalAggregation:
    """Test run_interval_aggregation function."""

    def test_run_interval_aggregation_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test running full tiered aggregation process."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Use date older than retention period
        old_date = _get_old_date(RETENTION_RAW_INTERVALS_DAYS)

        # Ensure area and entity exist first (foreign key requirements)
        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        with db.get_session() as session:
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                state="on",
                start_time=old_date,
                end_time=old_date + timedelta(hours=1),
                duration_seconds=3600.0,
                aggregation_level="raw",
            )
            session.add(interval)
            session.commit()
            interval_id = interval.id

        # Verify raw interval exists before aggregation
        with db.get_session() as session:
            raw_count_before = (
                session.query(db.Intervals).filter_by(id=interval_id).count()
            )
            assert raw_count_before == 1

        results = run_interval_aggregation(db, area_name, force=False)
        assert isinstance(results, dict)
        assert "daily" in results
        assert "weekly" in results
        assert "monthly" in results

        # Verify aggregation actually occurred
        assert results["daily"] > 0, "Should create daily aggregates"

        # Verify raw interval was deleted
        with db.get_session() as session:
            raw_count_after = (
                session.query(db.Intervals).filter_by(id=interval_id).count()
            )
            assert raw_count_after == 0, (
                "Raw intervals should be deleted after aggregation"
            )

        # Verify daily aggregates were created
        with db.get_session() as session:
            daily_count = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="daily")
                .count()
            )
            assert daily_count > 0, "Daily aggregates should exist"


class TestPruneOldAggregates:
    """Test prune_old_aggregates function."""

    @pytest.mark.parametrize(
        (
            "prune_func",
            "aggregate_class",
            "retention_days",
            "period_field",
            "entity_id",
            "entity_type",
            "period_end_delta",
        ),
        [
            (
                prune_old_aggregates,
                "IntervalAggregates",
                RETENTION_DAILY_AGGREGATES_DAYS,
                "daily",
                "binary_sensor.motion1",
                "motion",
                timedelta(days=1),
            ),
            (
                prune_old_numeric_aggregates,
                "NumericAggregates",
                RETENTION_HOURLY_NUMERIC_DAYS,
                "hourly",
                "sensor.temperature",
                "temperature",
                timedelta(hours=1),
            ),
        ],
    )
    def test_prune_old_aggregates_success(
        self,
        prune_func,
        aggregate_class,
        retention_days,
        period_field,
        entity_id,
        entity_type,
        period_end_delta,
        coordinator: AreaOccupancyCoordinator,
    ):
        """Test pruning old aggregates successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Create old aggregates (older than retention) and new aggregates (within retention)
        old_date = _get_old_date(retention_days, offset_days=1)
        new_date = dt_util.utcnow() - timedelta(days=retention_days - 1)

        # Ensure area and entity exist first (foreign key requirements)
        _setup_area_and_entity(db, area_name, entity_id, entity_type)

        # Get the aggregate class dynamically
        AggregateClass = getattr(db, aggregate_class)

        with db.get_session() as session:
            # Create old aggregate (should be pruned)
            if aggregate_class == "IntervalAggregates":
                old_aggregate = AggregateClass(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    aggregation_period=period_field,
                    period_start=old_date,
                    period_end=old_date + period_end_delta,
                    state="on",
                    interval_count=10,
                    total_duration_seconds=36000.0,
                )
                new_aggregate = AggregateClass(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    aggregation_period=period_field,
                    period_start=new_date,
                    period_end=new_date + period_end_delta,
                    state="on",
                    interval_count=5,
                    total_duration_seconds=18000.0,
                )
            else:  # NumericAggregates
                old_aggregate = AggregateClass(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    aggregation_period=period_field,
                    period_start=old_date,
                    period_end=old_date + period_end_delta,
                    min_value=20.0,
                    max_value=25.0,
                    avg_value=22.5,
                    median_value=22.5,
                    sample_count=10,
                    first_value=20.0,
                    last_value=25.0,
                    std_deviation=1.5,
                )
                new_aggregate = AggregateClass(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    aggregation_period=period_field,
                    period_start=new_date,
                    period_end=new_date + period_end_delta,
                    min_value=21.0,
                    max_value=26.0,
                    avg_value=23.5,
                    median_value=23.5,
                    sample_count=12,
                    first_value=21.0,
                    last_value=26.0,
                    std_deviation=1.8,
                )
            session.add(old_aggregate)
            session.add(new_aggregate)
            session.commit()
            old_id = old_aggregate.id
            new_id = new_aggregate.id

        # Verify both aggregates exist before pruning
        with db.get_session() as session:
            count_before = (
                session.query(AggregateClass)
                .filter(AggregateClass.id.in_([old_id, new_id]))
                .count()
            )
            assert count_before == 2

        results = prune_func(db, area_name)
        assert isinstance(results, dict)
        assert period_field in results

        # Verify exact count matches
        assert results[period_field] == 1, (
            f"Should prune exactly one old {period_field} aggregate"
        )

        # Verify old aggregate was pruned and new aggregate remains
        with db.get_session() as session:
            old_agg_exists = session.query(AggregateClass).filter_by(id=old_id).first()
            assert old_agg_exists is None, "Old aggregate should be pruned"

            new_agg_exists = session.query(AggregateClass).filter_by(id=new_id).first()
            assert new_agg_exists is not None, "New aggregate should NOT be pruned"

    def test_prune_old_aggregates_boundary(self, coordinator: AreaOccupancyCoordinator):
        """Test pruning at retention boundary - aggregate within retention should NOT be pruned."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Create aggregate slightly newer than cutoff (within retention period, should NOT be pruned)
        # Use cutoff + 1 day to ensure it's definitely within retention period (newer than cutoff)
        cutoff_date = dt_util.utcnow() - timedelta(days=RETENTION_DAILY_AGGREGATES_DAYS)
        boundary_date = cutoff_date + timedelta(days=1)

        _setup_area_and_entity(db, area_name, "binary_sensor.motion1", "motion")

        with db.get_session() as session:
            aggregate = db.IntervalAggregates(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                aggregation_period="daily",
                period_start=boundary_date,
                period_end=boundary_date + timedelta(days=1),
                state="on",
                interval_count=10,
                total_duration_seconds=36000.0,
            )
            session.add(aggregate)
            session.commit()
            agg_id = aggregate.id

        results = prune_old_aggregates(db, area_name)
        assert results["daily"] == 0, (
            "Aggregate within retention period should NOT be pruned"
        )

        # Verify aggregate still exists
        with db.get_session() as session:
            agg_exists = (
                session.query(db.IntervalAggregates).filter_by(id=agg_id).first()
            )
            assert agg_exists is not None, (
                "Aggregate within retention period should NOT be pruned"
            )


class TestPruneOldNumericSamples:
    """Test prune_old_numeric_samples function."""

    def test_prune_old_numeric_samples_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test pruning old numeric samples successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Create old samples (older than retention) and new samples (within retention)
        old_date = _get_old_date(RETENTION_RAW_NUMERIC_SAMPLES_DAYS)
        new_date = dt_util.utcnow() - timedelta(
            days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS - 1
        )

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        with db.get_session() as session:
            # Create entity first
            entity = db.Entities(
                entity_id="sensor.temperature",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            # Create old sample (should be pruned)
            old_sample = db.NumericSamples(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="sensor.temperature",
                timestamp=old_date,
                value=25.5,
            )
            session.add(old_sample)

            # Create new sample (should NOT be pruned)
            new_sample = db.NumericSamples(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="sensor.temperature",
                timestamp=new_date,
                value=22.0,
            )
            session.add(new_sample)
            session.commit()
            old_id = old_sample.id
            new_id = new_sample.id

        # Verify both samples exist before pruning
        with db.get_session() as session:
            count_before = (
                session.query(db.NumericSamples)
                .filter(db.NumericSamples.id.in_([old_id, new_id]))
                .count()
            )
            assert count_before == 2

        result = prune_old_numeric_samples(db, area_name)
        assert result == 1, "Should prune exactly one old sample"

        # Verify old sample was pruned and new sample remains
        with db.get_session() as session:
            old_sample_exists = (
                session.query(db.NumericSamples).filter_by(id=old_id).first()
            )
            assert old_sample_exists is None, "Old sample should be pruned"

            new_sample_exists = (
                session.query(db.NumericSamples).filter_by(id=new_id).first()
            )
            assert new_sample_exists is not None, "New sample should NOT be pruned"


class TestAggregateNumericSamplesToHourly:
    """Test aggregate_numeric_samples_to_hourly function."""

    def test_aggregate_numeric_samples_to_hourly_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful aggregation from numeric samples to hourly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Use date older than retention period
        old_date = _get_old_date(RETENTION_RAW_NUMERIC_SAMPLES_DAYS)

        # Ensure area and entity exist first (foreign key requirements)
        _setup_area_and_entity(db, area_name, "sensor.temperature", "temperature")

        # Create samples with known values for validation
        # Use values [10, 20, 30, 40, 50] for easy calculation verification
        sample_values = [10.0, 20.0, 30.0, 40.0, 50.0]

        with db.get_session() as session:
            samples = []
            for i, value in enumerate(sample_values):
                sample = db.NumericSamples(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="sensor.temperature",
                    timestamp=old_date + timedelta(hours=i),
                    value=value,
                    unit_of_measurement="°C",
                )
                samples.append(sample)
                session.add(sample)
            session.commit()
            sample_ids = [sample.id for sample in samples]

        # Verify samples exist before aggregation
        with db.get_session() as session:
            sample_count_before = (
                session.query(db.NumericSamples)
                .filter(db.NumericSamples.id.in_(sample_ids))
                .count()
            )
            assert sample_count_before == len(sample_values)

        result_count, result_ids = aggregate_numeric_samples_to_hourly(db, area_name)
        assert result_count > 0
        assert isinstance(result_ids, list)
        assert len(result_ids) == result_count

        # Verify hourly aggregates were created with correct statistics
        with db.get_session() as session:
            aggregates = (
                session.query(db.NumericAggregates)
                .filter_by(area_name=area_name, aggregation_period="hourly")
                .all()
            )
            assert len(aggregates) > 0

            # Verify statistics are calculated correctly for first aggregate (all samples in same hour)
            # Since samples are in different hours, check the aggregate with all 5 samples
            # Actually, samples are in different hours, so we'll have 5 separate aggregates
            # Let's verify the first one has correct values
            first_agg = aggregates[0]
            assert first_agg.min_value is not None
            assert first_agg.max_value is not None
            assert first_agg.avg_value is not None
            assert first_agg.median_value is not None
            assert first_agg.sample_count > 0
            assert first_agg.std_deviation is not None

            # Verify all aggregates have valid statistics
            for agg in aggregates:
                assert agg.min_value is not None
                assert agg.max_value is not None
                assert agg.avg_value is not None
                assert agg.median_value is not None
                assert agg.sample_count > 0
                assert agg.std_deviation is not None

        # Verify samples were deleted after aggregation
        with db.get_session() as session:
            sample_count_after = (
                session.query(db.NumericSamples)
                .filter(db.NumericSamples.id.in_(sample_ids))
                .count()
            )
            assert sample_count_after == 0, (
                "Numeric samples should be deleted after aggregation"
            )

    def test_aggregate_numeric_samples_to_hourly_multiple_hours(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test aggregation with samples spanning multiple hours."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        old_date = _get_old_date(RETENTION_RAW_NUMERIC_SAMPLES_DAYS)
        hour_start = old_date.replace(minute=0, second=0, microsecond=0)

        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="sensor.temperature",
                entity_type="temperature",
            )
            session.add(entity)
            session.commit()

        # Create samples spanning 3 hours
        samples_per_hour = 3
        with db.get_session() as session:
            for hour_offset in range(3):
                for minute_offset in range(samples_per_hour):
                    sample = db.NumericSamples(
                        entry_id=db.coordinator.entry_id,
                        area_name=area_name,
                        entity_id="sensor.temperature",
                        timestamp=hour_start
                        + timedelta(hours=hour_offset, minutes=minute_offset * 20),
                        value=20.0 + hour_offset + minute_offset,
                        unit_of_measurement="°C",
                    )
                    session.add(sample)
            session.commit()

        result_count, _result_ids = aggregate_numeric_samples_to_hourly(db, area_name)
        assert result_count == 3, "Should create one hourly aggregate per hour"

        # Verify multiple hourly aggregates were created
        with db.get_session() as session:
            aggregates = (
                session.query(db.NumericAggregates)
                .filter_by(area_name=area_name, aggregation_period="hourly")
                .all()
            )
            assert len(aggregates) == 3, "Should have 3 hourly aggregates"

            # Verify each aggregate has correct sample count
            for agg in aggregates:
                assert agg.sample_count == samples_per_hour

    def test_aggregate_numeric_samples_to_hourly_single_sample(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test aggregation with single sample per hour."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        old_date = _get_old_date(RETENTION_RAW_NUMERIC_SAMPLES_DAYS)

        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="sensor.temperature",
                entity_type="temperature",
            )
            session.add(entity)
            session.commit()

        with db.get_session() as session:
            sample = db.NumericSamples(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="sensor.temperature",
                timestamp=old_date,
                value=25.5,
                unit_of_measurement="°C",
            )
            session.add(sample)
            session.commit()

        result_count, _result_ids = aggregate_numeric_samples_to_hourly(db, area_name)
        assert result_count == 1, "Should create one hourly aggregate"

        # Verify aggregate has correct values for single sample
        with db.get_session() as session:
            aggregate = (
                session.query(db.NumericAggregates)
                .filter_by(area_name=area_name, aggregation_period="hourly")
                .first()
            )
            assert aggregate is not None
            assert aggregate.sample_count == 1
            assert aggregate.min_value == 25.5
            assert aggregate.max_value == 25.5
            assert aggregate.avg_value == 25.5
            assert aggregate.median_value == 25.5
            assert aggregate.std_deviation == 0.0  # Single sample has no std dev


class TestAggregateHourlyToWeekly:
    """Test aggregate_hourly_to_weekly function."""

    def test_aggregate_hourly_to_weekly_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful aggregation from hourly to weekly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Start further back to ensure all hourly aggregates are old enough
        # Need to be older than RETENTION_HOURLY_NUMERIC_DAYS, so start at RETENTION_HOURLY_NUMERIC_DAYS + 8 days
        old_date = _get_old_date(RETENTION_HOURLY_NUMERIC_DAYS, offset_days=8)

        # Ensure area and entity exist first (foreign key requirements)
        _setup_area_and_entity(db, area_name, "sensor.temperature", "temperature")

        # Create hourly aggregates with known values
        hourly_min_values = [20.0, 21.0, 19.0, 22.0, 20.5]
        hourly_max_values = [25.0, 26.0, 24.0, 27.0, 25.5]
        hourly_avg_values = [22.5, 23.5, 21.5, 24.5, 23.0]
        hourly_sample_counts = [10, 12, 8, 15, 11]

        with db.get_session() as session:
            hourly_aggregates = []
            # Create hourly aggregates spanning a week (7 days * 24 hours = 168 hours)
            # But we'll create a smaller set for testing
            for i in range(7 * 24):  # 7 days * 24 hours
                hour_start = old_date + timedelta(hours=i)
                # Cycle through test values
                idx = i % len(hourly_min_values)
                aggregate = db.NumericAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="sensor.temperature",
                    aggregation_period="hourly",
                    period_start=hour_start,
                    period_end=hour_start + timedelta(hours=1),
                    min_value=hourly_min_values[idx],
                    max_value=hourly_max_values[idx],
                    avg_value=hourly_avg_values[idx],
                    median_value=hourly_avg_values[idx],
                    sample_count=hourly_sample_counts[idx],
                    first_value=hourly_min_values[idx],
                    last_value=hourly_max_values[idx],
                    std_deviation=1.5,
                )
                hourly_aggregates.append(aggregate)
                session.add(aggregate)
            session.commit()
            hourly_ids = [agg.id for agg in hourly_aggregates]

        # Verify hourly aggregates exist before aggregation
        with db.get_session() as session:
            hourly_count_before = (
                session.query(db.NumericAggregates)
                .filter(db.NumericAggregates.id.in_(hourly_ids))
                .count()
            )
            assert hourly_count_before == len(hourly_aggregates)

        result_count, result_ids = aggregate_hourly_to_weekly(db, area_name)
        assert result_count > 0
        assert isinstance(result_ids, list)
        assert len(result_ids) == result_count

        # Verify weekly aggregates were created
        with db.get_session() as session:
            aggregates = (
                session.query(db.NumericAggregates)
                .filter_by(area_name=area_name, aggregation_period="weekly")
                .all()
            )
            assert len(aggregates) > 0

            # Verify calculation correctness
            weekly_agg = aggregates[0]
            assert weekly_agg.sample_count > 0
            assert weekly_agg.min_value is not None
            assert weekly_agg.max_value is not None
            assert weekly_agg.avg_value is not None
            assert weekly_agg.median_value is not None
            assert weekly_agg.std_deviation is not None

        # Verify hourly aggregates were deleted after aggregation
        with db.get_session() as session:
            hourly_count_after = (
                session.query(db.NumericAggregates)
                .filter(db.NumericAggregates.id.in_(hourly_ids))
                .count()
            )
            assert hourly_count_after == 0, (
                "Hourly aggregates should be deleted after aggregation"
            )

    def test_aggregate_hourly_to_weekly_multiple_weeks(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test aggregation across multiple weeks."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Start further back to ensure all hourly aggregates are old enough
        # Need to be older than RETENTION_HOURLY_NUMERIC_DAYS, so start at RETENTION_HOURLY_NUMERIC_DAYS + 15 days
        old_date = _get_old_date(RETENTION_HOURLY_NUMERIC_DAYS, offset_days=15)
        # Start on a Monday
        monday_start = _get_monday_start(old_date)

        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="sensor.temperature",
                entity_type="temperature",
            )
            session.add(entity)
            session.commit()

        # Create hourly aggregates spanning 2 weeks
        with db.get_session() as session:
            for i in range(14 * 24):  # 14 days * 24 hours
                hour_start = monday_start + timedelta(hours=i)
                aggregate = db.NumericAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="sensor.temperature",
                    aggregation_period="hourly",
                    period_start=hour_start,
                    period_end=hour_start + timedelta(hours=1),
                    min_value=20.0,
                    max_value=25.0,
                    avg_value=22.5,
                    median_value=22.5,
                    sample_count=10,
                    first_value=20.0,
                    last_value=25.0,
                    std_deviation=1.5,
                )
                session.add(aggregate)
            session.commit()

        result_count, _result_ids = aggregate_hourly_to_weekly(db, area_name)
        assert result_count >= 2, (
            "Should create at least 2 weekly aggregates for 2 weeks"
        )

        # Verify weekly aggregates were created
        with db.get_session() as session:
            aggregates = (
                session.query(db.NumericAggregates)
                .filter_by(area_name=area_name, aggregation_period="weekly")
                .all()
            )
            assert len(aggregates) >= 2, "Should have at least 2 weekly aggregates"


class TestRunNumericAggregation:
    """Test run_numeric_aggregation function."""

    def test_run_numeric_aggregation_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test running full tiered numeric aggregation process."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        # Use 8 days ago to ensure samples are older than retention period (7 days)
        old_date = _get_old_date(RETENTION_RAW_NUMERIC_SAMPLES_DAYS, offset_days=1)

        # Ensure area and entity exist first (foreign key requirements)
        _setup_area_and_entity(db, area_name, "sensor.temperature", "temperature")

        with db.get_session() as session:
            sample = db.NumericSamples(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="sensor.temperature",
                timestamp=old_date,
                value=25.5,
                unit_of_measurement="°C",
            )
            session.add(sample)
            session.commit()

        # Verify sample exists before aggregation
        with db.get_session() as session:
            sample_count_before = (
                session.query(db.NumericSamples)
                .filter_by(area_name=area_name, entity_id="sensor.temperature")
                .count()
            )
            assert sample_count_before == 1

        results = run_numeric_aggregation(db, area_name, force=False)
        assert isinstance(results, dict)
        assert "hourly" in results
        assert "weekly" in results

        # Verify aggregation actually occurred
        assert results["hourly"] > 0, "Should create hourly aggregates"

        # Verify sample was deleted
        with db.get_session() as session:
            sample_count_after = (
                session.query(db.NumericSamples)
                .filter_by(area_name=area_name, entity_id="sensor.temperature")
                .count()
            )
            assert sample_count_after == 0, (
                "Numeric samples should be deleted after aggregation"
            )

        # Verify hourly aggregates were created
        with db.get_session() as session:
            hourly_count = (
                session.query(db.NumericAggregates)
                .filter_by(area_name=area_name, aggregation_period="hourly")
                .count()
            )
            assert hourly_count > 0, "Hourly aggregates should exist"
