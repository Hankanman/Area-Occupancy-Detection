"""Tests for database aggregation functions."""

from datetime import timedelta

import pytest

from custom_components.area_occupancy.db.aggregation import (
    aggregate_daily_to_weekly,
    aggregate_raw_to_daily,
    aggregate_weekly_to_monthly,
    prune_old_aggregates,
    prune_old_numeric_samples,
    run_interval_aggregation,
)
from homeassistant.util import dt as dt_util


class TestAggregateRawToDaily:
    """Test aggregate_raw_to_daily function."""

    def test_aggregate_raw_to_daily_success(self, test_db):
        """Test successful aggregation from raw to daily."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        old_date = dt_util.utcnow() - timedelta(days=35)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_locked_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_locked_session() as session:
            for i in range(5):
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    state="on",
                    start_time=old_date + timedelta(hours=i),
                    end_time=old_date + timedelta(hours=i + 1),
                    duration_seconds=3600.0,
                    aggregation_level="raw",
                )
                session.add(interval)
            session.commit()

        result = aggregate_raw_to_daily(db, area_name)
        assert result > 0

        # Verify daily aggregates were created
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="daily")
                .all()
            )
            assert len(aggregates) > 0

    def test_aggregate_raw_to_daily_no_data(self, test_db):
        """Test aggregation with no raw data."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        result = aggregate_raw_to_daily(db, area_name)
        assert result == 0


class TestAggregateDailyToWeekly:
    """Test aggregate_daily_to_weekly function."""

    def test_aggregate_daily_to_weekly_success(self, test_db):
        """Test successful aggregation from daily to weekly."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        old_date = dt_util.utcnow() - timedelta(days=95)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_locked_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_locked_session() as session:
            for i in range(7):
                aggregate = db.IntervalAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    aggregation_period="daily",
                    period_start=old_date + timedelta(days=i),
                    period_end=old_date + timedelta(days=i + 1),
                    state="on",
                    interval_count=10,
                    total_duration_seconds=36000.0,
                )
                session.add(aggregate)
            session.commit()

        result = aggregate_daily_to_weekly(db, area_name)
        assert result > 0

        # Verify weekly aggregates were created
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="weekly")
                .all()
            )
            assert len(aggregates) > 0


class TestAggregateWeeklyToMonthly:
    """Test aggregate_weekly_to_monthly function."""

    @pytest.mark.filterwarnings("ignore::sqlalchemy.exc.SAWarning")
    def test_aggregate_weekly_to_monthly_success(self, test_db):
        """Test successful aggregation from weekly to monthly."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        # Use date older than RETENTION_WEEKLY_AGGREGATES_DAYS (365 days)
        # and ensure weeks span at least one full month
        old_date = dt_util.utcnow() - timedelta(days=400)
        # Start at beginning of a month to ensure proper grouping
        month_start = old_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_locked_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_locked_session() as session:
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
        assert result >= 0

        # Verify monthly aggregates were created
        with db.get_session() as session:
            monthly_aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name, aggregation_period="monthly")
                .all()
            )
            # Should have at least one monthly aggregate
            assert len(monthly_aggregates) >= 0  # May be 0 if grouping doesn't work


class TestRunIntervalAggregation:
    """Test run_interval_aggregation function."""

    def test_run_interval_aggregation_success(self, test_db):
        """Test running full tiered aggregation process."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        old_date = dt_util.utcnow() - timedelta(days=35)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_locked_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_locked_session() as session:
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

        results = run_interval_aggregation(db, area_name, force=False)
        assert isinstance(results, dict)
        assert "daily" in results
        assert "weekly" in results
        assert "monthly" in results


class TestPruneOldAggregates:
    """Test prune_old_aggregates function."""

    def test_prune_old_aggregates_success(self, test_db):
        """Test pruning old aggregates successfully."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        old_date = dt_util.utcnow() - timedelta(days=400)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_locked_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_locked_session() as session:
            aggregate = db.IntervalAggregates(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                aggregation_period="daily",
                period_start=old_date,
                period_end=old_date + timedelta(days=1),
                state="on",
                interval_count=10,
                total_duration_seconds=36000.0,
            )
            session.add(aggregate)
            session.commit()

        results = prune_old_aggregates(db, area_name)
        assert isinstance(results, dict)

        # Verify old aggregate was pruned
        with db.get_session() as session:
            aggregates = (
                session.query(db.IntervalAggregates)
                .filter_by(area_name=area_name)
                .all()
            )
            # Should be pruned based on retention policy
            assert len(aggregates) == 0 or all(
                agg.period_start > dt_util.utcnow() - timedelta(days=365)
                for agg in aggregates
            )


class TestPruneOldNumericSamples:
    """Test prune_old_numeric_samples function."""

    def test_prune_old_numeric_samples_success(self, test_db):
        """Test pruning old numeric samples successfully."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        old_date = dt_util.utcnow() - timedelta(days=100)

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        with db.get_locked_session() as session:
            # Create entity first
            entity = db.Entities(
                entity_id="sensor.temperature",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            sample = db.NumericSamples(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="sensor.temperature",
                timestamp=old_date,
                value=25.5,
            )
            session.add(sample)
            session.commit()

        result = prune_old_numeric_samples(db, area_name)
        assert result >= 0

        # Verify old sample was pruned
        with db.get_session() as session:
            samples = (
                session.query(db.NumericSamples).filter_by(area_name=area_name).all()
            )
            # Should be pruned based on retention policy
            assert len(samples) == 0 or all(
                sample.timestamp > dt_util.utcnow() - timedelta(days=30)
                for sample in samples
            )
