"""Tests for database query functions."""

from datetime import datetime, timedelta

from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.db.operations import (
    save_global_prior,
    save_occupied_intervals_cache,
)
from custom_components.area_occupancy.db.queries import (
    build_appliance_query,
    build_base_filters,
    build_media_query,
    build_motion_query,
    get_area_data,
    get_global_prior,
    get_latest_interval,
    get_occupied_intervals,
    get_occupied_intervals_cache,
    get_time_bounds,
    get_time_prior,
    get_total_occupied_seconds_sql,
    is_occupied_intervals_cache_valid,
)
from homeassistant.util import dt as dt_util


class TestGetAreaData:
    """Test get_area_data function."""

    def test_get_area_data_success(self, test_db):
        """Test get_area_data with existing area."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]

        # Save area data first
        db.save_area_data(area_name)

        result = get_area_data(db, db.coordinator.entry_id)
        assert result is not None
        assert result["entry_id"] == db.coordinator.entry_id
        assert result["area_name"] == area_name

    def test_get_area_data_not_found(self, test_db):
        """Test get_area_data when area doesn't exist."""
        db = test_db
        result = get_area_data(db, "nonexistent_entry")
        assert result is None

    def test_get_area_data_error(self, test_db, monkeypatch):
        """Test get_area_data with database error."""
        db = test_db

        def bad_session():
            raise SQLAlchemyError("Error")

        monkeypatch.setattr(db, "get_session", bad_session)
        result = get_area_data(db, "test")
        assert result is None


class TestGetLatestInterval:
    """Test get_latest_interval function."""

    def test_get_latest_interval_with_data(self, test_db):
        """Test get_latest_interval when intervals exist."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        end = dt_util.utcnow()
        start = end - timedelta(seconds=60)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_locked_session() as session:
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

        result = get_latest_interval(db)
        assert isinstance(result, datetime)
        # Should be end_time - 1 hour
        assert result <= end.replace(tzinfo=None)

    def test_get_latest_interval_no_data(self, test_db):
        """Test get_latest_interval when no intervals exist."""
        db = test_db
        result = get_latest_interval(db)
        assert isinstance(result, datetime)
        # Should return default (now - 10 days)
        # Both result and expected are timezone-aware, so compare directly
        expected = dt_util.now() - timedelta(days=10)
        # Compare as naive datetimes to avoid timezone issues
        result_naive = result.replace(tzinfo=None) if result.tzinfo else result
        expected_naive = expected.replace(tzinfo=None) if expected.tzinfo else expected
        assert abs((result_naive - expected_naive).total_seconds()) < 60

    def test_get_latest_interval_error(self, test_db, monkeypatch):
        """Test get_latest_interval with database error."""
        db = test_db

        def bad_session():
            raise SQLAlchemyError("no such table")

        monkeypatch.setattr(db, "get_session", bad_session)
        result = get_latest_interval(db)
        assert isinstance(result, datetime)


class TestGetTotalOccupiedSecondsSql:
    """Test get_total_occupied_seconds_sql function."""

    def test_get_total_occupied_seconds_sql_success(self, test_db):
        """Test successful calculation."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_locked_session() as session:
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                start_time=start,
                end_time=end,
                state="on",
                duration_seconds=3600,
            )
            session.add(interval)
            session.commit()

        result = get_total_occupied_seconds_sql(
            db, db.coordinator.entry_id, area_name=area_name, lookback_days=90
        )
        assert result is not None
        assert result >= 3600

    def test_get_total_occupied_seconds_sql_empty(self, test_db):
        """Test calculation with no intervals."""
        db = test_db
        result = get_total_occupied_seconds_sql(
            db, "test_entry", area_name=None, lookback_days=90
        )
        assert result == 0.0 or result is None


class TestGetTimePrior:
    """Test get_time_prior function."""

    def test_get_time_prior_with_data(self, test_db):
        """Test get_time_prior when prior exists."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]

        with db.get_locked_session() as session:
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

    def test_get_time_prior_default(self, test_db):
        """Test get_time_prior returns default when prior doesn't exist."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        result = get_time_prior(db, db.coordinator.entry_id, area_name, 1, 14, 0.5)
        assert result == 0.5


class TestGetOccupiedIntervals:
    """Test get_occupied_intervals function."""

    def test_get_occupied_intervals_success(self, test_db):
        """Test successful retrieval of occupied intervals."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)

            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                start_time=start,
                end_time=end,
                state="on",
                duration_seconds=3600,
            )
            session.add(interval)
            session.commit()

        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )
        assert isinstance(result, list)
        if result:
            assert len(result) > 0
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2

    def test_get_occupied_intervals_empty(self, test_db):
        """Test get_occupied_intervals with no intervals."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=90,
            motion_timeout_seconds=0,
        )
        assert result == []

    def test_get_occupied_intervals_with_media_and_appliance(self, test_db):
        """Test retrieval including media/appliance union path."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        start = now - timedelta(hours=2)

        with db.get_locked_session() as session:
            entities = [
                db.Entities(
                    entity_id="binary_sensor.motion1",
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_type="motion",
                ),
                db.Entities(
                    entity_id="media_player.tv",
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_type="media",
                ),
                db.Entities(
                    entity_id="switch.appliance1",
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_type="appliance",
                ),
            ]
            session.add_all(entities)
            session.commit()

        with db.get_locked_session() as session:
            intervals = [
                db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="binary_sensor.motion1",
                    start_time=start,
                    end_time=start + timedelta(minutes=30),
                    state="on",
                    duration_seconds=1800,
                ),
                db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="media_player.tv",
                    start_time=start + timedelta(minutes=40),
                    end_time=start + timedelta(minutes=80),
                    state="playing",
                    duration_seconds=2400,
                ),
                db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="switch.appliance1",
                    start_time=start + timedelta(minutes=90),
                    end_time=start + timedelta(minutes=120),
                    state="on",
                    duration_seconds=1800,
                ),
            ]
            session.add_all(intervals)
            session.commit()

        result = get_occupied_intervals(
            db,
            db.coordinator.entry_id,
            area_name,
            lookback_days=1,
            motion_timeout_seconds=0,
            include_media=True,
            include_appliance=True,
            media_sensor_ids=["media_player.tv"],
            appliance_sensor_ids=["switch.appliance1"],
        )

        assert len(result) == 3


class TestGetTimeBounds:
    """Test get_time_bounds function."""

    def test_get_time_bounds_with_data(self, test_db):
        """Test get_time_bounds when intervals exist."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion1",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)

            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                start_time=start,
                end_time=end,
                state="on",
                duration_seconds=3600,
            )
            session.add(interval)
            session.commit()

        first, last = get_time_bounds(db, db.coordinator.entry_id, area_name)
        assert first is not None
        assert last is not None

    def test_get_time_bounds_no_data(self, test_db):
        """Test get_time_bounds when no intervals exist."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        first, last = get_time_bounds(db, db.coordinator.entry_id, area_name)
        assert first is None
        assert last is None


class TestBuildFilters:
    """Test filter building functions."""

    def test_build_base_filters(self, test_db):
        """Test build_base_filters function."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        lookback_date = dt_util.utcnow() - timedelta(days=90)

        filters = build_base_filters(
            db, db.coordinator.entry_id, lookback_date, area_name
        )
        assert isinstance(filters, list)
        assert len(filters) > 0

    def test_build_motion_query(self, test_db):
        """Test build_motion_query function."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        lookback_date = dt_util.utcnow() - timedelta(days=90)
        base_filters = build_base_filters(
            db, db.coordinator.entry_id, lookback_date, area_name
        )

        with db.get_session() as session:
            query = build_motion_query(session, db, base_filters)
            assert query is not None

    def test_build_media_query(self, test_db):
        """Test build_media_query function."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        lookback_date = dt_util.utcnow() - timedelta(days=90)
        base_filters = build_base_filters(
            db, db.coordinator.entry_id, lookback_date, area_name
        )

        with db.get_session() as session:
            query = build_media_query(session, db, base_filters, ["media.player"])
            assert query is not None

    def test_build_appliance_query(self, test_db):
        """Test build_appliance_query function."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        lookback_date = dt_util.utcnow() - timedelta(days=90)
        base_filters = build_base_filters(
            db, db.coordinator.entry_id, lookback_date, area_name
        )

        with db.get_session() as session:
            query = build_appliance_query(
                session, db, base_filters, ["switch.appliance"]
            )
            assert query is not None


class TestGetGlobalPrior:
    """Test get_global_prior function."""

    def test_get_global_prior_with_data(self, test_db):
        """Test get_global_prior when data exists."""
        db = test_db
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
        assert result["prior_value"] == 0.35

    def test_get_global_prior_no_data(self, test_db):
        """Test get_global_prior when no data exists."""
        db = test_db
        result = get_global_prior(db, "nonexistent_area")
        assert result is None


class TestOccupiedIntervalsCache:
    """Test occupied intervals cache functions."""

    def test_get_occupied_intervals_cache(self, test_db):
        """Test get_occupied_intervals_cache function."""
        db = test_db
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

    def test_is_occupied_intervals_cache_valid(self, test_db):
        """Test is_occupied_intervals_cache_valid function."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]

        # Initially invalid
        assert is_occupied_intervals_cache_valid(db, area_name) is False

        # Save cache
        intervals = [(dt_util.utcnow() - timedelta(hours=1), dt_util.utcnow())]
        save_occupied_intervals_cache(db, area_name, intervals)

        # Should be valid now
        assert is_occupied_intervals_cache_valid(db, area_name) is True
