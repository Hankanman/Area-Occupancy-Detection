"""Tests for database utility functions."""

from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db.utils import (
    is_intervals_empty,
    is_valid_state,
    safe_is_intervals_empty,
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


class TestSafeIsIntervalsEmpty:
    """Test safe_is_intervals_empty function."""

    def test_safe_empty_intervals(self, coordinator: AreaOccupancyCoordinator):
        """Test safe_is_intervals_empty with empty intervals."""
        db = coordinator.db
        result = safe_is_intervals_empty(db)
        assert result is True

    def test_safe_non_empty_intervals(self, coordinator: AreaOccupancyCoordinator):
        """Test safe_is_intervals_empty with non-empty intervals."""
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

        result = safe_is_intervals_empty(db)
        assert result is False

    def test_safe_corruption_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test safe_is_intervals_empty with corruption error."""
        db = coordinator.db

        @contextmanager
        def mock_session():
            raise SQLAlchemyError("database disk image is malformed")

        monkeypatch.setattr(db, "get_session", mock_session)

        # safe_is_intervals_empty catches all exceptions and returns True
        result = safe_is_intervals_empty(db)
        assert result is True  # Should return True to trigger data population

    def test_safe_other_error(self, coordinator: AreaOccupancyCoordinator, monkeypatch):
        """Test safe_is_intervals_empty with other error."""
        db = coordinator.db

        @contextmanager
        def mock_session():
            raise RuntimeError("Other error")

        monkeypatch.setattr(db, "get_session", mock_session)

        with patch(
            "custom_components.area_occupancy.db.utils.maintenance.is_database_corrupted"
        ) as mock_corrupted:
            mock_corrupted.return_value = False
            result = safe_is_intervals_empty(db)
            assert result is True  # Should return True as fallback


# Note: The following functions are comprehensively tested in test_analysis_helpers.py:
# - merge_overlapping_intervals
# - find_overlapping_motion_intervals
# - segment_interval_with_motion
# - apply_motion_timeout
#
# Those tests are kept in test_analysis_helpers.py since they're more comprehensive.
# This file focuses on testing db/utils.py functions that are NOT tested elsewhere.
