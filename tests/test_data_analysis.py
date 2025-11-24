"""Tests for data analysis module."""

from datetime import timedelta
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.data.analysis import PriorAnalyzer
from custom_components.area_occupancy.db.core import AreaOccupancyDB
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

    def test_prior_analyzer_init_with_invalid_area(self, coordinator: Mock) -> None:
        """Test PriorAnalyzer raises ValueError for invalid area."""
        with pytest.raises(ValueError, match="Area 'Invalid Area' not found"):
            PriorAnalyzer(coordinator, "Invalid Area")


class TestPriorAnalyzerLogic:
    """Test PriorAnalyzer calculation logic with mocks."""

    # Note: PriorAnalyzer logic has been simplified to rely on DB methods
    # or other components, so we test integration more than internal logic here.


class TestPriorAnalyzerWithRealDB:
    """Integration tests for PriorAnalyzer with real database."""

    def test_get_occupied_intervals_with_real_data(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test get_occupied_intervals with real database data."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entity and interval
        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with test_db.get_locked_session() as session:
            entity = test_db.Entities(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with test_db.get_session() as session:
            interval = test_db.Intervals(
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
        intervals = analyzer.get_occupied_intervals(
            include_media=False, include_appliance=False
        )
        assert len(intervals) > 0
