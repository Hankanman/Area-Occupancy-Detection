"""Tests for time-based priors functionality."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.area_occupancy.schema import AreaTimePriorRecord
from custom_components.area_occupancy.utils import (
    datetime_to_time_slot,
    get_current_time_slot,
    get_time_slot_name,
    time_slot_to_datetime_range,
)


# ruff: noqa: SLF001, PLC0415
class TestTimeBasedPriorUtilities:
    """Test time-based prior utility functions."""

    def test_datetime_to_time_slot(self):
        """Test datetime to time slot conversion."""
        # Monday 13:15 (Monday=0, 13:15 = slot 26)
        dt = datetime(2024, 1, 1, 13, 15)  # Monday
        day_of_week, time_slot = datetime_to_time_slot(dt)
        assert day_of_week == 0  # Monday
        assert time_slot == 26  # 13:00-13:29

        # Sunday 23:45 (Sunday=6, 23:45 = slot 47)
        dt = datetime(2024, 1, 7, 23, 45)  # Sunday
        day_of_week, time_slot = datetime_to_time_slot(dt)
        assert day_of_week == 6  # Sunday
        assert time_slot == 47  # 23:30-23:59

        # Wednesday 00:00 (Wednesday=2, 00:00 = slot 0)
        dt = datetime(2024, 1, 3, 0, 0)  # Wednesday
        day_of_week, time_slot = datetime_to_time_slot(dt)
        assert day_of_week == 2  # Wednesday
        assert time_slot == 0  # 00:00-00:29

    def test_time_slot_to_datetime_range(self):
        """Test time slot to datetime range conversion."""
        base_date = datetime(2024, 1, 1, 12, 0)  # Monday 12:00

        # Monday 13:00-13:29 (slot 26)
        start_dt, end_dt = time_slot_to_datetime_range(0, 26, base_date)
        assert start_dt.hour == 13
        assert start_dt.minute == 0
        assert end_dt.hour == 13
        assert end_dt.minute == 30

        # Sunday 23:30-23:59 (slot 47)
        start_dt, end_dt = time_slot_to_datetime_range(6, 47, base_date)
        assert start_dt.hour == 23
        assert start_dt.minute == 30
        assert end_dt.hour == 0  # Next day
        assert end_dt.minute == 0

    def test_get_time_slot_name(self):
        """Test time slot name generation."""
        # Monday 13:00-13:29
        name = get_time_slot_name(0, 26)
        assert name == "Monday 13:00-13:30"

        # Sunday 23:30-23:59
        name = get_time_slot_name(6, 47)
        assert name == "Sunday 23:30-00:00"

        # Wednesday 00:00-00:29
        name = get_time_slot_name(2, 0)
        assert name == "Wednesday 00:00-00:30"

    def test_get_current_time_slot(self):
        """Test getting current time slot."""
        with patch("custom_components.area_occupancy.utils.dt_util.utcnow") as mock_now:
            # Mock current time to Monday 14:15
            mock_now.return_value = datetime(2024, 1, 1, 14, 15)
            day_of_week, time_slot = get_current_time_slot()
            assert day_of_week == 0  # Monday
            assert time_slot == 28  # 14:00-14:29


class TestAreaTimePriorRecord:
    """Test AreaTimePriorRecord data class."""

    def test_area_time_prior_record_creation(self):
        """Test creating AreaTimePriorRecord."""
        record = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=1,  # Tuesday
            time_slot=26,  # 13:00-13:29
            prior_value=0.75,
            data_points=10,
        )

        assert record.entry_id == "test_entry"
        assert record.day_of_week == 1
        assert record.time_slot == 26
        assert record.prior_value == 0.75
        assert record.data_points == 10

    def test_time_range_properties(self):
        """Test time range properties."""
        record = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=0,  # Monday
            time_slot=26,  # 13:00-13:29
            prior_value=0.5,
        )

        start_hour, start_minute = record.time_range
        assert start_hour == 13
        assert start_minute == 0

        end_hour, end_minute = record.end_time_range
        assert end_hour == 13
        assert end_minute == 30

    def test_time_range_edge_cases(self):
        """Test time range edge cases."""
        # Slot 47 (23:30-23:59)
        record = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=6,  # Sunday
            time_slot=47,
            prior_value=0.1,
        )

        start_hour, start_minute = record.time_range
        assert start_hour == 23
        assert start_minute == 30

        end_hour, end_minute = record.end_time_range
        assert end_hour == 0  # Next day
        assert end_minute == 0


class TestTimeBasedPriorIntegration:
    """Test time-based prior integration with coordinator."""

    @pytest.mark.asyncio
    async def test_prior_get_time_prior(self, mock_coordinator):
        """Test getting time-based prior from coordinator."""
        from custom_components.area_occupancy.data.prior import Prior

        # Mock coordinator
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.sqlite_store = AsyncMock()
        mock_coordinator.sqlite_store.get_time_prior.return_value = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=0,
            time_slot=26,
            prior_value=0.75,
            data_points=5,
        )

        prior = Prior(mock_coordinator)

        # Test getting time prior
        result = await prior.get_time_prior()
        assert result == 0.75

        # Verify the database was queried
        mock_coordinator.sqlite_store.get_time_prior.assert_called_once_with(
            "test_entry", 0, 26
        )

    @pytest.mark.asyncio
    async def test_prior_get_current_time_prior(self, mock_coordinator):
        """Test getting current time-based prior."""
        from custom_components.area_occupancy.data.prior import Prior

        # Mock coordinator
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.sqlite_store = AsyncMock()
        mock_coordinator.sqlite_store.get_time_prior.return_value = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=0,
            time_slot=26,
            prior_value=0.8,
            data_points=3,
        )

        prior = Prior(mock_coordinator)

        with patch(
            "custom_components.area_occupancy.utils.get_current_time_slot"
        ) as mock_current:
            mock_current.return_value = (0, 26)  # Monday 13:00-13:29

            result = await prior.get_time_prior()
            assert result == 0.8

    @pytest.mark.asyncio
    async def test_prior_fallback_to_global(self, mock_coordinator):
        """Test fallback to global prior when time-based prior not available."""
        from custom_components.area_occupancy.data.prior import Prior

        # Mock coordinator
        mock_coordinator.entry_id = "test_entry"
        mock_coordinator.sqlite_store = AsyncMock()
        mock_coordinator.sqlite_store.get_time_prior.return_value = (
            None  # No time-based prior
        )

        prior = Prior(mock_coordinator)
        prior._current_value = 0.6  # Set global prior

        result = await prior.get_time_prior()
        assert result == 0.6  # Should fall back to global prior
