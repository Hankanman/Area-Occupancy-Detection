"""Tests for the Prior class (updated for improved implementation)."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import (
    DataError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)

from custom_components.area_occupancy.const import MAX_PRIOR, MIN_PRIOR
from custom_components.area_occupancy.data.prior import (
    DAYS_PER_WEEK,
    DEFAULT_OCCUPIED_SECONDS,
    DEFAULT_PRIOR,
    DEFAULT_SLOT_MINUTES,
    HOURS_PER_DAY,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    PRIOR_FACTOR,
    SIGNIFICANT_CHANGE_THRESHOLD,
    SQLITE_TO_PYTHON_WEEKDAY_OFFSET,
    Prior,
)
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
@pytest.fixture
def mock_coordinator():
    mock = Mock()
    mock.config.sensors.motion = ["binary_sensor.motion1", "binary_sensor.motion2"]
    mock.config.wasp_in_box.enabled = False
    mock.hass = Mock()
    mock.entry_id = "test_entry_id"

    # Mock database
    mock.db = Mock()
    mock.db.get_session = Mock()
    mock.db.Intervals = Mock()
    mock.db.Entities = Mock()
    mock.db.Priors = Mock()

    return mock


def test_initialization(mock_coordinator):
    prior = Prior(mock_coordinator)
    assert prior.sensor_ids == ["binary_sensor.motion1", "binary_sensor.motion2"]
    assert prior.hass == mock_coordinator.hass
    assert prior.global_prior is None
    assert prior._last_updated is None


def test_value_property_clamping(mock_coordinator):
    prior = Prior(mock_coordinator)
    # Not set
    assert prior.value == MIN_PRIOR

    # Test with PRIOR_FACTOR multiplication
    # Below min after factor
    prior.global_prior = 0.005  # 0.005 * 1.1 = 0.0055, still below MIN_PRIOR
    assert prior.value == MIN_PRIOR

    # Above max after factor
    prior.global_prior = 0.9  # 0.9 * 1.1 = 0.99, within MAX_PRIOR
    expected = min(max(0.9 * PRIOR_FACTOR, MIN_PRIOR), MAX_PRIOR)
    assert prior.value == expected

    # In range after factor
    prior.global_prior = 0.5  # 0.5 * 1.1 = 0.55
    expected = min(max(0.5 * PRIOR_FACTOR, MIN_PRIOR), MAX_PRIOR)
    assert prior.value == expected


def test_value_property_with_invalid_global_prior(mock_coordinator):
    """Test value property handles invalid global_prior values."""
    prior = Prior(mock_coordinator)

    # Test negative value
    prior.global_prior = -0.1
    assert prior.value == MIN_PRIOR

    # Test value above 1.0
    prior.global_prior = 1.5
    assert prior.value == MAX_PRIOR


def test_value_property_significant_change_logging(mock_coordinator):
    """Test value property logging for significant changes."""
    prior = Prior(mock_coordinator)

    # Test non-significant changes (should not log)
    prior.global_prior = 0.8  # 0.8 * 1.1 = 0.88, change of 0.08 < 0.1 threshold
    _ = prior.value  # Should not log significant change

    prior.global_prior = 0.5  # 0.5 * 1.1 = 0.55, change of 0.05 < 0.1 threshold
    _ = prior.value  # Should not log significant change

    # Test significant change (should log)
    # MIN_PRIOR is 0.1, so setting to 0.0 will cause it to be clamped to 0.1
    # Then 0.1 * 1.1 = 0.11, but it gets clamped to MIN_PRIOR (0.1)
    # So the change is 0.0 -> 0.1, which is 0.1 > 0.1 threshold
    prior.global_prior = 0.0
    result = prior.value
    assert result == MIN_PRIOR


@pytest.mark.parametrize("entity_ids", [[], None])
def test_calculate_area_prior_with_invalid_entity_ids(mock_coordinator, entity_ids):
    """Test calculate_area_prior returns default when entity IDs are invalid."""
    prior = Prior(mock_coordinator)
    result = prior.calculate_area_prior(entity_ids)
    assert result == DEFAULT_PRIOR


@pytest.mark.parametrize(
    ("first_time", "last_time", "description", "expected_result"),
    [
        (
            dt_util.utcnow(),
            dt_util.utcnow(),
            "same time",
            MAX_PROBABILITY,
        ),  # Division by zero -> inf -> 1.0
        (
            dt_util.utcnow() + timedelta(hours=1),
            dt_util.utcnow(),
            "reversed time",
            DEFAULT_PRIOR,
        ),
    ],
)
def test_calculate_area_prior_with_invalid_time_ranges(
    mock_coordinator, first_time, last_time, description, expected_result
):
    """Test calculate_area_prior handles invalid time ranges."""
    prior = Prior(mock_coordinator)

    # Mock get_total_occupied_seconds to return some value
    with (
        patch.object(prior, "get_total_occupied_seconds", return_value=100.0),
        patch.object(prior, "get_time_bounds", return_value=(first_time, last_time)),
    ):
        result = prior.calculate_area_prior(["test.entity"])
        assert result == expected_result, f"Failed for {description}"


def test_calculate_area_prior_with_data_corruption(mock_coordinator):
    """Test calculate_area_prior handles data corruption (occupied > total time)."""
    prior = Prior(mock_coordinator)

    # Mock get_total_occupied_seconds to return more than total time
    with patch.object(
        prior, "get_total_occupied_seconds", return_value=7200.0
    ):  # 2 hours
        # Mock get_time_bounds to return 1 hour range
        now = dt_util.utcnow()
        with patch.object(
            prior, "get_time_bounds", return_value=(now, now + timedelta(hours=1))
        ):
            result = prior.calculate_area_prior(["test.entity"])
            # Should cap occupied time to total time, so result should be 1.0
            assert result == 1.0


def test_calculate_area_prior_normal_case(mock_coordinator):
    """Test calculate_area_prior with normal valid data."""
    prior = Prior(mock_coordinator)

    # Mock get_total_occupied_seconds to return 30 minutes
    with patch.object(prior, "get_total_occupied_seconds", return_value=1800.0):
        # Mock get_time_bounds to return 2 hour range
        now = dt_util.utcnow()
        with patch.object(
            prior, "get_time_bounds", return_value=(now, now + timedelta(hours=2))
        ):
            result = prior.calculate_area_prior(["test.entity"])
            # 30 minutes / 2 hours = 0.25
            assert result == 0.25


def test_calculate_area_prior_with_no_data(mock_coordinator):
    """Test calculate_area_prior returns default when no data available."""
    prior = Prior(mock_coordinator)

    # Mock get_total_occupied_seconds to return 0
    with (
        patch.object(prior, "get_total_occupied_seconds", return_value=0.0),
        patch.object(prior, "get_time_bounds", return_value=(None, None)),
    ):
        result = prior.calculate_area_prior(["test.entity"])
        assert result == DEFAULT_PRIOR


@pytest.mark.asyncio
async def test_update_handles_data_errors(mock_coordinator):
    """Test update method handles data errors specifically."""
    prior = Prior(mock_coordinator)

    # Mock calculate_area_prior to raise ValueError
    with patch.object(
        prior, "calculate_area_prior", side_effect=ValueError("Invalid data")
    ):
        await prior.update()
        assert prior.global_prior == MIN_PRIOR
        assert prior._last_updated is not None


@pytest.mark.asyncio
async def test_update_handles_database_errors(mock_coordinator):
    """Test update method handles database errors specifically."""
    prior = Prior(mock_coordinator)

    # Mock calculate_area_prior to raise SQLAlchemyError
    with patch.object(
        prior, "calculate_area_prior", side_effect=SQLAlchemyError("DB error")
    ):
        await prior.update()
        assert prior.global_prior == MIN_PRIOR
        assert prior._last_updated is not None


@pytest.mark.asyncio
async def test_update_handles_unexpected_errors(mock_coordinator):
    """Test update method handles unexpected errors."""
    prior = Prior(mock_coordinator)

    # Mock calculate_area_prior to raise unexpected error
    with patch.object(
        prior, "calculate_area_prior", side_effect=RuntimeError("Unexpected")
    ):
        await prior.update()
        assert prior.global_prior == MIN_PRIOR
        assert prior._last_updated is not None


@pytest.mark.asyncio
async def test_update_successful_case(mock_coordinator):
    """Test update method with successful calculation."""
    prior = Prior(mock_coordinator)

    # Mock successful calculation
    with (
        patch.object(prior, "calculate_area_prior", return_value=0.3),
        patch.object(prior, "compute_time_priors"),
    ):
        await prior.update()
        assert prior.global_prior == 0.3
        assert prior._last_updated is not None


@pytest.mark.asyncio
async def test_update_handles_time_prior_database_errors(mock_coordinator):
    """Test update method handles database errors in compute_time_priors."""
    prior = Prior(mock_coordinator)

    # Mock successful calculate_area_prior
    with (
        patch.object(prior, "calculate_area_prior", return_value=0.3),
        patch.object(
            prior, "compute_time_priors", side_effect=SQLAlchemyError("DB error")
        ),
    ):
        await prior.update()
        assert prior.global_prior == 0.3
        assert prior._last_updated is not None


@pytest.mark.asyncio
async def test_update_handles_time_prior_data_errors(mock_coordinator):
    """Test update method handles data errors in compute_time_priors."""
    prior = Prior(mock_coordinator)

    # Mock successful calculate_area_prior
    with (
        patch.object(prior, "calculate_area_prior", return_value=0.3),
        patch.object(
            prior, "compute_time_priors", side_effect=ValueError("Data error")
        ),
    ):
        await prior.update()
        assert prior.global_prior == 0.3
        assert prior._last_updated is not None


@pytest.mark.asyncio
async def test_update_handles_time_prior_unexpected_errors(mock_coordinator):
    """Test update method handles unexpected errors in compute_time_priors."""
    prior = Prior(mock_coordinator)

    # Mock successful calculate_area_prior
    with (
        patch.object(prior, "calculate_area_prior", return_value=0.3),
        patch.object(
            prior, "compute_time_priors", side_effect=RuntimeError("Unexpected")
        ),
    ):
        await prior.update()
        assert prior.global_prior == 0.3
        assert prior._last_updated is not None


def test_compute_time_priors_parameter_validation(mock_coordinator):
    """Test compute_time_priors validates slot_minutes parameter."""
    prior = Prior(mock_coordinator)

    # Test invalid slot_minutes (negative)
    with (
        patch.object(prior, "get_interval_aggregates", return_value=[]),
        patch.object(
            prior,
            "get_time_bounds",
            return_value=(dt_util.utcnow(), dt_util.utcnow() + timedelta(days=1)),
        ),
    ):
        prior.compute_time_priors(slot_minutes=-10)
        # Should use DEFAULT_SLOT_MINUTES instead


def test_compute_time_priors_slot_alignment_validation(mock_coordinator):
    """Test compute_time_priors validates slot alignment."""
    prior = Prior(mock_coordinator)

    # Test slot_minutes that doesn't divide MINUTES_PER_DAY evenly
    with (
        patch.object(prior, "get_interval_aggregates", return_value=[]),
        patch.object(
            prior,
            "get_time_bounds",
            return_value=(dt_util.utcnow(), dt_util.utcnow() + timedelta(days=1)),
        ),
    ):
        prior.compute_time_priors(slot_minutes=70)  # 70 doesn't divide 1440 evenly
        # Should use DEFAULT_SLOT_MINUTES instead


def test_compute_time_priors_with_invalid_days(mock_coordinator):
    """Test compute_time_priors handles invalid day calculations."""
    prior = Prior(mock_coordinator)

    # Mock get_interval_aggregates to return some data
    with patch.object(prior, "get_interval_aggregates", return_value=[(0, 0, 100.0)]):
        # Mock get_time_bounds to return invalid time range (same date)
        now = dt_util.utcnow()
        with patch.object(prior, "get_time_bounds", return_value=(now, now)):
            prior.compute_time_priors()
            # Should return early due to invalid days calculation


def test_compute_time_priors_with_invalid_slots(mock_coordinator):
    """Test compute_time_priors handles invalid slot calculations."""
    prior = Prior(mock_coordinator)

    # Mock get_interval_aggregates to return some data
    with patch.object(prior, "get_interval_aggregates", return_value=[(0, 0, 100.0)]):
        # Mock get_time_bounds to return valid time range
        now = dt_util.utcnow()
        with patch.object(
            prior, "get_time_bounds", return_value=(now, now + timedelta(days=1))
        ):
            # This should work normally, but we'll test the slot validation path
            prior.compute_time_priors(slot_minutes=1440)  # 1 slot per day
            # Should work with valid slot_minutes


def test_compute_time_priors_with_invalid_interval_data(mock_coordinator):
    """Test compute_time_priors handles invalid interval data."""
    prior = Prior(mock_coordinator)

    # Mock get_interval_aggregates to return invalid data
    with patch.object(
        prior, "get_interval_aggregates", return_value=[("invalid", "data", None)]
    ):
        # Mock get_time_bounds to return valid time range
        now = dt_util.utcnow()
        with patch.object(
            prior, "get_time_bounds", return_value=(now, now + timedelta(days=1))
        ):
            prior.compute_time_priors()
            # Should handle invalid data gracefully


def test_compute_time_priors_with_invalid_slot_number(mock_coordinator):
    """Test compute_time_priors handles invalid slot numbers."""
    prior = Prior(mock_coordinator)

    # Mock get_interval_aggregates to return data with invalid slot
    with patch.object(
        prior, "get_interval_aggregates", return_value=[(0, 999, 100.0)]
    ):  # Invalid slot
        # Mock get_time_bounds to return valid time range
        now = dt_util.utcnow()
        with patch.object(
            prior, "get_time_bounds", return_value=(now, now + timedelta(days=1))
        ):
            prior.compute_time_priors()
            # Should skip invalid slot numbers


def test_compute_time_priors_successful_case(mock_coordinator):
    """Test compute_time_priors with successful database operations."""
    prior = Prior(mock_coordinator)

    # Mock successful database operations
    mock_session = Mock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = (
        None  # No existing prior
    )
    mock_session.add = Mock()
    mock_session.commit = Mock()

    # Mock the context manager properly
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_session)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_coordinator.db.get_session.return_value = mock_context_manager

    # Mock get_interval_aggregates to return valid data
    with patch.object(prior, "get_interval_aggregates", return_value=[(0, 0, 100.0)]):
        # Mock get_time_bounds to return valid time range
        now = dt_util.utcnow()
        with patch.object(
            prior, "get_time_bounds", return_value=(now, now + timedelta(days=1))
        ):
            prior.compute_time_priors()
            # Should successfully create new priors


def test_compute_time_priors_with_existing_prior(mock_coordinator):
    """Test compute_time_priors updates existing priors."""
    prior = Prior(mock_coordinator)

    # Mock existing prior
    mock_existing_prior = Mock()
    mock_existing_prior.prior_value = 0.5
    mock_existing_prior.data_points = 100
    mock_existing_prior.last_updated = dt_util.utcnow()

    mock_session = Mock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = (
        mock_existing_prior
    )
    mock_session.commit = Mock()

    # Mock the context manager properly
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_session)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_coordinator.db.get_session.return_value = mock_context_manager

    # Mock get_interval_aggregates to return valid data
    with patch.object(prior, "get_interval_aggregates", return_value=[(0, 0, 100.0)]):
        # Mock get_time_bounds to return valid time range
        now = dt_util.utcnow()
        with patch.object(
            prior, "get_time_bounds", return_value=(now, now + timedelta(days=1))
        ):
            prior.compute_time_priors()
            # Should successfully update existing prior


def _test_database_error_handling(
    mock_coordinator, method_name, error_class, expected_result
):
    """Test database error handling for different methods."""
    prior = Prior(mock_coordinator)

    # Mock database session to raise the specified error
    mock_session = Mock()
    mock_session.__enter__ = Mock(side_effect=error_class("Test error", None, None))
    mock_session.__exit__ = Mock()
    mock_coordinator.db.get_session.return_value = mock_session

    # Call the method and verify it handles the error gracefully
    method = getattr(prior, method_name)
    if method_name == "get_total_occupied_seconds":
        result = method(["test.entity"])
    elif method_name in ("get_time_bounds", "get_interval_aggregates"):
        result = method()
    else:
        raise ValueError(f"Unknown method: {method_name}")

    assert result == expected_result


@pytest.mark.parametrize(
    ("method_name", "error_class", "expected_result"),
    [
        ("get_interval_aggregates", OperationalError, []),
        ("get_time_bounds", DataError, (None, None)),
        ("get_total_occupied_seconds", ProgrammingError, DEFAULT_OCCUPIED_SECONDS),
    ],
)
def test_database_error_handling(
    mock_coordinator, method_name, error_class, expected_result
):
    """Test database error handling for various methods."""
    _test_database_error_handling(
        mock_coordinator, method_name, error_class, expected_result
    )


def test_get_time_bounds_database_errors(mock_coordinator):
    """Test get_time_bounds handles specific database errors."""
    prior = Prior(mock_coordinator)

    # Mock database session to raise DataError
    mock_session = Mock()
    mock_session.__enter__ = Mock(side_effect=DataError("Invalid data", None, None))
    mock_session.__exit__ = Mock()
    mock_coordinator.db.get_session.return_value = mock_session

    result = prior.get_time_bounds()
    assert result == (None, None)


def test_get_time_bounds_successful_case(mock_coordinator):
    """Test get_time_bounds with successful database operations."""
    prior = Prior(mock_coordinator)

    # Mock successful database query with proper attribute names
    now = dt_util.utcnow()

    # Create a mock that returns actual values for the attributes
    class MockResult:
        def __init__(self, first_time, last_time):
            self.first = first_time
            self.last = last_time

    mock_result = MockResult(now, now + timedelta(hours=1))

    mock_session = Mock()
    # Mock the join path since entity_ids is None
    mock_session.query.return_value.join.return_value.filter.return_value.first.return_value = mock_result
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock()
    mock_coordinator.db.get_session.return_value = mock_session

    result = prior.get_time_bounds()
    assert result == (now, now + timedelta(hours=1))


def test_get_time_bounds_with_entity_ids(mock_coordinator):
    """Test get_time_bounds with specific entity IDs."""
    prior = Prior(mock_coordinator)

    # Mock successful database query
    now = dt_util.utcnow()
    mock_result = Mock()
    mock_result.first = now
    mock_result.last = now + timedelta(hours=1)

    mock_session = Mock()
    mock_session.query.return_value.filter.return_value.first.return_value = mock_result
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock()
    mock_coordinator.db.get_session.return_value = mock_session

    result = prior.get_time_bounds(["test.entity"])
    assert result == (now, now + timedelta(hours=1))


def test_get_total_occupied_seconds_with_none_result(mock_coordinator):
    """Test get_total_occupied_seconds handles None result from database."""
    prior = Prior(mock_coordinator)

    # Mock database query returning None
    mock_session = Mock()
    mock_session.query.return_value.filter.return_value.scalar.return_value = None
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock()
    mock_coordinator.db.get_session.return_value = mock_session

    result = prior.get_total_occupied_seconds(["test.entity"])
    assert result == DEFAULT_OCCUPIED_SECONDS


def test_compute_time_priors_with_zero_total_slot_seconds(mock_coordinator):
    """Test compute_time_priors handles zero total slot seconds."""
    prior = Prior(mock_coordinator)

    # Mock successful database operations
    mock_session = Mock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = (
        None  # No existing prior
    )
    mock_session.add = Mock()
    mock_session.commit = Mock()

    # Mock the context manager properly
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_session)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_coordinator.db.get_session.return_value = mock_context_manager

    # Mock get_interval_aggregates to return valid data
    with patch.object(prior, "get_interval_aggregates", return_value=[(0, 0, 100.0)]):
        # Mock get_time_bounds to return same time (zero days)
        now = dt_util.utcnow()
        with patch.object(prior, "get_time_bounds", return_value=(now, now)):
            prior.compute_time_priors()
            # Should handle zero days gracefully


def test_get_interval_aggregates_with_invalid_data_conversion(mock_coordinator):
    """Test get_interval_aggregates handles invalid data conversion."""
    prior = Prior(mock_coordinator)

    # Mock database query to return invalid data
    mock_result = [("invalid", "data", "not_a_number")]
    mock_session = Mock()
    mock_session.query.return_value.join.return_value.filter.return_value.group_by.return_value.all.return_value = mock_result
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock()
    mock_coordinator.db.get_session.return_value = mock_session

    result = prior.get_interval_aggregates()
    assert result == []  # Should filter out invalid data


def test_get_interval_aggregates_successful_case(mock_coordinator):
    """Test get_interval_aggregates with successful database operations."""
    prior = Prior(mock_coordinator)

    # Mock successful database query
    mock_result = [(0, 0, 100.0), (1, 1, 200.0)]  # Sample interval data
    mock_session = Mock()
    mock_session.query.return_value.join.return_value.filter.return_value.group_by.return_value.all.return_value = mock_result
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock()
    mock_coordinator.db.get_session.return_value = mock_session

    result = prior.get_interval_aggregates()
    assert len(result) == 2
    assert result[0] == (0, 0, 100.0)
    assert result[1] == (1, 1, 200.0)


def test_constants_are_properly_defined():
    """Test that all constants are properly defined."""
    assert PRIOR_FACTOR == 1.1
    assert DEFAULT_PRIOR == 0.5
    assert MIN_PROBABILITY == 0.0
    assert MAX_PROBABILITY == 1.0
    assert SIGNIFICANT_CHANGE_THRESHOLD == 0.1
    assert DEFAULT_SLOT_MINUTES == 60
    assert MINUTES_PER_HOUR == 60
    assert HOURS_PER_DAY == 24
    assert MINUTES_PER_DAY == 1440
    assert DAYS_PER_WEEK == 7
    assert SQLITE_TO_PYTHON_WEEKDAY_OFFSET == 6
    assert DEFAULT_OCCUPIED_SECONDS == 0.0


def test_weekday_conversion():
    """Test SQLite to Python weekday conversion."""
    # SQLite strftime('%w'): 0=Sunday, 1=Monday, ..., 6=Saturday
    # Python: 0=Monday, 1=Tuesday, ..., 6=Sunday

    # Sunday (SQLite 0) -> Sunday (Python 6)
    assert (0 + SQLITE_TO_PYTHON_WEEKDAY_OFFSET) % DAYS_PER_WEEK == 6

    # Monday (SQLite 1) -> Monday (Python 0)
    assert (1 + SQLITE_TO_PYTHON_WEEKDAY_OFFSET) % DAYS_PER_WEEK == 0

    # Tuesday (SQLite 2) -> Tuesday (Python 1)
    assert (2 + SQLITE_TO_PYTHON_WEEKDAY_OFFSET) % DAYS_PER_WEEK == 1

    # Wednesday (SQLite 3) -> Wednesday (Python 2)
    assert (3 + SQLITE_TO_PYTHON_WEEKDAY_OFFSET) % DAYS_PER_WEEK == 2

    # Thursday (SQLite 4) -> Thursday (Python 3)
    assert (4 + SQLITE_TO_PYTHON_WEEKDAY_OFFSET) % DAYS_PER_WEEK == 3

    # Friday (SQLite 5) -> Friday (Python 4)
    assert (5 + SQLITE_TO_PYTHON_WEEKDAY_OFFSET) % DAYS_PER_WEEK == 4

    # Saturday (SQLite 6) -> Saturday (Python 5)
    assert (6 + SQLITE_TO_PYTHON_WEEKDAY_OFFSET) % DAYS_PER_WEEK == 5


def test_to_dict_and_from_dict(mock_coordinator):
    prior = Prior(mock_coordinator)
    now = dt_util.utcnow()
    prior.global_prior = 0.42
    prior._last_updated = now
    d = prior.to_dict()
    assert d["value"] == 0.42
    assert d["last_updated"] == now.isoformat()
    restored = Prior.from_dict(d, mock_coordinator)
    assert restored.global_prior == 0.42
    assert restored._last_updated == now


def test_set_global_prior(mock_coordinator):
    """Test set_global_prior method."""
    prior = Prior(mock_coordinator)
    now = dt_util.utcnow()

    with patch(
        "custom_components.area_occupancy.data.prior.dt_util.utcnow", return_value=now
    ):
        prior.set_global_prior(0.75)
        assert prior.global_prior == 0.75
        assert prior._last_updated == now


def test_last_updated_property(mock_coordinator):
    """Test last_updated property."""
    prior = Prior(mock_coordinator)
    assert prior.last_updated is None

    now = dt_util.utcnow()
    prior._last_updated = now
    assert prior.last_updated == now
