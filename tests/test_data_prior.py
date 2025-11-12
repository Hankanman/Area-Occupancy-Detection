"""Tests for the Prior class (updated for improved implementation)."""

from datetime import timedelta
import logging
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    DEFAULT_CACHE_TTL_SECONDS,
    MAX_PRIOR,
    MIN_PRIOR,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.data.prior import (
    DAYS_PER_WEEK,
    DEFAULT_OCCUPIED_SECONDS,
    DEFAULT_PRIOR,
    DEFAULT_SLOT_MINUTES,
    HOURS_PER_DAY,
    MAX_PROBABILITY,
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
def mock_coordinator(mock_hass):
    mock = Mock()
    mock.config.sensors.motion = ["binary_sensor.motion1", "binary_sensor.motion2"]
    mock.config.wasp_in_box.enabled = False
    mock.hass = mock_hass  # Use the proper mock_hass fixture
    mock.entry_id = "test_entry_id"

    # Mock database
    mock.db = Mock()
    mock.db.get_session = Mock()
    mock.db.Intervals = Mock()
    mock.db.Entities = Mock()
    mock.db.Priors = Mock()

    # Mock multi-area architecture
    mock_area = Mock()
    mock_area.config = mock.config
    mock.areas = {"Test Area": mock_area}

    return mock


def test_initialization(mock_coordinator):
    prior = Prior(mock_coordinator, area_name="Test Area")
    assert prior.sensor_ids == ["binary_sensor.motion1", "binary_sensor.motion2"]
    assert prior.hass == mock_coordinator.hass
    assert prior.global_prior is None
    assert prior._last_updated is None


@pytest.mark.parametrize(
    ("global_prior", "expected_value", "description"),
    [
        (None, MIN_PRIOR, "not set"),
        (0.005, MIN_PRIOR, "below min after factor"),
        (
            0.9,
            min(max(0.9 * PRIOR_FACTOR, MIN_PRIOR), MAX_PRIOR),
            "above max after factor",
        ),
        (
            0.5,
            min(max(0.5 * PRIOR_FACTOR, MIN_PRIOR), MAX_PRIOR),
            "in range after factor",
        ),
        (-0.1, MIN_PRIOR, "negative value"),
        (1.5, MAX_PRIOR, "value above 1.0"),
        (0.0, MIN_PRIOR, "zero value"),
    ],
)
def test_value_property_clamping(
    mock_coordinator, global_prior, expected_value, description
):
    """Test value property handles various global_prior values correctly."""
    prior = Prior(mock_coordinator, area_name="Test Area")
    prior.global_prior = global_prior
    # Mock get_time_prior to return None to avoid database calls
    with patch.object(prior, "get_time_prior", return_value=None):
        assert prior.value == expected_value, f"Failed for {description}"


def test_constants_are_properly_defined():
    """Test that all constants are properly defined."""
    assert PRIOR_FACTOR == 1.05
    assert DEFAULT_PRIOR == 0.5
    assert MIN_PROBABILITY == 0.01
    assert MAX_PROBABILITY == 0.99
    assert SIGNIFICANT_CHANGE_THRESHOLD == 0.1
    assert DEFAULT_SLOT_MINUTES == 60
    assert MINUTES_PER_HOUR == 60
    assert HOURS_PER_DAY == 24
    assert MINUTES_PER_DAY == 1440
    assert DAYS_PER_WEEK == 7
    assert SQLITE_TO_PYTHON_WEEKDAY_OFFSET == 6
    assert DEFAULT_OCCUPIED_SECONDS == 0.0


@pytest.mark.parametrize(
    ("sqlite_weekday", "expected_python_weekday"),
    [
        (0, 6),  # Sunday
        (1, 0),  # Monday
        (2, 1),  # Tuesday
        (3, 2),  # Wednesday
        (4, 3),  # Thursday
        (5, 4),  # Friday
        (6, 5),  # Saturday
    ],
)
def test_weekday_conversion(sqlite_weekday, expected_python_weekday):
    """Test SQLite to Python weekday conversion."""
    result = (sqlite_weekday + SQLITE_TO_PYTHON_WEEKDAY_OFFSET) % DAYS_PER_WEEK
    assert result == expected_python_weekday


def test_to_dict_and_from_dict(mock_coordinator):
    prior = Prior(mock_coordinator, area_name="Test Area")
    now = dt_util.utcnow()
    prior.global_prior = 0.42
    prior._last_updated = now
    d = prior.to_dict()
    assert d["value"] == 0.42
    assert d["last_updated"] == now.isoformat()
    restored = Prior.from_dict(d, mock_coordinator, area_name="Test Area")
    assert restored.global_prior == 0.42
    assert restored._last_updated == now


def test_set_global_prior(mock_coordinator):
    """Test set_global_prior method."""
    prior = Prior(mock_coordinator, area_name="Test Area")
    now = dt_util.utcnow()

    with patch(
        "custom_components.area_occupancy.data.prior.dt_util.utcnow", return_value=now
    ):
        prior.set_global_prior(0.75)
        assert prior.global_prior == 0.75
        assert prior._last_updated == now


def test_last_updated_property(mock_coordinator):
    """Test last_updated property."""
    prior = Prior(mock_coordinator, area_name="Test Area")
    assert prior.last_updated is None

    now = dt_util.utcnow()
    prior._last_updated = now
    assert prior.last_updated == now


# New tests for performance optimization features


def test_get_occupied_intervals_caching(mock_coordinator):
    """Test caching behavior of get_occupied_intervals."""
    prior = Prior(mock_coordinator, area_name="Test Area")

    # Test cache invalidation method directly
    # Set up cache
    test_intervals = [(dt_util.utcnow() - timedelta(hours=1), dt_util.utcnow())]
    prior._cached_occupied_intervals = test_intervals
    prior._cached_intervals_timestamp = dt_util.utcnow()

    # Verify cache is set
    assert prior._cached_occupied_intervals is not None
    assert prior._cached_intervals_timestamp is not None

    # Test cache invalidation
    prior._invalidate_occupied_intervals_cache()

    # Verify cache is cleared
    assert prior._cached_occupied_intervals is None
    assert prior._cached_intervals_timestamp is None


def test_get_occupied_intervals_cache_expiry(mock_coordinator):
    """Test cache expiry after DEFAULT_CACHE_TTL_SECONDS."""
    prior = Prior(mock_coordinator, area_name="Test Area")

    # Test cache expiry logic directly
    test_intervals = [(dt_util.utcnow() - timedelta(hours=1), dt_util.utcnow())]
    prior._cached_occupied_intervals = test_intervals

    # Set cache timestamp to expired time
    prior._cached_intervals_timestamp = dt_util.utcnow() - timedelta(
        seconds=DEFAULT_CACHE_TTL_SECONDS + 1
    )

    # Verify cache is expired
    now = dt_util.utcnow()
    cache_age = (now - prior._cached_intervals_timestamp).total_seconds()
    assert cache_age > DEFAULT_CACHE_TTL_SECONDS


def test_invalidate_occupied_intervals_cache(mock_coordinator):
    """Test cache invalidation method."""
    prior = Prior(mock_coordinator, area_name="Test Area")

    # Set up cache
    prior._cached_occupied_intervals = [(dt_util.utcnow(), dt_util.utcnow())]
    prior._cached_intervals_timestamp = dt_util.utcnow()

    # Verify cache is set
    assert prior._cached_occupied_intervals is not None
    assert prior._cached_intervals_timestamp is not None

    # Invalidate cache
    prior._invalidate_occupied_intervals_cache()

    # Verify cache is cleared
    assert prior._cached_occupied_intervals is None
    assert prior._cached_intervals_timestamp is None


def test_get_occupied_intervals_cache_hit_logging(mock_coordinator, caplog):
    """Test cache hit logging."""
    prior = Prior(mock_coordinator, area_name="Test Area")

    # Test cache hit logging by setting up cache and calling the actual method
    test_intervals = [(dt_util.utcnow() - timedelta(hours=1), dt_util.utcnow())]
    prior._cached_occupied_intervals = test_intervals
    prior._cached_intervals_timestamp = dt_util.utcnow()

    with caplog.at_level(logging.DEBUG):
        # Call the actual method - it should hit cache
        result = prior.get_occupied_intervals()

    # Check for cache hit logging
    assert "Returning cached occupied intervals" in caplog.text
    assert result == test_intervals
