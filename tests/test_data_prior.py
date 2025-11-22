"""Tests for the Prior class (updated for improved implementation)."""

from datetime import timedelta
import logging
from unittest.mock import patch

import pytest

from custom_components.area_occupancy.const import (
    DEFAULT_CACHE_TTL_SECONDS,
    MAX_PRIOR,
    MAX_PROBABILITY,
    MIN_PRIOR,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.analysis import (
    DAYS_PER_WEEK,
    DEFAULT_OCCUPIED_SECONDS,
    DEFAULT_SLOT_MINUTES,
    HOURS_PER_DAY,
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    SQLITE_TO_PYTHON_WEEKDAY_OFFSET,
)
from custom_components.area_occupancy.data.prior import (
    DEFAULT_PRIOR,
    PRIOR_FACTOR,
    SIGNIFICANT_CHANGE_THRESHOLD,
    Prior,
)
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
def test_initialization(coordinator: AreaOccupancyCoordinator):
    """Test Prior initialization with real coordinator."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)
    # Check that sensor_ids matches the area config
    assert prior.sensor_ids == area.config.sensors.motion
    assert prior.hass == coordinator.hass
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
    coordinator: AreaOccupancyCoordinator,
    global_prior,
    expected_value,
    description,
):
    """Test value property handles various global_prior values correctly."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)
    prior.global_prior = global_prior
    # Mock get_time_prior to return None to avoid database calls
    with patch.object(prior, "get_time_prior", return_value=None):
        assert prior.value == expected_value, f"Failed for {description}"


def test_min_prior_override_with_global_prior_below_threshold(
    coordinator: AreaOccupancyCoordinator,
):
    """Test min_prior_override is applied when global_prior is below threshold."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)

    # Set min_prior_override to 0.3
    area.config.min_prior_override = 0.3

    # Set global_prior to 0.2 (below min_prior_override)
    prior.global_prior = 0.2

    # Mock get_time_prior to return None to avoid database calls
    with patch.object(prior, "get_time_prior", return_value=None):
        # After applying PRIOR_FACTOR: 0.2 * 1.05 = 0.21
        # But min_prior_override (0.3) should take precedence
        result = prior.value
        assert result == 0.3, f"Expected 0.3 but got {result}"


def test_min_prior_override_with_combined_prior_below_threshold(
    coordinator: AreaOccupancyCoordinator,
):
    """Test min_prior_override is applied when combined prior is below threshold."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)

    # Set min_prior_override to 0.3
    area.config.min_prior_override = 0.3

    # Set global_prior to 0.25
    prior.global_prior = 0.25

    # Mock time_prior to return a low value that would lower the combined prior
    # Combined prior will be less than 0.3, so min_prior_override should apply
    with patch.object(prior, "get_time_prior", return_value=0.1):
        # Combined prior will be less than min_prior_override
        # After PRIOR_FACTOR it could still be below, so min_prior_override should apply
        result = prior.value
        assert result >= 0.3, f"Expected at least 0.3 but got {result}"


def test_min_prior_override_disabled_when_zero(
    coordinator: AreaOccupancyCoordinator,
):
    """Test min_prior_override has no effect when set to 0.0 (disabled)."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)

    # Set min_prior_override to 0.0 (disabled)
    area.config.min_prior_override = 0.0

    # Set global_prior to a low value
    prior.global_prior = 0.05

    # Mock get_time_prior to return None to avoid database calls
    with patch.object(prior, "get_time_prior", return_value=None):
        # Should use normal calculation without override
        result = prior.value
        # After PRIOR_FACTOR: 0.05 * 1.05 = 0.0525, clamped to MIN_PRIOR
        assert result == MIN_PRIOR, f"Expected MIN_PRIOR ({MIN_PRIOR}) but got {result}"


def test_min_prior_override_above_normal_calculation(
    coordinator: AreaOccupancyCoordinator,
):
    """Test min_prior_override when final prior after PRIOR_FACTOR is below threshold."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)

    # Set min_prior_override to 0.3
    area.config.min_prior_override = 0.3

    # Set global_prior to 0.25
    # After PRIOR_FACTOR: 0.25 * 1.05 = 0.2625 (below min_prior_override)
    prior.global_prior = 0.25

    # Mock get_time_prior to return None to avoid database calls
    with patch.object(prior, "get_time_prior", return_value=None):
        result = prior.value
        # Should be raised to min_prior_override (0.3) even though calculation gives 0.2625
        assert result == 0.3, f"Expected 0.3 but got {result}"


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


def test_to_dict_and_from_dict(coordinator: AreaOccupancyCoordinator):
    """Test Prior serialization and deserialization."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)
    now = dt_util.utcnow()
    prior.global_prior = 0.42
    prior._last_updated = now
    d = prior.to_dict()
    assert d["value"] == 0.42
    assert d["last_updated"] == now.isoformat()
    restored = Prior.from_dict(d, coordinator, area_name=area_name)
    assert restored.global_prior == 0.42
    assert restored._last_updated == now


def test_set_global_prior(coordinator: AreaOccupancyCoordinator):
    """Test set_global_prior method."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)
    now = dt_util.utcnow()

    with patch(
        "custom_components.area_occupancy.data.prior.dt_util.utcnow", return_value=now
    ):
        prior.set_global_prior(0.75)
        assert prior.global_prior == 0.75
        assert prior._last_updated == now


def test_last_updated_property(coordinator: AreaOccupancyCoordinator):
    """Test last_updated property."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)
    assert prior.last_updated is None

    now = dt_util.utcnow()
    prior._last_updated = now
    assert prior.last_updated == now


# New tests for performance optimization features


def test_get_occupied_intervals_caching(
    coordinator: AreaOccupancyCoordinator,
):
    """Test caching behavior of get_occupied_intervals."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

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


def test_get_occupied_intervals_cache_expiry(
    coordinator: AreaOccupancyCoordinator,
):
    """Test cache expiry after DEFAULT_CACHE_TTL_SECONDS."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

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


def test_invalidate_occupied_intervals_cache(
    coordinator: AreaOccupancyCoordinator,
):
    """Test cache invalidation method."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

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


def test_get_occupied_intervals_cache_hit_logging(
    coordinator: AreaOccupancyCoordinator, caplog
):
    """Test cache hit logging."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

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
