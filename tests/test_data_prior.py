"""Tests for the Prior class (updated for improved implementation)."""

from unittest.mock import PropertyMock, patch

import pytest

from custom_components.area_occupancy.const import (
    DEFAULT_TIME_PRIOR,
    MAX_PRIOR,
    MIN_PRIOR,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.prior import (
    DEFAULT_SLOT_MINUTES,
    PRIOR_FACTOR,
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
    # Mock time_prior property to return None to avoid database calls
    # This simulates the case where time_prior is not available
    with patch.object(
        Prior, "time_prior", new_callable=PropertyMock, return_value=None
    ):
        assert prior.value == expected_value, f"Failed for {description}"


@pytest.mark.parametrize(
    (
        "global_prior",
        "time_prior_value",
        "min_override",
        "expected_result",
        "description",
    ),
    [
        # Test 1: global_prior below threshold, no time_prior
        (0.2, None, 0.3, 0.3, "global_prior below threshold, no time_prior"),
        # Test 2: global_prior below threshold after PRIOR_FACTOR
        (0.25, None, 0.3, 0.3, "global_prior below threshold after PRIOR_FACTOR"),
        # Test 3: combined prior below threshold
        (0.25, 0.1, 0.3, 0.3, "combined prior below threshold"),
        # Test 4: min_override disabled (0.0)
        (0.05, None, 0.0, MIN_PRIOR, "min_override disabled"),
    ],
)
def test_min_prior_override_scenarios(
    coordinator: AreaOccupancyCoordinator,
    global_prior: float | None,
    time_prior_value: float | None,
    min_override: float,
    expected_result: float,
    description: str,
):
    """Test min_prior_override in various scenarios."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)

    # Set min_prior_override
    area.config.min_prior_override = min_override

    # Set global_prior
    prior.global_prior = global_prior

    # Set up time_prior
    if time_prior_value is None:
        # Mock time_prior property to return None
        with patch.object(
            Prior, "time_prior", new_callable=PropertyMock, return_value=None
        ):
            result = prior.value
            assert result == expected_result, (
                f"Failed for {description}: expected {expected_result}, got {result}"
            )
    else:
        # Set time_prior via cache
        current_day = prior.day_of_week
        current_slot = prior.time_slot
        slot_key = (current_day, current_slot)
        prior._cached_time_priors = {slot_key: time_prior_value}
        result = prior.value
        assert result >= expected_result, (
            f"Failed for {description}: expected at least {expected_result}, got {result}"
        )


def test_min_prior_override_when_global_prior_is_none(
    coordinator: AreaOccupancyCoordinator,
):
    """Test min_prior_override is applied when global_prior is None."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)

    # Set min_prior_override to 0.3
    area.config.min_prior_override = 0.3

    # Ensure global_prior is None (default state before prior is calculated)
    prior.global_prior = None

    # When global_prior is None, it should default to MIN_PRIOR
    # But min_prior_override (0.3) should take precedence
    result = prior.value
    assert result == 0.3, f"Expected 0.3 but got {result}"


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
        # Verify cache is invalidated
        assert prior._cached_time_priors is None


def test_time_prior_property(coordinator: AreaOccupancyCoordinator):
    """Test time_prior property loads from cache correctly."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

    # Initially cache is None, should load from database
    # Mock _load_time_priors to avoid database call
    test_cache = {(0, 14): 0.6, (1, 10): 0.4}
    with patch.object(
        prior,
        "_load_time_priors",
        side_effect=lambda: setattr(prior, "_cached_time_priors", test_cache),
    ):
        # Set cache manually to simulate loaded state
        prior._cached_time_priors = test_cache.copy()

        # Test getting time_prior for a slot in cache
        current_day = prior.day_of_week
        current_slot = prior.time_slot
        slot_key = (current_day, current_slot)

        if slot_key in test_cache:
            assert prior.time_prior == test_cache[slot_key]
        else:
            # If current slot not in test cache, should return DEFAULT_TIME_PRIOR
            assert prior.time_prior == DEFAULT_TIME_PRIOR


def test_day_of_week_property(coordinator: AreaOccupancyCoordinator):
    """Test day_of_week property returns correct weekday."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

    # Should return Python weekday (0=Monday, 6=Sunday)
    weekday = prior.day_of_week
    assert 0 <= weekday <= 6
    # Verify it matches Python's weekday
    assert weekday == dt_util.utcnow().weekday()


def test_time_slot_property(coordinator: AreaOccupancyCoordinator):
    """Test time_slot property calculates correct slot."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

    # Should return hour-based slot (0-23)
    slot = prior.time_slot
    assert 0 <= slot <= 23
    # Verify it matches expected calculation
    now = dt_util.utcnow()
    expected_slot = (now.hour * 60 + now.minute) // DEFAULT_SLOT_MINUTES
    assert slot == expected_slot


def test_clear_cache(coordinator: AreaOccupancyCoordinator):
    """Test clear_cache method clears all cached data."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

    # Set up some cached data
    prior.global_prior = 0.5
    prior._last_updated = dt_util.utcnow()
    prior._cached_time_priors = {(0, 0): 0.3}

    # Clear cache
    prior.clear_cache()

    # Verify all caches are cleared
    assert prior.global_prior is None
    assert prior._last_updated is None
    assert prior._cached_time_priors is None
