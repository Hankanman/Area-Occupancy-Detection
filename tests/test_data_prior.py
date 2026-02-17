"""Tests for the Prior class (updated for improved implementation)."""

from datetime import UTC
from unittest.mock import PropertyMock, patch

import pytest

from custom_components.area_occupancy.const import (
    DEFAULT_TIME_PRIOR,
    MAX_PRIOR,
    MAX_PROBABILITY,
    MIN_PRIOR,
    MIN_PROBABILITY,
    TIME_PRIOR_MAX_BOUND,
    TIME_PRIOR_MIN_BOUND,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.prior import (
    PRIOR_FACTOR,
    PURPOSE_MIN_PRIORS,
    Prior,
)
from custom_components.area_occupancy.data.purpose import AreaPurpose, Purpose
from custom_components.area_occupancy.utils import combine_priors
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001, PLC0415
def test_initialization(coordinator: AreaOccupancyCoordinator):
    """Test Prior initialization with real coordinator."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)
    # Check that sensor_ids matches the area config
    assert prior.sensor_ids == area.config.sensors.motion
    assert prior.media_sensor_ids == area.config.sensors.media
    assert prior.appliance_sensor_ids == area.config.sensors.appliance
    assert prior.hass == coordinator.hass
    assert prior.db == coordinator.db
    assert prior.config == area.config
    assert prior.area_name == area_name
    assert prior.coordinator == coordinator
    assert prior.global_prior is None
    assert prior._last_updated is None
    assert prior._cached_time_priors is None


@pytest.mark.parametrize(
    ("global_prior", "expected_value", "description"),
    [
        (None, MIN_PRIOR, "not set"),
        (0.005, MIN_PRIOR, "below min after factor"),
        (
            0.9,
            min(max(0.9 * PRIOR_FACTOR, MIN_PRIOR), MAX_PRIOR),
            "in range after factor (0.9 * 1.05 = 0.945)",
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
    """Test value property handles various global_prior values correctly.

    Note: This test focuses on basic clamping without time_prior combination.
    For time_prior combination tests, see test_value_property_with_time_prior_combination.
    For complex clamping logic tests, see test_value_property_complex_clamping.
    """
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
        "combined_prior",
        "expected_result",
        "check_exact",
        "description",
    ),
    [
        (
            0.5,
            0.6,
            0.55,
            max(MIN_PRIOR, min(MAX_PRIOR, 0.55 * PRIOR_FACTOR)),
            True,
            "normal combination",
        ),
        (
            0.5,
            0.0,
            0.01,  # MIN_PROBABILITY
            None,  # Will check >= MIN_PRIOR
            False,
            "time_prior = 0.0 (converted to MIN_PROBABILITY)",
        ),
        (
            0.5,
            1.0,
            0.99,  # MAX_PROBABILITY
            None,  # Will check <= MAX_PRIOR
            False,
            "time_prior = 1.0 (converted to MAX_PROBABILITY)",
        ),
        (
            0.5,
            0.5,
            0.5,  # Identical priors return the same value
            max(MIN_PRIOR, min(MAX_PRIOR, 0.5 * PRIOR_FACTOR)),
            True,
            "identical priors",
        ),
    ],
)
def test_value_property_with_time_prior_combination(
    coordinator: AreaOccupancyCoordinator,
    global_prior: float,
    time_prior_value: float,
    combined_prior: float,
    expected_result: float | None,
    check_exact: bool,
    description: str,
):
    """Test value property combines global_prior and time_prior correctly."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)
    current_day = prior.day_of_week
    current_slot = prior.time_slot
    slot_key = (current_day, current_slot)

    prior.global_prior = global_prior
    prior._cached_time_priors = {slot_key: time_prior_value}

    with patch(
        "custom_components.area_occupancy.data.prior.combine_priors",
        return_value=combined_prior,
    ) as mock_combine:
        result = prior.value

        # Verify combine_priors was called with correct arguments
        if time_prior_value in {0.0, 1.0}:
            # For edge cases, just verify it was called
            mock_combine.assert_called_once()
        else:
            # For normal cases, verify exact arguments
            mock_combine.assert_called_once_with(global_prior, time_prior_value)

        # Verify result
        if check_exact:
            assert result == expected_result, f"Failed for {description}"
        # For edge cases, check bounds
        elif expected_result is None:
            if time_prior_value == 0.0:
                assert result >= MIN_PRIOR, f"Failed for {description}"
            elif time_prior_value == 1.0:
                assert result <= MAX_PRIOR, f"Failed for {description}"
        else:
            assert result == expected_result, f"Failed for {description}"


@pytest.mark.parametrize(
    (
        "global_prior",
        "time_prior_value",
        "clamp_return_value",
        "combine_return_value",
        "expected_result",
        "description",
    ),
    [
        (
            -0.1,
            None,
            MIN_PROBABILITY,
            None,
            MIN_PRIOR,
            "Prior clamped to MIN_PROBABILITY before PRIOR_FACTOR",
        ),
        (
            1.5,
            None,
            MAX_PROBABILITY,
            None,
            MAX_PRIOR,
            "Prior clamped to MAX_PROBABILITY before PRIOR_FACTOR",
        ),
        (
            0.5,
            None,
            None,
            None,
            max(MIN_PRIOR, min(MAX_PRIOR, 0.5 * PRIOR_FACTOR)),
            "Prior within bounds, PRIOR_FACTOR applied normally",
        ),
        (
            0.5,
            0.9,
            MAX_PROBABILITY,
            1.5,  # Exceeds MAX_PROBABILITY
            MAX_PRIOR,
            "Combined prior exceeds MAX_PROBABILITY after combination",
        ),
    ],
)
def test_value_property_complex_clamping(
    coordinator: AreaOccupancyCoordinator,
    global_prior: float,
    time_prior_value: float | None,
    clamp_return_value: float | None,
    combine_return_value: float | None,
    expected_result: float,
    description: str,
):
    """Test complex clamping logic when prior is clamped before PRIOR_FACTOR."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)
    prior.global_prior = global_prior

    # Set up time_prior if needed
    if time_prior_value is not None:
        current_day = prior.day_of_week
        current_slot = prior.time_slot
        slot_key = (current_day, current_slot)
        prior._cached_time_priors = {slot_key: time_prior_value}
        use_time_prior_patch = False
    else:
        use_time_prior_patch = True

    # Build context managers for patches
    patches = []
    if use_time_prior_patch:
        patches.append(
            patch.object(
                Prior, "time_prior", new_callable=PropertyMock, return_value=None
            )
        )
    if clamp_return_value is not None:
        patches.append(
            patch(
                "custom_components.area_occupancy.data.prior.clamp_probability",
                return_value=clamp_return_value,
            )
        )
    if combine_return_value is not None:
        patches.append(
            patch(
                "custom_components.area_occupancy.data.prior.combine_priors",
                return_value=combine_return_value,
            )
        )

    # Apply all patches using nested context managers
    if len(patches) == 1:
        with patches[0]:
            result = prior.value
    elif len(patches) == 2:
        with patches[0], patches[1]:
            result = prior.value
    elif len(patches) == 3:
        with patches[0], patches[1], patches[2]:
            result = prior.value
    else:
        result = prior.value

    assert result == expected_result, f"Failed for {description}"


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
        (0.05, None, 0.0, 0.05, "min_override disabled"),
        # Test 5: combined prior + PRIOR_FACTOR exceeds min_override
        (0.5, 0.6, 0.4, None, "combined prior + PRIOR_FACTOR exceeds min_override"),
        # Test 6: min_override when global_prior is None
        (None, None, 0.3, 0.3, "min_override when global_prior is None"),
    ],
)
def test_min_prior_override_scenarios(
    coordinator: AreaOccupancyCoordinator,
    global_prior: float | None,
    time_prior_value: float | None,
    min_override: float,
    expected_result: float | None,
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

        if expected_result is None:
            # Test 5: Calculate expected result
            # Combine priors, apply PRIOR_FACTOR, then apply min_override
            combined = combine_priors(global_prior, time_prior_value)
            adjusted = combined * PRIOR_FACTOR
            clamped = max(MIN_PRIOR, min(MAX_PRIOR, adjusted))
            expected_result = max(clamped, min_override)

        assert result == expected_result, (
            f"Failed for {description}: expected {expected_result}, got {result}"
        )


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


@pytest.mark.parametrize(
    ("input_prior", "expected_clamped"),
    [
        (0.5, 0.5),  # Normal value - no clamping
        (0.01, 0.01),  # At MIN_PROBABILITY - no clamping
        (0.99, 0.99),  # At MAX_PROBABILITY - no clamping
        (-0.1, MIN_PROBABILITY),  # Below minimum - clamped to MIN_PROBABILITY
        (1.5, MAX_PROBABILITY),  # Above maximum - clamped to MAX_PROBABILITY
        (0.0, MIN_PROBABILITY),  # Zero - clamped to MIN_PROBABILITY
        (1.0, MAX_PROBABILITY),  # One - clamped to MAX_PROBABILITY
        (float("inf"), MAX_PROBABILITY),  # Infinity - clamped to MAX_PROBABILITY
        (
            float("-inf"),
            MIN_PROBABILITY,
        ),  # Negative infinity - clamped to MIN_PROBABILITY
    ],
)
def test_set_global_prior_clamping(
    coordinator: AreaOccupancyCoordinator, input_prior: float, expected_clamped: float
):
    """Test that set_global_prior clamps values to valid probability bounds."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

    prior.set_global_prior(input_prior)
    assert prior.global_prior == expected_clamped


def test_time_prior_property(coordinator: AreaOccupancyCoordinator):
    """Test time_prior property loads from cache correctly."""
    original_tz = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(dt_util.UTC)
    try:
        area_name = coordinator.get_area_names()[0]
        prior = Prior(coordinator, area_name=area_name)

        # Initially cache is None
        assert prior._cached_time_priors is None

        # Mock database get_all_time_priors to return test data.
        # Derive "other" slot keys relative to current time so they never collide
        # with the current slot (avoids time-dependent test failures).
        current_day = prior.day_of_week
        current_slot = prior.time_slot
        slot_key = (current_day, current_slot)
        other_day = (current_day + 1) % 7
        other_slot = (current_slot + 3) % 24
        missing_day = (current_day + 3) % 7
        missing_slot = (current_slot + 7) % 24
        test_cache = {
            slot_key: 0.6,
            (other_day, other_slot): 0.4,
            ((current_day + 2) % 7, (current_slot + 5) % 24): 0.3,
        }

        with patch.object(
            prior.db,
            "get_all_time_priors",
            return_value=test_cache.copy(),
        ) as mock_get_all:
            # First access should trigger _load_time_priors
            result = prior.time_prior
            mock_get_all.assert_called_once_with(
                area_name=area_name, default_prior=DEFAULT_TIME_PRIOR
            )
            # Cache should now be populated
            assert prior._cached_time_priors is not None
            assert result == 0.6  # Should return value for current slot

            # Second access should use cache (no additional database call)
            mock_get_all.reset_mock()
            result2 = prior.time_prior
            mock_get_all.assert_not_called()
            assert result2 == 0.6

        # Test accessing a different slot by patching dt_util.utcnow()
        from datetime import datetime

        # Build a datetime that maps to (other_day, other_slot)
        # Monday 2024-01-01 is weekday=0, so add other_day days
        other_dt = datetime(2024, 1, 1 + other_day, other_slot, 0, 0, tzinfo=UTC)
        prior._cached_time_priors = test_cache.copy()
        with patch(
            "custom_components.area_occupancy.data.prior.dt_util.utcnow",
            return_value=other_dt,
        ):
            result3 = prior.time_prior
            assert result3 == 0.4  # Should return value for the other slot

        # Test accessing a slot not in cache (should return DEFAULT_TIME_PRIOR)
        missing_dt = datetime(2024, 1, 1 + missing_day, missing_slot, 0, 0, tzinfo=UTC)
        prior._cached_time_priors = test_cache.copy()
        with patch(
            "custom_components.area_occupancy.data.prior.dt_util.utcnow",
            return_value=missing_dt,
        ):
            result4 = prior.time_prior
            assert result4 == DEFAULT_TIME_PRIOR
    finally:
        dt_util.set_default_time_zone(original_tz)


def test_load_time_priors_bounds_checking(coordinator: AreaOccupancyCoordinator):
    """Test _load_time_priors applies bounds checking correctly."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

    # Test data with values outside bounds
    test_data = {
        (0, 0): 0.02,  # Below TIME_PRIOR_MIN_BOUND (0.03).
        (0, 1): 0.95,  # Above TIME_PRIOR_MAX_BOUND (0.9).
        (0, 2): 0.5,  # Within bounds.
        (1, 0): TIME_PRIOR_MIN_BOUND,  # At minimum bound.
        (1, 1): TIME_PRIOR_MAX_BOUND,  # At maximum bound.
    }

    with patch.object(
        prior.db,
        "get_all_time_priors",
        return_value=test_data.copy(),
    ):
        # Trigger _load_time_priors by accessing time_prior
        prior._load_time_priors()

        # Verify bounds are applied
        assert prior._cached_time_priors[(0, 0)] == TIME_PRIOR_MIN_BOUND
        assert prior._cached_time_priors[(0, 1)] == TIME_PRIOR_MAX_BOUND
        assert prior._cached_time_priors[(0, 2)] == 0.5
        assert prior._cached_time_priors[(1, 0)] == TIME_PRIOR_MIN_BOUND
        assert prior._cached_time_priors[(1, 1)] == TIME_PRIOR_MAX_BOUND

    # Test default fallback when slot not in database
    with patch.object(
        prior.db,
        "get_all_time_priors",
        return_value={},  # Empty dict - no data in database
    ):
        prior._load_time_priors()
        # Cache should be empty dict
        assert prior._cached_time_priors == {}
        # Accessing time_prior should return DEFAULT_TIME_PRIOR
        result = prior.time_prior
        assert result == DEFAULT_TIME_PRIOR


def test_time_prior_cache_invalidation(coordinator: AreaOccupancyCoordinator):
    """Test time_prior cache invalidation triggers reload."""
    area_name = coordinator.get_area_names()[0]
    prior = Prior(coordinator, area_name=area_name)

    current_day = prior.day_of_week
    current_slot = prior.time_slot
    slot_key = (current_day, current_slot)

    # First cache load
    cache1 = {slot_key: 0.6}
    with patch.object(
        prior.db,
        "get_all_time_priors",
        return_value=cache1.copy(),
    ) as mock_get_all:
        result1 = prior.time_prior
        assert result1 == 0.6
        assert mock_get_all.call_count == 1

    # Invalidate cache
    prior._invalidate_time_prior_cache()
    assert prior._cached_time_priors is None

    # Second cache load (should trigger reload)
    cache2 = {slot_key: 0.7}
    with patch.object(
        prior.db,
        "get_all_time_priors",
        return_value=cache2.copy(),
    ) as mock_get_all2:
        result2 = prior.time_prior
        assert result2 == 0.7
        assert mock_get_all2.call_count == 1

    # Verify set_global_prior invalidates cache
    prior._cached_time_priors = {slot_key: 0.5}
    prior.set_global_prior(0.8)
    assert prior._cached_time_priors is None

    # Verify clear_cache invalidates cache
    prior._cached_time_priors = {slot_key: 0.4}
    prior.clear_cache()
    assert prior._cached_time_priors is None


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


def test_passageway_purpose_min_prior(coordinator: AreaOccupancyCoordinator):
    """Test that passageway areas get a minimum prior floor."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)

    # Set a very low global prior (typical for passageways)
    prior.global_prior = 0.015

    # With social purpose (default), the prior should stay low
    with patch.object(
        Prior, "time_prior", new_callable=PropertyMock, return_value=None
    ):
        result_social = prior.value
    assert result_social < 0.1

    # Switch area purpose to passageway
    area._purpose = Purpose(purpose=AreaPurpose.PASSAGEWAY)
    with patch.object(
        Prior, "time_prior", new_callable=PropertyMock, return_value=None
    ):
        result_passageway = prior.value

    assert result_passageway == PURPOSE_MIN_PRIORS[AreaPurpose.PASSAGEWAY]
    assert result_passageway == 0.1


def test_purpose_min_prior_does_not_override_higher_value(
    coordinator: AreaOccupancyCoordinator,
):
    """Test that purpose min prior doesn't lower an already-higher prior."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    prior = Prior(coordinator, area_name=area_name)

    area._purpose = Purpose(purpose=AreaPurpose.PASSAGEWAY)
    prior.global_prior = 0.5

    with patch.object(
        Prior, "time_prior", new_callable=PropertyMock, return_value=None
    ):
        result = prior.value

    assert result == 0.5
