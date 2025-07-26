"""Tests for the Prior class (updated for simplified implementation)."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.const import MAX_PRIOR, MIN_PRIOR
from custom_components.area_occupancy.data.prior import PRIOR_FACTOR, Prior
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
@pytest.fixture
def mock_coordinator():
    mock = Mock()
    mock.config.sensors.motion = ["binary_sensor.motion1", "binary_sensor.motion2"]
    mock.config.wasp_in_box.enabled = False
    mock.hass = Mock()
    mock.sqlite_store = Mock()
    mock.wasp_entity_id = "binary_sensor.wasp"
    mock.occupancy_entity_id = "binary_sensor.occupancy"
    return mock


def test_initialization(mock_coordinator):
    prior = Prior(mock_coordinator)
    assert prior.sensor_ids == ["binary_sensor.motion1", "binary_sensor.motion2"]
    assert prior.hass == mock_coordinator.hass
    assert prior.global_prior is None
    assert prior.primary_sensors_prior is None
    assert prior.occupancy_prior is None
    assert prior._prior_intervals is None
    assert prior._last_updated is None


def test_value_property_clamping(mock_coordinator):
    prior = Prior(mock_coordinator)
    # Not set
    assert prior.value == MIN_PRIOR

    # Test with PRIOR_FACTOR multiplication
    # Below min after factor
    prior.global_prior = 0.005  # 0.005 * 1.2 = 0.006, still below MIN_PRIOR
    assert prior.value == MIN_PRIOR

    # Above max after factor
    prior.global_prior = 0.9  # 0.9 * 1.2 = 1.08, above MAX_PRIOR
    assert prior.value == MAX_PRIOR

    # In range after factor
    prior.global_prior = 0.5  # 0.5 * 1.2 = 0.6
    expected = min(max(0.5 * PRIOR_FACTOR, MIN_PRIOR), MAX_PRIOR)
    assert prior.value == expected


def test_prior_intervals_property(mock_coordinator):
    prior = Prior(mock_coordinator)
    assert prior.prior_intervals is None
    # Set intervals
    now = dt_util.utcnow()
    intervals = [
        {
            "state": "on",
            "start": now,
            "end": now + timedelta(minutes=10),
            "entity_id": "test.entity",
        },
        {
            "state": "off",
            "start": now + timedelta(minutes=10),
            "end": now + timedelta(minutes=20),
            "entity_id": "test.entity",
        },
    ]
    prior._prior_intervals = intervals
    assert prior.prior_intervals == intervals


@pytest.mark.asyncio
async def test_update_with_wasp_disabled(mock_coordinator):
    # Mock get_historical_intervals to return different intervals for different entities
    now = dt_util.utcnow()

    # Primary sensors intervals (higher occupancy)
    primary_intervals = [
        {
            "state": "on",
            "start": now,
            "end": now + timedelta(minutes=30),
            "entity_id": "binary_sensor.motion1",
        },
    ]

    # Occupancy intervals (lower occupancy)
    occupancy_intervals = [
        {
            "state": "on",
            "start": now,
            "end": now + timedelta(minutes=10),
            "entity_id": "binary_sensor.occupancy",
        },
    ]

    def mock_get_intervals(entity_id, start_time, end_time, state_filter=None):
        if entity_id == "binary_sensor.occupancy":
            return occupancy_intervals
        return primary_intervals

    mock_coordinator.sqlite_store.get_historical_intervals = AsyncMock(
        side_effect=mock_get_intervals
    )
    mock_coordinator.config.wasp_in_box.enabled = False

    prior = Prior(mock_coordinator)
    await prior.update()

    # Should set all prior values
    assert prior.global_prior is not None
    assert prior.primary_sensors_prior is not None
    assert prior.occupancy_prior is not None
    assert isinstance(prior._prior_intervals, list)
    assert prior._last_updated is not None

    # Primary sensors should have higher prior (30 min out of total)
    # Occupancy should have lower prior (10 min out of total)
    assert prior.primary_sensors_prior > prior.occupancy_prior

    # Global prior should be the higher one (primary sensors)
    assert prior.global_prior == prior.primary_sensors_prior

    # Verify call count: motion1, motion2, occupancy (wasp disabled)
    calls = mock_coordinator.sqlite_store.get_historical_intervals.call_args_list
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_update_with_wasp_enabled(mock_coordinator):
    # Mock get_historical_intervals
    now = dt_util.utcnow()
    intervals = [
        {
            "state": "on",
            "start": now,
            "end": now + timedelta(minutes=10),
            "entity_id": "binary_sensor.motion1",
        },
    ]

    mock_coordinator.sqlite_store.get_historical_intervals = AsyncMock(
        return_value=intervals
    )
    mock_coordinator.config.wasp_in_box.enabled = True

    prior = Prior(mock_coordinator)
    await prior.update()

    # Verify that get_historical_intervals was called with wasp entity included
    calls = mock_coordinator.sqlite_store.get_historical_intervals.call_args_list

    # Should be called 4 times: motion1, motion2, wasp (for primary sensors), and occupancy
    assert len(calls) == 4

    # Check if wasp entity was called
    called_entities = [
        call[0][0] for call in calls
    ]  # First positional argument of each call
    assert "binary_sensor.wasp" in called_entities
    assert "binary_sensor.motion1" in called_entities
    assert "binary_sensor.motion2" in called_entities
    assert "binary_sensor.occupancy" in called_entities


@pytest.mark.asyncio
async def test_update_occupancy_higher_than_primary(mock_coordinator):
    # Test case where occupancy prior is higher than primary sensors
    now = dt_util.utcnow()

    # Primary sensors intervals (lower occupancy)
    primary_intervals = [
        {
            "state": "on",
            "start": now,
            "end": now + timedelta(minutes=5),
            "entity_id": "binary_sensor.motion1",
        },
    ]

    # Occupancy intervals (higher occupancy)
    occupancy_intervals = [
        {
            "state": "on",
            "start": now,
            "end": now + timedelta(minutes=30),
            "entity_id": "binary_sensor.occupancy",
        },
    ]

    def mock_get_intervals(entity_id, start_time, end_time, state_filter=None):
        if entity_id == "binary_sensor.occupancy":
            return occupancy_intervals
        return primary_intervals

    mock_coordinator.sqlite_store.get_historical_intervals = AsyncMock(
        side_effect=mock_get_intervals
    )

    prior = Prior(mock_coordinator)
    await prior.update()

    # Occupancy should have higher prior
    assert prior.occupancy_prior > prior.primary_sensors_prior

    # Global prior should be the higher one (occupancy)
    assert prior.global_prior == prior.occupancy_prior

    # Verify call count: motion1, motion2, occupancy (wasp disabled by default)
    calls = mock_coordinator.sqlite_store.get_historical_intervals.call_args_list
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_update_handles_exception_and_sets_min_prior(mock_coordinator):
    # Simulate error in get_historical_intervals
    mock_coordinator.sqlite_store.get_historical_intervals = AsyncMock(
        side_effect=Exception("fail")
    )
    prior = Prior(mock_coordinator)
    await prior.update()
    assert prior.global_prior == MIN_PRIOR
    assert prior._last_updated is not None


@pytest.mark.asyncio
async def test_calculate_prior_method(mock_coordinator):
    # Test the _calculate_prior method directly
    now = dt_util.utcnow()
    intervals = [
        {
            "state": "on",
            "start": now,
            "end": now + timedelta(minutes=10),
            "entity_id": "test.entity",
        },
        {
            "state": "on",
            "start": now + timedelta(minutes=20),
            "end": now + timedelta(minutes=30),
            "entity_id": "test.entity",
        },
    ]

    mock_coordinator.sqlite_store.get_historical_intervals = AsyncMock(
        return_value=intervals
    )

    prior = Prior(mock_coordinator)
    entity_ids = ["test.entity"]

    prior_value, merged_intervals = await prior._calculate_prior(entity_ids)

    # Should return a tuple
    assert isinstance(prior_value, float)
    assert isinstance(merged_intervals, list)

    # Verify get_historical_intervals was called with state_filter
    mock_coordinator.sqlite_store.get_historical_intervals.assert_called_once()
    call_kwargs = mock_coordinator.sqlite_store.get_historical_intervals.call_args[1]
    assert call_kwargs.get("state_filter") == "on"


def test_to_dict_and_from_dict(mock_coordinator):
    prior = Prior(mock_coordinator)
    now = dt_util.utcnow()
    prior.global_prior = 0.42
    prior._last_updated = now
    d = prior.to_dict()
    assert d["value"] == 0.42
    assert d["last_updated"] == now.isoformat()
    # from_dict
    restored = Prior.from_dict(d, mock_coordinator)
    assert restored.global_prior == 0.42
    assert restored._last_updated == now


def test_prior_factor_constant():
    """Test that PRIOR_FACTOR is properly defined."""
    assert PRIOR_FACTOR == 1.2
