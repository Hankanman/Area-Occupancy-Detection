"""Tests for the Prior class (updated for simplified implementation)."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.const import MAX_PRIOR, MIN_PRIOR
from custom_components.area_occupancy.data.prior import Prior
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
@pytest.fixture
def mock_coordinator():
    mock = Mock()
    mock.config.sensors.motion = ["binary_sensor.motion1", "binary_sensor.motion2"]
    mock.hass = Mock()
    mock.sqlite_store = Mock()
    return mock


def test_initialization(mock_coordinator):
    prior = Prior(mock_coordinator)
    assert prior.sensor_ids == ["binary_sensor.motion1", "binary_sensor.motion2"]
    assert prior.hass == mock_coordinator.hass
    assert prior.global_prior is None
    assert prior._prior_intervals is None
    assert prior._last_updated is None


def test_value_property_clamping(mock_coordinator):
    prior = Prior(mock_coordinator)
    # Not set
    assert prior.value == MIN_PRIOR
    # Below min
    prior.global_prior = 0.01
    assert prior.value == MIN_PRIOR
    # Above max
    prior.global_prior = 2.0
    assert prior.value == MAX_PRIOR
    # In range
    prior.global_prior = 0.5
    assert prior.value == 0.5


def test_prior_intervals_property(mock_coordinator):
    prior = Prior(mock_coordinator)
    assert prior.prior_intervals is None
    # Set intervals
    now = dt_util.utcnow()
    intervals = [
        {"state": "on", "start": now, "end": now + timedelta(minutes=10)},
        {
            "state": "off",
            "start": now + timedelta(minutes=10),
            "end": now + timedelta(minutes=20),
        },
    ]
    prior._prior_intervals = intervals
    assert prior.prior_intervals == intervals


@pytest.mark.asyncio
async def test_update_sets_prior_and_intervals(mock_coordinator):
    # Mock get_historical_intervals to return intervals
    now = dt_util.utcnow()
    intervals = [
        {"state": "on", "start": now, "end": now + timedelta(minutes=10)},
        {
            "state": "on",
            "start": now + timedelta(minutes=20),
            "end": now + timedelta(minutes=30),
        },
    ]
    mock_coordinator.sqlite_store.get_historical_intervals = AsyncMock(
        return_value=intervals
    )
    prior = Prior(mock_coordinator)
    await prior.update()
    # Should set global_prior and _prior_intervals
    assert prior.global_prior is not None
    assert isinstance(prior._prior_intervals, list)
    assert prior._last_updated is not None


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
