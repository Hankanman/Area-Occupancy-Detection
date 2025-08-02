"""Tests for the Prior class (updated for simplified implementation)."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

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

    mock_coordinator.db.get_historical_intervals = AsyncMock(
        side_effect=mock_get_intervals
    )

    # Mock the database session properly
    mock_session = Mock()
    mock_coordinator.db.get_session = Mock(return_value=mock_session)
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=None)
    mock_session.query.return_value.filter.return_value.scalar.return_value = (
        600  # 10 minutes in seconds
    )

    prior = Prior(mock_coordinator)
    await prior.update()

    assert prior.global_prior is not None


@pytest.mark.asyncio
async def test_update_handles_exception_and_sets_min_prior(mock_coordinator):
    # Simulate error in get_historical_intervals
    mock_coordinator.db.get_historical_intervals = AsyncMock(
        side_effect=Exception("fail")
    )
    prior = Prior(mock_coordinator)
    await prior.update()
    assert prior.global_prior == MIN_PRIOR
    assert prior._last_updated is not None


def test_calculate_prior_method(mock_coordinator):
    # Test the calculate_area_prior method directly
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

    # Mock the db.get_session method to return a proper context manager
    mock_session = Mock()
    mock_coordinator.db.get_session = Mock(return_value=mock_session)
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=None)
    mock_session.query.return_value.filter.return_value.scalar.return_value = (
        600  # 10 minutes in seconds
    )

    # Mock get_time_bounds to return proper datetime objects
    mock_coordinator.db.get_historical_intervals = AsyncMock(return_value=intervals)

    prior = Prior(mock_coordinator)
    entity_ids = ["test.entity"]

    # Mock the get_time_bounds method to return proper datetime objects
    with patch.object(
        prior, "get_time_bounds", return_value=(now, now + timedelta(hours=1))
    ):
        prior_value = prior.calculate_area_prior(entity_ids)
        assert isinstance(prior_value, float)
        assert 0.0 <= prior_value <= 1.0


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


def test_prior_factor_constant():
    """Test that PRIOR_FACTOR is properly defined."""
    assert PRIOR_FACTOR == 1.1
