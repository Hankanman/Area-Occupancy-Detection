"""Unit tests for the PriorCalculator in calculate_prior.py.

These tests verify the correct calculation of prior probabilities for Home Assistant entities
based on historical state data, as well as fallback/default handling and error cases.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError

from custom_components.area_occupancy.calculate_prior import (  # noqa: TID252
    PriorCalculator,
    TimeInterval,
)
from custom_components.area_occupancy.const import (  # noqa: TID252
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.types import PriorData  # noqa: TID252

# Note: Using fixtures from conftest.py:
# - mock_hass
# - mock_probabilities
# - mock_sensor_inputs


@pytest.mark.asyncio
async def test_calculate_prior_primary_sensor_valid(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test that calculate_prior returns correct prior data for a valid primary sensor with a simple ON/OFF state history (30 min ON, 30 min OFF)."""
    # Arrange
    entity_id = "binary_sensor.motion_living_room"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    mock_sensor_inputs.primary_sensor = entity_id
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True
    # Fake state history: 30 min ON, 30 min OFF
    states = [
        State(entity_id, STATE_ON, last_changed=start),
        State(entity_id, STATE_OFF, last_changed=start + timedelta(minutes=30)),
    ]
    with patch(
        "custom_components.area_occupancy.calculate_prior.PriorCalculator._get_states_from_recorder",
        new=AsyncMock(return_value=states),
    ):
        calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
        result = await calc.calculate_prior(entity_id, start, end)
    # Assert
    assert isinstance(result, PriorData)
    assert result.prob_given_true == 0.9
    assert result.prob_given_false == 0.1
    assert abs(result.prior - 0.5) < 0.01  # 30min/60min


@pytest.mark.asyncio
async def test_calculate_prior_invalid_entity_id(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test that calculate_prior returns None for an invalid entity_id."""
    # Arrange
    entity_id = "invalid_entity"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    mock_sensor_inputs.is_valid_entity_id = lambda eid: False
    calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
    # Act
    result = await calc.calculate_prior(entity_id, start, end)
    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_calculate_prior_invalid_time_range(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test that calculate_prior returns None if the time range is invalid (start > end)."""
    # Arrange
    entity_id = "binary_sensor.motion_living_room"
    start = datetime.now()
    end = start - timedelta(hours=1)
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True
    calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
    # Act
    result = await calc.calculate_prior(entity_id, start, end)
    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_calculate_prior_recorder_error_returns_fallback(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test that calculate_prior returns fallback/default values if the recorder raises an error."""
    # Arrange
    entity_id = "binary_sensor.motion_living_room"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    mock_sensor_inputs.primary_sensor = entity_id
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True
    with patch(
        "custom_components.area_occupancy.calculate_prior.PriorCalculator._get_states_from_recorder",
        new=AsyncMock(side_effect=HomeAssistantError("recorder error")),
    ):
        calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
        result = await calc.calculate_prior(entity_id, start, end)
    # Assert
    assert isinstance(result, PriorData)
    assert result.prob_given_true == DEFAULT_PROB_GIVEN_TRUE
    assert result.prob_given_false == DEFAULT_PROB_GIVEN_FALSE
    assert result.prior == 0.5


@pytest.mark.asyncio
async def test_calculate_prior_non_primary_sensor(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test calculation of prior probabilities for a non-primary sensor."""
    # Arrange
    primary_sensor = "binary_sensor.motion_living_room"
    entity_id = "binary_sensor.door_sensor"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()

    mock_sensor_inputs.primary_sensor = primary_sensor
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True

    # Create test states with overlapping active periods
    primary_states = [
        State(primary_sensor, STATE_ON, last_changed=start),
        State(primary_sensor, STATE_OFF, last_changed=start + timedelta(minutes=30)),
    ]

    entity_states = [
        State(entity_id, STATE_ON, last_changed=start),
        State(entity_id, STATE_OFF, last_changed=start + timedelta(minutes=15)),
    ]

    with patch(
        "custom_components.area_occupancy.calculate_prior.PriorCalculator._get_states_from_recorder",
        new=AsyncMock(side_effect=[primary_states, entity_states]),
    ):
        calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
        result = await calc.calculate_prior(entity_id, start, end)

    # Assert
    assert isinstance(result, PriorData)
    if (
        result.prob_given_true is not None
        and result.prob_given_false is not None
        and result.prior is not None
    ):
        assert MIN_PROBABILITY <= float(result.prob_given_true) <= MAX_PROBABILITY
        assert MIN_PROBABILITY <= float(result.prob_given_false) <= MAX_PROBABILITY
        assert MIN_PROBABILITY <= float(result.prior) <= MAX_PROBABILITY


@pytest.mark.asyncio
async def test_calculate_prior_no_valid_intervals(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test handling of cases where no valid intervals are found."""
    # Arrange
    entity_id = "binary_sensor.test_sensor"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True

    # Mock empty state lists
    with patch(
        "custom_components.area_occupancy.calculate_prior.PriorCalculator._get_states_from_recorder",
        new=AsyncMock(return_value=[]),
    ):
        calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
        result = await calc.calculate_prior(entity_id, start, end)

    # Assert
    assert isinstance(result, PriorData)
    assert result.prob_given_true == DEFAULT_PROB_GIVEN_TRUE
    assert result.prob_given_false == DEFAULT_PROB_GIVEN_FALSE
    assert result.prior == 0.5


@pytest.mark.asyncio
async def test_calculate_conditional_probability_with_intervals(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test calculation of conditional probabilities using intervals."""
    # Arrange
    calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
    entity_id = "binary_sensor.test_sensor"
    start_time = datetime.now() - timedelta(hours=1)

    # Create test intervals with proper TimeInterval type
    entity_intervals: list[TimeInterval] = [
        {
            "start": start_time,
            "end": start_time + timedelta(minutes=30),
            "state": STATE_ON,
        },
        {
            "start": start_time + timedelta(minutes=30),
            "end": start_time + timedelta(minutes=60),
            "state": STATE_OFF,
        },
    ]

    motion_intervals_by_sensor: dict[str, list[TimeInterval]] = {
        mock_sensor_inputs.primary_sensor: [
            {
                "start": start_time,
                "end": start_time + timedelta(minutes=45),
                "state": STATE_ON,
            },
            {
                "start": start_time + timedelta(minutes=45),
                "end": start_time + timedelta(minutes=60),
                "state": STATE_OFF,
            },
        ]
    }

    # Test calculation for both ON and OFF states
    prob_given_true = calc._calculate_conditional_probability_with_intervals(
        entity_id, entity_intervals, motion_intervals_by_sensor, STATE_ON
    )
    prob_given_false = calc._calculate_conditional_probability_with_intervals(
        entity_id, entity_intervals, motion_intervals_by_sensor, STATE_OFF
    )

    # Assert - ensure probabilities are not None and cast to float for comparison
    assert prob_given_true is not None
    assert prob_given_false is not None
    if prob_given_true is not None and prob_given_false is not None:
        prob_given_true_float = float(prob_given_true)
        prob_given_false_float = float(prob_given_false)
        assert MIN_PROBABILITY <= prob_given_true_float <= MAX_PROBABILITY
        assert MIN_PROBABILITY <= prob_given_false_float <= MAX_PROBABILITY


@pytest.mark.asyncio
async def test_calculate_prior_with_empty_states(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test handling of empty state lists."""
    # Arrange
    entity_id = "binary_sensor.test_sensor"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True

    # Mock states with empty lists and None values
    with patch(
        "custom_components.area_occupancy.calculate_prior.PriorCalculator._get_states_from_recorder",
        new=AsyncMock(return_value=None),
    ):
        calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
        result = await calc.calculate_prior(entity_id, start, end)

    # Assert
    assert isinstance(result, PriorData)
    assert result.prob_given_true == DEFAULT_PROB_GIVEN_TRUE
    assert result.prob_given_false == DEFAULT_PROB_GIVEN_FALSE
    assert result.prior == 0.5


@pytest.mark.asyncio
async def test_calculate_prior_with_invalid_states(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test handling of invalid state objects."""
    # Arrange
    entity_id = "binary_sensor.test_sensor"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True

    # Create invalid state objects (dicts instead of State objects)
    invalid_states = [
        {"entity_id": entity_id, "state": STATE_ON},
        {"entity_id": entity_id, "state": STATE_OFF},
    ]

    with patch(
        "custom_components.area_occupancy.calculate_prior.PriorCalculator._get_states_from_recorder",
        new=AsyncMock(return_value=invalid_states),
    ):
        calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
        result = await calc.calculate_prior(entity_id, start, end)

    # Assert
    assert isinstance(result, PriorData)
    assert result.prob_given_true == DEFAULT_PROB_GIVEN_TRUE
    assert result.prob_given_false == DEFAULT_PROB_GIVEN_FALSE
    assert result.prior == 0.5


@pytest.mark.asyncio
async def test_calculate_prior_with_insufficient_data(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test handling of insufficient historical data."""
    entity_id = "binary_sensor.test_sensor"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True

    # Mock states with insufficient data points
    states = [
        State(entity_id, STATE_ON, last_changed=start),
    ]

    with patch(
        "custom_components.area_occupancy.calculate_prior.PriorCalculator._get_states_from_recorder",
        new=AsyncMock(return_value=states),
    ):
        calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
        result = await calc.calculate_prior(entity_id, start, end)

    assert isinstance(result, PriorData)
    assert result.prob_given_true == MAX_PROBABILITY
    assert result.prob_given_false == DEFAULT_PROB_GIVEN_FALSE
    assert result.prior == MAX_PROBABILITY


@pytest.mark.asyncio
async def test_calculate_prior_with_invalid_state_values(
    mock_hass,
    mock_probabilities,
    mock_sensor_inputs,
):
    """Test handling of invalid state values in historical data."""
    entity_id = "binary_sensor.test_sensor"
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    mock_sensor_inputs.is_valid_entity_id = lambda eid: True

    # Mock states with invalid values
    states = [
        State(entity_id, "invalid_state", last_changed=start),
        State(entity_id, STATE_UNAVAILABLE, last_changed=start + timedelta(minutes=30)),
    ]

    with patch(
        "custom_components.area_occupancy.calculate_prior.PriorCalculator._get_states_from_recorder",
        new=AsyncMock(return_value=states),
    ):
        calc = PriorCalculator(mock_hass, mock_probabilities, mock_sensor_inputs)
        result = await calc.calculate_prior(entity_id, start, end)

    assert isinstance(result, PriorData)
    assert result.prob_given_true == DEFAULT_PROB_GIVEN_TRUE
    assert result.prob_given_false == DEFAULT_PROB_GIVEN_FALSE
    assert result.prior == MIN_PROBABILITY
