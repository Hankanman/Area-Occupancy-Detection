"""Tests for Area Occupancy Detection probability calculations."""

import logging
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from homeassistant.const import STATE_OFF, STATE_ON

from custom_components.area_occupancy.calculate_prob import (
    ProbabilityCalculator,
    bayesian_update,
)
from custom_components.area_occupancy.const import MAX_PROBABILITY, MIN_PROBABILITY
from custom_components.area_occupancy.probabilities import Probabilities
from custom_components.area_occupancy.types import EntityType, PriorState, SensorInfo

_LOGGER = logging.getLogger(__name__)


@pytest.fixture
def mock_probabilities():
    """Create a mock probabilities provider."""
    probabilities = MagicMock(spec=Probabilities)
    probabilities.get_sensor_config.return_value = {
        "prob_given_true": 0.9,
        "prob_given_false": 0.1,
        "default_prior": 0.5,
        "weight": 1.0,
        "active_states": {"on"},
    }
    probabilities.is_entity_active.side_effect = lambda entity_id, state: state == "on"
    probabilities.get_entity_type.return_value = "motion"  # Default mock type
    probabilities.entity_types = {
        "binary_sensor.test": "motion",
        "binary_sensor.test1": "motion",
        "binary_sensor.test2": "motion",
        "binary_sensor.test3": "light",  # Example for different types
    }
    return probabilities


@pytest.fixture
def mock_prior_state() -> PriorState:
    """Create a mock PriorState object."""
    state = PriorState()
    # Add some sample learned priors if needed for specific tests
    state.update_entity_prior(
        "binary_sensor.test",
        prob_given_true=0.85,
        prob_given_false=0.15,
        prior=0.55,
        timestamp=datetime.now().isoformat(),
    )
    return state


@pytest.fixture
def calculator(mock_probabilities):
    """Create a ProbabilityCalculator instance."""
    return ProbabilityCalculator(probabilities=mock_probabilities)


# --- Tests for bayesian_update ---


def test_bayesian_update_basic():
    """Test basic Bayesian probability update."""
    # Test with standard values
    result = bayesian_update(prior=0.5, prob_given_true=0.8, prob_given_false=0.2)
    assert 0.7 < result < 0.9

    # Test with extreme values
    result = bayesian_update(
        prior=MIN_PROBABILITY,
        prob_given_true=MIN_PROBABILITY,
        prob_given_false=MIN_PROBABILITY,
    )
    assert result == MIN_PROBABILITY

    result = bayesian_update(
        prior=MAX_PROBABILITY,
        prob_given_true=MAX_PROBABILITY,
        prob_given_false=MIN_PROBABILITY,
    )
    assert result == MAX_PROBABILITY

    # Test invalid probabilities (outside 0-1 range)
    with pytest.raises(ValueError):
        bayesian_update(0.5, 1.1, 0.6)  # P(E|H) > 1
    with pytest.raises(ValueError):
        bayesian_update(0.5, -0.1, 0.6)  # P(E|H) < 0
    with pytest.raises(ValueError):
        bayesian_update(0.5, 0.8, 1.2)  # P(E|~H) > 1
    with pytest.raises(ValueError):
        bayesian_update(0.5, 0.8, -0.2)  # P(E|~H) < 0
    with pytest.raises(ValueError):
        bayesian_update(1.5, 0.8, 0.6)  # Prior > 1
    with pytest.raises(ValueError):
        bayesian_update(-0.5, 0.8, 0.6)  # Prior < 0

    # Test zero denominator (should return MIN_PROBABILITY)
    # P(E|H)*P(H) + P(E|~H)*(1-P(H)) = 0
    # Only when both prob_given_true and prob_given_false are 0, denominator is 0
    result = bayesian_update(0.0, 0.0, 0.0)
    assert result == MIN_PROBABILITY
    result = bayesian_update(1.0, 0.0, 0.0)
    assert result == MAX_PROBABILITY
    result = bayesian_update(0.5, 0.0, 0.0)
    assert result == 0.5


def test_bayesian_update_edge_cases():
    """Test edge cases for probability update."""
    # Test with zero denominator case - should return MIN_PROBABILITY
    result = bayesian_update(0.0, 0.0, 0.0)
    assert result == MIN_PROBABILITY

    # Test with values exceeding bounds (should raise ValueError)
    with pytest.raises(ValueError):
        bayesian_update(1.5, 0.8, 0.2)
    with pytest.raises(ValueError):
        bayesian_update(-0.5, 0.8, 0.2)
    with pytest.raises(ValueError):
        bayesian_update(0.5, 1.8, 0.2)
    with pytest.raises(ValueError):
        bayesian_update(0.5, 0.8, -0.2)


# --- Tests for ProbabilityCalculator methods ---


def test_calculate_sensor_probability(calculator, mock_prior_state):
    """Test probability calculation for a single sensor."""
    entity_id = "binary_sensor.test"
    state_on: SensorInfo = {"state": "on", "availability": True, "last_changed": ""}
    state_off: SensorInfo = {"state": "off", "availability": True, "last_changed": ""}
    state_unavail: SensorInfo = {
        "state": None,
        "availability": False,
        "last_changed": "",
    }

    # Test normal case (active state 'on', using learned priors)
    calc_result = calculator._calculate_sensor_probability(
        entity_id, state_on, mock_prior_state
    )
    assert calc_result.is_active is True
    assert calc_result.weighted_probability > MIN_PROBABILITY
    assert calc_result.details["probability"] == pytest.approx(
        bayesian_update(0.55, 0.85, 0.15)
    )  # Check against learned prior
    assert calc_result.details["weight"] == 1.0

    # Test inactive state ('off')
    calc_result_off = calculator._calculate_sensor_probability(
        entity_id, state_off, mock_prior_state
    )
    assert calc_result_off.is_active is False
    assert calc_result_off.weighted_probability == 0.0

    # Test unavailable sensor
    calc_result_unavail = calculator._calculate_sensor_probability(
        entity_id, state_unavail, mock_prior_state
    )
    assert calc_result_unavail.is_active is False
    assert calc_result_unavail.weighted_probability == 0.0

    # Test with missing sensor config
    calculator.probabilities.get_sensor_config.return_value = None
    calc_result_no_config = calculator._calculate_sensor_probability(
        "binary_sensor.missing", state_on, mock_prior_state
    )
    assert calc_result_no_config.is_active is False
    assert calc_result_no_config.weighted_probability == 0.0


def test_calculate_complementary_probability(calculator, mock_prior_state):
    """Test complementary probability calculation with multiple sensors."""
    sensor_states: dict[str, SensorInfo] = {
        "binary_sensor.test1": {
            "state": STATE_ON,
            "availability": True,
            "last_changed": "",
        },
        "binary_sensor.test2": {
            "state": STATE_ON,
            "availability": True,
            "last_changed": "",
        },
    }
    sensor_probs = {}

    # Mock entity types
    calculator.probabilities.entity_types = {
        "binary_sensor.test1": EntityType.MOTION,
        "binary_sensor.test2": EntityType.MOTION,
    }

    # Mock configs for the sensors
    calculator.probabilities.get_sensor_config.side_effect = lambda eid: {
        "binary_sensor.test1": {
            "prob_given_true": 0.9,
            "prob_given_false": 0.1,
            "default_prior": 0.5,
            "weight": 0.8,
            "active_states": {STATE_ON},
        },
        "binary_sensor.test2": {
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "default_prior": 0.6,
            "weight": 0.7,
            "active_states": {STATE_ON},
        },
    }.get(eid)

    result = calculator._calculate_complementary_probability(
        sensor_states, sensor_probs, mock_prior_state
    )

    assert MIN_PROBABILITY <= result <= MAX_PROBABILITY
    assert len(sensor_probs) == 2
    assert "binary_sensor.test1" in sensor_probs
    assert "binary_sensor.test2" in sensor_probs

    # Manual calculation check (for illustration)
    prob1 = bayesian_update(0.5, 0.9, 0.1)
    w_prob1 = prob1 * 0.8
    comp1 = 1.0 - w_prob1

    prob2 = bayesian_update(0.6, 0.8, 0.2)
    w_prob2 = prob2 * 0.7
    comp2 = 1.0 - w_prob2

    expected_complementary = comp1 * comp2
    expected_final_prob = 1.0 - expected_complementary

    assert result == pytest.approx(
        max(MIN_PROBABILITY, min(expected_final_prob, MAX_PROBABILITY))
    )


def test_calculate_occupancy_probability(calculator, mock_prior_state):
    """Test the main calculation logic."""
    current_states: dict[str, SensorInfo] = {
        "binary_sensor.test1": {
            "state": STATE_ON,
            "availability": True,
            "last_changed": "",
        },
        "binary_sensor.test2": {
            "state": STATE_OFF,
            "availability": True,
            "last_changed": "",
        },
    }

    # Mock entity types
    calculator.probabilities.entity_types = {
        "binary_sensor.test1": EntityType.MOTION,
        "binary_sensor.test2": EntityType.MOTION,
    }

    # Mock configs
    calculator.probabilities.get_sensor_config.side_effect = lambda eid: {
        "binary_sensor.test1": {
            "prob_given_true": 0.9,
            "prob_given_false": 0.1,
            "default_prior": 0.5,
            "weight": 0.8,
            "active_states": {STATE_ON},
        },
        "binary_sensor.test2": {
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "default_prior": 0.6,
            "weight": 0.7,
            "active_states": {STATE_ON},
        },
    }.get(eid)

    result = calculator.calculate_occupancy_probability(
        current_states, mock_prior_state
    )

    assert MIN_PROBABILITY <= result.calculated_probability <= MAX_PROBABILITY
    assert MIN_PROBABILITY <= result.prior_probability <= MAX_PROBABILITY
    assert "binary_sensor.test1" in result.sensor_probabilities
    assert "binary_sensor.test2" not in result.sensor_probabilities

    # Check prior calculation (only test1 is active)
    assert result.prior_probability == pytest.approx(0.55)

    # Remove mock_prior_state.get_entity_prior usage, use direct dict access
    # The following block is commented out because get_entity_prior does not exist:
    # mock_prior_state = MagicMock(spec=PriorState)
    # mock_prior_state.entity_priors = {}  # Use defaults initially
    # mock_prior_state.type_priors = {
    #     "motion": {"prior": 0.55, "last_updated": "", "analysis_period": 7},
    #     "door": {"prior": 0.25, "last_updated": "", "analysis_period": 7},
    # }
    # mock_prior_state.get_entity_prior.side_effect = lambda entity_id, default: default
    # calculator.prior_state = mock_prior_state
    # mock_probabilities.get_sensor_config.return_value = {
    #     "default_prior": 0.5,
    #     "prob_given_true": 0.9,
    #     "prob_given_false": 0.1,
    #     "weight": 1.0,  # Assume weight 1 for simplicity unless overridden
    # }
    # mock_probabilities.entity_types = {"test1": "motion", "test2": "door"}
    # --- Test Case 1: Simple case with one active sensor ---
    # now = dt_util.utcnow()
    # current_states: dict[str, SensorInfo] = {
    #     "test1": {
    #         "state": STATE_ON,
    #         "last_changed": now.isoformat(),
    #         "availability": True,
    #     },
    #     "test2": {
    #         "state": STATE_OFF,
    #         "last_changed": now.isoformat(),
    #         "availability": True,
    #     },
    # }
    # result = calculator.calculate_occupancy_probability(current_states)
    # expected_prob_test1 = 0.495 / 0.54
    # assert isinstance(result, OccupancyCalculationResult)
    # assert result.calculated_probability == pytest.approx(expected_prob_test1)
    # assert result.prior_probability == pytest.approx(0.55)  # Prior from motion type
    # assert "test1" in result.sensor_probabilities
    # assert result.sensor_probabilities["test1"]["probability"] == pytest.approx(expected_prob_test1)
    # assert calculator.probabilities == mock_probabilities
    # assert calculator.logger == _LOGGER


# Remove old, irrelevant tests
# test_update_probability_basic, test_update_probability_edge_cases
# test_calculate_base_probability, test_perform_calculation_logic
# test_decay_handling, test_threshold_behavior
# test_calculate_composite_probability_*
