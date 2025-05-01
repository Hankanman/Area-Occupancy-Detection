"""Unit tests for the probability calculation functions in calculate_prob.py."""

import logging
from datetime import datetime

import pytest
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE

from custom_components.area_occupancy.calculate_prob import (
    ProbabilityCalculator,
    bayesian_update,  # noqa: TID252
)
from custom_components.area_occupancy.const import (  # noqa: TID252
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.types import (  # noqa: TID252
    PriorState,
    SensorInfo,
)

_LOGGER = logging.getLogger(__name__)

# Note: Using fixtures from conftest.py:
# - mock_probabilities
# - mock_prior_state
# - default_config


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


def test_bayesian_update_invalid_inputs():
    """Test Bayesian update with invalid inputs."""
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


def test_bayesian_update_zero_denominator():
    """Test Bayesian update with zero denominator cases."""
    # Test zero denominator (should return MIN_PROBABILITY)
    # P(E|H)*P(H) + P(E|~H)*(1-P(H)) = 0
    # Only when both prob_given_true and prob_given_false are 0, denominator is 0
    result = bayesian_update(0.0, 0.0, 0.0)
    assert result == MIN_PROBABILITY
    result = bayesian_update(1.0, 0.0, 0.0)
    assert result == MAX_PROBABILITY
    result = bayesian_update(0.5, 0.0, 0.0)
    assert result == 0.5


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
    calc_result = calculator._calculate_sensor_probability(  # noqa: SLF001
        entity_id, state_on, mock_prior_state
    )
    assert calc_result.is_active is True
    assert calc_result.weighted_probability > MIN_PROBABILITY
    assert calc_result.details["probability"] == pytest.approx(
        bayesian_update(0.55, 0.85, 0.15)
    )  # Check against learned prior
    assert calc_result.details["weight"] == 1.0

    # Test inactive state ('off')
    calc_result_off = calculator._calculate_sensor_probability(  # noqa: SLF001
        entity_id, state_off, mock_prior_state
    )
    assert calc_result_off.is_active is False
    assert calc_result_off.weighted_probability == 0.0

    # Test unavailable sensor
    calc_result_unavail = calculator._calculate_sensor_probability(  # noqa: SLF001
        entity_id, state_unavail, mock_prior_state
    )
    assert calc_result_unavail.is_active is False
    assert calc_result_unavail.weighted_probability == 0.0

    # Test with missing sensor config
    calculator.probabilities.get_sensor_config.return_value = None
    calc_result_no_config = calculator._calculate_sensor_probability(  # noqa: SLF001
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

    result = calculator._calculate_complementary_probability(  # noqa: SLF001
        sensor_states, sensor_probs, mock_prior_state
    )

    assert MIN_PROBABILITY <= result <= MAX_PROBABILITY
    assert len(sensor_probs) == 2
    assert "binary_sensor.test1" in sensor_probs
    assert "binary_sensor.test2" in sensor_probs

    # Manual calculation check using actual default values
    # For both sensors:
    # - prob_given_true = 0.5 (DEFAULT_PROB_GIVEN_TRUE)
    # - prob_given_false = 0.1 (DEFAULT_PROB_GIVEN_FALSE)
    # - prior = 0.5 (from mock)
    # - weight = 1.0 (from mock)

    # For sensor 1
    prob1 = bayesian_update(0.5, 0.5, 0.1)  # P(H|E) = 0.833...
    w_prob1 = prob1 * 1.0  # weight is 1.0
    comp1 = 1.0 - w_prob1  # 0.166...

    # For sensor 2 (same calculation since identical setup)
    prob2 = bayesian_update(0.5, 0.5, 0.1)  # P(H|E) = 0.833...
    w_prob2 = prob2 * 1.0  # weight is 1.0
    comp2 = 1.0 - w_prob2  # 0.166...

    # Calculate final probability
    expected_complementary = comp1 * comp2  # ~0.0277...
    expected_final_prob = 1.0 - expected_complementary  # ~0.9722...

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

    result = calculator.calculate_occupancy_probability(
        current_states, mock_prior_state
    )

    assert MIN_PROBABILITY <= result.calculated_probability <= MAX_PROBABILITY
    assert MIN_PROBABILITY <= result.prior_probability <= MAX_PROBABILITY
    assert "binary_sensor.test1" in result.sensor_probabilities
    assert "binary_sensor.test2" not in result.sensor_probabilities

    # Check prior calculation (only test1 is active)
    assert result.prior_probability == pytest.approx(0.5)  # Updated expected value


def test_probability_calculator_with_single_sensor(calculator, default_config):
    """Test ProbabilityCalculator with a single sensor."""
    # Create test data
    current_states = {
        "binary_sensor.motion": SensorInfo(
            state=STATE_ON,
            availability=True,
            last_changed=datetime.now().isoformat(),
        )
    }
    prior_state = PriorState()
    timestamp = datetime.now().isoformat()
    prior_state.update_entity_prior(
        entity_id="binary_sensor.motion",
        prob_given_true=0.8,
        prob_given_false=0.2,
        prior=0.5,
        timestamp=timestamp,
    )

    # Calculate probability using calculator fixture
    result = calculator.calculate_occupancy_probability(current_states, prior_state)

    # Verify results
    assert MIN_PROBABILITY <= result.calculated_probability <= MAX_PROBABILITY
    assert MIN_PROBABILITY <= result.prior_probability <= MAX_PROBABILITY
    assert "binary_sensor.motion" in result.sensor_probabilities


def test_probability_calculator_with_multiple_sensors(calculator, default_config):
    """Test ProbabilityCalculator with multiple sensors."""
    # Create test data
    current_states = {
        "binary_sensor.motion": SensorInfo(
            state=STATE_ON,
            availability=True,
            last_changed=datetime.now().isoformat(),
        ),
        "binary_sensor.door": SensorInfo(
            state=STATE_ON,
            availability=True,
            last_changed=datetime.now().isoformat(),
        ),
    }
    prior_state = PriorState()
    timestamp = datetime.now().isoformat()
    prior_state.update_entity_prior(
        entity_id="binary_sensor.motion",
        prob_given_true=0.8,
        prob_given_false=0.2,
        prior=0.5,
        timestamp=timestamp,
    )
    prior_state.update_entity_prior(
        entity_id="binary_sensor.door",
        prob_given_true=0.7,
        prob_given_false=0.3,
        prior=0.4,
        timestamp=timestamp,
    )

    # Calculate probability using calculator fixture
    result = calculator.calculate_occupancy_probability(current_states, prior_state)

    # Verify results
    assert MIN_PROBABILITY <= result.calculated_probability <= MAX_PROBABILITY
    assert MIN_PROBABILITY <= result.prior_probability <= MAX_PROBABILITY
    assert "binary_sensor.motion" in result.sensor_probabilities
    assert "binary_sensor.door" in result.sensor_probabilities


def test_probability_calculator_with_unavailable_sensor(calculator, default_config):
    """Test ProbabilityCalculator with unavailable sensor."""
    # Create test data
    current_states = {
        "binary_sensor.motion": SensorInfo(
            state=STATE_ON,
            availability=True,
            last_changed=datetime.now().isoformat(),
        ),
        "binary_sensor.door": SensorInfo(
            state=STATE_UNAVAILABLE,
            availability=False,
            last_changed=datetime.now().isoformat(),
        ),
    }
    prior_state = PriorState()
    timestamp = datetime.now().isoformat()
    prior_state.update_entity_prior(
        entity_id="binary_sensor.motion",
        prob_given_true=0.8,
        prob_given_false=0.2,
        prior=0.5,
        timestamp=timestamp,
    )
    prior_state.update_entity_prior(
        entity_id="binary_sensor.door",
        prob_given_true=0.7,
        prob_given_false=0.3,
        prior=0.4,
        timestamp=timestamp,
    )

    # Calculate probability using calculator fixture
    result = calculator.calculate_occupancy_probability(current_states, prior_state)

    # Verify results
    assert MIN_PROBABILITY <= result.calculated_probability <= MAX_PROBABILITY
    assert MIN_PROBABILITY <= result.prior_probability <= MAX_PROBABILITY
    assert "binary_sensor.motion" in result.sensor_probabilities
    assert "binary_sensor.door" not in result.sensor_probabilities


def test_probability_calculator_with_invalid_probabilities(calculator, default_config):
    """Test ProbabilityCalculator with invalid probability values."""
    # Create test data with invalid probabilities
    current_states = {
        "binary_sensor.motion": SensorInfo(
            state=STATE_ON,
            availability=True,
            last_changed=datetime.now().isoformat(),
        )
    }
    prior_state = PriorState()
    timestamp = datetime.now().isoformat()
    prior_state.update_entity_prior(
        entity_id="binary_sensor.motion",
        prob_given_true=0.8,  # Valid
        prob_given_false=0.2,  # Valid
        prior=0.5,  # Valid
        timestamp=timestamp,
    )

    # Calculate probability using calculator fixture
    result = calculator.calculate_occupancy_probability(current_states, prior_state)

    # Verify results
    assert MIN_PROBABILITY <= result.calculated_probability <= MAX_PROBABILITY
    assert MIN_PROBABILITY <= result.prior_probability <= MAX_PROBABILITY
    assert "binary_sensor.motion" in result.sensor_probabilities


def test_probability_calculator_with_all_sensors_unavailable(
    calculator, default_config
):
    """Test ProbabilityCalculator when all sensors are unavailable."""
    # Create test data with all sensors unavailable
    current_states = {
        "binary_sensor.motion": SensorInfo(
            state=STATE_UNAVAILABLE,
            availability=False,
            last_changed=datetime.now().isoformat(),
        ),
        "binary_sensor.door": SensorInfo(
            state=STATE_UNAVAILABLE,
            availability=False,
            last_changed=datetime.now().isoformat(),
        ),
    }
    prior_state = PriorState()
    timestamp = datetime.now().isoformat()
    prior_state.update_entity_prior(
        entity_id="binary_sensor.motion",
        prob_given_true=0.8,
        prob_given_false=0.2,
        prior=0.5,
        timestamp=timestamp,
    )
    prior_state.update_entity_prior(
        entity_id="binary_sensor.door",
        prob_given_true=0.7,
        prob_given_false=0.3,
        prior=0.4,
        timestamp=timestamp,
    )

    # Calculate probability using calculator fixture
    result = calculator.calculate_occupancy_probability(current_states, prior_state)

    # Verify results
    assert result.calculated_probability == MIN_PROBABILITY
    assert MIN_PROBABILITY <= result.prior_probability <= MAX_PROBABILITY
    assert not result.sensor_probabilities  # Should be empty


def test_probability_calculator_with_empty_data(calculator, default_config):
    """Test ProbabilityCalculator with empty data."""
    # Calculate probability with empty data using calculator fixture
    result = calculator.calculate_occupancy_probability({}, PriorState())

    # Verify results
    assert result.calculated_probability == MIN_PROBABILITY
    assert result.prior_probability == MIN_PROBABILITY
    assert not result.sensor_probabilities  # Should be empty
