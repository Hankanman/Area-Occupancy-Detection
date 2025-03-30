"""Tests for Area Occupancy Detection probability calculations."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.area_occupancy.calculate_prob import (
    ProbabilityCalculator,
    update_probability,
)
from custom_components.area_occupancy.const import (
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_MIN_DELAY,
)


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator."""
    coordinator = MagicMock()
    coordinator.config = {
        CONF_DECAY_ENABLED: True,
        CONF_DECAY_WINDOW: 300,  # 5 minutes
        CONF_DECAY_MIN_DELAY: 60,  # 1 minute
        "threshold": 50,
    }
    coordinator.threshold = 0.5  # Convert from percentage to decimal
    coordinator.learned_priors = {}
    return coordinator


@pytest.fixture
def mock_probabilities():
    """Create a mock probabilities provider."""
    probabilities = MagicMock()
    probabilities.get_sensor_config.return_value = {
        "prob_given_true": 0.9,
        "prob_given_false": 0.1,
        "default_prior": 0.5,
        "weight": 1.0,
    }
    probabilities.is_entity_active.return_value = True
    return probabilities


@pytest.fixture
def calculator(mock_coordinator, mock_probabilities):
    """Create a ProbabilityCalculator instance."""
    return ProbabilityCalculator(mock_coordinator, mock_probabilities)


def test_update_probability_basic():
    """Test basic Bayesian probability update."""
    # Test with standard values
    result = update_probability(prior=0.5, prob_given_true=0.8, prob_given_false=0.2)
    assert 0.7 < result < 0.9

    # Test with extreme values
    assert update_probability(1.0, 1.0, 0.0) == MAX_PROBABILITY
    assert update_probability(0.0, 0.0, 1.0) == MIN_PROBABILITY


def test_update_probability_edge_cases():
    """Test edge cases for probability update."""
    # Test with zero denominator case
    result = update_probability(0.5, 0.0, 0.0)
    assert result == 0.5  # Should return prior

    # Test with values exceeding bounds
    result = update_probability(1.5, 0.8, 0.2)  # Prior > 1
    assert result <= MAX_PROBABILITY

    result = update_probability(-0.5, 0.8, 0.2)  # Prior < 0
    assert result >= MIN_PROBABILITY


def test_calculate_sensor_probability(calculator):
    """Test probability calculation for a single sensor."""
    entity_id = "binary_sensor.test"
    state = {"state": "on", "availability": True}

    # Test normal case
    weighted_prob, is_active, prob_details = calculator._calculate_sensor_probability(
        entity_id, state
    )
    assert weighted_prob > 0
    assert is_active is True
    assert all(
        k in prob_details for k in ["probability", "weight", "weighted_probability"]
    )

    # Test unavailable sensor
    state["availability"] = False
    weighted_prob, is_active, prob_details = calculator._calculate_sensor_probability(
        entity_id, state
    )
    assert weighted_prob == 0
    assert is_active is False

    # Test inactive state
    calculator.probabilities.is_entity_active.return_value = False
    state["availability"] = True
    weighted_prob, is_active, prob_details = calculator._calculate_sensor_probability(
        entity_id, state
    )
    assert weighted_prob == 0
    assert is_active is False


def test_calculate_base_probability(calculator):
    """Test base probability calculation with multiple sensors."""
    sensor_states = {
        "binary_sensor.test1": {"state": "on", "availability": True},
        "binary_sensor.test2": {"state": "on", "availability": True},
    }
    active_triggers = []
    sensor_probs = {}
    now = datetime.now()

    result = calculator._calculate_base_probability(
        sensor_states, active_triggers, sensor_probs, now
    )

    assert MIN_PROBABILITY <= result <= MAX_PROBABILITY
    assert len(active_triggers) == 2
    assert len(sensor_probs) == 2


def test_perform_calculation_logic(calculator):
    """Test the main calculation logic."""
    sensor_states = {
        "binary_sensor.test1": {"state": "on", "availability": True},
        "binary_sensor.test2": {"state": "off", "availability": True},
    }
    now = datetime.now()

    # First calculation
    result = calculator.perform_calculation_logic(sensor_states, now)
    assert isinstance(result, dict)
    assert all(
        k in result
        for k in [
            "probability",
            "potential_probability",
            "prior_probability",
            "active_triggers",
            "sensor_probabilities",
            "device_states",
            "decay_status",
            "sensor_availability",
            "is_occupied",
        ]
    )

    # Test decay
    later = now + timedelta(seconds=30)
    second_result = calculator.perform_calculation_logic(sensor_states, later)
    assert second_result["probability"] <= result["probability"]


def test_decay_handling(calculator):
    """Test decay behavior over time."""
    # Set a very low threshold to ensure decay can happen
    calculator.coordinator.threshold = 0.1  # 10%

    # Configure sensor probabilities for high probability
    calculator.probabilities.get_sensor_config.return_value = {
        "prob_given_true": 0.99,
        "prob_given_false": 0.01,
        "default_prior": 0.9,
        "weight": 1.0,
    }

    # Start with a single active sensor
    sensor_states = {
        "binary_sensor.test": {"state": "on", "availability": True},
    }
    calculator.probabilities.is_entity_active.return_value = True

    # First calculation - establish high probability
    now = datetime.now()
    initial_result = calculator.perform_calculation_logic(sensor_states, now)
    initial_prob = initial_result["probability"]
    assert initial_prob > 0.9  # Should be very high

    # Store this high probability as previous
    calculator.previous_probability = initial_prob

    # Now configure for much lower probability to trigger decay
    calculator.probabilities.get_sensor_config.return_value = {
        "prob_given_true": 0.2,
        "prob_given_false": 0.8,
        "default_prior": 0.2,
        "weight": 0.2,  # Much lower weight to ensure lower probability
    }

    # After short delay (within decay_min_delay)
    short_delay = now + timedelta(seconds=30)
    after_short_delay = calculator.perform_calculation_logic(sensor_states, short_delay)

    # Should start decaying from previous high probability
    # The probability should be the same as initial_prob since we're within decay_min_delay
    assert after_short_delay["probability"] == initial_prob
    assert after_short_delay["decay_status"]["global_decay"] == 0.0

    # Store the probability for next calculation
    calculator.previous_probability = after_short_delay["probability"]

    # After significant delay (beyond decay_min_delay)
    long_delay = now + timedelta(seconds=180)  # 3 minutes
    after_long_delay = calculator.perform_calculation_logic(sensor_states, long_delay)

    # Should have decayed
    assert after_long_delay["probability"] < initial_prob
    assert after_long_delay["probability"] > calculator.coordinator.threshold
    assert after_long_delay["decay_status"]["global_decay"] > 0.0


def test_threshold_behavior(calculator):
    """Test behavior around the occupancy threshold."""
    sensor_states = {
        "binary_sensor.test": {"state": "on", "availability": True},
    }

    # Configure a high probability case (should be well above 50% threshold)
    calculator.probabilities.get_sensor_config.return_value = {
        "prob_given_true": 0.99,
        "prob_given_false": 0.01,
        "default_prior": 0.9,
        "weight": 1.0,
    }
    calculator.probabilities.is_entity_active.return_value = True

    result = calculator.perform_calculation_logic(sensor_states, datetime.now())
    assert result["probability"] > calculator.coordinator.threshold
    assert result["is_occupied"] is True

    # Configure a low probability case (should be well below 50% threshold)
    calculator.probabilities.get_sensor_config.return_value = {
        "prob_given_true": 0.1,
        "prob_given_false": 0.9,
        "default_prior": 0.1,
        "weight": 1.0,
    }

    result = calculator.perform_calculation_logic(sensor_states, datetime.now())
    assert result["probability"] < calculator.coordinator.threshold
    assert result["is_occupied"] is False
