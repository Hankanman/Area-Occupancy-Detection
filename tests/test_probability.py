"""Tests for the Bayesian probability calculations (continued)."""
from unittest.mock import patch
import pytest
import numpy as np

from custom_components.room_occupancy.probability import BayesianProbability

def test_decay_integration():
    """Test probability calculations with decay factors."""
    calc = BayesianProbability()
    
    # Test probabilities with decay factors
    base_probs = [0.9, 0.8, 0.7]
    decay_factors = [1.0, 0.5, 0.2]
    
    decayed_probs = [p * d for p, d in zip(base_probs, decay_factors)]
    result = calc.calculate_probability(decayed_probs)
    
    # Result should be lower than without decay
    no_decay_result = calc.calculate_probability(base_probs)
    assert result < no_decay_result

def test_probability_threshold_effects():
    """Test probability calculations near thresholds."""
    calc = BayesianProbability()
    
    # Test near-threshold probabilities
    threshold = 0.5
    slightly_above = [0.51, 0.52, 0.51]
    result_above = calc.calculate_probability(slightly_above)
    
    slightly_below = [0.49, 0.48, 0.49]
    result_below = calc.calculate_probability(slightly_below)
    
    assert result_above > threshold
    assert result_below < threshold

def test_sensor_type_weighting():
    """Test probability calculations with different sensor types."""
    calc = BayesianProbability()
    
    # Motion sensor (high weight)
    motion_prob = 0.9
    motion_weight = 0.8
    
    # Environmental sensor (lower weight)
    env_prob = 0.7
    env_weight = 0.2
    
    result = calc.calculate_weighted_probability(
        [motion_prob, env_prob],
        [motion_weight, env_weight]
    )
    
    # Result should be closer to motion sensor probability
    assert abs(result - motion_prob) < abs(result - env_prob)

def test_probability_stability():
    """Test stability of probability calculations over time."""
    calc = BayesianProbability()
    
    # Test repeated calculations with same inputs
    probs = [0.7, 0.6, 0.8]
    results = [calc.calculate_probability(probs) for _ in range(10)]
    
    # All results should be identical
    assert all(r == results[0] for r in results)
    
    # Test stability with small variations
    varied_probs = [[p + np.random.uniform(-0.01, 0.01) for p in probs] 
                   for _ in range(10)]
    varied_results = [calc.calculate_probability(vp) for vp in varied_probs]
    
    # Results should be similar but not identical
    mean_result = np.mean(varied_results)
    assert all(abs(r - mean_result) < 0.1 for r in varied_results)

def test_contradictory_evidence():
    """Test handling of contradictory probability evidence."""
    calc = BayesianProbability()
    
    # Strong contradictory evidence
    high_probs = [0.9, 0.95, 0.9]  # Strong positive evidence
    low_probs = [0.1, 0.05, 0.1]   # Strong negative evidence
    
    # Calculate separately
    high_result = calc.calculate_probability(high_probs)
    low_result = calc.calculate_probability(low_probs)
    
    # Combined calculation
    combined_result = calc.calculate_probability(high_probs + low_probs)
    
    # Combined result should be moderate
    assert low_result < combined_result < high_result
    assert abs(combined_result - 0.5) < abs(high_result - 0.5)

def test_missing_data_handling():
    """Test probability calculations with missing or invalid data."""
    calc = BayesianProbability()
    
    # Test with mixed valid and invalid data
    probs = [0.8, None, 0.7, float('nan'), 0.6]
    valid_probs = [p for p in probs if p is not None and not np.isnan(p)]
    
    result = calc.calculate_probability(valid_probs)
    assert 0.0 <= result <= 1.0
    
    # Test confidence reduction with missing data
    full_data = [0.8, 0.7, 0.6]
    full_result = calc.calculate_probability(full_data)
    
    assert abs(full_result - 0.5) >= abs(result - 0.5)

def test_temporal_correlation():
    """Test handling of temporally correlated probabilities."""
    calc = BayesianProbability()
    
    # Simulate sequence of related probabilities
    sequence = [0.8, 0.78, 0.82, 0.79, 0.81]
    
    # Calculate with temporal weights
    temporal_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing weight with time
    
    result = calc.calculate_weighted_probability(sequence, temporal_weights)
    
    # Result should be closer to recent values
    assert abs(result - sequence[0]) < abs(result - sequence[-1])

def test_probability_convergence():
    """Test convergence of probability calculations."""
    calc = BayesianProbability()
    
    # Test convergence with repeated evidence
    initial_prob = 0.5
    evidence_prob = 0.8
    
    results = []
    calc.update_prior(initial_prob)
    
    for _ in range(10):
        result = calc.calculate_probability([evidence_prob])
        results.append(result)
        calc.update_prior(result)
    
    # Results should converge towards evidence probability
    assert abs(results[-1] - evidence_prob) < abs(results[0] - evidence_prob)

def test_confidence_thresholds():
    """Test probability calculations with confidence thresholds."""
    calc = BayesianProbability()
    
    # Test high confidence scenario
    high_conf_probs = [0.85, 0.82, 0.88]
    high_conf_weights = [0.9, 0.95, 0.92]
    
    high_conf_result = calc.calculate_weighted_probability(
        high_conf_probs,
        high_conf_weights
    )
    
    # Test low confidence scenario
    low_conf_probs = [0.85, 0.82, 0.88]
    low_conf_weights = [0.3, 0.4, 0.35]
    
    low_conf_result = calc.calculate_weighted_probability(
        low_conf_probs,
        low_conf_weights
    )
    
    # High confidence should give more extreme results
    assert abs(high_conf_result - 0.5) > abs(low_conf_result - 0.5)

def test_probability_bounds():
    """Test enforcement of probability bounds."""
    calc = BayesianProbability()
    
    # Test with extreme values
    extreme_probs = [1.5, -0.5, 2.0, -1.0]
    result = calc.calculate_probability(extreme_probs)
    
    # Result should be bounded
    assert 0.0 <= result <= 1.0
    
    # Test with extreme weights
    extreme_weights = [1000, -500, 2000, -1000]
    result = calc.calculate_weighted_probability(
        [0.7, 0.6, 0.8, 0.5],
        extreme_weights
    )
    
    # Result should be bounded
    assert 0.0 <= result <= 1.0
