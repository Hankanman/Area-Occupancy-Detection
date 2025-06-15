"""Tests for utils module."""

from datetime import datetime

from custom_components.area_occupancy.utils import (
    bayesian_probability,
    format_float,
    validate_datetime,
    validate_decay_factor,
    validate_prior,
    validate_prob,
    validate_weight,
)
from homeassistant.util import dt as dt_util


class TestValidateProb:
    """Test validate_prob function."""

    def test_valid_probabilities(self) -> None:
        """Test validation of valid probability values."""
        assert validate_prob(0.0) == 0.001  # Minimum enforced to avoid division by zero
        assert validate_prob(1.0) == 1.0
        assert validate_prob(0.5) == 0.5
        assert validate_prob(0.001) == 0.001
        assert validate_prob(0.999) == 0.999

    def test_clamp_invalid_probabilities(self) -> None:
        """Test clamping of invalid probability values."""
        assert validate_prob(-0.1) == 0.001  # Minimum enforced
        assert validate_prob(1.1) == 1.0
        assert validate_prob(-999) == 0.001  # Minimum enforced
        assert validate_prob(999) == 1.0

    def test_type_conversion(self) -> None:
        """Test type conversion to float."""
        assert validate_prob(1) == 1.0
        assert validate_prob(0) == 0.001  # Minimum enforced


class TestValidatePrior:
    """Test validate_prior function."""

    def test_valid_priors(self) -> None:
        """Test validation of valid prior values."""
        assert validate_prior(0.001) == 0.001
        assert validate_prior(0.5) == 0.5
        assert validate_prior(0.999) == 0.999

    def test_clamp_invalid_priors(self) -> None:
        """Test clamping of invalid prior values."""
        assert validate_prior(0.0) == 0.001  # Minimum enforced
        assert validate_prior(-0.1) == 0.001  # Minimum enforced
        assert validate_prior(1.0) == 1.0
        assert validate_prior(1.1) == 1.0


class TestValidateWeight:
    """Test validate_weight function."""

    def test_valid_weights(self) -> None:
        """Test validation of valid weight values."""
        assert validate_weight(0.01) == 0.01
        assert validate_weight(0.5) == 0.5
        assert validate_weight(0.99) == 0.99

    def test_clamp_invalid_weights(self) -> None:
        """Test clamping of invalid weight values."""
        assert validate_weight(0.0) == 0.01
        assert validate_weight(-0.1) == 0.01
        assert validate_weight(1.0) == 0.99
        assert validate_weight(1.1) == 0.99


class TestValidateDecayFactor:
    """Test validate_decay_factor function."""

    def test_valid_decay_factors(self) -> None:
        """Test validation of valid decay factor values."""
        assert validate_decay_factor(0.0) == 0.0
        assert validate_decay_factor(0.5) == 0.5
        assert validate_decay_factor(1.0) == 1.0

    def test_clamp_invalid_decay_factors(self) -> None:
        """Test clamping of invalid decay factor values."""
        assert validate_decay_factor(-0.1) == 0.0
        assert validate_decay_factor(1.1) == 1.0


class TestValidateDatetime:
    """Test validate_datetime function."""

    def test_valid_datetime(self) -> None:
        """Test validation of valid datetime objects."""
        now = dt_util.utcnow()
        assert validate_datetime(now) == now

    def test_none_datetime(self) -> None:
        """Test handling of None datetime."""
        result = validate_datetime(None)
        assert isinstance(result, datetime)
        # Should be recent (within last minute)
        assert (dt_util.utcnow() - result).total_seconds() < 60


class TestFormatFloat:
    """Test format_float function."""

    def test_formatting(self) -> None:
        """Test float formatting to 2 decimal places."""
        assert format_float(1.234567) == 1.23
        assert format_float(1.0) == 1.0
        assert format_float(0.999) == 1.0
        assert format_float(0.001) == 0.0


class TestBayesianProbability:
    """Test bayesian_probability function."""

    def test_active_state_calculation(self) -> None:
        """Test Bayesian calculation when sensor is active."""
        # P(occupied | sensor active) = P(sensor active | occupied) * P(occupied) / P(sensor active)
        prior = 0.3
        prob_given_true = 0.8
        prob_given_false = 0.1

        # Expected: (0.8 * 0.3) / (0.8 * 0.3 + 0.1 * 0.7) = 0.24 / 0.31 ≈ 0.774
        result = bayesian_probability(prior, prob_given_true, prob_given_false, True)
        assert abs(result - 0.774) < 0.001

    def test_inactive_state_calculation(self) -> None:
        """Test Bayesian calculation when sensor is inactive."""
        prior = 0.3
        prob_given_true = 0.8
        prob_given_false = 0.1

        # For inactive state: P(occupied | inactive) = P(inactive | occupied) * P(occupied) / P(inactive)
        # numerator = prob_given_false * prior = 0.1 * 0.3 = 0.03
        # denominator = (prob_given_false * prior) + ((1 - prob_given_false) * (1 - prior))
        #             = (0.1 * 0.3) + (0.9 * 0.7) = 0.03 + 0.63 = 0.66
        # result = 0.03 / 0.66 ≈ 0.045
        result = bayesian_probability(prior, prob_given_true, prob_given_false, False)
        assert abs(result - 0.045) < 0.001

    def test_edge_cases(self) -> None:
        """Test edge cases for Bayesian calculation."""
        # Test with extreme values
        result = bayesian_probability(0.0001, 0.99, 0.01, True)
        assert 0 <= result <= 1

        result = bayesian_probability(0.9999, 0.99, 0.01, False)
        assert 0 <= result <= 1

    def test_probability_bounds(self) -> None:
        """Test that results are always within [0, 1] bounds."""
        # Test various combinations
        test_cases = [
            (0.1, 0.9, 0.05, True),
            (0.1, 0.9, 0.05, False),
            (0.9, 0.1, 0.95, True),
            (0.9, 0.1, 0.95, False),
            (0.5, 0.5, 0.5, True),
            (0.5, 0.5, 0.5, False),
        ]

        for prior, prob_true, prob_false, is_active in test_cases:
            result = bayesian_probability(prior, prob_true, prob_false, is_active)
            assert 0 <= result <= 1, (
                f"Result {result} out of bounds for inputs {prior}, {prob_true}, {prob_false}, {is_active}"
            )

    def test_validation_of_inputs(self) -> None:
        """Test that invalid inputs are handled properly."""
        # The function should handle out-of-bounds inputs gracefully
        result = bayesian_probability(-0.1, 1.1, -0.1, True)
        assert 0 <= result <= 1

        result = bayesian_probability(1.1, -0.1, 1.1, False)
        assert 0 <= result <= 1
