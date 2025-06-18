"""Tests for utils module."""

from datetime import datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

from custom_components.area_occupancy.utils import (
    bayesian_probability,
    format_float,
    overall_probability,
    validate_datetime,
    validate_decay_factor,
    validate_prior,
    validate_prob,
    validate_weight,
)
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from custom_components.area_occupancy.data.entity import Entity


# ruff: noqa: SLF001
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
        result = bayesian_probability(
            prior=prior,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            evidence=True,
            weight=1.0,
            decay_factor=1.0,
        )
        assert abs(result - 0.774) < 0.001

    def test_inactive_state_calculation(self) -> None:
        """Test Bayesian calculation when sensor is inactive."""
        prior = 0.3
        prob_given_true = 0.8
        prob_given_false = 0.1

        # For inactive state: P(occupied | inactive)
        # P(inactive|occupied) = 1 - prob_given_true = 1 - 0.8 = 0.2
        # P(inactive|empty) = 1 - prob_given_false = 1 - 0.1 = 0.9
        # numerator = P(inactive|occupied) * P(occupied) = 0.2 * 0.3 = 0.06
        # denominator = (0.2 * 0.3) + (0.9 * 0.7) = 0.06 + 0.63 = 0.69
        # result = 0.06 / 0.69 ≈ 0.087
        result = bayesian_probability(
            prior=prior,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            evidence=False,
            weight=1.0,
            decay_factor=1.0,
        )
        # Correct calculation:
        # P(OFF|occupied) = 0.2, P(OFF|unoccupied) = 0.9
        # P(evidence) = 0.2 * 0.3 + 0.9 * 0.7 = 0.69
        # P(occupied|OFF) = (0.2 * 0.3) / 0.69 ≈ 0.087
        assert abs(result - 0.087) < 0.01

    def test_edge_cases(self) -> None:
        """Test edge cases for Bayesian calculation."""
        # Test with extreme values - function should clamp to valid range
        result = bayesian_probability(
            prior=0.0001,
            prob_given_true=0.99,
            prob_given_false=0.01,
            evidence=True,
            weight=1.0,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0

        result = bayesian_probability(
            prior=0.9999,
            prob_given_true=0.99,
            prob_given_false=0.01,
            evidence=False,
            weight=1.0,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0

    def test_validation_of_inputs(self) -> None:
        """Test that invalid inputs are handled properly."""
        # The function should handle out-of-bounds inputs gracefully
        result = bayesian_probability(
            prior=-0.1,
            prob_given_true=1.1,
            prob_given_false=-0.1,
            evidence=True,
            weight=1.0,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0

        result = bayesian_probability(
            prior=1.1,
            prob_given_true=-0.1,
            prob_given_false=1.1,
            evidence=False,
            weight=1.0,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0

    def test_bayesian_probability_fractional_weight(self) -> None:
        """Test Bayesian probability with fractional weight."""
        result = bayesian_probability(
            prior=0.5,
            prob_given_true=0.8,
            prob_given_false=0.2,
            evidence=True,
            weight=0.5,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0
        # With fractional weight, result should be between prior and full update
        full_result = bayesian_probability(
            prior=0.5,
            prob_given_true=0.8,
            prob_given_false=0.2,
            evidence=True,
            weight=1.0,
            decay_factor=1.0,
        )
        assert 0.5 <= result <= full_result

    def test_bayesian_probability_fractional_decay(self) -> None:
        """Test Bayesian probability with fractional decay factor."""
        result = bayesian_probability(
            prior=0.5,
            prob_given_true=0.8,
            prob_given_false=0.2,
            evidence=False,
            weight=1.0,
            decay_factor=0.5,
        )
        assert 0.0 <= result <= 1.0


class TestOverallProbability:
    """Test overall_probability function."""

    def test_single_entity(self) -> None:
        """Test probability calculation with a single entity."""
        # Create a mock entity with known probabilities
        mock_entity = Mock()
        mock_entity.evidence = True
        mock_entity.decay.is_decaying = False
        mock_entity.type.weight = 1.0
        mock_entity.prior.prob_given_true = 0.8
        mock_entity.prior.prob_given_false = 0.1

        entities = {"test_entity": cast("Entity", mock_entity)}
        prior = 0.3

        # Expected: (0.8 * 0.3) / (0.8 * 0.3 + 0.1 * 0.7) = 0.24 / 0.31 ≈ 0.774
        result = overall_probability(entities, prior)
        assert abs(result - 0.774) < 0.001

    def test_multiple_entities(self) -> None:
        """Test probability calculation with multiple entities."""
        # Create mock entities with different probabilities
        mock_entity1 = Mock()
        mock_entity1.evidence = True
        mock_entity1.decay.is_decaying = False
        mock_entity1.type.weight = 0.8
        mock_entity1.prior.prob_given_true = 0.8
        mock_entity1.prior.prob_given_false = 0.1

        mock_entity2 = Mock()
        mock_entity2.evidence = False
        mock_entity2.decay.is_decaying = False
        mock_entity2.type.weight = 0.6
        mock_entity2.prior.prob_given_true = 0.7
        mock_entity2.prior.prob_given_false = 0.2

        entities = {
            "entity1": cast("Entity", mock_entity1),
            "entity2": cast("Entity", mock_entity2),
        }
        prior = 0.3

        # Sequential Bayesian updates - just ensure valid probability range
        result = overall_probability(entities, prior)
        assert 0.0 <= result <= 1.0

    def test_decaying_entity(self) -> None:
        """Test probability calculation with a decaying entity."""
        mock_entity = Mock()
        mock_entity.evidence = False
        mock_entity.decay.is_decaying = True
        mock_entity.decay.decay_factor = 0.5  # Half decay
        mock_entity.type.weight = 1.0
        mock_entity.prior.prob_given_true = 0.8
        mock_entity.prior.prob_given_false = 0.1

        entities = {"test_entity": cast("Entity", mock_entity)}
        prior = 0.3

        # With decay factor of 0.5, the evidence weight is reduced
        result = overall_probability(entities, prior)
        assert 0.0 <= result <= 1.0
        # Result should be between the prior and what it would be with full weight
        # but closer to prior due to decay

    def test_no_entities(self) -> None:
        """Test probability calculation with no entities."""
        entities = {}
        prior = 0.3

        # With no entities, should return the prior unchanged
        result = overall_probability(entities, prior)
        assert result == prior

    def test_mixed_states(self) -> None:
        """Test probability calculation with mixed entity states."""
        # Create entities with different states
        mock_active = Mock()
        mock_active.evidence = True
        mock_active.decay.is_decaying = False
        mock_active.type.weight = 1.0
        mock_active.prior.prob_given_true = 0.8
        mock_active.prior.prob_given_false = 0.1

        mock_inactive = Mock()
        mock_inactive.evidence = False
        mock_inactive.decay.is_decaying = False
        mock_inactive.type.weight = 1.0
        mock_inactive.prior.prob_given_true = 0.8
        mock_inactive.prior.prob_given_false = 0.1

        mock_decaying = Mock()
        mock_decaying.evidence = False
        mock_decaying.decay.is_decaying = True
        mock_decaying.decay.decay_factor = 0.5
        mock_decaying.type.weight = 1.0
        mock_decaying.prior.prob_given_true = 0.8
        mock_decaying.prior.prob_given_false = 0.1

        entities = {
            "active": cast("Entity", mock_active),
            "inactive": cast("Entity", mock_inactive),
            "decaying": cast("Entity", mock_decaying),
        }
        prior = 0.3

        # Should combine all evidence appropriately and return valid probability
        result = overall_probability(entities, prior)
        assert 0.0 <= result <= 1.0
