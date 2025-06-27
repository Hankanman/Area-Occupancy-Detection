"""Tests for utils module."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.utils import (
    apply_decay,
    bayesian_probability,
    format_float,
    overall_probability,
    states_to_intervals,
    validate_datetime,
    validate_decay_factor,
    validate_prior,
    validate_prob,
    validate_weight,
)
from homeassistant.core import State
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from custom_components.area_occupancy.data.entity import Entity


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
        assert validate_prior(0.0) == 0.000001  # Minimum enforced
        assert validate_prior(-0.1) == 0.000001  # Minimum enforced
        assert validate_prior(1.0) == 0.999999
        assert validate_prior(1.1) == 0.999999


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
        )
        assert 0.0 <= result <= 1.0

        result = bayesian_probability(
            prior=0.9999,
            prob_given_true=0.99,
            prob_given_false=0.01,
            evidence=False,
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
        )
        assert 0.0 <= result <= 1.0

        result = bayesian_probability(
            prior=1.1,
            prob_given_true=-0.1,
            prob_given_false=1.1,
            evidence=False,
        )
        assert 0.0 <= result <= 1.0

    def test_bayesian_probability_fractional_weight(self) -> None:
        """Test Bayesian probability with fractional weight (now handled in likelihood)."""
        # Weight is now applied in likelihood calculation, not in bayesian_probability
        result = bayesian_probability(
            prior=0.5,
            prob_given_true=0.8,
            prob_given_false=0.2,
            evidence=True,
        )
        assert 0.0 <= result <= 1.0
        # Just test that it returns a valid probability
        assert 0.5 <= result <= 1.0

    def test_bayesian_probability_with_decay_preprocessing(self) -> None:
        """Test Bayesian probability with decay applied via preprocessing."""
        # Apply decay to probabilities first, then call bayesian_probability
        original_prob_true = 0.8
        original_prob_false = 0.2
        decay_factor = 0.5

        # Apply decay to get effective probabilities
        effective_prob_true, effective_prob_false = apply_decay(
            original_prob_true, original_prob_false, decay_factor
        )

        # Now call bayesian_probability with effective probabilities
        result = bayesian_probability(
            prior=0.5,
            prob_given_true=effective_prob_true,
            prob_given_false=effective_prob_false,
            evidence=False,
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
        mock_entity.likelihood.prob_given_true = 0.8
        mock_entity.likelihood.prob_given_false = 0.1
        mock_entity.effective_prob_given_true = 0.8  # No decay
        mock_entity.effective_prob_given_false = 0.1  # No decay

        entities = {"test_entity": cast("Entity", mock_entity)}
        prior = 0.3
        expected = bayesian_probability(
            prior=prior,
            prob_given_true=0.8,
            prob_given_false=0.1,
            evidence=True,
        )

        result = overall_probability(entities, prior)
        assert abs(result - expected) < 0.001

    def test_multiple_entities(self) -> None:
        """Test probability calculation with multiple entities."""
        # Create mock entities with different probabilities
        mock_entity1 = Mock()
        mock_entity1.evidence = True
        mock_entity1.decay.is_decaying = False
        mock_entity1.type.weight = 0.8
        mock_entity1.likelihood.prob_given_true = 0.8
        mock_entity1.likelihood.prob_given_false = 0.1
        mock_entity1.effective_prob_given_true = 0.8  # No decay
        mock_entity1.effective_prob_given_false = 0.1  # No decay

        mock_entity2 = Mock()
        mock_entity2.evidence = False
        mock_entity2.decay.is_decaying = False
        mock_entity2.type.weight = 0.6
        mock_entity2.likelihood.prob_given_true = 0.7
        mock_entity2.likelihood.prob_given_false = 0.2
        mock_entity2.effective_prob_given_true = 0.7  # No decay
        mock_entity2.effective_prob_given_false = 0.2  # No decay

        entities = {
            "entity1": cast("Entity", mock_entity1),
            "entity2": cast("Entity", mock_entity2),
        }
        prior = 0.3

        expected = bayesian_probability(
            prior=prior,
            prob_given_true=0.8,
            prob_given_false=0.1,
            evidence=True,
        )

        result = overall_probability(entities, prior)
        assert abs(result - expected) < 0.001

    def test_decaying_entity(self) -> None:
        """Test probability calculation with a decaying entity."""
        # Calculate effective probabilities with decay applied
        original_prob_true = 0.8
        original_prob_false = 0.1
        decay_factor = 0.5
        effective_prob_true, effective_prob_false = apply_decay(
            original_prob_true, original_prob_false, decay_factor
        )

        mock_entity = Mock()
        mock_entity.evidence = False
        mock_entity.decay.is_decaying = True
        mock_entity.decay.decay_factor = decay_factor
        mock_entity.type.weight = 1.0
        mock_entity.likelihood.prob_given_true = original_prob_true
        mock_entity.likelihood.prob_given_false = original_prob_false
        mock_entity.effective_prob_given_true = effective_prob_true
        mock_entity.effective_prob_given_false = effective_prob_false

        entities = {"test_entity": cast("Entity", mock_entity)}
        prior = 0.3
        expected = bayesian_probability(
            prior=prior,
            prob_given_true=effective_prob_true,
            prob_given_false=effective_prob_false,
            evidence=True,
        )

        result = overall_probability(entities, prior)
        assert abs(result - expected) < 0.001

    def test_no_entities(self) -> None:
        """Test probability calculation with no entities."""
        entities = {}
        prior = 0.3

        # With no entities, should return the prior unchanged
        result = overall_probability(entities, prior)
        assert result == prior

    def test_inactive_sensor_ignored(self) -> None:
        """Ensure inactive sensors do not influence the result."""
        mock_entity = Mock()
        mock_entity.evidence = False
        mock_entity.decay.is_decaying = False
        mock_entity.type.weight = 1.0
        mock_entity.likelihood.prob_given_true = 0.8
        mock_entity.likelihood.prob_given_false = 0.1
        mock_entity.effective_prob_given_true = 0.8  # No decay
        mock_entity.effective_prob_given_false = 0.1  # No decay

        entities = {"sensor": cast("Entity", mock_entity)}
        prior = 0.3

        result = overall_probability(entities, prior)
        assert result == prior

    def test_mixed_states(self) -> None:
        """Test probability calculation with mixed entity states."""
        # Create entities with different states
        mock_active = Mock()
        mock_active.evidence = True
        mock_active.decay.is_decaying = False
        mock_active.type.weight = 1.0
        mock_active.likelihood.prob_given_true = 0.8
        mock_active.likelihood.prob_given_false = 0.1
        mock_active.effective_prob_given_true = 0.8  # No decay
        mock_active.effective_prob_given_false = 0.1  # No decay

        mock_inactive = Mock()
        mock_inactive.evidence = False
        mock_inactive.decay.is_decaying = False
        mock_inactive.type.weight = 1.0
        mock_inactive.likelihood.prob_given_true = 0.8
        mock_inactive.likelihood.prob_given_false = 0.1
        mock_inactive.effective_prob_given_true = 0.8  # No decay
        mock_inactive.effective_prob_given_false = 0.1  # No decay

        # Calculate effective probabilities for decaying entity
        decay_factor = 0.5
        effective_prob_true, effective_prob_false = apply_decay(0.8, 0.1, decay_factor)

        mock_decaying = Mock()
        mock_decaying.evidence = False
        mock_decaying.decay.is_decaying = True
        mock_decaying.decay.decay_factor = decay_factor
        mock_decaying.type.weight = 1.0
        mock_decaying.likelihood.prob_given_true = 0.8
        mock_decaying.likelihood.prob_given_false = 0.1
        mock_decaying.effective_prob_given_true = effective_prob_true
        mock_decaying.effective_prob_given_false = effective_prob_false

        entities = {
            "active": cast("Entity", mock_active),
            "inactive": cast("Entity", mock_inactive),
            "decaying": cast("Entity", mock_decaying),
        }
        prior = 0.3

        result = overall_probability(entities, prior)
        assert 0.0 <= result <= 1.0


class TestStatesToIntervals:
    """Test the states_to_intervals helper."""

    @pytest.mark.asyncio
    async def test_intervals_cover_full_range(self) -> None:
        """Intervals should span start to end even if first change is later."""
        start = dt_util.utcnow() - timedelta(minutes=30)
        end = dt_util.utcnow()

        states = [
            State(
                "binary_sensor.test", "off", last_changed=start - timedelta(minutes=5)
            ),
            State(
                "binary_sensor.test", "on", last_changed=start + timedelta(minutes=10)
            ),
            State(
                "binary_sensor.test", "off", last_changed=start + timedelta(minutes=20)
            ),
        ]

        intervals = await states_to_intervals(states, start, end)

        assert intervals[0]["start"] == start
        assert intervals[-1]["end"] == end
        assert intervals[0]["state"] == "off"


class TestApplyDecayToLikelihood:
    """Test apply_decay function."""

    def test_no_decay(self) -> None:
        """Test that decay_factor=1.0 returns original probabilities."""
        prob_true = 0.8
        prob_false = 0.2

        result_true, result_false = apply_decay(prob_true, prob_false, 1.0)

        assert result_true == prob_true
        assert result_false == prob_false

    def test_full_decay(self) -> None:
        """Test that decay_factor=0.0 returns neutral probabilities."""
        prob_true = 0.8
        prob_false = 0.2

        result_true, result_false = apply_decay(prob_true, prob_false, 0.0)

        assert result_true == 0.5
        assert result_false == 0.5

    def test_partial_decay(self) -> None:
        """Test that partial decay moves probabilities toward neutral."""
        prob_true = 0.8
        prob_false = 0.2
        decay_factor = 0.5

        result_true, result_false = apply_decay(prob_true, prob_false, decay_factor)

        # Should be between original and neutral (0.5)
        assert 0.5 < result_true < 0.8
        assert 0.2 < result_false < 0.5

        # Should maintain mathematical equivalence with original bayes factor exponentiation
        original_bf = prob_true / prob_false
        decayed_bf = original_bf**decay_factor
        result_bf = result_true / result_false

        assert abs(result_bf - decayed_bf) < 0.001

    def test_mathematical_equivalence(self) -> None:
        """Test that apply_decay maintains mathematical equivalence."""
        prob_true = 0.9
        prob_false = 0.1
        decay_factor = 0.3

        # Original approach: apply decay as exponent to bayes factor
        original_bf = prob_true / prob_false
        expected_decayed_bf = original_bf**decay_factor

        # New approach: apply decay to probabilities
        result_true, result_false = apply_decay(prob_true, prob_false, decay_factor)
        actual_decayed_bf = result_true / result_false

        # Should be mathematically equivalent
        assert abs(actual_decayed_bf - expected_decayed_bf) < 0.001
