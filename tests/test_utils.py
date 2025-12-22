"""Tests for utils module."""

import math
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.const import MAX_PROBABILITY, MIN_PROBABILITY
from custom_components.area_occupancy.utils import (
    bayesian_probability,
    clamp_probability,
    combine_priors,
    format_float,
    format_percentage,
    map_binary_state_to_semantic,
)


def _create_mock_entity(
    evidence: bool | None = True,
    prob_given_true: float = 0.8,
    prob_given_false: float = 0.1,
    weight: float = 1.0,
    is_decaying: bool = False,
    decay_factor: float = 1.0,
    is_continuous: bool = False,
) -> Mock:
    """Create a mock entity for testing bayesian_probability.

    Args:
        evidence: Entity evidence state (True/False/None)
        prob_given_true: Probability given true
        prob_given_false: Probability given false
        weight: Entity weight
        is_decaying: Whether entity is decaying
        decay_factor: Decay factor (0.0 to 1.0)
        is_continuous: Whether entity uses continuous likelihood

    Returns:
        Mock entity object
    """
    entity = Mock()
    entity.evidence = evidence
    entity.decay.decay_factor = decay_factor
    entity.decay.is_decaying = is_decaying
    # decay_factor property returns 1.0 when evidence is True, otherwise decay.decay_factor
    entity.decay_factor = 1.0 if evidence is True else decay_factor
    entity.prob_given_true = prob_given_true
    entity.prob_given_false = prob_given_false
    entity.weight = weight
    entity.is_continuous_likelihood = is_continuous
    return entity


class TestUtils:
    """Test utility functions."""

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            # Basic formatting
            (1.234567, 1.23),
            (1.0, 1.0),
            (0.999, 1.0),
            (0.001, 0.0),
            # Edge cases
            (0.0, 0.0),
            (-1.234567, -1.23),
            (999.999, 1000.0),
            # Very large numbers
            (1234567.89, 1234567.89),
            # Very small numbers
            (0.0001, 0.0),
            # String conversion (format_float can handle strings)
            ("1.234567", 1.23),
            ("0", 0.0),
        ],
    )
    def test_format_float(self, input_value, expected) -> None:
        """Test float formatting to 2 decimal places."""
        assert format_float(input_value) == expected

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            # Basic percentage formatting
            (0.5, "50.00%"),
            (0.123, "12.30%"),
            (1.0, "100.00%"),
            (0.0, "0.00%"),
            # Edge cases
            (0.999, "99.90%"),
            (0.001, "0.10%"),
            (1.5, "150.00%"),
            (-0.1, "-10.00%"),
            # Very large percentages
            (10.0, "1000.00%"),
            # Very small percentages
            (0.0001, "0.01%"),
            # Negative percentages
            (-0.5, "-50.00%"),
        ],
    )
    def test_format_percentage(self, input_value, expected) -> None:
        """Test percentage formatting."""
        assert format_percentage(input_value) == expected

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            # Test values within range
            (0.5, 0.5),
            (0.0, MIN_PROBABILITY),
            (1.0, MAX_PROBABILITY),
            # Test values outside range
            (-0.1, MIN_PROBABILITY),
            (1.5, MAX_PROBABILITY),
            (0.01, MIN_PROBABILITY),  # Assuming MIN_PROBABILITY > 0.01
            (0.99, MAX_PROBABILITY),  # Assuming MAX_PROBABILITY < 0.99
        ],
    )
    def test_clamp_probability(self, input_value, expected) -> None:
        """Test clamp_probability function with various input values."""
        assert clamp_probability(input_value) == expected

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            (float("inf"), MAX_PROBABILITY),
            (float("-inf"), MIN_PROBABILITY),
            (float("nan"), MAX_PROBABILITY),  # NaN clamped to MAX_PROBABILITY
        ],
    )
    def test_clamp_probability_edge_cases(self, input_value, expected) -> None:
        """Test clamp_probability handles edge cases (inf, nan) correctly."""
        result = clamp_probability(input_value)
        if math.isnan(input_value):
            assert not math.isnan(result)
            assert not math.isinf(result)
            assert result == expected
        else:
            assert result == expected


class TestCombinePriors:
    """Test combine_priors function.

    Tests verify that area and time priors are correctly combined using weighted
    averaging in logit space, with proper handling of edge cases.
    """

    def test_basic_combine_priors(self) -> None:
        """Test basic prior combination with explicit expected behavior."""
        # With equal priors, result should be the same
        result = combine_priors(0.5, 0.5)
        assert abs(result - 0.5) < 1e-6

        # With different priors, result should be between them
        result = combine_priors(0.3, 0.7)
        assert 0.3 < result < 0.7  # Should be between the two priors

        # With default time_weight (0.2), result should be closer to area_prior
        result = combine_priors(0.2, 0.8)
        assert 0.2 < result < 0.8
        # Should be closer to area_prior (0.2) than time_prior (0.8)
        assert abs(result - 0.2) < abs(result - 0.8)

    def test_combine_priors_edge_cases(self) -> None:
        """Test combine_priors handles edge cases correctly."""
        # Zero time_weight should return area_prior only
        result = combine_priors(0.3, 0.7, time_weight=0.0)
        assert abs(result - clamp_probability(0.3)) < 1e-6

        # Full time_weight should return time_prior only
        result = combine_priors(0.3, 0.7, time_weight=1.0)
        assert abs(result - clamp_probability(0.7)) < 1e-6

        # Zero priors should be clamped to MIN_PROBABILITY
        result = combine_priors(0.0, 0.0)
        assert abs(result - MIN_PROBABILITY) < 1e-6

        # Maximum priors should be clamped to MAX_PROBABILITY
        result = combine_priors(1.0, 1.0)
        assert abs(result - MAX_PROBABILITY) < 1e-6

        # Identical priors should return the same value
        result = combine_priors(0.5, 0.5)
        assert abs(result - 0.5) < 1e-6

        # Extreme time_weight values should be clamped
        result_neg = combine_priors(0.3, 0.7, time_weight=-0.1)
        result_over = combine_priors(0.3, 0.7, time_weight=1.5)
        expected_zero = combine_priors(0.3, 0.7, time_weight=0.0)
        expected_one = combine_priors(0.3, 0.7, time_weight=1.0)
        assert abs(result_neg - expected_zero) < 1e-10
        assert abs(result_over - expected_one) < 1e-10


class TestBayesianProbability:
    """Test bayesian_probability function.

    Tests verify the core Bayesian probability calculation, including:
    - Correct combination of multiple sensor inputs
    - Proper handling of edge cases (empty entities, invalid likelihoods)
    - Numerical stability with extreme values
    - Correct behavior for different sensor states (active, inactive, unavailable, decaying)
    - Continuous vs binary sensor handling
    """

    def test_basic_bayesian_calculation(self) -> None:
        """Test basic Bayesian probability calculation with explicit expected values.

        Verifies that active sensors increase probability and inactive sensors decrease it.
        """
        # Active entity with high prob_given_true should increase probability
        entity1 = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=0.1, weight=0.5
        )

        # Inactive entity should decrease probability (uses inverse likelihoods)
        entity2 = _create_mock_entity(
            evidence=False, prob_given_true=0.7, prob_given_false=0.2, weight=0.3
        )

        entities = {"entity1": entity1, "entity2": entity2}

        # With prior 0.5, active entity should increase probability
        result = bayesian_probability(entities, prior=0.5)
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # Active entity should increase probability

        # Test with lower prior - should still be increased by active entity
        result_low_prior = bayesian_probability(entities, prior=0.2)
        assert result_low_prior > 0.2  # Active entity increases from low prior
        assert (
            result_low_prior < result
        )  # Lower prior should result in lower final probability

    def test_bayesian_with_decay(self) -> None:
        """Test Bayesian probability with decaying entities.

        Verifies that decay interpolation affects the result correctly.
        """
        # Entity with no current evidence but decaying (half decay)
        entity_decaying = _create_mock_entity(
            evidence=False,
            prob_given_true=0.8,
            prob_given_false=0.1,
            weight=1.0,
            is_decaying=True,
            decay_factor=0.5,
        )

        # Entity with full evidence (no decay)
        entity_active = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=0.1, weight=1.0
        )

        # Entity with no evidence and no decay
        entity_inactive = _create_mock_entity(
            evidence=False, prob_given_true=0.8, prob_given_false=0.1, weight=1.0
        )

        prior = 0.5
        result_decaying = bayesian_probability({"entity": entity_decaying}, prior=prior)
        result_active = bayesian_probability({"entity": entity_active}, prior=prior)
        result_inactive = bayesian_probability({"entity": entity_inactive}, prior=prior)

        # Decaying entity should have effect between active and inactive
        assert result_active > result_decaying > result_inactive

    def test_bayesian_empty_entities(self) -> None:
        """Test Bayesian probability with empty entities returns prior."""
        prior = 0.7
        result = bayesian_probability({}, prior=prior)
        assert abs(result - clamp_probability(prior)) < 1e-6

    def test_bayesian_numerical_stability(self) -> None:
        """Test Bayesian probability numerical stability with many entities.

        Verifies that calculations remain stable and don't produce NaN/inf with many entities.
        """
        entities = {}

        # Create many entities with varying probabilities
        for i in range(10):
            entity = _create_mock_entity(
                evidence=(i % 2 == 0),  # Alternate evidence
                prob_given_true=0.8,
                prob_given_false=0.1,
                weight=0.1,
            )
            entities[f"entity_{i}"] = entity

        result = bayesian_probability(entities, prior=0.5)
        assert 0.0 <= result <= 1.0
        assert not (math.isnan(result) or math.isinf(result))
        # With mixed evidence, result should be reasonable
        assert 0.1 < result < 0.9

    def test_bayesian_zero_weight_entities(self) -> None:
        """Test that entities with zero weight are correctly ignored."""
        entity_zero_weight = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=0.1, weight=0.0
        )

        entity_with_weight = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=0.1, weight=0.5
        )

        entities = {
            "zero_weight": entity_zero_weight,
            "with_weight": entity_with_weight,
        }

        result = bayesian_probability(entities, prior=0.5)
        result_without_zero = bayesian_probability(
            {"with_weight": entity_with_weight}, prior=0.5
        )

        # Should behave exactly the same as if zero-weight entity wasn't present
        assert abs(result - result_without_zero) < 1e-6

    def test_bayesian_invalid_likelihoods_filtered(self) -> None:
        """Test that entities with invalid likelihoods are filtered out correctly."""
        # Invalid entities (should be filtered)
        entity_invalid1 = _create_mock_entity(
            evidence=True, prob_given_true=0.0, prob_given_false=0.1, weight=0.5
        )  # prob_given_true = 0 is invalid
        entity_invalid2 = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=1.0, weight=0.5
        )  # prob_given_false = 1 is invalid
        entity_invalid3 = _create_mock_entity(
            evidence=True, prob_given_true=1.5, prob_given_false=0.1, weight=0.5
        )  # prob_given_true > 1 is invalid

        # Valid entity
        entity_valid = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=0.1, weight=0.5
        )

        entities_mixed = {
            "invalid1": entity_invalid1,
            "invalid2": entity_invalid2,
            "invalid3": entity_invalid3,
            "valid": entity_valid,
        }

        result_mixed = bayesian_probability(entities_mixed, prior=0.5)
        result_valid_only = bayesian_probability({"valid": entity_valid}, prior=0.5)

        # Should behave exactly the same as if only valid entity was present
        assert abs(result_mixed - result_valid_only) < 1e-6

    def test_bayesian_numerical_overflow(self) -> None:
        """Test Bayesian probability handles extreme probabilities without overflow.

        Verifies that probabilities very close to 0 or 1 don't cause numerical issues.
        """
        entity = _create_mock_entity(
            evidence=True,
            prob_given_true=0.999999,  # Very close to 1
            prob_given_false=0.000001,  # Very close to 0
            weight=1.0,
        )

        entities = {"entity1": entity}

        result = bayesian_probability(entities, prior=0.5)
        assert 0.0 <= result <= 1.0
        assert not (math.isnan(result) or math.isinf(result))
        # With such extreme probabilities, result should be very high
        assert result > 0.9

    def test_bayesian_all_invalid_entities(self) -> None:
        """Test that when all entities are invalid, function returns clamped prior."""
        # All entities with invalid likelihoods
        entity1 = _create_mock_entity(
            evidence=True, prob_given_true=0.0, prob_given_false=0.1, weight=0.5
        )  # Invalid: prob_given_true = 0
        entity2 = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=1.0, weight=0.5
        )  # Invalid: prob_given_false = 1

        entities = {"entity1": entity1, "entity2": entity2}

        # Should return clamped prior when all entities are invalid
        prior = combine_priors(0.3, 0.7)
        result = bayesian_probability(entities, prior=prior)
        assert abs(result - clamp_probability(prior)) < 1e-6

    def test_bayesian_decay_interpolation(self) -> None:
        """Test that decay interpolation correctly affects probability calculation.

        When evidence=False and is_decaying=True:
        - effective_evidence becomes True (because value or is_decaying)
        - Uses original likelihoods (0.8, 0.1), NOT inverse
        - Applies decay interpolation to those original likelihoods

        With decay factor 0.5: p_t = 0.5 + (0.8 - 0.5) * 0.5 = 0.65, p_f = 0.5 + (0.1 - 0.5) * 0.5 = 0.3
        With decay factor 0.0: p_t = 0.5, p_f = 0.5 (neutral)
        """
        entity_no_decay = _create_mock_entity(
            evidence=False, prob_given_true=0.8, prob_given_false=0.1, weight=1.0
        )
        entity_half_decay = _create_mock_entity(
            evidence=False,
            prob_given_true=0.8,
            prob_given_false=0.1,
            weight=1.0,
            is_decaying=True,
            decay_factor=0.5,
        )
        entity_full_decay = _create_mock_entity(
            evidence=False,
            prob_given_true=0.8,
            prob_given_false=0.1,
            weight=1.0,
            is_decaying=True,
            decay_factor=0.0,
        )

        prior = 0.5
        result_no_decay = bayesian_probability({"entity": entity_no_decay}, prior=prior)
        result_half_decay = bayesian_probability(
            {"entity": entity_half_decay}, prior=prior
        )
        result_full_decay = bayesian_probability(
            {"entity": entity_full_decay}, prior=prior
        )

        # entity_no_decay: uses inverse likelihoods (0.2, 0.9) → suggests NOT occupied → low probability
        # entity_full_decay: uses original likelihoods with full decay → neutral (0.5, 0.5) → close to prior
        # entity_half_decay: uses original likelihoods with half decay → suggests occupied → higher probability
        assert abs(result_full_decay - prior) < 0.1
        assert result_no_decay < result_full_decay
        assert result_full_decay < result_half_decay
        assert result_half_decay > prior

    def test_bayesian_inactive_sensor_inverse_likelihoods(self) -> None:
        """Test that inactive sensors correctly use inverse likelihoods.

        When a sensor is inactive, it uses (1 - prob_given_true, 1 - prob_given_false).
        This means an inactive sensor with high prob_given_true suggests not occupied.
        """
        # Entity with prob_given_true=0.8, prob_given_false=0.1
        # When inactive, uses p_t=0.2, p_f=0.9 (inverse)
        entity_active = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=0.1, weight=1.0
        )
        entity_inactive = _create_mock_entity(
            evidence=False, prob_given_true=0.8, prob_given_false=0.1, weight=1.0
        )

        prior = 0.5
        result_active = bayesian_probability({"entity": entity_active}, prior=prior)
        result_inactive = bayesian_probability({"entity": entity_inactive}, prior=prior)

        # Active sensor should increase probability, inactive should decrease it
        assert result_active > prior
        assert result_inactive < prior
        assert result_active > result_inactive

    def test_bayesian_motion_sensor_with_inactive_others(self) -> None:
        """Test that active motion sensor dominates inactive sensors.

        Verifies that a strong active sensor (motion) increases probability significantly
        even when other sensors are inactive.
        """
        # Motion sensor: active, high reliability
        motion = _create_mock_entity(
            evidence=True, prob_given_true=0.95, prob_given_false=0.02, weight=1.0
        )

        # Other sensors: inactive
        media = _create_mock_entity(
            evidence=False, prob_given_true=0.65, prob_given_false=0.02, weight=0.85
        )
        door = _create_mock_entity(
            evidence=False, prob_given_true=0.2, prob_given_false=0.02, weight=0.3
        )
        window = _create_mock_entity(
            evidence=False, prob_given_true=0.2, prob_given_false=0.02, weight=0.2
        )

        entities = {"motion": motion, "media": media, "door": door, "window": window}

        prior = 0.3
        result = bayesian_probability(entities, prior=prior)

        # Motion sensor's strong positive evidence should dominate
        assert result > prior
        assert result > 0.5  # Should be significantly higher than prior
        # Should be higher than prior alone
        assert result > clamp_probability(prior)

    def test_bayesian_unavailable_sensors_skipped(self) -> None:
        """Test that unavailable sensors (evidence=None) are correctly skipped."""
        inactive = _create_mock_entity(
            evidence=False, prob_given_true=0.8, prob_given_false=0.1, weight=1.0
        )

        unavailable = _create_mock_entity(
            evidence=None, prob_given_true=0.8, prob_given_false=0.1, weight=1.0
        )

        entities = {"inactive": inactive, "unavailable": unavailable}

        # Should behave exactly the same as if only inactive sensor was present
        result_with_unavailable = bayesian_probability(entities, prior=0.5)
        result_without_unavailable = bayesian_probability(
            {"inactive": inactive}, prior=0.5
        )

        assert abs(result_with_unavailable - result_without_unavailable) < 1e-6

    def test_bayesian_evidence_true_with_decay_active(self) -> None:
        """Test that decay is not applied when evidence is True.

        When evidence=True, entity.decay_factor property returns 1.0, preventing
        decay interpolation even if is_decaying=True (inconsistent state).
        """
        entity_with_evidence = _create_mock_entity(
            evidence=True,
            prob_given_true=0.8,
            prob_given_false=0.1,
            weight=1.0,
            is_decaying=True,  # Inconsistent state
            decay_factor=0.5,
        )

        entity_no_decay = _create_mock_entity(
            evidence=True, prob_given_true=0.8, prob_given_false=0.1, weight=1.0
        )

        prior = 0.5
        result_with_decay_flag = bayesian_probability(
            {"entity": entity_with_evidence}, prior=prior
        )
        result_no_decay = bayesian_probability({"entity": entity_no_decay}, prior=prior)

        # Both should produce same result since evidence=True prevents decay
        assert abs(result_with_decay_flag - result_no_decay) < 1e-6
        assert result_with_decay_flag > 0.5

    def test_bayesian_continuous_sensor_inactive_state(self) -> None:
        """Test that continuous sensors use get_likelihoods() for inactive states.

        Continuous sensors (Gaussian densities) should call get_likelihoods() even
        when evidence=False, not use inverse probabilities like binary sensors.
        """
        entity = _create_mock_entity(
            evidence=False,
            prob_given_true=0.8,
            prob_given_false=0.2,
            weight=1.0,
            is_continuous=True,
        )
        # Mock get_likelihoods to return densities for inactive state
        entity.get_likelihoods = Mock(return_value=(0.3, 0.7))

        entities = {"entity1": entity}

        result = bayesian_probability(entities, prior=0.5)

        # Should use get_likelihoods() for inactive continuous sensor
        entity.get_likelihoods.assert_called_once()
        assert 0.0 <= result <= 1.0
        # Result should reflect the densities returned by get_likelihoods()
        assert result < 0.5  # Lower density for true suggests not occupied

    def test_bayesian_continuous_sensor_unavailable_state(self) -> None:
        """Test that unavailable continuous sensors are skipped.

        Unavailable sensors (evidence=None) are skipped unless decaying,
        so get_likelihoods() should not be called.
        """
        entity = _create_mock_entity(
            evidence=None,
            prob_given_true=0.8,
            prob_given_false=0.2,
            weight=1.0,
            is_continuous=True,
        )
        entity.get_likelihoods = Mock(return_value=(0.5, 0.5))

        entities = {"entity1": entity}

        prior = 0.5
        result = bayesian_probability(entities, prior=prior)

        # Should return prior since entity is skipped
        assert abs(result - clamp_probability(prior)) < 1e-6
        # get_likelihoods should not be called since entity is skipped
        entity.get_likelihoods.assert_not_called()

    def test_bayesian_gaussian_std_zero_edge_case(self) -> None:
        """Test that continuous sensors handle edge cases gracefully.

        Verifies that get_likelihoods() returns valid densities that don't cause
        numerical issues in the calculation.
        """
        entity = _create_mock_entity(
            evidence=True,
            prob_given_true=0.8,
            prob_given_false=0.2,
            weight=1.0,
            is_continuous=True,
        )
        # Mock get_likelihoods to return valid densities
        entity.get_likelihoods = Mock(return_value=(0.6, 0.4))

        entities = {"entity1": entity}

        result = bayesian_probability(entities, prior=0.5)

        # Should use get_likelihoods() and produce valid result
        entity.get_likelihoods.assert_called_once()
        assert 0.0 <= result <= 1.0
        assert not (math.isnan(result) or math.isinf(result))
        assert result > 0.0

    @pytest.mark.parametrize(
        "return_value",
        [
            (float("nan"), 0.5),
            (0.5, float("inf")),
        ],
        ids=["NaN", "inf"],
    )
    def test_bayesian_get_likelihoods_invalid_fallback(self, return_value) -> None:
        """Test that get_likelihoods() returning NaN/inf falls back to static values."""
        entity = _create_mock_entity(
            evidence=True,
            prob_given_true=0.8,
            prob_given_false=0.1,
            weight=1.0,
            is_continuous=True,
        )
        # Mock get_likelihoods to return invalid value
        entity.get_likelihoods = Mock(return_value=return_value)

        entities = {"entity1": entity}

        result = bayesian_probability(entities, prior=0.5)

        # Should fallback to static values and produce valid result
        assert 0.0 <= result <= 1.0
        assert not (math.isnan(result) or math.isinf(result))
        entity.get_likelihoods.assert_called_once()
        # Result should be based on static prob_given_true/prob_given_false
        assert result > 0.5


class TestMapBinaryStateToSemantic:
    """Test map_binary_state_to_semantic function.

    Tests mapping of binary sensor states ('on'/'off') to semantic states
    ('open'/'closed') for door and window sensors.
    """

    @pytest.mark.parametrize(
        ("input_state", "active_states", "expected_result", "description"),
        [
            ("off", ["closed"], "closed", "door closed (off -> closed)"),
            ("on", ["open"], "open", "door open (on -> open)"),
            ("on", ["open"], "open", "window open (on -> open)"),
            ("off", ["closed"], "closed", "window closed (off -> closed)"),
        ],
    )
    def test_map_binary_state_to_semantic(
        self, input_state, active_states, expected_result, description
    ):
        """Test mapping binary states to semantic states."""
        result = map_binary_state_to_semantic(input_state, active_states)
        assert result == expected_result

    @pytest.mark.parametrize(
        ("input_state", "active_states", "expected_result"),
        [
            ("off", ["on"], "off"),  # No mapping when semantic not in active_states
            ("on", ["off"], "on"),  # No mapping when semantic not in active_states
        ],
    )
    def test_no_mapping_when_semantic_not_present(
        self, input_state, active_states, expected_result
    ):
        """Test that no mapping occurs when semantic states not in active_states."""
        result = map_binary_state_to_semantic(input_state, active_states)
        assert result == expected_result

    def test_mapping_preserves_other_states(self):
        """Test that non-binary states are preserved."""
        result = map_binary_state_to_semantic("playing", ["playing", "paused"])
        assert result == "playing"
