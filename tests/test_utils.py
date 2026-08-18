"""Tests for utils module."""

import math
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.const import MAX_PROBABILITY, MIN_PROBABILITY
from custom_components.area_occupancy.data.entity_type import DEFAULT_TYPES, InputType
from custom_components.area_occupancy.utils import (
    apply_activity_boost,
    clamp_probability,
    combine_priors,
    combined_probability,
    environmental_confidence,
    format_float,
    format_percentage,
    logit,
    map_binary_state_to_semantic,
    presence_probability,
    sigmoid,
    sigmoid_probability,
)


def _create_mock_entity(
    evidence: bool | None = True,
    prob_given_true: float = 0.8,
    prob_given_false: float = 0.1,
    weight: float = 1.0,
    is_decaying: bool = False,
    decay_factor: float = 1.0,
    is_continuous: bool = False,
    input_type: InputType = InputType.MOTION,
    effective_weight: float | None = None,
) -> Mock:
    """Create a mock entity for testing the sigmoid probability model.

    Args:
        evidence: Entity evidence state (True/False/None)
        prob_given_true: Probability given true
        prob_given_false: Probability given false
        weight: Entity weight
        is_decaying: Whether entity is decaying
        decay_factor: Decay factor (0.0 to 1.0)
        is_continuous: Whether entity uses continuous likelihood
        input_type: The type of input (motion, temperature, etc.)
        effective_weight: Effective weight (defaults to weight if not specified)

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
    # effective_weight defaults to weight (full information gain).
    entity.effective_weight = (
        effective_weight if effective_weight is not None else weight
    )
    entity.is_continuous_likelihood = is_continuous
    entity.type = Mock()
    entity.type.input_type = input_type
    entity.type.strength_multiplier = DEFAULT_TYPES.get(input_type, {}).get(
        "strength_multiplier", 2.0
    )
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

    def test_format_float_custom_precision(self) -> None:
        """Test float formatting with custom precision settings."""
        # Test default precision (2)
        assert format_float(1.234567) == 1.23

        # Test precision = 1
        assert format_float(1.234567, 1) == 1.2
        assert format_float(1.2789, 1) == 1.3

        # Test precision = 0
        assert format_float(1.234567, 0) == 1.0
        assert format_float(1.789, 0) == 2.0
        assert format_float(0.0, 0) == 0.0

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

        # With default time_weight (0.4), result should be closer to area_prior
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


class TestSigmoidFunctions:
    """Test sigmoid-based probability functions.

    Tests verify the weighted sigmoid probability model including:
    - Basic sigmoid and logit mathematical properties
    - Additive contributions from multiple sensors
    - Correlation weight integration
    - Decay factor handling
    - Presence vs environmental sensor separation
    """

    def test_sigmoid_basic_properties(self) -> None:
        """Test that sigmoid has correct mathematical properties."""
        # sigmoid(0) = 0.5
        assert abs(sigmoid(0) - 0.5) < 1e-6

        # sigmoid is bounded (0, 1)
        assert 0 < sigmoid(-10) < 0.5
        assert 0.5 < sigmoid(10) < 1

        # sigmoid is monotonically increasing
        assert sigmoid(-2) < sigmoid(-1) < sigmoid(0) < sigmoid(1) < sigmoid(2)

        # Symmetry: sigmoid(-x) = 1 - sigmoid(x)
        for x in [-3, -1, 0, 1, 3]:
            assert abs(sigmoid(-x) - (1 - sigmoid(x))) < 1e-6

    def test_sigmoid_numerical_stability(self) -> None:
        """Test sigmoid handles extreme values without overflow."""
        # Very large negative values should approach 0
        result_neg = sigmoid(-100)
        assert result_neg >= 0  # May be exactly 0 due to floating point
        assert result_neg < 0.01
        assert not math.isnan(result_neg)
        assert not math.isinf(result_neg)

        # Very large positive values should approach 1
        result_pos = sigmoid(100)
        assert result_pos <= 1  # May be exactly 1 due to floating point
        assert result_pos > 0.99
        assert not math.isnan(result_pos)
        assert not math.isinf(result_pos)

    def test_logit_basic_properties(self) -> None:
        """Test that logit has correct mathematical properties."""
        # logit(0.5) = 0
        assert abs(logit(0.5)) < 1e-6

        # logit is the inverse of sigmoid
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(sigmoid(logit(p)) - p) < 1e-6

        # logit is monotonically increasing
        assert logit(0.2) < logit(0.5) < logit(0.8)

    def test_logit_clamping(self) -> None:
        """Test that logit clamps values to valid range."""
        # Values at or beyond bounds are clamped to MIN/MAX_PROBABILITY
        result_zero = logit(0.0)
        result_one = logit(1.0)

        # Should produce finite values (not -inf or +inf)
        assert not math.isinf(result_zero)
        assert not math.isinf(result_one)

    def test_sigmoid_probability_empty_entities(self) -> None:
        """Test that sigmoid_probability returns prior when no entities."""
        prior = 0.3
        result = sigmoid_probability({}, prior=prior)
        assert abs(result - clamp_probability(prior)) < 1e-6

    def test_sigmoid_probability_single_active_sensor(self) -> None:
        """Test single active motion sensor significantly increases probability."""
        motion = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )

        prior = 0.3
        result = sigmoid_probability({"motion": motion}, prior=prior)

        # Active motion sensor should significantly increase probability
        assert result > prior
        assert result > 0.5  # Should be well above neutral

    def test_sigmoid_probability_inactive_sensor_no_penalty(self) -> None:
        """Test that inactive sensors don't penalize probability (OR-like behavior)."""
        motion = _create_mock_entity(
            evidence=False,  # Inactive
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )

        prior = 0.5
        result = sigmoid_probability({"motion": motion}, prior=prior)

        # Inactive sensor should NOT decrease probability (key difference from Bayesian)
        # With no active sensors, result should be close to prior
        assert abs(result - clamp_probability(prior)) < 0.1

    def test_sigmoid_probability_multiple_sensors_additive(self) -> None:
        """Test that multiple active sensors are additive (OR-like, not AND-like)."""
        motion1 = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )
        motion2 = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )
        motion3 = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )

        prior = 0.3
        result_1 = sigmoid_probability({"m1": motion1}, prior=prior)
        result_2 = sigmoid_probability({"m1": motion1, "m2": motion2}, prior=prior)
        result_3 = sigmoid_probability(
            {"m1": motion1, "m2": motion2, "m3": motion3}, prior=prior
        )

        # Each additional sensor should increase probability (or hit ceiling)
        assert result_1 > prior
        assert result_2 >= result_1
        assert result_3 >= result_2

    def test_sigmoid_probability_with_decay(self) -> None:
        """Test that decaying sensors contribute proportionally to decay factor."""
        # Sensor with full evidence
        motion_active = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )

        # Sensor that has decayed to 50%
        motion_decay_50 = _create_mock_entity(
            evidence=False,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            is_decaying=True,
            decay_factor=0.5,
            input_type=InputType.MOTION,
        )

        # Sensor that has fully decayed
        motion_decay_0 = _create_mock_entity(
            evidence=False,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            is_decaying=True,
            decay_factor=0.0,
            input_type=InputType.MOTION,
        )

        prior = 0.3
        result_active = sigmoid_probability({"m": motion_active}, prior=prior)
        result_decay_50 = sigmoid_probability({"m": motion_decay_50}, prior=prior)
        result_decay_0 = sigmoid_probability({"m": motion_decay_0}, prior=prior)

        # Active > 50% decay > 0% decay
        assert result_active > result_decay_50 > result_decay_0
        # 0% decay should be close to prior
        assert abs(result_decay_0 - clamp_probability(prior)) < 0.1

    def test_sigmoid_probability_with_correlations(self) -> None:
        """Test that correlation weights scale contributions."""
        motion = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )

        prior = 0.3

        # No correlation data (default 1.0)
        result_no_corr = sigmoid_probability({"motion": motion}, prior=prior)

        # Strong correlation (1.0)
        result_strong_corr = sigmoid_probability(
            {"motion": motion}, prior=prior, correlations={"motion": 1.0}
        )

        # Weak correlation (0.3)
        result_weak_corr = sigmoid_probability(
            {"motion": motion}, prior=prior, correlations={"motion": 0.3}
        )

        # No correlation should equal strong correlation
        assert abs(result_no_corr - result_strong_corr) < 1e-6

        # Weak correlation should contribute less
        assert result_strong_corr > result_weak_corr
        assert result_weak_corr > clamp_probability(prior)

    def test_sigmoid_probability_zero_weight_ignored(self) -> None:
        """Test that zero-weight sensors are ignored."""
        motion_zero = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=0.0,
            input_type=InputType.MOTION,
        )

        motion_normal = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )

        prior = 0.5
        result_zero = sigmoid_probability({"m": motion_zero}, prior=prior)
        result_normal = sigmoid_probability({"m": motion_normal}, prior=prior)

        # Zero weight should return prior
        assert abs(result_zero - clamp_probability(prior)) < 1e-6
        # Normal weight should be different
        assert result_normal > result_zero

    def test_sigmoid_probability_motion_strength_multiplier(self) -> None:
        """Test that motion sensors (multiplier 3.0) produce higher results than 2.0."""
        # Motion sensor with default multiplier (3.0)
        motion_high = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )

        # Same sensor but with multiplier overridden to 2.0
        motion_low = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )
        motion_low.type.strength_multiplier = 2.0

        prior = 0.15
        result_high = sigmoid_probability({"m": motion_high}, prior=prior)
        result_low = sigmoid_probability({"m": motion_low}, prior=prior)

        # 3.0 multiplier should produce higher probability than 2.0
        assert result_high > result_low
        # Both should be above prior
        assert result_high > prior
        assert result_low > prior


class TestPresenceEnvironmentalSplit:
    """Test presence and environmental probability separation."""

    def test_presence_probability_filters_to_presence_types(self) -> None:
        """Test that presence_probability only considers presence sensor types."""
        motion = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )
        temperature = _create_mock_entity(
            evidence=True,
            prob_given_true=0.09,
            prob_given_false=0.01,
            weight=0.1,
            input_type=InputType.TEMPERATURE,
        )

        entities = {"motion": motion, "temperature": temperature}
        prior = 0.3

        result = presence_probability(entities, prior=prior)
        result_motion_only = sigmoid_probability({"motion": motion}, prior=prior)

        # Should only use motion (presence type), ignoring temperature
        assert abs(result - result_motion_only) < 1e-6

    def test_presence_probability_no_presence_sensors(self) -> None:
        """Test presence_probability with no presence sensors returns reduced prior."""
        temperature = _create_mock_entity(
            evidence=True,
            prob_given_true=0.09,
            prob_given_false=0.01,
            weight=0.1,
            input_type=InputType.TEMPERATURE,
        )

        prior = 0.6
        result = presence_probability({"temp": temperature}, prior=prior)

        # Should return prior * 0.5 (reduced due to no presence sensors)
        expected = clamp_probability(prior * 0.5)
        assert abs(result - expected) < 1e-6

    def test_environmental_confidence_filters_to_env_types(self) -> None:
        """Test that environmental_confidence only considers environmental types."""
        motion = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )
        temperature = _create_mock_entity(
            evidence=True,
            prob_given_true=0.09,
            prob_given_false=0.01,
            weight=0.1,
            input_type=InputType.TEMPERATURE,
        )

        entities = {"motion": motion, "temperature": temperature}

        result = environmental_confidence(entities)
        result_temp_only = sigmoid_probability({"temperature": temperature}, prior=0.5)

        # Should only use temperature (environmental type), ignoring motion
        assert abs(result - result_temp_only) < 1e-6

    def test_environmental_confidence_no_env_sensors(self) -> None:
        """Test environmental_confidence with no env sensors returns 0.5 (neutral)."""
        motion = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )

        result = environmental_confidence({"motion": motion})

        # Should return 0.5 (neutral) when no environmental sensors
        assert result == 0.5

    def test_wifi_clients_counted_in_presence_not_environmental(self) -> None:
        """WIFI_CLIENTS is a presence-channel type, not environmental (issue #515).

        This is the load-bearing assertion for the feature: wifi_clients must
        be picked up by presence_probability() and explicitly excluded from
        environmental_confidence(), since it was deliberately added to
        PRESENCE_INPUT_TYPES rather than ENVIRONMENTAL_INPUT_TYPES.
        """
        wifi_clients = _create_mock_entity(
            evidence=True,
            prob_given_true=0.3,
            prob_given_false=0.03,
            weight=0.35,
            input_type=InputType.WIFI_CLIENTS,
        )
        temperature = _create_mock_entity(
            evidence=True,
            prob_given_true=0.09,
            prob_given_false=0.01,
            weight=0.1,
            input_type=InputType.TEMPERATURE,
        )

        entities = {"wifi_clients": wifi_clients, "temperature": temperature}
        prior = 0.4

        # presence_probability() must include wifi_clients and ignore temperature
        presence_result = presence_probability(entities, prior=prior)
        presence_wifi_only = sigmoid_probability(
            {"wifi_clients": wifi_clients}, prior=prior
        )
        assert abs(presence_result - presence_wifi_only) < 1e-6

        # environmental_confidence() must ignore wifi_clients and only use temperature
        env_result = environmental_confidence(entities)
        env_temp_only = sigmoid_probability({"temperature": temperature}, prior=0.5)
        assert abs(env_result - env_temp_only) < 1e-6

        # Sanity: a wifi_clients-only entity dict must NOT be neutral under
        # environmental_confidence() being falsely non-neutral - it must be
        # exactly neutral (0.5) since wifi_clients contributes nothing there.
        assert environmental_confidence({"wifi_clients": wifi_clients}) == 0.5


class TestCombinedProbability:
    """Test combined probability function."""

    def test_combined_probability_equal_inputs(self) -> None:
        """Test combined probability with equal presence and environmental."""
        result = combined_probability(presence=0.5, environmental=0.5)
        assert abs(result - 0.5) < 1e-6

    def test_combined_probability_presence_dominant(self) -> None:
        """Test that presence dominates the environmental adjustment."""
        # High presence, low environmental
        result_high_presence = combined_probability(presence=0.8, environmental=0.2)
        # Low presence, high environmental
        result_high_env = combined_probability(presence=0.2, environmental=0.8)

        # High presence should result in higher overall probability
        assert result_high_presence > result_high_env

        # The environmental term is a damped logit contribution, so presence
        # decides which side of 0.5 the result lands on.
        assert result_high_presence > 0.5

    def test_combined_probability_environmental_influence(self) -> None:
        """Test that environmental has some influence on result."""
        result_env_low = combined_probability(presence=0.5, environmental=0.2)
        result_env_high = combined_probability(presence=0.5, environmental=0.8)

        # Environmental still moves the result via its damped logit contribution
        assert result_env_high > result_env_low

    def test_combined_probability_clamping(self) -> None:
        """Test that combined probability is properly clamped."""
        result_extreme_high = combined_probability(presence=0.99, environmental=0.99)
        result_extreme_low = combined_probability(presence=0.01, environmental=0.01)

        # Should be within MIN/MAX bounds
        assert MIN_PROBABILITY <= result_extreme_high <= MAX_PROBABILITY
        assert MIN_PROBABILITY <= result_extreme_low <= MAX_PROBABILITY

    def test_combined_probability_neutral_env_returns_presence(self) -> None:
        """Neutral environmental (0.5) must leave presence untouched.

        Area._base_probability() short-circuits to presence when there are no
        environmental sensors (env == 0.5 exactly). This asserts the formula
        agrees with that short-circuit, so the two branches stay continuous.
        """
        for presence in (0.01, 0.2, 0.5, 0.8, 0.945, 0.99):
            assert combined_probability(presence, 0.5) == pytest.approx(
                presence, abs=1e-9
            )

    def test_combined_probability_supporting_env_never_lowers(self) -> None:
        """Supporting environmental evidence must never lower the presence estimate."""
        # Environmental contributions are non-negative, so environmental >= 0.5
        # always. Averaging used to pull confident presence back toward 0.5,
        # so adding a supporting environmental sensor reduced the probability.
        for presence in (0.6, 0.8, 0.9, 0.95, 0.99):
            for environmental in (0.5, 0.505, 0.6, 0.9):
                combined = combined_probability(presence, environmental)
                # 1e-9 absorbs logit/sigmoid float round-trip error.
                assert combined >= min(presence, MAX_PROBABILITY) - 1e-9


class TestSigmoidModelBehavior:
    """Verify characteristic behaviors of the sigmoid probability model."""

    def test_inactive_sensors_no_penalty_sigmoid(self) -> None:
        """Verify sigmoid doesn't penalize inactive sensors."""
        active_sensor = _create_mock_entity(
            evidence=True,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MOTION,
        )
        inactive_sensor = _create_mock_entity(
            evidence=False,
            prob_given_true=0.95,
            prob_given_false=0.02,
            weight=1.0,
            input_type=InputType.MEDIA,
        )

        prior = 0.5
        entities = {"active": active_sensor, "inactive": inactive_sensor}

        result_sigmoid = sigmoid_probability(entities, prior=prior)
        result_active_only = sigmoid_probability({"active": active_sensor}, prior=prior)

        # In sigmoid model, inactive sensor should NOT decrease probability
        # Result should be very similar to active-only
        assert abs(result_sigmoid - result_active_only) < 0.1

    def test_full_dynamic_range_sigmoid(self) -> None:
        """Test that sigmoid achieves fuller dynamic range than typical Bayesian."""
        # Create multiple strongly active sensors
        sensors = {}
        for i in range(3):
            sensors[f"motion_{i}"] = _create_mock_entity(
                evidence=True,
                prob_given_true=0.95,
                prob_given_false=0.02,
                weight=1.0,
                input_type=InputType.MOTION,
            )

        prior = 0.3
        result = sigmoid_probability(sensors, prior=prior)

        # With 3 strong active sensors, should achieve high probability
        assert result > 0.9  # Should approach upper range


class TestApplyActivityBoost:
    """Test apply_activity_boost function."""

    def test_zero_boost_returns_base(self) -> None:
        """Zero boost should return base probability unchanged."""
        base = 0.5
        result = apply_activity_boost(base, activity_boost=0.0, activity_confidence=1.0)
        assert result == base

    def test_zero_confidence_returns_base(self) -> None:
        """Zero confidence should return base probability unchanged."""
        base = 0.5
        result = apply_activity_boost(base, activity_boost=1.5, activity_confidence=0.0)
        assert result == base

    def test_showering_boost_from_half(self) -> None:
        """Showering boost (1.5) at full confidence from base=0.5 should increase significantly."""
        result = apply_activity_boost(0.5, activity_boost=1.5, activity_confidence=1.0)
        # logit(0.5)=0, 0+1.5=1.5, sigmoid(1.5)≈0.82
        assert result == pytest.approx(0.82, abs=0.02)

    def test_watching_tv_boost(self) -> None:
        """Watching TV boost (1.2) at full confidence from base=0.5."""
        result = apply_activity_boost(0.5, activity_boost=1.2, activity_confidence=1.0)
        # logit(0.5)=0, 0+1.2=1.2, sigmoid(1.2)≈0.77
        assert result == pytest.approx(0.77, abs=0.02)

    def test_partial_confidence_scales_boost(self) -> None:
        """Partial confidence should scale the boost proportionally."""
        full = apply_activity_boost(0.5, activity_boost=1.5, activity_confidence=1.0)
        half = apply_activity_boost(0.5, activity_boost=1.5, activity_confidence=0.5)
        # Half confidence → half effective boost → smaller increase
        assert half < full
        assert half > 0.5  # Still a boost

    def test_boost_from_high_base(self) -> None:
        """Boost from a high base probability should still increase but clamped."""
        result = apply_activity_boost(0.9, activity_boost=1.5, activity_confidence=1.0)
        assert result > 0.9
        assert result <= MAX_PROBABILITY

    def test_boost_from_low_base(self) -> None:
        """Boost from a low base should still increase probability."""
        result = apply_activity_boost(0.3, activity_boost=1.0, activity_confidence=1.0)
        assert result > 0.3

    def test_result_always_clamped(self) -> None:
        """Result should always be within MIN_PROBABILITY to MAX_PROBABILITY."""
        result = apply_activity_boost(0.99, activity_boost=5.0, activity_confidence=1.0)
        assert MIN_PROBABILITY <= result <= MAX_PROBABILITY
