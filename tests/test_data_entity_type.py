"""Tests for data.entity_type module."""

import logging

import pytest

from custom_components.area_occupancy.data.entity_type import (
    DEFAULT_TYPES,
    EntityType,
    InputType,
)
from homeassistant.const import STATE_ON


class TestInputType:
    """Test InputType enum."""

    @pytest.mark.parametrize(
        ("input_type", "expected_value"),
        [(input_type, input_type.value) for input_type in InputType],
    )
    def test_input_type_values(self, input_type, expected_value) -> None:
        """Test that InputType has expected values."""
        assert input_type.value == expected_value


class TestEntityType:
    """Test EntityType class."""

    def test_initialization_with_states(self) -> None:
        """Test EntityType initialization with active_states."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=0.25,
            prob_given_false=0.05,
            active_states=[STATE_ON],
        )

        assert entity_type.weight == 0.8
        assert entity_type.prob_given_true == 0.25
        assert entity_type.prob_given_false == 0.05
        assert entity_type.active_states == [STATE_ON]
        assert entity_type.active_range is None

    def test_initialization_with_range(self) -> None:
        """Test EntityType initialization with active_range."""
        entity_type = EntityType(
            input_type=InputType.ENVIRONMENTAL,
            weight=0.3,
            prob_given_true=0.09,
            prob_given_false=0.01,
            active_range=(0.0, 0.2),
        )

        assert entity_type.active_range == (0.0, 0.2)
        assert entity_type.active_states is None

    @pytest.mark.parametrize(
        ("input_type", "expected_config"),
        [(input_type, config) for input_type, config in DEFAULT_TYPES.items()],
    )
    def test_initialization_with_defaults(self, input_type, expected_config) -> None:
        """Test initialization for different input types with default values."""
        entity_type = EntityType(input_type)

        assert entity_type.weight == expected_config["weight"]
        assert entity_type.prob_given_true == expected_config["prob_given_true"]
        assert entity_type.prob_given_false == expected_config["prob_given_false"]
        assert entity_type.active_states == expected_config["active_states"]
        assert entity_type.active_range == expected_config["active_range"]
        assert entity_type.strength_multiplier == expected_config["strength_multiplier"]

    @pytest.mark.parametrize(
        (
            "override_type",
            "input_type",
            "override_value",
            "expected_weight",
            "expected_states",
            "expected_range",
        ),
        [
            (
                "weight",
                InputType.MOTION,
                {"weight": 0.9},
                0.9,
                DEFAULT_TYPES[InputType.MOTION]["active_states"],
                None,
            ),
            (
                "active_states",
                InputType.MOTION,
                {"active_states": ["on", "detected"]},
                DEFAULT_TYPES[InputType.MOTION]["weight"],
                ["on", "detected"],
                None,
            ),
            (
                "active_range",
                InputType.ENVIRONMENTAL,
                {"weight": 0.2, "active_range": (0.1, 0.3)},
                0.2,
                None,
                (0.1, 0.3),
            ),
        ],
    )
    def test_initialization_with_overrides(
        self,
        override_type,
        input_type,
        override_value,
        expected_weight,
        expected_states,
        expected_range,
    ) -> None:
        """Test initialization with various parameter overrides."""
        entity_type = EntityType(input_type, **override_value)

        assert entity_type.weight == expected_weight
        assert entity_type.active_states == expected_states
        assert entity_type.active_range == expected_range

    @pytest.mark.parametrize(
        ("test_case", "params", "expected_error"),
        [
            # Mutually exclusive parameters
            (
                "both_active_states_and_range",
                {
                    "input_type": InputType.MOTION,
                    "weight": 0.8,
                    "prob_given_true": 0.25,
                    "prob_given_false": 0.05,
                    "active_states": [STATE_ON],
                    "active_range": (0.0, 1.0),
                },
                "Cannot provide both active_states and active_range",
            ),
            # Weight validation errors
            (
                "invalid_weight_negative",
                {
                    "input_type": InputType.MOTION,
                    "weight": -0.1,
                    "active_states": [STATE_ON],
                },
                "Invalid weight for motion: -0.1",
            ),
            (
                "invalid_weight_too_large",
                {
                    "input_type": InputType.MOTION,
                    "weight": 1.5,
                    "active_states": [STATE_ON],
                },
                "Invalid weight for motion: 1.5",
            ),
            (
                "invalid_weight_string",
                {
                    "input_type": InputType.MOTION,
                    "weight": "0.8",
                    "active_states": [STATE_ON],
                },
                "Invalid weight for motion: 0.8",
            ),
            # active_states validation errors
            (
                "invalid_states_not_list",
                {
                    "input_type": InputType.MOTION,
                    "active_states": "invalid",
                },
                "Invalid active states for motion: invalid",
            ),
            (
                "invalid_states_non_string",
                {
                    "input_type": InputType.MOTION,
                    "active_states": [123, "on"],
                },
                "Invalid active states for motion:",
            ),
            (
                "invalid_states_mixed_types",
                {
                    "input_type": InputType.MOTION,
                    "active_states": ["on", 456, None],
                },
                "Invalid active states for motion:",
            ),
            # active_range validation errors
            (
                "invalid_range_not_tuple",
                {
                    "input_type": InputType.ENVIRONMENTAL,
                    "active_range": "invalid",
                },
                "Invalid active range for environmental: invalid",
            ),
            (
                "invalid_range_wrong_length_0",
                {
                    "input_type": InputType.ENVIRONMENTAL,
                    "active_range": (),
                },
                "Invalid active range for environmental:",
            ),
            (
                "invalid_range_wrong_length_1",
                {
                    "input_type": InputType.ENVIRONMENTAL,
                    "active_range": (0.0,),
                },
                "Invalid active range for environmental:",
            ),
            (
                "invalid_range_wrong_length_3",
                {
                    "input_type": InputType.ENVIRONMENTAL,
                    "active_range": (0.0, 0.2, 0.5),
                },
                "Invalid active range for environmental:",
            ),
        ],
    )
    def test_initialization_validation_errors(
        self, test_case, params, expected_error
    ) -> None:
        """Test initialization with invalid parameter values."""
        with pytest.raises(ValueError, match=expected_error):
            EntityType(**params)

    @pytest.mark.parametrize(
        ("input_type", "weight", "expected_states", "expected_range"),
        [
            (
                InputType.MOTION,
                0.9,
                DEFAULT_TYPES[InputType.MOTION]["active_states"],
                None,
            ),
            (
                InputType.TEMPERATURE,
                0.2,
                None,
                DEFAULT_TYPES[InputType.TEMPERATURE]["active_range"],
            ),
        ],
    )
    def test_initialization_with_empty_states_list(
        self, input_type, weight, expected_states, expected_range
    ) -> None:
        """Test that empty active_states list uses defaults instead of crashing."""
        entity_type = EntityType(
            input_type,
            weight=weight,
            active_states=[],  # Empty list - should use defaults
        )

        # Should use defaults from DEFAULT_TYPES, not empty list
        assert entity_type.weight == weight  # Weight override should still work
        assert entity_type.active_states == expected_states
        assert entity_type.active_range == expected_range

    @pytest.mark.parametrize(
        "weight",
        [
            0.0,  # Boundary: minimum valid weight
            1.0,  # Boundary: maximum valid weight
            0.5,  # Valid middle value
        ],
    )
    def test_weight_boundary_values(self, weight) -> None:
        """Test weight boundary values (0.0 and 1.0 should pass)."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=weight,
            active_states=[STATE_ON],
        )
        assert entity_type.weight == weight

    @pytest.mark.parametrize(
        ("input_type", "active_range"),
        [
            (InputType.ENVIRONMENTAL, (0.0, 0.0)),  # Equal min/max should be valid
            (InputType.TEMPERATURE, (18.0, 24.0)),  # Normal range should work
        ],
    )
    def test_active_range_boundary_values(self, input_type, active_range) -> None:
        """Test active_range boundary values including equal min/max."""
        entity_type = EntityType(
            input_type=input_type,
            active_range=active_range,
        )
        assert entity_type.active_range == active_range

    def test_missing_input_type_fallback(self, caplog) -> None:
        """Test fallback behavior when InputType is missing from DEFAULT_TYPES."""
        # Create a temporary InputType that doesn't exist in DEFAULT_TYPES
        # We'll need to temporarily remove one from DEFAULT_TYPES for testing
        # Since we can't easily create new enum values, we'll test the fallback
        # by temporarily modifying DEFAULT_TYPES

        # Temporarily remove a type (we'll use CO2 as it's not critical)
        test_type = InputType.CO2
        original_value = DEFAULT_TYPES.pop(test_type, None)

        try:
            with caplog.at_level(logging.WARNING):
                entity_type = EntityType(test_type)

            # Should log a warning
            assert any(
                "missing from DEFAULT_TYPES" in record.message
                and str(test_type) in record.message
                for record in caplog.records
            )

            # Should fallback to UNKNOWN defaults
            assert entity_type.weight == DEFAULT_TYPES[InputType.UNKNOWN]["weight"]
            assert (
                entity_type.active_states
                == DEFAULT_TYPES[InputType.UNKNOWN]["active_states"]
            )
        finally:
            # Restore original DEFAULT_TYPES
            if original_value is not None:
                DEFAULT_TYPES[test_type] = original_value

    def test_strength_multiplier_override(self) -> None:
        """Test that strength_multiplier can be overridden from default."""
        entity_type = EntityType(InputType.MOTION, strength_multiplier=4.0)
        # Default for MOTION is 3.0, but override should take precedence
        assert entity_type.strength_multiplier == 4.0

    def test_missing_unknown_fallback(self, caplog) -> None:
        """Test ultimate fallback when UNKNOWN is also missing from DEFAULT_TYPES."""
        # Save original values
        original_unknown = DEFAULT_TYPES.pop(InputType.UNKNOWN, None)
        test_type = InputType.CO2
        original_co2 = DEFAULT_TYPES.pop(test_type, None)

        try:
            with caplog.at_level(logging.WARNING):
                entity_type = EntityType(test_type)

            # Should still work with ultimate fallback
            assert entity_type.weight == 0.5  # Ultimate fallback value
            assert entity_type.active_states == [STATE_ON]  # Ultimate fallback
        finally:
            # Restore original values
            if original_unknown is not None:
                DEFAULT_TYPES[InputType.UNKNOWN] = original_unknown
            if original_co2 is not None:
                DEFAULT_TYPES[test_type] = original_co2
