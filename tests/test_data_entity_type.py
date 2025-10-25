"""Tests for data.entity_type module."""

from unittest.mock import Mock

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
        [
            (InputType.MOTION, "motion"),
            (InputType.MEDIA, "media"),
            (InputType.APPLIANCE, "appliance"),
            (InputType.DOOR, "door"),
            (InputType.WINDOW, "window"),
            (InputType.TEMPERATURE, "temperature"),
            (InputType.HUMIDITY, "humidity"),
            (InputType.ILLUMINANCE, "illuminance"),
            (InputType.ENVIRONMENTAL, "environmental"),
            (InputType.UNKNOWN, "unknown"),
        ],
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

        assert entity_type.input_type == InputType.MOTION
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

        assert entity_type.input_type == InputType.ENVIRONMENTAL
        assert entity_type.active_range == (0.0, 0.2)
        assert entity_type.active_states is None

    @pytest.mark.parametrize(
        ("test_case", "params", "expected_error"),
        [
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
            (
                "neither_active_states_nor_range",
                {
                    "input_type": InputType.MOTION,
                    "weight": 0.8,
                    "prob_given_true": 0.25,
                    "prob_given_false": 0.05,
                },
                "Either active_states or active_range must be provided",
            ),
        ],
    )
    def test_initialization_errors(self, test_case, params, expected_error) -> None:
        """Test initialization errors for invalid configurations."""
        with pytest.raises(ValueError, match=expected_error):
            EntityType(**params)

    @pytest.mark.parametrize(
        ("input_type", "expected_config"),
        [(input_type, config) for input_type, config in DEFAULT_TYPES.items()],
    )
    def test_create_classmethod_defaults(self, input_type, expected_config) -> None:
        """Test the create classmethod for different input types with default values."""
        entity_type = EntityType.create(input_type)

        assert entity_type.input_type == input_type
        assert entity_type.weight == expected_config["weight"]
        assert entity_type.prob_given_true == expected_config["prob_given_true"]
        assert entity_type.prob_given_false == expected_config["prob_given_false"]
        assert entity_type.active_states == expected_config["active_states"]
        assert entity_type.active_range == expected_config["active_range"]

    def test_create_classmethod_with_config_override(self) -> None:
        """Test the create classmethod with configuration overrides."""
        mock_config = Mock()
        mock_config.weights = Mock()
        mock_config.weights.motion = 0.9
        mock_config.sensor_states = Mock()
        mock_config.sensor_states.motion = ["on", "detected"]
        # Ensure no unexpected attributes exist
        mock_config.motion_active_range = None

        entity_type = EntityType.create(InputType.MOTION, mock_config)

        assert entity_type.input_type == InputType.MOTION
        assert entity_type.weight == 0.9  # Overridden
        assert entity_type.active_states == ["on", "detected"]  # Overridden
        assert entity_type.active_range is None

    def test_create_classmethod_with_active_range_override(self) -> None:
        """Test the create classmethod with active range override."""
        mock_config = Mock()
        mock_config.weights = Mock()
        mock_config.weights.environmental = 0.2  # Override weight
        # Explicitly set sensor_states to None to avoid Mock creating unexpected attributes
        mock_config.sensor_states = None
        mock_config.environmental_active_range = (0.1, 0.3)  # Override range

        entity_type = EntityType.create(InputType.ENVIRONMENTAL, mock_config)

        assert entity_type.input_type == InputType.ENVIRONMENTAL
        assert entity_type.weight == 0.2  # Overridden
        assert entity_type.active_states is None  # Cleared when range is set
        assert entity_type.active_range == (0.1, 0.3)  # Overridden

    @pytest.mark.parametrize(
        ("test_case", "config_setup", "expected_error"),
        [
            (
                "invalid_weight",
                lambda: Mock(
                    weights=Mock(motion=1.5),  # Invalid weight > 1
                ),
                "Invalid weight for motion: 1.5",
            ),
            (
                "invalid_states",
                lambda: Mock(
                    weights=Mock(motion=0.8),  # Valid weight
                    sensor_states=Mock(motion="invalid"),  # Should be list
                ),
                "Invalid active states for motion: invalid",
            ),
            (
                "invalid_active_range",
                lambda: Mock(
                    weights=Mock(environmental=0.1),  # Valid weight
                    sensor_states=None,  # Explicitly set to None
                    environmental_active_range="invalid",  # Should be tuple
                ),
                "Invalid active range for environmental: invalid",
            ),
        ],
    )
    def test_create_classmethod_config_errors(
        self, test_case, config_setup, expected_error
    ) -> None:
        """Test the create classmethod with invalid configuration."""
        mock_config = config_setup()

        # Use the appropriate input type based on the test case
        if test_case == "invalid_active_range":
            input_type = InputType.ENVIRONMENTAL
        else:
            input_type = InputType.MOTION

        with pytest.raises(ValueError, match=expected_error):
            EntityType.create(input_type, mock_config)

    def test_create_classmethod_with_empty_states_list(self) -> None:
        """Test that empty active_states list uses defaults instead of crashing."""
        mock_config = Mock()
        mock_config.weights = Mock()
        mock_config.weights.motion = 0.9  # Override weight
        mock_config.sensor_states = Mock()
        mock_config.sensor_states.motion = []  # Empty list - should use defaults
        mock_config.motion_active_range = None

        entity_type = EntityType.create(InputType.MOTION, mock_config)

        # Should use default active_states from DEFAULT_TYPES, not empty list
        assert entity_type.input_type == InputType.MOTION
        assert entity_type.weight == 0.9  # Weight override should still work
        assert (
            entity_type.active_states
            == DEFAULT_TYPES[InputType.MOTION]["active_states"]
        )  # Default states
        assert entity_type.active_range is None

    def test_create_classmethod_with_empty_states_list_for_range_type(self) -> None:
        """Test that empty active_states list uses defaults for range-based types."""
        mock_config = Mock()
        mock_config.weights = Mock()
        mock_config.weights.temperature = 0.2  # Override weight
        mock_config.sensor_states = Mock()
        mock_config.sensor_states.temperature = []  # Empty list - should use defaults
        mock_config.temperature_active_range = None

        entity_type = EntityType.create(InputType.TEMPERATURE, mock_config)

        # Should use default active_range from DEFAULT_TYPES, not empty list
        assert entity_type.input_type == InputType.TEMPERATURE
        assert entity_type.weight == 0.2  # Weight override should still work
        assert entity_type.active_states is None  # Default for temperature
        assert (
            entity_type.active_range
            == DEFAULT_TYPES[InputType.TEMPERATURE]["active_range"]
        )  # Default range
