"""Tests for data.entity_type module."""

from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util


# ruff: noqa: PLC0415
class TestInputType:
    """Test InputType enum."""

    def test_input_type_values(self) -> None:
        """Test that InputType has expected values."""
        assert InputType.MOTION.value == "motion"
        assert InputType.MEDIA.value == "media"
        assert InputType.APPLIANCE.value == "appliance"
        assert InputType.DOOR.value == "door"
        assert InputType.WINDOW.value == "window"
        assert InputType.ENVIRONMENTAL.value == "environmental"


class TestEntityType:
    """Test EntityType class."""

    def test_initialization_with_states(self) -> None:
        """Test EntityType initialization with active_states."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_true=0.25,
            prob_false=0.05,
            active_states=[STATE_ON],
        )

        assert entity_type.input_type == InputType.MOTION
        assert entity_type.weight == 0.8
        assert entity_type.prob_true == 0.25
        assert entity_type.prob_false == 0.05
        assert entity_type.active_states == [STATE_ON]
        assert entity_type.active_range is None

    def test_initialization_with_range(self) -> None:
        """Test EntityType initialization with active_range."""
        entity_type = EntityType(
            input_type=InputType.ENVIRONMENTAL,
            weight=0.3,
            prob_true=0.09,
            prob_false=0.01,
            active_range=(0.0, 0.2),
        )

        assert entity_type.input_type == InputType.ENVIRONMENTAL
        assert entity_type.active_range == (0.0, 0.2)
        assert entity_type.active_states is None

    def test_initialization_validation(self) -> None:
        """Test that invalid values are validated during initialization."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=-0.1,  # Invalid
            prob_true=1.5,  # Invalid
            prob_false=-0.1,  # Invalid
            active_states=[STATE_ON],
        )

        assert entity_type.weight == 0.01  # Clamped to minimum
        assert entity_type.prob_true == 1.0  # Clamped to maximum
        assert entity_type.prob_false == 0.001  # Clamped to minimum (0.001, not 0.0)

    def test_initialization_errors(self) -> None:
        """Test initialization errors for invalid configurations."""
        # Both active_states and active_range provided
        with pytest.raises(
            ValueError, match="Cannot provide both active_states and active_range"
        ):
            EntityType(
                input_type=InputType.MOTION,
                weight=0.8,
                prob_true=0.25,
                prob_false=0.05,
                active_states=[STATE_ON],
                active_range=(0.0, 1.0),
            )

        # Neither active_states nor active_range provided
        with pytest.raises(
            ValueError, match="Either active_states or active_range must be provided"
        ):
            EntityType(
                input_type=InputType.MOTION,
                weight=0.8,
                prob_true=0.25,
                prob_false=0.05,
            )

    def test_has_evidence_with_states(self) -> None:
        """Test Entity evidence detection with active_states."""
        from custom_components.area_occupancy.data.decay import Decay
        from custom_components.area_occupancy.data.entity import Entity

        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_true=0.25,
            prob_false=0.05,
            active_states=[STATE_ON],
        )

        # Create mock coordinator with proper state mocking
        mock_coordinator = Mock()
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_state.attributes = {"friendly_name": "Test Sensor"}
        mock_coordinator.hass.states.get.return_value = mock_state

        # Create a test entity
        test_entity = Entity(
            entity_id="binary_sensor.test",
            type=entity_type,
            prob_given_true=0.25,
            prob_given_false=0.05,
            decay=Decay(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
        )

        # Should have evidence when state is "on"
        assert test_entity.evidence

        # Change state to "off" and test again
        mock_state.state = "off"
        assert not test_entity.evidence

    def test_has_evidence_with_range(self) -> None:
        """Test Entity evidence detection with active_range."""
        from custom_components.area_occupancy.data.decay import Decay
        from custom_components.area_occupancy.data.entity import Entity

        entity_type = EntityType(
            input_type=InputType.ENVIRONMENTAL,
            weight=0.3,
            prob_true=0.09,
            prob_false=0.01,
            active_range=(0.0, 1.0),
        )

        # Create mock coordinator with proper state mocking
        mock_coordinator = Mock()
        mock_state = Mock()
        mock_state.state = "0.5"  # Within range
        mock_state.attributes = {"friendly_name": "Test Sensor"}
        mock_coordinator.hass.states.get.return_value = mock_state

        # Create a test entity
        test_entity = Entity(
            entity_id="sensor.test",
            type=entity_type,
            prob_given_true=0.09,
            prob_given_false=0.01,
            decay=Decay(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
        )

        # Should have evidence when value is within range (0.0, 1.0)
        assert test_entity.evidence

        # Change state to outside range
        mock_state.state = "2.0"
        assert not test_entity.evidence

        # Change state to invalid value
        mock_state.state = "invalid"
        assert not test_entity.evidence

    def test_to_dict(self) -> None:
        """Test converting EntityType to dictionary."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_true=0.25,
            prob_false=0.05,
            active_states=[STATE_ON],
        )

        result = entity_type.to_dict()
        expected = {
            "input_type": "motion",
            "weight": 0.8,
            "prob_true": 0.25,
            "prob_false": 0.05,
            "active_states": [STATE_ON],
            "active_range": None,
        }

        assert result == expected
