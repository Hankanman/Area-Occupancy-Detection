"""Tests for data.entity_type module."""

from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.data.entity_type import (
    EntityType,
    EntityTypeManager,
    InputType,
)
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import State


# ruff: noqa: SLF001
class TestInputType:
    """Test InputType enum."""

    def test_input_type_values(self) -> None:
        """Test that InputType has expected values."""
        assert InputType.MOTION.value == "motion"
        assert InputType.MEDIA.value == "media"
        assert InputType.APPLIANCE.value == "appliance"
        assert InputType.DOOR.value == "door"
        assert InputType.WINDOW.value == "window"
        assert InputType.LIGHT.value == "light"
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
            prior=0.35,
            active_states=[STATE_ON],
        )

        assert entity_type.input_type == InputType.MOTION
        assert entity_type.weight == 0.8
        assert entity_type.prob_true == 0.25
        assert entity_type.prob_false == 0.05
        assert entity_type.prior == 0.35
        assert entity_type.active_states == [STATE_ON]
        assert entity_type.active_range is None

    def test_initialization_with_range(self) -> None:
        """Test EntityType initialization with active_range."""
        entity_type = EntityType(
            input_type=InputType.ENVIRONMENTAL,
            weight=0.3,
            prob_true=0.09,
            prob_false=0.01,
            prior=0.0769,
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
            prior=0.0,  # Invalid for prior
            active_states=[STATE_ON],
        )

        assert entity_type.weight == 0.01  # Clamped to minimum
        assert entity_type.prob_true == 1.0  # Clamped to maximum
        assert entity_type.prob_false == 0.001  # Clamped to minimum (0.001, not 0.0)
        assert (
            entity_type.prior == 0.001
        )  # Clamped to minimum for priors (0.001, not 0.0001)

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
                prior=0.35,
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
                prior=0.35,
            )

    def test_is_active_with_states(self) -> None:
        """Test is_active method with active_states."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_true=0.25,
            prob_false=0.05,
            prior=0.35,
            active_states=[STATE_ON],
        )

        # Create mock states
        active_state = Mock(spec=State)
        active_state.state = STATE_ON

        inactive_state = Mock(spec=State)
        inactive_state.state = STATE_OFF

        assert entity_type.is_active(active_state)
        assert not entity_type.is_active(inactive_state)

    def test_is_active_with_range(self) -> None:
        """Test is_active method with active_range."""
        entity_type = EntityType(
            input_type=InputType.ENVIRONMENTAL,
            weight=0.3,
            prob_true=0.09,
            prob_false=0.01,
            prior=0.0769,
            active_range=(0.0, 0.2),
        )

        # Create mock states
        active_state = Mock(spec=State)
        active_state.state = "0.1"  # Within range

        inactive_state = Mock(spec=State)
        inactive_state.state = "0.5"  # Outside range

        invalid_state = Mock(spec=State)
        invalid_state.state = "not_a_number"

        assert entity_type.is_active(active_state)
        assert not entity_type.is_active(inactive_state)
        assert not entity_type.is_active(invalid_state)

    def test_to_dict(self) -> None:
        """Test converting EntityType to dictionary."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_true=0.25,
            prob_false=0.05,
            prior=0.35,
            active_states=[STATE_ON],
        )

        result = entity_type.to_dict()
        expected = {
            "input_type": "motion",
            "weight": 0.8,
            "prob_true": 0.25,
            "prob_false": 0.05,
            "prior": 0.35,
            "active_states": [STATE_ON],
            "active_range": None,
        }

        assert result == expected

    def test_from_dict(self) -> None:
        """Test creating EntityType from dictionary."""
        data = {
            "input_type": "motion",
            "weight": 0.8,
            "prob_true": 0.25,
            "prob_false": 0.05,
            "prior": 0.35,
            "active_states": [STATE_ON],
            "active_range": None,
        }

        entity_type = EntityType.from_dict(data)

        assert entity_type.input_type == InputType.MOTION
        assert entity_type.weight == 0.8
        assert entity_type.prob_true == 0.25
        assert entity_type.prob_false == 0.05
        assert entity_type.prior == 0.35
        assert entity_type.active_states == [STATE_ON]
        assert entity_type.active_range is None


class TestEntityTypeManager:
    """Test EntityTypeManager class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test EntityTypeManager initialization."""
        manager = EntityTypeManager(mock_coordinator)

        assert manager.coordinator == mock_coordinator
        assert manager.config == mock_coordinator.config_manager.config
        assert manager._entity_types == {}

    async def test_async_initialize(self, mock_coordinator: Mock) -> None:
        """Test async initialization of entity types."""
        manager = EntityTypeManager(mock_coordinator)
        await manager.async_initialize()

        # Should have created entity types for all InputType values
        assert len(manager._entity_types) == len(InputType)
        assert InputType.MOTION in manager._entity_types
        assert InputType.MEDIA in manager._entity_types

    def test_get_entity_type(self, mock_coordinator: Mock) -> None:
        """Test getting entity type by InputType."""
        manager = EntityTypeManager(mock_coordinator)
        manager._entity_types[InputType.MOTION] = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_true=0.25,
            prob_false=0.05,
            prior=0.35,
            active_states=[STATE_ON],
        )

        entity_type = manager.get_entity_type(InputType.MOTION)
        assert entity_type.input_type == InputType.MOTION
