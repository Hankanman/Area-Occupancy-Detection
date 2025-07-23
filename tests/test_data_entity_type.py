"""Tests for data.entity_type module."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.const import (
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
)
from custom_components.area_occupancy.data.entity_type import (
    EntityType,
    EntityTypeManager,
    InputType,
)
from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001, PLC0415
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
            entity_type.prior == 0.000001
        )  # Clamped to minimum for priors (0.000001, not 0.001)

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

    def test_has_evidence_with_states(self) -> None:
        """Test Entity evidence detection with active_states."""
        from custom_components.area_occupancy.data.decay import Decay
        from custom_components.area_occupancy.data.entity import Entity

        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_true=0.25,
            prob_false=0.05,
            prior=0.35,
            active_states=[STATE_ON],
        )

        # Create mock coordinator with proper state mocking
        mock_coordinator = Mock()
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_state.attributes = {"friendly_name": "Test Sensor"}
        mock_coordinator.hass.states.get.return_value = mock_state

        # Create a test entity
        from custom_components.area_occupancy.data.likelihood import Likelihood

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.test",
            active_states=[STATE_ON],
            default_prob_true=0.25,
            default_prob_false=0.05,
            weight=0.8,
        )

        test_entity = Entity(
            entity_id="binary_sensor.test",
            type=entity_type,
            likelihood=likelihood,
            decay=Decay(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
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
            prior=0.0769,
            active_range=(0.0, 1.0),
        )

        # Create mock coordinator with proper state mocking
        mock_coordinator = Mock()
        mock_state = Mock()
        mock_state.state = "0.5"  # Within range
        mock_state.attributes = {"friendly_name": "Test Sensor"}
        mock_coordinator.hass.states.get.return_value = mock_state

        # Create a test entity
        from custom_components.area_occupancy.data.likelihood import Likelihood

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="sensor.test",
            active_states=[],  # Empty list instead of None
            default_prob_true=0.09,
            default_prob_false=0.01,
            weight=0.3,
        )

        test_entity = Entity(
            entity_id="sensor.test",
            type=entity_type,
            likelihood=likelihood,
            decay=Decay(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
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


class TestEntityTypeManagerOverrides:
    """Test configuration override helpers."""

    def _make_manager(self, config: Any) -> EntityTypeManager:
        coordinator = Mock()
        coordinator.config_manager = Mock(config=config)
        return EntityTypeManager(coordinator)

    def test_apply_weight_valid(self) -> None:
        config = SimpleNamespace(weights=SimpleNamespace(motion=0.4))
        manager = self._make_manager(config)
        params = {"weight": 0.8}
        manager._apply_weight(InputType.MOTION, params)
        assert params["weight"] == 0.4

    def test_apply_weight_environmental_sensors(self) -> None:
        """Test that environmental sensor types use the environmental weight."""
        config = SimpleNamespace(weights=SimpleNamespace(environmental=0.25))
        manager = self._make_manager(config)

        # Test each environmental sensor type
        for input_type in [
            InputType.ILLUMINANCE,
            InputType.HUMIDITY,
            InputType.TEMPERATURE,
        ]:
            params = {"weight": 0.8}
            manager._apply_weight(input_type, params)
            assert params["weight"] == 0.25, f"Failed for {input_type}"

    def test_apply_weight_environmental_sensors_no_config(self) -> None:
        """Test that environmental sensor types don't change weight when no environmental config."""
        config = SimpleNamespace(
            weights=SimpleNamespace(motion=0.4)
        )  # No environmental weight
        manager = self._make_manager(config)

        # Test each environmental sensor type
        for input_type in [
            InputType.ILLUMINANCE,
            InputType.HUMIDITY,
            InputType.TEMPERATURE,
        ]:
            params = {"weight": 0.8}
            manager._apply_weight(input_type, params)
            assert params["weight"] == 0.8, (
                f"Should not change for {input_type} when no environmental config"
            )

    def test_apply_weight_invalid(self) -> None:
        config = SimpleNamespace(weights=SimpleNamespace(motion=1.5))
        manager = self._make_manager(config)
        with pytest.raises(ValueError):
            manager._apply_weight(InputType.MOTION, {})

    def test_apply_states_valid(self) -> None:
        config = SimpleNamespace(sensor_states=SimpleNamespace(motion=["on", "open"]))
        manager = self._make_manager(config)
        params = {"active_states": None, "active_range": (0, 1)}
        manager._apply_states(InputType.MOTION, params)
        assert params["active_states"] == ["on", "open"]
        assert params["active_range"] is None

    def test_apply_states_invalid(self) -> None:
        config = SimpleNamespace(sensor_states=SimpleNamespace(motion="bad"))
        manager = self._make_manager(config)
        with pytest.raises(ValueError):
            manager._apply_states(InputType.MOTION, {})

    def test_apply_range_valid(self) -> None:
        config = SimpleNamespace(motion_active_range=(0.1, 0.9))
        manager = self._make_manager(config)
        params = {"active_range": None, "active_states": ["on"]}
        manager._apply_range(InputType.MOTION, params)
        assert params["active_range"] == (0.1, 0.9)
        assert params["active_states"] is None

    def test_apply_range_invalid(self) -> None:
        config = SimpleNamespace(motion_active_range=(1,))
        manager = self._make_manager(config)
        with pytest.raises(ValueError):
            manager._apply_range(InputType.MOTION, {})

    def test_to_dict_roundtrip(self, mock_coordinator: Mock) -> None:
        manager = self._make_manager(SimpleNamespace())
        manager._entity_types = {
            InputType.MOTION: EntityType(
                input_type=InputType.MOTION,
                weight=0.5,
                prob_true=0.2,
                prob_false=0.1,
                prior=0.3,
                active_states=[STATE_ON],
            )
        }
        data = manager.to_dict()
        loaded = EntityTypeManager.from_dict(data, mock_coordinator)
        assert loaded.entity_types[InputType.MOTION].weight == 0.5

    def test_from_dict_errors(self, mock_coordinator: Mock) -> None:
        with pytest.raises(ValueError):
            EntityTypeManager.from_dict({}, mock_coordinator)

        bad_data = {"entity_types": {"motion": {"weight": 1}}}
        with pytest.raises(ValueError):
            EntityTypeManager.from_dict(bad_data, mock_coordinator)

    def test_cleanup(self) -> None:
        """Test cleanup method."""
        manager = self._make_manager(None)
        manager._entity_types = {"test": "data"}
        manager.cleanup()
        assert manager._entity_types == {}

    def test_build_entity_types_with_environmental_weight(self) -> None:
        """Test that environmental sensor types get the correct weight from config."""
        # Create a config with custom environmental weight
        config = SimpleNamespace(
            weights=SimpleNamespace(
                motion=0.9,
                media=0.8,
                appliance=0.7,
                door=0.6,
                window=0.5,
                environmental=0.25,  # Custom environmental weight
                wasp=0.85,
            )
        )
        manager = self._make_manager(config)

        # Build entity types
        entity_types = manager._build_entity_types()

        # Check that environmental sensor types use the environmental weight
        assert entity_types[InputType.ILLUMINANCE].weight == 0.25
        assert entity_types[InputType.HUMIDITY].weight == 0.25
        assert entity_types[InputType.TEMPERATURE].weight == 0.25

        # Check that other types use their specific weights
        assert entity_types[InputType.MOTION].weight == 0.9
        assert entity_types[InputType.MEDIA].weight == 0.8
        assert entity_types[InputType.APPLIANCE].weight == 0.7
        assert entity_types[InputType.DOOR].weight == 0.6
        assert entity_types[InputType.WINDOW].weight == 0.5

    def test_build_entity_types_without_config_uses_defaults(self) -> None:
        """Test that entity types use default weights when no config is provided."""
        manager = self._make_manager(None)

        # Build entity types
        entity_types = manager._build_entity_types()

        # Check that all types use their default weights from constants
        assert entity_types[InputType.MOTION].weight == DEFAULT_WEIGHT_MOTION
        assert entity_types[InputType.MEDIA].weight == DEFAULT_WEIGHT_MEDIA
        assert entity_types[InputType.APPLIANCE].weight == DEFAULT_WEIGHT_APPLIANCE
        assert entity_types[InputType.DOOR].weight == DEFAULT_WEIGHT_DOOR
        assert entity_types[InputType.WINDOW].weight == DEFAULT_WEIGHT_WINDOW

        # Check that environmental sensor types use the environmental default
        assert (
            entity_types[InputType.ILLUMINANCE].weight == DEFAULT_WEIGHT_ENVIRONMENTAL
        )
        assert entity_types[InputType.HUMIDITY].weight == DEFAULT_WEIGHT_ENVIRONMENTAL
        assert (
            entity_types[InputType.TEMPERATURE].weight == DEFAULT_WEIGHT_ENVIRONMENTAL
        )
