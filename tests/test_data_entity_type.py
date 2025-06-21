"""Tests for data.entity_type module."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

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

    def test_is_entity_active_with_states(self) -> None:
        """Test EntityManager.is_entity_active method with active_states."""
        from custom_components.area_occupancy.data.decay import Decay
        from custom_components.area_occupancy.data.entity import Entity
        from custom_components.area_occupancy.data.prior import Prior
        from homeassistant.util import dt as dt_util

        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_true=0.25,
            prob_false=0.05,
            prior=0.35,
            active_states=[STATE_ON],
        )

        # Create a test entity
        test_entity = Entity(
            entity_id="binary_sensor.test",
            type=entity_type,
            prior=Prior(
                prob_given_true=0.25,
                prob_given_false=0.05,
                last_updated=dt_util.utcnow(),
            ),
            decay=Decay(),
            coordinator=Mock(),
        )

        assert test_entity.evidence
        assert not test_entity.evidence

    def test_is_entity_active_with_range(self) -> None:
        """Test EntityManager.is_entity_active method with active_range."""
        from custom_components.area_occupancy.data.decay import Decay
        from custom_components.area_occupancy.data.entity import Entity
        from custom_components.area_occupancy.data.prior import Prior
        from homeassistant.util import dt as dt_util

        entity_type = EntityType(
            input_type=InputType.ENVIRONMENTAL,
            weight=0.3,
            prob_true=0.09,
            prob_false=0.01,
            prior=0.0769,
            active_range=(0.0, 1.0),
        )

        # Create a test entity
        test_entity = Entity(
            entity_id="sensor.test",
            type=entity_type,
            prior=Prior(
                prob_given_true=0.09,
                prob_given_false=0.01,
                last_updated=dt_util.utcnow(),
            ),
            decay=Decay(),
            coordinator=Mock(),
        )

        assert test_entity.evidence
        assert not test_entity.evidence
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

    def test_learn_from_entities_with_entity_objects(
        self, mock_coordinator: Mock
    ) -> None:
        """Test learning from Entity objects."""
        manager = EntityTypeManager(mock_coordinator)

        # Initialize entity types
        manager._entity_types = {
            InputType.MOTION: EntityType(
                input_type=InputType.MOTION,
                weight=0.8,
                prob_true=0.25,
                prob_false=0.05,
                prior=0.35,
                active_states=[STATE_ON],
            ),
            InputType.MEDIA: EntityType(
                input_type=InputType.MEDIA,
                weight=0.7,
                prob_true=0.2,
                prob_false=0.03,
                prior=0.3,
                active_states=[STATE_ON],
            ),
        }

        # Create mock entities with priors
        from custom_components.area_occupancy.data.entity import Entity
        from custom_components.area_occupancy.data.prior import Prior

        motion_entity = Mock(spec=Entity)
        motion_entity.type = manager._entity_types[InputType.MOTION]
        motion_entity.prior = Prior(
            prob_given_true=0.3,
            prob_given_false=0.06,
            last_updated=dt_util.utcnow(),
        )

        media_entity = Mock(spec=Entity)
        media_entity.type = manager._entity_types[InputType.MEDIA]
        media_entity.prior = Prior(
            prob_given_true=0.25,
            prob_given_false=0.04,
            last_updated=dt_util.utcnow(),
        )

        # Test learning from entities
        manager.learn_from_entities(
            {
                "motion1": motion_entity,
                "media1": media_entity,
            }
        )

        # Verify motion type was updated with averages
        motion_type = manager._entity_types[InputType.MOTION]
        assert motion_type.prior == 0.3
        assert motion_type.prob_true == 0.3
        assert motion_type.prob_false == 0.06

        # Verify media type was updated with averages
        media_type = manager._entity_types[InputType.MEDIA]
        assert media_type.prior == 0.25
        assert media_type.prob_true == 0.25
        assert media_type.prob_false == 0.04

    def test_learn_from_entities_empty(self, mock_coordinator: Mock) -> None:
        """Test learning from empty entity list."""
        manager = EntityTypeManager(mock_coordinator)

        # Initialize entity types with known values
        initial_values = {
            InputType.MOTION: EntityType(
                input_type=InputType.MOTION,
                weight=0.8,
                prob_true=0.25,
                prob_false=0.05,
                prior=0.35,
                active_states=[STATE_ON],
            ),
        }
        manager._entity_types = initial_values.copy()

        # Test learning from empty entities
        manager.learn_from_entities({})

        # Verify values were not changed
        motion_type = manager._entity_types[InputType.MOTION]
        assert motion_type.prior == initial_values[InputType.MOTION].prior
        assert motion_type.prob_true == initial_values[InputType.MOTION].prob_true
        assert motion_type.prob_false == initial_values[InputType.MOTION].prob_false

    def test_learn_from_entities_invalid_type(self, mock_coordinator: Mock) -> None:
        """Test learning from entities with invalid type."""
        manager = EntityTypeManager(mock_coordinator)

        # Initialize entity types
        manager._entity_types = {
            InputType.MOTION: EntityType(
                input_type=InputType.MOTION,
                weight=0.8,
                prob_true=0.25,
                prob_false=0.05,
                prior=0.35,
                active_states=[STATE_ON],
            ),
        }

        # Create entity with invalid type
        entities = {
            "invalid": {
                "type": "invalid_type",
                "prior": {
                    "prob_given_true": 0.3,
                    "prob_given_false": 0.06,
                },
            },
        }

        # Test learning from invalid entity (should skip invalid type)
        manager.learn_from_entities(entities)

        # Verify motion type was not changed
        motion_type = manager._entity_types[InputType.MOTION]
        assert motion_type.prior == 0.35
        assert motion_type.prob_true == 0.25
        assert motion_type.prob_false == 0.05


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
        manager = self._make_manager(SimpleNamespace())
        manager._entity_types = {InputType.MOTION: Mock()}
        manager.cleanup()
        assert manager.entity_types == {}
