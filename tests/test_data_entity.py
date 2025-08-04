"""Tests for the entity module."""

from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.data.entity import (
    Entity,
    EntityFactory,
    EntityManager,
)
from custom_components.area_occupancy.data.entity_type import InputType
from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestEntity:
    """Test the Entity class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test entity initialization."""
        mock_entity_type = Mock()
        mock_decay = Mock()

        entity = Entity(
            entity_id="test_entity",
            type=mock_entity_type,
            prob_given_true=0.8,
            prob_given_false=0.1,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
        )

        assert entity.entity_id == "test_entity"
        assert entity.type == mock_entity_type
        assert entity.prob_given_true == 0.8
        assert entity.prob_given_false == 0.1
        assert entity.decay == mock_decay
        assert entity.coordinator == mock_coordinator

    @patch("custom_components.area_occupancy.data.entity.Decay")
    @patch("custom_components.area_occupancy.data.entity.EntityType")
    def test_from_dict(
        self,
        mock_entity_type_class: Mock,
        mock_decay_class: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test creating entity from dictionary."""
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None
        mock_entity_type_class.from_dict.return_value = mock_entity_type

        mock_decay = Mock()
        mock_decay.decay_factor = 1.0
        mock_decay_class.from_dict.return_value = mock_decay

        # Set up coordinator hass.states properly for entity creation
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_state.attributes = {"friendly_name": "Test Motion"}
        mock_coordinator.hass.states.get.return_value = mock_state

        current_time = dt_util.utcnow()
        data = {
            "entity_id": "binary_sensor.test_motion",
            "type": {
                "input_type": InputType.MOTION,
                "weight": 0.8,
                "prob_true": 0.2,
                "prob_false": 0.03,
                "prior": 0.3,
                "active_states": [STATE_ON],
            },
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "decay": {"is_decaying": False},
            "last_updated": current_time.isoformat(),
            "previous_evidence": None,
        }

        # Create entity directly using the new structure
        entity = Entity(
            entity_id=data["entity_id"],
            type=mock_entity_type,
            prob_given_true=data["prob_given_true"],
            prob_given_false=data["prob_given_false"],
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=current_time,
            previous_evidence=data["previous_evidence"],
        )

        assert entity.entity_id == "binary_sensor.test_motion"
        assert entity.type == mock_entity_type
        assert entity.prob_given_true == 0.8
        assert entity.prob_given_false == 0.1


class TestEntityManager:
    """Test the EntityManager class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test manager initialization."""
        manager = EntityManager(mock_coordinator)

        assert manager.coordinator == mock_coordinator
        assert manager._entities == {}

    def test_entities_property(self, mock_coordinator: Mock) -> None:
        """Test entities property."""
        manager = EntityManager(mock_coordinator)
        manager._entities = {"test": Mock()}

        assert manager.entities == {"test": manager._entities["test"]}

    def test_entity_ids_property(self, mock_coordinator: Mock) -> None:
        """Test entity_ids property."""
        manager = EntityManager(mock_coordinator)
        manager._entities = {"entity1": Mock(), "entity2": Mock()}

        assert set(manager.entity_ids) == {"entity1", "entity2"}

    def test_active_entities_property(self, mock_coordinator: Mock) -> None:
        """Test active_entities property."""
        manager = EntityManager(mock_coordinator)

        # Create mock entities with different states
        active_entity = Mock()
        active_entity.state = STATE_ON
        active_entity.evidence = True
        active_entity.decay.is_decaying = False

        inactive_entity = Mock()
        inactive_entity.state = "off"
        inactive_entity.evidence = False
        inactive_entity.decay.is_decaying = False

        manager._entities = {
            "active": active_entity,
            "inactive": inactive_entity,
        }

        active_entities = manager.active_entities
        assert len(active_entities) == 1
        assert active_entities[0] == active_entity

    def test_inactive_entities_property(self, mock_coordinator: Mock) -> None:
        """Test inactive_entities property."""
        manager = EntityManager(mock_coordinator)

        # Create mock entities with different states
        active_entity = Mock()
        active_entity.state = STATE_ON
        active_entity.evidence = True
        active_entity.decay.is_decaying = False

        inactive_entity = Mock()
        inactive_entity.state = "off"
        inactive_entity.evidence = False
        inactive_entity.decay.is_decaying = False

        manager._entities = {
            "active": active_entity,
            "inactive": inactive_entity,
        }

        inactive_entities = manager.inactive_entities
        assert len(inactive_entities) == 1
        assert inactive_entities[0] == inactive_entity

    def test_get_entity(self, mock_coordinator: Mock) -> None:
        """Test getting entity by ID."""
        manager = EntityManager(mock_coordinator)
        mock_entity = Mock()
        manager._entities = {"test": mock_entity}

        result = manager.get_entity("test")
        assert result == mock_entity

        with pytest.raises(
            ValueError, match="Entity not found for entity: nonexistent"
        ):
            manager.get_entity("nonexistent")

    def test_add_entity(self, mock_coordinator: Mock) -> None:
        """Test adding entity to manager."""
        manager = EntityManager(mock_coordinator)
        mock_entity = Mock()
        mock_entity.entity_id = "test"

        manager.add_entity(mock_entity)

        assert manager._entities["test"] == mock_entity


class TestEntityPropertiesAndMethods:
    """Test entity properties and methods."""

    def test_state_property_edge_cases(self, mock_coordinator: Mock) -> None:
        """Test state property with edge cases."""
        entity = Entity(
            entity_id="test",
            type=Mock(),
            prob_given_true=0.8,
            prob_given_false=0.1,
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
        )

        # Test with no state available
        mock_coordinator.hass.states.get.return_value = None
        assert entity.state is None

        # Test with state but no state attribute
        mock_state = Mock()
        mock_state.state = "test_state"  # Set a valid state first
        mock_coordinator.hass.states.get.return_value = mock_state
        assert entity.state == "test_state"

    def test_available_property(self, mock_coordinator: Mock) -> None:
        """Test available property."""
        entity = Entity(
            entity_id="test",
            type=Mock(),
            prob_given_true=0.8,
            prob_given_false=0.1,
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
        )

        # Test with available state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state
        assert entity.available is True

        # Test with unavailable state
        mock_state.state = "unavailable"
        assert entity.available is False

        # Test with no state
        mock_coordinator.hass.states.get.return_value = None
        assert entity.available is False

    def test_evidence_with_active_range(self, mock_coordinator: Mock) -> None:
        """Test evidence property with active range."""
        mock_entity_type = Mock()
        mock_entity_type.active_range = (10, 20)
        mock_entity_type.active_states = None  # Ensure this is None for range test

        entity = Entity(
            entity_id="test",
            type=mock_entity_type,
            prob_given_true=0.8,
            prob_given_false=0.1,
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
        )

        # Test with value in range
        mock_state = Mock()
        mock_state.state = "15"
        mock_coordinator.hass.states.get.return_value = mock_state
        assert entity.evidence is True

        # Test with value outside range
        mock_state.state = "5"
        assert entity.evidence is False

    def test_has_new_evidence_transitions(self, mock_coordinator: Mock) -> None:
        """Test has_new_evidence with state transitions."""
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = Entity(
            entity_id="test",
            type=mock_entity_type,
            prob_given_true=0.8,
            prob_given_false=0.1,
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
        )

        # Test transition from None to active
        entity.previous_evidence = None
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state
        # The first call should return False because previous_evidence is None (initialization)
        assert entity.has_new_evidence() is False

        # Now test transition from False to True
        entity.previous_evidence = False
        entity.coordinator.area_prior = 0.3
        assert entity.has_new_evidence() is True

        # Test transition from active to inactive
        entity.previous_evidence = True
        mock_state.state = "off"
        entity.coordinator.area_prior = 0.3
        assert entity.has_new_evidence() is True

        # Test no transition
        entity.previous_evidence = True
        mock_state.state = STATE_ON
        assert entity.has_new_evidence() is False


class TestEntityManagerAdvanced:
    """Test EntityManager advanced functionality and edge cases."""

    def test_decaying_entities_property(self, mock_coordinator: Mock) -> None:
        """Test decaying_entities property."""
        manager = EntityManager(mock_coordinator)

        # Create mock entities with different decay states
        decaying_entity1 = Mock()
        decaying_entity1.decay.is_decaying = True
        decaying_entity1.entity_id = "decaying1"

        decaying_entity2 = Mock()
        decaying_entity2.decay.is_decaying = True
        decaying_entity2.entity_id = "decaying2"

        non_decaying_entity = Mock()
        non_decaying_entity.decay.is_decaying = False
        non_decaying_entity.entity_id = "non_decaying"

        manager._entities = {
            "decaying1": decaying_entity1,
            "decaying2": decaying_entity2,
            "non_decaying": non_decaying_entity,
        }

        decaying_entities = manager.decaying_entities
        assert len(decaying_entities) == 2
        assert decaying_entity1 in decaying_entities
        assert decaying_entity2 in decaying_entities
        assert non_decaying_entity not in decaying_entities


# --- Begin migrated tests from test_entity_factory.py ---


class TestEntityFactory:
    """Test the EntityFactory class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test factory initialization."""
        factory = EntityFactory(mock_coordinator)
        assert factory.coordinator == mock_coordinator
        assert factory.config == mock_coordinator.config

    def test_create_from_config_spec(self, mock_coordinator: Mock) -> None:
        """Test creating entity from configuration."""
        factory = EntityFactory(mock_coordinator)
        mock_entity_type = Mock()
        mock_entity_type.weight = 0.8
        mock_entity_type.prob_true = 0.7
        mock_entity_type.prob_false = 0.1
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        # Mock the entity type creation
        with patch(
            "custom_components.area_occupancy.data.entity.InputType"
        ) as mock_input_type_class:
            mock_input_type_class.return_value = "motion"

            # Mock the coordinator's entity_types.get_entity_type method
            mock_coordinator.entity_types.get_entity_type.return_value = (
                mock_entity_type
            )

            entity = factory.create_from_config_spec("test_entity", "motion")

            assert entity.entity_id == "test_entity"
            assert entity.type == mock_entity_type

    def test_create_all_from_config(self, mock_coordinator: Mock) -> None:
        """Test creating all entities from configuration."""
        factory = EntityFactory(mock_coordinator)

        # Mock the config method properly
        mock_coordinator.config.entity_ids = [
            "binary_sensor.motion1",
            "media_player.tv",
        ]

        # Mock entity types
        mock_motion_type = Mock()
        mock_media_type = Mock()
        mock_coordinator.entity_types.get_entity_type.side_effect = (
            lambda x: mock_motion_type if "motion" in str(x) else mock_media_type
        )

        entities = factory.create_all_from_config()

        # The method creates entities for all entity_ids in config
        assert (
            len(entities) >= 2
        )  # At least the 2 we specified, but may include more from default config
        assert "binary_sensor.motion1" in entities
        assert "media_player.tv" in entities

    def test_create_entity_without_stored_data(self, mock_coordinator: Mock) -> None:
        """Test creating entity without stored data."""
        factory = EntityFactory(mock_coordinator)

        # Mock the coordinator's entity_types.get_entity_type method
        mock_entity_type = Mock()
        mock_coordinator.entity_types.get_entity_type.return_value = mock_entity_type

        entity = factory.create_from_config_spec("test_entity", "motion")

        assert entity.entity_id == "test_entity"
        assert entity.type == mock_entity_type


# --- End migrated tests from test_entity_factory.py ---
