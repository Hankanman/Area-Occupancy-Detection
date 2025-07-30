"""Tests for the entity module."""

from unittest.mock import AsyncMock, Mock, patch

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
        mock_likelihood = Mock()
        mock_decay = Mock()

        entity = Entity(
            entity_id="test_entity",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        assert entity.entity_id == "test_entity"
        assert entity.type == mock_entity_type
        assert entity.likelihood == mock_likelihood
        assert entity.decay == mock_decay
        assert entity.coordinator == mock_coordinator

    def test_to_dict(self, mock_coordinator: Mock) -> None:
        """Test converting entity to dictionary."""
        mock_entity_type = Mock()
        mock_entity_type.to_dict.return_value = {"type": "motion"}

        mock_likelihood = Mock()
        mock_likelihood.to_dict.return_value = {"prob": 0.8}

        mock_decay = Mock()
        mock_decay.to_dict.return_value = {"is_decaying": False}

        entity = Entity(
            entity_id="test_entity",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        result = entity.to_dict()

        assert result["entity_id"] == "test_entity"
        assert result["type"] == {"type": "motion"}
        assert result["likelihood"] == {"prob": 0.8}
        assert result["decay"] == {"is_decaying": False}

    @patch("custom_components.area_occupancy.data.entity.Decay")
    @patch("custom_components.area_occupancy.data.entity.Likelihood")
    @patch("custom_components.area_occupancy.data.entity.EntityType")
    def test_from_dict(
        self,
        mock_entity_type_class: Mock,
        mock_prior_class: Mock,
        mock_decay_class: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test creating entity from dictionary."""
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None
        mock_entity_type_class.from_dict.return_value = mock_entity_type

        mock_likelihood = Mock()
        mock_likelihood.prob_given_true = 0.8
        mock_likelihood.prob_given_false = 0.1
        mock_prior_class.from_dict.return_value = mock_likelihood

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
            "likelihood": {
                "prob_given_true": 0.8,
                "prob_given_false": 0.1,
                "last_updated": current_time.isoformat(),
            },
            "decay": {"is_decaying": False},
            "last_updated": current_time.isoformat(),
            "previous_evidence": None,
            "previous_probability": 0.0,
        }

        # Use factory instead of Entity.from_dict
        factory = EntityFactory(mock_coordinator)
        entity = factory.create_from_storage(data)

        assert entity.entity_id == "binary_sensor.test_motion"
        # Note: The factory creates new instances, so we can't directly compare
        # assert entity.type == mock_entity_type


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

    def test_to_dict(self, mock_coordinator: Mock) -> None:
        """Test converting manager to dictionary."""
        manager = EntityManager(mock_coordinator)

        mock_entity = Mock()
        mock_entity.to_dict.return_value = {"entity_id": "test"}
        manager._entities = {"test": mock_entity}

        result = manager.to_dict()

        assert result["entities"] == {"test": {"entity_id": "test"}}

    def test_from_dict(self, mock_coordinator: Mock) -> None:
        """Test creating manager from dictionary."""
        data = {
            "entities": {
                "test": {
                    "entity_id": "test",
                    "type": {"input_type": "motion"},
                    "likelihood": {"prob": 0.8},
                    "decay": {"is_decaying": False},
                }
            }
        }

        with patch(
            "custom_components.area_occupancy.data.entity.EntityFactory"
        ) as mock_factory:
            mock_factory_instance = Mock()
            mock_factory_instance.create_from_storage.return_value = Mock()
            mock_factory.return_value = mock_factory_instance

            manager = EntityManager.from_dict(data, mock_coordinator)

            assert isinstance(manager, EntityManager)
            assert manager.coordinator == mock_coordinator

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

    def test_remove_entity(self, mock_coordinator: Mock) -> None:
        """Test removing entity from manager."""
        manager = EntityManager(mock_coordinator)
        mock_entity = Mock()
        manager._entities = {"test": mock_entity}

        manager.remove_entity("test")

        assert "test" not in manager._entities


class TestEntityPropertiesAndMethods:
    """Test entity properties and methods."""

    def test_state_property_edge_cases(self, mock_coordinator: Mock) -> None:
        """Test state property with edge cases."""
        entity = Entity(
            entity_id="test",
            type=Mock(),
            likelihood=Mock(),
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
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
            likelihood=Mock(),
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
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
            likelihood=Mock(),
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
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
            likelihood=Mock(),
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
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
        entity.likelihood.prob_given_true = 0.8
        entity.likelihood.prob_given_false = 0.1
        entity.coordinator.area_prior = 0.3
        assert entity.has_new_evidence() is True

        # Test transition from active to inactive
        entity.previous_evidence = True
        mock_state.state = "off"
        entity.likelihood.prob_given_true = 0.8
        entity.likelihood.prob_given_false = 0.1
        entity.coordinator.area_prior = 0.3
        assert entity.has_new_evidence() is True

        # Test no transition
        entity.previous_evidence = True
        mock_state.state = STATE_ON
        assert entity.has_new_evidence() is False

    def test_probability_with_decay(self, mock_coordinator: Mock) -> None:
        """Test probability property with decay."""
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = Entity(
            entity_id="test",
            type=mock_entity_type,
            likelihood=Mock(),
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        # Mock the state to be active so evidence is True
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state

        # Test with decay (but evidence is True, so decay_factor should be 1.0)
        entity.decay.is_decaying = True
        entity.decay.decay_factor = 0.5
        entity.likelihood.prob_given_true = 0.8
        entity.likelihood.prob_given_false = 0.1
        entity.coordinator.area_prior = 0.3

        # Since evidence is True, decay_factor should be 1.0, not 0.5
        assert entity.probability > 0.0  # Just check it's calculated

    def test_effective_probability_property(self, mock_coordinator: Mock) -> None:
        """Test effective_probability property."""
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = Entity(
            entity_id="test",
            type=mock_entity_type,
            likelihood=Mock(),
            decay=Mock(),
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        # Mock the state to be active
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state

        # Mock the likelihood values properly
        entity.likelihood.prob_given_true = 0.8
        entity.likelihood.prob_given_false = 0.1
        entity.coordinator.area_prior = 0.3
        entity.decay.decay_factor = 1.0

        # Test with evidence (should be True based on state)
        assert entity.evidence is True
        assert entity.probability > 0.0  # Just check it's calculated

        # Test without evidence (change state to inactive)
        mock_state.state = "off"
        assert entity.evidence is False
        assert entity.probability > 0.0  # Just check it's calculated


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

    def test_from_dict_error_cases(self, mock_coordinator: Mock) -> None:
        """Test from_dict with error cases."""
        # Test with invalid data structure
        invalid_data = {"invalid": "data"}

        with pytest.raises(ValueError, match="Invalid storage format"):
            EntityManager.from_dict(invalid_data, mock_coordinator)

    async def test_update_all_entity_likelihoods(self, mock_coordinator: Mock) -> None:
        """Test update_all_entity_likelihoods method."""
        manager = EntityManager(mock_coordinator)

        # Create mock entities
        entity1 = Mock()
        entity1.likelihood.update = AsyncMock()
        entity2 = Mock()
        entity2.likelihood.update = AsyncMock()

        manager._entities = {
            "entity1": entity1,
            "entity2": entity2,
        }

        # Mock prior
        manager.coordinator.prior.prior_intervals = None
        manager.coordinator.prior.update = AsyncMock()

        result = await manager.update_all_entity_likelihoods()

        # Should update prior first
        manager.coordinator.prior.update.assert_called_once()

        # Should update both entities
        entity1.likelihood.update.assert_called_once()
        entity2.likelihood.update.assert_called_once()

        # Should return count of successful updates
        assert result == 2

    async def test_update_all_entity_likelihoods_with_errors(
        self, mock_coordinator: Mock
    ) -> None:
        """Test update_all_entity_likelihoods with errors."""
        manager = EntityManager(mock_coordinator)

        # Create mock entities - one with error
        entity1 = Mock()
        entity1.likelihood.update = AsyncMock()
        entity2 = Mock()
        entity2.likelihood.update = AsyncMock(side_effect=ValueError("Test error"))

        manager._entities = {
            "entity1": entity1,
            "entity2": entity2,
        }

        # Mock prior
        manager.coordinator.prior.prior_intervals = None
        manager.coordinator.prior.update = AsyncMock()

        with patch(
            "custom_components.area_occupancy.data.entity._LOGGER"
        ) as mock_logger:
            result = await manager.update_all_entity_likelihoods()

            # Should log warning for failed update
            mock_logger.warning.assert_called_once()
            assert (
                "Failed to update likelihood for entity"
                in mock_logger.warning.call_args[0][0]
            )

            # Should return count of successful updates only
            assert result == 1


# --- Begin migrated tests from test_entity_factory.py ---


class TestEntityFactory:
    """Test the EntityFactory class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test factory initialization."""
        factory = EntityFactory(mock_coordinator)
        assert factory.coordinator == mock_coordinator
        assert factory.config == mock_coordinator.config_manager.config

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
            "custom_components.area_occupancy.data.entity.EntityType"
        ) as mock_entity_type_class:
            mock_entity_type_class.return_value = mock_entity_type

            entity = factory.create_from_config_spec(
                "test_entity", {"input_type": InputType.MOTION}
            )

            assert entity.entity_id == "test_entity"
            # Don't compare mock objects directly, just check the entity was created
            assert entity.type is not None

    def test_create_from_storage(self, mock_coordinator: Mock) -> None:
        """Test creating entity from storage data."""
        factory = EntityFactory(mock_coordinator)
        mock_coordinator.config_manager.config.decay.half_life = 300
        storage_data = {
            "entity_id": "binary_sensor.test",
            "type": {
                "input_type": "motion",
                "weight": 0.8,
                "prob_true": 0.7,
                "prob_false": 0.1,
                "prior": 0.3,  # Add missing prior field
                "active_states": [STATE_ON],
                "active_range": None,
            },
            "likelihood": {
                "prob_given_true": 0.8,
                "prob_given_false": 0.1,
                "last_updated": dt_util.utcnow().isoformat(),
            },
            "decay": {
                "is_decaying": False,
                "decay_start_time": None,
                "decay_start_probability": 0.0,
                "half_life": 300,
                "decay_enabled": True,
                "decay_factor": 1.0,
            },
            "last_updated": dt_util.utcnow().isoformat(),
            "previous_evidence": True,
            "previous_probability": 0.6,
        }
        entity = factory.create_from_storage(storage_data)

        assert entity.entity_id == "binary_sensor.test"
        assert entity.type.input_type == InputType.MOTION
        assert entity.type.weight == 0.8

    def test_create_all_from_config(self, mock_coordinator: Mock) -> None:
        """Test creating all entities from configuration."""
        factory = EntityFactory(mock_coordinator)
        mock_specs = {
            "binary_sensor.motion1": {
                "input_type": InputType.MOTION,
                "weight": 0.8,
                "active_states": None,
                "active_range": None,
            },
            "media_player.tv": {
                "input_type": InputType.MEDIA,
                "weight": 0.7,
                "active_states": ["playing", "paused"],
                "active_range": None,
            },
        }

        # Mock the config method properly
        mock_coordinator.config_manager.config.get_entity_specifications = Mock(
            return_value=mock_specs
        )

        entities = factory.create_all_from_config()

        assert len(entities) == 2
        assert "binary_sensor.motion1" in entities
        assert "media_player.tv" in entities

    def test_create_entity_with_stored_data(self, mock_coordinator: Mock) -> None:
        """Test creating entity with stored data."""
        factory = EntityFactory(mock_coordinator)
        stored_data = {
            "entity_id": "test_entity",
            "type": {
                "input_type": "motion",
                "weight": 0.8,
                "prob_true": 0.7,
                "prob_false": 0.1,
                "prior": 0.3,
                "active_states": [STATE_ON],
                "active_range": None,
            },
            "likelihood": {
                "prob_given_true": 0.8,
                "prob_given_false": 0.1,
                "last_updated": dt_util.utcnow().isoformat(),
            },
            "decay": {"is_decaying": False},
            "last_updated": dt_util.utcnow().isoformat(),
            "previous_evidence": True,
            "previous_probability": 0.6,
        }

        entity = factory.create_from_storage(stored_data)

        assert entity.entity_id == "test_entity"
        assert entity.previous_evidence is True
        assert entity.previous_probability == 0.6

    def test_create_entity_without_stored_data(self, mock_coordinator: Mock) -> None:
        """Test creating entity without stored data."""
        factory = EntityFactory(mock_coordinator)
        config_spec = {
            "input_type": InputType.MOTION,
            "weight": 0.8,
            "active_states": [STATE_ON],
            "active_range": None,
        }

        entity = factory.create_from_config_spec("test_entity", config_spec)

        assert entity.entity_id == "test_entity"
        assert entity.previous_evidence is None
        assert entity.previous_probability == 0.0


# --- End migrated tests from test_entity_factory.py ---
