"""Tests for the entity module."""

from datetime import datetime
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity, EntityManager
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.data.prior import Prior
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestEntity:
    """Test the Entity class."""

    def test_initialization(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test entity initialization."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
            state=STATE_ON,
            is_active=True,
            available=True,
        )

        assert entity.entity_id == "binary_sensor.test_motion"
        assert entity.type == mock_entity_type
        assert entity.probability == 0.5
        assert entity.prior == mock_prior
        assert entity.decay == mock_decay
        assert entity.state == STATE_ON
        assert entity.is_active is True
        assert entity.available is True
        assert entity._coordinator is None
        assert entity.previous_probability == 0.0
        assert entity.previous_is_active is False

    def test_set_coordinator(
        self,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test setting coordinator reference."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )

        entity.set_coordinator(mock_coordinator)
        assert entity._coordinator == mock_coordinator

    def test_to_dict(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test converting entity to dictionary."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
            state=STATE_ON,
            is_active=True,
            available=True,
        )

        data = entity.to_dict()
        assert data["entity_id"] == "binary_sensor.test_motion"
        assert data["type"] == mock_entity_type.input_type.value
        assert data["probability"] == 0.5
        assert data["prior"] == mock_prior.to_dict.return_value
        assert data["decay"] == mock_decay.to_dict.return_value
        assert data["state"] == STATE_ON
        assert data["is_active"] is True
        assert data["available"] is True

    @patch("custom_components.area_occupancy.data.entity.Decay")
    @patch("custom_components.area_occupancy.data.entity.Prior")
    def test_from_dict(
        self, mock_prior_class: Mock, mock_decay_class: Mock, mock_coordinator: Mock
    ) -> None:
        """Test creating entity from dictionary."""
        mock_entity_type = Mock()
        mock_entity_type.input_type = InputType.MOTION
        mock_coordinator.entity_types.get_entity_type.return_value = mock_entity_type

        mock_prior = Mock()
        mock_prior_class.from_dict.return_value = mock_prior

        mock_decay = Mock()
        mock_decay_class.from_dict.return_value = mock_decay

        current_time = dt_util.utcnow()
        data = {
            "entity_id": "binary_sensor.test_motion",
            "type": "motion",
            "probability": 0.5,
            "prior": {
                "prior": 0.3,
                "prob_given_true": 0.8,
                "prob_given_false": 0.1,
                "last_updated": current_time.isoformat(),
            },
            "decay": {"is_decaying": False},
            "state": STATE_ON,
            "is_active": True,
            "available": True,
            "last_updated": current_time.isoformat(),
        }

        entity = Entity.from_dict(data, mock_coordinator)

        assert entity.entity_id == "binary_sensor.test_motion"
        assert entity.type == mock_entity_type
        assert entity.probability == 0.5
        assert entity.prior == mock_prior
        assert entity.decay == mock_decay
        assert entity.state == STATE_ON
        assert entity.is_active is True
        assert entity.available is True
        assert entity._coordinator == mock_coordinator
        assert entity.previous_probability == 0.5
        assert entity.previous_is_active is True

    def test_stop_decay_completely(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test stopping decay completely."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )

        entity.stop_decay_completely()
        assert mock_decay.is_decaying is False

    def test_cleanup(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test entity cleanup."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )

        entity.cleanup()
        assert mock_decay.is_decaying is False

    def test_get_state_edge(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test state edge detection."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )

        # No change
        entity.is_active = False
        entity.previous_is_active = False
        assert entity.get_state_edge() is None

        # Rising edge (OFF->ON)
        entity.is_active = True
        entity.previous_is_active = False
        assert entity.get_state_edge() is True

        # Falling edge (ON->OFF)
        entity.is_active = False
        entity.previous_is_active = True
        assert entity.get_state_edge() is False

    async def test_update_probability_no_change(
        self,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test probability update with no state change."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
            is_active=True,
            previous_is_active=True,
        )
        entity.set_coordinator(mock_coordinator)

        # Configure mocks
        mock_prior.prob_given_true = 0.8
        mock_prior.prob_given_false = 0.1
        mock_entity_type.weight = 1.0
        mock_decay.is_decaying = False
        mock_decay.decay_factor = 1.0

        # No state change
        await entity.update_probability()

        # Verify no decay changes
        assert mock_decay.is_decaying is False
        mock_coordinator.async_notify_decay_started.assert_not_called()
        mock_coordinator.async_notify_decay_stopped.assert_not_called()

        # Verify probability unchanged
        assert entity.probability == 0.5
        assert entity.previous_probability == 0.5
        assert entity.previous_is_active is True

    def test_start_decay_timer(
        self,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test starting decay timer."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )
        entity.set_coordinator(mock_coordinator)

        # Test with decay enabled
        mock_decay.is_decaying = True
        entity.start_decay_timer()
        mock_coordinator.async_notify_decay_started.assert_called_once()

        # Test with decay disabled
        mock_decay.is_decaying = False
        mock_coordinator.async_notify_decay_started.reset_mock()
        entity.start_decay_timer()
        mock_coordinator.async_notify_decay_started.assert_not_called()

        # Test without coordinator
        entity._coordinator = None
        entity.start_decay_timer()  # Should not raise

    def test_stop_decay_timer(
        self,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test stopping decay timer."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )
        entity.set_coordinator(mock_coordinator)

        entity._stop_decay_timer()
        mock_coordinator.async_notify_decay_stopped.assert_called_once()

        # Test without coordinator
        entity._coordinator = None
        entity._stop_decay_timer()  # Should not raise

    async def test_async_state_changed_listener(self, mock_coordinator: Mock) -> None:
        """Test state change listener with various scenarios."""
        manager = EntityManager(mock_coordinator)

        # Create a real Entity instance with proper initialization
        from custom_components.area_occupancy.data.entity import Entity
        from custom_components.area_occupancy.data.entity_type import (
            EntityType,
            InputType,
        )

        # Create required components
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=1.0,
            active_states=[STATE_ON],
            prob_true=0.8,
            prob_false=0.1,
            prior=0.5,
        )
        prior = Prior(
            prior=0.5,
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
        )
        decay = Decay(last_trigger_ts=time.time(), half_life=60.0)

        # Create the entity
        entity = Entity(
            entity_id="test_entity",
            type=entity_type,
            probability=0.5,
            prior=prior,
            decay=decay,
            state=None,
            is_active=False,
            previous_is_active=False,
            available=True,
        )
        entity.set_coordinator(mock_coordinator)

        manager._entities = {"test_entity": entity}

        # Test state change to ON
        event = Mock()
        new_state = Mock()
        new_state.state = STATE_ON
        new_state.last_changed = dt_util.utcnow()
        new_state.last_updated = dt_util.utcnow()
        event.data = {
            "entity_id": "test_entity",
            "new_state": new_state,
        }
        await manager.async_state_changed_listener(event)

        assert entity.state == STATE_ON
        assert entity.available is True
        assert entity.is_active is True

        # Test state change to unavailable
        event = Mock()  # Create a new event for the unavailable state
        event.data = {
            "entity_id": "test_entity",
            "new_state": None,  # Home Assistant sets new_state to None for unavailable entities
        }
        await manager.async_state_changed_listener(event)

        assert entity.state is None  # State should be None for unavailable entities
        assert entity.available is False
        assert entity.is_active is False

        # Test unknown entity
        event = Mock()
        event.data = {
            "entity_id": "unknown_entity",
            "new_state": Mock(state=STATE_ON),
        }
        await manager.async_state_changed_listener(event)
        # Should not raise or modify any entities

    async def test_create_entity(
        self, mock_coordinator: Mock, mock_entity_type: Mock
    ) -> None:
        """Test entity creation with various scenarios."""
        manager = EntityManager(mock_coordinator)

        # Test creating entity with current HA state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state

        entity = await manager._create_entity(
            entity_id="test_entity",
            entity_type=mock_entity_type,
        )

        assert entity.entity_id == "test_entity"
        assert entity.type == mock_entity_type
        assert entity.state == STATE_ON
        assert entity.available is True
        assert entity.is_active is True
        assert entity._coordinator == mock_coordinator

        # Test creating entity with no HA state
        mock_coordinator.hass.states.get.return_value = None
        entity = await manager._create_entity(
            entity_id="test_entity",
            entity_type=mock_entity_type,
        )

        assert entity.state is None
        assert entity.available is False
        assert entity.is_active is False

        # Test creating entity with provided state
        entity = await manager._create_entity(
            entity_id="test_entity",
            entity_type=mock_entity_type,
            state=STATE_OFF,
            is_active=False,
            available=True,
        )

        assert entity.state == STATE_OFF
        assert entity.available is True
        assert entity.is_active is False

    async def test_calculate_initial_prior(
        self, mock_coordinator: Mock, mock_entity_type: Mock
    ) -> None:
        """Test initial prior calculation."""
        manager = EntityManager(mock_coordinator)

        # Create a mock entity for the manager
        mock_entity = Mock()
        mock_entity.entity_id = "test_entity"
        manager._entities = {"test_entity": mock_entity}

        # Test successful prior calculation
        mock_prior = Mock()
        mock_prior.prior = 0.35
        mock_prior.prob_given_true = 0.25
        mock_prior.prob_given_false = 0.05
        mock_prior.last_updated = dt_util.utcnow()
        mock_coordinator.priors.calculate.return_value = mock_prior

        prior = await manager._calculate_initial_prior("test_entity", mock_entity_type)
        assert prior == mock_prior
        mock_coordinator.priors.calculate.assert_called_once()

        # Test fallback to defaults on error
        mock_coordinator.priors.calculate.side_effect = ValueError("Test error")
        prior = await manager._calculate_initial_prior("test_entity", mock_entity_type)

        assert prior.prior == mock_entity_type.prior
        assert prior.prob_given_true == mock_entity_type.prob_true
        assert prior.prob_given_false == mock_entity_type.prob_false
        assert isinstance(prior.last_updated, datetime)


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
        manager._entities = {"test1": Mock(), "test2": Mock()}
        assert set(manager.entity_ids) == {"test1", "test2"}

    def test_active_entities_property(self, mock_coordinator: Mock) -> None:
        """Test active_entities property."""
        manager = EntityManager(mock_coordinator)

        # Create mock entities
        active_entity = Mock()
        active_entity.is_active = True
        active_entity.decay.is_decaying = False
        inactive_entity = Mock()
        inactive_entity.is_active = False
        inactive_entity.decay.is_decaying = False
        decaying_entity = Mock()
        decaying_entity.is_active = False
        decaying_entity.decay.is_decaying = True

        manager._entities = {
            "active": active_entity,
            "inactive": inactive_entity,
            "decaying": decaying_entity,
        }

        active_entities = manager.active_entities
        assert len(active_entities) == 2
        assert active_entity in active_entities
        assert decaying_entity in active_entities

    def test_inactive_entities_property(self, mock_coordinator: Mock) -> None:
        """Test inactive_entities property."""
        manager = EntityManager(mock_coordinator)

        # Create mock entities
        active_entity = Mock()
        active_entity.is_active = True
        active_entity.decay.is_decaying = False
        inactive_entity = Mock()
        inactive_entity.is_active = False
        inactive_entity.decay.is_decaying = False
        decaying_entity = Mock()
        decaying_entity.is_active = False
        decaying_entity.decay.is_decaying = True

        manager._entities = {
            "active": active_entity,
            "inactive": inactive_entity,
            "decaying": decaying_entity,
        }

        inactive_entities = manager.inactive_entities
        assert len(inactive_entities) == 1
        assert inactive_entity in inactive_entities

    def test_to_dict(self, mock_coordinator: Mock) -> None:
        """Test converting manager to dictionary."""
        manager = EntityManager(mock_coordinator)

        # Add mock entity
        mock_entity = Mock()
        mock_entity.to_dict.return_value = {"test": "data"}
        manager._entities = {"test_entity": mock_entity}

        data = manager.to_dict()
        assert data == {"entities": {"test_entity": {"test": "data"}}}

    @patch("custom_components.area_occupancy.data.entity.Entity.from_dict")
    def test_from_dict(self, mock_from_dict: Mock, mock_coordinator: Mock) -> None:
        """Test creating manager from dictionary."""

        # Create mock entity
        mock_entity = Mock()
        mock_from_dict.return_value = mock_entity

        data = {
            "entities": {
                "test_entity": {
                    "entity_id": "test_entity",
                    "type": "motion",
                }
            }
        }

        result = EntityManager.from_dict(data, mock_coordinator)

        assert isinstance(result, EntityManager)
        assert result.coordinator == mock_coordinator
        assert "test_entity" in result._entities
        assert result._entities["test_entity"] == mock_entity

    async def test_async_initialize_with_existing_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test async initialization with existing entities."""
        manager = EntityManager(mock_coordinator)
        manager._entities = {"test": Mock()}  # Add an existing entity

        # Mock the update and create methods
        manager._update_entities_from_config = AsyncMock()
        manager._create_entities_from_config = AsyncMock()

        await manager.async_initialize()

        manager._update_entities_from_config.assert_called_once()
        manager._create_entities_from_config.assert_not_called()

    async def test_async_initialize_without_existing_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test async initialization without existing entities."""
        manager = EntityManager(mock_coordinator)
        manager._entities = {}  # No existing entities

        # Create a mock entity to be returned
        mock_entity = Mock()
        mock_entity.entity_id = "new"

        # Mock the update and create methods
        manager._update_entities_from_config = AsyncMock()
        manager._create_entities_from_config = AsyncMock(
            return_value={"new": mock_entity}
        )

        await manager.async_initialize()

        manager._update_entities_from_config.assert_not_called()
        manager._create_entities_from_config.assert_called_once()
        assert "new" in manager._entities
        assert manager._entities["new"] is mock_entity

    def test_get_entity(self, mock_coordinator: Mock) -> None:
        """Test getting entity."""
        manager = EntityManager(mock_coordinator)

        # Add mock entity
        mock_entity = Mock()
        manager._entities = {"test_entity": mock_entity}

        assert manager.get_entity("test_entity") == mock_entity
        with pytest.raises(
            ValueError, match="Entity not found for entity: nonexistent"
        ):
            manager.get_entity("nonexistent")

    def test_add_entity(self, mock_coordinator: Mock) -> None:
        """Test adding entity."""
        manager = EntityManager(mock_coordinator)

        # Create mock entity
        mock_entity = Mock()
        mock_entity.entity_id = "test_entity"

        manager.add_entity(mock_entity)

        assert "test_entity" in manager._entities
        assert manager._entities["test_entity"] == mock_entity

    def test_remove_entity(self, mock_coordinator: Mock) -> None:
        """Test removing entity."""
        manager = EntityManager(mock_coordinator)

        # Add mock entity
        mock_entity = Mock()
        manager._entities = {"test_entity": mock_entity}

        manager.remove_entity("test_entity")

        assert "test_entity" not in manager._entities

    def test_cleanup(self, mock_coordinator: Mock) -> None:
        """Test manager cleanup."""
        manager = EntityManager(mock_coordinator)

        # Add mock entities
        mock_entity1 = Mock()
        mock_entity1.stop_decay_completely = Mock()
        mock_entity1.cleanup = Mock()
        mock_entity2 = Mock()
        mock_entity2.stop_decay_completely = Mock()
        mock_entity2.cleanup = Mock()

        manager._entities = {
            "entity1": mock_entity1,
            "entity2": mock_entity2,
        }

        manager.cleanup()

        mock_entity1.stop_decay_completely.assert_called_once()
        mock_entity1.cleanup.assert_called_once()
        mock_entity2.stop_decay_completely.assert_called_once()
        mock_entity2.cleanup.assert_called_once()

    @patch("homeassistant.helpers.event.async_track_state_change_event")
    async def test_initialize_states(
        self, mock_track: Mock, mock_coordinator: Mock
    ) -> None:
        """Test initializing entity states."""
        manager = EntityManager(mock_coordinator)

        # Add mock entities with proper type configuration
        mock_entity1 = Mock()
        mock_entity1.entity_id = "test1"
        mock_entity1.update_probability = AsyncMock()
        mock_entity1.type = Mock()
        mock_entity1.type.active_states = [STATE_ON]
        mock_entity1.type.active_range = None

        mock_entity2 = Mock()
        mock_entity2.entity_id = "test2"
        mock_entity2.update_probability = AsyncMock()
        mock_entity2.type = Mock()
        mock_entity2.type.active_states = [STATE_ON]
        mock_entity2.type.active_range = None

        manager._entities = {
            "test1": mock_entity1,
            "test2": mock_entity2,
        }

        # Mock state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state

        await manager.initialize_states()

        # Should update probabilities
        mock_entity1.update_probability.assert_called_once()
        mock_entity2.update_probability.assert_called_once()

        # Verify entity states were updated
        assert mock_entity1.state == STATE_ON
        assert mock_entity1.available is True
        assert mock_entity1.is_active is True
        assert mock_entity2.state == STATE_ON
        assert mock_entity2.available is True
        assert mock_entity2.is_active is True
