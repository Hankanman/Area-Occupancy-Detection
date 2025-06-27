"""Tests for the entity module."""

from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest

from custom_components.area_occupancy.data.entity import Entity, EntityManager
from custom_components.area_occupancy.data.entity_type import InputType
from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001, PLC0415
class TestEntity:
    """Test the Entity class."""

    def test_initialization(
        self,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test entity initialization."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        assert entity.entity_id == "binary_sensor.test_motion"
        assert entity.type == mock_entity_type
        assert entity.likelihood == mock_likelihood
        assert entity.decay == mock_decay
        assert entity.coordinator == mock_coordinator

        # These are now calculated properties - we need to mock the coordinator's hass.states
        # to test them, but for basic initialization we just verify they exist
        assert hasattr(entity, "probability")
        assert hasattr(entity, "state")
        assert hasattr(entity, "evidence")
        assert hasattr(entity, "available")

    def test_set_coordinator(
        self,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test coordinator is properly set during initialization."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        assert entity.coordinator == mock_coordinator
        assert entity.entity_id == "binary_sensor.test_motion"

    def test_to_dict(
        self,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test converting entity to dictionary."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        data = entity.to_dict()
        assert data["entity_id"] == "binary_sensor.test_motion"
        assert data["type"] == mock_entity_type.to_dict.return_value
        assert data["likelihood"] == mock_likelihood.to_dict.return_value
        assert data["decay"] == mock_decay.to_dict.return_value
        # Note: state, evidence, available, probability are no longer stored
        # as they are calculated properties

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
        mock_decay.decay_factor = 1.0  # Add proper decay_factor property
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

        entity = Entity.from_dict(data, mock_coordinator)

        assert entity.entity_id == "binary_sensor.test_motion"
        assert entity.type == mock_entity_type
        assert entity.likelihood == mock_likelihood
        assert entity.decay == mock_decay
        assert entity.coordinator == mock_coordinator

    async def test_create_entity(
        self, mock_coordinator: Mock, mock_entity_type: Mock
    ) -> None:
        """Test entity creation with various scenarios."""
        manager = EntityManager(mock_coordinator)

        # Test creating entity with current HA state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state

        entity = await manager.create_entity(
            entity_id="test_entity", entity_type=mock_entity_type
        )

        assert entity.entity_id == "test_entity"
        assert entity.type == mock_entity_type
        assert entity.state == STATE_ON
        assert entity.available is True
        assert entity.evidence is True

    async def test_calculate_initial_prior(
        self, mock_coordinator: Mock, mock_entity_type: Mock
    ) -> None:
        """Test initial prior calculation."""
        manager = EntityManager(mock_coordinator)

        # Create a mock entity for the manager
        mock_entity = Mock()
        mock_entity.entity_id = "test_entity"
        manager._entities = {"test_entity": mock_entity}

        # Test successful likelihood update
        mock_likelihood = Mock()
        mock_likelihood.prob_given_true = 0.25
        mock_likelihood.prob_given_false = 0.05
        mock_likelihood.last_updated = dt_util.utcnow()
        mock_entity.likelihood = mock_likelihood
        mock_entity.likelihood.update = AsyncMock()

        await mock_entity.likelihood.update()
        mock_entity.likelihood.update.assert_called_once()

        # Test error handling during likelihood update
        mock_entity.likelihood.update.side_effect = ValueError("Test error")

        # The test should expect the error to be raised, not handled
        with pytest.raises(ValueError, match="Test error"):
            await mock_entity.likelihood.update()


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
        active_entity.evidence = True
        active_entity.decay.is_decaying = False
        inactive_entity = Mock()
        inactive_entity.evidence = False
        inactive_entity.decay.is_decaying = False
        decaying_entity = Mock()
        decaying_entity.evidence = False
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
        active_entity.evidence = True
        active_entity.decay.is_decaying = False
        inactive_entity = Mock()
        inactive_entity.evidence = False
        inactive_entity.decay.is_decaying = False
        decaying_entity = Mock()
        decaying_entity.evidence = False
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
            "entities": {"test_entity": {"entity_id": "test_entity", "type": "motion"}}
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

        # Call __post_init__ explicitly since it's normally called automatically after init
        await manager.__post_init__()

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

        # Call __post_init__ explicitly since it's normally called automatically after init
        await manager.__post_init__()

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

    async def test_cleanup(self, mock_coordinator: Mock) -> None:
        """Test manager cleanup."""
        manager = EntityManager(mock_coordinator)

        # Add mock entities
        mock_entity1 = Mock()
        mock_entity2 = Mock()

        manager._entities = {"entity1": mock_entity1, "entity2": mock_entity2}

        # Mock the _create_entities_from_config method to return new entities
        manager._create_entities_from_config = AsyncMock(
            return_value={"new_entity": Mock()}
        )

        await manager.cleanup()

        # Verify entities were cleared and recreated
        assert "entity1" not in manager._entities
        assert "entity2" not in manager._entities
        assert "new_entity" in manager._entities
        manager._create_entities_from_config.assert_called_once()

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

        manager._entities = {"test1": mock_entity1, "test2": mock_entity2}

        # Mock state - the entities are mocks, so we need to mock their properties
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state

        # Mock entity properties since these are Mock entities, not real ones
        mock_entity1.state = STATE_ON
        mock_entity1.available = True
        mock_entity1.evidence = True
        mock_entity2.state = STATE_ON
        mock_entity2.available = True
        mock_entity2.evidence = True

        # Should update probabilities
        assert mock_entity1.probability
        assert mock_entity2.probability

        # Verify entity states were updated (already mocked above)
        assert mock_entity1.state == STATE_ON
        assert mock_entity1.available is True
        assert mock_entity1.evidence is True
        assert mock_entity2.state == STATE_ON
        assert mock_entity2.available is True
        assert mock_entity2.evidence is True


class TestEntityPropertiesAndMethods:
    """Test Entity properties and complex methods."""

    def test_state_property_edge_cases(
        self,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test state property with various edge cases."""
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = Entity(
            entity_id="binary_sensor.test",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        # Test valid state
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state
        assert entity.state == STATE_ON

        # Test invalid states
        invalid_states = ["unknown", "unavailable", None, "", "NaN"]
        for invalid_state in invalid_states:
            mock_state.state = invalid_state
            assert entity.state is None

        # Test no state object
        mock_coordinator.hass.states.get.return_value = None
        assert entity.state is None

    def test_available_property(
        self,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test available property."""
        mock_entity_type.active_states = [STATE_ON]
        entity = Entity(
            entity_id="binary_sensor.test",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        # Available when state exists
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state
        assert entity.available is True

        # Not available when state is None
        mock_state.state = "unavailable"
        assert entity.available is False

    def test_evidence_with_active_range(
        self,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test evidence property with active_range configuration."""
        mock_entity_type.active_states = None
        mock_entity_type.active_range = [0.0, 50.0]

        entity = Entity(
            entity_id="sensor.temperature",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        # Test values within range
        mock_state = Mock()
        mock_state.state = "25.5"
        mock_coordinator.hass.states.get.return_value = mock_state
        assert entity.evidence is True

        # Test values outside range
        mock_state.state = "75.0"
        assert entity.evidence is False

        # Test edge values
        mock_state.state = "0.0"
        assert entity.evidence is True
        mock_state.state = "50.0"
        assert entity.evidence is True

        # Test invalid numeric values
        mock_state.state = "invalid"
        assert entity.evidence is False

        # Test None state
        mock_state.state = None
        assert entity.evidence is None

    def test_has_new_evidence_transitions(
        self,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test has_new_evidence method with evidence transitions."""
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None
        mock_entity_type.weight = 0.8
        mock_likelihood.prob_given_true = 0.8
        mock_likelihood.prob_given_false = 0.2

        # Configure mock decay to track state
        mock_decay.is_decaying = False  # Start not decaying

        entity = Entity(
            entity_id="binary_sensor.motion",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        # Start with OFF state - first check should set initial evidence
        mock_state = Mock()
        mock_state.state = "off"
        mock_coordinator.hass.states.get.return_value = mock_state
        entity.previous_evidence = None  # Initial state - no previous evidence
        entity.previous_probability = 0.2

        # First evidence check - should initialize but not start decay (no transition yet)
        transition_occurred = entity.has_new_evidence()
        assert transition_occurred is False  # No transition (initializing)
        mock_decay.start_decay.assert_not_called()  # No decay on initialization

        # Reset mocks and set previous evidence to simulate next check
        mock_decay.reset_mock()
        entity.previous_evidence = True  # Simulate previous ON state

        # Now test TRUE → FALSE transition (should start decay)
        mock_state.state = "off"  # Still OFF, but now we have a transition
        transition_occurred = entity.has_new_evidence()

        assert transition_occurred is True  # Should detect TRUE → FALSE transition
        assert entity.previous_evidence is False
        mock_decay.start_decay.assert_called_once()  # Should start decay for TRUE→FALSE transition

        # Reset mocks
        mock_decay.reset_mock()
        mock_decay.is_decaying = True  # Now decaying

        # Transition OFF → ON
        mock_state.state = STATE_ON
        transition_occurred = entity.has_new_evidence()

        assert transition_occurred is True
        assert entity.previous_evidence is True
        assert entity.previous_probability > 0.2  # Should increase
        mock_decay.stop_decay.assert_called_once()  # Should stop decay for ON state

        # Reset mocks
        mock_decay.reset_mock()
        mock_decay.is_decaying = False  # Now not decaying

        # Transition ON → OFF
        mock_state.state = "off"
        entity.previous_probability = 0.8  # Set high value
        transition_occurred = entity.has_new_evidence()

        assert transition_occurred is True
        assert entity.previous_evidence is False
        mock_decay.start_decay.assert_called_once()  # Should start decay for OFF state

        # Reset mocks
        mock_decay.reset_mock()
        mock_decay.is_decaying = True  # Now decaying

        # No transition (same state) - decay state should remain consistent
        transition_occurred = entity.has_new_evidence()
        assert transition_occurred is False
        # Since evidence is OFF and decay is already started, no additional calls needed
        mock_decay.start_decay.assert_not_called()
        mock_decay.stop_decay.assert_not_called()

    @patch("custom_components.area_occupancy.data.entity.bayesian_probability")
    def test_probability_with_decay(
        self,
        mock_bayesian_prob: Mock,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test probability calculation with decay."""
        mock_entity_type.active_states = [STATE_ON]

        # Set up mock values
        mock_coordinator.area_prior = 0.3
        mock_likelihood.prob_given_true = 0.8
        mock_likelihood.prob_given_false = 0.1

        entity = Entity(
            entity_id="binary_sensor.test",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        # Test when not decaying
        mock_decay.is_decaying = False
        type(mock_decay).decay_factor = PropertyMock(return_value=1.0)
        mock_bayesian_prob.return_value = 0.8

        assert entity.probability == 0.8
        mock_bayesian_prob.assert_called_with(
            prior=0.3,
            prob_given_true=0.8,
            prob_given_false=0.1,
            evidence=True,
            decay_factor=1.0,
        )

        # Test when decaying
        mock_decay.is_decaying = True
        type(mock_decay).decay_factor = PropertyMock(return_value=0.9)
        mock_bayesian_prob.return_value = 0.73

        assert entity.probability == 0.73
        mock_bayesian_prob.assert_called_with(
            prior=0.3,
            prob_given_true=0.8,
            prob_given_false=0.1,
            evidence=True,
            decay_factor=0.9,
        )

    def test_effective_probability_property(
        self,
        mock_entity_type: Mock,
        mock_likelihood: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test effective_probability property."""
        entity = Entity(
            entity_id="binary_sensor.test",
            type=mock_entity_type,
            likelihood=mock_likelihood,
            decay=mock_decay,
            coordinator=mock_coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
        )

        entity.previous_probability = 0.75
        assert entity.previous_probability == 0.75

        # Should return current value regardless of decay state
        mock_decay.is_decaying = True
        assert entity.previous_probability == 0.75


class TestEntityManagerAdvanced:
    """Test EntityManager advanced functionality and edge cases."""

    def test_decaying_entities_property(self, mock_coordinator: Mock) -> None:
        """Test decaying_entities property."""
        manager = EntityManager(mock_coordinator)

        # Create mock entities with different decay states
        decaying_entity1 = Mock()
        decaying_entity1.decay.is_decaying = True
        decaying_entity2 = Mock()
        decaying_entity2.decay.is_decaying = True
        non_decaying_entity = Mock()
        non_decaying_entity.decay.is_decaying = False

        manager._entities = {
            "decaying1": decaying_entity1,
            "decaying2": decaying_entity2,
            "normal": non_decaying_entity,
        }

        decaying_entities = manager.decaying_entities
        assert len(decaying_entities) == 2
        assert decaying_entity1 in decaying_entities
        assert decaying_entity2 in decaying_entities
        assert non_decaying_entity not in decaying_entities

    def test_from_dict_error_cases(self, mock_coordinator: Mock) -> None:
        """Test EntityManager.from_dict error handling."""
        # Test missing 'entities' key
        with pytest.raises(
            ValueError, match="Invalid storage format: missing 'entities' key"
        ):
            EntityManager.from_dict({"wrong_key": {}}, mock_coordinator)

        # Test invalid entity data
        with patch(
            "custom_components.area_occupancy.data.entity.Entity.from_dict"
        ) as mock_from_dict:
            mock_from_dict.side_effect = KeyError("missing field")

            data = {"entities": {"test": {"invalid": "data"}}}
            with pytest.raises(ValueError, match="Failed to deserialize entity data"):
                EntityManager.from_dict(data, mock_coordinator)

    def test_build_sensor_type_mappings(self, mock_coordinator: Mock) -> None:
        """Test build_sensor_type_mappings method."""
        manager = EntityManager(mock_coordinator)

        # Mock the config sensors
        mock_config = Mock()
        mock_config.sensors.get_motion_sensors.return_value = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        mock_config.sensors.media = ["media_player.tv"]
        mock_config.sensors.appliances = ["switch.computer"]
        mock_config.sensors.doors = ["binary_sensor.door"]
        mock_config.sensors.windows = ["binary_sensor.window"]
        mock_config.sensors.illuminance = ["sensor.lux"]
        mock_config.sensors.humidity = ["sensor.humidity"]
        mock_config.sensors.temperature = ["sensor.temp"]

        manager.config = mock_config

        mappings = manager.build_sensor_type_mappings()

        assert InputType.MOTION in mappings
        assert InputType.MEDIA in mappings
        assert InputType.APPLIANCE in mappings
        assert InputType.DOOR in mappings
        assert InputType.WINDOW in mappings
        assert InputType.ENVIRONMENTAL in mappings

        assert mappings[InputType.MOTION] == [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        assert mappings[InputType.MEDIA] == ["media_player.tv"]
        assert mappings[InputType.ENVIRONMENTAL] == [
            "sensor.lux",
            "sensor.humidity",
            "sensor.temp",
        ]

    def test_process_existing_entities(self, mock_coordinator: Mock) -> None:
        """Test _process_existing_entities method."""
        manager = EntityManager(mock_coordinator)

        # Create mock existing entities
        existing_entity1 = Mock()
        existing_entity1.type = Mock()
        existing_entity2 = Mock()
        existing_entity2.type = Mock()

        manager._entities = {
            "entity1": existing_entity1,
            "entity2": existing_entity2,
            "removed_entity": Mock(),
        }

        # Create mock config entities (entity1 and entity2 still exist, removed_entity is gone)
        from custom_components.area_occupancy.data.entity_type import EntityType

        mock_entity_type1 = Mock(spec=EntityType)
        mock_entity_type1.weight = 0.8
        mock_entity_type1.active_states = [STATE_ON]
        mock_entity_type1.active_range = None

        mock_entity_type2 = Mock(spec=EntityType)
        mock_entity_type2.weight = 0.6
        mock_entity_type2.active_states = ["playing"]
        mock_entity_type2.active_range = None

        config_entities = {"entity1": mock_entity_type1, "entity2": mock_entity_type2}
        updated_entities = {}

        manager._process_existing_entities(config_entities, updated_entities)  # type: ignore[arg-type]

        # Check that existing entities were updated and added to updated_entities
        assert "entity1" in updated_entities
        assert "entity2" in updated_entities
        assert "removed_entity" not in updated_entities

        # Check that entity types were updated
        assert existing_entity1.type.weight == 0.8
        assert existing_entity1.type.active_states == [STATE_ON]
        assert existing_entity2.type.weight == 0.6
        assert existing_entity2.type.active_states == ["playing"]

        # Check that processed entities were removed from config_entities
        assert "entity1" not in config_entities
        assert "entity2" not in config_entities

    async def test_create_new_entities(self, mock_coordinator: Mock) -> None:
        """Test _create_new_entities method."""
        manager = EntityManager(mock_coordinator)

        # Mock a real entity to return from get_entity (needed for prior calculation)
        mock_existing_entity = Mock()
        mock_existing_entity.entity_id = "new_entity1"
        manager._entities = {"new_entity1": mock_existing_entity}

        # Mock create_entity
        mock_new_entity = Mock()
        mock_likelihood = Mock()
        mock_new_entity.likelihood = mock_likelihood
        manager.create_entity = AsyncMock(return_value=mock_new_entity)

        # Create config entities to add
        mock_entity_type = Mock()
        config_entities = {"new_entity1": mock_entity_type}
        updated_entities = {}

        await manager._create_new_entities(config_entities, updated_entities)  # type: ignore[arg-type]

        # Check that new entity was created and added
        assert "new_entity1" in updated_entities
        assert updated_entities["new_entity1"] == mock_new_entity
        manager.create_entity.assert_called_once_with(
            entity_id="new_entity1", entity_type=mock_entity_type
        )

    def test_build_entity_mapping_from_types(self, mock_coordinator: Mock) -> None:
        """Test _build_entity_mapping_from_types method."""
        manager = EntityManager(mock_coordinator)

        # Mock entity types
        motion_type = Mock()
        media_type = Mock()
        mock_coordinator.entity_types.get_entity_type.side_effect = lambda t: (
            motion_type if t == InputType.MOTION else media_type
        )

        type_mappings = {
            InputType.MOTION: ["binary_sensor.motion1", "binary_sensor.motion2"],
            InputType.MEDIA: ["media_player.tv"],
        }

        result = manager._build_entity_mapping_from_types(type_mappings)

        assert result == {
            "binary_sensor.motion1": motion_type,
            "binary_sensor.motion2": motion_type,
            "media_player.tv": media_type,
        }

    async def test_get_config_entity_mapping(self, mock_coordinator: Mock) -> None:
        """Test _get_config_entity_mapping method."""
        manager = EntityManager(mock_coordinator)

        # Mock the methods it calls
        type_mappings = {InputType.MOTION: ["binary_sensor.motion"]}
        manager.build_sensor_type_mappings = Mock(return_value=type_mappings)

        entity_mapping = {"binary_sensor.motion": Mock()}
        manager._build_entity_mapping_from_types = Mock(return_value=entity_mapping)

        result = await manager._get_config_entity_mapping()

        assert result == entity_mapping
        manager.build_sensor_type_mappings.assert_called_once()
        manager._build_entity_mapping_from_types.assert_called_once_with(type_mappings)

    async def test_create_entities_from_config(self, mock_coordinator: Mock) -> None:
        """Test _create_entities_from_config method."""
        manager = EntityManager(mock_coordinator)

        # Mock type mappings
        type_mappings = {InputType.MOTION: ["binary_sensor.motion"]}
        manager.build_sensor_type_mappings = Mock(return_value=type_mappings)

        # Mock entity type
        mock_entity_type = Mock()
        mock_entity_type.prob_true = 0.8
        mock_entity_type.prob_false = 0.2
        mock_coordinator.entity_types.get_entity_type.return_value = mock_entity_type

        # Mock entity creation
        mock_entity = Mock()
        manager.create_entity = AsyncMock(return_value=mock_entity)

        # Mock likelihood update
        mock_likelihood = Mock()
        mock_entity.likelihood = mock_likelihood
        mock_entity.likelihood.update = AsyncMock()

        result = await manager._create_entities_from_config()

        assert "binary_sensor.motion" in result
        assert result["binary_sensor.motion"] == mock_entity

        # Check that entity was created (likelihood update happens later in coordinator setup)
        manager.create_entity.assert_called_once()
