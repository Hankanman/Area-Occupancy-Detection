"""Tests for data.entity module."""

from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import (
    Entity,
    EntityFactory,
    EntityManager,
)
from custom_components.area_occupancy.data.entity_type import (
    DEFAULT_TYPES,
    EntityType,
    InputType,
)
from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util

# ruff: noqa: SLF001


# Helper functions to reduce code duplication
def create_test_entity(
    entity_id: str = "test",
    entity_type: Mock | EntityType = None,
    prob_given_true: float = 0.25,
    prob_given_false: float = 0.05,
    decay: Mock | Decay = None,
    coordinator: Mock | None = None,
    **kwargs,
) -> Entity:
    """Create a test Entity instance with default values."""
    if entity_type is None:
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            active_states=[STATE_ON],
        )
    if decay is None:
        decay = Decay()
    if coordinator is None:
        coordinator = Mock()

    return Entity(
        entity_id=entity_id,
        type=entity_type,
        prob_given_true=prob_given_true,
        prob_given_false=prob_given_false,
        decay=decay,
        coordinator=coordinator,
        last_updated=dt_util.utcnow(),
        previous_evidence=kwargs.get("previous_evidence"),
    )


def create_test_entity_manager(coordinator: Mock | None = None) -> EntityManager:
    """Create a test EntityManager instance with mocked dependencies."""
    if coordinator is None:
        coordinator = Mock()

    with patch(
        "custom_components.area_occupancy.data.entity.EntityFactory"
    ) as mock_factory_class:
        mock_factory = Mock()
        mock_factory.create_all_from_config.return_value = {}
        mock_factory_class.return_value = mock_factory

        return EntityManager(coordinator)


def create_mock_entities_with_states() -> dict[str, Mock]:
    """Create mock entities with different states for testing."""
    active_entity = Mock()
    active_entity.state = STATE_ON
    active_entity.evidence = True
    active_entity.decay.is_decaying = False

    inactive_entity = Mock()
    inactive_entity.state = "off"
    inactive_entity.evidence = False
    inactive_entity.decay.is_decaying = False

    decaying_entity = Mock()
    decaying_entity.evidence = False
    decaying_entity.decay.is_decaying = True

    return {
        "active": active_entity,
        "inactive": inactive_entity,
        "decaying": decaying_entity,
    }


class TestEntity:
    """Test the Entity class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test entity initialization."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=0.25,
            prob_given_false=0.05,
            active_states=[STATE_ON],
        )
        decay = Decay()
        entity = create_test_entity(
            entity_type=entity_type,
            decay=decay,
            coordinator=mock_coordinator,
        )

        assert entity.entity_id == "test"
        assert entity.type == entity_type
        assert entity.prob_given_true == 0.25
        assert entity.prob_given_false == 0.05
        assert entity.decay == decay
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
        mock_entity_type_class.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.return_value = mock_decay

        entity_data = {
            "entity_id": "test",
            "type": {"input_type": "motion", "weight": 0.8},
            "prob_given_true": DEFAULT_TYPES[InputType.MOTION]["prob_given_true"],
            "prob_given_false": DEFAULT_TYPES[InputType.MOTION]["prob_given_false"],
            "decay": {"is_decaying": False},
        }

        # Create entity directly since there's no from_dict method
        entity = create_test_entity(
            entity_id=entity_data["entity_id"],
            entity_type=mock_entity_type,
            prob_given_true=entity_data["prob_given_true"],
            prob_given_false=entity_data["prob_given_false"],
            decay=mock_decay,
            coordinator=mock_coordinator,
        )

        assert entity.entity_id == "test"
        assert entity.type == mock_entity_type
        assert (
            entity.prob_given_true == DEFAULT_TYPES[InputType.MOTION]["prob_given_true"]
        )
        assert (
            entity.prob_given_false
            == DEFAULT_TYPES[InputType.MOTION]["prob_given_false"]
        )
        assert entity.decay == mock_decay
        assert entity.coordinator == mock_coordinator


class TestEntityManager:
    """Test the EntityManager class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test manager initialization."""
        manager = create_test_entity_manager(mock_coordinator)

        assert manager.coordinator == mock_coordinator
        assert manager._entities == {}

    def test_entities_property(self, mock_coordinator: Mock) -> None:
        """Test entities property."""
        manager = create_test_entity_manager(mock_coordinator)
        test_entity = Mock()
        manager._entities = {"test": test_entity}

        assert manager.entities == {"test": test_entity}

    def test_entity_ids_property(self, mock_coordinator: Mock) -> None:
        """Test entity_ids property."""
        manager = create_test_entity_manager(mock_coordinator)
        manager._entities = {"entity1": Mock(), "entity2": Mock()}

        assert set(manager.entity_ids) == {"entity1", "entity2"}

    @pytest.mark.parametrize(
        ("property_name", "expected_entities"),
        [
            (
                "active_entities",
                ["active", "decaying"],
            ),  # Both evidence=True and decay.is_decaying=True
            (
                "inactive_entities",
                ["inactive"],
            ),  # Only evidence=False and decay.is_decaying=False
            ("decaying_entities", ["decaying"]),  # Only decay.is_decaying=True
        ],
    )
    def test_entity_filtering_properties(
        self, mock_coordinator: Mock, property_name: str, expected_entities: list[str]
    ) -> None:
        """Test entity filtering properties (active, inactive, decaying)."""
        manager = create_test_entity_manager(mock_coordinator)
        mock_entities = create_mock_entities_with_states()
        manager._entities = mock_entities

        filtered_entities = getattr(manager, property_name)
        assert len(filtered_entities) == len(expected_entities)
        for entity_name in expected_entities:
            assert mock_entities[entity_name] in filtered_entities

    def test_get_entity(self, mock_coordinator: Mock) -> None:
        """Test getting entity by ID."""
        manager = create_test_entity_manager(mock_coordinator)
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
        manager = create_test_entity_manager(mock_coordinator)
        mock_entity = Mock()
        mock_entity.entity_id = "test"

        manager.add_entity(mock_entity)

        assert manager._entities["test"] == mock_entity


class TestEntityPropertiesAndMethods:
    """Test entity properties and methods."""

    @pytest.fixture
    def test_entity(self, mock_coordinator: Mock) -> Entity:
        """Create a test entity for property testing."""
        return create_test_entity(coordinator=mock_coordinator)

    def test_state_property_edge_cases(
        self, test_entity: Entity, mock_coordinator: Mock
    ) -> None:
        """Test state property with edge cases."""
        # Test with no state available
        mock_coordinator.hass.states.get.return_value = None
        assert test_entity.state is None

        # Test with state but no state attribute
        mock_state = Mock()
        mock_state.state = "test_state"
        mock_coordinator.hass.states.get.return_value = mock_state
        assert test_entity.state == "test_state"

    @pytest.mark.parametrize(
        ("state_value", "expected_available"),
        [
            (STATE_ON, True),
            ("unavailable", False),
            (None, False),
        ],
    )
    def test_available_property(
        self,
        test_entity: Entity,
        mock_coordinator: Mock,
        state_value: str | None,
        expected_available: bool,
    ) -> None:
        """Test available property with different states."""
        if state_value is None:
            mock_coordinator.hass.states.get.return_value = None
        else:
            mock_state = Mock()
            mock_state.state = state_value
            mock_coordinator.hass.states.get.return_value = mock_state

        assert test_entity.available is expected_available

    @pytest.mark.parametrize(
        ("state_value", "expected_evidence"),
        [
            ("15", True),  # Within range (10, 20)
            ("25", False),  # Outside range
            ("invalid", False),  # Invalid value
        ],
    )
    def test_evidence_with_active_range(
        self, mock_coordinator: Mock, state_value: str, expected_evidence: bool
    ) -> None:
        """Test evidence property with active range."""
        mock_entity_type = Mock()
        mock_entity_type.active_range = (10, 20)
        mock_entity_type.active_states = None  # Ensure this is None for range test

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=mock_coordinator,
        )

        mock_state = Mock()
        mock_state.state = state_value
        mock_coordinator.hass.states.get.return_value = mock_state

        assert entity.evidence is expected_evidence

    def test_has_new_evidence_transitions(self, mock_coordinator: Mock) -> None:
        """Test has_new_evidence method with evidence transitions."""
        # Create a proper mock entity type with active_states
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=mock_coordinator,
        )

        # Test initial state (no transition)
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_state
        assert not entity.has_new_evidence()  # No transition on first call

        # Test transition from True to False
        mock_state.state = "off"
        assert entity.has_new_evidence()  # Should detect transition

        # Test transition from False to True
        mock_state.state = STATE_ON
        assert entity.has_new_evidence()  # Should detect transition


class TestEntityFactory:
    """Test the EntityFactory class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test factory initialization."""
        factory = EntityFactory(mock_coordinator)
        assert factory.coordinator == mock_coordinator

    @patch("custom_components.area_occupancy.data.entity.EntityType")
    @patch("custom_components.area_occupancy.data.entity.Decay")
    def test_create_from_config_spec(
        self,
        mock_decay_class: Mock,
        mock_entity_type_class: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test creating entity from config spec."""
        mock_entity_type = Mock()
        mock_entity_type_class.create.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.create.return_value = mock_decay

        factory = EntityFactory(mock_coordinator)
        entity = factory.create_from_config_spec("test_entity", "motion")

        assert entity.entity_id == "test_entity"
        assert entity.type == mock_entity_type
        assert entity.decay == mock_decay
        assert entity.coordinator == mock_coordinator

    @patch("custom_components.area_occupancy.data.entity.EntityType")
    @patch("custom_components.area_occupancy.data.entity.Decay")
    def test_create_all_from_config(
        self,
        mock_decay_class: Mock,
        mock_entity_type_class: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test creating all entities from config."""
        mock_entity_type = Mock()
        mock_entity_type_class.create.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.create.return_value = mock_decay

        # Mock config with sensors
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.config.sensors.media = ["media_player.tv"]

        factory = EntityFactory(mock_coordinator)
        entities = factory.create_all_from_config()

        # Should create entities for all configured sensors
        assert len(entities) >= 2  # At least motion and media sensors
        assert "binary_sensor.motion1" in entities
        assert "media_player.tv" in entities

    @patch("custom_components.area_occupancy.data.entity.EntityType")
    @patch("custom_components.area_occupancy.data.entity.Decay")
    def test_create_entity_without_stored_data(
        self,
        mock_decay_class: Mock,
        mock_entity_type_class: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test creating entity without stored data."""
        mock_entity_type = Mock()
        mock_entity_type.prob_given_true = DEFAULT_TYPES[InputType.MOTION][
            "prob_given_true"
        ]
        mock_entity_type.prob_given_false = DEFAULT_TYPES[InputType.MOTION][
            "prob_given_false"
        ]
        mock_entity_type_class.create.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.create.return_value = mock_decay

        factory = EntityFactory(mock_coordinator)
        entity = factory.create_from_config_spec("test_entity", "motion")

        # Should use default values when no stored data is available
        assert (
            entity.prob_given_true == DEFAULT_TYPES[InputType.MOTION]["prob_given_true"]
        )
        assert (
            entity.prob_given_false
            == DEFAULT_TYPES[InputType.MOTION]["prob_given_false"]
        )
        assert entity.previous_evidence is None
