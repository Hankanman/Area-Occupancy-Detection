"""Tests for data.entity module."""

from datetime import timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
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


def _set_states_get(hass, mock_get):
    """Helper to set hass.states.get by replacing the entire states object."""
    # Replace the entire states object with a mock that has a get method
    mock_states = MagicMock()
    mock_states.get = mock_get
    mock_states.async_set = hass.states.async_set  # Preserve async_set
    mock_states.async_all = hass.states.async_all  # Preserve other methods if needed
    object.__setattr__(hass, "states", mock_states)


# Helper functions to reduce code duplication
def create_test_entity(
    entity_id: str = "test",
    entity_type: Mock | EntityType = None,
    prob_given_true: float = 0.25,
    prob_given_false: float = 0.05,
    decay: Mock | Decay = None,
    coordinator: Mock | None = None,
    hass: Mock | None = None,
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
        decay = Decay(half_life=60.0)  # half_life is required
    if hass is None:
        if coordinator is not None:
            hass = coordinator.hass
        else:
            hass = Mock()

    return Entity(
        entity_id=entity_id,
        type=entity_type,
        prob_given_true=prob_given_true,
        prob_given_false=prob_given_false,
        decay=decay,
        hass=hass,
        last_updated=dt_util.utcnow(),
        previous_evidence=kwargs.get("previous_evidence"),
    )


def create_test_entity_manager(
    coordinator: AreaOccupancyCoordinator | None = None, area_name: str | None = None
) -> EntityManager:
    """Create a test EntityManager instance with real coordinator.

    Args:
        coordinator: Real coordinator instance (will use coordinator_with_areas if None)
        area_name: Area name to use (will use first area from coordinator if None)
    """
    if coordinator is None:
        # This will be provided by fixture
        raise ValueError("coordinator must be provided")

    if area_name is None:
        area_name = coordinator.get_area_names()[0]

    with patch(
        "custom_components.area_occupancy.data.entity.EntityFactory"
    ) as mock_factory_class:
        mock_factory = Mock()
        mock_factory.create_all_from_config.return_value = {}
        mock_factory_class.return_value = mock_factory

        return EntityManager(coordinator, area_name=area_name)


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

    def test_initialization(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test entity initialization."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=0.25,
            prob_given_false=0.05,
            active_states=[STATE_ON],
        )
        decay = Decay(half_life=60.0)  # half_life is required
        entity = create_test_entity(
            entity_type=entity_type,
            decay=decay,
            coordinator=coordinator_with_areas,
        )

        assert entity.entity_id == "test"
        assert entity.type == entity_type
        assert entity.prob_given_true == 0.25
        assert entity.prob_given_false == 0.05
        assert entity.decay == decay
        assert entity.hass == coordinator_with_areas.hass

    @patch("custom_components.area_occupancy.data.entity.Decay")
    @patch("custom_components.area_occupancy.data.entity.EntityType")
    def test_from_dict(
        self,
        mock_entity_type_class: Mock,
        mock_decay_class: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
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
            coordinator=coordinator_with_areas,
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
        assert entity.hass == coordinator_with_areas.hass


class TestEntityManager:
    """Test the EntityManager class."""

    def test_initialization(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test manager initialization."""
        area_name = coordinator_with_areas.get_area_names()[0]
        manager = create_test_entity_manager(coordinator_with_areas, area_name)

        assert manager.coordinator == coordinator_with_areas
        assert manager._entities == {}

    def test_entities_property(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test entities property."""
        area_name = coordinator_with_areas.get_area_names()[0]
        manager = create_test_entity_manager(coordinator_with_areas, area_name)
        test_entity = Mock()
        manager._entities = {"test": test_entity}

        assert manager.entities == {"test": test_entity}

    def test_entity_ids_property(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test entity_ids property."""
        area_name = coordinator_with_areas.get_area_names()[0]
        manager = create_test_entity_manager(coordinator_with_areas, area_name)
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
        self,
        coordinator_with_areas: AreaOccupancyCoordinator,
        property_name: str,
        expected_entities: list[str],
    ) -> None:
        """Test entity filtering properties (active, inactive, decaying)."""
        area_name = coordinator_with_areas.get_area_names()[0]
        manager = create_test_entity_manager(coordinator_with_areas, area_name)
        mock_entities = create_mock_entities_with_states()
        manager._entities = mock_entities

        filtered_entities = getattr(manager, property_name)
        assert len(filtered_entities) == len(expected_entities)
        for entity_name in expected_entities:
            assert mock_entities[entity_name] in filtered_entities

    def test_get_entity(self, coordinator_with_areas: AreaOccupancyCoordinator) -> None:
        """Test getting entity by ID."""
        area_name = coordinator_with_areas.get_area_names()[0]
        manager = create_test_entity_manager(coordinator_with_areas, area_name)
        mock_entity = Mock()
        manager._entities = {"test": mock_entity}

        result = manager.get_entity("test")
        assert result == mock_entity

        with pytest.raises(
            ValueError, match="Entity not found for entity: nonexistent"
        ):
            manager.get_entity("nonexistent")

    def test_add_entity(self, coordinator_with_areas: AreaOccupancyCoordinator) -> None:
        """Test adding entity to manager."""
        area_name = coordinator_with_areas.get_area_names()[0]
        manager = create_test_entity_manager(coordinator_with_areas, area_name)
        mock_entity = Mock()
        mock_entity.entity_id = "test"

        manager.add_entity(mock_entity)

        assert manager._entities["test"] == mock_entity


class TestEntityPropertiesAndMethods:
    """Test entity properties and methods."""

    @pytest.fixture
    def test_entity(self, coordinator_with_areas: AreaOccupancyCoordinator) -> Entity:
        """Create a test entity for property testing."""
        return create_test_entity(coordinator=coordinator_with_areas)

    def test_state_property_edge_cases(
        self, test_entity: Entity, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test state property with edge cases."""
        # Test with no state available
        original_states = coordinator_with_areas.hass.states
        _set_states_get(coordinator_with_areas.hass, lambda _: None)
        try:
            assert test_entity.state is None
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

        # Test with state but no state attribute
        mock_state = Mock()
        mock_state.state = "test_state"
        _set_states_get(coordinator_with_areas.hass, lambda _: mock_state)
        try:
            assert test_entity.state == "test_state"
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

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
        coordinator_with_areas: AreaOccupancyCoordinator,
        state_value: str | None,
        expected_available: bool,
    ) -> None:
        """Test available property with different states."""
        original_states = coordinator_with_areas.hass.states
        try:
            if state_value is None:
                _set_states_get(coordinator_with_areas.hass, lambda _: None)
            else:
                mock_state = Mock()
                mock_state.state = state_value
                _set_states_get(coordinator_with_areas.hass, lambda _: mock_state)
            assert test_entity.available is expected_available
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

    @pytest.mark.parametrize(
        ("state_value", "expected_evidence"),
        [
            ("15", True),  # Within range (10, 20)
            ("25", False),  # Outside range
            ("invalid", False),  # Invalid value
        ],
    )
    def test_evidence_with_active_range(
        self,
        coordinator_with_areas: AreaOccupancyCoordinator,
        state_value: str,
        expected_evidence: bool,
    ) -> None:
        """Test evidence property with active range."""
        mock_entity_type = Mock()
        mock_entity_type.active_range = (10, 20)
        mock_entity_type.active_states = None  # Ensure this is None for range test

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator_with_areas,
        )

        original_states = coordinator_with_areas.hass.states
        mock_state = Mock()
        mock_state.state = state_value
        _set_states_get(coordinator_with_areas.hass, lambda _: mock_state)
        try:
            assert entity.evidence is expected_evidence
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

    def test_has_new_evidence_transitions(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test has_new_evidence method with evidence transitions."""
        # Create a proper mock entity type with active_states
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator_with_areas,
        )

        original_states = coordinator_with_areas.hass.states
        # Test initial state (no transition)
        mock_state = Mock()
        mock_state.state = STATE_ON
        _set_states_get(coordinator_with_areas.hass, lambda _: mock_state)
        try:
            assert not entity.has_new_evidence()  # No transition on first call

            # Test transition from True to False
            mock_state.state = "off"
            assert entity.has_new_evidence()  # Should detect transition

            # Test transition from False to True
            mock_state.state = STATE_ON
            assert entity.has_new_evidence()  # Should detect transition
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

    def test_entity_properties_comprehensive(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test comprehensive entity properties including name, weight, active, decay_factor."""
        # Create entity with proper type - only use active_states, not both
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.85,
            prob_given_true=0.25,
            prob_given_false=0.05,
            active_states=[STATE_ON],
            active_range=None,  # Don't provide both active_states and active_range
        )

        # Create decay with specific behavior - set up for decay_factor < 1.0
        decay = Decay(half_life=60.0)  # half_life is required
        decay.is_decaying = True
        # Use timezone-aware datetime to match dt_util.utcnow() in Decay class
        decay.decay_start = dt_util.utcnow() - timedelta(seconds=60)  # 1 minute ago
        decay.half_life = 30.0  # 30 second half-life

        entity = create_test_entity(
            entity_type=entity_type,
            decay=decay,
            coordinator=coordinator_with_areas,
        )

        original_states = coordinator_with_areas.hass.states
        # Test name property
        mock_state = Mock()
        mock_state.name = "Test Motion Sensor"
        _set_states_get(coordinator_with_areas.hass, lambda _: mock_state)
        try:
            assert entity.name == "Test Motion Sensor"

            # Test weight property
            assert entity.weight == 0.85

            # Test active_states and active_range properties
            assert entity.active_states == [STATE_ON]
            assert entity.active_range is None

            # Test active property when evidence is True
            mock_state.state = STATE_ON
            assert entity.active is True

            # Test decay_factor when evidence is True (should return 1.0)
            assert entity.decay_factor == 1.0

            # Test decay_factor when evidence is False (should return decay.decay_factor)
            mock_state.state = "off"
            # decay_factor should be < 1.0 since decay is running and started 1 minute ago
            assert entity.decay_factor < 1.0
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

    def test_entity_methods_update_likelihood_and_decay(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test update_likelihood and update_decay methods."""
        entity = create_test_entity(coordinator=coordinator_with_areas)
        original_updated = entity.last_updated

        # Test update_likelihood
        entity.update_likelihood(0.9, 0.1)
        assert entity.prob_given_true == 0.9
        assert entity.prob_given_false == 0.1
        assert entity.last_updated > original_updated

        # Test update_decay
        decay_start = dt_util.utcnow()
        entity.update_decay(decay_start, True)
        assert entity.decay.decay_start == decay_start
        assert entity.decay.is_decaying is True

    def test_has_new_evidence_edge_cases(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test has_new_evidence method with edge cases and decay interactions."""
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator_with_areas,
            previous_evidence=None,  # Start with None
        )

        original_states = coordinator_with_areas.hass.states
        # Test with current evidence None (entity unavailable)
        _set_states_get(coordinator_with_areas.hass, lambda _: None)
        try:
            assert not entity.has_new_evidence()
            assert entity.previous_evidence is None
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

        # Test with previous evidence None but current evidence available
        mock_state = Mock()
        mock_state.state = STATE_ON
        _set_states_get(coordinator_with_areas.hass, lambda _: mock_state)
        try:
            assert not entity.has_new_evidence()  # No transition when previous is None
            assert entity.previous_evidence is True
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

        # Test decay interaction when evidence becomes True
        entity.decay.is_decaying = True
        entity.decay.stop_decay = Mock()

        # Set state to off to establish previous_evidence as False
        mock_state_off = Mock()
        mock_state_off.state = "off"
        _set_states_get(coordinator_with_areas.hass, lambda _: mock_state_off)
        try:
            entity.has_new_evidence()  # This sets previous_evidence to False
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

        # Reset the mock to count only the next call
        entity.decay.stop_decay.reset_mock()

        # Now change to on - this should trigger stop_decay
        mock_state_on = Mock()
        mock_state_on.state = STATE_ON
        _set_states_get(coordinator_with_areas.hass, lambda _: mock_state_on)
        try:
            assert entity.has_new_evidence()  # Should detect transition and stop decay
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)
        # stop_decay is called twice: once for inconsistent state, once for transition
        assert entity.decay.stop_decay.call_count == 2

    def test_entity_factory_create_from_db(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test EntityFactory.create_from_db method."""
        with (
            patch(
                "custom_components.area_occupancy.data.entity.EntityType"
            ) as mock_entity_type_class,
            patch(
                "custom_components.area_occupancy.data.entity.Decay"
            ) as mock_decay_class,
        ):
            # Mock the create methods
            mock_entity_type = Mock()
            # EntityType is instantiated directly in create_from_db, not via create()
            mock_entity_type_class.return_value = mock_entity_type

            mock_decay = Mock()
            mock_decay_class.return_value = (
                mock_decay  # Decay uses __init__, not create()
            )

            # Mock database entity object
            mock_db_entity = Mock()
            mock_db_entity.entity_id = "binary_sensor.test"
            mock_db_entity.entity_type = "motion"
            mock_db_entity.prob_given_true = 0.8
            mock_db_entity.prob_given_false = 0.1
            mock_db_entity.decay_start = dt_util.utcnow()
            mock_db_entity.is_decaying = False
            mock_db_entity.last_updated = dt_util.utcnow()
            mock_db_entity.evidence = True

            # Add to_dict method to return proper dictionary
            def mock_to_dict():
                return {
                    "entity_id": "binary_sensor.test",
                    "entity_type": "motion",
                    "prob_given_true": 0.8,
                    "prob_given_false": 0.1,
                    "decay_start": mock_db_entity.decay_start,
                    "is_decaying": False,
                    "last_updated": mock_db_entity.last_updated,
                    "evidence": True,
                    "weight": 0.5,  # Add default weight
                }

            mock_db_entity.to_dict = mock_to_dict

            # Mock config - get area for config access
            area_name = coordinator_with_areas.get_area_names()[0]
            area = coordinator_with_areas.get_area(area_name)
            area.config.decay.half_life = 300.0

            # Set configured motion sensor likelihoods (motion sensors use configured values, not database values)
            # Default values are 0.95 and 0.02, but we'll set explicit test values
            area.config.sensors.motion_prob_given_true = 0.95
            area.config.sensors.motion_prob_given_false = 0.02

            # EntityFactory requires area_name
            factory = EntityFactory(coordinator_with_areas, area_name=area_name)
            entity = factory.create_from_db(mock_db_entity)

            # Verify entity creation
            # Motion sensors use configured likelihoods, not database values
            assert entity.entity_id == "binary_sensor.test"
            assert entity.type == mock_entity_type
            assert entity.decay == mock_decay
            assert (
                entity.prob_given_true == 0.95
            )  # Uses configured value, not database value (0.8)
            assert (
                entity.prob_given_false == 0.02
            )  # Uses configured value, not database value (0.1)
            assert entity.previous_evidence is True

            # Verify factory calls - EntityType is instantiated directly
            # The actual call uses EntityType(input_type_enum, ...) not EntityType.create()
            mock_entity_type_class.assert_called()
            mock_decay_class.assert_called_once()
            # Decay is now created with __init__, not create()
            decay_call_args = mock_decay_class.call_args
            assert decay_call_args.kwargs.get("half_life") == 300.0

    def test_entity_manager_get_entities_by_input_type(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test EntityManager.get_entities_by_input_type method."""
        # Create manager with mocked factory
        with patch(
            "custom_components.area_occupancy.data.entity.EntityFactory"
        ) as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_all_from_config.return_value = {}
            mock_factory_class.return_value = mock_factory

            # EntityManager requires area_name
            area_name = coordinator_with_areas.get_area_names()[0]
            manager = EntityManager(coordinator_with_areas, area_name=area_name)

            # Create test entities with different input types
            motion_entity = create_test_entity(
                "motion_1", coordinator=coordinator_with_areas
            )
            motion_entity.type.input_type = InputType.MOTION

            media_entity = create_test_entity(
                "media_1", coordinator=coordinator_with_areas
            )
            media_entity.type.input_type = InputType.MEDIA

            door_entity = create_test_entity(
                "door_1", coordinator=coordinator_with_areas
            )
            door_entity.type.input_type = InputType.DOOR

            manager._entities = {
                "motion_1": motion_entity,
                "media_1": media_entity,
                "door_1": door_entity,
            }

            # Test filtering by motion type
            motion_entities = manager.get_entities_by_input_type(InputType.MOTION)
            assert len(motion_entities) == 1
            assert "motion_1" in motion_entities
            assert motion_entities["motion_1"] == motion_entity

            # Test filtering by media type
            media_entities = manager.get_entities_by_input_type(InputType.MEDIA)
            assert len(media_entities) == 1
            assert "media_1" in media_entities
            assert media_entities["media_1"] == media_entity

            # Test filtering by non-existent type
            empty_entities = manager.get_entities_by_input_type(InputType.APPLIANCE)
            assert len(empty_entities) == 0

    def test_evidence_property_edge_case_no_active_config(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test evidence property when neither active_states nor active_range is configured."""
        # Create entity type with no active configuration
        mock_entity_type = Mock()
        mock_entity_type.active_states = None
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator_with_areas,
        )

        # Set up a valid state
        original_states = coordinator_with_areas.hass.states
        mock_state = Mock()
        mock_state.state = "some_state"
        _set_states_get(coordinator_with_areas.hass, lambda _: mock_state)
        try:
            # Should return None when neither active_states nor active_range is configured
            assert entity.evidence is None
        finally:
            object.__setattr__(coordinator_with_areas.hass, "states", original_states)

    @pytest.mark.asyncio
    async def test_entity_manager_cleanup(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test EntityManager cleanup method."""
        # Create manager with mocked factory
        with patch(
            "custom_components.area_occupancy.data.entity.EntityFactory"
        ) as mock_factory_class:
            mock_factory = Mock()
            new_entity = Mock()
            mock_factory.create_all_from_config.return_value = {
                "new_entity": new_entity
            }
            mock_factory_class.return_value = mock_factory

            # Create manager and verify it uses the factory
            # EntityManager requires area_name
            area_name = coordinator_with_areas.get_area_names()[0]
            manager = EntityManager(coordinator_with_areas, area_name=area_name)

            # Test cleanup method - this should call the factory again
            await manager.cleanup()
            # The cleanup method calls create_all_from_config again
            assert mock_factory.create_all_from_config.call_count >= 2


class TestEntityFactory:
    """Test the EntityFactory class."""

    def test_initialization(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test factory initialization."""
        # EntityFactory requires area_name
        area_name = coordinator_with_areas.get_area_names()[0]
        factory = EntityFactory(coordinator_with_areas, area_name=area_name)
        assert factory.coordinator == coordinator_with_areas

    @patch("custom_components.area_occupancy.data.entity.EntityType")
    @patch("custom_components.area_occupancy.data.entity.Decay")
    def test_create_from_config_spec(
        self,
        mock_decay_class: Mock,
        mock_entity_type_class: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test creating entity from config spec."""
        mock_entity_type = Mock()
        # EntityType is instantiated directly, not via create()
        mock_entity_type_class.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.return_value = mock_decay  # Decay uses __init__, not create()

        # EntityFactory requires area_name
        area_name = coordinator_with_areas.get_area_names()[0]
        factory = EntityFactory(coordinator_with_areas, area_name=area_name)
        entity = factory.create_from_config_spec("test_entity", "motion")

        assert entity.entity_id == "test_entity"
        assert entity.type == mock_entity_type
        assert entity.decay == mock_decay
        assert entity.hass == coordinator_with_areas.hass

    @patch("custom_components.area_occupancy.data.entity.EntityType")
    @patch("custom_components.area_occupancy.data.entity.Decay")
    def test_create_all_from_config(
        self,
        mock_decay_class: Mock,
        mock_entity_type_class: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test creating all entities from config."""
        mock_entity_type = Mock()
        # EntityType is instantiated directly, not via create()
        mock_entity_type_class.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.return_value = mock_decay  # Decay uses __init__, not create()

        # Mock config with sensors - use area config if available
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area(area_name)
        area.config.sensors.motion = ["binary_sensor.motion1"]
        area.config.sensors.media = ["media_player.tv"]
        # EntityFactory requires area_name
        factory = EntityFactory(coordinator_with_areas, area_name=area_name)
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
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test creating entity without stored data."""
        mock_entity_type = Mock()
        mock_entity_type.prob_given_true = DEFAULT_TYPES[InputType.MOTION][
            "prob_given_true"
        ]
        mock_entity_type.prob_given_false = DEFAULT_TYPES[InputType.MOTION][
            "prob_given_false"
        ]
        # EntityType is instantiated directly, not via create()
        mock_entity_type_class.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.return_value = mock_decay  # Decay uses __init__, not create()

        # EntityFactory requires area_name
        area_name = coordinator_with_areas.get_area_names()[0]
        factory = EntityFactory(coordinator_with_areas, area_name=area_name)
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
