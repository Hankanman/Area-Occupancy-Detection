"""Tests for data.entity module."""

from datetime import timedelta
import math
from unittest.mock import MagicMock, Mock, patch

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import (
    Entity,
    EntityFactory,
    EntityManager,
)
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.data.types import GaussianParams
from homeassistant.const import STATE_OFF, STATE_ON
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

    # Set default analysis_error based on entity type
    if "analysis_error" not in kwargs:
        if entity_type.input_type == InputType.MOTION:
            kwargs["analysis_error"] = "motion_sensor_excluded"
        else:
            kwargs["analysis_error"] = "not_analyzed"

    return Entity(
        entity_id=entity_id,
        type=entity_type,
        prob_given_true=prob_given_true,
        prob_given_false=prob_given_false,
        decay=decay,
        hass=hass,
        last_updated=dt_util.utcnow(),
        previous_evidence=kwargs.get("previous_evidence"),
        analysis_error=kwargs.get("analysis_error"),
    )


def create_test_entity_manager(
    coordinator: AreaOccupancyCoordinator | None = None, area_name: str | None = None
) -> EntityManager:
    """Create a test EntityManager instance with real coordinator.

    Args:
        coordinator: Real coordinator instance (will use coordinator if None)
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

    def test_initialization(self, coordinator: AreaOccupancyCoordinator) -> None:
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
            coordinator=coordinator,
        )

        assert entity.entity_id == "test"
        assert entity.type == entity_type
        assert entity.prob_given_true == 0.25
        assert entity.prob_given_false == 0.05
        assert entity.decay == decay
        assert entity.hass == coordinator.hass

    def test_initialization_both_hass_and_state_provider_raises_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that providing both hass and state_provider raises ValueError."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=0.25,
            prob_given_false=0.05,
            active_states=[STATE_ON],
        )
        decay = Decay(half_life=60.0)

        def state_provider(_):
            return None

        with pytest.raises(
            ValueError, match="Cannot provide both hass and state_provider"
        ):
            Entity(
                entity_id="test",
                type=entity_type,
                prob_given_true=0.25,
                prob_given_false=0.05,
                decay=decay,
                hass=coordinator.hass,
                state_provider=state_provider,
            )

    def test_initialization_neither_hass_nor_state_provider_raises_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that providing neither hass nor state_provider raises ValueError."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=0.25,
            prob_given_false=0.05,
            active_states=[STATE_ON],
        )
        decay = Decay(half_life=60.0)

        with pytest.raises(
            ValueError, match="Either hass or state_provider must be provided"
        ):
            Entity(
                entity_id="test",
                type=entity_type,
                prob_given_true=0.25,
                prob_given_false=0.05,
                decay=decay,
                hass=None,
                state_provider=None,
            )


class TestEntityManager:
    """Test the EntityManager class."""

    def test_initialization(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test manager initialization."""
        area_name = coordinator.get_area_names()[0]
        manager = create_test_entity_manager(coordinator, area_name)

        assert manager.coordinator == coordinator
        assert manager._entities == {}

    @pytest.mark.parametrize(
        ("area_name", "error_message"),
        [
            (None, "Area name is required in multi-area architecture"),
            ("nonexistent", "Area 'nonexistent' not found"),
        ],
    )
    def test_initialization_invalid_area_name(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str | None,
        error_message: str,
    ) -> None:
        """Test EntityManager initialization with invalid area_name raises ValueError."""
        with pytest.raises(ValueError, match=error_message):
            EntityManager(coordinator, area_name=area_name)

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
        coordinator: AreaOccupancyCoordinator,
        property_name: str,
        expected_entities: list[str],
    ) -> None:
        """Test entity filtering properties (active, inactive, decaying)."""
        area_name = coordinator.get_area_names()[0]
        manager = create_test_entity_manager(coordinator, area_name)
        mock_entities = create_mock_entities_with_states()
        manager._entities = mock_entities

        filtered_entities = getattr(manager, property_name)
        assert len(filtered_entities) == len(expected_entities)
        for entity_name in expected_entities:
            assert mock_entities[entity_name] in filtered_entities

    def test_get_entity(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test getting entity by ID."""
        area_name = coordinator.get_area_names()[0]
        manager = create_test_entity_manager(coordinator, area_name)
        mock_entity = Mock()
        manager._entities = {"test": mock_entity}

        result = manager.get_entity("test")
        assert result == mock_entity

        with pytest.raises(
            ValueError, match="Entity not found for entity: nonexistent"
        ):
            manager.get_entity("nonexistent")

    def test_add_entity(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test adding entity to manager."""
        area_name = coordinator.get_area_names()[0]
        manager = create_test_entity_manager(coordinator, area_name)
        mock_entity = Mock()
        mock_entity.entity_id = "test"

        manager.add_entity(mock_entity)

        assert manager._entities["test"] == mock_entity


class TestEntityPropertiesAndMethods:
    """Test entity properties and methods."""

    @pytest.fixture
    def test_entity(self, coordinator: AreaOccupancyCoordinator) -> Entity:
        """Create a test entity for property testing."""
        return create_test_entity(coordinator=coordinator)

    def test_state_property_edge_cases(
        self, test_entity: Entity, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test state property with edge cases."""
        # Test with no state available
        original_states = coordinator.hass.states
        _set_states_get(coordinator.hass, lambda _: None)
        try:
            assert test_entity.state is None
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

        # Test with state but no state attribute
        mock_state = Mock()
        mock_state.state = "test_state"
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            assert test_entity.state == "test_state"
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_state_property_with_state_provider(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test state property with state_provider instead of hass."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=0.25,
            prob_given_false=0.05,
            active_states=[STATE_ON],
        )
        decay = Decay(half_life=60.0)

        # Create state provider that returns object with .state attribute
        mock_state_obj = Mock()
        mock_state_obj.state = "on"

        def state_provider(entity_id):
            if entity_id == "test":
                return mock_state_obj
            return None

        entity = Entity(
            entity_id="test",
            type=entity_type,
            prob_given_true=0.25,
            prob_given_false=0.05,
            decay=decay,
            hass=None,
            state_provider=state_provider,
        )

        assert entity.state == "on"

        # Test with state_provider returning direct value
        def state_provider_direct(entity_id):
            if entity_id == "test":
                return "off"
            return None

        entity.state_provider = state_provider_direct
        assert entity.state == "off"

        # Test with state_provider returning None
        def state_provider_none(entity_id):
            return None

        entity.state_provider = state_provider_none
        assert entity.state is None

    def test_name_property_with_state_provider(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test name property with state_provider."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=0.25,
            prob_given_false=0.05,
            active_states=[STATE_ON],
        )
        decay = Decay(half_life=60.0)

        # Create state provider that returns object with .name attribute
        mock_state_obj = Mock()
        mock_state_obj.name = "Test Sensor"

        def state_provider(entity_id):
            if entity_id == "test":
                return mock_state_obj
            return None

        entity = Entity(
            entity_id="test",
            type=entity_type,
            prob_given_true=0.25,
            prob_given_false=0.05,
            decay=decay,
            hass=None,
            state_provider=state_provider,
        )

        assert entity.name == "Test Sensor"

        # Test with state_provider returning object without name
        mock_state_obj_no_name = Mock()
        del mock_state_obj_no_name.name

        def state_provider_no_name(entity_id):
            if entity_id == "test":
                return mock_state_obj_no_name
            return None

        entity.state_provider = state_provider_no_name
        assert entity.name is None

        # Test with state_provider returning None
        def state_provider_none(entity_id):
            return None

        entity.state_provider = state_provider_none
        assert entity.name is None

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
        coordinator: AreaOccupancyCoordinator,
        state_value: str | None,
        expected_available: bool,
    ) -> None:
        """Test available property with different states."""
        original_states = coordinator.hass.states
        try:
            if state_value is None:
                _set_states_get(coordinator.hass, lambda _: None)
            else:
                mock_state = Mock()
                mock_state.state = state_value
                _set_states_get(coordinator.hass, lambda _: mock_state)
            assert test_entity.available is expected_available
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

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
        coordinator: AreaOccupancyCoordinator,
        state_value: str,
        expected_evidence: bool,
    ) -> None:
        """Test evidence property with active range."""
        mock_entity_type = Mock()
        mock_entity_type.active_range = (10, 20)
        mock_entity_type.active_states = None  # Ensure this is None for range test

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator,
        )

        original_states = coordinator.hass.states
        mock_state = Mock()
        mock_state.state = state_value
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            assert entity.evidence is expected_evidence
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_has_new_evidence_transitions(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test has_new_evidence method with evidence transitions."""
        # Create a proper mock entity type with active_states
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator,
        )

        original_states = coordinator.hass.states
        # Test initial state (no transition)
        mock_state = Mock()
        mock_state.state = STATE_ON
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            assert not entity.has_new_evidence()  # No transition on first call

            # Test transition from True to False
            mock_state.state = "off"
            assert entity.has_new_evidence()  # Should detect transition

            # Test transition from False to True
            mock_state.state = STATE_ON
            assert entity.has_new_evidence()  # Should detect transition
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_entity_properties_comprehensive(
        self, coordinator: AreaOccupancyCoordinator
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
        decay = Decay(half_life=30.0)  # 30 second half-life
        decay.is_decaying = True
        # Use timezone-aware datetime to match dt_util.utcnow() in Decay class
        decay.decay_start = dt_util.utcnow() - timedelta(seconds=60)  # 1 minute ago

        entity = create_test_entity(
            entity_type=entity_type,
            decay=decay,
            coordinator=coordinator,
        )

        original_states = coordinator.hass.states
        # Test name property
        mock_state = Mock()
        mock_state.name = "Test Motion Sensor"
        _set_states_get(coordinator.hass, lambda _: mock_state)
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
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_information_gain_highly_informative(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test information_gain for highly informative sensors (large pgt/pgf gap)."""
        # Motion sensor: prob_given_true=0.95, prob_given_false=0.02
        entity = create_test_entity(
            prob_given_true=0.95,
            prob_given_false=0.02,
            coordinator=coordinator,
        )
        gain = entity.information_gain
        assert gain > 0.9
        assert gain <= 1.0

    def test_information_gain_uninformative(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test information_gain for uninformative sensors (pgt ≈ pgf)."""
        entity_type = EntityType(
            input_type=InputType.DOOR,
            weight=0.3,
            prob_given_true=0.50,
            prob_given_false=0.50,
            active_states=["closed"],
        )
        entity = create_test_entity(
            entity_type=entity_type,
            prob_given_true=0.50,
            prob_given_false=0.50,
            coordinator=coordinator,
            analysis_error=None,
        )
        gain = entity.information_gain
        assert gain == 0.0

    def test_information_gain_with_analysis_error(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test information_gain is clamped to 0.1 for sensors with analysis errors."""
        entity = create_test_entity(
            prob_given_true=0.95,
            prob_given_false=0.02,
            coordinator=coordinator,
            analysis_error="no_correlation",
        )
        assert entity.information_gain == 0.1

    def test_information_gain_not_analyzed_uses_likelihoods(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test information_gain for not_analyzed and motion_excluded uses likelihoods."""
        # NOT_ANALYZED should not trigger the low-gain clamp
        entity = create_test_entity(
            prob_given_true=0.95,
            prob_given_false=0.02,
            coordinator=coordinator,
            analysis_error="not_analyzed",
        )
        assert entity.information_gain > 0.9

        # MOTION_EXCLUDED should not trigger the low-gain clamp
        entity2 = create_test_entity(
            prob_given_true=0.95,
            prob_given_false=0.02,
            coordinator=coordinator,
            analysis_error="motion_sensor_excluded",
        )
        assert entity2.information_gain > 0.9

    def test_effective_weight(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test effective_weight = weight × information_gain."""
        # Highly informative sensor
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.85,
            prob_given_true=0.95,
            prob_given_false=0.02,
            active_states=[STATE_ON],
        )
        entity = create_test_entity(
            entity_type=entity_type,
            prob_given_true=0.95,
            prob_given_false=0.02,
            coordinator=coordinator,
        )
        assert entity.effective_weight == pytest.approx(
            entity.weight * entity.information_gain, abs=1e-6
        )
        assert entity.effective_weight > 0.8  # Nearly full weight

    def test_effective_weight_uninformative_is_zero(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test effective_weight is zero for uninformative sensors."""
        entity_type = EntityType(
            input_type=InputType.DOOR,
            weight=0.3,
            prob_given_true=0.50,
            prob_given_false=0.50,
            active_states=["closed"],
        )
        entity = create_test_entity(
            entity_type=entity_type,
            prob_given_true=0.50,
            prob_given_false=0.50,
            coordinator=coordinator,
            analysis_error=None,
        )
        assert entity.effective_weight == 0.0

    def test_entity_methods_update_decay(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_decay method."""
        entity = create_test_entity(coordinator=coordinator)

        # Test update_decay
        decay_start = dt_util.utcnow()
        entity.update_decay(decay_start, True)
        assert entity.decay.decay_start == decay_start
        assert entity.decay.is_decaying is True

    def test_has_new_evidence_edge_cases(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test has_new_evidence method with edge cases and decay interactions."""
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator,
            previous_evidence=None,  # Start with None
        )

        original_states = coordinator.hass.states
        # Test with current evidence None (entity unavailable)
        _set_states_get(coordinator.hass, lambda _: None)
        try:
            assert not entity.has_new_evidence()
            assert entity.previous_evidence is None
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

        # Test with previous evidence None but current evidence available
        mock_state = Mock()
        mock_state.state = STATE_ON
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            assert not entity.has_new_evidence()  # No transition when previous is None
            assert entity.previous_evidence is True
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

        # Test decay interaction when evidence becomes True
        entity.decay.is_decaying = True
        entity.decay.stop_decay = Mock()

        # Set state to off to establish previous_evidence as False
        mock_state_off = Mock()
        mock_state_off.state = "off"
        _set_states_get(coordinator.hass, lambda _: mock_state_off)
        try:
            entity.has_new_evidence()  # This sets previous_evidence to False
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

        # Reset the mock to count only the next call
        entity.decay.stop_decay.reset_mock()

        # Now change to on - this should trigger stop_decay
        mock_state_on = Mock()
        mock_state_on.state = STATE_ON
        _set_states_get(coordinator.hass, lambda _: mock_state_on)
        try:
            assert entity.has_new_evidence()  # Should detect transition and stop decay
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)
        # stop_decay is called twice: once for inconsistent state, once for transition
        assert entity.decay.stop_decay.call_count == 2

    def test_has_new_evidence_unavailable_transitions(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test has_new_evidence handles entity becoming unavailable correctly.

        When an entity with positive evidence becomes unavailable, decay should start.
        """
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator,
            previous_evidence=True,  # Entity previously had evidence
        )

        # Mock decay methods
        entity.decay.start_decay = Mock()
        entity.decay.stop_decay = Mock()
        entity.decay.is_decaying = False

        original_states = coordinator.hass.states

        # Test: Entity becomes unavailable (None state)
        _set_states_get(coordinator.hass, lambda _: None)
        try:
            result = entity.has_new_evidence()

            # Should return False (no "new evidence" to trigger refresh)
            assert result is False

            # But decay should have started since we lost positive evidence
            entity.decay.start_decay.assert_called_once()

            # previous_evidence should be updated to None
            assert entity.previous_evidence is None
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_has_new_evidence_unavailable_from_false(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test has_new_evidence when entity with no evidence becomes unavailable.

        When an entity without evidence becomes unavailable, no decay change needed.
        """
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator,
            previous_evidence=False,  # Entity previously had no evidence
        )

        # Mock decay methods
        entity.decay.start_decay = Mock()
        entity.decay.stop_decay = Mock()
        entity.decay.is_decaying = False

        original_states = coordinator.hass.states

        # Test: Entity becomes unavailable (None state)
        _set_states_get(coordinator.hass, lambda _: None)
        try:
            result = entity.has_new_evidence()

            # Should return False
            assert result is False

            # Decay should NOT have started (no evidence to lose)
            entity.decay.start_decay.assert_not_called()

            # previous_evidence should be updated to None
            assert entity.previous_evidence is None
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_has_new_evidence_available_with_evidence(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test has_new_evidence when unavailable entity becomes available with evidence.

        When an entity becomes available with positive evidence, decay should stop.
        """
        mock_entity_type = Mock()
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator,
            previous_evidence=None,  # Entity was unavailable
        )

        # Mock decay methods
        entity.decay.start_decay = Mock()
        entity.decay.stop_decay = Mock()
        entity.decay.is_decaying = True  # Decay was running

        original_states = coordinator.hass.states

        # Test: Entity becomes available with evidence
        mock_state = Mock()
        mock_state.state = STATE_ON
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            result = entity.has_new_evidence()

            # Should return False (entity just became available, not a transition)
            assert result is False

            # Decay should be stopped since entity has positive evidence
            entity.decay.stop_decay.assert_called_once()

            # previous_evidence should be updated to True
            assert entity.previous_evidence is True
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_entity_factory_create_from_db(
        self, coordinator: AreaOccupancyCoordinator
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
            area_name = coordinator.get_area_names()[0]
            area = coordinator.get_area(area_name)
            area.config.decay.half_life = 300.0

            # Set configured motion sensor likelihoods (motion sensors use configured values, not database values)
            # Use non-default values (0.88/0.03) to prove configured values override both DB values (0.8/0.1)
            # and static defaults (0.95/0.02)
            area.config.sensors.motion_prob_given_true = 0.88
            area.config.sensors.motion_prob_given_false = 0.03

            # EntityFactory requires area_name
            factory = EntityFactory(coordinator, area_name=area_name)
            entity = factory.create_from_db(mock_db_entity)

            # Verify entity creation
            # Motion sensors use configured likelihoods, not database values or defaults
            assert entity.entity_id == "binary_sensor.test"
            assert entity.type == mock_entity_type
            assert entity.decay == mock_decay
            assert (
                entity.prob_given_true == 0.88
            )  # Uses configured value (0.88), not database value (0.8) or default (0.95)
            assert (
                entity.prob_given_false == 0.03
            )  # Uses configured value (0.03), not database value (0.1) or default (0.02)
            assert entity.previous_evidence is True

            # Verify factory calls - EntityType is instantiated directly
            # The actual call uses EntityType(input_type_enum, ...) not EntityType.create()
            mock_entity_type_class.assert_called()
            mock_decay_class.assert_called_once()
            # Decay is now created with __init__, not create()
            decay_call_args = mock_decay_class.call_args
            assert decay_call_args.kwargs.get("half_life") == 300.0

    @pytest.mark.parametrize(
        ("field_name", "invalid_value", "exception_type", "error_message"),
        [
            (
                "entity_id",
                "",
                ValueError,
                "Entity ID cannot be empty",
            ),
            (
                "entity_type",
                "",
                ValueError,
                "Entity type cannot be empty",
            ),
            (
                "prob_given_true",
                "invalid",
                TypeError,
                "Failed to convert numeric fields",
            ),
        ],
    )
    def test_create_from_db_validation_errors(
        self,
        coordinator: AreaOccupancyCoordinator,
        field_name: str,
        invalid_value: str,
        exception_type: type[Exception],
        error_message: str,
    ) -> None:
        """Test create_from_db validation errors."""
        area_name = coordinator.get_area_names()[0]
        factory = EntityFactory(coordinator, area_name=area_name)

        mock_db_entity = Mock()

        # Build base dict with valid values
        base_dict = {
            "entity_id": "test",
            "entity_type": "motion",
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "weight": 0.5,
        }
        # Override the invalid field
        base_dict[field_name] = invalid_value

        def mock_to_dict():
            return base_dict

        mock_db_entity.to_dict = mock_to_dict

        with pytest.raises(exception_type, match=error_message):
            factory.create_from_db(mock_db_entity)

    @pytest.mark.parametrize(
        "creation_method", ["create_from_db", "create_from_config_spec"]
    )
    def test_wasp_entity_sets_half_life(
        self,
        coordinator: AreaOccupancyCoordinator,
        creation_method: str,
    ) -> None:
        """Test WASP entity sets half_life to 0.1 regardless of creation method."""
        with (
            patch(
                "custom_components.area_occupancy.data.entity.EntityType"
            ) as mock_entity_type_class,
            patch(
                "custom_components.area_occupancy.data.entity.Decay"
            ) as mock_decay_class,
        ):
            mock_entity_type = Mock()
            mock_entity_type_class.return_value = mock_entity_type
            mock_decay = Mock()
            mock_decay_class.return_value = mock_decay

            area_name = coordinator.get_area_names()[0]
            area = coordinator.get_area(area_name)
            # Set WASP entity ID
            area.wasp_entity_id = "binary_sensor.wasp"

            factory = EntityFactory(coordinator, area_name=area_name)
            area.config.decay.half_life = 300.0

            if creation_method == "create_from_db":
                mock_db_entity = Mock()
                mock_db_entity.entity_id = "binary_sensor.wasp"
                mock_db_entity.entity_type = "motion"
                mock_db_entity.prob_given_true = 0.8
                mock_db_entity.prob_given_false = 0.1
                mock_db_entity.decay_start = dt_util.utcnow()
                mock_db_entity.is_decaying = False
                mock_db_entity.last_updated = dt_util.utcnow()
                mock_db_entity.evidence = True

                def mock_to_dict():
                    return {
                        "entity_id": "binary_sensor.wasp",
                        "entity_type": "motion",
                        "prob_given_true": 0.8,
                        "prob_given_false": 0.1,
                        "decay_start": mock_db_entity.decay_start,
                        "is_decaying": False,
                        "last_updated": mock_db_entity.last_updated,
                        "evidence": True,
                        "weight": 0.5,
                    }

                mock_db_entity.to_dict = mock_to_dict
                factory.create_from_db(mock_db_entity)
            else:  # create_from_config_spec
                factory.create_from_config_spec("binary_sensor.wasp", "motion")

            # Verify Decay was created with half_life=0.1 for WASP entity
            decay_call_args = mock_decay_class.call_args
            assert decay_call_args.kwargs.get("half_life") == 0.1

    def test_entity_manager_get_entities_by_input_type(
        self, coordinator: AreaOccupancyCoordinator
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
            area_name = coordinator.get_area_names()[0]
            manager = EntityManager(coordinator, area_name=area_name)

            # Create test entities with different input types
            motion_entity = create_test_entity("motion_1", coordinator=coordinator)
            motion_entity.type.input_type = InputType.MOTION

            media_entity = create_test_entity("media_1", coordinator=coordinator)
            media_entity.type.input_type = InputType.MEDIA

            door_entity = create_test_entity("door_1", coordinator=coordinator)
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
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test evidence property when neither active_states nor active_range is configured."""
        # Create entity type with no active configuration
        mock_entity_type = Mock()
        mock_entity_type.active_states = None
        mock_entity_type.active_range = None

        entity = create_test_entity(
            entity_type=mock_entity_type,
            coordinator=coordinator,
        )

        # Set up a valid state
        original_states = coordinator.hass.states
        mock_state = Mock()
        mock_state.state = "some_state"
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            # Should return None when neither active_states nor active_range is configured
            assert entity.evidence is None
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_update_correlation(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test update_correlation method."""
        entity = create_test_entity(coordinator=coordinator)

        # Test with valid positive correlation
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "strong_positive",
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
        }
        # Store original values before update
        original_prob_true = entity.prob_given_true
        original_prob_false = entity.prob_given_false

        entity.update_correlation(correlation_data)
        # Active > 10 + 2*2 = 14
        assert entity.learned_active_range == (14.0, float("inf"))
        assert entity.active_range == (14.0, float("inf"))
        # Since occupied stats are missing, Gaussian params should not be set
        assert entity.learned_gaussian_params is None
        # prob_given_true/false should NOT be updated (no fallback)
        assert entity.prob_given_true == original_prob_true
        assert entity.prob_given_false == original_prob_false

        # Test with valid negative correlation
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "strong_negative",
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
        }
        entity.update_correlation(correlation_data)
        # Active < 10 - 2*2 = 6
        assert entity.learned_active_range == (float("-inf"), 6.0)
        assert entity.active_range == (float("-inf"), 6.0)
        # Since occupied stats are missing, Gaussian params should not be set
        assert entity.learned_gaussian_params is None
        # prob_given_true/false should NOT be updated (no fallback)
        assert entity.prob_given_true == original_prob_true
        assert entity.prob_given_false == original_prob_false

        # Test with low confidence - should set range but NOT update probabilities
        correlation_data = {
            "confidence": 0.2,
            "correlation_type": "strong_positive",
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
        }
        entity.update_correlation(correlation_data)
        # Range should be set even with low confidence
        assert entity.learned_active_range == (14.0, float("inf"))
        # Since occupied stats are missing, Gaussian params should not be set
        assert entity.learned_gaussian_params is None
        # prob_given_true/false should NOT be updated (no fallback)
        assert entity.prob_given_true == original_prob_true
        assert entity.prob_given_false == original_prob_false

        # Test with invalid data (missing keys)
        correlation_data = {
            "confidence": 0.9,
            "correlation_type": "strong_positive",
            # missing mean/std
        }
        entity.update_correlation(correlation_data)
        assert entity.learned_active_range is None

        # Test with 'none' correlation type
        correlation_data = {
            "confidence": 0.9,
            "correlation_type": "none",
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
        }
        entity.update_correlation(correlation_data)
        assert entity.learned_active_range is None

        # Test with None data
        entity.update_correlation(None)
        assert entity.learned_active_range is None

    def test_update_correlation_with_occupied_stats(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_correlation with occupied stats stores Gaussian params."""
        entity = create_test_entity(coordinator=coordinator)

        # Test with occupied stats provided
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "strong_positive",
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
            "mean_value_when_occupied": 20.0,
            "std_dev_when_occupied": 3.0,
        }

        entity.update_correlation(correlation_data)

        # Should store Gaussian params
        assert entity.learned_gaussian_params is not None
        assert entity.learned_gaussian_params.mean_occupied == 20.0
        assert entity.learned_gaussian_params.std_occupied == 3.0
        assert entity.learned_gaussian_params.mean_unoccupied == 10.0
        assert entity.learned_gaussian_params.std_unoccupied == 2.0

        # Should also set learned_active_range
        assert entity.learned_active_range == (14.0, 26.0)  # 10+2*2, 20+2*3

    def test_update_correlation_binary_sensor(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_correlation with binary sensor type doesn't set learned_active_range."""
        # Create a binary sensor entity (APPLIANCE)
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.APPLIANCE),
        )

        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "strong_positive",
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
            "mean_value_when_occupied": 20.0,
            "std_dev_when_occupied": 3.0,
        }

        entity.update_correlation(correlation_data)

        # Binary sensors shouldn't have learned_active_range
        assert entity.learned_active_range is None
        # But should still store Gaussian params if provided
        assert entity.learned_gaussian_params is not None

    @pytest.mark.parametrize(
        ("correlation_type", "expected_range"),
        [
            ("positive", (14.0, float("inf"))),
            ("negative", (float("-inf"), 6.0)),
        ],
    )
    def test_update_correlation_weak_correlation_types(
        self,
        coordinator: AreaOccupancyCoordinator,
        correlation_type: str,
        expected_range: tuple[float, float],
    ) -> None:
        """Test update_correlation with weak correlation types (positive/negative, not just strong_*)."""
        entity = create_test_entity(coordinator=coordinator)

        correlation_data = {
            "confidence": 0.6,
            "correlation_type": correlation_type,
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
        }

        entity.update_correlation(correlation_data)

        # Should set range same as strong_* types
        assert entity.learned_active_range == expected_range

    @pytest.mark.parametrize(
        ("correlation_type", "mean_occupied", "std_occupied", "expected_range"),
        [
            # Positive correlation: upper_bound <= lower_bound -> fallback to open-ended
            # unoccupied: mean=10, std=2 -> lower_bound = 10 + 2*2 = 14
            # occupied: mean=12, std=1 -> upper_bound = 12 + 2*1 = 14
            # Since upper_bound (14) <= lower_bound (14), should fallback to open-ended
            (
                "strong_positive",
                12.0,
                1.0,
                (14.0, float("inf")),
            ),
            # Negative correlation: lower_bound >= upper_bound -> fallback to open-ended
            # unoccupied: mean=10, std=2 -> upper_bound = 10 - 2*2 = 6
            # occupied: mean=8, std=1 -> lower_bound = 8 - 2*1 = 6
            # Since lower_bound (6) >= upper_bound (6), should fallback to open-ended
            (
                "strong_negative",
                8.0,
                1.0,
                (float("-inf"), 6.0),
            ),
        ],
    )
    def test_update_correlation_overlapping_stats(
        self,
        coordinator: AreaOccupancyCoordinator,
        correlation_type: str,
        mean_occupied: float,
        std_occupied: float,
        expected_range: tuple[float, float],
    ) -> None:
        """Test update_correlation with overlapping stats falls back to open-ended range."""
        entity = create_test_entity(coordinator=coordinator)

        correlation_data = {
            "confidence": 0.8,
            "correlation_type": correlation_type,
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
            "mean_value_when_occupied": mean_occupied,  # Very close to unoccupied
            "std_dev_when_occupied": std_occupied,  # Small std
        }

        entity.update_correlation(correlation_data)

        # Should fallback to open-ended range when stats overlap
        assert entity.learned_active_range == expected_range

    def test_update_correlation_unknown_type(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_correlation with unknown/invalid correlation_type."""
        entity = create_test_entity(coordinator=coordinator)

        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "invalid_type",  # Unknown correlation type
            "mean_value_when_unoccupied": 10.0,
            "std_dev_when_unoccupied": 2.0,
        }

        entity.update_correlation(correlation_data)

        # Should set learned_active_range to None for unknown types
        assert entity.learned_active_range is None

    def test_update_binary_likelihoods(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update_binary_likelihoods method."""
        # Create a binary sensor entity (light)
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.APPLIANCE),
        )

        # Test with valid likelihood data
        likelihood_data = {
            "prob_given_true": 0.75,
            "prob_given_false": 0.15,
            "analysis_error": None,
            "correlation_type": "binary_likelihood",
        }

        entity.update_binary_likelihoods(likelihood_data)

        assert entity.prob_given_true == 0.75
        assert entity.prob_given_false == 0.15
        assert entity.analysis_error is None
        assert entity.correlation_type == "binary_likelihood"
        assert entity.learned_gaussian_params is None
        assert entity.learned_active_range is None

        # Test with error
        likelihood_data = {
            "prob_given_true": None,
            "prob_given_false": None,
            "analysis_error": "no_occupied_intervals",
        }

        entity.update_binary_likelihoods(likelihood_data)

        # Should reset to EntityType defaults
        assert entity.prob_given_true == entity.type.prob_given_true
        assert entity.prob_given_false == entity.type.prob_given_false
        assert entity.analysis_error == "no_occupied_intervals"
        assert entity.learned_gaussian_params is None
        assert entity.learned_active_range is None

    @pytest.mark.parametrize("empty_input", [{}, None])
    def test_update_binary_likelihoods_empty_input(
        self,
        coordinator: AreaOccupancyCoordinator,
        empty_input: dict | None,
    ) -> None:
        """Test update_binary_likelihoods with empty input returns early."""
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.APPLIANCE),
        )

        # Set some initial values
        entity.prob_given_true = 0.8
        entity.prob_given_false = 0.2

        # Call with empty input (empty dict or None)
        entity.update_binary_likelihoods(empty_input)

        # Values should remain unchanged (early return)
        assert entity.prob_given_true == 0.8
        assert entity.prob_given_false == 0.2

    def test_get_likelihoods_binary_sensor(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods for binary sensors."""
        # Create a binary sensor entity (light)
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.APPLIANCE),
        )

        # Initially not analyzed - should use EntityType defaults
        assert entity.analysis_error == "not_analyzed"
        prob_true, prob_false = entity.get_likelihoods()
        assert prob_true == entity.type.prob_given_true
        assert prob_false == entity.type.prob_given_false

        # After analysis - should use learned probabilities
        entity.update_binary_likelihoods(
            {
                "prob_given_true": 0.8,
                "prob_given_false": 0.1,
                "analysis_error": None,
            }
        )

        prob_true, prob_false = entity.get_likelihoods()
        assert prob_true == 0.8
        assert prob_false == 0.1

        # After failed analysis - should use EntityType defaults
        entity.update_binary_likelihoods(
            {
                "prob_given_true": None,
                "prob_given_false": None,
                "analysis_error": "no_occupied_intervals",
            }
        )

        prob_true, prob_false = entity.get_likelihoods()
        assert prob_true == entity.type.prob_given_true
        assert prob_false == entity.type.prob_given_false

    def test_get_likelihoods_motion_sensor(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods for motion sensors - always uses configured values."""
        # Create a motion sensor entity
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.MOTION),
            prob_given_true=0.95,
            prob_given_false=0.02,
        )

        # Motion sensors always use configured values, not EntityType defaults
        prob_true, prob_false = entity.get_likelihoods()
        assert prob_true == 0.95
        assert prob_false == 0.02

        # Even if analysis_error changes, motion sensors still use configured values
        entity.analysis_error = "some_error"
        prob_true, prob_false = entity.get_likelihoods()
        assert prob_true == 0.95
        assert prob_false == 0.02

    def test_get_likelihoods_numeric_sensor_with_gaussian(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods for numeric sensor with Gaussian params."""
        # Create a numeric sensor entity (CO2)
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        # Set up Gaussian parameters
        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=100.0,
            mean_unoccupied=400.0,
            std_unoccupied=50.0,
        )

        # Set current state value
        original_states = coordinator.hass.states
        mock_state = Mock()
        mock_state.state = "800.0"  # Between occupied and unoccupied means
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            prob_true, prob_false = entity.get_likelihoods()

            # Should calculate Gaussian PDFs
            # Both probabilities should be positive
            assert prob_true > 0.0
            assert prob_false > 0.0
            # Value 800 is between means (400 and 1000), so both PDFs should be reasonable
            # The exact relationship depends on std devs, so just verify they're calculated
            assert isinstance(prob_true, float)
            assert isinstance(prob_false, float)
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_get_likelihoods_numeric_sensor_nan_in_mean(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods with NaN/inf in mean values falls back to defaults."""

        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        # Set Gaussian params with NaN in mean_occupied
        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=float("nan"),
            std_occupied=100.0,
            mean_unoccupied=400.0,
            std_unoccupied=50.0,
        )

        prob_true, prob_false = entity.get_likelihoods()
        # Should fall back to EntityType defaults
        assert prob_true == entity.type.prob_given_true
        assert prob_false == entity.type.prob_given_false

        # Test with inf in mean_unoccupied
        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=100.0,
            mean_unoccupied=float("inf"),
            std_unoccupied=50.0,
        )

        prob_true, prob_false = entity.get_likelihoods()
        assert prob_true == entity.type.prob_given_true
        assert prob_false == entity.type.prob_given_false

    def test_get_likelihoods_numeric_sensor_nan_in_state(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods with NaN/inf in state value uses mean of means."""

        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=100.0,
            mean_unoccupied=400.0,
            std_unoccupied=50.0,
        )

        original_states = coordinator.hass.states
        # Test with NaN state (string that converts to NaN float)
        mock_state = Mock()
        mock_state.state = "nan"  # Lowercase nan converts to NaN float
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            prob_true, prob_false = entity.get_likelihoods()
            # Should use mean of means (1000 + 400) / 2 = 700
            assert prob_true > 0.0
            assert prob_false > 0.0
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

        # Test with inf state (string that converts to inf float)
        mock_state_inf = Mock()
        mock_state_inf.state = "inf"
        _set_states_get(coordinator.hass, lambda _: mock_state_inf)
        try:
            prob_true, prob_false = entity.get_likelihoods()
            # Should use mean of means when state is inf
            assert prob_true > 0.0
            assert prob_false > 0.0
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_get_likelihoods_numeric_sensor_invalid_state(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods with invalid (non-numeric) state uses mean of means."""
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=100.0,
            mean_unoccupied=400.0,
            std_unoccupied=50.0,
        )

        original_states = coordinator.hass.states
        # Test with non-numeric state
        mock_state = Mock()
        mock_state.state = "invalid"
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            prob_true, prob_false = entity.get_likelihoods()
            # Should use mean of means as fallback
            assert prob_true > 0.0
            assert prob_false > 0.0
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_get_likelihoods_numeric_sensor_no_state(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods with no state available uses mean of means."""
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=100.0,
            mean_unoccupied=400.0,
            std_unoccupied=50.0,
        )

        original_states = coordinator.hass.states
        _set_states_get(coordinator.hass, lambda _: None)
        try:
            prob_true, prob_false = entity.get_likelihoods()
            # Should use mean of means when state is unavailable
            assert prob_true > 0.0
            assert prob_false > 0.0
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_get_likelihoods_numeric_sensor_std_zero(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods with std <= 0 clamps to minimum 0.05."""
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=0.0,  # Zero std - will be clamped to 0.05
            mean_unoccupied=400.0,
            std_unoccupied=-1.0,  # Negative std - will be clamped to 0.05
        )

        original_states = coordinator.hass.states
        mock_state = Mock()
        mock_state.state = "800.0"
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            prob_true, prob_false = entity.get_likelihoods()
            # Should handle zero/negative std by clamping to 0.05
            # Both probabilities should be valid floats (not NaN/inf)
            assert isinstance(prob_true, float)
            assert isinstance(prob_false, float)
            assert prob_true >= 0.0
            assert prob_false >= 0.0
            # With clamped std, calculations should succeed
            # (exact values depend on Gaussian PDF calculation)
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)

    def test_get_likelihoods_numeric_sensor_nan_in_density(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods with NaN/inf in calculated densities falls back to defaults."""

        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=100.0,
            mean_unoccupied=400.0,
            std_unoccupied=50.0,
        )

        # Mock _calculate_gaussian_density to return inf for p_true to test inf handling
        original_calc = entity._calculate_gaussian_density

        def mock_calc_inf(value, mean, std):
            if mean == entity.learned_gaussian_params.mean_occupied:
                return float("inf")
            return original_calc(value, mean, std)

        entity._calculate_gaussian_density = mock_calc_inf

        original_states = coordinator.hass.states
        mock_state = Mock()
        mock_state.state = "800.0"
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            prob_true, prob_false = entity.get_likelihoods()
            # Should fall back to EntityType defaults when p_true is inf
            assert prob_true == entity.type.prob_given_true
            assert prob_false == entity.type.prob_given_false
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)
            entity._calculate_gaussian_density = original_calc

    @pytest.mark.parametrize(
        ("target_mean", "mean_key"),
        [
            ("mean_occupied", "mean_occupied"),
            ("mean_unoccupied", "mean_unoccupied"),
        ],
    )
    def test_get_likelihoods_numeric_sensor_nan_density(
        self,
        coordinator: AreaOccupancyCoordinator,
        target_mean: str,
        mean_key: str,
    ) -> None:
        """Test get_likelihoods when density is NaN/inf falls back to defaults."""

        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=100.0,
            mean_unoccupied=400.0,
            std_unoccupied=50.0,
        )

        # Mock _calculate_gaussian_density to return NaN for target mean
        original_calc = entity._calculate_gaussian_density

        def mock_calc_nan(value, mean, std):
            if mean == getattr(entity.learned_gaussian_params, mean_key):
                return float("nan")
            return original_calc(value, mean, std)

        entity._calculate_gaussian_density = mock_calc_nan

        original_states = coordinator.hass.states
        mock_state = Mock()
        mock_state.state = "800.0"
        _set_states_get(coordinator.hass, lambda _: mock_state)
        try:
            prob_true, prob_false = entity.get_likelihoods()
            # Should fall back to EntityType defaults when density is NaN
            assert prob_true == entity.type.prob_given_true
            assert prob_false == entity.type.prob_given_false
        finally:
            object.__setattr__(coordinator.hass, "states", original_states)
            entity._calculate_gaussian_density = original_calc

    def test_get_likelihoods_numeric_sensor_no_gaussian_params(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_likelihoods for numeric sensor without Gaussian params uses defaults."""
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        # No Gaussian params set
        assert entity.learned_gaussian_params is None

        prob_true, prob_false = entity.get_likelihoods()
        # Should use EntityType defaults
        assert prob_true == entity.type.prob_given_true
        assert prob_false == entity.type.prob_given_false

    def test_is_continuous_likelihood_property(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test is_continuous_likelihood property."""
        entity = create_test_entity(
            coordinator=coordinator,
            entity_type=EntityType(input_type=InputType.CO2),
        )

        # Without Gaussian params, should be False
        assert entity.is_continuous_likelihood is False

        # With Gaussian params, should be True
        entity.learned_gaussian_params = GaussianParams(
            mean_occupied=1000.0,
            std_occupied=100.0,
            mean_unoccupied=400.0,
            std_unoccupied=50.0,
        )
        assert entity.is_continuous_likelihood is True

    def test_calculate_gaussian_density_std_zero(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _calculate_gaussian_density with std <= 0 returns 0.0."""
        entity = create_test_entity(coordinator=coordinator)

        # Test with std = 0
        density = entity._calculate_gaussian_density(100.0, 50.0, 0.0)
        assert density == 0.0

        # Test with std < 0
        density = entity._calculate_gaussian_density(100.0, 50.0, -1.0)
        assert density == 0.0

    def test_calculate_gaussian_density_normal_calculation(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _calculate_gaussian_density with normal values."""
        entity = create_test_entity(coordinator=coordinator)

        # Test with value equal to mean (should give highest density)
        density_at_mean = entity._calculate_gaussian_density(50.0, 50.0, 10.0)
        assert density_at_mean > 0.0

        # Test with value away from mean (should give lower density)
        density_away = entity._calculate_gaussian_density(70.0, 50.0, 10.0)
        assert density_away > 0.0
        assert density_away < density_at_mean

    def test_calculate_gaussian_density_extreme_values(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _calculate_gaussian_density with extreme values."""
        entity = create_test_entity(coordinator=coordinator)

        # Test with very large numbers
        density = entity._calculate_gaussian_density(1e100, 1e100, 1e50)
        assert isinstance(density, float)
        assert density >= 0.0

        # Test with very small numbers
        density = entity._calculate_gaussian_density(1e-100, 1e-100, 1e-50)
        assert isinstance(density, float)
        assert density >= 0.0

    @pytest.mark.asyncio
    async def test_entity_manager_cleanup(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test EntityManager cleanup method."""
        # Create manager with mocked factory
        with patch(
            "custom_components.area_occupancy.data.entity.EntityFactory"
        ) as mock_factory_class:
            original_entity = Mock()
            original_entity.entity_id = "original"
            new_entity = Mock()
            new_entity.entity_id = "new"

            mock_factory = Mock()
            # First call during initialization
            mock_factory.create_all_from_config.return_value = {
                "original": original_entity
            }
            mock_factory_class.return_value = mock_factory

            # EntityManager requires area_name
            area_name = coordinator.get_area_names()[0]
            manager = EntityManager(coordinator, area_name=area_name)

            # Verify initial state
            assert "original" in manager._entities
            assert manager._entities["original"] == original_entity

            # Update factory to return new entities after cleanup
            mock_factory.create_all_from_config.return_value = {"new": new_entity}

            # Test cleanup method - should clear and recreate entities
            await manager.cleanup()

            # Verify entities were cleared and recreated
            assert "original" not in manager._entities
            assert "new" in manager._entities
            assert manager._entities["new"] == new_entity
            # Verify factory was called (initialization + cleanup)
            assert mock_factory.create_all_from_config.call_count == 2


class TestEntityFactory:
    """Test the EntityFactory class."""

    def test_initialization(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test factory initialization."""
        # EntityFactory requires area_name
        area_name = coordinator.get_area_names()[0]
        factory = EntityFactory(coordinator, area_name=area_name)
        assert factory.coordinator == coordinator

    def test_initialization_invalid_area_name(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test EntityFactory initialization with invalid area_name raises ValueError."""
        with pytest.raises(ValueError, match="Area 'nonexistent' not found"):
            EntityFactory(coordinator, area_name="nonexistent")

    @patch("custom_components.area_occupancy.data.entity.EntityType")
    @patch("custom_components.area_occupancy.data.entity.Decay")
    def test_create_from_config_spec(
        self,
        mock_decay_class: Mock,
        mock_entity_type_class: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test creating entity from config spec."""
        mock_entity_type = Mock()
        # EntityType is instantiated directly, not via create()
        mock_entity_type_class.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.return_value = mock_decay  # Decay uses __init__, not create()

        # EntityFactory requires area_name
        area_name = coordinator.get_area_names()[0]
        factory = EntityFactory(coordinator, area_name=area_name)
        entity = factory.create_from_config_spec("test_entity", "motion")

        assert entity.entity_id == "test_entity"
        assert entity.type == mock_entity_type
        assert entity.decay == mock_decay
        assert entity.hass == coordinator.hass

    @patch("custom_components.area_occupancy.data.entity.EntityType")
    @patch("custom_components.area_occupancy.data.entity.Decay")
    def test_create_all_from_config(
        self,
        mock_decay_class: Mock,
        mock_entity_type_class: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test creating all entities from config."""
        mock_entity_type = Mock()
        # EntityType is instantiated directly, not via create()
        mock_entity_type_class.return_value = mock_entity_type
        mock_decay = Mock()
        mock_decay_class.return_value = mock_decay  # Decay uses __init__, not create()

        # Mock config with sensors - use area config if available
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.motion = ["binary_sensor.motion1"]
        area.config.sensors.media = ["media_player.tv"]
        # EntityFactory requires area_name
        factory = EntityFactory(coordinator, area_name=area_name)
        entities = factory.create_all_from_config()

        # Should create entities for all configured sensors
        assert len(entities) >= 2  # At least motion and media sensors
        assert "binary_sensor.motion1" in entities
        assert "media_player.tv" in entities

    def test_get_entity_type_mapping(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_entity_type_mapping method."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.motion = ["binary_sensor.motion1", "binary_sensor.motion2"]
        area.config.sensors.media = ["media_player.tv"]
        area.config.sensors.appliance = ["switch.light"]
        area.config.sensors.door = ["binary_sensor.door"]
        area.config.sensors.window = ["binary_sensor.window"]
        area.config.sensors.co2 = ["sensor.co2"]
        area.config.sensors.co = ["sensor.co"]

        factory = EntityFactory(coordinator, area_name=area_name)
        mapping = factory.get_entity_type_mapping()

        # Verify mapping contains all configured sensors
        assert "binary_sensor.motion1" in mapping
        assert mapping["binary_sensor.motion1"] == "motion"
        assert "binary_sensor.motion2" in mapping
        assert mapping["binary_sensor.motion2"] == "motion"
        assert "media_player.tv" in mapping
        assert mapping["media_player.tv"] == "media"
        assert "switch.light" in mapping
        assert mapping["switch.light"] == "appliance"
        assert "binary_sensor.door" in mapping
        assert mapping["binary_sensor.door"] == "door"
        assert "binary_sensor.window" in mapping
        assert mapping["binary_sensor.window"] == "window"
        assert "sensor.co2" in mapping
        assert mapping["sensor.co2"] == "co2"
        assert "sensor.co" in mapping
        assert mapping["sensor.co"] == "co"


# ruff: noqa: SLF001
@pytest.fixture
def mock_numeric_entity():
    """Create a mock numeric entity for testing."""
    entity_type = EntityType(
        input_type=InputType.TEMPERATURE,
        weight=0.1,
        prob_given_true=0.5,
        prob_given_false=0.5,
        active_range=None,
    )
    decay = Decay(half_life=60.0)

    return Entity(
        entity_id="sensor.temp",
        type=entity_type,
        prob_given_true=0.5,
        prob_given_false=0.5,
        decay=decay,
        state_provider=lambda x: "20.0",
        last_updated=dt_util.utcnow(),
    )


@pytest.fixture
def mock_binary_entity():
    """Create a mock binary entity for testing."""
    entity_type = EntityType(
        input_type=InputType.MEDIA,
        weight=0.7,
        prob_given_true=0.5,
        prob_given_false=0.5,
        active_states=[STATE_ON],
    )
    decay = Decay(half_life=60.0)

    return Entity(
        entity_id="media_player.tv",
        type=entity_type,
        prob_given_true=0.5,
        prob_given_false=0.5,
        decay=decay,
        state_provider=lambda x: STATE_ON,
        last_updated=dt_util.utcnow(),
    )


class TestGaussianLikelihood:
    """Test Gaussian likelihood calculation for Entity class."""

    def test_is_continuous_likelihood_property(self, mock_numeric_entity):
        """Test is_continuous_likelihood property."""
        # Initially false
        assert not mock_numeric_entity.is_continuous_likelihood

        # Set gaussian params
        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=22.0,
            std_occupied=1.0,
            mean_unoccupied=20.0,
            std_unoccupied=1.0,
        )

        # Now true
        assert mock_numeric_entity.is_continuous_likelihood

    @pytest.mark.parametrize(
        ("value", "mean", "std", "expected_density", "tolerance"),
        [
            # Peak density (at mean): 1 / (sqrt(2*pi) * 1) ≈ 0.3989
            (20.0, 20.0, 1.0, 0.3989, 0.0001),
            # 1 std dev away: 0.3989 * exp(-0.5 * 1^2) ≈ 0.2420
            (21.0, 20.0, 1.0, 0.2420, 0.0001),
            # 2 std dev away: 0.3989 * exp(-0.5 * 2^2) ≈ 0.0540
            (22.0, 20.0, 1.0, 0.0540, 0.0001),
            # Small std dev (higher peak): 1 / (sqrt(2*pi) * 0.1) ≈ 3.989
            (20.0, 20.0, 0.1, 3.989, 0.001),
        ],
    )
    def test_calculate_gaussian_density(
        self, mock_numeric_entity, value, mean, std, expected_density, tolerance
    ):
        """Test _calculate_gaussian_density method."""
        density = mock_numeric_entity._calculate_gaussian_density(value, mean, std)
        assert abs(density - expected_density) < tolerance

    @pytest.mark.parametrize(
        ("value", "expected_p_t", "expected_p_f", "comparison"),
        [
            # Value = 22 (Occupied Mean): P(val|Occ) peak (~0.3989), P(val|Unocc) 2 std (~0.054)
            ("22.0", 0.3989, 0.0540, "gt"),
            # Value = 20 (Unoccupied Mean): P(val|Occ) 2 std (~0.054), P(val|Unocc) peak (~0.3989)
            ("20.0", 0.0540, 0.3989, "lt"),
            # Value = 21 (Middle): equal distance from both means, densities equal (~0.2420)
            ("21.0", 0.2420, 0.2420, "eq"),
        ],
    )
    def test_get_likelihoods_continuous_numeric(
        self, mock_numeric_entity, value, expected_p_t, expected_p_f, comparison
    ):
        """Test get_likelihoods with continuous parameters for numeric sensor."""
        # Setup: Occupied Mean 22, Std 1; Unoccupied Mean 20, Std 1
        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=22.0,
            std_occupied=1.0,
            mean_unoccupied=20.0,
            std_unoccupied=1.0,
        )

        mock_numeric_entity.state_provider = lambda x: value
        p_t, p_f = mock_numeric_entity.get_likelihoods()

        assert abs(p_t - expected_p_t) < 0.001
        assert abs(p_f - expected_p_f) < 0.001

        if comparison == "gt":
            assert p_t > p_f  # Favors occupied
        elif comparison == "lt":
            assert p_t < p_f  # Favors unoccupied
        else:  # eq
            assert abs(p_t - p_f) < 0.0001  # Equal densities

    @pytest.mark.parametrize(
        ("state", "analysis_error", "expected_p_t", "expected_p_f"),
        [
            # Analyzed successfully - should return learned probabilities regardless of state
            (STATE_ON, None, 0.8, 0.1),
            (STATE_OFF, None, 0.8, 0.1),
            # Not analyzed - should use EntityType defaults
            (STATE_ON, "not_analyzed", 0.5, 0.5),
        ],
    )
    def test_get_likelihoods_binary_sensor_static(
        self, mock_binary_entity, state, analysis_error, expected_p_t, expected_p_f
    ):
        """Test get_likelihoods for binary sensor using static probabilities."""
        # Binary sensors use static probabilities, not Gaussian PDF
        mock_binary_entity.prob_given_true = 0.8
        mock_binary_entity.prob_given_false = 0.1
        mock_binary_entity.analysis_error = analysis_error
        mock_binary_entity.learned_gaussian_params = None

        mock_binary_entity.state_provider = lambda x: state
        p_t, p_f = mock_binary_entity.get_likelihoods()

        if analysis_error is None:
            # Should return learned probabilities
            assert p_t == expected_p_t
            assert p_f == expected_p_f
        else:
            # Should use EntityType defaults
            assert p_t == mock_binary_entity.type.prob_given_true
            assert p_f == mock_binary_entity.type.prob_given_false

    def test_get_likelihoods_fallback(self, mock_numeric_entity):
        """Test get_likelihoods fallback behavior uses EntityType defaults."""
        # No params -> returns EntityType defaults (not stored prob_given_true/false)
        mock_numeric_entity.learned_gaussian_params = None
        # Change stored values to verify we use EntityType defaults
        mock_numeric_entity.prob_given_true = 0.9
        mock_numeric_entity.prob_given_false = 0.1
        p_t, p_f = mock_numeric_entity.get_likelihoods()
        # Should use EntityType defaults (0.5, 0.5), not stored values
        assert p_t == 0.5
        assert p_f == 0.5

        # With params but invalid state -> uses representative value (average of means)
        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=22.0,
            std_occupied=1.0,
            mean_unoccupied=20.0,
            std_unoccupied=1.0,
        )
        mock_numeric_entity.state_provider = lambda x: "unavailable"
        p_t, p_f = mock_numeric_entity.get_likelihoods()
        # Should use representative value (average of means = 21.0) to calculate probabilities
        # This will give non-zero probabilities based on Gaussian PDF
        assert p_t > 0.0
        assert p_f > 0.0
        assert p_t != 0.5  # Should be calculated, not default
        assert p_f != 0.5  # Should be calculated, not default

    def test_update_correlation_populates_params(self, mock_numeric_entity):
        """Test update_correlation populates Gaussian params."""
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "strong_positive",
            "mean_value_when_occupied": 22.0,
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_occupied": 1.5,
            "std_dev_when_unoccupied": 1.2,
        }

        mock_numeric_entity.update_correlation(correlation_data)

        assert mock_numeric_entity.learned_gaussian_params is not None
        assert mock_numeric_entity.learned_gaussian_params.mean_occupied == 22.0
        assert mock_numeric_entity.learned_gaussian_params.std_occupied == 1.5
        assert mock_numeric_entity.learned_gaussian_params.mean_unoccupied == 20.0
        assert mock_numeric_entity.learned_gaussian_params.std_unoccupied == 1.2

        # Should also populate learned_active_range for UI
        assert mock_numeric_entity.learned_active_range is not None

    def test_update_correlation_missing_params(self, mock_numeric_entity):
        """Test update_correlation handles missing occupied stats."""
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "strong_positive",
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_unoccupied": 1.2,
            # Missing occupied stats
        }

        mock_numeric_entity.update_correlation(correlation_data)

        # Should NOT populate gaussian params
        assert mock_numeric_entity.learned_gaussian_params is None
        # Should still populate active range (open-ended)
        assert mock_numeric_entity.learned_active_range is not None
        assert mock_numeric_entity.learned_active_range[1] == float("inf")
        # Should NOT update stored prob_given_true/false (no fallback)

    def test_get_likelihoods_nan_mean_occupied(self, mock_numeric_entity):
        """Test get_likelihoods with NaN mean_occupied falls back to EntityType defaults."""

        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=float("nan"),
            std_occupied=1.0,
            mean_unoccupied=20.0,
            std_unoccupied=1.0,
        )

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should fallback to EntityType defaults
        assert p_t == mock_numeric_entity.type.prob_given_true
        assert p_f == mock_numeric_entity.type.prob_given_false

    def test_get_likelihoods_nan_mean_unoccupied(self, mock_numeric_entity):
        """Test get_likelihoods with NaN mean_unoccupied falls back to EntityType defaults."""

        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=22.0,
            std_occupied=1.0,
            mean_unoccupied=float("nan"),
            std_unoccupied=1.0,
        )

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should fallback to EntityType defaults
        assert p_t == mock_numeric_entity.type.prob_given_true
        assert p_f == mock_numeric_entity.type.prob_given_false

    def test_get_likelihoods_inf_mean_occupied(self, mock_numeric_entity):
        """Test get_likelihoods with inf mean_occupied falls back to EntityType defaults."""

        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=float("inf"),
            std_occupied=1.0,
            mean_unoccupied=20.0,
            std_unoccupied=1.0,
        )

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should fallback to EntityType defaults
        assert p_t == mock_numeric_entity.type.prob_given_true
        assert p_f == mock_numeric_entity.type.prob_given_false

    def test_get_likelihoods_inf_mean_unoccupied(self, mock_numeric_entity):
        """Test get_likelihoods with inf mean_unoccupied falls back to EntityType defaults."""

        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=22.0,
            std_occupied=1.0,
            mean_unoccupied=float("inf"),
            std_unoccupied=1.0,
        )

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should fallback to EntityType defaults
        assert p_t == mock_numeric_entity.type.prob_given_true
        assert p_f == mock_numeric_entity.type.prob_given_false

    def test_get_likelihoods_nan_state_value(self, mock_numeric_entity):
        """Test get_likelihoods with NaN state value uses mean of means."""

        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=22.0,
            std_occupied=1.0,
            mean_unoccupied=20.0,
            std_unoccupied=1.0,
        )

        # Set state to NaN
        mock_numeric_entity.state_provider = lambda x: float("nan")

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should use mean of means (21.0) and calculate valid densities
        assert not math.isnan(p_t)
        assert not math.isnan(p_f)
        assert not math.isinf(p_t)
        assert not math.isinf(p_f)
        assert p_t > 0.0
        assert p_f > 0.0

    def test_get_likelihoods_inf_state_value(self, mock_numeric_entity):
        """Test get_likelihoods with inf state value uses mean of means."""

        mock_numeric_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=22.0,
            std_occupied=1.0,
            mean_unoccupied=20.0,
            std_unoccupied=1.0,
        )

        # Set state to inf
        mock_numeric_entity.state_provider = lambda x: float("inf")

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should use mean of means (21.0) and calculate valid densities
        assert not math.isnan(p_t)
        assert not math.isnan(p_f)
        assert not math.isinf(p_t)
        assert not math.isinf(p_f)
        assert p_t > 0.0
        assert p_f > 0.0

    def test_get_likelihoods_motion_sensor_uses_configured_values(self):
        """Test that motion sensors always use configured prob_given_true/false."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.85,
            prob_given_true=0.95,  # EntityType default
            prob_given_false=0.02,  # EntityType default
            active_states=[STATE_ON],
        )
        decay = Decay(half_life=60.0)

        # Create motion sensor with different configured values
        motion_entity = Entity(
            entity_id="binary_sensor.motion",
            type=entity_type,
            prob_given_true=0.9,  # Configured value (different from EntityType)
            prob_given_false=0.05,  # Configured value (different from EntityType)
            decay=decay,
            state_provider=lambda x: STATE_ON,
            last_updated=dt_util.utcnow(),
        )

        # Even with Gaussian params, motion sensors should use configured values
        motion_entity.learned_gaussian_params = GaussianParams(
            mean_occupied=0.9,
            std_occupied=0.3,
            mean_unoccupied=0.1,
            std_unoccupied=0.3,
        )

        p_t, p_f = motion_entity.get_likelihoods()
        # Should use configured values, not Gaussian params or EntityType defaults
        assert p_t == 0.9
        assert p_f == 0.05
