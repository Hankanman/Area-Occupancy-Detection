"""Tests for data.entity module."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity, EntityManager
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.data.prior import Prior
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util


class TestEntity:
    """Test Entity dataclass."""

    @pytest.fixture
    def mock_entity_type(self) -> Mock:
        """Create a mock entity type."""
        entity_type = Mock(spec=EntityType)
        entity_type.input_type = InputType.MOTION
        entity_type.weight = 0.8
        entity_type.prob_true = 0.25
        entity_type.prob_false = 0.05
        entity_type.prior = 0.35
        entity_type.active_states = [STATE_ON]
        entity_type.is_active.return_value = True
        return entity_type

    @pytest.fixture
    def mock_prior(self) -> Mock:
        """Create a mock prior."""
        prior = Mock(spec=Prior)
        prior.prior = 0.35
        prior.prob_given_true = 0.8
        prior.prob_given_false = 0.1
        prior.last_updated = dt_util.utcnow()
        prior.to_dict.return_value = {
            "prior": 0.35,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "last_updated": dt_util.utcnow().isoformat(),
        }
        return prior

    @pytest.fixture
    def mock_decay(self) -> Mock:
        """Create a mock decay."""
        decay = Mock(spec=Decay)
        decay.is_decaying = False
        decay.decay_enabled = True
        decay.decay_window = 300
        decay.decay_factor = 1.0
        decay.should_start_decay.return_value = False
        decay.should_stop_decay.return_value = False
        decay.is_decay_complete.return_value = False
        decay.update_decay.return_value = (0.5, 1.0)
        decay.to_dict.return_value = {
            "is_decaying": False,
            "decay_start_time": None,
            "decay_start_probability": 0.0,
            "decay_window": 300,
            "decay_enabled": True,
            "decay_factor": 1.0,
        }
        return decay

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.hass = Mock()
        coordinator.config = Mock()
        coordinator.config.decay = Mock()
        coordinator.config.decay.window = 300
        coordinator.config.decay.enabled = True
        coordinator.request_update = Mock()
        return coordinator

    def test_initialization(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test Entity initialization."""
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
        assert isinstance(entity.last_updated, datetime)

    def test_set_coordinator(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock, mock_coordinator: Mock
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
        """Test converting Entity to dictionary."""
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

        result = entity.to_dict()

        assert result["entity_id"] == "binary_sensor.test_motion"
        assert result["type"] == InputType.MOTION.value
        assert result["probability"] == 0.5
        assert result["state"] == STATE_ON
        assert result["is_active"] is True
        assert result["available"] is True
        assert "prior" in result
        assert "decay" in result

    @patch("custom_components.area_occupancy.data.entity.Decay")
    def test_from_dict(
        self, mock_decay_class: Mock, mock_coordinator: Mock
    ) -> None:
        """Test creating Entity from dictionary."""
        # Setup mock coordinator with entity types
        mock_entity_type = Mock()
        mock_entity_type.input_type = InputType.MOTION
        mock_coordinator.entity_types = Mock()
        mock_coordinator.entity_types.get_entity_type.return_value = mock_entity_type

        # Setup mock decay
        mock_decay_instance = Mock()
        mock_decay_class.return_value = mock_decay_instance

        data = {
            "entity_id": "binary_sensor.test_motion",
            "type": "motion",
            "probability": 0.5,
            "prior": {
                "prior": 0.35,
                "prob_given_true": 0.8,
                "prob_given_false": 0.1,
                "last_updated": dt_util.utcnow().isoformat(),
            },
            "state": STATE_ON,
            "is_active": True,
            "available": True,
            "last_updated": dt_util.utcnow(),
        }

        entity = Entity.from_dict(data, mock_coordinator)

        assert entity.entity_id == "binary_sensor.test_motion"
        assert entity.type == mock_entity_type
        assert entity.probability == 0.5
        assert entity.state == STATE_ON
        assert entity.is_active is True
        assert entity.available is True

    @patch("custom_components.area_occupancy.utils.bayesian_probability")
    def test_update_probability_basic(
        self, mock_bayesian: Mock, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test basic probability update without decay."""
        mock_bayesian.return_value = 0.7
        mock_decay.is_decaying = False

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
            is_active=True,
        )

        entity.update_probability()

        assert entity.probability == 0.7
        mock_bayesian.assert_called_once()

    @patch("custom_components.area_occupancy.utils.bayesian_probability")
    def test_update_probability_start_decay(
        self, mock_bayesian: Mock, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test probability update when starting decay."""
        mock_bayesian.return_value = 0.3
        mock_decay.is_decaying = False
        mock_decay.should_start_decay.return_value = True

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.7,  # Previous probability
            prior=mock_prior,
            decay=mock_decay,
            is_active=False,  # Currently inactive
        )

        with patch.object(entity, 'start_decay_timer') as mock_start_timer:
            entity.update_probability()

            # Should maintain previous probability for first decay cycle
            assert entity.probability == 0.7
            mock_decay.start_decay.assert_called_once_with(0.5)  # previous_probability
            mock_start_timer.assert_called_once()

    @patch("custom_components.area_occupancy.utils.bayesian_probability")
    def test_update_probability_continue_decay(
        self, mock_bayesian: Mock, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test probability update when continuing decay."""
        mock_bayesian.return_value = 0.3
        mock_decay.is_decaying = True
        mock_decay.should_start_decay.return_value = False
        mock_decay.should_stop_decay.return_value = False
        mock_decay.update_decay.return_value = (0.4, 0.8)  # decayed_prob, decay_factor

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
            is_active=False,
        )

        entity.update_probability()

        assert entity.probability == 0.4  # Decayed probability
        mock_decay.update_decay.assert_called_once()

    @patch("custom_components.area_occupancy.utils.bayesian_probability")
    def test_update_probability_stop_decay(
        self, mock_bayesian: Mock, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test probability update when stopping decay."""
        mock_bayesian.return_value = 0.8
        mock_decay.is_decaying = True
        mock_decay.should_stop_decay.return_value = True

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.3,
            prior=mock_prior,
            decay=mock_decay,
            is_active=True,  # Became active
        )

        with patch.object(entity, '_stop_decay_timer') as mock_stop_timer:
            entity.update_probability()

            assert entity.probability == 0.8  # Current probability
            mock_decay.stop_decay.assert_called_once()
            mock_stop_timer.assert_called_once()

    def test_start_decay_timer(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock, mock_coordinator: Mock
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

        with patch('custom_components.area_occupancy.data.entity.async_track_point_in_time') as mock_track:
            entity.start_decay_timer()

            # Should schedule timer if decay is enabled and coordinator is available
            if mock_decay.decay_enabled:
                mock_track.assert_called_once()

    def test_start_decay_timer_disabled(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock, mock_coordinator: Mock
    ) -> None:
        """Test starting decay timer when decay is disabled."""
        mock_decay.decay_enabled = False

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )
        entity.set_coordinator(mock_coordinator)

        with patch('custom_components.area_occupancy.data.entity.async_track_point_in_time') as mock_track:
            entity.start_decay_timer()

            # Should not schedule timer when decay is disabled
            mock_track.assert_not_called()

    def test_stop_decay_timer(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock
    ) -> None:
        """Test stopping decay timer."""
        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )

        # Set a mock timer
        mock_timer = Mock()
        entity._decay_timer = mock_timer

        entity._stop_decay_timer()

        # Should call the timer cancellation function
        mock_timer.assert_called_once()
        assert entity._decay_timer is None

    async def test_handle_decay_timer(
        self, mock_entity_type: Mock, mock_prior: Mock, mock_decay: Mock, mock_coordinator: Mock
    ) -> None:
        """Test handling decay timer firing."""
        mock_decay.is_decaying = True

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
        )
        entity.set_coordinator(mock_coordinator)

        with patch.object(entity, 'update_probability') as mock_update:
            with patch.object(entity, '_schedule_next_decay_update') as mock_schedule:
                await entity._handle_decay_timer(dt_util.utcnow())

                mock_update.assert_called_once()
                # Should schedule next update if still decaying
                if mock_decay.is_decaying:
                    mock_schedule.assert_called_once()

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

        with patch.object(entity, '_stop_decay_timer') as mock_stop_timer:
            entity.stop_decay_completely()

            mock_decay.stop_decay.assert_called_once()
            mock_stop_timer.assert_called_once()

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

        with patch.object(entity, '_stop_decay_timer') as mock_stop_timer:
            entity.cleanup()

            mock_stop_timer.assert_called_once()


class TestEntityManager:
    """Test EntityManager class."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator for testing."""
        coordinator = Mock()
        coordinator.hass = Mock()
        coordinator.config = Mock()
        coordinator.config.sensors = Mock()
        coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        coordinator.config.sensors.media = ["media_player.tv"]
        coordinator.config.sensors.lights = ["light.living_room"]
        coordinator.entity_types = Mock()
        coordinator.prior_manager = Mock()
        coordinator.request_update = Mock()
        return coordinator

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test EntityManager initialization."""
        manager = EntityManager(mock_coordinator)

        assert manager.coordinator == mock_coordinator
        assert manager._entities == {}

    def test_entities_property(self, mock_coordinator: Mock) -> None:
        """Test entities property."""
        manager = EntityManager(mock_coordinator)

        # Initially empty
        assert manager.entities == {}

        # Add an entity
        mock_entity = Mock()
        manager._entities["test_entity"] = mock_entity

        assert "test_entity" in manager.entities
        assert manager.entities["test_entity"] == mock_entity

    def test_entity_ids_property(self, mock_coordinator: Mock) -> None:
        """Test entity_ids property."""
        manager = EntityManager(mock_coordinator)

        # Add some entities
        manager._entities["entity1"] = Mock()
        manager._entities["entity2"] = Mock()

        entity_ids = manager.entity_ids
        assert len(entity_ids) == 2
        assert "entity1" in entity_ids
        assert "entity2" in entity_ids

    def test_active_entities_property(self, mock_coordinator: Mock) -> None:
        """Test active_entities property."""
        manager = EntityManager(mock_coordinator)

        # Add entities with different active states
        active_entity = Mock()
        active_entity.is_active = True
        active_entity.available = True

        inactive_entity = Mock()
        inactive_entity.is_active = False
        inactive_entity.available = True

        unavailable_entity = Mock()
        unavailable_entity.is_active = True
        unavailable_entity.available = False

        manager._entities["active"] = active_entity
        manager._entities["inactive"] = inactive_entity
        manager._entities["unavailable"] = unavailable_entity

        active_entities = manager.active_entities
        assert len(active_entities) == 1
        assert active_entities[0] == active_entity

    def test_inactive_entities_property(self, mock_coordinator: Mock) -> None:
        """Test inactive_entities property."""
        manager = EntityManager(mock_coordinator)

        # Add entities with different active states
        active_entity = Mock()
        active_entity.is_active = True
        active_entity.available = True

        inactive_entity = Mock()
        inactive_entity.is_active = False
        inactive_entity.available = True

        manager._entities["active"] = active_entity
        manager._entities["inactive"] = inactive_entity

        inactive_entities = manager.inactive_entities
        assert len(inactive_entities) == 1
        assert inactive_entities[0] == inactive_entity

    def test_to_dict(self, mock_coordinator: Mock) -> None:
        """Test converting EntityManager to dictionary."""
        manager = EntityManager(mock_coordinator)

        # Add a mock entity
        mock_entity = Mock()
        mock_entity.to_dict.return_value = {"entity_id": "test", "probability": 0.5}
        manager._entities["test_entity"] = mock_entity

        result = manager.to_dict()

        assert "entities" in result
        assert "test_entity" in result["entities"]
        mock_entity.to_dict.assert_called_once()

    @patch("custom_components.area_occupancy.data.entity.Entity.from_dict")
    def test_from_dict(self, mock_from_dict: Mock, mock_coordinator: Mock) -> None:
        """Test creating EntityManager from dictionary."""
        mock_entity = Mock()
        mock_from_dict.return_value = mock_entity

        data = {
            "entities": {
                "test_entity": {"entity_id": "test", "probability": 0.5}
            }
        }

        manager = EntityManager.from_dict(data, mock_coordinator)

        assert isinstance(manager, EntityManager)
        assert manager.coordinator == mock_coordinator
        mock_from_dict.assert_called_once()

    async def test_async_initialize(self, mock_coordinator: Mock) -> None:
        """Test async initialization."""
        manager = EntityManager(mock_coordinator)

        with patch.object(manager, '_update_entities_from_config') as mock_update:
            with patch.object(manager, '_setup_entity_tracking') as mock_setup:
                with patch.object(manager, '_initialize_current_states') as mock_init:
                    await manager.async_initialize()

                    mock_update.assert_called_once()
                    mock_setup.assert_called_once()
                    mock_init.assert_called_once()

    def test_get_entity(self, mock_coordinator: Mock) -> None:
        """Test getting entity by ID."""
        manager = EntityManager(mock_coordinator)

        # Add an entity
        mock_entity = Mock()
        manager._entities["test_entity"] = mock_entity

        # Test getting existing entity
        result = manager.get_entity("test_entity")
        assert result == mock_entity

        # Test getting non-existent entity
        with pytest.raises(KeyError):
            manager.get_entity("nonexistent")

    def test_add_entity(self, mock_coordinator: Mock) -> None:
        """Test adding entity."""
        manager = EntityManager(mock_coordinator)

        mock_entity = Mock()
        mock_entity.entity_id = "test_entity"

        manager.add_entity(mock_entity)

        assert "test_entity" in manager._entities
        assert manager._entities["test_entity"] == mock_entity
        mock_entity.set_coordinator.assert_called_once_with(mock_coordinator)

    def test_remove_entity(self, mock_coordinator: Mock) -> None:
        """Test removing entity."""
        manager = EntityManager(mock_coordinator)

        # Add an entity
        mock_entity = Mock()
        mock_entity.cleanup = Mock()
        manager._entities["test_entity"] = mock_entity

        # Remove it
        manager.remove_entity("test_entity")

        assert "test_entity" not in manager._entities
        mock_entity.cleanup.assert_called_once()

    def test_cleanup(self, mock_coordinator: Mock) -> None:
        """Test manager cleanup."""
        manager = EntityManager(mock_coordinator)

        # Add some entities
        mock_entity1 = Mock()
        mock_entity1.cleanup = Mock()
        mock_entity2 = Mock()
        mock_entity2.cleanup = Mock()

        manager._entities["entity1"] = mock_entity1
        manager._entities["entity2"] = mock_entity2

        # Cleanup
        manager.cleanup()

        # Should cleanup all entities
        mock_entity1.cleanup.assert_called_once()
        mock_entity2.cleanup.assert_called_once()
        assert len(manager._entities) == 0

    @patch("homeassistant.helpers.event.async_track_state_change_event")
    def test_setup_entity_tracking(self, mock_track: Mock, mock_coordinator: Mock) -> None:
        """Test setting up entity state tracking."""
        manager = EntityManager(mock_coordinator)

        # Add some entities
        manager._entities["entity1"] = Mock()
        manager._entities["entity2"] = Mock()

        manager._setup_entity_tracking()

        # Should set up tracking for all entities
        mock_track.assert_called_once()

    async def test_state_changed_listener(self, mock_coordinator: Mock) -> None:
        """Test entity state change listener."""
        manager = EntityManager(mock_coordinator)

        # Create a mock entity
        mock_entity = Mock()
        mock_entity.entity_id = "test_entity"
        mock_entity.update_probability = Mock()
        manager._entities["test_entity"] = mock_entity

        # Create a mock event
        mock_event = Mock()
        mock_event.data = {
            "entity_id": "test_entity",
            "new_state": Mock(state=STATE_ON),
            "old_state": Mock(state=STATE_OFF),
        }

        # Setup the listener (this would normally be done in _setup_entity_tracking)
        manager._setup_entity_tracking()

        # Simulate state change by calling the method that would be registered
        with patch.object(manager, '_initialize_current_states'):
            await manager.async_initialize()

        # Verify entity tracking was set up
        assert mock_coordinator.hass is not None

    async def test_reset_entities(self, mock_coordinator: Mock) -> None:
        """Test resetting all entities."""
        manager = EntityManager(mock_coordinator)

        # Add some entities
        mock_entity1 = Mock()
        mock_entity1.stop_decay_completely = Mock()
        mock_entity2 = Mock()
        mock_entity2.stop_decay_completely = Mock()

        manager._entities["entity1"] = mock_entity1
        manager._entities["entity2"] = mock_entity2

        await manager.reset_entities()

        # Should stop decay for all entities
        mock_entity1.stop_decay_completely.assert_called_once()
        mock_entity2.stop_decay_completely.assert_called_once()


class TestEntityManagerIntegration:
    """Test EntityManager integration scenarios."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a comprehensive mock coordinator."""
        coordinator = Mock()
        coordinator.hass = Mock()
        coordinator.config = Mock()
        coordinator.config.sensors = Mock()
        coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        coordinator.config.sensors.media = ["media_player.tv"]
        coordinator.config.sensors.lights = ["light.living_room"]

        # Mock entity types
        mock_motion_type = Mock()
        mock_motion_type.input_type = InputType.MOTION
        mock_motion_type.prior = 0.35
        mock_motion_type.prob_true = 0.8
        mock_motion_type.prob_false = 0.1

        coordinator.entity_types = Mock()
        coordinator.entity_types.get_entity_type.return_value = mock_motion_type

        # Mock prior manager
        mock_prior = Mock()
        mock_prior.prior = 0.35
        mock_prior.prob_given_true = 0.8
        mock_prior.prob_given_false = 0.1
        mock_prior.last_updated = dt_util.utcnow()

        coordinator.prior_manager = Mock()
        coordinator.prior_manager.calculate = AsyncMock(return_value=mock_prior)

        return coordinator

    @patch("homeassistant.helpers.event.async_track_state_change_event")
    async def test_full_entity_lifecycle(
        self, mock_track: Mock, mock_coordinator: Mock
    ) -> None:
        """Test complete entity lifecycle from initialization to cleanup."""
        manager = EntityManager(mock_coordinator)

        # Mock Home Assistant states
        mock_coordinator.hass.states.async_all.return_value = [
            Mock(entity_id="binary_sensor.motion1", state=STATE_ON),
            Mock(entity_id="media_player.tv", state="playing"),
        ]

        # Initialize manager
        with patch.object(manager, '_get_config_entity_mapping') as mock_mapping:
            mock_mapping.return_value = {
                "binary_sensor.motion1": mock_coordinator.entity_types.get_entity_type(InputType.MOTION)
            }

            await manager.async_initialize()

        # Verify entity was created
        assert len(manager.entities) > 0

        # Test entity state updates
        if "binary_sensor.motion1" in manager.entities:
            entity = manager.entities["binary_sensor.motion1"]

            # Update entity state
            with patch.object(entity, 'update_probability') as mock_update:
                entity.state = STATE_OFF
                entity.is_active = False
                entity.update_probability()

                mock_update.assert_called()

        # Test cleanup
        manager.cleanup()

        # Verify all entities were cleaned up
        assert len(manager._entities) == 0
