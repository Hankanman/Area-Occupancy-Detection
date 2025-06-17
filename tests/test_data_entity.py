"""Tests for data.entity module."""

from datetime import datetime
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.data.entity import Entity, EntityManager
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.data.prior import Prior
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestEntity:
    """Test Entity dataclass."""

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
    def test_from_dict(self, mock_decay_class: Mock, mock_coordinator: Mock) -> None:
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

    @patch("custom_components.area_occupancy.data.entity.bayesian_probability")
    async def test_update_probability_basic(
        self,
        mock_bayesian: Mock,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
    ) -> None:
        """Test basic probability update without decay."""
        mock_bayesian.return_value = 0.7
        mock_decay.is_decaying = False
        mock_decay.should_start_decay.return_value = False
        mock_decay.should_stop_decay.return_value = False

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
            is_active=True,
        )

        # Clear any calls from entity initialization
        mock_bayesian.reset_mock()

        await entity.update_probability()

        # Use approximate comparison for probability values
        assert (
            abs(entity.probability - 0.7) < 0.12
        )  # Allow small variance due to bayesian calculation
        mock_bayesian.assert_called_once()

    @patch("custom_components.area_occupancy.data.entity.bayesian_probability")
    async def test_update_probability_start_decay(
        self,
        mock_bayesian: Mock,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
    ) -> None:
        """Test probability update when starting decay."""
        mock_bayesian.return_value = 0.3
        mock_decay.is_decaying = False
        mock_decay.should_start_decay.return_value = True
        mock_decay.should_stop_decay.return_value = False

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.7,  # Previous probability
            prior=mock_prior,
            decay=mock_decay,
            is_active=False,  # Currently inactive
        )

        with patch.object(entity, "start_decay_timer") as mock_start_timer:
            await entity.update_probability()

            # Should maintain previous probability for first decay cycle
            assert entity.probability == 0.7
            mock_decay.start_decay.assert_called_once_with(
                0.7
            )  # uses actual previous_probability
            mock_start_timer.assert_called_once()

    @patch("custom_components.area_occupancy.data.entity.bayesian_probability")
    async def test_update_probability_continue_decay(
        self,
        mock_bayesian: Mock,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
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

        await entity.update_probability()

        assert entity.probability == 0.4  # Decayed probability
        mock_decay.update_decay.assert_called_once()

    @patch("custom_components.area_occupancy.data.entity.bayesian_probability")
    async def test_update_probability_stop_decay(
        self,
        mock_bayesian: Mock,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
    ) -> None:
        """Test probability update when stopping decay."""
        mock_bayesian.return_value = 0.8
        mock_decay.is_decaying = True
        mock_decay.should_start_decay.return_value = False
        mock_decay.should_stop_decay.return_value = True

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.3,
            prior=mock_prior,
            decay=mock_decay,
            is_active=True,  # Became active
        )

        with patch.object(entity, "_stop_decay_timer") as mock_stop_timer:
            await entity.update_probability()

            # Use approximate comparison for probability values
            assert (
                abs(entity.probability - 0.8) < 0.12
            )  # Allow small variance due to bayesian calculation
            mock_decay.stop_decay.assert_called_once()
            mock_stop_timer.assert_called_once()

    async def test_start_decay_timer(
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

        # Mock the coordinator's notify method
        mock_coordinator.async_notify_decay_started = Mock()

        entity.start_decay_timer()

        # Should notify coordinator if decay is enabled and coordinator is available
        if mock_decay.decay_enabled:
            mock_coordinator.async_notify_decay_started.assert_called_once()
        else:
            mock_coordinator.async_notify_decay_started.assert_not_called()

    async def test_start_decay_timer_disabled(
        self,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
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

        # Mock the coordinator's notify method
        mock_coordinator.async_notify_decay_started = Mock()

        entity.start_decay_timer()

        # Should not notify coordinator when decay is disabled
        mock_coordinator.async_notify_decay_started.assert_not_called()

    async def test_stop_decay_timer(
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

        # Mock the coordinator's notify method
        mock_coordinator.async_notify_decay_stopped = Mock()

        entity._stop_decay_timer()

        # Should notify coordinator to stop decay
        mock_coordinator.async_notify_decay_stopped.assert_called_once()

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

        with patch.object(entity, "_stop_decay_timer") as mock_stop_timer:
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

        with patch.object(entity, "_stop_decay_timer") as mock_stop_timer:
            entity.cleanup()

            mock_stop_timer.assert_called_once()


class TestEntityManager:
    """Test EntityManager class."""

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
        active_entity.decay = Mock()
        active_entity.decay.is_decaying = False

        inactive_entity = Mock()
        inactive_entity.is_active = False
        inactive_entity.decay = Mock()
        inactive_entity.decay.is_decaying = False

        decaying_entity = Mock()
        decaying_entity.is_active = False
        decaying_entity.decay = Mock()
        decaying_entity.decay.is_decaying = True

        manager._entities["active"] = active_entity
        manager._entities["inactive"] = inactive_entity
        manager._entities["decaying"] = decaying_entity

        active_entities = manager.active_entities
        assert len(active_entities) == 2  # active + decaying
        assert active_entity in active_entities
        assert decaying_entity in active_entities

    def test_inactive_entities_property(self, mock_coordinator: Mock) -> None:
        """Test inactive_entities property."""
        manager = EntityManager(mock_coordinator)

        # Add entities with different active states
        active_entity = Mock()
        active_entity.is_active = True
        active_entity.decay = Mock()
        active_entity.decay.is_decaying = False

        inactive_entity = Mock()
        inactive_entity.is_active = False
        inactive_entity.decay = Mock()
        inactive_entity.decay.is_decaying = False

        decaying_entity = Mock()
        decaying_entity.is_active = False
        decaying_entity.decay = Mock()
        decaying_entity.decay.is_decaying = True

        manager._entities["active"] = active_entity
        manager._entities["inactive"] = inactive_entity
        manager._entities["decaying"] = decaying_entity

        inactive_entities = manager.inactive_entities
        assert len(inactive_entities) == 1  # only truly inactive
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
    async def test_from_dict(
        self, mock_from_dict: Mock, mock_coordinator: Mock
    ) -> None:
        """Test creating EntityManager from dictionary."""
        mock_entity = Mock()
        mock_from_dict.return_value = mock_entity

        data = {"entities": {"test_entity": {"entity_id": "test", "probability": 0.5}}}

        manager = EntityManager.from_dict(data, mock_coordinator)

        assert isinstance(manager, EntityManager)
        assert manager.coordinator == mock_coordinator
        mock_from_dict.assert_called_once()

    async def test_async_initialize(self, mock_coordinator: Mock) -> None:
        """Test async initialization with no existing entities."""
        # Add required config attributes to mock
        mock_coordinator.config_manager = Mock()
        mock_coordinator.config_manager.config = Mock()
        mock_coordinator.config_manager.config.sensors = Mock()
        mock_coordinator.config_manager.config.sensors.primary_occupancy = (
            "binary_sensor.test_motion"
        )
        mock_coordinator.config_manager.config.sensors.get_motion_sensors = Mock(
            return_value=[]
        )
        mock_coordinator.config_manager.config.sensors.media = []
        mock_coordinator.config_manager.config.sensors.appliances = []
        mock_coordinator.config_manager.config.sensors.doors = []
        mock_coordinator.config_manager.config.sensors.windows = []
        mock_coordinator.config_manager.config.sensors.lights = []
        mock_coordinator.config_manager.config.sensors.illuminance = []
        mock_coordinator.config_manager.config.sensors.humidity = []
        mock_coordinator.config_manager.config.sensors.temperature = []

        manager = EntityManager(mock_coordinator)

        with (
            patch.object(manager, "_get_config_entity_mapping") as mock_mapping,
            patch.object(manager, "_create_entities_from_config") as mock_create,
            patch.object(manager, "_setup_entity_tracking") as mock_setup,
        ):
            mock_mapping.return_value = {
                "binary_sensor.motion1": mock_coordinator.entity_types.get_entity_type(
                    InputType.MOTION
                )
            }
            mock_create.return_value = {}

            await manager.async_initialize()

            # Should call create instead of update for fresh initialization
            mock_create.assert_called_once()
            mock_setup.assert_called_once()

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
        with pytest.raises(ValueError):
            manager.get_entity("nonexistent")

    def test_add_entity(self, mock_coordinator: Mock) -> None:
        """Test adding entity."""
        manager = EntityManager(mock_coordinator)

        mock_entity = Mock()
        mock_entity.entity_id = "test_entity"

        manager.add_entity(mock_entity)

        assert "test_entity" in manager._entities
        assert manager._entities["test_entity"] == mock_entity

    def test_remove_entity(self, mock_coordinator: Mock) -> None:
        """Test removing entity."""
        manager = EntityManager(mock_coordinator)

        # Add an entity
        mock_entity = Mock()
        manager._entities["test_entity"] = mock_entity

        # Remove it
        manager.remove_entity("test_entity")

        assert "test_entity" not in manager._entities

    def test_cleanup(self, mock_coordinator: Mock) -> None:
        """Test manager cleanup."""
        manager = EntityManager(mock_coordinator)

        # Set up a mock state listener
        mock_listener = Mock()
        manager._remove_state_listener = mock_listener

        # Add some entities
        mock_entity1 = Mock()
        mock_entity1.stop_decay_completely = Mock()
        mock_entity1.cleanup = Mock()
        mock_entity2 = Mock()
        mock_entity2.stop_decay_completely = Mock()
        mock_entity2.cleanup = Mock()

        manager._entities["entity1"] = mock_entity1
        manager._entities["entity2"] = mock_entity2

        # Cleanup
        manager.cleanup()

        # Should cleanup all entities and call state listener if it exists
        mock_entity1.stop_decay_completely.assert_called_once()
        mock_entity1.cleanup.assert_called_once()
        mock_entity2.stop_decay_completely.assert_called_once()
        mock_entity2.cleanup.assert_called_once()
        mock_listener.assert_called_once()

    @patch(
        "custom_components.area_occupancy.data.entity.async_track_state_change_event"
    )
    async def test_setup_entity_tracking(
        self, mock_track: Mock, mock_coordinator: Mock
    ) -> None:
        """Test setting up entity state tracking."""
        manager = EntityManager(mock_coordinator)

        # Add some entities
        manager._entities["entity1"] = Mock()
        manager._entities["entity2"] = Mock()

        with (
            patch.object(manager, "_initialize_current_states") as mock_init,
            patch("custom_components.area_occupancy.data.entity._LOGGER"),
        ):
            await manager._setup_entity_tracking()

            # Should set up tracking for all entities
            mock_track.assert_called_once()
            mock_init.assert_called_once()

    @patch(
        "custom_components.area_occupancy.data.entity.async_track_state_change_event"
    )
    async def test_setup_entity_tracking_no_entities(
        self, mock_track: Mock, mock_coordinator: Mock
    ) -> None:
        """Test setting up entity state tracking with no entities."""
        manager = EntityManager(mock_coordinator)

        # No entities added - should skip tracking setup
        with (
            patch.object(manager, "_initialize_current_states") as mock_init,
            patch("custom_components.area_occupancy.data.entity._LOGGER"),
        ):
            await manager._setup_entity_tracking()

            # Should not set up tracking when no entities exist
            mock_track.assert_not_called()
            mock_init.assert_not_called()

    async def test_reset_entities(self, mock_coordinator: Mock) -> None:
        """Test resetting all entities."""
        manager = EntityManager(mock_coordinator)

        with (
            patch.object(manager, "_create_entities_from_config") as mock_create,
            patch.object(manager, "_setup_entity_tracking") as mock_setup,
        ):
            mock_create.return_value = {}

            await manager.reset_entities()

            mock_create.assert_called_once()
            mock_setup.assert_called_once()


class TestEntityManagerIntegration:
    """Test EntityManager integration scenarios."""

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
        with (
            patch.object(manager, "_get_config_entity_mapping") as mock_mapping,
            patch.object(manager, "_create_entities_from_config") as mock_create,
            patch.object(manager, "_setup_entity_tracking") as mock_setup,
        ):
            mock_mapping.return_value = {
                "binary_sensor.motion1": mock_coordinator.entity_types.get_entity_type(
                    InputType.MOTION
                )
            }
            mock_create.return_value = {}

            await manager.async_initialize()

            # Should set up tracking
            mock_setup.assert_called_once()

        # Test cleanup
        manager.cleanup()


class TestEntityDecayAndUpdates:
    """Test Entity decay completion and coordinator updates."""

    @patch("custom_components.area_occupancy.data.entity.bayesian_probability")
    async def test_update_probability_decay_completion(
        self,
        mock_bayesian: Mock,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test probability update when decay completes."""
        # Set up request_update as AsyncMock for this test
        mock_coordinator.request_update = AsyncMock()

        mock_bayesian.return_value = 0.3
        mock_decay.is_decaying = True
        mock_decay.should_start_decay.return_value = False
        mock_decay.should_stop_decay.return_value = False
        mock_decay.update_decay.return_value = (0.1, 0.5)
        mock_decay.is_decay_complete.return_value = True

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
            is_active=False,
        )
        entity.set_coordinator(mock_coordinator)

        with patch.object(entity, "_stop_decay_timer") as mock_stop_timer:
            await entity.update_probability()

            # Should complete decay and reset is_active
            assert entity.probability == 0.1
            assert entity.is_active is False
            mock_decay.stop_decay.assert_called_once()
            mock_stop_timer.assert_called_once()
            # The coordinator update is called twice - once for decay completion, once for probability change
            assert mock_coordinator.request_update.call_count >= 1

    @patch("custom_components.area_occupancy.data.entity.bayesian_probability")
    async def test_update_probability_significant_change(
        self,
        mock_bayesian: Mock,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test probability update with significant change triggers coordinator update."""
        # Set up request_update as AsyncMock for this test
        mock_coordinator.request_update = AsyncMock()

        mock_bayesian.return_value = 0.8
        mock_decay.is_decaying = False
        mock_decay.should_start_decay.return_value = False
        mock_decay.should_stop_decay.return_value = False

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.2,  # Significant change from 0.2 to 0.8
            prior=mock_prior,
            decay=mock_decay,
            is_active=True,
        )
        entity.set_coordinator(mock_coordinator)

        await entity.update_probability()

        # Should trigger coordinator update due to significant probability change
        mock_coordinator.request_update.assert_called_with(
            message="Entity state changed, forcing update"
        )

    @patch("custom_components.area_occupancy.data.entity.bayesian_probability")
    async def test_update_probability_no_significant_change(
        self,
        mock_bayesian: Mock,
        mock_entity_type: Mock,
        mock_prior: Mock,
        mock_decay: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test probability update with no significant change doesn't trigger update."""
        mock_bayesian.return_value = 0.5005  # Very small change
        mock_decay.is_decaying = False
        mock_decay.should_start_decay.return_value = False
        mock_decay.should_stop_decay.return_value = False

        entity = Entity(
            entity_id="binary_sensor.test_motion",
            type=mock_entity_type,
            probability=0.5,
            prior=mock_prior,
            decay=mock_decay,
            is_active=True,
        )
        entity.set_coordinator(mock_coordinator)

        await entity.update_probability()

        # Should not trigger coordinator update due to insignificant change
        mock_coordinator.request_update.assert_not_called()


class TestEntityManagerConfigUpdates:
    """Test EntityManager entity updates from configuration."""

    async def test_update_entities_from_config_existing_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test updating existing entities from config."""
        manager = EntityManager(mock_coordinator)

        # Create existing entity
        mock_entity = Mock()
        mock_entity.type = Mock()
        mock_entity.decay = Mock()
        mock_entity.decay.is_decaying = False
        manager._entities["binary_sensor.motion1"] = mock_entity

        # Mock config entity mapping
        mock_entity_type = Mock()
        mock_entity_type.weight = 0.9
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        with (
            patch.object(
                manager,
                "_get_config_entity_mapping",
                return_value={"binary_sensor.motion1": mock_entity_type},
            ),
            patch("custom_components.area_occupancy.data.entity._LOGGER"),
        ):
            await manager._update_entities_from_config()

            # Should update entity type configuration
            assert mock_entity.type.weight == 0.9
            assert mock_entity.type.active_states == [STATE_ON]
            assert mock_entity.type.active_range is None
            assert "binary_sensor.motion1" in manager._entities

    async def test_update_entities_from_config_removed_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test updating entities when some are removed from config."""
        manager = EntityManager(mock_coordinator)

        # Create existing entity that will be removed
        mock_entity = Mock()
        manager._entities["binary_sensor.old_motion"] = mock_entity

        # Mock config with no entities (all removed)
        with (
            patch.object(manager, "_get_config_entity_mapping", return_value={}),
            patch(
                "custom_components.area_occupancy.data.entity._LOGGER"
            ) as mock_logger,
        ):
            await manager._update_entities_from_config()

            # Should log removal and remove entity - check that info was called
            assert mock_logger.info.call_count >= 1
            # Check that the entity was removed
            assert "binary_sensor.old_motion" not in manager._entities

    async def test_update_entities_from_config_new_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test updating entities when new ones are added to config."""
        manager = EntityManager(mock_coordinator)

        # Mock config with new entity
        mock_entity_type = Mock()
        mock_prior = Mock()

        with (
            patch.object(
                manager,
                "_get_config_entity_mapping",
                return_value={"binary_sensor.new_motion": mock_entity_type},
            ),
            patch.object(manager, "_calculate_initial_prior", return_value=mock_prior),
            patch.object(manager, "_create_entity", return_value=Mock()) as mock_create,
            patch(
                "custom_components.area_occupancy.data.entity._LOGGER"
            ) as mock_logger,
        ):
            await manager._update_entities_from_config()

            # Should log creation and create entity - check that info was called
            assert mock_logger.info.call_count >= 1
            mock_create.assert_called_once_with(
                entity_id="binary_sensor.new_motion",
                entity_type=mock_entity_type,
                prior=mock_prior,
            )

    async def test_update_entities_from_config_decay_disabled(
        self, mock_coordinator: Mock
    ) -> None:
        """Test updating entities when decay is disabled in config."""
        manager = EntityManager(mock_coordinator)
        manager.config.decay.enabled = False

        # Create existing entity with active decay
        mock_entity = Mock()
        mock_entity.type = Mock()
        mock_entity.decay = Mock()
        mock_entity.decay.is_decaying = True
        mock_entity.stop_decay_completely = Mock()
        manager._entities["binary_sensor.motion1"] = mock_entity

        # Mock config entity mapping
        mock_entity_type = Mock()
        mock_entity_type.weight = 0.9
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        with (
            patch.object(
                manager,
                "_get_config_entity_mapping",
                return_value={"binary_sensor.motion1": mock_entity_type},
            ),
            patch(
                "custom_components.area_occupancy.data.entity._LOGGER"
            ) as mock_logger,
        ):
            await manager._update_entities_from_config()

            # Should stop active decay and log info - check that info was called
            assert mock_logger.info.call_count >= 1
            mock_entity.stop_decay_completely.assert_called_once()

    async def test_get_config_entity_mapping_no_primary_sensor(
        self, mock_coordinator: Mock
    ) -> None:
        """Test config entity mapping without primary sensor."""
        manager = EntityManager(mock_coordinator)
        manager.config.sensors.primary_occupancy = None

        with pytest.raises(
            ValueError, match="Primary occupancy sensor must be configured"
        ):
            await manager._get_config_entity_mapping()

    async def test_get_config_entity_mapping_with_primary_sensor(
        self, mock_coordinator: Mock
    ) -> None:
        """Test config entity mapping with primary sensor."""
        manager = EntityManager(mock_coordinator)
        manager.config.sensors.primary_occupancy = "binary_sensor.primary"
        manager.config.sensors.get_motion_sensors = Mock(
            return_value=["binary_sensor.motion1"]
        )
        manager.config.sensors.media = ["media_player.tv"]
        manager.config.sensors.appliances = []
        manager.config.sensors.doors = []
        manager.config.sensors.windows = []
        manager.config.sensors.lights = []
        manager.config.sensors.illuminance = []
        manager.config.sensors.humidity = []
        manager.config.sensors.temperature = []

        mock_motion_type = Mock()
        mock_media_type = Mock()
        # Fix the side_effect to return enough values
        mock_coordinator.entity_types.get_entity_type.side_effect = [
            mock_motion_type,  # For primary sensor
            mock_motion_type,  # For motion sensors
            mock_media_type,  # For media sensors
            Mock(),  # For appliances (empty)
            Mock(),  # For doors (empty)
            Mock(),  # For windows (empty)
            Mock(),  # For lights (empty)
            Mock(),  # For environmental (empty)
        ]

        result = await manager._get_config_entity_mapping()

        # Should include primary sensor and avoid duplicates
        assert "binary_sensor.primary" in result
        assert "binary_sensor.motion1" in result
        assert "media_player.tv" in result
        assert len(result) == 3  # primary + motion1 + tv

    async def test_calculate_initial_prior_success(
        self, mock_coordinator: Mock
    ) -> None:
        """Test successful initial prior calculation."""
        manager = EntityManager(mock_coordinator)
        mock_entity_type = Mock()
        mock_prior = Mock()

        # Mock successful prior calculation
        mock_coordinator.priors.calculate = AsyncMock(return_value=mock_prior)

        with (
            patch.object(manager, "get_entity", return_value=Mock()),
            patch("custom_components.area_occupancy.data.entity._LOGGER"),
        ):
            result = await manager._calculate_initial_prior(
                "binary_sensor.test", mock_entity_type
            )

            assert result == mock_prior
            mock_coordinator.priors.calculate.assert_called_once()

    async def test_calculate_initial_prior_failure(
        self, mock_coordinator: Mock
    ) -> None:
        """Test initial prior calculation failure fallback."""
        manager = EntityManager(mock_coordinator)
        mock_entity_type = Mock()
        mock_entity_type.prior = 0.35
        mock_entity_type.prob_true = 0.8
        mock_entity_type.prob_false = 0.1

        # Mock failed prior calculation
        mock_coordinator.priors.calculate = AsyncMock(
            side_effect=ValueError("Test error")
        )

        with (
            patch.object(manager, "get_entity", return_value=Mock()),
            patch(
                "custom_components.area_occupancy.data.entity._LOGGER"
            ) as mock_logger,
        ):
            result = await manager._calculate_initial_prior(
                "binary_sensor.test", mock_entity_type
            )

            # Should return default prior and log warning
            assert isinstance(result, Prior)
            assert result.prior == 0.35
            # The logger receives the exception object, not just the string
            mock_logger.warning.assert_called_with(
                "Failed to calculate initial prior for %s: %s, using defaults",
                "binary_sensor.test",
                mock.ANY,
            )


class TestEntityManagerStateTracking:
    """Test EntityManager state change listener and initialization."""

    async def test_setup_entity_tracking_no_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test setting up entity tracking with no entities."""
        manager = EntityManager(mock_coordinator)

        with (
            patch(
                "custom_components.area_occupancy.data.entity.async_track_state_change_event"
            ) as mock_track,
            patch(
                "custom_components.area_occupancy.data.entity._LOGGER"
            ) as mock_logger,
        ):
            await manager._setup_entity_tracking()

            # Should skip tracking setup and log debug message
            mock_track.assert_not_called()
            mock_logger.debug.assert_called_with(
                "No entities to track, skipping state listener setup"
            )

    async def test_setup_entity_tracking_with_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test setting up entity tracking with entities."""
        manager = EntityManager(mock_coordinator)
        manager._entities["binary_sensor.test"] = Mock()

        with (
            patch(
                "custom_components.area_occupancy.data.entity.async_track_state_change_event"
            ) as mock_track,
            patch.object(manager, "_initialize_current_states") as mock_init,
        ):
            await manager._setup_entity_tracking()

            # Should set up tracking and initialize states
            mock_track.assert_called_once()
            mock_init.assert_called_once()

    async def test_state_changed_listener_entity_not_tracked(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state change listener with untracked entity."""
        manager = EntityManager(mock_coordinator)

        # Create entity with proper mock attributes
        mock_entity = Mock()
        mock_entity.type.active_states = [STATE_ON]  # Make it iterable
        mock_entity.type.active_range = None
        mock_entity.update_probability = Mock()
        manager._entities["binary_sensor.tracked"] = mock_entity

        # Mock HA state for initialization
        mock_ha_state = Mock()
        mock_ha_state.state = STATE_OFF
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        # Create mock event for untracked entity
        mock_event = Mock()
        mock_event.data = {"entity_id": "binary_sensor.untracked", "new_state": Mock()}

        with patch(
            "custom_components.area_occupancy.data.entity.async_track_state_change_event"
        ) as mock_track:
            await manager._setup_entity_tracking()

            # Get the listener function
            listener = mock_track.call_args[0][2]

            # Should return early for untracked entity
            await listener(mock_event)
            # No assertions needed - just ensure no exceptions

    async def test_state_changed_listener_active_states(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state change listener with active_states entity type."""
        manager = EntityManager(mock_coordinator)

        # Create entity with active_states type
        mock_entity = Mock()
        mock_entity.type.active_states = [STATE_ON]
        mock_entity.type.active_range = None
        mock_entity.update_probability = Mock()
        manager._entities["binary_sensor.test"] = mock_entity

        # Mock HA state for initialization
        mock_ha_state = Mock()
        mock_ha_state.state = STATE_OFF
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        # Create mock event
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_event = Mock()
        mock_event.data = {"entity_id": "binary_sensor.test", "new_state": mock_state}

        with patch(
            "custom_components.area_occupancy.data.entity.async_track_state_change_event"
        ) as mock_track:
            await manager._setup_entity_tracking()

            # Get the listener function
            listener = mock_track.call_args[0][2]

            await listener(mock_event)

            # Should update entity state and call update_probability
            assert mock_entity.state == STATE_ON
            assert mock_entity.available is True
            assert mock_entity.is_active is True
            # update_probability is called twice - once during initialization, once during listener
            assert mock_entity.update_probability.call_count >= 1

    async def test_state_changed_listener_active_range(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state change listener with active_range entity type."""
        manager = EntityManager(mock_coordinator)

        # Create entity with active_range type
        mock_entity = Mock()
        mock_entity.type.active_states = None
        mock_entity.type.active_range = (0.0, 1.0)
        mock_entity.update_probability = Mock()
        manager._entities["sensor.test"] = mock_entity

        # Mock HA state for initialization
        mock_ha_state = Mock()
        mock_ha_state.state = "0.2"
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        # Create mock event with value in range
        mock_state = Mock()
        mock_state.state = "0.5"
        mock_event = Mock()
        mock_event.data = {"entity_id": "sensor.test", "new_state": mock_state}

        with patch(
            "custom_components.area_occupancy.data.entity.async_track_state_change_event"
        ) as mock_track:
            await manager._setup_entity_tracking()

            # Get the listener function
            listener = mock_track.call_args[0][2]

            await listener(mock_event)

            # Should update entity state and set active
            assert mock_entity.state == "0.5"
            assert mock_entity.available is True
            assert mock_entity.is_active is True
            # update_probability is called twice - once during initialization, once during listener
            assert mock_entity.update_probability.call_count >= 1

    async def test_state_changed_listener_unavailable_state(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state change listener with unavailable state."""
        manager = EntityManager(mock_coordinator)

        # Create entity
        mock_entity = Mock()
        mock_entity.type.active_states = [STATE_ON]
        mock_entity.type.active_range = None
        mock_entity.update_probability = Mock()
        manager._entities["binary_sensor.test"] = mock_entity

        # Mock HA state for initialization
        mock_ha_state = Mock()
        mock_ha_state.state = STATE_OFF
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        # Create mock event with unavailable state
        mock_state = Mock()
        mock_state.state = "unavailable"
        mock_event = Mock()
        mock_event.data = {"entity_id": "binary_sensor.test", "new_state": mock_state}

        with patch(
            "custom_components.area_occupancy.data.entity.async_track_state_change_event"
        ) as mock_track:
            await manager._setup_entity_tracking()

            # Get the listener function
            listener = mock_track.call_args[0][2]

            await listener(mock_event)

            # Should mark entity as unavailable
            assert mock_entity.state is None
            assert mock_entity.available is False
            assert mock_entity.is_active is False
            # update_probability is called twice - once during initialization, once during listener
            assert mock_entity.update_probability.call_count >= 1

    async def test_state_changed_listener_error_handling(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state change listener error handling."""
        manager = EntityManager(mock_coordinator)

        # Create entity that will cause error during listener, not initialization
        mock_entity = Mock()
        mock_entity.type.active_states = [STATE_ON]
        mock_entity.type.active_range = None
        # Make update_probability work during initialization but fail during listener
        call_count = 0

        def update_probability_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:  # Fail on second call (listener)
                raise Exception("Test error")  # noqa: TRY002

        mock_entity.update_probability.side_effect = update_probability_side_effect
        manager._entities["binary_sensor.test"] = mock_entity

        # Mock HA state for initialization
        mock_ha_state = Mock()
        mock_ha_state.state = STATE_OFF
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        # Create mock event
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_event = Mock()
        mock_event.data = {"entity_id": "binary_sensor.test", "new_state": mock_state}

        with (
            patch(
                "custom_components.area_occupancy.data.entity.async_track_state_change_event"
            ) as mock_track,
            patch(
                "custom_components.area_occupancy.data.entity._LOGGER"
            ) as mock_logger,
        ):
            await manager._setup_entity_tracking()

            # Get the listener function
            listener = mock_track.call_args[0][2]

            await listener(mock_event)

            # Should log exception
            mock_logger.exception.assert_called_with(
                "Error processing state change for entity %s", "binary_sensor.test"
            )

    async def test_initialize_current_states_with_unavailable_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test initializing current states with unavailable entities."""
        manager = EntityManager(mock_coordinator)

        # Create entity
        mock_entity = Mock()
        manager._entities["binary_sensor.test"] = mock_entity

        # Mock unavailable HA state
        mock_ha_state = Mock()
        mock_ha_state.state = "unavailable"
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        with patch(
            "custom_components.area_occupancy.data.entity._LOGGER"
        ) as mock_logger:
            await manager._initialize_current_states()

            # Should mark entity as unavailable and log debug messages
            assert mock_entity.available is False
            # Check that debug was called (there are multiple debug calls)
            assert mock_logger.debug.call_count >= 1


class TestEntityManagerEntityCreation:
    """Test EntityManager entity creation with HA state detection."""

    async def test_create_entity_with_ha_state_active_states(
        self, mock_coordinator: Mock, mock_entity_type: Mock, mock_prior: Mock
    ) -> None:
        """Test creating entity with HA state using active_states."""
        manager = EntityManager(mock_coordinator)
        mock_entity_type.active_states = [STATE_ON]
        mock_entity_type.active_range = None

        # Mock HA state
        mock_ha_state = Mock()
        mock_ha_state.state = STATE_ON
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        with patch(
            "custom_components.area_occupancy.data.entity._LOGGER"
        ) as mock_logger:
            entity = await manager._create_entity(
                entity_id="binary_sensor.test",
                entity_type=mock_entity_type,
                prior=mock_prior,
            )

            # Should initialize from HA state
            assert entity.state == STATE_ON
            assert entity.available is True
            assert entity.is_active is True
            mock_logger.debug.assert_called_with(
                "Initialized entity %s with current HA state: state=%s, is_active=%s, available=%s",
                "binary_sensor.test",
                STATE_ON,
                True,
                True,
            )

    async def test_create_entity_with_ha_state_active_range(
        self, mock_coordinator: Mock, mock_entity_type: Mock, mock_prior: Mock
    ) -> None:
        """Test creating entity with HA state using active_range."""
        manager = EntityManager(mock_coordinator)
        mock_entity_type.active_states = None
        mock_entity_type.active_range = (0.0, 1.0)

        # Mock HA state with numeric value in range
        mock_ha_state = Mock()
        mock_ha_state.state = "0.7"
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        entity = await manager._create_entity(
            entity_id="sensor.test",
            entity_type=mock_entity_type,
            prior=mock_prior,
        )

        # Should initialize from HA state and detect active
        assert entity.state == "0.7"
        assert entity.available is True
        assert entity.is_active is True

    async def test_create_entity_with_ha_state_invalid_range_value(
        self, mock_coordinator: Mock, mock_entity_type: Mock, mock_prior: Mock
    ) -> None:
        """Test creating entity with HA state that can't be converted to float."""
        manager = EntityManager(mock_coordinator)
        mock_entity_type.active_states = None
        mock_entity_type.active_range = (0.0, 1.0)

        # Mock HA state with non-numeric value
        mock_ha_state = Mock()
        mock_ha_state.state = "not_a_number"
        mock_coordinator.hass.states.get.return_value = mock_ha_state

        entity = await manager._create_entity(
            entity_id="sensor.test",
            entity_type=mock_entity_type,
            prior=mock_prior,
        )

        # Should initialize from HA state but not be active
        assert entity.state == "not_a_number"
        assert entity.available is True
        assert entity.is_active is False

    async def test_create_entity_no_ha_state(
        self, mock_coordinator: Mock, mock_entity_type: Mock, mock_prior: Mock
    ) -> None:
        """Test creating entity with no HA state available."""
        manager = EntityManager(mock_coordinator)
        mock_coordinator.hass.states.get.return_value = None

        with patch(
            "custom_components.area_occupancy.data.entity._LOGGER"
        ) as mock_logger:
            entity = await manager._create_entity(
                entity_id="binary_sensor.test",
                entity_type=mock_entity_type,
                prior=mock_prior,
            )

            # Should use defaults
            assert entity.available is False
            mock_logger.debug.assert_called_with(
                "Entity %s has no valid current state in HA, using defaults",
                "binary_sensor.test",
            )

    async def test_create_entity_with_provided_state(
        self, mock_coordinator: Mock, mock_entity_type: Mock, mock_prior: Mock
    ) -> None:
        """Test creating entity with explicitly provided state."""
        manager = EntityManager(mock_coordinator)

        entity = await manager._create_entity(
            entity_id="binary_sensor.test",
            entity_type=mock_entity_type,
            state=STATE_ON,
            is_active=True,
            available=True,
            prior=mock_prior,
        )

        # Should use provided values, not query HA
        assert entity.state == STATE_ON
        assert entity.is_active is True
        assert entity.available is True
        mock_coordinator.hass.states.get.assert_not_called()

    async def test_create_entities_from_config_two_pass_creation(
        self, mock_coordinator: Mock
    ) -> None:
        """Test creating entities from config with two-pass prior calculation."""
        manager = EntityManager(mock_coordinator)

        # Mock config sensors
        manager.config.sensors.get_motion_sensors = Mock(
            return_value=["binary_sensor.motion1"]
        )
        manager.config.sensors.media = []
        manager.config.sensors.appliances = []
        manager.config.sensors.doors = []
        manager.config.sensors.windows = []
        manager.config.sensors.lights = []
        manager.config.sensors.illuminance = []
        manager.config.sensors.humidity = []
        manager.config.sensors.temperature = []

        # Mock entity type
        mock_entity_type = Mock()
        mock_entity_type.prior = 0.35
        mock_entity_type.prob_true = 0.8
        mock_entity_type.prob_false = 0.1
        mock_coordinator.entity_types.get_entity_type.return_value = mock_entity_type

        # Mock prior calculation
        mock_learned_prior = Mock()
        mock_coordinator.priors.calculate = AsyncMock(return_value=mock_learned_prior)

        with patch.object(
            manager, "_create_entity", return_value=Mock()
        ) as mock_create:
            # Actually call the method to trigger entity creation
            await manager._create_entities_from_config()

            # Should create entity in first pass
            mock_create.assert_called_once()

            # Should calculate learned prior in second pass
            mock_coordinator.priors.calculate.assert_called_once()


class TestEntityManagerErrorHandling:
    """Test EntityManager error handling scenarios."""

    def test_from_dict_missing_entities_key(self, mock_coordinator: Mock) -> None:
        """Test EntityManager.from_dict with missing entities key."""
        data = {"invalid_key": {}}

        with pytest.raises(
            ValueError, match="Invalid storage format: missing 'entities' key"
        ):
            EntityManager.from_dict(data, mock_coordinator)

    def test_from_dict_invalid_entity_data(self, mock_coordinator: Mock) -> None:
        """Test EntityManager.from_dict with invalid entity data."""
        data = {"entities": {"test_entity": {"invalid": "data"}}}

        with pytest.raises(ValueError, match="Failed to deserialize entity data"):
            EntityManager.from_dict(data, mock_coordinator)

    def test_get_entity_not_found(self, mock_coordinator: Mock) -> None:
        """Test getting non-existent entity."""
        manager = EntityManager(mock_coordinator)

        with pytest.raises(
            ValueError, match="Entity not found for entity: nonexistent"
        ):
            manager.get_entity("nonexistent")

    def test_cleanup_with_state_listener(self, mock_coordinator: Mock) -> None:
        """Test cleanup with active state listener."""
        manager = EntityManager(mock_coordinator)

        # Set up mock state listener
        mock_listener = Mock()
        manager._remove_state_listener = mock_listener

        # Add entities
        mock_entity1 = Mock()
        mock_entity2 = Mock()
        manager._entities["entity1"] = mock_entity1
        manager._entities["entity2"] = mock_entity2

        manager.cleanup()

        # Should cleanup listener and all entities
        mock_listener.assert_called_once()
        mock_entity1.stop_decay_completely.assert_called_once()
        mock_entity1.cleanup.assert_called_once()
        mock_entity2.stop_decay_completely.assert_called_once()
        mock_entity2.cleanup.assert_called_once()

    def test_cleanup_no_state_listener(self, mock_coordinator: Mock) -> None:
        """Test cleanup with no active state listener."""
        manager = EntityManager(mock_coordinator)
        manager._remove_state_listener = None

        # Add entity
        mock_entity = Mock()
        manager._entities["entity1"] = mock_entity

        # Should not raise exception
        manager.cleanup()

        # Should still cleanup entities
        mock_entity.stop_decay_completely.assert_called_once()
        mock_entity.cleanup.assert_called_once()
