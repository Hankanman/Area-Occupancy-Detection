"""Tests for coordinator module."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import DOMAIN
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util


class TestAreaOccupancyCoordinator:
    """Test AreaOccupancyCoordinator class."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.config = Mock()
        hass.config.path = Mock(return_value="/config")
        hass.states = Mock()
        hass.config_entries = Mock()
        hass.data = {}
        hass.bus = Mock()
        hass.services = Mock()
        hass.loop = Mock()
        hass.async_create_task = Mock()
        hass.async_add_executor_job = AsyncMock()
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry_id"
        entry.version = 1
        entry.minor_version = 1
        entry.domain = DOMAIN
        entry.data = {
            "name": "Test Area",
            "threshold": 50,
            "motion_sensors": ["binary_sensor.motion1"],
        }
        entry.options = {}
        entry.runtime_data = None
        entry.add_update_listener = Mock()
        entry.async_on_unload = Mock()
        return entry

    def test_initialization(self, mock_hass: Mock, mock_config_entry: Mock) -> None:
        """Test coordinator initialization."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        assert coordinator.hass == mock_hass
        assert coordinator.config_entry == mock_config_entry
        assert coordinator.entry_id == "test_entry_id"
        assert coordinator.name == "Test Area"
        assert coordinator.available is True

    def test_device_info_property(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test device_info property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        device_info = coordinator.device_info

        assert "identifiers" in device_info
        assert "name" in device_info
        assert "manufacturer" in device_info
        assert "model" in device_info
        assert device_info["name"] == "Test Area"

    def test_probability_property(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test probability property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock internal state
        coordinator._probability = 0.75

        assert coordinator.probability == 0.75

    def test_prior_property(self, mock_hass: Mock, mock_config_entry: Mock) -> None:
        """Test prior property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock internal state
        coordinator._prior = 0.35

        assert coordinator.prior == 0.35

    def test_decay_property(self, mock_hass: Mock, mock_config_entry: Mock) -> None:
        """Test decay property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock internal state
        coordinator._decay = 0.85

        assert coordinator.decay == 0.85

    def test_is_occupied_property(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test is_occupied property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock probability and threshold
        coordinator._probability = 0.7
        coordinator._threshold = 0.6

        assert coordinator.is_occupied is True

        # Test below threshold
        coordinator._probability = 0.5
        assert coordinator.is_occupied is False

    def test_threshold_property(self, mock_hass: Mock, mock_config_entry: Mock) -> None:
        """Test threshold property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock internal state
        coordinator._threshold = 0.6

        assert coordinator.threshold == 0.6

    def test_last_updated_property(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test last_updated property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        now = dt_util.utcnow()
        coordinator._last_updated = now

        assert coordinator.last_updated == now

    def test_last_changed_property(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test last_changed property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        now = dt_util.utcnow()
        coordinator._last_changed = now

        assert coordinator.last_changed == now

    def test_binary_sensor_entity_ids_property(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test binary_sensor_entity_ids property."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock internal state
        coordinator._occupancy_entity_id = "binary_sensor.area_occupancy"
        coordinator._wasp_entity_id = "binary_sensor.wasp_box"

        entity_ids = coordinator.binary_sensor_entity_ids

        assert entity_ids["occupancy"] == "binary_sensor.area_occupancy"
        assert entity_ids["wasp"] == "binary_sensor.wasp_box"

    def test_request_update_basic(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test basic request_update functionality."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        with patch.object(coordinator, "_async_immediate_update") as mock_update:
            coordinator.request_update()

            # Should schedule immediate update
            mock_hass.async_create_task.assert_called_once()

    def test_request_update_with_message(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test request_update with debug message."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        with patch.object(coordinator, "_async_immediate_update") as mock_update:
            coordinator.request_update(force=True, message="Test update")

            # Should still schedule update
            mock_hass.async_create_task.assert_called_once()

    async def test_async_immediate_update(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _async_immediate_update method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        with patch.object(coordinator, "async_request_refresh") as mock_refresh:
            await coordinator._async_immediate_update()

            mock_refresh.assert_called_once()

    async def test_async_setup(self, mock_hass: Mock, mock_config_entry: Mock) -> None:
        """Test _async_setup method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock all the managers and their initialization
        with patch(
            "custom_components.area_occupancy.data.config.ConfigManager"
        ) as mock_config_mgr:
            with patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ) as mock_entity_type_mgr:
                with patch(
                    "custom_components.area_occupancy.data.prior.PriorManager"
                ) as mock_prior_mgr:
                    with patch(
                        "custom_components.area_occupancy.data.entity.EntityManager"
                    ) as mock_entity_mgr:
                        with patch(
                            "custom_components.area_occupancy.storage.StorageManager"
                        ) as mock_storage_mgr:
                            # Mock async methods
                            mock_entity_type_mgr.return_value.async_initialize = (
                                AsyncMock()
                            )
                            mock_entity_mgr.return_value.async_initialize = AsyncMock()
                            mock_storage_mgr.return_value.async_initialize = AsyncMock()

                            await coordinator._async_setup()

                            # Verify managers were created and initialized
                            assert coordinator.config_manager is not None
                            assert coordinator.entity_types is not None
                            assert coordinator.prior_manager is not None
                            assert coordinator.entity_manager is not None
                            assert coordinator.storage_manager is not None

    async def test_async_shutdown(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_shutdown method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock managers
        coordinator.entity_manager = Mock()
        coordinator.entity_manager.cleanup = Mock()
        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_shutdown = AsyncMock()
        coordinator._prior_update_timer = Mock()

        await coordinator.async_shutdown()

        # Verify cleanup was called
        coordinator.entity_manager.cleanup.assert_called_once()
        coordinator.storage_manager.async_shutdown.assert_called_once()
        coordinator._prior_update_timer.assert_called_once()

    async def test_async_update_options(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_update_options method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock config manager
        coordinator.config_manager = Mock()
        coordinator.config_manager.update_config = AsyncMock()

        # Mock storage manager for saving
        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_save_instance_data = AsyncMock()
        coordinator.entity_manager = Mock()

        new_options = {"threshold": 70, "decay_enabled": False}

        with patch.object(coordinator, "async_refresh") as mock_refresh:
            await coordinator.async_update_options(new_options)

            coordinator.config_manager.update_config.assert_called_once_with(
                new_options
            )
            coordinator.storage_manager.async_save_instance_data.assert_called_once()
            mock_refresh.assert_called_once()

    async def test_async_update_threshold(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_update_threshold method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock config manager
        coordinator.config_manager = Mock()
        coordinator.config_manager.update_config = AsyncMock()

        # Mock storage manager for saving
        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_save_instance_data = AsyncMock()
        coordinator.entity_manager = Mock()

        with patch.object(coordinator, "async_refresh") as mock_refresh:
            await coordinator.async_update_threshold(0.75)

            # Should convert percentage and update config
            coordinator.config_manager.update_config.assert_called_once()
            call_args = coordinator.config_manager.update_config.call_args[0][0]
            assert call_args["threshold"] == 75  # Converted to percentage

            coordinator.storage_manager.async_save_instance_data.assert_called_once()
            mock_refresh.assert_called_once()

    async def test_async_load_stored_data_new_setup(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_load_stored_data for new setup."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock storage manager returning None (new setup)
        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_load_with_compatibility_check = AsyncMock(
            return_value=(None, False)
        )

        # Mock entity manager
        coordinator.entity_manager = Mock()
        coordinator.entity_manager.async_initialize = AsyncMock()

        await coordinator.async_load_stored_data()

        coordinator.entity_manager.async_initialize.assert_called_once()

    async def test_async_load_stored_data_existing_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_load_stored_data with existing data."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock storage manager returning existing data
        existing_data = {
            "entity_manager": {
                "entities": {
                    "test_entity": {"entity_id": "test_entity", "probability": 0.5}
                }
            }
        }
        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_load_with_compatibility_check = AsyncMock(
            return_value=(existing_data, False)
        )

        # Mock EntityManager.from_dict
        with patch(
            "custom_components.area_occupancy.data.entity.EntityManager.from_dict"
        ) as mock_from_dict:
            mock_entity_manager = Mock()
            mock_from_dict.return_value = mock_entity_manager

            await coordinator.async_load_stored_data()

            mock_from_dict.assert_called_once_with(
                existing_data["entity_manager"], coordinator
            )
            assert coordinator.entity_manager == mock_entity_manager

    async def test_update_learned_priors(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test update_learned_priors method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock prior manager
        coordinator.prior_manager = Mock()
        coordinator.prior_manager.update_all_entity_priors = AsyncMock(return_value=5)

        # Mock storage manager for saving
        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_save_instance_data = AsyncMock()
        coordinator.entity_manager = Mock()

        with patch.object(coordinator, "_schedule_next_prior_update") as mock_schedule:
            await coordinator.update_learned_priors()

            coordinator.prior_manager.update_all_entity_priors.assert_called_once()
            coordinator.storage_manager.async_save_instance_data.assert_called_once()
            mock_schedule.assert_called_once()

    async def test_schedule_next_prior_update(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _schedule_next_prior_update method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock config
        coordinator.config_manager = Mock()
        coordinator.config_manager.config = Mock()
        coordinator.config_manager.config.history = Mock()
        coordinator.config_manager.config.history.enabled = True

        with patch(
            "homeassistant.helpers.event.async_track_point_in_time"
        ) as mock_track:
            await coordinator._schedule_next_prior_update()

            mock_track.assert_called_once()

    async def test_handle_prior_update(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _handle_prior_update method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        coordinator._prior_update_timer = None

        with patch.object(coordinator, "update_learned_priors") as mock_update:
            await coordinator._handle_prior_update(dt_util.utcnow())

            mock_update.assert_called_once()

    def test_async_refresh_finished(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _async_refresh_finished callback."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock storage manager for saving
        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_save_instance_data = AsyncMock()
        coordinator.entity_manager = Mock()

        coordinator._async_refresh_finished()

        # Should schedule save
        mock_hass.async_create_task.assert_called_once()

    def test_async_set_updated_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_set_updated_data method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock internal state
        coordinator._probability = 0.5
        coordinator._last_changed = dt_util.utcnow() - timedelta(minutes=1)

        test_data = {
            "probability": 0.7,
            "prior": 0.35,
            "decay": 0.9,
            "threshold": 0.6,
            "is_occupied": True,
            "last_updated": dt_util.utcnow(),
        }

        with patch.object(
            coordinator, "async_update_listeners"
        ) as mock_update_listeners:
            coordinator.async_set_updated_data(test_data)

            # Should update internal state
            assert coordinator._probability == 0.7
            assert coordinator._prior == 0.35
            assert coordinator._decay == 0.9
            assert coordinator._threshold == 0.6

            # Should update last_changed when is_occupied changes
            assert coordinator._last_changed is not None

            mock_update_listeners.assert_called_once()

    def test_async_add_listener(self, mock_hass: Mock, mock_config_entry: Mock) -> None:
        """Test async_add_listener method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        callback = Mock()
        context = {"test": "context"}

        # Mock parent method
        with patch(
            "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_add_listener"
        ) as mock_parent:
            mock_parent.return_value = Mock()  # Mock unsub function

            unsub = coordinator.async_add_listener(callback, context)

            mock_parent.assert_called_once_with(callback, context)
            assert unsub is not None

    async def test_async_update_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _async_update_data method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock config
        coordinator.config_manager = Mock()
        coordinator.config_manager.config = Mock()
        coordinator.config_manager.config.threshold = 0.6

        # Mock entity aggregates calculation
        with patch.object(coordinator, "_calculate_entity_aggregates") as mock_calc:
            mock_calc.return_value = {
                "probability": 0.7,
                "prior": 0.35,
                "decay": 0.9,
            }

            result = await coordinator._async_update_data()

            assert result["probability"] == 0.7
            assert result["prior"] == 0.35
            assert result["decay"] == 0.9
            assert result["threshold"] == 0.6
            assert result["is_occupied"] is True  # 0.7 > 0.6
            assert "last_updated" in result

    async def test_async_save_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _async_save_data method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock storage manager and entity manager
        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_save_instance_data = AsyncMock()
        coordinator.entity_manager = Mock()

        await coordinator._async_save_data(force=True)

        coordinator.storage_manager.async_save_instance_data.assert_called_once_with(
            "test_entry_id", coordinator.entity_manager, force=True
        )

    def test_calculate_entity_aggregates(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _calculate_entity_aggregates method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock entity manager with test entities
        mock_entity1 = Mock()
        mock_entity1.probability = 0.7
        mock_entity1.type.weight = 0.8
        mock_entity1.available = True

        mock_entity2 = Mock()
        mock_entity2.probability = 0.5
        mock_entity2.type.weight = 0.6
        mock_entity2.available = True

        coordinator.entity_manager = Mock()
        coordinator.entity_manager.entities = {
            "entity1": mock_entity1,
            "entity2": mock_entity2,
        }

        result = coordinator._calculate_entity_aggregates()

        assert "probability" in result
        assert "prior" in result
        assert "decay" in result
        assert 0 <= result["probability"] <= 1
        assert 0 <= result["prior"] <= 1
        assert 0 <= result["decay"] <= 1


class TestCoordinatorIntegration:
    """Test coordinator integration scenarios."""

    @pytest.fixture
    def comprehensive_coordinator(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> AreaOccupancyCoordinator:
        """Create a coordinator with comprehensive mocked dependencies."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock all managers
        coordinator.config_manager = Mock()
        coordinator.config_manager.config = Mock()
        coordinator.config_manager.config.threshold = 0.6
        coordinator.config_manager.config.history = Mock()
        coordinator.config_manager.config.history.enabled = True
        coordinator.config_manager.update_config = AsyncMock()

        coordinator.entity_types = Mock()
        coordinator.prior_manager = Mock()
        coordinator.prior_manager.update_all_entity_priors = AsyncMock(return_value=3)

        coordinator.entity_manager = Mock()
        coordinator.entity_manager.entities = {}
        coordinator.entity_manager.async_initialize = AsyncMock()

        coordinator.storage_manager = Mock()
        coordinator.storage_manager.async_initialize = AsyncMock()
        coordinator.storage_manager.async_save_instance_data = AsyncMock()
        coordinator.storage_manager.async_shutdown = AsyncMock()

        return coordinator

    async def test_full_coordinator_lifecycle(
        self, comprehensive_coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test complete coordinator lifecycle."""
        coordinator = comprehensive_coordinator

        # Test setup
        await coordinator._async_setup()

        # Test first refresh
        with patch.object(coordinator, "_calculate_entity_aggregates") as mock_calc:
            mock_calc.return_value = {
                "probability": 0.7,
                "prior": 0.35,
                "decay": 0.9,
            }

            await coordinator.async_config_entry_first_refresh()

        # Test option updates
        await coordinator.async_update_options({"threshold": 70})

        # Test prior learning
        await coordinator.update_learned_priors()

        # Test shutdown
        await coordinator.async_shutdown()

        # Verify all components were properly called
        coordinator.storage_manager.async_initialize.assert_called_once()
        coordinator.storage_manager.async_shutdown.assert_called_once()

    async def test_error_handling_during_setup(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test error handling during coordinator setup."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock one of the managers to fail during initialization
        with patch(
            "custom_components.area_occupancy.data.entity.EntityManager"
        ) as mock_entity_mgr:
            mock_entity_mgr.return_value.async_initialize = AsyncMock(
                side_effect=Exception("Initialization failed")
            )

            # Should handle the error gracefully
            with pytest.raises(Exception, match="Initialization failed"):
                await coordinator._async_setup()

    async def test_data_update_with_real_calculations(
        self, comprehensive_coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test data update with realistic entity calculations."""
        coordinator = comprehensive_coordinator

        # Create realistic mock entities
        motion_entity = Mock()
        motion_entity.probability = 0.8
        motion_entity.type.weight = 0.85
        motion_entity.type.prior = 0.35
        motion_entity.decay.decay_factor = 0.9
        motion_entity.available = True

        media_entity = Mock()
        media_entity.probability = 0.6
        media_entity.type.weight = 0.7
        media_entity.type.prior = 0.15
        media_entity.decay.decay_factor = 1.0
        media_entity.available = True

        coordinator.entity_manager.entities = {
            "binary_sensor.motion1": motion_entity,
            "media_player.tv": media_entity,
        }

        # Test data update
        result = await coordinator._async_update_data()

        # Verify realistic results
        assert 0 <= result["probability"] <= 1
        assert 0 <= result["prior"] <= 1
        assert 0 <= result["decay"] <= 1
        assert result["threshold"] == 0.6
        assert isinstance(result["is_occupied"], bool)
        assert "last_updated" in result

    def test_threshold_boundary_conditions(
        self, comprehensive_coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test is_occupied calculation at threshold boundaries."""
        coordinator = comprehensive_coordinator

        # Test exactly at threshold
        coordinator._probability = 0.6
        coordinator._threshold = 0.6
        assert coordinator.is_occupied is False  # Should be False for equal values

        # Test just above threshold
        coordinator._probability = 0.601
        assert coordinator.is_occupied is True

        # Test just below threshold
        coordinator._probability = 0.599
        assert coordinator.is_occupied is False

        # Test edge cases
        coordinator._probability = 0.0
        coordinator._threshold = 0.0
        assert coordinator.is_occupied is False

        coordinator._probability = 1.0
        coordinator._threshold = 1.0
        assert coordinator.is_occupied is False

    def test_listener_management(
        self, comprehensive_coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test listener management functionality."""
        coordinator = comprehensive_coordinator

        # Mock listener functions
        listener1 = Mock()
        listener2 = Mock()

        # Mock parent class methods
        with patch(
            "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_add_listener"
        ) as mock_add:
            unsub1 = Mock()
            unsub2 = Mock()
            mock_add.side_effect = [unsub1, unsub2]

            # Add listeners
            result1 = coordinator.async_add_listener(listener1)
            result2 = coordinator.async_add_listener(listener2, {"context": "test"})

            assert result1 == unsub1
            assert result2 == unsub2
            assert mock_add.call_count == 2
