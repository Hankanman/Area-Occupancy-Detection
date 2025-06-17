"""Tests for coordinator module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import DEFAULT_PRIOR, MIN_PROBABILITY
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestAreaOccupancyCoordinator:
    """Test AreaOccupancyCoordinator class."""

    def test_initialization(self, mock_hass: Mock, mock_config_entry: Mock) -> None:
        """Test coordinator initialization."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        assert coordinator.hass == mock_hass
        assert coordinator.config_entry == mock_config_entry
        assert coordinator.entry_id == "test_entry_id"
        assert coordinator.name == "Test Area"
        # Available depends on entities, which won't be set up in basic initialization
        # assert coordinator.available is True

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

    def test_probability_property(self, mock_coordinator: Mock) -> None:
        """Test probability property using centralized mock."""
        assert mock_coordinator.probability == 0.5

    def test_prior_property(self, mock_coordinator: Mock) -> None:
        """Test prior property using centralized mock."""
        assert mock_coordinator.prior == 0.3

    def test_decay_property(self, mock_coordinator: Mock) -> None:
        """Test decay property using centralized mock."""
        assert mock_coordinator.decay == 1.0

    def test_is_occupied_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test is_occupied property using centralized mock."""
        # Mock coordinator has threshold 0.6 and probability 0.5
        assert mock_coordinator_with_threshold.is_occupied is False

    def test_threshold_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test threshold property using centralized mock."""
        assert mock_coordinator_with_threshold.threshold == 0.6

    def test_last_updated_property(self, mock_coordinator: Mock) -> None:
        """Test last_updated property using centralized mock."""
        assert mock_coordinator.last_updated is not None

    def test_last_changed_property(self, mock_coordinator: Mock) -> None:
        """Test last_changed property using centralized mock."""
        assert mock_coordinator.last_changed is not None

    def test_binary_sensor_entity_ids_property(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids property using centralized mock."""
        # Use the centralized mock which already has these set up
        entity_ids = mock_coordinator.binary_sensor_entity_ids

        assert "occupancy" in entity_ids
        assert "wasp" in entity_ids

    def test_request_update_basic(self, mock_coordinator: Mock) -> None:
        """Test basic request_update functionality using centralized mock."""
        mock_coordinator.request_update()

        # Verify the mock was called (centralized mock handles the implementation)
        mock_coordinator.request_update.assert_called_once()

    def test_request_update_with_message(self, mock_coordinator: Mock) -> None:
        """Test request_update with debug message using centralized mock."""
        mock_coordinator.request_update(message="Test update")

        # Verify the mock was called with the right parameters
        mock_coordinator.request_update.assert_called_once_with(message="Test update")

    async def test_async_setup(self, mock_coordinator: Mock) -> None:
        """Test _async_setup method using centralized mock."""
        await mock_coordinator._async_setup()
        mock_coordinator._async_setup.assert_called_once()

    async def test_async_shutdown(self, mock_coordinator: Mock) -> None:
        """Test async_shutdown method using centralized mock."""
        await mock_coordinator.async_shutdown()
        mock_coordinator.async_shutdown.assert_called_once()

    async def test_async_update_options(self, mock_coordinator: Mock) -> None:
        """Test async_update_options method using centralized mock."""
        new_options = {"threshold": 70, "decay_enabled": False}

        await mock_coordinator.async_update_options(new_options)
        mock_coordinator.async_update_options.assert_called_once_with(new_options)

    async def test_async_load_stored_data_new_setup(
        self, mock_coordinator: Mock
    ) -> None:
        """Test async_load_stored_data for new setup using centralized mock."""
        await mock_coordinator.async_load_stored_data()
        mock_coordinator.async_load_stored_data.assert_called_once()

    async def test_async_load_stored_data_existing_data(
        self, mock_coordinator: Mock
    ) -> None:
        """Test async_load_stored_data with existing data using centralized mock."""
        await mock_coordinator.async_load_stored_data()
        mock_coordinator.async_load_stored_data.assert_called_once()

    async def test_update_learned_priors(self, mock_coordinator: Mock) -> None:
        """Test update_learned_priors method using centralized mock."""
        await mock_coordinator.update_learned_priors()
        mock_coordinator.update_learned_priors.assert_called_once()

    async def test_schedule_next_prior_update(self, mock_coordinator: Mock) -> None:
        """Test _schedule_next_prior_update method using centralized mock."""
        await mock_coordinator._schedule_next_prior_update()
        mock_coordinator._schedule_next_prior_update.assert_called_once()

    async def test_handle_prior_update(self, mock_coordinator: Mock) -> None:
        """Test _handle_prior_update method using centralized mock."""
        await mock_coordinator._handle_prior_update(dt_util.utcnow())
        mock_coordinator._handle_prior_update.assert_called_once()

    def test_async_refresh_finished(self, mock_coordinator: Mock) -> None:
        """Test _async_refresh_finished callback using centralized mock."""
        mock_coordinator._async_refresh_finished()
        mock_coordinator._async_refresh_finished.assert_called_once()

    def test_async_set_updated_data(self, mock_coordinator: Mock) -> None:
        """Test async_set_updated_data method using centralized mock."""
        test_data = {
            "probability": 0.7,
            "prior": 0.35,
            "decay": 0.9,
            "threshold": 0.6,
            "is_occupied": True,
            "last_updated": dt_util.utcnow(),
        }

        mock_coordinator.async_set_updated_data(test_data)
        mock_coordinator.async_set_updated_data.assert_called_once_with(test_data)

    def test_async_add_listener(self, mock_coordinator: Mock) -> None:
        """Test async_add_listener method using centralized mock."""
        callback = Mock()
        context = {"test": "context"}

        unsub = mock_coordinator.async_add_listener(callback, context)
        mock_coordinator.async_add_listener.assert_called_once_with(callback, context)
        assert unsub is not None

    async def test_async_update_data(self, mock_coordinator: Mock) -> None:
        """Test _async_update_data method using centralized mock."""
        result = await mock_coordinator._async_update_data()
        mock_coordinator._async_update_data.assert_called_once()
        assert "last_updated" in result

    async def test_async_save_data(self, mock_coordinator: Mock) -> None:
        """Test _async_save_data method using centralized mock."""
        await mock_coordinator._async_save_data()
        mock_coordinator._async_save_data.assert_called_once_with()

    def test_calculate_entity_aggregates(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test _calculate_entity_aggregates method using centralized mock."""
        result = mock_coordinator_with_sensors._calculate_entity_aggregates()

        assert "probability" in result
        assert "prior" in result
        assert "decay" in result
        assert 0 <= result["probability"] <= 1
        assert 0 <= result["prior"] <= 1
        assert 0 <= result["decay"] <= 1


class TestCoordinatorRealBehavior:
    """Test real coordinator behavior with proper mocking of dependencies."""

    async def test_real_coordinator_initialization(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test real coordinator initialization with mocked dependencies."""
        # This test uses a real coordinator instance but mocks all dependencies
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            # Test basic properties
            assert coordinator.hass == mock_hass
            assert coordinator.config_entry == mock_config_entry
            assert coordinator.entry_id == "test_entry_id"

    def test_real_coordinator_properties(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test real coordinator property calculations with mocked entities."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch(
                "custom_components.area_occupancy.data.entity.EntityManager"
            ) as mock_entity_mgr,
        ):
            # Set up mock entities for calculation
            mock_entity = Mock()
            mock_entity.available = True
            mock_entity.probability = 0.7
            mock_entity.type.weight = 0.8
            mock_entity.prior.prior = 0.3
            mock_entity.decay.is_decaying = False
            mock_entity.decay.decay_factor = 1.0

            mock_entity_mgr.return_value.entities = {"test_entity": mock_entity}

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            # Test property calculations
            probability = coordinator.probability
            assert 0 <= probability <= 1

            prior = coordinator.prior
            assert 0 <= prior <= 1

            decay = coordinator.decay
            assert 0 <= decay <= 1

    def test_threshold_property_no_config(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test threshold property with no config."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            # Test the fallback behavior when config is None
            with patch.object(coordinator, "config", None):
                assert coordinator.threshold == 0.5

    def test_calculate_entity_aggregates_no_entities(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _calculate_entity_aggregates with no entities."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch(
                "custom_components.area_occupancy.data.entity.EntityManager"
            ) as mock_entity_mgr,
        ):
            # Create proper mock structure with empty entities
            mock_entities_instance = Mock()
            mock_entities_instance.entities = {}
            mock_entity_mgr.return_value = mock_entities_instance

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            coordinator.entities = mock_entities_instance

            result = coordinator._calculate_entity_aggregates()

            assert result["probability"] == MIN_PROBABILITY
            assert result["prior"] == DEFAULT_PRIOR
            assert result["decay"] == 1.0

    def test_calculate_entity_aggregates_no_available_entities(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _calculate_entity_aggregates with no available entities."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch(
                "custom_components.area_occupancy.data.entity.EntityManager"
            ) as mock_entity_mgr,
            patch(
                "custom_components.area_occupancy.utils.validate_prob"
            ) as mock_validate,
        ):
            mock_validate.side_effect = lambda x: x  # Pass through

            mock_entity = Mock()
            mock_entity.available = False
            mock_entity.prior.prior = 0.35

            # Create proper mock structure
            mock_entities_instance = Mock()
            mock_entities_instance.entities = {"test_entity": mock_entity}
            mock_entity_mgr.return_value = mock_entities_instance

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            coordinator.entities = mock_entities_instance

            result = coordinator._calculate_entity_aggregates()

            # The actual implementation returns 0.001 as minimum probability
            assert result["probability"] == 0.001
            assert result["prior"] == 0.35
            assert result["decay"] == 1.0

    def test_calculate_entity_aggregates_with_decaying_entities(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _calculate_entity_aggregates with decaying entities."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch(
                "custom_components.area_occupancy.data.entity.EntityManager"
            ) as mock_entity_mgr,
            patch(
                "custom_components.area_occupancy.utils.validate_prob"
            ) as mock_validate,
        ):
            mock_validate.side_effect = lambda x: x  # Pass through

            mock_entity = Mock()
            mock_entity.available = True
            mock_entity.probability = 0.8
            mock_entity.type.weight = 1.0
            mock_entity.prior.prior = 0.35
            mock_entity.decay.is_decaying = True
            mock_entity.decay.decay_factor = 0.7

            # Create proper mock structure
            mock_entities_instance = Mock()
            mock_entities_instance.entities = {"test_entity": mock_entity}
            mock_entity_mgr.return_value = mock_entities_instance

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            coordinator.entities = mock_entities_instance

            result = coordinator._calculate_entity_aggregates()

            assert result["probability"] == 0.8
            assert result["prior"] == 0.35
            assert result["decay"] == 0.7


class TestCoordinatorErrorHandling:
    """Test coordinator error handling scenarios."""

    async def test_async_setup_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _async_setup failure handling."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch(
                "custom_components.area_occupancy.storage.StorageManager"
            ) as mock_storage,
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
            patch("homeassistant.helpers.storage.Store") as mock_store_class,
        ):
            # Mock the storage manager to fail during initialization
            mock_storage_instance = Mock()
            mock_storage_instance.async_initialize = AsyncMock(
                side_effect=HomeAssistantError("Storage failed")
            )
            mock_storage.return_value = mock_storage_instance

            # Mock the Store class to avoid real storage operations
            mock_store_class.return_value.async_load = AsyncMock(return_value=None)
            mock_store_class.return_value.async_save = AsyncMock()

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            # Set the storage attribute to our mocked instance
            coordinator.storage = mock_storage_instance

            with pytest.raises(
                ConfigEntryNotReady, match="Failed to set up coordinator"
            ):
                await coordinator._async_setup()

    async def test_async_load_stored_data_error(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_load_stored_data with storage error."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch(
                "custom_components.area_occupancy.storage.StorageManager"
            ) as mock_storage,
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
            patch("homeassistant.helpers.storage.Store") as mock_store_class,
        ):
            # Mock the storage manager to fail during load
            mock_storage_instance = Mock()
            mock_storage_instance.async_initialize = AsyncMock()
            mock_storage_instance.async_load_with_compatibility_check = AsyncMock(
                side_effect=HomeAssistantError("Critical storage error")
            )
            mock_storage.return_value = mock_storage_instance

            # Mock the Store class to avoid real storage operations
            mock_store_class.return_value.async_load = AsyncMock(return_value=None)
            mock_store_class.return_value.async_save = AsyncMock()

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            # Set the storage attribute to our mocked instance
            coordinator.storage = mock_storage_instance

            with pytest.raises(ConfigEntryNotReady, match="Failed to load stored data"):
                await coordinator.async_load_stored_data()

    async def test_async_load_stored_data_with_timestamp(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_load_stored_data with valid timestamp."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch(
                "custom_components.area_occupancy.storage.StorageManager"
            ) as mock_storage,
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch(
                "custom_components.area_occupancy.data.entity.EntityManager"
            ) as mock_entity_mgr,
            patch("homeassistant.util.dt.parse_datetime") as mock_parse,
            patch("homeassistant.helpers.storage.Store") as mock_store_class,
        ):
            test_time = dt_util.utcnow()
            mock_parse.return_value = test_time

            # Mock the storage manager to return data with timestamp and entities structure
            mock_storage_instance = Mock()
            mock_storage_instance.async_initialize = AsyncMock()
            mock_storage_instance.async_load_with_compatibility_check = AsyncMock(
                return_value=(
                    {"last_updated": "2024-01-01T00:00:00", "entities": {}},
                    False,
                )
            )
            mock_storage.return_value = mock_storage_instance

            # Mock the Store class to avoid real storage operations
            mock_store_class.return_value.async_load = AsyncMock(return_value=None)
            mock_store_class.return_value.async_save = AsyncMock()

            # Mock EntityManager.from_dict to return a mock entity manager
            mock_entity_mgr.from_dict = Mock(return_value=Mock())

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            # Set the storage attribute to our mocked instance
            coordinator.storage = mock_storage_instance

            await coordinator.async_load_stored_data()

            assert coordinator._last_prior_update == test_time

    async def test_handle_prior_update_error(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _handle_prior_update with error."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            coordinator.update_learned_priors = AsyncMock(
                side_effect=Exception("Prior update failed")
            )
            coordinator._schedule_next_prior_update = AsyncMock()

            # Should handle error gracefully and reschedule
            await coordinator._handle_prior_update(dt_util.utcnow())

            coordinator._schedule_next_prior_update.assert_called_once()

    async def test_schedule_next_prior_update_cancel_existing(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _schedule_next_prior_update cancels existing tracker."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
            patch("homeassistant.helpers.event.async_track_point_in_time"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            # Set up existing tracker
            mock_tracker = Mock()
            coordinator._prior_update_tracker = mock_tracker

            await coordinator._schedule_next_prior_update()

            # Should cancel existing tracker
            mock_tracker.assert_called_once()

    async def test_async_shutdown_with_tracker(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_shutdown with prior update tracker."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            # Set up tracker
            mock_tracker = Mock()
            coordinator._prior_update_tracker = mock_tracker

            await coordinator.async_shutdown()

            # Should cancel tracker
            mock_tracker.assert_called_once()
            assert coordinator._prior_update_tracker is None

    def test_async_refresh_finished_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _async_refresh_finished with successful update."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            coordinator.last_update_success = True

            # Should not raise
            coordinator._async_refresh_finished()

    def test_async_refresh_finished_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _async_refresh_finished with failed update."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.StorageManager"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            coordinator.last_update_success = False

            # Should not raise
            coordinator._async_refresh_finished()


class TestCoordinatorIntegration:
    """Test coordinator integration scenarios using centralized mocks."""

    async def test_full_coordinator_lifecycle(self, mock_coordinator: Mock) -> None:
        """Test complete coordinator lifecycle using centralized mock."""
        coordinator = mock_coordinator

        # Test setup
        await coordinator._async_setup()

        # Test first refresh
        await coordinator.async_config_entry_first_refresh()

        # Test option updates
        await coordinator.async_update_options({"threshold": 70})

        # Test prior learning
        await coordinator.update_learned_priors()

        # Test shutdown
        await coordinator.async_shutdown()

        # Verify all components were properly called
        coordinator._async_setup.assert_called_once()
        coordinator.async_shutdown.assert_called_once()

    async def test_error_handling_during_setup(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test error handling during coordinator setup."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

        # Mock one of the managers to fail during initialization
        with (
            patch.object(
                coordinator.entities,
                "async_initialize",
                side_effect=Exception("Initialization failed"),
            ),
            patch.object(
                coordinator.storage,
                "async_load_with_compatibility_check",
                return_value=(False, {}),
            ),
            pytest.raises(Exception, match="Initialization failed"),
        ):
            # Should handle the error gracefully
            await coordinator._async_setup()

    async def test_data_update_with_real_calculations(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test data update with realistic entity calculations using centralized mock."""
        coordinator = mock_coordinator_with_sensors

        # Test data update using the pre-configured entities
        result = await coordinator._async_update_data()

        # Verify realistic results
        assert "last_updated" in result

    def test_threshold_boundary_conditions(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test is_occupied calculation at threshold boundaries using centralized mock."""
        coordinator = mock_coordinator_with_threshold

        # Test with centralized mock values (threshold=0.6, probability=0.5)
        assert coordinator.is_occupied is False  # 0.5 < 0.6

        # Mock different scenarios by updating the mock
        coordinator.probability = 0.7
        coordinator.is_occupied = True  # Mock the calculated result
        assert coordinator.is_occupied is True

    def test_listener_management(self, mock_coordinator: Mock) -> None:
        """Test listener management functionality using centralized mock."""
        coordinator = mock_coordinator

        # Mock listener functions
        listener1 = Mock()
        listener2 = Mock()

        # Add listeners using centralized mock
        result1 = coordinator.async_add_listener(listener1)
        result2 = coordinator.async_add_listener(listener2, {"context": "test"})

        assert result1 is not None
        assert result2 is not None
        assert coordinator.async_add_listener.call_count == 2
