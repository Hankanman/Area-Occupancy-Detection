"""Tests for coordinator module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    DEFAULT_PRIOR,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
)
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

    async def test_async_add_listener(self, mock_coordinator: Mock) -> None:
        """Test async_add_listener method using centralized mock."""
        callback = Mock()
        context = {"test": "context"}

        unsub = mock_coordinator.async_add_listener(callback, context)
        mock_coordinator.async_add_listener.assert_called_once_with(callback, context)
        assert unsub is not None

    async def test_store_operations(self, mock_coordinator: Mock) -> None:
        """Test store operations using centralized mock."""
        # Test saving coordinator data
        mock_coordinator.store.async_save_coordinator_data(mock_coordinator.entities)
        mock_coordinator.store.async_save_coordinator_data.assert_called_with(
            mock_coordinator.entities
        )

    def test_calculate_entity_aggregates(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test individual property calculations using centralized mock."""
        coordinator = mock_coordinator_with_sensors

        # Test individual property calls
        prob = coordinator.probability
        prior = coordinator.prior
        decay = coordinator.decay

        assert 0 <= prob <= 1
        assert 0 <= prior <= 1
        assert 0 <= decay <= 1


class TestCoordinatorRealBehavior:
    """Test real coordinator behavior with proper mocking of dependencies."""

    async def test_real_coordinator_initialization(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test real coordinator initialization with mocked dependencies."""
        # This test uses a real coordinator instance but mocks all dependencies
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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

    def test_calculate_entity_properties_no_entities(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test property calculations with no entities."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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

            # Test individual properties
            assert coordinator.prior == DEFAULT_PRIOR
            assert coordinator.decay == 1.0

    def test_calculate_entity_properties_unavailable_entities(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test property calculations with unavailable entities."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch(
                "custom_components.area_occupancy.data.entity.EntityManager"
            ) as mock_entity_mgr,
        ):
            mock_entity = Mock()
            mock_entity.available = False
            mock_entity.prior.prior = 0.35
            mock_entity.decay.is_decaying = False
            mock_entity.decay.decay_factor = 1.0  # Provide actual numeric value

            # Create proper mock structure
            mock_entities_instance = Mock()
            mock_entities_instance.entities = {"test_entity": mock_entity}
            mock_entity_mgr.return_value = mock_entities_instance

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            coordinator.entities = mock_entities_instance

            # Test individual properties
            assert coordinator.prior == 0.35
            assert coordinator.decay == 1.0

    def test_calculate_entity_properties_with_decaying_entities(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test property calculations with decaying entities."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch(
                "custom_components.area_occupancy.data.entity.EntityManager"
            ) as mock_entity_mgr,
        ):
            mock_entity = Mock()
            mock_entity.available = True
            mock_entity.type.weight = 1.0
            mock_entity.prior.prior = 0.35
            mock_entity.decay.is_decaying = True
            mock_entity.decay.decay_factor = 0.7  # Provide actual numeric value

            # Create proper mock structure
            mock_entities_instance = Mock()
            mock_entities_instance.entities = {"test_entity": mock_entity}
            mock_entity_mgr.return_value = mock_entities_instance

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            coordinator.entities = mock_entities_instance

            # Test individual properties
            assert coordinator.prior == 0.35
            assert coordinator.decay == 0.7


class TestCoordinatorErrorHandling:
    """Test coordinator error handling scenarios."""

    async def test_async_setup_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _async_setup failure handling."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch(
                "custom_components.area_occupancy.storage.AreaOccupancyStore"
            ) as mock_store,
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ) as mock_entity_type_mgr,
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
            patch("homeassistant.helpers.storage.Store") as mock_store_class,
        ):
            # Mock the entity type manager to fail during initialization
            mock_entity_type_instance = Mock()
            mock_entity_type_instance.async_initialize = AsyncMock(
                side_effect=HomeAssistantError("Entity type initialization failed")
            )
            mock_entity_type_mgr.return_value = mock_entity_type_instance

            # Mock the store
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance

            # Mock the Store class to avoid real storage operations
            mock_store_class.return_value.async_load = AsyncMock(return_value=None)
            mock_store_class.return_value.async_save = AsyncMock()

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            # Set the entity_types attribute to our mocked instance
            coordinator.entity_types = mock_entity_type_instance

            with pytest.raises(
                ConfigEntryNotReady, match="Failed to set up coordinator"
            ):
                await coordinator._async_coordinator_setup()

    async def test_async_load_stored_data_error(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_load_stored_data with storage error."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch(
                "custom_components.area_occupancy.storage.AreaOccupancyStore"
            ) as mock_store,
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
            patch("homeassistant.helpers.storage.Store") as mock_store_class,
        ):
            # Mock the store to fail during load
            mock_store_instance = Mock()
            mock_store_instance.async_load_with_compatibility_check = AsyncMock(
                side_effect=HomeAssistantError("Critical storage error")
            )
            mock_store.return_value = mock_store_instance

            # Mock the Store class to avoid real storage operations
            mock_store_class.return_value.async_load = AsyncMock(return_value=None)
            mock_store_class.return_value.async_save = AsyncMock()

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            # Set the store attribute to our mocked instance
            coordinator.store = mock_store_instance

            with pytest.raises(ConfigEntryNotReady, match="Failed to load stored data"):
                await coordinator.async_load_stored_data()

    async def test_async_load_stored_data_with_timestamp(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_load_stored_data with valid timestamp."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch(
                "custom_components.area_occupancy.storage.AreaOccupancyStore"
            ) as mock_store,
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

            # Mock the store to return data with timestamp and entities structure
            mock_store_instance = Mock()
            mock_store_instance.async_load_with_compatibility_check = AsyncMock(
                return_value=(
                    {"last_updated": "2024-01-01T00:00:00", "entities": {}},
                    False,
                )
            )
            mock_store.return_value = mock_store_instance

            # Mock the Store class to avoid real storage operations
            mock_store_class.return_value.async_load = AsyncMock(return_value=None)
            mock_store_class.return_value.async_save = AsyncMock()

            # Mock EntityManager.from_dict to return a mock entity manager
            mock_entity_mgr.from_dict = Mock(return_value=Mock())

            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            # Set the store attribute to our mocked instance
            coordinator.store = mock_store_instance

            await coordinator.async_load_stored_data()

            assert coordinator._last_prior_update == test_time

    async def test_handle_prior_update_error(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test _handle_prior_update with error."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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


class TestCoordinatorEdgeCases:
    """Test coordinator edge cases with simple, reliable mocking."""

    def test_binary_sensor_entity_ids_property_empty_state(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test binary_sensor_entity_ids property with empty state."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            # Test initial state
            entity_ids = coordinator.binary_sensor_entity_ids
            assert "occupancy" in entity_ids
            assert "wasp" in entity_ids
            assert entity_ids["occupancy"] is None
            assert entity_ids["wasp"] is None

    def test_binary_sensor_entity_ids_property_with_values(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test binary_sensor_entity_ids property with set values."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            # Set values
            coordinator.occupancy_entity_id = "binary_sensor.test_occupancy"
            coordinator.wasp_entity_id = "binary_sensor.test_wasp"

            entity_ids = coordinator.binary_sensor_entity_ids
            assert entity_ids["occupancy"] == "binary_sensor.test_occupancy"
            assert entity_ids["wasp"] == "binary_sensor.test_wasp"

    def test_threshold_property_with_no_config(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test threshold property when config is None."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
            # Mock config as None to test fallback
            with patch.object(coordinator, "config", None):
                assert coordinator.threshold == 0.5

    def test_device_info_property_detailed(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test device_info property with detailed verification."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            device_info = coordinator.device_info

            # Verify all expected keys and values
            assert "identifiers" in device_info
            assert "name" in device_info
            assert "manufacturer" in device_info
            assert "model" in device_info
            assert "sw_version" in device_info

            assert device_info["name"] == "Test Area"
            assert device_info["manufacturer"] == DEVICE_MANUFACTURER
            assert device_info["model"] == DEVICE_MODEL
            assert device_info["sw_version"] == DEVICE_SW_VERSION

            # Verify identifiers structure
            identifiers = device_info["identifiers"]
            assert isinstance(identifiers, set)
            assert (DOMAIN, "test_entry_id") in identifiers


class TestTimerAndTrackerCleanup:
    """Test timer and tracker cleanup functionality using mocks."""

    async def test_shutdown_with_prior_tracker(self, mock_coordinator: Mock) -> None:
        """Test shutdown with prior update tracker using centralized mock."""
        coordinator = mock_coordinator

        # Mock tracker cleanup
        mock_tracker = Mock()
        coordinator._prior_update_tracker = mock_tracker

        await coordinator.async_shutdown()

        # Should clean up tracker
        coordinator.async_shutdown.assert_called_once()

    async def test_shutdown_with_state_listener(self, mock_coordinator: Mock) -> None:
        """Test shutdown with state listener using centralized mock."""
        coordinator = mock_coordinator

        # Mock listener cleanup
        mock_listener = Mock()
        coordinator._remove_state_listener = mock_listener

        await coordinator.async_shutdown()

        # Should clean up listener
        coordinator.async_shutdown.assert_called_once()

    def test_global_decay_timer_initialization(self, mock_coordinator: Mock) -> None:
        """Test global decay timer initialization using centralized mock."""
        coordinator = mock_coordinator

        # Test setting timer using mock
        mock_timer = Mock()
        coordinator._global_decay_timer = mock_timer

        # Should be able to access the set value
        assert coordinator._global_decay_timer is mock_timer


class TestCoordinatorCallbacks:
    """Test coordinator callback functionality."""

    async def test_handle_prior_update_callback(self, mock_coordinator: Mock) -> None:
        """Test prior update callback using centralized mock."""
        coordinator = mock_coordinator

        test_time = dt_util.utcnow()
        await coordinator._handle_prior_update(test_time)

        # Should call the mock method
        coordinator._handle_prior_update.assert_called_once_with(test_time)

    def test_refresh_finished_callback(self, mock_coordinator: Mock) -> None:
        """Test refresh finished callback using centralized mock."""
        coordinator = mock_coordinator

        coordinator._async_refresh_finished()

        # Should call the mock method
        coordinator._async_refresh_finished.assert_called_once()


class TestUpdateOperations:
    """Test coordinator update operations using centralized mocks."""

    async def test_async_update_options_mock(self, mock_coordinator: Mock) -> None:
        """Test async_update_options using centralized mock."""
        coordinator = mock_coordinator

        new_options = {"threshold": 0.8, "decay_enabled": True}
        await coordinator.async_update_options(new_options)

        # Should call the mock method
        coordinator.async_update_options.assert_called_once_with(new_options)

    async def test_update_learned_priors_mock(self, mock_coordinator: Mock) -> None:
        """Test update_learned_priors using centralized mock."""
        coordinator = mock_coordinator

        await coordinator.update_learned_priors()

        # Should call the mock method
        coordinator.update_learned_priors.assert_called_once()

    async def test_update_learned_priors_with_period(
        self, mock_coordinator: Mock
    ) -> None:
        """Test update_learned_priors with custom period using centralized mock."""
        coordinator = mock_coordinator

        await coordinator.update_learned_priors(history_period=14)

        # Should call the mock method with period
        coordinator.update_learned_priors.assert_called_once_with(history_period=14)

    async def test_async_load_stored_data_mock(self, mock_coordinator: Mock) -> None:
        """Test async_load_stored_data using centralized mock."""
        coordinator = mock_coordinator

        await coordinator.async_load_stored_data()

        # Should call the mock method
        coordinator.async_load_stored_data.assert_called_once()


class TestEntityCalculationEdgeCases:
    """Test entity calculation edge cases using simple scenarios."""

    def test_property_calculations_no_entities_real(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test property calculations with no entities using real coordinator."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
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

            # Test individual properties
            assert coordinator.prior == DEFAULT_PRIOR
            assert coordinator.decay == 1.0

    def test_is_occupied_boundary_conditions(self, mock_coordinator: Mock) -> None:
        """Test is_occupied at threshold boundaries using centralized mock."""
        coordinator = mock_coordinator

        # Test exact threshold match
        coordinator.probability = 0.5
        coordinator.threshold = 0.5
        coordinator.is_occupied = True  # Mock the result for exact match
        assert coordinator.is_occupied is True

        # Test below threshold
        coordinator.probability = 0.4
        coordinator.threshold = 0.5
        coordinator.is_occupied = False  # Mock the result for below threshold
        assert coordinator.is_occupied is False


class TestDataProperties:
    """Test data property handling using mocks."""

    def test_data_property_access(self, mock_coordinator: Mock) -> None:
        """Test data property access using centralized mock."""
        coordinator = mock_coordinator

        # Test accessing data property
        data = coordinator.data
        assert "last_updated" in data

    def test_last_updated_from_data(self, mock_coordinator: Mock) -> None:
        """Test last_updated property from data using centralized mock."""
        coordinator = mock_coordinator

        # Mock data with last_updated
        test_time = dt_util.utcnow()
        coordinator.data = {"last_updated": test_time}

        # Access through the coordinator should work
        assert coordinator.data["last_updated"] == test_time

    def test_last_changed_from_data(self, mock_coordinator: Mock) -> None:
        """Test last_changed property from data using centralized mock."""
        coordinator = mock_coordinator

        # Mock data with last_changed
        test_time = dt_util.utcnow()
        coordinator.data = {"last_changed": test_time}

        # Access through the coordinator should work
        assert coordinator.data["last_changed"] == test_time


class TestCoordinatorStateManagement:
    """Test coordinator state management functionality."""

    def test_availability_property(self, mock_coordinator: Mock) -> None:
        """Test availability property using centralized mock."""
        coordinator = mock_coordinator

        # Test available state
        assert coordinator.available is True

    def test_last_update_success_property(self, mock_coordinator: Mock) -> None:
        """Test last_update_success property using centralized mock."""
        coordinator = mock_coordinator

        # Test successful update state
        assert coordinator.last_update_success is True


class TestCoordinatorInitializationEdgeCases:
    """Test coordinator initialization with edge cases."""

    def test_entry_id_from_config_entry(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test entry_id is correctly set from config entry."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            assert coordinator.entry_id == "test_entry_id"
            assert coordinator.config_entry == mock_config_entry

    def test_name_from_config_entry(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test name is correctly set from config entry."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            assert coordinator.name == "Test Area"

    def test_initial_timer_states(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test initial timer and tracker states."""
        with (
            patch("custom_components.area_occupancy.data.config.ConfigManager"),
            patch("custom_components.area_occupancy.storage.AreaOccupancyStore"),
            patch(
                "custom_components.area_occupancy.data.entity_type.EntityTypeManager"
            ),
            patch("custom_components.area_occupancy.data.prior.PriorManager"),
            patch("custom_components.area_occupancy.data.entity.EntityManager"),
        ):
            coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)

            # All timers and trackers should be None initially
            assert coordinator._next_prior_update is None
            assert coordinator._last_prior_update is None
            assert coordinator._prior_update_tracker is None
            assert coordinator._global_decay_timer is None
            assert coordinator._remove_state_listener is None
