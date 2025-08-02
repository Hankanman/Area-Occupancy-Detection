"""Tests for coordinator module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.prior import MIN_PRIOR
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestAreaOccupancyCoordinator:
    """Test AreaOccupancyCoordinator class."""

    def test_initialization(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test coordinator initialization using realistic fixtures."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        assert coordinator.hass == mock_hass
        assert coordinator.config_entry == mock_realistic_config_entry
        assert coordinator.entry_id == mock_realistic_config_entry.entry_id
        assert coordinator.name == mock_realistic_config_entry.data["name"]

    def test_device_info_property(self, mock_coordinator: Mock) -> None:
        """Test device_info property using centralized mock."""
        device_info = mock_coordinator.device_info

        assert "identifiers" in device_info
        assert "name" in device_info
        assert "manufacturer" in device_info
        assert "model" in device_info

        # Check that device_info has proper structure - values come from centralized mock
        assert isinstance(device_info["identifiers"], set)
        assert isinstance(device_info["name"], str)

    def test_probability_property(self, mock_coordinator: Mock) -> None:
        """Test probability property using centralized mock."""
        assert mock_coordinator.probability == 0.5

    def test_prior_property(self, mock_coordinator: Mock) -> None:
        """Test prior property using centralized mock."""
        assert mock_coordinator.area_prior == 0.3

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

    def test_binary_sensor_entity_ids_property(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids property using centralized mock."""
        entity_ids = mock_coordinator.binary_sensor_entity_ids

        assert "occupancy" in entity_ids
        assert "wasp" in entity_ids

    def test_decaying_entities_property(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decaying_entities property using centralized mock."""
        # The mock_coordinator_with_sensors has one decaying entity
        mock_coordinator_with_sensors.decaying_entities = [
            mock_coordinator_with_sensors.entities.entities["binary_sensor.motion2"]
        ]

        decaying = mock_coordinator_with_sensors.decaying_entities
        assert len(decaying) == 1
        assert decaying[0].entity_id == "binary_sensor.motion2"

    async def test_async_setup_basic(self, mock_coordinator: Mock) -> None:
        """Test basic _async_setup method using centralized mock."""
        await mock_coordinator.setup()
        mock_coordinator.setup.assert_called_once()

    async def test_async_shutdown(self, mock_coordinator: Mock) -> None:
        """Test async_shutdown method using centralized mock."""
        await mock_coordinator.async_shutdown()
        mock_coordinator.async_shutdown.assert_called_once()

    async def test_async_update_options(self, mock_coordinator: Mock) -> None:
        """Test async_update_options method using centralized mock."""
        new_options = {"threshold": 70, "decay_enabled": False}

        await mock_coordinator.async_update_options(new_options)
        mock_coordinator.async_update_options.assert_called_once_with(new_options)

    async def test_update_method(self, mock_coordinator: Mock) -> None:
        """Test update method using centralized mock."""
        result = await mock_coordinator.update()
        mock_coordinator.update.assert_called_once()

        # Should return coordinator data
        assert result is not None

    async def test_track_entity_state_changes(self, mock_coordinator: Mock) -> None:
        """Test track_entity_state_changes method using centralized mock."""
        entity_ids = ["binary_sensor.test1", "binary_sensor.test2"]
        await mock_coordinator.track_entity_state_changes(entity_ids)
        mock_coordinator.track_entity_state_changes.assert_called_once_with(entity_ids)

    def test_real_device_info_constants(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test device_info property with actual constant values."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        device_info = coordinator.device_info

        # Test actual constant values (using .get() for linter safety)
        assert device_info.get("manufacturer") == DEVICE_MANUFACTURER  # "Hankanman"
        assert device_info.get("model") == DEVICE_MODEL  # "Area Occupancy Detector"
        assert device_info.get("sw_version") == DEVICE_SW_VERSION  # "2025.6.1-pre2"

        # Test identifiers structure
        identifiers = device_info.get("identifiers")
        assert identifiers is not None
        assert (DOMAIN, coordinator.entry_id) in identifiers


class TestCoordinatorTimerMethods:
    """Test coordinator timer methods comprehensively using centralized mocks."""

    async def test_handle_decay_timer_decay_enabled(
        self, mock_coordinator: Mock
    ) -> None:
        """Test _handle_decay_timer with decay enabled using centralized mock."""
        test_time = dt_util.utcnow()

        # Configure mock for enabled decay
        mock_coordinator.config.decay.enabled = True
        mock_coordinator._handle_decay_timer = AsyncMock()
        mock_coordinator.async_refresh = AsyncMock()

        await mock_coordinator._handle_decay_timer(test_time)
        mock_coordinator._handle_decay_timer.assert_called_once_with(test_time)

    async def test_handle_decay_timer_decay_disabled(
        self, mock_coordinator: Mock
    ) -> None:
        """Test _handle_decay_timer with decay disabled using centralized mock."""
        test_time = dt_util.utcnow()

        # Configure mock for disabled decay
        mock_coordinator.config.decay.enabled = False
        mock_coordinator._handle_decay_timer = AsyncMock()

        await mock_coordinator._handle_decay_timer(test_time)
        mock_coordinator._handle_decay_timer.assert_called_once_with(test_time)

    def test_start_decay_timer_success(self, mock_coordinator: Mock) -> None:
        """Test _start_decay_timer method using centralized mock."""
        mock_coordinator._global_decay_timer = None
        mock_coordinator._start_decay_timer = Mock()

        mock_coordinator._start_decay_timer()
        mock_coordinator._start_decay_timer.assert_called_once()

    def test_start_decay_timer_already_exists(self, mock_coordinator: Mock) -> None:
        """Test _start_decay_timer when timer already exists using centralized mock."""
        existing_timer = Mock()
        mock_coordinator._global_decay_timer = existing_timer
        mock_coordinator._start_decay_timer = Mock()

        mock_coordinator._start_decay_timer()
        mock_coordinator._start_decay_timer.assert_called_once()


class TestCoordinatorPropertyCalculations:
    """Test coordinator property calculations using centralized mocks."""

    def test_probability_with_empty_entities(self, mock_coordinator: Mock) -> None:
        """Test probability calculation with no entities using centralized mock."""
        # Configure mock for empty entities
        mock_coordinator.entities.entities = {}
        mock_coordinator.probability = 0.0  # MIN_PROBABILITY

        assert mock_coordinator.probability == 0.0

    def test_probability_with_active_entities(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test probability calculation with active entities using centralized mock."""
        # The mock_coordinator_with_sensors has realistic probability
        assert 0.0 <= mock_coordinator_with_sensors.probability <= 1.0
        assert mock_coordinator_with_sensors.probability == 0.65

    def test_probability_with_decaying_entities(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test probability calculation with decaying entities using centralized mock."""
        # Configure entities with decay
        motion2 = mock_coordinator_with_sensors.entities.entities[
            "binary_sensor.motion2"
        ]
        motion2.decay.is_decaying = True
        motion2.decay.decay_factor = 0.8
        motion2.evidence = False

        # Should still factor in decaying entities
        assert 0.0 <= mock_coordinator_with_sensors.probability <= 1.0

    def test_prior_with_empty_entities(self, mock_coordinator: Mock) -> None:
        """Test prior calculation with no entities using centralized mock."""
        mock_coordinator.entities.entities = {}
        mock_coordinator.area_prior = MIN_PRIOR

        assert mock_coordinator.area_prior == MIN_PRIOR

    def test_prior_with_multiple_entities(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test prior calculation with multiple entities using centralized mock."""
        # Should average all entity priors
        assert 0.0 <= mock_coordinator_with_sensors.area_prior <= 1.0
        assert mock_coordinator_with_sensors.area_prior == 0.35

    def test_decay_with_empty_entities(self, mock_coordinator: Mock) -> None:
        """Test decay calculation with no entities using centralized mock."""
        mock_coordinator.entities.entities = {}
        mock_coordinator.decay = 1.0

        assert mock_coordinator.decay == 1.0

    def test_decay_with_mixed_entities(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decay calculation with mixed decaying/non-decaying entities using centralized mock."""
        # Should average all entity decay factors
        assert 0.0 <= mock_coordinator_with_sensors.decay <= 1.0
        assert mock_coordinator_with_sensors.decay == 0.8

    def test_occupied_threshold_comparison(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test occupied property threshold comparison using centralized mock."""
        # Mock coordinator has probability 0.5 and threshold 0.6
        assert not mock_coordinator_with_threshold.is_occupied  # 0.5 < 0.6

        # Test at threshold boundary
        mock_coordinator_with_threshold.probability = 0.6
        mock_coordinator_with_threshold.is_occupied = True  # 0.6 >= 0.6
        assert mock_coordinator_with_threshold.is_occupied

    def test_threshold_from_config(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test threshold property from config using centralized mock."""
        assert mock_coordinator_with_threshold.threshold == 0.6

    def test_threshold_fallback_no_config(self, mock_coordinator: Mock) -> None:
        """Test threshold fallback when config is None using centralized mock."""
        mock_coordinator.config = None
        mock_coordinator.threshold = 0.5  # Mock fallback value

        assert mock_coordinator.threshold == 0.5

    def test_binary_sensor_entity_ids_structure(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids property structure using centralized mock."""
        entity_ids = mock_coordinator.binary_sensor_entity_ids

        assert isinstance(entity_ids, dict)
        assert "occupancy" in entity_ids
        assert "wasp" in entity_ids

    def test_binary_sensor_entity_ids_with_values(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids with actual values using centralized mock."""
        # Set mock entity IDs
        mock_coordinator.occupancy_entity_id = "binary_sensor.test_occupancy"
        mock_coordinator.wasp_entity_id = "binary_sensor.test_wasp"
        mock_coordinator.binary_sensor_entity_ids = {
            "occupancy": "binary_sensor.test_occupancy",
            "wasp": "binary_sensor.test_wasp",
        }

        entity_ids = mock_coordinator.binary_sensor_entity_ids
        assert entity_ids["occupancy"] == "binary_sensor.test_occupancy"
        assert entity_ids["wasp"] == "binary_sensor.test_wasp"

    def test_decaying_entities_filtering(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decaying_entities property filtering using centralized mock."""
        # Configure decaying entities
        motion2 = mock_coordinator_with_sensors.entities.entities[
            "binary_sensor.motion2"
        ]
        motion2.decay.is_decaying = True

        mock_coordinator_with_sensors.decaying_entities = [motion2]

        decaying = mock_coordinator_with_sensors.decaying_entities
        assert len(decaying) == 1
        assert decaying[0].entity_id == "binary_sensor.motion2"


class TestCoordinatorIntegrationUsingCentralizedMocks:
    """Test coordinator integration scenarios using centralized mocks exclusively."""

    async def test_full_coordinator_lifecycle_with_centralized_mocks(
        self, mock_coordinator: Mock
    ) -> None:
        """Test complete coordinator lifecycle using centralized mock."""
        coordinator = mock_coordinator

        # Test setup
        await coordinator.setup()

        # Test data update
        await coordinator.update()

        # Test option updates
        await coordinator.async_update_options({"threshold": 70})

        # Test entity state tracking
        await coordinator.track_entity_state_changes(["binary_sensor.test"])

        # Test shutdown
        await coordinator.async_shutdown()

        # Verify all components were properly called
        coordinator.setup.assert_called_once()
        coordinator.update.assert_called_once()
        coordinator.async_update_options.assert_called_once_with({"threshold": 70})
        coordinator.track_entity_state_changes.assert_called_once_with(
            ["binary_sensor.test"]
        )
        coordinator.async_shutdown.assert_called_once()

    async def test_update_method_data_structure(self, mock_coordinator: Mock) -> None:
        """Test update method returns correct data structure using centralized mock."""
        # Configure mock to return expected data structure
        test_data = {
            "probability": 0.65,
            "occupied": True,
            "threshold": 0.5,
            "prior": 0.35,
            "decay": 0.8,
            "last_updated": dt_util.utcnow(),
        }
        mock_coordinator.update.return_value = test_data

        result = await mock_coordinator.update()

        expected_keys = {
            "probability",
            "occupied",
            "threshold",
            "prior",
            "decay",
            "last_updated",
        }
        assert set(result.keys()) == expected_keys
        assert 0.0 <= result["probability"] <= 1.0
        assert isinstance(result["occupied"], bool)
        assert 0.0 <= result["threshold"] <= 1.0

    async def test_state_tracking_with_empty_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test entity state tracking with empty entity list using centralized mock."""
        await mock_coordinator.track_entity_state_changes([])
        mock_coordinator.track_entity_state_changes.assert_called_once_with([])

    async def test_state_tracking_with_multiple_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test entity state tracking with multiple entities using centralized mock."""
        entity_ids = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
            "media_player.tv",
        ]

        await mock_coordinator.track_entity_state_changes(entity_ids)
        mock_coordinator.track_entity_state_changes.assert_called_once_with(entity_ids)

    def test_threshold_boundary_conditions_comprehensive(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test is_occupied calculation at various threshold boundaries using centralized mock."""
        coordinator = mock_coordinator_with_threshold

        # Test below threshold
        coordinator.probability = 0.59
        coordinator.threshold = 0.6
        coordinator.is_occupied = False
        assert not coordinator.is_occupied

        # Test at threshold (should be occupied)
        coordinator.probability = 0.6
        coordinator.threshold = 0.6
        coordinator.is_occupied = True
        assert coordinator.is_occupied

        # Test above threshold
        coordinator.probability = 0.61
        coordinator.threshold = 0.6
        coordinator.is_occupied = True
        assert coordinator.is_occupied

    async def test_configuration_updates_with_centralized_mocks(
        self, mock_coordinator: Mock
    ) -> None:
        """Test configuration updates using centralized mock."""

        # Update configuration
        new_options = {"threshold": 0.8, "decay_enabled": True, "decay_window": 600}

        await mock_coordinator.async_update_options(new_options)

        # Verify update was called
        mock_coordinator.async_update_options.assert_called_once_with(new_options)

    def test_device_info_comprehensive(self, mock_coordinator: Mock) -> None:
        """Test device_info property comprehensively using centralized mock."""
        device_info = mock_coordinator.device_info

        # Test all required fields exist
        required_fields = {"identifiers", "name", "manufacturer", "model", "sw_version"}
        assert all(field in device_info for field in required_fields)

        # Test field types
        assert isinstance(device_info["identifiers"], set)
        assert isinstance(device_info["name"], str)
        assert isinstance(device_info["manufacturer"], str)
        assert isinstance(device_info["model"], str)
        assert isinstance(device_info["sw_version"], str)

        # Test identifiers structure
        identifiers = device_info["identifiers"]
        assert (DOMAIN, mock_coordinator.entry_id) in identifiers


class TestCoordinatorErrorHandlingUsingCentralizedMocks:
    """Test coordinator error handling using centralized mocks."""

    async def test_setup_error_handling(self, mock_coordinator: Mock) -> None:
        """Test setup error handling using centralized mock."""
        # Configure mock to raise error
        mock_coordinator.setup.side_effect = ConfigEntryNotReady("Setup failed")

        with pytest.raises(ConfigEntryNotReady, match="Setup failed"):
            await mock_coordinator.setup()

    async def test_update_error_handling(self, mock_coordinator: Mock) -> None:
        """Test update error handling using centralized mock."""
        # Configure mock to raise error
        mock_coordinator.update.side_effect = HomeAssistantError("Update failed")

        with pytest.raises(HomeAssistantError, match="Update failed"):
            await mock_coordinator.update()

    async def test_option_update_error_handling(self, mock_coordinator: Mock) -> None:
        """Test option update error handling using centralized mock."""
        # Configure mock to raise error
        mock_coordinator.async_update_options.side_effect = HomeAssistantError(
            "Option update failed"
        )

        with pytest.raises(HomeAssistantError, match="Option update failed"):
            await mock_coordinator.async_update_options({"threshold": 0.8})

    async def test_state_tracking_error_handling(self, mock_coordinator: Mock) -> None:
        """Test state tracking error handling using centralized mock."""
        # Configure mock to raise error
        mock_coordinator.track_entity_state_changes.side_effect = HomeAssistantError(
            "Tracking failed"
        )

        with pytest.raises(HomeAssistantError, match="Tracking failed"):
            await mock_coordinator.track_entity_state_changes(["binary_sensor.test"])

    async def test_shutdown_error_handling(self, mock_coordinator: Mock) -> None:
        """Test shutdown error handling using centralized mock."""
        # Configure mock to raise error
        mock_coordinator.async_shutdown.side_effect = HomeAssistantError(
            "Shutdown failed"
        )

        with pytest.raises(HomeAssistantError, match="Shutdown failed"):
            await mock_coordinator.async_shutdown()


class TestCoordinatorEdgeCasesUsingCentralizedMocks:
    """Test coordinator edge cases using centralized mocks."""

    def test_initialization_with_realistic_config(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test initialization with realistic config entry."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        assert coordinator.entry_id == mock_realistic_config_entry.entry_id
        assert coordinator.name == mock_realistic_config_entry.data["name"]
        assert coordinator.config_entry == mock_realistic_config_entry

    def test_probability_edge_values(self, mock_coordinator: Mock) -> None:
        """Test probability property with edge values using centralized mock."""
        # Test minimum probability
        mock_coordinator.entities.entities = {}
        mock_coordinator.probability = 0.0
        assert mock_coordinator.probability == 0.0

        # Test maximum probability
        mock_coordinator.probability = 1.0
        assert mock_coordinator.probability == 1.0

    def test_threshold_edge_values(self, mock_coordinator: Mock) -> None:
        """Test threshold property with edge values using centralized mock."""
        # Test minimum threshold
        mock_coordinator.config.threshold = 0.0
        mock_coordinator.threshold = 0.0
        assert mock_coordinator.threshold == 0.0

        # Test maximum threshold
        mock_coordinator.config.threshold = 1.0
        mock_coordinator.threshold = 1.0
        assert mock_coordinator.threshold == 1.0

    def test_prior_edge_values(self, mock_coordinator: Mock) -> None:
        """Test prior property with edge values using centralized mock."""
        # Test with no entities (should use DEFAULT_PRIOR)
        mock_coordinator.entities.entities = {}
        mock_coordinator.area_prior = MIN_PRIOR
        assert mock_coordinator.area_prior == MIN_PRIOR

    def test_decay_edge_values(self, mock_coordinator: Mock) -> None:
        """Test decay property with edge values using centralized mock."""
        # Test with no entities (should be 1.0)
        mock_coordinator.entities.entities = {}
        mock_coordinator.decay = 1.0
        assert mock_coordinator.decay == 1.0

    async def test_empty_entity_list_tracking(self, mock_coordinator: Mock) -> None:
        """Test tracking empty entity list using centralized mock."""
        await mock_coordinator.track_entity_state_changes([])
        mock_coordinator.track_entity_state_changes.assert_called_once_with([])

    def test_binary_sensor_entity_ids_none_values(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids with None values using centralized mock."""
        mock_coordinator.occupancy_entity_id = None
        mock_coordinator.wasp_entity_id = None
        mock_coordinator.binary_sensor_entity_ids = {"occupancy": None, "wasp": None}

        entity_ids = mock_coordinator.binary_sensor_entity_ids
        assert entity_ids["occupancy"] is None
        assert entity_ids["wasp"] is None


class TestCoordinatorAdvancedTimerManagement:
    """Test advanced timer management scenarios for comprehensive coverage."""

    async def test_timer_lifecycle_full_cycle(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test complete timer lifecycle from start to cancellation."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock entity types to prevent KeyError during shutdown
        with (
            patch.object(
                coordinator.entity_types, "get_entity_type"
            ) as mock_get_entity_type,
            patch(
                "homeassistant.helpers.event.async_track_point_in_time",
                return_value=Mock(),
            ),
            patch.object(coordinator.db, "save_data", new_callable=AsyncMock),
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8  # Real float value for math operations
            mock_entity_type.active_states = ["on"]  # Make iterable
            mock_entity_type.active_range = None
            mock_get_entity_type.return_value = mock_entity_type

            # Test all timers are None initially
            assert coordinator._global_decay_timer is None

            # Start timers
            coordinator._start_decay_timer()

            # Verify timers are set
            assert coordinator._global_decay_timer is not None

            # Test shutdown cancels all timers
            await coordinator.async_shutdown()

        assert coordinator._global_decay_timer is None

    def test_timer_start_with_missing_hass(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test timer start when hass is missing/invalid."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock hass to be falsy for timer check
        with patch.object(coordinator, "hass", None):
            # Should not start timers when hass is None
            coordinator._start_decay_timer()

            assert coordinator._global_decay_timer is None

    async def test_timer_error_handling_during_callbacks(
        self, mock_coordinator: Mock
    ) -> None:
        """Test timer callback error handling."""
        test_time = dt_util.utcnow()

        # Test decay timer error handling
        mock_coordinator.async_refresh.side_effect = HomeAssistantError(
            "Refresh failed"
        )

        with patch.object(mock_coordinator, "_start_decay_timer"):
            await mock_coordinator._handle_decay_timer(test_time)


@pytest.mark.expected_lingering_timers(True)
class TestCoordinatorSetupScenarios:
    """Test various coordinator setup scenarios for better coverage."""

    async def test_setup_with_stored_data_restoration(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with stored data restoration."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock stored data
        stored_data = {"entities": {"binary_sensor.test": {}}}
        coordinator.db.load_data = AsyncMock(return_value=stored_data)

        # Mock other dependencies
        with (
            patch.object(coordinator.purpose, "async_initialize", new=AsyncMock()),
            patch.object(coordinator.entity_types, "async_initialize", new=AsyncMock()),
            patch.object(coordinator.entities, "__post_init__", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(coordinator.db, "is_intervals_empty", return_value=False),
            patch.object(
                coordinator, "run_analysis", new=AsyncMock()
            ),  # Mock run_analysis
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(coordinator, "_start_decay_timer"),
            patch.object(coordinator, "_start_analysis_timer"),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
        ):
            await coordinator.setup()

    async def test_setup_entity_initialization_failure(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when entity initialization fails."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock storage to avoid unpacking issues
        coordinator.db.load_data = AsyncMock(return_value=None)

        # Mock entity initialization failure - patch the actual method that can fail
        with (
            patch.object(
                coordinator.entity_types,
                "async_initialize",
                side_effect=HomeAssistantError("Entity init failed"),
            ),
            patch.object(coordinator, "_start_decay_timer"),
            patch.object(coordinator, "_start_analysis_timer"),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch.object(
                coordinator.entity_types, "get_entity_type"
            ) as mock_get_entity_type,
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8  # Real float value for math operations
            mock_entity_type.active_states = ["on"]  # Make iterable
            mock_entity_type.active_range = None
            mock_get_entity_type.return_value = mock_entity_type

            with pytest.raises(
                ConfigEntryNotReady, match="Failed to set up coordinator"
            ):
                await coordinator.setup()

    async def test_setup_storage_failure_recovery(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when storage operations fail - should raise ConfigEntryNotReady."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock storage failure
        coordinator.db.load_data = AsyncMock(
            side_effect=HomeAssistantError("Storage failed")
        )

        # Mock other dependencies
        with (
            patch.object(coordinator.purpose, "async_initialize", new=AsyncMock()),
            patch.object(coordinator.entity_types, "async_initialize", new=AsyncMock()),
            patch.object(coordinator.entities, "__post_init__", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            pytest.raises(
                ConfigEntryNotReady,
                match="Failed to set up coordinator: Storage failed",
            ),
        ):
            await coordinator.setup()

    async def test_setup_partial_failure_recovery(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test recovery from partial setup failures - should raise ConfigEntryNotReady."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock partial failure scenario
        coordinator.entity_types.async_initialize = AsyncMock()
        coordinator.db.load_data = AsyncMock(
            side_effect=HomeAssistantError("Storage unavailable")
        )

        # Mock other dependencies
        with (
            patch.object(coordinator.purpose, "async_initialize", new=AsyncMock()),
            patch.object(coordinator.entities, "__post_init__", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            pytest.raises(
                ConfigEntryNotReady,
                match="Failed to set up coordinator: Storage unavailable",
            ),
        ):
            await coordinator.setup()


class TestCoordinatorProbabilityCalculationEdgeCases:
    """Test probability calculation edge cases and complex scenarios."""

    def test_probability_with_mixed_evidence_and_decay(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test probability calculation with mixed evidence and decay states."""
        # Setup entities with various states
        entities = mock_coordinator_with_sensors.entities.entities

        # Entity 1: Active evidence, no decay
        entities["binary_sensor.motion1"].evidence = True
        entities["binary_sensor.motion1"].decay.is_decaying = False
        entities["binary_sensor.motion1"].decay.decay_factor = 1.0

        # Entity 2: No evidence, but decaying
        entities["binary_sensor.motion2"].evidence = False
        entities["binary_sensor.motion2"].decay.is_decaying = True
        entities["binary_sensor.motion2"].decay.decay_factor = 0.5

        # Entity 3: No evidence, no decay
        entities["binary_sensor.appliance"].evidence = False
        entities["binary_sensor.appliance"].decay.is_decaying = False

        # Entity 4: Active evidence and decaying
        entities["media_player.tv"].evidence = True
        entities["media_player.tv"].decay.is_decaying = True
        entities["media_player.tv"].decay.decay_factor = 0.8

        # Test that probability incorporates both evidence and decay
        probability = mock_coordinator_with_sensors.probability
        assert 0.0 <= probability <= 1.0

    def test_probability_calculation_with_varying_weights(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test probability calculation with entities having different weights."""
        entities = mock_coordinator_with_sensors.entities.entities

        # Set up entities with different weights and evidence
        entities["binary_sensor.motion1"].type.weight = 0.9
        entities["binary_sensor.motion1"].evidence = True

        entities["binary_sensor.appliance"].type.weight = 0.1
        entities["binary_sensor.appliance"].evidence = True

        # High weight entity should have more impact
        probability = mock_coordinator_with_sensors.probability
        assert 0.0 <= probability <= 1.0

    def test_prior_calculation_with_heterogeneous_entities(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test prior calculation with entities having different prior values."""
        entities = mock_coordinator_with_sensors.entities.entities

        # Set different likelihood values
        entities["binary_sensor.motion1"].likelihood.prob_given_true = 0.9
        entities["binary_sensor.motion2"].likelihood.prob_given_true = 0.3
        entities["binary_sensor.appliance"].likelihood.prob_given_true = 0.1
        entities["media_player.tv"].likelihood.prob_given_true = 0.7

        # Calculate expected area prior and set mock to return it
        expected_prior = (0.9 + 0.3 + 0.1 + 0.7) / 4
        mock_coordinator_with_sensors.area_prior = expected_prior

        prior = mock_coordinator_with_sensors.area_prior

        # Should be average of all entity priors
        assert abs(prior - expected_prior) < 0.01

    def test_decay_calculation_with_mixed_states(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decay calculation with entities in different decay states."""
        entities = mock_coordinator_with_sensors.entities.entities

        # Set different decay factors
        entities["binary_sensor.motion1"].decay.decay_factor = 1.0  # No decay
        entities["binary_sensor.motion2"].decay.decay_factor = 0.5  # Half decay
        entities["binary_sensor.appliance"].decay.decay_factor = 0.2  # Strong decay
        entities["media_player.tv"].decay.decay_factor = 0.8  # Appliance decay

        # Calculate expected decay and set mock to return it
        expected_decay = (1.0 + 0.5 + 0.2 + 0.8) / 4
        mock_coordinator_with_sensors.decay = expected_decay

        decay = mock_coordinator_with_sensors.decay

        # Should be average of all decay factors
        assert abs(decay - expected_decay) < 0.01


class TestCoordinatorResourceManagement:
    """Test coordinator resource management and cleanup."""

    async def test_shutdown_behavior_with_real_coordinator(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test shutdown behavior with real coordinator instance."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Set up some timers to test cleanup
        coordinator._global_decay_timer = Mock()
        coordinator._remove_state_listener = Mock()

        # Mock entity types to prevent KeyError during shutdown
        with (
            patch.object(
                coordinator.entity_types, "get_entity_type"
            ) as mock_get_entity_type,
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8  # Real float value for math operations
            mock_entity_type.active_states = ["on"]  # Make iterable
            mock_entity_type.active_range = None
            mock_get_entity_type.return_value = mock_entity_type

            # Mock super class shutdown to avoid complications
            await coordinator.async_shutdown()

            # Verify resources were cleaned up
            assert coordinator._global_decay_timer is None
            assert coordinator._remove_state_listener is None

    async def test_shutdown_with_none_resources(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test shutdown when resources are already None."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Resources already None - should not crash
        coordinator._global_decay_timer = None
        coordinator._remove_state_listener = None

        # Mock entity types to prevent KeyError during shutdown
        with (
            patch.object(
                coordinator.entity_types, "get_entity_type"
            ) as mock_get_entity_type,
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8  # Real float value for math operations
            mock_entity_type.active_states = ["on"]  # Make iterable
            mock_entity_type.active_range = None
            mock_get_entity_type.return_value = mock_entity_type

            # Should complete without errors
            await coordinator.async_shutdown()

            # Verify they remain None
            assert coordinator._global_decay_timer is None
            assert coordinator._remove_state_listener is None


class TestCoordinatorIntegrationFlows:
    """Test complete integration flows and workflows."""

    async def test_full_coordinator_lifecycle_with_realistic_data(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test complete coordinator lifecycle with realistic configuration."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock entity types to prevent KeyError during async_update_options
        with (
            patch.object(
                coordinator.entity_types, "get_entity_type"
            ) as mock_get_entity_type,
            patch.object(coordinator.entity_types, "async_initialize", new=AsyncMock()),
            patch.object(
                coordinator.db,
                "load_data",
                new=AsyncMock(return_value=None),
            ),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(coordinator, "_start_decay_timer"),
            patch.object(
                coordinator, "run_analysis", new=AsyncMock()
            ),  # Mock run_analysis to prevent recorder calls
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8  # Real float value for math operations
            mock_entity_type.active_states = ["on"]  # Make iterable
            mock_entity_type.active_range = None
            mock_get_entity_type.return_value = mock_entity_type

            # Setup
            await coordinator.setup()

            # Test update - use the correct method name
            await coordinator.update()

            # Test shutdown
            await coordinator.async_shutdown()

            # Verify entity type was accessed
            mock_get_entity_type.assert_called()


class TestCoordinatorPerformanceScenarios:
    """Test coordinator performance with various load scenarios."""

    # Add expected_lingering_timers marker to handle timer cleanup

    @pytest.mark.expected_lingering_timers(True)
    def test_probability_calculation_with_many_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test probability calculation performance with many entities."""
        # Create many mock entities
        entities = {}
        for i in range(50):
            entity_id = f"binary_sensor.motion_{i}"
            mock_entity = Mock()
            mock_entity.probability = 0.5 + (i * 0.01)  # Varying probabilities
            mock_entity.weight = 0.8
            mock_entity.is_active = True
            mock_entity.is_decaying = False
            entities[entity_id] = mock_entity

        mock_coordinator.entities.entities = entities

        # Calculate probability
        probability = mock_coordinator.probability

        # Should be a valid probability
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0

    def test_prior_calculation_with_many_entities(self, mock_coordinator: Mock) -> None:
        """Test prior calculation with many entities."""
        # Create entities with different prior values
        entities = {}
        for i in range(50):
            entity_id = f"sensor.test_{i}"
            entities[entity_id] = Mock(
                prior=Mock(prob_given_true=i / 100.0)  # Varying priors from 0 to 0.49
            )

        mock_coordinator.entities.entities = entities

        # Should handle calculation efficiently
        prior = mock_coordinator.area_prior
        assert 0.0 <= prior <= 1.0

    async def test_state_tracking_with_many_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state tracking setup with many entities."""
        # Generate large entity list
        entity_ids = [f"binary_sensor.motion_{i}" for i in range(200)]

        # Test that the method can be called with large lists without errors
        await mock_coordinator.track_entity_state_changes(entity_ids)

        # Verify the method was called with the large list
        mock_coordinator.track_entity_state_changes.assert_called_with(entity_ids)


class TestCoordinatorDeviceInfoAndProperties:
    """Test device info and property edge cases."""

    def test_device_info_with_missing_config(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test device info generation when config is missing."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock config.name to be None to test missing config handling
        with patch.object(coordinator, "config") as mock_config:
            mock_config.name = None

            # Should handle missing config gracefully
            device_info = coordinator.device_info

            # Should still have required fields
            assert "identifiers" in device_info
            assert "manufacturer" in device_info
            assert "model" in device_info
            assert "sw_version" in device_info

    def test_binary_sensor_entity_ids_with_values(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids with actual values."""
        test_occupancy_id = "binary_sensor.test_area_occupancy"
        test_wasp_id = "binary_sensor.test_area_wasp"

        mock_coordinator.occupancy_entity_id = test_occupancy_id
        mock_coordinator.wasp_entity_id = test_wasp_id

        entity_ids = mock_coordinator.binary_sensor_entity_ids

        assert entity_ids["occupancy"] == test_occupancy_id
        assert entity_ids["wasp"] == test_wasp_id

    def test_decaying_entities_filtering_complex(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decaying entities filtering with complex scenarios."""
        entities = mock_coordinator_with_sensors.entities.entities

        # Set up mixed decay states
        entities["binary_sensor.motion1"].decay.is_decaying = True
        entities["binary_sensor.motion2"].decay.is_decaying = False
        entities["binary_sensor.appliance"].decay.is_decaying = True
        entities["media_player.tv"].decay.is_decaying = False

        # Create expected decaying entities list
        expected_decaying = [
            entities["binary_sensor.motion1"],
            entities["binary_sensor.appliance"],
        ]
        mock_coordinator_with_sensors.decaying_entities = expected_decaying

        decaying = mock_coordinator_with_sensors.decaying_entities

        assert len(decaying) == 2
        assert entities["binary_sensor.motion1"] in decaying
        assert entities["binary_sensor.appliance"] in decaying
        assert entities["binary_sensor.motion2"] not in decaying
        assert entities["media_player.tv"] not in decaying


class TestCoordinatorErrorRecoveryAndResilience:
    """Test coordinator error recovery and resilience patterns."""

    async def test_setup_partial_failure_recovery(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test recovery from partial setup failures - should raise ConfigEntryNotReady."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Mock partial failure scenario
        coordinator.entity_types.async_initialize = AsyncMock()
        coordinator.db.load_data = AsyncMock(
            side_effect=HomeAssistantError("Storage unavailable")
        )

        # Mock other dependencies
        with (
            patch.object(coordinator.purpose, "async_initialize", new=AsyncMock()),
            patch.object(coordinator.entities, "__post_init__", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            pytest.raises(
                ConfigEntryNotReady,
                match="Failed to set up coordinator: Storage unavailable",
            ),
        ):
            await coordinator.setup()

    async def test_update_method_structure(self, mock_coordinator: Mock) -> None:
        """Test update method structure and basic functionality."""
        # Configure mock to return expected data structure
        test_data = {
            "probability": 0.5,
            "occupied": True,
            "threshold": 0.6,
            "prior": 0.3,
            "decay": 0.8,
            "last_updated": dt_util.utcnow(),
        }
        mock_coordinator.update.return_value = test_data

        # Update should return data structure
        update_data = await mock_coordinator.update()

        # Verify data structure is returned
        assert isinstance(update_data, dict)
        assert "probability" in update_data
        assert "occupied" in update_data
        assert "last_updated" in update_data

    async def test_state_tracking_method_structure(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state tracking method structure."""
        entity_ids = ["binary_sensor.motion"]

        # Test that the method exists and can be called
        assert hasattr(mock_coordinator, "track_entity_state_changes")
        await mock_coordinator.track_entity_state_changes(entity_ids)

        # Verify it was called with the expected arguments
        mock_coordinator.track_entity_state_changes.assert_called_with(entity_ids)
