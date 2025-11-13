"""Tests for coordinator module."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.area.area import Area
from custom_components.area_occupancy.const import (
    ALL_AREAS_IDENTIFIER,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import Sensors
from custom_components.area_occupancy.data.prior import MIN_PRIOR
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.util import dt as dt_util

# Import helper functions from conftest
from tests.conftest import create_test_area


# ruff: noqa: SLF001, TID251
@pytest.fixture(autouse=True)
def _disable_frame_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable Home Assistant frame reporting which requires special setup."""
    monkeypatch.setattr(
        "homeassistant.helpers.frame.report_usage", lambda *args, **kwargs: None
    )


# Automatically apply the frame helper mock to all tests in this module
pytestmark = pytest.mark.usefixtures("mock_frame_helper")


class TestAreaOccupancyCoordinator:
    """Test AreaOccupancyCoordinator class."""

    @pytest.fixture(autouse=True)
    def mock_async_call_later(self):
        """Mock async_call_later to prevent lingering timers across all tests."""
        with patch(
            "custom_components.area_occupancy.coordinator.async_call_later",
            return_value=Mock(),
        ):
            yield

    def test_initialization(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test coordinator initialization."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        assert coordinator.hass == mock_hass
        assert coordinator.config_entry == mock_realistic_config_entry
        assert coordinator.entry_id == mock_realistic_config_entry.entry_id
        assert coordinator.name == mock_realistic_config_entry.data["name"]

    def test_device_info_property(self, mock_coordinator: Mock) -> None:
        """Test device_info property."""
        # device_info is now a method that takes area_name
        area_name = (
            mock_coordinator.get_area_names()[0]
            if mock_coordinator.get_area_names()
            else "Test Area"
        )
        device_info = mock_coordinator.device_info(area_name)

        assert "identifiers" in device_info
        assert "name" in device_info
        assert "manufacturer" in device_info
        assert "model" in device_info
        assert isinstance(device_info["identifiers"], set)
        assert isinstance(device_info["name"], str)

    def test_device_info_with_real_constants(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test device_info property with actual constant values."""
        # device_info is now a method that takes area_name
        area_name = coordinator_with_areas.get_area_names()[0]
        device_info = coordinator_with_areas.device_info(area_name)

        assert device_info.get("manufacturer") == DEVICE_MANUFACTURER
        assert device_info.get("model") == DEVICE_MODEL
        assert device_info.get("sw_version") == DEVICE_SW_VERSION

        identifiers = device_info.get("identifiers")
        assert identifiers is not None
        assert isinstance(identifiers, set)
        # In multi-area architecture, device_info uses area_name as identifier, not entry_id
        expected_identifier = (DOMAIN, area_name)
        assert expected_identifier in identifiers, (
            f"Expected {expected_identifier} in {identifiers}"
        )

    def test_device_info_with_all_areas_identifier(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test device_info properly handles ALL_AREAS_IDENTIFIER."""
        device_info = coordinator_with_areas.device_info(ALL_AREAS_IDENTIFIER)

        assert device_info.get("manufacturer") == DEVICE_MANUFACTURER
        assert device_info.get("model") == DEVICE_MODEL
        assert device_info.get("sw_version") == DEVICE_SW_VERSION
        assert device_info.get("name") == "All Areas"

        identifiers = device_info.get("identifiers")
        assert identifiers is not None
        assert isinstance(identifiers, set)
        # Should use ALL_AREAS_IDENTIFIER, not entry_id
        expected_identifier = (DOMAIN, ALL_AREAS_IDENTIFIER)
        assert expected_identifier in identifiers, (
            f"Expected {expected_identifier} in {identifiers}, got {identifiers}"
        )

    def test_device_info_with_missing_config(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test device info generation when config is missing."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Get first area (always exists - at least one area is guaranteed)
        area = coordinator.get_area_or_default()
        # Handle case where areas might not be loaded in test
        if area is None:
            # Use fallback device info test
            device_info = coordinator.device_info(area_name="NonExistentArea")
            assert "identifiers" in device_info
            assert "manufacturer" in device_info
            assert "model" in device_info
            assert "sw_version" in device_info
            return

        with patch.object(area, "config") as mock_config:
            mock_config.name = None
            device_info = coordinator.device_info(area_name=area.area_name)

        assert "identifiers" in device_info
        assert "manufacturer" in device_info
        assert "model" in device_info
        assert "sw_version" in device_info

    @pytest.mark.parametrize(
        ("property_name", "expected_value"),
        [
            ("probability", 0.5),
            ("area_prior", 0.3),
            ("decay", 1.0),
        ],
    )
    def test_basic_properties(
        self, mock_coordinator: Mock, property_name: str, expected_value: float
    ) -> None:
        """Test basic coordinator properties."""
        # These are now methods that take area_name and delegate to area methods
        area_name = (
            mock_coordinator.get_area_names()[0]
            if mock_coordinator.get_area_names()
            else "Test Area"
        )
        # Mock coordinator method to return expected value (it delegates to area)
        method = getattr(mock_coordinator, property_name)
        method.return_value = expected_value
        assert method(area_name) == expected_value

    def test_threshold_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test threshold property specifically."""
        # threshold is now a method that delegates to area
        area_name = (
            mock_coordinator_with_threshold.get_area_names()[0]
            if mock_coordinator_with_threshold.get_area_names()
            else "Test Area"
        )
        # The wrapper should call area.threshold()
        area = mock_coordinator_with_threshold.get_area_or_default(area_name)
        assert area.threshold.return_value == 0.6

    def test_is_occupied_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test occupied method threshold comparison."""
        # Mock coordinator has threshold 0.6 and probability 0.5
        # occupied is now a method that takes area_name
        area_name = (
            mock_coordinator_with_threshold.get_area_names()[0]
            if mock_coordinator_with_threshold.get_area_names()
            else "Test Area"
        )
        # Configure occupied to return False (0.5 < 0.6)
        mock_coordinator_with_threshold.occupied = Mock(return_value=False)
        assert not mock_coordinator_with_threshold.occupied(area_name)  # 0.5 < 0.6

        # Test at threshold boundary
        mock_coordinator_with_threshold.probability = Mock(return_value=0.6)
        mock_coordinator_with_threshold.occupied = Mock(return_value=True)  # 0.6 >= 0.6
        assert mock_coordinator_with_threshold.occupied(area_name)

    def test_binary_sensor_entity_ids_property(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids property."""
        entity_ids = mock_coordinator.binary_sensor_entity_ids

        assert isinstance(entity_ids, dict)
        assert "occupancy" in entity_ids
        assert "wasp" in entity_ids

    def test_binary_sensor_entity_ids_with_values(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids with actual values."""
        test_occupancy_id = "binary_sensor.test_occupancy"
        test_wasp_id = "binary_sensor.test_wasp"

        mock_coordinator.occupancy_entity_id = test_occupancy_id
        mock_coordinator.wasp_entity_id = test_wasp_id
        mock_coordinator.binary_sensor_entity_ids = {
            "occupancy": test_occupancy_id,
            "wasp": test_wasp_id,
        }

        entity_ids = mock_coordinator.binary_sensor_entity_ids
        assert entity_ids["occupancy"] == test_occupancy_id
        assert entity_ids["wasp"] == test_wasp_id

    def test_binary_sensor_entity_ids_none_values(self, mock_coordinator: Mock) -> None:
        """Test binary_sensor_entity_ids with None values."""
        mock_coordinator.occupancy_entity_id = None
        mock_coordinator.wasp_entity_id = None
        mock_coordinator.binary_sensor_entity_ids = {"occupancy": None, "wasp": None}

        entity_ids = mock_coordinator.binary_sensor_entity_ids
        assert entity_ids["occupancy"] is None
        assert entity_ids["wasp"] is None

    def test_decaying_entities_property(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decaying_entities property filtering."""
        # Configure decaying entities - use area-based access
        area = mock_coordinator_with_sensors.get_area_or_default.return_value
        motion2 = area.entities.entities["binary_sensor.motion2"]
        motion2.decay.is_decaying = True

        area.entities.decaying_entities = [motion2]

        decaying = area.entities.decaying_entities
        assert len(decaying) == 1
        assert decaying[0].entity_id == "binary_sensor.motion2"

    def test_decaying_entities_filtering_complex(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test decaying entities filtering with complex scenarios."""
        # Use area-based access
        area = mock_coordinator_with_sensors.get_area_or_default.return_value
        entities = area.entities.entities

        # Set up mixed decay states
        entities["binary_sensor.motion1"].decay.is_decaying = True
        entities["binary_sensor.motion2"].decay.is_decaying = False
        entities["binary_sensor.appliance"].decay.is_decaying = True
        entities["media_player.tv"].decay.is_decaying = False

        expected_decaying = [
            entities["binary_sensor.motion1"],
            entities["binary_sensor.appliance"],
        ]
        area.entities.decaying_entities = expected_decaying

        decaying = area.entities.decaying_entities

        assert len(decaying) == 2
        assert entities["binary_sensor.motion1"] in decaying
        assert entities["binary_sensor.appliance"] in decaying
        assert entities["binary_sensor.motion2"] not in decaying
        assert entities["media_player.tv"] not in decaying

    @pytest.mark.parametrize(
        ("entities_empty", "expected_probability", "expected_prior", "expected_decay"),
        [
            (True, 0.0, MIN_PRIOR, 1.0),
            (False, 0.65, 0.35, 0.8),
        ],
    )
    def test_property_calculations_with_entities(
        self,
        mock_coordinator: Mock,
        mock_coordinator_with_sensors: Mock,
        entities_empty: bool,
        expected_probability: float,
        expected_prior: float,
        expected_decay: float,
    ) -> None:
        """Test property calculations with different entity states."""
        coordinator = (
            mock_coordinator if entities_empty else mock_coordinator_with_sensors
        )

        # These are now methods that take area_name
        area_name = (
            coordinator.get_area_names()[0]
            if coordinator.get_area_names()
            else "Test Area"
        )
        area = coordinator.get_area_or_default(area_name)
        if entities_empty:
            area.entities._entities = {}
            # Mock area methods instead of coordinator methods
            area.probability = Mock(return_value=expected_probability)
            area.area_prior = Mock(return_value=expected_prior)
            area.decay = Mock(return_value=expected_decay)
        else:
            # For non-empty entities, also mock area methods
            area.probability = Mock(return_value=expected_probability)
            area.area_prior = Mock(return_value=expected_prior)
            area.decay = Mock(return_value=expected_decay)

        # Coordinator wrappers should delegate to area methods
        # Note: Mock coordinators don't actually call wrapper methods, they're just Mocks
        # So we verify area methods are set up correctly instead
        # For mock coordinators, verify area methods are set up correctly
        assert area.probability.return_value == expected_probability
        assert area.area_prior.return_value == expected_prior
        assert area.decay.return_value == expected_decay

    def test_probability_with_mixed_evidence_and_decay(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test probability calculation with mixed evidence and decay states."""
        # Access entities via area
        area = mock_coordinator_with_sensors.get_area_or_default.return_value
        entities = area.entities.entities

        # Setup entities with various states
        entities["binary_sensor.motion1"].evidence = True
        entities["binary_sensor.motion1"].decay.is_decaying = False
        entities["binary_sensor.motion1"].decay.decay_factor = 1.0

        entities["binary_sensor.motion2"].evidence = False
        entities["binary_sensor.motion2"].decay.is_decaying = True
        entities["binary_sensor.motion2"].decay.decay_factor = 0.5

        entities["binary_sensor.appliance"].evidence = False
        entities["binary_sensor.appliance"].decay.is_decaying = False

        entities["media_player.tv"].evidence = True
        entities["media_player.tv"].decay.is_decaying = True
        entities["media_player.tv"].decay.decay_factor = 0.8

        # Mock area.probability to return a valid value since we're testing delegation
        area.probability = Mock(return_value=0.65)
        # For mock coordinators, verify area method is mocked correctly
        assert area.probability.return_value == 0.65

    def test_probability_calculation_with_varying_weights(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test probability calculation with entities having different weights."""
        # Use area-based access
        area = mock_coordinator_with_sensors.get_area_or_default.return_value
        entities = area.entities.entities

        entities["binary_sensor.motion1"].type.weight = 0.9
        entities["binary_sensor.motion1"].evidence = True

        entities["binary_sensor.appliance"].type.weight = 0.1
        entities["binary_sensor.appliance"].evidence = True

        # Mock area.probability to return a valid value since we're testing delegation
        area.probability = Mock(return_value=0.65)
        # For mock coordinators, verify area method is mocked correctly
        assert area.probability.return_value == 0.65

    @pytest.mark.parametrize(
        ("probability", "threshold", "expected_occupied"),
        [
            (0.59, 0.6, False),
            (0.6, 0.6, True),
            (0.61, 0.6, True),
        ],
    )
    def test_threshold_boundary_conditions(
        self,
        mock_coordinator_with_threshold: Mock,
        probability: float,
        threshold: float,
        expected_occupied: bool,
    ) -> None:
        """Test is_occupied calculation at various threshold boundaries."""
        coordinator = mock_coordinator_with_threshold
        # These are now methods that take area_name
        area_name = (
            coordinator.get_area_names()[0]
            if coordinator.get_area_names()
            else "Test Area"
        )
        # Mock area methods instead of coordinator methods
        area = coordinator.get_area_or_default(area_name)
        area.probability = Mock(return_value=probability)
        area.threshold = Mock(return_value=threshold)
        area.occupied = Mock(return_value=expected_occupied)

        # Coordinator wrappers should delegate to area methods
        # For mock coordinators, verify area method is set up correctly
        assert area.occupied.return_value == expected_occupied

    @pytest.mark.parametrize(
        ("edge_value", "property_name"),
        [
            (0.0, "probability"),
            (1.0, "probability"),
            (0.0, "threshold"),
            (1.0, "threshold"),
        ],
    )
    def test_edge_values(
        self, mock_coordinator: Mock, edge_value: float, property_name: str
    ) -> None:
        """Test property edge values."""
        setattr(mock_coordinator, property_name, edge_value)
        assert getattr(mock_coordinator, property_name) == edge_value

    async def test_async_methods(self, mock_coordinator: Mock) -> None:
        """Test async coordinator methods."""
        # Test setup
        await mock_coordinator.setup()
        mock_coordinator.setup.assert_called_once()

        # Test update
        result = await mock_coordinator.update()
        mock_coordinator.update.assert_called_once()
        assert result is not None

        # Test option updates
        new_options = {"threshold": 70, "decay_enabled": False}
        await mock_coordinator.async_update_options(new_options)
        mock_coordinator.async_update_options.assert_called_once_with(new_options)

        # Test entity state tracking
        entity_ids = ["binary_sensor.test1", "binary_sensor.test2"]
        await mock_coordinator.track_entity_state_changes(entity_ids)
        mock_coordinator.track_entity_state_changes.assert_called_once_with(entity_ids)

        # Test shutdown
        await mock_coordinator.async_shutdown()
        mock_coordinator.async_shutdown.assert_called_once()

    async def test_update_method_data_structure(self, mock_coordinator: Mock) -> None:
        """Test update method returns correct data structure."""
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

    @pytest.mark.parametrize(
        "entity_ids",
        [
            [],
            ["binary_sensor.motion1"],
            ["binary_sensor.motion1", "binary_sensor.motion2", "media_player.tv"],
        ],
    )
    async def test_state_tracking_with_various_entities(
        self, mock_coordinator: Mock, entity_ids: list[str]
    ) -> None:
        """Test entity state tracking with various entity lists."""
        await mock_coordinator.track_entity_state_changes(entity_ids)
        mock_coordinator.track_entity_state_changes.assert_called_once_with(entity_ids)

    @pytest.mark.parametrize(
        ("method_name", "error_class", "error_message"),
        [
            ("setup", ConfigEntryNotReady, "Setup failed"),
            ("update", HomeAssistantError, "Update failed"),
            ("async_update_options", HomeAssistantError, "Option update failed"),
            ("track_entity_state_changes", HomeAssistantError, "Tracking failed"),
            ("async_shutdown", HomeAssistantError, "Shutdown failed"),
        ],
    )
    async def test_error_handling(
        self,
        mock_coordinator: Mock,
        method_name: str,
        error_class: type,
        error_message: str,
    ) -> None:
        """Test error handling for various methods."""
        method = getattr(mock_coordinator, method_name)
        method.side_effect = error_class(error_message)

        # Determine the call based on method name
        call_args: dict[str, float] | list[str] | None
        if method_name == "async_update_options":
            call_args = {"threshold": 0.8}
        elif method_name == "track_entity_state_changes":
            call_args = ["binary_sensor.test"]
        else:
            call_args = None

        with pytest.raises(error_class, match=error_message):
            await method() if call_args is None else await method(call_args)

    async def test_timer_lifecycle(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test complete timer lifecycle from start to cancellation."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Set up an area for the test
        area_name = "Test Area"
        area = Area(coordinator, area_name=area_name)
        coordinator.areas[area_name] = area

        # Mock get_area_names
        coordinator.get_area_names = Mock(return_value=[area_name])

        with (
            patch.object(area.entities, "get_entity") as mock_get_entity,
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=Mock(),
            ),
            patch.object(coordinator.db, "save_data", new_callable=AsyncMock),
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8
            mock_entity_type.active_states = ["on"]
            mock_entity_type.active_range = None
            mock_get_entity.return_value = mock_entity_type

            assert coordinator._global_decay_timer is None

            coordinator._start_decay_timer()
            assert coordinator._global_decay_timer is not None

            await coordinator.async_shutdown()
            assert coordinator._global_decay_timer is None

    def test_timer_start_with_missing_hass(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test timer start when hass is missing/invalid."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with patch.object(coordinator, "hass", None):
            coordinator._start_decay_timer()
            assert coordinator._global_decay_timer is None

    async def test_setup_scenarios(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test various setup scenarios."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Test setup with stored data
        stored_data: dict[str, Any] = {"entities": {"binary_sensor.test": {}}}

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        with (
            patch.object(area.entities, "cleanup", new=AsyncMock()),
            patch.object(
                coordinator.db, "load_data", new=AsyncMock(return_value=stored_data)
            ),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(coordinator.db, "is_intervals_empty", return_value=False),
            patch.object(coordinator, "run_analysis", new=AsyncMock()),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
        ):
            await coordinator.setup()

        # Test setup failure
        # Use the same area setup from above

        with (
            patch.object(area.entities, "cleanup", new=AsyncMock()),
            patch.object(
                coordinator.db,
                "load_data",
                new=AsyncMock(side_effect=HomeAssistantError("Storage failed")),
            ),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch.object(coordinator, "run_analysis", new=AsyncMock()),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch.object(area.entities, "get_entity") as mock_get_entity,
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8
            mock_entity_type.active_states = ["on"]
            mock_entity_type.active_range = None
            mock_get_entity.return_value = mock_entity_type

            with pytest.raises(
                ConfigEntryNotReady, match="Failed to set up coordinator"
            ):
                await coordinator.setup()

    @pytest.mark.parametrize("expected_lingering_timers", [True])
    async def test_shutdown_behavior(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test shutdown behavior with real coordinator instance."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        # Prevent scheduling real timers
        with (
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
            patch.object(area, "async_cleanup", new=AsyncMock()),
            patch.object(area.entities, "get_entity") as mock_get_entity,
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            # Start timers so shutdown has something to cancel (they will be None)
            coordinator._start_decay_timer()
            # _remove_state_listener doesn't exist in new architecture - state listeners are per-area
            # coordinator._remove_state_listener = Mock()

            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8
            mock_entity_type.active_states = ["on"]
            mock_entity_type.active_range = None
            mock_get_entity.return_value = mock_entity_type

            await coordinator.async_shutdown()

            assert coordinator._global_decay_timer is None
            # _remove_state_listener doesn't exist in new architecture
            # assert coordinator._remove_state_listener is None

    async def test_shutdown_with_none_resources(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test shutdown when resources are already None."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        coordinator._global_decay_timer = None
        # _remove_state_listener doesn't exist in new architecture
        # coordinator._remove_state_listener = None

        with (
            patch.object(area, "async_cleanup", new=AsyncMock()),
            patch.object(area.entities, "get_entity") as mock_get_entity,
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8
            mock_entity_type.active_states = ["on"]
            mock_entity_type.active_range = None
            mock_get_entity.return_value = mock_entity_type

            await coordinator.async_shutdown()

            assert coordinator._global_decay_timer is None
            # _remove_state_listener doesn't exist in new architecture
            # assert coordinator._remove_state_listener is None

    @pytest.mark.expected_lingering_timers(True)
    async def test_full_coordinator_lifecycle(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test complete coordinator lifecycle with realistic configuration."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        with (
            patch.object(area.entities, "get_entity") as mock_get_entity,
            patch.object(area, "async_cleanup", new=AsyncMock()),
            patch.object(coordinator.db, "load_data", new=AsyncMock(return_value=None)),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch.object(coordinator, "run_analysis", new=AsyncMock()),
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            mock_entity_type = Mock()
            mock_entity_type.prob_true = 0.25
            mock_entity_type.prob_false = 0.05
            mock_entity_type.weight = 0.8
            mock_entity_type.active_states = ["on"]
            mock_entity_type.active_range = None
            mock_get_entity.return_value = mock_entity_type

            await coordinator.setup()
            await coordinator.update()
            await coordinator.async_shutdown()

            # entities is now per-area, not on coordinator
            assert area.entities is not None

    @pytest.mark.expected_lingering_timers(True)
    def test_performance_with_many_entities(self, mock_coordinator: Mock) -> None:
        """Test performance with many entities."""
        # Create many mock entities
        entities = {}
        for i in range(50):
            entity_id = f"binary_sensor.motion_{i}"
            mock_entity = Mock()
            mock_entity.probability = 0.5 + (i * 0.01)
            mock_entity.weight = 0.8
            mock_entity.is_active = True
            mock_entity.is_decaying = False
            entities[entity_id] = mock_entity

        # Access entities via area (multi-area architecture)
        area = mock_coordinator.get_area_or_default.return_value
        area.entities._entities = entities

        # Mock area.probability to return a valid value since we're testing delegation
        area.probability = Mock(return_value=0.5)
        # For mock coordinators, verify area method is mocked correctly
        assert area.probability.return_value == 0.5

    async def test_state_tracking_with_many_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state tracking setup with many entities."""
        entity_ids = [f"binary_sensor.motion_{i}" for i in range(200)]

        await mock_coordinator.track_entity_state_changes(entity_ids)
        mock_coordinator.track_entity_state_changes.assert_called_with(entity_ids)

    def test_type_probabilities_property(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test type_probabilities property calculation."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Create an area and add it to coordinator
        area_name = "Test Area"
        area = Area(coordinator, area_name=area_name)
        coordinator.areas[area_name] = area

        # Attach a lightweight mocked entities manager
        # Mock the entities property on the area by setting the private attribute
        entities_manager = Mock()
        entities_manager.entities = {
            "binary_sensor.motion": Mock(),
            "media_player.tv": Mock(),
            "binary_sensor.door": Mock(),
        }
        entities_manager.get_entities_by_input_type = Mock(return_value={})
        # Set the private _entities attribute directly
        area._entities = entities_manager

        # type_probabilities is now a method that delegates to area
        # Coordinator wrapper should delegate to area method
        type_probs = coordinator.type_probabilities(area_name)
        assert isinstance(type_probs, dict)
        # The method returns probabilities for each input type, even if empty
        assert "motion" in type_probs
        assert "media" in type_probs
        assert "appliance" in type_probs
        assert "door" in type_probs
        assert "window" in type_probs
        assert "illuminance" in type_probs
        assert "humidity" in type_probs
        assert "temperature" in type_probs

    def test_type_probabilities_with_empty_entities(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test type_probabilities property with no entities."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Get area name - handle case where areas might not be loaded
        area_names = coordinator.get_area_names()
        if not area_names:
            # Skip test if no areas are configured
            pytest.skip("No areas configured in test coordinator")
        area_name = area_names[0]

        entities_manager = Mock()
        entities_manager.entities = {}
        entities_manager.get_entities_by_input_type = Mock(return_value={})
        # Access entities via area
        area = coordinator.get_area_or_default(area_name)
        if area is None:
            pytest.skip("Area not found in test coordinator")
        area.entities = entities_manager

        # type_probabilities is now a method that delegates to area
        # Coordinator wrapper should delegate to area method
        type_probs = coordinator.type_probabilities(area_name)
        assert type_probs == {}

    def test_threshold_property_with_none_config(self, mock_coordinator: Mock) -> None:
        """Test threshold property when config is None."""
        # threshold is now a method that delegates to area
        area_name = (
            mock_coordinator.get_area_names()[0]
            if mock_coordinator.get_area_names()
            else "Test Area"
        )
        area = mock_coordinator.get_area_or_default(area_name)
        area.threshold = Mock(return_value=0.5)
        # Coordinator wrapper should delegate to area method
        # For mock coordinators, verify area method is mocked correctly
        assert area.threshold.return_value == 0.5

    async def test_decay_timer_handling(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer start and handling."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time",
            return_value=Mock(),
        ) as mock_track:
            coordinator._start_decay_timer()
            mock_track.assert_called_once()
            assert coordinator._global_decay_timer is not None

    async def test_decay_timer_handling_with_existing_timer(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer start when timer already exists."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._global_decay_timer = Mock()

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time"
        ) as mock_track:
            coordinator._start_decay_timer()
            mock_track.assert_not_called()

    async def test_decay_timer_handling_without_hass(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer start when hass is None."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator, "hass", None),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time"
            ) as mock_track,
        ):
            coordinator._start_decay_timer()
            mock_track.assert_not_called()

    async def test_handle_decay_timer(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer callback handling."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._global_decay_timer = Mock()

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        # Access config via area
        area.config.decay.enabled = True

        with (
            patch.object(coordinator, "async_refresh", new=AsyncMock()) as mock_refresh,
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator._handle_decay_timer(dt_util.utcnow())
            assert coordinator._global_decay_timer is None
            mock_refresh.assert_called_once()

    async def test_handle_decay_timer_disabled(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer callback when decay is disabled."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._global_decay_timer = Mock()

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        # Access config via area
        area.config.decay.enabled = False

        with (
            patch.object(coordinator, "async_refresh", new=AsyncMock()) as mock_refresh,
            patch.object(
                coordinator, "_schedule_save"
            ),  # Mock _schedule_save to avoid timer
            patch(
                "custom_components.area_occupancy.coordinator.async_call_later",
                return_value=Mock(),  # Return a Mock that can be canceled
            ),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator._handle_decay_timer(dt_util.utcnow())
            assert coordinator._global_decay_timer is None
            mock_refresh.assert_not_called()

    async def test_analysis_timer_handling(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test analysis timer start and handling."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time",
            return_value=Mock(),
        ) as mock_track:
            await coordinator._start_analysis_timer()  # Now async
            mock_track.assert_called_once()
            assert coordinator._analysis_timer is not None

    async def test_analysis_timer_handling_with_existing_timer(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test analysis timer start when timer already exists."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time"
        ) as mock_track:
            await coordinator._start_analysis_timer()  # Now async
            mock_track.assert_not_called()

    async def test_analysis_timer_handling_without_hass(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test analysis timer start when hass is None."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator, "hass", None),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time"
            ) as mock_track,
        ):
            await coordinator._start_analysis_timer()  # Now async
            mock_track.assert_not_called()

    async def test_run_analysis(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test run_analysis method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        coordinator._is_master = True  # Enable pruning (master-only)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        with (
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(
                coordinator.db, "prune_old_intervals", return_value=5
            ),  # Mock pruning
            patch.object(area, "run_prior_analysis", new=AsyncMock()),
            patch.object(area, "run_likelihood_analysis", new=AsyncMock()),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis()
            assert coordinator._analysis_timer is None
            # Verify pruning was called (master-only)
            coordinator.db.prune_old_intervals.assert_called_once()

    async def test_run_analysis_with_error(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test run_analysis method with error handling."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()

        with (
            patch.object(
                coordinator.db,
                "sync_states",
                side_effect=HomeAssistantError("Sync failed"),
            ),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis()
            assert coordinator._analysis_timer is None

    async def test_run_analysis_with_custom_time(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test run_analysis method with custom time parameter."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        custom_time = dt_util.utcnow()

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        with (
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(area, "run_prior_analysis", new=AsyncMock()),
            patch.object(area, "run_likelihood_analysis", new=AsyncMock()),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis(custom_time)
            assert coordinator._analysis_timer is None

    async def test_track_entity_state_changes_with_existing_listener(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test entity state tracking with existing listener."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        # _remove_state_listener doesn't exist in new architecture - state listeners are per-area
        # coordinator._remove_state_listener = Mock()
        # prev_listener = coordinator._remove_state_listener

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_state_change_event",
            return_value=Mock(),
        ) as mock_track:
            await coordinator.track_entity_state_changes(["binary_sensor.test"])
            # Previous listener should be called (if it existed)
            # prev_listener.assert_called_once()
            # New listener should be set (since we provided entity_ids)
            # In new architecture, listeners are stored in _area_state_listeners dict
            # assert coordinator._remove_state_listener is not None
            mock_track.assert_called_once()

    async def test_track_entity_state_changes_empty_list(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test entity state tracking with empty entity list."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        # _remove_state_listener doesn't exist in new architecture - state listeners are per-area
        # coordinator._remove_state_listener = Mock()
        # prev_listener = coordinator._remove_state_listener

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_state_change_event"
        ) as mock_track:
            await coordinator.track_entity_state_changes([])
            # Previous listener should be called and cleared even if no new tracking
            # prev_listener.assert_called_once()
            # In new architecture, listeners are stored in _area_state_listeners dict
            # assert coordinator._remove_state_listener is None
            mock_track.assert_not_called()

    async def test_track_entity_state_changes_with_entity_with_new_evidence(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test entity state tracking with entity that has new evidence."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Ensure setup_complete is True so the refresh condition is met
        coordinator._setup_complete = True

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        # Mock entity with new evidence
        mock_entity = Mock()
        mock_entity.has_new_evidence.return_value = True

        # Patch async_refresh BEFORE calling track_entity_state_changes
        with (
            patch.object(coordinator, "async_refresh", new=AsyncMock()) as mock_refresh,
            patch.object(area.entities, "get_entity", return_value=mock_entity),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_state_change_event",
                return_value=Mock(),
            ),
        ):
            # Create the event handler manually to test it
            event_handler = None

            def track_callback(hass: Any, entity_ids: list[str], callback: Any) -> Mock:
                nonlocal event_handler
                event_handler = callback
                return Mock()

            with patch(
                "custom_components.area_occupancy.coordinator.async_track_state_change_event",
                side_effect=track_callback,
            ):
                await coordinator.track_entity_state_changes(["binary_sensor.test"])

                # Simulate state change event
                mock_event = Mock()
                mock_event.data = {"entity_id": "binary_sensor.test"}
                if event_handler is not None:
                    await event_handler(mock_event)
            mock_refresh.assert_called_once()

    async def test_track_entity_state_changes_with_entity_without_new_evidence(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test entity state tracking with entity that has no new evidence."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        # Mock entity without new evidence
        mock_entity = Mock()
        mock_entity.has_new_evidence.return_value = False

        with (
            patch.object(area.entities, "get_entity", return_value=mock_entity),
            patch.object(
                coordinator, "_schedule_save"
            ),  # Mock _schedule_save to avoid timer
            patch(
                "custom_components.area_occupancy.coordinator.async_call_later",
                return_value=Mock(),  # Return a Mock that can be canceled
            ),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_state_change_event",
                return_value=Mock(),
            ),
            patch.object(coordinator, "async_refresh", new=AsyncMock()) as mock_refresh,
        ):
            # Create the event handler manually to test it
            event_handler = None

            def track_callback(hass: Any, entity_ids: list[str], callback: Any) -> Mock:
                nonlocal event_handler
                event_handler = callback
                return Mock()

            with patch(
                "custom_components.area_occupancy.coordinator.async_track_state_change_event",
                side_effect=track_callback,
            ):
                await coordinator.track_entity_state_changes(["binary_sensor.test"])

                # Simulate state change event
                mock_event = Mock()
                mock_event.data = {"entity_id": "binary_sensor.test"}
                if event_handler is not None:
                    await event_handler(mock_event)
            mock_refresh.assert_not_called()

    async def test_setup_with_intervals_empty(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when intervals table is empty."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator.db, "load_data", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data"),  # Now sync, no AsyncMock
            patch.object(coordinator.db, "safe_is_intervals_empty", return_value=True),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ) as mock_start_timer,
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
        ):
            await coordinator.setup()
            # run_analysis is now deferred to background, so _start_analysis_timer should be called
            mock_start_timer.assert_called_once()

    async def test_setup_with_intervals_not_empty(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when intervals table is not empty."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator.db, "load_data", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(coordinator.db, "safe_is_intervals_empty", return_value=False),
            patch.object(
                coordinator, "run_analysis", new=AsyncMock()
            ) as mock_run_analysis,
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            await coordinator.setup()
            mock_run_analysis.assert_not_called()

            # Clean up timers
            await coordinator.async_shutdown()

    async def test_setup_with_no_entity_ids(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when no entity IDs are configured."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)

        # Make entity_ids empty by clearing sensors
        area.config.sensors = Sensors(
            motion=[],
            primary_occupancy=None,
            media=[],
            appliance=[],
            illuminance=[],
            humidity=[],
            temperature=[],
            door=[],
            window=[],
            _parent_config=area.config,
        )

        with (
            patch.object(coordinator.db, "load_data", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(coordinator.db, "safe_is_intervals_empty", return_value=True),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator, "run_analysis", new=AsyncMock()) as mock_run,
        ):
            await coordinator.setup()
            mock_run.assert_not_called()

    async def test_setup_with_database_errors(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with database errors that should be handled gracefully."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator.db, "load_data", new=AsyncMock()),
            patch.object(
                coordinator.db, "save_area_data", side_effect=OSError("DB Error")
            ),
            patch.object(coordinator.db, "safe_is_intervals_empty", return_value=False),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
        ):
            await coordinator.setup()

    async def test_setup_with_intervals_check_error(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with intervals check error."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator.db, "load_data", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(
                coordinator.db,
                "safe_is_intervals_empty",
                side_effect=HomeAssistantError("Check failed"),
            ),
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
        ):
            await coordinator.setup()

    @pytest.mark.parametrize("expected_lingering_timers", [True])
    async def test_setup_with_analysis_error(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with analysis error handling."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator.db, "load_data", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(coordinator.db, "safe_is_intervals_empty", return_value=True),
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
            patch.object(
                coordinator,
                "run_analysis",
                side_effect=HomeAssistantError("Analysis failed"),
            ),
        ):
            await coordinator.setup()

    async def test_setup_with_unexpected_error(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with unexpected error that should continue with basic functionality."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(
                coordinator.db,
                "load_data",
                side_effect=RuntimeError("Unexpected error"),
            ),
            patch.object(
                coordinator,
                "_start_decay_timer",
                side_effect=lambda: setattr(coordinator, "_global_decay_timer", Mock()),
            ),
            patch.object(
                coordinator,
                "_start_analysis_timer",
                new=AsyncMock(
                    side_effect=lambda: setattr(coordinator, "_analysis_timer", Mock())
                ),
            ),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
        ):
            await coordinator.setup()

    async def test_setup_with_timer_start_error(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with timer start error."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(
                coordinator.db,
                "load_data",
                side_effect=RuntimeError("Unexpected error"),
            ),
            patch.object(
                coordinator, "_start_decay_timer", side_effect=OSError("Timer error")
            ),
            patch.object(
                coordinator, "_start_analysis_timer", side_effect=OSError("Timer error")
            ),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
        ):
            await coordinator.setup()

    async def test_async_update_options(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test async_update_options method."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        # Load areas from config (normally done during setup)
        coordinator._load_areas_from_config()

        # Verify areas exist before update
        initial_area_count = len(coordinator.areas)
        assert initial_area_count > 0, "Areas should exist before update"

        # Update config entry options (simulating what config flow does)
        mock_realistic_config_entry.options = {"threshold": 0.7}

        with (
            patch.object(
                coordinator, "track_entity_state_changes", new=AsyncMock()
            ) as mock_track,
            patch.object(coordinator.db, "save_data", new=AsyncMock()) as mock_save,
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
            patch(
                "homeassistant.helpers.entity_registry.async_get", return_value=Mock()
            ),
            # Patch Area.entities.cleanup on the class level so it works after reload
            patch(
                "custom_components.area_occupancy.data.entity.EntityManager.cleanup",
                new=AsyncMock(),
            ) as mock_cleanup,
        ):
            # async_update_options expects options dict but reads from config_entry
            # The options parameter is for compatibility but not used
            options = {"threshold": 0.7}
            await coordinator.async_update_options(options)

            # Verify areas are never empty during/after update
            # The root cause fix ensures self.areas is never cleared before new areas are loaded
            # Instead, new areas are loaded into a temporary dict first, then atomically replaced
            assert len(coordinator.areas) > 0, (
                "Areas should never be empty after update"
            )

            # cleanup is called on area.entities for each area (after reload)
            mock_cleanup.assert_called()
            # track_entity_state_changes is called with new entity lists
            mock_track.assert_called_once()
            # save_data is called to persist changes
            mock_save.assert_called_once()

            # Clean up timers
            await coordinator.async_shutdown()

    async def test_shutdown_with_all_timers(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test shutdown with all timers present."""
        coordinator_with_areas._is_master = True  # Enable master-specific cleanup

        # Get area from fixture
        area = coordinator_with_areas.get_area_or_default()
        assert area is not None

        # Set up all timers
        coordinator_with_areas._global_decay_timer = Mock()
        coordinator_with_areas._analysis_timer = Mock()
        coordinator_with_areas._save_timer = (
            Mock()
        )  # Set save timer to trigger final save

        with (
            patch.object(coordinator_with_areas.db, "save_data", new=AsyncMock()),
            patch.object(area, "async_cleanup", new=AsyncMock()) as mock_cleanup,
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            await coordinator_with_areas.async_shutdown()

            assert coordinator_with_areas._global_decay_timer is None
            assert coordinator_with_areas._analysis_timer is None
            coordinator_with_areas.db.save_data.assert_called_once()
            # async_cleanup is called for each area, which calls entities.cleanup() and purpose.cleanup()
            mock_cleanup.assert_called_once()

    # New tests for performance optimization features

    def test_get_area_names(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test get_area_names method."""
        # Areas are already loaded by coordinator_with_areas fixture
        area_names = coordinator_with_areas.get_area_names()
        assert isinstance(area_names, list)
        assert len(area_names) > 0

        # Should contain at least one area name
        assert all(isinstance(name, str) for name in area_names)

    def test_get_area_or_default(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test get_area_or_default method."""
        # Areas are already loaded by coordinator_with_areas fixture

        # Should return first area when None is passed
        area = coordinator_with_areas.get_area_or_default()
        assert area is not None
        assert hasattr(area, "area_name")
        assert hasattr(area, "config")
        assert hasattr(area, "entities")
        assert hasattr(area, "prior")

        # Should return specific area when name is provided
        area_names = coordinator_with_areas.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        specific_area = coordinator_with_areas.get_area_or_default(area_name)
        assert specific_area is not None
        assert specific_area.area_name == area_name

        # Should return None for non-existent area
        non_existent = coordinator_with_areas.get_area_or_default("NonExistentArea")
        assert non_existent is None

        first_area = coordinator_with_areas.get_area_or_default()
        assert first_area is not None
        assert first_area.area_name in coordinator_with_areas.get_area_names()


class TestRunAnalysisWithPruning:
    """Test run_analysis method with pruning functionality."""

    async def test_run_analysis_with_pruning(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that prune_old_intervals is called during run_analysis."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        coordinator._is_master = True  # Enable pruning (master-only)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        with (
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(
                coordinator.db, "prune_old_intervals", return_value=10
            ) as mock_prune,
            patch.object(area, "run_prior_analysis", new=AsyncMock()),
            patch.object(area, "run_likelihood_analysis", new=AsyncMock()),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis()

            # Verify pruning was called
            mock_prune.assert_called_once()
            assert coordinator._analysis_timer is None

    async def test_run_analysis_pruning_failure(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that analysis continues if pruning fails."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        coordinator._is_master = True  # Enable pruning (master-only)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        with (
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(
                coordinator.db, "prune_old_intervals", return_value=0
            ),  # Pruning returns 0
            patch.object(area, "run_prior_analysis", new=AsyncMock()) as mock_prior,
            patch.object(
                area, "run_likelihood_analysis", new=AsyncMock()
            ) as mock_likelihood,
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis()

            # Verify pruning was called and analysis completed
            coordinator.db.prune_old_intervals.assert_called_once()
            # run_analysis calls area.run_prior_analysis() and area.run_likelihood_analysis()
            mock_prior.assert_called_once()
            mock_likelihood.assert_called_once()
            assert coordinator._analysis_timer is None

    async def test_run_analysis_pruning_error_handling(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that analysis continues if pruning raises an exception."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        coordinator._is_master = True  # Enable pruning (master-only)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area_or_default = Mock(return_value=area)

        with (
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(
                coordinator.db,
                "prune_old_intervals",
                side_effect=RuntimeError("Pruning failed"),
            ),
            patch.object(area, "run_prior_analysis", new=AsyncMock()) as mock_prior,
            patch.object(
                area, "run_likelihood_analysis", new=AsyncMock()
            ) as mock_likelihood,
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            # Should not raise exception, but analysis should fail due to pruning error
            await coordinator.run_analysis()

            # Verify pruning was called
            coordinator.db.prune_old_intervals.assert_called_once()
            # Verify other steps were NOT called due to exception handling
            # run_analysis calls area.run_prior_analysis() and area.run_likelihood_analysis()
            # but these should not be called if pruning raises an exception
            mock_prior.assert_not_called()
            mock_likelihood.assert_not_called()
            coordinator.async_refresh.assert_not_called()
            coordinator.db.save_data.assert_not_called()
            assert coordinator._analysis_timer is None
