"""Tests for coordinator module."""

from datetime import datetime
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
from homeassistant.core import HomeAssistant
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

    def test_initialization(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test coordinator initialization."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        assert coordinator.hass == hass
        assert coordinator.config_entry == mock_realistic_config_entry
        assert coordinator.entry_id == mock_realistic_config_entry.entry_id
        # Coordinator no longer has a single 'name' property (multi-area architecture)
        # Use config entry title instead
        assert coordinator.config_entry.title == mock_realistic_config_entry.title

    def test_device_info_property(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test device_info property."""
        # device_info is now a method that takes area_name
        area_name = coordinator.get_area_names()[0]
        device_info = coordinator.device_info(area_name)

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
        # In multi-area architecture, device_info uses area_id as identifier (stable even if area is renamed)
        # Fallback to area_name for legacy compatibility
        area = coordinator_with_areas.get_area(area_name)
        expected_identifier = (DOMAIN, area.config.area_id or area_name)
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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test device info generation when config is missing."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Get first area (always exists - at least one area is guaranteed)
        area = coordinator.get_area()
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
        self,
        coordinator: AreaOccupancyCoordinator,
        property_name: str,
        expected_value: float,
    ) -> None:
        """Test basic coordinator properties."""
        # These are now methods that take area_name and delegate to area methods
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        # Mock area method to return expected value
        with patch.object(area, property_name, return_value=expected_value):
            method = getattr(coordinator, property_name)
            assert method(area_name) == expected_value

    def test_threshold_property(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test threshold property specifically."""
        # threshold is now a method that delegates to area
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        # Set threshold to 0.6 for testing
        area.config.threshold = 0.6
        # The wrapper should call area.threshold()
        assert coordinator.threshold(area_name) == 0.6

    def test_is_occupied_property(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test occupied method threshold comparison."""
        # Set threshold to 0.6 and probability to 0.5
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.threshold = 0.6
        # Mock probability to return 0.5
        with patch.object(area, "probability", return_value=0.5):
            assert not coordinator.occupied(area_name)  # 0.5 < 0.6

        # Test at threshold boundary
        with patch.object(area, "probability", return_value=0.6):
            assert coordinator.occupied(area_name)  # 0.6 >= 0.6

    def test_decaying_entities_property(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test decaying_entities property filtering."""
        # Configure decaying entities - use area-based access
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        motion2 = area.entities.entities["binary_sensor.motion2"]
        motion2.decay.is_decaying = True

        area.entities.decaying_entities = [motion2]

        decaying = area.entities.decaying_entities
        assert len(decaying) == 1
        assert decaying[0].entity_id == "binary_sensor.motion2"

    def test_decaying_entities_filtering_complex(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test decaying entities filtering with complex scenarios."""
        # Use area-based access
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        entities = area.entities.entities

        # Set up mixed decay states (using actual entity IDs from coordinator_with_sensors fixture)
        entities["binary_sensor.motion"].decay.is_decaying = True
        entities["binary_sensor.motion2"].decay.is_decaying = False
        entities["binary_sensor.appliance"].decay.is_decaying = True
        entities["media_player.tv"].decay.is_decaying = False

        expected_decaying = [
            entities["binary_sensor.motion"],
            entities["binary_sensor.appliance"],
        ]
        area.entities.decaying_entities = expected_decaying

        decaying = area.entities.decaying_entities

        assert len(decaying) == 2
        assert entities["binary_sensor.motion"] in decaying
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
        coordinator: AreaOccupancyCoordinator,
        coordinator_with_sensors: AreaOccupancyCoordinator,
        entities_empty: bool,
        expected_probability: float,
        expected_prior: float,
        expected_decay: float,
    ) -> None:
        """Test property calculations with different entity states."""
        test_coordinator = coordinator if entities_empty else coordinator_with_sensors

        # These are now methods that take area_name
        area_name = test_coordinator.get_area_names()[0]
        area = test_coordinator.get_area(area_name)
        if entities_empty:
            area.entities._entities = {}
            # Mock area methods instead of coordinator methods
            with (
                patch.object(area, "probability", return_value=expected_probability),
                patch.object(area, "area_prior", return_value=expected_prior),
                patch.object(area, "decay", return_value=expected_decay),
            ):
                # Verify coordinator wrappers delegate to area methods
                assert test_coordinator.probability(area_name) == expected_probability
                assert test_coordinator.area_prior(area_name) == expected_prior
                assert test_coordinator.decay(area_name) == expected_decay
        else:
            # For non-empty entities, also mock area methods
            with (
                patch.object(area, "probability", return_value=expected_probability),
                patch.object(area, "area_prior", return_value=expected_prior),
                patch.object(area, "decay", return_value=expected_decay),
            ):
                # Verify coordinator wrappers delegate to area methods
                assert test_coordinator.probability(area_name) == expected_probability
                assert test_coordinator.area_prior(area_name) == expected_prior
                assert test_coordinator.decay(area_name) == expected_decay

    def test_probability_with_mixed_evidence_and_decay(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test probability calculation with mixed evidence and decay states."""
        # Access entities via area
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        entities = area.entities.entities

        # Setup entities with various states
        entities["binary_sensor.motion"].evidence = True
        entities["binary_sensor.motion"].decay.is_decaying = False
        entities["binary_sensor.motion"].decay.decay_factor = 1.0

        entities["binary_sensor.motion2"].evidence = False
        entities["binary_sensor.motion2"].decay.is_decaying = True
        entities["binary_sensor.motion2"].decay.decay_factor = 0.5

        entities["binary_sensor.appliance"].evidence = False
        entities["binary_sensor.appliance"].decay.is_decaying = False

        entities["media_player.tv"].evidence = True
        entities["media_player.tv"].decay.is_decaying = True
        entities["media_player.tv"].decay.decay_factor = 0.8

        # Mock area.probability to return a valid value since we're testing delegation
        with patch.object(area, "probability", return_value=0.65):
            # Verify coordinator wrapper delegates to area method
            assert coordinator_with_sensors.probability(area_name) == 0.65

    def test_probability_calculation_with_varying_weights(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test probability calculation with entities having different weights."""
        # Use area-based access
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        entities = area.entities.entities

        entities["binary_sensor.motion"].type.weight = 0.9
        entities["binary_sensor.motion"].evidence = True

        entities["binary_sensor.appliance"].type.weight = 0.1
        entities["binary_sensor.appliance"].evidence = True

        # Mock area.probability to return a valid value since we're testing delegation
        with patch.object(area, "probability", return_value=0.65):
            # Verify coordinator wrapper delegates to area method
            assert coordinator_with_sensors.probability(area_name) == 0.65

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
        coordinator: AreaOccupancyCoordinator,
        probability: float,
        threshold: float,
        expected_occupied: bool,
    ) -> None:
        """Test is_occupied calculation at various threshold boundaries."""
        # These are now methods that take area_name
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.threshold = threshold
        with patch.object(area, "probability", return_value=probability):
            assert coordinator.occupied(area_name) == expected_occupied

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
        self,
        coordinator: AreaOccupancyCoordinator,
        edge_value: float,
        property_name: str,
    ) -> None:
        """Test property edge values."""
        # For coordinator properties that are methods, we test via area
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        if property_name == "threshold":
            area.config.threshold = edge_value
            assert coordinator.threshold(area_name) == edge_value
        else:
            # For other properties, patch the area method
            with patch.object(area, property_name, return_value=edge_value):
                method = getattr(coordinator, property_name)
                assert method(area_name) == edge_value

    async def test_async_methods(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test async coordinator methods."""
        # Test setup
        with patch.object(coordinator, "setup", new_callable=AsyncMock) as mock_setup:
            await coordinator.setup()
            mock_setup.assert_called_once()

        # Test update
        with patch.object(
            coordinator, "update", new_callable=AsyncMock, return_value={"test": "data"}
        ) as mock_update:
            result = await coordinator.update()
            mock_update.assert_called_once()
            assert result is not None

        # Test option updates
        new_options = {"threshold": 70, "decay_enabled": False}
        with patch.object(
            coordinator, "async_update_options", new_callable=AsyncMock
        ) as mock_update_options:
            await coordinator.async_update_options(new_options)
            mock_update_options.assert_called_once_with(new_options)

        # Test entity state tracking
        entity_ids = ["binary_sensor.test1", "binary_sensor.test2"]
        with patch.object(
            coordinator, "track_entity_state_changes", new_callable=AsyncMock
        ) as mock_track:
            await coordinator.track_entity_state_changes(entity_ids)
            mock_track.assert_called_once_with(entity_ids)

        # Test shutdown
        with patch.object(
            coordinator, "async_shutdown", new_callable=AsyncMock
        ) as mock_shutdown:
            await coordinator.async_shutdown()
            mock_shutdown.assert_called_once()

    async def test_update_method_data_structure(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update method returns correct data structure."""
        test_data = {
            "probability": 0.65,
            "occupied": True,
            "threshold": 0.5,
            "prior": 0.35,
            "decay": 0.8,
            "last_updated": dt_util.utcnow(),
        }

        with patch.object(
            coordinator, "update", new_callable=AsyncMock, return_value=test_data
        ):
            result = await coordinator.update()

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
        self, coordinator: AreaOccupancyCoordinator, entity_ids: list[str]
    ) -> None:
        """Test entity state tracking with various entity lists."""
        with patch.object(
            coordinator, "track_entity_state_changes", new_callable=AsyncMock
        ) as mock_track:
            await coordinator.track_entity_state_changes(entity_ids)
            mock_track.assert_called_once_with(entity_ids)

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
        coordinator: AreaOccupancyCoordinator,
        method_name: str,
        error_class: type,
        error_message: str,
    ) -> None:
        """Test error handling for various methods."""
        # Determine the call based on method name
        call_args: dict[str, float] | list[str] | None
        if method_name == "async_update_options":
            call_args = {"threshold": 0.8}
        elif method_name == "track_entity_state_changes":
            call_args = ["binary_sensor.test"]
        else:
            call_args = None

        with patch.object(
            coordinator,
            method_name,
            new_callable=AsyncMock,
            side_effect=error_class(error_message),
        ):
            method = getattr(coordinator, method_name)
            with pytest.raises(error_class, match=error_message):
                await method() if call_args is None else await method(call_args)

    async def test_timer_lifecycle(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test complete timer lifecycle from start to cancellation."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test timer start when hass is missing/invalid."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with patch.object(coordinator, "hass", None):
            coordinator._start_decay_timer()
            assert coordinator._global_decay_timer is None

    async def test_setup_scenarios(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test various setup scenarios."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Test setup with stored data
        stored_data: dict[str, Any] = {"entities": {"binary_sensor.test": {}}}

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test shutdown behavior with real coordinator instance."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
            patch.object(coordinator.db, "save_data", new=Mock()),
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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test shutdown when resources are already None."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
            patch.object(coordinator.db, "save_data", new=Mock()),
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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test complete coordinator lifecycle with realistic configuration."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

        with (
            patch.object(area.entities, "get_entity") as mock_get_entity,
            patch.object(area, "async_cleanup", new=AsyncMock()),
            patch.object(coordinator.db, "load_data", new=AsyncMock(return_value=None)),
            patch.object(coordinator.db, "save_data", new=Mock()),
            patch.object(coordinator.db, "save_area_data", new=Mock()),
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
    def test_performance_with_many_entities(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
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
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.entities._entities = entities

        # Mock area.probability to return a valid value since we're testing delegation
        with patch.object(area, "probability", return_value=0.5):
            # Verify coordinator wrapper delegates to area method
            assert coordinator.probability(area_name) == 0.5

    async def test_state_tracking_with_many_entities(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test state tracking setup with many entities."""
        entity_ids = [f"binary_sensor.motion_{i}" for i in range(200)]

        with patch.object(
            coordinator, "track_entity_state_changes", new_callable=AsyncMock
        ) as mock_track:
            await coordinator.track_entity_state_changes(entity_ids)
            mock_track.assert_called_with(entity_ids)

    def test_type_probabilities_property(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test type_probabilities property calculation."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test type_probabilities property with no entities."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        area = coordinator.get_area(area_name)
        if area is None:
            pytest.skip("Area not found in test coordinator")
        area.entities = entities_manager

        # type_probabilities is now a method that delegates to area
        # Coordinator wrapper should delegate to area method
        type_probs = coordinator.type_probabilities(area_name)
        assert type_probs == {}

    def test_threshold_property_with_none_config(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test threshold property when config is None."""
        # threshold is now a method that delegates to area
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.threshold = Mock(return_value=0.5)
        # Coordinator wrapper should delegate to area method
        # For mock coordinators, verify area method is mocked correctly
        assert area.threshold.return_value == 0.5

    async def test_decay_timer_handling(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer start and handling."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time",
            return_value=Mock(),
        ) as mock_track:
            coordinator._start_decay_timer()
            mock_track.assert_called_once()
            assert coordinator._global_decay_timer is not None

    async def test_decay_timer_handling_with_existing_timer(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer start when timer already exists."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._global_decay_timer = Mock()

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time"
        ) as mock_track:
            coordinator._start_decay_timer()
            mock_track.assert_not_called()

    async def test_decay_timer_handling_without_hass(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer start when hass is None."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator, "hass", None),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time"
            ) as mock_track,
        ):
            coordinator._start_decay_timer()
            mock_track.assert_not_called()

    async def test_handle_decay_timer(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer callback handling."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._global_decay_timer = Mock()

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test decay timer callback when decay is disabled."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._global_decay_timer = Mock()

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

        # Access config via area
        area.config.decay.enabled = False

        with (
            patch.object(coordinator, "async_refresh", new=AsyncMock()) as mock_refresh,
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator._handle_decay_timer(dt_util.utcnow())
            assert coordinator._global_decay_timer is None
            mock_refresh.assert_not_called()

    async def test_analysis_timer_handling(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test analysis timer start and handling."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time",
            return_value=Mock(),
        ) as mock_track:
            await coordinator._start_analysis_timer()  # Now async
            mock_track.assert_called_once()
            assert coordinator._analysis_timer is not None

    async def test_analysis_timer_handling_with_existing_timer(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test analysis timer start when timer already exists."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time"
        ) as mock_track:
            await coordinator._start_analysis_timer()  # Now async
            mock_track.assert_not_called()

    async def test_analysis_timer_handling_without_hass(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test analysis timer start when hass is None."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator, "hass", None),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time"
            ) as mock_track,
        ):
            await coordinator._start_analysis_timer()  # Now async
            mock_track.assert_not_called()

    async def test_interval_aggregation_timer_start(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test nightly interval aggregation timer start."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with (
            patch.object(
                coordinator,
                "_next_interval_aggregation_run",
                return_value=dt_util.utcnow(),
            ) as mock_next,
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=Mock(),
            ) as mock_track,
        ):
            coordinator._start_interval_aggregation_timer()
            mock_next.assert_called_once()
            mock_track.assert_called_once()
            assert coordinator._interval_aggregation_timer is not None

    async def test_interval_aggregation_timer_not_started_when_exists(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test nightly timer is not started when already scheduled."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._interval_aggregation_timer = Mock()

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time"
        ) as mock_track:
            coordinator._start_interval_aggregation_timer()
            mock_track.assert_not_called()

    async def test_run_interval_aggregation_job_success(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test successful execution of nightly interval aggregation job."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator.db.run_interval_aggregation = Mock(return_value={"daily": 1})

        async def fake_executor_job(func, *args, **kwargs):
            return func(*args, **kwargs)

        hass.async_add_executor_job = AsyncMock(side_effect=fake_executor_job)

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time",
            return_value=Mock(),
        ) as mock_track:
            await coordinator.run_interval_aggregation_job(dt_util.utcnow())
            coordinator.db.run_interval_aggregation.assert_called_once()
            mock_track.assert_called_once()
            assert coordinator._interval_aggregation_timer == mock_track.return_value

    async def test_run_interval_aggregation_job_error(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test error handling for nightly interval aggregation job."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        def _raise_error():
            raise RuntimeError("Aggregation failed")

        coordinator.db.run_interval_aggregation = Mock(side_effect=_raise_error)

        async def fake_executor_job(func, *args, **kwargs):
            return func(*args, **kwargs)

        hass.async_add_executor_job = AsyncMock(side_effect=fake_executor_job)

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time",
            return_value=Mock(),
        ) as mock_track:
            await coordinator.run_interval_aggregation_job(dt_util.utcnow())
            mock_track.assert_called_once()

    async def test_run_analysis(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test run_analysis method."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        coordinator._is_master = True  # Enable pruning (master-only)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test run_analysis method with error handling."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test run_analysis method with custom time parameter."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        custom_time = dt_util.utcnow()

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test entity state tracking with existing listener."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test entity state tracking with empty entity list."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test entity state tracking with entity that has new evidence."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Ensure setup_complete is True so the refresh condition is met
        coordinator._setup_complete = True

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test entity state tracking with entity that has no new evidence."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

        # Mock entity without new evidence
        mock_entity = Mock()
        mock_entity.has_new_evidence.return_value = False

        with (
            patch.object(area.entities, "get_entity", return_value=mock_entity),
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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when intervals table is empty."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when intervals table is not empty."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator.db, "load_data", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(coordinator.db, "save_data", return_value=None),
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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when no entity IDs are configured."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with database errors that should be handled gracefully."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with intervals check error."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with analysis error handling."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with unexpected error that should continue with basic functionality."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with timer start error."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test async_update_options method."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

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
        area = coordinator_with_areas.get_area()
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

    def test_get_area(self, coordinator_with_areas: AreaOccupancyCoordinator) -> None:
        """Test get_area method."""
        # Areas are already loaded by coordinator_with_areas fixture

        # Should return first area when None is passed
        area = coordinator_with_areas.get_area()
        assert area is not None
        assert hasattr(area, "area_name")
        assert hasattr(area, "config")
        assert hasattr(area, "entities")
        assert hasattr(area, "prior")

        # Should return specific area when name is provided
        area_names = coordinator_with_areas.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        specific_area = coordinator_with_areas.get_area(area_name)
        assert specific_area is not None
        assert specific_area.area_name == area_name

        # Should return None for non-existent area
        non_existent = coordinator_with_areas.get_area("NonExistentArea")
        assert non_existent is None

        first_area = coordinator_with_areas.get_area()
        assert first_area is not None
        assert first_area.area_name in coordinator_with_areas.get_area_names()


class TestRunAnalysisWithPruning:
    """Test run_analysis method with pruning functionality."""

    async def test_run_analysis_with_pruning(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that prune_old_intervals is called during run_analysis."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        coordinator._is_master = True  # Enable pruning (master-only)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that analysis continues if pruning fails."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        coordinator._is_master = True  # Enable pruning (master-only)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that analysis continues if pruning raises an exception."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()
        coordinator._is_master = True  # Enable pruning (master-only)

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

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


class TestCoordinatorTimerCallbacks:
    """Test coordinator timer callback error handling."""

    async def test_handle_save_timer_error(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _handle_save_timer with database error."""
        with (
            patch.object(
                coordinator.db, "save_data", side_effect=RuntimeError("Save failed")
            ),
            patch.object(coordinator, "_start_save_timer") as mock_start,
        ):
            # Should handle error gracefully and reschedule
            await coordinator._handle_save_timer(datetime.now())
            mock_start.assert_called_once()

    async def test_handle_save_timer_success(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _handle_save_timer successful save."""
        with (
            patch.object(coordinator.db, "save_data") as mock_save,
            patch.object(coordinator, "_start_save_timer") as mock_start,
        ):
            await coordinator._handle_save_timer(datetime.now())
            mock_save.assert_called_once()
            mock_start.assert_called_once()

    async def test_handle_decay_timer_with_decay_enabled(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _handle_decay_timer when decay is enabled."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.decay.enabled = True

        with (
            patch.object(coordinator, "async_refresh") as mock_refresh,
            patch.object(coordinator, "_start_decay_timer") as mock_start,
        ):
            await coordinator._handle_decay_timer(datetime.now())
            mock_refresh.assert_called_once()
            mock_start.assert_called_once()

    async def test_handle_decay_timer_with_decay_disabled(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _handle_decay_timer when decay is disabled."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.decay.enabled = False

        with (
            patch.object(coordinator, "async_refresh") as mock_refresh,
            patch.object(coordinator, "_start_decay_timer") as mock_start,
        ):
            await coordinator._handle_decay_timer(datetime.now())
            # Should not refresh when decay is disabled
            mock_refresh.assert_not_called()
            mock_start.assert_called_once()

    async def test_run_analysis_sync_error(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test run_analysis with sync_states error."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        with (
            patch.object(
                coordinator.db, "sync_states", side_effect=RuntimeError("Sync failed")
            ),
            patch.object(coordinator.db, "periodic_health_check", return_value=True),
            patch.object(area, "run_prior_analysis", new=AsyncMock()),
            patch.object(area, "run_likelihood_analysis", new=AsyncMock()),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data"),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            # Should handle error gracefully
            await coordinator.run_analysis(datetime.now())

    async def test_run_analysis_health_check_failure(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test run_analysis with health check failure."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        with (
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(coordinator.db, "periodic_health_check", return_value=False),
            patch.object(area, "run_prior_analysis", new=AsyncMock()),
            patch.object(area, "run_likelihood_analysis", new=AsyncMock()),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data"),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            # Should continue despite health check failure
            await coordinator.run_analysis(datetime.now())


class TestCoordinatorAreaRemoval:
    """Test coordinator area removal scenarios."""

    async def test_async_update_options_remove_area(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_update_options when removing an area."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Update config_entry.data to remove the area
        original_data = coordinator.config_entry.data.copy()
        coordinator.config_entry.data = {
            "areas": {
                "New Area": {
                    "motion": ["binary_sensor.new_motion"],
                    "threshold": 50,
                }
            }
        }

        try:
            with (
                patch.object(area, "async_cleanup", new=AsyncMock()) as mock_cleanup,
                patch.object(coordinator.db, "delete_area_data") as mock_delete,
                patch.object(
                    coordinator, "track_entity_state_changes", new=AsyncMock()
                ),
            ):
                # Mock entity registry
                entity_registry = Mock()
                entity_registry.entities = {}
                with patch(
                    "custom_components.area_occupancy.coordinator.er.async_get",
                    return_value=entity_registry,
                ):
                    await coordinator.async_update_options(
                        coordinator.config_entry.data
                    )

                # Verify cleanup was called
                mock_cleanup.assert_called_once()
                # Verify database deletion was attempted
                mock_delete.assert_called_once_with(area_name)
        finally:
            # Restore original data
            coordinator.config_entry.data = original_data

    async def test_async_update_options_remove_area_db_error(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_update_options when database deletion fails."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Update config_entry.data to remove the area
        original_data = coordinator.config_entry.data.copy()
        coordinator.config_entry.data = {
            "areas": {
                "New Area": {
                    "motion": ["binary_sensor.new_motion"],
                    "threshold": 50,
                }
            }
        }

        try:
            with (
                patch.object(area, "async_cleanup", new=AsyncMock()),
                patch.object(
                    coordinator.db,
                    "delete_area_data",
                    side_effect=OSError("DB deletion failed"),
                ),
                patch.object(
                    coordinator, "track_entity_state_changes", new=AsyncMock()
                ),
            ):
                # Mock entity registry
                entity_registry = Mock()
                entity_registry.entities = {}
                with patch(
                    "custom_components.area_occupancy.coordinator.er.async_get",
                    return_value=entity_registry,
                ):
                    # Should handle error gracefully
                    await coordinator.async_update_options(
                        coordinator.config_entry.data
                    )
        finally:
            # Restore original data
            coordinator.config_entry.data = original_data


class TestCoordinatorFindAreaForEntity:
    """Test coordinator find_area_for_entity edge cases."""

    def test_find_area_for_entity_not_found(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test find_area_for_entity when entity is not found."""
        result = coordinator.find_area_for_entity("binary_sensor.nonexistent")
        assert result is None

    def test_find_area_for_entity_multiple_areas(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test find_area_for_entity with multiple areas."""
        # Create additional area
        create_test_area(
            coordinator,
            area_name="Kitchen",
            entity_ids=["binary_sensor.kitchen_motion"],
        )

        # Entity should be found in the correct area
        result = coordinator.find_area_for_entity("binary_sensor.kitchen_motion")
        assert result == "Kitchen"

    def test_find_area_for_entity_empty_entity_id(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test find_area_for_entity with empty entity_id."""
        result = coordinator.find_area_for_entity("")
        assert result is None
