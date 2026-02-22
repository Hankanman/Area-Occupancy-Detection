"""Tests for coordinator module."""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.area.area import Area
from custom_components.area_occupancy.const import (
    ALL_AREAS_IDENTIFIER,
    CONF_AREA_ID,
    CONF_AREAS,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import Sensors
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.util import dt as dt_util

# Import helper functions from conftest
from tests.conftest import create_test_area

# ruff: noqa: SLF001, TID251


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

    def test_device_info_with_all_areas_identifier(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test device_info properly handles ALL_AREAS_IDENTIFIER."""
        device_info = coordinator.get_all_areas().device_info()

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

    async def test_update_method_real_implementation(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test update() method with real implementation and areas."""
        # Call the real update() method (not mocked)
        result = await coordinator.update()

        # Verify result structure
        assert isinstance(result, dict)
        assert len(result) > 0, "update() should return data for at least one area"

        # Verify structure for each area
        area_names = coordinator.get_area_names()
        for area_name in area_names:
            assert area_name in result, (
                f"Result should contain data for area: {area_name}"
            )

            area_data = result[area_name]
            expected_keys = {
                "probability",
                "occupied",
                "threshold",
                "prior",
                "decay",
                "last_updated",
            }
            assert set(area_data.keys()) == expected_keys, (
                f"Area {area_name} data should contain all expected keys"
            )

            # Verify value types and ranges
            assert isinstance(area_data["probability"], float)
            assert 0.0 <= area_data["probability"] <= 1.0

            assert isinstance(area_data["occupied"], bool)

            assert isinstance(area_data["threshold"], float)
            assert 0.0 <= area_data["threshold"] <= 1.0

            assert isinstance(area_data["prior"], float)
            assert 0.0 <= area_data["prior"] <= 1.0

            assert isinstance(area_data["decay"], float)
            assert 0.0 <= area_data["decay"] <= 1.0

            assert isinstance(area_data["last_updated"], datetime)

            # Verify values match actual area properties
            area = coordinator.get_area(area_name)
            assert area_data["probability"] == area.probability()
            assert area_data["occupied"] == area.occupied()
            assert area_data["threshold"] == area.threshold()
            assert area_data["prior"] == area.area_prior()
            assert area_data["decay"] == area.decay()

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
            patch.object(coordinator.db, "save_data"),
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
            patch.object(coordinator.db, "save_data"),
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
            patch.object(coordinator.db, "save_data"),
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

    @pytest.mark.parametrize(
        (
            "timer_type",
            "condition",
            "should_call_track",
            "timer_attr_name",
            "start_method_name",
        ),
        [
            ("decay", "normal", True, "_global_decay_timer", "_start_decay_timer"),
            (
                "decay",
                "existing_timer",
                False,
                "_global_decay_timer",
                "_start_decay_timer",
            ),
            ("decay", "no_hass", False, "_global_decay_timer", "_start_decay_timer"),
            ("analysis", "normal", True, "_analysis_timer", "_start_analysis_timer"),
            (
                "analysis",
                "existing_timer",
                False,
                "_analysis_timer",
                "_start_analysis_timer",
            ),
            ("analysis", "no_hass", False, "_analysis_timer", "_start_analysis_timer"),
        ],
    )
    async def test_timer_start_handling(
        self,
        hass: HomeAssistant,
        mock_realistic_config_entry: Mock,
        timer_type: str,
        condition: str,
        should_call_track: bool,
        timer_attr_name: str,
        start_method_name: str,
    ) -> None:
        """Test timer start handling for different timer types and conditions."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Set up initial state based on condition
        if condition == "existing_timer":
            setattr(coordinator, timer_attr_name, Mock())
        elif condition == "no_hass":
            coordinator.hass = None

        with patch(
            "custom_components.area_occupancy.coordinator.async_track_point_in_time",
            return_value=Mock() if should_call_track else None,
        ) as mock_track:
            start_method = getattr(coordinator, start_method_name)
            if timer_type == "analysis":
                await start_method()
            else:
                start_method()

            if should_call_track:
                mock_track.assert_called_once()
                assert getattr(coordinator, timer_attr_name) is not None
            else:
                mock_track.assert_not_called()

    @pytest.mark.parametrize(
        ("decay_enabled", "should_refresh"), [(True, True), (False, False)]
    )
    async def test_handle_decay_timer(
        self,
        hass: HomeAssistant,
        mock_realistic_config_entry: Mock,
        decay_enabled: bool,
        should_refresh: bool,
    ) -> None:
        """Test decay timer callback handling with decay enabled/disabled."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._global_decay_timer = Mock()

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

        # Access config via area
        area.config.decay.enabled = decay_enabled

        with (
            patch.object(coordinator, "async_refresh", new=AsyncMock()) as mock_refresh,
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator._handle_decay_timer(dt_util.utcnow())
            assert coordinator._global_decay_timer is None
            if should_refresh:
                mock_refresh.assert_called_once()
            else:
                mock_refresh.assert_not_called()

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

        # Create a mock function that matches the signature of prune_old_intervals
        # The wrapper from __getattr__ calls func(self, *args, **kwargs)
        # So our mock needs to accept db as first argument
        mock_prune_func = Mock(return_value=5)

        with (
            patch.object(coordinator.db, "sync_states", new=AsyncMock()),
            patch.object(coordinator.db, "periodic_health_check", return_value=True),
            # Patch the delegated method in the _delegated_methods dictionary
            # This ensures the wrapper returned by __getattr__ uses our mock
            patch.dict(
                coordinator.db._delegated_methods,
                {"prune_old_intervals": mock_prune_func},
            ),
            patch(
                "custom_components.area_occupancy.data.analysis.ensure_occupied_intervals_cache",
                new=AsyncMock(),
            ) as mock_cache,
            patch(
                "custom_components.area_occupancy.data.analysis.run_interval_aggregation",
                new=AsyncMock(),
            ) as mock_aggregation,
            patch.object(area, "run_prior_analysis", new=AsyncMock()) as mock_prior,
            patch(
                "custom_components.area_occupancy.db.correlation.run_correlation_analysis",
                new=AsyncMock(),
            ) as mock_correlation,
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data") as mock_save,
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis()
            assert coordinator._analysis_timer is None
            # Verify pruning was called (master-only)
            # The wrapper from __getattr__ will call our mock with (db, *args)
            mock_prune_func.assert_called_once()
            # Verify cache ensure was called before aggregation
            mock_cache.assert_called_once()
            # Verify interval aggregation was called before prior analysis
            mock_aggregation.assert_called_once()
            # Verify prior analysis was called
            mock_prior.assert_called_once()
            # Verify correlation analysis was called after prior analysis
            mock_correlation.assert_called_once()
            # Verify save was called twice:
            # 1. Before async_refresh() to preserve decay state
            # 2. After async_refresh() to persist all changes
            assert mock_save.call_count == 2

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
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data"),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis(custom_time)
            assert coordinator._analysis_timer is None

    async def test_run_analysis_concurrent_guard(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that concurrent analysis runs are prevented."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()

        # Simulate analysis already running
        coordinator._analysis_running = True

        with (
            patch(
                "custom_components.area_occupancy.data.analysis.run_full_analysis",
                new=AsyncMock(),
            ) as mock_full_analysis,
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=Mock(),
            ) as mock_track_time,
        ):
            await coordinator.run_analysis()

            # Analysis should NOT have run because _analysis_running was True
            mock_full_analysis.assert_not_called()

            # Timer should be rescheduled for retry
            mock_track_time.assert_called_once()

            # Flag should still be True (the concurrent call didn't change it)
            assert coordinator._analysis_running is True

    async def test_run_analysis_flag_reset_on_completion(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that _analysis_running flag is reset after analysis completes."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()

        # Flag should start as False
        assert coordinator._analysis_running is False

        with (
            patch(
                "custom_components.area_occupancy.coordinator.run_full_analysis",
                new=AsyncMock(),
            ),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis()

            # Flag should be reset to False after completion
            assert coordinator._analysis_running is False

    async def test_run_analysis_flag_reset_on_error(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that _analysis_running flag is reset even if analysis fails."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        coordinator._analysis_timer = Mock()

        # Flag should start as False
        assert coordinator._analysis_running is False

        with (
            patch(
                "custom_components.area_occupancy.coordinator.run_full_analysis",
                side_effect=HomeAssistantError("Analysis failed"),
            ),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=Mock(),
            ),
        ):
            await coordinator.run_analysis()

            # Flag should be reset to False even after error (via finally block)
            assert coordinator._analysis_running is False

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

    @pytest.mark.parametrize(
        ("has_new_evidence", "should_refresh"), [(True, True), (False, False)]
    )
    async def test_track_entity_state_changes_with_evidence(
        self,
        hass: HomeAssistant,
        mock_realistic_config_entry: Mock,
        has_new_evidence: bool,
        should_refresh: bool,
    ) -> None:
        """Test entity state tracking with entity that has/doesn't have new evidence."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Ensure setup_complete is True so the refresh condition is met
        coordinator._setup_complete = True

        # Set up an area for the test using helper
        area_name = "Test Area"
        area = create_test_area(coordinator, area_name=area_name)
        coordinator.get_area = Mock(return_value=area)

        # Mock entity with/without new evidence
        mock_entity = Mock()
        mock_entity.has_new_evidence.return_value = has_new_evidence

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
            if should_refresh:
                mock_refresh.assert_called_once()
            else:
                mock_refresh.assert_not_called()

    async def test_setup_with_intervals_empty(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup when intervals table is empty."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator.db, "load_data", new=AsyncMock()),
            patch.object(coordinator.db, "save_data"),  # Now sync, no AsyncMock
            patch.object(coordinator.db, "is_intervals_empty", return_value=True),
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
            patch.object(coordinator.db, "save_data", return_value=None),
            patch.object(coordinator.db, "is_intervals_empty", return_value=False),
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
            patch.object(coordinator.db, "save_data"),
            patch.object(coordinator.db, "is_intervals_empty", return_value=True),
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
            patch.object(coordinator.db, "save_data", side_effect=OSError("DB Error")),
            patch.object(coordinator.db, "is_intervals_empty", return_value=False),
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
            patch.object(coordinator.db, "save_data"),
            patch.object(
                coordinator.db,
                "is_intervals_empty",
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
            patch.object(coordinator.db, "save_data"),
            patch.object(coordinator.db, "is_intervals_empty", return_value=True),
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
        """Test setup with unexpected RuntimeError raises ConfigEntryNotReady."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with (
            patch.object(
                coordinator.db,
                "load_data",
                side_effect=RuntimeError("Unexpected error"),
            ),
            pytest.raises(ConfigEntryNotReady),
        ):
            await coordinator.setup()

    async def test_setup_with_timer_start_error(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test setup with OSError raises ConfigEntryNotReady."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        with (
            patch.object(
                coordinator.db,
                "load_data",
                side_effect=OSError("Load error"),
            ),
            pytest.raises(ConfigEntryNotReady),
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
            patch.object(coordinator.db, "load_data", new=AsyncMock()) as mock_load,
            patch.object(coordinator.db, "save_data") as mock_save,
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

            # cleanup is called on area.entities for each area (before database reload)
            mock_cleanup.assert_called()
            # load_data is called to restore priors and entity states from database
            mock_load.assert_called_once()
            # track_entity_state_changes is called with new entity lists
            mock_track.assert_called_once()
            # save_data is called to persist changes
            mock_save.assert_called_once()

            # Clean up timers
            await coordinator.async_shutdown()

    async def test_shutdown_with_all_timers(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test shutdown with all timers present."""
        coordinator._is_master = True  # Enable master-specific cleanup

        # Get area from fixture
        area = coordinator.get_area()
        assert area is not None

        # Set up all timers
        coordinator._global_decay_timer = Mock()
        coordinator._analysis_timer = Mock()
        coordinator._save_timer = Mock()  # Set save timer to trigger final save

        with (
            patch.object(coordinator.db, "save_data"),
            patch.object(area, "async_cleanup", new=AsyncMock()) as mock_cleanup,
            patch(
                "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.async_shutdown",
                new=AsyncMock(),
            ),
        ):
            await coordinator.async_shutdown()

            assert coordinator._global_decay_timer is None
            assert coordinator._analysis_timer is None
            coordinator.db.save_data.assert_called_once()
            # async_cleanup is called for each area, which calls entities.cleanup() and purpose.cleanup()
            mock_cleanup.assert_called_once()

    # New tests for performance optimization features

    def test_get_area_names(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test get_area_names method."""
        # Areas are already loaded by coordinator fixture
        area_names = coordinator.get_area_names()
        assert isinstance(area_names, list)
        assert len(area_names) > 0

        # Should contain at least one area name
        assert all(isinstance(name, str) for name in area_names)

    def test_get_area(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test get_area method."""
        # Areas are already loaded by coordinator fixture

        # Should return first area when None is passed
        area = coordinator.get_area()
        assert area is not None
        assert hasattr(area, "area_name")
        assert hasattr(area, "config")
        assert hasattr(area, "entities")
        assert hasattr(area, "prior")

        # Should return specific area when name is provided
        area_names = coordinator.get_area_names()
        assert len(area_names) > 0
        area_name = area_names[0]
        specific_area = coordinator.get_area(area_name)
        assert specific_area is not None
        assert specific_area.area_name == area_name

        # Should return None for non-existent area
        non_existent = coordinator.get_area("NonExistentArea")
        assert non_existent is None

        first_area = coordinator.get_area()
        assert first_area is not None
        assert first_area.area_name in coordinator.get_area_names()

    def test_load_areas_from_config(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test _load_areas_from_config() loads areas correctly."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Load areas from config
        coordinator._load_areas_from_config()

        # Verify areas were loaded
        area_names = coordinator.get_area_names()
        assert len(area_names) > 0

        # Verify each area has correct structure
        for area_name in area_names:
            area = coordinator.get_area(area_name)
            assert area is not None
            assert area.area_name == area_name
            assert area.coordinator == coordinator

    @pytest.mark.parametrize(
        "error_type",
        ["invalid_area_id", "duplicate_areas"],
    )
    def test_load_areas_from_config_error_cases(
        self,
        hass: HomeAssistant,
        mock_realistic_config_entry: Mock,
        error_type: str,
    ) -> None:
        """Test _load_areas_from_config() handles error cases."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)
        original_data = mock_realistic_config_entry.data.copy()

        try:
            if error_type == "invalid_area_id":
                # Add invalid area ID to config
                invalid_area_data = {
                    CONF_AREAS: [
                        *original_data.get(CONF_AREAS, []),
                        {CONF_AREA_ID: "invalid_area_id_that_does_not_exist"},
                    ]
                }
                mock_realistic_config_entry.data = invalid_area_data

                # Should handle invalid area ID gracefully
                coordinator._load_areas_from_config()

                # Valid areas should still be loaded
                area_names = coordinator.get_area_names()
                assert len(area_names) > 0

                # Invalid area should not be in the list
                assert "invalid_area_id_that_does_not_exist" not in area_names

            elif error_type == "duplicate_areas":
                # Get first area from config
                areas_list = original_data.get(CONF_AREAS, [])
                if areas_list:
                    # Duplicate the first area
                    duplicate_area = areas_list[0].copy()
                    duplicate_config = {CONF_AREAS: [*areas_list, duplicate_area]}
                    mock_realistic_config_entry.data = duplicate_config

                    coordinator._load_areas_from_config()

                    # Should only load one instance of the duplicate area
                    area_names = coordinator.get_area_names()
                    # Count occurrences of the first area's name
                    first_area_id = areas_list[0].get(CONF_AREA_ID)
                    if first_area_id:
                        area_reg = ar.async_get(hass)
                        area_entry = area_reg.async_get_area(first_area_id)
                        if area_entry:
                            area_name = area_entry.name
                            assert area_names.count(area_name) == 1, (
                                "Duplicate area should only be loaded once"
                            )
        finally:
            # Restore original data
            mock_realistic_config_entry.data = original_data

    def test_load_areas_from_config_with_target_dict(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test _load_areas_from_config() with target_dict parameter."""
        coordinator = AreaOccupancyCoordinator(hass, mock_realistic_config_entry)

        # Load into a target dict instead of self.areas
        target_dict: dict[str, Area] = {}
        coordinator._load_areas_from_config(target_dict=target_dict)

        # Verify areas were loaded into target_dict
        assert len(target_dict) > 0

        # Verify self.areas was not modified
        assert len(coordinator.areas) == 0

        # Verify target_dict contains correct areas
        for area_name, area in target_dict.items():
            assert area.area_name == area_name
            assert area.coordinator == coordinator

    def test_get_area_handle(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test get_area_handle() returns stable handles."""
        area_name = coordinator.get_area_names()[0]

        # Get handle first time
        handle1 = coordinator.get_area_handle(area_name)
        assert handle1 is not None

        # Get handle second time - should return same instance
        handle2 = coordinator.get_area_handle(area_name)
        assert handle2 is handle1, (
            "get_area_handle() should return same instance for same area"
        )

        # Get handle for different area - should return different instance
        if len(coordinator.get_area_names()) > 1:
            area_name2 = coordinator.get_area_names()[1]
            handle3 = coordinator.get_area_handle(area_name2)
            assert handle3 is not handle1, (
                "Different areas should have different handles"
            )
            assert handle3 is not None

    def test_get_all_areas_lazy_initialization(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_all_areas() lazy initialization."""
        # Initially _all_areas should be None
        assert coordinator._all_areas is None

        # First call should create instance
        all_areas1 = coordinator.get_all_areas()
        assert all_areas1 is not None
        assert coordinator._all_areas is all_areas1

        # Second call should return same instance
        all_areas2 = coordinator.get_all_areas()
        assert all_areas2 is all_areas1, "get_all_areas() should return same instance"

    @pytest.mark.parametrize(
        ("has_areas", "should_raise"), [(True, False), (False, True)]
    )
    def test_validate_areas_configured(
        self,
        hass: HomeAssistant,
        mock_realistic_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator | None,
        has_areas: bool,
        should_raise: bool,
    ) -> None:
        """Test _validate_areas_configured() with and without areas."""
        if has_areas:
            # Use coordinator fixture which has areas loaded
            assert coordinator is not None
            test_coordinator = coordinator
        else:
            # Create new coordinator without areas for this test case
            test_coordinator = AreaOccupancyCoordinator(
                hass, mock_realistic_config_entry
            )
            test_coordinator.areas.clear()

        if should_raise:
            with pytest.raises(HomeAssistantError, match="No areas configured"):
                test_coordinator._validate_areas_configured()
        else:
            # Should not raise exception
            test_coordinator._validate_areas_configured()

    @pytest.mark.parametrize(
        ("db_error", "expected_delete_calls"),
        [(None, 1), (OSError("Database error"), 1)],
    )
    async def test_cleanup_removed_area(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        db_error: Exception | None,
        expected_delete_calls: int,
    ) -> None:
        """Test _cleanup_removed_area() with normal and error cases."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        with (
            patch.object(area, "async_cleanup", new=AsyncMock()) as mock_cleanup,
            patch.object(
                coordinator.db,
                "delete_area_data",
                return_value=5 if db_error is None else None,
                side_effect=db_error,
            ) as mock_delete,
            patch(
                "custom_components.area_occupancy.coordinator.dr.async_get"
            ) as mock_dr_get,
            patch(
                "custom_components.area_occupancy.coordinator.er.async_get"
            ) as mock_er_get,
        ):
            # Mock device registry
            mock_device_registry = Mock()
            if db_error is None:
                mock_device = Mock()
                mock_device.id = "device_id"
                mock_device_registry.async_get_device.return_value = mock_device
                mock_device_registry.async_remove_device = Mock()
            else:
                mock_device_registry.async_get_device.return_value = None
            mock_dr_get.return_value = mock_device_registry

            # Mock entity registry
            mock_entity_registry = Mock()
            mock_entity_registry.entities = {}  # Empty for simplicity
            mock_er_get.return_value = mock_entity_registry

            # Call cleanup - should handle errors gracefully
            await coordinator._cleanup_removed_area(area_name, area)

            # Verify cleanup was called
            mock_cleanup.assert_called_once()

            # Verify database deletion was attempted
            assert mock_delete.call_count == expected_delete_calls
            mock_delete.assert_called_with(area_name)

            # Verify device registry lookup was attempted
            mock_device_registry.async_get_device.assert_called_once()


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
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data"),
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
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator.db, "save_data"),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
        ):
            await coordinator.run_analysis()

            # Verify pruning was called and analysis completed
            coordinator.db.prune_old_intervals.assert_called_once()
            # run_analysis calls area.run_prior_analysis()
            mock_prior.assert_called_once()
            assert coordinator._analysis_timer is None

    async def test_run_analysis_pruning_error_handling(
        self, hass: HomeAssistant, mock_realistic_config_entry: Mock
    ) -> None:
        """Test that analysis continues if pruning raises an exception.

        With step-level error tracking, individual step failures don't stop
        the pipeline — subsequent steps still execute.
        """
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
                "periodic_health_check",
                side_effect=RuntimeError("Health check failed"),
            ),
            patch.object(area, "run_prior_analysis", new=AsyncMock()) as mock_prior,
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
            patch.object(coordinator, "async_refresh_correlations", new=AsyncMock()),
            patch.object(coordinator.db, "save_data"),
            patch(
                "custom_components.area_occupancy.coordinator.async_track_point_in_time",
                return_value=None,
            ),
            patch(
                "custom_components.area_occupancy.data.analysis.ensure_occupied_intervals_cache",
                new=AsyncMock(),
            ),
            patch(
                "custom_components.area_occupancy.data.analysis.run_interval_aggregation",
                new=AsyncMock(),
            ),
            patch(
                "custom_components.area_occupancy.data.analysis.run_numeric_aggregation",
                new=AsyncMock(),
            ),
            patch(
                "custom_components.area_occupancy.db.correlation.run_correlation_analysis",
                new=AsyncMock(),
            ),
        ):
            # Step 2 fails, but pipeline continues.
            await coordinator.run_analysis()

            # Verify subsequent steps still executed despite step 2 failure
            mock_prior.assert_called_once()
            coordinator.async_refresh.assert_called_once()
            coordinator.db.save_data.assert_called()


class TestCoordinatorTimerCallbacks:
    """Test coordinator timer callback error handling."""

    @pytest.mark.parametrize(
        ("should_error", "error_class"),
        [(False, None), (True, RuntimeError)],
    )
    async def test_handle_save_timer(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        should_error: bool,
        error_class: type[Exception] | None,
    ) -> None:
        """Test _handle_save_timer with success and error cases."""
        with (
            patch.object(
                coordinator.db,
                "save_data",
                side_effect=error_class("Save failed") if should_error else None,
            ) as mock_save,
            patch.object(coordinator, "_start_save_timer") as mock_start,
        ):
            await coordinator._handle_save_timer(datetime.now())
            if not should_error:
                mock_save.assert_called_once()
            mock_start.assert_called_once()

    @pytest.mark.parametrize(
        ("decay_enabled", "should_refresh"), [(True, True), (False, False)]
    )
    async def test_handle_decay_timer_callback(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        decay_enabled: bool,
        should_refresh: bool,
    ) -> None:
        """Test _handle_decay_timer callback with decay enabled/disabled."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.decay.enabled = decay_enabled

        with (
            patch.object(coordinator, "async_refresh") as mock_refresh,
            patch.object(coordinator, "_start_decay_timer") as mock_start,
        ):
            await coordinator._handle_decay_timer(datetime.now())
            if should_refresh:
                mock_refresh.assert_called_once()
            else:
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

        # Remove all areas from CONF_AREAS to simulate area removal.
        original_data = coordinator.config_entry.data
        coordinator.config_entry.data = {CONF_AREAS: []}

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
                        coordinator.config_entry.options
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

        # Remove all areas from CONF_AREAS to simulate area removal.
        original_data = coordinator.config_entry.data
        coordinator.config_entry.data = {CONF_AREAS: []}

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
                        coordinator.config_entry.options
                    )
        finally:
            # Restore original data
            coordinator.config_entry.data = original_data


class TestCoordinatorFindAreaForEntity:
    """Test coordinator find_area_for_entity edge cases."""

    @pytest.mark.parametrize(
        ("scenario", "entity_id", "expected_result"),
        [
            ("not_found", "binary_sensor.nonexistent", None),
            ("multiple_areas", "binary_sensor.kitchen_motion", "Kitchen"),
            ("empty_entity_id", "", None),
        ],
    )
    def test_find_area_for_entity(
        self,
        coordinator: AreaOccupancyCoordinator,
        scenario: str,
        entity_id: str,
        expected_result: str | None,
    ) -> None:
        """Test find_area_for_entity with various scenarios."""
        if scenario == "multiple_areas":
            # Create additional area for this scenario
            create_test_area(
                coordinator,
                area_name="Kitchen",
                entity_ids=["binary_sensor.kitchen_motion"],
            )

        result = coordinator.find_area_for_entity(entity_id)
        assert result == expected_result
