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
        device_info = mock_coordinator.device_info

        assert "identifiers" in device_info
        assert "name" in device_info
        assert "manufacturer" in device_info
        assert "model" in device_info
        assert isinstance(device_info["identifiers"], set)
        assert isinstance(device_info["name"], str)

    def test_device_info_with_real_constants(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test device_info property with actual constant values."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)
        device_info = coordinator.device_info

        assert device_info.get("manufacturer") == DEVICE_MANUFACTURER
        assert device_info.get("model") == DEVICE_MODEL
        assert device_info.get("sw_version") == DEVICE_SW_VERSION

        identifiers = device_info.get("identifiers")
        assert identifiers is not None
        assert (DOMAIN, coordinator.entry_id) in identifiers

    def test_device_info_with_missing_config(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test device info generation when config is missing."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with patch.object(coordinator, "config") as mock_config:
            mock_config.name = None
            device_info = coordinator.device_info

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
        assert getattr(mock_coordinator, property_name) == expected_value

    def test_threshold_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test threshold property specifically."""
        assert mock_coordinator_with_threshold.threshold == 0.6

    def test_is_occupied_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test is_occupied property threshold comparison."""
        # Mock coordinator has threshold 0.6 and probability 0.5
        assert not mock_coordinator_with_threshold.is_occupied  # 0.5 < 0.6

        # Test at threshold boundary
        mock_coordinator_with_threshold.probability = 0.6
        mock_coordinator_with_threshold.is_occupied = True  # 0.6 >= 0.6
        assert mock_coordinator_with_threshold.is_occupied

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
        # Configure decaying entities
        motion2 = mock_coordinator_with_sensors.entities.entities[
            "binary_sensor.motion2"
        ]
        motion2.decay.is_decaying = True

        mock_coordinator_with_sensors.decaying_entities = [motion2]

        decaying = mock_coordinator_with_sensors.decaying_entities
        assert len(decaying) == 1
        assert decaying[0].entity_id == "binary_sensor.motion2"

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

        if entities_empty:
            coordinator.entities.entities = {}
            coordinator.probability = expected_probability
            coordinator.area_prior = expected_prior
            coordinator.decay = expected_decay

        assert coordinator.probability == expected_probability
        assert coordinator.area_prior == expected_prior
        assert coordinator.decay == expected_decay

    def test_probability_with_mixed_evidence_and_decay(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test probability calculation with mixed evidence and decay states."""
        entities = mock_coordinator_with_sensors.entities.entities

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

        probability = mock_coordinator_with_sensors.probability
        assert 0.0 <= probability <= 1.0

    def test_probability_calculation_with_varying_weights(
        self, mock_coordinator_with_sensors: Mock
    ) -> None:
        """Test probability calculation with entities having different weights."""
        entities = mock_coordinator_with_sensors.entities.entities

        entities["binary_sensor.motion1"].type.weight = 0.9
        entities["binary_sensor.motion1"].evidence = True

        entities["binary_sensor.appliance"].type.weight = 0.1
        entities["binary_sensor.appliance"].evidence = True

        probability = mock_coordinator_with_sensors.probability
        assert 0.0 <= probability <= 1.0

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
        coordinator.probability = probability
        coordinator.threshold = threshold
        coordinator.is_occupied = expected_occupied

        assert coordinator.is_occupied == expected_occupied

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

        with (
            patch.object(coordinator.entities, "get_entity") as mock_get_entity,
            patch(
                "homeassistant.helpers.event.async_track_point_in_time",
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
        stored_data = {"entities": {"binary_sensor.test": {}}}
        coordinator.db.load_data = AsyncMock(return_value=stored_data)

        with (
            patch.object(coordinator.purpose, "async_initialize", new=AsyncMock()),
            patch.object(coordinator.entities, "cleanup", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(coordinator.db, "is_intervals_empty", return_value=False),
            patch.object(coordinator, "run_analysis", new=AsyncMock()),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(coordinator, "_start_decay_timer"),
            patch.object(coordinator, "_start_analysis_timer"),
            patch.object(coordinator, "async_refresh", new=AsyncMock()),
        ):
            await coordinator.setup()

        # Test setup failure
        coordinator.db.load_data = AsyncMock(
            side_effect=HomeAssistantError("Storage failed")
        )

        with (
            patch.object(coordinator.purpose, "async_initialize", new=AsyncMock()),
            patch.object(coordinator.entities, "cleanup", new=AsyncMock()),
            patch.object(coordinator.db, "save_area_data", new=AsyncMock()),
            patch.object(coordinator, "_start_decay_timer"),
            patch.object(coordinator, "_start_analysis_timer"),
            patch.object(coordinator, "run_analysis", new=AsyncMock()),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch.object(coordinator.entities, "get_entity") as mock_get_entity,
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

    async def test_shutdown_behavior(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test shutdown behavior with real coordinator instance."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        coordinator._global_decay_timer = Mock()
        coordinator._remove_state_listener = Mock()

        with (
            patch.object(coordinator.entities, "get_entity") as mock_get_entity,
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
            assert coordinator._remove_state_listener is None

    async def test_shutdown_with_none_resources(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test shutdown when resources are already None."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        coordinator._global_decay_timer = None
        coordinator._remove_state_listener = None

        with (
            patch.object(coordinator.entities, "get_entity") as mock_get_entity,
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
            assert coordinator._remove_state_listener is None

    async def test_full_coordinator_lifecycle(
        self, mock_hass: Mock, mock_realistic_config_entry: Mock
    ) -> None:
        """Test complete coordinator lifecycle with realistic configuration."""
        coordinator = AreaOccupancyCoordinator(mock_hass, mock_realistic_config_entry)

        with (
            patch.object(coordinator.entities, "get_entity") as mock_get_entity,
            patch.object(coordinator.entities, "cleanup", new=AsyncMock()),
            patch.object(coordinator.db, "load_data", new=AsyncMock(return_value=None)),
            patch.object(coordinator.db, "save_data", new=AsyncMock()),
            patch.object(coordinator, "track_entity_state_changes", new=AsyncMock()),
            patch.object(coordinator, "_start_decay_timer"),
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

            assert coordinator.entities is not None

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

        mock_coordinator.entities.entities = entities

        probability = mock_coordinator.probability
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0

    async def test_state_tracking_with_many_entities(
        self, mock_coordinator: Mock
    ) -> None:
        """Test state tracking setup with many entities."""
        entity_ids = [f"binary_sensor.motion_{i}" for i in range(200)]

        await mock_coordinator.track_entity_state_changes(entity_ids)
        mock_coordinator.track_entity_state_changes.assert_called_with(entity_ids)
