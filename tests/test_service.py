"""Tests for service module."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.service import (
    _get_area_status,
    _get_coordinator,
    _get_entity_metrics,
    _get_problematic_entities,
    _reset_entities,
    _run_analysis,
)
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001, PLC0415
# Helper functions to reduce code duplication
def _setup_coordinator_test(
    hass: HomeAssistant,
    mock_config_entry: Mock,
    coordinator: AreaOccupancyCoordinator,
    entry_id: str = "test_entry_id",
) -> None:
    """Set up common coordinator test configuration."""
    from custom_components.area_occupancy.const import DOMAIN

    mock_config_entry.runtime_data = coordinator
    # Set coordinator in hass.data for service functions that use _get_coordinator()
    hass.data[DOMAIN] = coordinator


def _create_service_call(area_name: str | None = None, **kwargs) -> Mock:
    """Create a mock service call with common data.

    Args:
        area_name: Area name to use (defaults to None for backward compatibility tests)
        **kwargs: Additional data to include in service call
    """
    mock_call = Mock(spec=ServiceCall)
    # Support both new area_name and deprecated entry_id for backward compatibility
    if area_name is not None:
        mock_call.data = {"area_name": area_name, **kwargs}
    else:
        # For backward compatibility tests, use entry_id
        mock_call.data = {"entry_id": "test_entry_id", **kwargs}
    return mock_call


def _create_missing_entry_service_call() -> Mock:
    """Create a service call with missing entry_id (for backward compatibility tests)."""
    mock_call = Mock(spec=ServiceCall)
    mock_call.data = {"entry_id": None}
    return mock_call


class TestGetCoordinator:
    """Test _get_coordinator helper function."""

    def test_get_coordinator_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful coordinator retrieval."""
        from custom_components.area_occupancy.const import DOMAIN

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        hass.data[DOMAIN] = coordinator_with_areas

        result = _get_coordinator(hass)
        assert result == coordinator_with_areas

    def test_get_coordinator_missing_domain(self, hass: HomeAssistant) -> None:
        """Test coordinator retrieval with missing domain."""
        from custom_components.area_occupancy.const import DOMAIN

        hass.data[DOMAIN] = None

        with pytest.raises(
            HomeAssistantError, match="Area Occupancy coordinator not found"
        ):
            _get_coordinator(hass)


class TestRunAnalysis:
    """Test _run_analysis service function."""

    async def test_run_analysis_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful analysis run."""
        # Set up coordinator with test data - use area-based access
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)

        # Create test entity
        mock_entity = Mock()
        mock_entity.type.input_type.value = "motion"
        mock_entity.type.weight = 0.85
        mock_entity.prob_given_true = 0.8
        mock_entity.prob_given_false = 0.1
        area._entities = type(
            "obj", (object,), {"entities": {"binary_sensor.motion1": mock_entity}}
        )()
        coordinator_with_areas.db.import_stats = {"binary_sensor.motion1": 100}

        # Mock run_analysis method - it's now area-based, so we need to mock it properly
        # The service calls coordinator.run_analysis(area_name) which calls area.run_analysis()
        area.run_analysis = AsyncMock(return_value={"status": "success"})
        coordinator_with_areas.run_analysis = AsyncMock(
            return_value={"status": "success"}
        )

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        mock_service_call = _create_service_call(area_name=area_name)

        result = await _run_analysis(hass, mock_service_call)

        assert isinstance(result, dict)
        assert len(result) > 0

    async def test_run_analysis_missing_entry_id(self, hass: HomeAssistant) -> None:
        """Test analysis run with missing entry_id (backward compatibility)."""
        from custom_components.area_occupancy.const import DOMAIN

        hass.data[DOMAIN] = None
        mock_service_call = _create_missing_entry_service_call()

        with pytest.raises(
            HomeAssistantError,
            match="Area Occupancy coordinator not found",
        ):
            await _run_analysis(hass, mock_service_call)

    async def test_run_analysis_coordinator_error(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test analysis run with coordinator error."""
        coordinator_with_areas.run_analysis = AsyncMock(
            side_effect=RuntimeError("Analysis failed")
        )

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        area_name = coordinator_with_areas.get_area_names()[0]
        mock_service_call = _create_service_call(area_name=area_name)

        with pytest.raises(
            HomeAssistantError,
            match="Failed to run analysis.*Analysis failed",
        ):
            await _run_analysis(hass, mock_service_call)


class TestResetEntities:
    """Test _reset_entities service function."""

    async def test_reset_entities_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful entity reset."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        # Mock cleanup as async method
        area.entities.cleanup = AsyncMock()
        coordinator_with_areas.async_refresh = AsyncMock()

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        mock_service_call = _create_service_call(area_name=area_name)

        await _reset_entities(hass, mock_service_call)

        area.entities.cleanup.assert_called_once()
        coordinator_with_areas.async_refresh.assert_called_once()

    async def test_reset_entities_missing_entry_id(self, hass: HomeAssistant) -> None:
        """Test entity reset with missing entry_id (backward compatibility)."""
        from custom_components.area_occupancy.const import DOMAIN

        hass.data[DOMAIN] = None
        mock_service_call = _create_missing_entry_service_call()

        with pytest.raises(
            HomeAssistantError,
            match="Area Occupancy coordinator not found",
        ):
            await _reset_entities(hass, mock_service_call)


class TestGetEntityMetrics:
    """Test _get_entity_metrics service function."""

    async def test_get_entity_metrics_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
        mock_active_entity: Mock,
        mock_inactive_entity: Mock,
    ) -> None:
        """Test successful entity metrics retrieval."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        area._entities = type(
            "obj",
            (object,),
            {
                "entities": {
                    "binary_sensor.motion1": mock_active_entity,
                    "binary_sensor.appliance": mock_inactive_entity,
                }
            },
        )()

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        area_name = coordinator_with_areas.get_area_names()[0]
        mock_service_call = _create_service_call(area_name=area_name)

        result = await _get_entity_metrics(hass, mock_service_call)

        assert "metrics" in result
        metrics = result["metrics"]
        assert metrics["total_entities"] == 2
        assert metrics["active_entities"] == 1
        assert metrics["available_entities"] == 2
        assert metrics["unavailable_entities"] == 0

    async def test_get_entity_metrics_missing_entry_id(
        self, hass: HomeAssistant
    ) -> None:
        """Test entity metrics with missing entry_id (backward compatibility)."""
        from custom_components.area_occupancy.const import DOMAIN

        hass.data[DOMAIN] = None
        mock_service_call = _create_missing_entry_service_call()

        with pytest.raises(
            HomeAssistantError,
            match="Area Occupancy coordinator not found",
        ):
            await _get_entity_metrics(hass, mock_service_call)

    async def test_get_entity_metrics_coordinator_error(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test entity metrics with coordinator error."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        mock_entities = Mock()
        mock_entities.__len__ = Mock(side_effect=Exception("Access error"))
        area._entities = type("obj", (object,), {"entities": mock_entities})()

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        area_name = coordinator_with_areas.get_area_names()[0]
        mock_service_call = _create_service_call(area_name=area_name)

        with pytest.raises(
            HomeAssistantError,
            match="Failed to get entity metrics.*Access error",
        ):
            await _get_entity_metrics(hass, mock_service_call)


class TestGetProblematicEntities:
    """Test _get_problematic_entities service function."""

    async def test_get_problematic_entities_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
        mock_unavailable_entity: Mock,
        mock_stale_entity: Mock,
    ) -> None:
        """Test successful problematic entities retrieval."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        area._entities = type(
            "obj",
            (object,),
            {
                "entities": {
                    "binary_sensor.motion1": mock_unavailable_entity,
                    "binary_sensor.appliance": mock_stale_entity,
                }
            },
        )()

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        area_name = coordinator_with_areas.get_area_names()[0]
        mock_service_call = _create_service_call(area_name=area_name)

        result = await _get_problematic_entities(hass, mock_service_call)

        assert "problems" in result
        problems = result["problems"]
        assert "unavailable" in problems
        assert "stale_updates" in problems
        assert "binary_sensor.motion1" in problems["unavailable"]
        assert "binary_sensor.appliance" in problems["stale_updates"]

    async def test_get_problematic_entities_no_issues(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
        mock_active_entity: Mock,
    ) -> None:
        """Test problematic entities with no issues."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        mock_active_entity.last_updated = dt_util.utcnow() - timedelta(minutes=30)
        area._entities = type(
            "obj",
            (object,),
            {"entities": {"binary_sensor.motion1": mock_active_entity}},
        )()

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        area_name = coordinator_with_areas.get_area_names()[0]
        mock_service_call = _create_service_call(area_name=area_name)

        result = await _get_problematic_entities(hass, mock_service_call)

        assert "problems" in result
        problems = result["problems"]
        assert len(problems["unavailable"]) == 0
        assert len(problems["stale_updates"]) == 0

    async def test_get_problematic_entities_coordinator_error(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test problematic entities with coordinator error."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        mock_entities = Mock()
        mock_entities.items = Mock(side_effect=Exception("Access error"))
        area._entities = type("obj", (object,), {"entities": mock_entities})()

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        area_name = coordinator_with_areas.get_area_names()[0]
        mock_service_call = _create_service_call(area_name=area_name)

        with pytest.raises(
            HomeAssistantError,
            match="Failed to get problematic entities.*Access error",
        ):
            await _get_problematic_entities(hass, mock_service_call)


class TestGetAreaStatus:
    """Test _get_area_status service function."""

    async def test_get_area_status_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
        mock_last_updated: Mock,
    ) -> None:
        """Test successful area status retrieval."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        # Area name is already set from coordinator
        area._entities = type("obj", (object,), {"entities": {}})()
        # Prior value is already set from coordinator

        # Mock area methods that service accesses
        area.probability = Mock(return_value=0.9)
        area.occupied = Mock(return_value=True)
        area.area_prior = Mock(return_value=0.3)

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        area_name = coordinator_with_areas.get_area_names()[0]
        mock_service_call = _create_service_call(area_name=area_name)

        result = await _get_area_status(hass, mock_service_call)

        assert "area_status" in result
        status = result["area_status"]
        area_name = coordinator_with_areas.get_area_names()[0]
        assert status["area_name"] == area_name
        assert status["occupied"] is True
        assert status["occupancy_probability"] == 0.9
        assert status["confidence_level"] == "high"

    async def test_get_area_status_no_occupancy_state(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
        mock_last_updated: Mock,
    ) -> None:
        """Test area status with no occupancy state."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)
        # Area name is already set from coordinator
        area._entities = type("obj", (object,), {"entities": {}})()
        # Prior value is already set from coordinator

        # Mock area methods that service accesses
        area.probability = Mock(return_value=None)
        area.occupied = Mock(return_value=False)
        area.area_prior = Mock(return_value=0.3)

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        area_name = coordinator_with_areas.get_area_names()[0]
        mock_service_call = _create_service_call(area_name=area_name)

        result = await _get_area_status(hass, mock_service_call)

        assert "area_status" in result
        status = result["area_status"]
        area_name = coordinator_with_areas.get_area_names()[0]
        assert status["area_name"] == area_name
        assert status["occupied"] is False
        assert status["occupancy_probability"] is None
        assert status["confidence_level"] == "unknown"


class TestAsyncSetupServices:
    """Test async_setup_services function."""

    @pytest.mark.skip(
        reason=(
            "Cannot test service registration error handling because "
            "hass.services.async_register is read-only and cannot be mocked. "
            "The ServiceRegistry uses __slots__ or descriptors that prevent "
            "patching even with object.__setattr__ or monkeypatch."
        )
    )
    async def test_async_setup_services_registration_error(
        self, hass: HomeAssistant
    ) -> None:
        """Test service setup with registration error."""
        # This test is skipped because hass.services.async_register cannot be mocked.
        # The ServiceRegistry attribute is read-only and protected at a low level,
        # preventing us from testing error handling during service registration.
