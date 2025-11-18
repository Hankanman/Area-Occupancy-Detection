"""Tests for service module."""

from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.const import DOMAIN
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.service import _run_analysis, async_setup_services
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError


# ruff: noqa: SLF001
# Helper functions to reduce code duplication
def _setup_coordinator_test(
    hass: HomeAssistant,
    mock_config_entry: Mock,
    coordinator: AreaOccupancyCoordinator,
    entry_id: str = "test_entry_id",
) -> None:
    """Set up common coordinator test configuration."""
    mock_config_entry.runtime_data = coordinator
    # Set coordinator in hass.data for service functions that use _get_coordinator()
    hass.data[DOMAIN] = coordinator


def _create_service_call(**kwargs) -> Mock:
    """Create a mock service call with common data.

    Args:
        **kwargs: Additional data to include in service call
    """
    mock_call = Mock(spec=ServiceCall)
    mock_call.data = kwargs
    return mock_call


def _create_missing_entry_service_call() -> Mock:
    """Create a service call with missing entry_id (for backward compatibility tests)."""
    mock_call = Mock(spec=ServiceCall)
    mock_call.data = {"entry_id": None}
    return mock_call


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
        area = coordinator_with_areas.get_area(area_name)

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

        # Mock run_analysis method - it always runs for all areas
        coordinator_with_areas.run_analysis = AsyncMock()

        _setup_coordinator_test(hass, mock_config_entry, coordinator_with_areas)
        mock_service_call = _create_service_call()

        result = await _run_analysis(hass, mock_service_call)

        assert isinstance(result, dict)
        assert "areas" in result
        assert "update_timestamp" in result
        assert isinstance(result["areas"], dict)

    async def test_run_analysis_missing_entry_id(self, hass: HomeAssistant) -> None:
        """Test analysis run with missing entry_id (backward compatibility)."""
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
        mock_service_call = _create_service_call()

        with pytest.raises(
            HomeAssistantError,
            match="Failed to run analysis.*Analysis failed",
        ):
            await _run_analysis(hass, mock_service_call)


class TestAsyncSetupServices:
    """Test async_setup_services function."""

    async def test_async_setup_services_registers_one_service(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that only run_analysis service is registered."""

        hass.data[DOMAIN] = coordinator
        await async_setup_services(hass)

        # Verify only run_analysis service is registered
        services = hass.services.async_services().get(DOMAIN, {})
        assert "run_analysis" in services
        assert "reset_entities" not in services
        assert "get_entity_metrics" not in services
        assert "get_problematic_entities" not in services
        assert "get_area_status" not in services

        # Verify service has no schema (no parameters)
        service = services["run_analysis"]
        assert service.schema is None
