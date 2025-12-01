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
    _entry_id: str = "test_entry_id",
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


class TestRunAnalysis:
    """Test _run_analysis service function."""

    async def test_run_analysis_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful analysis run."""
        # Set up coordinator with test data - use area-based access
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Create test entity
        mock_entity = Mock()
        mock_entity.type.input_type.value = "motion"
        mock_entity.type.weight = 0.85
        mock_entity.prob_given_true = 0.8
        mock_entity.prob_given_false = 0.1
        mock_entity.active_range = (
            float("inf"),
            float("-inf"),
        )  # Should result in [None, None]
        mock_entity.active_states = ["on"]
        mock_entity.learned_gaussian_params = None
        mock_entity.analysis_error = None
        mock_entity.get_likelihoods = Mock(return_value=(0.8, 0.1))
        area._entities = type(
            "obj", (object,), {"entities": {"binary_sensor.motion1": mock_entity}}
        )()
        coordinator.db.import_stats = {"binary_sensor.motion1": 100}

        # Mock area methods that are called by _build_analysis_data
        area.probability = Mock(return_value=0.5)
        area.occupied = Mock(return_value=False)
        area.threshold = Mock(return_value=0.5)
        area.area_prior = Mock(return_value=0.3)

        # Mock prior attributes
        area.prior.global_prior = 0.3
        # Set cached time_prior values (time_prior is a property with cache)
        area.prior._cached_time_prior = None
        area.prior.sensor_ids = ["binary_sensor.motion1"]

        # Mock run_analysis method - it always runs for all areas
        coordinator.run_analysis = AsyncMock()

        _setup_coordinator_test(hass, mock_config_entry, coordinator)
        mock_service_call = _create_service_call()

        result = await _run_analysis(hass, mock_service_call)

        assert isinstance(result, dict)
        assert "areas" in result
        assert "update_timestamp" in result
        assert isinstance(result["areas"], dict)

        # Verify output structure
        area_data = result["areas"][area_name]
        entity_data = area_data["likelihoods"]["binary_sensor.motion1"]

        # Binary sensors (with active_states) don't have active_range
        assert "active_range" not in entity_data

        # Verify analysis_data and analysis_error are always included (even if None)
        assert "analysis_data" in entity_data
        assert entity_data["analysis_data"] is None
        assert "analysis_error" in entity_data
        assert entity_data["analysis_error"] is None

    async def test_run_analysis_missing_entry_id(self, hass: HomeAssistant) -> None:
        """Test analysis run with missing entry_id."""
        hass.data[DOMAIN] = None
        mock_service_call = Mock(spec=ServiceCall)
        mock_service_call.data = {"entry_id": None}

        with pytest.raises(
            HomeAssistantError,
            match="Area Occupancy coordinator not found",
        ):
            await _run_analysis(hass, mock_service_call)

    async def test_run_analysis_coordinator_error(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test analysis run with coordinator error."""
        coordinator.run_analysis = AsyncMock(
            side_effect=RuntimeError("Analysis failed")
        )

        _setup_coordinator_test(hass, mock_config_entry, coordinator)
        mock_service_call = _create_service_call()

        with pytest.raises(
            HomeAssistantError,
            match="Failed to run analysis.*Analysis failed",
        ):
            await _run_analysis(hass, mock_service_call)


class TestAsyncSetupServices:
    """Test async_setup_services function."""

    async def test_async_setup_services_registers_services(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that services are registered."""

        hass.data[DOMAIN] = coordinator
        await async_setup_services(hass)

        # Verify services are registered
        services = hass.services.async_services().get(DOMAIN, {})
        assert "run_analysis" in services
        assert "reset_entities" not in services
        assert "get_entity_metrics" not in services
        assert "get_problematic_entities" not in services
        assert "get_area_status" not in services

        # Verify service has no schema (no parameters)
        service = services["run_analysis"]
        assert service.schema is None
