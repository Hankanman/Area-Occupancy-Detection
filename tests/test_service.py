"""Tests for service module."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.const import DOMAIN
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.decay import Decay as DecayClass
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.service import (
    _build_analysis_data,
    _collect_entity_states,
    _collect_likelihood_data,
    _run_analysis,
    async_setup_services,
)
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError


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


def _create_test_entity_type(
    input_type: InputType = InputType.MOTION,
    weight: float = 0.85,
    prob_given_true: float = 0.8,
    prob_given_false: float = 0.1,
    active_states: list[str] | None = None,
    active_range: tuple[float, float] | None = None,
) -> EntityType:
    """Create a test EntityType with default values.

    Args:
        input_type: The input type enum
        weight: Weight for this entity type
        prob_given_true: Probability given true occupancy
        prob_given_false: Probability given false occupancy
        active_states: List of active states (for binary sensors)
        active_range: Tuple of (min, max) for active range (for numeric sensors)

    Returns:
        EntityType instance
    """
    if active_states is None and active_range is None:
        active_states = ["on"]  # Default for binary sensors
    return EntityType(
        input_type=input_type,
        weight=weight,
        prob_given_true=prob_given_true,
        prob_given_false=prob_given_false,
        active_states=active_states,
        active_range=active_range,
    )


def _create_test_entity(
    entity_id: str,
    entity_type: EntityType,
    hass: HomeAssistant,
    prob_given_true: float | None = None,
    prob_given_false: float | None = None,
    **kwargs,
) -> Entity:
    """Create a test Entity with default values.

    Args:
        entity_id: The entity ID
        entity_type: The EntityType instance
        hass: Home Assistant instance
        prob_given_true: Probability given true (defaults to entity_type value)
        prob_given_false: Probability given false (defaults to entity_type value)
        **kwargs: Additional entity attributes (learned_gaussian_params, analysis_error, etc.)

    Returns:
        Entity instance
    """
    return Entity(
        entity_id=entity_id,
        type=entity_type,
        prob_given_true=prob_given_true or entity_type.prob_given_true,
        prob_given_false=prob_given_false or entity_type.prob_given_false,
        decay=DecayClass(half_life=60.0),
        hass=hass,
        **kwargs,
    )


def _add_entity_to_area(
    area: Any,
    entity: Entity,
    hass: HomeAssistant,
    state: str | None = None,
) -> None:
    """Add an entity to an area and optionally set up hass.states.

    Args:
        area: The area to add the entity to
        entity: The entity to add
        hass: Home Assistant instance
        state: Optional state to set in hass.states for the entity
    """
    area.entities.entities[entity.entity_id] = entity
    if state is not None:
        hass.states.async_set(entity.entity_id, state)


class TestRunAnalysis:
    """Test _run_analysis service function."""

    async def test_run_analysis_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful analysis run with real coordinator and area."""
        # Set up coordinator with test data - use area-based access
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Set up real entity states in hass
        entity_id = "binary_sensor.motion1"
        hass.states.async_set(entity_id, "on")

        # Ensure area has entities (from coordinator fixture)
        # The coordinator fixture loads areas from config, so entities should exist
        assert len(area.entities.entities) > 0

        # Mock run_analysis method - it always runs for all areas
        coordinator.run_analysis = AsyncMock()

        _setup_coordinator_test(hass, mock_config_entry, coordinator)
        mock_service_call = _create_service_call()

        result = await _run_analysis(hass, mock_service_call)

        assert isinstance(result, dict)
        assert "areas" in result
        assert "update_timestamp" in result
        assert isinstance(result["areas"], dict)

        # Verify output structure for the area
        assert area_name in result["areas"]
        area_data = result["areas"][area_name]

        # Verify required fields are present
        assert "area_name" in area_data
        assert "purpose" in area_data
        assert "current_probability" in area_data
        assert "current_occupied" in area_data
        assert "current_threshold" in area_data
        assert "entity_states" in area_data
        assert "likelihoods" in area_data
        assert "total_entities" in area_data

        # Verify entity states were collected
        assert isinstance(area_data["entity_states"], dict)
        assert len(area_data["entity_states"]) == len(area.entities.entities)

        # Verify likelihoods were collected
        assert isinstance(area_data["likelihoods"], dict)
        assert len(area_data["likelihoods"]) == len(area.entities.entities)

        # Verify timestamp format
        assert isinstance(result["update_timestamp"], str)
        assert "T" in result["update_timestamp"] or "Z" in result["update_timestamp"]

    async def test_run_analysis_missing_coordinator(self, hass: HomeAssistant) -> None:
        """Test analysis run with missing coordinator."""
        hass.data[DOMAIN] = None
        mock_service_call = Mock(spec=ServiceCall)
        mock_service_call.data = {}

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


class TestCollectEntityStates:
    """Test _collect_entity_states helper function."""

    def test_collect_entity_states_with_existing_entities(
        self, hass: HomeAssistant, default_area
    ) -> None:
        """Test collecting states for entities that exist in hass.states."""
        # Set up entities
        entity_id1 = "binary_sensor.motion1"
        entity_id2 = "binary_sensor.motion2"
        entity_type = _create_test_entity_type()
        entity1 = _create_test_entity(entity_id1, entity_type, hass)
        entity2 = _create_test_entity(entity_id2, entity_type, hass)

        _add_entity_to_area(default_area, entity1, hass, state="on")
        _add_entity_to_area(default_area, entity2, hass, state="off")

        result = _collect_entity_states(hass, default_area)

        assert isinstance(result, dict)
        assert result[entity_id1] == "on"
        assert result[entity_id2] == "off"

    def test_collect_entity_states_with_missing_entities(
        self, hass: HomeAssistant, default_area
    ) -> None:
        """Test collecting states for entities that don't exist in hass.states."""
        entity_id = "binary_sensor.missing"
        entity_type = _create_test_entity_type()
        entity = _create_test_entity(entity_id, entity_type, hass)

        _add_entity_to_area(default_area, entity, hass, state=None)

        result = _collect_entity_states(hass, default_area)

        assert isinstance(result, dict)
        assert result[entity_id] == "NOT_FOUND"

    def test_collect_entity_states_with_empty_list(
        self, hass: HomeAssistant, default_area
    ) -> None:
        """Test collecting states when area has no entities."""
        # Clear entities
        default_area.entities.entities.clear()

        result = _collect_entity_states(hass, default_area)

        assert isinstance(result, dict)
        assert len(result) == 0


class TestCollectLikelihoodData:
    """Test _collect_likelihood_data helper function."""

    @pytest.mark.parametrize(
        (
            "entity_id",
            "input_type",
            "active_states",
            "active_range",
            "has_active_range",
            "expected_active_range",
            "analysis_data",
            "analysis_error",
            "correlation_type",
        ),
        [
            (
                "binary_sensor.motion1",
                InputType.MOTION,
                ["on"],
                None,
                False,
                None,
                None,
                None,
                None,
            ),
            (
                "sensor.temperature",
                InputType.TEMPERATURE,
                None,
                (float("-inf"), float("inf")),
                True,
                [None, None],
                None,
                None,
                None,
            ),
            (
                "binary_sensor.motion1",
                InputType.MOTION,
                ["on"],
                None,
                False,
                None,
                {"mean": 25.0, "std": 5.0},
                None,
                "positive",
            ),
        ],
    )
    def test_collect_likelihood_data_entity_configurations(
        self,
        hass: HomeAssistant,
        default_area,
        entity_id: str,
        input_type: InputType,
        active_states: list[str] | None,
        active_range: tuple[float, float] | None,
        has_active_range: bool,
        expected_active_range: list | None,
        analysis_data: dict | None,
        analysis_error: str | None,
        correlation_type: str | None,
    ) -> None:
        """Test collecting likelihood data for different entity configurations."""
        entity_type = _create_test_entity_type(
            input_type=input_type,
            active_states=active_states,
            active_range=active_range,
        )
        entity = _create_test_entity(
            entity_id,
            entity_type,
            hass,
            learned_gaussian_params=analysis_data,
            analysis_error=analysis_error,
            correlation_type=correlation_type,
        )

        _add_entity_to_area(default_area, entity, hass)

        result = _collect_likelihood_data(default_area)

        assert isinstance(result, dict)
        assert entity_id in result
        entity_data = result[entity_id]

        # Verify active_range handling
        if has_active_range:
            assert "active_range" in entity_data
            assert entity_data["active_range"] == expected_active_range
        else:
            assert "active_range" not in entity_data

        # Verify required fields
        assert "type" in entity_data
        assert "weight" in entity_data
        assert "prob_given_true" in entity_data
        assert "prob_given_false" in entity_data
        assert "is_active" in entity_data

        # Verify active_states if provided (binary sensors)
        if active_states is not None:
            assert "active_states" in entity_data
            assert entity_data["active_states"] == active_states

        # Verify analysis fields are always included
        assert "analysis_data" in entity_data
        assert "analysis_error" in entity_data
        assert "correlation_type" in entity_data

        # Verify analysis data values if provided
        if analysis_data is not None:
            assert entity_data["analysis_data"] == analysis_data
        if correlation_type is not None:
            assert entity_data["correlation_type"] == correlation_type

    def test_collect_likelihood_data_filters_none_values(
        self, hass: HomeAssistant, default_area
    ) -> None:
        """Test that None values are filtered except for analysis fields."""
        entity_id = "binary_sensor.motion1"
        entity_type = _create_test_entity_type()
        entity = _create_test_entity(entity_id, entity_type, hass)

        _add_entity_to_area(default_area, entity, hass)

        result = _collect_likelihood_data(default_area)

        assert isinstance(result, dict)
        entity_data = result[entity_id]

        # active_range should not be present (None value filtered)
        assert "active_range" not in entity_data

        # But analysis fields should still be present even if None
        assert "analysis_data" in entity_data
        assert "analysis_error" in entity_data
        assert "correlation_type" in entity_data


class TestBuildAnalysisData:
    """Test _build_analysis_data helper function."""

    def test_build_analysis_data_with_real_area(
        self, hass: HomeAssistant, default_area
    ) -> None:
        """Test building analysis data with real area."""
        area_name = default_area.area_name

        # Set up entity state
        entity_id = "binary_sensor.motion1"
        hass.states.async_set(entity_id, "on")

        result = _build_analysis_data(hass, default_area, area_name)

        assert isinstance(result, dict)
        assert result["area_name"] == area_name
        assert "purpose" in result
        assert "current_probability" in result
        assert "current_occupied" in result
        assert "current_threshold" in result
        assert "entity_states" in result
        assert "likelihoods" in result
        assert "total_entities" in result

    def test_build_analysis_data_filters_none_values(
        self, hass: HomeAssistant, default_area
    ) -> None:
        """Test that None values are filtered from result."""
        area_name = default_area.area_name

        # Set prior values to None
        default_area.prior.global_prior = None
        # Note: time_prior is a property that loads from DB if cache is None,
        # so it will always return a value (DEFAULT_TIME_PRIOR), not None
        default_area.prior.sensor_ids = []

        result = _build_analysis_data(hass, default_area, area_name)

        # None values should be filtered
        assert "global_prior" not in result
        # time_prior will always have a value (property loads from DB if needed)
        # so it should be present in the result
        assert "time_prior" in result
        assert result["time_prior"] is not None
        # Empty lists are NOT filtered (only None values are filtered)
        assert "prior_entity_ids" in result
        assert result["prior_entity_ids"] == []

    def test_build_analysis_data_with_empty_entities(
        self, hass: HomeAssistant, default_area
    ) -> None:
        """Test building analysis data when area has no entities."""
        area_name = default_area.area_name

        # Clear entities
        default_area.entities.entities.clear()

        result = _build_analysis_data(hass, default_area, area_name)

        assert isinstance(result, dict)
        assert result["total_entities"] == 0
        assert result["entity_states"] == {}
        assert result["likelihoods"] == {}


class TestRunAnalysisMultipleAreas:
    """Test _run_analysis with multiple areas."""

    async def test_run_analysis_processes_all_areas(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test that _run_analysis processes all areas."""
        # Set up entity states
        hass.states.async_set("binary_sensor.kitchen_motion", "on")

        # Mock run_analysis method
        coordinator.run_analysis = AsyncMock()

        _setup_coordinator_test(hass, mock_config_entry, coordinator)
        mock_service_call = _create_service_call()

        result = await _run_analysis(hass, mock_service_call)

        assert isinstance(result, dict)
        assert "areas" in result
        assert isinstance(result["areas"], dict)

        # Verify all areas are in result
        area_names = coordinator.get_area_names()
        assert len(result["areas"]) == len(area_names)

        for area_name in area_names:
            assert area_name in result["areas"]
            area_data = result["areas"][area_name]
            assert "area_name" in area_data
            assert area_data["area_name"] == area_name
            assert "entity_states" in area_data
            assert "likelihoods" in area_data


class TestAsyncSetupServices:
    """Test async_setup_services function."""

    async def test_async_setup_services_registers_and_can_be_called(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that services are registered and can be called."""
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

        # Mock run_analysis to avoid actual execution
        coordinator.run_analysis = AsyncMock()

        # Call service through Home Assistant
        response = await hass.services.async_call(
            DOMAIN,
            "run_analysis",
            {},
            blocking=True,
            return_response=True,
        )

        # Verify response structure
        assert isinstance(response, dict)
        assert "areas" in response
        assert "update_timestamp" in response

        # Verify run_analysis was called
        coordinator.run_analysis.assert_called_once()

    async def test_async_setup_services_supports_response_only(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that service is registered with SupportsResponse.ONLY."""
        hass.data[DOMAIN] = coordinator
        await async_setup_services(hass)

        services = hass.services.async_services().get(DOMAIN, {})
        service = services["run_analysis"]

        # Verify SupportsResponse.ONLY is set
        assert service.supports_response == SupportsResponse.ONLY
