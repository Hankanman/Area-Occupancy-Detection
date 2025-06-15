"""Tests for service module."""

from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.const import DOMAIN
from custom_components.area_occupancy.service import (
    _force_entity_update,
    _get_area_status,
    _get_coordinator,
    _get_entity_details,
    _get_entity_metrics,
    _get_entity_type_learned_data,
    _get_problematic_entities,
    _reset_entities,
    _update_priors,
    async_setup_services,
)
from homeassistant.core import ServiceCall
from homeassistant.exceptions import ServiceValidationError


class TestGetCoordinator:
    """Test _get_coordinator helper function."""

    def test_get_coordinator_success(self, mock_hass: Mock) -> None:
        """Test successful coordinator retrieval."""
        mock_coordinator = Mock()
        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        result = _get_coordinator(mock_hass, "test_entry_id")
        assert result == mock_coordinator

    def test_get_coordinator_missing_domain(self, mock_hass: Mock) -> None:
        """Test coordinator retrieval with missing domain."""
        mock_hass.data = {}

        with pytest.raises(ServiceValidationError, match="Integration not found"):
            _get_coordinator(mock_hass, "test_entry_id")

    def test_get_coordinator_missing_entry(self, mock_hass: Mock) -> None:
        """Test coordinator retrieval with missing entry."""
        mock_hass.data[DOMAIN] = {}

        with pytest.raises(ServiceValidationError, match="Entry not found"):
            _get_coordinator(mock_hass, "test_entry_id")


class TestUpdatePriors:
    """Test _update_priors service function."""

    @pytest.fixture
    def mock_call(self) -> Mock:
        """Create a mock service call."""
        call = Mock(spec=ServiceCall)
        call.data = {"entry_id": "test_entry_id"}
        return call

    async def test_update_priors_success(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test successful prior update."""
        mock_coordinator = Mock()
        mock_coordinator.update_learned_priors = AsyncMock(return_value=5)
        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        await _update_priors(mock_hass, mock_call)

        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        coordinator.update_learned_priors.assert_called_once()

    async def test_update_priors_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test prior update with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        with pytest.raises(ServiceValidationError, match="entry_id is required"):
            await _update_priors(mock_hass, mock_call)

    async def test_update_priors_coordinator_error(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test prior update with coordinator error."""
        mock_coordinator = Mock()
        mock_coordinator.update_learned_priors = AsyncMock(
            side_effect=Exception("Update failed")
        )
        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        with pytest.raises(Exception, match="Update failed"):
            await _update_priors(mock_hass, mock_call)


class TestResetEntities:
    """Test _reset_entities service function."""

    @pytest.fixture
    def mock_call(self) -> Mock:
        """Create a mock service call."""
        call = Mock(spec=ServiceCall)
        call.data = {"entry_id": "test_entry_id"}
        return call

    async def test_reset_entities_success(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test successful entity reset."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()
        mock_coordinator.entity_manager.reset_entities = AsyncMock()
        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        await _reset_entities(mock_hass, mock_call)

        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        coordinator.entity_manager.reset_entities.assert_called_once()

    async def test_reset_entities_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test entity reset with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        with pytest.raises(ServiceValidationError, match="entry_id is required"):
            await _reset_entities(mock_hass, mock_call)

    async def test_reset_entities_no_entity_manager(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test entity reset with no entity manager."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = None
        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        with pytest.raises(
            ServiceValidationError, match="Entity manager not available"
        ):
            await _reset_entities(mock_hass, mock_call)


class TestGetEntityMetrics:
    """Test _get_entity_metrics service function."""

    @pytest.fixture
    def mock_call(self) -> Mock:
        """Create a mock service call."""
        call = Mock(spec=ServiceCall)
        call.data = {"entry_id": "test_entry_id"}
        call.return_response = True
        return call

    async def test_get_entity_metrics_success(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test successful entity metrics retrieval."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entities with metrics
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.probability = 0.75
        mock_entity1.is_active = True
        mock_entity1.available = True

        mock_entity2 = Mock()
        mock_entity2.entity_id = "light.test_light"
        mock_entity2.probability = 0.25
        mock_entity2.is_active = False
        mock_entity2.available = True

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity1,
            "light.test_light": mock_entity2,
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        result = await _get_entity_metrics(mock_hass, mock_call)

        assert "entities" in result
        assert len(result["entities"]) == 2

        # Check entity data structure
        entity_data = result["entities"]
        assert "binary_sensor.motion1" in entity_data
        assert "light.test_light" in entity_data

        motion_data = entity_data["binary_sensor.motion1"]
        assert motion_data["probability"] == 75.0  # Converted to percentage
        assert motion_data["is_active"] is True
        assert motion_data["available"] is True

    async def test_get_entity_metrics_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test entity metrics with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}
        mock_call.return_response = True

        with pytest.raises(ServiceValidationError, match="entry_id is required"):
            await _get_entity_metrics(mock_hass, mock_call)

    async def test_get_entity_metrics_no_entity_manager(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test entity metrics with no entity manager."""
        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        coordinator.entity_manager = None

        result = await _get_entity_metrics(mock_hass, mock_call)
        assert result["entities"] == {}


class TestGetProblematicEntities:
    """Test _get_problematic_entities service function."""

    @pytest.fixture
    def mock_call(self) -> Mock:
        """Create a mock service call."""
        call = Mock(spec=ServiceCall)
        call.data = {"entry_id": "test_entry_id"}
        call.return_response = True
        return call

    async def test_get_problematic_entities_success(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test successful problematic entities retrieval."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entities with issues
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.probability = 0.95
        mock_entity1.is_active = True
        mock_entity1.available = True

        mock_entity2 = Mock()
        mock_entity2.entity_id = "light.test_light"
        mock_entity2.probability = 0.05
        mock_entity2.is_active = False
        mock_entity2.available = True

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity1,
            "light.test_light": mock_entity2,
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        result = await _get_problematic_entities(mock_hass, mock_call)

        assert "issues" in result
        assert len(result["issues"]) == 2

    async def test_get_problematic_entities_with_threshold(
        self, mock_hass: Mock
    ) -> None:
        """Test problematic entities with custom threshold."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entities with issues
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.probability = 0.95
        mock_entity.is_active = True
        mock_entity.available = True

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity,
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id", "threshold": 0.99}
        mock_call.return_response = True

        result = await _get_problematic_entities(mock_hass, mock_call)

        assert "issues" in result
        assert len(result["issues"]) == 0

    async def test_get_problematic_entities_no_issues(self, mock_hass: Mock) -> None:
        """Test problematic entities with no issues."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entities with no issues
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.probability = 0.5
        mock_entity.is_active = True
        mock_entity.available = True

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity,
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        result = await _get_problematic_entities(mock_hass, mock_call)

        assert "issues" in result
        assert len(result["issues"]) == 0


class TestGetEntityDetails:
    """Test _get_entity_details service function."""

    @pytest.fixture
    def mock_call(self) -> Mock:
        """Create a mock service call."""
        call = Mock(spec=ServiceCall)
        call.data = {
            "entry_id": "test_entry_id",
            "entity_id": "binary_sensor.motion1",
        }
        call.return_response = True
        return call

    async def test_get_entity_details_success(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test successful entity details retrieval."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entity with details
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.probability = 0.75
        mock_entity.is_active = True
        mock_entity.available = True
        mock_entity.state = "on"
        mock_entity.last_updated = "2024-01-01T00:00:00"
        mock_entity.last_changed = "2024-01-01T00:00:00"

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity,
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        result = await _get_entity_details(mock_hass, mock_call)

        assert "entity" in result
        assert result["entity"]["entity_id"] == "binary_sensor.motion1"
        assert result["entity"]["probability"] == 0.75
        assert result["entity"]["is_active"] is True
        assert result["entity"]["available"] is True
        assert result["entity"]["state"] == "on"

    async def test_get_entity_details_missing_entity_id(self, mock_hass: Mock) -> None:
        """Test entity details with missing entity_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        with pytest.raises(ServiceValidationError, match="entity_id is required"):
            await _get_entity_details(mock_hass, mock_call)

    async def test_get_entity_details_entity_not_found(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test entity details with entity not found."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()
        mock_coordinator.entity_manager.entities = {}

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        with pytest.raises(ServiceValidationError, match="Entity not found"):
            await _get_entity_details(mock_hass, mock_call)


class TestForceEntityUpdate:
    """Test _force_entity_update service function."""

    @pytest.fixture
    def mock_call(self) -> Mock:
        """Create a mock service call."""
        call = Mock(spec=ServiceCall)
        call.data = {
            "entry_id": "test_entry_id",
            "entity_id": "binary_sensor.motion1",
        }
        call.return_response = True
        return call

    async def test_force_entity_update_success(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test successful entity update."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entity
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.force_update = AsyncMock()

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity,
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        await _force_entity_update(mock_hass, mock_call)

        mock_entity.force_update.assert_called_once()

    async def test_force_entity_update_all_entities(self, mock_hass: Mock) -> None:
        """Test force update for all entities."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entities
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.force_update = AsyncMock()

        mock_entity2 = Mock()
        mock_entity2.entity_id = "light.test_light"
        mock_entity2.force_update = AsyncMock()

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity1,
            "light.test_light": mock_entity2,
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        await _force_entity_update(mock_hass, mock_call)

        mock_entity1.force_update.assert_called_once()
        mock_entity2.force_update.assert_called_once()

    async def test_force_entity_update_entity_not_found(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test force update with entity not found."""
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()
        mock_coordinator.entity_manager.entities = {}

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        with pytest.raises(ServiceValidationError, match="Entity not found"):
            await _force_entity_update(mock_hass, mock_call)


class TestGetAreaStatus:
    """Test _get_area_status service function."""

    @pytest.fixture
    def mock_call(self) -> Mock:
        """Create a mock service call."""
        call = Mock(spec=ServiceCall)
        call.data = {"entry_id": "test_entry_id"}
        call.return_response = True
        return call

    async def test_get_area_status_success(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test successful area status retrieval."""
        mock_coordinator = Mock()
        mock_coordinator.config = Mock()
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities = Mock()
        mock_coordinator.entities.entities = {}

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        # Mock occupancy state
        mock_state = Mock()
        mock_state.state = "on"
        mock_state.attributes = {"probability": 0.8}
        mock_state.last_updated = "2024-01-01T00:00:00"
        mock_hass.states.get = Mock(return_value=mock_state)

        result = await _get_area_status(mock_hass, mock_call)

        assert "area_name" in result
        assert result["area_name"] == "Test Area"
        assert result["is_occupied"] is True
        assert result["occupancy_probability"] == 0.8
        assert result["confidence_level"] == "high"

    async def test_get_area_status_unavailable_coordinator(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test area status with unavailable coordinator."""
        mock_coordinator = Mock()
        mock_coordinator.available = False

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        with pytest.raises(ServiceValidationError, match="Coordinator not available"):
            await _get_area_status(mock_hass, mock_call)


class TestGetEntityTypeLearned:
    """Test _get_entity_type_learned_data service function."""

    @pytest.fixture
    def mock_call(self) -> Mock:
        """Create a mock service call."""
        call = Mock(spec=ServiceCall)
        call.data = {"entry_id": "test_entry_id"}
        call.return_response = True
        return call

    async def test_get_entity_type_learned_data_success(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test successful entity type learned data retrieval."""
        mock_coordinator = Mock()
        mock_coordinator.entity_types = Mock()
        mock_coordinator.entity_types.entity_types = {
            "motion": Mock(
                prior=0.3,
                prob_true=0.8,
                prob_false=0.2,
                weight=1.0,
                active_states=["on"],
                active_range=None,
            )
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        result = await _get_entity_type_learned_data(mock_hass, mock_call)

        assert "motion" in result
        assert result["motion"]["prior"] == 0.3
        assert result["motion"]["prob_true"] == 0.8
        assert result["motion"]["prob_false"] == 0.2
        assert result["motion"]["weight"] == 1.0
        assert result["motion"]["active_states"] == ["on"]
        assert result["motion"]["active_range"] is None

    async def test_get_entity_type_learned_data_no_entity_types(
        self, mock_hass: Mock, mock_call: Mock
    ) -> None:
        """Test entity type learned data with no entity types."""
        mock_coordinator = Mock()
        mock_coordinator.entity_types = None

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        with pytest.raises(ServiceValidationError, match="Entity types not available"):
            await _get_entity_type_learned_data(mock_hass, mock_call)


class TestAsyncSetupServices:
    """Test async_setup_services function."""

    async def test_async_setup_services_success(self, mock_hass: Mock) -> None:
        """Test successful service setup."""
        await async_setup_services(mock_hass)

        mock_hass.services.async_register.assert_called()

    async def test_async_setup_services_registration_error(
        self, mock_hass: Mock
    ) -> None:
        """Test service setup with registration error."""
        mock_hass.services.async_register.side_effect = Exception("Registration failed")

        with pytest.raises(Exception, match="Registration failed"):
            await async_setup_services(mock_hass)


class TestServiceIntegration:
    """Test service integration."""

    @pytest.fixture
    def comprehensive_hass(self, mock_hass: Mock) -> Mock:
        """Create a comprehensive mock Home Assistant instance."""
        mock_coordinator = Mock()
        mock_coordinator.config = Mock()
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities = Mock()
        mock_coordinator.entities.entities = {}
        mock_coordinator.entity_types = Mock()
        mock_coordinator.entity_types.entity_types = {
            "motion": Mock(
                prior=0.3,
                prob_true=0.8,
                prob_false=0.2,
                weight=1.0,
                active_states=["on"],
                active_range=None,
            )
        }

        mock_hass.data[DOMAIN] = {"test_entry_id": mock_coordinator}

        # Mock occupancy state
        mock_state = Mock()
        mock_state.state = "on"
        mock_state.attributes = {"probability": 0.8}
        mock_state.last_updated = "2024-01-01T00:00:00"
        mock_hass.states.get = Mock(return_value=mock_state)

        return mock_hass

    async def test_service_workflow_integration(self, comprehensive_hass: Mock) -> None:
        """Test complete service workflow integration."""
        # 1. Get area status
        status_call = Mock(spec=ServiceCall)
        status_call.data = {"entry_id": "test_entry_id"}
        status_call.return_response = True

        status_result = await _get_area_status(comprehensive_hass, status_call)

        assert "area_name" in status_result
        assert status_result["area_name"] == "Test Area"
        assert status_result["is_occupied"] is True
        assert status_result["occupancy_probability"] == 0.8
        assert status_result["confidence_level"] == "high"

        # 2. Get entity type learned data
        learned_call = Mock(spec=ServiceCall)
        learned_call.data = {"entry_id": "test_entry_id"}
        learned_call.return_response = True

        learned_result = await _get_entity_type_learned_data(
            comprehensive_hass, learned_call
        )

        assert "motion" in learned_result
        assert learned_result["motion"]["prior"] == 0.3
        assert learned_result["motion"]["prob_true"] == 0.8
        assert learned_result["motion"]["prob_false"] == 0.2

    async def test_error_handling_across_services(
        self, comprehensive_hass: Mock
    ) -> None:
        """Test error handling across different services."""
        # Test with invalid entry_id
        invalid_call = Mock(spec=ServiceCall)
        invalid_call.data = {"entry_id": "nonexistent"}
        invalid_call.return_response = True

        with pytest.raises(ServiceValidationError):
            await _get_area_status(comprehensive_hass, invalid_call)

        # Test with missing required parameters
        empty_call = Mock(spec=ServiceCall)
        empty_call.data = {}
        empty_call.return_response = True

        with pytest.raises(ServiceValidationError, match="entry_id is required"):
            await _get_entity_details(comprehensive_hass, empty_call)

    async def test_service_parameter_validation(self, comprehensive_hass: Mock) -> None:
        """Test parameter validation across services."""
        # Test missing required parameters
        empty_call = Mock(spec=ServiceCall)
        empty_call.data = {}
        empty_call.return_response = True

        with pytest.raises(ServiceValidationError, match="entry_id is required"):
            await _get_entity_details(comprehensive_hass, empty_call)

    async def test_service_return_value_consistency(
        self, comprehensive_hass: Mock
    ) -> None:
        """Test return value consistency across services."""
        # All services that return data should have consistent structure
        call = Mock(spec=ServiceCall)
        call.data = {"entry_id": "test_entry_id"}
        call.return_response = True

        # Test return value structures
        status_result = await _get_area_status(comprehensive_hass, call)
        assert isinstance(status_result, dict)
        assert "area_name" in status_result
        assert "is_occupied" in status_result
        assert "occupancy_probability" in status_result
        assert "confidence_level" in status_result
