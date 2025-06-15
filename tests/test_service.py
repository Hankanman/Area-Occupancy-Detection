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
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError


class TestGetCoordinator:
    """Test _get_coordinator helper function."""

    def test_get_coordinator_success(self) -> None:
        """Test successful coordinator retrieval."""
        mock_hass = Mock(spec=HomeAssistant)
        mock_coordinator = Mock()

        mock_hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}

        result = _get_coordinator(mock_hass, "test_entry_id")
        assert result == mock_coordinator

    def test_get_coordinator_missing_domain(self) -> None:
        """Test coordinator retrieval with missing domain."""
        mock_hass = Mock(spec=HomeAssistant)
        mock_hass.data = {}

        with pytest.raises(ServiceValidationError, match="Integration not found"):
            _get_coordinator(mock_hass, "test_entry_id")

    def test_get_coordinator_missing_entry(self) -> None:
        """Test coordinator retrieval with missing entry."""
        mock_hass = Mock(spec=HomeAssistant)
        mock_hass.data = {DOMAIN: {}}

        with pytest.raises(ServiceValidationError, match="Entry not found"):
            _get_coordinator(mock_hass, "test_entry_id")


class TestUpdatePriors:
    """Test _update_priors service function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        mock_coordinator = Mock()
        mock_coordinator.update_learned_priors = AsyncMock(return_value=5)

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        return hass

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
        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        coordinator.update_learned_priors.side_effect = Exception("Update failed")

        with pytest.raises(Exception, match="Update failed"):
            await _update_priors(mock_hass, mock_call)


class TestResetEntities:
    """Test _reset_entities service function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()
        mock_coordinator.entity_manager.reset_entities = AsyncMock()

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        return hass

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
        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        coordinator.entity_manager = None

        with pytest.raises(
            ServiceValidationError, match="Entity manager not available"
        ):
            await _reset_entities(mock_hass, mock_call)


class TestGetEntityMetrics:
    """Test _get_entity_metrics service function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
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

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        return hass

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
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entities with various states
        mock_entity1 = Mock()
        mock_entity1.entity_id = "binary_sensor.motion1"
        mock_entity1.available = False  # Problematic - unavailable

        mock_entity2 = Mock()
        mock_entity2.entity_id = "light.test_light"
        mock_entity2.available = True
        mock_entity2.probability = 0.95  # Problematic - very high probability

        mock_entity3 = Mock()
        mock_entity3.entity_id = "sensor.normal"
        mock_entity3.available = True
        mock_entity3.probability = 0.5  # Normal

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity1,
            "light.test_light": mock_entity2,
            "sensor.normal": mock_entity3,
        }

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        return hass

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
        result = await _get_problematic_entities(mock_hass, mock_call)

        assert "problematic_entities" in result
        problematic = result["problematic_entities"]

        # Should identify unavailable entity
        assert "binary_sensor.motion1" in problematic
        assert "unavailable" in problematic["binary_sensor.motion1"]["issues"]

        # Should identify high probability entity
        assert "light.test_light" in problematic
        assert "high_probability" in problematic["light.test_light"]["issues"]

        # Should not include normal entity
        assert "sensor.normal" not in problematic

    async def test_get_problematic_entities_with_threshold(
        self, mock_hass: Mock
    ) -> None:
        """Test problematic entities with custom threshold."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {
            "entry_id": "test_entry_id",
            "probability_threshold": 90,  # Custom threshold
        }
        mock_call.return_response = True

        result = await _get_problematic_entities(mock_hass, mock_call)

        # With 90% threshold, 95% probability should still be flagged
        problematic = result["problematic_entities"]
        assert "light.test_light" in problematic

    async def test_get_problematic_entities_no_issues(self, mock_hass: Mock) -> None:
        """Test problematic entities when no issues exist."""
        # Update entities to have no issues
        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        for entity in coordinator.entity_manager.entities.values():
            entity.available = True
            entity.probability = 0.5

        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        result = await _get_problematic_entities(mock_hass, mock_call)
        assert result["problematic_entities"] == {}


class TestGetEntityDetails:
    """Test _get_entity_details service function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entity with detailed information
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.probability = 0.75
        mock_entity.is_active = True
        mock_entity.available = True
        mock_entity.state = "on"
        mock_entity.type.input_type.value = "motion"
        mock_entity.type.weight = 0.8
        mock_entity.prior.prior = 0.3
        mock_entity.prior.prob_given_true = 0.9
        mock_entity.prior.prob_given_false = 0.1
        mock_entity.decay.is_decaying = False
        mock_entity.decay.decay_factor = 1.0

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity,
        }

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        return hass

    async def test_get_entity_details_success(self, mock_hass: Mock) -> None:
        """Test successful entity details retrieval."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {
            "entry_id": "test_entry_id",
            "entity_id": "binary_sensor.motion1",
        }
        mock_call.return_response = True

        result = await _get_entity_details(mock_hass, mock_call)

        assert "entity_details" in result
        details = result["entity_details"]

        assert details["entity_id"] == "binary_sensor.motion1"
        assert details["probability"] == 75.0
        assert details["is_active"] is True
        assert details["available"] is True
        assert details["state"] == "on"
        assert details["type"] == "motion"
        assert details["weight"] == 80.0
        assert "prior" in details
        assert "decay" in details

    async def test_get_entity_details_missing_entity_id(self, mock_hass: Mock) -> None:
        """Test entity details with missing entity_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        with pytest.raises(ServiceValidationError, match="entity_id is required"):
            await _get_entity_details(mock_hass, mock_call)

    async def test_get_entity_details_entity_not_found(self, mock_hass: Mock) -> None:
        """Test entity details with entity not found."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {
            "entry_id": "test_entry_id",
            "entity_id": "binary_sensor.nonexistent",
        }
        mock_call.return_response = True

        with pytest.raises(ServiceValidationError, match="Entity not found"):
            await _get_entity_details(mock_hass, mock_call)


class TestForceEntityUpdate:
    """Test _force_entity_update service function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        mock_coordinator = Mock()
        mock_coordinator.entity_manager = Mock()

        # Mock entity
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.update_probability = Mock()

        mock_coordinator.entity_manager.entities = {
            "binary_sensor.motion1": mock_entity,
        }

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        return hass

    async def test_force_entity_update_success(self, mock_hass: Mock) -> None:
        """Test successful forced entity update."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {
            "entry_id": "test_entry_id",
            "entity_id": "binary_sensor.motion1",
        }

        await _force_entity_update(mock_hass, mock_call)

        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        entity = coordinator.entity_manager.entities["binary_sensor.motion1"]
        entity.update_probability.assert_called_once()

    async def test_force_entity_update_all_entities(self, mock_hass: Mock) -> None:
        """Test forced update for all entities."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}

        await _force_entity_update(mock_hass, mock_call)

        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        entity = coordinator.entity_manager.entities["binary_sensor.motion1"]
        entity.update_probability.assert_called_once()

    async def test_force_entity_update_entity_not_found(self, mock_hass: Mock) -> None:
        """Test forced update with entity not found."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {
            "entry_id": "test_entry_id",
            "entity_id": "binary_sensor.nonexistent",
        }

        with pytest.raises(ServiceValidationError, match="Entity not found"):
            await _force_entity_update(mock_hass, mock_call)


class TestGetAreaStatus:
    """Test _get_area_status service function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        mock_coordinator = Mock()
        mock_coordinator.probability = 0.75
        mock_coordinator.is_occupied = True
        mock_coordinator.threshold = 0.6
        mock_coordinator.prior = 0.35
        mock_coordinator.decay = 0.9
        mock_coordinator.available = True
        mock_coordinator.last_updated = "2023-01-01T12:00:00+00:00"
        mock_coordinator.last_changed = "2023-01-01T11:30:00+00:00"

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        return hass

    async def test_get_area_status_success(self, mock_hass: Mock) -> None:
        """Test successful area status retrieval."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        result = await _get_area_status(mock_hass, mock_call)

        assert "area_status" in result
        status = result["area_status"]

        assert status["probability"] == 75.0
        assert status["is_occupied"] is True
        assert status["threshold"] == 60.0
        assert status["prior"] == 35.0
        assert status["decay"] == 90.0
        assert status["available"] is True
        assert "last_updated" in status
        assert "last_changed" in status

    async def test_get_area_status_unavailable_coordinator(
        self, mock_hass: Mock
    ) -> None:
        """Test area status with unavailable coordinator."""
        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        coordinator.available = False

        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        result = await _get_area_status(mock_hass, mock_call)

        status = result["area_status"]
        assert status["available"] is False


class TestGetEntityTypeLearned:
    """Test _get_entity_type_learned_data service function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        mock_coordinator = Mock()
        mock_coordinator.entity_types = Mock()

        # Mock entity type
        mock_entity_type = Mock()
        mock_entity_type.input_type.value = "motion"
        mock_entity_type.weight = 0.8
        mock_entity_type.prob_true = 0.9
        mock_entity_type.prob_false = 0.1
        mock_entity_type.prior = 0.3

        mock_coordinator.entity_types.entity_types = {"motion": mock_entity_type}

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        return hass

    async def test_get_entity_type_learned_data_success(self, mock_hass: Mock) -> None:
        """Test successful entity type learned data retrieval."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        result = await _get_entity_type_learned_data(mock_hass, mock_call)

        assert "entity_types" in result
        types_data = result["entity_types"]

        assert "motion" in types_data
        motion_data = types_data["motion"]
        assert motion_data["weight"] == 80.0
        assert motion_data["prob_true"] == 90.0
        assert motion_data["prob_false"] == 10.0
        assert motion_data["prior"] == 30.0

    async def test_get_entity_type_learned_data_no_entity_types(
        self, mock_hass: Mock
    ) -> None:
        """Test entity type learned data with no entity types."""
        coordinator = mock_hass.data[DOMAIN]["test_entry_id"]
        coordinator.entity_types = None

        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}
        mock_call.return_response = True

        result = await _get_entity_type_learned_data(mock_hass, mock_call)
        assert result["entity_types"] == {}


class TestAsyncSetupServices:
    """Test async_setup_services function."""

    async def test_async_setup_services_success(self) -> None:
        """Test successful service setup."""
        mock_hass = Mock(spec=HomeAssistant)
        mock_hass.services.async_register = Mock()

        await async_setup_services(mock_hass)

        # Verify all services were registered
        assert mock_hass.services.async_register.call_count == 8

        # Check that all expected services were registered
        registered_services = [
            call[0][1] for call in mock_hass.services.async_register.call_args_list
        ]
        expected_services = [
            "update_priors",
            "reset_entities",
            "get_entity_metrics",
            "get_problematic_entities",
            "get_entity_details",
            "force_entity_update",
            "get_area_status",
            "get_entity_type_learned_data",
        ]

        for service in expected_services:
            assert service in registered_services

    async def test_async_setup_services_registration_error(self) -> None:
        """Test service setup with registration error."""
        mock_hass = Mock(spec=HomeAssistant)
        mock_hass.services.async_register.side_effect = Exception("Registration failed")

        # Should not raise exception, but log error
        await async_setup_services(mock_hass)


class TestServiceIntegration:
    """Test service integration scenarios."""

    @pytest.fixture
    def comprehensive_hass(self) -> Mock:
        """Create a comprehensive mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)

        # Mock coordinator with comprehensive data
        mock_coordinator = Mock()
        mock_coordinator.probability = 0.65
        mock_coordinator.is_occupied = True
        mock_coordinator.threshold = 0.6
        mock_coordinator.prior = 0.35
        mock_coordinator.decay = 0.85
        mock_coordinator.available = True
        mock_coordinator.update_learned_priors = AsyncMock(return_value=3)

        # Mock entity manager
        mock_coordinator.entity_manager = Mock()
        mock_coordinator.entity_manager.reset_entities = AsyncMock()

        # Mock entities
        mock_entities = {}
        for entity_id, probability, is_active, available in [
            ("binary_sensor.motion1", 0.8, True, True),
            ("binary_sensor.motion2", 0.3, False, True),
            (
                "light.test_light",
                0.95,
                True,
                False,
            ),  # Problematic - unavailable and high prob
        ]:
            mock_entity = Mock()
            mock_entity.entity_id = entity_id
            mock_entity.probability = probability
            mock_entity.is_active = is_active
            mock_entity.available = available
            mock_entity.update_probability = Mock()
            mock_entities[entity_id] = mock_entity

        mock_coordinator.entity_manager.entities = mock_entities

        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}

        return hass

    async def test_service_workflow_integration(self, comprehensive_hass: Mock) -> None:
        """Test complete service workflow integration."""
        # 1. Get area status
        status_call = Mock(spec=ServiceCall)
        status_call.data = {"entry_id": "test_entry_id"}
        status_call.return_response = True

        status_result = await _get_area_status(comprehensive_hass, status_call)
        assert status_result["area_status"]["is_occupied"] is True

        # 2. Get problematic entities
        problematic_call = Mock(spec=ServiceCall)
        problematic_call.data = {"entry_id": "test_entry_id"}
        problematic_call.return_response = True

        problematic_result = await _get_problematic_entities(
            comprehensive_hass, problematic_call
        )
        assert "light.test_light" in problematic_result["problematic_entities"]

        # 3. Force entity update
        update_call = Mock(spec=ServiceCall)
        update_call.data = {
            "entry_id": "test_entry_id",
            "entity_id": "light.test_light",
        }

        await _force_entity_update(comprehensive_hass, update_call)

        coordinator = comprehensive_hass.data[DOMAIN]["test_entry_id"]
        entity = coordinator.entity_manager.entities["light.test_light"]
        entity.update_probability.assert_called_once()

        # 4. Update priors
        priors_call = Mock(spec=ServiceCall)
        priors_call.data = {"entry_id": "test_entry_id"}

        await _update_priors(comprehensive_hass, priors_call)
        coordinator.update_learned_priors.assert_called_once()

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

        with pytest.raises(ServiceValidationError):
            await _update_priors(comprehensive_hass, invalid_call)

        with pytest.raises(ServiceValidationError):
            await _reset_entities(comprehensive_hass, invalid_call)

    async def test_service_parameter_validation(self, comprehensive_hass: Mock) -> None:
        """Test parameter validation across services."""
        # Test missing required parameters
        empty_call = Mock(spec=ServiceCall)
        empty_call.data = {}
        empty_call.return_response = True

        with pytest.raises(ServiceValidationError, match="entry_id is required"):
            await _get_entity_details(comprehensive_hass, empty_call)

        with pytest.raises(ServiceValidationError, match="entry_id is required"):
            await _force_entity_update(comprehensive_hass, empty_call)

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
        assert "area_status" in status_result

        metrics_result = await _get_entity_metrics(comprehensive_hass, call)
        assert isinstance(metrics_result, dict)
        assert "entities" in metrics_result

        problematic_result = await _get_problematic_entities(comprehensive_hass, call)
        assert isinstance(problematic_result, dict)
        assert "problematic_entities" in problematic_result
