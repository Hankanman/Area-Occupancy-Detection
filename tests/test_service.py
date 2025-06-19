"""Tests for service module."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest

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
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


# ruff: noqa: PLC0415
class TestGetCoordinator:
    """Test _get_coordinator helper function."""

    def test_get_coordinator_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful coordinator retrieval."""
        mock_coordinator = Mock()
        mock_config_entry.runtime_data = mock_coordinator

        # Mock hass.config_entries.async_entries to return list with our config entry
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]

        result = _get_coordinator(mock_hass, "test_entry_id")
        assert result == mock_coordinator

    def test_get_coordinator_missing_domain(self, mock_hass: Mock) -> None:
        """Test coordinator retrieval with missing domain."""
        # Mock async_entries to return empty list
        mock_hass.config_entries.async_entries.return_value = []

        with pytest.raises(
            HomeAssistantError, match="Config entry test_entry_id not found"
        ):
            _get_coordinator(mock_hass, "test_entry_id")

    def test_get_coordinator_missing_entry(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test coordinator retrieval with missing entry."""
        # Different entry_id than what we're looking for
        mock_config_entry.entry_id = "different_entry_id"
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]

        with pytest.raises(
            HomeAssistantError, match="Config entry test_entry_id not found"
        ):
            _get_coordinator(mock_hass, "test_entry_id")


class TestUpdatePriors:
    """Test _update_priors service function."""

    async def test_update_priors_success(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test successful prior update."""
        mock_coordinator = Mock()
        mock_coordinator.config.history.period = (
            30  # Set as real number instead of Mock
        )
        mock_coordinator.update_learned_priors = AsyncMock(return_value=5)
        mock_coordinator.async_refresh = AsyncMock()

        # Mock entities with proper structure for return data
        mock_entity = Mock()
        mock_entity.prior.prior = 0.35
        mock_entity.prior.prob_given_true = 0.8
        mock_entity.prior.prob_given_false = 0.1
        mock_entity.prior.last_updated.isoformat.return_value = "2024-01-01T00:00:00"
        mock_entity.type.input_type.value = "motion"

        mock_coordinator.entities.entities = {"binary_sensor.motion1": mock_entity}

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]

        # The service expects the service call to have entry_id
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _update_priors(mock_hass, mock_service_call)

        # Verify the result structure
        assert "updated_priors" in result
        assert "history_period" in result
        assert "total_entities" in result
        assert "update_timestamp" in result

        # Verify the values
        assert result["history_period"] == 30
        assert result["total_entities"] == 1
        assert isinstance(result["update_timestamp"], str)

        # Verify the prior data
        priors = result["updated_priors"]
        assert "binary_sensor.motion1" in priors
        prior_data = priors["binary_sensor.motion1"]
        assert prior_data["prior"] == 0.35
        assert prior_data["prob_given_true"] == 0.8
        assert prior_data["prob_given_false"] == 0.1
        assert prior_data["entity_type"] == "motion"

        # Verify the coordinator was called correctly with the configured history period
        mock_coordinator.update_learned_priors.assert_called_once_with(30)
        mock_coordinator.async_refresh.assert_called_once()

    async def test_update_priors_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test prior update with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        # The actual service implementation will raise KeyError for missing entry_id
        with pytest.raises(KeyError):
            await _update_priors(mock_hass, mock_call)

    async def test_update_priors_coordinator_error(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test prior update with coordinator error."""
        mock_coordinator = Mock()
        mock_coordinator.config.history.period = 30  # Set as real number
        mock_coordinator.update_learned_priors = AsyncMock(
            side_effect=RuntimeError("Update failed")
        )

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        # The service catches RuntimeError and wraps it in HomeAssistantError
        with pytest.raises(
            HomeAssistantError,
            match="Failed to update priors for test_entry_id: Update failed",
        ):
            await _update_priors(mock_hass, mock_service_call)

        # Verify the coordinator was called with the correct history period
        mock_coordinator.update_learned_priors.assert_called_once_with(30)


class TestResetEntities:
    """Test _reset_entities service function."""

    async def test_reset_entities_success(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test successful entity reset."""
        mock_coordinator = Mock()
        mock_coordinator.entities.reset_entities = AsyncMock()
        mock_coordinator.async_refresh = AsyncMock()

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        await _reset_entities(mock_hass, mock_service_call)

        mock_coordinator.entities.reset_entities.assert_called_once()
        mock_coordinator.async_refresh.assert_called_once()

    async def test_reset_entities_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test entity reset with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        # The actual service implementation will raise KeyError for missing entry_id
        with pytest.raises(KeyError):
            await _reset_entities(mock_hass, mock_call)

    async def test_reset_entities_with_clear_storage(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test entity reset with storage clearing."""
        mock_coordinator = Mock()
        mock_coordinator.entities.reset_entities = AsyncMock()
        mock_coordinator.storage.async_reset = AsyncMock()
        mock_coordinator.async_refresh = AsyncMock()

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id", "clear_storage": True}

        await _reset_entities(mock_hass, mock_service_call)

        mock_coordinator.entities.reset_entities.assert_called_once()
        mock_coordinator.storage.async_reset.assert_called_once()
        mock_coordinator.async_refresh.assert_called_once()


class TestGetEntityMetrics:
    """Test _get_entity_metrics service function."""

    async def test_get_entity_metrics_success(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test successful entity metrics retrieval."""
        mock_coordinator = Mock()

        # Mock entities with metrics
        mock_entity1 = Mock()
        mock_entity1.evidence = True
        mock_entity1.available = True
        mock_entity1.decay.is_decaying = False

        mock_entity2 = Mock()
        mock_entity2.evidence = False
        mock_entity2.available = True
        mock_entity2.decay.is_decaying = True

        mock_coordinator.entities.entities = {
            "binary_sensor.motion1": mock_entity1,
            "light.test_light": mock_entity2,
        }

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _get_entity_metrics(mock_hass, mock_service_call)

        # The service returns metrics summary, not individual entity data
        assert "metrics" in result
        metrics = result["metrics"]
        assert metrics["total_entities"] == 2
        assert metrics["active_entities"] == 1
        assert metrics["available_entities"] == 2
        assert metrics["unavailable_entities"] == 0
        assert metrics["decaying_entities"] == 1

    async def test_get_entity_metrics_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test entity metrics with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        # The actual service implementation will raise KeyError for missing entry_id
        with pytest.raises(KeyError):
            await _get_entity_metrics(mock_hass, mock_call)

    async def test_get_entity_metrics_coordinator_error(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test entity metrics with coordinator error."""
        mock_coordinator = Mock()
        # Create a mock that will raise an exception when len() is called on it
        mock_entities = Mock()
        mock_entities.__len__ = Mock(side_effect=Exception("Access error"))
        mock_coordinator.entities.entities = mock_entities

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        with pytest.raises(
            HomeAssistantError,
            match="Failed to get entity metrics for test_entry_id: Access error",
        ):
            await _get_entity_metrics(mock_hass, mock_service_call)


class TestGetProblematicEntities:
    """Test _get_problematic_entities service function."""

    async def test_get_problematic_entities_success(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test successful problematic entities retrieval."""
        mock_coordinator = Mock()

        # Mock entities with issues
        mock_entity1 = Mock()
        mock_entity1.available = False  # Unavailable entity
        mock_entity1.last_updated = None

        mock_entity2 = Mock()
        mock_entity2.available = True
        # Mock stale update (more than 1 hour ago)
        mock_entity2.last_updated = dt_util.utcnow() - timedelta(hours=2)

        mock_coordinator.entities.entities = {
            "binary_sensor.motion1": mock_entity1,
            "light.test_light": mock_entity2,
        }

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _get_problematic_entities(mock_hass, mock_service_call)

        assert "problems" in result
        problems = result["problems"]
        assert "unavailable" in problems
        assert "stale_updates" in problems
        assert "binary_sensor.motion1" in problems["unavailable"]
        assert "light.test_light" in problems["stale_updates"]

    async def test_get_problematic_entities_no_issues(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test problematic entities with no issues."""
        mock_coordinator = Mock()

        # Mock entities with no issues
        mock_entity = Mock()
        mock_entity.available = True
        # Recent update
        mock_entity.last_updated = dt_util.utcnow() - timedelta(minutes=30)

        mock_coordinator.entities.entities = {
            "binary_sensor.motion1": mock_entity,
        }

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _get_problematic_entities(mock_hass, mock_service_call)

        assert "problems" in result
        problems = result["problems"]
        assert len(problems["unavailable"]) == 0
        assert len(problems["stale_updates"]) == 0

    async def test_get_problematic_entities_coordinator_error(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test problematic entities with coordinator error."""
        mock_coordinator = Mock()
        # Create a mock that will raise an exception when .items() is called on it
        mock_entities = Mock()
        mock_entities.items = Mock(side_effect=Exception("Access error"))
        mock_coordinator.entities.entities = mock_entities

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        with pytest.raises(
            HomeAssistantError,
            match="Failed to get problematic entities for test_entry_id: Access error",
        ):
            await _get_problematic_entities(mock_hass, mock_service_call)


class TestGetEntityDetails:
    """Test _get_entity_details service function."""

    async def test_get_entity_details_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call_with_entity: Mock,
    ) -> None:
        """Test successful entity details retrieval."""
        mock_coordinator = Mock()
        mock_entities = Mock()

        # Mock entity with details
        mock_entity = Mock()
        mock_entity.state = "on"
        mock_entity.evidence = True
        mock_entity.available = True
        mock_entity.last_updated.isoformat.return_value = "2024-01-01T00:00:00"
        mock_entity.probability = 0.75
        mock_entity.decay.decay_factor = 1.0
        mock_entity.decay.is_decaying = False
        mock_entity.decay.decay_start_time = None
        mock_entity.type.input_type.value = "motion"
        mock_entity.type.weight = 0.8
        mock_entity.type.prob_true = 0.8
        mock_entity.type.prob_false = 0.1
        mock_entity.type.prior = 0.35
        mock_entity.type.active_states = ["on"]
        mock_entity.type.active_range = None
        mock_entity.prior.prior = 0.35
        mock_entity.prior.prob_given_true = 0.8
        mock_entity.prior.prob_given_false = 0.1
        mock_entity.prior.last_updated.isoformat.return_value = "2024-01-01T00:00:00"

        mock_entities.get_entity.return_value = mock_entity
        mock_entities.entities = {"binary_sensor.motion1": mock_entity}
        mock_coordinator.entities = mock_entities

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call_with_entity.data = {
            "entry_id": "test_entry_id",
            "entity_ids": ["binary_sensor.motion1"],
        }

        result = await _get_entity_details(mock_hass, mock_service_call_with_entity)

        assert "entity_details" in result
        assert "binary_sensor.motion1" in result["entity_details"]
        entity_detail = result["entity_details"]["binary_sensor.motion1"]
        assert entity_detail["state"] == "on"
        assert entity_detail["evidence"] is True
        assert entity_detail["available"] is True
        assert entity_detail["probability"] == 0.75

    async def test_get_entity_details_missing_entity_id(self, mock_hass: Mock) -> None:
        """Test entity details with missing entity_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}

        # The actual service implementation doesn't require entity_id, it returns all entities if none specified
        # So this test should actually work and return empty details
        mock_config_entry = Mock()
        mock_config_entry.entry_id = "test_entry_id"  # Set the correct entry_id
        mock_coordinator = Mock()
        mock_entities = Mock()
        mock_entities.entities = {}
        mock_coordinator.entities = mock_entities
        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]

        result = await _get_entity_details(mock_hass, mock_call)
        assert "entity_details" in result
        assert result["entity_details"] == {}

    async def test_get_entity_details_entity_not_found(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call_with_entity: Mock,
    ) -> None:
        """Test entity details with entity not found."""
        mock_coordinator = Mock()
        mock_entities = Mock()
        mock_entities.get_entity.side_effect = ValueError("Entity not found")
        mock_entities.entities = [
            "binary_sensor.motion1"
        ]  # Has entity in list but get_entity fails
        mock_coordinator.entities = mock_entities

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call_with_entity.data = {
            "entry_id": "test_entry_id",
            "entity_ids": ["binary_sensor.motion1"],
        }

        result = await _get_entity_details(mock_hass, mock_service_call_with_entity)

        # The service handles ValueError and returns {"error": "Entity not found"}
        assert "entity_details" in result
        assert (
            result["entity_details"]["binary_sensor.motion1"]["error"]
            == "Entity not found"
        )


class TestForceEntityUpdate:
    """Test _force_entity_update service function."""

    async def test_force_entity_update_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call_with_entity: Mock,
    ) -> None:
        """Test successful entity update."""
        mock_coordinator = Mock()
        mock_entities = Mock()

        # Mock entity
        mock_entity = Mock()
        mock_entity.async_update = AsyncMock()

        mock_entities.get_entity.return_value = mock_entity
        mock_entities.entities = {"binary_sensor.motion1": mock_entity}
        mock_coordinator.entities = mock_entities
        mock_coordinator.async_refresh = AsyncMock()

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call_with_entity.data = {
            "entry_id": "test_entry_id",
            "entity_ids": ["binary_sensor.motion1"],
        }

        result = await _force_entity_update(mock_hass, mock_service_call_with_entity)

        mock_entity.async_update.assert_called_once()
        mock_coordinator.async_refresh.assert_called_once()
        assert result["updated_entities"] == 1

    async def test_force_entity_update_all_entities(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test force update for all entities."""
        mock_coordinator = Mock()
        mock_entities = Mock()

        # Mock entities
        mock_entity1 = Mock()
        mock_entity1.async_update = AsyncMock()
        mock_entity2 = Mock()
        mock_entity2.async_update = AsyncMock()

        mock_entities.get_entity.side_effect = [mock_entity1, mock_entity2]
        mock_entities.entities = {
            "binary_sensor.motion1": mock_entity1,
            "light.test_light": mock_entity2,
        }
        mock_coordinator.entities = mock_entities
        mock_coordinator.async_refresh = AsyncMock()

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _force_entity_update(mock_hass, mock_service_call)

        mock_entity1.async_update.assert_called_once()
        mock_entity2.async_update.assert_called_once()
        mock_coordinator.async_refresh.assert_called_once()
        assert result["updated_entities"] == 2

    async def test_force_entity_update_entity_not_found(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call_with_entity: Mock,
    ) -> None:
        """Test force update with entity not found."""
        mock_coordinator = Mock()
        mock_entities = Mock()
        mock_entities.get_entity.side_effect = ValueError("Entity not found")
        mock_entities.entities = ["binary_sensor.motion1"]
        mock_coordinator.entities = mock_entities
        mock_coordinator.async_refresh = AsyncMock()

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call_with_entity.data = {
            "entry_id": "test_entry_id",
            "entity_ids": ["binary_sensor.motion1"],
        }

        result = await _force_entity_update(mock_hass, mock_service_call_with_entity)

        # Service handles ValueError gracefully and logs warning
        mock_coordinator.async_refresh.assert_called_once()
        assert result["updated_entities"] == 0


class TestGetAreaStatus:
    """Test _get_area_status service function."""

    async def test_get_area_status_success(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test successful area status retrieval."""
        mock_coordinator = Mock()
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities.entities = {}

        # Mock probability and is_occupied properties
        mock_coordinator.probability = 0.9  # High confidence (> 0.8)
        mock_coordinator.occupied = True

        # Mock last_updated with a Mock object
        mock_last_updated = Mock()
        mock_last_updated.isoformat.return_value = "2024-01-01T00:00:00"
        mock_coordinator.last_updated = mock_last_updated

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _get_area_status(mock_hass, mock_service_call)

        assert "area_status" in result
        status = result["area_status"]
        assert status["area_name"] == "Test Area"
        assert status["occupied"] is True
        assert status["occupancy_probability"] == 0.9
        assert status["confidence_level"] == "high"

    async def test_get_area_status_no_occupancy_state(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test area status with no occupancy state."""
        mock_coordinator = Mock()
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities.entities = {}

        # Mock properties for no occupancy state
        mock_coordinator.probability = None  # No probability available
        mock_coordinator.occupied = False

        # Mock last_updated with a Mock object
        mock_last_updated = Mock()
        mock_last_updated.isoformat.return_value = "2024-01-01T00:00:00"
        mock_coordinator.last_updated = mock_last_updated

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _get_area_status(mock_hass, mock_service_call)

        assert "area_status" in result
        status = result["area_status"]
        assert status["area_name"] == "Test Area"
        assert status["occupied"] is False
        assert status["occupancy_probability"] is None
        assert status["confidence_level"] == "unknown"


class TestGetEntityTypeLearned:
    """Test _get_entity_type_learned_data service function."""

    async def test_get_entity_type_learned_data_success(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test successful entity type learned data retrieval."""
        mock_coordinator = Mock()

        # Mock entity types with proper structure
        from custom_components.area_occupancy.data.entity_type import InputType

        mock_motion_type = Mock()
        mock_motion_type.prior = 0.3
        mock_motion_type.prob_true = 0.8
        mock_motion_type.prob_false = 0.2
        mock_motion_type.weight = 1.0
        mock_motion_type.active_states = ["on"]
        mock_motion_type.active_range = None

        mock_coordinator.entity_types.entity_types = {
            InputType.MOTION: mock_motion_type
        }

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _get_entity_type_learned_data(mock_hass, mock_service_call)

        assert "entity_types" in result
        # The service converts InputType.MOTION.value to string key
        assert InputType.MOTION.value in result["entity_types"]
        motion_data = result["entity_types"][InputType.MOTION.value]
        assert motion_data["prior"] == 0.3
        assert motion_data["prob_true"] == 0.8
        assert motion_data["prob_false"] == 0.2
        assert motion_data["weight"] == 1.0
        assert motion_data["active_states"] == ["on"]
        assert motion_data["active_range"] is None

    async def test_get_entity_type_learned_data_coordinator_error(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test entity type learned data with coordinator error."""
        mock_coordinator = Mock()
        # Create a mock that will raise an exception when .items() is called on it
        mock_entity_types = Mock()
        mock_entity_types.items = Mock(side_effect=Exception("Access error"))
        mock_coordinator.entity_types.entity_types = mock_entity_types

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        with pytest.raises(
            HomeAssistantError,
            match="Failed to get entity_type learned data for test_entry_id: Access error",
        ):
            await _get_entity_type_learned_data(mock_hass, mock_service_call)


class TestAsyncSetupServices:
    """Test async_setup_services function."""

    async def test_async_setup_services_success(self, mock_hass: Mock) -> None:
        """Test successful service setup."""
        await async_setup_services(mock_hass)

        # Verify services were registered (8 services total)
        assert mock_hass.services.async_register.call_count == 8

    async def test_async_setup_services_registration_error(
        self, mock_hass: Mock
    ) -> None:
        """Test service setup with registration error."""
        mock_hass.services.async_register.side_effect = Exception("Registration failed")

        with pytest.raises(Exception, match="Registration failed"):
            await async_setup_services(mock_hass)


class TestServiceIntegration:
    """Test service integration scenarios."""

    async def test_service_workflow_integration(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test complete service workflow integration."""
        mock_coordinator = Mock()
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities.entities = {}

        # Mock coordinator properties for high confidence state
        mock_coordinator.probability = 0.9  # High confidence (> 0.8)
        mock_coordinator.occupied = True

        # Mock last_updated with a Mock object
        mock_last_updated = Mock()
        mock_last_updated.isoformat.return_value = "2024-01-01T00:00:00"
        mock_coordinator.last_updated = mock_last_updated

        # Mock entity types
        from custom_components.area_occupancy.data.entity_type import InputType

        mock_motion_type = Mock()
        mock_motion_type.prior = 0.3
        mock_motion_type.prob_true = 0.8
        mock_motion_type.prob_false = 0.2
        mock_motion_type.weight = 1.0
        mock_motion_type.active_states = ["on"]
        mock_motion_type.active_range = None

        mock_coordinator.entity_types.entity_types = {
            InputType.MOTION: mock_motion_type
        }

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]

        # 1. Get area status
        mock_service_call.data = {"entry_id": "test_entry_id"}
        status_result = await _get_area_status(mock_hass, mock_service_call)

        assert "area_status" in status_result
        status = status_result["area_status"]
        assert status["area_name"] == "Test Area"
        assert status["occupied"] is True
        assert status["occupancy_probability"] == 0.9
        assert status["confidence_level"] == "high"

        # 2. Get entity metrics
        metrics_result = await _get_entity_metrics(mock_hass, mock_service_call)
        assert "metrics" in metrics_result
        metrics = metrics_result["metrics"]
        assert metrics["total_entities"] == 0
        assert metrics["active_entities"] == 0
        assert metrics["available_entities"] == 0
        assert metrics["unavailable_entities"] == 0
        assert metrics["decaying_entities"] == 0

        # 3. Get entity type learned data
        learned_result = await _get_entity_type_learned_data(
            mock_hass, mock_service_call
        )
        assert "entity_types" in learned_result
        entity_types = learned_result["entity_types"]
        assert "motion" in entity_types
        motion_type = entity_types["motion"]
        assert motion_type["prior"] == 0.3
        assert motion_type["prob_true"] == 0.8
        assert motion_type["prob_false"] == 0.2
        assert motion_type["weight"] == 1.0
        assert motion_type["active_states"] == ["on"]
        assert motion_type["active_range"] is None

    async def test_error_handling_across_services(
        self, mock_hass: Mock, mock_service_call: Mock
    ) -> None:
        """Test error handling across different services."""
        # Test with invalid entry_id
        mock_hass.config_entries.async_entries.return_value = []
        mock_service_call.data = {"entry_id": "nonexistent"}

        with pytest.raises(HomeAssistantError):
            await _get_area_status(mock_hass, mock_service_call)

        # Test with missing required parameters
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        with pytest.raises(KeyError):
            await _get_entity_details(mock_hass, mock_call)

    async def test_service_parameter_validation(self, mock_hass: Mock) -> None:
        """Test parameter validation across services."""
        # Test missing required parameters
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        with pytest.raises(KeyError):
            await _get_entity_details(mock_hass, mock_call)

    async def test_service_return_value_consistency(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test return value consistency across services."""
        mock_coordinator = Mock()
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities.entities = {}

        # Mock coordinator properties for medium confidence state
        mock_coordinator.probability = 0.8  # Medium confidence (0.2 < 0.8 <= 0.8)
        mock_coordinator.occupied = True

        # Mock last_updated with a Mock object
        mock_last_updated = Mock()
        mock_last_updated.isoformat.return_value = "2024-01-01T00:00:00"
        mock_coordinator.last_updated = mock_last_updated

        # Mock entity types
        from custom_components.area_occupancy.data.entity_type import InputType

        mock_motion_type = Mock()
        mock_motion_type.prior = 0.3
        mock_motion_type.prob_true = 0.8
        mock_motion_type.prob_false = 0.2
        mock_motion_type.weight = 1.0
        mock_motion_type.active_states = ["on"]
        mock_motion_type.active_range = None

        mock_coordinator.entity_types.entity_types = {
            InputType.MOTION: mock_motion_type
        }

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        # Test return value structures
        status_result = await _get_area_status(mock_hass, mock_service_call)
        assert "area_status" in status_result
        status = status_result["area_status"]
        assert status["area_name"] == "Test Area"
        assert status["occupied"] is True
        assert status["occupancy_probability"] == 0.8
        assert status["confidence_level"] == "medium"

        # Test metrics consistency
        metrics_result = await _get_entity_metrics(mock_hass, mock_service_call)
        assert "metrics" in metrics_result
        metrics = metrics_result["metrics"]
        assert metrics["total_entities"] == 0
        assert metrics["active_entities"] == 0
        assert metrics["available_entities"] == 0
        assert metrics["unavailable_entities"] == 0
        assert metrics["decaying_entities"] == 0

        # Test entity type data consistency
        learned_result = await _get_entity_type_learned_data(
            mock_hass, mock_service_call
        )
        assert "entity_types" in learned_result
        entity_types = learned_result["entity_types"]
        assert "motion" in entity_types
        motion_type = entity_types["motion"]
        assert motion_type["prior"] == 0.3
        assert motion_type["prob_true"] == 0.8
        assert motion_type["prob_false"] == 0.2
        assert motion_type["weight"] == 1.0
        assert motion_type["active_states"] == ["on"]
        assert motion_type["active_range"] is None
