"""Tests for service module."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, PropertyMock

import pytest

from custom_components.area_occupancy.service import (
    _debug_database_state,
    _debug_import_intervals,
    _get_area_status,
    _get_coordinator,
    _get_entity_details,
    _get_entity_metrics,
    _get_entity_type_learned_data,
    _get_problematic_entities,
    _reset_entities,
    _update_area_prior,
    _update_likelihoods,
    async_setup_services,
)
from homeassistant.core import ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


# ruff: noqa: PLC0415
class TestGetCoordinator:
    """Test _get_coordinator helper function."""

    def test_get_coordinator_success(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_coordinator: Mock
    ) -> None:
        """Test successful coordinator retrieval."""
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


class TestUpdateAreaPrior:
    """Test _update_area_prior service function."""

    async def test_update_area_prior_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful area prior update."""
        # Override specific properties needed for this test
        mock_coordinator.prior.update.return_value = 0.35

        # Mock the prior calculation details
        mock_coordinator.prior.sensor_ids = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]

        # Create mock data dictionaries for calculation details
        from datetime import datetime, timedelta

        now = datetime.now()
        start_time = now - timedelta(days=30)

        mock_prior_data_1 = {
            "ratio": 0.32,
            "occupied_seconds": 27648,  # About 32% of 86400 seconds/day * 30 days
            "states_count": 150,
            "intervals": [Mock() for _ in range(75)],  # Mock intervals list
            "start_time": start_time,
            "end_time": now,
        }

        mock_prior_data_2 = {
            "ratio": 0.35,
            "occupied_seconds": 30240,  # About 35% of 86400 seconds/day * 30 days
            "states_count": 120,
            "intervals": [Mock() for _ in range(60)],  # Mock intervals list
            "start_time": start_time,
            "end_time": now,
        }

        mock_coordinator.prior.data = {
            "binary_sensor.motion1": mock_prior_data_1,
            "binary_sensor.motion2": mock_prior_data_2,
        }

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]

        # The service expects the service call to have entry_id
        mock_service_call.data = {"entry_id": "test_entry_id"}

        # Patch get_time_priors_for_entry as AsyncMock with a mock object having .prior_value

        mock_record = Mock()
        mock_record.prior_value = 0.1
        mock_coordinator.sqlite_store.get_time_priors_for_entry = AsyncMock(
            return_value=[mock_record]
        )

        result = await _update_area_prior(mock_hass, mock_service_call)

        # Verify the result structure
        assert "area_prior" in result
        assert "last_updated" in result

        # Verify the values
        assert result["area_prior"] == 0.3
        assert isinstance(result["last_updated"], str)

        # Verify the coordinator was called correctly
        mock_coordinator.prior.update.assert_called_once_with()
        mock_coordinator.async_refresh.assert_called_once()

    async def test_update_area_prior_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test area prior update with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        # The actual service implementation will raise KeyError for missing entry_id
        with pytest.raises(KeyError):
            await _update_area_prior(mock_hass, mock_call)

    async def test_update_area_prior_coordinator_error(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test area prior update with coordinator error."""
        # Override specific properties needed for this test
        mock_coordinator.prior.update.side_effect = RuntimeError("Update failed")

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        # The service catches RuntimeError and wraps it in HomeAssistantError
        with pytest.raises(
            HomeAssistantError,
            match="Failed to update and get time-based priors for test_entry_id: Update failed",
        ):
            await _update_area_prior(mock_hass, mock_service_call)


class TestUpdateLikelihoods:
    """Test _update_likelihoods service function."""

    async def test_update_likelihoods_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful likelihood update."""
        # Override specific properties needed for this test
        mock_coordinator.entities.update_all_entity_likelihoods.return_value = 5

        # Mock entities with proper structure for return data
        mock_entity = Mock()
        mock_entity.likelihood.prob_given_true = 0.8
        mock_entity.likelihood.prob_given_false = 0.1
        mock_entity.likelihood.prob_given_true_raw = 0.75
        mock_entity.likelihood.prob_given_false_raw = 0.05
        mock_entity.likelihood.last_updated.isoformat.return_value = (
            "2024-01-01T00:00:00"
        )
        mock_entity.type.input_type.value = "motion"
        mock_entity.type.weight = 0.85
        # Add filtering statistics
        mock_entity.likelihood.total_on_intervals = 50
        mock_entity.likelihood.valid_intervals = 45
        mock_entity.likelihood.filtered_short_intervals = 3
        mock_entity.likelihood.filtered_long_intervals = 2
        mock_entity.likelihood.max_filtered_duration_seconds = 18 * 3600  # 18 hours

        mock_coordinator.entities.entities = {"binary_sensor.motion1": mock_entity}

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]

        # The service expects the service call to have entry_id
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _update_likelihoods(mock_hass, mock_service_call)

        # Verify the result structure
        assert "likelihoods" in result
        assert "total_entities" in result
        assert "update_timestamp" in result

        # Verify the values
        assert result["total_entities"] == 1
        assert isinstance(result["update_timestamp"], str)

        # Verify the likelihood data
        likelihoods = result["likelihoods"]
        assert "binary_sensor.motion1" in likelihoods
        likelihood_data = likelihoods["binary_sensor.motion1"]
        assert likelihood_data["type"] == "motion"
        assert likelihood_data["weight"] == 0.85
        assert likelihood_data["prob_given_true"] == 0.8
        assert likelihood_data["prob_given_false"] == 0.1
        assert likelihood_data["prob_given_true_raw"] == 0.75
        assert likelihood_data["prob_given_false_raw"] == 0.05

        # Verify the coordinator was called correctly
        mock_coordinator.entities.update_all_entity_likelihoods.assert_called_once_with()
        mock_coordinator.async_refresh.assert_called_once()

    async def test_update_likelihoods_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test likelihood update with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        # The actual service implementation will raise KeyError for missing entry_id
        with pytest.raises(KeyError):
            await _update_likelihoods(mock_hass, mock_call)

    async def test_update_likelihoods_coordinator_error(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test likelihood update with coordinator error."""
        # Override specific properties needed for this test
        mock_coordinator.entities.update_all_entity_likelihoods.side_effect = (
            RuntimeError("Update failed")
        )

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        # The service catches RuntimeError and wraps it in HomeAssistantError
        with pytest.raises(
            HomeAssistantError,
            match="Failed to update likelihoods for test_entry_id: Update failed",
        ):
            await _update_likelihoods(mock_hass, mock_service_call)

        # Verify the coordinator was called with the correct history period and force=True
        mock_coordinator.entities.update_all_entity_likelihoods.assert_called_once_with()


class TestDebugImportIntervals:
    """Test _debug_import_intervals service function."""

    async def test_debug_import_intervals_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful debug import intervals."""
        mock_import_result = {
            "binary_sensor.motion1": 100,
            "binary_sensor.door1": 50,
        }
        # Mock the async methods properly
        mock_coordinator.sqlite_store.import_intervals_from_recorder = AsyncMock()
        mock_coordinator.sqlite_store.import_intervals_from_recorder.return_value = (
            mock_import_result
        )
        mock_coordinator.sqlite_store.get_total_intervals_count = AsyncMock()
        mock_coordinator.sqlite_store.get_total_intervals_count.return_value = 150

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _debug_import_intervals(mock_hass, mock_service_call)

        # The service returns import results
        assert "import_counts" in result
        assert "days_imported" in result
        assert "entity_ids" in result
        assert result["import_counts"] == mock_import_result
        assert result["days_imported"] == 10

        # The method is called with entity_ids and days, so we just verify it was called
        mock_coordinator.sqlite_store.import_intervals_from_recorder.assert_called_once()

    async def test_debug_import_intervals_missing_entry_id(
        self, mock_hass: Mock
    ) -> None:
        """Test debug import intervals with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        with pytest.raises(KeyError):
            await _debug_import_intervals(mock_hass, mock_call)

    async def test_debug_import_intervals_error(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test debug import intervals with error."""
        # Mock the async method properly
        mock_coordinator.sqlite_store.import_intervals_from_recorder = AsyncMock()
        mock_coordinator.sqlite_store.import_intervals_from_recorder.side_effect = (
            RuntimeError("Import failed")
        )

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        with pytest.raises(
            HomeAssistantError,
            match="Debug import failed for test_entry_id: Import failed",
        ):
            await _debug_import_intervals(mock_hass, mock_service_call)


class TestDebugDatabaseState:
    """Test _debug_database_state service function."""

    async def test_debug_database_state_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful debug database state."""
        mock_stats = {
            "total_entities": 5,
            "total_areas": 2,
            "total_intervals": 1000,
            "total_time_priors": 500,
            "db_size_mb": 1.5,
        }
        # Mock the async methods properly
        mock_coordinator.sqlite_store.is_state_intervals_empty = AsyncMock()
        mock_coordinator.sqlite_store.is_state_intervals_empty.return_value = False
        mock_coordinator.sqlite_store.get_total_intervals_count = AsyncMock()
        mock_coordinator.sqlite_store.get_total_intervals_count.return_value = 1000
        mock_coordinator.sqlite_store.get_historical_intervals = AsyncMock()
        mock_coordinator.sqlite_store.get_historical_intervals.return_value = []
        mock_coordinator.sqlite_store.async_get_stats = AsyncMock()
        mock_coordinator.sqlite_store.async_get_stats.return_value = mock_stats

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _debug_database_state(mock_hass, mock_service_call)

        # The service returns database state information
        assert "configuration" in result
        assert "database_state" in result
        assert "state_intervals_empty" in result["database_state"]
        assert "sample_entities" in result["database_state"]
        assert "database_stats" in result["database_state"]
        assert "schema_info" in result["database_state"]

        mock_coordinator.sqlite_store.async_get_stats.assert_called_once()

    async def test_debug_database_state_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test debug database state with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        with pytest.raises(KeyError):
            await _debug_database_state(mock_hass, mock_call)

    async def test_debug_database_state_error(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test debug database state with error."""
        # Mock the async methods properly
        mock_coordinator.sqlite_store.is_state_intervals_empty = AsyncMock()
        mock_coordinator.sqlite_store.is_state_intervals_empty.return_value = False
        mock_coordinator.sqlite_store.get_total_intervals_count = AsyncMock()
        mock_coordinator.sqlite_store.get_total_intervals_count.return_value = 1000
        mock_coordinator.sqlite_store.get_historical_intervals = AsyncMock()
        mock_coordinator.sqlite_store.get_historical_intervals.return_value = []
        mock_coordinator.sqlite_store.async_get_stats = AsyncMock()
        mock_coordinator.sqlite_store.async_get_stats.side_effect = RuntimeError(
            "Stats failed"
        )

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        with pytest.raises(
            HomeAssistantError,
            match="Debug database state failed for test_entry_id: Stats failed",
        ):
            await _debug_database_state(mock_hass, mock_service_call)


class TestResetEntities:
    """Test _reset_entities service function."""

    async def test_reset_entities_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful entity reset."""
        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        await _reset_entities(mock_hass, mock_service_call)

        mock_coordinator.entities.cleanup.assert_called_once()
        mock_coordinator.async_refresh.assert_called_once()

    async def test_reset_entities_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test entity reset with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        # The actual service implementation will raise KeyError for missing entry_id
        with pytest.raises(KeyError):
            await _reset_entities(mock_hass, mock_call)


class TestGetEntityMetrics:
    """Test _get_entity_metrics service function."""

    async def test_get_entity_metrics_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
        mock_active_entity: Mock,
        mock_inactive_entity: Mock,
    ) -> None:
        """Test successful entity metrics retrieval."""
        # Use centralized entity fixtures
        mock_coordinator.entities.entities = {
            "binary_sensor.motion1": mock_active_entity,
            "binary_sensor.appliance": mock_inactive_entity,
        }

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await _get_entity_metrics(mock_hass, mock_service_call)

        # The service returns metrics summary, not individual entity data
        assert "metrics" in result
        metrics = result["metrics"]
        assert metrics["total_entities"] == 2
        assert metrics["active_entities"] == 1  # mock_active_entity has evidence=True
        assert metrics["available_entities"] == 2  # both entities are available
        assert metrics["unavailable_entities"] == 0
        assert (
            metrics["decaying_entities"] == 1
        )  # mock_inactive_entity has decay.is_decaying=True

    async def test_get_entity_metrics_missing_entry_id(self, mock_hass: Mock) -> None:
        """Test entity metrics with missing entry_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {}

        # The actual service implementation will raise KeyError for missing entry_id
        with pytest.raises(KeyError):
            await _get_entity_metrics(mock_hass, mock_call)

    async def test_get_entity_metrics_coordinator_error(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test entity metrics with coordinator error."""
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
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
        mock_unavailable_entity: Mock,
        mock_stale_entity: Mock,
    ) -> None:
        """Test successful problematic entities retrieval."""
        # Use centralized entity fixtures that already have problematic states
        mock_coordinator.entities.entities = {
            "binary_sensor.motion1": mock_unavailable_entity,  # available=False
            "binary_sensor.appliance": mock_stale_entity,  # last_updated > 1 hour ago
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
        assert "binary_sensor.appliance" in problems["stale_updates"]

    async def test_get_problematic_entities_no_issues(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
        mock_active_entity: Mock,
    ) -> None:
        """Test problematic entities with no issues."""
        # Use active entity that has no issues - need to update its last_updated to recent
        mock_active_entity.last_updated = dt_util.utcnow() - timedelta(minutes=30)

        mock_coordinator.entities.entities = {
            "binary_sensor.motion1": mock_active_entity
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
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test problematic entities with coordinator error."""
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
        mock_coordinator: Mock,
        mock_comprehensive_entity: Mock,
        mock_empty_entity_manager: Mock,
    ) -> None:
        """Test successful entity details retrieval."""
        # Set up the comprehensive entity with proper attribute values
        mock_comprehensive_entity.state = "on"
        mock_comprehensive_entity.evidence = True
        mock_comprehensive_entity.available = True
        mock_comprehensive_entity.probability = 0.5
        # Set up last_updated as a Mock that has an isoformat method
        mock_last_updated = Mock()
        mock_last_updated.isoformat.return_value = "2024-01-01T00:00:00"
        mock_comprehensive_entity.last_updated = mock_last_updated
        mock_comprehensive_entity.decay.decay_factor = 0.8
        mock_comprehensive_entity.decay.is_decaying = False

        # Set up type with proper mock structure
        mock_type = Mock()
        mock_input_type = Mock()
        mock_input_type.value = "motion"
        mock_type.input_type = mock_input_type
        mock_type.weight = 1.0
        mock_type.prob_true = 0.8
        mock_type.prob_false = 0.2
        mock_type.prior = 0.3
        mock_type.active_states = ["on"]
        mock_type.active_range = None
        mock_comprehensive_entity.type = mock_type

        # Set up likelihood
        mock_likelihood = Mock()
        mock_likelihood.prob_given_true = 0.8
        mock_likelihood.prob_given_false = 0.1
        mock_comprehensive_entity.likelihood = mock_likelihood

        # Use centralized comprehensive entity fixture which has all the required properties
        def mock_get_entity(entity_id):
            if entity_id == "binary_sensor.motion1":
                return mock_comprehensive_entity
            raise ValueError("Entity not found")

        mock_empty_entity_manager.get_entity.side_effect = mock_get_entity
        mock_empty_entity_manager.entities = {
            "binary_sensor.motion1": mock_comprehensive_entity
        }
        mock_coordinator.entities = mock_empty_entity_manager

        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call_with_entity.data = {
            "entry_id": "test_entry_id",
            "entity_ids": ["binary_sensor.motion1"],
        }

        result = await _get_entity_details(mock_hass, mock_service_call_with_entity)

        assert "entity_details" in result
        assert (
            "binary_sensor.motion1" in result["entity_details"]
        )  # Use the entity_id from service call
        entity_detail = result["entity_details"]["binary_sensor.motion1"]
        assert entity_detail["state"] == "on"
        assert entity_detail["evidence"] is True
        assert entity_detail["available"] is True
        assert entity_detail["probability"] == 0.5

    async def test_get_entity_details_missing_entity_id(
        self, mock_hass: Mock, mock_coordinator: Mock
    ) -> None:
        """Test entity details with missing entity_id."""
        mock_call = Mock(spec=ServiceCall)
        mock_call.data = {"entry_id": "test_entry_id"}

        # The actual service implementation doesn't require entity_id, it returns all entities if none specified
        # So this test should actually work and return empty details
        mock_config_entry = Mock()
        mock_config_entry.entry_id = "test_entry_id"  # Set the correct entry_id
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
        mock_coordinator: Mock,
    ) -> None:
        """Test entity details with entity not found."""
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


class TestGetAreaStatus:
    """Test _get_area_status service function."""

    async def test_get_area_status_success(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
        mock_last_updated: Mock,
    ) -> None:
        """Test successful area status retrieval."""
        # Override specific properties needed for this test
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities.entities = {}
        mock_coordinator.probability = 0.9  # High confidence (> 0.8)
        mock_coordinator.occupied = True

        # Use centralized mock_last_updated fixture
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
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
        mock_last_updated: Mock,
    ) -> None:
        """Test area status with no occupancy state."""
        # Override specific properties needed for this test
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities.entities = {}
        mock_coordinator.probability = None  # No probability available
        mock_coordinator.occupied = False

        # Use centralized mock_last_updated fixture
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
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
        mock_motion_entity_type: Mock,
    ) -> None:
        """Test successful entity type learned data retrieval."""
        # Mock entity types with proper structure
        from custom_components.area_occupancy.data.entity_type import InputType

        # Override the entity_types property using centralized fixture
        type(mock_coordinator.entity_types).entity_types = PropertyMock(
            return_value={InputType.MOTION: mock_motion_entity_type}
        )

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
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test entity type learned data with coordinator error."""
        # Create a mock that will raise an exception when .items() is called on it
        mock_entity_types = Mock()
        mock_entity_types.items = Mock(side_effect=Exception("Access error"))

        # Override the entity_types property
        type(mock_coordinator.entity_types).entity_types = PropertyMock(
            return_value=mock_entity_types
        )

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
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
        mock_last_updated: Mock,
        mock_motion_entity_type: Mock,
    ) -> None:
        """Test complete service workflow integration."""
        # Override specific properties needed for this test
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities.entities = {}
        mock_coordinator.probability = 0.9  # High confidence (> 0.8)
        mock_coordinator.occupied = True

        # Use centralized mock_last_updated fixture
        mock_coordinator.last_updated = mock_last_updated

        # Mock entity types using centralized fixture
        from custom_components.area_occupancy.data.entity_type import InputType

        # Override the entity_types property using centralized fixture
        type(mock_coordinator.entity_types).entity_types = PropertyMock(
            return_value={InputType.MOTION: mock_motion_entity_type}
        )

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
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        mock_service_call: Mock,
        mock_coordinator: Mock,
        mock_last_updated: Mock,
        mock_motion_entity_type: Mock,
    ) -> None:
        """Test return value consistency across services."""
        # Override specific properties needed for this test
        mock_coordinator.config.name = "Test Area"
        mock_coordinator.entities.entities = {}
        mock_coordinator.probability = 0.8  # Medium confidence (0.2 < 0.8 <= 0.8)
        mock_coordinator.occupied = True

        # Use centralized mock_last_updated fixture
        mock_coordinator.last_updated = mock_last_updated

        # Mock entity types using centralized fixture
        from custom_components.area_occupancy.data.entity_type import InputType

        # Override the entity_types property using centralized fixture
        type(mock_coordinator.entity_types).entity_types = PropertyMock(
            return_value={InputType.MOTION: mock_motion_entity_type}
        )

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
        assert status["confidence_level"] == "medium-high"

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
