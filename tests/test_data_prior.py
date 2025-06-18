"""Tests for data.prior module."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.data.prior import (
    Prior,
    PriorManager,
    TimeInterval,
)
from custom_components.area_occupancy.const import DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestTimeInterval:
    """Test TimeInterval TypedDict."""

    def test_time_interval_structure(self) -> None:
        """Test TimeInterval has correct structure."""
        interval: TimeInterval = {
            "start": dt_util.utcnow(),
            "end": dt_util.utcnow() + timedelta(hours=1),
            "state": STATE_ON,
        }

        assert "start" in interval
        assert "end" in interval
        assert "state" in interval
        assert isinstance(interval["start"], datetime)
        assert isinstance(interval["end"], datetime)
        assert isinstance(interval["state"], str)


class TestPrior:
    """Test Prior dataclass."""

    def test_initialization(self) -> None:
        """Test Prior initialization with valid parameters."""
        now = dt_util.utcnow()
        prior = Prior(
            prior=0.3,
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=now,
        )

        assert prior.prior == 0.3
        assert prior.prob_given_true == 0.8
        assert prior.prob_given_false == 0.1
        assert prior.last_updated == now

    def test_initialization_with_validation(self) -> None:
        """Test initialization with invalid values that should be validated."""
        prior = Prior(
            prior=0.0,  # Invalid for prior, should be clamped
            prob_given_true=1.5,  # Invalid, should be clamped
            prob_given_false=-0.1,  # Invalid, should be clamped
            last_updated=dt_util.utcnow(),
        )

        assert prior.prior == 0.001  # Clamped to minimum for priors (0.001, not 0.0001)
        assert prior.prob_given_true == 1.0  # Clamped to maximum
        assert prior.prob_given_false == 0.001  # Clamped to minimum (0.001, not 0.0)

    def test_initialization_with_none_datetime(self) -> None:
        """Test initialization with None datetime."""
        # Test with a timestamp from the past, not None
        # The validate_datetime function handles None by returning current time
        prior = Prior(
            prior=0.3,
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow() - timedelta(seconds=30),
        )

        assert isinstance(prior.last_updated, datetime)
        # Should be recent (within last minute)
        assert (dt_util.utcnow() - prior.last_updated).total_seconds() < 60

    def test_to_dict(self) -> None:
        """Test converting Prior to dictionary."""
        now = dt_util.utcnow()
        prior = Prior(
            prior=0.3,
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=now,
        )

        result = prior.to_dict()
        expected = {
            "prior": 0.3,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "last_updated": now.isoformat(),
        }

        assert result == expected

    def test_from_dict(self) -> None:
        """Test creating Prior from dictionary."""
        now = dt_util.utcnow()
        data = {
            "prior": 0.3,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "last_updated": now.isoformat(),
        }

        prior = Prior.from_dict(data)

        assert prior.prior == 0.3
        assert prior.prob_given_true == 0.8
        assert prior.prob_given_false == 0.1
        assert prior.last_updated == now

    def test_from_dict_with_invalid_datetime(self) -> None:
        """Test creating Prior from dictionary with invalid datetime."""
        data = {
            "prior": 0.3,
            "prob_given_true": 0.8,
            "prob_given_false": 0.1,
            "last_updated": "invalid_datetime",
        }

        prior = Prior.from_dict(data)

        # Should handle invalid datetime gracefully
        assert isinstance(prior.last_updated, datetime)


class TestPriorManager:
    """Test PriorManager class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test PriorManager initialization."""
        manager = PriorManager(mock_coordinator)

        assert manager.coordinator == mock_coordinator
        assert manager._priors == {}

    def test_priors_property(self, mock_coordinator: Mock) -> None:
        """Test priors property."""
        manager = PriorManager(mock_coordinator)

        # Initially empty
        assert manager.priors == {}

        # Add a prior
        prior = Prior(0.3, 0.8, 0.1, dt_util.utcnow())
        manager._priors["test_entity"] = prior

        assert "test_entity" in manager.priors
        assert manager.priors["test_entity"] == prior

    def test_get_prior(self, mock_coordinator: Mock) -> None:
        """Test getting prior for entity."""
        manager = PriorManager(mock_coordinator)

        # Non-existent entity
        assert manager.get_prior("nonexistent") is None

        # Existing entity
        prior = Prior(0.3, 0.8, 0.1, dt_util.utcnow())
        manager._priors["test_entity"] = prior

        assert manager.get_prior("test_entity") == prior

    def test_update_prior(self, mock_coordinator: Mock) -> None:
        """Test updating prior for entity."""
        manager = PriorManager(mock_coordinator)

        prior = Prior(0.3, 0.8, 0.1, dt_util.utcnow())
        manager.update_prior("test_entity", prior)

        assert manager.get_prior("test_entity") == prior

    def test_remove_prior(self, mock_coordinator: Mock) -> None:
        """Test removing prior for entity."""
        manager = PriorManager(mock_coordinator)

        # Add a prior
        prior = Prior(0.3, 0.8, 0.1, dt_util.utcnow())
        manager._priors["test_entity"] = prior

        # Remove it
        manager.remove_prior("test_entity")

        assert manager.get_prior("test_entity") is None

    def test_clear_priors(self, mock_coordinator: Mock) -> None:
        """Test clearing all priors."""
        manager = PriorManager(mock_coordinator)

        # Add some priors
        prior1 = Prior(0.3, 0.8, 0.1, dt_util.utcnow())
        prior2 = Prior(0.4, 0.7, 0.2, dt_util.utcnow())
        manager._priors["entity1"] = prior1
        manager._priors["entity2"] = prior2

        # Clear all
        manager.clear_priors()

        assert len(manager._priors) == 0

    @patch("custom_components.area_occupancy.data.prior.PriorManager.calculate")
    async def test_update_all_entity_priors(
        self, mock_calculate: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test updating all entity priors."""
        # Setup mock calculate responses
        prior1 = Prior(0.3, 0.8, 0.1, dt_util.utcnow())
        prior2 = Prior(0.4, 0.7, 0.2, dt_util.utcnow())
        mock_calculate.side_effect = [prior1, prior2]

        manager = PriorManager(mock_coordinator)

        # Update all priors
        updated_count = await manager.update_all_entity_priors()

        assert updated_count == 2
        assert mock_calculate.call_count == 2

    @patch("custom_components.area_occupancy.data.prior.PriorManager.calculate")
    async def test_update_all_entity_priors_with_calculation_error(
        self, mock_calculate: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test updating all entity priors with calculation error."""
        from homeassistant.exceptions import HomeAssistantError

        # Setup mock calculate to fail for first entity, succeed for second
        prior2 = Prior(0.4, 0.7, 0.2, dt_util.utcnow())
        mock_calculate.side_effect = [HomeAssistantError("Calculation failed"), prior2]

        manager = PriorManager(mock_coordinator)

        # Update all priors
        updated_count = await manager.update_all_entity_priors()

        assert updated_count == 1  # Only second entity succeeded
        assert mock_calculate.call_count == 2


class TestPriorCalculation:
    """Test prior calculation methods."""

    @patch(
        "custom_components.area_occupancy.data.prior.PriorManager._get_states_from_recorder"
    )
    async def test_calculate_with_valid_data(
        self,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
        mock_entity_for_prior_tests: Mock,
    ) -> None:
        """Test prior calculation with valid historical data."""
        # Setup mock states
        now = dt_util.utcnow()
        mock_states = [
            Mock(state=STATE_ON, last_changed=now - timedelta(hours=2)),
            Mock(state=STATE_OFF, last_changed=now - timedelta(hours=1)),
        ]
        mock_get_states.return_value = mock_states

        manager = PriorManager(mock_coordinator)

        # Test calculation
        result = await manager.calculate(mock_entity_for_prior_tests)

        assert isinstance(result, Prior)
        assert 0 <= result.prior <= 1
        assert 0 <= result.prob_given_true <= 1
        assert 0 <= result.prob_given_false <= 1
        assert isinstance(result.last_updated, datetime)

    @patch(
        "custom_components.area_occupancy.data.prior.PriorManager._get_states_from_recorder"
    )
    async def test_calculate_with_no_data(
        self,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
        mock_entity_for_prior_tests: Mock,
    ) -> None:
        """Test prior calculation with no historical data."""
        # Setup mock to return None (no data)
        mock_get_states.return_value = None

        manager = PriorManager(mock_coordinator)

        # Test calculation
        result = await manager.calculate(mock_entity_for_prior_tests)

        assert isinstance(result, Prior)
        # Should return default values
        assert result.prior == mock_entity_for_prior_tests.type.prior
        assert result.prob_given_true == mock_entity_for_prior_tests.type.prob_true
        assert result.prob_given_false == mock_entity_for_prior_tests.type.prob_false

    @patch(
        "custom_components.area_occupancy.data.prior.PriorManager._get_states_from_recorder"
    )
    async def test_calculate_with_history_disabled(
        self,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
        mock_entity_for_prior_tests: Mock,
    ) -> None:
        """Test prior calculation with history disabled."""
        # Disable history
        mock_coordinator.config.history.enabled = False

        manager = PriorManager(mock_coordinator)

        # Test calculation
        result = await manager.calculate(mock_entity_for_prior_tests)

        assert isinstance(result, Prior)
        # Should return default values without calling recorder
        assert result.prior == mock_entity_for_prior_tests.type.prior
        assert result.prob_given_true == mock_entity_for_prior_tests.type.prob_true
        assert result.prob_given_false == mock_entity_for_prior_tests.type.prob_false

        # Should not have called recorder
        mock_get_states.assert_not_called()

    async def test_states_to_intervals(self) -> None:
        """Test converting states to time intervals."""
        now = dt_util.utcnow()
        start = now - timedelta(hours=2)
        end = now

        # Create mock states
        states = [
            Mock(state=STATE_ON, last_changed=start),
            Mock(state=STATE_OFF, last_changed=start + timedelta(hours=1)),
            Mock(state=STATE_ON, last_changed=start + timedelta(hours=1.5)),
        ]

        # Test conversion (this is an async method)
        intervals = await PriorManager._states_to_intervals(states, start, end)

        assert len(intervals) == 3
        assert intervals[0]["state"] == STATE_ON
        assert intervals[0]["start"] == start
        assert intervals[0]["end"] == start + timedelta(hours=1)

        assert intervals[1]["state"] == STATE_OFF
        assert intervals[1]["start"] == start + timedelta(hours=1)
        assert intervals[1]["end"] == start + timedelta(hours=1.5)

        assert intervals[2]["state"] == STATE_ON
        assert intervals[2]["start"] == start + timedelta(hours=1.5)
        assert intervals[2]["end"] == end

    def test_calculate_conditional_probability(self) -> None:
        """Test conditional probability calculation."""
        now = dt_util.utcnow()

        # Create entity intervals using proper TimeInterval type
        entity_intervals: list[TimeInterval] = [
            TimeInterval(
                start=now - timedelta(hours=2),
                end=now - timedelta(hours=1),
                state=STATE_ON,
            ),
            TimeInterval(
                start=now - timedelta(hours=1),
                end=now,
                state=STATE_OFF,
            ),
        ]

        # Create motion intervals using proper TimeInterval type
        motion_intervals: dict[str, list[TimeInterval]] = {
            "binary_sensor.motion1": [
                TimeInterval(
                    start=now - timedelta(hours=2),
                    end=now - timedelta(hours=1.5),
                    state=STATE_ON,
                ),
                TimeInterval(
                    start=now - timedelta(hours=1.5),
                    end=now,
                    state=STATE_OFF,
                ),
            ]
        }

        # Test calculation
        prob = PriorManager._calculate_conditional_probability_with_intervals(
            entity_id="light.test",
            entity_intervals=entity_intervals,
            motion_intervals_by_sensor=motion_intervals,
            motion_state_filter=STATE_ON,
            entity_active_states=[STATE_ON],
        )

        assert isinstance(prob, float)
        assert 0 <= prob <= 1

    def test_calculate_prior_probability(self) -> None:
        """Test prior probability calculation."""
        now = dt_util.utcnow()

        # Create primary motion intervals using proper TimeInterval type
        primary_intervals: list[TimeInterval] = [
            TimeInterval(
                start=now - timedelta(hours=2),
                end=now - timedelta(hours=1),
                state=STATE_ON,
            ),
            TimeInterval(
                start=now - timedelta(hours=1),
                end=now,
                state=STATE_OFF,
            ),
        ]

        # Test calculation
        prior = PriorManager._calculate_prior_probability(primary_intervals)

        assert isinstance(prior, float)
        assert 0 <= prior <= 1

    def test_calculate_prior_probability_empty_intervals(self) -> None:
        """Test prior probability calculation with empty intervals."""
        prior = PriorManager._calculate_prior_probability([])

        # Should return default prior (0.5, not 0.35)
        assert prior == 0.5


class TestRecorderIntegration:
    """Test recorder integration methods."""

    @patch("custom_components.area_occupancy.data.prior.get_instance")
    @patch("custom_components.area_occupancy.data.prior.get_significant_states")
    async def test_get_states_from_recorder_success(
        self,
        mock_get_states: AsyncMock,
        mock_get_instance: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful state retrieval from recorder."""
        # Setup mocks
        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock()
        mock_get_instance.return_value = mock_recorder

        # Mock states
        mock_state = Mock()
        mock_state.state = STATE_ON
        mock_state.last_changed = dt_util.utcnow()
        mock_get_states.return_value = {"test_entity": [mock_state]}

        # Mock the executor job to return the mocked states
        mock_recorder.async_add_executor_job.return_value = {
            "test_entity": [mock_state]
        }

        manager = PriorManager(mock_coordinator)

        # Test retrieval
        start = dt_util.utcnow() - timedelta(hours=1)
        end = dt_util.utcnow()

        result = await manager._get_states_from_recorder(
            mock_coordinator.hass, "test_entity", start, end
        )

        assert result == [mock_state]

    @patch("custom_components.area_occupancy.data.prior.get_instance")
    async def test_get_states_from_recorder_no_recorder(
        self, mock_get_instance: Mock, mock_coordinator: Mock
    ) -> None:
        """Test state retrieval when recorder is not available."""
        # Setup mock to return None (no recorder)
        mock_get_instance.return_value = None

        manager = PriorManager(mock_coordinator)

        # Test retrieval
        start = dt_util.utcnow() - timedelta(hours=1)
        end = dt_util.utcnow()

        result = await manager._get_states_from_recorder(
            mock_coordinator.hass, "test_entity", start, end
        )

        assert result is None

    @patch("custom_components.area_occupancy.data.prior.get_instance")
    @patch("custom_components.area_occupancy.data.prior.get_significant_states")
    async def test_get_states_from_recorder_exception(
        self,
        mock_get_states: AsyncMock,
        mock_get_instance: Mock,
        mock_coordinator: Mock,
    ) -> None:
        """Test state retrieval with exception."""
        # Setup mocks
        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock()
        mock_get_instance.return_value = mock_recorder

        # Mock to raise a HomeAssistantError (one of the documented exceptions)
        from homeassistant.exceptions import HomeAssistantError

        mock_recorder.async_add_executor_job.side_effect = HomeAssistantError(
            "Database error"
        )

        manager = PriorManager(mock_coordinator)

        # Test retrieval
        start = dt_util.utcnow() - timedelta(hours=1)
        end = dt_util.utcnow()

        # Test that the exception is raised (not swallowed)
        with pytest.raises(HomeAssistantError, match="Database error"):
            await manager._get_states_from_recorder(
                mock_coordinator.hass, "test_entity", start, end
            )


class TestPriorManagerAdditional:
    """Additional tests for PriorManager methods."""

    async def test_calculate_uses_cached_prior(
        self, mock_coordinator: Mock, mock_entity_for_prior_tests: Mock
    ) -> None:
        manager = PriorManager(mock_coordinator)
        cached = Prior(
            prior=0.2,
            prob_given_true=0.3,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
        )
        manager.update_prior(mock_entity_for_prior_tests.entity_id, cached)
        result = await manager.calculate(mock_entity_for_prior_tests)
        assert result is cached

    @patch(
        "custom_components.area_occupancy.data.prior.PriorManager._get_states_from_recorder"
    )
    async def test_calculate_non_primary(
        self, mock_get: AsyncMock, mock_coordinator: Mock, mock_entity_for_prior_tests: Mock
    ) -> None:
        now = dt_util.utcnow()
        primary_states = [
            Mock(state=STATE_ON, last_changed=now - timedelta(minutes=10)),
            Mock(state=STATE_OFF, last_changed=now - timedelta(minutes=5)),
        ]
        entity_states = [
            Mock(state=STATE_ON, last_changed=now - timedelta(minutes=9)),
            Mock(state=STATE_OFF, last_changed=now - timedelta(minutes=4)),
        ]
        mock_get.side_effect = [primary_states, entity_states]
        manager = PriorManager(mock_coordinator)
        result = await manager.calculate(mock_entity_for_prior_tests)
        assert isinstance(result, Prior)

    @patch(
        "custom_components.area_occupancy.data.prior.PriorManager._get_states_from_recorder"
    )
    async def test_calculate_primary_sensor(
        self, mock_get: AsyncMock, mock_coordinator: Mock, mock_entity_for_prior_tests: Mock
    ) -> None:
        mock_entity_for_prior_tests.entity_id = (
            mock_coordinator.config.sensors.primary_occupancy
        )
        now = dt_util.utcnow()
        states = [
            Mock(state=STATE_ON, last_changed=now - timedelta(minutes=2)),
            Mock(state=STATE_OFF, last_changed=now - timedelta(minutes=1)),
        ]
        mock_get.return_value = states
        manager = PriorManager(mock_coordinator)
        result = await manager.calculate(mock_entity_for_prior_tests)
        assert isinstance(result, Prior)


class TestConditionalProbabilityEdgeCases:
    """Tests for _calculate_conditional_probability_with_intervals edge cases."""

    def test_no_filtered_motion_intervals(self) -> None:
        now = dt_util.utcnow()
        motion_intervals = {
            "sensor": [
                TimeInterval(start=now, end=now + timedelta(seconds=10), state=STATE_ON)
            ]
        }

        prob = PriorManager._calculate_conditional_probability_with_intervals(
            entity_id="entity",
            entity_intervals=[],
            motion_intervals_by_sensor=motion_intervals,
            motion_state_filter=STATE_OFF,
            entity_active_states=[STATE_ON],
        )
        assert prob == DEFAULT_PROB_GIVEN_FALSE

    def test_zero_motion_duration(self) -> None:
        now = dt_util.utcnow()
        motion_intervals = {
            "sensor": [TimeInterval(start=now, end=now, state=STATE_ON)]
        }

        prob = PriorManager._calculate_conditional_probability_with_intervals(
            entity_id="entity",
            entity_intervals=[],
            motion_intervals_by_sensor=motion_intervals,
            motion_state_filter=STATE_ON,
            entity_active_states=[STATE_ON],
        )
        assert prob == DEFAULT_PROB_GIVEN_TRUE
