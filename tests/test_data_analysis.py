"""Tests for data analysis module."""

from datetime import UTC, timedelta
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.data.analysis import (
    IntervalData,
    LikelihoodAnalyzer,
    PriorAnalyzer,
    is_timestamp_occupied,
    start_likelihood_analysis,
)
from custom_components.area_occupancy.data.entity import EntityManager
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.db.core import AreaOccupancyDB
from homeassistant.util import dt as dt_util

# ruff: noqa: SLF001


class TestIntervalData:
    """Test IntervalData named tuple."""

    def test_interval_data_creation(self) -> None:
        """Test creating IntervalData."""
        now = dt_util.utcnow()
        data = IntervalData(
            entity_id="binary_sensor.test",
            start_time=now,
            duration_seconds=60.0,
            state="on",
        )
        assert data.entity_id == "binary_sensor.test"
        assert data.start_time == now
        assert data.duration_seconds == 60.0
        assert data.state == "on"


class TestPriorAnalyzerParameterValidation:
    """Test PriorAnalyzer parameter validation."""

    def test_prior_analyzer_init(self, coordinator: Mock) -> None:
        """Test PriorAnalyzer initialization."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)
        assert analyzer.coordinator == coordinator
        assert analyzer.db == coordinator.db
        assert analyzer.area_name == area_name
        assert analyzer.config == coordinator.areas[area_name].config

    def test_prior_analyzer_init_invalid_area(self, coordinator: Mock) -> None:
        """Test PriorAnalyzer initialization with invalid area."""
        with pytest.raises(ValueError, match="Area 'Invalid Area' not found"):
            PriorAnalyzer(coordinator, "Invalid Area")

    def test_analyze_area_prior_no_entities(self, coordinator: Mock) -> None:
        """Test analyze_area_prior with no entities."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)
        result = analyzer.analyze_area_prior([])
        assert result == 0.5  # Default prior

    def test_analyze_time_priors_parameter_validation(self, coordinator: Mock) -> None:
        """Test analyze_time_priors validates slot_minutes parameter."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock dependencies
        analyzer.get_time_bounds = Mock(return_value=(None, None))

        # Test with valid parameter (should proceed to time bounds check)
        analyzer.analyze_time_priors(slot_minutes=60)
        analyzer.get_time_bounds.assert_called_once()

        # Reset mock
        analyzer.get_time_bounds.reset_mock()

        # Test with invalid parameter <= 0 (should use default)
        analyzer.analyze_time_priors(slot_minutes=0)
        # Should still call get_time_bounds but log warning
        analyzer.get_time_bounds.assert_called_once()

        # Reset mock
        analyzer.get_time_bounds.reset_mock()

        # Test with invalid parameter > minutes_per_day (should use default)
        analyzer.analyze_time_priors(slot_minutes=1441)
        analyzer.get_time_bounds.assert_called_once()

        # Reset mock
        analyzer.get_time_bounds.reset_mock()

        # Test with invalid parameter not dividing evenly (should use default)
        analyzer.analyze_time_priors(slot_minutes=77)
        analyzer.get_time_bounds.assert_called_once()


class TestPriorAnalyzerLogic:
    """Test PriorAnalyzer calculation logic with mocks."""

    def test_analyze_area_prior_various_scenarios(self, coordinator: Mock) -> None:
        """Test analyze_area_prior handles various scenarios."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Scenario 1: Normal calculation
        analyzer.get_total_occupied_seconds = Mock(return_value=3600.0)
        analyzer.get_time_bounds = Mock(
            return_value=(
                dt_util.utcnow() - timedelta(days=1),
                dt_util.utcnow(),
            )
        )
        # Should be roughly 3600 / 86400 = 0.0416
        result = analyzer.analyze_area_prior(["binary_sensor.motion"])
        assert 0.04 <= result <= 0.05

        # Scenario 2: Zero total seconds (should return 0.0, but clamped to 0.01)
        analyzer.get_total_occupied_seconds = Mock(return_value=0.0)
        result = analyzer.analyze_area_prior(["binary_sensor.motion"])
        assert result == 0.01  # Clamped to MIN_PROBABILITY

        # Scenario 3: Occupied > Total (should cap at 0.99)
        analyzer.get_total_occupied_seconds = Mock(return_value=100000.0)
        # Total time is 1 day = 86400 seconds
        result = analyzer.analyze_area_prior(["binary_sensor.motion"])
        assert result == 0.99  # Clamped to MAX_PROBABILITY

    def test_analyze_time_priors_various_scenarios(self, coordinator: Mock) -> None:
        """Test analyze_time_priors handles various scenarios."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock dependencies
        analyzer.db = Mock()
        analyzer.db.get_session = Mock()
        analyzer.get_time_bounds = Mock(
            return_value=(
                dt_util.utcnow() - timedelta(days=1),
                dt_util.utcnow(),
            )
        )

        # Mock get_occupied_intervals to return datetime objects
        start = dt_util.utcnow() - timedelta(minutes=30)
        end = dt_util.utcnow()
        # Ensure timezone info for tests
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)

        analyzer.get_occupied_intervals = Mock(return_value=[(start, end)])

        # Test execution
        analyzer.analyze_time_priors()

        # Verify get_occupied_intervals was called
        analyzer.get_occupied_intervals.assert_called()

    def test_analyze_time_priors_with_existing_prior(self, coordinator: Mock) -> None:
        """Test analyze_time_priors updates existing priors."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock dependencies
        analyzer.db = Mock()
        session_mock = Mock()
        analyzer.db.get_session = Mock(return_value=session_mock)
        # Context manager support
        session_mock.__enter__ = Mock(return_value=session_mock)
        session_mock.__exit__ = Mock(return_value=None)

        analyzer.get_time_bounds = Mock(
            return_value=(
                dt_util.utcnow() - timedelta(days=1),
                dt_util.utcnow(),
            )
        )
        analyzer.get_occupied_intervals = Mock(return_value=[])

        # Mock existing prior query
        prior_mock = Mock()
        session_mock.query.return_value.filter_by.return_value.first.return_value = (
            prior_mock
        )

        # Test execution
        analyzer.analyze_time_priors()

        # Should update existing prior
        assert prior_mock.prior_value is not None
        assert prior_mock.data_points is not None
        assert prior_mock.last_updated is not None


class TestLikelihoodAnalyzer:
    """Test LikelihoodAnalyzer."""

    def test_analyze_likelihoods_empty(self, coordinator: Mock) -> None:
        """Test analyze_likelihoods with empty inputs."""
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)
        analyzer.db = Mock()
        session_mock = Mock()
        analyzer.db.get_session = Mock(return_value=session_mock)
        session_mock.__enter__ = Mock(return_value=session_mock)
        session_mock.__exit__ = Mock(return_value=None)

        # Mock empty sensors query
        session_mock.query.return_value.filter_by.return_value.all.return_value = []

        result = analyzer.analyze_likelihoods([], Mock())
        assert result == {}

    def test_is_occupied(self, coordinator: Mock) -> None:
        """Test _is_occupied helper."""
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        now = dt_util.utcnow()
        intervals = [
            (now - timedelta(minutes=30), now - timedelta(minutes=15)),
            (now - timedelta(minutes=10), now),
        ]

        # Test occupied timestamps
        assert analyzer._is_occupied(now - timedelta(minutes=20), intervals)
        assert analyzer._is_occupied(now - timedelta(minutes=5), intervals)

        # Test unoccupied timestamps
        assert not analyzer._is_occupied(now - timedelta(minutes=40), intervals)
        assert not analyzer._is_occupied(now - timedelta(minutes=12), intervals)
        assert not analyzer._is_occupied(now + timedelta(minutes=1), intervals)


class TestIsTimestampOccupied:
    """Test is_timestamp_occupied function."""

    def test_is_timestamp_occupied_empty(self) -> None:
        """Test with empty intervals."""
        assert not is_timestamp_occupied(dt_util.utcnow(), [])

    def test_is_timestamp_occupied_true(self) -> None:
        """Test when timestamp is occupied."""
        now = dt_util.utcnow()
        intervals = [
            (now - timedelta(minutes=30), now - timedelta(minutes=15)),
            (now - timedelta(minutes=10), now),
        ]
        assert is_timestamp_occupied(now - timedelta(minutes=20), intervals)

    def test_is_timestamp_occupied_false(self) -> None:
        """Test when timestamp is not occupied."""
        now = dt_util.utcnow()
        intervals = [
            (now - timedelta(minutes=30), now - timedelta(minutes=15)),
            (now - timedelta(minutes=10), now),
        ]
        assert not is_timestamp_occupied(now - timedelta(minutes=12), intervals)


class TestPriorAnalyzerWithRealDB:
    """Integration tests for PriorAnalyzer with real database."""

    def test_get_total_occupied_seconds_with_real_data(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test get_total_occupied_seconds with real database data."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entity and interval
        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with test_db.get_locked_session() as session:
            entity = test_db.Entities(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with test_db.get_session() as session:
            interval = test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                state="on",
                start_time=start,
                end_time=end,
                duration_seconds=3600.0,
                aggregation_level="raw",
            )
            session.add(interval)
            session.commit()

        # Get total occupied seconds
        total = analyzer.get_total_occupied_seconds()
        assert total >= 0.0

    def test_get_total_occupied_seconds_pure_python(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test get_total_occupied_seconds uses pure Python path."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists
        test_db.save_area_data(area_name)

        # Create test entity and interval
        end = dt_util.utcnow()
        start = end - timedelta(hours=1)

        with test_db.get_locked_session() as session:
            entity = test_db.Entities(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with test_db.get_session() as session:
            interval = test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                state="on",
                start_time=start,
                end_time=end,
                duration_seconds=3600.0,
                aggregation_level="raw",
            )
            session.add(interval)
            session.commit()

        # Verify result
        total = analyzer.get_total_occupied_seconds()
        assert total == 3600.0

    def test_prior_analyzer_init_with_invalid_area(self, coordinator: Mock) -> None:
        """Test PriorAnalyzer raises ValueError for invalid area."""
        with pytest.raises(ValueError, match="Area 'Invalid Area' not found"):
            PriorAnalyzer(coordinator, "Invalid Area")


class TestLikelihoodAnalyzerExtended:
    """Extended tests for LikelihoodAnalyzer."""

    def test_likelihood_analyzer_init_with_invalid_area(
        self, coordinator: Mock
    ) -> None:
        """Test LikelihoodAnalyzer raises ValueError for invalid area."""
        with pytest.raises(ValueError, match="Area 'Invalid Area' not found"):
            LikelihoodAnalyzer(coordinator, "Invalid Area")

    def test_analyze_likelihoods_with_real_data(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test analyze_likelihoods with real database data."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entity and interval
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)
        end_time = start_time + timedelta(hours=2)

        entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(entity)
        session.commit()

        interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start_time.replace(tzinfo=UTC),
            end_time=end_time.replace(tzinfo=UTC),
            duration_seconds=7200.0,
        )
        session.add(interval)
        session.commit()

        # Analyze likelihoods
        occupied_times = [(start_time, end_time)]
        likelihoods = analyzer.analyze_likelihoods(occupied_times, area.entities)
        assert isinstance(likelihoods, dict)

    def test_analyze_likelihoods_skips_correlation_entities(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test that entities with learned correlation ranges are skipped during likelihood update."""

    @pytest.mark.asyncio
    async def test_start_likelihood_analysis_skips_correlation_entities(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test that entities with learned correlation ranges are skipped during likelihood update."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]

        # Setup entity manager with mocked entity
        # Create entity manager manually to ensure clean state

        # Mock factory inside EntityManager
        with patch(
            "custom_components.area_occupancy.data.entity.EntityFactory"
        ) as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_all_from_config.return_value = {}
            mock_factory_class.return_value = mock_factory

            entity_manager = EntityManager(coordinator, area_name=area_name)

            # Add a mock entity with learned correlation
            mock_entity = Mock()
            mock_entity.entity_id = "sensor.correlated_temp"
            mock_entity.learned_active_range = (20.0, 25.0)  # Has learned range
            mock_entity.update_likelihood = Mock()
            entity_manager.add_entity(mock_entity)

            # Add a mock area to coordinator
            mock_area = Mock()
            mock_area.config = Mock()
            # Ensure area has config for motion sensor check
            mock_area.config.sensors.motion_prob_given_true = 0.95
            mock_area.config.sensors.motion_prob_given_false = 0.02
            mock_area.prior.get_occupied_intervals.return_value = []  # Return empty occupied intervals

            # Patch coordinator.get_area to return our mock area
            with (
                patch.object(coordinator, "get_area", return_value=mock_area),
                patch(
                    "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
                ) as MockAnalyzer,
                patch(
                    "custom_components.area_occupancy.data.analysis._update_likelihoods_in_db",
                    return_value=["sensor.correlated_temp"],
                ),
            ):
                # Mock dependencies
                # 1. Mock analyzer.analyze_likelihoods to return new values
                # 2. Mock _update_likelihoods_in_db to simulate DB update success (returning the entity ID)
                analyzer_instance = MockAnalyzer.return_value
                # Return new calculated values that should be ignored
                analyzer_instance.analyze_likelihoods.return_value = {
                    "sensor.correlated_temp": (0.123, 0.456)
                }

                await start_likelihood_analysis(coordinator, area_name, entity_manager)

            # Verify update_likelihood was NOT called because learned_active_range is present
            mock_entity.update_likelihood.assert_not_called()

            # Now test the negative case: clear learned range and ensure it IS called
            mock_entity.learned_active_range = None
            mock_entity.update_likelihood.reset_mock()

            # Re-mock dependencies for the second call
            with (
                patch.object(coordinator, "get_area", return_value=mock_area),
                patch(
                    "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
                ) as MockAnalyzer,
                patch(
                    "custom_components.area_occupancy.data.analysis._update_likelihoods_in_db",
                    return_value=["sensor.correlated_temp"],
                ),
            ):
                analyzer_instance = MockAnalyzer.return_value
                analyzer_instance.analyze_likelihoods.return_value = {
                    "sensor.correlated_temp": (0.123, 0.456)
                }

                # Mock the DB update to return the entity ID so the loop proceeds to update_likelihood
                await start_likelihood_analysis(coordinator, area_name, entity_manager)

            # Verify update_likelihood WAS called
            mock_entity.update_likelihood.assert_called_once_with(0.123, 0.456)

    def test_analyze_likelihoods_with_no_sensors(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test analyze_likelihoods with no sensors."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Analyze likelihoods with empty session (no sensors)
        likelihoods = analyzer.analyze_likelihoods([], area.entities, db_test_session)
        assert likelihoods == {}
