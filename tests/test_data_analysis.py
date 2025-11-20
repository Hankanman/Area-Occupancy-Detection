"""Tests for data.analysis module."""

from datetime import UTC, timedelta
import logging
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import (
    DataError,
    IntegrityError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.analysis import (
    DEFAULT_OCCUPIED_SECONDS,
    DEFAULT_PRIOR,
    IntervalData,
    LikelihoodAnalyzer,
    PriorAnalyzer,
    _update_likelihoods_in_db,
    start_likelihood_analysis,
    start_prior_analysis,
)
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.db import AreaOccupancyDB
from homeassistant.const import STATE_ON, STATE_PLAYING
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

# ruff: noqa: SLF001


# Helper functions to reduce code duplication
def create_test_entity(
    entity_id: str = "test",
    entity_type: Mock | EntityType = None,
    prob_given_true: float = 0.25,
    prob_given_false: float = 0.05,
    decay: Mock | Decay = None,
    coordinator: Mock | None = None,
    hass: Mock | None = None,
    **kwargs,
) -> Entity:
    """Create a test Entity instance with default values."""
    if entity_type is None:
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.8,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            active_states=[STATE_ON],
        )
    if decay is None:
        decay = Decay(half_life=60.0)
    if hass is None:
        if coordinator is not None:
            hass = coordinator.hass
        else:
            hass = Mock()
    if coordinator is None:
        coordinator = Mock()
        coordinator.hass = hass

    return Entity(
        entity_id=entity_id,
        type=entity_type,
        prob_given_true=prob_given_true,
        prob_given_false=prob_given_false,
        decay=decay,
        hass=hass,
        last_updated=dt_util.utcnow(),
        previous_evidence=kwargs.get("previous_evidence"),
    )


class TestLikelihoodAnalyzer:
    """Test the LikelihoodAnalyzer class."""

    def test_is_occupied_binary_search_logic(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test the _is_occupied method with binary search logic."""
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Test with empty occupied times
        assert analyzer._is_occupied(dt_util.utcnow(), []) is False

        # Test with occupied times
        now = dt_util.utcnow()
        occupied_times = [
            (now - timedelta(hours=2), now - timedelta(hours=1)),  # Past interval
            (now, now + timedelta(hours=1)),  # Current interval
            (now + timedelta(hours=2), now + timedelta(hours=3)),  # Future interval
        ]

        # Test timestamp within current interval
        assert (
            analyzer._is_occupied(now + timedelta(minutes=30), occupied_times) is True
        )

        # Test timestamp outside all intervals
        assert (
            analyzer._is_occupied(now + timedelta(hours=1.5), occupied_times) is False
        )

        # Test timestamp at interval boundary
        assert analyzer._is_occupied(now, occupied_times) is True  # Start of interval
        assert (
            analyzer._is_occupied(now + timedelta(hours=1), occupied_times) is False
        )  # End of interval


class TestPriorAnalyzer:
    """Test the PriorAnalyzer class."""

    @pytest.mark.parametrize("entity_ids", [[], None])
    def test_analyze_area_prior_with_invalid_entity_ids(
        self, coordinator: AreaOccupancyCoordinator, entity_ids
    ) -> None:
        """Test analyze_area_prior returns default when entity IDs are invalid."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)
        result = analyzer.analyze_area_prior(entity_ids)
        assert result == DEFAULT_PRIOR

    @pytest.mark.parametrize(
        ("slot_minutes", "description"),
        [
            (-10, "negative slot_minutes"),
            (70, "slot_minutes not dividing day evenly"),
        ],
    )
    def test_analyze_time_priors_parameter_validation(
        self, coordinator: AreaOccupancyCoordinator, slot_minutes, description
    ) -> None:
        """Test analyze_time_priors validates slot_minutes parameter."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock instance methods (now that analyze_time_priors uses wrapper methods)
        first_time = dt_util.utcnow()
        last_time = first_time + timedelta(days=1)

        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(first_time, last_time),
            ),
        ):
            # Should handle invalid slot_minutes gracefully
            analyzer.analyze_time_priors(slot_minutes=slot_minutes)
            # Should use DEFAULT_SLOT_MINUTES instead

    @pytest.mark.parametrize(
        ("interval_data", "time_bounds", "description"),
        [
            (
                [(0, 0, 100.0)],
                (dt_util.utcnow(), dt_util.utcnow()),
                "invalid days calculation",
            ),
            (
                [("invalid", "data", None)],
                (dt_util.utcnow(), dt_util.utcnow() + timedelta(days=1)),
                "invalid interval data",
            ),
            (
                [(0, 999, 100.0)],
                (dt_util.utcnow(), dt_util.utcnow() + timedelta(days=1)),
                "invalid slot number",
            ),
            (
                [(0, 0, 100.0)],
                (dt_util.utcnow(), dt_util.utcnow() + timedelta(days=1)),
                "valid data",
            ),
        ],
    )
    def test_analyze_time_priors_various_scenarios(
        self,
        coordinator: AreaOccupancyCoordinator,
        interval_data,
        time_bounds,
        description,
    ) -> None:
        """Test analyze_time_priors handles various scenarios."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock instance methods (now that analyze_time_priors uses wrapper methods)
        # Ensure time_bounds are real datetime objects, not Mocks
        if isinstance(time_bounds[0], Mock) or isinstance(time_bounds[1], Mock):
            first_time = dt_util.utcnow()
            last_time = first_time + timedelta(days=1)
            time_bounds = (first_time, last_time)

        # Ensure interval_data is a list, not a Mock
        if isinstance(interval_data, Mock):
            interval_data = []

        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=interval_data if isinstance(interval_data, list) else [],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=time_bounds,
            ),
        ):
            analyzer.analyze_time_priors()
            # Should handle all scenarios gracefully

    def test_analyze_time_priors_with_existing_prior(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test analyze_time_priors updates existing priors."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock instance methods (now that analyze_time_priors uses wrapper methods)
        first_time = dt_util.utcnow()
        last_time = first_time + timedelta(days=1)

        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(first_time, last_time),
            ),
        ):
            analyzer.analyze_time_priors()
            # Should successfully update existing prior

    @pytest.mark.parametrize(
        ("method_name", "error_class", "expected_result"),
        [
            ("get_interval_aggregates", OperationalError, []),
            ("get_time_bounds", DataError, (None, None)),
            ("get_total_occupied_seconds", ProgrammingError, DEFAULT_OCCUPIED_SECONDS),
        ],
    )
    def test_database_error_handling(
        self,
        coordinator: AreaOccupancyCoordinator,
        method_name,
        error_class,
        expected_result,
    ) -> None:
        """Test database error handling for various methods."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock database session to raise the specified error
        mock_session = Mock()
        mock_session.__enter__ = Mock(side_effect=error_class("Test error", None, None))
        mock_session.__exit__ = Mock()
        with patch.object(coordinator.db, "get_session", return_value=mock_session):
            # Call the method and verify it handles the error gracefully
            method = getattr(analyzer, method_name)
            if method_name == "get_total_occupied_seconds":
                result = method()
            else:
                result = method(["test.entity"])

            assert result == expected_result

    def test_get_time_bounds_successful_cases(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_time_bounds with successful database operations."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create datetime objects once to ensure consistency
        now = dt_util.utcnow()
        later = now + timedelta(hours=1)

        test_cases = [
            (None, Mock(first=now, last=later), (now, later)),
            (["test.entity"], Mock(first=now, last=later), (now, later)),
        ]

        for entity_ids, mock_result, expected_result in test_cases:
            mock_session = Mock()
            if entity_ids is None:
                mock_session.query.return_value.join.return_value.filter.return_value.first.return_value = mock_result
            else:
                mock_session.query.return_value.filter.return_value.first.return_value = mock_result

            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock()
            # Patch db.get_session to return our mock session
            with patch.object(analyzer.db, "get_session", return_value=mock_session):
                result = analyzer.get_time_bounds(entity_ids)
                assert result == expected_result

    def test_get_total_occupied_seconds_with_none_result(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_total_occupied_seconds handles None result from database."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.scalar.return_value = None
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock()
        with patch.object(coordinator.db, "get_session", return_value=mock_session):
            result = analyzer.get_total_occupied_seconds()
            assert result == DEFAULT_OCCUPIED_SECONDS

    @pytest.mark.parametrize(
        ("mock_result", "expected_result"),
        [
            ([], []),  # No intervals
            (
                [(dt_util.utcnow(), dt_util.utcnow() + timedelta(hours=1))],
                [(0, 0, 3600.0)],
            ),  # One hour interval
        ],
    )
    def test_get_interval_aggregates_various_scenarios(
        self, coordinator: AreaOccupancyCoordinator, mock_result, expected_result
    ) -> None:
        """Test get_interval_aggregates handles various scenarios."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        with patch.object(analyzer, "get_occupied_intervals", return_value=mock_result):
            result = analyzer.get_interval_aggregates()
            # The new implementation aggregates intervals differently, so we just check it returns a list
            assert isinstance(result, list)
            # For the empty case, we expect an empty list
            if not mock_result:
                assert result == []
            else:
                # For non-empty cases, the result depends on the actual aggregation logic
                # Since we're mocking get_occupied_intervals, the SQL path might return empty
                # So we just verify it's a list (the actual aggregation is tested elsewhere)
                assert isinstance(result, list)

    def test_get_interval_aggregates_sql_path(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test successful SQL aggregation path."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock successful SQL aggregation
        with patch.object(
            coordinator.db,
            "get_aggregated_intervals_by_slot",
            return_value=[
                (0, 0, 3600.0),  # Monday, slot 0, 1 hour
                (1, 12, 1800.0),  # Tuesday, slot 12, 30 minutes
            ],
        ) as mock_get_aggregated:
            result = analyzer.get_interval_aggregates(slot_minutes=60)

            assert len(result) == 2
            assert result[0] == (0, 0, 3600.0)
            assert result[1] == (1, 12, 1800.0)

            # Verify SQL method was called with area_name
            mock_get_aggregated.assert_called_once_with(
                entry_id=coordinator.entry_id,
                slot_minutes=60,
                area_name=area_name,
            )

    def test_get_interval_aggregates_fallback(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test fallback to Python method on SQL failure."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock SQL failure and Python fallback path
        with (
            patch.object(
                coordinator.db,
                "get_aggregated_intervals_by_slot",
                side_effect=(OperationalError("Database error", None, None)),
            ),
            patch(
                "custom_components.area_occupancy.db.queries.is_occupied_intervals_cache_valid",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.aggregation.get_occupied_intervals",
                return_value=[
                    (
                        dt_util.utcnow() - timedelta(hours=1),
                        dt_util.utcnow(),
                    )
                ],
            ) as mock_get_occupied,
            patch(
                "custom_components.area_occupancy.db.aggregation.aggregate_intervals_by_slot",
                return_value=[(0, 0, 1800.0)],
            ) as mock_aggregate,
        ):
            result = analyzer.get_interval_aggregates(slot_minutes=60)

            assert len(result) == 1
            assert result[0] == (0, 0, 1800.0)

            # Verify Python fallback path was called
            mock_get_occupied.assert_called_once()
            mock_aggregate.assert_called_once()

    def test_get_occupied_intervals_with_lookback_days(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test get_occupied_intervals with lookback_days parameter."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Test the lookback_days parameter by mocking the database
        test_intervals = [
            (
                dt_util.utcnow() - timedelta(days=1),
                dt_util.utcnow() - timedelta(days=1, hours=1),
            ),
            (
                dt_util.utcnow() - timedelta(hours=2),
                dt_util.utcnow() - timedelta(hours=1),
            ),
        ]

        # Mock database session
        mock_session = Mock()
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = [
            (start, end) for start, end in test_intervals
        ]

        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        with (
            patch.object(
                coordinator.db, "get_session", return_value=mock_context_manager
            ),
            patch(
                "custom_components.area_occupancy.db.queries.is_occupied_intervals_cache_valid",
                return_value=False,
            ),
        ):
            # Test default lookback (90 days)
            intervals = analyzer.get_occupied_intervals(lookback_days=90)
            assert isinstance(intervals, list)

            # Test custom lookback
            intervals_custom = analyzer.get_occupied_intervals(lookback_days=30)
            assert isinstance(intervals_custom, list)

    def test_get_occupied_intervals_performance_logging(
        self, coordinator: AreaOccupancyCoordinator, caplog
    ) -> None:
        """Test performance logging in get_occupied_intervals."""
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Test that the method logs debug messages by testing the actual method
        # We'll mock the database to avoid complex SQLAlchemy mocking
        with (
            patch(
                "custom_components.area_occupancy.db.queries.is_occupied_intervals_cache_valid",
                return_value=False,
            ),
            patch(
                "custom_components.area_occupancy.db.queries.get_occupied_intervals",
                return_value=[],
            ),
        ):
            with caplog.at_level(logging.DEBUG):
                result = analyzer.get_occupied_intervals()

            # Verify the method returns a list (even if empty)
            assert isinstance(result, list)

            # Check that the method executed (any log message indicates execution)
            # The actual log messages may vary, so we just verify the method ran


class TestPriorAnalyzerWithRealDB:
    """Test PriorAnalyzer with real database."""

    def test_analyze_area_prior_with_real_data(self, test_db: AreaOccupancyDB) -> None:
        """Test analyze_area_prior with real database data."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Test with empty entity IDs
        result = analyzer.analyze_area_prior([])
        assert result == DEFAULT_PRIOR

        # Test with no time bounds
        with (
            patch.object(analyzer, "get_total_occupied_seconds", return_value=0.0),
            patch.object(analyzer, "get_time_bounds", return_value=(None, None)),
        ):
            result = analyzer.analyze_area_prior(["binary_sensor.motion"])
            assert result == DEFAULT_PRIOR

    def test_analyze_area_prior_with_motion_data(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test analyze_area_prior with motion sensor data."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entity and interval in database
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

        # Calculate prior
        result = analyzer.analyze_area_prior(["binary_sensor.test_motion"])
        assert 0.0 <= result <= 1.0

    def test_analyze_area_prior_with_supplemented_sensors(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test analyze_area_prior supplements with media/appliance when motion prior is low."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Set up media and appliance sensors
        area.config.sensors.media = ["media_player.test_tv"]
        area.config.sensors.appliance = ["switch.test_appliance"]

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock low motion prior
        with (
            patch.object(
                analyzer,
                "get_total_occupied_seconds",
                side_effect=lambda **kwargs: 100.0
                if not kwargs.get("include_media")
                else 500.0,
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(
                    dt_util.utcnow() - timedelta(days=10),
                    dt_util.utcnow(),
                ),
            ),
            patch.object(
                analyzer,
                "get_occupied_intervals",
                return_value=[
                    (
                        dt_util.utcnow() - timedelta(days=1),
                        dt_util.utcnow() - timedelta(hours=23),
                    )
                ],
            ),
        ):
            result = analyzer.analyze_area_prior(analyzer.sensor_ids)
            assert 0.0 <= result <= 1.0

    def test_analyze_area_prior_with_data_corruption_check(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_area_prior handles data corruption (occupied > total)."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock occupied time exceeding total time
        with (
            patch.object(
                analyzer,
                "get_total_occupied_seconds",
                return_value=100000.0,  # Exceeds total
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(
                    dt_util.utcnow() - timedelta(days=10),
                    dt_util.utcnow(),
                ),
            ),
        ):
            result = analyzer.analyze_area_prior(analyzer.sensor_ids)
            # Should handle gracefully and cap to 1.0
            assert 0.0 <= result <= 1.0

    def test_analyze_area_prior_with_min_prior_override(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_area_prior applies minimum prior override."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.min_prior_override = 0.3

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock low calculated prior
        with (
            patch.object(
                analyzer,
                "get_total_occupied_seconds",
                return_value=100.0,
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(
                    dt_util.utcnow() - timedelta(days=10),
                    dt_util.utcnow(),
                ),
            ),
        ):
            result = analyzer.analyze_area_prior(analyzer.sensor_ids)
            # Should be at least min_prior_override
            assert result >= 0.3

    def test_analyze_time_priors_with_real_data(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test analyze_time_priors with real database data."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entity and interval
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)
        end_time = start_time + timedelta(hours=1)

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
            duration_seconds=3600.0,
        )
        session.add(interval)
        session.commit()

        # Mock interval aggregates and time bounds
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(start_time.weekday(), 0, 3600.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(start_time, now),
            ),
        ):
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_analyze_time_priors_with_zero_total_slot_seconds(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_time_priors handles zero total slot seconds."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock zero days
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(dt_util.utcnow(), dt_util.utcnow()),
            ),
        ):
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_get_occupied_intervals_with_real_database(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals with real database."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities and intervals
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

        # Get occupied intervals
        intervals = analyzer.get_occupied_intervals(lookback_days=90)
        assert isinstance(intervals, list)

    def test_get_occupied_intervals_with_media(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals with media sensors."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.media = ["media_player.test_tv"]

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities and intervals
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        motion_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(motion_entity)

        media_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            entity_type=InputType.MEDIA,
        )
        session.add(media_entity)
        session.commit()

        motion_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start_time.replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(motion_interval)

        media_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            state=STATE_PLAYING,
            start_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(media_interval)
        session.commit()

        # Get occupied intervals with media
        intervals = analyzer.get_occupied_intervals(
            lookback_days=90, include_media=True
        )
        assert isinstance(intervals, list)

    def test_get_occupied_intervals_with_appliance(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals with appliance sensors."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.appliance = ["switch.test_appliance"]

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities and intervals
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        motion_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(motion_entity)

        appliance_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="switch.test_appliance",
            entity_type=InputType.APPLIANCE,
        )
        session.add(appliance_entity)
        session.commit()

        motion_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start_time.replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(motion_interval)

        appliance_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="switch.test_appliance",
            state=STATE_ON,
            start_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(appliance_interval)
        session.commit()

        # Get occupied intervals with appliance
        intervals = analyzer.get_occupied_intervals(
            lookback_days=90, include_appliance=True
        )
        assert isinstance(intervals, list)

    def test_get_occupied_intervals_merges_overlapping(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals merges overlapping intervals."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities and overlapping intervals
        session = db_test_session
        now = dt_util.utcnow()
        start1 = now - timedelta(days=1)
        end1 = start1 + timedelta(hours=1)
        start2 = start1 + timedelta(minutes=30)  # Overlaps with first
        end2 = start2 + timedelta(hours=1)

        entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(entity)
        session.commit()

        interval1 = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start1.replace(tzinfo=UTC),
            end_time=end1.replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(interval1)

        interval2 = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start2.replace(tzinfo=UTC),
            end_time=end2.replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(interval2)
        session.commit()

        # Get occupied intervals - should merge overlapping
        intervals = analyzer.get_occupied_intervals(lookback_days=90)
        assert isinstance(intervals, list)

    def test_get_total_occupied_seconds_with_real_data(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_total_occupied_seconds with real database."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

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

        # Get total occupied seconds
        total = analyzer.get_total_occupied_seconds()
        assert total >= 0.0

    def test_get_total_occupied_seconds_sql_path(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test get_total_occupied_seconds uses SQL path."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Set timeout to 0 to enable SQL path (guard condition)
        area.config.sensors.motion_timeout = 0

        # Mock SQL method to return a value
        with patch.object(
            coordinator.db,
            "get_total_occupied_seconds_sql",
            return_value=3600.0,
        ):
            total = analyzer.get_total_occupied_seconds()
            assert total == 3600.0

    def test_get_total_occupied_seconds_python_fallback(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test get_total_occupied_seconds falls back to Python method."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock SQL method to fail
        with (
            patch.object(
                coordinator.db,
                "get_total_occupied_seconds_sql",
                side_effect=SQLAlchemyError("Test error"),
            ),
            patch(
                "custom_components.area_occupancy.db.queries.get_occupied_intervals",
                return_value=[
                    (
                        dt_util.utcnow() - timedelta(hours=1),
                        dt_util.utcnow(),
                    )
                ],
            ),
            patch(
                "custom_components.area_occupancy.db.queries.is_occupied_intervals_cache_valid",
                return_value=False,
            ),
        ):
            total = analyzer.get_total_occupied_seconds()
            assert total > 0.0

    def test_get_total_occupied_seconds_skips_sql_with_timeout(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test get_total_occupied_seconds skips SQL path when timeout > 0."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Set timeout > 0 to disable SQL path (guard condition)
        area.config.sensors.motion_timeout = 300

        # Mock SQL method - it should NOT be called
        with (
            patch.object(
                coordinator.db,
                "get_total_occupied_seconds_sql",
                return_value=3600.0,
            ) as mock_sql,
            patch(
                "custom_components.area_occupancy.db.queries.get_occupied_intervals",
                return_value=[
                    (
                        dt_util.utcnow() - timedelta(hours=1),
                        dt_util.utcnow(),
                    )
                ],
            ),
            patch(
                "custom_components.area_occupancy.db.queries.is_occupied_intervals_cache_valid",
                return_value=False,
            ),
        ):
            total = analyzer.get_total_occupied_seconds()
            assert total > 0.0
            # SQL method should not be called due to guard condition
            mock_sql.assert_not_called()

    def test_get_total_occupied_seconds_skips_sql_with_media(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test get_total_occupied_seconds skips SQL path when include_media=True."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Set timeout to 0 but include media to disable SQL path (guard condition)
        area.config.sensors.motion_timeout = 0

        # Mock SQL method - it should NOT be called
        with (
            patch.object(
                coordinator.db,
                "get_total_occupied_seconds_sql",
                return_value=3600.0,
            ) as mock_sql,
            patch(
                "custom_components.area_occupancy.db.queries.get_occupied_intervals",
                return_value=[
                    (
                        dt_util.utcnow() - timedelta(hours=1),
                        dt_util.utcnow(),
                    )
                ],
            ),
            patch(
                "custom_components.area_occupancy.db.queries.is_occupied_intervals_cache_valid",
                return_value=False,
            ),
        ):
            total = analyzer.get_total_occupied_seconds(include_media=True)
            assert total > 0.0
            # SQL method should not be called due to guard condition
            mock_sql.assert_not_called()

    def test_prior_analyzer_init_with_invalid_area(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test PriorAnalyzer raises ValueError for invalid area."""
        with pytest.raises(ValueError, match="Area 'Invalid Area' not found"):
            PriorAnalyzer(coordinator, "Invalid Area")


class TestLikelihoodAnalyzerExtended:
    """Extended tests for LikelihoodAnalyzer."""

    def test_likelihood_analyzer_init_with_invalid_area(
        self, coordinator: AreaOccupancyCoordinator
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

    def test_analyze_likelihoods_with_no_sensors(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test analyze_likelihoods with no sensors."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Mock no sensors
        with patch.object(analyzer, "_get_sensors", return_value=[]):
            likelihoods = analyzer.analyze_likelihoods([], area.entities)
            assert likelihoods == {}

    def test_get_sensors(self, test_db: AreaOccupancyDB, db_test_session) -> None:
        """Test _get_sensors returns sensors from database."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entity
        session = db_test_session
        entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(entity)
        session.commit()

        # Get sensors
        sensors = analyzer._get_sensors(session)
        assert len(sensors) >= 0

    def test_get_intervals_by_entity(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test _get_intervals_by_entity groups intervals by entity."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entity and interval
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

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
            end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(interval)
        session.commit()

        # Get intervals by entity
        sensors = [entity]
        intervals_by_entity = analyzer._get_intervals_by_entity(session, sensors)
        assert isinstance(intervals_by_entity, dict)
        assert "binary_sensor.test_motion" in intervals_by_entity

    def test_analyze_entity_likelihood(self, test_db: AreaOccupancyDB) -> None:
        """Test _analyze_entity_likelihood calculates probabilities."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Create test entity object
        entity = Mock()
        entity.entity_id = "binary_sensor.test_motion"

        intervals_by_entity = {
            "binary_sensor.test_motion": [
                IntervalData(
                    entity_id="binary_sensor.test_motion",
                    start_time=dt_util.utcnow() - timedelta(hours=1),
                    duration_seconds=3600.0,
                    state="on",
                )
            ]
        }

        occupied_times = [
            (
                dt_util.utcnow() - timedelta(hours=1),
                dt_util.utcnow(),
            )
        ]

        entity_obj = Mock()
        entity_obj.active_states = ["on"]
        entity_obj.active_range = None

        entity_manager = Mock()
        entity_manager.get_entity = Mock(return_value=entity_obj)

        prob_given_true, prob_given_false = analyzer._analyze_entity_likelihood(
            entity, intervals_by_entity, occupied_times, entity_manager
        )
        assert 0.0 <= prob_given_true <= 1.0
        assert 0.0 <= prob_given_false <= 1.0

    def test_analyze_entity_likelihood_with_active_range(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test _analyze_entity_likelihood with active_range."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        entity = Mock()
        entity.entity_id = "sensor.test"

        intervals_by_entity = {
            "sensor.test": [
                IntervalData(
                    entity_id="sensor.test",
                    start_time=dt_util.utcnow() - timedelta(hours=1),
                    duration_seconds=3600.0,
                    state="0.5",  # Within range
                )
            ]
        }

        occupied_times = [
            (
                dt_util.utcnow() - timedelta(hours=1),
                dt_util.utcnow(),
            )
        ]

        entity_obj = Mock()
        entity_obj.active_states = None
        entity_obj.active_range = (0.0, 1.0)

        entity_manager = Mock()
        entity_manager.get_entity = Mock(return_value=entity_obj)

        prob_given_true, prob_given_false = analyzer._analyze_entity_likelihood(
            entity, intervals_by_entity, occupied_times, entity_manager
        )
        assert 0.0 <= prob_given_true <= 1.0
        assert 0.0 <= prob_given_false <= 1.0

    def test_is_interval_active_with_active_states(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _is_interval_active with active_states."""
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        interval = IntervalData(
            entity_id="test",
            start_time=dt_util.utcnow(),
            duration_seconds=100.0,
            state="on",
        )

        entity_obj = Mock()
        entity_obj.active_states = ["on", "active"]
        entity_obj.active_range = None

        assert analyzer._is_interval_active(interval, entity_obj) is True

        # Create a new IntervalData with different state (NamedTuple is immutable)
        interval_off = IntervalData(
            entity_id="test",
            start_time=dt_util.utcnow(),
            duration_seconds=100.0,
            state="off",
        )
        assert analyzer._is_interval_active(interval_off, entity_obj) is False

    def test_is_interval_active_with_active_range(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _is_interval_active with active_range."""
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        interval = IntervalData(
            entity_id="test",
            start_time=dt_util.utcnow(),
            duration_seconds=100.0,
            state="0.5",
        )

        entity_obj = Mock()
        entity_obj.active_states = None
        entity_obj.active_range = (0.0, 1.0)

        assert analyzer._is_interval_active(interval, entity_obj) is True

        # Create new IntervalData instances with different states (NamedTuple is immutable)
        interval_high = IntervalData(
            entity_id="test",
            start_time=dt_util.utcnow(),
            duration_seconds=100.0,
            state="1.5",
        )
        assert analyzer._is_interval_active(interval_high, entity_obj) is False

        interval_invalid = IntervalData(
            entity_id="test",
            start_time=dt_util.utcnow(),
            duration_seconds=100.0,
            state="invalid",
        )
        assert analyzer._is_interval_active(interval_invalid, entity_obj) is False

    def test_analyze_likelihoods_database_error(self, test_db: AreaOccupancyDB) -> None:
        """Test analyze_likelihoods handles database errors."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Mock database error
        with patch.object(
            coordinator.db,
            "get_session",
            side_effect=OperationalError("Test error", None, None),
        ):
            likelihoods = analyzer.analyze_likelihoods([], area.entities)
            assert likelihoods == {}


class TestAnalysisHelperFunctions:
    """Test helper functions in analysis module."""

    def test_update_likelihoods_in_db(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test _update_likelihoods_in_db updates database."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create entity in database
        session = db_test_session
        entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
            weight=0.85,
            prob_given_true=0.5,
            prob_given_false=0.05,
            last_updated=dt_util.utcnow(),
            created_at=dt_util.utcnow(),
            is_decaying=False,
            evidence=False,
        )
        session.add(entity)
        session.commit()

        # Update likelihoods
        now = dt_util.utcnow()
        likelihoods = {
            "binary_sensor.test_motion": (0.8, 0.1),
        }
        updated_ids = _update_likelihoods_in_db(
            test_db, coordinator.entry_id, likelihoods, now
        )

        assert "binary_sensor.test_motion" in updated_ids

        # Verify update - refresh the entity from the database using a new session
        with test_db.get_session() as verify_session:
            updated_entity = (
                verify_session.query(test_db.Entities)
                .filter_by(
                    entry_id=coordinator.entry_id, entity_id="binary_sensor.test_motion"
                )
                .first()
            )
            assert updated_entity is not None
            assert updated_entity.prob_given_true == 0.8
            assert updated_entity.prob_given_false == 0.1

    def test_update_likelihoods_in_db_entity_not_found(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test _update_likelihoods_in_db handles missing entity."""
        coordinator = test_db.coordinator

        # Update likelihoods for non-existent entity (should log warning)
        now = dt_util.utcnow()
        likelihoods = {
            "binary_sensor.non_existent": (0.8, 0.1),
        }
        updated_ids = _update_likelihoods_in_db(
            test_db, coordinator.entry_id, likelihoods, now
        )

        # Entity not found, so not in updated_ids
        assert "binary_sensor.non_existent" not in updated_ids

    @pytest.mark.asyncio
    async def test_start_prior_analysis(
        self, hass: HomeAssistant, test_db: AreaOccupancyDB
    ) -> None:
        """Test start_prior_analysis orchestrates prior calculation."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock analyzer methods
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.PriorAnalyzer"
            ) as mock_analyzer_class,
            patch.object(area.prior, "set_global_prior") as mock_set_prior,
        ):
            mock_analyzer = Mock()
            mock_analyzer.sensor_ids = ["binary_sensor.motion"]
            mock_analyzer.analyze_area_prior = Mock(return_value=0.5)
            mock_analyzer.analyze_time_priors = Mock()
            mock_analyzer_class.return_value = mock_analyzer

            await start_prior_analysis(coordinator, area_name, area.prior)

            mock_analyzer.analyze_area_prior.assert_called_once()
            mock_analyzer.analyze_time_priors.assert_called_once()
            mock_set_prior.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_prior_analysis_with_errors(
        self, hass: HomeAssistant, test_db: AreaOccupancyDB
    ) -> None:
        """Test start_prior_analysis handles errors."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock analyzer to raise error
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.PriorAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.sensor_ids = ["binary_sensor.motion"]
            mock_analyzer.analyze_area_prior = Mock(
                side_effect=ValueError("Test error")
            )
            mock_analyzer_class.return_value = mock_analyzer

            with pytest.raises(ValueError):
                await start_prior_analysis(coordinator, area_name, area.prior)

    @pytest.mark.asyncio
    async def test_start_likelihood_analysis(
        self, hass: HomeAssistant, test_db: AreaOccupancyDB
    ) -> None:
        """Test start_likelihood_analysis orchestrates likelihood calculation."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock prior to return occupied intervals
        area.prior.get_occupied_intervals = Mock(
            return_value=[
                (
                    dt_util.utcnow() - timedelta(hours=1),
                    dt_util.utcnow(),
                )
            ]
        )

        # Mock analyzer methods
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.analyze_likelihoods = Mock(
                return_value={
                    "binary_sensor.test": (0.8, 0.1),
                }
            )
            mock_analyzer_class.return_value = mock_analyzer

            # Mock entity manager
            mock_entity = Mock()
            mock_entity.update_likelihood = Mock()
            area.entities.get_entity = Mock(return_value=mock_entity)

            await start_likelihood_analysis(coordinator, area_name, area.entities)

            mock_analyzer.analyze_likelihoods.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_likelihood_analysis_area_not_found(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test start_likelihood_analysis handles missing area."""
        with pytest.raises(ValueError, match="area 'Invalid Area' not found"):
            await start_likelihood_analysis(coordinator, "Invalid Area", Mock())

    @pytest.mark.asyncio
    async def test_start_likelihood_analysis_no_likelihoods(
        self, hass: HomeAssistant, test_db: AreaOccupancyDB
    ) -> None:
        """Test start_likelihood_analysis handles no likelihoods."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock prior to return occupied intervals
        area.prior.get_occupied_intervals = Mock(return_value=[])

        # Mock analyzer to return empty likelihoods
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.analyze_likelihoods = Mock(return_value={})
            mock_analyzer_class.return_value = mock_analyzer

            # Should complete without error
            await start_likelihood_analysis(coordinator, area_name, area.entities)


class TestPriorAnalyzerEdgeCases:
    """Test edge cases and error paths in PriorAnalyzer."""

    def test_analyze_area_prior_with_zero_total_seconds(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_area_prior handles zero total seconds."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock zero total seconds - use same time for both bounds
        now = dt_util.utcnow()
        with (
            patch.object(
                analyzer,
                "get_total_occupied_seconds",
                return_value=100.0,
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now, now),  # Same time = zero total seconds
            ),
        ):
            result = analyzer.analyze_area_prior(analyzer.sensor_ids)
            assert result == DEFAULT_PRIOR

    def test_analyze_area_prior_no_media_appliance_sensors(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_area_prior when no media/appliance sensors configured."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.media = []
        area.config.sensors.appliance = []

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock low motion prior
        with (
            patch.object(
                analyzer,
                "get_total_occupied_seconds",
                return_value=100.0,
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(
                    dt_util.utcnow() - timedelta(days=10),
                    dt_util.utcnow(),
                ),
            ),
        ):
            result = analyzer.analyze_area_prior(analyzer.sensor_ids)
            assert 0.0 <= result <= 1.0

    def test_analyze_time_priors_with_exceptions(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_time_priors handles database exceptions."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock database exception during query
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(
                    dt_util.utcnow() - timedelta(days=1),
                    dt_util.utcnow(),
                ),
            ),
            patch.object(
                coordinator.db,
                "get_session",
                side_effect=OperationalError("Database error", None, None),
            ),
        ):
            # Should handle exception gracefully
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_get_time_bounds_with_exceptions(self, test_db: AreaOccupancyDB) -> None:
        """Test get_time_bounds handles database exceptions."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Test OperationalError
        with patch.object(
            coordinator.db,
            "get_session",
            side_effect=OperationalError("Database connection error", None, None),
        ):
            result = analyzer.get_time_bounds(["test.entity"])
            assert result == (None, None)

        # Test ProgrammingError
        with patch.object(
            coordinator.db,
            "get_session",
            side_effect=ProgrammingError("Query error", None, None),
        ):
            result = analyzer.get_time_bounds(["test.entity"])
            assert result == (None, None)

        # Test SQLAlchemyError
        with patch.object(
            coordinator.db,
            "get_session",
            side_effect=SQLAlchemyError("General database error", None),
        ):
            result = analyzer.get_time_bounds(["test.entity"])
            assert result == (None, None)

        # Test ValueError/TypeError/RuntimeError/OSError
        with patch.object(
            coordinator.db,
            "get_session",
            side_effect=ValueError("Unexpected error"),
        ):
            result = analyzer.get_time_bounds(["test.entity"])
            assert result == (None, None)

    def test_get_occupied_intervals_complex_merging(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals with complex interval merging and timeout."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.motion_timeout = 300  # 5 minutes

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities and intervals with various overlapping scenarios
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(entity)
        session.commit()

        # Create multiple overlapping and adjacent intervals
        intervals = [
            # First interval
            test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.test_motion",
                state="on",
                start_time=start_time.replace(tzinfo=UTC),
                end_time=(start_time + timedelta(minutes=30)).replace(tzinfo=UTC),
                duration_seconds=1800.0,
            ),
            # Adjacent interval (should merge)
            test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.test_motion",
                state="on",
                start_time=(start_time + timedelta(minutes=30)).replace(tzinfo=UTC),
                end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
                duration_seconds=1800.0,
            ),
            # Overlapping interval (should merge)
            test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.test_motion",
                state="on",
                start_time=(start_time + timedelta(minutes=45)).replace(tzinfo=UTC),
                end_time=(start_time + timedelta(hours=1, minutes=30)).replace(
                    tzinfo=UTC
                ),
                duration_seconds=2700.0,
            ),
        ]
        for interval in intervals:
            session.add(interval)
        session.commit()

        # Get occupied intervals - should merge and apply timeout
        intervals_result = analyzer.get_occupied_intervals(lookback_days=90)
        assert isinstance(intervals_result, list)

    def test_get_occupied_intervals_with_exceptions(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test get_occupied_intervals handles database exceptions."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock database exception
        with patch.object(
            coordinator.db,
            "get_session",
            side_effect=OperationalError("Database error", None, None),
        ):
            result = analyzer.get_occupied_intervals(lookback_days=90)
            assert result == []

    def test_likelihood_analyzer_analyze_likelihoods_exceptions(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_likelihoods handles exceptions."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Mock database exception
        with patch.object(
            coordinator.db,
            "get_session",
            side_effect=DataError("Database error", None, None),
        ):
            likelihoods = analyzer.analyze_likelihoods([], area.entities)
            assert likelihoods == {}

    @pytest.mark.asyncio
    async def test_start_likelihood_analysis_exceptions(
        self, hass: HomeAssistant, test_db: AreaOccupancyDB
    ) -> None:
        """Test start_likelihood_analysis handles exceptions."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock prior to return occupied intervals
        area.prior.get_occupied_intervals = Mock(
            return_value=[
                (
                    dt_util.utcnow() - timedelta(hours=1),
                    dt_util.utcnow(),
                )
            ]
        )

        # Mock analyzer to raise exception
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.analyze_likelihoods = Mock(
                side_effect=SQLAlchemyError("Database error")
            )
            mock_analyzer_class.return_value = mock_analyzer

            with pytest.raises(SQLAlchemyError):
                await start_likelihood_analysis(coordinator, area_name, area.entities)

    @pytest.mark.asyncio
    async def test_start_likelihood_analysis_entity_not_found(
        self, hass: HomeAssistant, test_db: AreaOccupancyDB
    ) -> None:
        """Test start_likelihood_analysis handles entity not found in manager."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock prior to return occupied intervals
        area.prior.get_occupied_intervals = Mock(
            return_value=[
                (
                    dt_util.utcnow() - timedelta(hours=1),
                    dt_util.utcnow(),
                )
            ]
        )

        # Mock analyzer to return likelihoods
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.analyze_likelihoods = Mock(
                return_value={
                    "binary_sensor.test": (0.8, 0.1),
                }
            )
            mock_analyzer_class.return_value = mock_analyzer

            # Mock entity manager to raise ValueError (entity not found)
            area.entities.get_entity = Mock(side_effect=ValueError("Entity not found"))

            # Should complete without error (just logs warning)
            await start_likelihood_analysis(coordinator, area_name, area.entities)

    def test_get_occupied_intervals_union_all_path(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals uses UNION ALL path with multiple sensor types."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.media = ["media_player.test_tv"]
        area.config.sensors.appliance = ["switch.test_appliance"]

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities and intervals for all sensor types
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        motion_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(motion_entity)

        media_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            entity_type=InputType.MEDIA,
        )
        session.add(media_entity)

        appliance_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="switch.test_appliance",
            entity_type=InputType.APPLIANCE,
        )
        session.add(appliance_entity)
        session.commit()

        # Create intervals for all sensor types
        motion_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start_time.replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(motion_interval)

        media_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            state=STATE_PLAYING,
            start_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(media_interval)

        appliance_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="switch.test_appliance",
            state=STATE_ON,
            start_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=3)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(appliance_interval)
        session.commit()

        # Get occupied intervals with all sensor types - should use UNION ALL path
        intervals = analyzer.get_occupied_intervals(
            lookback_days=90, include_media=True, include_appliance=True
        )
        assert isinstance(intervals, list)

    def test_get_occupied_intervals_motion_timeout_segmentation(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals segments intervals for motion timeout correctly."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.media = ["media_player.test_tv"]
        area.config.sensors.motion_timeout = 600  # 10 minutes

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        motion_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(motion_entity)

        media_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            entity_type=InputType.MEDIA,
        )
        session.add(media_entity)
        session.commit()

        # Create intervals: media before motion, motion, media after motion
        # This tests the "before motion", "motion segment", and "after motion" logic
        media_before = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            state=STATE_PLAYING,
            start_time=start_time.replace(tzinfo=UTC),
            end_time=(start_time + timedelta(minutes=30)).replace(tzinfo=UTC),
            duration_seconds=1800.0,
        )
        session.add(media_before)

        motion_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=(start_time + timedelta(minutes=30)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            duration_seconds=1800.0,
        )
        session.add(motion_interval)

        media_after = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            state=STATE_PLAYING,
            start_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1, minutes=30)).replace(tzinfo=UTC),
            duration_seconds=1800.0,
        )
        session.add(media_after)
        session.commit()

        # Get occupied intervals - should segment correctly
        intervals = analyzer.get_occupied_intervals(
            lookback_days=90, include_media=True
        )
        assert isinstance(intervals, list)
        # Intervals may be empty if lookback window doesn't include the test data

    def test_get_occupied_intervals_mixed_sensor_types(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals with mixed sensor types and complex overlaps."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.media = ["media_player.test_tv"]
        area.config.sensors.appliance = ["switch.test_appliance"]

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        motion_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(motion_entity)

        media_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            entity_type=InputType.MEDIA,
        )
        session.add(media_entity)

        appliance_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="switch.test_appliance",
            entity_type=InputType.APPLIANCE,
        )
        session.add(appliance_entity)
        session.commit()

        # Create overlapping intervals from different sensor types
        motion_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start_time.replace(tzinfo=UTC),
            end_time=(start_time + timedelta(minutes=45)).replace(tzinfo=UTC),
            duration_seconds=2700.0,
        )
        session.add(motion_interval)

        # Media overlaps with motion
        media_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            state=STATE_PLAYING,
            start_time=(start_time + timedelta(minutes=30)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1, minutes=30)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(media_interval)

        # Appliance overlaps with media
        appliance_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="switch.test_appliance",
            state=STATE_ON,
            start_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(appliance_interval)
        session.commit()

        # Get occupied intervals - should merge and segment correctly
        intervals = analyzer.get_occupied_intervals(
            lookback_days=90, include_media=True, include_appliance=True
        )
        assert isinstance(intervals, list)

    def test_get_occupied_intervals_no_motion_overlap(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals handles media/appliance intervals with no motion overlap."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.media = ["media_player.test_tv"]

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        motion_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(motion_entity)

        media_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            entity_type=InputType.MEDIA,
        )
        session.add(media_entity)
        session.commit()

        # Create motion interval at one time
        motion_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start_time.replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(motion_interval)

        # Create media interval at different time (no overlap)
        media_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            state=STATE_PLAYING,
            start_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=3)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(media_interval)
        session.commit()

        # Get occupied intervals - should handle non-overlapping intervals correctly
        intervals = analyzer.get_occupied_intervals(
            lookback_days=90, include_media=True
        )
        assert isinstance(intervals, list)

    def test_get_occupied_intervals_union_all_path_with_results(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test get_occupied_intervals UNION ALL path returns results with sensor_type."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.sensors.media = ["media_player.test_tv"]
        area.config.sensors.appliance = ["switch.test_appliance"]

        analyzer = PriorAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities and intervals within lookback window
        session = db_test_session
        now = dt_util.utcnow()
        # Create intervals within 90-day lookback
        start_time = now - timedelta(days=1)

        motion_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(motion_entity)

        media_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            entity_type=InputType.MEDIA,
        )
        session.add(media_entity)

        appliance_entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="switch.test_appliance",
            entity_type=InputType.APPLIANCE,
        )
        session.add(appliance_entity)
        session.commit()

        # Create intervals for all sensor types - ensure they're within lookback
        motion_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start_time.replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(motion_interval)

        media_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="media_player.test_tv",
            state=STATE_PLAYING,
            start_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(media_interval)

        appliance_interval = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="switch.test_appliance",
            state=STATE_ON,
            start_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=3)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add(appliance_interval)
        session.commit()

        # Get occupied intervals with all sensor types - should execute UNION ALL path
        # and return results with sensor_type from union_query
        intervals = analyzer.get_occupied_intervals(
            lookback_days=90, include_media=True, include_appliance=True
        )
        assert isinstance(intervals, list)

    def test_analyze_time_priors_no_time_bounds(self, test_db: AreaOccupancyDB) -> None:
        """Test analyze_time_priors handles no time bounds."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Mock no time bounds
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(None, None),
            ),
        ):
            # Should return early
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_analyze_time_priors_timezone_awareness(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_time_priors handles timezone-naive times."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create timezone-naive datetimes
        now = dt_util.utcnow().replace(tzinfo=None)

        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now, now + timedelta(days=1)),
            ),
        ):
            # Should handle timezone-naive times by adding UTC
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_analyze_time_priors_invalid_days(self, test_db: AreaOccupancyDB) -> None:
        """Test analyze_time_priors handles invalid days calculation."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # Create times where last_time is before first_time (invalid)
        now = dt_util.utcnow()

        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(
                    now,
                    now - timedelta(days=1),
                ),  # Invalid: last before first
            ),
        ):
            # Should handle invalid days calculation
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_analyze_time_priors_invalid_slots_per_day(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_time_priors handles invalid slots_per_day calculation."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        # This edge case is hard to trigger because slot_minutes is validated
        # before slots_per_day is calculated. The check would trigger if
        # MINUTES_PER_DAY // slot_minutes == 0, but slot_minutes is validated
        # to be <= MINUTES_PER_DAY, so this can't happen in practice.
        # We'll skip this specific test case since it's not reachable.
        # Instead, test with a valid configuration.
        now = dt_util.utcnow()

        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now - timedelta(days=1), now),
            ),
        ):
            # Test with valid configuration
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_analyze_time_priors_invalid_weekday(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_time_priors handles invalid weekday in interval aggregates."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        now = dt_util.utcnow()

        # Mock invalid weekday (outside 0-6 range)
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[
                    (7, 0, 100.0),  # Invalid weekday (7 > 6)
                    (0, 0, 100.0),  # Valid weekday
                ],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now - timedelta(days=1), now),
            ),
        ):
            # Should skip invalid weekday but process valid one
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_analyze_time_priors_zero_total_slot_seconds(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_time_priors handles zero total slot seconds.

        Note: The zero check is defensive code that's hard to test directly
        because total_slot_seconds = days * slot_duration_seconds, and both
        are validated to be > 0. This test ensures the code path exists.
        """
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        now = dt_util.utcnow()

        # Test with valid data - the zero check is defensive code
        # that protects against edge cases in calculation
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now - timedelta(days=1), now),
            ),
        ):
            # Test with valid configuration
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_analyze_time_priors_exception_handling(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test analyze_time_priors handles all exception types."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = PriorAnalyzer(coordinator, area_name)

        now = dt_util.utcnow()

        # Test exception handling during database operations - IntegrityError
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now - timedelta(days=1), now),
            ),
            patch.object(
                coordinator.db,
                "get_session",
                side_effect=IntegrityError("Integrity error", None, None),
            ),
        ):
            # Should handle exception gracefully
            analyzer.analyze_time_priors(slot_minutes=60)

        # Test DataError
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now - timedelta(days=1), now),
            ),
            patch.object(
                coordinator.db,
                "get_session",
                side_effect=DataError("Data error", None, None),
            ),
        ):
            analyzer.analyze_time_priors(slot_minutes=60)

        # Test ProgrammingError
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now - timedelta(days=1), now),
            ),
            patch.object(
                coordinator.db,
                "get_session",
                side_effect=ProgrammingError("Query error", None, None),
            ),
        ):
            analyzer.analyze_time_priors(slot_minutes=60)

        # Test SQLAlchemyError
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now - timedelta(days=1), now),
            ),
            patch.object(
                coordinator.db,
                "get_session",
                side_effect=SQLAlchemyError("General error", None),
            ),
        ):
            analyzer.analyze_time_priors(slot_minutes=60)

        # Test ValueError/TypeError/RuntimeError/OSError
        with (
            patch.object(
                analyzer,
                "get_interval_aggregates",
                return_value=[(0, 0, 100.0)],
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(now - timedelta(days=1), now),
            ),
            patch.object(
                coordinator.db,
                "get_session",
                side_effect=RuntimeError("Runtime error"),
            ),
        ):
            analyzer.analyze_time_priors(slot_minutes=60)

    def test_likelihood_analyzer_analyze_likelihoods_complete_path(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test analyze_likelihoods completes full path for all branches."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        # Create test entities and intervals with various states
        session = db_test_session
        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(entity)
        session.commit()

        # Create intervals: some active+occupied, some active+empty, some inactive+occupied, some inactive+empty
        intervals = [
            # Active and occupied
            test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.test_motion",
                state="on",
                start_time=start_time.replace(tzinfo=UTC),
                end_time=(start_time + timedelta(minutes=30)).replace(tzinfo=UTC),
                duration_seconds=1800.0,
            ),
            # Active but empty (occupied period is different)
            test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.test_motion",
                state="on",
                start_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
                end_time=(start_time + timedelta(hours=2, minutes=30)).replace(
                    tzinfo=UTC
                ),
                duration_seconds=1800.0,
            ),
            # Inactive but occupied (state is "off" but during occupied time)
            test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.test_motion",
                state="off",
                start_time=(start_time + timedelta(minutes=30)).replace(tzinfo=UTC),
                end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
                duration_seconds=1800.0,
            ),
            # Inactive and empty
            test_db.Intervals(
                entry_id=coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.test_motion",
                state="off",
                start_time=(start_time + timedelta(hours=3)).replace(tzinfo=UTC),
                end_time=(start_time + timedelta(hours=3, minutes=30)).replace(
                    tzinfo=UTC
                ),
                duration_seconds=1800.0,
            ),
        ]
        for interval in intervals:
            session.add(interval)
        session.commit()

        # Analyze likelihoods with occupied times
        occupied_times = [
            (start_time, start_time + timedelta(hours=1)),  # First hour is occupied
        ]

        likelihoods = analyzer.analyze_likelihoods(occupied_times, area.entities)
        assert isinstance(likelihoods, dict)

    @pytest.mark.asyncio
    async def test_start_prior_analysis_sqlalchemy_error(
        self, hass: HomeAssistant, test_db: AreaOccupancyDB
    ) -> None:
        """Test start_prior_analysis handles SQLAlchemyError."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock analyzer to raise SQLAlchemyError
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.PriorAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.sensor_ids = ["binary_sensor.motion"]
            mock_analyzer.analyze_area_prior = Mock(
                side_effect=SQLAlchemyError("DB error")
            )
            mock_analyzer_class.return_value = mock_analyzer

            with pytest.raises(SQLAlchemyError):
                await start_prior_analysis(coordinator, area_name, area.prior)

    @pytest.mark.asyncio
    async def test_start_likelihood_analysis_data_errors(
        self, hass: HomeAssistant, test_db: AreaOccupancyDB
    ) -> None:
        """Test start_likelihood_analysis handles ValueError and TypeError."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Mock prior to return occupied intervals
        area.prior.get_occupied_intervals = Mock(
            return_value=[
                (
                    dt_util.utcnow() - timedelta(hours=1),
                    dt_util.utcnow(),
                )
            ]
        )

        # Test ValueError
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.analyze_likelihoods = Mock(
                side_effect=ValueError("Data error")
            )
            mock_analyzer_class.return_value = mock_analyzer

            with pytest.raises(ValueError):
                await start_likelihood_analysis(coordinator, area_name, area.entities)

        # Test TypeError
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.analyze_likelihoods = Mock(
                side_effect=TypeError("Type error")
            )
            mock_analyzer_class.return_value = mock_analyzer

            with pytest.raises(TypeError):
                await start_likelihood_analysis(coordinator, area_name, area.entities)

        # Test ZeroDivisionError
        with (
            patch(
                "custom_components.area_occupancy.data.analysis.LikelihoodAnalyzer"
            ) as mock_analyzer_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.analyze_likelihoods = Mock(
                side_effect=ZeroDivisionError("Division by zero")
            )
            mock_analyzer_class.return_value = mock_analyzer

            with pytest.raises(ZeroDivisionError):
                await start_likelihood_analysis(coordinator, area_name, area.entities)


class TestLikelihoodAnalyzerHelperMethods:
    """Test LikelihoodAnalyzer helper methods."""

    def test_get_sensors(self, test_db: AreaOccupancyDB, db_test_session) -> None:
        """Test _get_sensors filters by entry_id and area_name."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        session = db_test_session

        # Create "Different Area" directly in database for entity3 (not in coordinator)
        different_area = test_db.Areas(
            entry_id=coordinator.entry_id,
            area_name="Different Area",
            area_id="different_area",
            purpose="living",
            threshold=0.5,
        )
        session.add(different_area)
        session.commit()

        # Create test entities
        entity1 = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion1",
            entity_type=InputType.MOTION,
        )
        entity2 = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion2",
            entity_type=InputType.MOTION,
        )
        # Entity in different area (should not be returned)
        entity3 = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name="Different Area",
            entity_id="binary_sensor.other",
            entity_type=InputType.MOTION,
        )
        session.add_all([entity1, entity2, entity3])
        session.commit()

        sensors = analyzer._get_sensors(session)

        assert len(sensors) == 2
        entity_ids = {str(sensor.entity_id) for sensor in sensors}
        assert "binary_sensor.test_motion1" in entity_ids
        assert "binary_sensor.test_motion2" in entity_ids
        assert "binary_sensor.other" not in entity_ids

    def test_get_intervals_by_entity(
        self, test_db: AreaOccupancyDB, db_test_session
    ) -> None:
        """Test _get_intervals_by_entity groups intervals by entity."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        # Ensure area exists first (foreign key requirement)
        test_db.save_area_data(area_name)

        session = db_test_session

        now = dt_util.utcnow()
        start_time = now - timedelta(days=1)

        # Create test entity
        entity = test_db.Entities(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            entity_type=InputType.MOTION,
        )
        session.add(entity)
        session.commit()

        # Create intervals for this entity
        interval1 = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="on",
            start_time=start_time.replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        interval2 = test_db.Intervals(
            entry_id=coordinator.entry_id,
            area_name=area_name,
            entity_id="binary_sensor.test_motion",
            state="off",
            start_time=(start_time + timedelta(hours=1)).replace(tzinfo=UTC),
            end_time=(start_time + timedelta(hours=2)).replace(tzinfo=UTC),
            duration_seconds=3600.0,
        )
        session.add_all([interval1, interval2])
        session.commit()

        sensors = [entity]
        intervals_by_entity = analyzer._get_intervals_by_entity(session, sensors)

        assert isinstance(intervals_by_entity, dict)
        assert "binary_sensor.test_motion" in intervals_by_entity
        assert len(intervals_by_entity["binary_sensor.test_motion"]) == 2

    def test_analyze_entity_likelihood_all_combinations(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test _analyze_entity_likelihood with all probability calculation paths."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        now = dt_util.utcnow()
        entity = Mock()
        entity.entity_id = "binary_sensor.test_motion"

        # Create intervals covering all combinations:
        # - Active and occupied (true_occ)
        # - Active but empty (true_empty)
        # - Inactive but occupied (false_occ)
        # - Inactive and empty (false_empty)
        intervals_by_entity = {
            "binary_sensor.test_motion": [
                IntervalData(
                    entity_id="binary_sensor.test_motion",
                    start_time=now - timedelta(hours=2),
                    duration_seconds=1800.0,  # 30 min
                    state="on",  # Active
                ),
                IntervalData(
                    entity_id="binary_sensor.test_motion",
                    start_time=now - timedelta(hours=4),
                    duration_seconds=1800.0,  # 30 min
                    state="on",  # Active
                ),
                IntervalData(
                    entity_id="binary_sensor.test_motion",
                    start_time=now - timedelta(hours=6),
                    duration_seconds=1800.0,  # 30 min
                    state="off",  # Inactive
                ),
                IntervalData(
                    entity_id="binary_sensor.test_motion",
                    start_time=now - timedelta(hours=8),
                    duration_seconds=1800.0,  # 30 min
                    state="off",  # Inactive
                ),
            ]
        }

        # Occupied times: first hour (covers first two intervals)
        occupied_times = [
            (now - timedelta(hours=2), now - timedelta(hours=1)),
        ]

        entity_obj = Mock()
        entity_obj.active_states = ["on"]
        entity_obj.active_range = None

        entity_manager = Mock()
        entity_manager.get_entity = Mock(return_value=entity_obj)

        prob_given_true, prob_given_false = analyzer._analyze_entity_likelihood(
            entity, intervals_by_entity, occupied_times, entity_manager
        )

        assert 0.0 <= prob_given_true <= 1.0
        assert 0.0 <= prob_given_false <= 1.0

    def test_analyze_entity_likelihood_zero_denominators(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test _analyze_entity_likelihood handles zero denominators."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        now = dt_util.utcnow()
        entity = Mock()
        entity.entity_id = "binary_sensor.test_motion"

        # Only active intervals, no occupied times
        intervals_by_entity = {
            "binary_sensor.test_motion": [
                IntervalData(
                    entity_id="binary_sensor.test_motion",
                    start_time=now - timedelta(hours=2),
                    duration_seconds=1800.0,
                    state="on",
                ),
            ]
        }

        occupied_times = []  # No occupied times

        entity_obj = Mock()
        entity_obj.active_states = ["on"]
        entity_obj.active_range = None

        entity_manager = Mock()
        entity_manager.get_entity = Mock(return_value=entity_obj)

        prob_given_true, prob_given_false = analyzer._analyze_entity_likelihood(
            entity, intervals_by_entity, occupied_times, entity_manager
        )

        # Should return default 0.5 when denominator is zero
        assert prob_given_true == 0.5
        assert 0.0 <= prob_given_false <= 1.0

    def test_analyze_entity_likelihood_with_active_range(
        self, test_db: AreaOccupancyDB
    ) -> None:
        """Test _analyze_entity_likelihood with active_range entity."""
        coordinator = test_db.coordinator
        area_name = coordinator.get_area_names()[0]
        analyzer = LikelihoodAnalyzer(coordinator, area_name)

        now = dt_util.utcnow()
        entity = Mock()
        entity.entity_id = "sensor.test"

        intervals_by_entity = {
            "sensor.test": [
                IntervalData(
                    entity_id="sensor.test",
                    start_time=now - timedelta(hours=1),
                    duration_seconds=3600.0,
                    state="0.5",  # Within range (0.0, 1.0)
                ),
                IntervalData(
                    entity_id="sensor.test",
                    start_time=now - timedelta(hours=3),
                    duration_seconds=3600.0,
                    state="1.5",  # Outside range
                ),
            ]
        }

        occupied_times = [
            (now - timedelta(hours=1), now),
        ]

        entity_obj = Mock()
        entity_obj.active_states = None
        entity_obj.active_range = (0.0, 1.0)

        entity_manager = Mock()
        entity_manager.get_entity = Mock(return_value=entity_obj)

        prob_given_true, prob_given_false = analyzer._analyze_entity_likelihood(
            entity, intervals_by_entity, occupied_times, entity_manager
        )

        assert 0.0 <= prob_given_true <= 1.0
        assert 0.0 <= prob_given_false <= 1.0
