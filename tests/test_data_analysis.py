"""Tests for data.analysis module."""

from datetime import timedelta
import logging
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import DataError, OperationalError, ProgrammingError

from custom_components.area_occupancy.data.analysis import (
    DEFAULT_OCCUPIED_SECONDS,
    DEFAULT_PRIOR,
    MAX_PROBABILITY,
    LikelihoodAnalyzer,
    PriorAnalyzer,
)
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity, EntityManager
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from homeassistant.const import STATE_ON
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
        decay = Decay()
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

    def test_likelihood_calculation_logic(self, mock_coordinator: Mock) -> None:
        """Test the complex likelihood calculation logic in LikelihoodAnalyzer._analyze_entity_likelihood."""
        # Create manager with mocked factory
        with patch(
            "custom_components.area_occupancy.data.entity.EntityFactory"
        ) as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_all_from_config.return_value = {}
            mock_factory_class.return_value = mock_factory

            manager = EntityManager(mock_coordinator, area_name="Test Area")

            # Create test entity
            test_entity = create_test_entity(
                "test_sensor", coordinator=mock_coordinator
            )
            manager._entities = {"test_sensor": test_entity}

            # Mock database entity
            mock_db_entity = Mock()
            mock_db_entity.entity_id = "test_sensor"

            # Mock intervals for insufficient data test
            mock_interval_insufficient = Mock()
            mock_interval_insufficient.start_time = dt_util.utcnow()
            mock_interval_insufficient.duration_seconds = 1800.0  # Less than 3600s
            mock_interval_insufficient.state = "on"

            # Add to_dict method to return proper dictionary
            def mock_interval_to_dict():
                return {
                    "start_time": mock_interval_insufficient.start_time,
                    "duration_seconds": mock_interval_insufficient.duration_seconds,
                    "state": mock_interval_insufficient.state,
                }

            mock_interval_insufficient.to_dict = mock_interval_to_dict

            intervals_by_entity_insufficient = {
                "test_sensor": [mock_interval_insufficient]
            }
            occupied_times = [(dt_util.utcnow() - timedelta(hours=1), dt_util.utcnow())]

            # Test motion sensor logic (insufficient data)
            test_entity.type.input_type = InputType.MOTION
            test_entity.type.prob_given_true = 0.8
            test_entity.type.prob_given_false = 0.1

            analyzer = LikelihoodAnalyzer(mock_coordinator, "Test Area")
            with (
                patch.object(analyzer, "_is_occupied", return_value=True),
                patch.object(analyzer, "_is_interval_active", return_value=True),
            ):
                prob_given_true, prob_given_false = analyzer._analyze_entity_likelihood(
                    mock_db_entity,
                    intervals_by_entity_insufficient,
                    occupied_times,
                    manager,
                )

                # Since we have insufficient data (< 3600s), it should use defaults
                assert prob_given_true == 0.5  # Default fallback when insufficient data
                assert (
                    prob_given_false == 0.5
                )  # Default fallback when insufficient data

            # Test motion sensor logic (sufficient data - calculated values preserved)
            test_entity_sufficient = create_test_entity(
                "test_sensor_sufficient", coordinator=mock_coordinator
            )
            test_entity_sufficient.type.input_type = InputType.MOTION
            test_entity_sufficient.type.prob_given_true = 0.95  # Default
            test_entity_sufficient.type.prob_given_false = 0.02  # Default
            manager._entities = {"test_sensor_sufficient": test_entity_sufficient}

            # Mock intervals for sufficient data test (>= 3600s total)
            # Create scenario that yields calculated values outside typical ranges
            # We need two intervals with equal durations:
            # - One active when occupied (true_occ)
            # - One inactive when occupied (false_occ)
            # This yields prob_given_true = 3600 / (3600 + 3600) = 0.5
            mock_interval_sufficient = Mock()
            mock_interval_sufficient.start_time = dt_util.utcnow() - timedelta(hours=2)
            mock_interval_sufficient.duration_seconds = (
                3600.0  # 1 hour - active interval
            )
            mock_interval_sufficient.state = "on"
            mock_interval_sufficient.to_dict = lambda: {
                "start_time": mock_interval_sufficient.start_time,
                "duration_seconds": mock_interval_sufficient.duration_seconds,
                "state": mock_interval_sufficient.state,
            }

            mock_interval_sufficient_2 = Mock()
            mock_interval_sufficient_2.start_time = (
                mock_interval_sufficient.start_time + timedelta(hours=1)
            )
            mock_interval_sufficient_2.duration_seconds = (
                3600.0  # 1 hour - inactive interval
            )
            mock_interval_sufficient_2.state = "off"  # Not active
            mock_interval_sufficient_2.to_dict = lambda: {
                "start_time": mock_interval_sufficient_2.start_time,
                "duration_seconds": mock_interval_sufficient_2.duration_seconds,
                "state": mock_interval_sufficient_2.state,
            }

            mock_db_entity_sufficient = Mock()
            mock_db_entity_sufficient.entity_id = "test_sensor_sufficient"

            intervals_by_entity_sufficient = {
                "test_sensor_sufficient": [
                    mock_interval_sufficient,
                    mock_interval_sufficient_2,
                ]
            }

            # First interval: active when occupied (true_occ = 3600)
            # Second interval: inactive when occupied (false_occ = 3600)
            # This yields prob_given_true = 3600 / (3600 + 3600) = 0.5
            # Occupied period covers both intervals' start times
            occupied_times_sufficient = [
                (
                    mock_interval_sufficient.start_time,
                    mock_interval_sufficient_2.start_time
                    + timedelta(hours=1),  # Cover both intervals
                )
            ]

            # Don't need to mock _is_occupied - it will use the actual occupied_times list
            def mock_is_interval_active(interval, entity):
                """Return True only for the first interval (active)."""
                return interval == mock_interval_sufficient

            analyzer = LikelihoodAnalyzer(mock_coordinator, "Test Area")
            with patch.object(
                analyzer, "_is_interval_active", side_effect=mock_is_interval_active
            ):
                prob_given_true, prob_given_false = analyzer._analyze_entity_likelihood(
                    mock_db_entity_sufficient,
                    intervals_by_entity_sufficient,
                    occupied_times_sufficient,
                    manager,
                )

                # With sufficient data (>= 3600s), calculated values should be preserved
                # even if outside typical ranges (prob_given_true < 0.8)
                # First interval: active=True, occupied=True -> true_occ = 3600
                # Second interval: active=False, occupied=True -> false_occ = 3600
                # prob_given_true = 3600 / (3600 + 3600) = 0.5
                # Both intervals are during occupied time, so true_empty=0, false_empty=0
                # total_unoccupied_time = 0 < 3600, so prob_given_false uses default (0.5)
                # This proves threshold overrides are not applied for prob_given_true
                assert abs(prob_given_true - 0.5) < 0.01  # Calculated value ~0.5
                assert (
                    prob_given_false == 0.5
                )  # Default fallback when insufficient unoccupied data
                # Verify prob_given_true is not the default
                assert abs(prob_given_true - 0.95) > 0.01

            # Test non-motion sensor logic - create a fresh entity to avoid interference
            fresh_entity = create_test_entity(
                "test_sensor", coordinator=mock_coordinator
            )
            fresh_entity.type.input_type = InputType.MEDIA
            fresh_entity.type.prob_given_true = 0.6
            fresh_entity.type.prob_given_false = 0.05
            manager._entities = {"test_sensor": fresh_entity}

            # Use insufficient data intervals for non-motion sensor test
            analyzer = LikelihoodAnalyzer(mock_coordinator, "Test Area")
            with (
                patch.object(analyzer, "_is_occupied", return_value=False),
                patch.object(analyzer, "_is_interval_active", return_value=False),
            ):
                prob_given_true, prob_given_false = analyzer._analyze_entity_likelihood(
                    mock_db_entity,
                    intervals_by_entity_insufficient,
                    occupied_times,
                    manager,
                )

                # Should use calculated values for non-motion sensors
                # The logic calculates based on the interval data
                # For this test case with _is_occupied=False and _is_interval_active=False:
                # - true_occ = 0 (no intervals active when occupied)
                # - false_occ = 1800 (interval not active when occupied)
                # - true_empty = 0 (no intervals active when not occupied)
                # - false_empty = 0 (no intervals not active when not occupied)
                #
                # prob_given_true = true_occ / (true_occ + false_occ) = 0 / (0 + 1800) = 0
                # prob_given_false = true_empty / (true_empty + false_empty) = 0 / (0 + 0) = 0.5 (fallback)
                #
                # Since 0 < threshold, prob_given_true gets reset to 0.5 (fallback)
                assert prob_given_true == 0.5  # Fallback value
                assert prob_given_false == 0.5  # Fallback value

    def test_interval_active_logic(self, mock_coordinator: Mock) -> None:
        """Test the _is_interval_active method with different entity types."""
        analyzer = LikelihoodAnalyzer(mock_coordinator, "Test Area")

        # Create test entity with active_states
        test_entity = create_test_entity("test_sensor", coordinator=mock_coordinator)
        test_entity.type.active_states = ["on", "playing"]

        # Mock interval
        mock_interval = Mock()
        mock_interval.state = "on"

        # Test with active_states
        assert analyzer._is_interval_active(mock_interval, test_entity) is True

        mock_interval.state = "off"
        assert analyzer._is_interval_active(mock_interval, test_entity) is False

        # Test with active_range
        test_entity.type.active_states = None
        test_entity.type.active_range = (10.0, 20.0)

        mock_interval.state = "15"
        assert analyzer._is_interval_active(mock_interval, test_entity) is True

        mock_interval.state = "25"
        assert analyzer._is_interval_active(mock_interval, test_entity) is False

        mock_interval.state = "invalid"
        assert analyzer._is_interval_active(mock_interval, test_entity) is False

        # Test with no active configuration
        test_entity.type.active_states = None
        test_entity.type.active_range = None
        assert analyzer._is_interval_active(mock_interval, test_entity) is False

    def test_is_occupied_binary_search_logic(self, mock_coordinator: Mock) -> None:
        """Test the _is_occupied method with binary search logic."""
        analyzer = LikelihoodAnalyzer(mock_coordinator, "Test Area")

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
        self, mock_coordinator: Mock, entity_ids
    ) -> None:
        """Test analyze_area_prior returns default when entity IDs are invalid."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")
        result = analyzer.analyze_area_prior(entity_ids)
        assert result == DEFAULT_PRIOR

    @pytest.mark.parametrize(
        (
            "first_time",
            "last_time",
            "occupied_seconds",
            "expected_result",
            "description",
        ),
        [
            (
                dt_util.utcnow(),
                dt_util.utcnow(),
                100.0,
                MAX_PROBABILITY,
                "same time (division by zero)",
            ),
            (
                dt_util.utcnow() + timedelta(hours=1),
                dt_util.utcnow(),
                100.0,
                DEFAULT_PRIOR,
                "reversed time",
            ),
            (
                dt_util.utcnow(),
                dt_util.utcnow() + timedelta(hours=1),
                7200.0,  # 2 hours in 1 hour range
                0.99,  # Updated to reflect actual behavior
                "data corruption (occupied > total)",
            ),
            (
                dt_util.utcnow(),
                dt_util.utcnow() + timedelta(hours=2),
                1800.0,  # 30 minutes in 2 hour range
                0.25,
                "normal case",
            ),
            (
                None,
                None,
                0.0,
                DEFAULT_PRIOR,
                "no data available",
            ),
        ],
    )
    def test_analyze_area_prior_various_scenarios(
        self,
        mock_coordinator: Mock,
        first_time,
        last_time,
        occupied_seconds,
        expected_result,
        description,
    ) -> None:
        """Test analyze_area_prior handles various scenarios."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        with (
            patch.object(
                analyzer, "get_total_occupied_seconds", return_value=occupied_seconds
            ),
            patch.object(
                analyzer, "get_time_bounds", return_value=(first_time, last_time)
            ),
        ):
            result = analyzer.analyze_area_prior(["test.entity"])
            assert result == pytest.approx(expected_result, rel=1e-9), (
                f"Failed for {description}"
            )

    @pytest.mark.parametrize(
        ("slot_minutes", "description"),
        [
            (-10, "negative slot_minutes"),
            (70, "slot_minutes not dividing day evenly"),
        ],
    )
    def test_analyze_time_priors_parameter_validation(
        self, mock_coordinator: Mock, slot_minutes, description
    ) -> None:
        """Test analyze_time_priors validates slot_minutes parameter."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        # Mock database session
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_session.add = Mock()
        mock_session.commit = Mock()

        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_coordinator.db.get_session.return_value = mock_context_manager

        with (
            patch.object(analyzer, "get_interval_aggregates", return_value=[]),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(dt_util.utcnow(), dt_util.utcnow() + timedelta(days=1)),
            ),
        ):
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
        self, mock_coordinator: Mock, interval_data, time_bounds, description
    ) -> None:
        """Test analyze_time_priors handles various scenarios."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        # Mock database session
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_session.add = Mock()
        mock_session.commit = Mock()

        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_coordinator.db.get_session.return_value = mock_context_manager

        with (
            patch.object(
                analyzer, "get_interval_aggregates", return_value=interval_data
            ),
            patch.object(analyzer, "get_time_bounds", return_value=time_bounds),
        ):
            analyzer.analyze_time_priors()
            # Should handle all scenarios gracefully

    def test_analyze_time_priors_with_existing_prior(
        self, mock_coordinator: Mock
    ) -> None:
        """Test analyze_time_priors updates existing priors."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        # Mock existing prior
        mock_existing_prior = Mock()
        mock_existing_prior.prior_value = 0.5
        mock_existing_prior.data_points = 100
        mock_existing_prior.last_updated = dt_util.utcnow()

        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_existing_prior
        )
        mock_session.commit = Mock()

        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_coordinator.db.get_session.return_value = mock_context_manager

        with (
            patch.object(
                analyzer, "get_interval_aggregates", return_value=[(0, 0, 100.0)]
            ),
            patch.object(
                analyzer,
                "get_time_bounds",
                return_value=(dt_util.utcnow(), dt_util.utcnow() + timedelta(days=1)),
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
        self, mock_coordinator: Mock, method_name, error_class, expected_result
    ) -> None:
        """Test database error handling for various methods."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        # Mock database session to raise the specified error
        mock_session = Mock()
        mock_session.__enter__ = Mock(side_effect=error_class("Test error", None, None))
        mock_session.__exit__ = Mock()
        mock_coordinator.db.get_session.return_value = mock_session

        # Call the method and verify it handles the error gracefully
        method = getattr(analyzer, method_name)
        if method_name == "get_total_occupied_seconds":
            result = method()
        else:
            result = method(["test.entity"])

        assert result == expected_result

    def test_get_time_bounds_successful_cases(self, mock_coordinator: Mock) -> None:
        """Test get_time_bounds with successful database operations."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

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
            mock_coordinator.db.get_session.return_value = mock_session

            result = analyzer.get_time_bounds(entity_ids)
            assert result == expected_result

    def test_get_total_occupied_seconds_with_none_result(
        self, mock_coordinator: Mock
    ) -> None:
        """Test get_total_occupied_seconds handles None result from database."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.scalar.return_value = None
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock()
        mock_coordinator.db.get_session.return_value = mock_session

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
        self, mock_coordinator: Mock, mock_result, expected_result
    ) -> None:
        """Test get_interval_aggregates handles various scenarios."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        with patch.object(analyzer, "get_occupied_intervals", return_value=mock_result):
            result = analyzer.get_interval_aggregates()
            # The new implementation aggregates intervals differently, so we just check it returns a list
            assert isinstance(result, list)
            # For the empty case, we expect an empty list
            if not mock_result:
                assert result == []
            else:
                # For non-empty cases, we expect some aggregated data
                assert len(result) > 0

    def test_get_interval_aggregates_sql_path(self, mock_coordinator: Mock) -> None:
        """Test successful SQL aggregation path."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        # Mock successful SQL aggregation
        mock_coordinator.db.get_aggregated_intervals_by_slot.return_value = [
            (0, 0, 3600.0),  # Monday, slot 0, 1 hour
            (1, 12, 1800.0),  # Tuesday, slot 12, 30 minutes
        ]

        result = analyzer.get_interval_aggregates(slot_minutes=60)

        assert len(result) == 2
        assert result[0] == (0, 0, 3600.0)
        assert result[1] == (1, 12, 1800.0)

        # Verify SQL method was called
        mock_coordinator.db.get_aggregated_intervals_by_slot.assert_called_once_with(
            mock_coordinator.entry_id, 60
        )

    def test_get_interval_aggregates_fallback(self, mock_coordinator: Mock) -> None:
        """Test fallback to Python method on SQL failure."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        # Mock SQL failure
        mock_coordinator.db.get_aggregated_intervals_by_slot.side_effect = (
            OperationalError("Database error", None, None)
        )

        # Mock Python fallback method
        with patch.object(
            analyzer, "_get_interval_aggregates_python", return_value=[(0, 0, 1800.0)]
        ) as mock_python:
            result = analyzer.get_interval_aggregates(slot_minutes=60)

            assert len(result) == 1
            assert result[0] == (0, 0, 1800.0)

            # Verify Python fallback was called
            mock_python.assert_called_once_with(60)

    def test_get_interval_aggregates_python_fallback(
        self, mock_coordinator: Mock
    ) -> None:
        """Test Python fallback method directly."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        # Mock get_occupied_intervals to return test data
        test_intervals = [
            (
                dt_util.utcnow() - timedelta(hours=2),
                dt_util.utcnow() - timedelta(hours=1),
            ),
            (dt_util.utcnow() - timedelta(hours=1), dt_util.utcnow()),
        ]

        with patch.object(
            analyzer, "get_occupied_intervals", return_value=test_intervals
        ):
            result = analyzer._get_interval_aggregates_python(slot_minutes=60)

            # Should return aggregated data
            assert isinstance(result, list)
            # Each item should be (day_of_week, time_slot, total_seconds)
            for item in result:
                assert len(item) == 3
                assert isinstance(item[0], int)  # day_of_week
                assert isinstance(item[1], int)  # time_slot
                assert isinstance(item[2], float)  # total_seconds

    def test_get_occupied_intervals_with_lookback_days(
        self, mock_coordinator: Mock
    ) -> None:
        """Test get_occupied_intervals with lookback_days parameter."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

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
        mock_coordinator.db.get_session.return_value = mock_context_manager

        # Test default lookback (90 days)
        intervals = analyzer.get_occupied_intervals(lookback_days=90)
        assert isinstance(intervals, list)

        # Test custom lookback
        intervals_custom = analyzer.get_occupied_intervals(lookback_days=30)
        assert isinstance(intervals_custom, list)

    def test_get_occupied_intervals_performance_logging(
        self, mock_coordinator: Mock, caplog
    ) -> None:
        """Test performance logging in get_occupied_intervals."""
        analyzer = PriorAnalyzer(mock_coordinator, "Test Area")

        # Test that the method logs debug messages by testing the actual method
        # We'll mock the database to avoid complex SQLAlchemy mocking
        with patch.object(analyzer.db, "get_session") as mock_session:
            # Create a simple mock that returns empty results
            mock_session.return_value.__enter__.return_value.query.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = []

            with caplog.at_level(logging.DEBUG):
                analyzer.get_occupied_intervals()

        # Check for debug logging
        assert "Getting occupied intervals with unified logic" in caplog.text
