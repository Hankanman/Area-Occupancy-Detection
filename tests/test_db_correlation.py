"""Tests for database correlation analysis functions."""

from datetime import datetime, timedelta
import math

import pytest

from custom_components.area_occupancy.const import CORRELATION_MONTHS_TO_KEEP
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity, EntityType
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.db.correlation import (
    _prune_old_correlations,
    analyze_and_save_correlation,
    analyze_binary_likelihoods,
    analyze_correlation,
    calculate_pearson_correlation,
    get_correlation_for_entity,
    save_correlation_result,
)
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util


# Helper functions to reduce boilerplate
def _create_numeric_entity_with_samples(
    db,
    area_name: str,
    entity_id: str,
    num_samples: int,
    value_generator,
    entity_type: str = "numeric",
    unit_of_measurement: str | None = None,
) -> None:
    """Create a numeric entity with samples for testing."""
    db.save_area_data(area_name)
    now = dt_util.utcnow()

    with db.get_session() as session:
        entity = db.Entities(
            entity_id=entity_id,
            entry_id=db.coordinator.entry_id,
            area_name=area_name,
            entity_type=entity_type,
        )
        session.add(entity)

        for i in range(num_samples):
            sample = db.NumericSamples(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=entity_id,
                timestamp=now - timedelta(hours=num_samples - i),
                value=value_generator(i),
                unit_of_measurement=unit_of_measurement,
            )
            session.add(sample)
        session.commit()


def _create_binary_entity_with_intervals(
    db,
    area_name: str,
    entity_id: str,
    num_intervals: int,
    state_generator,
    entity_type: str = "door",
) -> None:
    """Create a binary entity with intervals for testing."""
    db.save_area_data(area_name)
    now = dt_util.utcnow()

    with db.get_session() as session:
        entity = db.Entities(
            entity_id=entity_id,
            entry_id=db.coordinator.entry_id,
            area_name=area_name,
            entity_type=entity_type,
        )
        session.add(entity)

        for i in range(num_intervals):
            start = now - timedelta(hours=num_intervals - i)
            end = now - timedelta(hours=num_intervals - i - 1)
            state = state_generator(i)
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=entity_id,
                start_time=start,
                end_time=end,
                duration_seconds=(end - start).total_seconds(),
                state=state,
            )
            session.add(interval)
        session.commit()


def _create_occupied_intervals_cache(
    db, area_name: str, intervals: list[tuple], source: str = "motion_sensors"
) -> None:
    """Create occupied intervals cache for testing."""
    db.save_occupied_intervals_cache(area_name, intervals, source)


def _create_entity(
    db, area_name: str, entity_id: str, entity_type: str = "numeric"
) -> None:
    """Create an entity for testing."""
    db.save_area_data(area_name)
    with db.get_session() as session:
        entity = db.Entities(
            entity_id=entity_id,
            entry_id=db.coordinator.entry_id,
            area_name=area_name,
            entity_type=entity_type,
        )
        session.add(entity)
        session.commit()


def _create_motion_intervals(
    db,
    area_name: str,
    motion_entity_id: str,
    num_intervals: int,
    now: datetime | None = None,
) -> list[tuple]:
    """Create motion sensor intervals and return occupied intervals list.

    Args:
        db: Database instance
        area_name: Area name
        motion_entity_id: Motion sensor entity ID
        num_intervals: Number of intervals to create
        now: Optional datetime to use as reference (defaults to dt_util.utcnow())
    """
    if now is None:
        now = dt_util.utcnow()
    intervals = []
    with db.get_session() as session:
        for i in range(num_intervals):
            start = now - timedelta(hours=num_intervals - i)
            end = start + timedelta(hours=1)
            intervals.append((start, end))
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=motion_entity_id,
                start_time=start,
                end_time=end,
                state="on",
                duration_seconds=3600,
            )
            session.add(interval)
        session.commit()

    # Return occupied intervals list for cache (matching the intervals created)
    return intervals


class TestCalculatePearsonCorrelation:
    """Test calculate_pearson_correlation function."""

    @pytest.mark.parametrize(
        ("x_values", "y_values", "expected_corr", "expected_p", "p_tolerance"),
        [
            # Positive correlation
            (
                list(range(1, 51)),
                [x * 2 for x in range(1, 51)],
                1.0,
                None,
                0.01,
            ),
            # Negative correlation
            (
                list(range(1, 51)),
                [101 - x for x in range(1, 51)],
                -1.0,
                None,
                0.01,
            ),
            # No correlation (random values)
            ([1, 2, 3, 4, 5], [5, 2, 8, 1, 9], None, None, None),
            # Very close to perfect correlation
            (
                list(range(1, 51)),
                [x * 2 + 0.0000001 for x in range(1, 51)],
                None,
                None,
                0.01,
            ),
        ],
    )
    def test_calculate_pearson_correlation_variations(
        self, x_values, y_values, expected_corr, expected_p, p_tolerance
    ):
        """Test correlation calculation with various correlation types."""
        correlation, p_value = calculate_pearson_correlation(x_values, y_values)

        if expected_corr is not None:
            assert abs(correlation - expected_corr) < p_tolerance
        else:
            # For no correlation or very close to one, just check bounds
            assert -1.0 <= correlation <= 1.0

        if expected_p is not None:
            assert abs(p_value - expected_p) < 0.01
        else:
            assert 0.0 <= p_value <= 1.0

    @pytest.mark.parametrize(
        ("x_values", "y_values"),
        [
            # Insufficient samples
            ([1, 2], [3, 4]),
            # Mismatched lengths
            ([1, 2, 3], [4, 5]),
            # NaN values
            ([1, 2, float("nan"), 4, 5], [2, 4, 6, 8, 10]),
            # Invalid values (inf)
            ([1, 2, float("inf"), 4, 5], [2, 4, 6, 8, 10]),
            # Exception (invalid types)
            (["invalid", "data"], [1, 2]),
        ],
    )
    def test_calculate_pearson_correlation_error_cases(self, x_values, y_values):
        """Test correlation calculation with error cases."""
        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert correlation == 0.0
        assert p_value == 1.0


class TestAnalyzeCorrelation:
    """Test analyze_correlation function."""

    def test_analyze_correlation_success(self, coordinator: AreaOccupancyCoordinator):
        """Test successful correlation analysis for numeric sensors."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        _create_numeric_entity_with_samples(
            db,
            area_name,
            entity_id,
            100,
            lambda i: 20.0 + (i % 10),
            unit_of_measurement="°C",
        )

        intervals = [
            (now - timedelta(hours=50), now - timedelta(hours=40)),
            (now - timedelta(hours=20), now - timedelta(hours=10)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        # With sufficient samples and occupied intervals, should return correlation data
        assert isinstance(result, dict)
        assert "correlation_coefficient" in result
        assert "sample_count" in result

    def test_analyze_correlation_no_data(self, coordinator: AreaOccupancyCoordinator):
        """Test correlation analysis with no data."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        result = analyze_correlation(
            db, area_name, "sensor.nonexistent", analysis_period_days=30
        )
        assert result is not None
        assert result["analysis_error"] == "too_few_samples"

    def test_analyze_binary_correlation_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful binary correlation analysis."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "binary_sensor.door"
        now = dt_util.utcnow()

        # Create occupied intervals first - 50 intervals, each 1 hour long
        # These will be the "on" periods
        occupied_intervals = []
        for i in range(50):
            start = now - timedelta(hours=100 - (i * 2))
            end = start + timedelta(hours=1)
            occupied_intervals.append((start, end))
        _create_occupied_intervals_cache(db, area_name, occupied_intervals)

        # Create binary entity intervals - "on" during occupied periods, "off" otherwise
        # Create 100 intervals covering the same time period
        with db.get_session() as session:
            db.save_area_data(area_name)
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="door",
            )
            session.add(entity)

            for i in range(100):
                start = now - timedelta(hours=100 - i)
                end = start + timedelta(hours=1)
                # "on" during occupied periods (even indices: 0, 2, 4, ...)
                # "off" during unoccupied periods (odd indices: 1, 3, 5, ...)
                state = "on" if i % 2 == 0 else "off"
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    start_time=start,
                    end_time=end,
                    duration_seconds=3600,
                    state=state,
                )
                session.add(interval)
            session.commit()

        # Analyze with binary flag - use 5 days to match the 100 hours of data
        result = analyze_correlation(
            db,
            area_name,
            entity_id,
            analysis_period_days=5,
            is_binary=True,
            active_states=["on"],
        )

        assert isinstance(result, dict)
        assert result["correlation_coefficient"] > 0.9  # Should be perfect correlation
        assert result["sample_count"] > 0
        assert result["mean_value_when_occupied"] > 0.9  # Should be ~1.0
        assert result["mean_value_when_unoccupied"] < 0.1  # Should be ~0.0

    def test_analyze_binary_correlation_no_active_states(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary correlation without active states."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "binary_sensor.door"

        result = analyze_correlation(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
            is_binary=True,
            active_states=None,
        )

        assert result is None

    def test_analyze_correlation_no_occupied_intervals(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test analysis when no occupied intervals exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 100, lambda i: 20.0 + (i % 10)
        )

        result = analyze_correlation(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
            is_binary=False,
            active_states=None,
        )
        assert result is not None
        assert result["analysis_error"] == "no_occupied_samples"

    def test_analyze_correlation_insufficient_samples(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test analysis with insufficient samples."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 10, lambda i: 20.0
        )

        intervals = [(now - timedelta(hours=5), now - timedelta(hours=3))]
        _create_occupied_intervals_cache(db, area_name, intervals)

        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None
        assert result["analysis_error"] == "too_few_samples"

    def test_analyze_correlation_negative_correlation(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test analysis with negative correlation."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 100, lambda i: 30.0 - (i % 10)
        )

        intervals = [(now - timedelta(hours=50), now - timedelta(hours=40))]
        _create_occupied_intervals_cache(db, area_name, intervals)

        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        # With negative correlation pattern, may return None or dict with negative correlation
        if result is not None:
            assert isinstance(result, dict)
            # If correlation is found, verify it's a valid result
            assert "correlation_coefficient" in result


class TestSaveCorrelationResult:
    """Test save_correlation_result function."""

    def test_save_correlation_result_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving correlation result successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        correlation_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "input_type": InputType.TEMPERATURE.value,
            "correlation_coefficient": 0.75,
            "correlation_type": "strong_positive",
            "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
            "analysis_period_end": dt_util.utcnow(),
            "sample_count": 100,
            "confidence": 0.85,
            "mean_value_when_occupied": 22.5,
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_occupied": 1.5,
            "std_dev_when_unoccupied": 1.0,
            "threshold_active": 21.0,
            "threshold_inactive": 19.0,
            "calculation_date": dt_util.utcnow(),
        }

        result = save_correlation_result(db, correlation_data)
        assert result is True

        # Verify correlation was saved
        with db.get_session() as session:
            correlation = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .first()
            )
            assert correlation is not None
            assert correlation.correlation_coefficient == 0.75


class TestGetCorrelationForEntity:
    """Test get_correlation_for_entity function."""

    def test_get_correlation_for_entity_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test retrieving correlation for entity."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        correlation_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "input_type": InputType.TEMPERATURE.value,
            "correlation_coefficient": 0.8,
            "correlation_type": "strong_positive",
            "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
            "analysis_period_end": dt_util.utcnow(),
            "sample_count": 100,
        }
        save_correlation_result(db, correlation_data)

        result = get_correlation_for_entity(db, area_name, entity_id)
        assert result is not None
        assert result["correlation_coefficient"] == 0.8

    def test_get_correlation_for_entity_not_found(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test retrieving correlation when none exists."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        result = get_correlation_for_entity(db, area_name, "sensor.nonexistent")
        assert result is None

    def test_get_correlation_for_entity_multiple_results(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test getting correlation when multiple results exist."""
        db = coordinator.db
        db.init_db()
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Create multiple correlations
        with db.get_session() as session:
            # Create multiple correlations (most recent first due to desc ordering)
            now = dt_util.utcnow()
            for i in range(3):
                # Use different analysis_period_start to avoid unique constraint violation
                period_start = now - timedelta(days=30 + i)
                period_end = now - timedelta(days=i)
                correlation = db.Correlations(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    input_type=InputType.TEMPERATURE.value,
                    correlation_coefficient=0.5 + (i * 0.1),
                    correlation_type="strong_positive",
                    calculation_date=now - timedelta(days=i),
                    analysis_period_start=period_start,
                    analysis_period_end=period_end,
                    sample_count=100,
                )
                session.add(correlation)
            session.commit()

        # Should return most recent (i=0, coefficient=0.5)
        result = get_correlation_for_entity(db, area_name, entity_id)
        assert result is not None
        assert result["correlation_coefficient"] == 0.5  # Most recent (i=0)


class TestAnalyzeAndSaveCorrelation:
    """Test analyze_and_save_correlation function."""

    def test_analyze_and_save_correlation_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test analyzing and saving correlation in one call."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 100, lambda i: 20.0 + (i % 10)
        )

        intervals = [(now - timedelta(hours=50), now - timedelta(hours=40))]
        _create_occupied_intervals_cache(db, area_name, intervals)

        result = analyze_and_save_correlation(
            db, area_name, entity_id, analysis_period_days=30
        )
        # Should return correlation data (dict) if successful
        assert isinstance(result, dict)
        assert result["area_name"] == area_name
        assert result["entity_id"] == entity_id

    def test_analyze_and_save_correlation_no_data(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test analyze_and_save when no correlation data is generated."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        result = analyze_and_save_correlation(
            db, area_name, "sensor.nonexistent", analysis_period_days=30
        )
        # Should now return rejection result
        assert result is not None
        assert result["analysis_error"] == "too_few_samples"


class TestPruneOldCorrelations:
    """Test _prune_old_correlations function."""

    def test_prune_old_correlations_excess_records(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test pruning when there are more correlations than limit."""
        db = coordinator.db
        db.init_db()
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Create more correlations than CORRELATION_MONTHS_TO_KEEP
        # Spread them across multiple months to test pruning properly
        with db.get_session() as session:
            now = dt_util.utcnow()
            for i in range(CORRELATION_MONTHS_TO_KEEP + 5):
                # Create correlations spread across months (one per month)
                # Use i * 30 days to ensure they're in different months
                days_ago = i * 30
                period_start = now - timedelta(days=days_ago + 30)
                period_end = now - timedelta(days=days_ago)
                correlation = db.Correlations(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    input_type=InputType.TEMPERATURE.value,
                    correlation_coefficient=0.5 + (i * 0.01),
                    correlation_type="strong_positive",
                    calculation_date=now - timedelta(days=days_ago),
                    analysis_period_start=period_start,
                    analysis_period_end=period_end,
                    sample_count=100,
                )
                session.add(correlation)
            session.commit()

        # Call prune function
        with db.get_session() as session:
            _prune_old_correlations(db, session, area_name, entity_id)

        # Verify only CORRELATION_MONTHS_TO_KEEP remain (one per month)
        with db.get_session() as session:
            count = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .count()
            )
            assert count == CORRELATION_MONTHS_TO_KEEP

    def test_prune_old_correlations_no_excess(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test pruning when there are fewer correlations than limit."""
        db = coordinator.db
        db.init_db()
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Create fewer correlations than limit
        # Spread them across months to test pruning properly
        with db.get_session() as session:
            now = dt_util.utcnow()
            num_correlations = CORRELATION_MONTHS_TO_KEEP - 2
            for i in range(num_correlations):
                # Create correlations spread across months (one per month)
                # Use month boundaries to ensure they're in different months
                # Start from current month and go back i months
                target_date = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                for _ in range(i):
                    # Go back one month
                    if target_date.month == 1:
                        target_date = target_date.replace(
                            year=target_date.year - 1, month=12
                        )
                    else:
                        target_date = target_date.replace(month=target_date.month - 1)

                # Create period that starts at the beginning of the month
                period_start = target_date
                # Period ends at the beginning of next month
                if period_start.month == 12:
                    period_end = period_start.replace(
                        year=period_start.year + 1, month=1
                    )
                else:
                    period_end = period_start.replace(month=period_start.month + 1)

                correlation = db.Correlations(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    input_type=InputType.TEMPERATURE.value,
                    correlation_coefficient=0.5,
                    correlation_type="strong_positive",
                    calculation_date=period_end,
                    analysis_period_start=period_start,
                    analysis_period_end=period_end,
                    sample_count=100,
                )
                session.add(correlation)
            session.commit()

        # Call prune function
        with db.get_session() as session:
            _prune_old_correlations(db, session, area_name, entity_id)

        # Verify all correlations remain
        with db.get_session() as session:
            count = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .count()
            )
            assert count == num_correlations


class TestAnalyzeBinaryLikelihoods:
    """Test analyze_binary_likelihoods function."""

    def test_analyze_binary_likelihoods_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful binary likelihood analysis calculates probabilities correctly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"
        now = dt_util.utcnow()

        db.save_area_data(area_name)

        # Create motion sensor intervals (5 intervals is sufficient for testing)
        # Use a time slightly in the past to ensure intervals are definitely found by queries
        # that use dt_util.utcnow() which may be slightly after the 'now' used here
        base_time = now - timedelta(seconds=30)
        motion_entity_id = db.coordinator.get_area(area_name).config.sensors.motion[0]
        motion_intervals = _create_motion_intervals(
            db, area_name, motion_entity_id, 5, now=base_time
        )
        db.save_occupied_intervals_cache(area_name, motion_intervals, "motion_sensors")

        # Create light intervals (active during first 3 occupied periods)
        # Use the same base_time for consistency
        with db.get_session() as session:
            for i in range(5):
                start = base_time - timedelta(hours=5 - i)
                end = start + timedelta(hours=1)
                state = "on" if i < 3 else "off"
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    start_time=start,
                    end_time=end,
                    state=state,
                    duration_seconds=3600,
                )
                session.add(interval)
            session.commit()

        # Analyze binary likelihoods - use 1 day to match the 5 hours of data
        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=1,
            active_states=["on"],
        )

        assert result is not None
        assert result["prob_given_true"] is not None
        assert result["prob_given_false"] is not None
        assert result["analysis_error"] is None
        # Light should be more likely on when occupied
        assert result["prob_given_true"] > result["prob_given_false"]
        # Probabilities should be clamped
        assert 0.05 <= result["prob_given_true"] <= 0.95
        assert 0.05 <= result["prob_given_false"] <= 0.95

    def test_analyze_binary_likelihoods_no_active_states(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary likelihood analysis without active states."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"

        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
            active_states=None,
        )

        assert result is None

    def test_analyze_binary_likelihoods_no_occupied_intervals(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary likelihood analysis with no occupied intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"

        _create_binary_entity_with_intervals(
            db, area_name, entity_id, 5, lambda i: "on"
        )

        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
            active_states=["on"],
        )

        assert result is not None
        assert result["prob_given_true"] is None
        assert result["prob_given_false"] is None
        assert result["analysis_error"] == "no_occupied_intervals"

    def test_analyze_binary_likelihoods_no_sensor_data(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary likelihood analysis returns error when sensor has no intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"

        db.save_area_data(area_name)

        # Create motion intervals but no light intervals
        # Create intervals explicitly within the analysis period to ensure overlap
        now = dt_util.utcnow()
        motion_entity_id = db.coordinator.get_area(area_name).config.sensors.motion[0]

        # Create 3 motion intervals in the last 3 hours (definitely within 1-day period)
        motion_intervals = []
        with db.get_session() as session:
            for i in range(3):
                start = now - timedelta(hours=3 - i)
                end = start + timedelta(hours=1)
                motion_intervals.append((start, end))
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=motion_entity_id,
                    start_time=start,
                    end_time=end,
                    state="on",
                    duration_seconds=3600,
                )
                session.add(interval)
            session.commit()

        db.save_occupied_intervals_cache(area_name, motion_intervals, "motion_sensors")

        # Use 1 day analysis period - motion intervals are in last 3 hours, so they overlap
        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=1,
            active_states=["on"],
        )

        assert result is not None
        assert result["prob_given_true"] is None
        assert result["prob_given_false"] is None
        # Function checks for occupied time first, then sensor data
        # If motion intervals don't overlap with analysis period, we get "no_occupied_time"
        # Otherwise, we get "no_sensor_data" (no light intervals)
        assert result["analysis_error"] in ("no_sensor_data", "no_occupied_time")


# ruff: noqa: SLF001
@pytest.fixture
def mock_numeric_entity():
    """Create a mock numeric entity for testing."""
    entity_type = EntityType(
        input_type=InputType.TEMPERATURE,
        weight=0.1,
        prob_given_true=0.5,
        prob_given_false=0.5,
        active_range=None,
    )
    decay = Decay(half_life=60.0)

    return Entity(
        entity_id="sensor.temp",
        type=entity_type,
        prob_given_true=0.5,
        prob_given_false=0.5,
        decay=decay,
        state_provider=lambda x: "20.0",
        last_updated=dt_util.utcnow(),
    )


@pytest.fixture
def mock_binary_entity():
    """Create a mock binary entity for testing."""
    entity_type = EntityType(
        input_type=InputType.MEDIA,
        weight=0.7,
        prob_given_true=0.5,
        prob_given_false=0.5,
        active_states=[STATE_ON],
    )
    decay = Decay(half_life=60.0)

    return Entity(
        entity_id="media_player.tv",
        type=entity_type,
        prob_given_true=0.5,
        prob_given_false=0.5,
        decay=decay,
        state_provider=lambda x: STATE_ON,
        last_updated=dt_util.utcnow(),
    )


class TestGaussianLikelihood:
    """Test Gaussian likelihood calculation."""

    def test_is_continuous_likelihood_property(self, mock_numeric_entity):
        """Test is_continuous_likelihood property."""
        # Initially false
        assert not mock_numeric_entity.is_continuous_likelihood

        # Set gaussian params
        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        # Now true
        assert mock_numeric_entity.is_continuous_likelihood

    @pytest.mark.parametrize(
        ("value", "mean", "std", "expected_density", "tolerance"),
        [
            # Peak density (at mean): 1 / (sqrt(2*pi) * 1) ≈ 0.3989
            (20.0, 20.0, 1.0, 0.3989, 0.0001),
            # 1 std dev away: 0.3989 * exp(-0.5 * 1^2) ≈ 0.2420
            (21.0, 20.0, 1.0, 0.2420, 0.0001),
            # 2 std dev away: 0.3989 * exp(-0.5 * 2^2) ≈ 0.0540
            (22.0, 20.0, 1.0, 0.0540, 0.0001),
            # Small std dev (higher peak): 1 / (sqrt(2*pi) * 0.1) ≈ 3.989
            (20.0, 20.0, 0.1, 3.989, 0.001),
        ],
    )
    def test_calculate_gaussian_density(
        self, mock_numeric_entity, value, mean, std, expected_density, tolerance
    ):
        """Test _calculate_gaussian_density method."""
        density = mock_numeric_entity._calculate_gaussian_density(value, mean, std)
        assert abs(density - expected_density) < tolerance

    @pytest.mark.parametrize(
        ("value", "expected_p_t", "expected_p_f", "comparison"),
        [
            # Value = 22 (Occupied Mean): P(val|Occ) peak (~0.3989), P(val|Unocc) 2 std (~0.054)
            ("22.0", 0.3989, 0.0540, "gt"),
            # Value = 20 (Unoccupied Mean): P(val|Occ) 2 std (~0.054), P(val|Unocc) peak (~0.3989)
            ("20.0", 0.0540, 0.3989, "lt"),
            # Value = 21 (Middle): equal distance from both means, densities equal (~0.2420)
            ("21.0", 0.2420, 0.2420, "eq"),
        ],
    )
    def test_get_likelihoods_continuous_numeric(
        self, mock_numeric_entity, value, expected_p_t, expected_p_f, comparison
    ):
        """Test get_likelihoods with continuous parameters for numeric sensor."""
        # Setup: Occupied Mean 22, Std 1; Unoccupied Mean 20, Std 1
        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        mock_numeric_entity.state_provider = lambda x: value
        p_t, p_f = mock_numeric_entity.get_likelihoods()

        assert abs(p_t - expected_p_t) < 0.001
        assert abs(p_f - expected_p_f) < 0.001

        if comparison == "gt":
            assert p_t > p_f  # Favors occupied
        elif comparison == "lt":
            assert p_t < p_f  # Favors unoccupied
        else:  # eq
            assert abs(p_t - p_f) < 0.0001  # Equal densities

    @pytest.mark.parametrize(
        ("state", "analysis_error", "expected_p_t", "expected_p_f"),
        [
            # Analyzed successfully - should return learned probabilities regardless of state
            (STATE_ON, None, 0.8, 0.1),
            (STATE_OFF, None, 0.8, 0.1),
            # Not analyzed - should use EntityType defaults
            (STATE_ON, "not_analyzed", 0.5, 0.5),
        ],
    )
    def test_get_likelihoods_binary_sensor_static(
        self, mock_binary_entity, state, analysis_error, expected_p_t, expected_p_f
    ):
        """Test get_likelihoods for binary sensor using static probabilities."""
        # Binary sensors use static probabilities, not Gaussian PDF
        mock_binary_entity.prob_given_true = 0.8
        mock_binary_entity.prob_given_false = 0.1
        mock_binary_entity.analysis_error = analysis_error
        mock_binary_entity.learned_gaussian_params = None

        mock_binary_entity.state_provider = lambda x: state
        p_t, p_f = mock_binary_entity.get_likelihoods()

        if analysis_error is None:
            # Should return learned probabilities
            assert p_t == expected_p_t
            assert p_f == expected_p_f
        else:
            # Should use EntityType defaults
            assert p_t == mock_binary_entity.type.prob_given_true
            assert p_f == mock_binary_entity.type.prob_given_false

    def test_get_likelihoods_fallback(self, mock_numeric_entity):
        """Test get_likelihoods fallback behavior uses EntityType defaults."""
        # No params -> returns EntityType defaults (not stored prob_given_true/false)
        mock_numeric_entity.learned_gaussian_params = None
        # Change stored values to verify we use EntityType defaults
        mock_numeric_entity.prob_given_true = 0.9
        mock_numeric_entity.prob_given_false = 0.1
        p_t, p_f = mock_numeric_entity.get_likelihoods()
        # Should use EntityType defaults (0.5, 0.5), not stored values
        assert p_t == 0.5
        assert p_f == 0.5

        # With params but invalid state -> uses representative value (average of means)
        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }
        mock_numeric_entity.state_provider = lambda x: "unavailable"
        p_t, p_f = mock_numeric_entity.get_likelihoods()
        # Should use representative value (average of means = 21.0) to calculate probabilities
        # This will give non-zero probabilities based on Gaussian PDF
        assert p_t > 0.0
        assert p_f > 0.0
        assert p_t != 0.5  # Should be calculated, not default
        assert p_f != 0.5  # Should be calculated, not default

    def test_update_correlation_populates_params(self, mock_numeric_entity):
        """Test update_correlation populates Gaussian params."""
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "strong_positive",
            "mean_value_when_occupied": 22.0,
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_occupied": 1.5,
            "std_dev_when_unoccupied": 1.2,
        }

        mock_numeric_entity.update_correlation(correlation_data)

        assert mock_numeric_entity.learned_gaussian_params is not None
        assert mock_numeric_entity.learned_gaussian_params["mean_occupied"] == 22.0
        assert mock_numeric_entity.learned_gaussian_params["std_occupied"] == 1.5
        assert mock_numeric_entity.learned_gaussian_params["mean_unoccupied"] == 20.0
        assert mock_numeric_entity.learned_gaussian_params["std_unoccupied"] == 1.2

        # Should also populate learned_active_range for UI
        assert mock_numeric_entity.learned_active_range is not None

    def test_update_correlation_missing_params(self, mock_numeric_entity):
        """Test update_correlation handles missing occupied stats."""
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "strong_positive",
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_unoccupied": 1.2,
            # Missing occupied stats
        }

        mock_numeric_entity.update_correlation(correlation_data)

        # Should NOT populate gaussian params
        assert mock_numeric_entity.learned_gaussian_params is None
        # Should still populate active range (open-ended)
        assert mock_numeric_entity.learned_active_range is not None
        assert mock_numeric_entity.learned_active_range[1] == float("inf")
        # Should NOT update stored prob_given_true/false (no fallback)

    def test_get_likelihoods_nan_mean_occupied(self, mock_numeric_entity):
        """Test get_likelihoods with NaN mean_occupied falls back to EntityType defaults."""

        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": float("nan"),
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should fallback to EntityType defaults
        assert p_t == mock_numeric_entity.type.prob_given_true
        assert p_f == mock_numeric_entity.type.prob_given_false

    def test_get_likelihoods_nan_mean_unoccupied(self, mock_numeric_entity):
        """Test get_likelihoods with NaN mean_unoccupied falls back to EntityType defaults."""

        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": float("nan"),
            "std_unoccupied": 1.0,
        }

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should fallback to EntityType defaults
        assert p_t == mock_numeric_entity.type.prob_given_true
        assert p_f == mock_numeric_entity.type.prob_given_false

    def test_get_likelihoods_inf_mean_occupied(self, mock_numeric_entity):
        """Test get_likelihoods with inf mean_occupied falls back to EntityType defaults."""

        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": float("inf"),
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should fallback to EntityType defaults
        assert p_t == mock_numeric_entity.type.prob_given_true
        assert p_f == mock_numeric_entity.type.prob_given_false

    def test_get_likelihoods_inf_mean_unoccupied(self, mock_numeric_entity):
        """Test get_likelihoods with inf mean_unoccupied falls back to EntityType defaults."""

        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": float("inf"),
            "std_unoccupied": 1.0,
        }

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should fallback to EntityType defaults
        assert p_t == mock_numeric_entity.type.prob_given_true
        assert p_f == mock_numeric_entity.type.prob_given_false

    def test_get_likelihoods_nan_state_value(self, mock_numeric_entity):
        """Test get_likelihoods with NaN state value uses mean of means."""

        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        # Set state to NaN
        mock_numeric_entity.state_provider = lambda x: float("nan")

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should use mean of means (21.0) and calculate valid densities
        assert not math.isnan(p_t)
        assert not math.isnan(p_f)
        assert not math.isinf(p_t)
        assert not math.isinf(p_f)
        assert p_t > 0.0
        assert p_f > 0.0

    def test_get_likelihoods_inf_state_value(self, mock_numeric_entity):
        """Test get_likelihoods with inf state value uses mean of means."""

        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        # Set state to inf
        mock_numeric_entity.state_provider = lambda x: float("inf")

        p_t, p_f = mock_numeric_entity.get_likelihoods()

        # Should use mean of means (21.0) and calculate valid densities
        assert not math.isnan(p_t)
        assert not math.isnan(p_f)
        assert not math.isinf(p_t)
        assert not math.isinf(p_f)
        assert p_t > 0.0
        assert p_f > 0.0

    def test_get_likelihoods_motion_sensor_uses_configured_values(self):
        """Test that motion sensors always use configured prob_given_true/false."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.85,
            prob_given_true=0.95,  # EntityType default
            prob_given_false=0.02,  # EntityType default
            active_states=[STATE_ON],
        )
        decay = Decay(half_life=60.0)

        # Create motion sensor with different configured values
        motion_entity = Entity(
            entity_id="binary_sensor.motion",
            type=entity_type,
            prob_given_true=0.9,  # Configured value (different from EntityType)
            prob_given_false=0.05,  # Configured value (different from EntityType)
            decay=decay,
            state_provider=lambda x: STATE_ON,
            last_updated=dt_util.utcnow(),
        )

        # Even with Gaussian params, motion sensors should use configured values
        motion_entity.learned_gaussian_params = {
            "mean_occupied": 0.9,
            "std_occupied": 0.3,
            "mean_unoccupied": 0.1,
            "std_unoccupied": 0.3,
        }

        p_t, p_f = motion_entity.get_likelihoods()
        # Should use configured values, not Gaussian params or EntityType defaults
        assert p_t == 0.9
        assert p_f == 0.05


class TestCorrelationBugFixes:
    """Test fixes for correlation logic bugs."""

    def test_timezone_aware_numeric_query(self, coordinator: AreaOccupancyCoordinator):
        """Test that numeric sensor query uses timezone-aware datetimes."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        # Create samples with timezone-aware timestamps
        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 50, lambda i: 20.0 + (i % 10)
        )

        # Create occupied intervals
        intervals = [
            (now - timedelta(hours=30), now - timedelta(hours=20)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Test with timezone-aware period (should work correctly)
        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None
        assert "correlation_coefficient" in result
        assert result["sample_count"] > 0

    def test_entry_id_filtering(self, coordinator: AreaOccupancyCoordinator):
        """Test that numeric samples query filters by entry_id to prevent cross-entry leakage."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        # Create samples for the current entry
        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 50, lambda i: 20.0 + (i % 10)
        )

        # Create samples for a different entry_id (simulating another config entry)
        with db.get_session() as session:
            different_entry_id = "different_entry_id"
            for i in range(20):
                sample = db.NumericSamples(
                    entry_id=different_entry_id,  # Different entry_id
                    area_name=area_name,  # Same area_name
                    entity_id=entity_id,  # Same entity_id
                    timestamp=now - timedelta(hours=20 - i),
                    value=100.0 + i,  # Different values to make detection easier
                )
                session.add(sample)
            session.commit()

        # Create occupied intervals
        intervals = [
            (now - timedelta(hours=30), now - timedelta(hours=20)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Analyze correlation - should only use samples from current entry
        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None
        # Should have 50 samples (from current entry), not 70 (50 + 20)
        assert result["sample_count"] == 50

        # Verify the values are from the correct entry (should be around 20-30, not 100+)
        # Get the actual samples used
        with db.get_session() as session:
            samples = (
                session.query(db.NumericSamples)
                .filter(
                    db.NumericSamples.entry_id == db.coordinator.entry_id,
                    db.NumericSamples.area_name == area_name,
                    db.NumericSamples.entity_id == entity_id,
                )
                .all()
            )
            # All samples should be from the current entry
            assert all(sample.entry_id == db.coordinator.entry_id for sample in samples)
            # Values should be in the expected range (20-30 from our generator)
            assert all(20.0 <= sample.value <= 30.0 for sample in samples)

    def test_unoccupied_overlap_validation(self, coordinator: AreaOccupancyCoordinator):
        """Test that unoccupied_overlap calculation handles edge cases correctly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"
        now = dt_util.utcnow()

        db.save_area_data(area_name)

        # Create motion sensor intervals (occupied periods)
        base_time = now - timedelta(seconds=30)
        motion_entity_id = db.coordinator.get_area(area_name).config.sensors.motion[0]
        motion_intervals = _create_motion_intervals(
            db, area_name, motion_entity_id, 5, now=base_time
        )
        db.save_occupied_intervals_cache(area_name, motion_intervals, "motion_sensors")

        # Create light intervals that span both occupied and unoccupied periods
        # This tests the edge case where an interval partially overlaps with occupied periods
        with db.get_session() as session:
            # Create an interval that starts before occupied period and ends during it
            # This will test the unoccupied_overlap calculation
            interval_start = base_time - timedelta(hours=6)
            interval_end = base_time - timedelta(hours=4) + timedelta(minutes=30)
            # This interval spans: unoccupied -> occupied -> unoccupied
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=entity_id,
                start_time=interval_start,
                end_time=interval_end,
                state="on",
                duration_seconds=(interval_end - interval_start).total_seconds(),
            )
            session.add(interval)
            session.commit()

        # Analyze binary likelihoods - should handle the edge case correctly
        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=1,
            active_states=["on"],
        )

        assert result is not None
        assert result["prob_given_true"] is not None
        assert result["prob_given_false"] is not None
        # Probabilities should be valid (clamped between 0.05 and 0.95)
        assert 0.05 <= result["prob_given_true"] <= 0.95
        assert 0.05 <= result["prob_given_false"] <= 0.95
        # Unoccupied overlap should be non-negative and valid
        # (validated internally by the max/min clamping we added)
