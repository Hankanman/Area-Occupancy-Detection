"""Tests for database correlation analysis functions."""

from datetime import datetime, timedelta

import pytest

from custom_components.area_occupancy.const import (
    AGGREGATION_PERIOD_HOURLY,
    CORRELATION_MONTHS_TO_KEEP,
    RETENTION_RAW_NUMERIC_SAMPLES_DAYS,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.db.correlation import (
    _map_binary_state_to_semantic,
    _prune_old_correlations,
    analyze_and_save_correlation,
    analyze_binary_likelihoods,
    analyze_correlation,
    calculate_pearson_correlation,
    convert_hourly_aggregates_to_samples,
    convert_intervals_to_samples,
    get_correlatable_entities_by_area,
    get_correlation_for_entity,
    run_correlation_analysis,
    save_binary_likelihood_result,
    save_correlation_result,
)
from homeassistant.const import STATE_ON
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
        ("x_values", "y_values", "expected_corr", "corr_tolerance", "description"),
        [
            (
                list(range(1, 51)),
                [x * 2 for x in range(1, 51)],
                1.0,
                0.01,
                "positive correlation",
            ),
            (
                list(range(1, 51)),
                [101 - x for x in range(1, 51)],
                -1.0,
                0.01,
                "negative correlation",
            ),
            (
                [1, 2, 3, 4, 5],
                [5, 2, 8, 1, 9],
                None,  # No specific expected value, just check range
                None,
                "no correlation",
            ),
        ],
    )
    def test_calculate_pearson_correlation(
        self, x_values, y_values, expected_corr, corr_tolerance, description
    ):
        """Test correlation calculation with various correlation types."""
        correlation, p_value = calculate_pearson_correlation(x_values, y_values)

        if expected_corr is not None:
            # Verify correlation matches expected value within tolerance
            assert abs(correlation - expected_corr) < corr_tolerance
        else:
            # For no correlation, just verify it's in valid range
            assert -1.0 <= correlation <= 1.0

        # P-value should always be in valid range
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
        # Verify correlation coefficient is in valid range
        assert -1.0 <= result["correlation_coefficient"] <= 1.0
        # Verify sample_count matches input data
        assert result["sample_count"] == 100
        # Verify mean/std values are reasonable
        assert result["mean_value_when_occupied"] is not None
        assert result["mean_value_when_unoccupied"] is not None
        assert result["std_dev_when_occupied"] is not None
        assert result["std_dev_when_unoccupied"] is not None
        # Verify mean values are reasonable (should be around 20-30 based on generator)
        assert 15.0 <= result["mean_value_when_occupied"] <= 35.0
        assert 15.0 <= result["mean_value_when_unoccupied"] <= 35.0

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
        """Test analysis with negative correlation.

        Creates data where values are LOW during occupied periods and HIGH during
        unoccupied periods, resulting in negative correlation.
        """
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        # Create occupied interval: hours 50-40 ago (samples 50-59)
        # For negative correlation: LOW values during occupied, HIGH values when unoccupied
        def value_generator(i):
            # Samples are created with timestamp: now - timedelta(hours=100-i)
            # Occupied period is hours 50-40 ago, which corresponds to i=50-59
            if 50 <= i <= 59:
                return 15.0  # LOW value during occupied period
            return 25.0  # HIGH value during unoccupied period

        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 100, value_generator
        )

        intervals = [(now - timedelta(hours=50), now - timedelta(hours=40))]
        _create_occupied_intervals_cache(db, area_name, intervals)

        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        # With negative correlation pattern, should return dict with negative correlation
        assert result is not None
        assert isinstance(result, dict)
        assert "correlation_coefficient" in result
        # Verify negative correlation was detected
        assert result["correlation_coefficient"] < 0
        # Verify correlation_type reflects negative correlation
        assert result["correlation_type"] in ("negative", "strong_negative")
        # Verify sample_count matches input
        assert result["sample_count"] == 100


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

        # Verify data was actually saved to database
        with db.get_session() as session:
            correlation = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .first()
            )
            assert correlation is not None
            assert (
                correlation.correlation_coefficient == result["correlation_coefficient"]
            )
            assert correlation.sample_count == result["sample_count"]
            assert (
                correlation.mean_value_when_occupied
                == result["mean_value_when_occupied"]
            )
            assert (
                correlation.mean_value_when_unoccupied
                == result["mean_value_when_unoccupied"]
            )

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

    def test_analyze_binary_likelihoods_no_active_intervals(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary likelihood analysis when sensor has intervals but none are active."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "media_player.test_tv"
        now = dt_util.utcnow()

        db.save_area_data(area_name)

        # Create motion sensor intervals
        base_time = now - timedelta(seconds=30)
        motion_entity_id = db.coordinator.get_area(area_name).config.sensors.motion[0]
        motion_intervals = _create_motion_intervals(
            db, area_name, motion_entity_id, 5, now=base_time
        )
        db.save_occupied_intervals_cache(area_name, motion_intervals, "motion_sensors")

        # Create media player intervals with states that don't match active_states
        # Media player uses "playing"/"paused" as active, but we'll create "off"/"idle" intervals
        with db.get_session() as session:
            for i in range(5):
                start = base_time - timedelta(hours=5 - i)
                end = start + timedelta(hours=1)
                # Use states that don't match active_states ["playing", "paused"]
                state = "off" if i % 2 == 0 else "idle"
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

        # Analyze binary likelihoods with active_states that don't match any intervals
        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=1,
            active_states=["playing", "paused"],
        )

        assert result is not None
        assert result["prob_given_true"] is None
        assert result["prob_given_false"] is None
        assert result["analysis_error"] == "no_active_intervals"

    def test_analyze_binary_likelihoods_active_but_not_during_occupied(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary likelihood analysis when sensor is active but never during occupied periods."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"
        now = dt_util.utcnow()

        db.save_area_data(area_name)

        # Create motion sensor intervals (occupied periods)
        base_time = now - timedelta(seconds=30)
        motion_entity_id = db.coordinator.get_area(area_name).config.sensors.motion[0]
        # Create 3 motion intervals in hours 1, 2, 3
        motion_intervals = []
        with db.get_session() as session:
            for i in range(3):
                start = base_time - timedelta(hours=3 - i)
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

        # Create light intervals that are active but NOT during occupied periods
        # Light is on during hours 4, 5, 6 (after occupied periods)
        with db.get_session() as session:
            for i in range(3):
                start = base_time - timedelta(hours=6 - i)
                end = start + timedelta(hours=1)
                # Light is on, but this is after the occupied periods
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    start_time=start,
                    end_time=end,
                    state="on",
                    duration_seconds=3600,
                )
                session.add(interval)
            session.commit()

        # Analyze binary likelihoods
        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=1,
            active_states=["on"],
        )

        assert result is not None
        # When sensor was active but never during occupied periods, return error
        # so entity can use type defaults instead of clamped 0.05
        assert result["prob_given_true"] is None
        assert result["prob_given_false"] is None
        assert result["analysis_error"] == "no_active_during_occupied"

    @pytest.mark.parametrize(
        (
            "entity_id",
            "states",
            "active_states",
            "expected_active_count",
            "description",
        ),
        [
            (
                "binary_sensor.door_contact",
                [
                    "off",
                    "off",
                    "on",
                ],  # Door: 'off'=closed (active), 'on'=open (inactive)
                ["closed"],  # Semantic state
                2,  # Closed during 2 of 3 occupied periods
                "door sensors (off/on → closed/open)",
            ),
            (
                "binary_sensor.window_contact",
                [
                    "on",
                    "on",
                    "off",
                ],  # Window: 'on'=open (active), 'off'=closed (inactive)
                ["open"],  # Semantic state
                2,  # Open during 2 of 3 occupied periods
                "window sensors (off/on → closed/open)",
            ),
        ],
    )
    def test_analyze_binary_likelihoods_state_mapping(
        self,
        coordinator: AreaOccupancyCoordinator,
        entity_id,
        states,
        active_states,
        expected_active_count,
        description,
    ):
        """Test binary likelihood analysis with state mapping for door/window sensors."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        now = dt_util.utcnow()

        db.save_area_data(area_name)

        # Create motion sensor intervals (occupied periods)
        base_time = now - timedelta(seconds=30)
        motion_entity_id = db.coordinator.get_area(area_name).config.sensors.motion[0]
        motion_intervals = _create_motion_intervals(
            db, area_name, motion_entity_id, 3, now=base_time
        )
        db.save_occupied_intervals_cache(area_name, motion_intervals, "motion_sensors")

        # Create intervals using binary states but config expects semantic states
        with db.get_session() as session:
            for i, state in enumerate(states):
                start = base_time - timedelta(hours=3 - i)
                end = start + timedelta(hours=1)
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

        # Analyze with semantic states - should map binary states to semantic states
        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=1,
            active_states=active_states,
        )

        assert result is not None
        assert result["prob_given_true"] is not None
        assert result["prob_given_false"] is not None
        assert result["analysis_error"] is None
        # Entity was active during expected_active_count of 3 occupied periods
        assert result["prob_given_true"] > 0.0


class TestCorrelationBugFixes:
    """Test fixes for correlation logic bugs."""

    def test_timezone_aware_numeric_query(self, coordinator: AreaOccupancyCoordinator):
        """Test that numeric sensor query uses timezone-aware datetimes correctly."""
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

        # Verify timezone-aware datetimes are handled correctly
        # The function should handle both naive and aware datetimes via ensure_timezone_aware
        # Verify samples were found (proves timezone handling worked)
        assert result["sample_count"] == 50

        # Verify the analysis period boundaries are timezone-aware
        # This is tested implicitly by the successful query, but we can verify
        # that the result contains valid timestamps
        assert result["analysis_period_start"] is not None
        assert result["analysis_period_end"] is not None
        # Verify these are datetime objects (timezone-aware)
        assert isinstance(result["analysis_period_start"], datetime)
        assert isinstance(result["analysis_period_end"], datetime)

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

        # Verify overlap calculations are correct:
        # The interval spans from base_time - 6h to base_time - 4h + 30m (2.5 hours total)
        # Motion intervals are at base_time - 5h to base_time - 4h (1 hour)
        # So occupied overlap should be ~1 hour, unoccupied overlap should be ~1.5 hours
        # This means prob_given_true should be > prob_given_false (more time active during occupied)
        # But since the interval spans both occupied and unoccupied, both probabilities should be > 0
        assert result["prob_given_true"] > 0.0
        assert result["prob_given_false"] > 0.0
        # The interval overlaps with occupied period, so prob_given_true should be meaningful
        # (not just clamped minimum)
        assert result["prob_given_true"] >= 0.05


class TestNumericAggregatesInCorrelation:
    """Test correlation analysis using numeric aggregates for historical data."""

    def test_correlation_uses_recent_samples_only(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that correlation uses only raw samples when all data is within retention."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        # Create samples within retention period (last 3 days)
        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 50, lambda i: 20.0 + (i % 10)
        )

        # Create occupied intervals
        intervals = [
            (now - timedelta(days=2), now - timedelta(days=1)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Verify no aggregates exist in retention period
        with db.get_session() as session:
            aggregate_count = (
                session.query(db.NumericAggregates)
                .filter(
                    db.NumericAggregates.entry_id == db.coordinator.entry_id,
                    db.NumericAggregates.area_name == area_name,
                    db.NumericAggregates.entity_id == entity_id,
                    db.NumericAggregates.period_start
                    >= now - timedelta(days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS),
                )
                .count()
            )
            assert aggregate_count == 0

        # Analyze correlation - should use only raw samples
        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None
        assert result["sample_count"] == 50

        # Verify all samples are from NumericSamples table (not aggregates)
        with db.get_session() as session:
            sample_count = (
                session.query(db.NumericSamples)
                .filter(
                    db.NumericSamples.entry_id == db.coordinator.entry_id,
                    db.NumericSamples.area_name == area_name,
                    db.NumericSamples.entity_id == entity_id,
                )
                .count()
            )
            assert sample_count == 50

    def test_correlation_uses_aggregates_only(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that correlation uses only aggregates when all data is older than retention."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        old_date = dt_util.utcnow() - timedelta(
            days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS + 10
        )

        # Ensure area and entity exist
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=entity_id,
                entity_type="temperature",
            )
            session.add(entity)

            # Create hourly aggregates (older than retention)
            for i in range(24):  # 24 hours of aggregates
                hour_start = old_date + timedelta(hours=i)
                aggregate = db.NumericAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    aggregation_period=AGGREGATION_PERIOD_HOURLY,
                    period_start=hour_start,
                    period_end=hour_start + timedelta(hours=1),
                    min_value=20.0,
                    max_value=25.0,
                    avg_value=22.5,
                    median_value=22.5,
                    sample_count=10,
                    first_value=20.0,
                    last_value=25.0,
                    std_deviation=1.5,
                )
                session.add(aggregate)
            session.commit()

        # Create occupied intervals
        intervals = [
            (old_date, old_date + timedelta(hours=12)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Verify no raw samples exist in old period
        with db.get_session() as session:
            raw_sample_count = (
                session.query(db.NumericSamples)
                .filter(
                    db.NumericSamples.entry_id == db.coordinator.entry_id,
                    db.NumericSamples.area_name == area_name,
                    db.NumericSamples.entity_id == entity_id,
                    db.NumericSamples.timestamp
                    < dt_util.utcnow()
                    - timedelta(days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS),
                )
                .count()
            )
            assert raw_sample_count == 0

        # Analyze correlation - should use only aggregates
        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None
        assert result["sample_count"] == 24  # 24 hourly aggregates

        # Verify all samples are from NumericAggregates table (not raw samples)
        with db.get_session() as session:
            aggregate_count = (
                session.query(db.NumericAggregates)
                .filter(
                    db.NumericAggregates.entry_id == db.coordinator.entry_id,
                    db.NumericAggregates.area_name == area_name,
                    db.NumericAggregates.entity_id == entity_id,
                )
                .count()
            )
            assert aggregate_count == 24

    def test_correlation_combines_both_sources(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that correlation combines raw samples and aggregates when period spans both."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()
        old_date = now - timedelta(days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS + 5)

        # Ensure area and entity exist
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=entity_id,
                entity_type="temperature",
            )
            session.add(entity)

            # Create recent raw samples (within retention)
            for i in range(20):
                sample = db.NumericSamples(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    timestamp=now - timedelta(hours=20 - i),
                    value=20.0 + (i % 10),
                )
                session.add(sample)

            # Create old hourly aggregates (older than retention)
            for i in range(24):  # 24 hours of aggregates
                hour_start = old_date + timedelta(hours=i)
                aggregate = db.NumericAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    aggregation_period=AGGREGATION_PERIOD_HOURLY,
                    period_start=hour_start,
                    period_end=hour_start + timedelta(hours=1),
                    min_value=20.0,
                    max_value=25.0,
                    avg_value=22.5,
                    median_value=22.5,
                    sample_count=10,
                    first_value=20.0,
                    last_value=25.0,
                    std_deviation=1.5,
                )
                session.add(aggregate)
            session.commit()

        # Create occupied intervals spanning both periods
        intervals = [
            (old_date, old_date + timedelta(hours=12)),
            (now - timedelta(hours=10), now - timedelta(hours=5)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Analyze correlation - should combine both sources
        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None
        # Should have both recent samples (20) and aggregates (24)
        assert result["sample_count"] == 44

    def test_convert_hourly_aggregates_to_samples(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test conversion of hourly aggregates to sample objects."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()
        period_start = now - timedelta(days=10)
        period_end = now - timedelta(days=5)

        # Ensure area and entity exist
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=entity_id,
                entity_type="temperature",
            )
            session.add(entity)

            # Create hourly aggregates
            for i in range(24):  # 24 hours of aggregates
                hour_start = period_start + timedelta(hours=i)
                aggregate = db.NumericAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    aggregation_period=AGGREGATION_PERIOD_HOURLY,
                    period_start=hour_start,
                    period_end=hour_start + timedelta(hours=1),
                    min_value=20.0,
                    max_value=25.0,
                    avg_value=22.5 + i,  # Varying avg_value
                    median_value=22.5,
                    sample_count=10,
                    first_value=20.0,
                    last_value=25.0,
                    std_deviation=1.5,
                )
                session.add(aggregate)
            session.commit()

        # Convert aggregates to samples
        with db.get_session() as session:
            samples = convert_hourly_aggregates_to_samples(
                db, area_name, entity_id, period_start, period_end, session
            )

        # Verify conversion
        assert len(samples) == 24
        # Verify samples are sorted by timestamp
        timestamps = [s.timestamp for s in samples]
        assert timestamps == sorted(timestamps)
        # Verify values match avg_value from aggregates
        assert samples[0].value == 22.5  # First aggregate avg_value
        assert samples[-1].value == 22.5 + 23  # Last aggregate avg_value

    def test_correlation_timestamp_ordering(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that combined samples are properly sorted by timestamp."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()
        old_date = now - timedelta(days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS + 2)

        # Ensure area and entity exist
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=entity_id,
                entity_type="temperature",
            )
            session.add(entity)

            # Create recent raw samples
            for i in range(5):
                sample = db.NumericSamples(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    timestamp=now - timedelta(hours=5 - i),
                    value=20.0 + i,
                )
                session.add(sample)

            # Create old hourly aggregates
            for i in range(3):
                hour_start = old_date + timedelta(hours=i)
                aggregate = db.NumericAggregates(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    aggregation_period=AGGREGATION_PERIOD_HOURLY,
                    period_start=hour_start,
                    period_end=hour_start + timedelta(hours=1),
                    min_value=20.0,
                    max_value=25.0,
                    avg_value=22.5,
                    median_value=22.5,
                    sample_count=10,
                    first_value=20.0,
                    last_value=25.0,
                    std_deviation=1.5,
                )
                session.add(aggregate)
            session.commit()

        # Create occupied intervals
        intervals = [
            (old_date, old_date + timedelta(hours=2)),
            (now - timedelta(hours=3), now - timedelta(hours=1)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Analyze correlation
        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None
        # Should have both sources combined
        assert result["sample_count"] == 8  # 5 recent + 3 aggregates


class TestConvertIntervalsToSamples:
    """Test convert_intervals_to_samples function."""

    def test_convert_intervals_to_samples_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful conversion of intervals to samples."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "binary_sensor.door"
        now = dt_util.utcnow()
        period_start = now - timedelta(hours=10)
        period_end = now

        # Create binary entity with intervals
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="door",
            )
            session.add(entity)

            # Create 5 intervals, 3 active ("on") and 2 inactive ("off")
            for i in range(5):
                start = now - timedelta(hours=5 - i)
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

        # Convert intervals to samples
        with db.get_session() as session:
            samples = convert_intervals_to_samples(
                db,
                area_name,
                entity_id,
                period_start,
                period_end,
                ["on"],
                session,
            )

        # Should have 5 samples (one per interval)
        assert len(samples) == 5
        # First 3 should be active (value=1.0), last 2 inactive (value=0.0)
        assert samples[0].value == 1.0
        assert samples[1].value == 1.0
        assert samples[2].value == 1.0
        assert samples[3].value == 0.0
        assert samples[4].value == 0.0
        # Samples should be sorted by timestamp
        timestamps = [s.timestamp for s in samples]
        assert timestamps == sorted(timestamps)

    def test_convert_intervals_to_samples_no_active_states(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test conversion with no active states returns empty list."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "binary_sensor.door"
        now = dt_util.utcnow()

        with db.get_session() as session:
            samples = convert_intervals_to_samples(
                db,
                area_name,
                entity_id,
                now - timedelta(hours=10),
                now,
                None,
                session,
            )
        assert samples == []

    def test_convert_intervals_to_samples_period_boundary_clamping(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that intervals are clamped to period boundaries."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "binary_sensor.door"
        now = dt_util.utcnow()
        period_start = now - timedelta(hours=5)
        period_end = now - timedelta(hours=1)

        # Create interval that spans before, during, and after period
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="door",
            )
            session.add(entity)

            # Interval starts before period, ends after period
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id=entity_id,
                start_time=now - timedelta(hours=10),
                end_time=now,
                state="on",
                duration_seconds=36000,
            )
            session.add(interval)
            session.commit()

        # Convert intervals to samples
        with db.get_session() as session:
            samples = convert_intervals_to_samples(
                db,
                area_name,
                entity_id,
                period_start,
                period_end,
                ["on"],
                session,
            )

        # Should have 1 sample at the midpoint of the clamped interval
        assert len(samples) == 1
        # Midpoint should be within period boundaries
        assert period_start <= samples[0].timestamp <= period_end


class TestMapBinaryStateToSemantic:
    """Test _map_binary_state_to_semantic function."""

    @pytest.mark.parametrize(
        ("input_state", "active_states", "expected_result", "description"),
        [
            ("off", ["closed"], "closed", "door closed (off → closed)"),
            ("on", ["open"], "open", "door open (on → open)"),
            ("on", ["open"], "open", "window open (on → open)"),
            ("off", ["closed"], "closed", "window closed (off → closed)"),
        ],
    )
    def test_map_binary_state_to_semantic(
        self, input_state, active_states, expected_result, description
    ):
        """Test mapping binary states to semantic states."""
        result = _map_binary_state_to_semantic(input_state, active_states)
        assert result == expected_result

    @pytest.mark.parametrize(
        ("input_state", "active_states", "expected_result"),
        [
            ("off", ["on"], "off"),  # No mapping when semantic not in active_states
            ("on", ["off"], "on"),  # No mapping when semantic not in active_states
        ],
    )
    def test_no_mapping_when_semantic_not_present(
        self, input_state, active_states, expected_result
    ):
        """Test that no mapping occurs when semantic states not in active_states."""
        result = _map_binary_state_to_semantic(input_state, active_states)
        assert result == expected_result

    def test_mapping_preserves_other_states(self):
        """Test that non-binary states are preserved."""
        result = _map_binary_state_to_semantic("playing", ["playing", "paused"])
        assert result == "playing"


class TestSaveBinaryLikelihoodResult:
    """Test save_binary_likelihood_result function."""

    def test_save_binary_likelihood_result_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving binary likelihood result successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"

        _create_entity(db, area_name, entity_id)

        likelihood_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "prob_given_true": 0.75,
            "prob_given_false": 0.15,
            "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
            "analysis_period_end": dt_util.utcnow(),
            "analysis_error": None,
            "calculation_date": dt_util.utcnow(),
        }

        result = save_binary_likelihood_result(db, likelihood_data, InputType.APPLIANCE)
        assert result is True

        # Verify likelihood was saved
        with db.get_session() as session:
            correlation = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .first()
            )
            assert correlation is not None
            assert correlation.correlation_type == "binary_likelihood"
            assert correlation.mean_value_when_occupied == 0.75
            assert correlation.mean_value_when_unoccupied == 0.15

    def test_save_binary_likelihood_result_updates_existing(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that saving updates existing record."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"

        _create_entity(db, area_name, entity_id)

        period_start = dt_util.utcnow() - timedelta(days=30)
        likelihood_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "prob_given_true": 0.75,
            "prob_given_false": 0.15,
            "analysis_period_start": period_start,
            "analysis_period_end": dt_util.utcnow(),
            "analysis_error": None,
            "calculation_date": dt_util.utcnow(),
        }

        # Save first time
        save_binary_likelihood_result(db, likelihood_data, InputType.APPLIANCE)

        # Update and save again
        likelihood_data["prob_given_true"] = 0.80
        result = save_binary_likelihood_result(db, likelihood_data, InputType.APPLIANCE)
        assert result is True

        # Verify updated value
        with db.get_session() as session:
            correlation = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .first()
            )
            assert correlation.mean_value_when_occupied == 0.80


class TestGetCorrelatableEntitiesByArea:
    """Test get_correlatable_entities_by_area function."""

    @pytest.mark.parametrize(
        (
            "entity_id",
            "input_type",
            "is_binary",
            "active_states",
            "active_range",
            "state_provider",
            "description",
        ),
        [
            (
                "media_player.tv",
                InputType.MEDIA,
                True,
                [STATE_ON],
                None,
                lambda x: STATE_ON,
                "binary sensors",
            ),
            (
                "sensor.temperature",
                InputType.TEMPERATURE,
                False,
                None,
                None,
                lambda x: "20.0",
                "numeric sensors",
            ),
        ],
    )
    def test_get_correlatable_entities(
        self,
        coordinator: AreaOccupancyCoordinator,
        entity_id,
        input_type,
        is_binary,
        active_states,
        active_range,
        state_provider,
        description,
    ):
        """Test discovery of correlatable entities (binary and numeric)."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Create entity type
        entity_type = EntityType(
            input_type=input_type,
            weight=0.7 if is_binary else 0.1,
            prob_given_true=0.5,
            prob_given_false=0.5,
            active_states=active_states,
            active_range=active_range,
        )

        # Create entity
        entity = Entity(
            entity_id=entity_id,
            type=entity_type,
            prob_given_true=0.5,
            prob_given_false=0.5,
            decay=Decay(half_life=60.0),
            state_provider=state_provider,
            last_updated=dt_util.utcnow(),
        )
        area.entities.entities[entity_id] = entity

        result = get_correlatable_entities_by_area(coordinator)

        assert area_name in result
        assert entity_id in result[area_name]
        assert result[area_name][entity_id]["is_binary"] is is_binary
        assert result[area_name][entity_id]["active_states"] == active_states

    def test_get_correlatable_entities_excludes_motion(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that motion sensors are excluded from correlatable entities."""
        area_name = coordinator.get_area_names()[0]

        # Motion sensors should not appear in correlatable entities
        result = get_correlatable_entities_by_area(coordinator)

        # Check that motion sensors are not included
        if area_name in result:
            for entity_id in result[area_name]:
                entity_info = result[area_name][entity_id]
                # Should not have motion sensors
                assert entity_info["input_type"] != InputType.MOTION


class TestRunCorrelationAnalysis:
    """Test run_correlation_analysis function."""

    async def test_run_correlation_analysis_binary_sensor(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test correlation analysis for binary sensors."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Add a media player entity
        media_type = EntityType(
            input_type=InputType.MEDIA,
            weight=0.7,
            prob_given_true=0.5,
            prob_given_false=0.5,
            active_states=[STATE_ON],
        )
        media_entity = Entity(
            entity_id="media_player.tv",
            type=media_type,
            prob_given_true=0.5,
            prob_given_false=0.5,
            decay=Decay(half_life=60.0),
            state_provider=lambda x: STATE_ON,
            last_updated=dt_util.utcnow(),
        )
        area.entities.entities["media_player.tv"] = media_entity

        # Create motion intervals for occupied periods
        db = coordinator.db
        now = dt_util.utcnow()
        base_time = now - timedelta(seconds=30)
        motion_entity_id = area.config.sensors.motion[0]
        motion_intervals = _create_motion_intervals(
            db, area_name, motion_entity_id, 5, now=base_time
        )
        db.save_occupied_intervals_cache(area_name, motion_intervals, "motion_sensors")

        # Create media player intervals
        with db.get_session() as session:
            for i in range(5):
                start = base_time - timedelta(hours=5 - i)
                end = start + timedelta(hours=1)
                state = "on" if i < 3 else "off"
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id="media_player.tv",
                    start_time=start,
                    end_time=end,
                    state=state,
                    duration_seconds=3600,
                )
                session.add(interval)
            session.commit()

        # Run correlation analysis
        results = await run_correlation_analysis(coordinator, return_results=True)

        # Should have analyzed the media player
        assert results is not None
        assert len(results) > 0
        media_result = next(
            (r for r in results if r.get("entity_id") == "media_player.tv"), None
        )
        assert media_result is not None
        assert media_result["type"] == "binary_likelihood"
        assert media_result["success"] is True

    async def test_run_correlation_analysis_numeric_sensor(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test correlation analysis for numeric sensors."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        entity_id = "sensor.temperature"

        # Add temperature sensor entity
        temp_type = EntityType(
            input_type=InputType.TEMPERATURE,
            weight=0.1,
            prob_given_true=0.5,
            prob_given_false=0.5,
            active_range=None,
        )
        temp_entity = Entity(
            entity_id=entity_id,
            type=temp_type,
            prob_given_true=0.5,
            prob_given_false=0.5,
            decay=Decay(half_life=60.0),
            state_provider=lambda x: "20.0",
            last_updated=dt_util.utcnow(),
        )
        area.entities.entities[entity_id] = temp_entity

        # Create numeric samples
        now = dt_util.utcnow()
        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 100, lambda i: 20.0 + (i % 10)
        )

        # Create occupied intervals
        intervals = [
            (now - timedelta(hours=50), now - timedelta(hours=40)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Run correlation analysis
        results = await run_correlation_analysis(coordinator, return_results=True)

        # Should have analyzed the temperature sensor
        assert results is not None
        assert len(results) > 0
        temp_result = next(
            (r for r in results if r.get("entity_id") == entity_id), None
        )
        assert temp_result is not None
        assert temp_result["type"] == "correlation"
        assert temp_result["success"] is True

    async def test_run_correlation_analysis_no_results_flag(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that return_results=False returns None."""
        result = await run_correlation_analysis(coordinator, return_results=False)
        assert result is None
