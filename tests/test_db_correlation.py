"""Tests for database correlation analysis functions."""

from datetime import datetime, timedelta
from unittest.mock import patch

from dateutil.relativedelta import relativedelta
import pytest
from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.const import (
    AGGREGATION_PERIOD_HOURLY,
    CORRELATION_MONTHS_TO_KEEP,
    MIN_CORRELATION_SAMPLES,
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
    """Test calculate_pearson_correlation function.

    Tests the core correlation calculation algorithm, including:
    - Positive and negative correlations
    - Edge cases (insufficient samples, mismatched lengths, invalid values)
    - P-value calculation
    """

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
    """Test analyze_correlation function.

    Tests the main correlation analysis function for both numeric and binary sensors.
    Verifies:
    - Successful correlation analysis with valid data
    - Error handling for missing data, insufficient samples, no occupied intervals
    - Binary sensor correlation analysis
    - Negative correlation detection
    - Correlation type classification
    - Confidence and threshold calculations
    """

    def test_analyze_correlation_success(self, coordinator: AreaOccupancyCoordinator):
        """Test successful correlation analysis for numeric sensors.

        Verifies that correlation analysis correctly:
        - Calculates correlation coefficient and statistics
        - Classifies correlation type based on coefficient strength
        - Computes mean and std dev for occupied vs unoccupied periods
        - Returns all required fields in the result dictionary
        """
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        # Create samples with values cycling 20-29
        # Samples are created with timestamps: now - timedelta(hours=100-i)
        # So i=0 is 100 hours ago, i=99 is 1 hour ago
        _create_numeric_entity_with_samples(
            db,
            area_name,
            entity_id,
            100,
            lambda i: 20.0 + (i % 10),
            unit_of_measurement="°C",
        )

        # Occupied intervals: hours 50-40 ago and hours 20-10 ago
        # These correspond to samples i=50-59 and i=80-89
        intervals = [
            (now - timedelta(hours=50), now - timedelta(hours=40)),
            (now - timedelta(hours=20), now - timedelta(hours=10)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)

        # Verify result structure
        assert isinstance(result, dict)
        assert "correlation_coefficient" in result
        assert "sample_count" in result
        assert "correlation_type" in result
        assert "mean_value_when_occupied" in result
        assert "mean_value_when_unoccupied" in result
        assert "std_dev_when_occupied" in result
        assert "std_dev_when_unoccupied" in result

        # Verify sample_count matches input data
        assert result["sample_count"] == 100

        # Verify correlation coefficient is in valid range
        assert -1.0 <= result["correlation_coefficient"] <= 1.0

        # Verify correlation type matches coefficient strength
        abs_corr = abs(result["correlation_coefficient"])
        if abs_corr >= 0.4:
            assert result["correlation_type"] in ("strong_positive", "strong_negative")
        elif abs_corr >= 0.15:
            assert result["correlation_type"] in ("positive", "negative")
        else:
            assert result["correlation_type"] == "none"

        # Verify mean values are within expected range (generator produces 20-29)
        assert 20.0 <= result["mean_value_when_occupied"] <= 29.0
        assert 20.0 <= result["mean_value_when_unoccupied"] <= 29.0

        # Verify std dev values are reasonable (for uniform distribution 20-29, std ~2.87)
        assert result["std_dev_when_occupied"] is not None
        assert result["std_dev_when_unoccupied"] is not None
        assert 0.0 <= result["std_dev_when_occupied"] <= 5.0
        assert 0.0 <= result["std_dev_when_unoccupied"] <= 5.0

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
        """Test successful binary correlation analysis.

        Creates a perfect correlation scenario where the binary sensor is "on"
        during all occupied periods and "off" during all unoccupied periods.
        Verifies that correlation analysis correctly identifies this strong
        positive correlation.
        """
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "binary_sensor.door"
        now = dt_util.utcnow()

        # Create 50 occupied intervals at even hours (0, 2, 4, ..., 98)
        # Each interval is 1 hour long
        occupied_intervals = []
        for i in range(50):
            start = now - timedelta(hours=100 - (i * 2))
            end = start + timedelta(hours=1)
            occupied_intervals.append((start, end))
        _create_occupied_intervals_cache(db, area_name, occupied_intervals)

        # Create 100 binary intervals covering the same time period
        # Even indices (0, 2, 4, ...) are "on" (matching occupied periods)
        # Odd indices (1, 3, 5, ...) are "off" (matching unoccupied periods)
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

        # Verify result structure
        assert isinstance(result, dict)
        assert "correlation_coefficient" in result
        assert "sample_count" in result
        assert "mean_value_when_occupied" in result
        assert "mean_value_when_unoccupied" in result

        # With perfect correlation (sensor on during all occupied, off during all unoccupied),
        # correlation should be very high (close to 1.0)
        assert result["correlation_coefficient"] > 0.95

        # Verify correlation type is strong_positive
        assert result["correlation_type"] == "strong_positive"

        # Verify sample count is reasonable (binary analysis uses 60-second chunks)
        assert result["sample_count"] > 0

        # Mean when occupied should be close to 1.0 (sensor is on during occupied periods)
        assert result["mean_value_when_occupied"] > 0.95

        # Mean when unoccupied should be close to 0.0 (sensor is off during unoccupied periods)
        assert result["mean_value_when_unoccupied"] < 0.05

        # Verify std dev values are reasonable for binary data
        assert result["std_dev_when_occupied"] is not None
        assert result["std_dev_when_unoccupied"] is not None
        assert 0.0 <= result["std_dev_when_occupied"] <= 1.0
        assert 0.0 <= result["std_dev_when_unoccupied"] <= 1.0

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

    @pytest.mark.parametrize(
        (
            "is_binary",
            "entity_id",
            "entity_type",
            "analysis_period_days",
            "active_states",
        ),
        [
            (False, "sensor.temperature", "numeric", 30, None),
            (True, "binary_sensor.door", "door", 5, ["on"]),
        ],
        ids=["numeric", "binary"],
    )
    def test_analyze_correlation_no_unoccupied_samples(
        self,
        coordinator: AreaOccupancyCoordinator,
        is_binary,
        entity_id,
        entity_type,
        analysis_period_days,
        active_states,
    ):
        """Test correlation analysis when all samples/intervals are in occupied periods."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        now = dt_util.utcnow()

        # Create occupied interval covering the entire period
        intervals = [
            (now - timedelta(hours=100), now),  # Covers all samples/intervals
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Setup entity and data based on type
        if is_binary:
            # Create binary entity intervals
            with db.get_session() as session:
                db.save_area_data(area_name)
                entity = db.Entities(
                    entity_id=entity_id,
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_type=entity_type,
                )
                session.add(entity)

                for i in range(100):
                    start = now - timedelta(hours=100 - i)
                    end = start + timedelta(hours=1)
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
        else:
            _create_numeric_entity_with_samples(
                db, area_name, entity_id, 100, lambda i: 20.0 + (i % 10)
            )

        result = analyze_correlation(
            db,
            area_name,
            entity_id,
            analysis_period_days=analysis_period_days,
            is_binary=is_binary,
            active_states=active_states,
        )
        assert result is not None
        assert result["analysis_error"] == "no_unoccupied_samples"
        if is_binary:
            assert result["sample_count"] > 0
        else:
            assert result["sample_count"] == 100

    @pytest.mark.parametrize(
        (
            "correlation_coefficient",
            "expected_type",
            "description",
        ),
        [
            (0.8, "strong_positive", "strong positive correlation"),
            (0.5, "strong_positive", "moderate positive correlation (>= 0.4)"),
            (0.4, "strong_positive", "threshold positive correlation"),
            (0.3, "positive", "weak positive correlation (0.15-0.4)"),
            (0.2, "positive", "weak positive correlation"),
            (0.15, "positive", "threshold weak positive correlation"),
            (0.1, "none", "very weak positive correlation (< 0.15)"),
            (0.0, "none", "no correlation"),
            (-0.1, "none", "very weak negative correlation (< 0.15)"),
            (-0.15, "negative", "threshold weak negative correlation"),
            (-0.2, "negative", "weak negative correlation"),
            (-0.3, "negative", "weak negative correlation (0.15-0.4)"),
            (-0.4, "strong_negative", "threshold negative correlation"),
            (-0.5, "strong_negative", "moderate negative correlation (>= 0.4)"),
            (-0.8, "strong_negative", "strong negative correlation"),
        ],
    )
    def test_analyze_correlation_type_classification(
        self,
        coordinator: AreaOccupancyCoordinator,
        correlation_coefficient,
        expected_type,
        description,
    ):
        """Test that correlation type is correctly classified based on coefficient strength."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        # Create samples and occupied intervals
        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 100, lambda i: 20.0 + (i % 10)
        )
        intervals = [(now - timedelta(hours=50), now - timedelta(hours=40))]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Mock calculate_pearson_correlation to return the test coefficient
        with patch(
            "custom_components.area_occupancy.db.correlation.calculate_pearson_correlation"
        ) as mock_calc:
            mock_calc.return_value = (correlation_coefficient, 0.01)

            result = analyze_correlation(
                db, area_name, entity_id, analysis_period_days=30
            )
            assert result is not None
            assert result["correlation_coefficient"] == correlation_coefficient
            assert result["correlation_type"] == expected_type, (
                f"Expected {expected_type} for coefficient {correlation_coefficient} "
                f"({description}), got {result['correlation_type']}"
            )

    def test_analyze_correlation_confidence_calculation(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that confidence calculation increases with stronger correlation and more samples."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 100, lambda i: 20.0 + (i % 10)
        )
        intervals = [(now - timedelta(hours=50), now - timedelta(hours=40))]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Test with strong correlation and many samples
        with patch(
            "custom_components.area_occupancy.db.correlation.calculate_pearson_correlation"
        ) as mock_calc:
            mock_calc.return_value = (0.8, 0.01)  # Strong correlation

            result = analyze_correlation(
                db, area_name, entity_id, analysis_period_days=30
            )
            assert result is not None
            assert result["correlation_coefficient"] == 0.8

            # Confidence formula: min(1.0, abs_correlation * (1.0 - (MIN_CORRELATION_SAMPLES / sample_count)))
            # With 100 samples and 0.8 correlation:
            # confidence = 0.8 * (1.0 - (MIN_CORRELATION_SAMPLES / 100))
            expected_confidence = min(
                1.0, 0.8 * (1.0 - (MIN_CORRELATION_SAMPLES / result["sample_count"]))
            )
            assert abs(result["confidence"] - expected_confidence) < 0.01

        # Test with weak correlation - confidence should be lower
        with patch(
            "custom_components.area_occupancy.db.correlation.calculate_pearson_correlation"
        ) as mock_calc:
            mock_calc.return_value = (0.2, 0.05)  # Weak correlation

            result = analyze_correlation(
                db, area_name, entity_id, analysis_period_days=30
            )
            assert result is not None
            assert result["correlation_coefficient"] == 0.2

            # Confidence should be lower due to weaker correlation
            expected_confidence = min(
                1.0, 0.2 * (1.0 - (MIN_CORRELATION_SAMPLES / result["sample_count"]))
            )
            assert abs(result["confidence"] - expected_confidence) < 0.01
            assert (
                result["confidence"] < 0.3
            )  # Should be much lower than strong correlation

    def test_analyze_correlation_threshold_calculation(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that thresholds are calculated as mean ± std_dev."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        # Create samples with known values for easier verification
        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 100, lambda i: 20.0 + (i % 10)
        )
        intervals = [(now - timedelta(hours=50), now - timedelta(hours=40))]
        _create_occupied_intervals_cache(db, area_name, intervals)

        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None

        # Verify thresholds are calculated correctly
        mean_occupied = result["mean_value_when_occupied"]
        mean_unoccupied = result["mean_value_when_unoccupied"]
        std_occupied = result["std_dev_when_occupied"]
        std_unoccupied = result["std_dev_when_unoccupied"]

        if mean_occupied is not None and std_occupied is not None:
            expected_threshold_active = mean_occupied + std_occupied
            assert result["threshold_active"] is not None
            assert abs(result["threshold_active"] - expected_threshold_active) < 0.01

        if mean_unoccupied is not None and std_unoccupied is not None:
            expected_threshold_inactive = mean_unoccupied - std_unoccupied
            assert result["threshold_inactive"] is not None
            assert (
                abs(result["threshold_inactive"] - expected_threshold_inactive) < 0.01
            )


class TestSaveResults:
    """Test save functions for correlation and binary likelihood results.

    Tests saving results to the database, including:
    - Successful save operations
    - Database error handling
    - Updating existing records (concurrent updates)

    Uses parametrization to test both save_correlation_result and
    save_binary_likelihood_result with the same test logic.
    """

    @pytest.mark.parametrize(
        (
            "save_func",
            "create_data_func",
            "entity_id",
            "input_type",
            "verify_field",
            "verify_value",
        ),
        [
            (
                save_correlation_result,
                lambda db, area_name, entity_id: {
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
                },
                "sensor.temperature",
                None,
                "correlation_coefficient",
                0.75,
            ),
            (
                save_binary_likelihood_result,
                lambda db, area_name, entity_id: {
                    "entry_id": db.coordinator.entry_id,
                    "area_name": area_name,
                    "entity_id": entity_id,
                    "prob_given_true": 0.75,
                    "prob_given_false": 0.15,
                    "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
                    "analysis_period_end": dt_util.utcnow(),
                    "analysis_error": None,
                    "calculation_date": dt_util.utcnow(),
                },
                "light.test_light",
                InputType.APPLIANCE,
                "mean_value_when_occupied",
                0.75,
            ),
        ],
        ids=["correlation", "binary_likelihood"],
    )
    def test_save_result_success(
        self,
        coordinator: AreaOccupancyCoordinator,
        save_func,
        create_data_func,
        entity_id,
        input_type,
        verify_field,
        verify_value,
    ):
        """Test saving result successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_entity(db, area_name, entity_id)

        data = create_data_func(db, area_name, entity_id)

        if input_type is not None:
            result = save_func(db, data, input_type)
        else:
            result = save_func(db, data)

        assert result is True

        # Verify result was saved
        with db.get_session() as session:
            correlation = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .first()
            )
            assert correlation is not None
            assert getattr(correlation, verify_field) == verify_value

    @pytest.mark.parametrize(
        ("save_func", "create_data_func", "entity_id", "input_type"),
        [
            (
                save_correlation_result,
                lambda db, area_name, entity_id: {
                    "entry_id": db.coordinator.entry_id,
                    "area_name": area_name,
                    "entity_id": entity_id,
                    "input_type": InputType.TEMPERATURE.value,
                    "correlation_coefficient": 0.75,
                    "correlation_type": "strong_positive",
                    "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
                    "analysis_period_end": dt_util.utcnow(),
                    "sample_count": 100,
                },
                "sensor.temperature",
                None,
            ),
            (
                save_binary_likelihood_result,
                lambda db, area_name, entity_id: {
                    "entry_id": db.coordinator.entry_id,
                    "area_name": area_name,
                    "entity_id": entity_id,
                    "prob_given_true": 0.75,
                    "prob_given_false": 0.15,
                    "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
                    "analysis_period_end": dt_util.utcnow(),
                    "analysis_error": None,
                    "calculation_date": dt_util.utcnow(),
                },
                "light.test_light",
                InputType.APPLIANCE,
            ),
        ],
        ids=["correlation", "binary_likelihood"],
    )
    def test_save_result_database_error(
        self,
        coordinator: AreaOccupancyCoordinator,
        save_func,
        create_data_func,
        entity_id,
        input_type,
    ):
        """Test that save functions handle database errors gracefully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_entity(db, area_name, entity_id)

        data = create_data_func(db, area_name, entity_id)

        # Mock database commit to raise an error
        with patch.object(db, "get_session") as mock_session:
            mock_session.return_value.__enter__.return_value.commit.side_effect = (
                SQLAlchemyError("Database error")
            )
            if input_type is not None:
                result = save_func(db, data, input_type)
            else:
                result = save_func(db, data)
            assert result is False

    @pytest.mark.parametrize(
        (
            "save_func",
            "create_data_func",
            "entity_id",
            "input_type",
            "update_field",
            "update_value",
            "verify_field",
            "verify_value",
        ),
        [
            (
                save_correlation_result,
                lambda db, area_name, entity_id, period_start: {
                    "entry_id": db.coordinator.entry_id,
                    "area_name": area_name,
                    "entity_id": entity_id,
                    "input_type": InputType.TEMPERATURE.value,
                    "correlation_coefficient": 0.75,
                    "correlation_type": "strong_positive",
                    "analysis_period_start": period_start,
                    "analysis_period_end": dt_util.utcnow(),
                    "sample_count": 100,
                },
                "sensor.temperature",
                None,
                "correlation_coefficient",
                0.80,
                "correlation_coefficient",
                0.80,
            ),
            (
                save_binary_likelihood_result,
                lambda db, area_name, entity_id, period_start: {
                    "entry_id": db.coordinator.entry_id,
                    "area_name": area_name,
                    "entity_id": entity_id,
                    "prob_given_true": 0.75,
                    "prob_given_false": 0.15,
                    "analysis_period_start": period_start,
                    "analysis_period_end": dt_util.utcnow(),
                    "analysis_error": None,
                    "calculation_date": dt_util.utcnow(),
                },
                "light.test_light",
                InputType.APPLIANCE,
                "prob_given_true",
                0.80,
                "mean_value_when_occupied",
                0.80,
            ),
        ],
        ids=["correlation", "binary_likelihood"],
    )
    def test_save_result_updates_existing(
        self,
        coordinator: AreaOccupancyCoordinator,
        save_func,
        create_data_func,
        entity_id,
        input_type,
        update_field,
        update_value,
        verify_field,
        verify_value,
    ):
        """Test that save functions update existing record with same analysis_period_start."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_entity(db, area_name, entity_id)

        period_start = dt_util.utcnow() - timedelta(days=30)
        data = create_data_func(db, area_name, entity_id, period_start)

        # Save first time
        if input_type is not None:
            result = save_func(db, data, input_type)
        else:
            result = save_func(db, data)
        assert result is True

        # Update and save again with same analysis_period_start
        data[update_field] = update_value
        # For correlation, also update correlation_type
        if input_type is None and "correlation_type" in data:
            data["correlation_type"] = "positive"
        if input_type is not None:
            result = save_func(db, data, input_type)
        else:
            result = save_func(db, data)
        assert result is True

        # Verify updated value
        with db.get_session() as session:
            correlation = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .first()
            )
            assert correlation is not None
            assert getattr(correlation, verify_field) == verify_value
            # For correlation, also verify correlation_type was updated
            if input_type is None:
                assert correlation.correlation_type == "positive"
            # Verify only one record exists (updated, not duplicated)
            count = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .count()
            )
            assert count == 1


class TestGetCorrelationForEntity:
    """Test get_correlation_for_entity function.

    Tests retrieving correlation results from the database, including:
    - Retrieving most recent correlation
    - Handling multiple correlations (selects most recent)
    - Handling missing correlations
    """

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
        """Test getting correlation when multiple results exist.

        Verifies that when multiple correlations exist, the function returns
        the most recent one by calculation_date (fallback path when current
        month's record doesn't exist).
        """
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Create multiple correlations with different calculation dates
        # None match current month, so fallback to most recent by calculation_date
        now = dt_util.utcnow()
        with db.get_session() as session:
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
                    calculation_date=now - timedelta(days=i),  # i=0 is most recent
                    analysis_period_start=period_start,
                    analysis_period_end=period_end,
                    sample_count=100,
                )
                session.add(correlation)
            session.commit()

        # Should return most recent by calculation_date (i=0, coefficient=0.5)
        result = get_correlation_for_entity(db, area_name, entity_id)
        assert result is not None
        assert result["correlation_coefficient"] == 0.5


class TestAnalyzeAndSaveCorrelation:
    """Test analyze_and_save_correlation function.

    Tests the combined analyze-and-save operation, including:
    - Successful analysis and save
    - Handling invalid correlation coefficients (NaN, infinity)
    - Preserving analysis errors even with invalid coefficients
    """

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

    def test_analyze_and_save_correlation_invalid_coefficient(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that analyze_and_save_correlation skips saving when coefficient is invalid."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Mock analyze_correlation to return invalid coefficient (NaN)
        with patch(
            "custom_components.area_occupancy.db.correlation.analyze_correlation"
        ) as mock_analyze:
            mock_analyze.return_value = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name,
                "entity_id": entity_id,
                "correlation_coefficient": float("nan"),  # Invalid coefficient
                "correlation_type": "none",
                "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
                "analysis_period_end": dt_util.utcnow(),
                "sample_count": 100,
            }

            result = analyze_and_save_correlation(
                db, area_name, entity_id, analysis_period_days=30
            )
            # Should return None (not saved due to invalid coefficient)
            assert result is None

            # Verify nothing was saved to database
            with db.get_session() as session:
                correlation = (
                    session.query(db.Correlations)
                    .filter_by(area_name=area_name, entity_id=entity_id)
                    .first()
                )
                assert correlation is None

    def test_analyze_and_save_correlation_invalid_coefficient_with_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that analyze_and_save_correlation saves error even with invalid coefficient."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Mock analyze_correlation to return invalid coefficient with analysis_error
        with patch(
            "custom_components.area_occupancy.db.correlation.analyze_correlation"
        ) as mock_analyze:
            mock_analyze.return_value = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name,
                "entity_id": entity_id,
                "correlation_coefficient": float("inf"),  # Invalid coefficient
                "correlation_type": "none",
                "analysis_error": "too_few_samples",  # Error to preserve
                "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
                "analysis_period_end": dt_util.utcnow(),
                "sample_count": 0,
            }

            result = analyze_and_save_correlation(
                db, area_name, entity_id, analysis_period_days=30
            )
            # Should return result (error is preserved even with invalid coefficient)
            assert result is not None
            assert result["analysis_error"] == "too_few_samples"
            # Coefficient should be replaced with 0.0 placeholder
            assert result["correlation_coefficient"] == 0.0

            # Verify error was saved to database
            with db.get_session() as session:
                correlation = (
                    session.query(db.Correlations)
                    .filter_by(area_name=area_name, entity_id=entity_id)
                    .first()
                )
                assert correlation is not None
                assert correlation.analysis_error == "too_few_samples"
                assert correlation.correlation_coefficient == 0.0


class TestPruneOldCorrelations:
    """Test _prune_old_correlations function.

    Tests the correlation pruning logic that keeps only the most recent N months:
    - Pruning when excess correlations exist
    - No pruning when fewer than limit
    - Handling duplicates within the same month
    - Edge cases (empty list, exactly at limit)
    """

    @pytest.fixture(autouse=True)
    def _set_default_tz_utc(self):
        """Correlation month grouping is local-time based; keep tests deterministic."""
        original = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(dt_util.UTC)
        try:
            yield
        finally:
            dt_util.set_default_time_zone(original)

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
                # Use month-based arithmetic to ensure each iteration falls into a distinct calendar month
                months_ago = i
                # Calculate target month: go back i months from now
                target_month_start = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                ) - relativedelta(months=months_ago)
                # Period spans the entire month
                period_start = target_month_start
                period_end = target_month_start + relativedelta(months=1)
                # Calculation date is at the end of the analysis period
                calculation_date = period_end
                correlation = db.Correlations(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    input_type=InputType.TEMPERATURE.value,
                    correlation_coefficient=0.5 + (i * 0.01),
                    correlation_type="strong_positive",
                    calculation_date=calculation_date,
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
        """Test pruning when there are fewer correlations than limit.

        Verifies that when there are fewer correlations than CORRELATION_MONTHS_TO_KEEP,
        none are pruned. The pruning function groups by year-month, so we need
        correlations in different months to properly test the logic.
        """
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Create fewer correlations than limit, spread across different months
        # Pruning groups by year-month, so we need proper month boundaries
        now = dt_util.utcnow()
        num_correlations = CORRELATION_MONTHS_TO_KEEP - 2
        with db.get_session() as session:
            for i in range(num_correlations):
                # Calculate target month: current month minus i months
                target_date = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                for _ in range(i):
                    # Go back one month, handling year boundary
                    if target_date.month == 1:
                        target_date = target_date.replace(
                            year=target_date.year - 1, month=12
                        )
                    else:
                        target_date = target_date.replace(month=target_date.month - 1)

                # Create period spanning the entire month
                period_start = target_date
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

        # Verify all correlations remain (none should be pruned)
        with db.get_session() as session:
            count = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .count()
            )
            assert count == num_correlations

    def test_prune_old_correlations_empty_list(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test pruning when no correlations exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Call prune function with no correlations
        with db.get_session() as session:
            _prune_old_correlations(db, session, area_name, entity_id)

        # Verify no errors occurred
        with db.get_session() as session:
            count = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .count()
            )
            assert count == 0

    def test_prune_old_correlations_exactly_limit(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test pruning when exactly CORRELATION_MONTHS_TO_KEEP correlations exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Create exactly CORRELATION_MONTHS_TO_KEEP correlations
        now = dt_util.utcnow()
        with db.get_session() as session:
            for i in range(CORRELATION_MONTHS_TO_KEEP):
                target_date = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                for _ in range(i):
                    if target_date.month == 1:
                        target_date = target_date.replace(
                            year=target_date.year - 1, month=12
                        )
                    else:
                        target_date = target_date.replace(month=target_date.month - 1)

                period_start = target_date
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

        # Verify all correlations remain (none should be pruned)
        with db.get_session() as session:
            count = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .count()
            )
            assert count == CORRELATION_MONTHS_TO_KEEP

    def test_prune_old_correlations_duplicates_in_month(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test pruning when multiple correlations exist in the same month."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        _create_entity(db, area_name, entity_id)

        # Create multiple correlations in the same month
        # Use different days within the same month to avoid UNIQUE constraint violation
        now = dt_util.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        with db.get_session() as session:
            # Create 3 correlations in the same month with different analysis_period_start
            # (different days) and different calculation dates
            for i in range(3):
                # Use different days within the same month (day 1, 5, 10)
                period_start = month_start.replace(day=1 + (i * 4))
                correlation = db.Correlations(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    input_type=InputType.TEMPERATURE.value,
                    correlation_coefficient=0.5 + (i * 0.1),
                    correlation_type="strong_positive",
                    calculation_date=now
                    - timedelta(days=i),  # Different calculation dates
                    analysis_period_start=period_start,  # Different days in same month
                    analysis_period_end=now,
                    sample_count=100,
                )
                session.add(correlation)
            session.commit()

        # Call prune function
        with db.get_session() as session:
            _prune_old_correlations(db, session, area_name, entity_id)

        # Verify only the most recent correlation for the month remains
        # (by calculation_date, which is now - 0 days = most recent)
        with db.get_session() as session:
            correlations = (
                session.query(db.Correlations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .all()
            )
            assert len(correlations) == 1
            # Should keep the one with most recent calculation_date (i=0, coefficient=0.5)
            assert correlations[0].correlation_coefficient == 0.5


class TestAnalyzeBinaryLikelihoods:
    """Test analyze_binary_likelihoods function.

    Tests binary sensor likelihood analysis using duration-based probability calculation:
    - Successful probability calculation
    - Error handling (no active states, no occupied intervals, no sensor data)
    - State mapping for door/window sensors (binary to semantic)
    - Edge cases (sensor active but never during occupied periods)
    """

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

        # Calculate expected probabilities:
        # - 5 occupied intervals (5 hours total)
        # - Light is "on" during first 3 occupied intervals (3 hours)
        # - Light is "off" during last 2 occupied intervals (2 hours)
        # - prob_given_true = 3/5 = 0.6
        # - During unoccupied periods, light has no intervals (considered off)
        # - prob_given_false = 0.0, but clamped to 0.05 minimum

        # Verify prob_given_true is approximately 0.6 (3 out of 5 occupied hours)
        assert 0.55 <= result["prob_given_true"] <= 0.65

        # Verify prob_given_false is clamped minimum (0.05) since light is never on during unoccupied
        assert result["prob_given_false"] == 0.05

        # Light should be more likely on when occupied
        assert result["prob_given_true"] > result["prob_given_false"]

        # Probabilities should be clamped to valid range
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
    """Test fixes for correlation logic bugs.

    Tests specific bug fixes and edge cases:
    - Timezone-aware datetime handling in queries
    - Entry ID filtering to prevent cross-entry data leakage
    - Unoccupied overlap calculation edge cases
    """

    def test_timezone_aware_numeric_query(self, coordinator: AreaOccupancyCoordinator):
        """Test that numeric sensor query uses timezone-aware datetimes correctly.

        Verifies that queries correctly handle timezone-aware datetimes and that
        samples are found when they fall within the analysis period, regardless
        of whether naive or aware datetimes are used.
        """
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        # Create samples with timezone-aware timestamps
        # Samples are created with: now - timedelta(hours=50-i)
        # So i=0 is 50 hours ago, i=49 is 1 hour ago
        _create_numeric_entity_with_samples(
            db, area_name, entity_id, 50, lambda i: 20.0 + (i % 10)
        )

        # Create occupied intervals that overlap with samples
        # Occupied period: hours 30-20 ago (samples i=20-30)
        intervals = [
            (now - timedelta(hours=30), now - timedelta(hours=20)),
        ]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Test with timezone-aware period (should work correctly)
        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        assert result is not None
        assert "correlation_coefficient" in result
        assert result["sample_count"] > 0

        # Verify all samples were found (proves timezone handling worked)
        assert result["sample_count"] == 50

        # Verify the analysis period boundaries are timezone-aware UTC
        assert result["analysis_period_start"] is not None
        assert result["analysis_period_end"] is not None
        assert isinstance(result["analysis_period_start"], datetime)
        assert isinstance(result["analysis_period_end"], datetime)

        # Verify they are timezone-aware (have tzinfo)
        assert result["analysis_period_start"].tzinfo is not None
        assert result["analysis_period_end"].tzinfo is not None

        # Verify analysis_period_end is approximately now (within 1 second)
        assert abs((result["analysis_period_end"] - now).total_seconds()) < 1.0

        # Verify analysis_period_start is approximately 30 days ago
        expected_start = now - timedelta(days=30)
        assert (
            abs((result["analysis_period_start"] - expected_start).total_seconds())
            < 1.0
        )

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
        assert result["analysis_error"] is None

        # Verify overlap calculations are correct:
        # - Light interval: base_time - 6h to base_time - 4h + 30m (2.5 hours total)
        # - Motion intervals create 5 occupied periods: base_time - 5h to base_time (5 hours total)
        # - The light interval overlaps with occupied periods from base_time - 5h to base_time - 3h30m
        # - The actual overlap depends on how intervals are clamped to the analysis period
        #
        # The key test is that:
        # 1. Both probabilities are calculated correctly (not None)
        # 2. prob_given_true > prob_given_false (light is more likely to be on during occupied periods)
        # 3. Both probabilities are within valid clamped range (0.05 to 0.95)

        # prob_given_true should be higher than prob_given_false since light overlaps with occupied periods
        assert result["prob_given_true"] > result["prob_given_false"]

        # Both probabilities should be valid (clamped between 0.05 and 0.95)
        assert 0.05 <= result["prob_given_true"] <= 0.95
        assert 0.05 <= result["prob_given_false"] <= 0.95

        # Verify that the light interval does overlap with occupied periods
        # (prob_given_true should be significantly higher than the minimum)
        assert result["prob_given_true"] > 0.05

        # Verify that there is some unoccupied overlap
        # (prob_given_false should be greater than 0, indicating light was on during unoccupied time)
        assert result["prob_given_false"] > 0.0


class TestNumericAggregatesInCorrelation:
    """Test correlation analysis using numeric aggregates for historical data.

    Tests the aggregation system that uses raw samples for recent data and
    hourly aggregates for older data:
    - Using only raw samples when all data is within retention period
    - Using only aggregates when all data is older than retention
    - Combining both sources when period spans retention boundary
    - Timestamp ordering of combined samples
    """

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
    """Test convert_intervals_to_samples function.

    Tests conversion of binary sensor intervals to numeric samples:
    - Successful conversion with proper value mapping (active=1.0, inactive=0.0)
    - Handling intervals that span period boundaries (clamping)
    - Handling missing active states
    """

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
    """Test _map_binary_state_to_semantic function.

    Tests mapping of binary sensor states ('on'/'off') to semantic states
    ('open'/'closed') for door and window sensors.
    """

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


class TestGetCorrelatableEntitiesByArea:
    """Test get_correlatable_entities_by_area function.

    Tests discovery of entities that can be analyzed for correlation:
    - Binary sensors (media, appliances, doors, windows)
    - Numeric sensors (temperature, humidity, etc.)
    - Exclusion of motion sensors (they define occupancy, not correlate with it)
    """

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
    """Test run_correlation_analysis function.

    Tests the high-level correlation analysis orchestration:
    - Analysis of binary sensors (uses binary likelihood analysis)
    - Analysis of numeric sensors (uses correlation analysis)
    - Error propagation (errors in one entity don't stop others)
    - Return results flag behavior
    """

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

    async def test_run_correlation_analysis_error_propagation(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that errors in individual entity analysis don't stop the entire analysis."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Add two entities: one that will succeed, one that will fail
        # Successful entity
        temp_type = EntityType(
            input_type=InputType.TEMPERATURE,
            weight=0.1,
            prob_given_true=0.5,
            prob_given_false=0.5,
        )
        temp_entity = Entity(
            entity_id="sensor.temperature",
            type=temp_type,
            prob_given_true=0.5,
            prob_given_false=0.5,
            decay=Decay(half_life=60.0),
            state_provider=lambda x: "20.0",
            last_updated=dt_util.utcnow(),
        )
        area.entities.entities["sensor.temperature"] = temp_entity

        # Entity that will fail (media player without intervals)
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

        # Create numeric samples for temperature sensor (will succeed)
        _create_numeric_entity_with_samples(
            db, area_name, "sensor.temperature", 100, lambda i: 20.0 + (i % 10)
        )
        intervals = [(now - timedelta(hours=50), now - timedelta(hours=40))]
        _create_occupied_intervals_cache(db, area_name, intervals)

        # Mock analyze_binary_likelihoods on the database instance to raise an error for media player
        # The function is called via coordinator.db.analyze_binary_likelihoods
        original_func = coordinator.db.analyze_binary_likelihoods

        def mock_analyze_binary_likelihoods(area_name, entity_id, *args, **kwargs):
            if entity_id == "media_player.tv":
                raise SQLAlchemyError("Database error for media player")
            # For other entities, call the real function
            return original_func(area_name, entity_id, *args, **kwargs)

        with patch.object(
            coordinator.db,
            "analyze_binary_likelihoods",
            side_effect=mock_analyze_binary_likelihoods,
        ):
            # Run analysis - should continue despite error
            results = await run_correlation_analysis(coordinator, return_results=True)

            # Should have results (may include other entities from the coordinator fixture)
            assert results is not None
            assert len(results) > 0

            # Find results for each entity we added
            temp_result = next(
                (r for r in results if r.get("entity_id") == "sensor.temperature"), None
            )
            media_result = next(
                (r for r in results if r.get("entity_id") == "media_player.tv"), None
            )

            # Temperature sensor should succeed
            assert temp_result is not None
            assert temp_result["success"] is True

            # Media player should have error recorded
            assert media_result is not None
            assert media_result["success"] is False
            assert "error" in media_result
            assert "Database error" in media_result["error"]
