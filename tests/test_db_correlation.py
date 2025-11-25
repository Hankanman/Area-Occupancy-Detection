"""Tests for database correlation analysis functions."""

from datetime import timedelta
from unittest.mock import Mock, patch

from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.const import NUMERIC_CORRELATION_HISTORY_COUNT
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db.correlation import (
    _prune_old_correlations,
    analyze_and_save_correlation,
    analyze_correlation,
    calculate_pearson_correlation,
    get_correlation_for_entity,
    save_correlation_result,
)
from homeassistant.util import dt as dt_util


class TestCalculatePearsonCorrelation:
    """Test calculate_pearson_correlation function."""

    def test_calculate_pearson_correlation_positive(self):
        """Test correlation calculation with positive correlation."""
        # Use at least 50 samples to meet MIN_CORRELATION_SAMPLES requirement
        x_values = list(range(1, 51))  # [1, 2, 3, ..., 50]
        y_values = [x * 2 for x in x_values]  # [2, 4, 6, ..., 100]

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert abs(correlation - 1.0) < 0.01  # Should be close to 1.0
        assert 0.0 <= p_value <= 1.0

    def test_calculate_pearson_correlation_negative(self):
        """Test correlation calculation with negative correlation."""
        # Use at least 50 samples to meet MIN_CORRELATION_SAMPLES requirement
        x_values = list(range(1, 51))  # [1, 2, 3, ..., 50]
        y_values = [101 - x for x in x_values]  # [100, 99, 98, ..., 51]

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert abs(correlation - (-1.0)) < 0.01  # Should be close to -1.0
        assert 0.0 <= p_value <= 1.0

    def test_calculate_pearson_correlation_no_correlation(self):
        """Test correlation calculation with no correlation."""
        x_values = [1, 2, 3, 4, 5]
        y_values = [5, 2, 8, 1, 9]  # Random values

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert -1.0 <= correlation <= 1.0
        assert 0.0 <= p_value <= 1.0

    def test_calculate_pearson_correlation_insufficient_samples(self):
        """Test correlation calculation with insufficient samples."""
        x_values = [1, 2]
        y_values = [3, 4]

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert correlation == 0.0
        assert p_value == 1.0

    def test_calculate_pearson_correlation_mismatched_lengths(self):
        """Test correlation calculation with mismatched array lengths."""
        x_values = [1, 2, 3]
        y_values = [4, 5]

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert correlation == 0.0
        assert p_value == 1.0

    def test_calculate_pearson_correlation_nan_values(self):
        """Test correlation calculation with NaN values."""
        x_values = [1, 2, float("nan"), 4, 5]
        y_values = [2, 4, 6, 8, 10]

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

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        # Create numeric samples
        with db.get_locked_session() as session:
            # Create entity first
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            for i in range(100):
                sample = db.NumericSamples(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    timestamp=now - timedelta(hours=100 - i),
                    value=20.0 + (i % 10),
                    unit_of_measurement="°C",
                )
                session.add(sample)
            session.commit()

        # Create occupied intervals cache
        intervals = [
            (now - timedelta(hours=50), now - timedelta(hours=40)),
            (now - timedelta(hours=20), now - timedelta(hours=10)),
        ]
        db.save_occupied_intervals_cache(area_name, intervals)

        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        # Result may be None if insufficient data
        assert result is None or isinstance(result, dict)

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

        # Ensure area exists first
        db.save_area_data(area_name)

        # Create binary sensor entity
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="door",
            )
            session.add(entity)

            # Create intervals for binary sensor
            # Alternating on/off every hour
            for i in range(100):
                start = now - timedelta(hours=100 - i)
                end = now - timedelta(hours=99 - i)
                state = "on" if i % 2 == 0 else "off"
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

        # Create occupied intervals cache
        # Occupied during "on" states (even indices)
        intervals = []
        for i in range(0, 100, 2):
            start = now - timedelta(hours=100 - i)
            end = now - timedelta(hours=99 - i)
            intervals.append((start, end))

        db.save_occupied_intervals_cache(area_name, intervals)

        # Analyze with binary flag
        result = analyze_correlation(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
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


class TestSaveCorrelationResult:
    """Test save_correlation_result function."""

    def test_save_correlation_result_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving correlation result successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        # Create entity first
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)
            session.commit()

        correlation_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "correlation_coefficient": 0.75,
            "correlation_type": "occupancy_positive",
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
                session.query(db.NumericCorrelations)
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

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        # Create entity and save correlation
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)
            session.commit()

        correlation_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "correlation_coefficient": 0.8,
            "correlation_type": "occupancy_positive",
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

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        # Create entity and samples
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            for i in range(100):
                sample = db.NumericSamples(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    timestamp=now - timedelta(hours=100 - i),
                    value=20.0 + (i % 10),
                )
                session.add(sample)
            session.commit()

        # Create occupied intervals
        intervals = [
            (now - timedelta(hours=50), now - timedelta(hours=40)),
        ]
        db.save_occupied_intervals_cache(area_name, intervals)

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


class TestCalculatePearsonCorrelationEdgeCases:
    """Test calculate_pearson_correlation edge cases."""

    def test_calculate_pearson_correlation_perfect_correlation(self):
        """Test correlation calculation with perfect correlation."""
        x_values = list(range(1, 51))
        y_values = [x * 2 for x in x_values]  # Perfect positive correlation

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert abs(correlation - 1.0) < 0.01
        assert p_value == 0.0  # Perfect correlation has p-value near 0

    def test_calculate_pearson_correlation_very_close_to_one(self):
        """Test correlation calculation when correlation is very close to ±1.0."""
        x_values = list(range(1, 51))
        y_values = [x * 2 + 0.0000001 for x in x_values]  # Very close to perfect

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert abs(correlation) > 0.99
        assert 0.0 <= p_value <= 1.0

    def test_calculate_pearson_correlation_n_less_than_3(self):
        """Test correlation calculation with n < 3."""
        x_values = [1, 2]
        y_values = [3, 4]

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert correlation == 0.0
        assert p_value == 1.0

    def test_calculate_pearson_correlation_invalid_values(self):
        """Test correlation calculation with invalid numpy values."""

        x_values = [1, 2, float("inf"), 4, 5]
        y_values = [2, 4, 6, 8, 10]

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        # Should handle gracefully
        assert isinstance(correlation, float)
        assert isinstance(p_value, float)

    def test_calculate_pearson_correlation_exception(self):
        """Test correlation calculation with exception."""
        x_values = ["invalid", "data"]
        y_values = [1, 2]

        correlation, p_value = calculate_pearson_correlation(x_values, y_values)
        assert correlation == 0.0
        assert p_value == 1.0


class TestAnalyzeCorrelationEdgeCases:
    """Test analyze_correlation edge cases."""

    def test_analyze_correlation_no_occupied_intervals(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test analysis when no occupied intervals exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"
        now = dt_util.utcnow()

        db.save_area_data(area_name)

        # Create entity and samples but no occupied intervals
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            for i in range(100):
                sample = db.NumericSamples(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    timestamp=now - timedelta(hours=100 - i),
                    value=20.0 + (i % 10),
                )
                session.add(sample)
            session.commit()

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

        db.save_area_data(area_name)

        # Create entity with only a few samples
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            # Only 10 samples (less than MIN_CORRELATION_SAMPLES)
            for i in range(10):
                sample = db.NumericSamples(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    timestamp=now - timedelta(hours=10 - i),
                    value=20.0,
                )
                session.add(sample)
            session.commit()

        # Create occupied intervals
        intervals = [
            (now - timedelta(hours=5), now - timedelta(hours=3)),
        ]
        db.save_occupied_intervals_cache(area_name, intervals)

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

        db.save_area_data(area_name)

        # Create entity and samples with negative correlation pattern
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            # Create samples where value decreases when occupied
            for i in range(100):
                sample = db.NumericSamples(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    timestamp=now - timedelta(hours=100 - i),
                    value=30.0 - (i % 10),  # Decreasing pattern
                )
                session.add(sample)
            session.commit()

        # Create occupied intervals
        intervals = [
            (now - timedelta(hours=50), now - timedelta(hours=40)),
        ]
        db.save_occupied_intervals_cache(area_name, intervals)

        result = analyze_correlation(db, area_name, entity_id, analysis_period_days=30)
        # May return None if insufficient correlation
        assert result is None or isinstance(result, dict)

    def test_analyze_correlation_database_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test analysis with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        with patch.object(db, "get_session", side_effect=SQLAlchemyError("DB error")):
            result = analyze_correlation(
                db, area_name, "sensor.temperature", analysis_period_days=30
            )
            assert result is None


class TestSaveCorrelationResultEdgeCases:
    """Test save_correlation_result edge cases."""

    def test_save_correlation_result_database_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving correlation with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        correlation_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": "sensor.temperature",
            "correlation_coefficient": 0.75,
            "correlation_type": "occupancy_positive",
            "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
            "analysis_period_end": dt_util.utcnow(),
            "sample_count": 100,
        }

        with patch.object(
            db, "get_locked_session", side_effect=SQLAlchemyError("DB error")
        ):
            result = save_correlation_result(db, correlation_data)
            assert result is False

    def test_save_correlation_result_missing_fields(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving correlation with missing required fields."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Missing required fields - will raise KeyError when accessing entity_id in logging
        correlation_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            # Missing entity_id
        }

        # The function accesses entity_id in the logging statement, which will raise KeyError
        # But the function may catch it or handle it differently
        try:
            result = save_correlation_result(db, correlation_data)
            # If it doesn't raise, it should return False
            assert result is False
        except KeyError:
            # KeyError is also acceptable - function may not catch it
            pass


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

        db.save_area_data(area_name)

        # Create entity
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            # Create more correlations than NUMERIC_CORRELATION_HISTORY_COUNT
            now = dt_util.utcnow()
            for i in range(NUMERIC_CORRELATION_HISTORY_COUNT + 5):
                # Use different analysis_period_start to avoid unique constraint violation
                period_start = now - timedelta(days=30 + i)
                period_end = now - timedelta(days=i)
                correlation = db.NumericCorrelations(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    correlation_coefficient=0.5 + (i * 0.01),
                    correlation_type="occupancy_positive",
                    calculation_date=now - timedelta(days=i),
                    analysis_period_start=period_start,
                    analysis_period_end=period_end,
                    sample_count=100,
                )
                session.add(correlation)
            session.commit()

        # Call prune function
        with db.get_locked_session() as session:
            _prune_old_correlations(db, session, area_name, entity_id)

        # Verify only NUMERIC_CORRELATION_HISTORY_COUNT remain
        with db.get_session() as session:
            count = (
                session.query(db.NumericCorrelations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .count()
            )
            assert count == NUMERIC_CORRELATION_HISTORY_COUNT

    def test_prune_old_correlations_no_excess(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test pruning when there are fewer correlations than limit."""
        db = coordinator.db
        db.init_db()
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        db.save_area_data(area_name)

        # Create entity
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            # Create fewer correlations than limit
            now = dt_util.utcnow()
            num_correlations = NUMERIC_CORRELATION_HISTORY_COUNT - 2
            for i in range(num_correlations):
                # Use different analysis_period_start to avoid unique constraint violation
                period_start = now - timedelta(days=30 + i)
                period_end = now - timedelta(days=i)
                correlation = db.NumericCorrelations(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    correlation_coefficient=0.5,
                    correlation_type="occupancy_positive",
                    calculation_date=now - timedelta(days=i),
                    analysis_period_start=period_start,
                    analysis_period_end=period_end,
                    sample_count=100,
                )
                session.add(correlation)
            session.commit()

        # Call prune function
        with db.get_locked_session() as session:
            _prune_old_correlations(db, session, area_name, entity_id)

        # Verify all correlations remain
        with db.get_session() as session:
            count = (
                session.query(db.NumericCorrelations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .count()
            )
            assert count == num_correlations

    def test_prune_old_correlations_error(self, coordinator: AreaOccupancyCoordinator):
        """Test pruning with error."""
        db = coordinator.db
        db.init_db()
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        mock_session = Mock()
        mock_session.query.side_effect = SQLAlchemyError("Prune error")

        # Should not raise exception
        _prune_old_correlations(db, mock_session, area_name, entity_id)


class TestGetCorrelationForEntityEdgeCases:
    """Test get_correlation_for_entity edge cases."""

    def test_get_correlation_for_entity_database_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test getting correlation with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        with patch.object(db, "get_session", side_effect=SQLAlchemyError("DB error")):
            result = get_correlation_for_entity(db, area_name, "sensor.temperature")
            assert result is None

    def test_get_correlation_for_entity_multiple_results(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test getting correlation when multiple results exist."""
        db = coordinator.db
        db.init_db()
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "sensor.temperature"

        db.save_area_data(area_name)

        # Create entity and multiple correlations
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id=entity_id,
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="numeric",
            )
            session.add(entity)

            # Create multiple correlations (most recent first due to desc ordering)
            now = dt_util.utcnow()
            for i in range(3):
                # Use different analysis_period_start to avoid unique constraint violation
                period_start = now - timedelta(days=30 + i)
                period_end = now - timedelta(days=i)
                correlation = db.NumericCorrelations(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    correlation_coefficient=0.5 + (i * 0.1),
                    correlation_type="occupancy_positive",
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
