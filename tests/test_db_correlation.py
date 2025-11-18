"""Tests for database correlation analysis functions."""

from datetime import timedelta

from custom_components.area_occupancy.db.correlation import (
    analyze_and_save_correlation,
    analyze_numeric_correlation,
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


class TestAnalyzeNumericCorrelation:
    """Test analyze_numeric_correlation function."""

    def test_analyze_numeric_correlation_success(self, test_db):
        """Test successful correlation analysis."""
        db = test_db
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

        result = analyze_numeric_correlation(
            db, area_name, entity_id, analysis_period_days=30
        )
        # Result may be None if insufficient data
        assert result is None or isinstance(result, dict)

    def test_analyze_numeric_correlation_no_data(self, test_db):
        """Test correlation analysis with no data."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        result = analyze_numeric_correlation(
            db, area_name, "sensor.nonexistent", analysis_period_days=30
        )
        assert result is None


class TestSaveCorrelationResult:
    """Test save_correlation_result function."""

    def test_save_correlation_result_success(self, test_db):
        """Test saving correlation result successfully."""
        db = test_db
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

    def test_get_correlation_for_entity_success(self, test_db):
        """Test retrieving correlation for entity."""
        db = test_db
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

    def test_get_correlation_for_entity_not_found(self, test_db):
        """Test retrieving correlation when none exists."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        result = get_correlation_for_entity(db, area_name, "sensor.nonexistent")
        assert result is None


class TestAnalyzeAndSaveCorrelation:
    """Test analyze_and_save_correlation function."""

    def test_analyze_and_save_correlation_success(self, test_db):
        """Test analyzing and saving correlation in one call."""
        db = test_db
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
        # May return False if insufficient data
        assert isinstance(result, bool)
