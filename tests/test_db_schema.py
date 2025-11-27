"""Tests for database schema and ORM models."""

from datetime import timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from custom_components.area_occupancy.db.schema import (
    AreaRelationships,
    Areas,
    Correlations,
    CrossAreaStats,
    Entities,
    EntityStatistics,
    GlobalPriors,
    IntervalAggregates,
    Intervals,
    Metadata,
    NumericAggregates,
    NumericSamples,
    OccupiedIntervalsCache,
    Priors,
)
from homeassistant.util import dt as dt_util


# Helper functions for creating test data
def create_test_area_data(entry_id="test_entry_001", **overrides):
    """Create standardized test area data."""
    data = {
        "entry_id": entry_id,
        "area_name": "Test Living Room",
        "area_id": "test_living_room",
        "purpose": "living",
        "threshold": 0.5,
        "area_prior": 0.3,
        "created_at": dt_util.utcnow(),
        "updated_at": dt_util.utcnow(),
    }
    data.update(overrides)
    return data


def create_test_entity_data(
    entry_id="test_entry_001",
    entity_id="binary_sensor.motion_1",
    area_name="Test Living Room",
    **overrides,
):
    """Create standardized test entity data."""
    data = {
        "entry_id": entry_id,
        "area_name": area_name,
        "entity_id": entity_id,
        "entity_type": "motion",
        "weight": 0.85,
        "prob_given_true": 0.8,
        "prob_given_false": 0.1,
        "last_updated": dt_util.utcnow(),
        "created_at": dt_util.utcnow(),
    }
    data.update(overrides)
    return data


def create_test_prior_data(
    entry_id="test_entry_001", area_name="Test Living Room", **overrides
):
    """Create standardized test prior data."""
    data = {
        "entry_id": entry_id,
        "area_name": area_name,
        "day_of_week": 1,  # Monday
        "time_slot": 14,  # 2 PM
        "prior_value": 0.35,
        "data_points": 10,
        "last_updated": dt_util.utcnow(),
    }
    data.update(overrides)
    return data


def create_test_interval_data(
    entity_id="binary_sensor.motion_1",
    entry_id="test_entry_001",
    area_name="Test Living Room",
    **overrides,
):
    """Create standardized test interval data."""
    start_time = dt_util.utcnow()
    end_time = start_time + timedelta(hours=1)
    data = {
        "entry_id": entry_id,
        "area_name": area_name,
        "entity_id": entity_id,
        "state": "on",
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": 3600.0,
        "created_at": dt_util.utcnow(),
    }
    data.update(overrides)
    return data


class TestAreasModel:
    """Test Areas model."""

    def test_areas_creation(self, db_session: Session):
        """Test creating an Areas instance."""
        area_data = create_test_area_data()
        area = Areas.from_dict(area_data)
        db_session.add(area)
        db_session.commit()

        assert area.entry_id == "test_entry_001"
        assert area.area_name == "Test Living Room"
        assert area.threshold == 0.5

    def test_areas_to_dict(self, db_session: Session):
        """Test Areas.to_dict method."""
        area_data = create_test_area_data()
        area = Areas.from_dict(area_data)
        db_session.add(area)
        db_session.commit()

        result = area.to_dict()
        assert result["entry_id"] == "test_entry_001"
        assert result["area_name"] == "Test Living Room"
        assert result["threshold"] == 0.5

    def test_areas_unique_constraint(self, db_session: Session):
        """Test that area_name is unique (primary key)."""
        area1 = Areas.from_dict(create_test_area_data())
        db_session.add(area1)
        db_session.commit()

        # Expunge area1 from session to prevent SQLAlchemy from merging instances
        db_session.expunge(area1)

        # Try to create duplicate area_name
        area2 = Areas.from_dict(create_test_area_data())
        db_session.add(area2)

        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_areas_defaults(self, db_session: Session):
        """Test Areas default values."""
        area = Areas(
            entry_id="test",
            area_name="Test Area",
            area_id="test_id",
            purpose="living",
            threshold=0.5,
        )
        db_session.add(area)
        db_session.commit()

        assert area.created_at is not None
        assert area.updated_at is not None


class TestEntitiesModel:
    """Test Entities model."""

    def test_entities_creation(self, db_session: Session):
        """Test creating an Entities instance."""
        # Create area first (foreign key requirement)
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity_data = create_test_entity_data()
        entity = Entities.from_dict(entity_data)
        db_session.add(entity)
        db_session.commit()

        assert entity.entity_id == "binary_sensor.motion_1"
        assert entity.entity_type == "motion"
        assert entity.weight == 0.85

    def test_entities_to_dict(self, db_session: Session):
        """Test Entities.to_dict method."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity_data = create_test_entity_data()
        entity = Entities.from_dict(entity_data)
        db_session.add(entity)
        db_session.commit()

        result = entity.to_dict()
        assert result["entity_id"] == "binary_sensor.motion_1"
        assert result["entity_type"] == "motion"

    def test_entities_foreign_key(self, db_session: Session):
        """Test Entities foreign key constraint."""
        entity_data = create_test_entity_data()
        entity = Entities.from_dict(entity_data)
        db_session.add(entity)

        # Should fail because area doesn't exist
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_entities_unique_constraint(self, db_session: Session):
        """Test Entities unique constraint on (area_name, entity_id)."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity1 = Entities.from_dict(create_test_entity_data())
        db_session.add(entity1)
        db_session.commit()

        # Expunge entity1 from session to prevent SQLAlchemy from merging instances
        db_session.expunge(entity1)

        # Try to create duplicate with same (area_name, entity_id)
        entity2 = Entities.from_dict(
            create_test_entity_data()
        )  # Same area_name and entity_id
        db_session.add(entity2)

        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_entities_defaults(self, db_session: Session):
        """Test Entities default values."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities(
            entry_id="test",
            area_name="Test Living Room",
            entity_id="sensor.test",
            entity_type="motion",
        )
        db_session.add(entity)
        db_session.commit()

        assert entity.weight == 0.85  # DEFAULT_ENTITY_WEIGHT
        assert entity.prob_given_true == 0.8  # DEFAULT_ENTITY_PROB_GIVEN_TRUE
        assert entity.prob_given_false == 0.05  # DEFAULT_ENTITY_PROB_GIVEN_FALSE
        assert entity.is_shared is False
        assert entity.is_decaying is False
        assert entity.evidence is False


class TestPriorsModel:
    """Test Priors model."""

    def test_priors_creation(self, db_session: Session):
        """Test creating a Priors instance."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        prior_data = create_test_prior_data()
        prior = Priors.from_dict(prior_data)
        db_session.add(prior)
        db_session.commit()

        assert prior.day_of_week == 1
        assert prior.time_slot == 14
        assert prior.prior_value == 0.35

    def test_priors_to_dict(self, db_session: Session):
        """Test Priors.to_dict method."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        prior_data = create_test_prior_data()
        prior = Priors.from_dict(prior_data)
        db_session.add(prior)
        db_session.commit()

        result = prior.to_dict()
        assert result["day_of_week"] == 1
        assert result["time_slot"] == 14

    def test_priors_foreign_key(self, db_session: Session):
        """Test Priors foreign key constraint."""
        prior_data = create_test_prior_data()
        prior = Priors.from_dict(prior_data)
        db_session.add(prior)

        # Should fail because area doesn't exist
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_priors_unique_constraint(self, db_session: Session):
        """Test Priors unique constraint on (area_name, day_of_week, time_slot)."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        prior1 = Priors.from_dict(create_test_prior_data())
        db_session.add(prior1)
        db_session.commit()

        # Expunge prior1 from session to prevent SQLAlchemy from merging instances
        db_session.expunge(prior1)

        # Try to create duplicate
        prior2 = Priors.from_dict(create_test_prior_data())
        db_session.add(prior2)

        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestIntervalsModel:
    """Test Intervals model."""

    def test_intervals_creation(self, db_session: Session):
        """Test creating an Intervals instance."""
        # Create area and entity first
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        interval_data = create_test_interval_data()
        interval = Intervals.from_dict(interval_data)
        db_session.add(interval)
        db_session.commit()

        assert interval.entity_id == "binary_sensor.motion_1"
        assert interval.state == "on"
        assert interval.duration_seconds == 3600.0

    def test_intervals_to_dict(self, db_session: Session):
        """Test Intervals.to_dict method."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        interval_data = create_test_interval_data()
        interval = Intervals.from_dict(interval_data)
        db_session.add(interval)
        db_session.commit()

        result = interval.to_dict()
        assert result["entity_id"] == "binary_sensor.motion_1"
        assert result["state"] == "on"

    def test_intervals_foreign_key(self, db_session: Session):
        """Test Intervals foreign key constraint - removed, FK constraint no longer enforced."""
        # Foreign key constraint was removed due to SQLite composite PK limitations
        # Relationships are validated at application level through joins
        # This test is kept for documentation but will not raise IntegrityError
        interval_data = create_test_interval_data()
        interval = Intervals.from_dict(interval_data)
        db_session.add(interval)

        # Without FK constraint, commit will succeed
        # Application-level validation should catch invalid entity references
        db_session.commit()
        db_session.rollback()

    def test_intervals_unique_constraint(self, db_session: Session):
        """Test Intervals unique constraint."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        # Use fixed timestamps to create actual duplicates
        fixed_start = dt_util.utcnow()
        fixed_end = fixed_start + timedelta(hours=1)
        interval1 = Intervals.from_dict(
            create_test_interval_data(start_time=fixed_start, end_time=fixed_end)
        )
        db_session.add(interval1)
        db_session.commit()

        # Expunge interval1 from session to prevent SQLAlchemy from merging instances
        db_session.expunge(interval1)

        # Try to create duplicate with same entity_id, start_time, end_time, aggregation_level
        interval2 = Intervals.from_dict(
            create_test_interval_data(start_time=fixed_start, end_time=fixed_end)
        )
        db_session.add(interval2)

        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_intervals_default_aggregation_level(self, db_session: Session):
        """Test Intervals default aggregation_level."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        interval = Intervals.from_dict(create_test_interval_data())
        db_session.add(interval)
        db_session.commit()

        assert interval.aggregation_level == "raw"


class TestMetadataModel:
    """Test Metadata model."""

    def test_metadata_creation(self, db_session: Session):
        """Test creating a Metadata instance."""
        metadata = Metadata(key="test_key", value="test_value")
        db_session.add(metadata)
        db_session.commit()

        assert metadata.key == "test_key"
        assert metadata.value == "test_value"

    def test_metadata_unique_constraint(self, db_session: Session):
        """Test Metadata unique constraint on key."""
        metadata1 = Metadata(key="test_key", value="value1")
        db_session.add(metadata1)
        db_session.commit()

        # Expunge metadata1 from session to prevent SQLAlchemy from merging instances
        db_session.expunge(metadata1)

        metadata2 = Metadata(key="test_key", value="value2")
        db_session.add(metadata2)

        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestOtherModels:
    """Test other schema models."""

    def test_interval_aggregates_creation(self, db_session: Session):
        """Test creating IntervalAggregates instance."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        aggregate = IntervalAggregates(
            entry_id="test",
            area_name="Test Living Room",
            entity_id="binary_sensor.motion_1",
            aggregation_period="daily",
            period_start=dt_util.utcnow(),
            period_end=dt_util.utcnow() + timedelta(days=1),
            state="on",
            interval_count=10,
            total_duration_seconds=3600.0,
        )
        db_session.add(aggregate)
        db_session.commit()

        assert aggregate.aggregation_period == "daily"
        assert aggregate.interval_count == 10

    def test_occupied_intervals_cache_creation(self, db_session: Session):
        """Test creating OccupiedIntervalsCache instance."""
        # Create area first for data integrity
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        cache = OccupiedIntervalsCache(
            entry_id="test",
            area_name="Test Living Room",
            start_time=dt_util.utcnow(),
            end_time=dt_util.utcnow() + timedelta(hours=1),
            duration_seconds=3600.0,
            calculation_date=dt_util.utcnow(),
            data_source="merged",
        )
        db_session.add(cache)
        db_session.commit()

        assert cache.data_source == "merged"
        assert cache.duration_seconds == 3600.0

    def test_global_priors_creation(self, db_session: Session):
        """Test creating GlobalPriors instance."""
        # Create area first for data integrity
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        global_prior = GlobalPriors(
            entry_id="test",
            area_name="Test Living Room",
            prior_value=0.5,
            calculation_date=dt_util.utcnow(),
            data_period_start=dt_util.utcnow() - timedelta(days=30),
            data_period_end=dt_util.utcnow(),
            total_occupied_seconds=7200.0,
            total_period_seconds=2592000.0,
            interval_count=100,
        )
        db_session.add(global_prior)
        db_session.commit()

        assert global_prior.prior_value == 0.5
        assert global_prior.area_name == "Test Living Room"

    def test_global_priors_unique_constraint(self, db_session: Session):
        """Test GlobalPriors unique constraint on area_name."""
        # Create area first for data integrity
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        global_prior1 = GlobalPriors(
            entry_id="test",
            area_name="Test Living Room",
            prior_value=0.5,
            calculation_date=dt_util.utcnow(),
            data_period_start=dt_util.utcnow() - timedelta(days=30),
            data_period_end=dt_util.utcnow(),
            total_occupied_seconds=7200.0,
            total_period_seconds=2592000.0,
            interval_count=100,
        )
        db_session.add(global_prior1)
        db_session.commit()

        # Expunge global_prior1 from session to prevent SQLAlchemy from merging instances
        db_session.expunge(global_prior1)

        global_prior2 = GlobalPriors(
            entry_id="test",
            area_name="Test Living Room",
            prior_value=0.6,
            calculation_date=dt_util.utcnow(),
            data_period_start=dt_util.utcnow() - timedelta(days=30),
            data_period_end=dt_util.utcnow(),
            total_occupied_seconds=7200.0,
            total_period_seconds=2592000.0,
            interval_count=100,
        )
        db_session.add(global_prior2)

        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_numeric_samples_creation(self, db_session: Session):
        """Test creating NumericSamples instance."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        sample = NumericSamples(
            entry_id="test",
            area_name="Test Living Room",
            entity_id="binary_sensor.motion_1",
            timestamp=dt_util.utcnow(),
            value=25.5,
            unit_of_measurement="°C",
            state="25.5",
        )
        db_session.add(sample)
        db_session.commit()

        assert sample.value == 25.5
        assert sample.unit_of_measurement == "°C"

    def test_numeric_aggregates_creation(self, db_session: Session):
        """Test creating NumericAggregates instance."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        aggregate = NumericAggregates(
            entry_id="test",
            area_name="Test Living Room",
            entity_id="binary_sensor.motion_1",
            aggregation_period="daily",
            period_start=dt_util.utcnow(),
            period_end=dt_util.utcnow() + timedelta(days=1),
            min_value=20.0,
            max_value=30.0,
            avg_value=25.0,
            sample_count=100,
        )
        db_session.add(aggregate)
        db_session.commit()

        assert aggregate.aggregation_period == "daily"
        assert aggregate.avg_value == 25.0

    def test_numeric_correlations_creation(self, db_session: Session):
        """Test creating Correlations instance."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        correlation = Correlations(
            entry_id="test",
            area_name="Test Living Room",
            entity_id="binary_sensor.motion_1",
            input_type="motion",
            correlation_coefficient=0.8,
            correlation_type="occupancy_positive",
            analysis_period_start=dt_util.utcnow() - timedelta(days=30),
            analysis_period_end=dt_util.utcnow(),
            sample_count=1000,
            calculation_date=dt_util.utcnow(),
        )
        db_session.add(correlation)
        db_session.commit()

        assert correlation.correlation_coefficient == 0.8
        assert correlation.correlation_type == "occupancy_positive"

    def test_entity_statistics_creation(self, db_session: Session):
        """Test creating EntityStatistics instance."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        stat = EntityStatistics(
            entry_id="test",
            area_name="Test Living Room",
            entity_id="binary_sensor.motion_1",
            statistic_type="operational",
            statistic_name="total_activations",
            statistic_value=100.0,
            period_start=dt_util.utcnow() - timedelta(days=30),
            period_end=dt_util.utcnow(),
        )
        db_session.add(stat)
        db_session.commit()

        assert stat.statistic_type == "operational"
        assert stat.statistic_name == "total_activations"

    def test_area_relationships_creation(self, db_session: Session):
        """Test creating AreaRelationships instance."""
        area1 = Areas.from_dict(create_test_area_data(area_name="Area1"))
        db_session.add(area1)
        area2 = Areas.from_dict(create_test_area_data(area_name="Area2"))
        db_session.add(area2)
        db_session.commit()

        relationship = AreaRelationships(
            entry_id="test",
            area_name="Area1",
            related_area_name="Area2",
            relationship_type="adjacent",
            influence_weight=0.5,
            distance=10.0,
        )
        db_session.add(relationship)
        db_session.commit()

        assert relationship.relationship_type == "adjacent"
        assert relationship.influence_weight == 0.5

    def test_cross_area_stats_creation(self, db_session: Session):
        """Test creating CrossAreaStats instance."""
        stat = CrossAreaStats(
            entry_id="test",
            statistic_type="combined_occupancy",
            statistic_name="total_occupied_time",
            involved_areas=["Area1", "Area2"],
            aggregation_period="daily",
            period_start=dt_util.utcnow(),
            period_end=dt_util.utcnow() + timedelta(days=1),
            statistic_value=7200.0,
        )
        db_session.add(stat)
        db_session.commit()

        assert stat.statistic_type == "combined_occupancy"
        assert stat.involved_areas == ["Area1", "Area2"]


class TestModelRelationships:
    """Test ORM relationships between models."""

    def test_areas_entities_relationship(self, db_session: Session):
        """Test relationship between Areas and Entities."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        # Refresh to load relationships
        db_session.refresh(area)
        db_session.refresh(entity)

        assert entity in area.entities
        assert entity.area == area

    def test_areas_priors_relationship(self, db_session: Session):
        """Test relationship between Areas and Priors."""
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        prior = Priors.from_dict(create_test_prior_data())
        db_session.add(prior)
        db_session.commit()

        # Refresh to load relationships
        db_session.refresh(area)
        db_session.refresh(prior)

        assert prior in area.priors
        assert prior.area == area

    def test_entities_intervals_relationship(self, db_session: Session):
        """Test relationship between Entities and Intervals - relationships removed."""
        # Relationships were removed due to SQLite composite FK limitations
        # Use manual joins in queries instead (see db/queries.py)
        area = Areas.from_dict(create_test_area_data())
        db_session.add(area)
        db_session.commit()

        entity = Entities.from_dict(create_test_entity_data())
        db_session.add(entity)
        db_session.commit()

        interval = Intervals.from_dict(create_test_interval_data())
        db_session.add(interval)
        db_session.commit()

        # Verify data was stored correctly (relationships removed, so can't test them)
        assert interval.entity_id == entity.entity_id
        assert interval.area_name == entity.area_name
        # Relationships are validated through manual joins in application code
