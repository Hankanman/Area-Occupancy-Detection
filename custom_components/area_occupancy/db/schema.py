"""Database schema definitions for all tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from homeassistant.util import dt as dt_util

from .constants import (
    DEFAULT_ENTITY_PROB_GIVEN_FALSE,
    DEFAULT_ENTITY_PROB_GIVEN_TRUE,
    DEFAULT_ENTITY_WEIGHT,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase

    Base = DeclarativeBase
else:
    Base = declarative_base()


class Areas(Base):
    """A table to store the area occupancy information."""

    __tablename__ = "areas"
    entry_id = Column(
        String, nullable=False, index=True
    )  # Same for all areas in single integration
    area_name = Column(String, primary_key=True)  # Primary key - unique per integration
    area_id = Column(String, nullable=False)
    purpose = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)
    adjacent_areas = Column(JSON, nullable=True)  # JSON array of adjacent area names
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)
    entities = relationship("Entities", back_populates="area")
    priors = relationship("Priors", back_populates="area")

    def to_dict(self) -> dict[str, Any]:
        """Convert the ORM object to a dictionary."""
        return {
            "entry_id": self.entry_id,
            "area_name": self.area_name,
            "area_id": self.area_id,
            "purpose": self.purpose,
            "threshold": self.threshold,
            "adjacent_areas": self.adjacent_areas,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Create an Areas instance from a dictionary."""
        return cls(
            entry_id=data["entry_id"],
            area_name=data["area_name"],
            area_id=data["area_id"],
            purpose=data["purpose"],
            threshold=data["threshold"],
            adjacent_areas=data.get("adjacent_areas"),
            created_at=data.get("created_at", dt_util.utcnow()),
            updated_at=data.get("updated_at", dt_util.utcnow()),
        )


class Entities(Base):
    """A table to store the entity information."""

    __tablename__ = "entities"
    entry_id = Column(
        String, nullable=False, index=True
    )  # Same for all entities in single integration
    area_name = Column(
        String, ForeignKey("areas.area_name"), nullable=False, primary_key=True
    )
    entity_id = Column(String, primary_key=True)
    entity_type = Column(String, nullable=False)
    weight = Column(Float, nullable=False, default=DEFAULT_ENTITY_WEIGHT)
    prob_given_true = Column(
        Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_TRUE
    )
    prob_given_false = Column(
        Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_FALSE
    )
    is_shared = Column(
        Boolean, nullable=False, default=False
    )  # Entity shared across multiple areas
    shared_with_areas = Column(
        JSON, nullable=True
    )  # JSON array of area names this entity is shared with
    last_updated = Column(
        DateTime(timezone=True), nullable=False, default=dt_util.utcnow
    )
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)
    is_decaying = Column(Boolean, nullable=False, default=False)
    decay_start = Column(DateTime(timezone=True), nullable=True)
    evidence = Column(Boolean, nullable=False, default=False)
    # Relationship removed - SQLite doesn't support composite FKs properly
    # Use manual joins in queries instead (see db/queries.py)
    area = relationship("Areas", back_populates="entities")

    __table_args__ = (
        Index("idx_entities_entry", "entry_id"),
        Index("idx_entities_type", "entry_id", "entity_type"),
        Index("idx_entities_area", "area_name"),
        Index("idx_entities_shared", "is_shared"),  # For filtering shared entities
        # Composite index for optimized queries filtering by entry_id, area_name, and entity_type
        Index(
            "idx_entities_entry_area_type",
            "entry_id",
            "area_name",
            "entity_type",
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the ORM object to a dictionary."""
        return {
            "entry_id": self.entry_id,
            "area_name": getattr(
                self, "area_name", None
            ),  # Handle existing rows that may not have area_name
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "weight": self.weight,
            "prob_given_true": self.prob_given_true,
            "prob_given_false": self.prob_given_false,
            "is_shared": getattr(self, "is_shared", False),
            "shared_with_areas": getattr(self, "shared_with_areas", None),
            "last_updated": self.last_updated,
            "created_at": self.created_at,
            "is_decaying": self.is_decaying,
            "decay_start": self.decay_start,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Create an Entities instance from a dictionary."""
        return cls(
            entry_id=data["entry_id"],
            area_name=data["area_name"],
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            weight=data.get("weight", DEFAULT_ENTITY_WEIGHT),
            prob_given_true=data.get("prob_given_true", DEFAULT_ENTITY_PROB_GIVEN_TRUE),
            prob_given_false=data.get(
                "prob_given_false", DEFAULT_ENTITY_PROB_GIVEN_FALSE
            ),
            is_shared=data.get("is_shared", False),
            shared_with_areas=data.get("shared_with_areas"),
            last_updated=data.get("last_updated", dt_util.utcnow()),
            created_at=data.get("created_at", dt_util.utcnow()),
            is_decaying=data.get("is_decaying", False),
            decay_start=data.get("decay_start"),
            evidence=data.get("evidence", False),
        )


class Priors(Base):
    """A table to store the area time priors."""

    __tablename__ = "priors"
    entry_id = Column(String, nullable=False, index=True)  # Same for all priors
    area_name = Column(
        String, ForeignKey("areas.area_name"), nullable=False, primary_key=True
    )
    day_of_week = Column(Integer, primary_key=True)
    time_slot = Column(Integer, primary_key=True)
    prior_value = Column(Float, nullable=False)
    data_points = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)  # Confidence in the prior calculation
    last_calculation_date = Column(
        DateTime(timezone=True), nullable=True
    )  # When prior was last calculated
    sample_period_start = Column(
        DateTime(timezone=True), nullable=True
    )  # Start of data period used
    sample_period_end = Column(
        DateTime(timezone=True), nullable=True
    )  # End of data period used
    calculation_method = Column(
        String, nullable=True
    )  # Method used (e.g., 'interval_analysis')
    last_updated = Column(
        DateTime(timezone=True), nullable=False, default=dt_util.utcnow
    )
    area = relationship("Areas", back_populates="priors")

    __table_args__ = (
        Index("idx_priors_entry", "entry_id"),
        Index("idx_priors_area", "area_name"),
        Index("idx_priors_entry_area", "entry_id", "area_name"),
        Index("idx_priors_day_slot", "day_of_week", "time_slot"),
        Index("idx_priors_last_updated", "last_updated"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the ORM object to a dictionary."""
        return {
            "entry_id": self.entry_id,
            "area_name": self.area_name,
            "day_of_week": self.day_of_week,
            "time_slot": self.time_slot,
            "prior_value": self.prior_value,
            "data_points": self.data_points,
            "confidence": getattr(self, "confidence", None),
            "last_calculation_date": getattr(self, "last_calculation_date", None),
            "sample_period_start": getattr(self, "sample_period_start", None),
            "sample_period_end": getattr(self, "sample_period_end", None),
            "calculation_method": getattr(self, "calculation_method", None),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Create a Priors instance from a dictionary."""
        return cls(
            entry_id=data["entry_id"],
            area_name=data["area_name"],
            day_of_week=data["day_of_week"],
            time_slot=data["time_slot"],
            prior_value=data["prior_value"],
            data_points=data["data_points"],
            confidence=data.get("confidence"),
            last_calculation_date=data.get("last_calculation_date"),
            sample_period_start=data.get("sample_period_start"),
            sample_period_end=data.get("sample_period_end"),
            calculation_method=data.get("calculation_method"),
            last_updated=data.get("last_updated", dt_util.utcnow()),
        )


class Intervals(Base):
    """A table to store the state intervals."""

    __tablename__ = "intervals"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)  # Same for all intervals
    area_name = Column(
        String, nullable=False, index=True
    )  # For efficient area-based queries
    # Note: Foreign key removed - SQLite doesn't support partial FKs for composite PKs
    # Relationships are validated at application level through joins
    entity_id = Column(String, nullable=False)
    state = Column(String, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    duration_seconds = Column(Float, nullable=False)
    aggregation_level = Column(
        String, nullable=False, default="raw"
    )  # 'raw', 'daily', 'weekly', 'monthly'
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)
    # Relationship removed - SQLite doesn't support composite FKs properly
    # Use manual joins in queries instead (see db/queries.py)

    # Add unique constraint on (entity_id, start_time, end_time, aggregation_level)
    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "start_time",
            "end_time",
            "aggregation_level",
            name="uq_intervals_entity_time_level",
        ),
        # Performance indexes
        Index("idx_intervals_entity", "entity_id"),
        Index("idx_intervals_area", "area_name"),  # For area-based queries
        Index("idx_intervals_area_time", "area_name", "start_time", "end_time"),
        Index("idx_intervals_entity_time", "entity_id", "start_time", "end_time"),
        Index("idx_intervals_start_time", "start_time"),
        Index("idx_intervals_end_time", "end_time"),
        Index(
            "idx_intervals_aggregation", "aggregation_level"
        ),  # For filtering by aggregation level
        # Composite index for optimized queries with entity join and filtering
        Index(
            "idx_intervals_entity_state_time",
            "entity_id",
            "state",
            "start_time",
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the ORM object to a dictionary."""
        return {
            "id": self.id,
            "entry_id": getattr(self, "entry_id", None),
            "area_name": getattr(self, "area_name", None),
            "entity_id": self.entity_id,
            "state": self.state,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "aggregation_level": getattr(self, "aggregation_level", "raw"),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Create an Intervals instance from a dictionary."""
        return cls(
            entry_id=data.get("entry_id", ""),
            area_name=data.get("area_name", ""),
            entity_id=data["entity_id"],
            state=data["state"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            duration_seconds=data["duration_seconds"],
            aggregation_level=data.get("aggregation_level", "raw"),
            created_at=data.get("created_at", dt_util.utcnow()),
        )


class Metadata(Base):
    """A table to store the metadata."""

    __tablename__ = "metadata"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)


class IntervalAggregates(Base):
    """A table to store aggregated interval statistics."""

    __tablename__ = "interval_aggregates"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    area_name = Column(String, nullable=False, index=True)
    # Note: Foreign key removed - SQLite doesn't support partial FKs for composite PKs
    # Relationships are validated at application level through joins
    entity_id = Column(String, nullable=False, index=True)
    aggregation_period = Column(
        String, nullable=False
    )  # 'daily', 'weekly', 'monthly', 'yearly'
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False)
    state = Column(String, nullable=False)
    interval_count = Column(Integer, nullable=False)
    total_duration_seconds = Column(Float, nullable=False)
    min_duration_seconds = Column(Float, nullable=True)
    max_duration_seconds = Column(Float, nullable=True)
    avg_duration_seconds = Column(Float, nullable=True)
    first_occurrence = Column(DateTime(timezone=True), nullable=True)
    last_occurrence = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "aggregation_period",
            "period_start",
            "state",
            name="uq_interval_aggregates_entity_period_state",
        ),
        Index(
            "idx_interval_aggregates_area_entity_period",
            "area_name",
            "entity_id",
            "aggregation_period",
            "period_start",
        ),
        Index(
            "idx_interval_aggregates_area_period",
            "area_name",
            "aggregation_period",
            "period_start",
        ),
    )


class OccupiedIntervalsCache(Base):
    """A table to store precomputed occupied intervals for fast prior calculations."""

    __tablename__ = "occupied_intervals_cache"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    area_name = Column(String, nullable=False, index=True)
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True), nullable=False, index=True)
    duration_seconds = Column(Float, nullable=False)
    calculation_date = Column(DateTime(timezone=True), nullable=False)
    data_source = Column(String, nullable=True)  # 'motion_sensors', 'merged'
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "area_name",
            "start_time",
            "end_time",
            name="uq_occupied_intervals_cache_area_time",
        ),
        Index(
            "idx_occupied_intervals_cache_area_time",
            "area_name",
            "start_time",
            "end_time",
        ),
        Index(
            "idx_occupied_intervals_cache_area_start",
            "area_name",
            "start_time",
        ),
    )


class GlobalPriors(Base):
    """A table to store global prior values with underlying data."""

    __tablename__ = "global_priors"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    area_name = Column(String, nullable=False, unique=True, index=True)
    prior_value = Column(Float, nullable=False)
    calculation_date = Column(DateTime(timezone=True), nullable=False, index=True)
    data_period_start = Column(DateTime(timezone=True), nullable=False)
    data_period_end = Column(DateTime(timezone=True), nullable=False)
    total_occupied_seconds = Column(Float, nullable=False)
    total_period_seconds = Column(Float, nullable=False)
    interval_count = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)
    calculation_method = Column(String, nullable=True)
    underlying_data_hash = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=dt_util.utcnow,
        onupdate=dt_util.utcnow,
    )


class NumericSamples(Base):
    """A table to store raw numeric sensor samples for correlation analysis."""

    __tablename__ = "numeric_samples"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    area_name = Column(String, nullable=False, index=True)
    # Note: Foreign key removed - SQLite doesn't support partial FKs for composite PKs
    # Relationships are validated at application level through joins
    entity_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit_of_measurement = Column(String, nullable=True)
    state = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "timestamp",
            name="uq_numeric_samples_entity_timestamp",
        ),
        Index(
            "idx_numeric_samples_area_entity_time",
            "area_name",
            "entity_id",
            "timestamp",
        ),
        Index(
            "idx_numeric_samples_entity_time",
            "entity_id",
            "timestamp",
        ),
    )


class NumericAggregates(Base):
    """A table to store aggregated numeric sensor data for trend analysis."""

    __tablename__ = "numeric_aggregates"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    area_name = Column(String, nullable=False, index=True)
    # Note: Foreign key removed - SQLite doesn't support partial FKs for composite PKs
    # Relationships are validated at application level through joins
    entity_id = Column(String, nullable=False, index=True)
    aggregation_period = Column(
        String, nullable=False
    )  # 'hourly', 'daily', 'weekly', 'monthly', 'yearly'
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    avg_value = Column(Float, nullable=True)
    median_value = Column(Float, nullable=True)
    sample_count = Column(Integer, nullable=False)
    first_value = Column(Float, nullable=True)
    last_value = Column(Float, nullable=True)
    std_deviation = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "aggregation_period",
            "period_start",
            name="uq_numeric_aggregates_entity_period",
        ),
        Index(
            "idx_numeric_aggregates_area_entity_period",
            "area_name",
            "entity_id",
            "aggregation_period",
            "period_start",
        ),
        Index(
            "idx_numeric_aggregates_entity_period",
            "entity_id",
            "aggregation_period",
            "period_start",
        ),
    )


class NumericCorrelations(Base):
    """A table to store calculated correlations between numeric sensor values and occupancy."""

    __tablename__ = "numeric_correlations"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    area_name = Column(String, nullable=False, index=True)
    # Note: Foreign key removed - SQLite doesn't support partial FKs for composite PKs
    # Relationships are validated at application level through joins
    entity_id = Column(String, nullable=False, index=True)
    correlation_coefficient = Column(
        Float, nullable=False
    )  # Pearson correlation (-1 to 1)
    correlation_type = Column(
        String, nullable=True
    )  # 'occupancy_positive', 'occupancy_negative', 'none'
    analysis_period_start = Column(DateTime(timezone=True), nullable=False)
    analysis_period_end = Column(DateTime(timezone=True), nullable=False)
    sample_count = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)
    mean_value_when_occupied = Column(Float, nullable=True)
    mean_value_when_unoccupied = Column(Float, nullable=True)
    std_dev_when_occupied = Column(Float, nullable=True)
    std_dev_when_unoccupied = Column(Float, nullable=True)
    threshold_active = Column(Float, nullable=True)
    threshold_inactive = Column(Float, nullable=True)
    rejection_reason = Column(
        String, nullable=True
    )  # Reason why correlation was rejected (e.g., 'no_correlation', 'too_few_samples')
    calculation_date = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=dt_util.utcnow,
        onupdate=dt_util.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "area_name",
            "entity_id",
            "analysis_period_start",
            name="uq_numeric_correlations_area_entity_period",
        ),
        Index(
            "idx_numeric_correlations_area_entity_date",
            "area_name",
            "entity_id",
            "calculation_date",
        ),
        Index(
            "idx_numeric_correlations_type_confidence",
            "correlation_type",
            "confidence",
        ),
    )


class EntityStatistics(Base):
    """A table to store per-entity operational and Bayesian statistics."""

    __tablename__ = "entity_statistics"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    area_name = Column(String, nullable=False, index=True)
    # Note: Foreign key removed - SQLite doesn't support partial FKs for composite PKs
    # Relationships are validated at application level through joins
    entity_id = Column(String, nullable=False, index=True)
    statistic_type = Column(String, nullable=False)  # 'operational' or 'bayesian'
    statistic_name = Column(
        String, nullable=False
    )  # e.g., 'total_activations', 'prob_given_true'
    statistic_value = Column(Float, nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=dt_util.utcnow,
        onupdate=dt_util.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "statistic_type",
            "statistic_name",
            "period_start",
            name="uq_entity_statistics_entity_type_name_period",
        ),
        Index(
            "idx_entity_statistics_area_entity_type_name",
            "area_name",
            "entity_id",
            "statistic_type",
            "statistic_name",
            "period_start",
        ),
        Index(
            "idx_entity_statistics_entity_type_name_end",
            "entity_id",
            "statistic_type",
            "statistic_name",
            "period_end",
        ),
    )


class AreaRelationships(Base):
    """A table to define and track relationships between areas."""

    __tablename__ = "area_relationships"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    area_name = Column(String, nullable=False, index=True)  # Source area
    related_area_name = Column(
        String, nullable=False, index=True
    )  # Related/adjacent area
    relationship_type = Column(
        String, nullable=False
    )  # 'adjacent', 'shared_wall', 'shared_entrance', etc.
    influence_weight = Column(
        Float, nullable=False
    )  # How much this area influences related area (0.0 to 1.0)
    distance = Column(Float, nullable=True)  # Physical distance if applicable
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=dt_util.utcnow,
        onupdate=dt_util.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "area_name",
            "related_area_name",
            name="uq_area_relationships_area_related",
        ),
        Index(
            "idx_area_relationships_area_related",
            "area_name",
            "related_area_name",
        ),
        Index(
            "idx_area_relationships_related_area",
            "related_area_name",
            "area_name",
        ),
    )


class CrossAreaStats(Base):
    """A table to store aggregated statistics that span multiple areas."""

    __tablename__ = "cross_area_stats"
    id = Column(Integer, primary_key=True)
    entry_id = Column(String, nullable=False, index=True)
    statistic_type = Column(
        String, nullable=False
    )  # e.g., 'combined_occupancy', 'shared_sensor_active'
    statistic_name = Column(String, nullable=False)
    involved_areas = Column(JSON, nullable=True)  # JSON array of area names
    aggregation_period = Column(
        String, nullable=False
    )  # 'hourly', 'daily', 'weekly', 'monthly'
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False)
    statistic_value = Column(Float, nullable=False)
    extra_metadata = Column(
        JSON, nullable=True
    )  # Additional metadata (renamed from 'metadata' - reserved by SQLAlchemy)
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt_util.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "statistic_type",
            "statistic_name",
            "aggregation_period",
            "period_start",
            name="uq_cross_area_stats_type_name_period",
        ),
        Index(
            "idx_cross_area_stats_type_period",
            "statistic_type",
            "aggregation_period",
            "period_start",
        ),
    )
