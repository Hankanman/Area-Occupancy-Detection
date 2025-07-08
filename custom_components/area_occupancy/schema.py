"""Database schema definitions using SQLAlchemy Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any

import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    schema,
)
from sqlalchemy.dialects import sqlite

from homeassistant.util import dt as dt_util

from .utils import StateInterval

# Database metadata
metadata = MetaData()

# Database schema version for migrations
DB_VERSION = 1

# ─────────────────── Global Tables ───────────────────

# Global entity master table (shared across all areas)
entities_table = Table(
    "entities",
    metadata,
    Column("entity_id", String, primary_key=True),
    Column("domain", String, nullable=False),  # (sensor, binary_sensor, etc.)
    Column("last_seen", DateTime, nullable=False, default=dt_util.utcnow),
    Column("created_at", DateTime, nullable=False, default=dt_util.utcnow),
)

# Global state intervals table (shared across all areas)
state_intervals_table = Table(
    "state_intervals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("entity_id", String, nullable=False),
    Column("state", String, nullable=False),
    Column("start_time", DateTime, nullable=False),
    Column("end_time", DateTime, nullable=False),
    Column("duration_seconds", Float, nullable=False),
    Column("created_at", DateTime, nullable=False, default=dt_util.utcnow),
    # Foreign key to entities table
    ForeignKey("entities.entity_id", name="fk_state_intervals_entity"),
    # Unique constraint to prevent duplicate intervals
    UniqueConstraint(
        "entity_id", "start_time", "end_time", name="uq_intervals_entity_time"
    ),
)

# Database metadata table
metadata_table = Table(
    "metadata",
    metadata,
    Column("key", String, primary_key=True),
    Column("value", String, nullable=False),
)

# ─────────────────── Area-Specific Tables ───────────────────

# Area configuration and state (one per integration instance)
area_occupancy_table = Table(
    "area_occupancy",
    metadata,
    Column("entry_id", String, primary_key=True),
    Column("area_name", String, nullable=False),
    Column("purpose", String, nullable=False),
    Column("probability", Float, nullable=False, default=0.0),
    Column("prior", Float, nullable=False, default=0.0),
    Column("threshold", Float, nullable=False, default=0.0),
    Column("occupied", Boolean, nullable=False, default=False),
    Column("created_at", DateTime, nullable=False, default=dt_util.utcnow),
    Column("updated_at", DateTime, nullable=False, default=dt_util.utcnow),
)

# Area-specific entity configuration and probability data
area_entity_config_table = Table(
    "area_entity_config",
    metadata,
    Column("entry_id", String, nullable=False),
    Column("entity_id", String, nullable=False),
    Column("entity_type", String, nullable=False),  # motion, media, door, etc.
    Column("weight", Float, nullable=False, default=1.0),
    Column("probability", Float, nullable=False, default=0.0),
    Column("prob_given_true", Float, nullable=False, default=0.5),
    Column("prob_given_false", Float, nullable=False, default=0.1),
    Column("last_state", String, nullable=True),
    Column("last_updated", DateTime, nullable=False, default=dt_util.utcnow),
    Column("attributes", Text, default="{}"),
    # Composite primary key
    sa.PrimaryKeyConstraint("entry_id", "entity_id", name="pk_area_entity_config"),
    # Foreign keys
    ForeignKey("area_occupancy.entry_id", name="fk_area_entity_entry"),
    ForeignKey("entities.entity_id", name="fk_area_entity_entity"),
)

# Area-specific history (probability changes per area)
area_history_table = Table(
    "area_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("entry_id", String, nullable=False),
    Column("entity_id", String, nullable=False),
    Column("probability_change", Float, nullable=False, default=0.0),
    Column("timestamp", DateTime, nullable=False, default=dt_util.utcnow),
    Column("context", Text, default="{}"),
    # Foreign keys
    ForeignKey("area_occupancy.entry_id", name="fk_area_history_entry"),
    ForeignKey("entities.entity_id", name="fk_area_history_entity"),
)

# Index definitions for performance
indexes = [
    # Global entity indexes
    sa.Index("idx_entities_domain", entities_table.c.domain),
    sa.Index("idx_entities_last_seen", entities_table.c.last_seen),
    # State intervals indexes (critical for performance)
    sa.Index("idx_state_intervals_entity", state_intervals_table.c.entity_id),
    sa.Index("idx_state_intervals_start_time", state_intervals_table.c.start_time),
    sa.Index("idx_state_intervals_end_time", state_intervals_table.c.end_time),
    sa.Index("idx_state_intervals_state", state_intervals_table.c.state),
    # Composite index for time range queries
    sa.Index(
        "idx_state_intervals_entity_time",
        state_intervals_table.c.entity_id,
        state_intervals_table.c.start_time,
        state_intervals_table.c.end_time,
    ),
    # Area occupancy indexes
    sa.Index("idx_area_occupancy_updated", area_occupancy_table.c.updated_at),
    # Area entity config indexes
    sa.Index("idx_area_entity_entry", area_entity_config_table.c.entry_id),
    sa.Index("idx_area_entity_entity", area_entity_config_table.c.entity_id),
    sa.Index("idx_area_entity_type", area_entity_config_table.c.entity_type),
    sa.Index("idx_area_entity_updated", area_entity_config_table.c.last_updated),
    # Area history indexes
    sa.Index("idx_area_history_entry", area_history_table.c.entry_id),
    sa.Index("idx_area_history_entity", area_history_table.c.entity_id),
    sa.Index("idx_area_history_timestamp", area_history_table.c.timestamp),
]


# ─────────────────── Data Models ───────────────────


@dataclass
class AreaOccupancyRecord:
    """Data class for area occupancy records."""

    entry_id: str = ""
    area_name: str = ""
    purpose: str = ""
    probability: float = 0.0
    prior: float = 0.0
    threshold: float = 0.0
    occupied: bool = False
    created_at: datetime = field(default_factory=dt_util.utcnow)
    updated_at: datetime = field(default_factory=dt_util.utcnow)


@dataclass
class EntityRecord:
    """Data class for global entity records."""

    entity_id: str = ""
    domain: str = ""
    last_seen: datetime = field(default_factory=dt_util.utcnow)
    created_at: datetime = field(default_factory=dt_util.utcnow)


@dataclass
class AreaEntityConfigRecord:
    """Data class for area-specific entity configuration."""

    entry_id: str = ""
    entity_id: str = ""
    entity_type: str = ""
    weight: float = 1.0
    probability: float = 0.0
    prob_given_true: float = 0.5
    prob_given_false: float = 0.1
    last_state: str | None = None
    last_updated: datetime = field(default_factory=dt_util.utcnow)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class AreaHistoryRecord:
    """Data class for area-specific historical data records."""

    id: int | None = None
    entry_id: str = ""
    entity_id: str = ""
    probability_change: float = 0.0
    timestamp: datetime = field(default_factory=dt_util.utcnow)
    context: dict[str, Any] = field(default_factory=dict)


# ─────────────────── Schema Converter ───────────────────
class SchemaConverter:
    """Convert between dataclass records and SQLAlchemy row objects."""

    @staticmethod
    def row_to_area_occupancy(row: Any) -> AreaOccupancyRecord:
        """Convert SQLAlchemy row to AreaOccupancyRecord."""
        return AreaOccupancyRecord(
            entry_id=row.entry_id,
            area_name=row.area_name,
            purpose=row.purpose,
            probability=row.probability,
            prior=row.prior,
            threshold=row.threshold,
            occupied=bool(row.occupied),
            created_at=row.created_at
            if isinstance(row.created_at, datetime)
            else dt_util.parse_datetime(row.created_at) or dt_util.utcnow(),
            updated_at=row.updated_at
            if isinstance(row.updated_at, datetime)
            else dt_util.parse_datetime(row.updated_at) or dt_util.utcnow(),
        )

    @staticmethod
    def area_occupancy_to_dict(record: AreaOccupancyRecord) -> dict[str, Any]:
        """Convert AreaOccupancyRecord to dictionary for database insertion."""
        return {
            "entry_id": record.entry_id,
            "area_name": record.area_name,
            "purpose": record.purpose,
            "probability": record.probability,
            "prior": record.prior,
            "threshold": record.threshold,
            "occupied": record.occupied,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def row_to_entity(row: Any) -> EntityRecord:
        """Convert SQLAlchemy row to EntityRecord."""
        return EntityRecord(
            entity_id=row.entity_id,
            domain=row.domain,
            last_seen=row.last_seen
            if isinstance(row.last_seen, datetime)
            else dt_util.parse_datetime(row.last_seen) or dt_util.utcnow(),
            created_at=row.created_at
            if isinstance(row.created_at, datetime)
            else dt_util.parse_datetime(row.created_at) or dt_util.utcnow(),
        )

    @staticmethod
    def entity_to_dict(record: EntityRecord) -> dict[str, Any]:
        """Convert EntityRecord to dictionary for database insertion."""
        return {
            "entity_id": record.entity_id,
            "domain": record.domain,
            "last_seen": record.last_seen,
            "created_at": record.created_at,
        }

    @staticmethod
    def row_to_area_entity_config(row: Any) -> AreaEntityConfigRecord:
        """Convert SQLAlchemy row to AreaEntityConfigRecord."""
        attributes = {}
        if row.attributes:
            try:
                attributes = json.loads(row.attributes)
            except (json.JSONDecodeError, TypeError):
                attributes = {}

        return AreaEntityConfigRecord(
            entry_id=row.entry_id,
            entity_id=row.entity_id,
            entity_type=row.entity_type,
            weight=row.weight,
            probability=row.probability,
            prob_given_true=row.prob_given_true,
            prob_given_false=row.prob_given_false,
            last_state=row.last_state,
            last_updated=row.last_updated
            if isinstance(row.last_updated, datetime)
            else dt_util.parse_datetime(row.last_updated) or dt_util.utcnow(),
            attributes=attributes,
        )

    @staticmethod
    def area_entity_config_to_dict(record: AreaEntityConfigRecord) -> dict[str, Any]:
        """Convert AreaEntityConfigRecord to dictionary for database insertion."""
        return {
            "entry_id": record.entry_id,
            "entity_id": record.entity_id,
            "entity_type": record.entity_type,
            "weight": record.weight,
            "probability": record.probability,
            "prob_given_true": record.prob_given_true,
            "prob_given_false": record.prob_given_false,
            "last_state": record.last_state,
            "last_updated": record.last_updated,
            "attributes": json.dumps(record.attributes),
        }

    @staticmethod
    def row_to_area_history(row: Any) -> AreaHistoryRecord:
        """Convert SQLAlchemy row to AreaHistoryRecord."""
        context = {}
        if row.context:
            try:
                context = json.loads(row.context)
            except (json.JSONDecodeError, TypeError):
                context = {}

        return AreaHistoryRecord(
            id=row.id,
            entry_id=row.entry_id,
            entity_id=row.entity_id,
            probability_change=row.probability_change,
            timestamp=row.timestamp
            if isinstance(row.timestamp, datetime)
            else dt_util.parse_datetime(row.timestamp) or dt_util.utcnow(),
            context=context,
        )

    @staticmethod
    def area_history_to_dict(record: AreaHistoryRecord) -> dict[str, Any]:
        """Convert AreaHistoryRecord to dictionary for database insertion."""
        return {
            "entry_id": record.entry_id,
            "entity_id": record.entity_id,
            "probability_change": record.probability_change,
            "timestamp": record.timestamp,
            "context": json.dumps(record.context),
        }

    @staticmethod
    def row_to_state_interval(row: Any) -> StateInterval:
        """Convert SQLAlchemy row to StateInterval."""
        return StateInterval(
            start=row.start_time
            if isinstance(row.start_time, datetime)
            else dt_util.parse_datetime(row.start_time) or dt_util.utcnow(),
            end=row.end_time
            if isinstance(row.end_time, datetime)
            else dt_util.parse_datetime(row.end_time) or dt_util.utcnow(),
            state=row.state,
            entity_id=row.entity_id,
        )

    @staticmethod
    def state_interval_to_dict(record: StateInterval) -> dict[str, Any]:
        """Convert StateInterval to dictionary for database insertion."""
        duration_seconds = (record["end"] - record["start"]).total_seconds()

        return {
            "entity_id": record["entity_id"],
            "state": record["state"],
            "start_time": record["start"],
            "end_time": record["end"],
            "duration_seconds": duration_seconds,
        }


def get_create_table_ddl(dialect_name: str = "sqlite") -> str:
    """Generate CREATE TABLE statements for the given dialect."""

    engine = create_engine(f"{dialect_name}://")
    ddl_statements = []

    # Create tables
    for table in metadata.tables.values():
        create_table = schema.CreateTable(table)
        ddl_statements.append(str(create_table.compile(engine)))

    # Create indexes
    ddl_statements.extend(
        [str(schema.CreateIndex(index).compile(engine)) for index in indexes]
    )

    # Insert initial metadata
    insert_stmt = metadata_table.insert().values(
        key="db_version", value=str(DB_VERSION)
    )
    ddl_statements.append(str(insert_stmt.compile(dialect=sqlite.dialect())))

    return ";\n".join(ddl_statements) + ";"
