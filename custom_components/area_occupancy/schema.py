"""Data models and converters for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from homeassistant.util import dt as dt_util

from .state_intervals import StateInterval

# ─────────────────── Data Models ───────────────────


@dataclass
class AreaOccupancyRecord:
    """Data class for area occupancy records."""

    entry_id: str = ""
    area_name: str = ""
    purpose: str = ""
    threshold: float = 0.0
    created_at: datetime = field(default_factory=dt_util.utcnow)
    updated_at: datetime = field(default_factory=dt_util.utcnow)


@dataclass
class EntityRecord:
    """Data class for global entity records."""

    entity_id: str = ""
    last_seen: datetime = field(default_factory=dt_util.utcnow)
    created_at: datetime = field(default_factory=dt_util.utcnow)

    @property
    def domain(self) -> str:
        """Get domain from entity_id."""
        return self.entity_id.split(".")[0] if "." in self.entity_id else "unknown"


@dataclass
class AreaEntityConfigRecord:
    """Data class for area-specific entity configuration."""

    entry_id: str = ""
    entity_id: str = ""
    entity_type: str = ""
    weight: float = 1.0
    prob_given_true: float = 0.5
    prob_given_false: float = 0.1
    last_updated: datetime = field(default_factory=dt_util.utcnow)


@dataclass
class AreaTimePriorRecord:
    """Data class for area-specific time-based prior records."""

    entry_id: str = ""
    day_of_week: int = 0  # 0=Monday, 6=Sunday
    time_slot: int = 0  # 0-47 (30-minute intervals)
    prior_value: float = 0.1
    data_points: int = 0
    last_updated: datetime = field(default_factory=dt_util.utcnow)

    @property
    def time_range(self) -> tuple[int, int]:
        """Get the time range for this slot (start_hour, start_minute)."""
        start_hour = self.time_slot // 2
        start_minute = (self.time_slot % 2) * 30
        return start_hour, start_minute

    @property
    def end_time_range(self) -> tuple[int, int]:
        """Get the end time range for this slot (end_hour, end_minute)."""
        start_hour, start_minute = self.time_range
        if start_minute == 30:
            end_hour = (start_hour + 1) % 24
            end_minute = 0
        else:
            end_hour = start_hour
            end_minute = 30
        return end_hour, end_minute


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
            threshold=row.threshold,
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
            "threshold": record.threshold,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def row_to_entity(row: Any) -> EntityRecord:
        """Convert SQLAlchemy row to EntityRecord."""
        return EntityRecord(
            entity_id=row.entity_id,
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
            "last_seen": record.last_seen,
            "created_at": record.created_at,
        }

    @staticmethod
    def row_to_area_entity_config(row: Any) -> AreaEntityConfigRecord:
        """Convert SQLAlchemy row to AreaEntityConfigRecord."""
        return AreaEntityConfigRecord(
            entry_id=row.entry_id,
            entity_id=row.entity_id,
            entity_type=row.entity_type,
            weight=row.weight,
            prob_given_true=row.prob_given_true,
            prob_given_false=row.prob_given_false,
            last_updated=row.last_updated
            if isinstance(row.last_updated, datetime)
            else dt_util.parse_datetime(row.last_updated) or dt_util.utcnow(),
        )

    @staticmethod
    def area_entity_config_to_dict(record: AreaEntityConfigRecord) -> dict[str, Any]:
        """Convert AreaEntityConfigRecord to dictionary for database insertion."""
        return {
            "entry_id": record.entry_id,
            "entity_id": record.entity_id,
            "entity_type": record.entity_type,
            "weight": record.weight,
            "prob_given_true": record.prob_given_true,
            "prob_given_false": record.prob_given_false,
            "last_updated": record.last_updated,
        }

    @staticmethod
    def row_to_area_time_prior(row: Any) -> AreaTimePriorRecord:
        """Convert SQLAlchemy row to AreaTimePriorRecord."""
        return AreaTimePriorRecord(
            entry_id=row.entry_id,
            day_of_week=row.day_of_week,
            time_slot=row.time_slot,
            prior_value=row.prior_value,
            data_points=row.data_points,
            last_updated=row.last_updated
            if isinstance(row.last_updated, datetime)
            else dt_util.parse_datetime(row.last_updated) or dt_util.utcnow(),
        )

    @staticmethod
    def area_time_prior_to_dict(record: AreaTimePriorRecord) -> dict[str, Any]:
        """Convert AreaTimePriorRecord to dictionary for database insertion."""
        return {
            "entry_id": record.entry_id,
            "day_of_week": record.day_of_week,
            "time_slot": record.time_slot,
            "prior_value": record.prior_value,
            "data_points": record.data_points,
            "last_updated": record.last_updated,
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
            "created_at": dt_util.utcnow(),
        }
