"""Database schema and functions to interact with the database."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase

    Base = DeclarativeBase
    Areas = "Areas"
    Entities = "Entities"
    Priors = "Priors"
    Intervals = "Intervals"
else:
    Base = declarative_base()

DEFAULT_AREA_PRIOR = 0.15
DEFAULT_ENTITY_WEIGHT = 0.85
DEFAULT_ENTITY_PROB_GIVEN_TRUE = 0.8
DEFAULT_ENTITY_PROB_GIVEN_FALSE = 0.05
DB_NAME = "area_occupancy.db"

# Database metadata for Core access
metadata = MetaData()

# Database schema version for migrations
DB_VERSION = 2


class AreaOccupancyDB:
    """A class to manage area occupancy database operations."""

    def __init__(
        self,
        hass: HomeAssistant = None,
    ):
        """Initialize SQLite storage.

        Args:
            hass: Home Assistant instance (optional for testing)

        """
        self.hass = hass
        self.storage_path = Path(hass.config.config_dir) / ".storage" if hass else None
        self.db_path = self.storage_path / DB_NAME if self.storage_path else None
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
        )

        # Ensure storage directory exists
        if self.storage_path:
            self.storage_path.mkdir(exist_ok=True)

        # Check if database exists and initialize if needed
        self._ensure_db_exists()

        # Create and store session
        self.session = self.get_session()

        # Create model classes dictionary for ORM
        self.model_classes = {
            "Areas": self.Areas,
            "Entities": self.Entities,
            "Priors": self.Priors,
            "Intervals": self.Intervals,
            "Metadata": self.Metadata,
        }

    # Table properties for cleaner access
    @property
    def areas(self):
        """Get the areas table."""
        return self.Areas.__table__

    @property
    def entities(self):
        """Get the entities table."""
        return self.Entities.__table__

    @property
    def intervals(self):
        """Get the intervals table."""
        return self.Intervals.__table__

    @property
    def priors(self):
        """Get the priors table."""
        return self.Priors.__table__

    @property
    def metadata(self):
        """Get the metadata table."""
        return self.Metadata.__table__

    class Areas(Base):
        """A table to store the area occupancy information."""

        __tablename__ = "areas"
        entry_id = Column(String, primary_key=True)
        area_name = Column(String, nullable=False)
        purpose = Column(String, nullable=False)
        threshold = Column(Float, nullable=False)
        area_prior = Column(
            Float,
            nullable=False,
            default=DEFAULT_AREA_PRIOR,
            server_default=text(str(DEFAULT_AREA_PRIOR)),
        )
        created_at = Column(DateTime, nullable=False, default=dt_util.utcnow)
        updated_at = Column(DateTime, nullable=False, default=dt_util.utcnow)
        entities = relationship("Entities", back_populates="area")
        priors = relationship("Priors", back_populates="area")

        def to_dict(self) -> dict[str, Any]:
            """Convert the ORM object to a dictionary."""
            return {
                "entry_id": self.entry_id,
                "area_name": self.area_name,
                "purpose": self.purpose,
                "threshold": self.threshold,
                "area_prior": self.area_prior,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Areas:
            """Create an Areas instance from a dictionary."""
            return cls(
                entry_id=data["entry_id"],
                area_name=data["area_name"],
                purpose=data["purpose"],
                threshold=data["threshold"],
                area_prior=data.get("area_prior", DEFAULT_AREA_PRIOR),
                created_at=data.get("created_at", dt_util.utcnow()),
                updated_at=data.get("updated_at", dt_util.utcnow()),
            )

    class Entities(Base):
        """A table to store the entity information."""

        __tablename__ = "entities"
        entry_id = Column(String, ForeignKey("areas.entry_id"), primary_key=True)
        entity_id = Column(String, primary_key=True)
        entity_type = Column(String, nullable=False)
        weight = Column(Float, nullable=False, default=DEFAULT_ENTITY_WEIGHT)
        prob_given_true = Column(
            Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_TRUE
        )
        prob_given_false = Column(
            Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_FALSE
        )
        last_updated = Column(DateTime, nullable=False, default=dt_util.utcnow)
        created_at = Column(DateTime, nullable=False, default=dt_util.utcnow)
        intervals = relationship("Intervals", back_populates="entity")
        area = relationship("Areas", back_populates="entities")

        __table_args__ = (
            Index("idx_entities_entry", "entry_id"),
            Index("idx_entities_type", "entry_id", "entity_type"),
        )

        def to_dict(self) -> dict[str, Any]:
            """Convert the ORM object to a dictionary."""
            return {
                "entry_id": self.entry_id,
                "entity_id": self.entity_id,
                "entity_type": self.entity_type,
                "weight": self.weight,
                "prob_given_true": self.prob_given_true,
                "prob_given_false": self.prob_given_false,
                "last_updated": self.last_updated,
                "created_at": self.created_at,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Entities:
            """Create an Entities instance from a dictionary."""
            return cls(
                entry_id=data["entry_id"],
                entity_id=data["entity_id"],
                entity_type=data["entity_type"],
                weight=data.get("weight", DEFAULT_ENTITY_WEIGHT),
                prob_given_true=data.get(
                    "prob_given_true", DEFAULT_ENTITY_PROB_GIVEN_TRUE
                ),
                prob_given_false=data.get(
                    "prob_given_false", DEFAULT_ENTITY_PROB_GIVEN_FALSE
                ),
                last_updated=data.get("last_updated", dt_util.utcnow()),
                created_at=data.get("created_at", dt_util.utcnow()),
            )

    class Priors(Base):
        """A table to store the area time priors."""

        __tablename__ = "priors"
        entry_id = Column(String, ForeignKey("areas.entry_id"), primary_key=True)
        day_of_week = Column(Integer, primary_key=True)
        time_slot = Column(Integer, primary_key=True)
        prior_value = Column(Float, nullable=False)
        data_points = Column(Integer, nullable=False)
        last_updated = Column(DateTime, nullable=False, default=dt_util.utcnow)
        area = relationship("Areas", back_populates="priors")

        __table_args__ = (
            Index("idx_priors_entry", "entry_id"),
            Index("idx_priors_day_slot", "day_of_week", "time_slot"),
            Index("idx_priors_last_updated", "last_updated"),
        )

        def to_dict(self) -> dict[str, Any]:
            """Convert the ORM object to a dictionary."""
            return {
                "entry_id": self.entry_id,
                "day_of_week": self.day_of_week,
                "time_slot": self.time_slot,
                "prior_value": self.prior_value,
                "data_points": self.data_points,
                "last_updated": self.last_updated,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Priors:
            """Create a Priors instance from a dictionary."""
            return cls(
                entry_id=data["entry_id"],
                day_of_week=data["day_of_week"],
                time_slot=data["time_slot"],
                prior_value=data["prior_value"],
                data_points=data["data_points"],
                last_updated=data.get("last_updated", dt_util.utcnow()),
            )

    class Intervals(Base):
        """A table to store the state intervals."""

        __tablename__ = "intervals"
        id = Column(Integer, primary_key=True)
        entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
        state = Column(String, nullable=False)
        start_time = Column(DateTime, nullable=False)
        end_time = Column(DateTime, nullable=False)
        duration_seconds = Column(Float, nullable=False)
        created_at = Column(DateTime, nullable=False, default=dt_util.utcnow)
        entity = relationship("Entities", back_populates="intervals")

        # Add unique constraint on (entity_id, start_time, end_time)
        __table_args__ = (
            UniqueConstraint(
                "entity_id", "start_time", "end_time", name="uq_intervals_entity_time"
            ),
            # Performance indexes
            Index("idx_intervals_entity", "entity_id"),
            Index("idx_intervals_entity_time", "entity_id", "start_time", "end_time"),
            Index("idx_intervals_start_time", "start_time"),
            Index("idx_intervals_end_time", "end_time"),
        )

        def to_dict(self) -> dict[str, Any]:
            """Convert the ORM object to a dictionary."""
            return {
                "id": self.id,
                "entity_id": self.entity_id,
                "state": self.state,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_seconds": self.duration_seconds,
                "created_at": self.created_at,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Intervals:
            """Create an Intervals instance from a dictionary."""
            return cls(
                entity_id=data["entity_id"],
                state=data["state"],
                start_time=data["start_time"],
                end_time=data["end_time"],
                duration_seconds=data["duration_seconds"],
                created_at=data["created_at"],
            )

    class Metadata(Base):
        """A table to store the metadata."""

        __tablename__ = "metadata"
        key = Column(String, primary_key=True)
        value = Column(String, nullable=False)

    def _ensure_db_exists(self):
        """Check if the database exists and initialize it if needed."""
        # Check if any tables exist by trying to query the metadata table
        try:
            with self.engine.connect() as conn:
                # Try to query a table to see if the database is initialized
                conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                )
        except Exception:  # noqa: BLE001
            # Database doesn't exist or is not initialized, create it
            self.init_db()

    def get_engine(self):
        """Get the engine for the database with optimized settings."""
        return self.engine

    def get_session(self):
        """Get the session for the database."""
        engine = self.get_engine()
        session = sessionmaker(bind=engine)
        return session()

    def commit(self):
        """Commit the current session."""
        if self.session:
            self.session.commit()

    def rollback(self):
        """Rollback the current session."""
        if self.session:
            self.session.rollback()

    def close(self):
        """Close the current session."""
        if self.session:
            self.session.close()
            self.session = None

    def refresh_session(self):
        """Create a new session if the current one is closed."""
        if not self.session:
            self.session = self.get_session()

    def init_db(self):
        """Initialize the database."""
        Base.metadata.create_all(self.engine)
