"""Database schema and functions to interact with the database."""

from pathlib import Path
from typing import TYPE_CHECKING

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

    class AreaOccupancy(Base):
        """A table to store the area occupancy information."""

        __tablename__ = "area_occupancy"
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
        entities = relationship("AreaEntityConfig", back_populates="area")
        priors = relationship("AreaTimePriors", back_populates="area")

    class Entity(Base):
        """A table to store the entity information."""

        __tablename__ = "entities"
        entity_id = Column(String, primary_key=True)
        last_seen = Column(DateTime, nullable=False, default=dt_util.utcnow)
        created_at = Column(DateTime, nullable=False, default=dt_util.utcnow)
        intervals = relationship("StateInterval", back_populates="entity")
        configs = relationship("AreaEntityConfig", back_populates="entity")

    class AreaEntityConfig(Base):
        """A table to store the area entity configuration."""

        __tablename__ = "area_entity_config"
        entry_id = Column(
            String, ForeignKey("area_occupancy.entry_id"), primary_key=True
        )
        entity_id = Column(String, ForeignKey("entities.entity_id"), primary_key=True)
        entity_type = Column(String, nullable=False)
        weight = Column(Float, nullable=False, default=DEFAULT_ENTITY_WEIGHT)
        prob_given_true = Column(
            Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_TRUE
        )
        prob_given_false = Column(
            Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_FALSE
        )
        last_updated = Column(DateTime, nullable=False, default=dt_util.utcnow)
        area = relationship("AreaOccupancy", back_populates="entities")
        entity = relationship("Entity", back_populates="configs")

        __table_args__ = (
            Index("idx_area_entity_entry", "entry_id"),
            Index("idx_area_entity_type", "entry_id", "entity_type"),
        )

    class AreaTimePriors(Base):
        """A table to store the area time priors."""

        __tablename__ = "area_time_priors"
        entry_id = Column(
            String, ForeignKey("area_occupancy.entry_id"), primary_key=True
        )
        day_of_week = Column(Integer, primary_key=True)
        time_slot = Column(Integer, primary_key=True)
        prior_value = Column(Float, nullable=False)
        data_points = Column(Integer, nullable=False)
        last_updated = Column(DateTime, nullable=False, default=dt_util.utcnow)
        area = relationship("AreaOccupancy", back_populates="priors")

        __table_args__ = (
            Index("idx_area_time_priors_entry", "entry_id"),
            Index("idx_area_time_priors_day_slot", "day_of_week", "time_slot"),
            Index("idx_area_time_priors_last_updated", "last_updated"),
        )

    class StateInterval(Base):
        """A table to store the state intervals."""

        __tablename__ = "state_intervals"
        id = Column(Integer, primary_key=True)
        entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
        state = Column(String, nullable=False)
        start_time = Column(DateTime, nullable=False)
        end_time = Column(DateTime, nullable=False)
        duration_seconds = Column(Float, nullable=False)
        created_at = Column(DateTime, nullable=False, default=dt_util.utcnow)
        entity = relationship("Entity", back_populates="intervals")

        # Add unique constraint on (entity_id, start_time, end_time)
        __table_args__ = (
            UniqueConstraint(
                "entity_id", "start_time", "end_time", name="uq_intervals_entity_time"
            ),
            # Performance indexes
            Index("idx_state_intervals_entity", "entity_id"),
            Index(
                "idx_state_intervals_entity_time", "entity_id", "start_time", "end_time"
            ),
            Index("idx_state_intervals_start_time", "start_time"),
            Index("idx_state_intervals_end_time", "end_time"),
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


# ─────────────────── Table References for Core Access ───────────────────

# Make table references available for SQLAlchemy Core operations
# These reference the ORM model tables for use by sqlite_storage.py
area_occupancy_table = AreaOccupancyDB.AreaOccupancy.__table__
entities_table = AreaOccupancyDB.Entity.__table__
area_entity_config_table = AreaOccupancyDB.AreaEntityConfig.__table__
area_time_priors_table = AreaOccupancyDB.AreaTimePriors.__table__
state_intervals_table = AreaOccupancyDB.StateInterval.__table__
metadata_table = AreaOccupancyDB.Metadata.__table__

# Update the global metadata with the ORM tables
metadata = Base.metadata
