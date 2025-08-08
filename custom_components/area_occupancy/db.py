"""Database schema and functions to interact with the database."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
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

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance
from homeassistant.util import dt as dt_util

from .const import (
    CONF_VERSION,
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PROBABILITY,
    MIN_WEIGHT,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase

    from .coordinator import AreaOccupancyCoordinator

    Base = DeclarativeBase
    Areas = "Areas"
    Entities = "Entities"
    Priors = "Priors"
    Intervals = "Intervals"
else:
    Base = declarative_base()

_LOGGER = logging.getLogger(__name__)

# Interval filtering thresholds to exclude anomalous data
# Exclude intervals shorter than 5 seconds (false triggers)
MIN_INTERVAL_SECONDS = 5
# Exclude intervals longer than 13 hours (stuck sensors)
MAX_INTERVAL_SECONDS = 13 * 3600
# States to exclude from intervals
INVALID_STATES = {"unknown", "unavailable", None, "", "NaN"}

RETENTION_DAYS = 365

DEFAULT_AREA_PRIOR = 0.15
DEFAULT_ENTITY_WEIGHT = 0.85
DEFAULT_ENTITY_PROB_GIVEN_TRUE = 0.8
DEFAULT_ENTITY_PROB_GIVEN_FALSE = 0.05
DB_NAME = "area_occupancy.db"

# Database metadata for Core access
metadata = MetaData()

# Database schema version for migrations
DB_VERSION = 3


class AreaOccupancyDB:
    """A class to manage area occupancy database operations."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
    ):
        """Initialize SQLite storage.

        Args:
            coordinator: AreaOccupancyCoordinator instance

        """
        self.coordinator = coordinator
        self.conf_version = coordinator.config_entry.data.get("version", CONF_VERSION)
        self.hass = coordinator.hass
        self.storage_path = (
            Path(self.hass.config.config_dir) / ".storage" if self.hass else None
        )
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

        # Create session maker
        self._session_maker = sessionmaker(bind=self.engine)

        # Create model classes dictionary for ORM
        self.model_classes = {
            "Areas": self.Areas,
            "Entities": self.Entities,
            "Priors": self.Priors,
            "Intervals": self.Intervals,
            "Metadata": self.Metadata,
        }

    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup.

        Yields:
            Session: A SQLAlchemy session

        Example:
            with self.get_session() as session:
                result = session.query(self.Areas).first()

        """
        session = self._session_maker()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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
        area_id = Column(String, nullable=False)
        purpose = Column(String, nullable=False)
        threshold = Column(Float, nullable=False)
        area_prior = Column(
            Float(precision=10),
            nullable=False,
            default=DEFAULT_AREA_PRIOR,
            server_default=str(DEFAULT_AREA_PRIOR),
        )
        created_at = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        updated_at = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
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
                "area_prior": self.area_prior,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Areas:
            """Create an Areas instance from a dictionary."""
            # Handle area_id fallback - use entry_id if area_id is None or empty
            area_id = data.get("area_id")
            if not area_id:
                area_id = data["entry_id"]

            return cls(
                entry_id=data["entry_id"],
                area_name=data["area_name"],
                area_id=area_id,
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
        last_updated = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        created_at = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
        is_decaying = Column(Boolean, nullable=False, default=False)
        decay_start = Column(DateTime(timezone=True), nullable=True)
        evidence = Column(Boolean, nullable=False, default=False)
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
                "is_decaying": self.is_decaying,
                "decay_start": self.decay_start,
                "evidence": self.evidence,
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
                is_decaying=data.get("is_decaying", False),
                decay_start=data.get("decay_start"),
                evidence=data.get("evidence", False),
            )

    class Priors(Base):
        """A table to store the area time priors."""

        __tablename__ = "priors"
        entry_id = Column(String, ForeignKey("areas.entry_id"), primary_key=True)
        day_of_week = Column(Integer, primary_key=True)
        time_slot = Column(Integer, primary_key=True)
        prior_value = Column(Float, nullable=False)
        data_points = Column(Integer, nullable=False)
        last_updated = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
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
        start_time = Column(DateTime(timezone=True), nullable=False)
        end_time = Column(DateTime(timezone=True), nullable=False)
        duration_seconds = Column(Float, nullable=False)
        created_at = Column(
            DateTime(timezone=True), nullable=False, default=dt_util.utcnow
        )
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
        """Check if the database exists and initialize it if needed, with locking."""
        try:
            # Use direct engine connection during initialization
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                ).fetchone()
                if not result:
                    # No tables exist, initialize the database
                    _LOGGER.debug("No tables found, initializing database")
                    self.init_db()
                    self.set_db_version()
        except sa.exc.SQLAlchemyError:
            # Database doesn't exist or is not initialized, create it
            _LOGGER.debug("Database error during table check, initializing database")
            self.init_db()
            self.set_db_version()

    def get_engine(self):
        """Get the engine for the database with optimized settings."""
        return self.engine

    def set_db_version(self):
        """Set the database version in the metadata table."""
        # Use direct engine connection during initialization
        with self.engine.begin() as conn:
            try:
                # Try to update if exists, else insert
                result = conn.execute(
                    text("SELECT value FROM metadata WHERE key = 'db_version'")
                ).fetchone()
                if result:
                    conn.execute(
                        text(
                            "UPDATE metadata SET value = :value WHERE key = 'db_version'"
                        ),
                        {"value": str(DB_VERSION)},
                    )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO metadata (key, value) VALUES ('db_version', :value)"
                        ),
                        {"value": str(DB_VERSION)},
                    )
            except Exception as e:
                _LOGGER.error("Failed to set db_version in metadata table: %s", e)
                raise

    def get_db_version(self) -> int:
        """Get the database version from the metadata table."""
        with self.get_session() as session:
            try:
                metadata_entry = (
                    session.query(self.Metadata).filter_by(key="db_version").first()
                )
                return int(metadata_entry.value) if metadata_entry else 0
            except Exception as e:
                _LOGGER.error("Failed to get db_version from metadata table: %s", e)
                raise

    def delete_db(self):
        """Delete the database file."""
        if self.db_path and self.db_path.exists():
            try:
                self.db_path.unlink()
                _LOGGER.info("Deleted database at %s", self.db_path)
            except (OSError, PermissionError) as e:
                _LOGGER.error("Failed to delete database file: %s", e)

    def force_reinitialize(self):
        """Force reinitialization of the database tables."""
        _LOGGER.debug("Forcing database reinitialization")
        self.init_db()
        self.set_db_version()

    def init_db(self) -> None:
        """Initialize the database with WAL mode and race condition handling."""
        _LOGGER.debug("Starting database initialization")
        try:
            # Enable WAL mode for better concurrent writes
            self._enable_wal_mode()
            # Create all tables with checkfirst to avoid race conditions
            _LOGGER.debug("Creating database tables")
            Base.metadata.create_all(self.engine, checkfirst=True)
            _LOGGER.debug("Database tables created successfully")
        except sa.exc.OperationalError as err:
            # Handle race condition when multiple instances try to create tables
            if err.orig and hasattr(err.orig, "sqlite_errno"):
                if err.orig.sqlite_errno == 1:
                    _LOGGER.debug(
                        "Table already exists (race condition), continuing: %s", err
                    )
                    # Continue - other tables might still need to be created
                    # Try to create remaining tables individually
                    self._create_tables_individually()
                else:
                    _LOGGER.error("Database initialization failed: %s", err)
                    raise
            else:
                _LOGGER.error("Database initialization failed: %s", err)
                raise
        except Exception as err:
            _LOGGER.error("Database initialization failed: %s", err)
            raise

    def _enable_wal_mode(self) -> None:
        """Enable SQLite WAL mode for better concurrent writes."""
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text("PRAGMA journal_mode=WAL"))
        except sa.exc.SQLAlchemyError as err:
            _LOGGER.debug("Failed to enable WAL mode: %s", err)

    def _create_tables_individually(self) -> None:
        """Create tables individually to handle race conditions."""
        for table in Base.metadata.tables.values():
            try:
                table.create(self.engine, checkfirst=True)
            except sa.exc.OperationalError as err:
                if err.orig and hasattr(err.orig, "sqlite_errno"):
                    if err.orig.sqlite_errno == 1:
                        _LOGGER.debug("Table %s already exists, skipping", table.name)
                        continue
                raise

    # --- Load Data ---

    async def load_data(self) -> None:
        """Load the data from the database."""
        try:
            with self.get_session() as session:
                area = (
                    session.query(self.Areas)
                    .filter_by(entry_id=self.coordinator.entry_id)
                    .first()
                )
                if area:
                    self.coordinator.prior.set_global_prior(area.area_prior)
                entities = (
                    session.query(self.Entities)
                    .filter_by(entry_id=self.coordinator.entry_id)
                    .order_by(self.Entities.entity_id)
                    .all()
                )
                if entities:
                    for entity_obj in entities:
                        # Try to get existing entity from coordinator
                        try:
                            existing_entity = self.coordinator.entities.get_entity(
                                entity_obj.entity_id
                            )
                            # Update existing entity with database values (preserve database timestamp)
                            existing_entity.update_decay(
                                entity_obj.decay_start,
                                entity_obj.is_decaying,
                            )
                            existing_entity.update_likelihood(
                                entity_obj.prob_given_true,
                                entity_obj.prob_given_false,
                            )
                            # DB weight takes priority over configured defaults when valid
                            if hasattr(existing_entity, "type") and hasattr(
                                existing_entity.type, "weight"
                            ):
                                try:
                                    weight_val = float(entity_obj.weight)
                                    if MIN_WEIGHT <= weight_val <= MAX_WEIGHT:
                                        existing_entity.type.weight = weight_val
                                except (TypeError, ValueError):
                                    pass
                            existing_entity.last_updated = entity_obj.last_updated
                            existing_entity.previous_evidence = entity_obj.evidence
                            _LOGGER.debug(
                                "Updated existing entity %s with database values",
                                entity_obj.entity_id,
                            )
                        except ValueError:
                            # Entity not found in coordinator, create new one from database
                            _LOGGER.warning(
                                "Entity %s not found in coordinator, creating from database",
                                entity_obj.entity_id,
                            )
                            new_entity = self.coordinator.factory.create_from_db(
                                entity_obj
                            )
                            self.coordinator.entities.add_entity(new_entity)
                _LOGGER.debug("Loaded area occupancy data")
        except Exception as err:
            _LOGGER.error("Failed to load area occupancy data: %s", err)
            raise

    # --- Save Data ---

    async def save_area_data(self) -> None:
        """Save the area data to the database."""
        try:
            with self.engine.begin() as conn:
                cfg = self.coordinator.config

                area_data = {
                    "entry_id": self.coordinator.entry_id,
                    "area_name": cfg.name,
                    "area_id": cfg.area_id,
                    "purpose": cfg.purpose,
                    "threshold": cfg.threshold,
                    "area_prior": self.coordinator.area_prior,
                    "updated_at": dt_util.utcnow(),
                }

                _LOGGER.debug("Attempting to insert area data: %s", area_data)

                # Validate required fields
                if not area_data["entry_id"]:
                    _LOGGER.error("entry_id is empty or None, cannot insert area")
                    return

                if not area_data["area_name"]:
                    _LOGGER.error("area_name is empty or None, cannot insert area")
                    return

                if not area_data["purpose"]:
                    _LOGGER.error("purpose is empty or None, cannot insert area")
                    return

                if area_data["threshold"] is None:
                    _LOGGER.error("threshold is None, cannot insert area")
                    return

                if area_data["area_prior"] is None:
                    _LOGGER.error("area_prior is None, cannot insert area")
                    return

                # Handle area_id - use entry_id as fallback if area_id is None or empty
                if not area_data["area_id"]:
                    _LOGGER.info(
                        "area_id is None or empty, using entry_id as fallback: %s",
                        area_data["entry_id"],
                    )
                    area_data["area_id"] = area_data["entry_id"]

                try:
                    result = conn.execute(
                        self.Areas.__table__.insert().prefix_with("OR REPLACE"),
                        area_data,
                    )
                    _LOGGER.debug("Area insert/replace result: %s", result)

                    if result.rowcount > 0:
                        _LOGGER.info(
                            "Successfully saved area data for entry_id: %s",
                            area_data["entry_id"],
                        )
                    else:
                        _LOGGER.warning("Area insert/replace affected 0 rows")

                except (
                    sa.exc.SQLAlchemyError,
                    HomeAssistantError,
                    TimeoutError,
                    OSError,
                ) as insert_err:
                    _LOGGER.error("Failed to insert/replace area data: %s", insert_err)
                    try:
                        conn.execute(
                            self.Areas.__table__.insert(),
                            area_data,
                        )
                        _LOGGER.info("Direct insert succeeded")
                    except Exception as direct_err:
                        _LOGGER.error("Direct insert also failed: %s", direct_err)
                        raise
            _LOGGER.debug("Saved area data")
        except Exception as err:
            _LOGGER.error("Failed to save area data: %s", err)
            raise

    async def save_entity_data(self) -> None:
        """Save the entity data to the database."""
        try:
            with self.engine.begin() as conn:
                entities = self.coordinator.entities.entities.values()
                for entity in entities:
                    # Skip entities with missing type information
                    if not hasattr(entity, "type") or not entity.type:
                        _LOGGER.warning(
                            "Entity %s has no type information, skipping",
                            entity.entity_id,
                        )
                        continue

                    entity_type = getattr(entity.type, "input_type", None)
                    if entity_type is None:
                        _LOGGER.warning(
                            "Entity %s has no input_type, skipping", entity.entity_id
                        )
                        continue

                    # Normalize values before persisting
                    try:
                        weight = float(
                            getattr(entity.type, "weight", DEFAULT_ENTITY_WEIGHT)
                        )
                    except (TypeError, ValueError):
                        weight = DEFAULT_ENTITY_WEIGHT
                    # Clamp weight to bounds
                    weight = max(MIN_WEIGHT, min(MAX_WEIGHT, weight))

                    # Clamp likelihoods to valid probability bounds
                    prob_true = max(
                        MIN_PROBABILITY,
                        min(MAX_PROBABILITY, float(entity.prob_given_true)),
                    )
                    prob_false = max(
                        MIN_PROBABILITY,
                        min(MAX_PROBABILITY, float(entity.prob_given_false)),
                    )

                    # Evidence must be a boolean for the DB schema
                    evidence_val = (
                        bool(entity.evidence) if entity.evidence is not None else False
                    )

                    last_updated = entity.last_updated or dt_util.utcnow()

                    conn.execute(
                        self.Entities.__table__.insert().prefix_with("OR REPLACE"),
                        {
                            "entry_id": self.coordinator.entry_id,
                            "entity_id": entity.entity_id,
                            "entity_type": entity_type,
                            "weight": weight,
                            "prob_given_true": prob_true,
                            "prob_given_false": prob_false,
                            "last_updated": last_updated,
                            "is_decaying": entity.decay.is_decaying,
                            "decay_start": entity.decay.decay_start,
                            "evidence": evidence_val,
                        },
                    )
            _LOGGER.debug("Saved entity data")
        except Exception as err:
            _LOGGER.error("Failed to save entity data: %s", err)
            raise

    async def save_data(self) -> None:
        """Save both area and entity data to the database."""
        await self.save_area_data()
        await self.save_entity_data()

    # --- Sync Data from Recorder ---

    def is_valid_state(self, state: Any) -> bool:
        """Check if a state is valid."""
        return state not in INVALID_STATES

    def is_intervals_empty(self) -> bool:
        """Check if the intervals table is empty using ORM."""
        try:
            with self.get_session() as session:
                count = session.query(self.Intervals).count()
                return count == 0
        except sa.exc.SQLAlchemyError as e:
            # If table doesn't exist, it's considered empty
            if "no such table" in str(e).lower():
                _LOGGER.debug("Intervals table doesn't exist yet, considering empty")
                return True
            _LOGGER.error("Failed to check if intervals empty: %s", e)
            raise

    def get_area_data(self, entry_id: str) -> dict[str, Any] | None:
        """Get area data for a specific entry_id."""
        try:
            with self.get_session() as session:
                area = session.query(self.Areas).filter_by(entry_id=entry_id).first()
                if area:
                    return area.to_dict()
                return None
        except sa.exc.SQLAlchemyError as e:
            _LOGGER.error("Failed to get area data: %s", e)
            return None

    async def ensure_area_exists(self) -> None:
        """Ensure that the area record exists in the database."""
        try:
            # Check if area exists
            existing_area = self.get_area_data(self.coordinator.entry_id)
            if existing_area:
                _LOGGER.debug(
                    "Area already exists for entry_id: %s", self.coordinator.entry_id
                )
                return

            # Area doesn't exist, force create it
            _LOGGER.info(
                "Area not found, forcing creation for entry_id: %s",
                self.coordinator.entry_id,
            )
            await self.save_data()

            # Verify it was created
            new_area = self.get_area_data(self.coordinator.entry_id)
            if new_area:
                _LOGGER.info("Successfully created area: %s", new_area)
            else:
                _LOGGER.error(
                    "Failed to create area for entry_id: %s", self.coordinator.entry_id
                )

        except (sa.exc.SQLAlchemyError, HomeAssistantError, TimeoutError, OSError) as e:
            _LOGGER.error("Error ensuring area exists: %s", e)

    def get_latest_interval(self) -> datetime | None:
        """Return the latest interval end time minus 1 hour, or default window if none."""
        try:
            with self.get_session() as session:
                result = session.execute(
                    sa.select(sa.func.max(self.Intervals.end_time))
                ).scalar()
                if result:
                    return result - timedelta(hours=1)
                return dt_util.now() - timedelta(days=10)
        except sa.exc.SQLAlchemyError as e:
            # If table doesn't exist, return a default time
            if "no such table" in str(e).lower():
                _LOGGER.debug("Intervals table doesn't exist yet, using default time")
                return dt_util.now() - timedelta(days=10)
            raise

    def _states_to_intervals(
        self, states: dict[str, list[State]], end_time: datetime
    ) -> list[dict[str, Any]]:
        """Convert states to intervals by processing consecutive state changes for each entity.

        Args:
            states: Dictionary mapping entity_id to list of State objects
            end_time: The end time for the analysis period

        Returns:
            List of interval dictionaries with proper start_time, end_time, and duration_seconds

        """
        intervals = []
        retention_time = dt_util.now() - timedelta(days=RETENTION_DAYS)

        for entity_id, state_list in states.items():
            if not state_list:
                continue

            # Sort states by last_changed time
            sorted_states = sorted(state_list, key=lambda s: s.last_changed)

            # Process each state to create intervals
            for i, state in enumerate(sorted_states):
                # Skip states outside retention period
                if state.last_changed < retention_time:
                    continue

                # Determine the end time for this interval
                if i + 1 < len(sorted_states):
                    # Use the start time of the next state as the end time
                    interval_end = sorted_states[i + 1].last_changed
                else:
                    # For the last state, use the analysis end time
                    interval_end = end_time

                # Calculate duration
                duration_seconds = (interval_end - state.last_changed).total_seconds()

                # Apply filtering based on state and duration
                if state.state == "on":
                    if duration_seconds <= MAX_INTERVAL_SECONDS:
                        intervals.append(
                            {
                                "entity_id": entity_id,
                                "state": state.state,
                                "start_time": state.last_changed,
                                "end_time": interval_end,
                                "duration_seconds": duration_seconds,
                            }
                        )
                elif (
                    self.is_valid_state(state.state)
                    and duration_seconds >= MIN_INTERVAL_SECONDS
                ):
                    intervals.append(
                        {
                            "entity_id": entity_id,
                            "state": state.state,
                            "start_time": state.last_changed,
                            "end_time": interval_end,
                            "duration_seconds": duration_seconds,
                        }
                    )

        _LOGGER.debug("Created %d intervals from states", len(intervals))
        return intervals

    async def sync_states(
        self,
    ) -> None:
        """Fetch states history from recorder and commit to Intervals table."""
        hass = self.coordinator.hass
        entity_ids = self.coordinator.config.entity_ids
        recorder = get_instance(hass)
        start_time = self.get_latest_interval()
        end_time = dt_util.now()

        try:
            states = await recorder.async_add_executor_job(
                lambda: get_significant_states(
                    hass,
                    start_time,
                    end_time,
                    entity_ids,
                    minimal_response=False,
                )
            )
            _LOGGER.debug("Found %d states", len(states))

            if states:
                # Convert states to proper intervals with correct duration calculation
                intervals = self._states_to_intervals(states, end_time)
                _LOGGER.debug("Syncing %d intervals", len(intervals))
                if intervals:
                    with self.engine.begin() as conn:
                        conn.execute(
                            self.Intervals.__table__.insert().prefix_with("OR IGNORE"),
                            intervals,
                        )
                    _LOGGER.debug("Synced %d intervals", len(intervals))
            else:
                _LOGGER.debug("No states found in recorder query result")

        except (HomeAssistantError, TimeoutError) as err:
            _LOGGER.error("Error getting states: %s", err)
            raise
