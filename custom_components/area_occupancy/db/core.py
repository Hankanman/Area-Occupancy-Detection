"""Core database management functionality."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import inspect
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker as create_sessionmaker

from homeassistant.util import dt as dt_util

from ..const import (
    CONF_VERSION,
    DB_NAME,
    DEFAULT_BACKUP_INTERVAL_HOURS,
    DEFAULT_ENABLE_AUTO_RECOVERY,
    DEFAULT_ENABLE_PERIODIC_BACKUPS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
)
from . import (
    aggregation,
    correlation,
    maintenance,
    operations,
    queries,
    relationships,
    sync,
    utils,
)
from .schema import (
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

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


def _create_delegated_methods() -> dict[str, Any]:
    """Create the delegation mapping for pure wrapper methods.

    Returns:
        Dictionary mapping method names to their module functions.
    """
    return {
        # Maintenance methods
        "check_database_integrity": maintenance.check_database_integrity,
        "check_database_accessibility": maintenance.check_database_accessibility,
        "is_database_corrupted": maintenance.is_database_corrupted,
        "attempt_database_recovery": maintenance.attempt_database_recovery,
        "backup_database": maintenance.backup_database,
        "restore_database_from_backup": maintenance.restore_database_from_backup,
        "handle_database_corruption": maintenance.handle_database_corruption,
        "periodic_health_check": maintenance.periodic_health_check,
        "set_db_version": maintenance.set_db_version,
        "get_db_version": maintenance.get_db_version,
        "delete_db": maintenance.delete_db,
        "force_reinitialize": maintenance.force_reinitialize,
        "init_db": maintenance.init_db,
        # Operations methods
        "load_data": operations.load_data,
        "save_area_data": operations.save_area_data,
        "save_entity_data": operations.save_entity_data,
        "save_data": operations.save_data,
        "cleanup_orphaned_entities": operations.cleanup_orphaned_entities,
        "delete_area_data": operations.delete_area_data,
        "ensure_area_exists": operations.ensure_area_exists,
        "prune_old_intervals": operations.prune_old_intervals,
        "save_global_prior": operations.save_global_prior,
        "save_occupied_intervals_cache": operations.save_occupied_intervals_cache,
        # Utility methods
        "is_intervals_empty": utils.is_intervals_empty,
        "safe_is_intervals_empty": utils.safe_is_intervals_empty,
        # Query methods (except get_time_prior which adds logic)
        "get_area_data": queries.get_area_data,
        "get_latest_interval": queries.get_latest_interval,
        "get_global_prior": queries.get_global_prior,
        "get_occupied_intervals_cache": queries.get_occupied_intervals_cache,
        "is_occupied_intervals_cache_valid": queries.is_occupied_intervals_cache_valid,
        # Sync methods
        "sync_states": sync.sync_states,
        # Aggregation methods
        "aggregate_raw_to_daily": aggregation.aggregate_raw_to_daily,
        "aggregate_daily_to_weekly": aggregation.aggregate_daily_to_weekly,
        "aggregate_weekly_to_monthly": aggregation.aggregate_weekly_to_monthly,
        "run_interval_aggregation": aggregation.run_interval_aggregation,
        "prune_old_aggregates": aggregation.prune_old_aggregates,
        "prune_old_numeric_samples": aggregation.prune_old_numeric_samples,
        # Correlation methods
        "analyze_correlation": correlation.analyze_correlation,
        "save_correlation_result": correlation.save_correlation_result,
        "save_binary_likelihood_result": correlation.save_binary_likelihood_result,
        "analyze_and_save_correlation": correlation.analyze_and_save_correlation,
        "analyze_binary_likelihoods": correlation.analyze_binary_likelihoods,
        "get_correlation_for_entity": correlation.get_correlation_for_entity,
        # Relationship methods
        "save_area_relationship": relationships.save_area_relationship,
        "get_adjacent_areas": relationships.get_adjacent_areas,
        "get_influence_weight": relationships.get_influence_weight,
        "calculate_adjacent_influence": relationships.calculate_adjacent_influence,
        "sync_adjacent_areas_from_config": relationships.sync_adjacent_areas_from_config,
    }


class AreaOccupancyDB:
    """A class to manage area occupancy database operations."""

    # Reference schema models as class attributes
    Areas = Areas
    Entities = Entities
    Intervals = Intervals
    Priors = Priors
    Metadata = Metadata
    IntervalAggregates = IntervalAggregates
    OccupiedIntervalsCache = OccupiedIntervalsCache
    GlobalPriors = GlobalPriors
    NumericSamples = NumericSamples
    NumericAggregates = NumericAggregates
    Correlations = Correlations
    EntityStatistics = EntityStatistics
    AreaRelationships = AreaRelationships
    CrossAreaStats = CrossAreaStats

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
    ):
        """Initialize SQLite storage.

        Args:
            coordinator: AreaOccupancyCoordinator instance
        """
        self.coordinator = coordinator
        if coordinator.config_entry is None:
            raise ValueError("Coordinator config_entry cannot be None")
        self.conf_version = coordinator.config_entry.data.get("version", CONF_VERSION)
        self.hass = coordinator.hass

        self._setup_paths()
        self._setup_engine()
        self._setup_delegation()
        self._setup_model_classes()

        if os.getenv("AREA_OCCUPANCY_AUTO_INIT_DB") == "1":
            self.initialize_database()

    def _setup_paths(self) -> None:
        """Set up storage and database paths."""
        self.storage_path = (
            Path(self.hass.config.config_dir) / ".storage" if self.hass else None
        )
        self.db_path = self.storage_path / DB_NAME if self.storage_path else None

        if self.storage_path:
            self.storage_path.mkdir(exist_ok=True)

    def _setup_engine(self) -> None:
        """Set up database engine and session maker."""
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            pool_pre_ping=True,
            poolclass=sa.pool.NullPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 10,
            },
        )
        self._session_maker = create_sessionmaker(bind=self.engine)

        self.enable_auto_recovery = DEFAULT_ENABLE_AUTO_RECOVERY
        self.max_recovery_attempts = DEFAULT_MAX_RECOVERY_ATTEMPTS
        self.enable_periodic_backups = DEFAULT_ENABLE_PERIODIC_BACKUPS
        self.backup_interval_hours = DEFAULT_BACKUP_INTERVAL_HOURS

        self.last_area_save_ts: float = 0.0
        self.last_entities_save_ts: float = 0.0
        self._save_debounce_seconds: float = 1.5

    def _setup_delegation(self) -> None:
        """Set up delegation mapping for pure wrapper methods."""
        self._delegated_methods = _create_delegated_methods()

    def _setup_model_classes(self) -> None:
        """Generate model_classes dictionary from schema model class attributes."""
        schema_model_names = [
            "Areas",
            "Entities",
            "Priors",
            "Intervals",
            "Metadata",
            "IntervalAggregates",
            "OccupiedIntervalsCache",
            "GlobalPriors",
            "NumericSamples",
            "NumericAggregates",
            "Correlations",
            "EntityStatistics",
            "AreaRelationships",
            "CrossAreaStats",
        ]
        self.model_classes = {name: getattr(self, name) for name in schema_model_names}

    def initialize_database(self) -> None:
        """Initialize the database by checking if it exists and creating it if needed.

        This method performs blocking I/O operations.
        In production environments, it should be called via
        hass.async_add_executor_job() to avoid blocking the event loop.
        In test environments (when AREA_OCCUPANCY_AUTO_INIT_DB=1 is set),
        this method may be called directly.
        """
        maintenance.ensure_db_exists(self)

    def __getattr__(self, name: str) -> Any:
        """Dynamically delegate to module functions for pure wrapper methods.

        This method handles delegation for all pure wrapper methods that simply
        call module functions with self as the first argument. Methods that add
        logic (like get_occupied_intervals and get_time_prior) are kept as
        explicit class methods.

        Args:
            name: The name of the method being accessed

        Returns:
            A callable that delegates to the appropriate module function

        Raises:
            AttributeError: If the method name is not in the delegation mapping
        """
        if name in self._delegated_methods:
            func = self._delegated_methods[name]
            if inspect.iscoroutinefunction(func):

                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await func(self, *args, **kwargs)

                return async_wrapper

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(self, *args, **kwargs)

            return sync_wrapper

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def is_valid_state(self, state: Any) -> bool:
        """Check if a state is valid."""
        return utils.is_valid_state(state)

    def get_time_prior(
        self,
        area_name: str,
        day_of_week: int,
        time_slot: int,
        default_prior: float = 0.5,
    ) -> float:
        """Get the time prior for a specific time slot.

        Args:
            area_name: The area name to filter by
            day_of_week: Day of week (0=Monday, 6=Sunday)
            time_slot: Time slot index
            default_prior: Default prior value if not found

        Returns:
            Time prior value or default if not found
        """
        return queries.get_time_prior(
            self,
            self.coordinator.entry_id,
            area_name,
            day_of_week,
            time_slot,
            default_prior,
        )

    def get_occupied_intervals(
        self,
        area_name: str,
        start_time: datetime | None = None,
        motion_timeout: int = 300,  # Default 5 min if not specified
    ) -> list[tuple[datetime, datetime]]:
        """Get raw occupied intervals from motion sensors only (without using cache).

        This delegates to queries.get_occupied_intervals but adapts arguments
        to match what PriorAnalyzer expects. Occupied intervals are determined
        exclusively by motion sensors. All motion sensors for the area are
        automatically included in the query. The end time is always the current time.
        """
        # Determine lookback days if start_time provided
        lookback_days = DEFAULT_LOOKBACK_DAYS  # default
        if start_time:
            lookback_days = (dt_util.utcnow() - start_time).days + 1

        return queries.get_occupied_intervals(
            self,
            self.coordinator.entry_id,
            area_name,
            lookback_days,
            motion_timeout,
        )

    @contextmanager
    def get_session(self) -> Any:
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

    @property
    def areas(self) -> Any:
        """Get the areas table."""
        return self.Areas.__table__

    @property
    def entities(self) -> Any:
        """Get the entities table."""
        return self.Entities.__table__

    @property
    def intervals(self) -> Any:
        """Get the intervals table."""
        return self.Intervals.__table__

    @property
    def priors(self) -> Any:
        """Get the priors table."""
        return self.Priors.__table__

    @property
    def metadata(self) -> Any:
        """Get the metadata table."""
        return self.Metadata.__table__

    def get_engine(self) -> Any:
        """Get the engine for the database with optimized settings."""
        return self.engine

    def update_session_maker(self) -> None:
        """Update the session maker after engine changes (e.g., recovery/restore)."""
        self._session_maker = create_sessionmaker(bind=self.engine)
