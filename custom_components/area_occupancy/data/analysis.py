"""Analysis module for heavy database operations.

This module handles all database-intensive analysis operations for priors and likelihoods,
separating analysis logic from state management in Prior and EntityManager classes.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import UTC, datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import DEFAULT_LOOKBACK_DAYS, MIN_PROBABILITY
from ..db.queries import (
    get_occupied_intervals,
    get_occupied_intervals_cache,
    get_time_bounds,
    get_total_occupied_seconds,
    is_occupied_intervals_cache_valid,
)
from ..utils import clamp_probability
from .entity_type import DEFAULT_TYPES, InputType

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
else:
    AreaOccupancyCoordinator = Any

_LOGGER = logging.getLogger(__name__)


class IntervalData(NamedTuple):
    """Lightweight interval data for likelihood analysis."""

    entity_id: str
    start_time: datetime
    duration_seconds: float
    state: str


# Prior calculation constants
DEFAULT_PRIOR = 0.5
DEFAULT_OCCUPIED_SECONDS = 0.0

# Time slot constants
DEFAULT_SLOT_MINUTES = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR
DAYS_PER_WEEK = 7

# SQLite to Python weekday conversion
# SQLite strftime('%w'): 0=Sunday, 1=Monday, ..., 6=Saturday
# Python: 0=Monday, 1=Tuesday, ..., 6=Sunday
SQLITE_TO_PYTHON_WEEKDAY_OFFSET = 6


def get_occupied_intervals_with_cache(
    db: Any,
    entry_id: str,
    area_name: str,
    lookback_days: int,
    motion_timeout_seconds: int,
    include_media: bool = False,
    include_appliance: bool = False,
    media_sensor_ids: list[str] | None = None,
    appliance_sensor_ids: list[str] | None = None,
) -> list[tuple[datetime, datetime]]:
    """Fetch occupied intervals with caching and Python fallback.

    This function wraps the query function from queries module with prior-specific caching.
    Business logic for cache validation and retrieval is handled here.

    Args:
        db: Database instance
        entry_id: Entry ID
        area_name: Area name
        lookback_days: Number of days to look back
        motion_timeout_seconds: Motion timeout in seconds
        include_media: Whether to include media sensors
        include_appliance: Whether to include appliance sensors
        media_sensor_ids: List of media sensor IDs
        appliance_sensor_ids: List of appliance sensor IDs

    Returns:
        List of occupied intervals
    """
    now = dt_util.utcnow()
    lookback_date = now - timedelta(days=lookback_days)

    # Check cache only for motion-only queries (business logic decision)
    if not include_media and not include_appliance:
        if is_occupied_intervals_cache_valid(db, area_name, max_age_hours=24):
            cached_intervals = get_occupied_intervals_cache(
                db,
                area_name,
                period_start=lookback_date,
                period_end=now,
            )
            if cached_intervals:
                _LOGGER.debug(
                    "Using cached occupied intervals for %s: %d intervals",
                    area_name,
                    len(cached_intervals),
                )
                return cached_intervals

    # Cache miss or media/appliance included - use direct query
    return get_occupied_intervals(
        db=db,
        entry_id=entry_id,
        area_name=area_name,
        lookback_days=lookback_days,
        motion_timeout_seconds=motion_timeout_seconds,
        include_media=include_media,
        include_appliance=include_appliance,
        media_sensor_ids=media_sensor_ids,
        appliance_sensor_ids=appliance_sensor_ids,
    )


def is_timestamp_occupied(
    timestamp: datetime, occupied_intervals: list[tuple[datetime, datetime]]
) -> bool:
    """Check if timestamp falls within any occupied interval using binary search."""
    if not occupied_intervals:
        return False

    idx = bisect.bisect_right([start for start, _ in occupied_intervals], timestamp)

    if idx > 0:
        start, end = occupied_intervals[idx - 1]
        if start <= timestamp < end:
            return True

    return False


class PriorAnalyzer:
    """Analyzer for prior probability calculations from database."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
    ) -> None:
        """Initialize the PriorAnalyzer.

        Args:
            coordinator: The coordinator instance
            area_name: Area name for multi-area support
        """
        self.coordinator = coordinator
        self.db = coordinator.db
        self.area_name = area_name
        if area_name not in coordinator.areas:
            available = list(coordinator.areas.keys())
            raise ValueError(
                f"Area '{area_name}' not found. "
                f"Available areas: {available if available else '(none)'}"
            )
        self.config = coordinator.areas[area_name].config
        self.sensor_ids = self.config.sensors.motion
        self.media_sensor_ids = self.config.sensors.media
        self.appliance_sensor_ids = self.config.sensors.appliance
        self.entry_id = coordinator.entry_id

    def analyze_area_prior(self, entity_ids: list[str]) -> float:
        """Calculate the overall occupancy prior for an area based on historical sensor data.

        Calculates from motion sensors only. Priors are exclusively based on motion/presence
        sensors to ensure consistent ground truth for occupancy detection.

        Args:
            entity_ids: List of entity IDs to calculate prior for (motion sensors).

        Returns:
            float: Prior probability of occupancy (0.0 to 1.0)
        """
        _LOGGER.debug("Calculating area prior")

        # Validate input
        if not entity_ids:
            _LOGGER.warning("No entity IDs provided for prior calculation")
            return DEFAULT_PRIOR

        # Step 1: Calculate prior from motion sensors only
        total_occupied_seconds = self.get_total_occupied_seconds(
            include_media=False,
            include_appliance=False,
        )

        lookback_date = dt_util.utcnow() - timedelta(days=DEFAULT_LOOKBACK_DAYS)

        # Get total time period from motion sensors
        first_time, last_time = self.get_time_bounds(entity_ids=entity_ids)

        if not first_time or not last_time:
            _LOGGER.debug("No time bounds available, using default prior")
            return DEFAULT_PRIOR

        # Ensure times are timezone-aware
        if first_time.tzinfo is None:
            first_time = first_time.replace(tzinfo=UTC)
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=UTC)

        # Align denominator with lookback
        window_start = max(first_time, lookback_date)
        total_seconds = (last_time - window_start).total_seconds()

        # Validate time range before division
        if total_seconds <= 0:
            _LOGGER.warning(
                "Invalid time range for prior calculation: first_time=%s, last_time=%s, "
                "total_seconds=%.2f, using default prior",
                first_time,
                last_time,
                total_seconds,
            )
            return DEFAULT_PRIOR

        # Validate that occupied time doesn't exceed total time (data corruption check)
        if total_occupied_seconds > total_seconds:
            _LOGGER.warning(
                "Data corruption detected: occupied time (%.2f) exceeds total time (%.2f), "
                "capping occupied time to total time",
                total_occupied_seconds,
                total_seconds,
            )
            total_occupied_seconds = total_seconds

        prior = total_occupied_seconds / total_seconds if total_seconds > 0 else 0.0
        prior = clamp_probability(prior)

        _LOGGER.debug(
            "Motion-only prior: %.4f (occupied: %.2fs, total: %.2fs)",
            prior,
            total_occupied_seconds,
            total_seconds,
        )

        # Note: min_prior_override is NOT applied here - it's applied at runtime
        # in Prior.value. We save the actual calculated prior to the database.

        _LOGGER.debug("Final calculated prior: %.4f", prior)

        # Save to GlobalPriors table with metadata
        try:
            # Get motion-only intervals for cache
            all_intervals = self.get_occupied_intervals(
                lookback_days=DEFAULT_LOOKBACK_DAYS,
                include_media=False,
                include_appliance=False,
            )
            interval_count = len(all_intervals) if all_intervals else 0

            # Save global prior to database
            self.db.save_global_prior(
                area_name=self.area_name,
                prior_value=prior,
                data_period_start=window_start,
                data_period_end=last_time,
                total_occupied_seconds=total_occupied_seconds,
                total_period_seconds=total_seconds,
                interval_count=interval_count,
                calculation_method="interval_analysis",
                confidence=None,  # Could calculate confidence based on sample size
            )

            # Save occupied intervals to cache (motion-only)
            if all_intervals:
                self.db.save_occupied_intervals_cache(
                    area_name=self.area_name,
                    intervals=all_intervals,
                    data_source="motion_sensors",
                )

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            _LOGGER.warning("Error saving global prior to database: %s", e)
            # Don't fail the calculation if saving fails

        return prior

    def analyze_time_priors(self, slot_minutes: int = DEFAULT_SLOT_MINUTES) -> None:
        """Estimate P(occupied) per day_of_week and time_slot from motion sensor intervals."""
        _LOGGER.debug("Computing time priors")

        # Validate slot_minutes parameter
        if slot_minutes <= 0 or slot_minutes > MINUTES_PER_DAY:
            _LOGGER.warning(
                "Invalid slot_minutes: %d, using default %d",
                slot_minutes,
                DEFAULT_SLOT_MINUTES,
            )
            slot_minutes = DEFAULT_SLOT_MINUTES

        # Ensure slot_minutes divides MINUTES_PER_DAY evenly to avoid misaligned slots
        if MINUTES_PER_DAY % slot_minutes != 0:
            _LOGGER.warning(
                "slot_minutes=%d does not divide %d evenly, using %d instead",
                slot_minutes,
                MINUTES_PER_DAY,
                DEFAULT_SLOT_MINUTES,
            )
            slot_minutes = DEFAULT_SLOT_MINUTES

        # Step 1: Get time bounds to calculate total denominator
        first_time, last_time = self.get_time_bounds()

        if not first_time or not last_time:
            _LOGGER.warning("No time bounds available for time prior calculation")
            return

        # Ensure times are timezone-aware
        if first_time.tzinfo is None:
            first_time = first_time.replace(tzinfo=UTC)
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=UTC)

        # Calculate total time period with proper date arithmetic
        days = (last_time.date() - first_time.date()).days + 1
        slots_per_day = MINUTES_PER_DAY // slot_minutes
        slot_duration_seconds = slot_minutes * MINUTES_PER_HOUR

        if days <= 0 or slots_per_day <= 0:
            _LOGGER.warning(
                "Invalid calculation parameters: days=%d, slots=%d", days, slots_per_day
            )
            return

        # Step 2: Get aggregated interval data using cached intervals
        # We use get_occupied_intervals_with_cache to ensure we use the EXACT SAME logic
        # as global prior (including motion timeout, merged intervals).
        # This ensures time priors are calculated from the same "ground truth" as global prior.
        # Both use motion-only sensors for consistent occupancy detection.
        intervals = self.get_occupied_intervals(
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            include_media=False,
            include_appliance=False,
        )

        # Step 3: Aggregate occupied seconds per slot (Python side)
        # Initialize buckets for (day_of_week, slot_idx) -> total_seconds
        occupied_seconds: dict[tuple[int, int], float] = defaultdict(float)

        for start, end in intervals:
            # Ensure timezone awareness
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            if end.tzinfo is None:
                end = end.replace(tzinfo=UTC)

            current = start
            while current < end:
                # Determine current slot
                day_of_week = current.weekday()  # 0=Monday, 6=Sunday
                minutes_from_midnight = current.hour * 60 + current.minute
                slot_idx = minutes_from_midnight // slot_minutes

                # Determine end of current slot
                next_slot_minutes = (slot_idx + 1) * slot_minutes
                # Handle day rollover
                if next_slot_minutes >= MINUTES_PER_DAY:
                    next_slot_start = (current + timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                else:
                    next_slot_start = current.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ) + timedelta(minutes=next_slot_minutes)

                # Determine segment end (min of interval end or slot end)
                segment_end = min(end, next_slot_start)
                duration = (segment_end - current).total_seconds()

                if duration > 0:
                    occupied_seconds[(day_of_week, slot_idx)] += duration

                current = segment_end

        # Generate priors for all time slots
        now = dt_util.utcnow()
        db = self.db

        try:
            with self.db.get_session() as session:
                for day in range(DAYS_PER_WEEK):
                    for slot in range(slots_per_day):
                        total_slot_seconds = days * slot_duration_seconds
                        occupied_slot_seconds = occupied_seconds.get(
                            (day, slot), DEFAULT_OCCUPIED_SECONDS
                        )

                        # Calculate probability with validation
                        if total_slot_seconds > 0:
                            p = occupied_slot_seconds / total_slot_seconds
                            # Ensure probability is within valid bounds
                            p = clamp_probability(p)
                        else:
                            _LOGGER.warning(
                                "Zero total slot seconds for day=%d, slot=%d", day, slot
                            )
                            p = MIN_PROBABILITY

                        # Check if prior already exists
                        existing_prior = (
                            session.query(db.Priors)
                            .filter_by(
                                entry_id=self.entry_id,
                                area_name=self.area_name,
                                day_of_week=day,
                                time_slot=slot,
                            )
                            .first()
                        )

                        if existing_prior:
                            # Update existing prior
                            existing_prior.prior_value = p
                            existing_prior.data_points = int(total_slot_seconds)
                            existing_prior.last_updated = now
                        else:
                            # Create new prior
                            prior = db.Priors(
                                entry_id=self.entry_id,
                                area_name=self.area_name,
                                day_of_week=day,
                                time_slot=slot,
                                prior_value=p,
                                data_points=int(total_slot_seconds),
                                last_updated=now,
                            )
                            session.add(prior)

                # Commit the session to save all priors
                session.commit()
                _LOGGER.debug(
                    "Successfully computed time priors for %d days, %d slots per day",
                    days,
                    slots_per_day,
                )
        except (
            SQLAlchemyError,
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
        ) as e:
            _LOGGER.error("Error during time prior computation: %s", e)

    def get_total_occupied_seconds(
        self, include_media: bool = False, include_appliance: bool = False
    ) -> float:
        """Thin wrapper over database helper for backwards compatibility."""
        return get_total_occupied_seconds(
            db=self.db,
            entry_id=self.entry_id,
            area_name=self.area_name,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            motion_timeout_seconds=self.config.sensors.motion_timeout,
            include_media=include_media,
            include_appliance=include_appliance,
            media_sensor_ids=self.media_sensor_ids if include_media else None,
            appliance_sensor_ids=self.appliance_sensor_ids
            if include_appliance
            else None,
        )

    def get_occupied_intervals(
        self,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        include_media: bool = False,
        include_appliance: bool = False,
    ) -> list[tuple[datetime, datetime]]:
        """Get occupied intervals with caching logic.

        This method wraps the query function with prior-specific caching logic.
        """
        return get_occupied_intervals_with_cache(
            db=self.db,
            entry_id=self.entry_id,
            area_name=self.area_name,
            lookback_days=lookback_days,
            motion_timeout_seconds=self.config.sensors.motion_timeout,
            include_media=include_media,
            include_appliance=include_appliance,
            media_sensor_ids=self.media_sensor_ids if include_media else None,
            appliance_sensor_ids=self.appliance_sensor_ids
            if include_appliance
            else None,
        )

    def get_time_bounds(
        self, entity_ids: list[str] | None = None
    ) -> tuple[datetime | None, datetime | None]:
        """Thin wrapper over database helper for backwards compatibility."""
        return get_time_bounds(
            db=self.db,
            entry_id=self.entry_id,
            area_name=self.area_name,
            entity_ids=entity_ids,
        )


class LikelihoodAnalyzer:
    """Analyzer for likelihood probability calculations from database."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
    ) -> None:
        """Initialize the LikelihoodAnalyzer.

        Args:
            coordinator: The coordinator instance
            area_name: Area name for multi-area support
        """
        self.coordinator = coordinator
        self.db = coordinator.db
        self.area_name = area_name
        if area_name not in coordinator.areas:
            available = list(coordinator.areas.keys())
            raise ValueError(
                f"Area '{area_name}' not found. "
                f"Available areas: {available if available else '(none)'}"
            )
        self.config = coordinator.areas[area_name].config
        self.entry_id = coordinator.entry_id

    def analyze_likelihoods(
        self,
        occupied_times: list[tuple[datetime, datetime]],
        entity_manager: Any,
        session: Any | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Compute P(sensor=true|occupied) and P(sensor=true|empty) per sensor.

        Args:
            occupied_times: List of (start, end) tuples representing occupied periods
            entity_manager: EntityManager instance for accessing entity configurations
            session: Optional database session. If None, creates a new session.

        Returns:
            Dictionary mapping entity_id to (P(sensor=true|occupied), P(sensor=true|empty))
        """
        _LOGGER.debug("Analyzing likelihoods")

        try:
            if session is not None:
                # Use provided session (don't use context manager)
                return self._analyze_likelihoods_from_session(
                    session, occupied_times, entity_manager
                )

            # Create new session
            with self.db.get_session() as new_session:
                return self._analyze_likelihoods_from_session(
                    new_session, occupied_times, entity_manager
                )
        except (
            SQLAlchemyError,
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
        ) as e:
            _LOGGER.error("Error in analyze_likelihoods: %s", e)
            return {}

    def _analyze_likelihoods_from_session(
        self,
        session: Any,
        occupied_times: list[tuple[datetime, datetime]],
        entity_manager: Any,
    ) -> dict[str, tuple[float, float]]:
        """Analyze likelihoods using an existing session."""
        likelihoods: dict[str, tuple[float, float]] = {}
        try:
            sensors = self._get_sensors(session)
            if not sensors:
                return likelihoods

            intervals_by_entity = self._get_intervals_by_entity(session, sensors)

            # Track motion sensors for logging
            motion_sensors_found = []

            for entity in sensors:
                entity_id = str(entity.entity_id)
                entity_type = (
                    str(entity.entity_type) if hasattr(entity, "entity_type") else None
                )

                # Exclude motion sensors from likelihood learning
                # Motion sensors are used as ground truth to determine occupied intervals,
                # so they should use default likelihoods, not learned ones.
                # This breaks the circular dependency where motion sensors determine
                # occupied intervals and then calculate their own likelihoods from those intervals.
                if entity_type == InputType.MOTION.value:
                    motion_sensors_found.append(entity_id)
                    _LOGGER.debug(
                        "Excluding motion sensor %s from likelihood learning "
                        "(motion sensors use default likelihoods as ground truth)",
                        entity_id,
                    )
                    continue

                # Check if entity has intervals
                if entity_id not in intervals_by_entity:
                    _LOGGER.debug(
                        "Skipping entity %s: no intervals found in database",
                        entity_id,
                    )
                    continue

                # Attempt to fetch entity from entity_manager
                try:
                    _ = entity_manager.get_entity(entity_id)
                except (KeyError, ValueError) as e:
                    _LOGGER.warning(
                        "Skipping entity %s: failed to get entity from manager: %s",
                        entity_id,
                        e,
                    )
                    continue

                # Analyze entity likelihood with error handling
                try:
                    prob_given_true, prob_given_false = self._analyze_entity_likelihood(
                        entity, intervals_by_entity, occupied_times, entity_manager
                    )
                    likelihoods[entity_id] = (prob_given_true, prob_given_false)
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    ValueError,
                ) as e:
                    _LOGGER.warning(
                        "Error analyzing likelihood for entity %s: %s. Skipping.",
                        entity_id,
                        e,
                    )
                    continue

            # Log summary of motion sensor exclusion
            if motion_sensors_found:
                _LOGGER.info(
                    "Excluded %d motion sensor(s) from likelihood learning: %s. "
                    "Motion sensors use default likelihoods (prob_given_true=0.95, prob_given_false=0.02) "
                    "as they are used as ground truth for determining occupied intervals.",
                    len(motion_sensors_found),
                    ", ".join(motion_sensors_found),
                )

            _LOGGER.debug("Likelihoods analyzed for %d entities", len(likelihoods))
        except (
            SQLAlchemyError,
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
        ) as e:
            _LOGGER.error("Error analyzing likelihoods: %s", e)

        return likelihoods

    def _get_sensors(self, session: Any) -> list[Any]:
        """Get all sensor configs for this area."""
        return list(
            session.query(self.db.Entities)
            .filter_by(entry_id=self.entry_id, area_name=self.area_name)
            .all()
        )

    def _get_intervals_by_entity(
        self,
        session: Any,
        sensors: list[Any],
    ) -> dict[str, list[IntervalData]]:
        """Get all intervals grouped by entity_id, selecting only needed columns."""
        sensor_entity_ids = [entity.entity_id for entity in sensors]

        # Select only needed columns instead of full ORM objects
        all_intervals = (
            session.query(
                self.db.Intervals.entity_id,
                self.db.Intervals.start_time,
                self.db.Intervals.duration_seconds,
                self.db.Intervals.state,
            )
            .filter(self.db.Intervals.entity_id.in_(sensor_entity_ids))
            .all()
        )

        intervals_by_entity: dict[str, list[IntervalData]] = defaultdict(list)
        for entity_id, start_time, duration_seconds, state in all_intervals:
            interval_data = IntervalData(
                entity_id=str(entity_id),
                start_time=start_time,
                duration_seconds=float(duration_seconds),
                state=str(state),
            )
            intervals_by_entity[str(entity_id)].append(interval_data)

        return intervals_by_entity

    def _analyze_entity_likelihood(
        self,
        entity: Any,
        intervals_by_entity: dict[str, list[IntervalData]],
        occupied_times: list[tuple[datetime, datetime]],
        entity_manager: Any,
    ) -> tuple[float, float]:
        """Analyze likelihoods for a single entity.

        Args:
            entity: SQLAlchemy Entities object
            intervals_by_entity: Dictionary mapping entity_id to list of intervals
            occupied_times: List of (start, end) tuples representing occupied periods
            entity_manager: EntityManager instance to get entity objects

        Returns:
            Tuple of (prob_given_true, prob_given_false)
        """
        # Convert SQLAlchemy entity to Python types
        entity_id = str(entity.entity_id)
        intervals = intervals_by_entity[entity_id]
        entity_obj = entity_manager.get_entity(entity_id)

        # Count interval states
        true_occ: float = 0.0
        false_occ: float = 0.0
        true_empty: float = 0.0
        false_empty: float = 0.0

        for interval in intervals:
            # Use interval data directly (already converted from ORM)
            start_time = interval.start_time
            duration_seconds = interval.duration_seconds

            occ = self._is_occupied(start_time, occupied_times)
            is_active = self._is_interval_active(interval, entity_obj)

            if is_active:
                if occ:
                    true_occ += duration_seconds
                else:
                    true_empty += duration_seconds
            elif occ:
                false_occ += duration_seconds
            else:
                false_empty += duration_seconds

        # Calculate probabilities
        prob_given_true = (
            true_occ / (true_occ + false_occ) if (true_occ + false_occ) > 0 else 0.5
        )
        prob_given_false = (
            true_empty / (true_empty + false_empty)
            if (true_empty + false_empty) > 0
            else 0.5
        )

        return (prob_given_true, prob_given_false)

    def _is_occupied(
        self, ts: datetime, occupied_times: list[tuple[datetime, datetime]]
    ) -> bool:
        """Check if timestamp falls within any occupied interval."""
        return is_timestamp_occupied(ts, occupied_times)

    def _is_interval_active(self, interval: IntervalData, entity_obj: Any) -> bool:
        """Determine if interval state is active based on entity type."""
        if entity_obj.active_states:
            return interval.state in entity_obj.active_states
        if entity_obj.active_range:
            min_val, max_val = entity_obj.active_range
            try:
                state_val = float(interval.state)
            except (ValueError, TypeError):
                return False
            else:
                return min_val <= state_val <= max_val
        return False


def _update_likelihoods_in_db(
    db: Any,
    entry_id: str,
    likelihoods: dict[str, tuple[float, float]],
    now: datetime,
) -> list[str]:
    """Update likelihoods in database (synchronous helper for executor).

    Args:
        db: Database instance
        entry_id: Config entry ID
        likelihoods: Dictionary mapping entity_id to (prob_given_true, prob_given_false)
        now: Current timestamp for last_updated field

    Returns:
        List of entity_ids that were successfully updated in the database
    """
    updated_entity_ids: list[str] = []
    with db.get_session() as session:
        for entity_id, (prob_given_true, prob_given_false) in likelihoods.items():
            # Update database
            entity_db = (
                session.query(db.Entities)
                .filter_by(entry_id=entry_id, entity_id=entity_id)
                .first()
            )
            if entity_db:
                # Validate: Motion sensors should not have learned likelihoods stored
                # They use default likelihoods as ground truth
                if entity_db.entity_type == InputType.MOTION.value:
                    _LOGGER.warning(
                        "Attempted to store learned likelihoods for motion sensor '%s'. "
                        "Motion sensors use default likelihoods (prob_given_true=0.95, prob_given_false=0.02) "
                        "and should not have learned values. Skipping database update.",
                        entity_id,
                    )
                    continue

                entity_db.prob_given_true = prob_given_true
                entity_db.prob_given_false = prob_given_false
                entity_db.last_updated = now
                updated_entity_ids.append(entity_id)
            else:
                _LOGGER.warning(
                    "Entity '%s' not found in database, skipping likelihood update",
                    entity_id,
                )
        session.commit()
    return updated_entity_ids


async def start_prior_analysis(
    coordinator: AreaOccupancyCoordinator,
    area_name: str,
    prior_instance: Any,
) -> None:
    """Start prior analysis and update database and memory.

    This function orchestrates the full prior analysis workflow:
    1. Calculates global prior from sensor data
    2. Writes global prior to Areas.area_prior in database
    3. Updates Prior.global_prior in memory
    4. Calculates time priors and writes to Priors table
    5. Invalidates caches

    Args:
        coordinator: The coordinator instance
        area_name: Area name for multi-area support
        prior_instance: Prior instance to update

    Raises:
        ValueError: If area_name is invalid or analysis fails
        TypeError: If data conversion fails
        SQLAlchemyError: If database operations fail
        Exception: For any other unexpected errors
    """
    _LOGGER.debug("Starting prior analysis for area: %s", area_name)

    # Create analyzer for heavy database operations
    analyzer = PriorAnalyzer(coordinator, area_name)

    # Run heavy calculations in executor to avoid blocking the event loop
    try:
        # Calculate global prior
        global_prior = await coordinator.hass.async_add_executor_job(
            analyzer.analyze_area_prior, analyzer.sensor_ids
        )
        _LOGGER.debug("Area prior calculated: %.2f", global_prior)

        # Calculate and write time priors
        await coordinator.hass.async_add_executor_job(analyzer.analyze_time_priors)
        _LOGGER.debug("Time priors calculated")

        # Update in-memory state
        prior_instance.set_global_prior(global_prior)

        _LOGGER.debug("Prior analysis completed successfully for area: %s", area_name)

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        RuntimeError,
        OSError,
    ):
        _LOGGER.exception("Prior calculation failed")
        raise


async def start_likelihood_analysis(
    coordinator: AreaOccupancyCoordinator,
    area_name: str,
    entity_manager: Any,
) -> None:
    """Start likelihood analysis and update database and memory.

    This function orchestrates the full likelihood analysis workflow:
    1. Gets occupied intervals from Prior
    2. Calculates likelihoods for all entities
    3. Writes prob_given_true and prob_given_false to Entities table
    4. Updates Entity objects in memory

    Args:
        coordinator: The coordinator instance
        area_name: Area name for multi-area support
        entity_manager: EntityManager instance to update

    Raises:
        ValueError: If area_name is invalid or analysis fails
        TypeError: If data conversion fails
        SQLAlchemyError: If database operations fail
        Exception: For any other unexpected errors
    """
    _LOGGER.debug("Starting likelihood analysis for area: %s", area_name)

    # Get the area's prior for this EntityManager's area
    area = coordinator.get_area(area_name)
    if area is None:
        error_msg = f"Cannot update likelihoods: area '{area_name}' not found"
        _LOGGER.warning(error_msg)
        raise ValueError(error_msg)

    # Get occupied intervals from prior
    occupied_times = area.prior.get_occupied_intervals()

    # Create analyzer for heavy database operations
    analyzer = LikelihoodAnalyzer(coordinator, area_name)

    # Run heavy calculations in executor to avoid blocking the event loop
    try:
        # Calculate likelihoods
        likelihoods = await coordinator.hass.async_add_executor_job(
            analyzer.analyze_likelihoods, occupied_times, entity_manager
        )

        if not likelihoods:
            _LOGGER.warning("No likelihoods calculated for area: %s", area_name)
            return

        # Write likelihoods to database (run in executor to avoid blocking event loop)
        db = coordinator.db
        entry_id = coordinator.entry_id
        now = dt_util.utcnow()

        updated_entity_ids = await coordinator.hass.async_add_executor_job(
            _update_likelihoods_in_db,
            db,
            entry_id,
            likelihoods,
            now,
        )

        # Update in-memory state for entities that were successfully updated in database
        for entity_id in updated_entity_ids:
            prob_given_true, prob_given_false = likelihoods[entity_id]
            try:
                entity_obj = entity_manager.get_entity(entity_id)

                # Skip update if entity has a learned active range from correlation
                # This prevents overwriting dynamic correlation-based likelihoods with standard analysis results
                if entity_obj.learned_active_range is not None:
                    _LOGGER.debug(
                        "Skipping standard likelihood update for %s (using learned correlation)",
                        entity_id,
                    )
                    continue

                entity_obj.update_likelihood(prob_given_true, prob_given_false)
            except ValueError as e:
                _LOGGER.warning(
                    "Entity '%s' not found in EntityManager: %s", entity_id, e
                )
                continue

        # Reset motion sensor likelihoods to configured values (user-configurable per area)
        # Get configured values from area config, fall back to defaults if not configured
        area = coordinator.get_area(area_name)
        if area and hasattr(area.config, "sensors"):
            motion_prob_given_true = getattr(
                area.config.sensors,
                "motion_prob_given_true",
                DEFAULT_TYPES[InputType.MOTION]["prob_given_true"],
            )
            motion_prob_given_false = getattr(
                area.config.sensors,
                "motion_prob_given_false",
                DEFAULT_TYPES[InputType.MOTION]["prob_given_false"],
            )
        else:
            # Fallback to defaults if area/config not available
            default_motion = DEFAULT_TYPES[InputType.MOTION]
            motion_prob_given_true = float(default_motion["prob_given_true"])
            motion_prob_given_false = float(default_motion["prob_given_false"])

        motion_prob_given_true = float(motion_prob_given_true)
        motion_prob_given_false = float(motion_prob_given_false)

        motion_entities = entity_manager.get_entities_by_input_type(InputType.MOTION)
        for motion_entity in motion_entities.values():
            motion_entity.update_likelihood(
                motion_prob_given_true, motion_prob_given_false
            )
            _LOGGER.debug(
                "Reset motion sensor %s to configured likelihoods: prob_given_true=%.2f, prob_given_false=%.2f",
                motion_entity.entity_id,
                motion_prob_given_true,
                motion_prob_given_false,
            )

        _LOGGER.debug(
            "Likelihoods updated for %d entities in area: %s (%d motion sensors reset to defaults)",
            len(updated_entity_ids),
            area_name,
            len(motion_entities),
        )

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        RuntimeError,
        OSError,
    ):
        _LOGGER.exception("Likelihood calculation failed")
        raise
