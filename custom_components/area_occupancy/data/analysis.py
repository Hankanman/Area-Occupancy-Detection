"""Analysis module for heavy database operations.

This module handles all database-intensive analysis operations for priors and likelihoods,
separating analysis logic from state management in Prior and EntityManager classes.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import DEFAULT_LOOKBACK_DAYS, MIN_PROBABILITY
from ..db.aggregation import get_interval_aggregates
from ..db.priors import (
    get_occupied_intervals,
    get_time_bounds,
    get_total_occupied_seconds,
)
from ..utils import clamp_probability

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
        session_provider: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> None:
        """Initialize the PriorAnalyzer.

        Args:
            coordinator: The coordinator instance
            area_name: Area name for multi-area support
            session_provider: Optional callable that returns a context manager for sessions.
                           If None, uses db.get_session(). Useful for testing.
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
        self._session_provider = session_provider or coordinator.db.get_session

    def analyze_area_prior(self, entity_ids: list[str]) -> float:
        """Calculate the overall occupancy prior for an area based on historical sensor data.

        First calculates from motion sensors only. If the result is below 0.10 (10%),
        supplements with media players (playing state only) and appliances (on state only).

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
        total_occupied_seconds = get_total_occupied_seconds(
            db=self.db,
            entry_id=self.entry_id,
            area_name=self.area_name,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            motion_timeout_seconds=self.config.sensors.motion_timeout,
            include_media=False,
            include_appliance=False,
            session_provider=self._session_provider,
        )

        lookback_date = dt_util.utcnow() - timedelta(days=DEFAULT_LOOKBACK_DAYS)

        # Get total time period from motion sensors
        first_time, last_time = get_time_bounds(
            db=self.db,
            entry_id=self.entry_id,
            area_name=self.area_name,
            entity_ids=entity_ids,
            session_provider=self._session_provider,
        )

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

        motion_prior = (
            total_occupied_seconds / total_seconds if total_seconds > 0 else 0.0
        )
        motion_prior = clamp_probability(motion_prior)

        _LOGGER.debug(
            "Motion-only prior: %.4f (occupied: %.2fs, total: %.2fs)",
            motion_prior,
            total_occupied_seconds,
            total_seconds,
        )

        # Step 2: If motion prior < 0.10, supplement with media/appliance sensors
        LOW_PRIOR_THRESHOLD = 0.10
        all_intervals = None
        total_all_occupied = total_occupied_seconds  # Default to motion-only

        if motion_prior >= LOW_PRIOR_THRESHOLD:
            # Motion prior is sufficient, use it
            prior = motion_prior
            _LOGGER.debug(
                "Motion prior %.4f >= %.2f, using motion-only prior",
                motion_prior,
                LOW_PRIOR_THRESHOLD,
            )
        else:
            # Motion prior is too low, supplement with media/appliance sensors
            _LOGGER.debug(
                "Motion prior %.4f < %.2f, supplementing with media/appliance sensors",
                motion_prior,
                LOW_PRIOR_THRESHOLD,
            )

            # Get intervals including media and appliances
            include_media = bool(self.media_sensor_ids)
            include_appliance = bool(self.appliance_sensor_ids)

            if not include_media and not include_appliance:
                # No additional sensors available, use motion prior
                prior = motion_prior
                _LOGGER.debug(
                    "No media or appliance sensors configured, using motion prior"
                )
            else:
                # Get total occupied time from all sensor types
                all_intervals = get_occupied_intervals(
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
                    session_provider=self._session_provider,
                )

                if not all_intervals:
                    prior = motion_prior
                    _LOGGER.debug(
                        "No intervals found when including media/appliances, using motion prior"
                    )
                else:
                    # Calculate total occupied seconds from all intervals
                    total_all_occupied = sum(
                        (end - start).total_seconds() for start, end in all_intervals
                    )

                    # Get time bounds from all sensor types if needed
                    # For simplicity, use the same time bounds (from motion sensors)
                    # since we're just supplementing the occupied time
                    prior = (
                        total_all_occupied / total_seconds if total_seconds > 0 else 0.0
                    )

                    # Ensure result is within valid probability bounds
                    prior = clamp_probability(prior)

                    _LOGGER.debug(
                        "Prior with media/appliances: %.4f (occupied: %.2fs, total: %.2fs, motion: %.4f)",
                        prior,
                        total_all_occupied,
                        total_seconds,
                        motion_prior,
                    )

        # Step 3: Apply minimum prior override if configured
        if self.config.min_prior_override > 0.0:
            original_prior = prior
            if prior < self.config.min_prior_override:
                prior = self.config.min_prior_override
                _LOGGER.debug(
                    "Applied minimum prior override: %.4f -> %.4f",
                    original_prior,
                    prior,
                )

        _LOGGER.debug("Final calculated prior: %.4f", prior)

        # Save to GlobalPriors table with metadata
        try:
            # Determine which intervals were used
            used_media = motion_prior < 0.10 and bool(self.media_sensor_ids)
            used_appliance = motion_prior < 0.10 and bool(self.appliance_sensor_ids)

            # Get all intervals used for calculation (if we haven't already)
            if used_media or used_appliance:
                # We already have all_intervals from above
                interval_count = len(all_intervals) if all_intervals else 0
                final_occupied_seconds = total_all_occupied
                data_source = "merged"
            else:
                # Motion-only, get intervals for cache
                if all_intervals is None:
                    all_intervals = get_occupied_intervals(
                        db=self.db,
                        entry_id=self.entry_id,
                        area_name=self.area_name,
                        lookback_days=DEFAULT_LOOKBACK_DAYS,
                        motion_timeout_seconds=self.config.sensors.motion_timeout,
                        include_media=False,
                        include_appliance=False,
                        session_provider=self._session_provider,
                    )
                interval_count = len(all_intervals) if all_intervals else 0
                final_occupied_seconds = total_occupied_seconds
                data_source = "motion_sensors"

            # Save global prior to database
            self.db.save_global_prior(
                area_name=self.area_name,
                prior_value=prior,
                data_period_start=window_start,
                data_period_end=last_time,
                total_occupied_seconds=final_occupied_seconds,
                total_period_seconds=total_seconds,
                interval_count=interval_count,
                calculation_method="interval_analysis",
                confidence=None,  # Could calculate confidence based on sample size
            )

            # Save occupied intervals to cache (motion-only for cache efficiency)
            if all_intervals and not used_media and not used_appliance:
                self.db.save_occupied_intervals_cache(
                    area_name=self.area_name,
                    intervals=all_intervals,
                    data_source=data_source,
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

        # Get aggregated interval data
        interval_aggregates = get_interval_aggregates(
            db=self.db,
            entry_id=self.entry_id,
            area_name=self.area_name,
            slot_minutes=slot_minutes,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            motion_timeout_seconds=self.config.sensors.motion_timeout,
            media_sensor_ids=self.media_sensor_ids,
            appliance_sensor_ids=self.appliance_sensor_ids,
            session_provider=self._session_provider,
        )

        # Get time bounds
        first_time, last_time = get_time_bounds(
            db=self.db,
            entry_id=self.entry_id,
            area_name=self.area_name,
            session_provider=self._session_provider,
        )

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

        # Validate calculated values
        if days <= 0:
            _LOGGER.warning(
                "Invalid day calculation: first_time=%s, last_time=%s, days=%d",
                first_time,
                last_time,
                days,
            )
            return

        if slots_per_day <= 0:
            _LOGGER.warning(
                "Invalid slots_per_day calculation: slot_minutes=%d, slots_per_day=%d",
                slot_minutes,
                slots_per_day,
            )
            return

        # Create lookup dictionary from aggregated results with safer conversion
        occupied_seconds = {}
        for day, slot, total_seconds in interval_aggregates:
            try:
                # Day is already in Python weekday format (0=Monday, 6=Sunday)
                # from both SQL and Python aggregation paths
                python_weekday = int(day)

                # Validate weekday is within valid range
                if not (0 <= python_weekday < DAYS_PER_WEEK):
                    _LOGGER.warning(
                        "Invalid weekday: %d (must be 0-6), skipping", python_weekday
                    )
                    continue

                # Validate slot number
                if 0 <= int(slot) < slots_per_day:
                    occupied_seconds[(python_weekday, int(slot))] = float(
                        total_seconds or DEFAULT_OCCUPIED_SECONDS
                    )
                else:
                    _LOGGER.warning(
                        "Invalid slot number: %d (max: %d), skipping",
                        int(slot),
                        slots_per_day - 1,
                    )
            except (ValueError, TypeError) as e:
                _LOGGER.warning(
                    "Invalid interval data: day=%s, slot=%s, error=%s", day, slot, e
                )
                continue

        # Generate priors for all time slots
        now = dt_util.utcnow()
        db = self.db

        try:
            with self._session_provider() as session:
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
            session_provider=self._session_provider,
        )

    def get_occupied_intervals(
        self,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        include_media: bool = False,
        include_appliance: bool = False,
    ) -> list[tuple[datetime, datetime]]:
        """Thin wrapper over database helper for backwards compatibility."""
        return get_occupied_intervals(
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
            session_provider=self._session_provider,
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
            session_provider=self._session_provider,
        )

    def get_interval_aggregates(
        self, slot_minutes: int = DEFAULT_SLOT_MINUTES
    ) -> list[tuple[int, int, float]]:
        """Thin wrapper over database helper for backwards compatibility."""
        return get_interval_aggregates(
            db=self.db,
            entry_id=self.entry_id,
            area_name=self.area_name,
            slot_minutes=slot_minutes,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            motion_timeout_seconds=self.config.sensors.motion_timeout,
            media_sensor_ids=self.media_sensor_ids,
            appliance_sensor_ids=self.appliance_sensor_ids,
            session_provider=self._session_provider,
        )


class LikelihoodAnalyzer:
    """Analyzer for likelihood probability calculations from database."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
        session_provider: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> None:
        """Initialize the LikelihoodAnalyzer.

        Args:
            coordinator: The coordinator instance
            area_name: Area name for multi-area support
            session_provider: Optional callable that returns a context manager for sessions.
                           If None, uses db.get_session(). Useful for testing.
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
        self._session_provider = session_provider or coordinator.db.get_session

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
            with self._session_provider() as new_session:
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

            for entity in sensors:
                entity_id = str(entity.entity_id)
                prob_given_true, prob_given_false = self._analyze_entity_likelihood(
                    entity, intervals_by_entity, occupied_times, entity_manager
                )
                likelihoods[entity_id] = (prob_given_true, prob_given_false)

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
    area = coordinator.get_area_or_default(area_name)
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
                entity_obj.update_likelihood(prob_given_true, prob_given_false)
            except ValueError as e:
                _LOGGER.warning(
                    "Entity '%s' not found in EntityManager: %s", entity_id, e
                )
                continue

        _LOGGER.debug(
            "Likelihoods updated for %d entities in area: %s",
            len(updated_entity_ids),
            area_name,
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
