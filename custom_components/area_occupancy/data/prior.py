"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

# Priors is accessed through the database instance, not imported directly
from sqlalchemy import func
from sqlalchemy.exc import (
    DataError,
    IntegrityError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)

from homeassistant.const import STATE_ON, STATE_PLAYING
from homeassistant.util import dt as dt_util

from ..const import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    MAX_PRIOR,
    MAX_PROBABILITY,
    MIN_PRIOR,
    MIN_PROBABILITY,
)
from ..data.entity_type import InputType
from ..utils import clamp_probability, combine_priors

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Prior calculation constants
PRIOR_FACTOR = 1.05
DEFAULT_PRIOR = 0.5
SIGNIFICANT_CHANGE_THRESHOLD = 0.1

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

# Database query constants
DEFAULT_OCCUPIED_SECONDS = 0.0


class Prior:
    """Compute the baseline probability for an Area entity."""

    def __init__(
        self, coordinator: AreaOccupancyCoordinator, area_name: str | None = None
    ) -> None:
        """Initialize the Prior class.

        Args:
            coordinator: The coordinator instance
            area_name: Optional area name for multi-area support
        """
        self.coordinator = coordinator
        self.db = coordinator.db
        self.area_name = area_name
        # Validate area_name and retrieve config from coordinator.areas
        if not area_name:
            raise ValueError("Area name is required in multi-area architecture")
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
        self.hass = coordinator.hass
        self.global_prior: float | None = None
        self._last_updated: datetime | None = None
        # Cache for time_prior
        self._cached_time_prior: float | None = None
        self._cached_time_prior_day: int | None = None
        self._cached_time_prior_slot: int | None = None
        # Cache for occupied intervals
        self._cached_occupied_intervals: list[tuple[datetime, datetime]] | None = None
        self._cached_intervals_timestamp: datetime | None = None

    @property
    def value(self) -> float:
        """Return the current prior value or minimum if not calculated."""
        if self.global_prior is None:
            return MIN_PRIOR

        # Use global_prior directly if time_prior is None, otherwise combine them
        if self.time_prior is None:
            prior = self.global_prior
        else:
            prior = combine_priors(self.global_prior, self.time_prior)

        # Track if we needed to clamp the prior
        was_clamped = False

        # Validate that prior is within reasonable bounds before applying factor
        if not (MIN_PROBABILITY <= prior <= MAX_PROBABILITY):
            _LOGGER.warning(
                "Prior %.10f is outside valid range [%.10f, %.10f], clamping to bounds",
                prior,
                MIN_PROBABILITY,
                MAX_PROBABILITY,
            )
            prior = clamp_probability(prior)
            was_clamped = True

        # Apply factor and clamp to bounds
        adjusted_prior = prior * PRIOR_FACTOR

        # If the prior was clamped to bounds, return the clamped prior value
        if was_clamped and prior == MIN_PROBABILITY:
            result = MIN_PRIOR
        elif was_clamped and prior == MAX_PROBABILITY:
            result = MAX_PRIOR
        else:
            result = max(MIN_PRIOR, min(MAX_PRIOR, adjusted_prior))

        return result

    @property
    def time_prior(self) -> float:
        """Return the current time prior value or minimum if not calculated."""
        current_day = self.day_of_week
        current_slot = self.time_slot

        # Check if we have a valid cache for the current time slot
        if (
            self._cached_time_prior is not None
            and self._cached_time_prior_day == current_day
            and self._cached_time_prior_slot == current_slot
        ):
            return self._cached_time_prior

        # Cache miss - get from database and cache the result
        self._cached_time_prior = self.get_time_prior()
        self._cached_time_prior_day = current_day
        self._cached_time_prior_slot = current_slot

        return self._cached_time_prior

    @property
    def day_of_week(self) -> int:
        """Return the current day of week (0=Monday, 6=Sunday)."""
        return dt_util.utcnow().weekday()

    @property
    def time_slot(self) -> int:
        """Return the current time slot based on DEFAULT_SLOT_MINUTES."""
        now = dt_util.utcnow()
        return (now.hour * 60 + now.minute) // DEFAULT_SLOT_MINUTES

    @property
    def last_updated(self) -> datetime | None:
        """Return the last updated timestamp."""
        return self._last_updated

    def set_global_prior(self, prior: float) -> None:
        """Set the global prior value."""
        self.global_prior = prior
        self._last_updated = dt_util.utcnow()

    def _invalidate_time_prior_cache(self) -> None:
        """Invalidate the time_prior cache."""
        self._cached_time_prior = None
        self._cached_time_prior_day = None
        self._cached_time_prior_slot = None

    def _invalidate_occupied_intervals_cache(self) -> None:
        """Invalidate the occupied intervals cache."""
        self._cached_occupied_intervals = None
        self._cached_intervals_timestamp = None

    async def update(self) -> None:
        """Calculate and update the prior value."""
        # Run heavy calculations in executor to avoid blocking the event loop
        try:
            self.global_prior = await self.hass.async_add_executor_job(
                self.calculate_area_prior, self.sensor_ids
            )
            _LOGGER.debug("Area prior calculated: %.2f", self.global_prior)
        except (ValueError, TypeError, ZeroDivisionError):
            _LOGGER.exception("Prior calculation failed due to data error")
            self.global_prior = MIN_PRIOR
        except SQLAlchemyError:
            _LOGGER.exception("Prior calculation failed due to database error")
            self.global_prior = MIN_PRIOR
        except Exception:
            _LOGGER.exception("Prior calculation failed due to unexpected error")
            self.global_prior = MIN_PRIOR

        try:
            await self.hass.async_add_executor_job(self.compute_time_priors)
            _LOGGER.debug("Time priors calculated")
            # Invalidate time_prior cache since we updated the time priors
            self._invalidate_time_prior_cache()
        except SQLAlchemyError:
            _LOGGER.exception("Time prior calculation failed due to database error")
        except (ValueError, TypeError):
            _LOGGER.exception("Time prior calculation failed due to data error")
        except Exception:
            _LOGGER.exception("Time prior calculation failed due to unexpected error")

        # Invalidate occupied intervals cache since we updated the data
        self._invalidate_occupied_intervals_cache()
        self._last_updated = dt_util.utcnow()

    def calculate_area_prior(self, entity_ids: list[str]) -> float:
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
        total_occupied_seconds = self.get_total_occupied_seconds()

        # Get total time period from motion sensors
        first_time, last_time = self.get_time_bounds(entity_ids)

        if not first_time or not last_time:
            _LOGGER.debug("No time bounds available, using default prior")
            return DEFAULT_PRIOR

        total_seconds = (last_time - first_time).total_seconds()

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
                all_intervals = self.get_occupied_intervals(
                    include_media=include_media, include_appliance=include_appliance
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
        return prior

    def get_time_prior(self) -> float:
        """Get the time prior for the current time slot.

        Returns:
            float: Time prior value or default 0.5 if not found

        """
        _LOGGER.debug("Getting time prior")
        db = self.db

        try:
            with db.get_session() as session:
                prior = (
                    session.query(db.Priors)
                    .filter_by(
                        entry_id=self.coordinator.entry_id,
                        day_of_week=self.day_of_week,
                        time_slot=self.time_slot,
                    )
                    .first()
                )
                return float(prior.prior_value) if prior else DEFAULT_PRIOR
        except OperationalError as e:
            _LOGGER.error("Database connection error getting time prior: %s", e)
            return DEFAULT_PRIOR
        except DataError as e:
            _LOGGER.error("Database data error getting time prior: %s", e)
            return DEFAULT_PRIOR
        except ProgrammingError as e:
            _LOGGER.error("Database query error getting time prior: %s", e)
            return DEFAULT_PRIOR
        except SQLAlchemyError as e:
            _LOGGER.error("Database error getting time prior: %s", e)
            return DEFAULT_PRIOR
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            _LOGGER.error("Unexpected error getting time prior: %s", e)
            return DEFAULT_PRIOR

    def compute_time_priors(self, slot_minutes: int = DEFAULT_SLOT_MINUTES) -> None:
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
        interval_aggregates = self.get_interval_aggregates(slot_minutes)

        # Get time bounds
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
                # Convert SQLite day_of_week (0=Sunday) to Python weekday (0=Monday)
                # Use modulo to handle any integer value safely
                python_weekday = (
                    int(day) + SQLITE_TO_PYTHON_WEEKDAY_OFFSET
                ) % DAYS_PER_WEEK

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
            with db.get_session() as session:
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
                                entry_id=self.coordinator.entry_id,
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
                                entry_id=self.coordinator.entry_id,
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
        except OperationalError as e:
            _LOGGER.error(
                "Database connection error during time prior computation: %s", e
            )
        except DataError as e:
            _LOGGER.error("Database data error during time prior computation: %s", e)
        except IntegrityError as e:
            _LOGGER.error(
                "Database integrity error during time prior computation: %s", e
            )
        except ProgrammingError as e:
            _LOGGER.error("Database query error during time prior computation: %s", e)
        except SQLAlchemyError as e:
            _LOGGER.error("Database error during time prior computation: %s", e)
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            _LOGGER.error("Unexpected error during time prior computation: %s", e)

    def get_interval_aggregates(
        self, slot_minutes: int = DEFAULT_SLOT_MINUTES
    ) -> list[tuple[int, int, float]]:
        """Get aggregated interval data for time prior computation using SQL optimization.

        Args:
            slot_minutes: Time slot size in minutes

        Returns:
            List of (day_of_week, time_slot, total_occupied_seconds) tuples

        """
        _LOGGER.debug("Getting interval aggregates using SQL GROUP BY optimization")

        # Use the new SQL-based aggregation method for much better performance
        try:
            start_time = dt_util.utcnow()
            result = self.db.get_aggregated_intervals_by_slot(
                self.coordinator.entry_id, slot_minutes
            )
            query_time = (dt_util.utcnow() - start_time).total_seconds()
            _LOGGER.debug(
                "SQL aggregation completed in %.3f seconds, returned %d slots",
                query_time,
                len(result),
            )
        except (
            SQLAlchemyError,
            OperationalError,
            DataError,
            ProgrammingError,
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
        ) as e:
            _LOGGER.error(
                "SQL aggregation failed, falling back to Python method: %s", e
            )
            # Fallback to the original Python method if SQL fails
            result = self._get_interval_aggregates_python(slot_minutes)

        return result

    def _get_interval_aggregates_python(
        self, slot_minutes: int = DEFAULT_SLOT_MINUTES
    ) -> list[tuple[int, int, float]]:
        """Fallback Python implementation for interval aggregation.

        Args:
            slot_minutes: Time slot size in minutes

        Returns:
            List of (day_of_week, time_slot, total_occupied_seconds) tuples

        """
        _LOGGER.debug("Using Python fallback for interval aggregation")

        # Use the unified method to get occupied intervals
        extended_intervals = self.get_occupied_intervals()

        if not extended_intervals:
            return []

        # Aggregate extended intervals by time slots
        slot_seconds: defaultdict[tuple[int, int], float] = defaultdict(float)
        for start_time, end_time in extended_intervals:
            # Calculate which slots this interval covers
            current_time = start_time
            while current_time < end_time:
                day_of_week = current_time.weekday()
                hour = current_time.hour
                minute = current_time.minute
                slot = (hour * MINUTES_PER_HOUR + minute) // slot_minutes

                # Calculate how much of this interval falls in this slot
                slot_start = current_time.replace(
                    minute=(slot * slot_minutes) % MINUTES_PER_HOUR,
                    second=0,
                    microsecond=0,
                )
                slot_end = slot_start + timedelta(minutes=slot_minutes)

                # Calculate overlap duration
                overlap_start = max(current_time, slot_start)
                overlap_end = min(end_time, slot_end)
                overlap_duration = (overlap_end - overlap_start).total_seconds()

                if overlap_duration > 0:
                    slot_seconds[(day_of_week, slot)] += overlap_duration

                # Move to next slot
                current_time = slot_end

        # Convert to result format
        result = []
        for (day, slot), seconds in slot_seconds.items():
            result.append((day, slot, seconds))

        return result

    def get_time_bounds(
        self, entity_ids: list[str] | None = None
    ) -> tuple[datetime | None, datetime | None]:
        """Get time bounds for specific entities or a specific area.

        Args:
            entity_ids: List of entity IDs to get bounds for. If None, uses all entities for the area.

        Returns:
            Tuple of (first_time, last_time) or (None, None) if no data

        """
        _LOGGER.debug("Getting time bounds")
        db = self.db

        try:
            with db.get_session() as session:
                query = session.query(
                    func.min(db.Intervals.start_time).label("first"),
                    func.max(db.Intervals.end_time).label("last"),
                )

                if entity_ids is not None:
                    # Filter by specific entity IDs
                    query = query.filter(db.Intervals.entity_id.in_(entity_ids))
                else:
                    # Filter by area entry_id
                    entry_id = self.coordinator.entry_id
                    query = query.join(
                        db.Entities,
                        db.Intervals.entity_id == db.Entities.entity_id,
                    ).filter(db.Entities.entry_id == entry_id)

                time_bounds = query.first()
                return (
                    (time_bounds.first, time_bounds.last)
                    if time_bounds
                    else (None, None)
                )
        except OperationalError as e:
            _LOGGER.error("Database connection error getting time bounds: %s", e)
            return (None, None)
        except DataError as e:
            _LOGGER.error("Database data error getting time bounds: %s", e)
            return (None, None)
        except ProgrammingError as e:
            _LOGGER.error("Database query error getting time bounds: %s", e)
            return (None, None)
        except SQLAlchemyError as e:
            _LOGGER.error("Database error getting time bounds: %s", e)
            return (None, None)
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            _LOGGER.error("Unexpected error getting time bounds: %s", e)
            return (None, None)

    def get_total_occupied_seconds(self) -> float:
        """Get total occupied seconds using unified occupancy logic."""
        _LOGGER.debug("Getting total occupied seconds using unified logic")

        # Use the unified method to get occupied intervals
        occupied_intervals = self.get_occupied_intervals()

        if not occupied_intervals:
            return DEFAULT_OCCUPIED_SECONDS

        # Calculate total duration from occupied intervals
        total_seconds = 0.0
        for start_time, end_time in occupied_intervals:
            duration = (end_time - start_time).total_seconds()
            total_seconds += duration

        _LOGGER.debug("Total occupied seconds: %.1f", total_seconds)
        return total_seconds

    def get_occupied_intervals(
        self,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        include_media: bool = False,
        include_appliance: bool = False,
    ) -> list[tuple[datetime, datetime]]:
        """Get occupied time intervals from motion sensors using unified logic.

        This method provides a single source of truth for determining occupancy
        intervals that can be used by both prior and likelihood calculations.

        Args:
            lookback_days: Number of days to look back for interval data (default: 90)
            include_media: If True, include media player intervals (playing state only)
            include_appliance: If True, include appliance intervals (on state only)

        Returns:
            List of (start_time, end_time) tuples representing occupied periods

        """
        # Check cache first (only for motion-only queries to keep cache simple)
        now = dt_util.utcnow()
        if (
            not include_media
            and not include_appliance
            and self._cached_occupied_intervals is not None
            and self._cached_intervals_timestamp is not None
            and (now - self._cached_intervals_timestamp).total_seconds()
            < DEFAULT_CACHE_TTL_SECONDS
        ):
            _LOGGER.debug(
                "Returning cached occupied intervals (%d intervals)",
                len(self._cached_occupied_intervals),
            )
            return self._cached_occupied_intervals

        _LOGGER.debug(
            "Getting occupied intervals with unified logic (lookback: %d days, media: %s, appliance: %s)",
            lookback_days,
            include_media,
            include_appliance,
        )
        entry_id = self.coordinator.entry_id
        db = self.db

        # Calculate lookback date
        lookback_date = now - timedelta(days=lookback_days)

        all_intervals: list[tuple[datetime, datetime]] = []

        try:
            start_time = dt_util.utcnow()
            with db.get_session() as session:
                # Get motion sensor intervals with state='on' within lookback period
                motion_intervals = (
                    session.query(
                        db.Intervals.start_time,
                        db.Intervals.end_time,
                    )
                    .join(
                        db.Entities,
                        db.Intervals.entity_id == db.Entities.entity_id,
                    )
                    .filter(
                        db.Entities.entry_id == entry_id,
                        db.Entities.entity_type == InputType.MOTION,
                        db.Intervals.state == "on",
                        db.Intervals.start_time >= lookback_date,
                    )
                    .order_by(db.Intervals.start_time)
                    .all()
                )

                motion_raw = [(start, end) for start, end in motion_intervals]
                all_intervals.extend(motion_raw)

                # Get media player intervals if requested (playing state only)
                if include_media and self.media_sensor_ids:
                    media_intervals = (
                        session.query(
                            db.Intervals.start_time,
                            db.Intervals.end_time,
                        )
                        .join(
                            db.Entities,
                            db.Intervals.entity_id == db.Entities.entity_id,
                        )
                        .filter(
                            db.Entities.entry_id == entry_id,
                            db.Entities.entity_type == InputType.MEDIA,
                            db.Intervals.entity_id.in_(self.media_sensor_ids),
                            db.Intervals.state == STATE_PLAYING,
                            db.Intervals.start_time >= lookback_date,
                        )
                        .order_by(db.Intervals.start_time)
                        .all()
                    )
                    media_raw = [(start, end) for start, end in media_intervals]
                    all_intervals.extend(media_raw)
                    _LOGGER.debug(
                        "Found %d media player intervals (playing state only)",
                        len(media_raw),
                    )

                # Get appliance intervals if requested (on state only)
                if include_appliance and self.appliance_sensor_ids:
                    appliance_intervals = (
                        session.query(
                            db.Intervals.start_time,
                            db.Intervals.end_time,
                        )
                        .join(
                            db.Entities,
                            db.Intervals.entity_id == db.Entities.entity_id,
                        )
                        .filter(
                            db.Entities.entry_id == entry_id,
                            db.Entities.entity_type == InputType.APPLIANCE,
                            db.Intervals.entity_id.in_(self.appliance_sensor_ids),
                            db.Intervals.state == STATE_ON,
                            db.Intervals.start_time >= lookback_date,
                        )
                        .order_by(db.Intervals.start_time)
                        .all()
                    )
                    appliance_raw = [(start, end) for start, end in appliance_intervals]
                    all_intervals.extend(appliance_raw)
                    _LOGGER.debug(
                        "Found %d appliance intervals (on state only)",
                        len(appliance_raw),
                    )

                query_time = (dt_util.utcnow() - start_time).total_seconds()
                _LOGGER.debug(
                    "Query executed in %.3f seconds, found %d total intervals (motion: %d)",
                    query_time,
                    len(all_intervals),
                    len(motion_raw),
                )

                if not all_intervals:
                    _LOGGER.debug("No intervals found for occupancy calculation")
                    # Only cache if motion-only
                    if not include_media and not include_appliance:
                        self._cached_occupied_intervals = []
                        self._cached_intervals_timestamp = now
                    return []

                # Sort all intervals by start time for merging
                all_intervals.sort(key=lambda x: x[0])

                # Merge overlapping intervals
                merged_intervals: list[tuple[datetime, datetime]] = []
                for start, end in all_intervals:
                    if not merged_intervals:
                        merged_intervals.append((start, end))
                    else:
                        last_start, last_end = merged_intervals[-1]
                        if start <= last_end:
                            # Overlapping or adjacent, merge them
                            merged_intervals[-1] = (last_start, max(last_end, end))
                        else:
                            # Non-overlapping, add as new interval
                            merged_intervals.append((start, end))

                # Apply motion timeout only to motion sensor intervals
                # For media/appliances, we use the raw intervals directly
                # We need to split merged intervals at motion boundaries to avoid
                # incorrectly applying timeout to media/appliance portions
                extended_intervals = []
                timeout_seconds = self.config.sensors.motion_timeout

                for merged_start, merged_end in merged_intervals:
                    # Find all motion intervals that overlap with this merged interval
                    overlapping_motion = [
                        (m_start, m_end)
                        for m_start, m_end in motion_raw
                        if not (merged_end < m_start or merged_start > m_end)
                    ]

                    if not overlapping_motion:
                        # No motion overlap, use interval as-is (media/appliance only)
                        extended_intervals.append((merged_start, merged_end))
                        continue

                    # Calculate the union of motion coverage within this merged interval
                    # Sort motion intervals by start time
                    overlapping_motion.sort(key=lambda x: x[0])
                    motion_union_start = min(
                        m_start for m_start, _ in overlapping_motion
                    )
                    motion_union_end = max(m_end for _, m_end in overlapping_motion)
                    # Clamp to merged interval boundaries
                    motion_union_start = max(motion_union_start, merged_start)
                    motion_union_end = min(motion_union_end, merged_end)

                    # Split the merged interval into segments:
                    # 1. Before motion (if any) - no timeout
                    # 2. Motion portion - apply timeout
                    # 3. After motion (if any) - no timeout

                    if merged_start < motion_union_start:
                        # Segment before motion: no timeout
                        extended_intervals.append((merged_start, motion_union_start))

                    # Motion segment: apply timeout
                    extended_intervals.append(
                        (
                            motion_union_start,
                            motion_union_end + timedelta(seconds=timeout_seconds),
                        )
                    )

                    if motion_union_end < merged_end:
                        # Segment after motion: no timeout
                        extended_intervals.append((motion_union_end, merged_end))

                # Merge again after extending motion intervals
                if extended_intervals:
                    final_intervals: list[tuple[datetime, datetime]] = []
                    extended_intervals.sort(key=lambda x: x[0])
                    for start, end in extended_intervals:
                        if not final_intervals:
                            final_intervals.append((start, end))
                        else:
                            last_start, last_end = final_intervals[-1]
                            if start <= last_end:
                                final_intervals[-1] = (
                                    last_start,
                                    max(last_end, end),
                                )
                            else:
                                final_intervals.append((start, end))
                    extended_intervals = final_intervals

                processing_time = (dt_util.utcnow() - start_time).total_seconds()

                _LOGGER.debug(
                    "Unified occupancy calculation: %d raw intervals -> %d merged intervals (processing: %.3fs)",
                    len(all_intervals),
                    len(extended_intervals),
                    processing_time,
                )

                # Cache the result only if motion-only
                if not include_media and not include_appliance:
                    self._cached_occupied_intervals = extended_intervals
                    self._cached_intervals_timestamp = now

                return extended_intervals

        except (OperationalError, SQLAlchemyError, DataError, ProgrammingError) as e:
            _LOGGER.error("Database error getting occupied intervals: %s", e)
            return []
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            _LOGGER.error("Unexpected error getting occupied intervals: %s", e)
            return []

    def to_dict(self) -> dict[str, Any]:
        """Convert prior to dictionary for storage."""
        return {
            "value": self.global_prior,
            "last_updated": (
                self._last_updated.isoformat() if self._last_updated else None
            ),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: AreaOccupancyCoordinator
    ) -> Prior:
        """Create prior from dictionary."""
        prior = cls(coordinator)
        prior.global_prior = data["value"]
        prior._last_updated = (
            datetime.fromisoformat(data["last_updated"])
            if data["last_updated"]
            else None
        )
        return prior
