"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from datetime import UTC, datetime
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

from homeassistant.util import dt as dt_util

from ..const import MAX_PRIOR, MAX_PROBABILITY, MIN_PRIOR, MIN_PROBABILITY
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

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the Prior class."""
        self.coordinator = coordinator
        self.db = coordinator.db
        self.sensor_ids = coordinator.config.sensors.motion
        self.hass = coordinator.hass
        self.global_prior: float | None = None
        self._last_updated: datetime | None = None
        # Cache for time_prior
        self._cached_time_prior: float | None = None
        self._cached_time_prior_day: int | None = None
        self._cached_time_prior_slot: int | None = None

    @property
    def value(self) -> float:
        """Return the current prior value or minimum if not calculated."""
        if self.global_prior is None:
            return MIN_PRIOR

        if self.time_prior is None:
            prior = self.global_prior
        else:
            prior = combine_priors(self.global_prior, self.time_prior)

        # Validate that global_prior is within reasonable bounds before applying factor
        if not (MIN_PROBABILITY <= prior <= MAX_PROBABILITY):
            _LOGGER.warning(
                "Global prior %.4f is outside valid range [%.1f, %.1f], clamping to bounds",
                prior,
                MIN_PROBABILITY,
                MAX_PROBABILITY,
            )
            prior = clamp_probability(prior)

        # Apply factor and clamp to bounds
        adjusted_prior = prior * PRIOR_FACTOR
        result = clamp_probability(adjusted_prior)

        # Log if the factor caused a significant change
        if abs(result - prior) > SIGNIFICANT_CHANGE_THRESHOLD:
            _LOGGER.debug(
                "Prior adjusted by factor: %.4f -> %.4f (factor: %.2f, bounds: [%.2f, %.2f])",
                prior,
                result,
                PRIOR_FACTOR,
                MIN_PRIOR,
                MAX_PRIOR,
            )

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

    async def update(self) -> None:
        """Calculate and update the prior value."""
        try:
            self.global_prior = self.calculate_area_prior(self.sensor_ids)
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
            self.compute_time_priors()
            _LOGGER.debug("Time priors calculated")
            # Invalidate time_prior cache since we updated the time priors
            self._invalidate_time_prior_cache()
        except SQLAlchemyError:
            _LOGGER.exception("Time prior calculation failed due to database error")
        except (ValueError, TypeError):
            _LOGGER.exception("Time prior calculation failed due to data error")
        except Exception:
            _LOGGER.exception("Time prior calculation failed due to unexpected error")

        self._last_updated = dt_util.utcnow()

    def calculate_area_prior(self, entity_ids: list[str]) -> float:
        """Calculate the overall occupancy prior for an area based on historical motion sensor data.

        Args:
            entity_ids: List of entity IDs to calculate prior for. If None, uses all motion sensors.

        Returns:
            float: Prior probability of occupancy (0.0 to 1.0)

        """
        _LOGGER.debug("Calculating area prior")

        # Validate input
        if not entity_ids:
            _LOGGER.warning("No entity IDs provided for prior calculation")
            return DEFAULT_PRIOR

        # Get total occupied time from motion sensors
        total_occupied_seconds = self.get_total_occupied_seconds(entity_ids)

        # Get total time period
        first_time, last_time = self.get_time_bounds(entity_ids)

        if not first_time or not last_time or total_occupied_seconds == 0:
            return DEFAULT_PRIOR  # Default prior if no data

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

        prior = total_occupied_seconds / total_seconds

        # Ensure result is within valid probability bounds
        prior = clamp_probability(prior)

        _LOGGER.debug(
            "Calculated prior: %.4f (occupied: %.2fs, total: %.2fs)",
            prior,
            total_occupied_seconds,
            total_seconds,
        )

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
                return prior.prior_value if prior else DEFAULT_PRIOR
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

    def compute_time_priors(self, slot_minutes: int = DEFAULT_SLOT_MINUTES):
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
        now = datetime.now(UTC)
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
        """Get aggregated interval data for time prior computation.

        Args:
            entry_id: Area entry ID
            slot_minutes: Time slot size in minutes

        Returns:
            List of (day_of_week, time_slot, total_occupied_seconds) tuples

        """
        _LOGGER.debug("Getting interval aggregates")
        entry_id = self.coordinator.entry_id
        db = self.db

        try:
            with db.get_session() as session:
                interval_aggregates = (
                    session.query(
                        func.extract("dow", db.Intervals.start_time).label(
                            "day_of_week"
                        ),
                        func.floor(
                            (
                                func.extract("hour", db.Intervals.start_time)
                                * MINUTES_PER_HOUR
                                + func.extract("minute", db.Intervals.start_time)
                            )
                            / slot_minutes
                        ).label("time_slot"),
                        func.sum(db.Intervals.duration_seconds).label(
                            "total_occupied_seconds"
                        ),
                    )
                    .join(
                        db.Entities,
                        db.Intervals.entity_id == db.Entities.entity_id,
                    )
                    .filter(
                        db.Entities.entry_id == entry_id,
                        db.Entities.entity_type == InputType.MOTION,
                        db.Intervals.state == "on",
                    )
                    .group_by("day_of_week", "time_slot")
                    .all()
                )

                # Validate and convert results safely
                result = []
                for day, slot, total_seconds in interval_aggregates:
                    try:
                        day_int = int(day) if day is not None else 0
                        slot_int = int(slot) if slot is not None else 0
                        seconds_float = float(total_seconds or DEFAULT_OCCUPIED_SECONDS)
                        result.append((day_int, slot_int, seconds_float))
                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(
                            "Invalid interval aggregate data: day=%s, slot=%s, seconds=%s, error=%s",
                            day,
                            slot,
                            total_seconds,
                            e,
                        )
                        continue

                return result
        except OperationalError as e:
            _LOGGER.error(
                "Database connection error getting interval aggregates: %s", e
            )
            return []
        except DataError as e:
            _LOGGER.error("Database data error getting interval aggregates: %s", e)
            return []
        except ProgrammingError as e:
            _LOGGER.error("Database query error getting interval aggregates: %s", e)
            return []
        except SQLAlchemyError as e:
            _LOGGER.error("Database error getting interval aggregates: %s", e)
            return []
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            _LOGGER.error("Unexpected error getting interval aggregates: %s", e)
            return []

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

    def get_total_occupied_seconds(self, entity_ids: list[str]) -> float:
        """Get total occupied seconds for specific entities."""
        _LOGGER.debug("Getting total occupied seconds")
        db = self.db

        try:
            with db.get_session() as session:
                query = session.query(func.sum(db.Intervals.duration_seconds)).filter(
                    db.Intervals.state == "on",
                    db.Intervals.entity_id.in_(entity_ids),
                )
                result = query.scalar()
                return float(result or DEFAULT_OCCUPIED_SECONDS)
        except OperationalError as e:
            _LOGGER.error(
                "Database connection error getting total occupied seconds: %s", e
            )
            return DEFAULT_OCCUPIED_SECONDS
        except DataError as e:
            _LOGGER.error("Database data error getting total occupied seconds: %s", e)
            return DEFAULT_OCCUPIED_SECONDS
        except ProgrammingError as e:
            _LOGGER.error("Database query error getting total occupied seconds: %s", e)
            return DEFAULT_OCCUPIED_SECONDS
        except SQLAlchemyError as e:
            _LOGGER.error("Database error getting total occupied seconds: %s", e)
            return DEFAULT_OCCUPIED_SECONDS
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            _LOGGER.error("Unexpected error getting total occupied seconds: %s", e)
            return DEFAULT_OCCUPIED_SECONDS

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
