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

from homeassistant.util import dt as dt_util

from ..const import MAX_PRIOR, MIN_PRIOR
from ..data.entity_type import InputType

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

PRIOR_FACTOR = 1.1


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

    @property
    def value(self) -> float:
        """Return the current prior value or minimum if not calculated."""
        if self.global_prior is None:
            return MIN_PRIOR
        return min(max(self.global_prior * PRIOR_FACTOR, MIN_PRIOR), MAX_PRIOR)

    @property
    def last_updated(self) -> datetime | None:
        """Return the last updated timestamp."""
        return self._last_updated

    def set_global_prior(self, prior: float) -> None:
        """Set the global prior value."""
        self.global_prior = prior
        self._last_updated = dt_util.utcnow()

    async def update(self) -> None:
        """Calculate and update the prior value."""
        try:
            self.global_prior = self.calculate_area_prior(self.sensor_ids)
        except Exception:
            _LOGGER.exception("Prior calculation failed, using default %.2f", MIN_PRIOR)
            self.global_prior = MIN_PRIOR

        self._last_updated = dt_util.utcnow()

    def calculate_area_prior(self, entity_ids: list[str]) -> float:
        """Calculate the overall occupancy prior for an area based on historical motion sensor data.

        Args:
            entity_ids: List of entity IDs to calculate prior for. If None, uses all motion sensors.

        Returns:
            float: Prior probability of occupancy (0.0 to 1.0)

        """
        _LOGGER.debug("Calculating area prior")

        # Get total occupied time from motion sensors
        total_occupied_seconds = self.get_total_occupied_seconds(entity_ids)

        # Get total time period
        first_time, last_time = self.get_time_bounds(entity_ids)

        if not first_time or not last_time or total_occupied_seconds == 0:
            return 0.5  # Default prior if no data

        total_seconds = (last_time - first_time).total_seconds()

        if total_seconds <= 0:
            return 0.5

        return total_occupied_seconds / total_seconds

    def compute_time_priors(self, slot_minutes: int = 60):
        """Estimate P(occupied) per day_of_week and time_slot from motion sensor intervals."""
        _LOGGER.debug("Computing time priors")

        # Get aggregated interval data
        interval_aggregates = self.get_interval_aggregates(slot_minutes)

        # Get time bounds
        first_time, last_time = self.get_time_bounds()

        if not first_time or not last_time:
            return

        # Calculate total time period
        days = (last_time.date() - first_time.date()).days + 1
        slots_per_day = (24 * 60) // slot_minutes
        slot_duration_seconds = slot_minutes * 60.0

        # Create lookup dictionary from aggregated results
        occupied_seconds = {}
        for day, slot, total_seconds in interval_aggregates:
            # Convert PostgreSQL day_of_week (0=Sunday) to Python weekday (0=Monday)
            python_weekday = (int(day) + 6) % 7
            occupied_seconds[(python_weekday, int(slot))] = float(total_seconds or 0)

        # Generate priors for all time slots
        now = datetime.now(UTC)
        db = self.db

        with db.get_session() as session:
            for day in range(7):
                for slot in range(slots_per_day):
                    total_slot_seconds = days * slot_duration_seconds
                    occupied_slot_seconds = occupied_seconds.get((day, slot), 0.0)

                    # Calculate probability
                    p = (
                        occupied_slot_seconds / total_slot_seconds
                        if total_slot_seconds > 0
                        else 0.0
                    )

                    # Create the prior object within the database's session
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

    def get_interval_aggregates(
        self, slot_minutes: int = 60
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

        with db.get_session() as session:
            interval_aggregates = (
                session.query(
                    func.extract("dow", db.Intervals.start_time).label("day_of_week"),
                    func.floor(
                        (
                            func.extract("hour", db.Intervals.start_time) * 60
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
            return [
                (day, slot, total_seconds)
                for day, slot, total_seconds in interval_aggregates
            ]

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
                (time_bounds.first, time_bounds.last) if time_bounds else (None, None)
            )

    def get_total_occupied_seconds(self, entity_ids: list[str]) -> float:
        """Get total occupied seconds for specific entities."""
        _LOGGER.debug("Getting total occupied seconds")
        db = self.db

        with db.get_session() as session:
            query = session.query(func.sum(db.Intervals.duration_seconds)).filter(
                db.Intervals.state == "on",
                db.Intervals.entity_id.in_(entity_ids),
            )
            result = query.scalar()
            return float(result or 0)

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
