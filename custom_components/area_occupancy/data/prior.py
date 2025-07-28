"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from ..const import HA_RECORDER_DAYS, MAX_PRIOR, MIN_PRIOR, STATE_ON
from ..state_intervals import StateInterval, merge_intervals

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

PRIOR_FACTOR = 1.2


class Prior:
    """Compute the baseline probability for an Area entity."""

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the Prior class."""
        self.coordinator = coordinator
        self.sensor_ids = coordinator.config.sensors.motion
        self.hass = coordinator.hass
        self.global_prior: float | None = None
        self.primary_sensors_prior: float | None = None
        self.occupancy_prior: float | None = None
        self._prior_intervals: list[StateInterval] | None = None
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

    @property
    def prior_intervals(self) -> list[StateInterval] | None:
        """Return the prior intervals."""
        return self._prior_intervals

    def set_global_prior(self, prior: float) -> None:
        """Set the global prior value."""
        self.global_prior = prior
        self._last_updated = dt_util.utcnow()

    async def update(self) -> None:
        """Calculate and update the prior value."""
        try:
            entities = self.sensor_ids
            if self.coordinator.config.wasp_in_box.enabled:
                entities.append(self.coordinator.wasp_entity_id)
            (
                primary_sensors_prior_value,
                primary_sensors_merged_intervals,
            ) = await self._calculate_prior(entities)
            (
                occupancy_prior_value,
                occupancy_merged_intervals,
            ) = await self._calculate_prior([self.coordinator.occupancy_entity_id])
            self.primary_sensors_prior = primary_sensors_prior_value
            self.occupancy_prior = occupancy_prior_value
            if primary_sensors_prior_value > occupancy_prior_value:
                self.global_prior = primary_sensors_prior_value
                self._prior_intervals = primary_sensors_merged_intervals
            else:
                self.global_prior = occupancy_prior_value
                self._prior_intervals = occupancy_merged_intervals
        except Exception:
            _LOGGER.exception("Prior calculation failed, using default %.2f", MIN_PRIOR)
            self.global_prior = MIN_PRIOR

        self._last_updated = dt_util.utcnow()

    async def _calculate_prior(
        self, entity_ids: list[str]
    ) -> tuple[float, list[StateInterval]]:
        """Calculate the global prior based on historical data."""
        start_time = dt_util.utcnow() - timedelta(days=HA_RECORDER_DAYS)
        end_time = dt_util.utcnow()
        total_seconds = int((end_time - start_time).total_seconds())

        all_intervals: list[StateInterval] = []
        for entity_id in entity_ids:
            intervals = await self.coordinator.storage.get_historical_intervals(
                entity_id,
                start_time,
                end_time,
                state_filter=STATE_ON,  # Only include "on" state intervals
            )
            all_intervals.extend(intervals)

        # Merge overlapping intervals across all entities
        merged_intervals = merge_intervals(all_intervals)

        # Calculate occupied seconds from merged intervals only
        occupied_seconds = int(
            sum(
                (interval["end"] - interval["start"]).total_seconds()
                for interval in merged_intervals
            )
        )

        if total_seconds <= 0:
            prior_value = MIN_PRIOR
        else:
            prior_value = occupied_seconds / total_seconds

        return prior_value, merged_intervals

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
