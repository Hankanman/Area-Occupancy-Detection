"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from ..utils import StateInterval, get_intervals_hybrid

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Minimum prior value to avoid division by zero (1%)
MIN_PRIOR = 0.1


# ─────────────────────────────────────────────────────────────────────────────
class Prior:  # exported name must stay identical
    """Compute and cache the baseline probability for an Area entity."""

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the Prior class."""
        self.coordinator = coordinator
        self.sensor_ids = coordinator.config.sensors.motion
        self.days = coordinator.config.history.period
        self.hass = coordinator.hass
        self.cache_ttl = timedelta(hours=2)
        self._current_value: float | None = None
        self._last_updated: datetime | None = None
        self._sensor_hash: int | None = None
        self._sensor_data: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------------------- #
    @property
    def value(self) -> float:
        """Return the current cached prior value, or default if not yet calculated."""
        if self._current_value is None or self._current_value < MIN_PRIOR:
            return MIN_PRIOR
        return self._current_value

    @property
    def state_intervals(self) -> list[StateInterval]:
        """Return the merged occupied intervals (state='on') deduplicated across all motion sensors with anomaly filtering."""
        # Collect all intervals from all sensors (already filtered during calculation)
        all_intervals = []
        for data in self._sensor_data.values():
            all_intervals.extend(data.get("intervals", []))

        if not all_intervals:
            return []

        # Sort intervals by start time
        sorted_intervals = sorted(all_intervals, key=lambda x: x["start"])

        # Merge overlapping or contiguous intervals
        merged: list[StateInterval] = []
        for interval in sorted_intervals:
            if not merged:
                merged.append(interval.copy())
            else:
                last = merged[-1]
                # If current interval starts before or at the end of the last, merge them
                if interval["start"] <= last["end"]:
                    last["end"] = max(last["end"], interval["end"])
                else:
                    merged.append(interval.copy())
        return merged

    @property
    def prior_total_seconds(self) -> int:
        """Return the total duration of the prior intervals."""
        return int(
            sum(
                (interval["end"] - interval["start"]).total_seconds()
                for interval in self.state_intervals
            )
        )

    @property
    def data(self) -> dict[str, Any]:
        """Return the data for the prior."""
        return self._sensor_data

    # --------------------------------------------------------------------- #
    async def update(
        self, force: bool = False, history_period: int | None = None
    ) -> float:
        """Return a baseline prior, re-computing if the cache is stale or forced.

        Args:
            force: If True, bypass cache validation and force recalculation
            history_period: Period in days for historical data (overrides coordinator default)

        Returns:
            The calculated or cached prior value

        """
        if not force and self._is_cache_valid():
            return self.value  # type: ignore[return-value]

        try:
            value = await self.calculate(history_period=history_period)
        except Exception:  # pragma: no cover
            _LOGGER.exception("Prior calculation failed, using default %.2f", MIN_PRIOR)
            value = MIN_PRIOR

        return value

    async def calculate(self, history_period: int | None = None) -> float:
        """Calculate the area prior with anomaly filtering.

        Args:
            history_period: Period in days for historical data (overrides coordinator default)

        """
        # Use provided history_period or fall back to coordinator default
        days_to_use = history_period if history_period is not None else self.days
        start_time = dt_util.utcnow() - timedelta(days=days_to_use)
        end_time = dt_util.utcnow()
        total_seconds = int((end_time - start_time).total_seconds())

        # --- Standard logic: all motion sensors ---
        all_sensors_prior, all_sensors_data = await self._calculate_prior_for_entities(
            self.sensor_ids, start_time, end_time, total_seconds
        )

        # --- Additional logic: occupancy_entity_id only ---
        occupancy_entity_id = self.coordinator.occupancy_entity_id
        occupancy_entity_data = {}
        if occupancy_entity_id:
            try:
                (
                    occupancy_entity_prior,
                    occupancy_entity_data,
                ) = await self._calculate_prior_for_entities(
                    [occupancy_entity_id], start_time, end_time, total_seconds
                )
            except (ValueError, TypeError, RuntimeError):
                _LOGGER.warning(
                    "Failed to calculate prior for occupancy_entity_id %s",
                    occupancy_entity_id,
                )
                occupancy_entity_prior = MIN_PRIOR
        else:
            occupancy_entity_prior = MIN_PRIOR

        # Take the max of both priors and store the corresponding data
        if occupancy_entity_prior > all_sensors_prior:
            final_prior = occupancy_entity_prior
            self._sensor_data = occupancy_entity_data
            self._prior_source = "occupancy_entity_id"
            self._prior_source_entity_ids = (
                [occupancy_entity_id] if occupancy_entity_id else []
            )
        else:
            final_prior = all_sensors_prior
            self._sensor_data = all_sensors_data
            self._prior_source = "input_sensors"
            self._prior_source_entity_ids = self.sensor_ids

        self._current_value = final_prior
        self._last_updated = dt_util.utcnow()
        self._sensor_hash = hash(frozenset(self._prior_source_entity_ids))

        # Store for debugging
        self._all_sensors_prior = all_sensors_prior
        self._occupancy_entity_prior = occupancy_entity_prior

        for sensor_id, data in self._sensor_data.items():
            _LOGGER.debug(
                "Sensor %s: %.3f (used %d intervals)",
                sensor_id,
                data["ratio"],
                len(data["intervals"]),
            )

        _LOGGER.debug(
            "Calculated new area prior: %d sensors, %.3f (all_sensors_prior=%.3f, occupancy_entity_prior=%.3f, source=%s)",
            len(self._sensor_data),
            self._current_value,
            all_sensors_prior,
            occupancy_entity_prior,
            self._prior_source,
        )

        return self._current_value

    async def _calculate_prior_for_entities(
        self,
        entity_ids: list[str],
        start_time: datetime,
        end_time: datetime,
        total_seconds: int,
    ) -> tuple[float, dict[str, dict[str, Any]]]:
        """Calculate the prior for a given list of entity_ids.

        Returns:
            Tuple[prior, data]: prior value and the data dictionary

        """
        data = {}
        for entity_id in entity_ids:
            # Get intervals using hybrid approach - checks our DB first, then recorder
            intervals = await get_intervals_hybrid(
                self.coordinator,
                entity_id,
                start_time,
                end_time,
            )

            # Intervals are already filtered by get_intervals_hybrid
            valid_intervals = intervals

            if valid_intervals:
                occupied_seconds = int(
                    sum(
                        (interval["end"] - interval["start"]).total_seconds()
                        for interval in valid_intervals
                    )
                )

                data[entity_id] = {
                    "entity_id": entity_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "states_count": len(
                        valid_intervals
                    ),  # Number of intervals instead of states
                    "intervals": valid_intervals,  # Store filtered intervals, not raw
                    "occupied_seconds": occupied_seconds,
                    "ratio": occupied_seconds / total_seconds,
                }
        if data:
            prior = (sum(d["ratio"] for d in data.values()) / len(data)) * 1.05
        else:
            prior = MIN_PRIOR
        return prior, data

    # ------------------------------------------------------------------ #
    def _is_cache_valid(self) -> bool:
        if self._current_value is None or self._last_updated is None:
            return False
        if (dt_util.utcnow() - self._last_updated) > self.cache_ttl:
            return False
        # in case sensors were added/removed
        return self._sensor_hash == hash(frozenset(self.sensor_ids))

    def to_dict(self) -> dict[str, Any]:
        """Convert prior to dictionary for storage."""
        return {
            "value": self._current_value,
            "last_updated": (
                self._last_updated.isoformat() if self._last_updated else None
            ),
            "sensor_hash": self._sensor_hash,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: AreaOccupancyCoordinator
    ) -> Prior:
        """Create prior from dictionary."""
        prior = cls(coordinator)
        prior._current_value = data["value"]
        prior._last_updated = (
            datetime.fromisoformat(data["last_updated"])
            if data["last_updated"]
            else None
        )
        prior._sensor_hash = data["sensor_hash"]
        return prior
