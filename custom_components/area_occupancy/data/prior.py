"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import State
from homeassistant.util import dt as dt_util

from ..utils import TimeInterval, get_states_from_recorder, states_to_intervals

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

DEFAULT_PRIOR = 0.15
MIN_PRIOR = 0.1


@dataclass
class PriorData:
    """Compute and cache the baseline probability for an Area entity."""

    entity_id: str
    start_time: datetime
    end_time: datetime
    states: list[State]
    intervals: list[TimeInterval]
    occupied_seconds: int
    ratio: float


# ─────────────────────────────────────────────────────────────────────────────
class Prior:  # exported name must stay identical
    """Compute and cache the baseline probability for an Area entity."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Initialize the Prior class."""
        self.sensor_ids = coordinator.config.sensors.motion
        self.days = coordinator.config.history.period
        self.hass = coordinator.hass
        self.cache_ttl = timedelta(hours=2)
        self.value: float | None = None
        self.last_updated: datetime | None = None
        self.sensor_hash: int | None = None
        self.data: dict[str, PriorData] = {}

    # --------------------------------------------------------------------- #
    @property
    def current_value(self) -> float:
        """Return the current cached prior value, or default if not yet calculated."""
        if self.value is None or self.value < MIN_PRIOR:
            return DEFAULT_PRIOR
        return self.value

    @property
    def prior_intervals(self) -> list[TimeInterval]:
        """Return the merged occupied intervals (state='on') deduplicated across all motion sensors."""
        # Collect all 'on' intervals from all sensors
        all_on_intervals = [
            interval
            for data in self.data.values()
            for interval in data.intervals
            if interval["state"] == "on"
        ]
        if not all_on_intervals:
            return []

        # Sort intervals by start time
        sorted_intervals = sorted(all_on_intervals, key=lambda x: x["start"])

        # Merge overlapping or contiguous intervals
        merged: list[TimeInterval] = []
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
                for interval in self.prior_intervals
            )
        )

    # --------------------------------------------------------------------- #
    async def update(self, force: bool = False) -> float:
        """Return a baseline prior, re-computing if the cache is stale or forced.

        Args:
            force: If True, bypass cache validation and force recalculation

        Returns:
            The calculated or cached prior value

        """
        if not force and self._is_cache_valid():
            return self.value  # type: ignore[return-value]

        try:
            value = await self.calculate()
        except Exception:  # pragma: no cover
            _LOGGER.exception(
                "Prior calculation failed, using default %.2f", DEFAULT_PRIOR
            )
            value = DEFAULT_PRIOR

        return value

    async def calculate(self) -> float:
        """Calculate the area prior."""
        start_time = dt_util.utcnow() - timedelta(days=self.days)
        end_time = dt_util.utcnow()
        total_seconds = int((end_time - start_time).total_seconds())
        for sensor_id in self.sensor_ids:
            states = await get_states_from_recorder(
                self.hass, sensor_id, start_time, end_time
            )
            if states:
                intervals = await states_to_intervals(
                    [s for s in states if isinstance(s, State)], start_time, end_time
                )
                occupied_seconds = int(
                    sum(
                        (interval["end"] - interval["start"]).total_seconds()
                        for interval in intervals
                        if interval["state"] == "on"
                    )
                )
                self.data[sensor_id] = PriorData(
                    entity_id=sensor_id,
                    start_time=start_time,
                    end_time=end_time,
                    states=[s for s in states if isinstance(s, State)],
                    intervals=intervals,
                    occupied_seconds=occupied_seconds,
                    ratio=occupied_seconds / total_seconds,
                )

        self.value = (
            sum(data.ratio for data in self.data.values()) / len(self.data)
        ) * 1.05  # 5% buffer to account for sensor noise
        self.last_updated = dt_util.utcnow()
        self.sensor_hash = hash(frozenset(self.sensor_ids))

        for sensor_id, data in self.data.items():
            _LOGGER.debug(
                "Sensor %s: %.3f",
                sensor_id,
                data.ratio,
            )

        _LOGGER.debug(
            "Calculated new area prior: %d sensors, %.3f",
            len(self.data),
            self.value,
        )

        return self.value

    # ------------------------------------------------------------------ #
    def _is_cache_valid(self) -> bool:
        if self.value is None or self.last_updated is None:
            return False
        if (dt_util.utcnow() - self.last_updated) > self.cache_ttl:
            return False
        # in case sensors were added/removed
        return self.sensor_hash == hash(frozenset(self.sensor_ids))

    def to_dict(self) -> dict[str, Any]:
        """Convert prior to dictionary for storage."""
        return {
            "value": self.value,
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
            "sensor_hash": self.sensor_hash,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: AreaOccupancyCoordinator
    ) -> Prior:
        """Create prior from dictionary."""
        prior = cls(coordinator)
        prior.value = data["value"]
        prior.last_updated = (
            datetime.fromisoformat(data["last_updated"])
            if data["last_updated"]
            else None
        )
        prior.sensor_hash = data["sensor_hash"]
        return prior
