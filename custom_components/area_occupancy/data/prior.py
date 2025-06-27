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

# Minimum prior value to avoid division by zero (1%)
MIN_PRIOR = 0.01

# Interval filtering thresholds to exclude anomalous data
# Exclude intervals shorter than 10 seconds (false triggers)
MIN_INTERVAL_SECONDS = 10
# Exclude intervals longer than 13 hours (stuck sensors)
MAX_INTERVAL_SECONDS = 13 * 3600


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
    # Filtering statistics
    total_on_intervals: int
    filtered_short_intervals: int  # Count of intervals < MIN_INTERVAL_SECONDS
    filtered_long_intervals: int  # Count of intervals > MAX_INTERVAL_SECONDS
    valid_intervals: int  # Count of intervals used in calculation
    max_filtered_duration_seconds: float | None  # Longest filtered interval duration


# ─────────────────────────────────────────────────────────────────────────────
class Prior:  # exported name must stay identical
    """Compute and cache the baseline probability for an Area entity."""

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
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
            return MIN_PRIOR
        return self.value

    @property
    def prior_intervals(self) -> list[TimeInterval]:
        """Return the merged occupied intervals (state='on') deduplicated across all motion sensors with anomaly filtering."""
        # Collect all 'on' intervals from all sensors, applying anomaly filtering
        all_on_intervals = []
        for data in self.data.values():
            for interval in data.intervals:
                if interval["state"] == "on":
                    duration_seconds = (
                        interval["end"] - interval["start"]
                    ).total_seconds()
                    # Apply the same filtering logic as in calculate()
                    if MIN_INTERVAL_SECONDS <= duration_seconds <= MAX_INTERVAL_SECONDS:
                        all_on_intervals.append(interval)

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

        for sensor_id in self.sensor_ids:
            states = await get_states_from_recorder(
                self.hass, sensor_id, start_time, end_time
            )
            if states:
                intervals = await states_to_intervals(
                    [s for s in states if isinstance(s, State)], start_time, end_time
                )

                # Filter and categorize intervals
                on_intervals = [
                    interval for interval in intervals if interval["state"] == "on"
                ]
                total_on_intervals = len(on_intervals)

                # Apply anomaly filtering
                valid_intervals = []
                filtered_short = 0
                filtered_long = 0
                max_filtered_duration = None

                for interval in on_intervals:
                    duration_seconds = (
                        interval["end"] - interval["start"]
                    ).total_seconds()

                    if duration_seconds < MIN_INTERVAL_SECONDS:
                        filtered_short += 1
                        _LOGGER.debug(
                            "Sensor %s: Filtered short interval (%.1fs) from %s to %s",
                            sensor_id,
                            duration_seconds,
                            interval["start"],
                            interval["end"],
                        )
                    elif duration_seconds > MAX_INTERVAL_SECONDS:
                        filtered_long += 1
                        # Track the maximum filtered duration
                        if (
                            max_filtered_duration is None
                            or duration_seconds > max_filtered_duration
                        ):
                            max_filtered_duration = duration_seconds
                        _LOGGER.debug(
                            "Sensor %s: Filtered long interval (%.1fh) from %s to %s",
                            sensor_id,
                            duration_seconds / 3600,
                            interval["start"],
                            interval["end"],
                        )
                    else:
                        valid_intervals.append(interval)

                # Calculate occupied seconds from valid intervals only
                occupied_seconds = int(
                    sum(
                        (interval["end"] - interval["start"]).total_seconds()
                        for interval in valid_intervals
                    )
                )

                # Log filtering results
                if filtered_short > 0 or filtered_long > 0:
                    _LOGGER.info(
                        "Sensor %s: Filtered %d short and %d long intervals, kept %d valid intervals",
                        sensor_id,
                        filtered_short,
                        filtered_long,
                        len(valid_intervals),
                    )

                self.data[sensor_id] = PriorData(
                    entity_id=sensor_id,
                    start_time=start_time,
                    end_time=end_time,
                    states=[s for s in states if isinstance(s, State)],
                    intervals=intervals,
                    occupied_seconds=occupied_seconds,
                    ratio=occupied_seconds / total_seconds,
                    total_on_intervals=total_on_intervals,
                    filtered_short_intervals=filtered_short,
                    filtered_long_intervals=filtered_long,
                    valid_intervals=len(valid_intervals),
                    max_filtered_duration_seconds=max_filtered_duration,
                )

        self.value = (
            sum(data.ratio for data in self.data.values()) / len(self.data)
        ) * 1.05  # 5% buffer to account for sensor noise
        self.last_updated = dt_util.utcnow()
        self.sensor_hash = hash(frozenset(self.sensor_ids))

        for sensor_id, data in self.data.items():
            _LOGGER.debug(
                "Sensor %s: %.3f (used %d/%d intervals)",
                sensor_id,
                data.ratio,
                data.valid_intervals,
                data.total_on_intervals,
            )

        _LOGGER.debug(
            "Calculated new area prior: %d sensors, %.3f", len(self.data), self.value
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
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
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
