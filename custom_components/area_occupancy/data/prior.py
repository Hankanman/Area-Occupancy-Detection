"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from ..const import HA_RECORDER_DAYS, MIN_PRIOR
from ..state_intervals import StateInterval
from ..utils import get_current_time_slot, time_slot_to_datetime_range

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

MIN_PRIOR_OCCUPIED_SECONDS = (
    120  # Minimum occupied seconds required for a prior to be considered valid
)

SMOOTHING_FACTOR = 1.0
MAX_PRIOR = 0.95
GLOBAL_PRIOR_FACTOR = 1.2
PRIOR_BUFFER_FACTOR = 1.05


@dataclass
class PriorCacheEntry:
    """Cache entry for a prior value."""

    prior: float
    occupied_seconds: int
    total_seconds: int


# ─────────────────────────────────────────────────────────────────────────────
class Prior:  # exported name must stay identical
    """Compute and cache the baseline probability for an Area entity."""

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the Prior class."""
        self.coordinator = coordinator
        self.sensor_ids = coordinator.config.sensors.motion
        self.days = HA_RECORDER_DAYS
        self.hass = coordinator.hass
        self.cache_ttl = timedelta(hours=2)
        self._current_value: float | None = None
        self._prior_intervals: list[StateInterval] | None = None
        self._last_updated: datetime | None = None
        self._sensor_hash: int | None = None
        # Time-based prior cache
        self._time_prior_cache: dict[tuple[int, int], PriorCacheEntry] = {}
        # Day-based prior cache
        self._day_prior_cache: dict[int, PriorCacheEntry] = {}
        # Global prior cache
        self._global_prior_cache: PriorCacheEntry | None = None

    # --------------------------------------------------------------------- #
    @property
    def value(self) -> float:
        """Return the best available prior: time-based, day, global, or minimum."""
        # 1. Try time slot prior
        slot = self.get_time_slot_prior()
        if (
            slot
            and slot.occupied_seconds >= MIN_PRIOR_OCCUPIED_SECONDS
            and slot.prior >= MIN_PRIOR
        ):
            return slot.prior
        # 2. Try day prior
        day = self.get_day_prior()
        if (
            day
            and day.occupied_seconds >= MIN_PRIOR_OCCUPIED_SECONDS
            and day.prior >= MIN_PRIOR
        ):
            return day.prior
        # 3. Try global prior
        global_ = self.get_global_prior()
        if global_ and global_.prior >= MIN_PRIOR:
            return global_.prior
        # 4. Fallback to MIN_PRIOR
        return MIN_PRIOR

    @property
    def global_prior(self) -> float:
        """Return the current cached global prior value, or default if not yet calculated."""
        if self._current_value is None:
            return MIN_PRIOR
        return max(self._current_value, MIN_PRIOR)

    @property
    def time_prior(self) -> float:
        """Return the time-based prior for the current time slot."""
        try:
            day_of_week, time_slot = get_current_time_slot()
            cache_key = (day_of_week, time_slot)

            # Check if we have a cached time-based prior
            if (
                cache_key in self._time_prior_cache
                and self._last_updated
                and (dt_util.utcnow() - self._last_updated) < self.cache_ttl
            ):
                entry = self._time_prior_cache[cache_key]
                if entry.prior >= MIN_PRIOR:
                    return entry.prior
        except Exception:  # noqa: BLE001
            pass  # Fall through to global prior

        # Fallback to global prior
        return self.global_prior

    @property
    def last_updated(self) -> datetime | None:
        """Return the last updated timestamp."""
        return self._last_updated

    @property
    def prior_intervals(self) -> list[StateInterval] | None:
        """Return the prior intervals."""
        return self._prior_intervals

    def get_time_slot_prior(self) -> PriorCacheEntry | None:
        """Return the time slot prior, its occupied seconds, and total seconds for the current slot, or None if not cached/valid."""
        day_of_week, time_slot = get_current_time_slot()
        cache_key = (day_of_week, time_slot)
        return self._get_cached_entry(self._time_prior_cache, cache_key)

    def get_day_prior(self) -> PriorCacheEntry | None:
        """Return the day prior, its occupied seconds, and total seconds for the current day of week, or None if not cached/valid."""
        day_of_week, _ = get_current_time_slot()
        return self._get_cached_entry(self._day_prior_cache, day_of_week)

    def get_global_prior(self) -> PriorCacheEntry | None:
        """Return the global prior, its occupied seconds, and total seconds, or None if not cached/valid."""
        return self._get_cached_entry(self._global_prior_cache)

    async def update(
        self, force: bool = False, history_period: int | None = None
    ) -> float:
        """Unified update: calculate all prior tiers and update caches with a single timestamp."""
        if not force and self._is_cache_valid():
            return self.value  # type: ignore[return-value]
        try:
            value = await self.calculate_all_priors(
                history_period=history_period, force=force
            )
        except Exception:  # pragma: no cover
            _LOGGER.exception("Prior calculation failed, using default %.2f", MIN_PRIOR)
            value = MIN_PRIOR
        return value

    async def calculate_all_priors(
        self, history_period: int | None = None, force: bool = False
    ) -> float:
        """Calculate and update all prior tiers (global, day, time slot) and caches, with a single last_updated timestamp."""
        # Use provided history_period or fall back to coordinator default
        days_to_use = history_period if history_period is not None else self.days
        start_time = dt_util.utcnow() - timedelta(days=days_to_use)
        end_time = dt_util.utcnow()
        total_seconds = int((end_time - start_time).total_seconds())

        # --- Global prior (max of all sensors and occupancy entity) ---
        (
            all_sensors_prior,
            all_sensors_data,
            all_sensors_intervals,
        ) = await self._calculate_prior_for_entities(
            self.sensor_ids, start_time, end_time, total_seconds
        )
        occupancy_entity_id = self.coordinator.occupancy_entity_id
        occupancy_entity_data = {}
        if occupancy_entity_id:
            try:
                (
                    occupancy_entity_prior,
                    occupancy_entity_data,
                    occupancy_entity_intervals,
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

        if occupancy_entity_prior > all_sensors_prior:
            final_prior = occupancy_entity_prior
            self._prior_intervals = occupancy_entity_intervals
            self._prior_source = "occupancy_entity_id"
            self._prior_source_entity_ids = (
                [occupancy_entity_id] if occupancy_entity_id else []
            )
            global_occupied = sum(
                d["occupied_seconds"] for d in occupancy_entity_data.values()
            )
            global_total = sum(
                (d["end_time"] - d["start_time"]).total_seconds()
                for d in occupancy_entity_data.values()
            )
        else:
            final_prior = all_sensors_prior
            self._prior_intervals = all_sensors_intervals
            self._prior_source = "input_sensors"
            self._prior_source_entity_ids = self.sensor_ids
            global_occupied = sum(
                d["occupied_seconds"] for d in all_sensors_data.values()
            )
            global_total = sum(
                (d["end_time"] - d["start_time"]).total_seconds()
                for d in all_sensors_data.values()
            )
        self._current_value = final_prior
        self._sensor_hash = hash(frozenset(self._prior_source_entity_ids))
        self._all_sensors_prior = all_sensors_prior
        self._occupancy_entity_prior = occupancy_entity_prior
        self._global_prior_cache = PriorCacheEntry(
            prior=final_prior,
            occupied_seconds=int(global_occupied),
            total_seconds=int(global_total),
        )

        # --- Day priors ---
        entity_ids = self.sensor_ids.copy()
        if self.coordinator.occupancy_entity_id:
            entity_ids.append(self.coordinator.occupancy_entity_id)
        day_priors: dict[int, PriorCacheEntry] = {}
        for day_of_week in range(7):
            total_occupied_seconds = 0
            total_analyzed_seconds = 0
            for entity_id in entity_ids:
                intervals = (
                    await self.coordinator.sqlite_store.get_historical_intervals(
                        entity_id,
                        start_time,
                        end_time,
                    )
                )
                if not intervals:
                    continue
                for interval in intervals:
                    interval_start = interval["start"]
                    interval_end = interval["end"]
                    if interval_start.weekday() == day_of_week:
                        day_start = interval_start.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                        day_end = day_start + timedelta(days=1)
                        overlap_start = max(interval_start, day_start)
                        overlap_end = min(interval_end, day_end)
                        if overlap_start < overlap_end:
                            overlap_seconds = (
                                overlap_end - overlap_start
                            ).total_seconds()
                            total_occupied_seconds += overlap_seconds
                total_analyzed_seconds += 86400 * (days_to_use // 7)
            global_prior = self.global_prior
            max_prior = min(global_prior * GLOBAL_PRIOR_FACTOR, MAX_PRIOR)
            if total_analyzed_seconds > 0:
                smoothed_prior = (
                    total_occupied_seconds
                    + SMOOTHING_FACTOR * global_prior * total_analyzed_seconds
                ) / (total_analyzed_seconds + SMOOTHING_FACTOR * total_analyzed_seconds)
                smoothed_prior = smoothed_prior * PRIOR_BUFFER_FACTOR
                prior = max(MIN_PRIOR, min(smoothed_prior, max_prior))
            else:
                prior = global_prior
            day_priors[day_of_week] = PriorCacheEntry(
                prior=prior,
                occupied_seconds=int(total_occupied_seconds),
                total_seconds=int(total_analyzed_seconds),
            )
        self._day_prior_cache = day_priors

        # --- Time slot priors ---
        time_priors: dict[tuple[int, int], PriorCacheEntry] = {}
        for day_of_week in range(7):
            for time_slot in range(48):
                (
                    prior_value,
                    occupied,
                    total,
                ) = await self._calculate_prior_for_time_slot_full(
                    entity_ids,
                    day_of_week,
                    time_slot,
                    start_time,
                    end_time,
                    days_to_use,
                )
                time_priors[(day_of_week, time_slot)] = PriorCacheEntry(
                    prior=prior_value,
                    occupied_seconds=occupied,
                    total_seconds=total,
                )
        self._time_prior_cache = time_priors

        # Set a single last_updated timestamp for all caches
        self._last_updated = dt_util.utcnow()
        # self._time_prior_last_updated = self._last_updated # REMOVE
        # self._day_prior_cache_last_updated = self._last_updated # REMOVE

        return final_prior

    async def _calculate_prior_for_time_slot_full(
        self,
        entity_ids: list[str],
        day_of_week: int,
        time_slot: int,
        start_time: datetime,
        end_time: datetime,
        days_to_use: int,
    ) -> tuple[float, int, int]:
        """Calculate prior, occupied seconds, and total seconds for a specific time slot."""
        slot_start_time, slot_end_time = time_slot_to_datetime_range(
            day_of_week, time_slot, start_time
        )
        slot_start_time = slot_start_time.time()
        slot_end_time = slot_end_time.time()
        slot_occurrences = []
        current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        while current <= end_time:
            if current.weekday() == day_of_week:
                slot_start_dt = current.replace(
                    hour=slot_start_time.hour,
                    minute=slot_start_time.minute,
                    second=slot_start_time.second,
                    microsecond=0,
                )
                slot_end_dt = current.replace(
                    hour=slot_end_time.hour,
                    minute=slot_end_time.minute,
                    second=slot_end_time.second,
                    microsecond=0,
                )
                if slot_end_dt <= slot_start_dt:
                    slot_end_dt += timedelta(days=1)
                if slot_end_dt >= start_time and slot_start_dt <= end_time:
                    slot_occurrences.append(
                        (
                            max(slot_start_dt, start_time),
                            min(slot_end_dt, end_time),
                        )
                    )
            current += timedelta(days=1)
        total_occupied_seconds = 0
        total_analyzed_seconds = 0
        for entity_id in entity_ids:
            intervals = await self.coordinator.sqlite_store.get_historical_intervals(
                entity_id,
                start_time,
                end_time,
            )
            if not intervals:
                continue
            for slot_start_dt, slot_end_dt in slot_occurrences:
                slot_duration = (slot_end_dt - slot_start_dt).total_seconds()
                total_analyzed_seconds += slot_duration
                for interval in intervals:
                    interval_start = interval["start"]
                    interval_end = interval["end"]
                    overlap_start = max(slot_start_dt, interval_start)
                    overlap_end = min(slot_end_dt, interval_end)
                    if overlap_start < overlap_end:
                        overlap_seconds = (overlap_end - overlap_start).total_seconds()
                        total_occupied_seconds += overlap_seconds
        global_prior = self.global_prior
        max_prior = min(global_prior * GLOBAL_PRIOR_FACTOR, MAX_PRIOR)
        if total_analyzed_seconds > 0:
            smoothed_prior = (
                total_occupied_seconds
                + SMOOTHING_FACTOR * global_prior * total_analyzed_seconds
            ) / (total_analyzed_seconds + SMOOTHING_FACTOR * total_analyzed_seconds)
            smoothed_prior = smoothed_prior * PRIOR_BUFFER_FACTOR
            prior = max(MIN_PRIOR, min(smoothed_prior, max_prior))
        else:
            prior = global_prior
        return prior, int(total_occupied_seconds), int(total_analyzed_seconds)

    async def _calculate_prior_for_entities(
        self,
        entity_ids: list[str],
        start_time: datetime,
        end_time: datetime,
        total_seconds: int,
    ) -> tuple[float, dict[str, dict[str, Any]], list[StateInterval]]:
        """Calculate the prior for a given list of entity_ids.

        Returns:
            Tuple[prior, data]: prior value and the data dictionary

        """
        data = {}
        if not entity_ids:
            return MIN_PRIOR, {}, []
        for entity_id in entity_ids:
            # Get intervals using only our DB
            intervals = await self.coordinator.sqlite_store.get_historical_intervals(
                entity_id,
                start_time,
                end_time,
            )

            if intervals:
                occupied_seconds = int(
                    sum(
                        (interval["end"] - interval["start"]).total_seconds()
                        for interval in intervals
                    )
                )

                data[entity_id] = {
                    "entity_id": entity_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "states_count": len(intervals),
                    "intervals": intervals,
                    "occupied_seconds": occupied_seconds,
                    "ratio": occupied_seconds / total_seconds,
                }

        if data:
            prior = (sum(d["ratio"] for d in data.values()) / len(data)) * 1.05
            prior = max(prior, MIN_PRIOR)
        else:
            prior = MIN_PRIOR

        return prior, data, intervals

    # ------------------------------------------------------------------ #
    def _is_cache_valid(self) -> bool:
        if self._current_value is None or self._last_updated is None:
            return False
        if (dt_util.utcnow() - self._last_updated) > self.cache_ttl:
            return False
        # in case sensors were added/removed
        return self._sensor_hash == hash(frozenset(self.sensor_ids))

    def _get_cached_entry(self, cache, key=None):
        entry = cache.get(key) if key is not None else (cache if cache else None)
        if (
            entry
            and self._last_updated
            and (dt_util.utcnow() - self._last_updated) < self.cache_ttl
        ):
            return entry
        return None

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
