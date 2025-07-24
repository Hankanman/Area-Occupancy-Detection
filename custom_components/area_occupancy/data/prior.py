"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import HA_RECORDER_DAYS, MIN_PRIOR
from ..schema import AreaTimePriorRecord
from ..state_intervals import StateInterval
from ..utils import (
    get_current_time_slot,
    get_time_slot_name,
    time_slot_to_datetime_range,
)

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

MIN_PRIOR_OCCUPIED_SECONDS = (
    120  # Minimum occupied seconds required for a prior to be considered valid
)


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
        self._last_updated: datetime | None = None
        self._sensor_hash: int | None = None
        self._sensor_data: dict[str, dict[str, Any]] = {}

        # Time-based prior cache
        self._time_prior_cache: dict[tuple[int, int], float] = {}
        self._time_prior_cache_ttl = timedelta(minutes=30)
        self._time_prior_last_updated: datetime | None = None

    # --------------------------------------------------------------------- #
    @property
    def value(self) -> float:
        """Return the best available prior: time-based, day, global, or minimum."""
        # 1. Try time slot prior
        slot_prior, slot_occupied = self.get_time_slot_prior_with_occupied()
        if slot_occupied >= MIN_PRIOR_OCCUPIED_SECONDS and slot_prior >= MIN_PRIOR:
            return slot_prior
        # 2. Try day prior
        day_prior, day_occupied = self.get_day_prior_with_occupied()
        if day_occupied >= MIN_PRIOR_OCCUPIED_SECONDS and day_prior >= MIN_PRIOR:
            return day_prior
        # 3. Try global prior
        if self.global_prior >= MIN_PRIOR:
            return self.global_prior
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
                and self._time_prior_last_updated
                and (dt_util.utcnow() - self._time_prior_last_updated)
                < self._time_prior_cache_ttl
            ):
                time_prior = self._time_prior_cache[cache_key]
                if time_prior >= MIN_PRIOR:
                    return time_prior
        except Exception:  # noqa: BLE001
            pass  # Fall through to global prior

        # Fallback to global prior
        return self.global_prior

    def get_time_slot_prior_with_occupied(self) -> tuple[float, int]:
        """Return the time slot prior and its occupied seconds for the current slot."""
        day_of_week, time_slot = get_current_time_slot()
        cache_key = (day_of_week, time_slot)
        if (
            cache_key in self._time_prior_cache
            and self._time_prior_last_updated
            and (dt_util.utcnow() - self._time_prior_last_updated)
            < self._time_prior_cache_ttl
        ):
            prior = self._time_prior_cache[cache_key]
            occupied = self._time_prior_cache.get((cache_key, "occupied"), 0)
            return prior, occupied
        # If not cached, fallback to async method (sync fallback)
        return self.global_prior, 0

    def get_day_prior_with_occupied(self) -> tuple[float, int]:
        """Return the day prior and its occupied seconds for the current day of week."""
        day_of_week, _ = get_current_time_slot()
        if (
            hasattr(self, "_day_prior_cache")
            and self._day_prior_cache_last_updated
            and (dt_util.utcnow() - self._day_prior_cache_last_updated)
            < self._time_prior_cache_ttl
        ):
            prior = self._day_prior_cache.get(day_of_week, MIN_PRIOR)
            occupied = self._day_prior_cache.get((day_of_week, "occupied"), 0)
            return prior, occupied
        # If not cached, fallback to async method (sync fallback)
        return self.global_prior, 0

    async def calculate_day_priors(
        self, history_period: int | None = None, force: bool = False
    ) -> dict[int, float]:
        """Calculate day-of-week priors for all days from historical data."""
        if (
            hasattr(self, "_day_prior_cache")
            and not force
            and self._day_prior_cache_last_updated
            and (dt_util.utcnow() - self._day_prior_cache_last_updated)
            < self._time_prior_cache_ttl
        ):
            return self._day_prior_cache
        days_to_use = history_period if history_period is not None else self.days
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=days_to_use)
        entity_ids = self.sensor_ids.copy()
        if self.coordinator.occupancy_entity_id:
            entity_ids.append(self.coordinator.occupancy_entity_id)
        if not entity_ids:
            return {}
        day_priors: dict[int, float] = {}
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
                # Each day is 86400 seconds
                total_analyzed_seconds += 86400 * (days_to_use // 7)
            alpha = 6.0
            global_prior = self.global_prior
            max_prior = min(global_prior * 1.2, 0.95)
            if total_analyzed_seconds > 0:
                smoothed_prior = (
                    total_occupied_seconds
                    + alpha * global_prior * total_analyzed_seconds
                ) / (total_analyzed_seconds + alpha * total_analyzed_seconds)
                smoothed_prior = smoothed_prior * 1.05
                prior = max(MIN_PRIOR, min(smoothed_prior, max_prior))
            else:
                prior = global_prior
            day_priors[day_of_week] = prior
            day_priors[(day_of_week, "occupied")] = total_occupied_seconds
        self._day_prior_cache = day_priors
        self._day_prior_cache_last_updated = dt_util.utcnow()
        return day_priors

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

    async def get_time_prior(self) -> float:
        """Get the prior for a specific time slot.

        Args:
            day_of_week: 0=Monday, 6=Sunday
            time_slot: 0-47 (30-minute intervals)

        Returns:
            Prior value for the specified time slot

        """
        day_of_week, time_slot = get_current_time_slot()
        # Check cache first
        cache_key = (day_of_week, time_slot)
        if (
            cache_key in self._time_prior_cache
            and self._time_prior_last_updated
            and (dt_util.utcnow() - self._time_prior_last_updated)
            < self._time_prior_cache_ttl
        ):
            return self._time_prior_cache[cache_key]

        # Try to get from database
        if self.coordinator.sqlite_store:
            try:
                record = await self.coordinator.sqlite_store.get_time_prior(
                    self.coordinator.entry_id, day_of_week, time_slot
                )
                if record and record.data_points > 0:
                    prior_value = record.prior_value
                    # Cache the result
                    self._time_prior_cache[cache_key] = prior_value
                    return prior_value
            except Exception:  # noqa: BLE001
                _LOGGER.debug(
                    "Failed to get time prior for %s, using fallback",
                    get_time_slot_name(day_of_week, time_slot),
                )

        # Fallback to current global prior
        return self.value

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

    async def calculate_time_based_priors(
        self, history_period: int | None = None, force: bool = False
    ) -> dict[tuple[int, int], float]:
        """Calculate time-based priors for all time slots from historical data.

        Args:
            history_period: Period in days for historical data (overrides coordinator default)
            force: If True, bypass cache validation and force recalculation

        Returns:
            Dictionary mapping (day_of_week, time_slot) to prior values

        """
        # Check cache first (unless forced)
        if (
            not force
            and self._time_prior_last_updated
            and (dt_util.utcnow() - self._time_prior_last_updated)
            < self._time_prior_cache_ttl
        ):
            _LOGGER.debug(
                "Using cached time-based priors for entry %s", self.coordinator.entry_id
            )
            return self._time_prior_cache

        # Check if we have recent data in the database first
        if not force:
            try:
                recent_priors = (
                    await self.coordinator.sqlite_store.get_recent_time_priors(
                        self.coordinator.entry_id, hours=24
                    )
                )
                if recent_priors:
                    _LOGGER.debug(
                        "Found %d recent time-based priors in database for entry %s",
                        len(recent_priors),
                        self.coordinator.entry_id,
                    )
                    # Convert to cache format and return
                    cache_data = {
                        (record.day_of_week, record.time_slot): record.prior_value
                        for record in recent_priors
                    }
                    self._time_prior_cache = cache_data
                    self._time_prior_last_updated = dt_util.utcnow()
                    return cache_data
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("Failed to get recent time priors from database: %s", err)

        _LOGGER.info(
            "Calculating time-based priors for entry %s", self.coordinator.entry_id
        )

        # Use provided history_period or fall back to coordinator default
        days_to_use = history_period if history_period is not None else self.days
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=days_to_use)

        # Get all entity IDs to analyze
        entity_ids = self.sensor_ids.copy()
        if self.coordinator.occupancy_entity_id:
            entity_ids.append(self.coordinator.occupancy_entity_id)

        if not entity_ids:
            _LOGGER.warning("No entities available for time-based prior calculation")
            return {}

        # Calculate priors in chunks to avoid blocking for too long
        time_priors: dict[tuple[int, int], float] = {}
        time_prior_records: list[AreaTimePriorRecord] = []

        # Process in smaller chunks to avoid blocking
        chunk_size = 12  # Process 12 time slots at a time (6 hours worth)
        processed_slots = 0

        for day_of_week in range(7):
            for time_slot in range(48):
                try:
                    prior_value = await self._calculate_prior_for_time_slot(
                        entity_ids,
                        day_of_week,
                        time_slot,
                        start_time,
                        end_time,
                        days_to_use,
                    )

                    time_priors[(day_of_week, time_slot)] = prior_value

                    # Create record for database storage
                    record = AreaTimePriorRecord(
                        entry_id=self.coordinator.entry_id,
                        day_of_week=day_of_week,
                        time_slot=time_slot,
                        prior_value=prior_value,
                        data_points=1,  # Will be updated with actual count
                        last_updated=dt_util.utcnow(),
                    )
                    time_prior_records.append(record)

                    processed_slots += 1

                    # Yield control every chunk_size slots to avoid blocking
                    if processed_slots % chunk_size == 0:
                        await asyncio.sleep(0)  # Yield control to event loop

                except HomeAssistantError as err:
                    _LOGGER.warning(
                        "Failed to calculate prior for %s: %s",
                        get_time_slot_name(day_of_week, time_slot),
                        err,
                    )
                    # Use default prior
                    time_priors[(day_of_week, time_slot)] = MIN_PRIOR
                    processed_slots += 1

        # Store in database in batches to avoid long database operations
        if time_prior_records and self.coordinator.sqlite_store:
            try:
                # Store in smaller batches
                batch_size = 50
                total_stored = 0
                for i in range(0, len(time_prior_records), batch_size):
                    batch = time_prior_records[i : i + batch_size]
                    stored_count = (
                        await self.coordinator.sqlite_store.save_time_priors_batch(
                            batch
                        )
                    )
                    total_stored += stored_count
                    await asyncio.sleep(0)  # Yield control between batches

                _LOGGER.info(
                    "Stored %d time-based priors for entry %s",
                    total_stored,
                    self.coordinator.entry_id,
                )
            except HomeAssistantError as err:
                _LOGGER.error(
                    "Failed to store time-based priors: %s",
                    err,
                )

        # Update cache
        self._time_prior_cache = time_priors
        self._time_prior_last_updated = dt_util.utcnow()

        _LOGGER.info(
            "Calculated %d time-based priors for entry %s",
            len(time_priors),
            self.coordinator.entry_id,
        )

        return time_priors

    async def _calculate_prior_for_time_slot(
        self,
        entity_ids: list[str],
        day_of_week: int,
        time_slot: int,
        start_time: datetime,
        end_time: datetime,
        days_to_use: int,
    ) -> float:
        """Calculate prior for a specific time slot.

        Args:
            entity_ids: List of entity IDs to analyze
            day_of_week: 0=Monday, 6=Sunday
            time_slot: 0-47 (30-minute intervals)
            start_time: Start of analysis period
            end_time: End of analysis period
            days_to_use: Number of days in analysis period

        Returns:
            Prior value for the time slot

        """

        # Get the time range for this slot (as time objects)
        slot_start_time, slot_end_time = time_slot_to_datetime_range(
            day_of_week, time_slot, start_time
        )
        slot_start_time = slot_start_time.time()
        slot_end_time = slot_end_time.time()

        # Find all slot occurrences in the analysis period
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
                # If slot_end_dt < slot_start_dt, it means the slot crosses midnight
                if slot_end_dt <= slot_start_dt:
                    slot_end_dt += timedelta(days=1)
                # Only include slots within the analysis period
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
        data_points = 0

        for entity_id in entity_ids:
            try:
                intervals = (
                    await self.coordinator.sqlite_store.get_historical_intervals(
                        entity_id,
                        start_time,
                        end_time,
                    )
                )
                if not intervals:
                    continue

                for slot_start_dt, slot_end_dt in slot_occurrences:
                    slot_duration = (slot_end_dt - slot_start_dt).total_seconds()
                    total_analyzed_seconds += slot_duration
                    for interval in intervals:
                        interval_start = interval["start"]
                        interval_end = interval["end"]
                        # Find overlap between interval and slot
                        overlap_start = max(slot_start_dt, interval_start)
                        overlap_end = min(slot_end_dt, interval_end)
                        if overlap_start < overlap_end:
                            overlap_seconds = (
                                overlap_end - overlap_start
                            ).total_seconds()
                            total_occupied_seconds += overlap_seconds
                            data_points += 1
                    await asyncio.sleep(0)  # Yield control
            except HomeAssistantError as err:
                _LOGGER.debug(
                    "Failed to analyze entity %s for time slot %s: %s",
                    entity_id,
                    get_time_slot_name(day_of_week, time_slot),
                    err,
                )

        # Calculate prior
        alpha = 6.0  # Smoothing parameter: 1.0 = one virtual week of global prior
        global_prior = self.global_prior  # Use the cached global prior
        max_prior = min(global_prior * 1.2, 0.95)

        if total_analyzed_seconds > 0:
            smoothed_prior = (
                total_occupied_seconds + alpha * global_prior * total_analyzed_seconds
            ) / (total_analyzed_seconds + alpha * total_analyzed_seconds)
            smoothed_prior = (
                smoothed_prior * 1.05
            )  # Optional: keep the 5% buffer if desired
            return max(MIN_PRIOR, min(smoothed_prior, max_prior))
        return global_prior  # Fallback to global prior if no data

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
        if not entity_ids:
            return MIN_PRIOR, {}
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
