"""Analysis classes for Area Occupancy Detection."""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime, timedelta
import logging
import time
from typing import TYPE_CHECKING

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import DEFAULT_LOOKBACK_DAYS, TIME_PRIOR_MAX_BOUND, TIME_PRIOR_MIN_BOUND
from ..db.queries import is_occupied_intervals_cache_valid
from ..time_utils import ensure_timezone_aware, ensure_utc_datetime, to_local, to_utc
from ..utils import format_area_names
from .entity_type import InputType
from .prior import Prior

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


async def run_full_analysis(
    coordinator: AreaOccupancyCoordinator, _now: datetime | None = None
) -> None:
    """Run the full analysis chain for all areas.

    This function orchestrates the complete analysis process:
    1. Sync states from recorder
    2. Database health check and pruning
    3. Populate occupied intervals cache
    4. Run interval aggregation
    5. Run numeric aggregation
    6. Recalculate priors for all areas
    7. Run correlation analysis
    8. Save data (preserve decay state before refresh)
    9. Refresh coordinator
    10. Save data (persist all changes)

    Args:
        coordinator: The coordinator instance containing areas and database
        _now: Optional timestamp for the analysis run (used by timer)
    """
    from ..db.correlation import run_correlation_analysis  # noqa: PLC0415

    if _now is None:
        _now = dt_util.utcnow()

    analysis_start_time = time.perf_counter()
    failed_steps: list[str] = []
    total_steps = 10

    async def _run_step(step_num: int, step_name: str, coro: Awaitable[None]) -> None:
        """Run a single analysis step with timing and error tracking."""
        start = time.perf_counter()
        try:
            await coro
            elapsed_ms = (time.perf_counter() - start) * 1000
            _LOGGER.info(
                "Step %d: %s completed in %.2f ms", step_num, step_name, elapsed_ms
            )
        except (HomeAssistantError, OSError, RuntimeError):
            elapsed_ms = (time.perf_counter() - start) * 1000
            failed_steps.append(step_name)
            _LOGGER.exception(
                "Step %d: %s FAILED in %.2f ms",
                step_num,
                step_name,
                elapsed_ms,
            )

    async def _sync_states() -> None:
        await coordinator.db.sync_states()

    async def _health_check_and_prune() -> None:
        health_ok = await coordinator.hass.async_add_executor_job(
            coordinator.db.periodic_health_check
        )
        if not health_ok:
            _LOGGER.warning(
                "Database health check found issues for areas: %s",
                format_area_names(coordinator),
            )
        await coordinator.hass.async_add_executor_job(
            coordinator.db.prune_old_intervals
        )

    async def _recalculate_priors() -> None:
        for area in coordinator.areas.values():
            await area.run_prior_analysis()

    async def _run_correlations() -> None:
        await run_correlation_analysis(coordinator)
        await coordinator.async_refresh_correlations()

    async def _save_data() -> None:
        await coordinator.hass.async_add_executor_job(coordinator.db.save_data)

    async def _refresh() -> None:
        await coordinator.async_refresh()

    try:
        await _run_step(1, "sync_states", _sync_states())
        await _run_step(2, "health_check_and_prune", _health_check_and_prune())
        await _run_step(
            3,
            "populate_occupied_intervals_cache",
            ensure_occupied_intervals_cache(coordinator),
        )
        await _run_step(
            4, "interval_aggregation", run_interval_aggregation(coordinator, _now)
        )
        await _run_step(
            5, "numeric_aggregation", run_numeric_aggregation(coordinator, _now)
        )
        await _run_step(6, "recalculate_priors", _recalculate_priors())
        await _run_step(7, "correlation_analysis", _run_correlations())
        await _run_step(8, "save_data_before_refresh", _save_data())
        await _run_step(9, "refresh_coordinator", _refresh())
        await _run_step(10, "save_data_after_refresh", _save_data())

    except Exception as err:
        _LOGGER.error("Fatal error during analysis pipeline: %s", err)
        failed_steps.append("FATAL")
        raise

    finally:
        succeeded = total_steps - len(failed_steps)
        final_elapsed_ms = (time.perf_counter() - analysis_start_time) * 1000
        if failed_steps:
            _LOGGER.warning(
                "Analysis completed: %d/%d steps succeeded (FAILED: %s) in %.2f ms",
                succeeded,
                total_steps,
                ", ".join(failed_steps),
                final_elapsed_ms,
            )
        else:
            _LOGGER.info(
                "Full analysis completed: %d/%d steps succeeded in %.2f ms",
                total_steps,
                total_steps,
                final_elapsed_ms,
            )


async def start_prior_analysis(
    coordinator: AreaOccupancyCoordinator,
    area_name: str,
    prior: Prior,
    analysis_period_days: int = DEFAULT_LOOKBACK_DAYS,
) -> None:
    """Start prior analysis for an area (wrapper for PriorAnalyzer)."""
    try:
        analyzer = PriorAnalyzer(coordinator, area_name)
        await coordinator.hass.async_add_executor_job(
            analyzer.calculate_and_update_prior, analysis_period_days
        )
    except (ValueError, TypeError, RuntimeError) as e:
        _LOGGER.error("Error during prior analysis for area %s: %s", area_name, e)


class PriorAnalyzer:
    """Analyzes historical data to calculate prior probabilities."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, area_name: str) -> None:
        """Initialize the analyzer."""
        self.coordinator = coordinator
        self.area_name = area_name
        self.hass = coordinator.hass
        self.db = coordinator.db
        if area_name not in coordinator.areas:
            raise ValueError(f"Area '{area_name}' not found")
        self.area = coordinator.areas[area_name]
        self.config = self.area.config

    def get_occupied_intervals(
        self,
        days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> list[tuple[datetime, datetime]]:
        """Get intervals where the area was occupied based on motion sensors only.

        Occupied intervals are determined exclusively by motion sensors to ensure
        consistent ground truth for prior calculations.
        """
        # Calculate time range
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=days)

        # Get occupied intervals from database (motion sensors only)
        # The query automatically includes all motion sensors for the area
        return self.db.get_occupied_intervals(
            area_name=self.area_name,
            start_time=start_time,
        )

    def calculate_and_update_prior(self, days: int = DEFAULT_LOOKBACK_DAYS) -> None:
        """Calculate and update the prior probability for the area."""
        _LOGGER.debug(
            "Starting prior analysis for area %s (lookback: %d days)",
            self.area_name,
            days,
        )

        try:
            # 1. Get occupied intervals based on motion sensors (ground truth)
            # Note: get_occupied_intervals already performs merging and timeout extension
            # The intervals returned are the final merged intervals
            occupied_intervals = self.get_occupied_intervals(days)

            if not occupied_intervals:
                _LOGGER.debug(
                    "No occupancy data found for prior calculation in area %s",
                    self.area_name,
                )
                return

            # Log interval statistics for debugging
            # Note: The intervals from get_occupied_intervals are already merged,
            # so we can't see the raw count here. The merge happens in queries.get_occupied_intervals.
            # We log what we have: the final merged interval count and duration.
            occupied_duration_before_calc = sum(
                (
                    ensure_timezone_aware(end) - ensure_timezone_aware(start)
                ).total_seconds()
                for start, end in occupied_intervals
            )
            _LOGGER.debug(
                "Prior calculation for area %s: %d merged intervals, %.1f hours total duration",
                self.area_name,
                len(occupied_intervals),
                occupied_duration_before_calc / 3600,
            )

            # 2. Calculate global prior using actual data period
            # Determine actual data period from intervals (not fixed lookback)
            # Ensure all datetime objects are timezone-aware UTC
            # Use ensure_timezone_aware for consistency (intervals are already timezone-aware)

            # Diagnostic logging: log interval details before calculation
            _LOGGER.debug(
                "Prior calculation for area %s: %d intervals",
                self.area_name,
                len(occupied_intervals),
            )
            if occupied_intervals:
                # Log first few intervals for debugging
                for i, (start, end) in enumerate(occupied_intervals[:5]):
                    start_aware = ensure_timezone_aware(start)
                    end_aware = ensure_timezone_aware(end)
                    _LOGGER.debug(
                        "Interval %d: start=%s (tz=%s), end=%s (tz=%s), duration=%.2f",
                        i,
                        start_aware,
                        start_aware.tzinfo,
                        end_aware,
                        end_aware.tzinfo,
                        (end_aware - start_aware).total_seconds(),
                    )

            # Validate interval data: check all intervals have start <= end
            invalid_intervals = []
            for i, (start, end) in enumerate(occupied_intervals):
                start_aware = ensure_utc_datetime(start)
                end_aware = ensure_utc_datetime(end)
                if start_aware > end_aware:
                    invalid_intervals.append((i, start_aware, end_aware))

            if invalid_intervals:
                _LOGGER.error(
                    "Invalid interval data for area %s: %d intervals have start > end",
                    self.area_name,
                    len(invalid_intervals),
                )
                for i, start, end in invalid_intervals[:5]:  # Log first 5
                    _LOGGER.error(
                        "  Interval %d: start=%s > end=%s (difference: %.2f seconds)",
                        i,
                        start,
                        end,
                        (start - end).total_seconds(),
                    )
                # Filter out invalid intervals
                occupied_intervals = [
                    (start, end)
                    for start, end in occupied_intervals
                    if ensure_utc_datetime(start) <= ensure_utc_datetime(end)
                ]
                if not occupied_intervals:
                    _LOGGER.error(
                        "All intervals invalid for area %s, using fallback prior",
                        self.area_name,
                    )
                    self.area.prior.set_global_prior(0.01)
                    return

            # Convert all intervals to UTC and find bounds
            first_interval_start = min(
                ensure_utc_datetime(start) for start, end in occupied_intervals
            )
            last_interval_end = max(
                ensure_utc_datetime(end) for start, end in occupied_intervals
            )
            now = ensure_utc_datetime(dt_util.utcnow())

            # Validate that intervals are chronologically valid
            if first_interval_start > last_interval_end:
                _LOGGER.error(
                    "Invalid interval data for area %s: first_interval_start (%s) > last_interval_end (%s). "
                    "This may indicate timezone issues or corrupted data.",
                    self.area_name,
                    first_interval_start,
                    last_interval_end,
                )
                # Use fallback prior
                self.area.prior.set_global_prior(0.01)
                _LOGGER.debug(
                    "Prior analysis completed for area %s: global_prior=0.010 (fallback due to invalid interval bounds)",
                    self.area_name,
                )
                return

            # Diagnostic logging: log key timestamps
            _LOGGER.debug(
                "Period calculation for area %s: first_interval_start=%s (tz=%s), "
                "last_interval_end=%s (tz=%s), now=%s (tz=%s)",
                self.area_name,
                first_interval_start,
                first_interval_start.tzinfo,
                last_interval_end,
                last_interval_end.tzinfo,
                now,
                now.tzinfo,
            )

            # Use actual period: from first interval to now (or last interval if very recent)
            # If last interval is more than 1 hour old, use it; otherwise use now
            # Ensure we use UTC-aware datetimes for all calculations
            if (now - last_interval_end).total_seconds() > 3600:
                actual_period_end = last_interval_end
            else:
                actual_period_end = now

            # Defensive check: ensure actual_period_end >= first_interval_start
            if actual_period_end < first_interval_start:
                _LOGGER.warning(
                    "actual_period_end (%s) < first_interval_start (%s) for area %s. "
                    "This may indicate timezone issues or clock skew. Using now instead.",
                    actual_period_end,
                    first_interval_start,
                    self.area_name,
                )
                actual_period_end = now
                # Double-check after using now
                if actual_period_end < first_interval_start:
                    _LOGGER.error(
                        "Even 'now' (%s) is before first_interval_start (%s) for area %s. "
                        "This indicates severe clock skew or timezone issues. Using fallback prior.",
                        actual_period_end,
                        first_interval_start,
                        self.area_name,
                    )
                    self.area.prior.set_global_prior(0.01)
                    _LOGGER.debug(
                        "Prior analysis completed for area %s: global_prior=0.010 (fallback due to clock skew)",
                        self.area_name,
                    )
                    return

            actual_period_duration = (
                actual_period_end - first_interval_start
            ).total_seconds()

            # Guard against zero or negative duration (bad timestamps or clock skew)
            if actual_period_duration <= 0:
                _LOGGER.warning(
                    "Invalid period duration (%.2f seconds) for area %s. "
                    "first_interval_start=%s, actual_period_end=%s. "
                    "This may indicate bad timestamps or clock skew. "
                    "Using safe fallback prior of 0.01.",
                    actual_period_duration,
                    self.area_name,
                    first_interval_start,
                    actual_period_end,
                )
                # Set safe fallback prior and return early
                self.area.prior.set_global_prior(0.01)
                _LOGGER.debug(
                    "Prior analysis completed for area %s: global_prior=0.010 (fallback due to invalid period)",
                    self.area_name,
                )
                return

            # Calculate occupied duration (ensure UTC-aware)
            # Use ensure_utc_datetime to ensure all datetimes are in UTC
            occupied_duration = sum(
                (ensure_utc_datetime(end) - ensure_utc_datetime(start)).total_seconds()
                for start, end in occupied_intervals
            )

            # Use actual period for prior calculation
            # Ensure valid probability (0.01 to 0.99)
            global_prior = max(
                0.01, min(0.99, occupied_duration / actual_period_duration)
            )

            # 3. Update the Prior object
            self.area.prior.set_global_prior(global_prior)

            _LOGGER.debug(
                "Prior analysis completed for area %s: global_prior=%.3f (occupied: %.1f hours over %.1f days, %d intervals)",
                self.area_name,
                global_prior,
                occupied_duration / 3600,
                actual_period_duration / 86400,
                len(occupied_intervals),
            )

            # 4. Save global prior to database
            try:
                success = self.db.save_global_prior(
                    area_name=self.area_name,
                    prior_value=global_prior,
                    data_period_start=first_interval_start,
                    data_period_end=actual_period_end,
                    total_occupied_seconds=occupied_duration,
                    total_period_seconds=actual_period_duration,
                    interval_count=len(occupied_intervals),
                    calculation_method="interval_analysis",
                )
                if success:
                    _LOGGER.info(
                        "Global prior saved for area %s: %.3f (period: %.1f days, %d intervals)",
                        self.area_name,
                        global_prior,
                        actual_period_duration / 86400,
                        len(occupied_intervals),
                    )
                else:
                    _LOGGER.warning(
                        "Failed to save global prior for area %s", self.area_name
                    )
            except Exception as e:  # noqa: BLE001
                _LOGGER.warning(
                    "Failed to save global prior for area %s: %s", self.area_name, e
                )
                # Don't fail the entire prior calculation if save fails

            # 5. Calculate and save time priors
            try:
                time_priors, data_points_per_slot = self.calculate_time_priors(
                    occupied_intervals,
                    first_interval_start,
                    actual_period_end,
                )
                if time_priors:
                    success = self.db.save_time_priors(
                        area_name=self.area_name,
                        time_priors=time_priors,
                        data_period_start=first_interval_start,
                        data_period_end=actual_period_end,
                        data_points_per_slot=data_points_per_slot,
                    )
                    if success:
                        _LOGGER.info(
                            "Time priors saved for area %s: %d slots populated",
                            self.area_name,
                            len(time_priors),
                        )
                        # Cache will be automatically reloaded on next time_prior access
                        # since it was invalidated when global_prior was set above
                    else:
                        _LOGGER.warning(
                            "Failed to save time priors for area %s", self.area_name
                        )
                else:
                    _LOGGER.debug(
                        "No time priors calculated for area %s (insufficient data)",
                        self.area_name,
                    )
            except Exception as e:  # noqa: BLE001
                _LOGGER.warning(
                    "Failed to calculate time priors for area %s: %s", self.area_name, e
                )
                # Don't fail the entire prior calculation if time priors fail

        except (ValueError, TypeError, RuntimeError) as e:
            _LOGGER.error(
                "Failed to calculate prior for area %s: %s", self.area_name, e
            )

    def _get_entity_ids_by_type(self, input_type: InputType) -> list[str]:
        """Get entity IDs for a specific input type in this area."""
        entities = self.area.entities.get_entities_by_input_type(input_type)
        return list(entities.keys())

    def calculate_time_priors(
        self,
        occupied_intervals: list[tuple[datetime, datetime]],
        period_start: datetime,
        period_end: datetime,
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
        """Calculate time priors for all 168 time slots (7 days × 24 hours).

        Args:
            occupied_intervals: List of (start_time, end_time) tuples representing
                occupied periods (already merged and extended)
            period_start: Start of the data period
            period_end: End of the data period

        Returns:
            Tuple of:
            - Dictionary mapping (day_of_week, time_slot) to prior_value
            - Dictionary mapping (day_of_week, time_slot) to data_points (weeks with data)
        """
        _LOGGER.debug(
            "Calculating time priors for area %s (%d intervals, period: %s to %s)",
            self.area_name,
            len(occupied_intervals),
            period_start,
            period_end,
        )

        # Policy: bucket by Home Assistant local wall-clock time.
        # We do overlap arithmetic in UTC, but derive slot keys from the corresponding local time.
        slot_occupied_seconds: dict[tuple[int, int], float] = {}

        period_start_utc = to_utc(period_start)
        period_end_utc = to_utc(period_end)

        # Track total possible seconds per slot over the period to handle DST correctly.
        # Keyed by (day_of_week, hour) in local time.
        slot_total_seconds: dict[tuple[int, int], float] = {}
        slot_weeks_total: dict[tuple[int, int], set[tuple[int, int]]] = {}

        # Build denominators by walking local hour slots across the analysis period.
        # Iterate in UTC to avoid ambiguity during DST fall-back (repeated local hours).
        current_utc = period_start_utc
        while current_utc < period_end_utc:
            current_local = to_local(current_utc)
            fold = getattr(current_local, "fold", 0)
            slot_start_local = current_local.replace(
                minute=0, second=0, microsecond=0, fold=fold
            )
            slot_end_local = slot_start_local + timedelta(hours=1)

            slot_start_utc = to_utc(slot_start_local)
            slot_end_utc = to_utc(slot_end_local)
            if slot_end_utc <= slot_start_utc:
                break

            slot_key = (slot_start_local.weekday(), slot_start_local.hour)
            overlap_start = max(period_start_utc, slot_start_utc)
            overlap_end = min(period_end_utc, slot_end_utc)
            slot_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
            if slot_seconds > 0:
                slot_total_seconds[slot_key] = (
                    slot_total_seconds.get(slot_key, 0.0) + slot_seconds
                )
                year, week_number, _ = slot_start_local.isocalendar()
                slot_weeks_total.setdefault(slot_key, set()).add((year, week_number))

            current_utc = slot_end_utc

        # Process each occupied interval
        for start_time, end_time in occupied_intervals:
            start_utc = to_utc(start_time)
            end_utc = to_utc(end_time)

            # Clamp to analysis period bounds
            start_utc = max(start_utc, period_start_utc)
            end_utc = min(end_utc, period_end_utc)
            if start_utc >= end_utc:
                continue

            current_utc = start_utc
            while current_utc < end_utc:
                current_local = to_local(current_utc)
                fold = getattr(current_local, "fold", 0)
                slot_start_local = current_local.replace(
                    minute=0, second=0, microsecond=0, fold=fold
                )
                slot_end_local = slot_start_local + timedelta(hours=1)

                slot_start_utc = to_utc(slot_start_local)
                slot_end_utc = to_utc(slot_end_local)
                if slot_end_utc <= slot_start_utc:
                    break

                slot_key = (slot_start_local.weekday(), slot_start_local.hour)
                overlap_start = max(start_utc, current_utc, slot_start_utc)
                overlap_end = min(end_utc, slot_end_utc)
                overlap_seconds = max(
                    0.0, (overlap_end - overlap_start).total_seconds()
                )
                if overlap_seconds > 0:
                    slot_occupied_seconds[slot_key] = (
                        slot_occupied_seconds.get(slot_key, 0.0) + overlap_seconds
                    )

                current_utc = slot_end_utc

        # Calculate prior values for each slot
        time_priors: dict[tuple[int, int], float] = {}
        data_points: dict[tuple[int, int], int] = {}

        for slot_key, occupied_seconds in slot_occupied_seconds.items():
            total_slot_seconds = slot_total_seconds.get(slot_key, 0.0)
            if total_slot_seconds <= 0:
                continue

            prior_value = occupied_seconds / total_slot_seconds
            prior_value = max(
                TIME_PRIOR_MIN_BOUND, min(TIME_PRIOR_MAX_BOUND, prior_value)
            )
            time_priors[slot_key] = prior_value
            data_points[slot_key] = len(slot_weeks_total.get(slot_key, set()))

        _LOGGER.debug(
            "Time priors calculated for area %s: %d slots populated out of 168 total",
            self.area_name,
            len(time_priors),
        )

        return time_priors, data_points


async def ensure_occupied_intervals_cache(
    coordinator: AreaOccupancyCoordinator,
) -> None:
    """Ensure OccupiedIntervalsCache is populated for all areas.

    This function checks cache validity and populates it from raw intervals
    if needed. This ensures the cache exists before interval aggregation
    deletes raw intervals older than the retention period.

    Args:
        coordinator: The coordinator instance containing areas and database
    """
    for area_name in coordinator.areas:
        # Check if cache is valid
        cache_valid = await coordinator.hass.async_add_executor_job(
            is_occupied_intervals_cache_valid, coordinator.db, area_name
        )

        if not cache_valid:
            _LOGGER.debug(
                "OccupiedIntervalsCache invalid or missing for %s, populating from raw intervals",
                area_name,
            )
            # Calculate occupied intervals from raw intervals (motion sensors only)
            analyzer = PriorAnalyzer(coordinator, area_name)
            intervals = await coordinator.hass.async_add_executor_job(
                analyzer.get_occupied_intervals,
                DEFAULT_LOOKBACK_DAYS,
            )

            # Defensive check: warn if no intervals found (may indicate missing entities)
            if not intervals:
                _LOGGER.warning(
                    "No occupied intervals found for area %s when populating cache. "
                    "This may indicate: (1) entities not saved to database yet, "
                    "(2) no motion sensor intervals synced, or (3) no motion sensors configured.",
                    area_name,
                )

            # Save to cache
            if intervals:
                success = await coordinator.hass.async_add_executor_job(
                    coordinator.db.save_occupied_intervals_cache,
                    area_name,
                    intervals,
                    "motion_sensors",
                )
                if success:
                    _LOGGER.debug(
                        "Populated OccupiedIntervalsCache for %s with %d intervals",
                        area_name,
                        len(intervals),
                    )
                else:
                    _LOGGER.warning(
                        "Failed to save OccupiedIntervalsCache for %s", area_name
                    )


async def run_interval_aggregation(
    coordinator: AreaOccupancyCoordinator,
    _now: datetime | None = None,
    return_results: bool = False,
) -> dict[str, int] | None:
    """Run interval aggregation.

    This function aggregates raw intervals older than the retention period
    into daily/weekly/monthly aggregates.

    Args:
        coordinator: The coordinator instance containing areas and database
        _now: Optional timestamp for the aggregation run
        return_results: If True, returns aggregation results dictionary

    Returns:
        Dictionary with aggregation results (daily, weekly, monthly counts) if
        return_results is True, None otherwise.
    """
    if _now is None:
        _now = dt_util.utcnow()

    try:
        results = await coordinator.hass.async_add_executor_job(
            coordinator.db.run_interval_aggregation
        )
        area_names = format_area_names(coordinator)
        _LOGGER.debug(
            "Interval aggregation completed for areas %s: %s",
            area_names,
            results,
        )
    except Exception as err:  # noqa: BLE001
        area_names = format_area_names(coordinator)
        _LOGGER.error(
            "Interval aggregation failed for areas %s: %s",
            area_names,
            err,
        )
        # Don't raise - allow analysis to continue even if aggregation fails
        return None
    else:
        return results if return_results else None


async def run_numeric_aggregation(
    coordinator: AreaOccupancyCoordinator,
    _now: datetime | None = None,
    return_results: bool = False,
) -> dict[str, int] | None:
    """Run numeric aggregation.

    This function aggregates raw numeric samples older than the retention period
    into hourly/weekly aggregates for seasonal trend analysis.

    Args:
        coordinator: The coordinator instance containing areas and database
        _now: Optional timestamp for the aggregation run
        return_results: If True, returns aggregation results dictionary

    Returns:
        Dictionary with aggregation results (hourly, weekly counts) if
        return_results is True, None otherwise.
    """
    if _now is None:
        _now = dt_util.utcnow()

    try:
        results = await coordinator.hass.async_add_executor_job(
            coordinator.db.run_numeric_aggregation
        )
        area_names = format_area_names(coordinator)
        _LOGGER.debug(
            "Numeric aggregation completed for areas %s: %s",
            area_names,
            results,
        )
    except Exception as err:  # noqa: BLE001
        area_names = format_area_names(coordinator)
        _LOGGER.error(
            "Numeric aggregation failed for areas %s: %s",
            area_names,
            err,
        )
        # Don't raise - allow analysis to continue even if aggregation fails
        return None
    else:
        return results if return_results else None
