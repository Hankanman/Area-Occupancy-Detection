"""Analysis classes for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import time
from typing import TYPE_CHECKING

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import DEFAULT_LOOKBACK_DAYS, TIME_PRIOR_MAX_BOUND, TIME_PRIOR_MIN_BOUND
from ..db.queries import is_occupied_intervals_cache_valid
from ..utils import ensure_timezone_aware, format_area_names
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

    try:
        analysis_start_time = time.perf_counter()
        # Step 1: Import recent data from recorder
        start_time = time.perf_counter()
        await coordinator.db.sync_states()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info(
            "Step 1: Sync states from recorder completed in %.2f ms", elapsed_ms
        )

        # Step 2: Prune old intervals and run health check
        start_time = time.perf_counter()
        health_ok = await coordinator.hass.async_add_executor_job(
            coordinator.db.periodic_health_check
        )
        if not health_ok:
            area_names = format_area_names(coordinator)
            _LOGGER.warning(
                "Database health check found issues for areas: %s",
                area_names,
            )

        pruned_count = await coordinator.hass.async_add_executor_job(
            coordinator.db.prune_old_intervals
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if pruned_count > 0:
            _LOGGER.info(
                "Step 2: Database health check and pruning completed in %.2f ms (pruned %d intervals)",
                elapsed_ms,
                pruned_count,
            )
        else:
            _LOGGER.info(
                "Step 2: Database health check and pruning completed in %.2f ms",
                elapsed_ms,
            )

        # Step 3: Ensure OccupiedIntervalsCache is populated before aggregation
        start_time = time.perf_counter()
        await ensure_occupied_intervals_cache(coordinator)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info(
            "Step 3: Populate occupied intervals cache completed in %.2f ms", elapsed_ms
        )

        # Step 4: Run interval aggregation (safe now that cache exists)
        start_time = time.perf_counter()
        await run_interval_aggregation(coordinator, _now)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info("Step 4: Interval aggregation completed in %.2f ms", elapsed_ms)

        # Step 5: Run numeric aggregation
        start_time = time.perf_counter()
        await run_numeric_aggregation(coordinator, _now)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info("Step 5: Numeric aggregation completed in %.2f ms", elapsed_ms)

        # Step 6: Recalculate priors with new data for all areas
        start_time = time.perf_counter()
        for area in coordinator.areas.values():
            await area.run_prior_analysis()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info(
            "Step 6: Recalculate priors for all areas completed in %.2f ms", elapsed_ms
        )

        # Step 7: Run correlation analysis (requires OccupiedIntervalsCache)
        start_time = time.perf_counter()
        await run_correlation_analysis(coordinator)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info("Step 7: Correlation analysis completed in %.2f ms", elapsed_ms)

        # Step 8: Save data (preserve decay state before refresh)
        # This ensures decay state is saved before async_refresh() potentially resets it
        start_time = time.perf_counter()
        await coordinator.hass.async_add_executor_job(coordinator.db.save_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info(
            "Step 8: Save data (before refresh) completed in %.2f ms", elapsed_ms
        )

        # Step 9: Refresh the coordinator
        start_time = time.perf_counter()
        await coordinator.async_refresh()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info("Step 9: Refresh coordinator completed in %.2f ms", elapsed_ms)

        # Step 10: Save data (persist all changes after refresh)
        start_time = time.perf_counter()
        await coordinator.hass.async_add_executor_job(coordinator.db.save_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _LOGGER.info(
            "Step 10: Save data (after refresh) completed in %.2f ms", elapsed_ms
        )

        final_elapsed_ms = (time.perf_counter() - analysis_start_time) * 1000
        _LOGGER.info("Full analysis completed in %.2f ms", final_elapsed_ms)

    except (HomeAssistantError, OSError, RuntimeError) as err:
        _LOGGER.error("Failed to run historical analysis: %s", err)
        raise


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
            first_interval_start = min(
                ensure_timezone_aware(start) for start, end in occupied_intervals
            )
            last_interval_end = max(
                ensure_timezone_aware(end) for start, end in occupied_intervals
            )
            now = dt_util.utcnow()

            # Use actual period: from first interval to now (or last interval if very recent)
            # If last interval is more than 1 hour old, use it; otherwise use now
            if (now - last_interval_end).total_seconds() > 3600:
                actual_period_end = last_interval_end
            else:
                actual_period_end = now

            actual_period_duration = (
                actual_period_end - first_interval_start
            ).total_seconds()

            # Guard against zero or negative duration (bad timestamps or clock skew)
            if actual_period_duration <= 0:
                _LOGGER.warning(
                    "Invalid period duration (%.2f seconds) for area %s. "
                    "This may indicate bad timestamps or clock skew. "
                    "Using safe fallback prior of 0.01.",
                    actual_period_duration,
                    self.area_name,
                )
                # Set safe fallback prior and return early
                self.area.prior.set_global_prior(0.01)
                _LOGGER.debug(
                    "Prior analysis completed for area %s: global_prior=0.010 (fallback due to invalid period)",
                    self.area_name,
                )
                return

            # Calculate occupied duration (ensure timezone-aware)
            # Use ensure_timezone_aware for consistency (intervals are already timezone-aware)
            occupied_duration = sum(
                (
                    ensure_timezone_aware(end) - ensure_timezone_aware(start)
                ).total_seconds()
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

        # Initialize slot tracking: (day_of_week, time_slot) -> (occupied_seconds, weeks_with_data)
        slot_occupied_seconds: dict[tuple[int, int], float] = {}
        # Track calendar weeks as (year, week_number) tuples using ISO calendar
        slot_weeks: dict[tuple[int, int], set[tuple[int, int]]] = {}

        # Process each occupied interval
        for start_time, end_time in occupied_intervals:
            start = ensure_timezone_aware(start_time)
            end = ensure_timezone_aware(end_time)

            # Get all time slots this interval overlaps with
            current_time = start
            while current_time < end:
                # Determine day of week (0=Monday, 6=Sunday)
                day_of_week = current_time.weekday()

                # Determine time slot (0-23 for hourly slots)
                # DEFAULT_SLOT_MINUTES = 60, so slot = hour
                time_slot = current_time.hour

                # Calculate slot start and end for this day
                slot_start = current_time.replace(minute=0, second=0, microsecond=0)
                slot_end = slot_start + timedelta(hours=1)

                # Calculate overlap duration
                overlap_start = max(start, slot_start)
                overlap_end = min(end, slot_end)
                overlap_seconds = max(0, (overlap_end - overlap_start).total_seconds())

                if overlap_seconds > 0:
                    slot_key = (day_of_week, time_slot)

                    # Initialize if needed
                    if slot_key not in slot_occupied_seconds:
                        slot_occupied_seconds[slot_key] = 0.0
                        slot_weeks[slot_key] = set()

                    # Add occupied seconds
                    slot_occupied_seconds[slot_key] += overlap_seconds

                    # Track which calendar week this data point belongs to
                    # Use ISO calendar week (year, week_number) to properly handle
                    # week boundaries and year transitions
                    # This ensures intervals on the same calendar week are grouped together
                    year, week_number, _ = overlap_start.isocalendar()
                    week_key = (year, week_number)
                    slot_weeks[slot_key].add(week_key)

                # Move to next slot
                current_time = slot_end

        # Calculate prior values for each slot
        time_priors: dict[tuple[int, int], float] = {}
        data_points: dict[tuple[int, int], int] = {}

        # Total seconds per slot per week (3600 seconds = 1 hour)
        seconds_per_slot_per_week = 3600.0

        for slot_key, occupied_seconds in slot_occupied_seconds.items():
            weeks_set = slot_weeks[slot_key]
            num_weeks_with_data = len(weeks_set)

            # Calculate total slot seconds: number of weeks with data × seconds per slot per week
            # We use weeks_with_data because we can only calculate occupancy for weeks we have data
            total_slot_seconds = num_weeks_with_data * seconds_per_slot_per_week

            if total_slot_seconds > 0:
                # Calculate occupancy percentage
                prior_value = occupied_seconds / total_slot_seconds

                # Apply safety bounds [0.1, 0.9]
                prior_value = max(
                    TIME_PRIOR_MIN_BOUND, min(TIME_PRIOR_MAX_BOUND, prior_value)
                )

                time_priors[slot_key] = prior_value
                # Store number of weeks with data as data_points
                data_points[slot_key] = num_weeks_with_data
            else:
                # No data for this slot - skip (will default to 0.5 at retrieval)
                _LOGGER.debug(
                    "No data for slot (day=%d, slot=%d) in area %s",
                    slot_key[0],
                    slot_key[1],
                    self.area_name,
                )

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
