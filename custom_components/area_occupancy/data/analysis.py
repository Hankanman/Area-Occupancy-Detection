"""Analysis classes for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import DEFAULT_LOOKBACK_DAYS
from ..data.entity_type import InputType
from ..db.queries import is_occupied_intervals_cache_valid
from ..utils import ensure_timezone_aware, format_area_names
from .prior import Prior

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


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
                _LOGGER.info(
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
            # Note: Time-based priors (hourly) could be calculated here in the future
            self.area.prior.set_global_prior(global_prior)

            _LOGGER.info(
                "Prior analysis completed for area %s: global_prior=%.3f (occupied: %.1f hours over %.1f days, %d intervals)",
                self.area_name,
                global_prior,
                occupied_duration / 3600,
                actual_period_duration / 86400,
                len(occupied_intervals),
            )

        except (ValueError, TypeError, RuntimeError) as e:
            _LOGGER.error(
                "Failed to calculate prior for area %s: %s", self.area_name, e
            )

    def _get_entity_ids_by_type(self, input_type: InputType) -> list[str]:
        """Get entity IDs for a specific input type in this area."""
        entities = self.area.entities.get_entities_by_input_type(input_type)
        return list(entities.keys())


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
        _LOGGER.info(
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


async def run_full_analysis(
    coordinator: AreaOccupancyCoordinator, _now: datetime | None = None
) -> None:
    """Run the full analysis chain for all areas.

    This function orchestrates the complete analysis process:
    1. Sync states from recorder
    2. Database health check and pruning
    3. Populate occupied intervals cache
    4. Run interval aggregation
    5. Recalculate priors for all areas
    6. Run correlation analysis
    7. Save data (preserve decay state before refresh)
    8. Refresh coordinator
    9. Save data (persist all changes)

    Args:
        coordinator: The coordinator instance containing areas and database
        _now: Optional timestamp for the analysis run (used by timer)
    """
    from ..db.correlation import run_correlation_analysis  # noqa: PLC0415

    if _now is None:
        _now = dt_util.utcnow()

    try:
        # Step 1: Import recent data from recorder
        await coordinator.db.sync_states()

        # Step 2: Prune old intervals and run health check
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
        if pruned_count > 0:
            _LOGGER.info("Pruned %d old intervals during analysis", pruned_count)

        # Step 3: Ensure OccupiedIntervalsCache is populated before aggregation
        await ensure_occupied_intervals_cache(coordinator)

        # Step 4: Run interval aggregation (safe now that cache exists)
        await run_interval_aggregation(coordinator, _now)

        # Step 5: Recalculate priors with new data for all areas
        for area in coordinator.areas.values():
            await area.run_prior_analysis()

        # Step 6: Run correlation analysis (requires OccupiedIntervalsCache)
        await run_correlation_analysis(coordinator)

        # Step 7: Save data (preserve decay state before refresh)
        # This ensures decay state is saved before async_refresh() potentially resets it
        await coordinator.hass.async_add_executor_job(coordinator.db.save_data)

        # Step 8: Refresh the coordinator
        await coordinator.async_refresh()

        # Step 9: Save data (persist all changes after refresh)
        await coordinator.hass.async_add_executor_job(coordinator.db.save_data)

    except (HomeAssistantError, OSError, RuntimeError) as err:
        _LOGGER.error("Failed to run historical analysis: %s", err)
        raise
