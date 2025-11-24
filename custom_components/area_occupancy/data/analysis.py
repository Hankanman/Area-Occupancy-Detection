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
        include_media: bool = False,
        include_appliance: bool = False,
    ) -> list[tuple[datetime, datetime]]:
        """Get intervals where the area was occupied based on motion/device activity."""
        # Get active motion sensors for this area
        motion_sensors = self._get_entity_ids_by_type(InputType.MOTION)

        if not motion_sensors:
            _LOGGER.debug("No motion sensors found for area %s", self.area_name)
            return []

        # Calculate time range
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=days)

        # Get media/appliance sensor IDs if needed
        media_sensor_ids = None
        if include_media:
            media_sensor_ids = self._get_entity_ids_by_type(InputType.MEDIA)

        appliance_sensor_ids = None
        if include_appliance:
            appliance_sensor_ids = self._get_entity_ids_by_type(InputType.APPLIANCE)

        # Get occupied intervals from database (raw calculation)
        return self.db.get_occupied_intervals(
            area_name=self.area_name,
            motion_sensor_ids=motion_sensors,
            start_time=start_time,
            end_time=end_time,
            include_media=include_media,
            include_appliance=include_appliance,
            media_sensor_ids=media_sensor_ids,
            appliance_sensor_ids=appliance_sensor_ids,
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
            # We don't include media/appliance for prior calculation to avoid circular dependencies
            occupied_intervals = self.get_occupied_intervals(
                days, include_media=False, include_appliance=False
            )

            if not occupied_intervals:
                _LOGGER.debug(
                    "No occupancy data found for prior calculation in area %s",
                    self.area_name,
                )
                return

            # 2. Calculate global prior using actual data period
            # Determine actual data period from intervals (not fixed lookback)
            # Ensure all datetime objects are timezone-aware UTC
            first_interval_start = min(
                dt_util.as_utc(start) for start, end in occupied_intervals
            )
            last_interval_end = max(
                dt_util.as_utc(end) for start, end in occupied_intervals
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
            occupied_duration = sum(
                (dt_util.as_utc(end) - dt_util.as_utc(start)).total_seconds()
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
                "Prior analysis completed for area %s: global_prior=%.3f (occupied: %.1f hours over %.1f days)",
                self.area_name,
                global_prior,
                occupied_duration / 3600,
                actual_period_duration / 86400,
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
            # Calculate occupied intervals from raw intervals
            analyzer = PriorAnalyzer(coordinator, area_name)
            intervals = await coordinator.hass.async_add_executor_job(
                analyzer.get_occupied_intervals,
                DEFAULT_LOOKBACK_DAYS,
                False,  # include_media
                False,  # include_appliance
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
    coordinator: AreaOccupancyCoordinator, _now: datetime | None = None
) -> None:
    """Run interval aggregation.

    This function aggregates raw intervals older than the retention period
    into daily/weekly/monthly aggregates.

    Args:
        coordinator: The coordinator instance containing areas and database
        _now: Optional timestamp for the aggregation run
    """
    if _now is None:
        _now = dt_util.utcnow()

    try:
        results = await coordinator.hass.async_add_executor_job(
            coordinator.db.run_interval_aggregation
        )
        area_names = ", ".join(coordinator.get_area_names())
        _LOGGER.info(
            "Interval aggregation completed for areas %s: %s",
            area_names,
            results,
        )
    except Exception as err:  # noqa: BLE001
        area_names = ", ".join(coordinator.get_area_names())
        _LOGGER.error(
            "Interval aggregation failed for areas %s: %s",
            area_names,
            err,
        )
        # Don't raise - allow analysis to continue even if aggregation fails


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
    7. Refresh coordinator and save data

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
            area_names = ", ".join(coordinator.get_area_names())
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

        # Step 7: Refresh the coordinator
        await coordinator.async_refresh()

        # Step 8: Save data (always save - no master check)
        await coordinator.hass.async_add_executor_job(coordinator.db.save_data)

    except (HomeAssistantError, OSError, RuntimeError) as err:
        _LOGGER.error("Failed to run historical analysis: %s", err)
        raise
