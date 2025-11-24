"""Analysis classes for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from ..const import DEFAULT_LOOKBACK_DAYS
from ..data.entity_type import InputType
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
