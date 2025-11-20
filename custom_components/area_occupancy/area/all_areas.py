"""AllAreas class for aggregating data across all areas.

The AllAreas class provides simple aggregation methods for the "All Areas" device,
which aggregates occupancy data from all individual areas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.helpers.entity import DeviceInfo

from ..const import (
    ALL_AREAS_IDENTIFIER,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator


class AllAreas:
    """Aggregates occupancy data from all areas.

    Provides simple aggregation methods for the "All Areas" device:
    - Average probability across all areas
    - OR logic for occupied status (any area occupied = occupied)
    - Average prior across all areas
    - Average decay across all areas
    """

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the AllAreas aggregator.

        Args:
            coordinator: The coordinator instance managing all areas
        """
        self.coordinator = coordinator

    def device_info(self) -> DeviceInfo:
        """Return device info for the "All Areas" device.

        Returns:
            DeviceInfo for the aggregated "All Areas" device
        """
        return DeviceInfo(
            identifiers={(DOMAIN, ALL_AREAS_IDENTIFIER)},
            name="All Areas",
            manufacturer=DEVICE_MANUFACTURER,
            model=DEVICE_MODEL,
            sw_version=DEVICE_SW_VERSION,
        )

    def probability(self) -> float:
        """Calculate average probability across all areas.

        Returns:
            Average probability (0.0-1.0) across all areas.
        """
        area_names = self.coordinator.get_area_names()
        # Safety check: return safe default if no areas exist
        # (should never happen due to root cause fix, but provides safety net)
        if not area_names:
            return MIN_PROBABILITY
        probabilities = [
            self.coordinator.probability(area_name) for area_name in area_names
        ]
        avg_prob = sum(probabilities) / len(probabilities)
        return max(MIN_PROBABILITY, min(1.0, avg_prob))

    def occupied(self) -> bool:
        """Check if ANY area is occupied.

        Returns:
            True if any area is occupied, False otherwise
        """
        area_names = self.coordinator.get_area_names()
        return any(self.coordinator.occupied(area_name) for area_name in area_names)

    def area_prior(self) -> float:
        """Calculate average prior across all areas.

        Returns:
            Average prior (0.0-1.0) across all areas.
        """
        area_names = self.coordinator.get_area_names()
        # Safety check: return safe default if no areas exist
        # (should never happen due to root cause fix, but provides safety net)
        if not area_names:
            return MIN_PROBABILITY
        priors = [self.coordinator.area_prior(area_name) for area_name in area_names]
        avg_prior = sum(priors) / len(priors)
        return max(MIN_PROBABILITY, min(1.0, avg_prior))

    def decay(self) -> float:
        """Calculate average decay across all areas.

        Returns:
            Average decay (0.0-1.0) across all areas.
        """
        area_names = self.coordinator.get_area_names()
        # Safety check: return safe default if no areas exist
        # (should never happen due to root cause fix, but provides safety net)
        if not area_names:
            return 1.0
        decays = [self.coordinator.decay(area_name) for area_name in area_names]
        avg_decay = sum(decays) / len(decays)
        return max(0.0, min(1.0, avg_decay))
