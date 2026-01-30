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
        areas = list(self.coordinator.areas.values())
        # Safety check: return safe default if no areas exist
        if not areas:
            return MIN_PROBABILITY
        probabilities = [area.probability() for area in areas]
        avg_prob = sum(probabilities) / len(areas)
        return max(MIN_PROBABILITY, min(1.0, avg_prob))

    def occupied(self) -> bool:
        """Check if ANY area is occupied.

        Returns:
            True if any area is occupied, False otherwise
        """
        return any(area.occupied() for area in self.coordinator.areas.values())

    def area_prior(self) -> float:
        """Calculate average prior across all areas.

        Returns:
            Average prior (0.0-1.0) across all areas.
        """
        areas = list(self.coordinator.areas.values())
        if not areas:
            return MIN_PROBABILITY
        priors = [area.area_prior() for area in areas]
        avg_prior = sum(priors) / len(areas)
        return max(MIN_PROBABILITY, min(1.0, avg_prior))

    def decay(self) -> float:
        """Calculate average decay across all areas.

        Returns:
            Average decay (0.0-1.0) across all areas.
        """
        areas = list(self.coordinator.areas.values())
        if not areas:
            return 1.0
        decays = [area.decay() for area in areas]
        avg_decay = sum(decays) / len(areas)
        return max(0.0, min(1.0, avg_decay))

    def presence_probability(self) -> float:
        """Calculate average presence probability across all areas.

        Returns:
            Average presence probability (0.0-1.0) across all areas.
        """
        areas = list(self.coordinator.areas.values())
        if not areas:
            return MIN_PROBABILITY
        probs = [area.presence_probability() for area in areas]
        avg_prob = sum(probs) / len(areas)
        return max(MIN_PROBABILITY, min(1.0, avg_prob))

    def environmental_confidence(self) -> float:
        """Calculate average environmental confidence across all areas.

        Returns:
            Average environmental confidence (0.0-1.0) across all areas.
            0.5 is neutral, >0.5 supports occupancy, <0.5 opposes.
        """
        areas = list(self.coordinator.areas.values())
        if not areas:
            return 0.5  # Neutral when no areas
        confs = [area.environmental_confidence() for area in areas]
        avg_conf = sum(confs) / len(areas)
        return max(0.0, min(1.0, avg_conf))
