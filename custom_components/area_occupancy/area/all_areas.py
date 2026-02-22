"""AllAreas and FloorAreas classes for aggregating data across areas.

The AllAreas class provides simple aggregation methods for the "All Areas" device,
which aggregates occupancy data from all individual areas (excluding opted-out areas).

The FloorAreas class provides the same aggregation scoped to a single HA floor.
"""

from __future__ import annotations

from collections.abc import Callable
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
    from ..area.area import Area
    from ..coordinator import AreaOccupancyCoordinator


def _avg(
    areas: list[Area],
    method: Callable[[Area], float],
    default: float,
    lo: float,
    hi: float,
) -> float:
    """Compute a clamped average of *method* over *areas*.

    Args:
        areas: List of areas to aggregate
        method: Callable that extracts a float from an Area
        default: Value to return when *areas* is empty
        lo: Lower clamp bound
        hi: Upper clamp bound

    Returns:
        Clamped average, or *default* when no areas are present
    """
    if not areas:
        return default
    values = [method(area) for area in areas]
    return max(lo, min(hi, sum(values) / len(values)))


class AllAreas:
    """Aggregates occupancy data from all areas.

    Provides simple aggregation methods for the "All Areas" device:
    - Average probability across all areas
    - OR logic for occupied status (any area occupied = occupied)
    - Average prior across all areas
    - Average decay across all areas

    Areas with ``config.exclude_from_all_areas`` set to ``True`` are excluded.
    """

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the AllAreas aggregator.

        Args:
            coordinator: The coordinator instance managing all areas
        """
        self.coordinator = coordinator

    def _included_areas(self) -> list[Area]:
        """Return areas that are not excluded from All Areas aggregation."""
        return [
            area
            for area in self.coordinator.areas.values()
            if not area.config.exclude_from_all_areas
        ]

    def areas(self) -> list[Area]:
        """Return the list of areas included in this aggregation."""
        return self._included_areas()

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
        """Calculate average probability across included areas."""
        return _avg(
            self._included_areas(),
            lambda a: a.probability(),
            MIN_PROBABILITY,
            MIN_PROBABILITY,
            1.0,
        )

    def occupied(self) -> bool:
        """Check if ANY included area is occupied."""
        return any(area.occupied() for area in self._included_areas())

    def area_prior(self) -> float:
        """Calculate average prior across included areas."""
        return _avg(
            self._included_areas(),
            lambda a: a.area_prior(),
            MIN_PROBABILITY,
            MIN_PROBABILITY,
            1.0,
        )

    def decay(self) -> float:
        """Calculate average decay across included areas."""
        return _avg(self._included_areas(), lambda a: a.decay(), 1.0, 0.0, 1.0)

    def presence_probability(self) -> float:
        """Calculate average presence probability across included areas."""
        return _avg(
            self._included_areas(),
            lambda a: a.presence_probability(),
            MIN_PROBABILITY,
            MIN_PROBABILITY,
            1.0,
        )

    def environmental_confidence(self) -> float:
        """Calculate average environmental confidence across included areas."""
        return _avg(
            self._included_areas(),
            lambda a: a.environmental_confidence(),
            0.5,
            0.0,
            1.0,
        )


class FloorAreas:
    """Aggregates occupancy data for areas on a single floor.

    Provides the same aggregation methods as AllAreas, but scoped to areas
    that belong to a specific Home Assistant floor.

    Floor assignments are resolved at startup / options update time.
    Changing floor assignments in HA requires an integration reload.
    """

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        floor_id: str,
        floor_name: str,
    ) -> None:
        """Initialize the FloorAreas aggregator.

        Args:
            coordinator: The coordinator instance managing all areas
            floor_id: Home Assistant floor ID
            floor_name: Human-readable floor name
        """
        self.coordinator = coordinator
        self.floor_id = floor_id
        self.floor_name = floor_name

    def _floor_areas(self) -> list[Area]:
        """Return areas that belong to this floor."""
        from homeassistant.helpers import area_registry as ar  # noqa: PLC0415

        area_reg = ar.async_get(self.coordinator.hass)
        result: list[Area] = []
        for area in self.coordinator.areas.values():
            if area.config.area_id:
                area_entry = area_reg.async_get_area(area.config.area_id)
                if area_entry and area_entry.floor_id == self.floor_id:
                    result.append(area)
        return result

    def areas(self) -> list[Area]:
        """Return the list of areas on this floor."""
        return self._floor_areas()

    def device_info(self) -> DeviceInfo:
        """Return device info for this floor's device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"floor_{self.floor_id}")},
            name=self.floor_name,
            manufacturer=DEVICE_MANUFACTURER,
            model=DEVICE_MODEL,
            sw_version=DEVICE_SW_VERSION,
        )

    def probability(self) -> float:
        """Calculate average probability across floor areas."""
        return _avg(
            self._floor_areas(),
            lambda a: a.probability(),
            MIN_PROBABILITY,
            MIN_PROBABILITY,
            1.0,
        )

    def occupied(self) -> bool:
        """Check if ANY area on this floor is occupied."""
        return any(area.occupied() for area in self._floor_areas())

    def area_prior(self) -> float:
        """Calculate average prior across floor areas."""
        return _avg(
            self._floor_areas(),
            lambda a: a.area_prior(),
            MIN_PROBABILITY,
            MIN_PROBABILITY,
            1.0,
        )

    def decay(self) -> float:
        """Calculate average decay across floor areas."""
        return _avg(self._floor_areas(), lambda a: a.decay(), 1.0, 0.0, 1.0)

    def presence_probability(self) -> float:
        """Calculate average presence probability across floor areas."""
        return _avg(
            self._floor_areas(),
            lambda a: a.presence_probability(),
            MIN_PROBABILITY,
            MIN_PROBABILITY,
            1.0,
        )

    def environmental_confidence(self) -> float:
        """Calculate average environmental confidence across floor areas."""
        return _avg(
            self._floor_areas(), lambda a: a.environmental_confidence(), 0.5, 0.0, 1.0
        )
