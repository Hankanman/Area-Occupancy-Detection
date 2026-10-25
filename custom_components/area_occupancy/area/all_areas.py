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
    from ..data.types import ZonePriors


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


def _max(areas: list[Area], method: Callable[[Area], float], default: float) -> float:
    """The highest *method* value over *areas*, clamped to [MIN_PROBABILITY, 1].

    A zone's occupancy probability is "is anyone in here", which can never be
    lower than the chance for any one of its rooms. An average could be, and
    read lower than an occupied room (#557).

    Args:
        areas: The zone's member areas.
        method: Callable that extracts a probability from an Area.
        default: Value when there are no members.

    Returns:
        The clamped maximum, or *default* without members.
    """
    if not areas:
        return default
    return max(MIN_PROBABILITY, min(1.0, max(method(area) for area in areas)))


def _zone_prior(areas: list[Area], empirical: ZonePriors | None) -> float:
    """A zone's prior: how often anyone is in it at this time of the week.

    From the empirical zone priors (the union of the rooms' occupied
    history, #557) combined exactly as a room's live prior is. Until the
    first analysis has computed them, the highest room prior stands in: a
    lower bound on "anyone", where an average is not.

    Args:
        areas: The zone's member areas.
        empirical: The zone priors from the last analysis, if any.

    Returns:
        The zone prior.
    """
    # Imported here: data.forecast and data.prior reach back into the area
    # package at import time.
    from ..data.forecast import forecast_prior  # noqa: PLC0415
    from ..data.prior import PRIOR_FACTOR  # noqa: PLC0415

    if empirical is None or not areas:
        return _max(areas, lambda a: a.area_prior(), MIN_PROBABILITY)
    clock = areas[0].prior
    slot = empirical.time_priors.get((clock.day_of_week, clock.time_slot))
    if slot is None:
        return max(MIN_PROBABILITY, min(1.0, empirical.global_prior))
    return forecast_prior(
        empirical.global_prior,
        slot,
        prior_factor=PRIOR_FACTOR,
        weeks=empirical.data_points.get((clock.day_of_week, clock.time_slot), 0),
    )


class AllAreas:
    """Aggregates occupancy data from all areas.

    Provides simple aggregation methods for the "All Areas" device:
    - Highest probability across all areas (anyone in any area, #557)
    - OR logic for occupied status (any area occupied = occupied)
    - Prior of anyone being in any area, from their combined history
    - Average decay across all areas

    Areas with ``config.exclude_from_all_areas`` set to ``True`` are excluded.
    """

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the AllAreas aggregator.

        Args:
            coordinator: The coordinator instance managing all areas
        """
        self.coordinator = coordinator
        # Set by the analysis (step 7) from the union of the members'
        # occupied history (#557); None until the first run.
        self.empirical: ZonePriors | None = None

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
        """Probability anyone is in an included area: the highest one (#557)."""
        return _max(self._included_areas(), lambda a: a.probability(), MIN_PROBABILITY)

    def occupied(self) -> bool:
        """Check if ANY included area is occupied."""
        return any(area.occupied() for area in self._included_areas())

    def area_prior(self) -> float:
        """Prior that anyone is in an included area (#557)."""
        return _zone_prior(self._included_areas(), self.empirical)

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
        # Set by the analysis (step 7) from the union of the floor's rooms'
        # occupied history (#557); None until the first run.
        self.empirical: ZonePriors | None = None

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
        """Probability anyone is on this floor: its highest room (#557)."""
        return _max(self._floor_areas(), lambda a: a.probability(), MIN_PROBABILITY)

    def occupied(self) -> bool:
        """Check if ANY area on this floor is occupied."""
        return any(area.occupied() for area in self._floor_areas())

    def area_prior(self) -> float:
        """Prior that anyone is on this floor (#557)."""
        return _zone_prior(self._floor_areas(), self.empirical)

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
