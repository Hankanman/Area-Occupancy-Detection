"""Area purpose definitions for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator


class AreaPurpose(StrEnum):
    """Area purpose types."""

    PASSAGEWAY = "passageway"
    UTILITY = "utility"
    FOOD_PREP = "food_prep"
    EATING = "eating"
    WORKING = "working"
    SOCIAL = "social"
    RELAXING = "relaxing"
    SLEEPING = "sleeping"


@dataclass
class Purpose:
    """Area purpose definition with associated decay properties."""

    purpose: AreaPurpose
    name: str
    description: str
    half_life: float  # in seconds


class PurposeManager:
    """Purpose manager for area purposes."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str | None = None,
    ) -> None:
        """Initialize the purpose manager.

        Args:
            coordinator: The coordinator instance
            area_name: Optional area name for multi-area support
        """
        self.coordinator = coordinator
        self.area_name = area_name
        # Get config from areas dict if area_name provided
        if (
            area_name
            and hasattr(coordinator, "areas")
            and area_name in coordinator.areas
        ):
            self.config = coordinator.areas[area_name].config
        else:
            # No area_name provided - raise error as config is required
            raise ValueError(
                "Area name is required to access config in multi-area architecture"
            )
        self._current_purpose: Purpose | None = None

    async def async_initialize(self) -> None:
        """Initialize the purpose manager."""
        # Get the purpose from configuration
        purpose_value = getattr(self.config, "purpose", None)
        if purpose_value:
            try:
                purpose_enum = AreaPurpose(purpose_value)
                self._current_purpose = PURPOSE_DEFINITIONS[purpose_enum]
            except (ValueError, KeyError):
                # Fallback to default purpose
                self._current_purpose = PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL]
        else:
            # Default purpose
            self._current_purpose = PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL]

    @property
    def current_purpose(self) -> Purpose:
        """Get the current purpose."""
        if self._current_purpose is None:
            return PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL]
        return self._current_purpose

    @property
    def half_life(self) -> float:
        """Get the half-life for the current purpose."""
        return self.current_purpose.half_life

    def get_purpose(self, purpose: AreaPurpose) -> Purpose:
        """Get purpose definition by enum."""
        return PURPOSE_DEFINITIONS[purpose]

    def get_all_purposes(self) -> dict[AreaPurpose, Purpose]:
        """Get all purpose definitions."""
        return PURPOSE_DEFINITIONS.copy()

    def set_purpose(self, purpose: AreaPurpose) -> None:
        """Set the current purpose."""
        self._current_purpose = PURPOSE_DEFINITIONS[purpose]

    def cleanup(self) -> None:
        """Clean up the purpose manager."""
        self._current_purpose = None


# Purpose definitions based on the provided table
PURPOSE_DEFINITIONS: dict[AreaPurpose, Purpose] = {
    AreaPurpose.PASSAGEWAY: Purpose(
        purpose=AreaPurpose.PASSAGEWAY,
        name="Passageway",
        description="Quick walk-through: halls, stair landings, entry vestibules. Motion evidence should disappear almost immediately after the last footstep.",
        half_life=60.0,
    ),
    AreaPurpose.UTILITY: Purpose(
        purpose=AreaPurpose.UTILITY,
        name="Utility",
        description="Laundry room, pantry, boot room. Short functional visits (grab the detergent, put on shoes) with little lingering.",
        half_life=120.0,
    ),
    AreaPurpose.FOOD_PREP: Purpose(
        purpose=AreaPurpose.FOOD_PREP,
        name="Food-Prep",
        description="Kitchen work zone around the hob or countertop. Residents step away to the fridge or sink and return; a few minutes of memory prevents flicker.",
        half_life=300.0,
    ),
    AreaPurpose.EATING: Purpose(
        purpose=AreaPurpose.EATING,
        name="Eating",
        description="Dining table, breakfast bar. Family members usually stay seated 10-20 minutes but may be fairly still between bites.",
        half_life=600.0,
    ),
    AreaPurpose.WORKING: Purpose(
        purpose=AreaPurpose.WORKING,
        name="Working / Studying",
        description='Home office, homework desk. Long seated sessions with occasional trips for coffee or printer; ten-minute half-life avoids premature "vacant".',
        half_life=600.0,
    ),
    AreaPurpose.SOCIAL: Purpose(
        purpose=AreaPurpose.SOCIAL,
        name="Social",
        description="Living room, play zone, game area. Conversations or board games create sporadic motion; evidence fades gently to ride out quiet pauses.",
        half_life=720.0,
    ),
    AreaPurpose.RELAXING: Purpose(
        purpose=AreaPurpose.RELAXING,
        name="Relaxing",
        description='TV lounge, reading nook, music corner. People can remain very still while watching or reading; a quarter-hour memory keeps the room "occupied" through stretches of calm.',
        half_life=900.0,
    ),
    AreaPurpose.SLEEPING: Purpose(
        purpose=AreaPurpose.SLEEPING,
        name="Sleeping",
        description='Bedrooms, nap pods. Motion is scarce; a long half-life prevents false vacancy during deep sleep yet lets the house revert to "empty" within a couple of hours after everyone gets up.',
        half_life=1800.0,
    ),
}


def get_purpose_options() -> list[dict[str, str]]:
    """Get purpose options for SelectSelector."""
    return [
        {"value": purpose.purpose.value, "label": purpose.name}
        for purpose in PURPOSE_DEFINITIONS.values()
    ]
