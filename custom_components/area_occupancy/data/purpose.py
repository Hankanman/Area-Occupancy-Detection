"""Area purpose definitions for Area Occupancy Detection."""

from __future__ import annotations

from enum import StrEnum


class AreaPurpose(StrEnum):
    """Area purpose types."""

    PASSAGEWAY = "passageway"
    UTILITY = "utility"
    BATHROOM = "bathroom"
    FOOD_PREP = "food_prep"
    EATING = "eating"
    WORKING = "working"
    SOCIAL = "social"
    RELAXING = "relaxing"
    SLEEPING = "sleeping"


class Purpose:
    """Area purpose definition with associated decay properties."""

    def __init__(
        self,
        purpose: AreaPurpose | str | None = None,
        *,
        _name: str | None = None,
        _description: str | None = None,
        _half_life: float | None = None,
    ) -> None:
        """Initialize the purpose.

        Args:
            purpose: The purpose enum or string value. Defaults to SOCIAL if None or invalid.
            _name: Internal parameter for direct creation (used by PURPOSE_DEFINITIONS).
            _description: Internal parameter for direct creation (used by PURPOSE_DEFINITIONS).
            _half_life: Internal parameter for direct creation (used by PURPOSE_DEFINITIONS).

        Note:
            If _name, _description, and _half_life are provided, creates instance directly.
            Otherwise, looks up from PURPOSE_DEFINITIONS.
        """
        # Direct creation from data (for PURPOSE_DEFINITIONS)
        if _name is not None and _description is not None and _half_life is not None:
            if purpose is None:
                raise ValueError("purpose must be provided when using direct creation")
            self.purpose = AreaPurpose(purpose) if isinstance(purpose, str) else purpose
            self.name = _name
            self.description = _description
            self.half_life = _half_life
            return

        # Lookup from PURPOSE_DEFINITIONS
        purpose_value = purpose if purpose is not None else AreaPurpose.SOCIAL

        try:
            purpose_enum = AreaPurpose(purpose_value)
            definition = PURPOSE_DEFINITIONS[purpose_enum]
        except (ValueError, KeyError):
            definition = PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL]

        self.purpose = definition.purpose
        self.name = definition.name
        self.description = definition.description
        self.half_life = definition.half_life

    @classmethod
    def get_purpose(cls, purpose: AreaPurpose) -> Purpose:
        """Get purpose definition by enum.

        Args:
            purpose: The purpose enum to get.

        Returns:
            Purpose instance for the given enum.
        """
        return PURPOSE_DEFINITIONS[purpose]

    @classmethod
    def get_all_purposes(cls) -> dict[AreaPurpose, Purpose]:
        """Get all purpose definitions.

        Returns:
            Dictionary mapping purpose enums to Purpose instances.
        """
        return PURPOSE_DEFINITIONS.copy()

    def cleanup(self) -> None:
        """Clean up the purpose (no-op for compatibility)."""


# Purpose definitions based on the provided table
PURPOSE_DEFINITIONS: dict[AreaPurpose, Purpose] = {
    AreaPurpose.PASSAGEWAY: Purpose(
        purpose=AreaPurpose.PASSAGEWAY,
        _name="Passageway",
        _description="Quick walk-through: halls, stair landings, entry vestibules. Motion evidence should disappear almost immediately after the last footstep.",
        _half_life=45.0,
    ),
    AreaPurpose.UTILITY: Purpose(
        purpose=AreaPurpose.UTILITY,
        _name="Utility",
        _description="Laundry room, pantry, boot room. Short functional visits (grab the detergent, put on shoes) with little lingering.",
        _half_life=90.0,
    ),
    AreaPurpose.BATHROOM: Purpose(
        purpose=AreaPurpose.BATHROOM,
        _name="Bathroom",
        _description="Showers, baths, getting ready. Motion can be obstructed or minimal; a moderate memory prevents darkness during a shower.",
        _half_life=450.0,
    ),
    AreaPurpose.FOOD_PREP: Purpose(
        purpose=AreaPurpose.FOOD_PREP,
        _name="Food-Prep",
        _description="Kitchen work zone around the hob or countertop. Residents step away to the fridge or sink and return; a few minutes of memory prevents flicker.",
        _half_life=240.0,
    ),
    AreaPurpose.EATING: Purpose(
        purpose=AreaPurpose.EATING,
        _name="Eating",
        _description="Dining table, breakfast bar. Family members usually stay seated 10-20 minutes but may be fairly still between bites.",
        _half_life=480.0,
    ),
    AreaPurpose.WORKING: Purpose(
        purpose=AreaPurpose.WORKING,
        _name="Working",
        _description='Home office, homework desk. Long seated sessions with occasional trips for coffee or printer; ten-minute half-life avoids premature "vacant".',
        _half_life=600.0,
    ),
    AreaPurpose.SOCIAL: Purpose(
        purpose=AreaPurpose.SOCIAL,
        _name="Social",
        _description="Living room, play zone, game area. Conversations or board games create sporadic motion; evidence fades gently to ride out quiet pauses.",
        _half_life=480.0,
    ),
    AreaPurpose.RELAXING: Purpose(
        purpose=AreaPurpose.RELAXING,
        _name="Relaxing",
        _description='TV lounge, reading nook, music corner. People can remain very still while watching or reading; a quarter-hour memory keeps the room "occupied" through stretches of calm.',
        _half_life=600.0,
    ),
    AreaPurpose.SLEEPING: Purpose(
        purpose=AreaPurpose.SLEEPING,
        _name="Sleeping",
        _description='Bedrooms, nap pods. Motion is scarce; a long half-life prevents false vacancy during deep sleep yet lets the house revert to "empty" within a couple of hours after everyone gets up.',
        _half_life=1200.0,
    ),
}


def get_purpose_options() -> list[dict[str, str]]:
    """Get purpose options for SelectSelector."""
    return [
        {"value": purpose.purpose.value, "label": purpose.name}
        for purpose in PURPOSE_DEFINITIONS.values()
    ]
