"""Minimal intelligent entity type builder."""

from dataclasses import dataclass
from enum import StrEnum
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.const import (
    STATE_CLOSED,
    STATE_ON,
    STATE_OPEN,
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STANDBY,
)

from ..utils import validate_prior, validate_prob, validate_weight

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


class InputType(StrEnum):
    """Input type."""

    MOTION = "motion"
    MEDIA = "media"
    APPLIANCE = "appliance"
    DOOR = "door"
    WINDOW = "window"
    ENVIRONMENTAL = "environmental"


@dataclass
class EntityType:
    """Entity type. active_states or active_range must be provided, but not both."""

    input_type: InputType
    weight: float
    prob_true: float
    prob_false: float
    prior: float
    active_states: list[str] | None = None
    active_range: tuple[float, float] | None = None

    def __post_init__(self):
        """Post init."""
        self.weight = validate_weight(self.weight)
        self.prob_true = validate_prob(self.prob_true)
        self.prob_false = validate_prob(self.prob_false)
        self.prior = validate_prior(self.prior)
        if self.active_states is None and self.active_range is None:
            raise ValueError("Either active_states or active_range must be provided")
        if self.active_states is not None and self.active_range is not None:
            raise ValueError("Cannot provide both active_states and active_range")

    def to_dict(self) -> dict[str, Any]:
        """Convert entity type to dictionary for storage."""
        return {
            "input_type": self.input_type.value,
            "weight": self.weight,
            "prob_true": self.prob_true,
            "prob_false": self.prob_false,
            "prior": self.prior,
            "active_states": self.active_states,
            "active_range": self.active_range,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityType":
        """Create entity type from dictionary."""
        return cls(
            input_type=InputType(data["input_type"]),
            weight=data["weight"],
            prob_true=data["prob_true"],
            prob_false=data["prob_false"],
            prior=data["prior"],
            active_states=data.get("active_states"),
            active_range=data.get("active_range"),
        )


class EntityTypeManager:
    """Entity type manager."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
    ) -> None:
        """Initialize the entity type manager."""
        self.coordinator = coordinator
        self.config = coordinator.config_manager.config
        self._entity_types: dict[InputType, EntityType] = {}

    async def async_initialize(self) -> None:
        """Initialize the entity type manager."""
        self._entity_types = self._build_entity_types()

    @property
    def entity_types(self) -> dict[InputType, EntityType]:
        """Get the entity types."""
        return self._entity_types

    def cleanup(self) -> None:
        """Clean up the entity type manager."""
        self._entity_types = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert entity type manager to dictionary for storage."""
        return {
            "entity_types": {
                input_type.value: entity_type.to_dict()
                for input_type, entity_type in self._entity_types.items()
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: "AreaOccupancyCoordinator"
    ) -> "EntityTypeManager":
        """Create entity type manager from dictionary."""
        manager = cls(coordinator=coordinator)
        if "entity_types" not in data:
            raise ValueError(
                "Invalid storage format: missing 'entity_types' key in data structure. "
                f"Available keys: {list(data.keys())}. "
                f"This should have been caught by storage validation."
            )

        try:
            manager._entity_types = {
                InputType(input_type): EntityType.from_dict(entity_type)
                for input_type, entity_type in data["entity_types"].items()
            }
        except Exception as err:
            raise ValueError(f"Invalid entity type data: {err}") from err

        return manager

    def get_entity_type(self, input_type: InputType) -> EntityType:
        """Get the entity type for an input type."""
        return self._entity_types[input_type]

    def _build_entity_types(self) -> dict[InputType, EntityType]:
        """Build the entity types with configuration overrides."""
        types: dict[InputType, EntityType] = {}

        for input_type, params in _ENTITY_TYPE_DATA.items():
            p = params.copy()

            # Apply configuration overrides if available
            if self.config:
                # Weight from config
                self._apply_weight(input_type, p)

                # Active states from config
                self._apply_states(input_type, p)

                # Active range from config (for range-based sensors)
                self._apply_range(input_type, p)

            try:
                types[input_type] = EntityType(input_type=input_type, **p)
            except Exception as err:
                raise ValueError(
                    f"Failed to create entity type for {input_type}: {err}"
                ) from err

        return types

    def _apply_weight(self, input_type: InputType, params: dict[str, Any]) -> None:
        """Apply weight override from configuration."""
        weights = getattr(self.config, "weights", None)
        if not weights:
            return

        weight_attr = getattr(weights, input_type.value, None)
        if weight_attr is None:
            return

        if not isinstance(weight_attr, (int, float)) or not 0 <= weight_attr <= 1:
            raise ValueError(f"Invalid weight for {input_type}: {weight_attr}")

        params["weight"] = weight_attr

    def _apply_states(self, input_type: InputType, params: dict[str, Any]) -> None:
        """Apply active states override from configuration."""
        sensor_states = getattr(self.config, "sensor_states", None)
        if not sensor_states:
            return

        states_attr = getattr(sensor_states, input_type.value, None)
        if states_attr is None:
            return

        if not isinstance(states_attr, list) or not all(
            isinstance(s, str) for s in states_attr
        ):
            raise ValueError(f"Invalid active states for {input_type}: {states_attr}")

        params["active_states"] = states_attr
        params["active_range"] = None  # Clear range when states are set

    def _apply_range(self, input_type: InputType, params: dict[str, Any]) -> None:
        """Apply active range override from configuration."""
        # Check for generic range override (extensible to other sensor types)
        range_config_attr = f"{input_type.value}_active_range"
        range_attr = getattr(self.config, range_config_attr, None)

        if range_attr is not None:
            if not isinstance(range_attr, tuple) or len(range_attr) != 2:
                raise ValueError(f"Invalid active range for {input_type}: {range_attr}")

            params["active_range"] = range_attr
            params["active_states"] = None  # Clear states when range is set


# Central definition of types—add or change defaults here.
_ENTITY_TYPE_DATA: dict[InputType, dict[str, Any]] = {
    InputType.MOTION: {
        "weight": 0.95,
        "prob_true": 0.25,
        "prob_false": 0.05,
        "prior": 0.35,
        "active_states": [STATE_ON],
        "active_range": None,
    },
    InputType.MEDIA: {
        "weight": 0.8,
        "prob_true": 0.25,
        "prob_false": 0.02,
        "prior": 0.3,
        "active_states": [STATE_PLAYING, STATE_PAUSED],
        "active_range": None,
    },
    InputType.APPLIANCE: {
        "weight": 0.6,
        "prob_true": 0.2,
        "prob_false": 0.02,
        "prior": 0.2356,
        "active_states": [STATE_ON, STATE_STANDBY],
        "active_range": None,
    },
    InputType.DOOR: {
        "weight": 0.5,
        "prob_true": 0.2,
        "prob_false": 0.02,
        "prior": 0.1356,
        "active_states": [STATE_CLOSED],
        "active_range": None,
    },
    InputType.WINDOW: {
        "weight": 0.4,
        "prob_true": 0.2,
        "prob_false": 0.02,
        "prior": 0.1569,
        "active_states": [STATE_OPEN],
        "active_range": None,
    },
    InputType.ENVIRONMENTAL: {
        "weight": 0.3,
        "prob_true": 0.09,
        "prob_false": 0.01,
        "prior": 0.0769,
        "active_states": None,
        "active_range": (0.0, 0.2),
    },
}
