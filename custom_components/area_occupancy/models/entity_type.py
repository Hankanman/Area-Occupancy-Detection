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
from homeassistant.core import State

from ..exceptions import StateError
from ..utils import validate_prior, validate_prob, validate_weight
from .config import Config

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass
class EntityType:
    """Entity type. active_states or active_range must be provided, but not both."""

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

    def is_active(self, state: State) -> bool:
        """Check if the entity state is active.

        Args:
            state: The current state of the entity

        Returns:
            bool: True if the state is active, False otherwise

        """
        try:
            if self.active_states is not None:
                return state.state in self.active_states
            if self.active_range is not None:
                try:
                    state_val = float(state.state)
                    min_val, max_val = self.active_range
                except (ValueError, TypeError):
                    return False
                else:
                    return min_val <= state_val <= max_val

        except Exception as err:
            raise StateError(f"Error checking active state: {err}") from err
        else:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert entity type to dictionary for storage."""
        return {
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
            weight=data["weight"],
            prob_true=data["prob_true"],
            prob_false=data["prob_false"],
            prior=data["prior"],
            active_states=data.get("active_states"),
            active_range=data.get("active_range"),
        )


class InputType(StrEnum):
    """Input type."""

    MOTION = "motion"
    MEDIA = "media"
    APPLIANCE = "appliance"
    DOOR = "door"
    WINDOW = "window"
    LIGHT = "light"
    ENVIRONMENTAL = "environmental"
    WASP_IN_BOX = "wasp_in_box"


# Central definition of types—add or change defaults here.
_ENTITY_TYPE_DATA: dict[InputType, dict[str, Any]] = {
    InputType.MOTION: {
        "weight": 0.8,
        "prob_true": 0.8,
        "prob_false": 0.2,
        "prior": 0.8,
        "active_states": [STATE_ON],
        "active_range": None,
    },
    InputType.MEDIA: {
        "weight": 0.7,
        "prob_true": 0.7,
        "prob_false": 0.3,
        "prior": 0.7,
        "active_states": [STATE_PLAYING, STATE_PAUSED],
        "active_range": None,
    },
    InputType.APPLIANCE: {
        "weight": 0.6,
        "prob_true": 0.6,
        "prob_false": 0.4,
        "prior": 0.6,
        "active_states": [STATE_ON, STATE_STANDBY],
        "active_range": None,
    },
    InputType.DOOR: {
        "weight": 0.5,
        "prob_true": 0.5,
        "prob_false": 0.5,
        "prior": 0.5,
        "active_states": [STATE_CLOSED],
        "active_range": None,
    },
    InputType.WINDOW: {
        "weight": 0.4,
        "prob_true": 0.4,
        "prob_false": 0.6,
        "prior": 0.4,
        "active_states": [STATE_OPEN],
        "active_range": None,
    },
    InputType.LIGHT: {
        "weight": 0.1,
        "prob_true": 0.1,
        "prob_false": 0.9,
        "prior": 0.1,
        "active_states": [STATE_ON],
        "active_range": None,
    },
    InputType.ENVIRONMENTAL: {
        "weight": 0.3,
        "prob_true": 0.3,
        "prob_false": 0.7,
        "prior": 0.3,
        "active_states": None,
        "active_range": (0.0, 0.2),
    },
    InputType.WASP_IN_BOX: {
        "weight": 0.2,
        "prob_true": 0.2,
        "prob_false": 0.8,
        "prior": 0.2,
        "active_states": [STATE_ON],
        "active_range": None,
    },
}


class EntityTypeManager:
    """Entity type manager."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
    ) -> None:
        """Initialize the entity type manager."""
        self.coordinator = coordinator
        self.config = coordinator.config_manager.config
        self._entity_types = self._build_entity_types()
        _LOGGER.debug(
            "EntityTypeManager initialized with entity types: %s", self._entity_types
        )

    @property
    def entity_types(self) -> dict[InputType, EntityType]:
        """Get the entity types."""
        return self._entity_types

    def get_entity_type(self, input_type: InputType) -> EntityType:
        """Get the entity type for an input type."""
        return self._entity_types[input_type]

    def update_entity_type(
        self, input_type: InputType, entity_type: dict[str, Any]
    ) -> None:
        """Update the entity type for an input type."""
        required_fields = {"weight", "prob_true", "prob_false", "prior"}
        if not all(field in entity_type for field in required_fields):
            raise ValueError(
                f"Entity type must contain all required fields: {required_fields}"
            )

        # Validate active states/range
        if "active_states" in entity_type and "active_range" in entity_type:
            raise ValueError("Cannot provide both active_states and active_range")

        if "active_states" not in entity_type and "active_range" not in entity_type:
            raise ValueError("Must provide either active_states or active_range")

        # Create and validate the entity type
        try:
            validated_type = EntityType(**entity_type)
            self._entity_types[input_type] = validated_type
        except Exception as err:
            raise ValueError(f"Invalid entity type data: {err}") from err

    def update_entity_types(self, config: Config) -> None:
        """Update the entity types."""
        self.config = config
        self._entity_types = self._build_entity_types()

    def reset_to_defaults(self) -> None:
        """Reset all entity types to their default values."""
        self._entity_types = self._build_entity_types()

    def _build_entity_types(self) -> dict[InputType, EntityType]:
        """Build the entity types."""
        types: dict[InputType, EntityType] = {}
        for input_type, params in _ENTITY_TYPE_DATA.items():
            p = params.copy()
            # Allow config to override weight/active_states
            if self.config:
                # Get the weights from the config's weights attribute
                weights = getattr(self.config, "weights", None)
                if weights:
                    weight_attr = getattr(weights, input_type.value, None)
                    if weight_attr is not None:
                        if (
                            not isinstance(weight_attr, (int, float))
                            or not 0 <= weight_attr <= 1
                        ):
                            raise ValueError(
                                f"Invalid weight for {input_type}: {weight_attr}"
                            )
                        p["weight"] = weight_attr

                # Get the sensor states from the config's sensor_states attribute
                sensor_states = getattr(self.config, "sensor_states", None)
                if sensor_states:
                    states_attr = getattr(sensor_states, input_type.value, None)
                    if states_attr is not None:
                        if not isinstance(states_attr, list) or not all(
                            isinstance(s, str) for s in states_attr
                        ):
                            raise ValueError(
                                f"Invalid active states for {input_type}: {states_attr}"
                            )
                        p["active_states"] = states_attr
                        p["active_range"] = None

                # For environmental sensors, check for active_range in config
                if input_type == InputType.ENVIRONMENTAL:
                    range_attr = getattr(
                        self.config, "environmental_active_range", None
                    )
                    if range_attr is not None:
                        if not isinstance(range_attr, tuple) or len(range_attr) != 2:
                            raise ValueError(
                                f"Invalid active range for {input_type}: {range_attr}"
                            )
                        p["active_range"] = range_attr
                        p["active_states"] = None

            try:
                types[input_type] = EntityType(**p)
            except Exception as err:
                raise ValueError(
                    f"Failed to create entity type for {input_type}: {err}"
                ) from err

        return types
