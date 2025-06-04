"""Minimal intelligent entity type builder."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from homeassistant.const import (
    STATE_CLOSED,
    STATE_ON,
    STATE_OPEN,
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STANDBY,
)

from ..utils import validate_prior, validate_prob, validate_weight

from ..coordinator import AreaOccupancyCoordinator

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
_ENTITY_TYPE_DATA = {
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
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Initialize the entity type manager."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self._entity_types = self._build_entity_types()

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
            raise ValueError(f"Entity type must contain all required fields: {required_fields}")
        
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

    def update_entity_types(self, config: dict[str, Any]) -> None:
        """Update the entity types."""
        
        self.config = config
        self._entity_types = self._build_entity_types()

    def reset_to_defaults(self) -> None:
        """Reset all entity types to their default values."""
        self.config = {}
        self._entity_types = self._build_entity_types()

    def _build_entity_types(self) -> dict[InputType, EntityType]:
        """Build the entity types."""
        types: dict[InputType, EntityType] = {}
        for input_type, params in _ENTITY_TYPE_DATA.items():
            p = params.copy()
            # Allow config to override weight/active_states
            if self.config:
                # Validate config overrides
                weight_key = f"{input_type.value}_weight"
                states_key = f"{input_type.value}_active_states"
                range_key = f"{input_type.value}_active_range"
                
                if weight_key in self.config:
                    weight = self.config[weight_key]
                    if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                        raise ValueError(f"Invalid weight for {input_type}: {weight}")
                    p["weight"] = weight
                
                if states_key in self.config:
                    states = self.config[states_key]
                    if not isinstance(states, list) or not all(isinstance(s, str) for s in states):
                        raise ValueError(f"Invalid active states for {input_type}: {states}")
                    p["active_states"] = states
                    p["active_range"] = None
                
                if range_key in self.config:
                    range_val = self.config[range_key]
                    if not isinstance(range_val, tuple) or len(range_val) != 2:
                        raise ValueError(f"Invalid active range for {input_type}: {range_val}")
                    p["active_range"] = range_val
                    p["active_states"] = None
            
            try:
                types[input_type] = EntityType(**p)
            except Exception as err:
                raise ValueError(f"Failed to create entity type for {input_type}: {err}") from err
                
        return types
