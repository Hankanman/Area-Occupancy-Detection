"""Minimal intelligent feature type builder."""

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


@dataclass
class FeatureType:
    """Feature type."""

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
_FEATURE_TYPE_DATA = {
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


class FeatureTypeManager:
    """Feature type manager."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the feature type manager."""
        self.config = config or {}
        self._feature_types = self._build_feature_types()

    @property
    def feature_types(self) -> dict[InputType, FeatureType]:
        """Get the feature types."""
        return self._feature_types

    def get_feature_type(self, input_type: InputType) -> FeatureType:
        """Get the feature type for an input type."""
        return self._feature_types[input_type]

    def update_feature_type(
        self, input_type: InputType, feature_type: dict[str, Any]
    ) -> None:
        """Update the feature type for an input type."""
        self._feature_types[input_type] = FeatureType(**feature_type)

    def update_feature_types(self, config: dict[str, Any]) -> None:
        """Update the feature types."""
        self.config = config
        self._feature_types = self._build_feature_types()

    def _build_feature_types(self) -> dict[InputType, FeatureType]:
        """Build the feature types."""
        types: dict[InputType, FeatureType] = {}
        for input_type, params in _FEATURE_TYPE_DATA.items():
            p = params.copy()
            # Allow config to override weight/active_states
            if self.config:
                p["weight"] = self.config.get(f"{input_type.value}_weight", p["weight"])
                p["active_states"] = self.config.get(
                    f"{input_type.value}_active_states", p["active_states"]
                )
                p["active_range"] = self.config.get(
                    f"{input_type.value}_active_range", p["active_range"]
                )
            types[input_type] = FeatureType(**p)
        return types
