"""Minimal intelligent entity type builder."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from homeassistant.const import (
    STATE_OFF,
    STATE_ON,
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STANDBY,
)


class InputType(StrEnum):
    """Input type."""

    MOTION = "motion"
    MEDIA = "media"
    APPLIANCE = "appliance"
    DOOR = "door"
    WINDOW = "window"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    ILLUMINANCE = "illuminance"
    ENVIRONMENTAL = "environmental"
    UNKNOWN = "unknown"


@dataclass
class EntityType:
    """Entity type. active_states or active_range must be provided, but not both."""

    input_type: InputType
    weight: float
    prob_given_true: float
    prob_given_false: float
    active_states: list[str] | None = None
    active_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Post init."""

        # Validate that we have exactly one of active_states or active_range
        has_active_states = (
            self.active_states is not None and len(self.active_states) > 0
        )
        has_active_range = self.active_range is not None

        if not has_active_states and not has_active_range:
            raise ValueError("Either active_states or active_range must be provided")
        if has_active_states and has_active_range:
            raise ValueError("Cannot provide both active_states and active_range")

    @classmethod
    def create(cls, input_type: InputType, config: Any = None) -> "EntityType":
        """Create an EntityType with optional configuration overrides."""
        # Default data for each input type
        data = DEFAULT_TYPES

        params = dict(data[input_type])

        # Apply configuration overrides if available
        if config:
            # Apply weight override
            weights = getattr(config, "weights", None)
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
                    params["weight"] = weight_attr

            # Apply active states override
            sensor_states = getattr(config, "sensor_states", None)
            if sensor_states:
                states_attr = getattr(sensor_states, input_type.value, None)
                if states_attr is not None:
                    if not isinstance(states_attr, list) or not all(
                        isinstance(s, str) for s in states_attr
                    ):
                        raise ValueError(
                            f"Invalid active states for {input_type}: {states_attr}"
                        )
                    params["active_states"] = states_attr
                    params["active_range"] = None  # Clear range when states are set

            # Apply active range override
            range_config_attr = f"{input_type.value}_active_range"
            range_attr = getattr(config, range_config_attr, None)
            if range_attr is not None:
                if not isinstance(range_attr, tuple) or len(range_attr) != 2:
                    raise ValueError(
                        f"Invalid active range for {input_type}: {range_attr}"
                    )
                params["active_range"] = range_attr
                params["active_states"] = None  # Clear states when range is set

        return cls(input_type=input_type, **params)


DEFAULT_TYPES: dict[InputType, dict[str, Any]] = {
    InputType.MOTION: {
        "weight": 1,
        "prob_given_true": 0.95,  # Much higher for ground truth
        "prob_given_false": 0.02,  # Lower false positive rate
        "active_states": [STATE_ON],
        "active_range": None,
    },
    InputType.MEDIA: {
        "weight": 0.85,
        "prob_given_true": 0.65,
        "prob_given_false": 0.02,
        "active_states": [STATE_PLAYING, STATE_PAUSED],
        "active_range": None,
    },
    InputType.APPLIANCE: {
        "weight": 0.4,
        "prob_given_true": 0.2,
        "prob_given_false": 0.02,
        "active_states": [STATE_ON, STATE_STANDBY],
        "active_range": None,
    },
    InputType.DOOR: {
        "weight": 0.3,
        "prob_given_true": 0.2,
        "prob_given_false": 0.02,
        "active_states": [STATE_OFF],
        "active_range": None,
    },
    InputType.WINDOW: {
        "weight": 0.2,
        "prob_given_true": 0.2,
        "prob_given_false": 0.02,
        "active_states": [STATE_ON],
        "active_range": None,
    },
    InputType.TEMPERATURE: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (18.0, 24.0),
    },
    InputType.HUMIDITY: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (70.0, 100.0),
    },
    InputType.ILLUMINANCE: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (30.00, 100000.0),
    },
    InputType.ENVIRONMENTAL: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (0.0, 0.2),
    },
    InputType.UNKNOWN: {
        "weight": 0.85,
        "prob_given_true": 0.15,
        "prob_given_false": 0.03,
        "active_states": [STATE_ON],
        "active_range": None,
    },
}
