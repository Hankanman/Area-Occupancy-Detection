"""Minimal intelligent entity type builder."""

from enum import StrEnum
import logging
from typing import Any

from homeassistant.const import (
    STATE_CLOSED,
    STATE_CLOSING,
    STATE_ON,
    STATE_OPEN,
    STATE_OPENING,
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STANDBY,
)

_LOGGER = logging.getLogger(__name__)


class CorrelationType(StrEnum):
    """Correlation type for sensor-occupancy analysis."""

    STRONG_POSITIVE = "strong_positive"
    POSITIVE = "positive"
    STRONG_NEGATIVE = "strong_negative"
    NEGATIVE = "negative"
    NONE = "none"
    BINARY_LIKELIHOOD = "binary_likelihood"


class AnalysisStatus(StrEnum):
    """Analysis status for entity correlation analysis."""

    NOT_ANALYZED = "not_analyzed"
    MOTION_EXCLUDED = "motion_sensor_excluded"
    ANALYZED = "analyzed"


class InputType(StrEnum):
    """Input type."""

    MOTION = "motion"
    MEDIA = "media"
    APPLIANCE = "appliance"
    DOOR = "door"
    WINDOW = "window"
    COVER = "cover"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    ILLUMINANCE = "illuminance"
    CO2 = "co2"
    CO = "co"
    SOUND_PRESSURE = "sound_pressure"
    PRESSURE = "pressure"
    AIR_QUALITY = "air_quality"
    VOC = "voc"
    PM25 = "pm25"
    PM10 = "pm10"
    POWER = "power"
    SLEEP = "sleep"
    ENVIRONMENTAL = "environmental"
    UNKNOWN = "unknown"


class EntityType:
    """Entity type. active_states or active_range must be provided, but not both."""

    def __init__(
        self,
        input_type: InputType,
        weight: float | None = None,
        prob_given_true: float | None = None,
        prob_given_false: float | None = None,
        active_states: list[str] | None = None,
        active_range: tuple[float, float] | None = None,
        strength_multiplier: float | None = None,
    ) -> None:
        """Initialize the entity type.

        Args:
            input_type: The input type enum.
            weight: Weight for this entity type. If None, uses default from DEFAULT_TYPES.
            prob_given_true: Probability given true occupancy. If None, uses default.
            prob_given_false: Probability given false occupancy. If None, uses default.
            active_states: List of active states (mutually exclusive with active_range).
                If None, uses default. Empty list is treated as "use defaults".
            active_range: Tuple of (min, max) for active range (mutually exclusive with active_states).
                If None, uses default.
            strength_multiplier: Logit-space multiplier for sensor strength.
                If None, uses default from DEFAULT_TYPES (e.g., 3.0 for MOTION, 2.0 for others).

        Raises:
            ValueError: If neither or both of active_states and active_range are provided,
                or if parameter values are invalid.
        """
        self.input_type = input_type

        # Validate weight if provided
        if weight is not None:
            if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                raise ValueError(f"Invalid weight for {input_type}: {weight}")

        # Validate active_states if provided
        if active_states is not None:
            if not isinstance(active_states, list) or not all(
                isinstance(s, str) for s in active_states
            ):
                raise ValueError(
                    f"Invalid active states for {input_type}: {active_states}"
                )

        # Validate active_range if provided
        if active_range is not None:
            if not isinstance(active_range, tuple) or len(active_range) != 2:
                raise ValueError(
                    f"Invalid active range for {input_type}: {active_range}"
                )

        # Get defaults from DEFAULT_TYPES with safe fallback
        default_type = DEFAULT_TYPES.get(input_type)
        if default_type is None:
            _LOGGER.warning(
                "InputType %s is missing from DEFAULT_TYPES, falling back to UNKNOWN. "
                "Please update DEFAULT_TYPES to include this input type.",
                input_type,
            )
            default_type = DEFAULT_TYPES.get(InputType.UNKNOWN)
            if default_type is None:
                # Ultimate fallback if UNKNOWN is also missing (should never happen)
                default_type = {
                    "weight": 0.5,
                    "prob_given_true": 0.5,
                    "prob_given_false": 0.05,
                    "active_states": [STATE_ON],
                    "active_range": None,
                    "strength_multiplier": 2.0,
                }
        defaults = dict(default_type)

        # Start with defaults, then override with explicit params if provided
        self.weight = weight if weight is not None else defaults["weight"]
        self.prob_given_true = (
            prob_given_true
            if prob_given_true is not None
            else defaults["prob_given_true"]
        )
        self.prob_given_false = (
            prob_given_false
            if prob_given_false is not None
            else defaults["prob_given_false"]
        )

        # Handle active_states and active_range
        # Empty list for active_states is treated as "use defaults"
        has_explicit_states = active_states is not None and len(active_states) > 0
        has_explicit_range = active_range is not None

        if has_explicit_states and has_explicit_range:
            raise ValueError("Cannot provide both active_states and active_range")
        if has_explicit_states:
            self.active_states = active_states
            self.active_range = None
        elif has_explicit_range:
            self.active_range = active_range
            self.active_states = None
        else:
            # Use defaults
            self.active_states = defaults["active_states"]
            self.active_range = defaults["active_range"]

        # Validate that we have exactly one of active_states or active_range
        has_active_states = (
            self.active_states is not None and len(self.active_states) > 0
        )
        has_active_range = self.active_range is not None

        if not has_active_states and not has_active_range:
            raise ValueError("Either active_states or active_range must be provided")
        if has_active_states and has_active_range:
            raise ValueError("Cannot provide both active_states and active_range")

        # Set strength multiplier (per-type scaling in logit space)
        self.strength_multiplier = (
            strength_multiplier
            if strength_multiplier is not None
            else defaults.get("strength_multiplier", 2.0)
        )


# Input type classifications for probability calculations
PRESENCE_INPUT_TYPES: set[InputType] = {
    InputType.MOTION,
    InputType.MEDIA,
    InputType.APPLIANCE,
    InputType.DOOR,
    InputType.WINDOW,
    InputType.COVER,
    InputType.POWER,
    InputType.SLEEP,
}

BINARY_INPUT_TYPES: set[InputType] = {
    InputType.MEDIA,
    InputType.APPLIANCE,
    InputType.DOOR,
    InputType.WINDOW,
}

ENVIRONMENTAL_INPUT_TYPES: set[InputType] = {
    InputType.TEMPERATURE,
    InputType.HUMIDITY,
    InputType.ILLUMINANCE,
    InputType.CO2,
    InputType.CO,
    InputType.SOUND_PRESSURE,
    InputType.PRESSURE,
    InputType.AIR_QUALITY,
    InputType.VOC,
    InputType.PM25,
    InputType.PM10,
    InputType.ENVIRONMENTAL,
}

DEFAULT_TYPES: dict[InputType, dict[str, Any]] = {
    InputType.MOTION: {
        "weight": 1,
        "prob_given_true": 0.95,  # Much higher for ground truth
        "prob_given_false": 0.005,  # PIR sensors rarely false-trigger
        "active_states": [STATE_ON],
        "active_range": None,
        "strength_multiplier": 3.0,
    },
    InputType.MEDIA: {
        "weight": 0.85,
        "prob_given_true": 0.65,
        "prob_given_false": 0.02,
        "active_states": [STATE_PLAYING, STATE_PAUSED],
        "active_range": None,
        "strength_multiplier": 2.0,
    },
    InputType.APPLIANCE: {
        "weight": 0.4,
        "prob_given_true": 0.2,
        "prob_given_false": 0.02,
        "active_states": [STATE_ON, STATE_STANDBY],
        "active_range": None,
        "strength_multiplier": 2.0,
    },
    InputType.DOOR: {
        "weight": 0.3,
        "prob_given_true": 0.2,
        "prob_given_false": 0.02,
        "active_states": [STATE_CLOSED],
        "active_range": None,
        "strength_multiplier": 2.0,
    },
    InputType.WINDOW: {
        "weight": 0.2,
        "prob_given_true": 0.2,
        "prob_given_false": 0.02,
        "active_states": [STATE_OPEN],
        "active_range": None,
        "strength_multiplier": 2.0,
    },
    InputType.COVER: {
        "weight": 0.5,
        "prob_given_true": 0.35,
        "prob_given_false": 0.02,
        "active_states": [STATE_OPENING, STATE_CLOSING],
        "active_range": None,
        "strength_multiplier": 2.0,
    },
    InputType.TEMPERATURE: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (18.0, 24.0),
        "strength_multiplier": 2.0,
    },
    InputType.HUMIDITY: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (70.0, 100.0),
        "strength_multiplier": 2.0,
    },
    InputType.ILLUMINANCE: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (30.00, 100000.0),
        "strength_multiplier": 2.0,
    },
    InputType.CO2: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (400.0, 1200.0),
        "strength_multiplier": 2.0,
    },
    InputType.CO: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (5.0, 50.0),  # ppm - elevated levels indicate human activity
        "strength_multiplier": 2.0,
    },
    InputType.SOUND_PRESSURE: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (40.0, 80.0),
        "strength_multiplier": 2.0,
    },
    InputType.PRESSURE: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (980.0, 1050.0),
        "strength_multiplier": 2.0,
    },
    InputType.AIR_QUALITY: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (50.0, 150.0),
        "strength_multiplier": 2.0,
    },
    InputType.VOC: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (200.0, 1000.0),
        "strength_multiplier": 2.0,
    },
    InputType.PM25: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (12.0, 55.0),
        "strength_multiplier": 2.0,
    },
    InputType.PM10: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (55.0, 155.0),
        "strength_multiplier": 2.0,
    },
    InputType.POWER: {
        "weight": 0.3,
        "prob_given_true": 0.2,
        "prob_given_false": 0.02,
        "active_states": None,
        "active_range": (0.1, 10.0),
        "strength_multiplier": 2.0,
    },
    InputType.SLEEP: {
        "weight": 0.9,
        "prob_given_true": 0.95,
        "prob_given_false": 0.02,
        "active_states": [STATE_ON],
        "active_range": None,
        "strength_multiplier": 3.0,
    },
    InputType.ENVIRONMENTAL: {
        "weight": 0.1,
        "prob_given_true": 0.09,
        "prob_given_false": 0.01,
        "active_states": None,
        "active_range": (0.0, 0.2),
        "strength_multiplier": 2.0,
    },
    InputType.UNKNOWN: {
        "weight": 0.85,
        "prob_given_true": 0.15,
        "prob_given_false": 0.03,
        "active_states": [STATE_ON],
        "active_range": None,
        "strength_multiplier": 2.0,
    },
}
