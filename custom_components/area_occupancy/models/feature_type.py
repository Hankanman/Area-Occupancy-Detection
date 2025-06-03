"""Hardcoded definition of entity types."""

from enum import StrEnum
from typing import Any, Final, TypedDict

from homeassistant.const import (
    STATE_CLOSED,
    STATE_ON,
    STATE_OPEN,
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STANDBY,
)

from ..const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_DOOR_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WASP,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
)

########################################################
# Default States
########################################################

DEFAULT_ACTIVE_STATES_APPLIANCE: Final[list[str]] = [STATE_ON, STATE_STANDBY]
DEFAULT_ACTIVE_STATES_DOOR: Final[list[str]] = [STATE_CLOSED]
DEFAULT_ACTIVE_STATES_ENVIRONMENTAL: Final[list[str]] = [STATE_ON]
DEFAULT_ACTIVE_STATES_LIGHT: Final[list[str]] = [STATE_ON]
DEFAULT_ACTIVE_STATES_MEDIA: Final[list[str]] = [STATE_PLAYING, STATE_PAUSED]
DEFAULT_ACTIVE_STATES_MOTION: Final[list[str]] = [STATE_ON]
DEFAULT_ACTIVE_STATES_WASP_IN_BOX: Final[list[str]] = [STATE_ON]
DEFAULT_ACTIVE_STATES_WINDOW: Final[list[str]] = [STATE_OPEN]

########################################################
# Default Weights
########################################################

DEFAULT_WEIGHT_APPLIANCE: Final = 0.6
DEFAULT_WEIGHT_DOOR: Final = 0.5
DEFAULT_WEIGHT_ENVIRONMENTAL: Final = 0.3
DEFAULT_WEIGHT_LIGHT: Final = 0.1
DEFAULT_WEIGHT_MEDIA: Final = 0.7
DEFAULT_WEIGHT_MOTION: Final = 0.8
DEFAULT_WEIGHT_WASP_IN_BOX: Final = 0.2
DEFAULT_WEIGHT_WINDOW: Final = 0.4

########################################################
# Default Priors
########################################################

DEFAULT_PRIOR_APPLIANCE: Final = 0.6
DEFAULT_PRIOR_DOOR: Final = 0.5
DEFAULT_PRIOR_ENVIRONMENTAL: Final = 0.3
DEFAULT_PRIOR_LIGHT: Final = 0.1
DEFAULT_PRIOR_MEDIA: Final = 0.7
DEFAULT_PRIOR_MOTION: Final = 0.8
DEFAULT_PRIOR_WASP_IN_BOX: Final = 0.2
DEFAULT_PRIOR_WINDOW: Final = 0.4

########################################################
# Default Probabilities
########################################################

DEFAULT_PROB_TRUE_APPLIANCE: Final = 0.6
DEFAULT_PROB_FALSE_APPLIANCE: Final = 0.4
DEFAULT_PROB_TRUE_DOOR: Final = 0.5
DEFAULT_PROB_FALSE_DOOR: Final = 0.5
DEFAULT_PROB_TRUE_ENVIRONMENTAL: Final = 0.3
DEFAULT_PROB_FALSE_ENVIRONMENTAL: Final = 0.7
DEFAULT_PROB_TRUE_LIGHT: Final = 0.1
DEFAULT_PROB_FALSE_LIGHT: Final = 0.9
DEFAULT_PROB_TRUE_MEDIA: Final = 0.7
DEFAULT_PROB_FALSE_MEDIA: Final = 0.3
DEFAULT_PROB_TRUE_MOTION: Final = 0.8
DEFAULT_PROB_FALSE_MOTION: Final = 0.2
DEFAULT_PROB_TRUE_WASP_IN_BOX: Final = 0.2
DEFAULT_PROB_FALSE_WASP_IN_BOX: Final = 0.8
DEFAULT_PROB_TRUE_WINDOW: Final = 0.4
DEFAULT_PROB_FALSE_WINDOW: Final = 0.6


class FeatureType(TypedDict):
    """Type for input type."""

    weight: float
    prob_true: float
    prob_false: float
    prior: float
    active_states: list[str]


class InputTypeEnum(StrEnum):
    """Enum representing the different types of entities used."""

    MOTION = "motion"
    MEDIA = "media"
    APPLIANCE = "appliance"
    DOOR = "door"
    WINDOW = "window"
    LIGHT = "light"
    ENVIRONMENTAL = "environmental"
    WASP_IN_BOX = "wasp_in_box"


class FeatureTypeManager:
    """Manages input types."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the input types."""
        self.config = config
        self._feature_types = self.build_feature_types()

    @property
    def feature_types(self) -> dict[InputTypeEnum, FeatureType]:
        """Get the input types."""
        return self._feature_types

    def build_feature_types(self) -> dict[InputTypeEnum, FeatureType]:
        """Build the input types."""
        return {
            InputTypeEnum.MOTION: FeatureType(
                weight=self.config.get(CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION),
                prob_true=DEFAULT_PROB_TRUE_MOTION,
                prob_false=DEFAULT_PROB_FALSE_MOTION,
                prior=DEFAULT_PRIOR_MOTION,
                active_states=DEFAULT_ACTIVE_STATES_MOTION,
            ),
            InputTypeEnum.MEDIA: FeatureType(
                weight=self.config.get(CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA),
                prob_true=DEFAULT_PROB_TRUE_MEDIA,
                prob_false=DEFAULT_PROB_FALSE_MEDIA,
                prior=DEFAULT_PRIOR_MEDIA,
                active_states=self.config.get(
                    CONF_MEDIA_ACTIVE_STATES, DEFAULT_ACTIVE_STATES_MEDIA
                ),
            ),
            InputTypeEnum.APPLIANCE: FeatureType(
                weight=self.config.get(CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE),
                prob_true=DEFAULT_PROB_TRUE_APPLIANCE,
                prob_false=DEFAULT_PROB_FALSE_APPLIANCE,
                prior=DEFAULT_PRIOR_APPLIANCE,
                active_states=self.config.get(
                    CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_ACTIVE_STATES_APPLIANCE
                ),
            ),
            InputTypeEnum.DOOR: FeatureType(
                weight=self.config.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR),
                prob_true=DEFAULT_PROB_TRUE_DOOR,
                prob_false=DEFAULT_PROB_FALSE_DOOR,
                prior=DEFAULT_PRIOR_DOOR,
                active_states=self.config.get(
                    CONF_DOOR_ACTIVE_STATE, DEFAULT_ACTIVE_STATES_DOOR
                ),
            ),
            InputTypeEnum.WINDOW: FeatureType(
                weight=self.config.get(CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW),
                prob_true=DEFAULT_PROB_TRUE_WINDOW,
                prob_false=DEFAULT_PROB_FALSE_WINDOW,
                prior=DEFAULT_PRIOR_WINDOW,
                active_states=self.config.get(
                    CONF_WINDOW_ACTIVE_STATE, DEFAULT_ACTIVE_STATES_WINDOW
                ),
            ),
            InputTypeEnum.LIGHT: FeatureType(
                weight=self.config.get(CONF_WEIGHT_LIGHT, DEFAULT_WEIGHT_LIGHT),
                prob_true=DEFAULT_PROB_TRUE_LIGHT,
                prob_false=DEFAULT_PROB_FALSE_LIGHT,
                prior=DEFAULT_PRIOR_LIGHT,
                active_states=DEFAULT_ACTIVE_STATES_LIGHT,
            ),
            InputTypeEnum.ENVIRONMENTAL: FeatureType(
                weight=self.config.get(
                    CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL
                ),
                prob_true=DEFAULT_PROB_TRUE_ENVIRONMENTAL,
                prob_false=DEFAULT_PROB_FALSE_ENVIRONMENTAL,
                prior=DEFAULT_PRIOR_ENVIRONMENTAL,
                active_states=DEFAULT_ACTIVE_STATES_ENVIRONMENTAL,
            ),
            InputTypeEnum.WASP_IN_BOX: FeatureType(
                weight=self.config.get(CONF_WEIGHT_WASP, DEFAULT_WEIGHT_WASP_IN_BOX),
                prob_true=DEFAULT_PROB_TRUE_WASP_IN_BOX,
                prob_false=DEFAULT_PROB_FALSE_WASP_IN_BOX,
                prior=DEFAULT_PRIOR_WASP_IN_BOX,
                active_states=DEFAULT_ACTIVE_STATES_WASP_IN_BOX,
            ),
        }

    def update_feature_types(self, config: dict[str, Any]) -> None:
        """Update the input types."""
        self.config = config
        self._feature_types = self.build_feature_types()
