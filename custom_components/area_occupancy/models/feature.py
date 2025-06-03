"""Feature model."""

import logging

_LOGGER = logging.getLogger(__name__)
_LOGGER.debug("Starting imports for feature.py")

from datetime import timedelta

from homeassistant.util import dt as dt_util

_LOGGER.debug("Imported datetime")

from typing import Any, TypedDict

_LOGGER.debug("Imported typing modules")

from homeassistant.core import HomeAssistant

_LOGGER.debug("Imported HomeAssistant")

from ..const import (
    CONF_APPLIANCES,
    CONF_DOOR_SENSORS,
    CONF_HISTORY_PERIOD,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_WINDOW_SENSORS,
)

_LOGGER.debug("Imported area_occupancy constants")

from ..models.prior import Prior

_LOGGER.debug("Imported Prior, get_prior_data")

from .feature_type import FeatureType, FeatureTypeManager, InputTypeEnum

_LOGGER.debug("Imported feature_type modules")

_LOGGER.debug("Completed all imports for feature.py")


class Feature(TypedDict):
    """Type for sensor state information."""

    type: FeatureType
    state: str | float | bool | None
    is_active: bool
    probability: float
    weighted_probability: float
    last_changed: str | None
    available: bool
    prior: Prior


class FeatureManager:
    """Manages features."""

    def __init__(self, config: dict[str, Any], hass: HomeAssistant) -> None:
        """Initialize the features."""
        self.config = config
        self.hass = hass
        self._features = {}
        self._feature_types = FeatureTypeManager(self.config).feature_types

    async def async_initialize(self) -> None:
        """Initialize the features."""
        self._features = await self.map_inputs_to_features()

    @property
    def features(self) -> dict[str, Feature]:
        """Get the features."""
        return self._features

    @property
    def feature_types(self) -> dict[InputTypeEnum, FeatureType]:
        """Get the feature types."""
        return self._feature_types

    async def update_features(self, config: dict[str, Any]) -> None:
        """Update the features."""
        self.config = config
        self._features = await self.map_inputs_to_features()
        self._feature_types = FeatureTypeManager(self.config).feature_types

    async def map_inputs_to_features(self) -> dict[str, Feature]:
        """Map inputs to features."""
        history_days = self.config.get(CONF_HISTORY_PERIOD, 7)
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=history_days)
        primary_sensor = self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor:
            raise ValueError("Primary occupancy sensor must be configured")
        type_mappings = {
            InputTypeEnum.MOTION: self.config.get(CONF_MOTION_SENSORS, []),
            InputTypeEnum.MEDIA: self.config.get(CONF_MEDIA_DEVICES, []),
            InputTypeEnum.APPLIANCE: self.config.get(CONF_APPLIANCES, []),
            InputTypeEnum.DOOR: self.config.get(CONF_DOOR_SENSORS, []),
            InputTypeEnum.WINDOW: self.config.get(CONF_WINDOW_SENSORS, []),
            InputTypeEnum.LIGHT: self.config.get(CONF_LIGHTS, []),
            InputTypeEnum.ENVIRONMENTAL: self.config.get(CONF_ILLUMINANCE_SENSORS, [])
            + self.config.get(CONF_HUMIDITY_SENSORS, [])
            + self.config.get(CONF_TEMPERATURE_SENSORS, []),
        }
        features: dict[str, Feature] = {}
        for input_type, inputs in type_mappings.items():
            for input in inputs:
                features[input] = Feature(
                    type=self._feature_types[input_type],
                    state=None,
                    is_active=False,
                    probability=0.0,
                    weighted_probability=0.0,
                    last_changed=None,
                    available=True,
                    prior=await Prior.calculate(
                        entity_id=input,
                        hass=self.hass,
                        default_prior=self._feature_types[input_type]["prior"],
                        default_prob_given_true=self._feature_types[input_type][
                            "prob_true"
                        ],
                        default_prob_given_false=self._feature_types[input_type][
                            "prob_false"
                        ],
                        entity_active_states=set(
                            self._feature_types[input_type]["active_states"]
                        ),
                        primary_sensor=primary_sensor,
                        start_time=start_time,
                        end_time=end_time,
                    ),
                )
        return features

    def get_feature_type(self, entity_id: str) -> FeatureType:
        """Get the type of an entity."""
        return self._features[entity_id]["type"]

    def get_entity_weight(self, entity_id: str) -> float:
        """Get the weight of a sensor."""
        return self._features[entity_id]["type"]["weight"]

    def get_entity_active_states(self, entity_id: str) -> set[str]:
        """Get the active states of an entity."""
        return set(self._features[entity_id]["type"]["active_states"])

    def get_feature(self, entity_id: str) -> Feature:
        """Get the feature from an entity ID."""
        return self._features[entity_id]
