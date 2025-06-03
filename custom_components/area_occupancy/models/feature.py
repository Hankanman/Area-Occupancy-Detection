"""Feature model."""

from datetime import datetime
from typing import Any, TypedDict

from homeassistant.core import HomeAssistant

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
from ..logic.prior import get_prior_data
from ..state.containers import PriorData
from .feature_type import FeatureType, FeatureTypeManager, InputTypeEnum


class Feature(TypedDict):
    """Type for sensor state information."""

    type: FeatureType
    state: str | float | bool | None
    is_active: bool
    probability: float
    weighted_probability: float
    last_changed: str | None
    available: bool
    prior: PriorData


class FeatureManager:
    """Manages features."""

    async def __init__(self, config: dict[str, Any], hass: HomeAssistant) -> None:
        """Initialize the features."""
        self.config = config
        self.hass = hass
        self._features = await self.map_inputs_to_features()
        self._feature_types = FeatureTypeManager(self.config).feature_types

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
        start_time = self.config.get(CONF_HISTORY_PERIOD, 7)
        end_time = datetime.now()
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
                    prior=await get_prior_data(
                        self.hass,
                        self,
                        primary_sensor,
                        input,
                        start_time,
                        end_time,
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
