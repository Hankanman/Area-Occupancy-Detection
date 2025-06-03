"""Feature model."""

from datetime import timedelta
from typing import Any, TypedDict

from homeassistant.util import dt as dt_util

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
from ..coordinator import AreaOccupancyCoordinator
from ..models.prior import Prior, PriorManager
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
    prior: Prior


class FeatureManager:
    """Manages features."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Initialize the features."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self.hass = coordinator.hass
        self.storage = coordinator.storage
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
            feature_type = self.get_feature_type(input_type)
            for input in inputs:
                features[input] = Feature(
                    type=feature_type,
                    state=None,
                    is_active=False,
                    probability=0.0,
                    weighted_probability=0.0,
                    last_changed=None,
                    available=True,
                    prior=await PriorManager(self.hass).calculate(
                        entity_id=input,
                        hass=self.hass,
                        feature_type=feature_type,
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
        return self.get_feature_type(entity_id).weight

    def get_entity_active_states(self, entity_id: str) -> set[str]:
        """Get the active states of an entity."""
        active_states = self.get_feature_type(entity_id).active_states
        return set(active_states) if active_states is not None else set()

    def get_feature(self, entity_id: str) -> Feature:
        """Get the feature from an entity ID."""
        return self._features[entity_id]
