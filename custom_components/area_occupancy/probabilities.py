"""Probability constants and defaults for Area Occupancy Detection."""

from __future__ import annotations
from typing import Final, Dict, Any

from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_CLOSED,
    STATE_OPEN,
    STATE_PLAYING,
    STATE_PAUSED,
)

from .types import EntityType

from .const import (
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_WINDOW,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_PRIOR,
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_APPLIANCES,
    CONF_DOOR_SENSORS,
    CONF_WINDOW_SENSORS,
    CONF_LIGHTS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
)

# Environmental detection baseline settings
ENVIRONMENTAL_BASELINE_PERCENT: Final[float] = 0.05  # 5% deviation allowed around mean
ENVIRONMENTAL_MIN_ACTIVE_DURATION: Final[int] = 300  # seconds of active data needed

# Motion sensor defaults
MOTION_PROB_GIVEN_TRUE: Final[float] = 0.25
MOTION_PROB_GIVEN_FALSE: Final[float] = 0.05

# Door sensor defaults
DOOR_PROB_GIVEN_TRUE: Final[float] = 0.2
DOOR_PROB_GIVEN_FALSE: Final[float] = 0.02

# Window sensor defaults
WINDOW_PROB_GIVEN_TRUE: Final[float] = 0.2
WINDOW_PROB_GIVEN_FALSE: Final[float] = 0.02

# Light sensor defaults
LIGHT_PROB_GIVEN_TRUE: Final[float] = 0.2
LIGHT_PROB_GIVEN_FALSE: Final[float] = 0.02

# Media device defaults
MEDIA_PROB_GIVEN_TRUE: Final[float] = 0.25
MEDIA_PROB_GIVEN_FALSE: Final[float] = 0.02

# Appliance defaults
APPLIANCE_PROB_GIVEN_TRUE: Final[float] = 0.2
APPLIANCE_PROB_GIVEN_FALSE: Final[float] = 0.02

# Environmental defaults
ENVIRONMENTAL_PROB_GIVEN_TRUE: Final[float] = 0.09
ENVIRONMENTAL_PROB_GIVEN_FALSE: Final[float] = 0.01

# Minimum active duration for storing learned priors
MIN_ACTIVE_DURATION_FOR_PRIORS: Final[int] = 300

# Baseline cache TTL (to avoid hitting DB repeatedly)
BASELINE_CACHE_TTL: Final[int] = 21600  # 6 hours in seconds

# Default Priors
MOTION_DEFAULT_PRIOR: Final[float] = 0.35
MEDIA_DEFAULT_PRIOR: Final[float] = 0.30
APPLIANCE_DEFAULT_PRIOR: Final[float] = 0.2356
DOOR_DEFAULT_PRIOR: Final[float] = 0.1356
WINDOW_DEFAULT_PRIOR: Final[float] = 0.1569
LIGHT_DEFAULT_PRIOR: Final[float] = 0.3846
ENVIRONMENTAL_DEFAULT_PRIOR: Final[float] = 0.0769

# Media device state probabilities
MEDIA_STATE_PROBABILITIES: Final[Dict[str, float]] = {
    "playing": 0.9,
    "paused": 0.7,
    "idle": 0.3,
    "off": 0.1,
    "default": 0.0,
}

# Appliance state probabilities
APPLIANCE_STATE_PROBABILITIES: Final[Dict[str, float]] = {
    "active": 0.8,
    "on": 0.8,
    "standby": 0.4,
    "off": 0.1,
    "default": 0.0,
}


class Probabilities:
    """Class to handle probability calculations and weights."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the probabilities handler."""
        self.config = config
        self._sensor_weights = self._get_sensor_weights()
        self._sensor_configs = self._build_sensor_configs()
        self.entity_types: dict[str, EntityType] = {}
        self._map_entities_to_types()

    def _map_entities_to_types(self) -> None:
        """Create mapping of entity IDs to their sensor types."""

        mappings = [
            (CONF_MOTION_SENSORS, "motion"),
            (CONF_MEDIA_DEVICES, "media"),
            (CONF_APPLIANCES, "appliance"),
            (CONF_DOOR_SENSORS, "door"),
            (CONF_WINDOW_SENSORS, "window"),
            (CONF_LIGHTS, "light"),
            (CONF_ILLUMINANCE_SENSORS, "environmental"),
            (CONF_HUMIDITY_SENSORS, "environmental"),
            (CONF_TEMPERATURE_SENSORS, "environmental"),
        ]

        for config_key, sensor_type in mappings:
            for entity_id in self.config.get(config_key, []):
                self.entity_types[entity_id] = sensor_type

    def _get_sensor_weights(self) -> dict[str, float]:
        """Get the configured sensor weights, falling back to defaults if not configured."""
        return {
            "motion": self.config.get(CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION),
            "media": self.config.get(CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA),
            "appliance": self.config.get(
                CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE
            ),
            "door": self.config.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR),
            "window": self.config.get(CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW),
            "light": self.config.get(CONF_WEIGHT_LIGHT, DEFAULT_WEIGHT_LIGHT),
            "environmental": self.config.get(
                CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL
            ),
        }

    def _build_sensor_configs(self) -> dict:
        """Build sensor configurations using current weights."""
        return {
            "motion": {
                "prob_given_true": MOTION_PROB_GIVEN_TRUE,
                "prob_given_false": MOTION_PROB_GIVEN_FALSE,
                "default_prior": MOTION_DEFAULT_PRIOR,
                "weight": self._sensor_weights["motion"],
                "active_states": {STATE_ON},
            },
            "media": {
                "prob_given_true": MEDIA_PROB_GIVEN_TRUE,
                "prob_given_false": MEDIA_PROB_GIVEN_FALSE,
                "default_prior": MEDIA_DEFAULT_PRIOR,
                "weight": self._sensor_weights["media"],
                "active_states": {STATE_PLAYING, STATE_PAUSED},
            },
            "appliance": {
                "prob_given_true": APPLIANCE_PROB_GIVEN_TRUE,
                "prob_given_false": APPLIANCE_PROB_GIVEN_FALSE,
                "default_prior": APPLIANCE_DEFAULT_PRIOR,
                "weight": self._sensor_weights["appliance"],
                "active_states": {STATE_ON},
            },
            "door": {
                "prob_given_true": DOOR_PROB_GIVEN_TRUE,
                "prob_given_false": DOOR_PROB_GIVEN_FALSE,
                "default_prior": DOOR_DEFAULT_PRIOR,
                "weight": self._sensor_weights["door"],
                "active_states": {STATE_OFF, STATE_CLOSED},
            },
            "window": {
                "prob_given_true": WINDOW_PROB_GIVEN_TRUE,
                "prob_given_false": WINDOW_PROB_GIVEN_FALSE,
                "default_prior": WINDOW_DEFAULT_PRIOR,
                "weight": self._sensor_weights["window"],
                "active_states": {STATE_ON, STATE_OPEN},
            },
            "light": {
                "prob_given_true": LIGHT_PROB_GIVEN_TRUE,
                "prob_given_false": LIGHT_PROB_GIVEN_FALSE,
                "default_prior": LIGHT_DEFAULT_PRIOR,
                "weight": self._sensor_weights["light"],
                "active_states": {STATE_ON},
            },
            "environmental": {
                "prob_given_true": ENVIRONMENTAL_PROB_GIVEN_TRUE,
                "prob_given_false": ENVIRONMENTAL_PROB_GIVEN_FALSE,
                "default_prior": ENVIRONMENTAL_DEFAULT_PRIOR,
                "weight": self._sensor_weights["environmental"],
                "active_states": {STATE_ON},
            },
        }

    @property
    def sensor_weights(self) -> dict[str, float]:
        """Get the current sensor weights."""
        return self._sensor_weights

    @property
    def sensor_configs(self) -> dict:
        """Get the current sensor configurations."""
        return self._sensor_configs

    def get_default_prior(self, entity_id: str) -> float:
        """Get the default prior for an entity."""
        sensor_type = self.entity_types.get(entity_id)
        return self._sensor_configs.get(sensor_type, {}).get(
            "default_prior", DEFAULT_PRIOR
        )

    def update_config(self, config: dict[str, Any]) -> None:
        """Update the configuration and recalculate weights and configs."""
        self.config = config
        self._sensor_weights = self._get_sensor_weights()
        self._sensor_configs = self._build_sensor_configs()

    def get_sensor_config(self, entity_id: str) -> dict[str, Any]:
        """Get sensor configuration based on entity type."""
        sensor_type = self.entity_types.get(entity_id)
        return self._sensor_configs.get(sensor_type, {})

    def is_entity_active(
        self,
        entity_id: str,
        state: str,
    ) -> bool:
        """Check if an entity is in an active state.

        Args:
            entity_id: The entity ID to check
            state: The current state of the entity

        Returns:
            bool: True if the entity is considered active, False otherwise
        """
        sensor_type = self.entity_types.get(entity_id)
        if not sensor_type:
            return False

        sensor_config = self.sensor_configs.get(sensor_type, {})
        if not sensor_config:
            return False

        return state in sensor_config.get("active_states", set())
