"""Probability constants and defaults for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import Any, Final

from homeassistant.const import STATE_OFF, STATE_ON, STATE_OPEN

from .const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
)
from .exceptions import ConfigurationError
from .types import EntityType, ProbabilityConfig

_LOGGER = logging.getLogger(__name__)

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
MEDIA_STATE_PROBABILITIES: Final[dict[str, float]] = {
    "playing": 0.9,
    "paused": 0.7,
    "idle": 0.3,
    "off": 0.1,
    "default": 0.0,
}

# Appliance state probabilities
APPLIANCE_STATE_PROBABILITIES: Final[dict[str, float]] = {
    "active": 0.8,
    "on": 0.8,
    "standby": 0.4,
    "off": 0.1,
    "default": 0.0,
}


class Probabilities:
    """Class to handle probability calculations and weights."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the probabilities handler.

        Args:
            config: Configuration dictionary

        Raises:
            ConfigurationError: If configuration is invalid

        """
        self.config = config
        self._sensor_weights = self._get_sensor_weights()
        self._sensor_configs = self._build_sensor_configs()
        self.entity_types: dict[str, EntityType] = {}
        self._map_entities_to_types()

        _LOGGER.debug(
            "Probabilities initialized with %d entity types and %d sensor configs",
            len(self.entity_types),
            len(self._sensor_configs),
        )

    def _map_entities_to_types(self) -> None:
        """Create mapping of entity IDs to their sensor types.

        Raises:
            ConfigurationError: If entity mapping fails

        """

        def _validate_entity_id(entity_id: str, config_key: str) -> None:
            if not entity_id:
                raise ConfigurationError(f"Empty entity ID in {config_key}")

        try:
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

            # Clear existing mappings first
            self.entity_types.clear()

            for config_key, sensor_type in mappings:
                entities = self.config.get(config_key, [])
                for entity_id in entities:
                    _validate_entity_id(entity_id, config_key)
                    self.entity_types[entity_id] = sensor_type

        except Exception as err:
            raise ConfigurationError(f"Failed to map entities to types: {err}") from err

    def _get_sensor_weights(self) -> dict[str, float]:
        """Get the configured sensor weights, falling back to defaults if not configured.

        Returns:
            Dictionary of sensor type to weight mapping

        Raises:
            ConfigurationError: If weights are invalid

        """

        def _validate_weight(sensor_type: str, weight: float) -> None:
            if not 0 <= weight <= 1:
                raise ConfigurationError(
                    f"Invalid weight for {sensor_type}: {weight}. Must be between 0 and 1."
                )

        try:
            weights = {
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

            # Validate weights
            for sensor_type, weight in weights.items():
                _validate_weight(sensor_type, weight)

        except Exception as err:
            raise ConfigurationError(f"Failed to get sensor weights: {err}") from err
        else:
            return weights

    def _translate_binary_sensor_active_state(self, configured_state: str) -> str:
        """Translate a configured state (e.g., 'open', 'closed') to a binary_sensor state ('on', 'off')."""
        # Default to STATE_OFF if the configured state isn't explicitly STATE_OPEN
        return STATE_ON if configured_state == STATE_OPEN else STATE_OFF

    def _build_sensor_configs(self) -> dict[str, ProbabilityConfig]:
        """Build sensor configurations using current weights and type priors.

        Returns:
            Dictionary of sensor configurations

        Raises:
            ConfigurationError: If sensor configurations are invalid

        """

        def _validate_probability(
            sensor_type: str, prob_name: str, value: float
        ) -> None:
            if not 0 <= value <= 1:
                raise ConfigurationError(
                    f"Invalid {prob_name} for {sensor_type}: {value}"
                )

        def _validate_config(sensor_type: str, config: dict) -> None:
            _validate_probability(
                sensor_type, "prob_given_true", config["prob_given_true"]
            )
            _validate_probability(
                sensor_type, "prob_given_false", config["prob_given_false"]
            )
            _validate_probability(sensor_type, "default_prior", config["default_prior"])
            _validate_probability(sensor_type, "weight", config["weight"])

        try:
            # Get the configured door active state (e.g., 'open' or 'closed')
            configured_door_active_state = self.config.get(
                CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE
            )
            # Translate to binary_sensor state (on/off)
            translated_door_active_state = self._translate_binary_sensor_active_state(
                configured_door_active_state
            )

            # Get the configured window active state (e.g., 'open' or 'closed')
            configured_window_active_state = self.config.get(
                CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE
            )
            # Translate to binary_sensor state (on/off)
            translated_window_active_state = self._translate_binary_sensor_active_state(
                configured_window_active_state
            )

            # Get the configured media active states
            media_active_states = set(
                self.config.get(CONF_MEDIA_ACTIVE_STATES, DEFAULT_MEDIA_ACTIVE_STATES)
            )

            # Get the configured appliance active states
            appliance_active_states = set(
                self.config.get(
                    CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_APPLIANCE_ACTIVE_STATES
                )
            )

            configs = {
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
                    "active_states": media_active_states,
                },
                "appliance": {
                    "prob_given_true": APPLIANCE_PROB_GIVEN_TRUE,
                    "prob_given_false": APPLIANCE_PROB_GIVEN_FALSE,
                    "default_prior": APPLIANCE_DEFAULT_PRIOR,
                    "weight": self._sensor_weights["appliance"],
                    "active_states": appliance_active_states,
                },
                "door": {
                    "prob_given_true": DOOR_PROB_GIVEN_TRUE,
                    "prob_given_false": DOOR_PROB_GIVEN_FALSE,
                    "default_prior": DOOR_DEFAULT_PRIOR,
                    "weight": self._sensor_weights["door"],
                    "active_states": {translated_door_active_state},
                },
                "window": {
                    "prob_given_true": WINDOW_PROB_GIVEN_TRUE,
                    "prob_given_false": WINDOW_PROB_GIVEN_FALSE,
                    "default_prior": WINDOW_DEFAULT_PRIOR,
                    "weight": self._sensor_weights["window"],
                    "active_states": {translated_window_active_state},
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

            # Validate configurations
            for sensor_type, config in configs.items():
                _validate_config(sensor_type, config)

        except Exception as err:
            raise ConfigurationError(f"Failed to build sensor configs: {err}") from err
        else:
            return configs

    @property
    def sensor_weights(self) -> dict[str, float]:
        """Get the current sensor weights."""
        return self._sensor_weights

    @property
    def sensor_configs(self) -> dict[str, ProbabilityConfig]:
        """Get the current sensor configurations."""
        return self._sensor_configs

    def get_default_prior(self, entity_id: str) -> float:
        """Get the default prior probability for an entity.

        Args:
            entity_id: The entity ID to get the prior for

        Returns:
            The default prior probability

        Raises:
            ValueError: If entity_id is not found

        """
        if entity_id not in self.entity_types:
            raise ValueError(f"Entity {entity_id} not found in entity types")

        sensor_type = self.entity_types[entity_id]
        if sensor_type not in self._sensor_configs:
            raise ValueError(
                f"Invalid sensor type {sensor_type} for entity {entity_id}"
            )

        return self._sensor_configs[sensor_type]["default_prior"]

    def update_config(self, config: dict[str, Any]) -> None:
        """Update the configuration.

        Args:
            config: New configuration dictionary

        Raises:
            ConfigurationError: If new configuration is invalid

        """
        try:
            self.config = config
            self._sensor_weights = self._get_sensor_weights()
            self._sensor_configs = self._build_sensor_configs()
            self.entity_types.clear()
            self._map_entities_to_types()

            _LOGGER.debug("Configuration updated successfully")
        except Exception as err:
            raise ConfigurationError(f"Failed to update configuration: {err}") from err

    def get_sensor_config(self, entity_id: str) -> ProbabilityConfig | None:
        """Get the configuration for a specific sensor.

        Args:
            entity_id: The entity ID to get the config for

        Returns:
            The sensor configuration or None if not found

        Raises:
            ValueError: If entity_id is not found

        """
        if entity_id not in self.entity_types:
            raise ValueError(f"Entity {entity_id} not found in entity types")

        sensor_type = self.entity_types[entity_id]
        if sensor_type not in self._sensor_configs:
            raise ValueError(
                f"Invalid sensor type {sensor_type} for entity {entity_id}"
            )

        return self._sensor_configs[sensor_type]

    def is_entity_active(
        self,
        entity_id: str,
        state: str,
    ) -> bool:
        """Check if an entity is in an active state.

        Args:
            entity_id: The entity ID to check
            state: The current state to check

        Returns:
            True if the entity is active, False otherwise

        Raises:
            ValueError: If entity_id is not found

        """
        if entity_id not in self.entity_types:
            _LOGGER.error("Entity %s not found in entity types mapping", entity_id)
            raise ValueError(f"Entity {entity_id} not found in entity types")

        sensor_type = self.entity_types[entity_id]
        if sensor_type not in self._sensor_configs:
            _LOGGER.error(
                "Invalid sensor type %s for entity %s in sensor configs",
                sensor_type,
                entity_id,
            )
            raise ValueError(
                f"Invalid sensor type {sensor_type} for entity {entity_id}"
            )

        active_states = self._sensor_configs[sensor_type]["active_states"]
        return state in active_states

    def get_initial_type_priors(self) -> dict[str, ProbabilityConfig]:
        """Get the initial type priors for all sensor types.

        Returns:
            Dictionary of sensor type to initial prior mapping

        """
        try:
            priors = {}
            for sensor_type, config in self._sensor_configs.items():
                priors[sensor_type] = {
                    "prob_given_true": config["prob_given_true"],
                    "prob_given_false": config["prob_given_false"],
                    "prior": config["default_prior"],
                }

        except (KeyError, ValueError, TypeError):
            _LOGGER.exception("Failed to get initial type priors: %s")
            return {}
        else:
            return priors

    def get_entity_type(self, entity_id: str) -> EntityType | None:
        """Get the type of an entity based on configuration.

        Args:
            entity_id: The entity ID to check

        Returns:
            The entity type or None if not found

        """
        if entity_id in self.entity_types:
            return self.entity_types[entity_id]
        return None
