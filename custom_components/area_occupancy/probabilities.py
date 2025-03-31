"""Probability constants and defaults for Area Occupancy Detection."""

from __future__ import annotations
import logging
from typing import Final, Dict, Any, Optional

from homeassistant.const import (
    STATE_ON,
)

from .types import EntityType, LearnedPrior, SensorConfig
from .exceptions import ConfigurationError

from .const import (
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_WINDOW,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_DOOR_ACTIVE_STATE,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_APPLIANCE_ACTIVE_STATES,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
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

    def __init__(
        self, config: dict[str, Any], coordinator: Optional[Any] = None
    ) -> None:
        """Initialize the probabilities handler.

        Args:
            config: Configuration dictionary
            coordinator: Optional coordinator instance for learned priors

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        self.coordinator = coordinator
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

            for config_key, sensor_type in mappings:
                for entity_id in self.config.get(config_key, []):
                    if not entity_id:
                        raise ConfigurationError(f"Empty entity ID in {config_key}")
                    self.entity_types[entity_id] = sensor_type

            _LOGGER.debug(
                "Mapped %d entities to types: %s",
                len(self.entity_types),
                {k: v for k, v in self.entity_types.items()},
            )
        except Exception as err:
            raise ConfigurationError(f"Failed to map entities to types: {err}") from err

    def _get_sensor_weights(self) -> dict[str, float]:
        """Get the configured sensor weights, falling back to defaults if not configured.

        Returns:
            Dictionary of sensor type to weight mapping

        Raises:
            ConfigurationError: If weights are invalid
        """
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
                if not 0 <= weight <= 1:
                    raise ConfigurationError(
                        f"Invalid weight for {sensor_type}: {weight}. Must be between 0 and 1."
                    )

            _LOGGER.debug("Sensor weights configured: %s", weights)
            return weights
        except Exception as err:
            raise ConfigurationError(f"Failed to get sensor weights: {err}") from err

    def _build_sensor_configs(self) -> dict[str, SensorConfig]:
        """Build sensor configurations using current weights and type priors.

        Returns:
            Dictionary of sensor configurations

        Raises:
            ConfigurationError: If sensor configurations are invalid
        """
        try:
            # Get the configured door active state
            door_active_state = self.config.get(
                CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE
            )

            # Get the configured window active state
            window_active_state = self.config.get(
                CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE
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
                    "active_states": {door_active_state},
                },
                "window": {
                    "prob_given_true": WINDOW_PROB_GIVEN_TRUE,
                    "prob_given_false": WINDOW_PROB_GIVEN_FALSE,
                    "default_prior": WINDOW_DEFAULT_PRIOR,
                    "weight": self._sensor_weights["window"],
                    "active_states": {window_active_state},
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

            # Update configs with learned type priors if available
            if self.coordinator and hasattr(self.coordinator, "type_priors"):
                for sensor_type, config in configs.items():
                    if type_prior := self.coordinator.type_priors.get(sensor_type):
                        config.update(
                            {
                                "prob_given_true": type_prior["prob_given_true"],
                                "prob_given_false": type_prior["prob_given_false"],
                                "default_prior": type_prior["prior"],
                            }
                        )

            # Validate configurations
            for sensor_type, config in configs.items():
                if not 0 <= config["prob_given_true"] <= 1:
                    raise ConfigurationError(
                        f"Invalid prob_given_true for {sensor_type}: {config['prob_given_true']}"
                    )
                if not 0 <= config["prob_given_false"] <= 1:
                    raise ConfigurationError(
                        f"Invalid prob_given_false for {sensor_type}: {config['prob_given_false']}"
                    )
                if not 0 <= config["default_prior"] <= 1:
                    raise ConfigurationError(
                        f"Invalid default_prior for {sensor_type}: {config['default_prior']}"
                    )
                if not 0 <= config["weight"] <= 1:
                    raise ConfigurationError(
                        f"Invalid weight for {sensor_type}: {config['weight']}"
                    )

            _LOGGER.debug("Built sensor configurations: %s", configs)
            return configs
        except Exception as err:
            raise ConfigurationError(f"Failed to build sensor configs: {err}") from err

    @property
    def sensor_weights(self) -> dict[str, float]:
        """Get the current sensor weights."""
        return self._sensor_weights

    @property
    def sensor_configs(self) -> dict[str, SensorConfig]:
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

    def get_sensor_config(self, entity_id: str) -> Optional[SensorConfig]:
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
            raise ValueError(f"Entity {entity_id} not found in entity types")

        sensor_type = self.entity_types[entity_id]
        if sensor_type not in self._sensor_configs:
            raise ValueError(
                f"Invalid sensor type {sensor_type} for entity {entity_id}"
            )

        active_states = self._sensor_configs[sensor_type]["active_states"]
        is_active = state in active_states

        return is_active

    def get_initial_type_priors(self) -> dict[str, LearnedPrior]:
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

            _LOGGER.debug("Initial type priors: %s", priors)
            return priors
        except (KeyError, ValueError, TypeError) as err:
            _LOGGER.error("Failed to get initial type priors: %s", err, exc_info=True)
            return {}
