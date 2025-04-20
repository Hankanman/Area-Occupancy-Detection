"""Handles probability configurations, weights, and state mappings for Area Occupancy Detection.

This module defines default probability values (priors, likelihoods) for different
sensor types, retrieves configured weights, builds the final probability
configuration used by calculators, and maps entity IDs to their types.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from homeassistant.const import STATE_OFF, STATE_ON

from .const import (
    APPLIANCE_DEFAULT_PRIOR,
    APPLIANCE_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
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
    CONF_WASP_IN_BOX_ENABLED,
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
    DEFAULT_WASP_IN_BOX_ENABLED,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    DOOR_DEFAULT_PRIOR,
    DOOR_PROB_GIVEN_FALSE,
    DOOR_PROB_GIVEN_TRUE,
    ENVIRONMENTAL_DEFAULT_PRIOR,
    ENVIRONMENTAL_PROB_GIVEN_FALSE,
    ENVIRONMENTAL_PROB_GIVEN_TRUE,
    LIGHT_DEFAULT_PRIOR,
    LIGHT_PROB_GIVEN_FALSE,
    LIGHT_PROB_GIVEN_TRUE,
    MEDIA_DEFAULT_PRIOR,
    MEDIA_PROB_GIVEN_FALSE,
    MEDIA_PROB_GIVEN_TRUE,
    MOTION_DEFAULT_PRIOR,
    MOTION_PROB_GIVEN_FALSE,
    MOTION_PROB_GIVEN_TRUE,
    WASP_DEFAULT_PRIOR,
    WASP_PROB_GIVEN_FALSE,
    WASP_PROB_GIVEN_TRUE,
    WASP_WEIGHT,
    WINDOW_DEFAULT_PRIOR,
    WINDOW_PROB_GIVEN_FALSE,
    WINDOW_PROB_GIVEN_TRUE,
)
from .exceptions import ConfigurationError
from .types import EntityType, PriorData, ProbabilityConfig

_LOGGER = logging.getLogger(__name__)

# Environmental detection baseline settings
ENVIRONMENTAL_BASELINE_PERCENT: Final[float] = 0.05  # 5% deviation allowed around mean
ENVIRONMENTAL_MIN_ACTIVE_DURATION: Final[int] = 300  # seconds of active data needed

# Minimum active duration for storing learned priors
MIN_ACTIVE_DURATION_FOR_PRIORS: Final[int] = 300

# Baseline cache TTL (to avoid hitting DB repeatedly)
BASELINE_CACHE_TTL: Final[int] = 21600  # 6 hours in seconds

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
    """Manages sensor type configurations including probabilities, weights, and active states.

    This class loads configuration values, applies defaults, validates inputs,
    and provides access to the processed probability settings for each sensor type
    and individual entity.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the probabilities handler with the provided configuration.

        Args:
            config: The configuration dictionary for the area occupancy instance.

        Raises:
            ConfigurationError: If the provided configuration is invalid (e.g.,
                incorrect types, invalid weights, missing required keys).

        """
        self.config = config
        self._sensor_weights = self._get_sensor_weights()
        self.entity_types: dict[str, EntityType] = {}
        self._map_entities_to_types()
        # Build configs *after* mapping types
        self._sensor_configs = self._build_sensor_configs()

        _LOGGER.debug(
            "Probabilities initialized with %d entity types and %d sensor configs",
            len(self.entity_types),
            len(self._sensor_configs),
        )

    def _map_entities_to_types(self) -> None:
        """Create a mapping of entity IDs to their corresponding sensor types based on configuration.

        Iterates through the configured sensor lists (motion, media, etc.) and populates
        the `self.entity_types` dictionary.

        Raises:
            ConfigurationError: If mapping fails due to invalid configuration
                (e.g., non-list value for a sensor type, invalid entity ID format).

        """

        def _validate_entity_id(entity_id: str, config_key: str) -> None:
            """Validate a single entity ID format."""
            if not entity_id or not isinstance(entity_id, str) or "." not in entity_id:
                # Raise directly
                raise ConfigurationError(
                    f"Invalid entity ID '{entity_id}' found in {config_key}"
                )

        try:
            mappings: list[tuple[str, EntityType]] = [
                (CONF_MOTION_SENSORS, EntityType.MOTION),
                (CONF_MEDIA_DEVICES, EntityType.MEDIA),
                (CONF_APPLIANCES, EntityType.APPLIANCE),
                (CONF_DOOR_SENSORS, EntityType.DOOR),
                (CONF_WINDOW_SENSORS, EntityType.WINDOW),
                (CONF_LIGHTS, EntityType.LIGHT),
                (CONF_ILLUMINANCE_SENSORS, EntityType.ENVIRONMENTAL),
                (CONF_HUMIDITY_SENSORS, EntityType.ENVIRONMENTAL),
                (CONF_TEMPERATURE_SENSORS, EntityType.ENVIRONMENTAL),
            ]

            # Clear existing mappings first
            self.entity_types.clear()

            for config_key, sensor_type in mappings:
                entities = self.config.get(config_key, [])
                if not isinstance(entities, list):
                    # Raise directly
                    raise ConfigurationError(
                        f"Configuration for {config_key} must be a list, got {type(entities)}"
                    )
                for entity_id in entities:
                    _validate_entity_id(entity_id, config_key)
                    self.entity_types[entity_id] = sensor_type

            # Add wasp virtual sensor if enabled
            if self.config.get(CONF_WASP_IN_BOX_ENABLED, DEFAULT_WASP_IN_BOX_ENABLED):
                self.entity_types["wasp.virtual"] = EntityType.WASP

        except (TypeError, ValueError) as err:
            # Catch potential unexpected type or value errors during entity processing
            # Raise directly, chaining the original exception
            raise ConfigurationError(
                f"Unexpected error mapping entities to types: {err}"
            ) from err
        # Note: ConfigurationError raised by _validate_entity_id will propagate naturally.

    def _get_sensor_weights(self) -> dict[str, float]:
        """Retrieve and validate sensor weights from the configuration, using defaults if necessary.

        Returns:
            A dictionary mapping sensor types ('motion', 'media', etc.) to their weights (0.0-1.0).

        Raises:
            ConfigurationError: If any configured weight is outside the valid range [0, 1]
                or if the configuration format is incorrect.

        """

        def _validate_weight(sensor_type: str, weight: float) -> None:
            """Validate a single weight value."""
            if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                raise ConfigurationError(
                    f"Invalid weight for {sensor_type}: {weight}. Must be a number between 0 and 1."
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

            # Add wasp weight if enabled
            if self.config.get(CONF_WASP_IN_BOX_ENABLED, DEFAULT_WASP_IN_BOX_ENABLED):
                weights["wasp"] = WASP_WEIGHT

            # Validate weights
            for sensor_type, weight in weights.items():
                _validate_weight(sensor_type, weight)

        except Exception as err:
            # Catch potential errors during config access or validation
            raise ConfigurationError(f"Failed to get sensor weights: {err}") from err
        else:
            return weights

    def _translate_binary_sensor_active_state(self, configured_state: str) -> str:
        """Translate a configured display state (e.g., 'Open', 'Detected', 'Low') to the corresponding internal binary_sensor state ('on', 'off') that represents this user-defined active condition.

        Args:
            configured_state: The user-configured state string (e.g., 'Open').

        Returns:
            The internal Home Assistant state ('on' or 'off') corresponding
            to the configured active display state.

        """
        # Lowercase for case-insensitive comparison
        configured_state_lower = configured_state.lower()

        # Define common display states that typically represent the 'on' (active/problem) internal state
        on_state_display_values = {
            # Common Active States
            "open",
            "detected",
            "occupied",
            "home",  # presence
            "wet",  # moisture
            "moving",
            "running",
            "connected",
            "charging",  # battery_charging
            "plugged in",
            "on",  # Generic fallback / some integrations might use this display value
            # Common Problem/Alert States
            "low",  # battery
            "cold",
            "hot",  # heat
            "unlocked",  # lock (ON = unsafe/unlocked)
            "problem",
            "unsafe",  # safety (ON = unsafe)
            "update available",
        }

        if configured_state_lower in on_state_display_values:
            return STATE_ON
        # Assume all other configured states (like "Closed", "Clear", "Normal", "Locked", "Safe", "OK", etc.)
        # mean the user wants the internal STATE_OFF to be the trigger condition.
        return STATE_OFF

    def _build_sensor_configs(self) -> dict[str, ProbabilityConfig]:
        """Build the final sensor configurations dictionary used for calculations.

        Combines default probabilities, configured weights, and processed active
        state configurations for each sensor type. Uses a dynamic approach based
        on `type_specifics` metadata.

        Returns:
            A dictionary where keys are sensor types and values are
            `ProbabilityConfig` dictionaries containing 'prob_given_true',
            'prob_given_false', 'default_prior', 'weight', and 'active_states'.

        Raises:
            ConfigurationError: If building the configurations fails due to invalid
                settings (e.g., incorrect types, invalid probabilities, missing keys).

        """

        # --- Helper function for validation (as before) ---
        def _validate_probability(
            sensor_type: str, prob_name: str, value: float
        ) -> None:
            """Validate a single probability value."""
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ConfigurationError(
                    f"Invalid {prob_name} for {sensor_type}: {value}. Must be a number between 0 and 1."
                )

        def _validate_config(sensor_type: str, config: dict) -> None:
            """Validate the core probability and weight values for a sensor type config."""
            for key in [
                "prob_given_true",
                "prob_given_false",
                "default_prior",
                "weight",
            ]:
                if key not in config:
                    raise ConfigurationError(
                        f"Missing key '{key}' in config for sensor type '{sensor_type}'"
                    )
                _validate_probability(sensor_type, key, config[key])

        # --- End Helper functions ---

        try:
            # --- Define Sensor Type Specifics ---
            # Map sensor types to their specific configuration keys and default values
            type_specifics = {
                "motion": {
                    "prob_true": MOTION_PROB_GIVEN_TRUE,
                    "prob_false": MOTION_PROB_GIVEN_FALSE,
                    "prior": MOTION_DEFAULT_PRIOR,
                    "active_states": {STATE_ON},
                },
                "media": {
                    "prob_true": MEDIA_PROB_GIVEN_TRUE,
                    "prob_false": MEDIA_PROB_GIVEN_FALSE,
                    "prior": MEDIA_DEFAULT_PRIOR,
                    "active_states_key": CONF_MEDIA_ACTIVE_STATES,
                    "active_states_default": DEFAULT_MEDIA_ACTIVE_STATES,
                    "multiple_states": True,
                },
                "appliance": {
                    "prob_true": APPLIANCE_PROB_GIVEN_TRUE,
                    "prob_false": APPLIANCE_PROB_GIVEN_FALSE,
                    "prior": APPLIANCE_DEFAULT_PRIOR,
                    "active_states_key": CONF_APPLIANCE_ACTIVE_STATES,
                    "active_states_default": DEFAULT_APPLIANCE_ACTIVE_STATES,
                    "multiple_states": True,
                },
                "door": {
                    "prob_true": DOOR_PROB_GIVEN_TRUE,
                    "prob_false": DOOR_PROB_GIVEN_FALSE,
                    "prior": DOOR_DEFAULT_PRIOR,
                    "active_states_key": CONF_DOOR_ACTIVE_STATE,
                    "active_states_default": DEFAULT_DOOR_ACTIVE_STATE,
                    "translate_state": True,
                },
                "window": {
                    "prob_true": WINDOW_PROB_GIVEN_TRUE,
                    "prob_false": WINDOW_PROB_GIVEN_FALSE,
                    "prior": WINDOW_DEFAULT_PRIOR,
                    "active_states_key": CONF_WINDOW_ACTIVE_STATE,
                    "active_states_default": DEFAULT_WINDOW_ACTIVE_STATE,
                    "translate_state": True,
                },
                "light": {
                    "prob_true": LIGHT_PROB_GIVEN_TRUE,
                    "prob_false": LIGHT_PROB_GIVEN_FALSE,
                    "prior": LIGHT_DEFAULT_PRIOR,
                    "active_states": {STATE_ON},
                },
                "environmental": {
                    "prob_true": ENVIRONMENTAL_PROB_GIVEN_TRUE,
                    "prob_false": ENVIRONMENTAL_PROB_GIVEN_FALSE,
                    "prior": ENVIRONMENTAL_DEFAULT_PRIOR,
                    "active_states": {STATE_ON},
                },  # Assuming environmental is active when 'on' (needs data)
            }
            # Add wasp config if enabled
            if self.config.get(CONF_WASP_IN_BOX_ENABLED, DEFAULT_WASP_IN_BOX_ENABLED):
                type_specifics["wasp"] = {
                    "prob_true": WASP_PROB_GIVEN_TRUE,
                    "prob_false": WASP_PROB_GIVEN_FALSE,
                    "prior": WASP_DEFAULT_PRIOR,
                    "active_states": {"on"},
                }
            # --- End Sensor Type Specifics ---

            configs = {}
            for sensor_type, specifics in type_specifics.items():
                # Determine active states dynamically
                active_states_set = set()
                if "active_states" in specifics:
                    # Hardcoded states (motion, light, environmental)
                    active_states_set = specifics["active_states"]
                elif "active_states_key" in specifics:
                    # Configurable states (media, appliance, door, window)
                    key = specifics["active_states_key"]
                    default = specifics["active_states_default"]
                    is_multiple = specifics.get("multiple_states", False)
                    needs_translation = specifics.get("translate_state", False)

                    configured_value = self.config.get(key, default)

                    if is_multiple:
                        # Handle multiple states (media, appliance)
                        if not isinstance(configured_value, list):
                            _LOGGER.warning(
                                "Expected list for %s, got %s. Using default: %s",
                                key,
                                type(configured_value),
                                default,
                            )
                            configured_value = default
                        active_states_set = set(configured_value)
                    else:
                        # Handle single state (door, window)
                        # Ensure configured_value is a string before processing/translation
                        if not isinstance(configured_value, str):
                            _LOGGER.warning(
                                "Expected string for %s, got %s. Using default: %s",
                                key,
                                type(configured_value),
                                default,
                            )
                            configured_value = str(
                                default
                            )  # Convert default to string as fallback

                        state_to_process = configured_value
                        if needs_translation:
                            state_to_process = (
                                self._translate_binary_sensor_active_state(
                                    state_to_process
                                )
                            )
                        active_states_set = {state_to_process}
                else:
                    # Should not happen if type_specifics is defined correctly
                    _LOGGER.error(
                        "Missing active state definition for sensor type: %s",
                        sensor_type,
                    )
                    active_states_set = {STATE_ON}  # Default fallback

                # Build the config dictionary
                current_config: ProbabilityConfig = {
                    "prob_given_true": specifics["prob_true"],
                    "prob_given_false": specifics["prob_false"],
                    "default_prior": specifics["prior"],
                    "weight": self._sensor_weights[sensor_type],
                    "active_states": active_states_set,
                }
                configs[sensor_type] = current_config

            # Validate configurations
            for sensor_type, config_data in configs.items():
                _validate_config(sensor_type, config_data)

        except Exception as err:
            # Catch potential errors during config access, processing, or validation
            raise ConfigurationError(f"Failed to build sensor configs: {err}") from err
        else:
            return configs

    @property
    def sensor_weights(self) -> dict[str, float]:
        """Return the dictionary of configured sensor weights by type."""
        return self._sensor_weights

    @property
    def sensor_configs(self) -> dict[str, ProbabilityConfig]:
        """Return the dictionary of final sensor configurations by type."""
        return self._sensor_configs

    def get_default_prior(self, entity_id: str) -> float:
        """Get the default prior probability defined for an entity's type.

        Args:
            entity_id: The entity ID to get the default prior for.

        Returns:
            The default prior probability (0.0-1.0) for the entity's type.

        Raises:
            ValueError: If the entity_id is not mapped to a known sensor type
                or if the sensor type configuration is missing.

        """
        if entity_id not in self.entity_types:
            raise ValueError(f"Entity '{entity_id}' not found in entity types mapping.")

        sensor_type = self.entity_types[entity_id]
        sensor_config = self._sensor_configs.get(sensor_type)

        if not sensor_config:
            # This should ideally not happen if initialization is correct
            raise ValueError(
                f"Configuration missing for sensor type '{sensor_type}' derived from entity '{entity_id}'."
            )

        # Ensure 'default_prior' key exists, though it should based on _build_sensor_configs logic
        if "default_prior" not in sensor_config:
            raise ValueError(
                f"Key 'default_prior' missing in configuration for sensor type '{sensor_type}'."
            )

        return sensor_config["default_prior"]

    def update_config(self, config: dict[str, Any]) -> None:
        """Update the internal configuration and rebuild derived settings.

        This should be called when the integration's configuration options change.

        Args:
            config: The new configuration dictionary.

        Raises:
            ConfigurationError: If the new configuration is invalid and causes errors
                during reprocessing (e.g., invalid weights, types).

        """
        try:
            self.config = config
            # Re-calculate weights, map entities, and build configs based on new config
            self._sensor_weights = self._get_sensor_weights()
            self.entity_types.clear()  # Clear before remapping
            self._map_entities_to_types()
            self._sensor_configs = self._build_sensor_configs()

            _LOGGER.debug("Probabilities configuration updated successfully")
        except Exception as err:
            # Catch errors during the reprocessing steps
            raise ConfigurationError(
                f"Failed to update probabilities configuration: {err}"
            ) from err

    def get_sensor_config(self, entity_id: str) -> ProbabilityConfig | None:
        """Get the final calculated configuration for a specific sensor entity.

        Args:
            entity_id: The entity ID to retrieve the configuration for.

        Returns:
            The `ProbabilityConfig` dictionary for the entity's type,
            or None if the entity or its type configuration is not found.

        Raises:
            ValueError: If the entity ID is not mapped to a known sensor type
                or if the type configuration is unexpectedly missing.

        """
        if entity_id not in self.entity_types:
            # Log potentially useful info instead of raising immediately? Or keep ValueError?
            _LOGGER.warning(
                "Entity '%s' not found in entity types mapping during config lookup",
                entity_id,
            )
            # Raise ValueError to signal a problem upstream that needs handling.
            raise ValueError(f"Entity '{entity_id}' not found in entity types mapping.")

        sensor_type = self.entity_types[entity_id]
        sensor_config = self._sensor_configs.get(sensor_type)

        if not sensor_config:
            # This indicates an internal inconsistency, likely during initialization or update.
            _LOGGER.error(
                "Internal Error: Configuration missing for sensor type '%s' derived from entity '%s'",
                sensor_type,
                entity_id,
            )
            raise ValueError(f"Configuration missing for sensor type '{sensor_type}'.")

        # Special handling for wasp.virtual
        if entity_id == "wasp.virtual":
            if self.config.get(CONF_WASP_IN_BOX_ENABLED, DEFAULT_WASP_IN_BOX_ENABLED):
                return self._sensor_configs.get("wasp")
            else:
                return None

        return sensor_config

    def is_entity_active(
        self,
        entity_id: str,
        state: str | None,  # Allow None state
    ) -> bool:
        """Check if the given state corresponds to an 'active' state for the entity's type.

        Args:
            entity_id: The entity ID to check.
            state: The current state value of the entity (can be None).

        Returns:
            True if the provided state is considered active for the entity's type,
            False otherwise (including if state is None or entity/config not found).

        Raises:
            ValueError: If the entity_id is not mapped to a known sensor type
                or if the sensor type configuration is missing.

        """
        if state is None:
            return False  # None state is never active

        sensor_config = self.get_sensor_config(entity_id)
        if not sensor_config:
            # get_sensor_config now raises ValueError if config missing
            # This path should ideally not be reached if exceptions are handled.
            _LOGGER.warning(
                "Could not find sensor config for '%s' to check active state",
                entity_id,
            )
            return False

        active_states = sensor_config.get("active_states")
        if not active_states:
            _LOGGER.warning(
                "No active states defined for sensor type of '%s'", entity_id
            )
            return False  # Treat as inactive if no active states defined

        # Special handling for wasp.virtual
        if entity_id == "wasp.virtual":
            return state == "on"

        return state in active_states

    def get_initial_type_priors(self) -> dict[str, PriorData]:
        """Get the default 'prob_given_true', 'prob_given_false', and 'prior' for all sensor types.

        Used for initializing the PriorState if no learned data exists.

        Returns:
            A dictionary where keys are sensor types and values are PriorData objects
            containing 'prob_given_true', 'prob_given_false', and 'prior'.

        Note: This uses PriorData structure for consistency, but values are defaults.

        """
        try:
            priors = {}
            for sensor_type, config in self._sensor_configs.items():
                # Ensure all required keys are present before adding
                if all(
                    k in config
                    for k in ("prob_given_true", "prob_given_false", "default_prior")
                ):
                    priors[sensor_type] = PriorData(
                        prob_given_true=config["prob_given_true"],
                        prob_given_false=config["prob_given_false"],
                        prior=config["default_prior"],
                        last_updated=None,  # Default priors have no update time initially
                    )
                else:
                    _LOGGER.warning(
                        "Skipping initial priors for type '%s' due to missing keys in config: %s",
                        sensor_type,
                        config,
                    )

            # Add wasp prior if enabled and not present
            if self.config.get(CONF_WASP_IN_BOX_ENABLED, DEFAULT_WASP_IN_BOX_ENABLED):
                if "wasp" not in priors:
                    priors["wasp"] = PriorData(
                        prob_given_true=WASP_PROB_GIVEN_TRUE,
                        prob_given_false=WASP_PROB_GIVEN_FALSE,
                        prior=WASP_DEFAULT_PRIOR,
                        last_updated=None,
                    )

        except (KeyError, ValueError, TypeError):
            _LOGGER.exception(
                "Failed to get initial type priors"
            )  # Keep general exception log
            return {}
        else:
            return priors

    def get_entity_type(self, entity_id: str) -> EntityType | None:
        """Get the configured sensor type for a given entity ID.

        Args:
            entity_id: The entity ID to look up.

        Returns:
            The sensor type (e.g., 'motion', 'media') as a string,
            or None if the entity ID is not found in the mapping.

        """
        return self.entity_types.get(entity_id)
