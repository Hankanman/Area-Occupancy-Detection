"""Handles probability configurations, weights, and state mappings for Area Occupancy Detection.

This module defines default probability values (priors, likelihoods) for different
sensor types, retrieves configured weights, builds the final probability
configuration used by calculators, and maps entity IDs to their types.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Final

from homeassistant.const import STATE_OFF, STATE_ON

from .const import (
    APPLIANCE_DEFAULT_PRIOR,
    APPLIANCE_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_DOOR_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_WASP_WEIGHT,
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
    """Manages sensor type configurations including probabilities and weights.

    This class loads configuration values, applies defaults, validates inputs,
    and provides access to the processed probability settings for each sensor type.

    Entity management (registry, types, active states) is handled by EntityManager.
    This class focuses purely on probability configuration and sensor type definitions.
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
        # Build configs *after* getting weights
        self._sensor_configs = self._build_sensor_configs()

        _LOGGER.debug(
            "Probabilities initialized with %d sensor configs",
            len(self._sensor_configs),
        )

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
                "wasp_in_box": self.config.get(CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT),
            }

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
                "wasp_in_box": {
                    "prob_true": WASP_PROB_GIVEN_TRUE,
                    "prob_false": WASP_PROB_GIVEN_FALSE,
                    "prior": WASP_DEFAULT_PRIOR,
                    "weight_key": CONF_WASP_WEIGHT,
                    "active_states": {STATE_ON},
                },
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

                # Determine weight dynamically based on key or default
                weight_key = specifics.get("weight_key")
                weight_value = (
                    self.config.get(
                        weight_key, self._sensor_weights.get(sensor_type)
                    )  # Try config first
                    if weight_key
                    else self._sensor_weights.get(
                        sensor_type
                    )  # Fallback to pre-calculated weight
                )
                if weight_value is None:
                    # Use a default if still None (e.g., for Wasp if key missing somehow)
                    weight_value = self._sensor_weights.get(
                        sensor_type, 0.5
                    )  # Fallback to 0.5
                    _LOGGER.warning(
                        "Could not determine weight for sensor type '%s' using key '%s' or default. Using fallback %.2f",
                        sensor_type,
                        weight_key,
                        weight_value,
                    )

                # Build the config dictionary
                current_config: ProbabilityConfig = {
                    "prob_given_true": specifics["prob_true"],
                    "prob_given_false": specifics["prob_false"],
                    "default_prior": specifics["prior"],
                    "weight": float(weight_value),  # Ensure weight is float
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

    def get_default_prior(self, sensor_type: str | EntityType) -> float:
        """Get the default prior probability defined for a sensor type.

        Args:
            sensor_type: The sensor type (string or EntityType) to get the default prior for.

        Returns:
            The default prior probability (0.0-1.0) for the sensor type.

        Raises:
            ValueError: If the sensor type configuration is missing.

        """
        if isinstance(sensor_type, EntityType):
            sensor_type = sensor_type.value

        sensor_config = self._sensor_configs.get(sensor_type)

        if not sensor_config:
            raise ValueError(f"Configuration missing for sensor type '{sensor_type}'.")

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
            # Re-calculate weights and build configs based on new config
            self._sensor_weights = self._get_sensor_weights()
            self._sensor_configs = self._build_sensor_configs()

            _LOGGER.debug("Probabilities configuration updated successfully")
        except Exception as err:
            # Catch errors during the reprocessing steps
            raise ConfigurationError(
                f"Failed to update probabilities configuration: {err}"
            ) from err

    def get_sensor_config_by_type(
        self, sensor_type: str | EntityType
    ) -> ProbabilityConfig | None:
        """Get the final calculated configuration for a specific sensor type.

        Args:
            sensor_type: The sensor type (string or EntityType) to retrieve the configuration for.

        Returns:
            The `ProbabilityConfig` dictionary for the sensor type,
            or None if the type configuration is not found.

        """
        if isinstance(sensor_type, EntityType):
            sensor_type = sensor_type.value

        return self._sensor_configs.get(sensor_type)

    @lru_cache(maxsize=256)
    def _get_cached_sensor_config(self, entity_type: str) -> ProbabilityConfig | None:
        """Get sensor config with LRU caching for better performance."""
        return self._sensor_configs.get(entity_type)

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

        except (KeyError, ValueError, TypeError):
            _LOGGER.exception(
                "Failed to get initial type priors"
            )  # Keep general exception log
            return {}
        else:
            return priors
