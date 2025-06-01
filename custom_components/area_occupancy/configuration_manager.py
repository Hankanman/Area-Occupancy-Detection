"""Manage configuration updates and component lifecycle for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import Callable

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryError,
    ConfigEntryNotReady,
    HomeAssistantError,
)

from .calculate_prior import PriorCalculator
from .const import (
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_THRESHOLD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_THRESHOLD,
    DOMAIN,
    STORAGE_KEY,
)
from .decay_manager import DecayManager
from .exceptions import ConfigurationError
from .prior_manager import PriorManager
from .probabilities import Probabilities
from .state_manager import StateManager
from .types import SensorInputs

_LOGGER = logging.getLogger(__name__)


class ConfigurationManager:
    """Manage configuration updates and component lifecycle.

    This class handles all configuration-related operations including:
    - Configuration initialization and validation
    - Options updates
    - Threshold updates
    - Component reinitialization
    - Configuration validation
    """

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        state_manager: StateManager,
        prior_manager: PriorManager,
        decay_manager: DecayManager,
        probabilities: Probabilities,
        update_callback: Callable[[], None],
        prior_calculator: PriorCalculator,
    ) -> None:
        """Initialize the configuration manager.

        Args:
            hass: Home Assistant instance
            config_entry: Config entry for this instance
            state_manager: State manager instance
            prior_manager: Prior manager instance
            decay_manager: Decay manager instance
            probabilities: Probabilities configuration handler
            update_callback: Callback to trigger updates
            prior_calculator: Prior calculator instance

        """
        self.hass = hass
        self.config_entry = config_entry
        self.state_manager = state_manager
        self.prior_manager = prior_manager
        self.decay_manager = decay_manager
        self.probabilities = probabilities
        self._update_callback = update_callback
        self.prior_calculator = prior_calculator

        # Initialize configuration
        self.config = {**config_entry.data, **config_entry.options}

        # Validate initial configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the current configuration.

        Raises:
            ConfigurationError: If configuration is invalid

        """
        try:
            # Validate sensor inputs
            SensorInputs.from_config(self.config)

            # Validate decay configuration
            decay_window = self.config.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)
            if not isinstance(decay_window, (int, float)):
                raise ConfigurationError(
                    f"Decay window must be a number, got {type(decay_window)}"
                )

            # Validate threshold
            threshold = self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
            if not isinstance(threshold, (int, float)):
                raise ConfigurationError(
                    f"Threshold must be a number, got {type(threshold)}"
                )
            if not 1 <= threshold <= 99:
                raise ConfigurationError(
                    f"Threshold must be between 1 and 99, got {threshold}"
                )

        except (ValueError, TypeError) as err:
            raise ConfigurationError(f"Invalid configuration: {err}") from err

    async def update_options(self) -> None:
        """Update configuration options and reinitialize components.

        Raises:
            ConfigEntryError: If configuration is invalid
            ConfigEntryNotReady: If component reinitialization fails

        """
        try:
            _LOGGER.debug(
                "ConfigurationManager update_options starting with config: %s",
                self.config,
            )

            # Shutdown the old PriorManager to clean up timers
            if self.prior_manager:
                await self.prior_manager.async_shutdown()

            # Update configuration first
            self.config = {**self.config_entry.data, **self.config_entry.options}
            _LOGGER.debug("Updated config: %s", self.config)

            # Validate new configuration
            self._validate_config()

            # Reinitialize components with new configuration
            await self._reinitialize_components()

            _LOGGER.info(
                "Configuration options successfully updated and components reinitialized"
            )

        except (ValueError, KeyError) as err:
            _LOGGER.error("Invalid configuration in update_options: %s", err)
            raise ConfigEntryError(f"Invalid configuration: {err}") from err
        except HomeAssistantError as err:
            _LOGGER.error("Failed to update configuration options: %s", err)
            raise ConfigEntryNotReady(
                f"Failed to update configuration options: {err}"
            ) from err

    async def update_threshold(self, value: float) -> None:
        """Update the threshold value.

        Args:
            value: The new threshold value as a percentage (1-99)

        Raises:
            HomeAssistantError: If there's an error updating the config entry

        """
        _LOGGER.debug("Updating threshold: %.2f", value)

        # Update config entry options
        new_options = dict(self.config_entry.options)
        new_options[CONF_THRESHOLD] = value

        try:
            # Only update the config entry, the listener will handle the rest
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                options=new_options,
            )

        except ValueError as err:
            raise HomeAssistantError(f"Failed to update threshold: {err}") from err
        except Exception as err:
            raise HomeAssistantError(f"Failed to update threshold: {err}") from err

    async def _reinitialize_components(self) -> None:
        """Reinitialize components with current configuration."""
        # Reset components that depend on config
        self.probabilities = Probabilities(config=self.config)

        # Update decay configuration
        self.config[CONF_DECAY_ENABLED] = self.config.get(
            CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED
        )
        self.config[CONF_DECAY_WINDOW] = self.config.get(
            CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
        )

        # Reinitialize decay manager
        self.decay_manager = DecayManager(
            hass=self.hass, config=self.config, update_callback=self._update_callback
        )

        # Reinitialize state manager
        self.state_manager = StateManager(
            hass=self.hass,
            config=self.config,
            probabilities=self.probabilities,
        )

        # Reinitialize prior manager
        self.prior_manager = PriorManager(
            hass=self.hass,
            config=self.config,
            storage=self.hass.data[DOMAIN][self.config_entry.entry_id][STORAGE_KEY],
            probabilities=self.probabilities,
            state_manager=self.state_manager,
            prior_calculator=self.prior_calculator,
            config_entry_id=self.config_entry.entry_id,
        )

        # Reinitialize prior manager
        self.probabilities = Probabilities(config=self.config)
