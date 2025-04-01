"""Area Occupancy Coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import (
    ConfigEntryError,
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .calculate_prior import PriorCalculator
from .calculate_prob import ProbabilityCalculator
from .const import (
    CONF_APPLIANCES,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DOOR_SENSORS,
    CONF_HISTORY_PERIOD,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_WINDOW_SENSORS,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_THRESHOLD,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)
from .decay_handler import DecayHandler
from .exceptions import (
    CalculationError,
    PriorCalculationError,
    StateError,
    StorageError,
)
from .probabilities import Probabilities
from .state_management import OccupancyStateManager
from .storage import AreaOccupancyStorage
from .types import DeviceInfo, PriorState, ProbabilityState

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityState]):
    """Manage fetching and combining data for area occupancy."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=config_entry.data.get(CONF_NAME, "Area Occupancy"),
            update_interval=timedelta(seconds=1),
            always_update=True,  # Always update listeners to ensure immediate response
        )

        # Initialize basic attributes
        self.config_entry = config_entry
        self.config = dict(config_entry.data)
        self._entity_ids: set[str] = set()
        self._state_lock = asyncio.Lock()
        self._storage_lock = asyncio.Lock()

        # Initialize timers and intervals
        self._prior_update_interval = timedelta(hours=1)
        self._save_interval = timedelta(seconds=10)
        self._last_save = dt_util.utcnow()

        # Initialize tracking
        self._prior_update_tracker = None
        self._remove_state_listener = None

        # Initialize state management
        self._state_cache: dict[str, dict] = {}
        self._state_manager = OccupancyStateManager()

        # Initialize data first
        self.data = ProbabilityState()
        self.data.update(
            probability=MIN_PROBABILITY,
            previous_probability=MIN_PROBABILITY,
            threshold=self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD) / 100.0,
            prior_probability=MIN_PROBABILITY,
            sensor_probabilities={},
            decay_status=0.0,
            current_states={},
            previous_states={},
            is_occupied=False,
        )

        # Initialize probabilities first
        self.probabilities = Probabilities(config=self.config)

        # Initialize prior state
        self.prior_state = PriorState()
        self.prior_state.initialize_from_defaults(self.probabilities)
        self.prior_state.update(
            analysis_period=self.config.get(
                CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
            ),
        )

        # Set decay configuration
        self.config[CONF_DECAY_ENABLED] = self.config.get(
            CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED
        )
        self.config[CONF_DECAY_WINDOW] = self.config.get(
            CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
        )

        # Initialize remaining components
        self.storage = AreaOccupancyStorage(self.hass, self.config_entry.entry_id)
        self.decay_handler = DecayHandler(self.config)
        self.calculator = ProbabilityCalculator(self, self.probabilities)
        self._prior_calculator = PriorCalculator(
            coordinator=self,
            probabilities=self.probabilities,
            hass=self.hass,
        )

    @property
    def entity_ids(self) -> set[str]:
        """Return the set of entity IDs being tracked."""
        return self._entity_ids

    @property
    def available(self) -> bool:
        """Check if the coordinator is available."""
        primary_sensor = self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor:
            return False
        # Use the cached state
        return self._state_cache.get(primary_sensor, {}).get("availability", False)

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for the coordinator."""
        return {
            "identifiers": {(DOMAIN, self.config_entry.entry_id)},
            "name": self.config[CONF_NAME],
            "manufacturer": DEVICE_MANUFACTURER,
            "model": DEVICE_MODEL,
            "sw_version": DEVICE_SW_VERSION,
        }

    @property
    def active_sensors_for_calc(self) -> dict[str, dict]:
        """Return active sensors from the synchronous cache."""
        return self._state_cache

    async def _async_setup(self) -> None:
        """Set up the coordinator and load stored data."""
        try:
            # Load stored data first
            await self.async_load_stored_data()

            # Initialize states after loading stored data using the new state manager
            sensors = self.get_configured_sensors()
            await self._state_manager.initialize_states(self.hass, sensors)
            self._state_cache = await self._state_manager.get_active_sensors()

            # Calculate initial priors before scheduling updates
            await self.update_learned_priors()

            # Schedule prior updates
            self.async_track_prior_updates()

            _LOGGER.info(
                "Successfully set up AreaOccupancyCoordinator for %s",
                self.config[CONF_NAME],
            )
        except (StorageError, StateError, CalculationError) as err:
            _LOGGER.error("Failed to set up coordinator: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up coordinator: {err}") from err

    @callback
    def _async_refresh_finished(self) -> None:
        """Handle when a refresh has finished with improved state management."""
        if self.last_update_success:
            # Save data if needed
            if (dt_util.utcnow() - self._last_save) >= self._save_interval:
                self.hass.async_create_task(self._save_debounced_data())

            # Update state cache
            if hasattr(self.data, "current_states"):
                self._state_cache = self.data.current_states.copy()

    @callback
    def async_set_updated_data(self, data: ProbabilityState) -> None:
        """Manually update data and notify listeners."""
        super().async_set_updated_data(data)
        # Additional state management specific to our use case
        if self.last_update_success:
            self._state_cache = data.current_states

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        await super().async_shutdown()
        # Additional cleanup specific to our use case
        if hasattr(self, "_prior_update_tracker"):
            self._prior_update_tracker()
            self._prior_update_tracker = None

        # Save final state
        await self._save_debounced_data()

        # Remove state listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Clear data
        self.data = None
        self.prior_state = None
        self._state_cache = {}
        self._entity_ids = set()
        self._last_save = dt_util.utcnow()

    async def async_load_stored_data(self) -> None:
        """Load and restore data from storage."""
        try:
            _, stored_prior_state = await self.storage.async_load_prior_state()
            if stored_prior_state:
                self.prior_state = stored_prior_state
            else:
                # Initialize from default priors if no stored prior state
                self.data.update(
                    probability=MIN_PROBABILITY,
                    previous_probability=MIN_PROBABILITY,
                    threshold=self.config.get("threshold", DEFAULT_THRESHOLD) / 100.0,
                    prior_probability=MIN_PROBABILITY,
                    sensor_probabilities={},
                    decay_status=0.0,
                    current_states={},
                    previous_states={},
                    is_occupied=False,
                )

                # Reset prior state
                self.prior_state = PriorState()
                self.prior_state.initialize_from_defaults(self.probabilities)
                self.prior_state.update(
                    analysis_period=self.config.get(
                        CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                    ),
                )

                self.decay_handler.reset()

            _LOGGER.debug("Successfully restored stored data")
        except Exception as err:
            raise StorageError(f"Failed to load stored data: {err}") from err

    async def async_update_options(self) -> None:
        """Update coordinator options with improved error handling."""
        try:
            self.config = {**self.config_entry.data, **self.config_entry.options}
            self.probabilities = Probabilities(config=self.config)
            self.calculator = ProbabilityCalculator(
                coordinator=self,
                probabilities=self.probabilities,
            )
            self._prior_calculator = PriorCalculator(
                coordinator=self,
                probabilities=self.probabilities,
                hass=self.hass,
            )

            # Update prior state analysis period
            self.prior_state.update(
                analysis_period=self.config.get(
                    CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                )
            )

            self._setup_entity_tracking()
            await self.async_initialize_states()
            await self.async_refresh()

        except (ValueError, KeyError) as err:
            raise ConfigEntryError(f"Invalid configuration: {err}") from err
        except HomeAssistantError as err:
            raise ConfigEntryNotReady(
                f"Failed to update coordinator options: {err}"
            ) from err

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes using state manager."""
        entities = self.get_configured_sensors()

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        @callback
        def async_state_changed_listener(event) -> None:
            try:
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")

                if not new_state or entity_id not in entities:
                    return

                # Store previous state before updating
                if entity_id in self.data.current_states:
                    self.data.previous_states[entity_id] = self.data.current_states[
                        entity_id
                    ]

                # Update current state
                self.data.current_states[entity_id] = {
                    "state": new_state.state,
                    "availability": True,
                }

                # Update sensor state in state manager
                self.hass.async_create_task(
                    self._state_manager.update_sensor(entity_id, new_state)
                )

                # Update local cache after state change
                async def update_cache():
                    try:
                        self._state_cache = (
                            await self._state_manager.get_active_sensors()
                        )
                    except (
                        TimeoutError,
                        ValueError,
                        AttributeError,
                        TypeError,
                        KeyError,
                    ) as err:
                        _LOGGER.error("Error updating state cache: %s", err)

                self.hass.async_create_task(update_cache())
                self.hass.async_create_task(self._debounced_refresh.async_call())

            except (
                TimeoutError,
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                HomeAssistantError,
                asyncio.CancelledError,
            ) as err:
                _LOGGER.error("Error in state change listener: %s", err)

        self._remove_state_listener = async_track_state_change_event(
            self.hass,
            entities,
            async_state_changed_listener,
        )

    async def async_initialize_states(self) -> None:
        """Initialize sensor states."""
        try:
            sensors = self.get_configured_sensors()
            await self._state_manager.initialize_states(self.hass, sensors)
            self._state_cache = await self._state_manager.get_active_sensors()
        except Exception as err:
            raise StateError(f"Failed to initialize states: {err}") from err

        self._setup_entity_tracking()

    def get_configured_sensors(self) -> list[str]:
        """Get all configured sensors including the primary occupancy sensor."""
        sensors = (
            self.config.get(CONF_MOTION_SENSORS, [])
            + self.config.get(CONF_MEDIA_DEVICES, [])
            + self.config.get(CONF_APPLIANCES, [])
            + self.config.get(CONF_ILLUMINANCE_SENSORS, [])
            + self.config.get(CONF_HUMIDITY_SENSORS, [])
            + self.config.get(CONF_TEMPERATURE_SENSORS, [])
            + self.config.get(CONF_DOOR_SENSORS, [])
            + self.config.get(CONF_WINDOW_SENSORS, [])
            + self.config.get(CONF_LIGHTS, [])
        )
        # Ensure primary sensor is included
        primary_sensor = self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if primary_sensor and primary_sensor not in sensors:
            sensors.append(primary_sensor)
        return sensors

    async def update_learned_priors(self, history_period: int | None = None) -> None:
        """Update learned priors using historical data."""
        try:
            period = history_period or self.config.get(
                CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
            )
            end_time = dt_util.utcnow()
            start_time = end_time - timedelta(days=period)

            # Update the analysis period in prior_state
            self.prior_state.update(analysis_period=period)

            # Get all sensors that need priors calculated
            sensors = self.get_configured_sensors()
            for sensor_id in sensors:
                if sensor_id != self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR):
                    try:
                        (
                            prob_given_true,
                            prob_given_false,
                            prior,
                        ) = await self._prior_calculator.calculate_prior(
                            sensor_id, start_time, end_time
                        )
                    except Exception as err:
                        raise PriorCalculationError(
                            f"Failed to calculate prior for {sensor_id}: {err}"
                        ) from err

                    # Update the prior state with entity prior
                    self.prior_state.update_entity_prior(
                        sensor_id,
                        prob_given_true,
                        prob_given_false,
                        prior,
                        dt_util.utcnow().isoformat(),
                    )

            # Calculate the overall prior after updating all entity priors
            overall_prior = self.prior_state.calculate_overall_prior()
            self.prior_state.update(overall_prior=overall_prior)

            # Save the updated priors
            await self._save_debounced_data()

            _LOGGER.info("Successfully updated learned priors")
        except Exception as err:
            raise CalculationError(f"Failed to update learned priors: {err}") from err

    def async_track_prior_updates(self) -> None:
        """Set up periodic prior updates using Home Assistant's async_track_time_interval."""

        async def _update_priors_wrapper(_):
            try:
                await self.update_learned_priors()
            except (TimeoutError, HomeAssistantError, ValueError, RuntimeError) as err:
                _LOGGER.error("Error in scheduled prior update: %s", err)

        if hasattr(self, "_prior_update_tracker"):
            self._prior_update_tracker()

        self._prior_update_tracker = async_track_time_interval(
            self.hass, _update_priors_wrapper, self._prior_update_interval
        )

    async def _async_update_data(self) -> ProbabilityState:
        """Update data with improved error handling."""
        # Check if there's a reason to update before performing expensive calculations
        should_update = False

        # Check if this is the first update or decay is active
        if not self.data.current_states or self.data.decaying:
            should_update = True

        # Check if sensor states or availability have changed
        if not should_update and self.data.current_states != self.data.previous_states:
            should_update = True

        # If no reason to update, return existing data
        if not should_update:
            return self.data

        try:
            # Calculate probabilities
            probability_state = self.calculator.calculate_occupancy_probability(
                self.data.current_states,
                datetime.now(),
            )

            # Check if probability has changed significantly
            if abs(probability_state.probability - self.data.probability) <= 0.01:
                # Still update decay status even when skipping main update
                self.data.update(decaying=probability_state.decaying)
                return self.data
            # Store previous states before updating
            previous_states = self.data.current_states.copy()

            # Update coordinator data
            self.data = probability_state
            self.data.current_states = self.data.current_states
            self.data.previous_states = previous_states

        except (
            TimeoutError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            HomeAssistantError,
            asyncio.CancelledError,
        ) as err:
            _LOGGER.error("Error updating occupancy data: %s", err)
            return self.data
        else:
            return probability_state

    async def _save_debounced_data(self) -> None:
        """Save data with debouncing."""
        try:
            await self.storage.async_save_prior_state(
                self.config[CONF_NAME],
                self.prior_state,
            )
        except Exception as err:
            raise StorageError(f"Failed to save data: {err}") from err

    async def async_update_threshold(self, value: float) -> None:
        """Update the threshold value.

        Args:
            value: The new threshold value as a percentage (1-99)

        Raises:
            ServiceValidationError: If the value is invalid
            HomeAssistantError: If there's an error updating the config entry

        """
        _LOGGER.debug("Updating threshold to %.2f", value)

        # Update config
        self.config[CONF_THRESHOLD] = value
        new_options = dict(self.config_entry.options)
        new_options[CONF_THRESHOLD] = value

        try:
            # Update config entry
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                options=new_options,
            )

            # Update state
            self.data.update(threshold=value / 100.0)

            # Trigger refresh
            await self.async_refresh()

        except ValueError as err:
            raise ServiceValidationError(f"Failed to update threshold: {err}") from err
        except Exception as err:
            raise HomeAssistantError(f"Failed to update threshold: {err}") from err

    @callback
    def async_add_listener(
        self, update_callback: CALLBACK_TYPE, context: Any = None
    ) -> Callable[[], None]:
        """Add a listener for data updates with improved tracking."""
        _LOGGER.debug("Adding listener for %s", self.name)
        return super().async_add_listener(update_callback, context)
