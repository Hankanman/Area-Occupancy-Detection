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
    async_track_point_in_time,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .calculate_prior import PriorCalculator
from .calculate_prob import ProbabilityCalculator
from .const import (
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_HISTORY_PERIOD,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
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
from .types import DeviceInfo, PriorState, ProbabilityState, SensorInputs

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
            update_interval=None,  # Disable automatic updates - we'll update via state changes only
            always_update=True,  # Always update listeners to ensure immediate response
        )

        # Initialize basic attributes
        self.config_entry = config_entry
        self.config = dict(config_entry.data)
        self._entity_ids: set[str] = set()
        self._state_lock = asyncio.Lock()
        self._storage_lock = asyncio.Lock()
        self._decay_unsub: CALLBACK_TYPE | None = None

        # Initialize sensor inputs with validation
        try:
            self.inputs = SensorInputs.from_config(self.config)
        except (ValueError, TypeError) as err:
            raise ConfigEntryError(f"Invalid sensor configuration: {err}") from err

        # Initialize timers and intervals
        self._prior_update_interval = timedelta(hours=1)
        self._prior_update_tracker = None
        self._next_prior_update = None
        self._save_interval = timedelta(seconds=10)
        self._last_save = dt_util.utcnow()

        # Initialize tracking
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
            decaying=False,
            decay_start_time=None,
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
        self.calculator = ProbabilityCalculator(
            decay_handler=self.decay_handler,
            probability_state=self.data,
            prior_state=self.prior_state,
            probabilities=self.probabilities,
        )
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

    @property
    def prior_update_interval(self) -> timedelta:
        """Return the interval between prior updates."""
        return self._prior_update_interval

    @property
    def next_prior_update(self) -> datetime | None:
        """Return the next scheduled prior update time."""
        return self._next_prior_update

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

            # Schedule prior updates at hour boundaries
            await self._schedule_next_prior_update()

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
        self._stop_decay_updates()

        # Cancel prior update tracker
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        await super().async_shutdown()

        # Additional cleanup specific to our use case
        if hasattr(self, "_remove_state_listener"):
            self._remove_state_listener()
            self._remove_state_listener = None

        # Save final state
        await self._save_debounced_data()

        # Clear data
        self.data = None
        self.prior_state = None
        self._state_cache = {}
        self._entity_ids = set()
        self._last_save = dt_util.utcnow()

    async def async_load_stored_data(self) -> None:
        """Load and restore data from storage."""
        try:
            _LOGGER.debug("Loading stored data from storage")

            # Attempt storage migration first
            try:
                await self.storage.async_migrate_storage()
            except StorageError as err:
                _LOGGER.warning(
                    "Storage migration failed, proceeding with load: %s", err
                )

            # Load prior state after migration attempt
            name, stored_prior_state = await self.storage.async_load_prior_state()

            if stored_prior_state:
                _LOGGER.debug(
                    "Found stored prior state for instance %s, restoring",
                    self.config_entry.entry_id,
                )
                self.prior_state = stored_prior_state
            else:
                _LOGGER.info(
                    "No stored prior state found for instance %s, initializing with defaults",
                    self.config_entry.entry_id,
                )
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
                    decaying=False,
                    decay_start_time=None,
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

                # Schedule initial save to ensure storage is created
                await self._save_debounced_data()

            _LOGGER.debug(
                "Successfully restored stored data for instance %s",
                self.config_entry.entry_id,
            )
        except StorageError as err:
            _LOGGER.warning(
                "Storage error for instance %s, initializing with defaults: %s",
                self.config_entry.entry_id,
                err,
            )
            # Initialize with defaults on storage error
            self.prior_state = PriorState()
            self.prior_state.initialize_from_defaults(self.probabilities)
            self.prior_state.update(
                analysis_period=self.config.get(
                    CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                ),
            )
            raise StorageError(f"Failed to load stored data: {err}") from err

    async def async_update_options(self) -> None:
        """Update coordinator options with improved error handling."""
        try:
            _LOGGER.debug(
                "Coordinator async_update_options starting with config: %s", self.config
            )

            # Update configuration first
            self.config = {**self.config_entry.data, **self.config_entry.options}
            _LOGGER.debug("Updated config: %s", self.config)

            # Reinitialize all components with new configuration
            _LOGGER.debug("Reinitializing all components with new configuration")

            # Reset state tracking
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
                decaying=False,
                decay_start_time=None,
            )

            # Reset all components
            self.inputs = SensorInputs.from_config(self.config)
            self.probabilities = Probabilities(config=self.config)

            # Reset prior state
            self.prior_state = PriorState()
            self.prior_state.initialize_from_defaults(self.probabilities)
            self.prior_state.update(
                analysis_period=self.config.get(
                    CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                ),
            )

            # Save the reset prior state immediately
            await self.storage.async_save_prior_state(
                self.config[CONF_NAME], self.prior_state, immediate=True
            )

            # Reset remaining components
            self.decay_handler = DecayHandler(self.config)
            self.calculator = ProbabilityCalculator(
                decay_handler=self.decay_handler,
                probability_state=self.data,
                prior_state=self.prior_state,
                probabilities=self.probabilities,
            )
            self._prior_calculator = PriorCalculator(
                coordinator=self,
                probabilities=self.probabilities,
                hass=self.hass,
            )

            # Clear state cache and state manager
            self._state_cache = {}
            self._state_manager = OccupancyStateManager()

            # Initialize states for current sensors
            await self.async_initialize_states()

            # Force immediate refresh to reflect changes
            await self.async_refresh()
            _LOGGER.debug("Coordinator async_update_options completed")

        except (ValueError, KeyError) as err:
            _LOGGER.error("Invalid configuration in async_update_options: %s", err)
            raise ConfigEntryError(f"Invalid configuration: {err}") from err
        except HomeAssistantError as err:
            _LOGGER.error("Failed to update coordinator options: %s", err)
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
                old_state = event.data.get("old_state")

                if not new_state or entity_id not in entities:
                    return

                # Log state change to help debug
                _LOGGER.debug(
                    "State change: %s -> %s",
                    old_state.state if old_state else "None",
                    new_state.state,
                )

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

                # Trigger an immediate update
                async def async_handle_state_change():
                    await update_cache()
                    await self.async_refresh()

                self.hass.async_create_task(async_handle_state_change())

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
        return self.inputs.get_all_sensors()

    async def update_learned_priors(self, history_period: int | None = None) -> None:
        """Update learned priors using historical data."""
        try:
            _LOGGER.debug(
                "Starting update_learned_priors with history_period: %s (type: %s)",
                history_period,
                type(history_period),
            )

            period = history_period or self.config.get(
                CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
            )
            _LOGGER.debug("Using period: %s (type: %s)", period, type(period))

            end_time = dt_util.utcnow()
            start_time = end_time - timedelta(days=period)
            _LOGGER.debug(
                "Calculated time range - start: %s, end: %s", start_time, end_time
            )

            # Update the analysis period in prior_state
            _LOGGER.debug("Updating prior_state analysis period to: %s", period)
            self.prior_state.update(analysis_period=period)

            # Get all sensors that need priors calculated
            sensors = self.get_configured_sensors()
            _LOGGER.debug("Calculating prior for %s sensors", len(sensors))

            for sensor_id in sensors:
                _LOGGER.debug("Calculating prior for sensor: %s", sensor_id)
                try:
                    (
                        prob_given_true,
                        prob_given_false,
                        prior,
                    ) = await self._prior_calculator.calculate_prior(
                        sensor_id, start_time, end_time
                    )
                    _LOGGER.debug(
                        "Calculated probabilities for %s - prob_given_true: %s (type: %s), "
                        "prob_given_false: %s (type: %s), prior: %s (type: %s)",
                        sensor_id,
                        prob_given_true,
                        type(prob_given_true),
                        prob_given_false,
                        type(prob_given_false),
                        prior,
                        type(prior),
                    )
                except Exception as err:
                    _LOGGER.exception("Error calculating prior for %s", sensor_id)
                    raise PriorCalculationError(
                        f"Failed to calculate prior for {sensor_id}: {err}"
                    ) from err

                # Update the prior state with entity prior
                _LOGGER.debug("Updating prior state for %s", sensor_id)
                self.prior_state.update_entity_prior(
                    sensor_id,
                    prob_given_true,
                    prob_given_false,
                    prior,
                    dt_util.utcnow().isoformat(),
                )

            # Calculate the overall prior after updating all entity priors
            _LOGGER.debug("Calculating overall prior")
            overall_prior = self.prior_state.calculate_overall_prior()
            _LOGGER.debug(
                "Overall prior calculated: %s (type: %s)",
                overall_prior,
                type(overall_prior),
            )
            self.prior_state.update(overall_prior=overall_prior)

            # Save the updated priors
            _LOGGER.debug("Starting prior update for %s", period)
            await self._save_debounced_data()
            _LOGGER.debug("Prior update complete, saving data")

            _LOGGER.info("Successfully updated learned priors")
        except Exception as err:
            _LOGGER.exception("Failed to update learned priors")
            raise CalculationError(f"Failed to update learned priors: {err}") from err

    async def _schedule_next_prior_update(self) -> None:
        """Schedule the next prior update at the start of the next hour."""
        now = dt_util.utcnow()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        self._next_prior_update = next_hour

        # Cancel any existing update
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()

        self._prior_update_tracker = async_track_point_in_time(
            self.hass, self._handle_prior_update, next_hour
        )
        _LOGGER.debug(
            "Scheduled next prior update for %s in area %s",
            next_hour,
            self.config[CONF_NAME],
        )

    async def _handle_prior_update(self, _now: datetime) -> None:
        """Handle the prior update and schedule the next one."""
        try:
            await self.update_learned_priors()
            _LOGGER.debug("Updated learned priors for area %s", self.config[CONF_NAME])
        except (
            CalculationError,
            PriorCalculationError,
            StorageError,
            HomeAssistantError,
        ) as err:
            _LOGGER.error(
                "Error updating learned priors for area %s: %s",
                self.config[CONF_NAME],
                err,
            )

        # Schedule next update
        await self._schedule_next_prior_update()

    def _start_decay_updates(self) -> None:
        """Start regular decay updates every 5 seconds."""
        self._stop_decay_updates()

        async def _do_decay_update(*_) -> None:
            """Execute decay update."""
            try:
                await self.async_refresh()
                _LOGGER.debug(
                    "Decay update: prob=%.3f, status=%.3f",
                    self.data.probability,
                    self.data.decay_status,
                )
            except (
                TimeoutError,
                ValueError,
                AttributeError,
                HomeAssistantError,
                asyncio.CancelledError,
            ) as err:
                _LOGGER.error("Error in decay update: %s", err)

        self._decay_unsub = async_track_time_interval(
            self.hass, _do_decay_update, timedelta(seconds=5)
        )

    def _stop_decay_updates(self) -> None:
        """Stop decay updates."""
        if self._decay_unsub is not None:
            self._decay_unsub()
            self._decay_unsub = None

    async def _async_update_data(self) -> ProbabilityState:
        """Update data with improved error handling."""
        try:
            # Check for OFF transitions (sensor turning from on to off)
            off_transition = False
            for entity_id, current in self.data.current_states.items():
                previous = self.data.previous_states.get(entity_id, {})
                if previous.get("state") == "on" and current.get("state") == "off":
                    _LOGGER.debug("Detected OFF transition for %s", entity_id)
                    off_transition = True
                    break

            # Calculate probabilities
            probability_state = self.calculator.calculate_occupancy_probability(
                self.data.current_states,
                datetime.now(),
            )

            # Store previous states for next update
            previous_states = self.data.current_states.copy()

            # Log current status
            _LOGGER.debug(
                "Status: probability=%.3f threshold=%.3f decay_status=%.3f decaying=%s is_occupied=%s",
                probability_state.probability,
                probability_state.threshold,
                probability_state.decay_status,
                probability_state.decaying,
                probability_state.is_occupied,
            )

            # Manage decay timer
            decay_active = probability_state.decay_status < 1.0 or off_transition
            if decay_active:
                probability_state.decaying = True
                if not self._decay_unsub:
                    _LOGGER.debug(
                        "Starting decay timer (decay_status=%.3f, off_transition=%s)",
                        probability_state.decay_status,
                        off_transition,
                    )
                    self._start_decay_updates()
            else:
                probability_state.decaying = False
                if self._decay_unsub:
                    _LOGGER.debug("Stopping decay timer (decay complete)")
                    self._stop_decay_updates()

            # Update coordinator data
            self.data = probability_state
            self.data.current_states = self.data.current_states
            self.data.previous_states = previous_states

            # Check if we should reset to MIN_PROBABILITY
            # Only reset when decay is complete AND no sensors are active
            should_reset = (
                not decay_active and probability_state.probability <= MIN_PROBABILITY
            )

            # Only reset if all sensors are off
            if should_reset:
                for info in self.data.current_states.values():
                    if info.get("state") == "on":
                        should_reset = False
                        break

            # Perform reset if needed
            if should_reset:
                _LOGGER.debug("Resetting state: all sensors off, decay complete")

                # Reset state tracking
                self.data.decaying = False
                self.data.decay_start_time = None
                self._stop_decay_updates()

                # Refresh sensor states
                await self._refresh_sensor_states()

                # Ensure minimum probability
                self.data.probability = MIN_PROBABILITY
                self.data.previous_probability = MIN_PROBABILITY
                self.data.is_occupied = False
            else:
                # Update occupancy state based on threshold
                self.data.is_occupied = self.data.probability >= self.data.threshold

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
            return self.data

    async def _refresh_sensor_states(self) -> None:
        """Refresh all sensor states from current values."""
        try:
            sensors = self.get_configured_sensors()
            _LOGGER.debug("Refreshing %d sensor states", len(sensors))

            # Get fresh states
            await self._state_manager.initialize_states(self.hass, sensors)
            fresh_states = await self._state_manager.get_active_sensors()

            # Update states
            self.data.current_states = fresh_states
            self.data.previous_states = {}
        except (TimeoutError, ValueError, AttributeError, HomeAssistantError) as err:
            _LOGGER.error("Error refreshing sensor states: %s", err)

    async def _save_debounced_data(self) -> None:
        """Save data with debouncing."""
        try:
            await self.storage.async_save_prior_state(
                self.config[CONF_NAME],
                self.prior_state,
            )
            self._last_save = dt_util.utcnow()
        except (TimeoutError, HomeAssistantError, ValueError, RuntimeError) as err:
            raise StorageError(f"Failed to save data: {err}") from err

    async def async_update_threshold(self, value: float) -> None:
        """Update the threshold value.

        Args:
            value: The new threshold value as a percentage (1-99)

        Raises:
            ServiceValidationError: If the value is invalid
            HomeAssistantError: If there's an error updating the config entry

        """
        _LOGGER.debug("Updating threshold: %.2f", value)

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
        return super().async_add_listener(update_callback, context)
