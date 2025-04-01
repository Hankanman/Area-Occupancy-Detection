"""Area Occupancy Coordinator."""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.debounce import Debouncer
from homeassistant.config_entries import ConfigEntry

from .const import (
    CONF_MOTION_SENSORS,
    CONF_HISTORY_PERIOD,
    CONF_NAME,
    DOMAIN,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    CONF_APPLIANCES,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_DOOR_SENSORS,
    CONF_WINDOW_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_DEVICES,
    CONF_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_THRESHOLD,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    MIN_PROBABILITY,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
)
from .storage import AreaOccupancyStorage
from .calculate_prob import ProbabilityCalculator
from .calculate_prior import PriorCalculator
from .probabilities import Probabilities
from .state_management import OccupancyStateManager
from .decay_handler import DecayHandler
from .exceptions import (
    StateError,
    CalculationError,
    StorageError,
)
from .types import (
    ProbabilityState,
    PriorState,
    DeviceInfo,
)

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
            update_interval=timedelta(
                seconds=config_entry.data.get("update_interval", 1)
            ),
        )
        self.config_entry = config_entry
        # Create a mutable copy of the config
        self.config = dict(config_entry.data)
        self.storage = AreaOccupancyStorage(hass, config_entry.entry_id)
        self.probabilities = Probabilities(config=self.config)

        # Initialize the state
        self.data = ProbabilityState()
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

        # Initialize prior state with default type priors
        self.prior_state = PriorState()
        self.prior_state.initialize_from_defaults(self.probabilities)
        self.prior_state.update(
            analysis_period=self.config.get(
                CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
            ),
        )

        self.decay_handler = DecayHandler(self.config)
        self.calculator = ProbabilityCalculator(self, self.probabilities)

        # Initialize state management
        self._state_manager = OccupancyStateManager()
        self._state_cache: dict[str, dict] = {}
        self._entity_ids: set[str] = set()
        self._state_lock = asyncio.Lock()
        self._storage_lock = asyncio.Lock()

        # Initialize timers and intervals
        self._prior_update_interval = timedelta(hours=1)
        self._save_interval = timedelta(seconds=10)
        self._last_save = dt_util.utcnow()

        # Initialize handlers and calculators
        self._prior_calculator = PriorCalculator(
            coordinator=self,
            probabilities=self.probabilities,
            hass=self.hass,
        )

        # Initialize debouncers
        self._debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=0.5,
            immediate=True,
            function=self.async_refresh,
        )
        self._save_debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=30.0,
            immediate=False,
            function=self._save_debounced_data,
        )

        # Initialize tracking
        self._prior_update_tracker = None
        self._remove_state_listener = None

        # Initialize decay-related attributes
        self.config[CONF_DECAY_ENABLED] = self.config.get(
            CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED
        )
        self.config[CONF_DECAY_WINDOW] = self.config.get(
            CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
        )

    @property
    def entity_ids(self) -> set[str]:
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

    async def async_setup(self) -> None:
        """Setup the coordinator and load stored data."""
        _LOGGER.debug(
            "Setting up AreaOccupancyCoordinator for %s", self.config[CONF_NAME]
        )

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
        except Exception as err:
            _LOGGER.error(
                "Failed to set up AreaOccupancyCoordinator: %s", err, exc_info=True
            )
            raise

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
            _LOGGER.error("Error loading stored data: %s", err, exc_info=True)
            raise StorageError(f"Failed to load stored data: {err}") from err

    async def async_refresh(self) -> None:
        """Refresh data and notify listeners."""
        try:
            await super().async_refresh()
        except (
            HomeAssistantError,
            ValueError,
            RuntimeError,
            asyncio.TimeoutError,
            asyncio.CancelledError,
        ) as err:
            _LOGGER.error("Error in async_refresh: %s", err, exc_info=True)

    async def async_update_options(self) -> None:
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

        except (ValueError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error updating coordinator options: %s", err)
            raise HomeAssistantError(
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
                        ValueError,
                        AttributeError,
                        TypeError,
                        KeyError,
                        asyncio.TimeoutError,
                    ) as err:
                        _LOGGER.error("Error updating state cache: %s", err)

                self.hass.async_create_task(update_cache())
                self.hass.async_create_task(self._debouncer.async_call())

            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                HomeAssistantError,
                asyncio.CancelledError,
                asyncio.TimeoutError,
            ) as err:
                _LOGGER.error("Error in state change listener: %s", err, exc_info=True)

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
            _LOGGER.error("Error initializing states: %s", err, exc_info=True)
            raise StateError(f"Failed to initialize states: {err}") from err

        self._setup_entity_tracking()

    async def async_unload(self) -> None:
        """Unload the coordinator and clean up resources."""
        try:
            # Cancel any pending prior updates
            if hasattr(self, "_prior_update_tracker"):
                self._prior_update_tracker()
                self._prior_update_tracker = None

            # Save final state
            await self._save_debounced_data()

            # Remove state listener
            if self._remove_state_listener is not None:
                self._remove_state_listener()
                self._remove_state_listener = None

            # Shutdown debouncers
            if self._debouncer is not None:
                await self._debouncer.async_shutdown()
                self._debouncer = None

            if self._save_debouncer is not None:
                await self._save_debouncer.async_shutdown()
                self._save_debouncer = None

            # Clear data
            self.data = None
            self.prior_state = None
            self._state_cache = {}
            self._entity_ids = set()
            self._last_save = dt_util.utcnow()

        except Exception as err:
            _LOGGER.error("Error unloading coordinator: %s", err, exc_info=True)
            raise

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
                    prob_given_true, prob_given_false, prior = (
                        await self._prior_calculator.calculate_prior(
                            sensor_id, start_time, end_time
                        )
                    )

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
            _LOGGER.error("Error updating learned priors: %s", err, exc_info=True)
            raise CalculationError(f"Failed to update learned priors: {err}") from err

    def async_track_prior_updates(self) -> None:
        """Set up periodic prior updates using Home Assistant's async_track_time_interval."""

        async def _update_priors_wrapper(_):
            try:
                await self.update_learned_priors()
            except (
                HomeAssistantError,
                ValueError,
                RuntimeError,
                asyncio.TimeoutError,
            ) as err:
                _LOGGER.error("Error in scheduled prior update: %s", err, exc_info=True)

        if hasattr(self, "_prior_update_tracker"):
            self._prior_update_tracker()

        self._prior_update_tracker = async_track_time_interval(
            self.hass, _update_priors_wrapper, self._prior_update_interval
        )

    async def _async_update_data(self) -> ProbabilityState:
        """Update data via library."""
        try:
            # Check if there's a reason to update before performing expensive calculations
            should_update = False

            # Check if this is the first update or decay is active
            if not self.data.current_states:
                should_update = True
            elif self.data.decaying:
                should_update = True

            # Check if sensor states or availability have changed
            if not should_update:
                if self.data.current_states != self.data.previous_states:
                    should_update = True

            # If no reason to update, return existing data
            if not should_update:
                return self.data

            # Only calculate probabilities if we need to update
            probability_state = self.calculator.calculate_occupancy_probability(
                self.data.current_states,
                datetime.now(),
            )

            # Check if probability has changed significantly
            if abs(probability_state.probability - self.data.probability) <= 0.01:
                # Still update decay status even when skipping main update
                self.data.update(decaying=probability_state.decaying)

                # Skip additional updates if change is minor
                if (
                    should_update is False
                ):  # Only true for first run or explicit changes
                    return self.data

            # Store previous states before updating
            previous_states = self.data.current_states.copy()

            # Update coordinator data
            self.data = probability_state
            self.data.current_states = self.data.current_states
            self.data.previous_states = previous_states

            return probability_state

        except (
            ValueError,
            TypeError,
            KeyError,
            HomeAssistantError,
            CalculationError,
        ) as err:
            _LOGGER.error("Error updating occupancy data: %s", err, exc_info=True)
            return self.data

    async def _save_debounced_data(self) -> None:
        """Save data with debouncing."""
        try:
            await self.storage.async_save_prior_state(
                self.config[CONF_NAME],
                self.prior_state,
            )
        except Exception as err:
            _LOGGER.error("Error saving data: %s", err, exc_info=True)
            raise StorageError(f"Failed to save data: {err}") from err

    async def async_update_threshold(self, value: float) -> None:
        _LOGGER.debug("Updating threshold to %.2f", value)
        self.config[CONF_THRESHOLD] = value
        new_options = dict(self.config_entry.options)
        new_options[CONF_THRESHOLD] = value

        try:
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                options=new_options,
            )
            self.data["threshold"] = value / 100.0
            await self.async_refresh()
        except ValueError as err:
            _LOGGER.error("Error updating threshold: %s", err)
            raise HomeAssistantError(f"Failed to update threshold: {err}") from err
