"""Area Occupancy Coordinator."""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta
import threading

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
    CONF_THRESHOLD,
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
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_THRESHOLD,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
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
)
from .types import (
    ProbabilityState,
    LearnedPrior,
    DeviceInfo,
)
from .storage import AreaOccupancyStorage
from .calculate_prob import ProbabilityCalculator
from .calculate_prior import PriorCalculator
from .probabilities import Probabilities
from .state_management import OccupancyStateManager
from .exceptions import (
    ConfigurationError,
    StateError,
    CalculationError,
    StorageError,
)
from .decay_handler import DecayHandler

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
            name=DOMAIN,
            update_interval=timedelta(seconds=1),
        )
        # Validate configuration
        self.entry_id = config_entry.entry_id
        self.config_entry = config_entry
        self.config = {**config_entry.data, **config_entry.options}
        self._validate_config()

        # Initialize storage and learned_priors
        self.storage = AreaOccupancyStorage(hass, config_entry.entry_id)
        self.learned_priors: dict[str, LearnedPrior] = {}
        self.type_priors: dict[str, LearnedPrior] = {}

        # Initialize threshold
        self.threshold = self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD) / 100.0

        # Initialize coordinator data
        self.data = ProbabilityState(
            probability=0.0,
            previous_probability=0.0,
            threshold=self.threshold,
            prior_probability=0.0,
            active_triggers=[],
            sensor_probabilities={},
            decay_status=0.0,
            device_states={},
            sensor_availability={},
            is_occupied=False,
        )

        # Initialize state management
        self._state_manager = OccupancyStateManager()
        self._state_cache: dict[str, dict] = {}
        self._entity_ids: set[str] = set()
        self._last_occupied: datetime | None = None
        self._last_positive_trigger = None
        self._state_lock = asyncio.Lock()
        self._storage_lock = asyncio.Lock()
        self._thread_lock = threading.Lock()

        # Initialize timers and intervals
        self._prior_update_interval = timedelta(hours=1)
        self._save_interval = timedelta(seconds=10)
        self._last_save = dt_util.utcnow()

        # Initialize handlers and calculators
        self._probabilities = Probabilities(config=self.config)
        self._decay_handler = DecayHandler(self.config)
        self._calculator = ProbabilityCalculator(
            coordinator=self,
            probabilities=self._probabilities,
        )
        self._prior_calculator = PriorCalculator(
            coordinator=self,
            probabilities=self._probabilities,
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
        self._stored_data = None

    def _validate_config(self) -> None:
        """Validate the configuration."""
        if not self.config.get(CONF_NAME):
            raise ConfigurationError("Name is required")

        motion_sensors = self.config.get(CONF_MOTION_SENSORS, [])
        if not motion_sensors:
            raise ConfigurationError("At least one motion sensor is required")

        primary_sensor = self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor:
            raise ConfigurationError("Primary occupancy sensor is required")
        if primary_sensor not in motion_sensors:
            raise ConfigurationError(
                "Primary occupancy sensor must be selected from motion sensors"
            )

        # Validate weights
        weights = [
            (
                CONF_WEIGHT_MOTION,
                self.config.get(CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION),
            ),
            (
                CONF_WEIGHT_MEDIA,
                self.config.get(CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA),
            ),
            (
                CONF_WEIGHT_APPLIANCE,
                self.config.get(CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE),
            ),
            (CONF_WEIGHT_DOOR, self.config.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR)),
            (
                CONF_WEIGHT_WINDOW,
                self.config.get(CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW),
            ),
            (
                CONF_WEIGHT_LIGHT,
                self.config.get(CONF_WEIGHT_LIGHT, DEFAULT_WEIGHT_LIGHT),
            ),
            (
                CONF_WEIGHT_ENVIRONMENTAL,
                self.config.get(
                    CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL
                ),
            ),
        ]
        for name, weight in weights:
            if not 0 <= weight <= 1:
                raise ConfigurationError(f"{name} must be between 0 and 1")

    @property
    def entity_ids(self) -> set[str]:
        return self._entity_ids

    @property
    def calculator(self) -> ProbabilityCalculator:
        """Get the probability calculator."""
        return self._calculator

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
            "identifiers": {(DOMAIN, self.entry_id)},
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
            stored_data = await self.storage.async_load()
            if not stored_data:
                _LOGGER.debug("No stored data found, initializing fresh state")
                self._reset_state()
                self.type_priors = self._probabilities.get_initial_type_priors()
                return

            self.learned_priors = stored_data.get("learned_priors", {})

            stored_type_priors = stored_data.get("type_priors", {})
            if stored_type_priors:
                self.type_priors = stored_type_priors
            else:
                self.type_priors = self._probabilities.get_initial_type_priors()

            _LOGGER.debug("Successfully restored stored data")
        except Exception as err:
            _LOGGER.error("Error loading stored data: %s", err, exc_info=True)
            raise StorageError(f"Failed to load stored data: {err}") from err

    async def async_initialize_states(self) -> None:
        """Initialize sensor states."""
        _LOGGER.debug("Initializing sensor states")
        try:
            sensors = self.get_configured_sensors()
            await self._state_manager.initialize_states(self.hass, sensors)
            self._state_cache = await self._state_manager.get_active_sensors()
            _LOGGER.info("Initialized states for sensors: %s", sensors)
        except Exception as err:
            _LOGGER.error("Error initializing states: %s", err, exc_info=True)
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

    async def update_learned_priors(self, history_period: int | None = None) -> None:
        """Update learned priors using historical data."""
        try:
            period = history_period or self.config.get(
                CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
            )
            end_time = dt_util.utcnow()
            start_time = end_time - timedelta(days=period)

            # Get all sensors that need priors calculated
            sensors = self.get_configured_sensors()
            for sensor_id in sensors:
                if sensor_id != self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR):
                    await self._prior_calculator.calculate_prior(
                        sensor_id, start_time, end_time
                    )

            # Save the updated priors
            await self._save_debounced_data()

            _LOGGER.info("Successfully updated learned priors")
        except Exception as err:
            _LOGGER.error("Error updating learned priors: %s", err, exc_info=True)
            raise CalculationError(f"Failed to update learned priors: {err}") from err

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes using state manager."""
        entities = self.get_configured_sensors()
        _LOGGER.debug("Setting up entity tracking for: %s", entities)

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

                _LOGGER.debug("Scheduling refresh due to state change of %s", entity_id)
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

    async def _async_update_data(self) -> None:
        """Update data from sensors and calculate probabilities."""
        try:
            # Get current states for all sensors
            sensor_states = {}

            for entity_id in self.get_configured_sensors():
                state = self.hass.states.get(entity_id)
                if state is not None and state.state not in [
                    "unavailable",
                    "unknown",
                    None,
                    "",
                ]:
                    sensor_states[entity_id] = {
                        "state": state.state,
                        "availability": True,
                        "last_changed": (
                            state.last_changed.isoformat()
                            if state.last_changed
                            else None
                        ),
                    }
                else:
                    sensor_states[entity_id] = {
                        "state": None,
                        "availability": False,
                        "last_changed": dt_util.utcnow().isoformat(),
                    }

            # Update sensor states
            self._state_cache = sensor_states
            _LOGGER.debug("Updated sensor states, found %s sensors", len(sensor_states))

            # Calculate probabilities
            _LOGGER.debug("Calculating probabilities...")
            result = self._calculator.perform_calculation_logic(
                sensor_states, dt_util.utcnow()
            )

            # Update coordinator data
            _LOGGER.debug(
                "Setting coordinator data with probability: %s",
                result["probability"],
            )
            self.data = result
            _LOGGER.debug("Data updated, data keys: %s", list(self.data.keys()))

            _LOGGER.debug(
                "Updated data - Probability: %.2f%%, Occupied: %s, Decay: %.2f%%",
                self.data["probability"] * 100,
                self.data["is_occupied"],
                self.data["decay_status"] * 100,
            )

        except Exception as err:
            _LOGGER.error("Error updating data: %s", err, exc_info=True)
            raise

        _LOGGER.debug("Completed _async_update_data method")
        return self.data

    async def _async_store_data(self) -> None:
        try:
            new_data = {
                "name": self.config[CONF_NAME],
                "learned_priors": self.learned_priors,
                "type_priors": self.type_priors,
            }

            if self._stored_data:
                old_data = self._stored_data.copy()
                new_data_compare = new_data.copy()
                old_data.pop("last_updated", None)
                new_data_compare.pop("last_updated", None)

                if old_data == new_data_compare:
                    _LOGGER.debug("No significant changes detected; skipping save")
                    return

            self._stored_data = new_data
            await self._save_debouncer.async_call()

        except (ValueError, TypeError, KeyError) as err:
            _LOGGER.error("Error preparing data for storage: %s", err, exc_info=True)
        except HomeAssistantError as err:
            _LOGGER.error("Error preparing data storage: %s", err, exc_info=True)

    async def async_reset(self) -> None:
        _LOGGER.debug("Resetting coordinator")
        self._state_cache = {}
        self._last_occupied = None
        self._last_save = dt_util.utcnow()
        self.learned_priors.clear()
        await self.async_refresh()

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        if self._debouncer is not None:
            await self._debouncer.async_shutdown()
            self._debouncer = None

    async def async_unload(self) -> None:
        """Unload the coordinator and clean up resources."""
        _LOGGER.debug("Unloading coordinator")

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
            self._state_cache = {}
            self._entity_ids = set()
            self._last_occupied = None
            self._last_save = dt_util.utcnow()
            self.learned_priors.clear()
            self.type_priors.clear()

            _LOGGER.debug("Successfully unloaded coordinator")
        except Exception as err:
            _LOGGER.error("Error unloading coordinator: %s", err, exc_info=True)
            raise

    async def async_update_options(self) -> None:
        _LOGGER.debug("Updating options")
        try:
            self.config = {**self.config_entry.data, **self.config_entry.options}
            self._probabilities = Probabilities(config=self.config)
            self._calculator = ProbabilityCalculator(
                coordinator=self,
                probabilities=self._probabilities,
            )
            self._prior_calculator = PriorCalculator(
                coordinator=self,
                probabilities=self._probabilities,
                hass=self.hass,
            )

            self._setup_entity_tracking()
            await self._async_reinitialize_states()

        except (ValueError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error updating coordinator options: %s", err)
            raise HomeAssistantError(
                f"Failed to update coordinator options: {err}"
            ) from err

    def _reset_state(self) -> None:
        """Reset the coordinator state."""
        self.learned_priors.clear()
        self.type_priors.clear()
        self._state_cache.clear()
        self._entity_ids.clear()
        self._last_positive_trigger = None
        self._last_occupied = None
        self.data = ProbabilityState(
            probability=0.0,
            previous_probability=0.0,
            threshold=self.threshold,
            prior_probability=0.0,
            active_triggers=[],
            sensor_probabilities={},
            decay_status=0.0,
            device_states={},
            sensor_availability={},
            is_occupied=False,
        )
        _LOGGER.debug("Reset coordinator state")

    def register_entity(self, entity_id: str) -> None:
        _LOGGER.debug("Registering entity: %s", entity_id)
        self._entity_ids.add(entity_id)

    def unregister_entity(self, entity_id: str) -> None:
        _LOGGER.debug("Unregistering entity: %s", entity_id)
        self._entity_ids.discard(entity_id)

    async def async_refresh(self) -> None:
        """Refresh data and notify listeners."""
        _LOGGER.debug("async_refresh called")
        try:
            await super().async_refresh()
            _LOGGER.debug(
                "super().async_refresh completed, data: %s",
                "available" if self.data is not None else "None",
            )
        except (
            HomeAssistantError,
            ValueError,
            RuntimeError,
            asyncio.TimeoutError,
            asyncio.CancelledError,
        ) as err:
            _LOGGER.error("Error in async_refresh: %s", err, exc_info=True)

    async def _async_reinitialize_states(self) -> None:
        await self.async_initialize_states()
        await self.async_refresh()

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

    def update_learned_prior(
        self, entity_id: str, p_true: float, p_false: float, prior: float
    ) -> None:
        _LOGGER.debug("Updating learned priors")
        self.learned_priors[entity_id] = {
            "prob_given_true": p_true,
            "prob_given_false": p_false,
            "prior": prior,
            "last_updated": dt_util.utcnow().isoformat(),
        }
        self.hass.async_create_task(self.async_save_state())
        self.async_set_updated_data(self.data)
        self._probabilities.update_config(self.config)

    async def async_save_state(self) -> None:
        _LOGGER.debug("Saving state")
        now = dt_util.utcnow()
        if now - self._last_save < self._save_interval:
            return

        async with self._storage_lock:
            try:
                storage_data = {
                    "name": self.config[CONF_NAME],
                    "learned_priors": self.learned_priors,
                }
                await self.storage.async_save(storage_data)
                self._last_save = now
            except (IOError, ValueError, HomeAssistantError) as err:
                _LOGGER.error("Failed to save state: %s", err)

    async def async_restore_state(self, stored_data: dict) -> None:
        _LOGGER.debug("Restoring state")
        try:
            if not isinstance(stored_data, dict):
                raise ValueError("Invalid storage data format")

            last_occupied = stored_data.get("last_occupied")
            self._last_occupied = (
                dt_util.parse_datetime(last_occupied) if last_occupied else None
            )
            self.learned_priors = stored_data.get("learned_priors", {})

        except (ValueError, TypeError, KeyError, HomeAssistantError) as err:
            _LOGGER.error("Error restoring state: %s", err)
            self._reset_state()

    async def _save_debounced_data(self) -> None:
        """Save data with debouncing."""
        try:
            async with self._storage_lock:
                data = {
                    "learned_priors": self.learned_priors,
                    "type_priors": self.type_priors,
                }
                await self.storage.async_save(data)
                self._last_save = dt_util.utcnow()
                _LOGGER.debug("Successfully saved data")
        except Exception as err:
            _LOGGER.error("Error saving data: %s", err, exc_info=True)
            raise StorageError(f"Failed to save data: {err}") from err

    async def calculate_sensor_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ):
        return await self._prior_calculator.calculate_prior(
            entity_id, start_time, end_time
        )

    def update_type_prior(
        self,
        sensor_type: str,
        p_true: float,
        p_false: float,
        prior: float,
    ) -> None:
        _LOGGER.debug("Updating type prior for %s", sensor_type)
        self.type_priors[sensor_type] = {
            "prob_given_true": p_true,
            "prob_given_false": p_false,
            "prior": prior,
            "last_updated": dt_util.utcnow().isoformat(),
        }
        self.hass.async_create_task(self.async_save_state())
        self._probabilities.update_config(self.config)

    async def _delayed_remove_sensor(self, entity_id: str) -> None:
        removal_delay = 10
        try:
            await asyncio.sleep(removal_delay)
        except asyncio.CancelledError:
            return

        state = self._state_cache.get(entity_id)
        if state is None:
            return

        last_changed = state.get("last_changed")
        if last_changed:
            last_changed_dt = dt_util.parse_datetime(last_changed)
        else:
            last_changed_dt = dt_util.utcnow()

        elapsed = (dt_util.utcnow() - last_changed_dt).total_seconds()
        if elapsed >= removal_delay:
            _LOGGER.debug(
                "Removing sensor %s after delay; elapsed %.1f seconds",
                entity_id,
                elapsed,
            )
            self.hass.async_create_task(self._state_manager.remove_sensor(entity_id))

            async def update_cache():
                self._state_cache = await self._state_manager.get_active_sensors()
                await self.async_refresh()

            self.hass.async_create_task(update_cache())
