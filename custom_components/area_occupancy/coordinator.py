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
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
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
)
from .types import (
    ProbabilityResult,
    LearnedPrior,
    DeviceInfo,
)
from .storage import AreaOccupancyStorage
from .calculate_prob import ProbabilityCalculator
from .calculate_prior import PriorCalculator
from .probabilities import Probabilities
from .state_management import OccupancyStateManager

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
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
        self.config_entry = config_entry
        self.config = {**config_entry.data, **config_entry.options}

        # Initialize storage and learned_priors
        self.storage = AreaOccupancyStorage(hass, config_entry.entry_id)
        self.learned_priors: dict[str, LearnedPrior] = {}
        self.type_priors: dict[str, LearnedPrior] = {}

        self.threshold = self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD) / 100.0

        # Initialize probabilities before calculator
        self._probabilities = Probabilities(config=self.config)

        self._prior_update_tracker = None

        # Initialize calculator after super().__init__
        self._calculator = ProbabilityCalculator(
            coordinator=self,
            probabilities=self._probabilities,
        )

        self.entry_id = config_entry.entry_id

        self._state_lock = asyncio.Lock()
        # Remove internal sensor state dictionaries, use state_manager instead
        self.state_manager = OccupancyStateManager()
        # Maintain a synchronous cache for active sensors for quick access
        self._state_cache: dict[str, dict] = {}

        # New: maintain a centralized set of entity IDs
        self._entity_ids: set[str] = set()
        self._last_positive_trigger = None

        self._prior_update_interval = timedelta(hours=1)

        self._last_occupied: datetime | None = None

        self._prior_calculator = PriorCalculator(
            coordinator=self,
            probabilities=self._probabilities,
            hass=self.hass,
        )

        self._storage_lock = asyncio.Lock()
        self._last_save = dt_util.utcnow()
        self._save_interval = timedelta(seconds=10)

        self._remove_state_listener = None

        # Initialize the debouncer
        self._debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=0.1,
            immediate=True,
            function=self.async_refresh,
        )

        # Initialize the save debouncer
        self._save_debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=30.0,
            immediate=False,
            function=self._save_debounced_data,
        )

        self._stored_data = None

        self._thread_lock = threading.Lock()

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
        _LOGGER.debug("Setting up AreaOccupancyCoordinator")

        # Load stored data first
        await self.async_load_stored_data()

        # Initialize states after loading stored data using the new state manager
        sensors = self.get_configured_sensors()
        await self.state_manager.initialize_states(self.hass, sensors)
        self._state_cache = await self.state_manager.get_active_sensors()

        # Calculate initial priors before scheduling updates
        await self.update_learned_priors()

        # Schedule prior updates
        self.async_track_prior_updates()

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

        except (ValueError, TypeError, KeyError) as err:
            _LOGGER.error("Error loading stored data: %s", err, exc_info=True)
            self._reset_state()
            self.type_priors = self._probabilities.get_initial_type_priors()

    async def async_initialize_states(self) -> None:
        """Initialize sensor states."""
        _LOGGER.debug("Initializing sensor states")
        try:
            sensors = self.get_configured_sensors()
            await self.state_manager.initialize_states(self.hass, sensors)
            self._state_cache = await self.state_manager.get_active_sensors()
            _LOGGER.info("Initialized states for sensors: %s", sensors)
        except (HomeAssistantError, ValueError, LookupError) as err:
            _LOGGER.error("Error initializing states: %s", err)

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
            period = history_period or self.config.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD)
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

        except Exception as err:
            _LOGGER.error("Error updating learned priors: %s", err, exc_info=True)

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes using state manager."""
        entities = self.get_configured_sensors()
        _LOGGER.debug("Setting up entity tracking for: %s", entities)

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        @callback
        def async_state_changed_listener(event) -> None:
            entity_id = event.data.get("entity_id")
            new_state = event.data.get("new_state")

            if not new_state or entity_id not in entities:
                return

            # Update sensor state in state manager
            self.hass.async_create_task(
                self.state_manager.update_sensor(entity_id, new_state)
            )

            # Update local cache after state change
            async def update_cache():
                self._state_cache = await self.state_manager.get_active_sensors()

            self.hass.async_create_task(update_cache())

            _LOGGER.debug("Scheduling refresh due to state change of %s", entity_id)
            self.hass.async_create_task(self._debouncer.async_call())

        self._remove_state_listener = async_track_state_change_event(
            self.hass,
            entities,
            async_state_changed_listener,
        )

    async def _async_update_data(self) -> ProbabilityResult:
        async with self._state_lock:
            try:
                if self.data and "probability" in self.data:
                    self._calculator.previous_probability = self.data["probability"]
                    _LOGGER.debug(
                        "Coordinator data probability: %s",
                        self._calculator.previous_probability,
                    )
                else:
                    self._calculator.previous_probability = (
                        self._probabilities.get_default_prior("dummy")
                    )
                    _LOGGER.debug(
                        "No coordinator data; using default prior: %s",
                        self._calculator.previous_probability,
                    )

                # Use cached active sensors from state manager
                active_sensors = self._state_cache
                now = dt_util.utcnow()
                _LOGGER.debug("Current time: %s", now)

                result = self._calculator.perform_calculation_logic(active_sensors, now)

                if result["probability"] >= self.threshold:
                    self._last_occupied = now
                await self._async_store_data()

                return result

            except (HomeAssistantError, ValueError, RuntimeError, KeyError) as err:
                _LOGGER.error("Error updating data: %s", err, exc_info=True)
                raise UpdateFailed(f"Error updating data: {err}") from err

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
        _LOGGER.debug("Unloading coordinator")

        if hasattr(self, "_prior_update_tracker"):
            self._prior_update_tracker()

        await self._save_debounced_data()

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        if self._debouncer is not None:
            await self._debouncer.async_shutdown()
            self._debouncer = None

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
        _LOGGER.debug("Resetting state")
        self._last_occupied = None
        self._last_save = dt_util.utcnow()
        self.learned_priors.clear()

    def register_entity(self, entity_id: str) -> None:
        _LOGGER.debug("Registering entity: %s", entity_id)
        self._entity_ids.add(entity_id)

    def unregister_entity(self, entity_id: str) -> None:
        _LOGGER.debug("Unregistering entity: %s", entity_id)
        self._entity_ids.discard(entity_id)

    async def async_refresh(self) -> None:
        await super().async_refresh()

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
            self.threshold = value / 100.0
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
        if not hasattr(self, "_stored_data") or self._stored_data is None:
            return

        try:
            async with self._storage_lock:
                await self.storage.async_save(self._stored_data)
                _LOGGER.debug("Successfully saved data to storage")
        except (ValueError, TypeError, KeyError) as err:
            _LOGGER.error("Error saving data to storage: %s", err, exc_info=True)
        except (IOError, HomeAssistantError) as err:
            _LOGGER.error("Storage I/O error: %s", err, exc_info=True)
        finally:
            self._stored_data = None

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
            self.hass.async_create_task(self.state_manager.remove_sensor(entity_id))

            async def update_cache():
                self._state_cache = await self.state_manager.get_active_sensors()
                await self.async_refresh()

            self.hass.async_create_task(update_cache())
