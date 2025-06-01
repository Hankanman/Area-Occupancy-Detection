"""Area Occupancy Coordinator."""

from __future__ import annotations

# Standard Library
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable

# Third Party
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .calculate_prior import PriorCalculator
from .calculate_prob import ProbabilityCalculator
from .configuration_manager import ConfigurationManager
from .const import (
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from .decay_manager import DecayManager
from .exceptions import CalculationError, StateError, StorageError
from .prior_manager import PriorManager
from .probabilities import Probabilities
from .state_manager import StateManager
from .storage import AreaOccupancyStorage, AreaOccupancyStore
from .types import MLHybridResult, ProbabilityState, SensorInfo, SensorInputs

# Conditional imports for ML functionality
try:
    from .ml_models import ModelManager

    ML_AVAILABLE = True
except ImportError:
    ModelManager = None
    ML_AVAILABLE = False

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
        self.hass = hass
        self.config_entry = config_entry
        self._storage_lock = asyncio.Lock()
        self._processing_state_change: bool = False
        self._state_change_pending: bool = False

        # Initialize configuration
        self.config = {**config_entry.data, **config_entry.options}

        # Initialize data first
        self.data = ProbabilityState()
        self.data.update(
            probability=MIN_PROBABILITY,
            previous_probability=MIN_PROBABILITY,
            threshold=self.config_entry.options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
            / 100.0,
            prior_probability=MIN_PROBABILITY,
            sensor_probabilities={},
            decay_status=0.0,
            is_occupied=False,
            decaying=False,
            decay_start_time=None,
            decay_start_probability=None,
        )

        # Create a synchronous wrapper for the async callback
        @callback
        def _sync_refresh() -> None:
            """Synchronous wrapper to trigger async refresh."""
            self.hass.async_create_task(self.async_request_refresh())

        # Initialize probabilities first
        self.probabilities = Probabilities(config=dict(self.config_entry.options))

        # Initialize unified state manager
        self.state_manager = StateManager(
            hass=self.hass,
            config=dict(self.config_entry.options),
            probabilities=self.probabilities,
        )

        # Initialize decay manager
        self.decay_manager = DecayManager(
            hass=self.hass,
            config=dict(self.config_entry.options),
            update_callback=_sync_refresh,
        )

        # Initialize remaining components
        self.storage = AreaOccupancyStore(self.hass)
        self.ml_storage = AreaOccupancyStorage(self.hass)

        # Initialize ML components if available
        self.model_manager = None
        if ML_AVAILABLE and ModelManager is not None:
            try:
                self.model_manager = ModelManager(hass, self.ml_storage)
            except Exception as err:
                _LOGGER.warning("Failed to initialize ML components: %s", err)

        self.calculator = ProbabilityCalculator(
            probabilities=self.probabilities,
            model_manager=self.model_manager,
            state_manager=self.state_manager,
        )
        self._prior_calculator = PriorCalculator(
            hass=self.hass,
            probabilities=self.probabilities,
            sensor_inputs=SensorInputs.from_config(dict(self.config_entry.options)),
            state_manager=self.state_manager,
        )

        # Initialize PriorManager
        prior_config = dict(self.config_entry.options)
        prior_config[CONF_NAME] = self.config_entry.data.get(
            CONF_NAME, "Area Occupancy"
        )
        self.prior_manager = PriorManager(
            hass=self.hass,
            config=prior_config,
            storage=self.storage,
            probabilities=self.probabilities,
            state_manager=self.state_manager,
            prior_calculator=self._prior_calculator,
            config_entry_id=self.config_entry.entry_id,
        )

        # Initialize ConfigurationManager last since it depends on all other components
        self.config_manager = ConfigurationManager(
            hass=self.hass,
            config_entry=self.config_entry,
            state_manager=self.state_manager,
            prior_manager=self.prior_manager,
            decay_manager=self.decay_manager,
            probabilities=self.probabilities,
            update_callback=_sync_refresh,
            prior_calculator=self._prior_calculator,
        )

    # --- Properties ---
    @property
    def entity_ids(self) -> set[str]:
        """Return the set of entity IDs being tracked."""
        return self.state_manager.get_tracked_entities()

    @property
    def available(self) -> bool:
        """Check if the coordinator is available."""
        if not self.data:
            return False
        primary_sensor = self.config_entry.options.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor:
            return False
        state_info = self.state_manager.get_entity_state(primary_sensor)
        return state_info.get("availability", False) if state_info else False

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for the coordinator."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.config_entry.entry_id)},
            name=self.config_entry.data.get(CONF_NAME, "Area Occupancy"),
            manufacturer=DEVICE_MANUFACTURER,
            model=DEVICE_MODEL,
            sw_version=DEVICE_SW_VERSION,
        )

    @property
    def prior_update_interval(self) -> timedelta:
        """Return the interval between prior updates."""
        return self.prior_manager.prior_update_interval

    @property
    def next_prior_update(self) -> datetime | None:
        """Return the next scheduled prior update time."""
        return self.prior_manager.next_prior_update

    @property
    def last_prior_update(self) -> str | None:
        """Return the timestamp the priors were last calculated."""
        return self.prior_manager.last_prior_update

    @property
    def probability(self) -> float:
        """Return the current occupancy probability (0.0-1.0)."""
        return self.data.probability if self.data else 0.0

    @property
    def is_occupied(self) -> bool:
        """Return the current occupancy state (True/False)."""
        return self.data.is_occupied if self.data else False

    @property
    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0)."""
        return self.data.threshold if self.data else 0.0

    # --- Public Methods ---
    async def async_setup(self) -> None:
        """Set up the coordinator, load data, initialize states, and setup prior management."""
        try:
            # Initialize states first
            sensors = SensorInputs.from_config(
                dict(self.config_entry.options)
            ).get_all_sensors()
            await self.state_manager.async_initialize_states(sensors)

            # Set up PriorManager - this handles loading stored data, initial calculations, and scheduling
            await self.prior_manager.async_setup()

            # Trigger an initial refresh after setup is complete
            await self.async_refresh()

            _LOGGER.debug(
                "Successfully set up AreaOccupancyCoordinator for %s",
                self.config_entry.data.get(CONF_NAME, "Area Occupancy"),
            )
        except (StorageError, StateError, CalculationError) as err:
            _LOGGER.error("Failed to set up coordinator: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up coordinator: {err}") from err

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # Shutdown DecayManager
        if self.decay_manager:
            self.decay_manager.shutdown()

        # Shutdown PriorManager
        if self.prior_manager:
            await self.prior_manager.async_shutdown()

        # Stop state tracking via StateManager
        self.state_manager.stop_state_tracking()

        await super().async_shutdown()

        # Clear data
        self.data = ProbabilityState()

    async def async_update_options(self) -> None:
        """Update coordinator options with improved error handling."""
        await self.config_manager.update_options()
        await self.async_refresh()

    async def async_update_threshold(self, value: float) -> None:
        """Update the threshold value."""
        await self.config_manager.update_threshold(value)

    async def update_learned_priors(self, history_period: int | None = None) -> None:
        """Update learned priors using historical data - delegated to PriorManager."""
        if self.prior_manager:
            await self.prior_manager.update_learned_priors(history_period)

    # --- Internal Update and State Handling Methods ---
    async def _async_update_data(self) -> ProbabilityState:
        """Update data with improved error handling."""
        if not self._can_update():
            return self.data if self.data else ProbabilityState()

        try:
            self._processing_state_change = True
            self._state_change_pending = False

            # Get current state and calculate probability
            current_states = self.state_manager.get_current_states()
            calculated_probability = await self._calculate_probability(current_states)

            # Apply decay and update state
            decay_result = self._apply_decay(calculated_probability)
            self._update_state(decay_result)

            # Handle reset if needed
            if self._should_reset(current_states):
                self._reset_state()

            return self.data

        except CalculationError as err:
            _LOGGER.error("Calculation Error caught in _async_update_data: %s", err)
            return self.data
        except Exception as err:
            _LOGGER.exception("Error updating occupancy data: %s", err)
            return self.data
        finally:
            self._processing_state_change = False

    def _can_update(self) -> bool:
        """Check if update can proceed."""
        if not self.data or not self.prior_manager:
            _LOGGER.warning(
                "_async_update_data called but coordinator data/prior_manager is not initialized"
            )
            return False
        if self._processing_state_change:
            _LOGGER.debug("Already processing state change, skipping update")
            return False
        return True

    async def _calculate_probability(
        self, current_states: dict[str, SensorInfo]
    ) -> float:
        """Calculate probability from current states."""
        calculation_result = await self.calculator.calculate_occupancy_probability(
            current_states, self.prior_manager.prior_state
        )
        return (
            calculation_result.final_probability
            if isinstance(calculation_result, MLHybridResult)
            else calculation_result.calculated_probability
        )

    def _apply_decay(
        self, calculated_probability: float
    ) -> tuple[float, float, bool, datetime | None, float | None]:
        """Apply decay to calculated probability."""
        return self.decay_manager.calculate_decay(
            current_probability=calculated_probability,
            previous_probability=self.data.probability,
            is_decaying=self.data.decaying,
            decay_start_time=self.data.decay_start_time,
            decay_start_probability=self.data.decay_start_probability,
        )

    def _update_state(
        self, decay_result: tuple[float, float, bool, datetime | None, float | None]
    ) -> None:
        """Update state with decay results."""
        (
            decayed_probability,
            decay_factor,
            new_is_decaying,
            new_decay_start_time,
            new_decay_start_probability,
        ) = decay_result

        final_probability = max(
            MIN_PROBABILITY, min(decayed_probability, MAX_PROBABILITY)
        )
        final_is_occupied = final_probability >= self.data.threshold
        decay_status_percent = (1.0 - decay_factor) * 100.0

        self.data.update(
            probability=final_probability,
            previous_probability=self.data.probability,
            decay_status=decay_status_percent,
            is_occupied=final_is_occupied,
            decaying=new_is_decaying,
            decay_start_time=new_decay_start_time,
            decay_start_probability=new_decay_start_probability,
        )

        _LOGGER.debug(
            "Status: probability=%.3f threshold=%.3f decay_status=%.3f%% decaying=%s is_occupied=%s",
            self.data.probability,
            self.data.threshold,
            self.data.decay_status,
            self.data.decaying,
            self.data.is_occupied,
        )

        self.decay_manager.start_decay_updates(self.data.decaying)

    def _should_reset(self, current_states: dict[str, SensorInfo]) -> bool:
        """Check if state should be reset."""
        if self.data.decaying:
            return False
        return not any(
            info.get("availability", False)
            and self.state_manager.is_entity_active(entity_id, info.get("state"))
            for entity_id, info in current_states.items()
            if info
        )

    def _reset_state(self) -> None:
        """Reset state to initial values."""
        _LOGGER.debug("Resetting state: no active sensors, decay complete/inactive")
        self.data.probability = MIN_PROBABILITY
        self.data.is_occupied = False
        self.data.decaying = False
        self.data.decay_start_time = None
        self.data.decay_status = 0.0
        self.decay_manager.stop_decay_updates()

    # --- Listener Handling ---
    @callback
    def _async_refresh_finished(self) -> None:
        """Handle when a refresh has finished."""
        if self.last_update_success:
            _LOGGER.debug("Coordinator refresh finished successfully")
        else:
            _LOGGER.warning("Coordinator refresh failed")

    @callback
    def async_set_updated_data(self, data: ProbabilityState) -> None:
        """Manually update data and notify listeners."""
        super().async_set_updated_data(data)

    @callback
    def async_add_listener(
        self, update_callback: CALLBACK_TYPE, context: Any = None
    ) -> Callable[[], None]:
        """Add a listener for data updates with improved tracking."""
        return super().async_add_listener(update_callback, context)
