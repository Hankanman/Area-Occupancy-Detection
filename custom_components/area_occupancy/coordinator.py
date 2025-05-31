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
from homeassistant.exceptions import (
    ConfigEntryError,
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .calculate_prior import PriorCalculator
from .calculate_prob import ProbabilityCalculator
from .const import (
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_THRESHOLD,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from .decay_handler import DecayHandler
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
        # Merge data and options for the effective configuration
        self.config = {**config_entry.data, **config_entry.options}
        self._storage_lock = asyncio.Lock()
        self._decay_unsub: CALLBACK_TYPE | None = None

        # Initialize sensor inputs with validation
        try:
            self.inputs = SensorInputs.from_config(self.config)
        except (ValueError, TypeError) as err:
            raise ConfigEntryError(f"Invalid sensor configuration: {err}") from err

        # Initialize tracking
        self._processing_state_change: bool = (
            False  # Flag to prevent recursive state changes
        )
        self._state_change_pending: bool = (
            False  # Flag to indicate state changes need processing
        )

        # Initialize data first
        self.data = ProbabilityState()
        self.data.update(
            probability=MIN_PROBABILITY,
            previous_probability=MIN_PROBABILITY,
            threshold=self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD) / 100.0,
            prior_probability=MIN_PROBABILITY,
            sensor_probabilities={},
            decay_status=0.0,
            is_occupied=False,
            decaying=False,
            decay_start_time=None,
            decay_start_probability=None,
        )

        # Initialize probabilities first
        self.probabilities = Probabilities(config=self.config)

        # Initialize unified state manager
        self.state_manager = StateManager(
            hass=self.hass,
            config=self.config,
            probabilities=self.probabilities,
        )

        # Set decay configuration
        self.config[CONF_DECAY_ENABLED] = self.config.get(
            CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED
        )
        self.config[CONF_DECAY_WINDOW] = self.config.get(
            CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW
        )

        # Initialize remaining components
        self.storage = AreaOccupancyStore(self.hass)
        self.ml_storage = AreaOccupancyStorage(self.hass)
        self.decay_handler = DecayHandler(self.config)

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
            sensor_inputs=self.inputs,
            state_manager=self.state_manager,
        )

        # Initialize PriorManager to handle all prior-related operations
        self.prior_manager: PriorManager = PriorManager(
            hass=self.hass,
            config=self.config,
            storage=self.storage,
            probabilities=self.probabilities,
            state_manager=self.state_manager,
            prior_calculator=self._prior_calculator,
            config_entry_id=self.config_entry.entry_id,
        )

    # --- Properties ---
    @property
    def entity_ids(self) -> set[str]:
        """Return the set of entity IDs being tracked."""
        return set(self.state_manager.get_tracked_entities())

    @property
    def available(self) -> bool:
        """Check if the coordinator is available."""
        primary_sensor = self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor or not self.data:
            return False
        # Check availability directly from StateManager
        primary_state_info = self.state_manager.get_entity_state(primary_sensor)
        return (
            primary_state_info.get("availability", False)
            if primary_state_info
            else False
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for the coordinator."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.config_entry.entry_id)},
            name=self.config[CONF_NAME],
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
            sensors = self.get_configured_sensors()
            await self.async_initialize_states(sensors)

            # Set up PriorManager - this handles loading stored data, initial calculations, and scheduling
            await self.prior_manager.async_setup()

            # Trigger an initial refresh after setup is complete
            await self.async_refresh()

            _LOGGER.debug(
                "Successfully set up AreaOccupancyCoordinator for %s",
                self.config[CONF_NAME],
            )
        except (StorageError, StateError, CalculationError) as err:
            _LOGGER.error("Failed to set up coordinator: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up coordinator: {err}") from err

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        self._stop_decay_updates()

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
        try:
            _LOGGER.debug(
                "Coordinator async_update_options starting with config: %s", self.config
            )

            # Shutdown the old PriorManager to clean up timers
            if self.prior_manager:
                await self.prior_manager.async_shutdown()

            # Update configuration first
            self.config = {**self.config_entry.data, **self.config_entry.options}
            _LOGGER.debug("Updated config: %s", self.config)

            # Reinitialize all components with new configuration
            _LOGGER.debug("Reinitializing all components with new configuration")

            # Reset state tracking (preserves current probability if available)
            current_prob = self.data.probability if self.data else MIN_PROBABILITY
            previous_prob = (
                self.data.previous_probability if self.data else MIN_PROBABILITY
            )
            is_occupied_state = self.data.is_occupied if self.data else False
            decay_status_state = self.data.decay_status if self.data else 0.0
            decaying_state = self.data.decaying if self.data else False
            decay_start_time_state = self.data.decay_start_time if self.data else None
            decay_start_probability_state = (
                self.data.decay_start_probability if self.data else None
            )
            sensor_probabilities_dict = (
                self.data.sensor_probabilities if self.data else {}
            )

            self.data = ProbabilityState()
            self.data.update(
                probability=current_prob,  # Keep current probability
                previous_probability=previous_prob,
                threshold=self.config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
                / 100.0,  # Update threshold
                prior_probability=self.prior_manager.prior_state.overall_prior
                if self.prior_manager and self.prior_manager.prior_state
                else MIN_PROBABILITY,
                sensor_probabilities=sensor_probabilities_dict,
                decay_status=decay_status_state,
                is_occupied=is_occupied_state,  # Keep current occupancy state initially
                decaying=decaying_state,
                decay_start_time=decay_start_time_state,
                decay_start_probability=decay_start_probability_state,
            )

            # Reset components that depend on config
            self.inputs = SensorInputs.from_config(self.config)
            self.probabilities = Probabilities(config=self.config)
            self.decay_handler = DecayHandler(self.config)
            # Note: PriorManager does not directly depend on config options, only data (history_period)
            # which isn't changeable via options flow yet. If it were, PriorManager would need reset here.

            # Re-initialize tracked sensor states based on the NEW configuration
            _LOGGER.debug("Re-initializing sensor states after options update")
            new_sensor_ids = self.get_configured_sensors()
            await self.async_initialize_states(new_sensor_ids)
            _LOGGER.debug("Sensor states re-initialized")

            # Reinitialize the calculators with updated references
            self.calculator = ProbabilityCalculator(
                probabilities=self.probabilities,
                model_manager=self.model_manager,
                state_manager=self.state_manager,
            )
            self._prior_calculator = PriorCalculator(
                hass=self.hass,
                probabilities=self.probabilities,
                sensor_inputs=self.inputs,
                state_manager=self.state_manager,
            )

            # Reinitialize PriorManager with updated references
            self.prior_manager = PriorManager(
                hass=self.hass,
                config=self.config,
                storage=self.storage,
                probabilities=self.probabilities,
                state_manager=self.state_manager,
                prior_calculator=self._prior_calculator,
                config_entry_id=self.config_entry.entry_id,
            )

            _LOGGER.info(
                "Coordinator options successfully updated and components reinitialized"
            )

        except (ValueError, KeyError) as err:
            _LOGGER.error("Invalid configuration in async_update_options: %s", err)
            raise ConfigEntryError(f"Invalid configuration: {err}") from err
        except HomeAssistantError as err:
            _LOGGER.error("Failed to update coordinator options: %s", err)
            raise ConfigEntryNotReady(
                f"Failed to update coordinator options: {err}"
            ) from err

    async def async_update_threshold(self, value: float) -> None:
        """Update the threshold value.

        Args:
            value: The new threshold value as a percentage (1-99)

        Raises:
            ServiceValidationError: If the value is invalid
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
            raise ServiceValidationError(f"Failed to update threshold: {err}") from err
        except Exception as err:
            raise HomeAssistantError(f"Failed to update threshold: {err}") from err

    async def update_learned_priors(self, history_period: int | None = None) -> None:
        """Update learned priors using historical data - delegated to PriorManager."""
        if self.prior_manager:
            await self.prior_manager.update_learned_priors(history_period)

    async def async_initialize_states(self, sensor_ids: list[str]) -> None:
        """Initialize sensor states using StateManager."""
        try:
            # Use StateManager for state initialization and tracking
            await self.state_manager.async_initialize_states(sensor_ids)

            # Set up callback for state changes to trigger calculations
            def state_change_callback(entity_id: str, sensor_info: SensorInfo) -> None:
                """Handle state changes from StateManager."""
                # Simple flag to prevent recursive calls
                if self._processing_state_change:
                    _LOGGER.debug(
                        "State change already being processed, skipping %s", entity_id
                    )
                    return

                try:
                    # Just set a flag - don't trigger immediate refresh
                    # Let the normal update cycle handle it
                    self._state_change_pending = True
                    _LOGGER.debug(
                        "State change detected for %s, will process on next update cycle",
                        entity_id,
                    )

                except Exception as err:
                    _LOGGER.exception(
                        "Error in state change callback for %s: %s", entity_id, err
                    )

            # Setup state tracking with callback
            self.state_manager.setup_state_tracking(sensor_ids, state_change_callback)

            _LOGGER.debug("State management initialized via StateManager")

        except Exception as err:
            raise StateError(f"Failed to initialize states: {err}") from err

    def get_configured_sensors(self) -> list[str]:
        """Get all configured sensors including the primary occupancy sensor."""
        return self.inputs.get_all_sensors()

    # --- Internal Update and State Handling Methods ---
    async def _async_update_data(self) -> ProbabilityState:
        """Update data with improved error handling."""
        if not self.data or not self.prior_manager:
            _LOGGER.warning(
                "_async_update_data called but coordinator data/prior_manager is not initialized"
            )
            return self.data if self.data else ProbabilityState()

        # Check if we're already processing to prevent loops
        if self._processing_state_change:
            _LOGGER.debug("Already processing state change, skipping update")
            return self.data

        try:
            # Set processing flag
            self._processing_state_change = True

            # Get current states directly from StateManager when needed
            # No need to store copies in coordinator data
            current_states_snapshot = self.state_manager.get_current_states()

            # Clear the pending flag since we're processing now
            self._state_change_pending = False

            # --- Store initial probability state before calculation ---
            initial_prob = self.data.probability
            initial_decaying = self.data.decaying
            initial_decay_start_time = self.data.decay_start_time
            initial_decay_start_prob = self.data.decay_start_probability

            # --- Calculate Undecayed Probability ---
            try:
                # Pass snapshot, prior state, and config to the calculator
                calc_result = await self.calculator.calculate_occupancy_probability(
                    current_states_snapshot,  # Pass the snapshot
                    self.prior_manager.prior_state,  # Pass the prior_state, not the manager
                    self.config,  # Pass the config for ML settings
                )
            except (CalculationError, ValueError, ZeroDivisionError) as calc_err:
                # Log the specific calculation error and return existing data
                _LOGGER.error("Error during probability calculation: %s", calc_err)
                # Keep existing data, but notify listeners
                self.async_set_updated_data(self.data)
                return self.data

            # Extract values from result based on type (OccupancyCalculationResult or MLHybridResult)
            if isinstance(calc_result, MLHybridResult):
                calculated_probability = calc_result.final_probability
                # For ML hybrid results, we need to get prior and sensor probabilities separately
                # Since they're not included in MLHybridResult, use bayesian probability
                prior_probability = (
                    calc_result.bayesian_probability
                )  # Use Bayesian as fallback
                sensor_probabilities = {}  # Empty for ML results
            else:  # OccupancyCalculationResult
                calculated_probability = calc_result.calculated_probability
                prior_probability = calc_result.prior_probability
                sensor_probabilities = calc_result.sensor_probabilities

            # Update self.data with intermediate results *before* decay
            self.data.update(
                probability=calculated_probability,  # Store undecayed prob temporarily
                prior_probability=prior_probability,
                sensor_probabilities=sensor_probabilities,
            )

            # --- Apply Decay ---
            (
                decayed_probability,
                decay_factor,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            ) = self.decay_handler.calculate_decay(
                current_probability=calculated_probability,  # Use the extracted probability
                previous_probability=initial_prob,  # Use the state before this update cycle
                is_decaying=initial_decaying,
                decay_start_time=initial_decay_start_time,
                decay_start_probability=initial_decay_start_prob,
            )
            decay_status_percent = (1.0 - decay_factor) * 100.0

            # --- Update Final State ---
            final_probability = max(
                MIN_PROBABILITY, min(decayed_probability, MAX_PROBABILITY)
            )
            final_is_occupied = final_probability >= self.data.threshold

            # Update the main data object with final results
            # Note: prior_probability and sensor_probabilities were already updated above
            self.data.update(
                probability=final_probability,  # Use the decayed probability
                previous_probability=initial_prob,  # Keep initial prob as previous
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

            # Manage decay timer
            if self.data.decaying:
                if not self._decay_unsub:
                    _LOGGER.debug("Starting decay timer (decaying is True)")
                    self._start_decay_updates()
            elif self._decay_unsub:
                _LOGGER.debug("Stopping decay timer (decaying is False)")
                self._stop_decay_updates()

            # --- Reset Logic ---
            # Determine active sensors based on the *snapshot* used for calculation
            active_sensors_exist = any(
                info.get("availability", False)
                and self.state_manager.is_entity_active(entity_id, info.get("state"))
                for entity_id, info in current_states_snapshot.items()  # Use snapshot
                if info
            )
            should_reset = not self.data.decaying and not active_sensors_exist

            if should_reset:
                _LOGGER.debug(
                    "Resetting state: no active sensors, decay complete/inactive"
                )
                self.data.probability = MIN_PROBABILITY
                self.data.is_occupied = False
                self.data.decaying = False
                self.data.decay_start_time = None
                self.data.decay_status = 0.0  # Reset decay status to 0
                self._stop_decay_updates()

        except CalculationError as err:
            _LOGGER.error("Calculation Error caught in _async_update_data: %s", err)
            # Return existing data
            return self.data
        except (
            TimeoutError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            HomeAssistantError,
            asyncio.CancelledError,
        ):  # Catch other potential errors
            _LOGGER.exception("Error updating occupancy data: %s")
            # Return existing data on general errors
            return self.data
        finally:
            # Always clear the processing flag
            self._processing_state_change = False

        # Return the updated self.data object on success
        return self.data

    # --- Decay Handling ---
    def _start_decay_updates(self) -> None:
        """Start regular decay updates every 5 seconds."""
        if not self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED):
            _LOGGER.debug("Decay updates disabled by configuration")
            return

        if self._decay_unsub is not None:
            _LOGGER.debug("Decay timer already running")
            return

        _LOGGER.debug("Starting decay update timer")
        interval = timedelta(seconds=5)

        async def _async_do_decay_update(*_) -> None:
            """Execute decay update."""
            try:
                if self.data and self.data.decaying:
                    _LOGGER.debug("Decay timer fired. Triggering refresh")
                    await self.async_request_refresh()
                else:
                    _LOGGER.debug(
                        "Decay timer fired, but decay is not active. Stopping timer"
                    )
                    self._stop_decay_updates()

            except Exception:
                _LOGGER.exception("Error in decay update task")
                self._stop_decay_updates()

        self._decay_unsub = async_track_time_interval(
            self.hass, _async_do_decay_update, interval
        )

    def _stop_decay_updates(self) -> None:
        """Stop decay updates."""
        if self._decay_unsub is not None:
            _LOGGER.debug("Stopping decay update timer")
            self._decay_unsub()
            self._decay_unsub = None

    # --- Data Saving ---
    async def _async_save_prior_state_data(self) -> None:
        """Save the current prior state data to storage."""
        if not self.prior_manager or not self.prior_manager.prior_state:
            _LOGGER.warning(
                "Attempted to save prior state, but prior_manager or prior_state is None"
            )
            return

        try:
            _LOGGER.debug("Attempting to save prior state data")
            await self.storage.async_save_instance_prior_state(
                self.config_entry.entry_id,
                self.config.get(CONF_NAME, "Unknown Area"),
                self.prior_manager.prior_state,  # Pass the prior_state, not the manager
            )
            _LOGGER.debug("Prior state data saved successfully")
        except (TimeoutError, HomeAssistantError, ValueError, RuntimeError) as err:
            _LOGGER.error("Failed to save prior state data: %s", err)
            raise StorageError(f"Failed to save prior state data: {err}") from err

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
