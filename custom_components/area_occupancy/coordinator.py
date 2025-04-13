"""Area Occupancy Coordinator."""

from __future__ import annotations

# Standard Library
import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
import logging
from typing import Any

# Third Party
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

# Local
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
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from .decay_handler import DecayHandler
from .exceptions import CalculationError, StateError, StorageError
from .probabilities import Probabilities
from .storage import AreaOccupancyStore
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
        self.hass = hass
        self.config_entry = config_entry
        # Merge data and options for the effective configuration
        self.config = {**config_entry.data, **config_entry.options}
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

        # Initialize tracking
        self._remove_state_listener = None
        self._last_prior_update: str | None = None  # Store last prior calc time

        # Initialize state management directly in the coordinator
        # self._all_sensor_states: dict[str, dict] = {}
        # self._active_sensors: dict[str, dict] = {}

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
            decay_start_probability=None,
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
        self.storage = AreaOccupancyStore(self.hass)
        self.decay_handler = DecayHandler(self.config)
        self.calculator = ProbabilityCalculator(
            prior_state=self.prior_state,
            probabilities=self.probabilities,
        )
        self._prior_calculator = PriorCalculator(
            coordinator=self,
            probabilities=self.probabilities,
            hass=self.hass,
        )

    # --- Properties ---
    @property
    def entity_ids(self) -> set[str]:
        """Return the set of entity IDs being tracked."""
        return self._entity_ids

    @property
    def available(self) -> bool:
        """Check if the coordinator is available."""
        primary_sensor = self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor or not self.data:
            return False
        # Check availability directly from self.data.current_states
        primary_state_info = self.data.current_states.get(primary_sensor, {})
        return primary_state_info.get("availability", False)

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
    def prior_update_interval(self) -> timedelta:
        """Return the interval between prior updates."""
        return self._prior_update_interval

    @property
    def next_prior_update(self) -> datetime | None:
        """Return the next scheduled prior update time."""
        return self._next_prior_update

    @property
    def last_prior_update(self) -> str | None:
        """Return the timestamp the priors were last calculated."""
        return self._last_prior_update

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
        """Set up the coordinator, load data, initialize states, check priors, and schedule updates."""
        try:
            # Load stored data first
            await self.async_load_stored_data()

            # Initialize states after loading stored data
            sensors = self.get_configured_sensors()
            await self.async_initialize_states(sensors)

            # Determine if priors need calculation at startup
            force_prior_update = False
            reason = ""

            if not self.prior_state:
                # Condition 1: Prior state object missing
                force_prior_update = True
                reason = "Prior state object not loaded or initialized"
            else:
                # Prior state object exists, check its completeness first
                is_incomplete = (
                    not self.prior_state.type_priors
                    or self.prior_state.type_priors == {}
                    or not self.prior_state.entity_priors
                    or self.prior_state.entity_priors == {}
                    or not self.prior_state.last_updated
                    # Corrected timestamp check - compare stored datetime with interval
                    or (
                        self.prior_state.last_updated
                        and isinstance(self.prior_state.last_updated, dict)
                        and not self.prior_state.last_updated
                    )  # Check if last_updated dict exists but is empty
                    or self.prior_state.overall_prior in (MIN_PROBABILITY, 0.0)
                )

                if is_incomplete:
                    # Condition 2: Prior state object exists but is incomplete
                    force_prior_update = True
                    reason = "Loaded prior state is incomplete or at minimum"
                # Condition 3 & 4: Prior state exists AND is complete, now check timestamp age
                # Use the _last_prior_update variable loaded alongside the state
                elif (
                    not self._last_prior_update
                ):  # Check coordinator's internal timestamp
                    # Timestamp wasn't loaded/found (should ideally not happen if state is complete)
                    force_prior_update = True
                    reason = "No last update timestamp found for complete prior state"
                else:
                    try:
                        last_update_dt = dt_util.parse_datetime(self._last_prior_update)
                        if not last_update_dt:
                            force_prior_update = True
                            reason = (
                                "Failed to get valid datetime from parsed timestamp"
                            )
                        # Check age against coordinator's interval property
                        elif (
                            dt_util.utcnow() - last_update_dt
                            >= self.prior_update_interval
                        ):
                            force_prior_update = True
                            reason = "Last update is older than the update interval"
                        # else: force_prior_update remains False (prior is complete and recent)

                    except (TypeError, ValueError):
                        _LOGGER.warning(
                            "Could not parse stored last prior update time: %s",
                            self._last_prior_update,
                        )
                        force_prior_update = True  # Force update if parsing fails
                        reason = "Failed to parse last update timestamp"

            # Perform prior update if needed
            if force_prior_update:
                _LOGGER.debug(
                    "Performing initial prior calculation at startup: %s", reason
                )
                # Call coordinator's method
                await self.update_learned_priors()
            else:
                # Add log message for skipping case
                _LOGGER.debug(
                    "Skipping initial prior calculation: Existing priors are complete and recent (last update: %s)",
                    self._last_prior_update or "N/A",
                )

            # Schedule periodic prior updates regardless of initial calculation
            await self._schedule_next_prior_update()
            # --- End Prior Logic ---

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

        # Cancel prior update tracker
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        await super().async_shutdown()

        # Additional cleanup specific to our use case
        if hasattr(self, "_remove_state_listener"):
            self._remove_state_listener()
            self._remove_state_listener = None

        # Save final state directly on shutdown
        # No need to save priors here, they are saved after calculation
        # await self._async_save_prior_state_data()

        # Clear data
        self.data = None
        self.prior_state = None
        # self._all_sensor_states = {}
        # self._active_sensors = {}
        self._entity_ids = set()

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

            # Reset state tracking (preserves current probability if available)
            current_prob = self.data.probability if self.data else MIN_PROBABILITY
            previous_prob = (
                self.data.previous_probability if self.data else MIN_PROBABILITY
            )
            is_occupied_state = self.data.is_occupied if self.data else False
            decay_status_state = self.data.decay_status if self.data else 0.0
            decaying_state = self.data.decaying if self.data else False
            decay_start_time_state = self.data.decay_start_time if self.data else None
            current_states_dict = self.data.current_states if self.data else {}
            previous_states_dict = self.data.previous_states if self.data else {}
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
                prior_probability=self.prior_state.overall_prior
                if self.prior_state
                else MIN_PROBABILITY,
                sensor_probabilities=sensor_probabilities_dict,
                decay_status=decay_status_state,
                current_states=current_states_dict,
                previous_states=previous_states_dict,
                is_occupied=is_occupied_state,  # Keep current occupancy state initially
                decaying=decaying_state,
                decay_start_time=decay_start_time_state,
                decay_start_probability=decay_start_probability_state,
            )

            # Reset components that depend on config
            self.inputs = SensorInputs.from_config(self.config)
            self.probabilities = Probabilities(config=self.config)
            self.decay_handler = DecayHandler(self.config)
            # Note: PriorState does not directly depend on config options, only data (history_period)
            # which isn't changeable via options flow yet. If it were, PriorState would need reset here.

            # Re-initialize tracked sensor states based on the NEW configuration
            _LOGGER.debug("Re-initializing sensor states after options update")
            new_sensor_ids = self.get_configured_sensors()
            await self.async_initialize_states(new_sensor_ids)
            _LOGGER.debug("Sensor states re-initialized")

            # Reinitialize the calculator with updated references
            self.calculator = ProbabilityCalculator(
                prior_state=self.prior_state,
                probabilities=self.probabilities,
            )

            # Don't save priors here, options change doesn't affect them directly
            # _LOGGER.debug("Saving prior state after options update (if reset occurred)")
            # await self._async_save_prior_state_data()
            # _LOGGER.debug("Prior state save complete after options update")

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
            # No direct state update here
            # No direct refresh trigger here

        except ValueError as err:
            raise ServiceValidationError(f"Failed to update threshold: {err}") from err
        except Exception as err:
            raise HomeAssistantError(f"Failed to update threshold: {err}") from err

    async def async_load_stored_data(self) -> None:
        """Load and restore data from storage."""
        try:
            _LOGGER.debug("Loading stored data from storage")

            # Use the store's instance-specific load method
            (
                name,
                stored_prior_state,
                last_updated_ts,
            ) = await self.storage.async_load_instance_prior_state(
                self.config_entry.entry_id
            )

            if stored_prior_state:
                _LOGGER.debug(
                    "Found stored prior state for instance %s, restoring (last saved: %s)",
                    self.config_entry.entry_id,
                    last_updated_ts,
                )
                self.prior_state = stored_prior_state
                self._last_prior_update = last_updated_ts  # Store the loaded timestamp
            else:
                _LOGGER.info(
                    "No stored prior state found for instance %s, initializing with defaults",
                    self.config_entry.entry_id,
                )
                self._last_prior_update = None  # Ensure it's None if no data loaded
                # Initialize data state (only if priors aren't loaded)
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
                    decay_start_probability=None,
                )

                # Reset prior state object
                self.prior_state = PriorState()
                self.prior_state.initialize_from_defaults(self.probabilities)
                self.prior_state.update(
                    analysis_period=self.config.get(
                        CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                    ),
                )

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
            self._last_prior_update = None  # Ensure it's None on error
            # Initialize with defaults on storage error
            self.prior_state = PriorState()
            self.prior_state.initialize_from_defaults(self.probabilities)
            self.prior_state.update(
                analysis_period=self.config.get(
                    CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                ),
            )
            # Re-raise as ConfigEntryNotReady if loading fails critically
            raise ConfigEntryNotReady(f"Failed to load stored data: {err}") from err

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
            if not isinstance(period, (int, float)) or period <= 0:
                _LOGGER.warning(
                    "Invalid history period configured: %s. Disabling prior calculation",
                    period,
                )
                # Set last update time to now to prevent immediate retry
                self._last_prior_update = dt_util.utcnow().isoformat()
                # Ensure schedule continues
                await self._schedule_next_prior_update()
                return

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
            if not sensors:
                _LOGGER.warning("No sensors configured for prior calculation")
                self._last_prior_update = dt_util.utcnow().isoformat()
                # Ensure schedule continues
                await self._schedule_next_prior_update()
                return

            _LOGGER.debug("Calculating prior for %s sensors", len(sensors))
            calculation_successful = True
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
                        "Calculated probabilities for %s - prob_given_true: %f, "
                        "prob_given_false: %f, prior: %f",
                        sensor_id,
                        prob_given_true,
                        prob_given_false,
                        prior,
                    )
                except Exception:
                    _LOGGER.exception("Error calculating prior for %s", sensor_id)
                    # Mark as unsuccessful but continue with other sensors
                    calculation_successful = False
                    # Skip updating this specific sensor's prior
                    continue
                    # raise PriorCalculationError(f"Failed to calculate prior for {sensor_id}: {err}") from err

                # Update the prior state with entity prior
                _LOGGER.debug("Updating prior state for %s", sensor_id)
                self.prior_state.update_entity_prior(
                    sensor_id,
                    prob_given_true,
                    prob_given_false,
                    prior,
                    dt_util.utcnow().isoformat(),
                )

            # Calculate the overall prior only if at least one sensor calculation succeeded
            if calculation_successful:
                _LOGGER.debug("Calculating overall prior")
                overall_prior = self.prior_state.calculate_overall_prior()
                _LOGGER.debug(
                    "Overall prior calculated: %f",
                    overall_prior,
                )
                self.prior_state.update(overall_prior=overall_prior)
            else:
                _LOGGER.warning(
                    "Overall prior not recalculated due to errors in sensor prior calculations"
                )

            # Save the updated priors directly if calculation was successful
            if calculation_successful:
                _LOGGER.debug("Saving prior state immediately after update")
                await self._async_save_prior_state_data()
                _LOGGER.debug("Prior update save complete")
                # Update the timestamp only after successful save
                self._last_prior_update = dt_util.utcnow().isoformat()
            else:
                _LOGGER.warning("Prior state not saved due to calculation errors")
                # Still update timestamp to prevent immediate retry loop
                self._last_prior_update = dt_util.utcnow().isoformat()

        except Exception:
            _LOGGER.exception("Unexpected error during update_learned_priors")
            # Update timestamp to prevent immediate retry loop even on unexpected errors
            self._last_prior_update = dt_util.utcnow().isoformat()
            # Re-raise the error so it can be handled upstream if needed
            # raise CalculationError(f"Failed to update learned priors: {err}") from err
        finally:
            # Always ensure the next update is scheduled, even if errors occurred
            # This prevents the schedule from stopping completely
            # Note: This was moved from _handle_prior_update
            await self._schedule_next_prior_update()

    async def async_initialize_states(self, sensor_ids: list[str]) -> None:
        """Initialize sensor states directly in the coordinator's data object."""
        try:
            async with self._state_lock:
                # Clear existing states in self.data
                self.data.current_states.clear()
                self.data.previous_states.clear()  # Also clear previous on full re-init

                for entity_id in sensor_ids:
                    state_obj = self.hass.states.get(entity_id)
                    is_available = bool(
                        state_obj
                        and state_obj.state not in ["unknown", "unavailable", None, ""]
                    )
                    # Use friendly name for logging if available
                    friendly_name = state_obj.name if state_obj else entity_id
                    _LOGGER.debug(
                        "Initializing state for %s (%s): available=%s, state=%s",
                        friendly_name,
                        entity_id,
                        is_available,
                        state_obj.state if state_obj else "None",
                    )
                    sensor_info = {
                        "state": state_obj.state if is_available else None,
                        "last_changed": (
                            state_obj.last_changed.isoformat()
                            if state_obj and state_obj.last_changed
                            else dt_util.utcnow().isoformat()  # Use UTC now for consistency
                        ),
                        "availability": is_available,
                    }
                    # Store directly in self.data.current_states
                    self.data.current_states[entity_id] = sensor_info
                    # Do not populate self.data.previous_states here, it's handled in the update cycle

            # Setup tracking after initialization
            self._setup_entity_tracking()

        except Exception as err:
            raise StateError(f"Failed to initialize states: {err}") from err

    def get_configured_sensors(self) -> list[str]:
        """Get all configured sensors including the primary occupancy sensor."""
        return self.inputs.get_all_sensors()

    # --- Internal Update and State Handling Methods ---
    async def _async_update_data(self) -> ProbabilityState:
        """Update data with improved error handling."""
        try:
            # --- Update self.data state dictionaries first (under lock) ---
            # This is now primarily handled by the listener. Here we just need to
            # manage the previous_states before calculation.
            async with self._state_lock:
                # Update previous_states in self.data based on its *current* current_states
                # before the calculation modifies anything
                self.data.previous_states = self.data.current_states.copy()
                # Capture a snapshot of current states for the calculation
                # Note: The calculator will now receive the self.data object directly

            # --- Store initial probability state before calculation ---
            initial_prob = self.data.probability
            initial_decaying = self.data.decaying
            initial_decay_start_time = self.data.decay_start_time
            initial_decay_start_prob = self.data.decay_start_probability

            # --- Calculate Undecayed Probability ---
            # Pass the *entire* self.data object to the calculator.
            # It will read current_states and update its internal fields.
            self.calculator.calculate_occupancy_probability(
                self.data,  # Pass the state object
                dt_util.now(),  # Use timezone-aware now
            )
            # The calculator modifies self.data directly now, so undecayed_prob is self.data.probability
            undecayed_prob = self.data.probability
            # The 'previous_probability' stored by the calculator reflects the state *before* its calculation.
            # We use the 'initial_prob' captured at the start of *this* method for decay.

            # --- Apply Decay ---
            (
                decayed_probability,
                decay_factor,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            ) = self.decay_handler.calculate_decay(
                current_probability=undecayed_prob,  # Use the undecayed result from self.data
                previous_probability=initial_prob,  # Use the state before this update cycle
                is_decaying=initial_decaying,
                decay_start_time=initial_decay_start_time,
                decay_start_probability=initial_decay_start_prob,
            )
            decay_status_percent = (1.0 - decay_factor) * 100.0

            # --- Update Final State ---
            # Clamp final probability
            final_probability = max(
                MIN_PROBABILITY, min(decayed_probability, MAX_PROBABILITY)
            )

            # Determine final occupancy state
            final_is_occupied = final_probability >= self.data.threshold

            # Update the main data object with merged results
            self.data.update(
                probability=final_probability,  # Use the decayed probability
                previous_probability=initial_prob,  # Store the probability from *before* this cycle
                # Threshold, Prior, Sensor Probs were updated by calculator
                decay_status=decay_status_percent,
                # current_states are updated by the listener
                # previous_states were updated earlier
                is_occupied=final_is_occupied,  # Use calculated value
                decaying=new_is_decaying,  # Use result from decay handler
                decay_start_time=new_decay_start_time,  # Use result from decay handler
                decay_start_probability=new_decay_start_probability,  # Use result from decay handler
            )

            _LOGGER.debug(
                "Status: probability=%.3f threshold=%.3f decay_status=%.3f%% decaying=%s is_occupied=%s",
                self.data.probability,
                self.data.threshold,
                self.data.decay_status,
                self.data.decaying,
                self.data.is_occupied,
            )

            # Manage decay timer based on the decay state set by the calculator
            if self.data.decaying:
                if not self._decay_unsub:  # Start timer only if not already running
                    _LOGGER.debug("Starting decay timer (decaying is True)")
                    self._start_decay_updates()
            elif self._decay_unsub:  # If not decaying, stop timer if running
                _LOGGER.debug("Stopping decay timer (decaying is False)")
                self._stop_decay_updates()

            # --- Reset Logic ---
            # Check if we should reset probability to MIN_PROBABILITY
            # Determine active sensors based on current_states in self.data
            active_sensors_exist = any(
                info.get("availability", False)
                and self.probabilities.is_entity_active(entity_id, info.get("state"))
                for entity_id, info in self.data.current_states.items()
                if info  # Check if info dict exists
            )
            should_reset = not self.data.decaying and not active_sensors_exist

            # Perform reset if needed
            if should_reset:
                _LOGGER.debug(
                    "Resetting state: no active sensors, decay complete/inactive"
                )
                # Update probability and occupancy state directly
                self.data.probability = MIN_PROBABILITY
                self.data.is_occupied = False
                # Ensure decay state is fully reset
                self.data.decaying = False
                self.data.decay_start_time = None
                self.data.decay_status = 1.0  # Explicitly set decay status to complete
                self._stop_decay_updates()  # Ensure timer is stopped
            # No 'else' needed as self.data.is_occupied was already set based on final_probability

        except CalculationError as err:
            _LOGGER.error("Error during probability calculation: %s", err)
            # Don't update probability/occupancy on calculation error, keep existing state
            # but still update listeners so they get other potential changes (like attributes)
            return self.data
        except (
            TimeoutError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            HomeAssistantError,
            asyncio.CancelledError,
        ):
            _LOGGER.exception("Error updating occupancy data")
            # Return existing data on general errors
            return self.data
        else:
            # Return the updated self.data object on success
            return self.data

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes."""

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Determine the list of entities to track from the current configuration
        entities_to_track = self.get_configured_sensors()
        # Store the set of currently tracked entities for the listener check
        self._entity_ids = set(entities_to_track)

        @callback
        def async_state_changed_listener(event) -> None:
            """Handle state changes for tracked entities."""
            try:
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")
                old_state = event.data.get("old_state")

                # Ensure the entity is one we are tracking
                if entity_id not in self._entity_ids:  # Use the stored set
                    _LOGGER.debug(
                        "Ignoring state change for untracked entity: %s", entity_id
                    )
                    return

                # Log state change details
                old_state_str = old_state.state if old_state else "None"
                new_state_str = new_state.state if new_state else "None"

                # Avoid logging redundant state changes (e.g., sensor updating timestamp but state is same)
                if old_state_str == new_state_str and new_state:
                    pass
                else:
                    _LOGGER.debug(
                        "State change for %s: %s -> %s",
                        entity_id,
                        old_state_str,
                        new_state_str,
                    )

                # Determine availability and state
                is_available = bool(
                    new_state
                    and new_state.state not in ["unknown", "unavailable", None, ""]
                )
                current_state = new_state.state if is_available else None
                last_changed = (
                    new_state.last_changed.isoformat()
                    if new_state and new_state.last_changed
                    else dt_util.utcnow().isoformat()
                )

                # --- Direct State Update ---
                sensor_info = {
                    "state": current_state,
                    "last_changed": last_changed,
                    "availability": is_available,
                }

                # Schedule async task to update self.data.current_states under lock
                async def update_data_state():
                    if not self.data:  # Check if coordinator/data still exists
                        _LOGGER.warning(
                            "Coordinator data object is None, cannot update state for %s",
                            entity_id,
                        )
                        return
                    async with self._state_lock:
                        # Update the current_states dictionary in self.data
                        self.data.current_states[entity_id] = sensor_info
                        _LOGGER.debug(
                            "Updated self.data.current_states for %s", entity_id
                        )

                self.hass.async_create_task(update_data_state())
                # --- End Async State Update Scheduling ---

                # Define async helper to calculate and push update
                async def async_calculate_and_update():
                    try:
                        # Use the updated state for calculation
                        new_data = await self._async_update_data()
                        # Only update if data calculation succeeded
                        if new_data is not None:
                            self.async_set_updated_data(new_data)
                        else:
                            _LOGGER.warning(
                                "Skipping update notification as data calculation failed"
                            )
                    except Exception:
                        _LOGGER.exception(
                            "Error during immediate calculation and update"
                        )

                # Schedule the immediate calculation and update
                self.hass.async_create_task(async_calculate_and_update())

            except (AttributeError, KeyError, TypeError, ValueError):
                _LOGGER.exception(
                    "Error processing state change for entity %s", entity_id
                )
            except Exception:
                _LOGGER.exception(
                    "Unexpected error in state change listener for %s", entity_id
                )

        self._remove_state_listener = async_track_state_change_event(
            self.hass,
            entities_to_track,  # Track entities from the current configuration
            async_state_changed_listener,
        )
        _LOGGER.debug(
            "State change listener set up for %d entities",
            len(entities_to_track),
        )

    # --- Prior Update Handling ---
    async def _schedule_next_prior_update(self) -> None:
        """Schedule the next prior update at the start of the next hour."""
        # Cancel any existing update first to prevent duplicates
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        now = dt_util.utcnow()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        self._next_prior_update = next_hour  # Store the exact time it's scheduled for

        self._prior_update_tracker = async_track_point_in_time(
            self.hass, self._handle_prior_update, self._next_prior_update
        )
        _LOGGER.debug(
            "Scheduled next prior update for %s in area %s",
            self._next_prior_update.isoformat(),  # Log the stored datetime object
            self.config[CONF_NAME],
        )

    async def _handle_prior_update(self, _now: datetime) -> None:
        """Handle the prior update task. Rescheduling is now handled in update_learned_priors finally block."""
        # Clear the tracker reference now that it has fired
        self._prior_update_tracker = None
        # Set next update to None as this one is now running
        self._next_prior_update = None

        try:
            _LOGGER.info(
                "Performing scheduled prior update for area %s", self.config[CONF_NAME]
            )
            # update_learned_priors will handle rescheduling in its finally block
            await self.update_learned_priors()
            # Timestamp update moved inside update_learned_priors upon success
            _LOGGER.info(
                "Finished scheduled prior update task trigger for area %s",
                self.config[CONF_NAME],
            )
        except Exception:
            # Log errors from update_learned_priors if they bubble up (should be caught inside)
            _LOGGER.exception(
                "Error occurred during scheduled prior update for area %s",
                self.config[CONF_NAME],
            )
            # Ensure rescheduling happens even if the update task itself failed unexpectedly
            # Note: update_learned_priors' finally block should handle this, but as a safeguard:
            if not self._prior_update_tracker:
                _LOGGER.warning(
                    "Update_learned_priors failed to reschedule, attempting fallback reschedule"
                )
                await self._schedule_next_prior_update()

        # Reschedule logic moved to the finally block of update_learned_priors
        # await self._schedule_next_prior_update()

    # --- Decay Handling ---
    def _start_decay_updates(self) -> None:
        """Start regular decay updates every 5 seconds."""
        if not self.config.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED):
            _LOGGER.debug("Decay updates disabled by configuration")
            return

        if self._decay_unsub is not None:
            _LOGGER.debug("Decay timer already running")
            return  # Already running

        _LOGGER.debug("Starting decay update timer")
        interval = timedelta(seconds=5)

        async def _async_do_decay_update(*_) -> None:
            """Execute decay update.

            This task runs independently and triggers a standard coordinator refresh.
            The main calculation logic in _async_update_data handles decay status.

            """
            try:
                # Only log if decay is actually active according to current state
                if self.data and self.data.decaying:
                    _LOGGER.debug("Decay timer fired. Triggering refresh")
                    # Request a refresh, let _async_update_data handle the logic
                    await self.async_request_refresh()
                    # No direct calculation here
                else:
                    _LOGGER.debug(
                        "Decay timer fired, but decay is not active. Stopping timer"
                    )
                    self._stop_decay_updates()  # Stop if decay became inactive

            except Exception:
                _LOGGER.exception("Error in decay update task")
                # Stop the timer on error to prevent loops
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
        else:
            _LOGGER.debug("Decay update timer was not running")

    # --- Data Saving ---
    async def _async_save_prior_state_data(self) -> None:
        """Save the current prior state data to storage.

        This is now only called explicitly after successful prior calculation.

        """
        if not self.prior_state:
            _LOGGER.warning("Attempted to save prior state, but prior_state is None")
            return

        try:
            _LOGGER.debug("Attempting to save prior state data")
            await self.storage.async_save_instance_prior_state(
                self.config_entry.entry_id,
                self.config.get(CONF_NAME, "Unknown Area"),
                self.prior_state,
            )
            # Update timestamp after successful save
            # This is now handled in update_learned_priors
            # self._last_prior_update = dt_util.utcnow().isoformat()
            _LOGGER.debug("Prior state data saved successfully")
        except (TimeoutError, HomeAssistantError, ValueError, RuntimeError) as err:
            # Raise specific StorageError for better upstream handling
            _LOGGER.error("Failed to save prior state data: %s", err)
            raise StorageError(f"Failed to save prior state data: {err}") from err

    # --- Listener Handling ---
    @callback
    def _async_refresh_finished(self) -> None:
        """Handle when a refresh has finished.

        Removed data saving from here, it's handled after prior calculation.
        """
        if self.last_update_success:
            _LOGGER.debug("Coordinator refresh finished successfully")
        # Save data unconditionally after successful refresh
        # self.hass.async_create_task(self._async_save_prior_state_data())
        else:
            _LOGGER.warning("Coordinator refresh failed")

    @callback
    def async_set_updated_data(self, data: ProbabilityState) -> None:
        """Manually update data and notify listeners."""
        _LOGGER.debug("Manually setting updated data")
        super().async_set_updated_data(data)

    @callback
    def async_add_listener(
        self, update_callback: CALLBACK_TYPE, context: Any = None
    ) -> Callable[[], None]:
        """Add a listener for data updates with improved tracking."""

        return super().async_add_listener(update_callback, context)
