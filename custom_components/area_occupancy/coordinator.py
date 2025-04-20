"""Area Occupancy Coordinator."""

from __future__ import annotations

# Standard Library
import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
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
from homeassistant.helpers.device_registry import DeviceInfo
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
    CONF_WASP_IN_BOX_ENABLED,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_IN_BOX_ENABLED,
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
from .types import PriorState, ProbabilityState, SensorInfo, SensorInputs, TypeAggregate

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
            probabilities=self.probabilities,
        )
        self._prior_calculator = PriorCalculator(
            hass=self.hass,
            probabilities=self.probabilities,
            sensor_inputs=self.inputs,
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
        if callable(self._remove_state_listener):
            self._remove_state_listener()
        self._remove_state_listener = None

        # Save final state directly on shutdown
        # No need to save priors here, they are saved after calculation
        # await self._async_save_prior_state_data()

        # Clear data
        self.data = ProbabilityState()
        self.prior_state = PriorState()
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

            # Reinitialize the calculators with updated references
            self.calculator = ProbabilityCalculator(
                probabilities=self.probabilities,
            )
            self._prior_calculator = PriorCalculator(
                hass=self.hass,
                probabilities=self.probabilities,
                sensor_inputs=self.inputs,
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

    async def async_load_stored_data(self) -> None:
        """Load and restore data from storage."""
        try:
            _LOGGER.debug("Loading stored data from storage")

            # Use the store's instance-specific load method
            loaded_data = await self.storage.async_load_instance_prior_state(
                self.config_entry.entry_id
            )

            if loaded_data and loaded_data.prior_state:
                _LOGGER.debug(
                    "Found stored prior state for instance %s, restoring (last saved: %s)",
                    self.config_entry.entry_id,
                    loaded_data.last_updated,
                )
                self.prior_state = loaded_data.prior_state
                self._last_prior_update = (
                    loaded_data.last_updated
                )  # Store the loaded timestamp
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
                self._last_prior_update = dt_util.utcnow().isoformat()
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

            sensors = self.get_configured_sensors()
            if not sensors:
                _LOGGER.warning("No sensors configured for prior calculation")
                self._last_prior_update = dt_util.utcnow().isoformat()
                await self._schedule_next_prior_update()
                return

            _LOGGER.debug("Calculating prior for %s sensors", len(sensors))
            # Use a temporary dict to store successful calculations before updating prior_state
            calculated_entity_priors = {}

            for sensor_id in sensors:
                _LOGGER.debug("Calculating prior for sensor: %s", sensor_id)
                try:
                    # Call calculator which now returns results or None
                    prior_result = await self._prior_calculator.calculate_prior(
                        sensor_id, start_time, end_time
                    )

                    if prior_result is None:
                        _LOGGER.warning(
                            "Prior calculation failed for %s, skipping update",
                            sensor_id,
                        )
                        continue  # Skip this sensor

                    # Extract values from PriorData object
                    prob_given_true = prior_result.prob_given_true
                    prob_given_false = prior_result.prob_given_false
                    prior = prior_result.prior

                    # Store successful calculation result temporarily
                    # Note: Need to ensure prob_given_true/false are not None here, although calculate_prior should ensure they are floats
                    # Add validation/fallback if necessary, but based on calculate_prior, they should be floats.
                    calculated_entity_priors[sensor_id] = {
                        "prob_given_true": float(prob_given_true)
                        if prob_given_true is not None
                        else MIN_PROBABILITY,  # Example fallback/casting
                        "prob_given_false": float(prob_given_false)
                        if prob_given_false is not None
                        else MIN_PROBABILITY,
                        "prior": float(prior),
                    }

                    _LOGGER.debug(
                        "Calculated probabilities for %s - P(T): %.3f, P(F): %.3f, Prior: %.3f",
                        sensor_id,
                        prob_given_true,
                        prob_given_false,
                        prior,
                    )

                except Exception:
                    _LOGGER.exception("Error calculating prior for %s", sensor_id)
                    continue

            # --- Update PriorState After Loop ---
            # Only update if we have some successful calculations
            if calculated_entity_priors:
                timestamp = dt_util.utcnow().isoformat()
                # Update entity priors in the main PriorState object
                for sensor_id, priors in calculated_entity_priors.items():
                    self.prior_state.update_entity_prior(
                        sensor_id,
                        priors["prob_given_true"],
                        priors["prob_given_false"],
                        priors["prior"],
                        timestamp,
                    )
                _LOGGER.debug(
                    "Updated entity priors in prior_state for %d sensors",
                    len(calculated_entity_priors),
                )

                # Calculate and update type priors based on the updated entity_priors
                await self._update_type_priors_from_entities()

                # Calculate the overall prior based on updated type priors
                _LOGGER.debug("Calculating overall prior based on updated type priors")
                overall_prior = self.prior_state.calculate_overall_prior(
                    self.probabilities
                )
                _LOGGER.debug("Overall prior calculated: %.3f", overall_prior)
                self.prior_state.update(overall_prior=overall_prior)

            else:
                # Handle case where no sensors could be calculated
                _LOGGER.warning(
                    "No entity priors were successfully calculated. Prior state not updated"
                )

            # Save the updated priors only if *all* requested calculations were successful
            # Or potentially save even if partially successful? Let's save if anything changed.
            if calculated_entity_priors:  # Save if we updated *any* entity
                _LOGGER.debug("Saving prior state after update (successful/partial)")
                await self._async_save_prior_state_data()
                _LOGGER.debug("Prior update save complete")
                # Update the timestamp only after successful save
                self._last_prior_update = dt_util.utcnow().isoformat()
            else:
                _LOGGER.warning(
                    "Prior state not saved as no calculations were successful"
                )
                # Still update timestamp to prevent immediate retry loop
                self._last_prior_update = dt_util.utcnow().isoformat()

        except Exception:
            _LOGGER.exception("Unexpected error during update_learned_priors")
            self._last_prior_update = dt_util.utcnow().isoformat()
        finally:
            await self._schedule_next_prior_update()

    async def _update_type_priors_from_entities(self) -> None:
        """Calculate and update type priors by averaging successfully learned entity priors."""
        _LOGGER.debug("Calculating type priors based on learned entity priors")
        timestamp = dt_util.utcnow().isoformat()
        type_aggregates: dict[str, TypeAggregate] = {}  # Use standard dict

        # Aggregate learned priors by type
        for entity_id, learned_priors in self.prior_state.entity_priors.items():
            if not learned_priors:  # Skip if empty dict somehow
                continue
            entity_type = self.probabilities.get_entity_type(entity_id)
            if entity_type:
                # Access attributes of PriorData object
                # Ensure required attributes are not None (prob_given_true/false can be None)
                if (
                    learned_priors.prob_given_true is not None
                    and learned_priors.prob_given_false is not None
                ):
                    # Get or create the TypeAggregate instance for this entity type
                    # Use entity_type.value as the key for the dictionary
                    aggregate = type_aggregates.setdefault(
                        entity_type.value, TypeAggregate()
                    )

                    aggregate.priors.append(learned_priors.prior)
                    aggregate.p_true.append(learned_priors.prob_given_true)
                    aggregate.p_false.append(learned_priors.prob_given_false)
                    aggregate.count += 1
                else:
                    _LOGGER.warning(
                        "Skipping entity %s for type prior calculation due to missing conditional probabilities in learned data: %s",
                        entity_id,
                        learned_priors,
                    )

        # Calculate averages and update PriorState
        for sensor_type, aggregate in type_aggregates.items():
            count = aggregate.count
            if count > 0:
                avg_prior = sum(aggregate.priors) / count
                avg_prob_given_true = sum(aggregate.p_true) / count
                avg_prob_given_false = sum(aggregate.p_false) / count

                _LOGGER.debug(
                    "Updating type %s priors - prior: %.3f, p_true: %.3f, p_false: %.3f (from %d sensors)",
                    sensor_type,
                    avg_prior,
                    avg_prob_given_true,
                    avg_prob_given_false,
                    count,
                )
                self.prior_state.update_type_prior(
                    sensor_type,
                    avg_prior,
                    timestamp,
                    avg_prob_given_true,
                    avg_prob_given_false,
                )
            else:
                _LOGGER.debug(
                    "No valid learned priors found to calculate average for type %s",
                    sensor_type,
                )

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
                    sensor_info: SensorInfo = {
                        "state": state_obj.state
                        if (state_obj and is_available)
                        else None,
                        "last_changed": (
                            state_obj.last_changed.isoformat()
                            if state_obj and state_obj.last_changed
                            else dt_util.utcnow().isoformat()
                        ),
                        "availability": is_available,
                    }
                    # Store directly in self.data.current_states
                    self.data.current_states[entity_id] = sensor_info

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
        if not self.data or not self.prior_state:
            _LOGGER.warning(
                "_async_update_data called but coordinator data/prior_state is not initialized"
            )
            # Cannot proceed without state objects
            # Return the potentially uninitialized self.data to avoid None propagation issues
            return (
                self.data if self.data else ProbabilityState()
            )  # Return empty if self.data is None

        try:
            current_states_snapshot = {}
            # --- Update self.data state dictionaries first (under lock) ---
            async with self._state_lock:
                # Update previous_states in self.data based on its *current* current_states
                self.data.previous_states = self.data.current_states.copy()
                # Capture a snapshot of current states for the calculation
                current_states_snapshot = self.data.current_states.copy()

            # --- Inject wasp-in-the-box virtual sensor if enabled ---
            if self.config.get(CONF_WASP_IN_BOX_ENABLED, DEFAULT_WASP_IN_BOX_ENABLED):
                wasp_entity_id = "wasp.virtual"
                # Determine wasp state: 'on' if last probability >= threshold, else 'off'
                # Use previous probability if available, else MIN_PROBABILITY
                last_prob = (
                    self.data.probability if hasattr(self.data, "probability") else 0.0
                )
                threshold = (
                    self.data.threshold if hasattr(self.data, "threshold") else 0.5
                )
                wasp_state = "on" if last_prob >= threshold else "off"
                prev_info = self.data.current_states.get(wasp_entity_id, None)
                prev_last_changed = (
                    prev_info["last_changed"]
                    if prev_info and prev_info.get("state") == wasp_state
                    else None
                )
                last_changed = prev_last_changed or dt_util.utcnow().isoformat()
                current_states_snapshot[wasp_entity_id] = SensorInfo(
                    state=wasp_state,
                    last_changed=last_changed,
                    availability=True,
                )

            # --- Store initial probability state before calculation ---
            initial_prob = self.data.probability
            initial_decaying = self.data.decaying
            initial_decay_start_time = self.data.decay_start_time
            initial_decay_start_prob = self.data.decay_start_probability

            # --- Calculate Undecayed Probability ---
            try:
                # Pass snapshot and prior state to the calculator
                calc_result = self.calculator.calculate_occupancy_probability(
                    current_states_snapshot,  # Pass the snapshot
                    self.prior_state,
                )
            except (CalculationError, ValueError, ZeroDivisionError) as calc_err:
                # Log the specific calculation error and return existing data
                _LOGGER.error("Error during probability calculation: %s", calc_err)
                # Keep existing data, but notify listeners
                self.async_set_updated_data(self.data)
                return self.data

            # Update self.data with intermediate results *before* decay
            self.data.update(
                probability=calc_result.calculated_probability,  # Store undecayed prob temporarily
                prior_probability=calc_result.prior_probability,
                sensor_probabilities=calc_result.sensor_probabilities,
            )

            # --- Apply Decay ---
            (
                decayed_probability,
                decay_factor,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            ) = self.decay_handler.calculate_decay(
                current_probability=calc_result.calculated_probability,  # Use the result from calculation
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
                and self.probabilities.is_entity_active(entity_id, info.get("state"))
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
        else:
            # Return the updated self.data object on success
            return self.data

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes."""

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        entities_to_track = self.get_configured_sensors()
        self._entity_ids = set(entities_to_track)

        @callback
        def async_state_changed_listener(event) -> None:
            """Handle state changes for tracked entities."""
            try:
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")
                old_state = event.data.get("old_state")

                if entity_id not in self._entity_ids:
                    _LOGGER.debug(
                        "Ignoring state change for untracked entity: %s", entity_id
                    )
                    return

                old_state_str = old_state.state if old_state else "None"
                new_state_str = new_state.state if new_state else "None"

                if (
                    old_state_str != new_state_str or not new_state
                ):  # Log if state changes or becomes unavailable
                    _LOGGER.debug(
                        "State change for %s: %s -> %s",
                        entity_id,
                        old_state_str,
                        new_state_str,
                    )

                is_available = bool(
                    new_state
                    and new_state.state not in ["unknown", "unavailable", None, ""]
                )
                current_state_val = new_state.state if is_available else None
                last_changed = (
                    new_state.last_changed.isoformat()
                    if new_state and new_state.last_changed
                    else dt_util.utcnow().isoformat()
                )

                sensor_info = SensorInfo(
                    state=current_state_val,
                    last_changed=last_changed,
                    availability=is_available,
                )

                async def update_state_and_calculate():
                    """Update state under lock then trigger calculation."""
                    if not self.data:
                        _LOGGER.warning(
                            "Coordinator data object is None, cannot update state for %s",
                            entity_id,
                        )
                        return
                    try:
                        # Update state under lock first
                        async with self._state_lock:
                            self.data.current_states[entity_id] = sensor_info
                            _LOGGER.debug(
                                "Updated self.data.current_states for %s", entity_id
                            )

                        # Now trigger the calculation and refresh
                        updated_data = await self._async_update_data()
                        if updated_data:  # Check if update succeeded
                            self.async_set_updated_data(updated_data)
                        else:
                            _LOGGER.warning(
                                "Skipping notification as _async_update_data failed for %s",
                                entity_id,
                            )

                    except Exception:
                        _LOGGER.exception(
                            "Error during state update and calculation for %s",
                            entity_id,
                        )

                # Schedule the combined state update and calculation task
                self.hass.async_create_task(update_state_and_calculate())

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
            entities_to_track,
            async_state_changed_listener,
        )
        _LOGGER.debug(
            "State change listener set up for %d entities",
            len(entities_to_track),
        )

    # --- Prior Update Handling ---
    async def _schedule_next_prior_update(self) -> None:
        """Schedule the next prior update at the start of the next hour."""
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        now = dt_util.utcnow()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        self._next_prior_update = next_hour

        self._prior_update_tracker = async_track_point_in_time(
            self.hass, self._handle_prior_update, self._next_prior_update
        )
        _LOGGER.debug(
            "Scheduled next prior update for %s in area %s",
            self._next_prior_update.isoformat(),
            self.config[CONF_NAME],
        )

    async def _handle_prior_update(self, _now: datetime) -> None:
        """Handle the prior update task."""
        self._prior_update_tracker = None
        self._next_prior_update = None

        try:
            _LOGGER.info(
                "Performing scheduled prior update for area %s", self.config[CONF_NAME]
            )
            await self.update_learned_priors()
            _LOGGER.info(
                "Finished scheduled prior update task trigger for area %s",
                self.config[CONF_NAME],
            )
        except Exception:
            _LOGGER.exception(
                "Error occurred during scheduled prior update for area %s",
                self.config[CONF_NAME],
            )
            # Ensure rescheduling happens even if the update task itself failed unexpectedly
            if not self._prior_update_tracker:
                _LOGGER.warning(
                    "Update_learned_priors failed to reschedule, attempting fallback reschedule"
                )
                await self._schedule_next_prior_update()

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
        else:
            _LOGGER.debug("Decay update timer was not running")

    # --- Data Saving ---
    async def _async_save_prior_state_data(self) -> None:
        """Save the current prior state data to storage."""
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
        _LOGGER.debug("Manually setting updated data")
        super().async_set_updated_data(data)

    @callback
    def async_add_listener(
        self, update_callback: CALLBACK_TYPE, context: Any = None
    ) -> Callable[[], None]:
        """Add a listener for data updates with improved tracking."""

        return super().async_add_listener(update_callback, context)
