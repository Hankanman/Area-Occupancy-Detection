"""Prior Probability Manager for Area Occupancy Detection.

This module manages all aspects of prior probability calculations and scheduling,
including loading/saving stored priors, scheduling updates, and coordinating
the calculation process.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.event import async_track_point_in_time
from homeassistant.util import dt as dt_util

from .const import (
    CONF_HISTORY_PERIOD,
    CONF_NAME,
    DEFAULT_HISTORY_PERIOD,
    MIN_PROBABILITY,
)
from .exceptions import StorageError
from .types import PriorState, TypeAggregate

if TYPE_CHECKING:
    from .calculate_prior import PriorCalculator
    from .probabilities import Probabilities
    from .state_manager import StateManager
    from .storage import AreaOccupancyStore

_LOGGER = logging.getLogger(__name__)


class PriorManager:
    """Manages prior probability calculations and scheduling for Area Occupancy Detection.

    This class centralizes all prior probability management including:
    - Loading and saving prior state data
    - Scheduling and executing prior updates
    - Managing prior update intervals and timing
    - Coordinating type and entity prior calculations
    """

    def __init__(
        self,
        hass: HomeAssistant,
        config: dict[str, Any],
        storage: "AreaOccupancyStore",
        probabilities: "Probabilities",
        state_manager: "StateManager",
        prior_calculator: "PriorCalculator",
        config_entry_id: str,
    ) -> None:
        """Initialize the prior manager.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary
            storage: Storage handler for persistence
            probabilities: Probabilities configuration handler
            state_manager: State manager for entity operations
            prior_calculator: Calculator for prior computations
            config_entry_id: Config entry ID for storage operations

        """
        self.hass = hass
        self.config = config
        self.storage = storage
        self.probabilities = probabilities
        self.state_manager = state_manager
        self.prior_calculator = prior_calculator
        self.config_entry_id = config_entry_id

        # Prior state management
        self.prior_state = PriorState()
        self._last_prior_update: str | None = None
        self._next_prior_update: datetime | None = None
        self._prior_update_tracker: CALLBACK_TYPE | None = None
        self._shutting_down: bool = False  # Track shutdown state

        # Scheduling management
        self._prior_update_interval = timedelta(hours=1)

        # Import here to avoid circular imports

    # --- Properties ---

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

    # --- Public Methods ---

    async def async_setup(self) -> None:
        """Set up the prior manager and determine if initial calculation is needed."""
        try:
            # Load stored data first
            await self.load_stored_priors()

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
                elif not self._last_prior_update:
                    # Timestamp wasn't loaded/found
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
                        # Check age against interval
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
                        force_prior_update = True
                        reason = "Failed to parse last update timestamp"

            # Perform prior update if needed
            if force_prior_update:
                _LOGGER.debug(
                    "Performing initial prior calculation at startup: %s", reason
                )
                await self.update_learned_priors()
            else:
                _LOGGER.debug(
                    "Skipping initial prior calculation: Existing priors are complete and recent (last update: %s)",
                    self._last_prior_update or "N/A",
                )

            # Schedule periodic prior updates regardless of initial calculation
            await self.schedule_next_update()

            _LOGGER.debug("PriorManager setup completed successfully")

        except (StorageError, HomeAssistantError) as err:
            _LOGGER.error("Failed to set up PriorManager: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up PriorManager: {err}") from err

    async def async_shutdown(self) -> None:
        """Shutdown the prior manager and cleanup resources."""
        # Set shutdown flag first to prevent any new timers from being scheduled
        self._shutting_down = True

        # Cancel prior update tracker
        if self._prior_update_tracker is not None:
            self._prior_update_tracker()
            self._prior_update_tracker = None

        # Clear state
        self.prior_state = PriorState()
        self._last_prior_update = None
        self._next_prior_update = None

        _LOGGER.debug("PriorManager shutdown completed")

    async def load_stored_priors(self) -> None:
        """Load and restore prior state data from storage."""
        try:
            _LOGGER.debug("Loading stored prior data from storage")

            # Use the store's instance-specific load method
            loaded_data = await self.storage.async_load_instance_prior_state(
                self.config_entry_id
            )

            if loaded_data and loaded_data.prior_state:
                _LOGGER.debug(
                    "Found stored prior state for instance %s, restoring (last saved: %s)",
                    self.config_entry_id,
                    loaded_data.last_updated,
                )
                self.prior_state = loaded_data.prior_state
                self._last_prior_update = loaded_data.last_updated
            else:
                _LOGGER.info(
                    "No stored prior state found for instance %s, initializing with defaults",
                    self.config_entry_id,
                )
                self._last_prior_update = None

                # Initialize with defaults
                self.prior_state = PriorState()
                self.prior_state.initialize_from_defaults(
                    self.probabilities, self.state_manager
                )
                self.prior_state.update(
                    analysis_period=self.config.get(
                        CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                    ),
                )

            _LOGGER.debug(
                "Successfully loaded prior state for instance %s",
                self.config_entry_id,
            )
        except StorageError as err:
            _LOGGER.warning(
                "Storage error for instance %s, initializing with defaults: %s",
                self.config_entry_id,
                err,
            )
            self._last_prior_update = None

            # Initialize with defaults on storage error
            self.prior_state = PriorState()
            self.prior_state.initialize_from_defaults(
                self.probabilities, self.state_manager
            )
            self.prior_state.update(
                analysis_period=self.config.get(
                    CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                ),
            )
            # Re-raise as ConfigEntryNotReady if loading fails critically
            raise ConfigEntryNotReady(
                f"Failed to load stored prior data: {err}"
            ) from err

    async def save_prior_state(self) -> None:
        """Save the current prior state data to storage."""
        if not self.prior_state:
            _LOGGER.warning("Attempted to save prior state, but prior_state is None")
            return

        try:
            _LOGGER.debug("Attempting to save prior state data")
            await self.storage.async_save_instance_prior_state(
                self.config_entry_id,
                self.config.get(CONF_NAME, "Unknown Area"),
                self.prior_state,
            )
            _LOGGER.debug("Prior state data saved successfully")
        except (TimeoutError, HomeAssistantError, ValueError, RuntimeError) as err:
            _LOGGER.error("Failed to save prior state data: %s", err)
            raise StorageError(f"Failed to save prior state data: {err}") from err

    async def update_learned_priors(self, history_period: int | None = None) -> None:
        """Update learned priors using historical data.

        Args:
            history_period: Period in days for historical analysis, or None to use config default

        """
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
                await self.schedule_next_update()
                return

            _LOGGER.debug("Using period: %s (type: %s)", period, type(period))

            end_time = dt_util.utcnow()
            start_time = end_time - timedelta(days=period)
            _LOGGER.debug(
                "Calculated time range - start: %s, end: %s", start_time, end_time
            )

            # Update the analysis period in prior_state
            _LOGGER.debug("Updating prior_state analysis period to: %s", period)
            self.prior_state.update(analysis_period=int(period))

            sensors = self._get_configured_sensors()
            if not sensors:
                _LOGGER.warning("No sensors configured for prior calculation")
                self._last_prior_update = dt_util.utcnow().isoformat()
                await self.schedule_next_update()
                return

            _LOGGER.debug("Calculating prior for %s sensors", len(sensors))
            # Use a temporary dict to store successful calculations
            calculated_entity_priors = {}

            for sensor_id in sensors:
                _LOGGER.debug("Calculating prior for sensor: %s", sensor_id)
                try:
                    # Call calculator which returns results or None
                    prior_result = await self.prior_calculator.calculate_prior(
                        sensor_id, start_time, end_time
                    )

                    if prior_result is None:
                        _LOGGER.warning(
                            "Prior calculation failed for %s, skipping update",
                            sensor_id,
                        )
                        continue

                    # Extract values from PriorData object
                    prob_given_true = prior_result.prob_given_true
                    prob_given_false = prior_result.prob_given_false
                    prior = prior_result.prior

                    # Store successful calculation result temporarily
                    calculated_entity_priors[sensor_id] = {
                        "prob_given_true": float(prob_given_true)
                        if prob_given_true is not None
                        else MIN_PROBABILITY,
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

            # Update PriorState after loop
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
                    self.probabilities, self.state_manager
                )
                _LOGGER.debug("Overall prior calculated: %.3f", overall_prior)
                self.prior_state.update(overall_prior=overall_prior)

            else:
                _LOGGER.warning(
                    "No entity priors were successfully calculated. Prior state not updated"
                )

            # Save the updated priors if any calculations were successful
            if calculated_entity_priors:
                _LOGGER.debug("Saving prior state after update (successful/partial)")
                await self.save_prior_state()
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
            # Only reschedule if we're not shutting down
            if not self._shutting_down:
                await self.schedule_next_update()

    async def schedule_next_update(self) -> None:
        """Schedule the next prior update at the start of the next hour."""
        # Don't schedule new timers if we're shutting down
        if self._shutting_down:
            _LOGGER.debug("Skipping timer scheduling during shutdown")
            return

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

    # --- Private Methods ---

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
            # Ensure rescheduling happens even if the update task itself failed
            if not self._prior_update_tracker:
                _LOGGER.warning(
                    "Update_learned_priors failed to reschedule, attempting fallback reschedule"
                )
                await self.schedule_next_update()

    async def _update_type_priors_from_entities(self) -> None:
        """Calculate and update type priors by averaging successfully learned entity priors."""
        _LOGGER.debug("Calculating type priors based on learned entity priors")
        timestamp = dt_util.utcnow().isoformat()
        type_aggregates: dict[str, TypeAggregate] = {}

        # Aggregate learned priors by type
        for entity_id, learned_priors in self.prior_state.entity_priors.items():
            if not learned_priors:  # Skip if empty dict somehow
                continue
            entity_type = self.state_manager.get_entity_type(entity_id)
            if entity_type:
                # Access attributes of PriorData object
                # Ensure required attributes are not None
                if (
                    learned_priors.prob_given_true is not None
                    and learned_priors.prob_given_false is not None
                ):
                    # Get or create the TypeAggregate instance for this entity type
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

    def _get_configured_sensors(self) -> list[str]:
        """Get all configured sensors for prior calculation.

        This should match the logic from the coordinator's get_configured_sensors method.
        """
        # Import here to avoid circular imports
        from .types import SensorInputs

        try:
            inputs = SensorInputs.from_config(self.config)
            return inputs.get_all_sensors()
        except (ValueError, TypeError) as err:
            _LOGGER.error("Failed to get configured sensors: %s", err)
            return []
