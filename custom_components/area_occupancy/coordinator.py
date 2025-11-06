"""Area Occupancy Coordinator."""

from __future__ import annotations

# Standard library imports
from datetime import datetime, timedelta
import logging
from typing import Any

# Home Assistant imports
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from homeassistant.helpers.event import (
    async_call_later,
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local imports
from .const import (
    ANALYSIS_INTERVAL,
    ANALYSIS_STAGGER_MINUTES,
    DECAY_INTERVAL,
    DEFAULT_NAME,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MASTER_HEALTH_CHECK_INTERVAL,
    MASTER_HEARTBEAT_INTERVAL,
    MIN_PROBABILITY,
    SAVE_DEBOUNCE_SECONDS,
    SIGNAL_MASTER_HEARTBEAT,
    SIGNAL_STATE_SAVE_REQUEST,
)
from .data.config import Config
from .data.entity import EntityFactory, EntityManager
from .data.entity_type import InputType
from .data.prior import Prior
from .data.purpose import PurposeManager
from .db import AreaOccupancyDB
from .utils import bayesian_probability

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Manage fetching and combining data for area occupancy."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=config_entry.data.get(CONF_NAME, DEFAULT_NAME),
            update_interval=None,
            setup_method=self.setup,
            update_method=self.update,
        )
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        self.db = AreaOccupancyDB(self)
        self.config = Config(self)
        self.factory = EntityFactory(self)
        self.prior = Prior(self)
        self.purpose = PurposeManager(self)
        self.entities = EntityManager(self)
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._remove_state_listener: CALLBACK_TYPE | None = None
        self._analysis_timer: CALLBACK_TYPE | None = None
        self._health_check_timer: CALLBACK_TYPE | None = None
        self._save_timer: CALLBACK_TYPE | None = None
        self._setup_complete: bool = False

        # Master instance management
        self._is_master: bool = False
        self._master_entry_id: str | None = None
        self._event_listeners: list[CALLBACK_TYPE] = []
        self._master_heartbeat_timer: CALLBACK_TYPE | None = None
        self._master_health_timer: CALLBACK_TYPE | None = None

    async def async_init_database(self) -> None:
        """Initialize the database asynchronously to avoid blocking the event loop.

        This method should be called after coordinator creation but before
        async_config_entry_first_refresh().

        This ensures database tables exist and basic integrity before setup() loads data.
        The initialization is idempotent - calling it multiple times is safe even if the
        database was pre-initialized.
        """
        try:
            await self.hass.async_add_executor_job(self.db.initialize_database)
            _LOGGER.debug(
                "Database initialization completed for entry %s", self.entry_id
            )
        except Exception as err:
            _LOGGER.error(
                "Failed to initialize database for entry %s: %s", self.entry_id, err
            )
            raise

    # --- Master Election and Coordination ---

    async def _elect_master(self) -> None:
        """Attempt to become master or identify existing master."""
        master_id = await self.hass.async_add_executor_job(self.db.elect_master)

        # If election failed (None), treat as non-master to avoid multiple masters
        if master_id is None:
            _LOGGER.warning(
                "Master election failed for %s, treating as non-master to avoid conflicts",
                self.config.name,
            )
            self._is_master = False
            self._master_entry_id = None
            return

        self._is_master = master_id == self.entry_id
        self._master_entry_id = master_id
        _LOGGER.info(
            "Master election complete for %s. Is master: %s",
            self.config.name,
            self._is_master,
        )

    def _send_state_change_request(self) -> None:
        """Request master to perform a save."""
        if not self._is_master:
            async_dispatcher_send(self.hass, SIGNAL_STATE_SAVE_REQUEST, self.entry_id)
            _LOGGER.debug("Sent save request to master from %s", self.entry_id)

    def _handle_state_change_request(self, entry_id: str) -> None:
        """Master only: Handle save requests from non-master instances."""
        if not self._is_master or entry_id == self.entry_id:
            return

        _LOGGER.debug("Received save request from %s", entry_id)
        self._schedule_save()  # Debounced save

    def _handle_master_heartbeat(self) -> None:
        """Non-master only: Acknowledge master heartbeat signal."""
        if self._is_master:
            return
        _LOGGER.debug("Received master heartbeat signal")
        # Heartbeat received, master is alive (no action needed)

    def _start_master_heartbeat_timer(self) -> None:
        """Start master heartbeat broadcasting (master only)."""
        if not self._is_master:
            return

        async def _send_heartbeat(_now: datetime) -> None:
            """Send heartbeat to database and via dispatcher."""
            await self.hass.async_add_executor_job(self.db.update_master_heartbeat)
            async_dispatcher_send(self.hass, SIGNAL_MASTER_HEARTBEAT)
            _LOGGER.debug("Sent master heartbeat")

            # Reschedule
            next_heartbeat = _now + timedelta(seconds=MASTER_HEARTBEAT_INTERVAL)
            self._master_heartbeat_timer = async_track_point_in_time(
                self.hass, _send_heartbeat, next_heartbeat
            )

        # Start first heartbeat
        self._master_heartbeat_timer = async_track_point_in_time(
            self.hass, _send_heartbeat, dt_util.utcnow() + timedelta(seconds=5)
        )

    def _start_master_health_timer(self) -> None:
        """Monitor master health and trigger re-election if needed (non-master only)."""
        if self._is_master:
            return

        async def _check_health(_now: datetime) -> None:
            """Check if master is still alive."""
            is_healthy = await self.hass.async_add_executor_job(
                self.db.check_master_health
            )

            if not is_healthy:
                _LOGGER.warning("Master instance unresponsive, triggering re-election")
                await self._elect_master()
                if self._is_master:
                    # Became new master, reconfigure
                    await self._become_master()

            # Reschedule
            self._master_health_timer = async_track_point_in_time(
                self.hass,
                _check_health,
                dt_util.utcnow() + timedelta(seconds=MASTER_HEALTH_CHECK_INTERVAL),
            )

        # Start health monitoring
        self._master_health_timer = async_track_point_in_time(
            self.hass,
            _check_health,
            dt_util.utcnow() + timedelta(seconds=MASTER_HEALTH_CHECK_INTERVAL),
        )

    async def _become_master(self) -> None:
        """Reconfigure coordinator to become master."""
        _LOGGER.info("Instance %s is becoming master", self.entry_id)

        # Clean up old listeners (from when this was a non-master)
        for listener in self._event_listeners:
            listener()
        self._event_listeners.clear()

        # Cancel old health monitoring timer if it exists
        if self._master_health_timer is not None:
            self._master_health_timer()
            self._master_health_timer = None

        # Start heartbeat broadcasting
        self._start_master_heartbeat_timer()

        # Setup event listeners (including save request listener for master)
        self._setup_event_listeners()

    def _setup_event_listeners(self) -> None:
        """Setup dispatcher listeners based on role."""
        if self._is_master:
            # Master: listen to save requests
            listener = async_dispatcher_connect(
                self.hass, SIGNAL_STATE_SAVE_REQUEST, self._handle_state_change_request
            )
            self._event_listeners.append(listener)
        else:
            # Non-master: listen to heartbeat signals
            listener = async_dispatcher_connect(
                self.hass, SIGNAL_MASTER_HEARTBEAT, self._handle_master_heartbeat
            )
            self._event_listeners.append(listener)

            # Start health monitoring
            self._start_master_health_timer()

    @property
    def is_master(self) -> bool:
        """Return whether this instance is the master."""
        return self._is_master

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry_id)},
            name=self.config.name,
            manufacturer=DEVICE_MANUFACTURER,
            model=DEVICE_MODEL,
            sw_version=DEVICE_SW_VERSION,
        )

    @property
    def probability(self) -> float:
        """Calculate and return the current occupancy probability (0.0-1.0)."""
        if not self.entities.entities:
            return MIN_PROBABILITY

        prior_value = self.prior.value
        return bayesian_probability(
            entities=self.entities.entities,
            prior=prior_value,
        )

    @property
    def type_probabilities(self) -> dict[str, float]:
        """Calculate and return the current occupancy probabilities for each entity type (0.0-1.0)."""
        if not self.entities.entities:
            return {}

        return {
            InputType.MOTION: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.MOTION),
                prior=self.prior.value,
            ),
            InputType.MEDIA: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.MEDIA),
                prior=self.prior.value,
            ),
            InputType.APPLIANCE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.APPLIANCE),
                prior=self.prior.value,
            ),
            InputType.DOOR: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.DOOR),
                prior=self.prior.value,
            ),
            InputType.WINDOW: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.WINDOW),
                prior=self.prior.value,
            ),
            InputType.ILLUMINANCE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(
                    InputType.ILLUMINANCE
                ),
                prior=self.prior.value,
            ),
            InputType.HUMIDITY: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.HUMIDITY),
                prior=self.prior.value,
            ),
            InputType.TEMPERATURE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(
                    InputType.TEMPERATURE
                ),
                prior=self.prior.value,
            ),
        }

    @property
    def area_prior(self) -> float:
        """Get the area's baseline occupancy prior from historical data.

        This returns the pure P(area occupied) without any sensor weighting.
        """
        # Use the dedicated area baseline prior calculation
        return self.prior.value

    @property
    def decay(self) -> float:
        """Calculate the current decay probability (0.0-1.0)."""
        if not self.entities.entities:
            return 1.0

        decay_sum = sum(
            entity.decay.decay_factor for entity in self.entities.entities.values()
        )
        return decay_sum / len(self.entities.entities)

    @property
    def occupied(self) -> bool:
        """Return the current occupancy state (True/False)."""
        return self.probability >= self.config.threshold

    @property
    def setup_complete(self) -> bool:
        """Return whether setup is complete."""
        return self._setup_complete

    @property
    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0)."""
        return self.config.threshold if self.config else 0.5

    def _verify_setup_complete(self) -> bool:
        """Verify that critical initialization components have started successfully.

        Returns:
            True if all critical components are initialized, False otherwise
        """
        # Check if hass is available
        if not self.hass:
            _LOGGER.error("Home Assistant instance not available")
            return False

        # Check if decay timer started
        if self._global_decay_timer is None:
            _LOGGER.warning("Decay timer not started for %s", self.config.name)
            return False

        # Check if analysis timer started (or is scheduled)
        if self._analysis_timer is None:
            _LOGGER.warning("Analysis timer not started for %s", self.config.name)
            return False

        return True

    # --- Public Methods ---
    async def setup(self) -> None:
        """Initialize the coordinator and its components (fast startup mode)."""
        try:
            _LOGGER.info(
                "Initializing Area Occupancy for %s (quick startup mode)",
                self.config.name,
            )

            # Elect or identify master instance
            await self._elect_master()

            # Initialize purpose manager
            _LOGGER.debug("Initializing purpose manager for %s", self.config.name)
            await self.purpose.async_initialize()

            # Note: Old interval pruning is handled by hourly analysis cycle, not during startup
            # This prevents lock contention when multiple instances start in parallel

            # Load stored data first to restore prior from DB
            # Database integrity checks are deferred to background (60s after startup)
            _LOGGER.debug(
                "Loading entity data from database (deferring heavy operations)"
            )
            await self.db.load_data()
            _LOGGER.info("Loaded entity data for %s", self.config.name)

            # Ensure area exists and persist current configuration/state
            try:
                await self.hass.async_add_executor_job(self.db.save_area_data)
            except (HomeAssistantError, OSError, RuntimeError) as e:
                _LOGGER.warning("Failed to save area data, continuing setup: %s", e)

            # Setup event listeners based on role
            self._setup_event_listeners()

            # Start master-specific functionality
            if self._is_master:
                _LOGGER.info(
                    "Starting master-specific functionality for %s", self.config.name
                )
                # Master: Start heartbeat and save timer
                self._start_master_heartbeat_timer()
                # Only master has save timer
                self._save_timer = None
            else:
                _LOGGER.info("Starting as non-master instance for %s", self.config.name)

            # Track entity state changes
            await self.track_entity_state_changes(self.entities.entity_ids)

            # Start timers only after everything is ready
            self._start_decay_timer()
            # Analysis timer is async and runs in background
            await self._start_analysis_timer()

            # Verify critical initialization succeeded before marking complete
            if not self._verify_setup_complete():
                _LOGGER.error("Critical initialization failed for %s", self.config.name)
                error_msg = "Failed to start critical timers"
                raise HomeAssistantError(error_msg)  # noqa: TRY301

            # Mark setup as complete before initial refresh to prevent debouncer conflicts
            self._setup_complete = True

            # Log instance information for multi-instance awareness
            all_instances = list(self.hass.config_entries.async_entries(DOMAIN))
            _LOGGER.info(
                "Successfully initialized %s with %d entities (instance %d of %d, all sharing database)",
                self.config.name,
                len(self.entities.entities),
                1,  # Could calculate position if needed
                len(all_instances),
            )
        except HomeAssistantError as err:
            _LOGGER.error("Failed to set up coordinator: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up coordinator: {err}") from err
        except (OSError, RuntimeError) as err:
            _LOGGER.error("Unexpected error during coordinator setup: %s", err)
            # Try to continue with basic functionality even if some parts fail
            _LOGGER.info(
                "Continuing with basic coordinator functionality despite errors"
            )
            try:
                # Setup event listeners
                self._setup_event_listeners()

                # Start master heartbeat if we became master
                if self._is_master:
                    self._start_master_heartbeat_timer()

                # Start basic timers
                self._start_decay_timer()
                # Analysis timer is async and runs in background
                await self._start_analysis_timer()

                # Verify critical initialization succeeded
                if not self._verify_setup_complete():
                    _LOGGER.error(
                        "Failed to start critical timers for %s after retry",
                        self.config.name,
                    )
                    # Still set complete to allow partial functionality
                    # but log the issue for debugging
                    self._setup_complete = True
                else:
                    # Only mark complete if verification passed
                    self._setup_complete = True

            except (HomeAssistantError, OSError, RuntimeError) as timer_err:
                _LOGGER.error(
                    "Failed to start basic timers for %s: %s",
                    self.config.name,
                    timer_err,
                )
                # Don't set _setup_complete if timers completely failed

    async def update(self) -> dict[str, Any]:
        """Update and return the current coordinator data (in-memory only).

        Database saves are debounced to avoid blocking on every state change.
        See _schedule_save() for the actual save logic.
        """
        # Return current state data (all calculations are in-memory)
        return {
            "probability": self.probability,
            "occupied": self.occupied,
            "threshold": self.threshold,
            "prior": self.area_prior,
            "decay": self.decay,
            "last_updated": dt_util.utcnow(),
        }

    def _schedule_save(self) -> None:
        """Schedule a debounced database save (master only).

        Cancels any pending save and schedules a new one. This ensures that
        rapid state changes result in a single database write after the
        activity settles.
        """
        # Only master saves to database
        if not self._is_master:
            _LOGGER.debug("Skipping save - not master instance")
            return

        # Cancel existing timer if any
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Schedule new save after debounce period
        async def _do_save(_now: datetime) -> None:
            """Perform the actual save operation."""
            self._save_timer = None
            try:
                await self.hass.async_add_executor_job(self.db.save_data)
                _LOGGER.debug(
                    "Debounced database save completed for %s", self.config.name
                )
            except (HomeAssistantError, OSError, RuntimeError) as err:
                _LOGGER.error("Failed to save data for %s: %s", self.config.name, err)

        self._save_timer = async_call_later(self.hass, SAVE_DEBOUNCE_SECONDS, _do_save)
        _LOGGER.debug(
            "Database save scheduled in %d seconds for %s",
            SAVE_DEBOUNCE_SECONDS,
            self.config.name,
        )

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # If master, release role and perform final save
        if self._is_master:
            # Release master role
            try:
                await self.hass.async_add_executor_job(self.db.release_master)
                _LOGGER.info("Released master role for %s", self.config.name)
            except (HomeAssistantError, OSError, RuntimeError) as err:
                _LOGGER.warning("Failed to release master role: %s", err)

            # Cancel pending save timer before cleanup
            if self._save_timer is not None:
                self._save_timer()
                self._save_timer = None
                # Perform final save to ensure no data loss
                try:
                    await self.hass.async_add_executor_job(self.db.save_data)
                    _LOGGER.info(
                        "Final database save completed for %s", self.config.name
                    )
                except (HomeAssistantError, OSError, RuntimeError) as err:
                    _LOGGER.error("Failed final save for %s: %s", self.config.name, err)

            # Cancel master heartbeat timer
            if self._master_heartbeat_timer is not None:
                self._master_heartbeat_timer()
                self._master_heartbeat_timer = None

        # Cancel master health timer
        if self._master_health_timer is not None:
            self._master_health_timer()
            self._master_health_timer = None

        # Cancel all event listeners
        for listener in self._event_listeners:
            listener()
        self._event_listeners.clear()

        # Cancel prior update tracker
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

        # Clean up state change listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Clean up historical timer
        if self._analysis_timer is not None:
            self._analysis_timer()
            self._analysis_timer = None

        # Clean up save timer (in case it wasn't handled in earlier logic)
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Clean up entity manager
        await self.entities.cleanup()

        # Clean up purpose manager
        self.purpose.cleanup()

        await super().async_shutdown()

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options."""
        # Update config
        await self.config.update_config(options)

        # Update purpose with new configuration
        await self.purpose.async_initialize()

        # Always re-initialize entities and entity types when configuration changes
        _LOGGER.info(
            "Configuration updated, re-initializing entities for %s", self.config.name
        )

        # Clean up existing entity tracking and re-initialize
        await self.entities.cleanup()

        # Re-establish entity state tracking with new entity list
        await self.track_entity_state_changes(self.entities.entity_ids)

        # Force immediate save after configuration changes (master only)
        if self._is_master:
            await self.hass.async_add_executor_job(self.db.save_data)

        # Only request refresh if setup is complete to avoid debouncer conflicts
        if self.setup_complete:
            await self.async_request_refresh()

    # --- Entity State Tracking ---
    async def track_entity_state_changes(self, entity_ids: list[str]) -> None:
        """Track state changes for a list of entity_ids."""
        # Clean up existing listener if it exists
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Only create new listener if we have entities to track
        if entity_ids:

            async def _refresh_on_state_change(event: Any) -> None:
                entity_id = event.data.get("entity_id")
                entity = self.entities.get_entity(entity_id)
                if entity and entity.has_new_evidence() and self.setup_complete:
                    await self.async_refresh()

                    # Master: debounced save
                    # Non-master: send state change request to master
                    if self._is_master:
                        self._schedule_save()
                    else:
                        self._send_state_change_request()

            self._remove_state_listener = async_track_state_change_event(
                self.hass, entity_ids, _refresh_on_state_change
            )

    # --- Decay Timer Handling ---
    def _start_decay_timer(self) -> None:
        """Start the global decay timer (always-on implementation)."""
        if self._global_decay_timer is not None or not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(seconds=DECAY_INTERVAL)

        self._global_decay_timer = async_track_point_in_time(
            self.hass, self._handle_decay_timer, next_update
        )

    async def _handle_decay_timer(self, _now: datetime) -> None:
        """Handle decay timer firing - refresh coordinator and always reschedule."""
        self._global_decay_timer = None

        # Refresh the coordinator if decay is enabled
        if self.config.decay.enabled:
            await self.async_refresh()
            self._schedule_save()  # Schedule save after decay update

        # Reschedule the timer
        self._start_decay_timer()

    # --- Historical Timer Handling ---
    async def _start_analysis_timer(self) -> None:
        """Start the historical data import timer with round-robin staggering."""
        if self._analysis_timer is not None or not self.hass:
            return

        # Get instance position for staggering
        position = await self.hass.async_add_executor_job(
            self.db.get_instance_position, self.entry_id
        )

        # Calculate stagger: position * 2 minutes
        stagger_seconds = position * (ANALYSIS_STAGGER_MINUTES * 60)

        # First analysis: 5 minutes + stagger
        # Subsequent analyses will also be staggered
        next_update = dt_util.utcnow() + timedelta(minutes=5, seconds=stagger_seconds)

        _LOGGER.info(
            "Starting analysis timer for %s with position %d (stagger: %d seconds)",
            self.config.name,
            position,
            stagger_seconds,
        )

        self._analysis_timer = async_track_point_in_time(
            self.hass, self.run_analysis, next_update
        )

    async def run_analysis(self, _now: datetime | None = None) -> None:
        """Handle the historical data import timer (all instances, staggered)."""
        if _now is None:
            _now = dt_util.utcnow()
        self._analysis_timer = None

        try:
            # Import recent data from recorder
            await self.db.sync_states()

            # Only master: prune old intervals and run health check
            if self._is_master:
                health_ok = await self.hass.async_add_executor_job(
                    self.db.periodic_health_check
                )
                if not health_ok:
                    _LOGGER.warning(
                        "Database health check found issues for %s", self.config.name
                    )

                pruned_count = await self.hass.async_add_executor_job(
                    self.db.prune_old_intervals
                )
                if pruned_count > 0:
                    _LOGGER.info(
                        "Pruned %d old intervals during analysis", pruned_count
                    )

            # All instances: Recalculate priors and likelihoods with new data
            await self.prior.update()
            await self.entities.update_likelihoods()

            # Refresh the coordinator
            await self.async_refresh()

            # Master saves, non-master sends state change request
            if self._is_master:
                await self.hass.async_add_executor_job(self.db.save_data)
            else:
                self._send_state_change_request()

            # Schedule next run (1 hour interval, maintaining stagger)
            # Note: stagger is maintained naturally since _now already includes the initial offset
            next_update = _now + timedelta(seconds=ANALYSIS_INTERVAL)
            self._analysis_timer = async_track_point_in_time(
                self.hass, self.run_analysis, next_update
            )

        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error("Failed to run historical analysis: %s", err)
            # Reschedule analysis even if it failed
            next_update = _now + timedelta(minutes=15)  # Retry sooner if failed
            self._analysis_timer = async_track_point_in_time(
                self.hass, self.run_analysis, next_update
            )
