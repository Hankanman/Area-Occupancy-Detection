"""Area Occupancy Coordinator."""

from __future__ import annotations

from datetime import datetime, timedelta

# Standard Library
import logging
import os
from typing import Any

from custom_components.area_occupancy.db import AreaOccupancyDB

# Third Party
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local
from .const import (
    DEFAULT_NAME,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)
from .data.config import Config
from .data.entity import EntityFactory, EntityManager
from .data.entity_type import InputType
from .data.prior import Prior
from .data.purpose import PurposeManager
from .utils import bayesian_probability

_LOGGER = logging.getLogger(__name__)

# Global timer intervals in seconds
DECAY_INTERVAL = 10
ANALYSIS_INTERVAL = 3600  # 1 hour in seconds


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
        self.hass = hass
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
        self._setup_complete: bool = False

    async def async_init_database(self) -> None:
        """Initialize the database asynchronously to avoid blocking the event loop.

        This method should be called after coordinator creation but before
        async_config_entry_first_refresh().

        Note: In test environments with AREA_OCCUPANCY_AUTO_INIT_DB=1, the database
        is already initialized in AreaOccupancyDB.__init__(), and this method returns early without performing initialization.
        """
        # In test environments, database is already initialized in __init__
        if os.getenv("AREA_OCCUPANCY_AUTO_INIT_DB") == "1":
            return

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

        return bayesian_probability(
            entities=self.entities.entities,
            area_prior=self.prior.value,
            time_prior=self.prior.time_prior,
        )

    @property
    def type_probabilities(self) -> dict[str, float]:
        """Calculate and return the current occupancy probabilities for each entity type (0.0-1.0)."""
        if not self.entities.entities:
            return {}

        return {
            InputType.MOTION: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.MOTION),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.MEDIA: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.MEDIA),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.APPLIANCE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.APPLIANCE),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.DOOR: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.DOOR),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.WINDOW: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.WINDOW),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.ILLUMINANCE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(
                    InputType.ILLUMINANCE
                ),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.HUMIDITY: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.HUMIDITY),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
            ),
            InputType.TEMPERATURE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(
                    InputType.TEMPERATURE
                ),
                area_prior=self.prior.value,
                time_prior=self.prior.time_prior,
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

    # --- Public Methods ---
    async def setup(self) -> None:
        """Initialize the coordinator and its components (fast startup mode)."""
        try:
            _LOGGER.info(
                "Initializing Area Occupancy for %s (quick startup mode)",
                self.config.name,
            )

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

            # Check if intervals table is empty (fast check)
            try:
                is_empty = await self.hass.async_add_executor_job(
                    self.db.safe_is_intervals_empty
                )
            except (HomeAssistantError, OSError, RuntimeError) as e:
                _LOGGER.warning(
                    "Failed to check intervals table, assuming empty: %s", e
                )
                is_empty = True

            if is_empty:
                _LOGGER.info(
                    "Database has no historical data for %s. Initial analysis scheduled for background after startup completes.",
                    self.entry_id,
                )
                # DON'T run analysis here - defer to background task (already scheduled by _start_analysis_timer)
                _LOGGER.info(
                    "Heavy historical analysis deferred to background task (will run in ~5 minutes)"
                )
            else:
                _LOGGER.info(
                    "Database has existing historical data for %s. Background health check scheduled for 60 seconds after startup.",
                    self.entry_id,
                )
                _LOGGER.debug(
                    "If database corruption is detected, automatic recovery will run in background "
                    "without blocking integration functionality."
                )

            # Track entity state changes
            await self.track_entity_state_changes(self.entities.entity_ids)

            # Start timers only after everything is ready
            self._start_decay_timer()
            self._start_analysis_timer()
            self._start_health_check_timer()

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
                # Start basic timers
                self._start_decay_timer()
                self._start_analysis_timer()
                self._start_health_check_timer()
                # Mark setup as complete - async_refresh() will be called by async_config_entry_first_refresh()
                self._setup_complete = True
            except (HomeAssistantError, OSError, RuntimeError) as timer_err:
                _LOGGER.error("Failed to start basic timers: %s", timer_err)

    async def update(self) -> dict[str, Any]:
        """Update and return the current coordinator data."""
        # Save current state to database
        await self.hass.async_add_executor_job(self.db.save_data)

        # Return current state data
        return {
            "probability": self.probability,
            "occupied": self.occupied,
            "threshold": self.threshold,
            "prior": self.area_prior,
            "decay": self.decay,
            "last_updated": dt_util.utcnow(),
        }

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
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

        # Clean up health check timer
        if self._health_check_timer is not None:
            self._health_check_timer()
            self._health_check_timer = None

        await self.hass.async_add_executor_job(self.db.save_data)

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

        # Force immediate save after configuration changes
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

        # Reschedule the timer
        self._start_decay_timer()

    # --- Historical Timer Handling ---
    def _start_analysis_timer(self) -> None:
        """Start the historical data import timer."""
        if self._analysis_timer is not None or not self.hass:
            return

        # Run first import shortly after startup (5 minutes)
        next_update = dt_util.utcnow() + timedelta(minutes=5)

        self._analysis_timer = async_track_point_in_time(
            self.hass, self.run_analysis, next_update
        )

    async def run_analysis(self, _now: datetime | None = None) -> None:
        """Handle the historical data import timer."""
        if _now is None:
            _now = dt_util.utcnow()
        self._analysis_timer = None

        try:
            # Import recent data from recorder
            await self.db.sync_states()

            # Prune old intervals to prevent database growth
            pruned_count = await self.hass.async_add_executor_job(
                self.db.prune_old_intervals
            )
            if pruned_count > 0:
                _LOGGER.info("Pruned %d old intervals during analysis", pruned_count)

            # Recalculate priors with new data
            await self.prior.update()

            # Recalculate likelihoods with new data
            await self.entities.update_likelihoods()

            # Refresh the coordinator
            await self.async_refresh()

            # Save data to database
            await self.hass.async_add_executor_job(self.db.save_data)

            # Schedule next analysis (every hour)
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

    # --- Health Check Timer Handling ---
    def _start_health_check_timer(self) -> None:
        """Start the database health check timer (staggered across instances).

        First check runs 60 seconds after startup. Subsequent checks run every
        6 hours, with each instance offset by its entry_id hash to prevent
        simultaneous checks from multiple instances.
        """
        if self._health_check_timer is not None or not self.hass:
            return

        # First check after startup
        first_check_delay = 60

        # Stagger subsequent checks based on entry_id hash
        # Spreads instances across 6-hour window (0-359 minutes apart)
        entry_hash = hash(self.entry_id) % 360  # 0-359 minutes
        stagger_minutes = entry_hash

        next_update = dt_util.utcnow() + timedelta(seconds=first_check_delay)

        _LOGGER.debug(
            "Health check scheduled for %s, with %d minute stagger for subsequent checks",
            next_update,
            stagger_minutes,
        )

        self._health_check_timer = async_track_point_in_time(
            self.hass, self._handle_health_check_timer, next_update
        )

        # Store stagger for next schedule
        self._health_check_stagger = stagger_minutes

    async def _handle_health_check_timer(self, _now: datetime) -> None:
        """Handle health check timer firing."""
        self._health_check_timer = None

        try:
            _LOGGER.info(
                "Running background database health check for %s", self.config.name
            )

            # Perform database health check in executor to avoid blocking
            health_ok = await self.hass.async_add_executor_job(
                self.db.periodic_health_check
            )

            if health_ok:
                _LOGGER.info("Database health check passed for %s", self.config.name)
            else:
                _LOGGER.warning(
                    "Database health check found issues for %s", self.config.name
                )

            # Schedule next check (6 hours + stagger to prevent simultaneous checks)
            base_interval = timedelta(hours=6)
            stagger = timedelta(minutes=getattr(self, "_health_check_stagger", 0))
            next_update = dt_util.utcnow() + base_interval + stagger

            self._health_check_timer = async_track_point_in_time(
                self.hass, self._handle_health_check_timer, next_update
            )

        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error("Health check timer failed: %s", err)
            # Reschedule with backoff (retry in 1 hour if failed)
            next_update = dt_util.utcnow() + timedelta(hours=1)
            self._health_check_timer = async_track_point_in_time(
                self.hass, self._handle_health_check_timer, next_update
            )
