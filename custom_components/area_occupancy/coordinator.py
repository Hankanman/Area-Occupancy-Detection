"""Area Occupancy Coordinator."""

from __future__ import annotations

# Standard library imports
from datetime import datetime, timedelta
import logging
from typing import Any

from custom_components.area_occupancy.data.entity_type import InputType

# Home Assistant imports
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local imports
from .area import AllAreas, Area, AreaDeviceHandle
from .const import CONF_AREA_ID, CONF_AREAS, DEFAULT_NAME, DOMAIN, SAVE_INTERVAL
from .data.config import IntegrationConfig
from .db import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)


class AreaOccupancyCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Manage fetching and combining data for area occupancy."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DEFAULT_NAME,
            update_interval=None,
            setup_method=self.setup,
            update_method=self.update,
        )
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        self.db = AreaOccupancyDB(self)

        # Integration-level configuration (global settings for entire integration)
        self.integration_config = IntegrationConfig(self, config_entry)

        # Multi-area architecture: dict[str, Area] keyed by area name
        self.areas: dict[str, Area] = {}
        self._area_handles: dict[str, AreaDeviceHandle] = {}

        # All Areas aggregator (lazy initialization)
        self._all_areas: AllAreas | None = None

        # Per-area state listeners (area_name -> callback)
        self._area_state_listeners: dict[str, CALLBACK_TYPE] = {}
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._analysis_timer: CALLBACK_TYPE | None = None
        self._save_timer: CALLBACK_TYPE | None = None
        self._nightly_timer: CALLBACK_TYPE | None = None
        self._setup_complete: bool = False

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

    def _load_areas_from_config(
        self, target_dict: dict[str, Area] | None = None
    ) -> None:
        """Load areas from config entry.

        Loads areas from CONF_AREAS list format.

        Args:
            target_dict: Optional dict to load areas into. If None, loads into self.areas.
        """
        merged = dict(self.config_entry.data)
        merged.update(self.config_entry.options)

        # Use target_dict if provided, otherwise use self.areas
        areas_dict = target_dict if target_dict is not None else self.areas

        area_reg = ar.async_get(self.hass)
        areas_to_remove: list[str] = []  # Track areas to remove (deleted or invalid)

        # Load areas from CONF_AREAS list
        if CONF_AREAS not in merged or not isinstance(merged[CONF_AREAS], list):
            _LOGGER.error(
                "Configuration must contain CONF_AREAS list. "
                "Please reconfigure the integration."
            )
            return

        areas_list = merged[CONF_AREAS]
        for area_data in areas_list:
            area_id = area_data.get(CONF_AREA_ID)

            if not area_id:
                _LOGGER.warning("Skipping area without area ID: %s", area_data)
                continue

            # Validate that area ID exists in Home Assistant
            area_entry = area_reg.async_get_area(area_id)
            if not area_entry:
                _LOGGER.warning(
                    "Area ID '%s' not found in Home Assistant registry. "
                    "Area may have been deleted. Removing from configuration.",
                    area_id,
                )
                areas_to_remove.append(area_id)
                continue

            # Resolve area name from ID
            area_name = area_entry.name

            # Check for duplicate area IDs
            if area_name in areas_dict:
                _LOGGER.warning("Duplicate area name %s, skipping", area_name)
                continue

            # Create Area for this area
            areas_dict[area_name] = Area(
                coordinator=self,
                area_name=area_name,
                area_data=area_data,
            )
            self.get_area_handle(area_name).attach(areas_dict[area_name])
            _LOGGER.debug("Loaded area: %s (ID: %s)", area_name, area_id)

        # Log warnings for deleted/invalid areas
        if areas_to_remove:
            _LOGGER.warning(
                "Found %d deleted or invalid area(s) in configuration. "
                "These areas will be skipped. Please reconfigure via options flow if needed.",
                len(areas_to_remove),
            )

    def get_area_handle(self, area_name: str) -> AreaDeviceHandle:
        """Return a stable handle for the requested area."""
        handle = self._area_handles.get(area_name)
        if handle is None:
            handle = AreaDeviceHandle(self, area_name)
            self._area_handles[area_name] = handle
        return handle

    def get_area(self, area_name: str | None = None) -> Area | None:
        """Get area by name, or return first area if None.

        Args:
            area_name: Optional area name, None returns first area

        Returns:
            Area instance (always returns first area when area_name is None),
            or None if the specified area_name doesn't exist or no areas exist
        """
        if area_name is None:
            # Return first area (at least one area always exists in normal operation)
            # Handle empty case for tests/edge cases
            if not self.areas:
                return None
            return next(iter(self.areas.values()))
        return self.areas.get(area_name)

    def get_area_names(self) -> list[str]:
        """Get list of all configured area names."""
        return list(self.areas.keys())

    def find_area_for_entity(self, entity_id: str) -> str | None:
        """Find which area contains a specific entity.

        Args:
            entity_id: The entity ID to search for

        Returns:
            Area name if entity is found, None otherwise
        """
        for area_name, area in self.areas.items():
            try:
                area.entities.get_entity(entity_id)
            except ValueError:
                continue
            else:
                return area_name
        return None

    def get_all_areas(self) -> AllAreas:
        """Get or create the AllAreas aggregator instance.

        Returns:
            AllAreas instance for aggregating data across all areas
        """
        if self._all_areas is None:
            self._all_areas = AllAreas(self)
        return self._all_areas

    @property
    def setup_complete(self) -> bool:
        """Return whether setup is complete."""
        return self._setup_complete

    def _format_area_names_for_logging(self) -> str:
        """Format area names for logging purposes.

        Returns:
            Comma-separated string of area names, or "no areas" if empty
        """
        return ", ".join(self.get_area_names()) if self.areas else "no areas"

    # --- Public Methods ---
    def _validate_areas_configured(self) -> None:
        """Validate that at least one area is configured.

        Raises:
            HomeAssistantError: If no areas are configured
        """
        if not self.areas:
            raise HomeAssistantError("No areas configured")

    async def setup(self) -> None:
        """Initialize the coordinator and its components (fast startup mode)."""
        try:
            # Load areas from config entry
            self._load_areas_from_config()

            self._validate_areas_configured()

            _LOGGER.info(
                "Initializing Area Occupancy for %d area(s): %s",
                len(self.areas),
                ", ".join(self.areas.keys()),
            )

            # Initialize each area
            for area_name in self.areas:
                _LOGGER.debug("Initializing area: %s", area_name)

                # Load stored data from database for this area
                # Note: Database load will restore priors and entities per area
                # Database integrity checks are deferred to background (60s after startup)
                _LOGGER.debug(
                    "Loading entity data from database for area %s (deferring heavy operations)",
                    area_name,
                )

            # Load data from database
            await self.db.load_data()

            # Ensure areas exist in database and persist configuration/state
            for area_name in self.areas:
                try:
                    # Save area data to database for this specific area
                    await self.hass.async_add_executor_job(
                        self.db.save_area_data, area_name
                    )
                except (HomeAssistantError, OSError, RuntimeError) as e:
                    _LOGGER.warning(
                        "Failed to save area data for %s, continuing setup: %s",
                        area_name,
                        e,
                    )

            # Track entity state changes for all areas
            all_entity_ids = []
            for area in self.areas.values():
                all_entity_ids.extend(area.entities.entity_ids)

            # Remove duplicates
            all_entity_ids = list(set(all_entity_ids))
            await self.track_entity_state_changes(all_entity_ids)

            # Start timers only after everything is ready
            self._start_decay_timer()
            self._start_save_timer()
            # Analysis timer is async and runs in background
            await self._start_analysis_timer()
            self._start_nightly_timer()

            # Mark setup as complete before initial refresh to prevent debouncer conflicts
            self._setup_complete = True

            # Log initialization summary
            total_entities = sum(
                len(area.entities.entities) for area in self.areas.values()
            )
            _LOGGER.info(
                "Successfully initialized %d area(s) with %d total entities",
                len(self.areas),
                total_entities,
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
                self._start_save_timer()
                # Analysis timer is async and runs in background
                await self._start_analysis_timer()
                self._start_nightly_timer()

                self._setup_complete = True

            except (HomeAssistantError, OSError, RuntimeError) as timer_err:
                _LOGGER.error(
                    "Failed to start basic timers for areas: %s: %s",
                    self._format_area_names_for_logging(),
                    timer_err,
                )
                # Don't set _setup_complete if timers completely failed

    async def update(self) -> dict[str, Any]:
        """Update and return the current coordinator data (in-memory only).

        Returns:
            Dictionary with area data keyed by area name
        """
        # Return current state data for all areas (all calculations are in-memory)
        result = {}
        for area_name, area in self.areas.items():
            result[area_name] = {
                "probability": area.probability(),
                "occupied": area.occupied(),
                "threshold": area.threshold(),
                "prior": area.area_prior(),
                "decay": area.decay(),
                "last_updated": dt_util.utcnow(),
            }

        # For backward compatibility, also include first area at root level
        # Only copy keys that won't overwrite area entries (i.e., keys that don't
        # already exist in result as a dict, which would indicate an area entry)
        if self.areas:
            first_area = next(iter(self.areas.keys()))
            first_area_data = result[first_area]
            for key, value in first_area_data.items():
                # Only copy if key doesn't already exist as an area entry (dict)
                # This prevents overwriting area dicts when area name equals a reserved key
                if not (key in result and isinstance(result[key], dict)):
                    result[key] = value

        # Only set root-level last_updated if it won't overwrite an area dict
        if not ("last_updated" in result and isinstance(result["last_updated"], dict)):
            result["last_updated"] = dt_util.utcnow()
        return result

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator.

        Cleanup order is important to prevent circular references and memory leaks:
        1. Cancel timers and listeners first
        2. Save final state
        3. Clean up areas (which clears their internal caches and references)
        4. Reset aggregators
        5. Dispose database engine (after all areas are cleaned up)
        6. Call parent shutdown
        """
        _LOGGER.info(
            "Starting coordinator shutdown for areas: %s",
            self._format_area_names_for_logging(),
        )

        # Step 1: Cancel periodic save timer before cleanup and perform final save
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Step 2: Perform final save to ensure no data loss
        try:
            await self.hass.async_add_executor_job(self.db.save_data)
            _LOGGER.info(
                "Final database save completed for areas: %s",
                self._format_area_names_for_logging(),
            )
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error(
                "Failed final save for areas: %s: %s",
                self._format_area_names_for_logging(),
                err,
            )

        # Step 3: Cancel all area state listeners
        for listener in self._area_state_listeners.values():
            listener()
        self._area_state_listeners.clear()

        # Step 4: Cancel prior update tracker
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

        # Step 5: Clean up historical timer
        if self._analysis_timer is not None:
            self._analysis_timer()
            self._analysis_timer = None

        if self._nightly_timer is not None:
            self._nightly_timer()
            self._nightly_timer = None

        # Step 6: Clean up periodic save timer (defensive check)
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Step 7: Clean up all areas (clears caches, entities, and internal references)
        for area in list(self.areas.values()):
            await area.async_cleanup()

        # Step 8: Reset AllAreas aggregator to release references to old areas
        # This must be done after areas are cleaned up to break circular references
        self._all_areas = None

        # Step 9: Dispose database engine to close all connections
        # This must be done after all areas are cleaned up to ensure no active sessions
        try:
            if hasattr(self.db, "engine") and self.db.engine is not None:
                self.db.engine.dispose(close=True)
        except (OSError, RuntimeError) as err:
            _LOGGER.warning("Error disposing database engine: %s", err)

        # Step 10: Clear areas dict to release all area references
        # This helps break any remaining circular references
        # Format area names before clearing (since we need them for logging)
        area_names_str = self._format_area_names_for_logging()
        self.areas.clear()

        _LOGGER.info("Coordinator shutdown completed for areas: %s", area_names_str)
        await super().async_shutdown()

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options.

        Args:
            options: Updated options dict (may contain CONF_AREAS for multi-area updates)
        """
        if self.config_entry is None:
            raise HomeAssistantError("Cannot update options: config_entry is None")

        # Load new areas into a temporary dict first to avoid race condition
        # where self.areas is empty while platform entities are still active
        new_areas: dict[str, Area] = {}
        self._load_areas_from_config(target_dict=new_areas)

        # Identify areas that will be removed by comparing old and new area names
        removed_area_names = set(self.areas.keys()) - set(new_areas.keys())
        if removed_area_names:
            _LOGGER.info(
                "Cleaning up %d removed area(s): %s",
                len(removed_area_names),
                ", ".join(removed_area_names),
            )

            # Get entity registry for cleanup
            entity_registry = er.async_get(self.hass)

            for area_name in removed_area_names:
                area = self.areas.get(area_name)
                if area:
                    # Clean up removed area
                    await area.async_cleanup()
                    self.get_area_handle(area_name).attach(None)

                    # Remove entities from entity registry
                    entities_removed = 0
                    area_prefix = f"{area_name}_"
                    for entity_id, entity_entry in list(
                        entity_registry.entities.items()
                    ):
                        if (
                            entity_entry.config_entry_id == self.entry_id
                            and entity_entry.unique_id
                            and str(entity_entry.unique_id).startswith(area_prefix)
                        ):
                            try:
                                entity_registry.async_remove(entity_id)
                                entities_removed += 1
                                _LOGGER.debug(
                                    "Removed entity %s from registry for removed area %s",
                                    entity_id,
                                    area_name,
                                )
                            except (ValueError, KeyError, AttributeError) as remove_err:
                                _LOGGER.warning(
                                    "Failed to remove entity %s from registry: %s",
                                    entity_id,
                                    remove_err,
                                )

                    if entities_removed > 0:
                        _LOGGER.info(
                            "Removed %d entities from registry for removed area %s",
                            entities_removed,
                            area_name,
                        )

                    # Remove device from device registry
                    device_registry = dr.async_get(self.hass)
                    device_identifiers = {(DOMAIN, area.config.area_id)}
                    device = device_registry.async_get_device(
                        identifiers=device_identifiers
                    )
                    if device:
                        try:
                            device_registry.async_remove_device(device.id)
                            _LOGGER.info(
                                "Removed device %s from registry for removed area %s",
                                device.id,
                                area_name,
                            )
                        except (ValueError, KeyError, AttributeError) as remove_err:
                            _LOGGER.warning(
                                "Failed to remove device %s from registry: %s",
                                device.id,
                                remove_err,
                            )
                    else:
                        _LOGGER.debug(
                            "No device found for removed area %s (area_id: %s)",
                            area_name,
                            area.config.area_id,
                        )

                    # Delete all database records for this area
                    try:
                        deleted_count = await self.hass.async_add_executor_job(
                            self.db.delete_area_data, area_name
                        )
                        _LOGGER.debug(
                            "Deleted %d database records for removed area %s",
                            deleted_count,
                            area_name,
                        )
                    except (HomeAssistantError, OSError, RuntimeError) as db_err:
                        _LOGGER.error(
                            "Failed to delete database records for removed area %s: %s",
                            area_name,
                            db_err,
                        )

                    _LOGGER.info("Cleaned up removed area: %s", area_name)

        # Cancel existing entity state listeners (will be recreated with new entity lists)
        for listener in self._area_state_listeners.values():
            listener()
        self._area_state_listeners.clear()

        # Reset AllAreas aggregator to release references to old areas
        self._all_areas = None
        _LOGGER.debug("Reset AllAreas aggregator after area update")

        # Atomically replace self.areas with new_areas
        # This ensures self.areas is never empty when platform entities can access it
        self.areas = new_areas

        # Update area handles to point to new Area objects
        # This ensures platform entities can access the updated areas
        for area_name, area in self.areas.items():
            self.get_area_handle(area_name).attach(area)

        # Update each area's configuration
        # Clean up and recreate entities from new config first
        for area_name, area in self.areas.items():
            _LOGGER.info(
                "Configuration updated, re-initializing entities for area: %s",
                area_name,
            )

            # Clean up existing entity tracking and recreate from new config
            # This ensures entities match the updated configuration
            await area.entities.cleanup()

            # Area components are now initialized synchronously in __init__

        # Reload database data for new areas (restores priors, entity states)
        # This must happen AFTER entities are recreated from config so database
        # state can be applied to the correctly configured entities
        # This is critical to restore state after config changes without requiring a full reload
        await self.db.load_data()

        # Re-establish entity state tracking with new entity lists
        all_entity_ids = []
        for area in self.areas.values():
            all_entity_ids.extend(area.entities.entity_ids)

        # Remove duplicates
        all_entity_ids = list(set(all_entity_ids))
        await self.track_entity_state_changes(all_entity_ids)

        # Force immediate save after configuration changes
        await self.hass.async_add_executor_job(self.db.save_data)

        # Only request refresh if setup is complete to avoid debouncer conflicts
        if self.setup_complete:
            await self.async_request_refresh()

    # --- Entity State Tracking ---
    async def track_entity_state_changes(self, entity_ids: list[str]) -> None:
        """Track state changes for a list of entity_ids across all areas."""
        # Clean up existing listeners
        for listener in self._area_state_listeners.values():
            listener()
        self._area_state_listeners.clear()

        # Only create listener if we have entities to track
        if entity_ids:

            async def _refresh_on_state_change(event: Any) -> None:
                entity_id = event.data.get("entity_id")
                if not entity_id:
                    return

                # Find which area(s) this entity belongs to
                affected_areas = []
                for area_name, area in self.areas.items():
                    try:
                        entity = area.entities.get_entity(entity_id)
                    except ValueError:
                        # Entity doesn't belong to this area, skip it
                        continue
                    if entity.has_new_evidence():
                        affected_areas.append(area_name)

                # If entity affects any area and setup is complete, refresh
                if affected_areas and self.setup_complete:
                    await self.async_refresh()

            # Create single listener for all entities (more efficient than per-area listeners)
            listener = async_track_state_change_event(
                self.hass, entity_ids, _refresh_on_state_change
            )
            # Store listener (using a single key since we have one listener for all)
            self._area_state_listeners["_all"] = listener

    # --- Save Timer Handling ---
    def _start_save_timer(self) -> None:
        """Start the periodic database save timer (runs every 10 minutes)."""
        if self._save_timer is not None or not self.hass:
            return

        next_save = dt_util.utcnow() + timedelta(seconds=SAVE_INTERVAL)

        self._save_timer = async_track_point_in_time(
            self.hass, self._handle_save_timer, next_save
        )

    async def _handle_save_timer(self, _now: datetime) -> None:
        """Handle periodic save timer firing - save data and reschedule."""
        self._save_timer = None

        try:
            await self.hass.async_add_executor_job(self.db.save_data)
            _LOGGER.debug(
                "Periodic database save completed for areas: %s",
                self._format_area_names_for_logging(),
            )
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error(
                "Failed periodic save for areas: %s: %s",
                self._format_area_names_for_logging(),
                err,
            )

        # Reschedule the timer
        self._start_save_timer()

    # --- Decay Timer Handling ---
    def _start_decay_timer(self) -> None:
        """Start the global decay timer (always-on implementation)."""
        if self._global_decay_timer is not None or not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(
            seconds=self.integration_config.decay_interval
        )

        self._global_decay_timer = async_track_point_in_time(
            self.hass, self._handle_decay_timer, next_update
        )

    async def _handle_decay_timer(self, _now: datetime) -> None:
        """Handle decay timer firing - refresh coordinator and always reschedule."""
        self._global_decay_timer = None

        # Refresh the coordinator if decay is enabled for any area
        decay_enabled = any(area.config.decay.enabled for area in self.areas.values())
        if decay_enabled:
            await self.async_refresh()

        # Reschedule the timer
        self._start_decay_timer()

    # --- Analysis Timer Handling ---
    async def _start_analysis_timer(self) -> None:
        """Start the historical data import timer.

        Note: No staggering needed with single-instance architecture.
        """
        if self._analysis_timer is not None or not self.hass:
            return

        # First analysis: 5 minutes after startup
        # Subsequent analyses: 1 hour interval
        next_update = dt_util.utcnow() + timedelta(minutes=5)

        _LOGGER.info(
            "Starting analysis timer for areas: %s",
            self._format_area_names_for_logging(),
        )

        self._analysis_timer = async_track_point_in_time(
            self.hass, self.run_analysis, next_update
        )

    async def run_analysis(self, _now: datetime | None = None) -> None:
        """Handle the historical data import timer.

        Always runs analysis for all areas.

        Args:
            _now: Optional timestamp for the analysis run (used by timer)
        """
        if _now is None:
            _now = dt_util.utcnow()
        self._analysis_timer = None

        try:
            # Import recent data from recorder
            await self.db.sync_states()

            # Prune old intervals and run health check (no master check needed)
            health_ok = await self.hass.async_add_executor_job(
                self.db.periodic_health_check
            )
            if not health_ok:
                _LOGGER.warning(
                    "Database health check found issues for areas: %s",
                    self._format_area_names_for_logging(),
                )

            pruned_count = await self.hass.async_add_executor_job(
                self.db.prune_old_intervals
            )
            if pruned_count > 0:
                _LOGGER.info("Pruned %d old intervals during analysis", pruned_count)

            # Recalculate priors and likelihoods with new data for all areas
            for area in self.areas.values():
                await area.run_prior_analysis()
                await area.run_likelihood_analysis()

            # Refresh the coordinator
            await self.async_refresh()

            # Save data (always save - no master check)
            await self.hass.async_add_executor_job(self.db.save_data)

            # Schedule next run (1 hour interval)
            next_update = _now + timedelta(
                seconds=self.integration_config.analysis_interval
            )
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

    # --- Nightly Timer Handling ---
    def _start_nightly_timer(self) -> None:
        """Start nightly interval aggregation timer (02:00 local time)."""
        if self._nightly_timer is not None or not self.hass:
            return

        next_run = self._next_interval_aggregation_run()
        self._nightly_timer = async_track_point_in_time(
            self.hass, self.run_interval_aggregation_job, next_run
        )

    def _next_interval_aggregation_run(
        self, reference_local: datetime | None = None
    ) -> datetime:
        """Return the next 02:00 local time run expressed in UTC."""
        if reference_local is None:
            reference_local = dt_util.now()

        target_time = reference_local.replace(hour=2, minute=0, second=0, microsecond=0)
        if target_time <= reference_local:
            target_time += timedelta(days=1)

        return dt_util.as_utc(target_time)

    def _get_numeric_entities_by_area(self) -> dict[str, list[str]]:
        """Return mapping of area_name to numeric entity IDs."""
        numeric_inputs = {
            InputType.TEMPERATURE,
            InputType.HUMIDITY,
            InputType.ILLUMINANCE,
            InputType.ENVIRONMENTAL,
        }
        numeric_entities: dict[str, list[str]] = {}

        for area_name, area in self.areas.items():
            entities_container = getattr(area.entities, "entities", {})
            area_numeric: list[str] = []
            for entity_id, entity in entities_container.items():
                entity_type = getattr(entity, "type", None)
                input_type = getattr(entity_type, "input_type", None)
                if input_type in numeric_inputs:
                    area_numeric.append(entity_id)
            if area_numeric:
                numeric_entities[area_name] = area_numeric

        return numeric_entities

    async def run_interval_aggregation_job(
        self, _now: datetime | None = None
    ) -> dict[str, Any]:
        """Handle nightly interval aggregation timer firing."""
        self._nightly_timer = None
        if _now is None:
            _now = dt_util.utcnow()

        summary: dict[str, Any] = {
            "aggregation": None,
            "correlations": [],
            "errors": [],
        }

        try:
            results = await self.hass.async_add_executor_job(
                self.db.run_interval_aggregation
            )
            summary["aggregation"] = results
            _LOGGER.info(
                "Interval aggregation completed for areas %s: %s",
                self._format_area_names_for_logging(),
                results,
            )

            numeric_entities = self._get_numeric_entities_by_area()
            if numeric_entities:
                _LOGGER.info(
                    "Starting numeric correlation analysis for %d area(s)",
                    len(numeric_entities),
                )
                for area_name, entity_ids in numeric_entities.items():
                    for entity_id in entity_ids:
                        try:
                            correlation_result = await self.hass.async_add_executor_job(
                                self.db.analyze_and_save_correlation,
                                area_name,
                                entity_id,
                            )
                            summary["correlations"].append(
                                {
                                    "area": area_name,
                                    "entity_id": entity_id,
                                    "success": bool(correlation_result),
                                }
                            )
                        except Exception as err:  # noqa: BLE001
                            _LOGGER.error(
                                "Correlation analysis failed for %s (%s): %s",
                                area_name,
                                entity_id,
                                err,
                            )
                            summary["correlations"].append(
                                {
                                    "area": area_name,
                                    "entity_id": entity_id,
                                    "success": False,
                                    "error": str(err),
                                }
                            )
            else:
                _LOGGER.debug(
                    "Skipping numeric correlation analysis - no numeric entities configured"
                )

        except Exception as err:  # noqa: BLE001
            _LOGGER.error(
                "Interval aggregation failed for areas %s: %s",
                self._format_area_names_for_logging(),
                err,
            )
            summary["errors"].append(str(err))

        next_reference_local = dt_util.as_local(_now)
        next_run = self._next_interval_aggregation_run(next_reference_local)
        self._nightly_timer = async_track_point_in_time(
            self.hass, self.run_interval_aggregation_job, next_run
        )
        return summary
