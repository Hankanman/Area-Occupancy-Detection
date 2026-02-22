"""Area Occupancy Coordinator."""

from __future__ import annotations

# Standard library imports
from datetime import datetime, timedelta
import logging
from typing import Any

# Home Assistant imports
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local imports
from .area import AllAreas, Area, AreaDeviceHandle, FloorAreas
from .const import CONF_AREA_ID, CONF_AREAS, DEFAULT_NAME, DOMAIN, SAVE_INTERVAL
from .data.analysis import run_full_analysis
from .data.config import IntegrationConfig
from .db import AreaOccupancyDB
from .utils import format_area_names

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

        # Floor-based aggregators: dict[floor_id, FloorAreas]
        self._floor_aggregators: dict[str, FloorAreas] = {}

        # Per-area state listeners (area_name -> callback)
        self._area_state_listeners: dict[str, CALLBACK_TYPE] = {}
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._analysis_timer: CALLBACK_TYPE | None = None
        self._save_timer: CALLBACK_TYPE | None = None
        self._setup_complete: bool = False
        self._analysis_running: bool = False
        self._cached_correlations: dict[str, dict[str, float]] = {}

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

    async def async_refresh_correlations(self) -> None:
        """Refresh cached entity correlations for all areas from database.

        Loads correlations via executor to avoid blocking the event loop,
        then stores them on the coordinator for synchronous access during
        probability calculations.
        """
        if self.db is None:
            return

        for area_name in self.areas:
            try:
                from .db.correlation import get_entity_correlations  # noqa: PLC0415

                correlations = await self.hass.async_add_executor_job(
                    get_entity_correlations, self.db, area_name
                )
                self._cached_correlations[area_name] = correlations
            except ImportError:
                _LOGGER.debug(
                    "Failed to import correlation module for area %s",
                    area_name,
                    exc_info=True,
                )
                self._cached_correlations[area_name] = {}
            except (AttributeError, ValueError, OSError, RuntimeError):
                _LOGGER.debug(
                    "Failed to load entity correlations for area %s",
                    area_name,
                    exc_info=True,
                )
                self._cached_correlations[area_name] = {}

    def get_cached_correlations(self, area_name: str) -> dict[str, float]:
        """Return cached correlation strengths for the given area.

        Args:
            area_name: Name of the area to get correlations for

        Returns:
            Dict of entity_id -> correlation strength. Empty dict if no data.
        """
        return self._cached_correlations.get(area_name, {})

    def _load_areas_from_config(
        self, target_dict: dict[str, Area] | None = None
    ) -> None:
        """Load areas from config entry CONF_AREAS list.

        Reads area configurations from the merged data+options CONF_AREAS list.

        Args:
            target_dict: Optional dict to load areas into. If None, loads into self.areas.
        """
        # Use target_dict if provided, otherwise use self.areas
        areas_dict = target_dict if target_dict is not None else self.areas

        area_reg = ar.async_get(self.hass)

        # Merge data and options to find CONF_AREAS.
        merged = dict(self.config_entry.data)
        merged.update(self.config_entry.options)
        areas_list = merged.get(CONF_AREAS, [])

        for area_data in areas_list:
            area_id = area_data.get(CONF_AREA_ID)

            if not area_id:
                _LOGGER.warning("Skipping area config without area ID")
                continue

            # Validate that area ID exists in Home Assistant.
            area_entry = area_reg.async_get_area(area_id)
            if not area_entry:
                _LOGGER.warning(
                    "Area ID '%s' not found in Home Assistant registry. "
                    "Area may have been deleted. Skipping.",
                    area_id,
                )
                continue

            # Resolve area name from ID.
            area_name = area_entry.name

            # Check for duplicate area names.
            if area_name in areas_dict:
                _LOGGER.warning("Duplicate area name %s, skipping", area_name)
                continue

            # Create Area for this area.
            areas_dict[area_name] = Area(
                coordinator=self,
                area_name=area_name,
                area_data=area_data,
            )
            self.get_area_handle(area_name).attach(areas_dict[area_name])
            _LOGGER.debug("Loaded area: %s (ID: %s)", area_name, area_id)

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

    def _build_floor_aggregators(self) -> None:
        """Discover floors from configured areas and create FloorAreas aggregators."""
        area_reg = ar.async_get(self.hass)
        floor_reg = fr.async_get(self.hass)

        seen_floors: dict[str, str] = {}  # floor_id -> floor_name
        for area in self.areas.values():
            if not area.config.area_id:
                continue
            area_entry = area_reg.async_get_area(area.config.area_id)
            if area_entry and area_entry.floor_id:
                if area_entry.floor_id not in seen_floors:
                    floor_entry = floor_reg.async_get_floor(area_entry.floor_id)
                    if floor_entry:
                        seen_floors[area_entry.floor_id] = floor_entry.name

        self._floor_aggregators = {
            floor_id: FloorAreas(self, floor_id, floor_name)
            for floor_id, floor_name in seen_floors.items()
        }

        if self._floor_aggregators:
            _LOGGER.debug(
                "Built %d floor aggregator(s): %s",
                len(self._floor_aggregators),
                ", ".join(f.floor_name for f in self._floor_aggregators.values()),
            )

    def get_floor_aggregators(self) -> dict[str, FloorAreas]:
        """Return floor-based aggregators keyed by floor_id."""
        return self._floor_aggregators

    @property
    def setup_complete(self) -> bool:
        """Return whether setup is complete."""
        return self._setup_complete

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

            # Load cached correlations for probability calculations
            await self.async_refresh_correlations()

            # Ensure areas and entities exist in database and persist configuration/state
            # This must happen before analysis runs so that get_occupied_intervals() can
            # properly JOIN Intervals with Entities table
            try:
                # Save both area and entity data for all areas in one operation
                await self.hass.async_add_executor_job(self.db.save_data)
            except (HomeAssistantError, OSError, RuntimeError) as e:
                _LOGGER.warning(
                    "Failed to save area and entity data, continuing setup: %s", e
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

            # Build floor-based aggregators from area floor assignments
            self._build_floor_aggregators()

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
            _LOGGER.exception("Unexpected error during coordinator setup")
            raise ConfigEntryNotReady(f"Setup failed: {err}. Will retry.") from err

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
            format_area_names(self),
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
                format_area_names(self),
            )
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error(
                "Failed final save for areas: %s: %s",
                format_area_names(self),
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

        # Step 6: Clean up periodic save timer (defensive check)
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Step 7: Clean up all areas (clears caches, entities, and internal references)
        for area in list(self.areas.values()):
            await area.async_cleanup()

        # Step 8: Reset AllAreas and floor aggregators to release references to old areas
        # This must be done after areas are cleaned up to break circular references
        self._all_areas = None
        self._floor_aggregators.clear()

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
        area_names_str = format_area_names(self)
        self.areas.clear()

        _LOGGER.info("Coordinator shutdown completed for areas: %s", area_names_str)
        await super().async_shutdown()

    async def _cleanup_removed_area(self, area_name: str, area: Area) -> None:
        """Clean up a removed area from registries and database.

        Handles cleanup of:
        - Area async cleanup
        - Entity registry entries
        - Device registry entries
        - Database records

        Args:
            area_name: Name of the area to clean up
            area: Area instance to clean up
        """
        # Clean up removed area
        await area.async_cleanup()
        self.get_area_handle(area_name).attach(None)

        # Look up device for this area (used for both entity and device removal)
        device_registry = dr.async_get(self.hass)
        device_identifiers = {(DOMAIN, area.config.area_id)}
        device = device_registry.async_get_device(identifiers=device_identifiers)

        # Remove entities from entity registry that belong to this device
        entity_registry = er.async_get(self.hass)
        entities_removed = 0
        target_device_id = device.id if device else None
        for entity_id, entity_entry in list(entity_registry.entities.items()):
            if (
                entity_entry.config_entry_id == self.entry_id
                and target_device_id is not None
                and entity_entry.device_id == target_device_id
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

        # Remove device from device registry (reusing device object from above)
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

            for area_name in removed_area_names:
                area = self.areas.get(area_name)
                if area:
                    await self._cleanup_removed_area(area_name, area)

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

        # Refresh cached correlations for new areas
        await self.async_refresh_correlations()

        # Re-establish entity state tracking with new entity lists
        all_entity_ids = []
        for area in self.areas.values():
            all_entity_ids.extend(area.entities.entity_ids)

        # Remove duplicates
        all_entity_ids = list(set(all_entity_ids))
        await self.track_entity_state_changes(all_entity_ids)

        # Rebuild floor-based aggregators for updated areas
        self._build_floor_aggregators()

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
                format_area_names(self),
            )
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error(
                "Failed periodic save for areas: %s: %s",
                format_area_names(self),
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

        # Tick decay for all areas to update state (e.g., stop decay when factor reaches zero)
        # This must be done before refresh to ensure state transitions happen
        for area in self.areas.values():
            if area.config.decay.enabled:
                area.tick_decay()

        # Refresh the coordinator if decay is enabled for any area
        decay_enabled = any(area.config.decay.enabled for area in self.areas.values())
        if decay_enabled:
            await self.async_refresh()

        # Reschedule the timer
        self._start_decay_timer()

    # --- Analysis Timer Handling ---
    async def _start_analysis_timer(self) -> None:
        """Start the historical data import timer.

        Uses async_at_started to ensure the analysis timer doesn't fire during
        Home Assistant bootstrap, which could cause startup timeouts.

        Note: No staggering needed with single-instance architecture.
        """
        if self._analysis_timer is not None or not self.hass:
            return

        async def _schedule_first_analysis(_: HomeAssistant) -> None:
            """Schedule the first analysis after HA is fully started."""
            if self._analysis_timer is not None:
                return  # Already scheduled by another call

            # First analysis: 5 minutes after HA is fully started
            # This ensures we don't block bootstrap
            next_update = dt_util.utcnow() + timedelta(minutes=5)

            _LOGGER.info(
                "Home Assistant started - scheduling analysis timer for areas: %s",
                format_area_names(self),
            )

            self._analysis_timer = async_track_point_in_time(
                self.hass, self.run_analysis, next_update
            )

        # Use async_at_started to defer analysis until after bootstrap completes
        # This prevents the analysis from blocking HA startup
        async_at_started(self.hass, _schedule_first_analysis)

    async def run_analysis(self, _now: datetime | None = None) -> None:
        """Handle the historical data import timer.

        Always runs analysis for all areas. Prevents concurrent runs by checking
        and setting the _analysis_running flag.

        Args:
            _now: Optional timestamp for the analysis run (used by timer)
        """
        if _now is None:
            _now = dt_util.utcnow()

        # Prevent concurrent analysis runs
        if self._analysis_running:
            _LOGGER.debug("Analysis already running, skipping this trigger")
            # Cancel existing timer before rescheduling to avoid orphaned timers
            if self._analysis_timer is not None:
                self._analysis_timer()
                self._analysis_timer = None
            # Reschedule to try again later
            next_update = _now + timedelta(minutes=5)
            self._analysis_timer = async_track_point_in_time(
                self.hass, self.run_analysis, next_update
            )
            return

        # Clear timer reference now that we're actually running
        # (timer has fired and called this function)
        self._analysis_timer = None

        self._analysis_running = True
        _failed = False
        try:
            # Run the full analysis chain
            await run_full_analysis(self, _now)
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error("Failed to run historical analysis: %s", err)
            _failed = True
        except Exception:
            # Last-resort safety net — keeps the analysis timer alive
            _LOGGER.exception("Unexpected analysis error")
            _failed = True
        finally:
            self._analysis_running = False
            # Always reschedule — retry sooner on failure
            if _failed:
                next_update = _now + timedelta(minutes=15)
            else:
                next_update = _now + timedelta(
                    seconds=self.integration_config.analysis_interval
                )
            self._analysis_timer = async_track_point_in_time(
                self.hass, self.run_analysis, next_update
            )
