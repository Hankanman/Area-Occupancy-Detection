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
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo

# Dispatcher imports removed - no longer needed without master election
from homeassistant.helpers.event import (
    async_call_later,
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local imports
from .area import AllAreas, Area
from .const import (
    ALL_AREAS_IDENTIFIER,
    CONF_AREAS,
    CONF_NAME,
    DEFAULT_NAME,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
    validate_and_sanitize_area_name,
)
from .data.integration_config import IntegrationConfig
from .db import AreaOccupancyDB

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

        # Integration-level configuration (global settings for entire integration)
        self.integration_config = IntegrationConfig(self, config_entry)

        # Multi-area architecture: dict[str, Area] keyed by area name
        self.areas: dict[str, Area] = {}

        # All Areas aggregator (lazy initialization)
        self._all_areas: AllAreas | None = None

        # Per-area state listeners (area_name -> callback)
        self._area_state_listeners: dict[str, CALLBACK_TYPE] = {}
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._analysis_timer: CALLBACK_TYPE | None = None
        self._save_timer: CALLBACK_TYPE | None = None
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

    def _load_areas_from_config(self) -> None:
        """Load areas from config entry.

        Supports both legacy (single area) and new (multi-area) formats.
        For legacy format, creates a single area from the config_entry data.
        For new format, loads areas from CONF_AREAS list.
        """
        merged = dict(self.config_entry.data)
        merged.update(self.config_entry.options)

        # Check if we have the new multi-area format
        if CONF_AREAS in merged and isinstance(merged[CONF_AREAS], list):
            # New multi-area format
            areas_list = merged[CONF_AREAS]
            for area_data in areas_list:
                area_name = area_data.get(CONF_NAME)
                if not area_name:
                    _LOGGER.warning("Skipping area without name: %s", area_data)
                    continue

                # Validate and sanitize area name
                try:
                    area_name = validate_and_sanitize_area_name(area_name)
                    # Update area_data with sanitized name
                    area_data[CONF_NAME] = area_name
                except ValueError as err:
                    _LOGGER.warning(
                        "Invalid area name '%s': %s. Skipping area.", area_name, err
                    )
                    continue

                if area_name in self.areas:
                    _LOGGER.warning("Duplicate area name %s, skipping", area_name)
                    continue

                # Create Area for this area
                self.areas[area_name] = Area(
                    coordinator=self,
                    area_name=area_name,
                    area_data=area_data,
                )
                _LOGGER.debug("Loaded area: %s", area_name)
        else:
            # Legacy single-area format - create area from entry data
            area_name = merged.get(CONF_NAME, DEFAULT_NAME)

            # Validate and sanitize area name
            try:
                area_name = validate_and_sanitize_area_name(area_name)
            except ValueError as err:
                _LOGGER.warning(
                    "Invalid area name '%s': %s. Using default name.", area_name, err
                )
                area_name = DEFAULT_NAME

            _LOGGER.info(
                "Legacy config format detected, creating single area: %s", area_name
            )

            # Create area for migration path
            self.areas[area_name] = Area(
                coordinator=self,
                area_name=area_name,
                area_data=merged,
            )
            _LOGGER.debug("Created area from legacy config: %s", area_name)

    def get_area_or_default(self, area_name: str | None = None) -> Area | None:
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

    def get_all_areas(self) -> AllAreas:
        """Get or create the AllAreas aggregator instance.

        Returns:
            AllAreas instance for aggregating data across all areas
        """
        if self._all_areas is None:
            self._all_areas = AllAreas(self)
        return self._all_areas

    # --- Area Management ---

    # Master election code removed - no longer needed with single-instance architecture

    def device_info(self, area_name: str | None = None) -> DeviceInfo:
        """Return device info for a specific area.

        Args:
            area_name: Area name, None returns first area (for backward compatibility)

        Returns:
            DeviceInfo for the specified area
        """
        # Handle "All Areas" aggregation
        if area_name == ALL_AREAS_IDENTIFIER:
            return self.get_all_areas().device_info()

        area = self.get_area_or_default(area_name)
        if area is None:
            return DeviceInfo(
                identifiers={(DOMAIN, self.entry_id)},
                name=DEFAULT_NAME,
                manufacturer=DEVICE_MANUFACTURER,
                model=DEVICE_MODEL,
                sw_version=DEVICE_SW_VERSION,
            )

        return area.device_info()

    def probability(self, area_name: str | None = None) -> float:
        """Calculate and return the current occupancy probability (0.0-1.0) for an area.

        Args:
            area_name: Area name, None returns first area (for backward compatibility)

        Returns:
            Probability value (0.0-1.0)
        """
        # Handle "All Areas" aggregation
        if area_name == ALL_AREAS_IDENTIFIER:
            return self.get_all_areas().probability()

        area = self.get_area_or_default(area_name)
        if area is None:
            return MIN_PROBABILITY

        return area.probability()

    def type_probabilities(self, area_name: str | None = None) -> dict[str, float]:
        """Calculate and return the current occupancy probabilities for each entity type (0.0-1.0).

        Args:
            area_name: Area name, None returns first area (for backward compatibility)

        Returns:
            Dictionary mapping input types to probabilities

        Raises:
            ValueError: If area_name is ALL_AREAS_IDENTIFIER (not supported for "All Areas")
        """
        # "All Areas" does not support type_probabilities aggregation
        if area_name == ALL_AREAS_IDENTIFIER:
            raise ValueError(
                "type_probabilities is not supported for 'All Areas' aggregation"
            )

        area = self.get_area_or_default(area_name)
        if area is None:
            return {}

        return area.type_probabilities()

    def area_prior(self, area_name: str | None = None) -> float:
        """Get the area's baseline occupancy prior from historical data.

        This returns the pure P(area occupied) without any sensor weighting.

        Args:
            area_name: Area name, None returns first area (for backward compatibility)

        Returns:
            Prior probability (0.0-1.0)
        """
        # Handle "All Areas" aggregation
        if area_name == ALL_AREAS_IDENTIFIER:
            return self.get_all_areas().area_prior()

        area = self.get_area_or_default(area_name)
        if area is None:
            return MIN_PROBABILITY
        return area.area_prior()

    def decay(self, area_name: str | None = None) -> float:
        """Calculate the current decay probability (0.0-1.0) for an area.

        Args:
            area_name: Area name, None returns first area (for backward compatibility)

        Returns:
            Decay probability (0.0-1.0)
        """
        # Handle "All Areas" aggregation
        if area_name == ALL_AREAS_IDENTIFIER:
            return self.get_all_areas().decay()

        area = self.get_area_or_default(area_name)
        if area is None:
            return 1.0

        return area.decay()

    def occupied(self, area_name: str | None = None) -> bool:
        """Return the current occupancy state (True/False) for an area.

        Args:
            area_name: Area name, None returns first area (for backward compatibility)

        Returns:
            True if occupied, False otherwise
        """
        # Handle "All Areas" aggregation
        if area_name == ALL_AREAS_IDENTIFIER:
            return self.get_all_areas().occupied()

        area = self.get_area_or_default(area_name)
        if area is None:
            return False
        return area.occupied()

    @property
    def setup_complete(self) -> bool:
        """Return whether setup is complete."""
        return self._setup_complete

    def threshold(self, area_name: str | None = None) -> float:
        """Return the current occupancy threshold (0.0-1.0) for an area.

        Args:
            area_name: Area name, None returns first area (for backward compatibility)

        Returns:
            Threshold value (0.0-1.0)
        """
        area = self.get_area_or_default(area_name)
        if area is None:
            return 0.5
        return area.threshold()

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
            area_names = ", ".join(self.get_area_names()) if self.areas else "no areas"
            _LOGGER.warning("Decay timer not started for areas: %s", area_names)
            return False

        # Check if analysis timer started (or is scheduled)
        if self._analysis_timer is None:
            area_names = ", ".join(self.get_area_names()) if self.areas else "no areas"
            _LOGGER.warning("Analysis timer not started for areas: %s", area_names)
            return False

        return True

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

            # Load data from database (legacy method - will need area-specific loading later)
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
            # Analysis timer is async and runs in background
            await self._start_analysis_timer()

            # Verify critical initialization succeeded before marking complete
            if not self._verify_setup_complete():
                area_names = ", ".join(self.areas.keys())
                _LOGGER.error(
                    "Critical initialization failed for areas: %s", area_names
                )
                error_msg = "Failed to start critical timers"
                raise HomeAssistantError(error_msg)  # noqa: TRY301

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
                # Analysis timer is async and runs in background
                await self._start_analysis_timer()

                # Verify critical initialization succeeded
                area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
                if not self._verify_setup_complete():
                    _LOGGER.error(
                        "Failed to start critical timers for areas: %s after retry",
                        area_names,
                    )
                    # Still set complete to allow partial functionality
                    # but log the issue for debugging
                    self._setup_complete = True
                else:
                    # Only mark complete if verification passed
                    self._setup_complete = True

            except (HomeAssistantError, OSError, RuntimeError) as timer_err:
                area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
                _LOGGER.error(
                    "Failed to start basic timers for areas: %s: %s",
                    area_names,
                    timer_err,
                )
                # Don't set _setup_complete if timers completely failed

    async def update(self) -> dict[str, Any]:
        """Update and return the current coordinator data (in-memory only).

        Database saves are debounced to avoid blocking on every state change.
        See _schedule_save() for the actual save logic.

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
        if self.areas:
            first_area = next(iter(self.areas.keys()))
            result.update(result[first_area])

        result["last_updated"] = dt_util.utcnow()
        return result

    def _schedule_save(self) -> None:
        """Schedule a debounced database save.

        Cancels any pending save and schedules a new one. This ensures that
        rapid state changes result in a single database write after the
        activity settles.

        Note: No master check needed - single instance always saves.
        """
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
                area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
                _LOGGER.debug(
                    "Debounced database save completed for areas: %s", area_names
                )
            except (HomeAssistantError, OSError, RuntimeError) as err:
                area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
                _LOGGER.error("Failed to save data for areas: %s: %s", area_names, err)

        self._save_timer = async_call_later(
            self.hass, self.integration_config.save_debounce, _do_save
        )
        area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
        _LOGGER.debug(
            "Database save scheduled in %d seconds for areas: %s",
            self.integration_config.save_debounce,
            area_names,
        )

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # Cancel pending save timer before cleanup and perform final save
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Perform final save to ensure no data loss
        try:
            await self.hass.async_add_executor_job(self.db.save_data)
            area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
            _LOGGER.info("Final database save completed for areas: %s", area_names)
        except (HomeAssistantError, OSError, RuntimeError) as err:
            area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
            _LOGGER.error("Failed final save for areas: %s: %s", area_names, err)

        # Cancel all area state listeners
        for listener in self._area_state_listeners.values():
            listener()
        self._area_state_listeners.clear()

        # Cancel prior update tracker
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

        # Clean up historical timer
        if self._analysis_timer is not None:
            self._analysis_timer()
            self._analysis_timer = None

        # Clean up save timer (in case it wasn't handled in earlier logic)
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Clean up all areas
        for area in self.areas.values():
            await area.async_cleanup()

        await super().async_shutdown()

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options.

        Args:
            options: Updated options dict (may contain CONF_AREAS for multi-area updates)
        """
        if self.config_entry is None:
            raise HomeAssistantError("Cannot update options: config_entry is None")

        # Identify areas that will be removed by checking what's in new config
        merged = dict(self.config_entry.data)
        merged.update(self.config_entry.options)

        # Get list of area names that will exist after reload
        new_area_names = set()
        if CONF_AREAS in merged and isinstance(merged[CONF_AREAS], list):
            new_area_names = {
                area_data.get(CONF_NAME)
                for area_data in merged[CONF_AREAS]
                if area_data.get(CONF_NAME)
            }
        else:
            # Legacy format - single area
            area_name = merged.get(CONF_NAME, DEFAULT_NAME)
            if area_name:
                new_area_names = {area_name}

        # Clean up areas that will be removed
        removed_area_names = set(self.areas.keys()) - new_area_names
        if removed_area_names:
            _LOGGER.info("Cleaning up removed areas: %s", ", ".join(removed_area_names))

            # Get entity registry for cleanup
            entity_registry = er.async_get(self.hass)

            for area_name in removed_area_names:
                area = self.areas.get(area_name)
                if area:
                    # Clean up removed area
                    await area.async_cleanup()

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

                    _LOGGER.debug("Cleaned up area: %s", area_name)

        # Clear existing areas before reloading (this ensures removed areas are gone
        # and prevents _load_areas_from_config from skipping areas that should be updated)
        self.areas.clear()

        # Cancel existing entity state listeners (will be recreated with new entity lists)
        for listener in self._area_state_listeners.values():
            listener()
        self._area_state_listeners.clear()

        # Reload areas from updated config
        self._load_areas_from_config()

        # Update each area's configuration
        for area_name, area in self.areas.items():
            _LOGGER.info(
                "Configuration updated, re-initializing entities for area: %s",
                area_name,
            )

            # Clean up existing entity tracking
            await area.entities.cleanup()

            # Area components are now initialized synchronously in __init__

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
                    entity = area.entities.get_entity(entity_id)
                    if entity and entity.has_new_evidence():
                        affected_areas.append(area_name)

                # If entity affects any area and setup is complete, refresh
                if affected_areas and self.setup_complete:
                    await self.async_refresh()
                    # Schedule debounced save
                    self._schedule_save()

            # Create single listener for all entities (more efficient than per-area listeners)
            listener = async_track_state_change_event(
                self.hass, entity_ids, _refresh_on_state_change
            )
            # Store listener (using a single key since we have one listener for all)
            self._area_state_listeners["_all"] = listener

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
            self._schedule_save()  # Schedule save after decay update

        # Reschedule the timer
        self._start_decay_timer()

    # --- Historical Timer Handling ---
    async def _start_analysis_timer(self) -> None:
        """Start the historical data import timer.

        Note: No staggering needed with single-instance architecture.
        """
        if self._analysis_timer is not None or not self.hass:
            return

        # First analysis: 5 minutes after startup
        # Subsequent analyses: 1 hour interval
        next_update = dt_util.utcnow() + timedelta(minutes=5)

        area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
        _LOGGER.info(
            "Starting analysis timer for areas: %s",
            area_names,
        )

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

            # Prune old intervals and run health check (no master check needed)
            health_ok = await self.hass.async_add_executor_job(
                self.db.periodic_health_check
            )
            if not health_ok:
                area_names = ", ".join(self.areas.keys()) if self.areas else "no areas"
                _LOGGER.warning(
                    "Database health check found issues for areas: %s", area_names
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
