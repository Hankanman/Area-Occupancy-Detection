"""Area Occupancy Coordinator."""

from __future__ import annotations

from datetime import datetime, timedelta

# Standard Library
import logging
from typing import Any

# Third Party
import sqlalchemy as sa

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
from .data.config import ConfigManager
from .data.entity import EntityManager
from .data.entity_type import EntityTypeManager
from .data.prior import Prior
from .data.purpose import PurposeManager
from .sqlite_storage import AreaOccupancyStorage

# from .storage import AreaOccupancyStore  # Replaced with SQLite storage
from .utils import conditional_sorted_probability

_LOGGER = logging.getLogger(__name__)

# Global timer intervals in seconds
DECAY_INTERVAL = 10
PRIOR_INTERVAL = 3600
HISTORICAL_INTERVAL = 86400  # 24 hours in seconds


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
        self.config_manager = ConfigManager(self)
        self.config = self.config_manager.config
        self.prior = Prior(self)
        self.sqlite_store = AreaOccupancyStorage(
            hass=self.hass, entry_id=self.entry_id, coordinator=self
        )
        self.entity_types = EntityTypeManager(self)
        self.purpose = PurposeManager(self)
        self.entities = EntityManager(self)
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None
        self._global_prior_timer: CALLBACK_TYPE | None = None
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._remove_state_listener: CALLBACK_TYPE | None = None
        self._historical_timer: CALLBACK_TYPE | None = None

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

        return conditional_sorted_probability(
            entities=self.entities.entities, prior=self.area_prior
        )

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
    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0)."""
        return self.config.threshold if self.config else 0.5

    @property
    def binary_sensor_entity_ids(self) -> dict[str, str | None]:
        """Return the entity_ids of the binary sensors created by this integration.

        Returns:
            dict[str, str | None]: Dictionary with 'occupancy' and 'wasp' keys
                                 containing their respective entity_ids or None if not set.

        """
        return {"occupancy": self.occupancy_entity_id, "wasp": self.wasp_entity_id}

    # --- Public Methods ---
    async def setup(self) -> None:
        """Initialize the coordinator and its components."""
        try:
            _LOGGER.debug("Starting coordinator setup for %s", self.config.name)

            # Initialize purpose manager
            await self.purpose.async_initialize()

            # Build Default Entity Types
            await self.entity_types.async_initialize()

            # Load stored data
            await self.sqlite_store.async_initialize()

            # Check if this is a new database and populate immediately if needed
            await self._check_and_populate_new_database()

            loaded_data = await self.sqlite_store.async_load_data()

            if loaded_data:
                # Create entity manager from loaded data
                self.entities = EntityManager.from_dict(dict(loaded_data), self)
                # Update restored entities with current configuration
                await self.entities.__post_init__()

                _LOGGER.debug(
                    "Successfully restored stored data for instance %s", self.entry_id
                )
            else:
                # No stored data - initialize entities from configuration
                self.entities = EntityManager(self)
                await self.entities.__post_init__()

                _LOGGER.debug(
                    "No stored data found, created fresh entities from configuration for %s",
                    self.entry_id,
                )

            # Calculate initial area baseline prior (this is fast and needed immediately)
            if self.config.history.enabled:
                await self.prior.update()

                # Defer time-based prior calculation to background task to avoid blocking startup
                self.hass.async_create_task(
                    self._calculate_time_priors_async(initial_setup=True)
                )

                # Defer likelihood updates to background task to avoid blocking startup
                self.hass.async_create_task(
                    self._update_likelihoods_async(initial_setup=True)
                )

            # Save current state to storage
            await self.sqlite_store.async_save_data(force=True)

            # Track entity state changes
            await self.track_entity_state_changes(self.entities.entity_ids)

            # Start the prior update timer
            self._start_prior_timer()

            # Start the global decay timer
            self._start_decay_timer()

            # Start the historical data import timer
            self._start_historical_timer()

            _LOGGER.debug(
                "Successfully set up AreaOccupancyCoordinator for %s with %d entities",
                self.config.name,
                len(self.entities.entities),
            )

        except HomeAssistantError as err:
            _LOGGER.error("Failed to set up coordinator: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up coordinator: {err}") from err

    async def update(self) -> dict[str, Any]:
        """Update and return the current coordinator data."""
        # Save current state to storage
        await self.sqlite_store.async_save_data()

        # Return current state data
        return {
            "probability": self.probability,
            "occupied": self.occupied,
            "threshold": self.threshold,
            "prior": self.prior,
            "decay": self.decay,
            "last_updated": dt_util.utcnow(),
        }

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        # Cancel prior update tracker
        if self._global_prior_timer is not None:
            self._global_prior_timer()
            self._global_prior_timer = None

        # Stop global decay timer
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

        # Clean up state change listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Clean up historical timer
        if self._historical_timer is not None:
            self._historical_timer()
            self._historical_timer = None

        await self.sqlite_store.async_save_data(force=True)

        # Clean up entity manager
        await self.entities.cleanup()

        # Clean up purpose manager
        self.purpose.cleanup()

        await super().async_shutdown()

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options."""
        # Update config
        await self.config_manager.update_config(options)
        self.config = self.config_manager.config

        # Update purpose with new configuration
        await self.purpose.async_initialize()

        # Always re-initialize entities and entity types when configuration changes
        _LOGGER.info(
            "Configuration updated, re-initializing entities for %s", self.config.name
        )

        # Update entity types with new configuration
        self.entity_types.cleanup()
        await self.entity_types.async_initialize()

        # Clean up existing entity tracking and re-initialize
        await self.entities.cleanup()

        # Re-establish entity state tracking with new entity list
        await self.track_entity_state_changes(self.entities.entity_ids)

        # Force immediate save after configuration changes
        await self.sqlite_store.async_save_data(force=True)

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

            async def _refresh_on_state_change(event):
                entity_id = event.data.get("entity_id")
                entity = self.entities.get_entity(entity_id)
                if entity and entity.has_new_evidence():
                    await self.async_refresh()

            self._remove_state_listener = async_track_state_change_event(
                self.hass, entity_ids, _refresh_on_state_change
            )

    # --- Prior Timer Handling ---
    def _start_prior_timer(self) -> None:
        """Start the prior update timer."""
        if self._global_prior_timer is not None or not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(seconds=PRIOR_INTERVAL)

        self._global_prior_timer = async_track_point_in_time(
            self.hass, self._handle_prior_timer, next_update
        )

    async def _handle_prior_timer(self, _now: datetime) -> None:
        """Handle the prior update timer."""
        self._global_prior_timer = None

        history_period = self.config.history.period

        # Update learned priors if history is enabled
        if self.config.history.enabled:
            # Update area baseline prior separately (unweighted)
            await self.prior.update()

            # Calculate time-based priors (this is more intensive, so do it less frequently)
            # Use configurable frequency (default: every 4th run = 4 hours)
            if not hasattr(self, "_prior_timer_count"):
                self._prior_timer_count = 0
            self._prior_timer_count += 1

            if (
                self._prior_timer_count
                % self.config.history.time_based_priors_frequency
                == 0
            ):
                # Use the async method to avoid blocking
                self.hass.async_create_task(
                    self._calculate_time_priors_async(initial_setup=False)
                )

            # Update individual sensor likelihoods
            if self.config.history.likelihood_updates_enabled:
                # Check if we should update likelihoods based on frequency
                if not hasattr(self, "_likelihood_timer_count"):
                    self._likelihood_timer_count = 0
                self._likelihood_timer_count += 1

                if (
                    self._likelihood_timer_count
                    % self.config.history.likelihood_updates_frequency
                    == 0
                ):
                    # Use background task to avoid blocking
                    self.hass.async_create_task(
                        self._update_likelihoods_async(
                            history_period, initial_setup=False
                        )
                    )

        # Reschedule the timer
        self._start_prior_timer()

        # Save current state to storage
        await self.sqlite_store.async_save_data(force=True)

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
    def _start_historical_timer(self) -> None:
        """Start the historical data import timer."""
        if self._historical_timer is not None or not self.hass:
            return

        # Run first import shortly after startup (5 minutes)
        next_update = dt_util.utcnow() + timedelta(minutes=5)

        self._historical_timer = async_track_point_in_time(
            self.hass, self._handle_historical_timer, next_update
        )

    async def _handle_historical_timer(self, _now: datetime) -> None:
        """Handle the historical data import timer."""
        self._historical_timer = None

        try:
            # Import recent data from recorder
            entity_ids = list(self.entities.entities.keys())
            if entity_ids:
                import_counts = await self.sqlite_store.import_intervals_from_recorder(
                    entity_ids, days=10
                )
                total_imported = sum(import_counts.values())

                if total_imported > 0:
                    _LOGGER.info(
                        "Historical import: %d intervals imported", total_imported
                    )

                    # Recalculate priors with new data
                    await self.prior.update(force=True)
                    # Use background task for likelihood updates
                    self.hass.async_create_task(
                        self._update_likelihoods_async(initial_setup=False)
                    )

                # Cleanup old data (yearly retention)
                await self.sqlite_store.cleanup_old_intervals(retention_days=365)

        except (sa.exc.SQLAlchemyError, OSError) as err:
            _LOGGER.error("Historical data import failed: %s", err)

        # Schedule next run (24 hours)
        next_update = dt_util.utcnow() + timedelta(seconds=HISTORICAL_INTERVAL)
        self._historical_timer = async_track_point_in_time(
            self.hass, self._handle_historical_timer, next_update
        )

    async def _check_and_populate_new_database(self) -> None:
        """Check if the state_intervals table is empty and populate it if needed."""
        if await self.sqlite_store.is_state_intervals_empty():
            _LOGGER.info(
                "State intervals table is empty for instance %s. Populating with initial data from recorder.",
                self.entry_id,
            )

            # Get entity IDs from configuration since entities haven't been created yet
            entity_ids = self.config.entity_ids

            # Remove duplicates and empty strings
            entity_ids = [eid for eid in set(entity_ids) if eid]

            if entity_ids:
                _LOGGER.info(
                    "Importing initial data for %d entities from recorder (last 10 days)",
                    len(entity_ids),
                )

                import_counts = await self.sqlite_store.import_intervals_from_recorder(
                    entity_ids, days=10
                )
                total_imported = sum(import_counts.values())

                _LOGGER.info("Import results by entity: %s", import_counts)
                _LOGGER.info(
                    "Populated state intervals table for instance %s with %d intervals.",
                    self.entry_id,
                    total_imported,
                )
            else:
                _LOGGER.warning(
                    "No entity IDs found in configuration to populate state intervals table for instance %s.",
                    self.entry_id,
                )

    async def _calculate_time_priors_async(self, initial_setup: bool = False) -> None:
        """Calculate time-based priors asynchronously without blocking startup.

        Args:
            initial_setup: If True, this is the initial setup calculation

        """
        if (
            not self.config.history.enabled
            or not self.config.history.time_based_priors_enabled
        ):
            return

        try:
            # Check if we already have recent time-based priors
            if not initial_setup:
                try:
                    current_prior = await self.prior.get_time_prior()
                    if current_prior > 0:
                        return
                except HomeAssistantError:
                    pass  # Continue with calculation if check fails

            # Calculate time-based priors in background
            await self.prior.calculate_time_based_priors()

        except HomeAssistantError as err:
            _LOGGER.warning(
                "Failed to calculate time-based priors for entry %s: %s",
                self.entry_id,
                err,
            )

    async def _update_likelihoods_async(
        self, history_period: int | None = None, initial_setup: bool = False
    ) -> None:
        """Update entity likelihoods asynchronously without blocking.

        Args:
            history_period: Period in days for historical data
            initial_setup: If True, this is the initial setup calculation

        """
        if (
            not self.config.history.enabled
            or not self.config.history.likelihood_updates_enabled
        ):
            return

        try:
            await self.entities.update_all_entity_likelihoods(
                history_period=history_period
            )

        except HomeAssistantError as err:
            _LOGGER.warning(
                "Failed to update likelihoods for entry %s: %s",
                self.entry_id,
                err,
            )
