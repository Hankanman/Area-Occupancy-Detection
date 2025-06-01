"""Unified State Manager for Area Occupancy Detection.

This module provides centralized state management for the integration,
maintaining a reliable list of entities and handling all state-related operations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, TypedDict

from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.typing import EventType
from homeassistant.util import dt as dt_util

from .const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_DOOR_ACTIVE_STATE,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_WINDOW_ACTIVE_STATE,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DEBOUNCE_TIME,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_WINDOW_ACTIVE_STATE,
)
from .exceptions import StateError
from .probabilities import Probabilities
from .types import EntityType, SensorInfo

if TYPE_CHECKING:
    from .probabilities import Probabilities

_LOGGER = logging.getLogger(__name__)


class EntityValidationResult:
    """Result of entity validation."""

    def __init__(
        self,
        entity_id: str,
        is_valid: bool,
        exists: bool = False,
        is_available: bool = False,
        domain: str | None = None,
        error_message: str | None = None,
    ):
        """Initialize validation result."""
        self.entity_id = entity_id
        self.is_valid = is_valid
        self.exists = exists
        self.is_available = is_available
        self.domain = domain
        self.error_message = error_message


class StateSnapshot(TypedDict):
    """Type for state snapshot."""

    states: Dict[str, State]
    timestamp: datetime


class StateManager:
    """Centralized state manager for Area Occupancy Detection.

    This class provides a unified interface for all state management operations,
    including entity registration, validation, active state determination, and
    maintaining reliable state tracking for the integration.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        config: dict[str, Any] | None = None,
        probabilities: Probabilities | None = None,
    ):
        """Initialize the state manager.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary with entity lists and active states
            probabilities: Probabilities handler for accessing entity types and configs

        """
        self.hass = hass
        self.config = config or {}
        self.probabilities = probabilities

        # State storage
        self._current_states: dict[str, SensorInfo] = {}
        self._previous_states: dict[str, SensorInfo] = {}
        self._state_snapshots: dict[str, dict[str, SensorInfo]] = {}
        self._tracked_entities: set[str] = set()
        self._state_change_callbacks: list[Callable[[str, SensorInfo], None]] = []
        self._state_lock = asyncio.Lock()
        self._last_state_change = 0.0
        self._debounce_time = self.config.get("debounce_time", DEFAULT_DEBOUNCE_TIME)

        # Entity tracking
        self._entity_types: dict[str, EntityType] = {}
        self._entity_registry: dict[str, dict[str, Any]] = {}
        self._active_states_cache: dict[str, set[str]] = {}

        # Initialize if config is provided
        if config:
            self._populate_entity_registry()
            self._populate_caches()

    def _populate_entity_registry(self) -> None:
        """Populate the entity registry from configuration."""
        self._entity_types.clear()
        self._entity_registry.clear()

        # Map configuration keys to entity types
        type_mapping = {
            CONF_MOTION_SENSORS: EntityType.MOTION,
            CONF_PRIMARY_OCCUPANCY_SENSOR: EntityType.MOTION,
            # Add other mappings as needed
        }

        # Populate entity types
        for conf_key, entity_type in type_mapping.items():
            if conf_key in self.config:
                entities = self.config[conf_key]
                if isinstance(entities, list):
                    for entity_id in entities:
                        self._entity_types[entity_id] = entity_type
                else:
                    self._entity_types[entities] = entity_type

    def _populate_caches(self) -> None:
        """Populate internal caches for fast lookups."""
        self._active_states_cache.clear()
        for entity_type in EntityType:
            self.get_active_states_for_type(entity_type)

    # --- Entity Registry Management ---

    def get_all_entities(self) -> list[str]:
        """Get list of all configured entity IDs.

        Returns:
            List of all entity IDs managed by this integration

        """
        return list(self._entity_registry.keys())

    def get_entities_by_type(self, entity_type: EntityType) -> list[str]:
        """Get list of entity IDs for a specific type.

        Args:
            entity_type: The EntityType to filter by

        Returns:
            List of entity IDs of the specified type

        """
        return [
            entity_id
            for entity_id, etype in self._entity_types.items()
            if etype == entity_type
        ]

    def get_entity_type(self, entity_id: str) -> EntityType | None:
        """Get the entity type for a given entity ID.

        Args:
            entity_id: Entity ID to look up

        Returns:
            EntityType if found, None otherwise

        """
        return self._entity_types.get(entity_id)

    def get_entity_types_mapping(self) -> dict[str, EntityType]:
        """Get the complete entity type mapping.

        Returns:
            Dictionary mapping entity IDs to their types

        """
        return self._entity_types.copy()

    def add_entity(self, entity_id: str, entity_type: EntityType) -> None:
        """Add an entity to the registry.

        Args:
            entity_id: Entity ID to add
            entity_type: Type of the entity

        """
        self._entity_types[entity_id] = entity_type
        # Update caches for the new entity
        try:
            active_states = self.get_active_states_for_type(entity_type)
            self._active_states_cache[entity_id] = active_states
        except Exception as err:
            _LOGGER.warning(
                "Failed to cache active states for new entity %s: %s", entity_id, err
            )

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the registry.

        Args:
            entity_id: Entity ID to remove

        """
        self._entity_types.pop(entity_id, None)
        self._active_states_cache.pop(entity_id, None)

    def update_entity_configuration(self, new_config: dict[str, Any]) -> None:
        """Update entity configuration and refresh registry.

        Args:
            new_config: New configuration dictionary

        """
        self.config = new_config
        self._populate_entity_registry()
        self._populate_caches()

    # --- Entity Validation ---

    def validate_entity(self, entity_id: str) -> EntityValidationResult:
        """Validate a single entity.

        Args:
            entity_id: Entity ID to validate

        Returns:
            EntityValidationResult with validation details

        """
        # Validate entity exists in Home Assistant
        state_obj = self.hass.states.get(entity_id)
        exists = state_obj is not None

        if not exists:
            return EntityValidationResult(
                entity_id=entity_id,
                is_valid=False,
                exists=False,
                error_message=f"Entity {entity_id} does not exist in Home Assistant",
            )

        # Extract domain and check availability
        domain = entity_id.split(".")[0]
        is_available = state_obj.state not in [STATE_UNAVAILABLE, STATE_UNKNOWN]

        return EntityValidationResult(
            entity_id=entity_id,
            is_valid=exists and is_available,
            exists=exists,
            is_available=is_available,
            domain=domain,
        )

    def validate_all_entities(self) -> dict[str, EntityValidationResult]:
        """Validate all configured entities.

        Returns:
            Dictionary mapping entity IDs to their validation results

        """
        results = {}
        for entity_id in self._entity_types:
            results[entity_id] = self.validate_entity(entity_id)
        return results

    def get_invalid_entities(self) -> list[str]:
        """Get list of invalid entity IDs.

        Returns:
            List of entity IDs that are invalid or unavailable

        """
        invalid_entities = []
        for entity_id in self._entity_types:
            result = self.validate_entity(entity_id)
            if not result.is_valid:
                invalid_entities.append(entity_id)
        return invalid_entities

    def get_available_entities(self) -> list[str]:
        """Get list of available entity IDs.

        Returns:
            List of entity IDs that are valid and available

        """
        available_entities = []
        for entity_id in self._entity_types:
            result = self.validate_entity(entity_id)
            if result.is_valid:
                available_entities.append(entity_id)
        return available_entities

    # --- Entity State Management ---

    def get_entity_state(self, entity_id: str) -> SensorInfo | None:
        """Get current state information for an entity.

        Args:
            entity_id: Entity ID to get state for

        Returns:
            SensorInfo object with state information, or None if unavailable

        """
        # Check cache first
        if entity_id in self._active_states_cache:
            cached_info = self._active_states_cache[entity_id]
            # Check if cache is still fresh (within last 30 seconds)
            if cached_info:
                try:
                    last_changed = dt_util.parse_datetime(
                        cached_info.get("last_changed")
                    )
                    if (
                        last_changed
                        and (dt_util.utcnow() - last_changed).total_seconds() < 30
                    ):
                        return cached_info
                except (ValueError, TypeError):
                    pass

        # Get fresh state from Home Assistant
        state_obj = self.hass.states.get(entity_id)
        if not state_obj:
            return None

        is_available = state_obj.state not in [
            STATE_UNAVAILABLE,
            STATE_UNKNOWN,
            None,
            "",
        ]
        sensor_info = SensorInfo(
            state=state_obj.state if is_available else None,
            last_changed=state_obj.last_changed.isoformat()
            if state_obj.last_changed
            else dt_util.utcnow().isoformat(),
            availability=is_available,
        )

        # Cache the result
        self._active_states_cache[entity_id] = sensor_info
        return sensor_info

    def get_entity_states_batch(
        self, entity_ids: list[str] | None = None
    ) -> dict[str, SensorInfo]:
        """Get state information for multiple entities.

        Args:
            entity_ids: List of entity IDs, or None for all entities

        Returns:
            Dictionary mapping entity IDs to their SensorInfo

        """
        if entity_ids is None:
            entity_ids = list(self._entity_types.keys())

        results = {}
        for entity_id in entity_ids:
            state_info = self.get_entity_state(entity_id)
            if state_info:
                results[entity_id] = state_info

        return results

    def refresh_entity_states(self) -> None:
        """Refresh all entity states in cache."""
        self._active_states_cache.clear()
        for entity_id in self._entity_types:
            self.get_entity_state(entity_id)

    # --- Active State Management (Existing Functionality) ---

    @lru_cache(maxsize=128)
    def _get_default_active_states_for_type(self, entity_type: str) -> set[str]:
        """Get default active states for a given entity type (cached).

        Args:
            entity_type: The entity type (motion, media, etc.)

        Returns:
            Set of states considered active for this entity type

        """
        default_active_states_map = {
            EntityType.MOTION.value: {"on", "detected"},
            EntityType.MEDIA.value: {"playing", "paused"},
            EntityType.DOOR.value: {"open", "closed"},  # Configurable, both by default
            EntityType.WINDOW.value: {"open"},
            EntityType.LIGHT.value: {"on"},
            EntityType.APPLIANCE.value: {"on", "standby"},
            EntityType.ENVIRONMENTAL.value: {"on"},
            EntityType.WASP_IN_BOX.value: {"on", "detected"},
        }
        return default_active_states_map.get(entity_type, {"on"})

    def _get_configured_active_states_for_type(
        self, entity_type: EntityType
    ) -> set[str]:
        """Get configured active states for an entity type from user configuration.

        Args:
            entity_type: The EntityType enum value

        Returns:
            Set of configured active states, or default if not configured

        """
        if entity_type == EntityType.MEDIA:
            configured = self.config.get(
                CONF_MEDIA_ACTIVE_STATES, DEFAULT_MEDIA_ACTIVE_STATES
            )
            return set(configured) if isinstance(configured, list) else set()

        elif entity_type == EntityType.APPLIANCE:
            configured = self.config.get(
                CONF_APPLIANCE_ACTIVE_STATES, DEFAULT_APPLIANCE_ACTIVE_STATES
            )
            return set(configured) if isinstance(configured, list) else set()

        elif entity_type == EntityType.DOOR:
            configured = self.config.get(
                CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE
            )
            return {self._translate_binary_sensor_active_state(str(configured))}

        elif entity_type == EntityType.WINDOW:
            configured = self.config.get(
                CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE
            )
            return {self._translate_binary_sensor_active_state(str(configured))}

        else:
            return self._get_default_active_states_for_type(entity_type.value)

    def _translate_binary_sensor_active_state(self, configured_state: str) -> str:
        """Translate a configured display state to internal HA state.

        Args:
            configured_state: User-configured state string (e.g., 'Open')

        Returns:
            Internal HA state ('on' or 'off') corresponding to the configured state

        """
        configured_state_lower = configured_state.lower()

        on_state_display_values = {
            "open",
            "detected",
            "occupied",
            "home",
            "wet",
            "moving",
            "running",
            "connected",
            "charging",
            "plugged in",
            "on",
            "low",
            "cold",
            "hot",
            "unlocked",
            "problem",
            "unsafe",
            "update available",
        }

        return (
            STATE_ON if configured_state_lower in on_state_display_values else STATE_OFF
        )

    def get_active_states_for_entity(self, entity_id: str) -> set[str]:
        """Get the set of active states for a specific entity.

        Args:
            entity_id: Entity ID to get active states for

        Returns:
            Set of states considered active for this entity

        """
        if entity_id in self._active_states_cache:
            return self._active_states_cache[entity_id].copy()

        entity_type = self.get_entity_type(entity_id)
        if entity_type:
            active_states = self._get_configured_active_states_for_type(entity_type)
            # Cache for future lookups
            self._active_states_cache[entity_id] = active_states
            return active_states.copy()

        # Fallback
        return {"on"}

    def get_active_states_for_type(self, entity_type: EntityType | str) -> set[str]:
        """Get the set of active states for an entity type.

        Args:
            entity_type: EntityType enum or string value

        Returns:
            Set of states considered active for this entity type

        """
        if isinstance(entity_type, str):
            try:
                entity_type = EntityType(entity_type)
            except ValueError:
                return self._get_default_active_states_for_type(entity_type)

        return self._get_configured_active_states_for_type(entity_type)

    # --- Statistics and Analysis ---

    def get_entity_statistics(self) -> dict[str, Any]:
        """Get statistics about managed entities.

        Returns:
            Dictionary with entity statistics

        """
        total_entities = len(self._entity_types)
        entities_by_type = {}
        available_count = 0
        active_count = 0

        for entity_type in EntityType:
            entities_by_type[entity_type.value] = len(
                self.get_entities_by_type(entity_type)
            )

        # Check availability and active status
        for entity_id in self._entity_types:
            validation = self.validate_entity(entity_id)
            if validation.is_valid:
                available_count += 1
                if self.is_entity_active(entity_id):
                    active_count += 1

        return {
            "total_entities": total_entities,
            "available_entities": available_count,
            "active_entities": active_count,
            "entities_by_type": entities_by_type,
            "availability_rate": available_count / total_entities
            if total_entities > 0
            else 0,
            "activity_rate": active_count / available_count
            if available_count > 0
            else 0,
        }

    def get_entity_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of all entities.

        Returns:
            Dictionary with detailed entity information

        """
        summary = {
            "statistics": self.get_entity_statistics(),
            "entities": {},
            "validation_results": {},
            "issues": [],
        }

        for entity_id in self._entity_types:
            entity_type = self.get_entity_type(entity_id)
            validation = self.validate_entity(entity_id)
            state_info = self.get_entity_state(entity_id)

            summary["entities"][entity_id] = {
                "type": entity_type.value if entity_type else "unknown",
                "validation": validation,
                "state_info": state_info,
                "active_states": list(self.get_active_states_for_entity(entity_id)),
                "is_active": self.is_entity_active(entity_id)
                if validation.is_valid
                else False,
            }

            if not validation.is_valid:
                summary["issues"].append(
                    {
                        "entity_id": entity_id,
                        "issue": validation.error_message or "Entity validation failed",
                        "type": "validation_error",
                    }
                )

        return summary

    # --- Cleanup and Maintenance ---

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self._active_states_cache.clear()
        self._entity_registry.clear()
        self._entity_types.clear()

    def refresh_all(self) -> None:
        """Refresh all entity data and caches."""
        self.clear_caches()
        self._populate_entity_registry()
        self._populate_caches()
        self.refresh_entity_states()

    def update_probabilities_handler(self, probabilities: Probabilities) -> None:
        """Update the probabilities handler and refresh caches.

        Args:
            probabilities: New probabilities handler

        """
        self.probabilities = probabilities
        self.clear_caches()
        self._populate_caches()

    # --- State Tracking Methods (moved from coordinator) ---

    async def async_initialize_states(self, sensor_ids: list[str]) -> None:
        """Initialize sensor states for tracking.

        Args:
            sensor_ids: List of entity IDs to initialize and track

        """
        try:
            # Get current states for all sensors
            current_states = self.get_entity_states_batch(sensor_ids)

            # Update current states
            async with self._state_lock:
                self._current_states = current_states
                self._previous_states = {}
                self._tracked_entities = set(sensor_ids)

            _LOGGER.debug("Initialized states for %d sensors", len(sensor_ids))

        except Exception as err:
            _LOGGER.error("Error initializing states: %s", err)
            raise StateError(f"Failed to initialize states: {err}") from err

    def setup_state_tracking(
        self,
        sensor_ids: list[str],
        callback: Callable[[str, SensorInfo], None] | None = None,
    ) -> None:
        """Set up state change tracking for specified entities.

        Args:
            sensor_ids: List of entity IDs to track
            callback: Optional callback function to call when states change

        """
        if callback:
            self._state_change_callbacks.append(callback)

        # Track state changes for all sensors
        self.hass.async_create_task(self._setup_state_tracking_internal(sensor_ids))

    async def _setup_state_tracking_internal(self, sensor_ids: list[str]) -> None:
        """Internal method to set up state tracking."""
        try:
            # Remove any existing listeners
            self.stop_state_tracking()

            # Add new listeners
            for entity_id in sensor_ids:
                self.hass.async_create_task(
                    async_track_state_change_event(
                        self.hass,
                        [entity_id],
                        self._async_state_changed_listener,
                    )
                )

            _LOGGER.debug("Set up state tracking for %d sensors", len(sensor_ids))

        except Exception as err:
            _LOGGER.error("Error setting up state tracking: %s", err)
            raise StateError(f"Failed to set up state tracking: {err}") from err

    @callback
    def _async_state_changed_listener(self, event: EventType) -> None:
        """Handle state changes."""
        try:
            # Get entity ID and new state
            entity_id = event.data["entity_id"]
            new_state = event.data["new_state"]

            # Create sensor info
            sensor_info = self._create_sensor_info(new_state)

            # Update state with debouncing
            current_time = time.time()
            if current_time - self._last_state_change < self._debounce_time:
                return

            self._last_state_change = current_time

            # Update state under lock
            self.hass.async_create_task(
                self._update_state_internal(entity_id, sensor_info)
            )

        except Exception as err:
            _LOGGER.error("Error handling state change: %s", err)

    async def _update_state_internal(
        self, entity_id: str, sensor_info: SensorInfo
    ) -> None:
        """Update state under lock."""
        async with self._state_lock:
            # Store previous state
            self._previous_states[entity_id] = self._current_states.get(entity_id, {})

            # Update current state
            self._current_states[entity_id] = sensor_info

            # Create snapshot
            self._state_snapshots[entity_id] = {
                "current": dict(self._current_states),
                "previous": dict(self._previous_states),
            }

            # Notify callbacks
            for callback in self._state_change_callbacks:
                try:
                    callback(entity_id, sensor_info)
                except Exception as err:
                    _LOGGER.error("Error in state change callback: %s", err)

    def _create_sensor_info(self, state: State | None) -> SensorInfo:
        """Create sensor info from state."""
        if not state:
            return {
                "state": None,
                "last_changed": datetime.now().isoformat(),
                "availability": False,
            }

        return {
            "state": state.state,
            "last_changed": state.last_changed.isoformat(),
            "availability": state.state != "unavailable",
        }

    def get_current_states(self) -> dict[str, SensorInfo]:
        """Get current states for all tracked entities."""
        return dict(self._current_states)

    def get_previous_states(self) -> dict[str, SensorInfo]:
        """Get previous states for all tracked entities."""
        return dict(self._previous_states)

    def get_state_snapshot(
        self, entity_id: str
    ) -> dict[str, dict[str, SensorInfo]] | None:
        """Get state snapshot for an entity."""
        return self._state_snapshots.get(entity_id)

    def get_tracked_entities(self) -> set[str]:
        """Get set of tracked entity IDs."""
        return set(self._tracked_entities)

    def stop_state_tracking(self) -> None:
        """Stop state tracking and cleanup resources."""
        self._state_change_callbacks.clear()
        self._tracked_entities.clear()
        self._current_states.clear()
        self._previous_states.clear()
        self._state_snapshots.clear()

    def is_entity_active(self, entity_id: str, state: str | None = None) -> bool:
        """Check if an entity is in an active state.

        Args:
            entity_id: The entity ID to check
            state: The current state of the entity (if None, will fetch current state)

        Returns:
            True if the entity is in an active state, False otherwise

        """
        if not self._tracked_entities or entity_id not in self._tracked_entities:
            return False

        entity_type = self.get_entity_type(entity_id)
        if not entity_type:
            return False

        if state is None:
            current_state = self.get_entity_state(entity_id)
            if not current_state:
                return False
            state = current_state.get("state")

        if state is None:
            return False

        active_states = self.get_active_states_for_entity(entity_id)
        return state in active_states


# Convenience function for backward compatibility
def create_state_manager(
    hass: HomeAssistant,
    config: dict[str, Any] | None = None,
    probabilities: Probabilities | None = None,
) -> StateManager:
    """Create a StateManager instance.

    Args:
        hass: Home Assistant instance
        config: Configuration dictionary
        probabilities: Probabilities handler

    Returns:
        StateManager instance

    """
    return StateManager(hass, config, probabilities)
