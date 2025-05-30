"""Unified State Manager for Area Occupancy Detection.

This module provides centralized state management for the integration,
maintaining a reliable list of entities and handling all state-related operations.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util

from .const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_WINDOW_ACTIVE_STATE,
)
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

        # Entity tracking
        self._entity_registry: dict[str, EntityType] = {}
        self._entity_states_cache: dict[str, SensorInfo] = {}
        self._active_states_cache: dict[str, set[str]] = {}

        # State tracking (new functionality from coordinator)
        self._current_states: dict[str, SensorInfo] = {}
        self._previous_states: dict[str, SensorInfo] = {}
        self._tracked_entities: set[str] = set()
        self._remove_state_listener: CALLBACK_TYPE | None = None
        self._state_change_callback: Callable[[str, SensorInfo], None] | None = None
        self._last_callback_time: dict[
            str, float
        ] = {}  # Track last callback time per entity

        # Initialize entity registry
        self._populate_entity_registry()
        self._populate_caches()

    def _populate_entity_registry(self) -> None:
        """Populate the entity registry from configuration."""
        self._entity_registry.clear()

        # Define entity type mappings from configuration
        entity_type_mappings = [
            (self.config.get(CONF_MOTION_SENSORS, []), EntityType.MOTION),
            (self.config.get(CONF_MEDIA_DEVICES, []), EntityType.MEDIA),
            (self.config.get(CONF_APPLIANCES, []), EntityType.APPLIANCE),
            (self.config.get(CONF_DOOR_SENSORS, []), EntityType.DOOR),
            (self.config.get(CONF_WINDOW_SENSORS, []), EntityType.WINDOW),
            (self.config.get(CONF_LIGHTS, []), EntityType.LIGHT),
            (self.config.get(CONF_ILLUMINANCE_SENSORS, []), EntityType.ENVIRONMENTAL),
            (self.config.get(CONF_HUMIDITY_SENSORS, []), EntityType.ENVIRONMENTAL),
            (self.config.get(CONF_TEMPERATURE_SENSORS, []), EntityType.ENVIRONMENTAL),
        ]

        for entity_list, entity_type in entity_type_mappings:
            if isinstance(entity_list, list):
                for entity_id in entity_list:
                    if isinstance(entity_id, str) and entity_id:
                        self._entity_registry[entity_id] = entity_type

        _LOGGER.debug(
            "Populated entity registry with %d entities", len(self._entity_registry)
        )

    def _populate_caches(self) -> None:
        """Populate internal caches for fast lookups."""
        self._active_states_cache.clear()
        self._entity_states_cache.clear()

        # Populate active states cache
        for entity_id, entity_type in self._entity_registry.items():
            try:
                active_states = self._get_active_states_for_entity_internal(
                    entity_id, entity_type
                )
                self._active_states_cache[entity_id] = active_states
            except Exception as err:
                _LOGGER.warning(
                    "Failed to cache active states for entity %s: %s", entity_id, err
                )

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
            for entity_id, etype in self._entity_registry.items()
            if etype == entity_type
        ]

    def get_entity_type(self, entity_id: str) -> EntityType | None:
        """Get the entity type for a given entity ID.

        Args:
            entity_id: Entity ID to look up

        Returns:
            EntityType if found, None otherwise

        """
        return self._entity_registry.get(entity_id)

    def get_entity_types_mapping(self) -> dict[str, EntityType]:
        """Get the complete entity type mapping.

        Returns:
            Dictionary mapping entity IDs to their types

        """
        return self._entity_registry.copy()

    def add_entity(self, entity_id: str, entity_type: EntityType) -> None:
        """Add an entity to the registry.

        Args:
            entity_id: Entity ID to add
            entity_type: Type of the entity

        """
        self._entity_registry[entity_id] = entity_type
        # Update caches for the new entity
        try:
            active_states = self._get_active_states_for_entity_internal(
                entity_id, entity_type
            )
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
        self._entity_registry.pop(entity_id, None)
        self._active_states_cache.pop(entity_id, None)
        self._entity_states_cache.pop(entity_id, None)

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
        for entity_id in self._entity_registry:
            results[entity_id] = self.validate_entity(entity_id)
        return results

    def get_invalid_entities(self) -> list[str]:
        """Get list of invalid entity IDs.

        Returns:
            List of entity IDs that are invalid or unavailable

        """
        invalid_entities = []
        for entity_id in self._entity_registry:
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
        for entity_id in self._entity_registry:
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
        if entity_id in self._entity_states_cache:
            cached_info = self._entity_states_cache[entity_id]
            # Check if cache is still fresh (within last 30 seconds)
            if cached_info.get("last_changed"):
                try:
                    last_changed = dt_util.parse_datetime(cached_info["last_changed"])
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
        self._entity_states_cache[entity_id] = sensor_info
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
            entity_ids = list(self._entity_registry.keys())

        results = {}
        for entity_id in entity_ids:
            state_info = self.get_entity_state(entity_id)
            if state_info:
                results[entity_id] = state_info

        return results

    def refresh_entity_states(self) -> None:
        """Refresh all entity states in cache."""
        self._entity_states_cache.clear()
        for entity_id in self._entity_registry:
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

    def _get_active_states_for_entity_internal(
        self, entity_id: str, entity_type: EntityType
    ) -> set[str]:
        """Internal method to get active states for a specific entity.

        Args:
            entity_id: Entity ID
            entity_type: EntityType enum value

        Returns:
            Set of active states for this entity

        """
        # First, try to get from probabilities configuration if available
        if self.probabilities:
            try:
                sensor_config = self.probabilities.get_sensor_config_by_type(
                    entity_type
                )
                if sensor_config and "active_states" in sensor_config:
                    return sensor_config["active_states"]
            except (ValueError, KeyError):
                pass

        # Fall back to configuration-based lookup
        return self._get_configured_active_states_for_type(entity_type)

    def is_entity_active(self, entity_id: str, state: str | None = None) -> bool:
        """Check if an entity is in an active state.

        Args:
            entity_id: The entity ID to check
            state: The current state of the entity (if None, will fetch current state)

        Returns:
            True if the entity is in an active state, False otherwise

        """
        if state is None:
            state_info = self.get_entity_state(entity_id)
            if not state_info or not state_info.get("availability", False):
                return False
            state = state_info.get("state")

        if state is None:
            return False

        # Try cached lookup first
        if entity_id in self._active_states_cache:
            active_states = self._active_states_cache[entity_id]
            return state in active_states

        # Get entity type and determine active states
        entity_type = self.get_entity_type(entity_id)
        if entity_type:
            active_states = self._get_active_states_for_entity_internal(
                entity_id, entity_type
            )
            # Cache for future lookups
            self._active_states_cache[entity_id] = active_states
            return state in active_states

        # Fallback: assume 'on' states are active
        _LOGGER.warning(
            "No entity type mapping found for %s, using fallback", entity_id
        )
        return state in {"on", "detected", "playing", "open"}

    def is_entity_type_active(
        self, entity_type: EntityType | str, state: str | None
    ) -> bool:
        """Check if a state is active for a given entity type.

        Args:
            entity_type: EntityType enum or string value
            state: The state to check

        Returns:
            True if the state is active for this entity type

        """
        if state is None:
            return False

        if isinstance(entity_type, str):
            try:
                entity_type = EntityType(entity_type)
            except ValueError:
                return state in self._get_default_active_states_for_type(entity_type)

        active_states = self._get_configured_active_states_for_type(entity_type)
        return state in active_states

    def check_multiple_entities_active(
        self, entity_states: dict[str, str | None] | None = None
    ) -> dict[str, bool]:
        """Check if multiple entities are active in a single batch operation.

        Args:
            entity_states: Dictionary mapping entity IDs to their current states,
                          or None to use current states from Home Assistant

        Returns:
            Dictionary mapping entity IDs to their active status (True/False)

        """
        if entity_states is None:
            # Get current states for all entities
            state_infos = self.get_entity_states_batch()
            entity_states = {
                entity_id: info.get("state") for entity_id, info in state_infos.items()
            }

        results = {}
        for entity_id, state in entity_states.items():
            results[entity_id] = self.is_entity_active(entity_id, state)

        return results

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
            active_states = self._get_active_states_for_entity_internal(
                entity_id, entity_type
            )
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
        total_entities = len(self._entity_registry)
        entities_by_type = {}
        available_count = 0
        active_count = 0

        for entity_type in EntityType:
            entities_by_type[entity_type.value] = len(
                self.get_entities_by_type(entity_type)
            )

        # Check availability and active status
        for entity_id in self._entity_registry:
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

        for entity_id in self._entity_registry:
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
        self._entity_states_cache.clear()
        self._active_states_cache.clear()

    def refresh_all(self) -> None:
        """Refresh all entity data and caches."""
        self.clear_caches()
        self._populate_caches()
        self.refresh_entity_states()

    def update_probabilities_handler(self, probabilities: Probabilities) -> None:
        """Update the probabilities handler and refresh caches.

        Args:
            probabilities: New probabilities handler

        """
        self.probabilities = probabilities
        self._populate_caches()

    # --- State Tracking Methods (moved from coordinator) ---

    async def async_initialize_states(self, sensor_ids: list[str]) -> None:
        """Initialize sensor states for tracking.

        Args:
            sensor_ids: List of entity IDs to initialize and track

        """
        try:
            # Clear existing states
            self._current_states.clear()
            self._previous_states.clear()

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
                    "state": state_obj.state if (state_obj and is_available) else None,
                    "last_changed": (
                        state_obj.last_changed.isoformat()
                        if state_obj and state_obj.last_changed
                        else dt_util.utcnow().isoformat()
                    ),
                    "availability": is_available,
                }
                # Store in state tracking
                self._current_states[entity_id] = sensor_info

            # Setup tracking after initialization
            self.setup_state_tracking(sensor_ids, None)

        except Exception as err:
            _LOGGER.error("Failed to initialize states: %s", err)
            raise

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
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        self._tracked_entities = set(sensor_ids)
        self._state_change_callback = callback

        def async_state_changed_listener(event) -> None:
            """Handle state changes for tracked entities."""
            try:
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")
                old_state = event.data.get("old_state")

                if entity_id not in self._tracked_entities:
                    return  # Skip silently for untracked entities

                old_state_str = old_state.state if old_state else "None"
                new_state_str = new_state.state if new_state else "None"

                # Skip if state hasn't actually changed
                if old_state_str == new_state_str:
                    return

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

                # Update states synchronously to avoid async complexity
                try:
                    # Simple synchronous update - no async_add_job needed
                    self._previous_states = self._current_states.copy()
                    self._current_states[entity_id] = sensor_info
                    _LOGGER.debug("Updated state tracking for %s", entity_id)

                    # Debounced callback - only call if enough time has passed
                    if self._state_change_callback is not None:
                        import time

                        current_time = time.time()
                        last_time = self._last_callback_time.get(entity_id, 0)

                        # Only call callback if at least 0.5 seconds have passed
                        if current_time - last_time >= 0.5:
                            self._last_callback_time[entity_id] = current_time
                            try:
                                self._state_change_callback(entity_id, sensor_info)
                            except Exception as cb_err:
                                _LOGGER.exception(
                                    "Error in state change callback for %s: %s",
                                    entity_id,
                                    cb_err,
                                )
                        else:
                            _LOGGER.debug(
                                "Debouncing state change callback for %s", entity_id
                            )

                except Exception as update_err:
                    _LOGGER.exception(
                        "Error updating state for %s: %s", entity_id, update_err
                    )

            except Exception:
                _LOGGER.exception(
                    "Unexpected error in state change listener for %s",
                    entity_id if "entity_id" in locals() else "unknown",
                )

        self._remove_state_listener = async_track_state_change_event(
            self.hass,
            list(self._tracked_entities),
            async_state_changed_listener,
        )
        _LOGGER.debug(
            "State change listener set up for %d entities",
            len(self._tracked_entities),
        )

    def get_current_states(self) -> dict[str, SensorInfo]:
        """Get copy of current tracked states.

        Returns:
            Dictionary mapping entity IDs to their current SensorInfo

        """
        return self._current_states.copy()

    def get_previous_states(self) -> dict[str, SensorInfo]:
        """Get copy of previous tracked states.

        Returns:
            Dictionary mapping entity IDs to their previous SensorInfo

        """
        return self._previous_states.copy()

    def get_tracked_entities(self) -> set[str]:
        """Get set of currently tracked entity IDs.

        Returns:
            Set of entity IDs being tracked

        """
        return self._tracked_entities.copy()

    async def update_entity_state(
        self, entity_id: str, sensor_info: SensorInfo
    ) -> None:
        """Manually update state for a specific entity.

        Args:
            entity_id: Entity ID to update
            sensor_info: New sensor information

        """
        # Update previous states before changing current
        self._previous_states = self._current_states.copy()
        self._current_states[entity_id] = sensor_info
        _LOGGER.debug("Manually updated state for %s", entity_id)

    def stop_state_tracking(self) -> None:
        """Stop state tracking and cleanup resources."""
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        self._tracked_entities.clear()
        self._state_change_callback = None
        _LOGGER.debug("Stopped state tracking")


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
