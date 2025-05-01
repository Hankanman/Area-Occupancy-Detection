"""Wasp in Box virtual sensor implementation.

This module implements a virtual binary sensor that detects occupancy based on door and motion
sensor states. The concept is that once someone enters a room with a single entry point
(door closes with motion detected), they remain in that room until the door opens again,
similar to a wasp trapped in a box.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

from custom_components.area_occupancy.const import (
    ATTR_DOOR_STATE,
    ATTR_LAST_DOOR_TIME,
    ATTR_LAST_MOTION_TIME,
    ATTR_MOTION_STATE,
    ATTR_MOTION_TIMEOUT,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_WEIGHT,
    NAME_WASP_IN_BOX,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.types import (
    EntityType,
    SensorInfo,
    WaspInBoxAttributes,
    WaspInBoxConfig,
)

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

ATTR_LAST_OCCUPIED_TIME = "last_occupied_time"
ATTR_MAX_DURATION = "max_duration"


async def async_setup_entry(
    hass: HomeAssistant,
    config: WaspInBoxConfig,
    async_add_entities: AddEntitiesCallback,
    coordinator: AreaOccupancyCoordinator,
) -> WaspInBoxSensor | None:
    """Set up the Wasp in Box sensor from config.

    Returns:
        The created WaspInBoxSensor instance, or None if disabled/error.

    """
    if not config.get("enabled", False):
        _LOGGER.debug("Wasp in Box sensor disabled, skipping setup")
        return None

    # Return the created sensor instance
    return WaspInBoxSensor(
        hass,
        config,
        coordinator,
        coordinator.config_entry.entry_id,
        coordinator.inputs.door_sensors,
        coordinator.inputs.motion_sensors,
    )


# Re-enable RestoreEntity inheritance
class WaspInBoxSensor(RestoreEntity, BinarySensorEntity):
    """Implementation of the Wasp in Box sensor."""

    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        config: WaspInBoxConfig,
        coordinator: AreaOccupancyCoordinator,
        entry_id: str,
        door_entities: list[str] | None = None,
        motion_entities: list[str] | None = None,
    ) -> None:
        """Initialize the sensor."""
        _LOGGER.debug("Initializing WaspInBoxSensor for entry_id: %s", entry_id)
        super().__init__()

        # Store references and configuration
        self.hass = hass
        self._coordinator = coordinator
        self._motion_timeout = config.get("motion_timeout", DEFAULT_WASP_MOTION_TIMEOUT)
        self._weight = config.get("weight", DEFAULT_WASP_WEIGHT)
        self._max_duration = config.get("max_duration", DEFAULT_WASP_MAX_DURATION)

        # Configure entity properties
        self._attr_has_entity_name = True
        self._attr_unique_id = (
            f"{entry_id}_{NAME_WASP_IN_BOX.lower().replace(' ', '_')}"
        )
        self._attr_name = NAME_WASP_IN_BOX
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._attr_device_info = coordinator.device_info
        self._attr_available = False
        self._attr_is_on = False

        # Initialize state tracking
        self._state = STATE_OFF
        self._door_state = STATE_OFF
        self._motion_state = STATE_OFF
        self._last_door_time = None
        self._last_motion_time = None
        self._last_occupied_time = None

        # Initialize tracking resources
        self._door_entities = door_entities or []
        self._motion_entities = motion_entities or []
        self._remove_state_listener = None
        self._remove_timer = None

        _LOGGER.debug(
            "WaspInBoxSensor initialized with unique_id: %s", self._attr_unique_id
        )

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        _LOGGER.debug(
            "WaspInBoxSensor async_added_to_hass started for %s", self.unique_id
        )
        try:
            await super().async_added_to_hass()
            await self._restore_previous_state()
            self._setup_entity_tracking()
            await self._register_with_coordinator()

            # Mark sensor as available
            self._attr_available = True
            self.async_write_ha_state()
            _LOGGER.debug("WaspInBoxSensor setup completed for %s", self.entity_id)
        except Exception:
            _LOGGER.exception(
                "Error during WaspInBoxSensor setup for %s", self.entity_id
            )
            self._attr_available = False
            self.async_write_ha_state()

    async def _restore_previous_state(self) -> None:
        """Restore state from previous run if available."""
        if (last_state := await self.async_get_last_state()) is not None:
            _LOGGER.debug(
                "Restoring previous state for %s: %s", self.entity_id, last_state.state
            )
            self._state = last_state.state
            self._attr_is_on = self._state == STATE_ON

            if last_state.attributes:
                if last_door_time := last_state.attributes.get(ATTR_LAST_DOOR_TIME):
                    self._last_door_time = dt_util.parse_datetime(last_door_time)
                if last_motion_time := last_state.attributes.get(ATTR_LAST_MOTION_TIME):
                    self._last_motion_time = dt_util.parse_datetime(last_motion_time)
                if last_occupied_time := last_state.attributes.get(
                    ATTR_LAST_OCCUPIED_TIME
                ):
                    self._last_occupied_time = dt_util.parse_datetime(
                        last_occupied_time
                    )

            # If restoring to occupied state, start a timer if max duration is set
            if self._state == STATE_ON:
                self._last_occupied_time = self._last_occupied_time or dt_util.utcnow()
                self._start_max_duration_timer()
        else:
            _LOGGER.debug("No previous state found for %s to restore", self.entity_id)

    async def _register_with_coordinator(self) -> None:
        """Register this sensor with the coordinator."""
        if not self.entity_id:
            return

        # Add to virtual sensors list
        if self.entity_id not in self._coordinator.inputs.virtual_sensors:
            self._coordinator.inputs.virtual_sensors.append(self.entity_id)
            _LOGGER.debug(
                "Added virtual sensor %s to coordinator inputs", self.entity_id
            )

        # Map the virtual sensor to its type in probabilities
        if self.entity_id not in self._coordinator.probabilities.entity_types:
            self._coordinator.probabilities.entity_types[self.entity_id] = (
                EntityType.WASP_IN_BOX
            )
            _LOGGER.debug(
                "Mapped virtual sensor %s to type %s in probabilities",
                self.entity_id,
                EntityType.WASP_IN_BOX.value,
            )

        # Initialize coordinator state
        current_state: SensorInfo = {
            "state": self._state,
            "last_changed": dt_util.utcnow().isoformat(),
            "availability": True,
        }

        # Update the coordinator's current_states
        if hasattr(self._coordinator, "data") and self._coordinator.data:
            if not hasattr(self._coordinator.data, "current_states"):
                self._coordinator.data.current_states = {}
            self._coordinator.data.current_states[self.entity_id] = current_state
            _LOGGER.debug(
                "Initialized state in coordinator: %s = %s",
                self.entity_id,
                current_state,
            )

            # Request refresh to apply the changes
            await self._coordinator.async_request_refresh()

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup when entity is removed."""
        _LOGGER.debug("Removing Wasp in Box sensor: %s", self.entity_id)
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        self._cancel_max_duration_timer()

    @property
    def extra_state_attributes(self) -> WaspInBoxAttributes:
        """Return the state attributes."""
        attrs: WaspInBoxAttributes = {
            ATTR_DOOR_STATE: self._door_state,
            ATTR_LAST_DOOR_TIME: self._last_door_time.isoformat()
            if self._last_door_time
            else None,
            ATTR_MOTION_STATE: self._motion_state,
            ATTR_LAST_MOTION_TIME: self._last_motion_time.isoformat()
            if self._last_motion_time
            else None,
            ATTR_MOTION_TIMEOUT: self._motion_timeout,
            ATTR_MAX_DURATION: self._max_duration,
            ATTR_LAST_OCCUPIED_TIME: self._last_occupied_time.isoformat()
            if self._last_occupied_time
            else None,
        }

        return attrs

    @property
    def weight(self) -> float:
        """Return the sensor weight for probability calculation."""
        return self._weight

    def _setup_entity_tracking(self) -> None:
        """Set up state tracking for door and motion entities."""
        if not self._door_entities and not self._motion_entities:
            _LOGGER.warning(
                "No door or motion entities configured for Wasp in Box sensor. Sensor will not function properly."
            )
            return

        # Clean up existing listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Get valid entities and set up tracking
        valid_entities = self._get_valid_entities()
        if not valid_entities:
            _LOGGER.warning(
                "No valid entities found to track. Sensor will not function."
            )
            return

        # Set up state change tracking
        self._remove_state_listener = async_track_state_change_event(
            self.hass, valid_entities["all"], self._handle_state_change
        )

        # Initialize from current states
        self._initialize_from_current_states(valid_entities)

        _LOGGER.debug(
            "Tracking %d entities (%d doors, %d motion)",
            len(valid_entities["all"]),
            len(valid_entities["doors"]),
            len(valid_entities["motion"]),
        )

    def _get_valid_entities(self) -> dict[str, list[str]]:
        """Filter and return valid entity IDs for tracking."""
        # Filter out invalid entities
        valid_door_entities = [
            entity_id
            for entity_id in self._door_entities
            if self.hass.states.get(entity_id) is not None
        ]

        valid_motion_entities = [
            entity_id
            for entity_id in self._motion_entities
            if self.hass.states.get(entity_id) is not None
        ]

        return {
            "doors": valid_door_entities,
            "motion": valid_motion_entities,
            "all": valid_door_entities + valid_motion_entities,
        }

    def _initialize_from_current_states(
        self, valid_entities: dict[str, list[str]]
    ) -> None:
        """Initialize sensor state from current entity states."""
        # Check current door states
        for entity_id in valid_entities["doors"]:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ["unknown", "unavailable"]:
                self._process_door_state(entity_id, state.state)

        # Check current motion states
        for entity_id in valid_entities["motion"]:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ["unknown", "unavailable"]:
                self._process_motion_state(entity_id, state.state)

    @callback
    def _handle_state_change(self, event) -> None:
        """Handle state changes for tracked entities."""
        entity_id = event.data.get("entity_id")
        new_state = event.data.get("new_state")

        if not new_state or new_state.state in ["unknown", "unavailable"]:
            return

        # Process based on entity type
        if entity_id in self._door_entities:
            self._process_door_state(entity_id, new_state.state)
        elif entity_id in self._motion_entities:
            self._process_motion_state(entity_id, new_state.state)

    def _process_door_state(self, entity_id: str, new_state: str) -> None:
        """Process a door state change event."""
        # Store previous door state for comparison
        previous_door_state = self._door_state

        # Update current door state
        self._door_state = new_state
        self._last_door_time = dt_util.utcnow()

        _LOGGER.debug(
            "Door state change: %s changed from %s to %s",
            entity_id,
            previous_door_state,
            new_state,
        )

        # Check if door has opened while room was occupied
        door_is_open = new_state == STATE_ON
        door_was_closed = previous_door_state == STATE_OFF

        if door_is_open and door_was_closed and self._state == STATE_ON:
            # Door opened while room was occupied - set to unoccupied
            _LOGGER.debug("Door opened while occupied - marking room as unoccupied")
            self._set_state(STATE_OFF)
        elif not door_is_open and self._motion_state == STATE_ON:
            # Door closed with active motion - set to occupied
            _LOGGER.debug("Door closed with motion detected - marking room as occupied")
            self._set_state(STATE_ON)
        else:
            # No state change, just update attributes
            self.async_write_ha_state()

    def _process_motion_state(self, entity_id: str, new_state: str) -> None:
        """Process a motion state change event."""
        # Update motion state
        old_motion = self._motion_state
        self._motion_state = new_state
        self._last_motion_time = dt_util.utcnow()

        _LOGGER.debug(
            "Motion state change: %s changed from %s to %s",
            entity_id,
            old_motion,
            new_state,
        )

        # Door closed + motion = occupied
        if new_state == STATE_ON and self._door_state == STATE_OFF:
            _LOGGER.debug("Motion detected with door closed - marking room as occupied")
            self._set_state(STATE_ON)
        elif new_state == STATE_OFF:
            # Motion stopped - maintain current state until door opens
            _LOGGER.debug("Motion stopped - maintaining current state until door opens")
            self.async_write_ha_state()

    def _set_state(self, new_state: str) -> None:
        """Set the sensor state and update all necessary components."""
        old_state = self._state
        self._state = new_state
        self._attr_is_on = new_state == STATE_ON

        if new_state == STATE_ON:
            # Record occupied time and start max duration timer
            self._last_occupied_time = dt_util.utcnow()
            self._start_max_duration_timer()
        else:
            # Cancel duration timer when becoming unoccupied
            self._cancel_max_duration_timer()

        # Update Home Assistant state
        self.async_write_ha_state()

        # Update coordinator state
        self._update_coordinator_state(new_state)

        _LOGGER.debug("State changed from %s to %s", old_state, new_state)

    def _update_coordinator_state(self, new_state: str) -> None:
        """Update the coordinator with this entity's current state."""
        if not self._coordinator or not self.entity_id:
            return

        # Skip if coordinator doesn't have data structure
        if not hasattr(self._coordinator, "data") or not self._coordinator.data:
            return

        # Ensure current_states dict exists
        if not hasattr(self._coordinator.data, "current_states"):
            self._coordinator.data.current_states = {}

        # Update coordinator state
        self._coordinator.data.current_states[self.entity_id] = {
            "state": new_state,
            "last_changed": dt_util.utcnow().isoformat(),
            "availability": self._attr_available,
        }

        # Request refresh to recalculate probabilities
        self.hass.async_create_task(self._coordinator.async_request_refresh())

    def _start_max_duration_timer(self) -> None:
        """Start a timer to reset occupancy after max duration."""
        self._cancel_max_duration_timer()

        # Skip if max duration is disabled (0 or None)
        if not self._max_duration:
            return

        # Calculate when the max duration will expire
        if self._last_occupied_time:
            max_duration_end = self._last_occupied_time + timedelta(
                seconds=self._max_duration
            )
            now = dt_util.utcnow()

            # If already expired, reset immediately
            if max_duration_end <= now:
                self._reset_after_max_duration()
                return

            # Schedule callback for expiration time
            self._remove_timer = async_track_point_in_time(
                self.hass, self._handle_max_duration_timeout, max_duration_end
            )
            _LOGGER.debug(
                "Max duration timer scheduled to expire at %s",
                max_duration_end.isoformat(),
            )

    def _cancel_max_duration_timer(self) -> None:
        """Cancel any scheduled max duration timer."""
        if self._remove_timer:
            self._remove_timer()
            self._remove_timer = None

    @callback
    def _handle_max_duration_timeout(self, _now: datetime) -> None:
        """Handle max duration timer expiration."""
        self._reset_after_max_duration()
        self._remove_timer = None

    def _reset_after_max_duration(self) -> None:
        """Reset occupancy state after max duration has elapsed."""
        if self._state == STATE_ON:
            _LOGGER.debug(
                "Max duration (%s seconds) exceeded, changing to unoccupied",
                self._max_duration,
            )
            self._set_state(STATE_OFF)
