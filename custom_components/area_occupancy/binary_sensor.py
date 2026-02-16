"""Binary sensor entities for Area Occupancy Detection."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_HOME, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .area import AllAreas, AreaDeviceHandle
from .const import (
    ALL_AREAS_IDENTIFIER,
    ATTR_DOOR_STATE,
    ATTR_LAST_DOOR_TIME,
    ATTR_LAST_MOTION_TIME,
    ATTR_LAST_OCCUPIED_TIME,
    ATTR_MAX_DURATION,
    ATTR_MOTION_STATE,
    ATTR_MOTION_TIMEOUT,
    ATTR_PEOPLE_DETAILS,
    ATTR_PEOPLE_SLEEPING,
    ATTR_PERSON_NAME,
    ATTR_PERSON_SLEEPING,
    ATTR_PERSON_STATE,
    ATTR_SLEEP_CONFIDENCE,
    ATTR_SLEEP_THRESHOLD,
    ATTR_VERIFICATION_DELAY,
    ATTR_VERIFICATION_PENDING,
    NAME_SLEEP_PRESENCE,
    NAME_WASP_IN_BOX,
)
from .utils import generate_entity_unique_id

if TYPE_CHECKING:
    from .area import Area
    from .coordinator import AreaOccupancyCoordinator
    from .data.config import PersonConfig

_LOGGER = logging.getLogger(__name__)

NAME_BINARY_SENSOR = "Occupancy Status"

# Door state constants
DOOR_OPEN = STATE_ON
DOOR_CLOSED = STATE_OFF


class Occupancy(CoordinatorEntity, BinarySensorEntity):
    """Binary sensor for the occupancy status."""

    def __init__(
        self,
        area_handle: AreaDeviceHandle | None = None,
        all_areas: AllAreas | None = None,
    ) -> None:
        """Initialize the sensor."""
        source = area_handle or all_areas
        if source is None:
            raise ValueError("area_handle or all_areas is required")
        super().__init__(source.coordinator)
        self._handle = area_handle
        self._all_areas = all_areas
        self._area_name = area_handle.area_name if area_handle else ALL_AREAS_IDENTIFIER
        self._attr_has_entity_name = True

        # Unique ID: use entry_id, device_id, and entity_name
        self._attr_unique_id = generate_entity_unique_id(
            source.coordinator.entry_id,
            (
                area_handle.device_info()
                if area_handle is not None
                else all_areas.device_info()
            ),
            NAME_BINARY_SENSOR,
        )
        self._attr_name = NAME_BINARY_SENSOR
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        # Get device_info directly from Area or AllAreas
        self._attr_device_info = (
            area_handle.device_info()
            if area_handle is not None
            else all_areas.device_info()
        )

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        # Let the coordinator know our entity_id (only for specific areas, not All Areas)
        if (
            self._area_name != ALL_AREAS_IDENTIFIER
            and (area := self._get_area()) is not None
        ):
            area.occupancy_entity_id = self.entity_id

            # Assign device to Home Assistant area if area_id is configured
            if area.config.area_id and self.device_info:
                device_registry = dr.async_get(self.hass)
                # DeviceInfo is a TypedDict, access identifiers directly
                identifiers = self.device_info.get("identifiers", set())
                device = device_registry.async_get_device(identifiers=identifiers)
                if device and device.area_id != area.config.area_id:
                    device_registry.async_update_device(
                        device.id, area_id=area.config.area_id
                    )

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity which will be removed."""
        # Clear the entity_id from coordinator (only for specific areas, not All Areas)
        if (
            self._area_name != ALL_AREAS_IDENTIFIER
            and (area := self._get_area()) is not None
            and area.occupancy_entity_id == self.entity_id
        ):
            area.occupancy_entity_id = None
        await super().async_will_remove_from_hass()

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend."""
        return "mdi:home-account" if self.is_on else "mdi:home-outline"

    @property
    def is_on(self) -> bool:
        """Return true if the area is occupied.

        Returns:
            bool: True if the area is currently occupied based on coordinator data,
                 False if no data is available or area is unoccupied.

        """
        if self._area_name == ALL_AREAS_IDENTIFIER:
            if self._all_areas is None:
                return False
            return self._all_areas.occupied()
        area = self._get_area()
        if area is None:
            return False
        return area.occupied()

    def _get_area(self) -> Area | None:
        """Resolve the current Area instance for this entity."""
        if self._handle is None:
            return None
        return self._handle.area

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        super()._handle_coordinator_update()


class WaspInBoxSensor(RestoreEntity, BinarySensorEntity):
    """Wasp in Box binary sensor implementation.

    This sensor detects occupancy based on door and motion sensor states.
    The concept is that once someone enters a room with a single entry point
    (door closes with motion detected), they remain in that room until the door opens again,
    similar to a wasp trapped in a box.
    """

    _attr_should_poll = False

    def __init__(
        self,
        area_handle: AreaDeviceHandle,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor.

        Args:
            area_handle: Stable reference to the parent Area.
            config_entry: The config entry (for backward compatibility with unique_id migration)
        """
        _LOGGER.debug(
            "Initializing WaspInBoxSensor for area: %s", area_handle.area_name
        )
        super().__init__()

        # Store references and configuration
        # Note: self.hass is automatically set by Home Assistant when entity is added
        self._handle = area_handle
        self._coordinator = area_handle.coordinator
        self._area_name = area_handle.area_name
        area = area_handle.resolve()
        if area is None:
            raise ValueError(f"Area '{area_handle.area_name}' is not available")
        self._config = area.config
        self._motion_timeout = self._config.wasp_in_box.motion_timeout
        self._weight = self._config.wasp_in_box.weight
        self._max_duration = self._config.wasp_in_box.max_duration
        self._verification_delay = self._config.wasp_in_box.verification_delay

        # Configure entity properties
        self._attr_has_entity_name = True
        # Unique ID: use entry_id, device_id, and entity_name
        self._attr_unique_id = generate_entity_unique_id(
            area_handle.coordinator.entry_id,
            area.device_info(),
            NAME_WASP_IN_BOX,
        )
        self._attr_name = NAME_WASP_IN_BOX
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        # Get device_info directly from Area
        self._attr_device_info = area.device_info() if area is not None else None
        self._attr_available = True
        self._attr_is_on = False

        # Initialize state tracking
        self._state = STATE_OFF
        self._door_state = DOOR_CLOSED
        self._motion_state = STATE_OFF
        self._last_door_time: datetime | None = None
        self._last_motion_time: datetime | None = None
        self._last_occupied_time: datetime | None = None

        # Initialize tracking resources
        self._door_entities = self._config.sensors.door or []
        self._motion_entities = self._config.sensors.motion or []
        self._remove_state_listener: Callable[[], None] | None = None
        self._remove_timer: Callable[[], None] | None = None
        self._remove_verification_timer: Callable[[], None] | None = None
        self._verification_pending: bool = False

        # Check if we have required entities configured
        if not self._door_entities or not self._motion_entities:
            _LOGGER.warning(
                "No door or motion entities configured for Wasp in Box sensor. Sensor will not function properly"
            )

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

            # Let the coordinator know our entity_id
            if (area := self._get_area()) is not None:
                area.wasp_entity_id = self.entity_id

            _LOGGER.debug("WaspInBoxSensor setup completed for %s", self.entity_id)
        except Exception:
            _LOGGER.exception(
                "Error during WaspInBoxSensor setup for %s", self.entity_id
            )

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

    def _cleanup_all_resources(self) -> None:
        """Defensive cleanup of all resources (timers, listeners, state).

        This method ensures all resources are cleaned up even if
        async_will_remove_from_hass() is not called, preventing memory leaks.
        """
        _LOGGER.debug(
            "Performing defensive cleanup for Wasp in Box sensor: %s", self.entity_id
        )

        # Cancel state listener
        if self._remove_state_listener is not None:
            try:
                self._remove_state_listener()
            except (RuntimeError, AttributeError) as err:
                _LOGGER.warning("Error canceling state listener: %s", err)
            self._remove_state_listener = None

        # Cancel all timers
        self._cancel_max_duration_timer()
        self._cancel_verification_timer()

        # Clear coordinator reference
        area = self._get_area()
        if area is not None and area.wasp_entity_id == self.entity_id:
            area.wasp_entity_id = None

        _LOGGER.debug(
            "Defensive cleanup completed for Wasp in Box sensor: %s", self.entity_id
        )

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup when entity is removed."""
        _LOGGER.debug("Removing Wasp in Box sensor: %s", self.entity_id)
        self._cleanup_all_resources()
        await super().async_will_remove_from_hass()

    @property
    def extra_state_attributes(self) -> dict[str, str | int | None | bool]:
        """Return the state attributes."""
        return {
            ATTR_DOOR_STATE: self._door_state,
            ATTR_LAST_DOOR_TIME: (
                self._last_door_time.isoformat() if self._last_door_time else None
            ),
            ATTR_MOTION_STATE: self._motion_state,
            ATTR_LAST_MOTION_TIME: (
                self._last_motion_time.isoformat() if self._last_motion_time else None
            ),
            ATTR_MOTION_TIMEOUT: self._motion_timeout,
            ATTR_MAX_DURATION: self._max_duration,
            ATTR_LAST_OCCUPIED_TIME: (
                self._last_occupied_time.isoformat()
                if self._last_occupied_time
                else None
            ),
            ATTR_VERIFICATION_DELAY: self._verification_delay,
            ATTR_VERIFICATION_PENDING: self._verification_pending,
        }

    @property
    def weight(self) -> float:
        """Return the sensor weight for probability calculation."""
        return self._weight

    def _setup_entity_tracking(self) -> None:
        """Set up state tracking for door and motion entities."""
        if not self._door_entities and not self._motion_entities:
            _LOGGER.warning(
                "No door or motion entities configured for Wasp in Box sensor. Sensor will not function properly"
            )
            return

        # Clean up existing listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Get valid entities and set up tracking
        valid_entities = self._get_valid_entities()
        if not valid_entities["all"]:
            _LOGGER.warning(
                "No valid entities found to track. Sensor will not function"
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
        """Return valid entity IDs for tracking.

        We return all configured entities regardless of whether they currently
        exist in the state machine, so that we can pick up their state changes
        if they become available later.
        """
        return {
            "doors": self._door_entities,
            "motion": self._motion_entities,
            "all": self._door_entities + self._motion_entities,
        }

    def _get_aggregate_door_state(self) -> str:
        """Calculate aggregate door state from all door sensors.

        Returns DOOR_OPEN if ANY door is open, DOOR_CLOSED if all are closed.
        For the "wasp in box" concept, any open door allows the wasp to escape.

        Returns:
            str: DOOR_OPEN if any door is open, DOOR_CLOSED if all are closed.
                 Returns DOOR_CLOSED if no door entities are configured.
        """
        if not self._door_entities:
            return DOOR_CLOSED

        # Check all door sensors
        for entity_id in self._door_entities:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ["unknown", "unavailable"]:
                # If ANY door is open, return DOOR_OPEN
                if state.state == DOOR_OPEN:
                    return DOOR_OPEN

        # All doors are closed (or unavailable/unknown)
        return DOOR_CLOSED

    def _get_aggregate_motion_state(self) -> str:
        """Calculate aggregate motion state from all motion sensors.

        Returns STATE_ON if ANY motion sensor is active, STATE_OFF if all are off.
        For the "wasp in box" concept, any motion indicates presence.

        Returns:
            str: STATE_ON if any motion sensor is active, STATE_OFF if all are off.
                 Returns STATE_OFF if no motion entities are configured.
        """
        if not self._motion_entities:
            return STATE_OFF

        # Check all motion sensors
        for entity_id in self._motion_entities:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ["unknown", "unavailable"]:
                # If ANY motion sensor is active, return STATE_ON
                if state.state == STATE_ON:
                    return STATE_ON

        # All motion sensors are off (or unavailable/unknown)
        return STATE_OFF

    def _initialize_from_current_states(
        self, valid_entities: dict[str, list[str]]
    ) -> None:
        """Initialize sensor state from current entity states."""
        # Check current door states
        for entity_id in valid_entities["doors"]:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ["unknown", "unavailable"]:
                self._process_door_state(entity_id, state.state)

        # Set aggregate door state after processing all door sensors
        self._door_state = self._get_aggregate_door_state()

        # Check current motion states
        for entity_id in valid_entities["motion"]:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ["unknown", "unavailable"]:
                self._process_motion_state(entity_id, state.state)

        # Set aggregate motion state after processing all motion sensors
        self._motion_state = self._get_aggregate_motion_state()

    @callback
    def _handle_state_change(self, event: Any) -> None:
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
        # Store previous aggregate door state for comparison
        previous_door_state = self._door_state

        # Update timestamp for this door event
        self._last_door_time = dt_util.utcnow()

        # Recalculate aggregate door state from all door sensors
        self._door_state = self._get_aggregate_door_state()

        _LOGGER.debug(
            "Door state change: %s changed to %s, aggregate door state: %s (was %s)",
            entity_id,
            new_state,
            self._door_state,
            previous_door_state,
        )

        # Check if ANY door has opened while room was occupied
        door_is_open = self._door_state == DOOR_OPEN
        door_was_closed = previous_door_state == DOOR_CLOSED

        if door_is_open and door_was_closed and self._state == STATE_ON:
            # Any door opened while room was occupied - set to unoccupied
            _LOGGER.debug("Door opened while occupied - marking room as unoccupied")
            self._cancel_verification_timer()
            self._set_state(STATE_OFF)
        elif not door_is_open:  # All doors closed
            # Pattern A: All doors closed with active motion
            if self._motion_state == STATE_ON:
                _LOGGER.debug(
                    "All doors closed with motion detected - marking room as occupied"
                )
                self._set_state(STATE_ON)
            # Pattern B: All doors closed with recent motion (within motion_timeout)
            elif self._last_motion_time:
                now = dt_util.utcnow()
                motion_age = (now - self._last_motion_time).total_seconds()
                if motion_age <= self._motion_timeout:
                    _LOGGER.debug(
                        "All doors closed with recent motion (%.1fs ago) - marking room as occupied",
                        motion_age,
                    )
                    self._set_state(STATE_ON)
                else:
                    _LOGGER.debug(
                        "All doors closed but motion is too old (%.1fs) - not marking as occupied",
                        motion_age,
                    )
                    self.async_write_ha_state()
            else:
                # No state change, just update attributes
                self.async_write_ha_state()
        else:
            # No state change, just update attributes
            self.async_write_ha_state()

    def _process_motion_state(self, entity_id: str, new_state: str) -> None:
        """Process a motion state change event."""
        # Store previous aggregate motion state
        old_motion = self._motion_state

        # Only update timestamp when any motion starts, not when it stops
        if new_state == STATE_ON:
            self._last_motion_time = dt_util.utcnow()

        # Recalculate aggregate motion state from all motion sensors
        self._motion_state = self._get_aggregate_motion_state()

        _LOGGER.debug(
            "Motion state change: %s changed to %s, aggregate motion state: %s (was %s)",
            entity_id,
            new_state,
            self._motion_state,
            old_motion,
        )

        # All doors closed + any motion = occupied
        if self._motion_state == STATE_ON and self._door_state == DOOR_CLOSED:
            _LOGGER.debug(
                "Motion detected with all doors closed - marking room as occupied"
            )
            self._set_state(STATE_ON)
        elif self._motion_state == STATE_OFF:
            # All motion stopped - maintain current state until door opens
            _LOGGER.debug(
                "All motion stopped - maintaining current state until door opens"
            )
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
            # Start verification timer if enabled
            self._start_verification_timer()
        else:
            # Cancel duration timer when becoming unoccupied
            self._cancel_max_duration_timer()
            self._cancel_verification_timer()

        # Update Home Assistant state
        self.async_write_ha_state()

        # WaspInBoxSensor works as a standalone sensor and doesn't interfere with coordinator entities

        _LOGGER.debug("State changed from %s to %s", old_state, new_state)

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

    def _start_verification_timer(self) -> None:
        """Start a timer to verify occupancy after a delay."""
        self._cancel_verification_timer()

        # Skip if verification is disabled (0 or None)
        # Also check if it's a valid integer to handle test mocks
        try:
            delay = int(self._verification_delay) if self._verification_delay else 0
        except (TypeError, ValueError):
            delay = 0

        if not delay:
            return

        # Schedule callback for verification time
        verification_time = dt_util.utcnow() + timedelta(seconds=delay)
        self._remove_verification_timer = async_track_point_in_time(
            self.hass, self._handle_verification_check, verification_time
        )
        self._verification_pending = True
        _LOGGER.debug(
            "Verification timer scheduled for %s seconds from now",
            delay,
        )

    def _cancel_verification_timer(self) -> None:
        """Cancel any scheduled verification timer."""
        if self._remove_verification_timer:
            self._remove_verification_timer()
            self._remove_verification_timer = None
        self._verification_pending = False

    @callback
    def _handle_verification_check(self, _now: datetime) -> None:
        """Handle verification timer expiration - check if motion is still present."""
        self._remove_verification_timer = None
        self._verification_pending = False

        # Only proceed if we're still in occupied state
        if self._state != STATE_ON:
            _LOGGER.debug("Verification check: already unoccupied, skipping")
            return

        # If no motion sensors are configured, skip verification and maintain occupancy
        # This supports "wasp in box" door-only setups
        if not self._motion_entities:
            _LOGGER.debug(
                "Verification check: no motion sensors configured, skipping verification"
            )
            # Motion-based verification doesn't apply, just update state
            self.async_write_ha_state()
            return

        # Check if any motion sensor is currently active
        has_active_motion = False
        for motion_entity in self._motion_entities:
            state = self.hass.states.get(motion_entity)
            if state and state.state == STATE_ON:
                has_active_motion = True
                break

        if has_active_motion:
            _LOGGER.debug(
                "Verification check: motion still detected, maintaining occupancy"
            )
            # Motion is confirmed - just update state
            self.async_write_ha_state()
        else:
            # No motion detected - this was likely a false positive
            _LOGGER.info(
                "Verification check: no motion detected after %s seconds, clearing occupancy",
                self._verification_delay,
            )
            self._set_state(STATE_OFF)

    def _get_area(self) -> Area | None:
        """Return the current Area backing this sensor."""
        return self._handle.resolve()


class SleepPresenceSensor(RestoreEntity, BinarySensorEntity):
    """Sleep Presence binary sensor implementation.

    This sensor detects sleep presence based on HA person entity state
    and phone sleep confidence sensor. When a person is home and their
    sleep confidence exceeds the threshold, the sensor turns ON, feeding
    into the area's occupancy calculation as a SLEEP input type.
    """

    _attr_should_poll = False

    def __init__(
        self,
        area_handle: AreaDeviceHandle,
        config_entry: ConfigEntry,
        people: list[PersonConfig],
    ) -> None:
        """Initialize the sensor."""
        _LOGGER.debug(
            "Initializing SleepPresenceSensor for area: %s with %d people",
            area_handle.area_name,
            len(people),
        )
        super().__init__()

        self._handle = area_handle
        self._coordinator = area_handle.coordinator
        self._area_name = area_handle.area_name
        self._people = people

        area = area_handle.resolve()
        if area is None:
            raise ValueError(f"Area '{area_handle.area_name}' is not available")

        # Configure entity properties
        self._attr_has_entity_name = True
        self._attr_unique_id = generate_entity_unique_id(
            area_handle.coordinator.entry_id,
            area.device_info(),
            NAME_SLEEP_PRESENCE,
        )
        self._attr_name = NAME_SLEEP_PRESENCE
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._attr_device_info = area.device_info()
        self._attr_available = True
        self._attr_is_on = False

        # State tracking
        self._state = STATE_OFF
        self._remove_state_listener: Callable[[], None] | None = None

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        _LOGGER.debug("SleepPresenceSensor async_added_to_hass for %s", self.unique_id)
        await super().async_added_to_hass()

        # Restore previous state
        if (last_state := await self.async_get_last_state()) is not None:
            self._state = last_state.state
            self._attr_is_on = self._state == STATE_ON

        # Register entity_id with the area and add to EntityManager
        if (area := self._get_area()) is not None:
            area.sleep_entity_id = self.entity_id

            # Add sleep entity to EntityManager so it's tracked for occupancy
            # calculation. EntityManager is created before platform setup, so
            # sleep_entity_id is None at that point — we must register it now.
            area.entities.register_entity(self.entity_id, "sleep")
            _LOGGER.debug("Registered sleep entity %s in EntityManager", self.entity_id)

        # Set up state tracking for all person + sleep confidence entities
        self._setup_entity_tracking()
        _LOGGER.debug("SleepPresenceSensor setup completed for %s", self.entity_id)

    def _setup_entity_tracking(self) -> None:
        """Set up state tracking for person and sleep confidence entities."""
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        tracked_entities: list[str] = []
        for person in self._people:
            tracked_entities.append(person.person_entity)
            tracked_entities.append(person.sleep_confidence_sensor)

        if not tracked_entities:
            _LOGGER.warning(
                "No entities to track for SleepPresenceSensor in area %s",
                self._area_name,
            )
            return

        self._remove_state_listener = async_track_state_change_event(
            self.hass, tracked_entities, self._handle_state_change
        )

        # Initialize from current states
        self._evaluate_and_update()

    @callback
    def _handle_state_change(self, event: Any) -> None:
        """Handle state changes for tracked entities."""
        self._evaluate_and_update()

    def _evaluate_sleep_state(self) -> bool:
        """Evaluate whether any person is home and sleeping."""
        for person in self._people:
            person_state = self.hass.states.get(person.person_entity)
            confidence_state = self.hass.states.get(person.sleep_confidence_sensor)
            if (
                person_state
                and person_state.state == STATE_HOME
                and confidence_state
                and confidence_state.state not in ("unknown", "unavailable", None, "")
            ):
                try:
                    confidence = float(confidence_state.state)
                    if confidence >= person.confidence_threshold:
                        return True
                except (ValueError, TypeError):
                    continue
        return False

    def _evaluate_and_update(self) -> None:
        """Evaluate sleep state and update if changed."""
        is_sleeping = self._evaluate_sleep_state()
        new_state = STATE_ON if is_sleeping else STATE_OFF

        if new_state != self._state:
            self._state = new_state
            self._attr_is_on = new_state == STATE_ON
            _LOGGER.debug(
                "Sleep presence changed to %s for area %s",
                new_state,
                self._area_name,
            )
        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        people_sleeping: list[str] = []
        person_details: list[dict[str, Any]] = []

        for person in self._people:
            person_state = self.hass.states.get(person.person_entity)
            confidence_state = self.hass.states.get(person.sleep_confidence_sensor)

            friendly_name = (
                person_state.attributes.get("friendly_name", person.person_entity)
                if person_state
                else person.person_entity
            )

            confidence_val: float | None = None
            if confidence_state and confidence_state.state not in (
                "unknown",
                "unavailable",
                None,
                "",
            ):
                with suppress(ValueError, TypeError):
                    confidence_val = float(confidence_state.state)

            is_home = person_state and person_state.state == STATE_HOME
            is_sleeping = (
                is_home
                and confidence_val is not None
                and confidence_val >= person.confidence_threshold
            )

            if is_sleeping:
                people_sleeping.append(friendly_name)

            person_details.append(
                {
                    ATTR_PERSON_NAME: friendly_name,
                    ATTR_PERSON_STATE: person_state.state
                    if person_state
                    else "unknown",
                    ATTR_SLEEP_CONFIDENCE: confidence_val,
                    ATTR_SLEEP_THRESHOLD: person.confidence_threshold,
                    ATTR_PERSON_SLEEPING: is_sleeping,
                }
            )

        return {
            ATTR_PEOPLE_SLEEPING: people_sleeping,
            ATTR_PEOPLE_DETAILS: person_details,
        }

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup when entity is removed."""
        _LOGGER.debug("Removing Sleep Presence sensor: %s", self.entity_id)
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        area = self._get_area()
        if area is not None and area.sleep_entity_id == self.entity_id:
            area.sleep_entity_id = None

        await super().async_will_remove_from_hass()

    def _get_area(self) -> Area | None:
        """Return the current Area backing this sensor."""
        return self._handle.resolve()


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: Any
) -> None:
    """Set up the Area Occupancy Detection binary sensors."""
    coordinator: AreaOccupancyCoordinator = config_entry.runtime_data

    entities: list[BinarySensorEntity] = []

    # Create occupancy sensors for each area
    for area_name in coordinator.get_area_names():
        handle = coordinator.get_area_handle(area_name)
        _LOGGER.debug("Creating occupancy sensor for area: %s", area_name)
        entities.append(Occupancy(area_handle=handle))

        # Create Wasp in Box sensor if enabled for this area
        area = coordinator.get_area(area_name)
        if area and area.config.wasp_in_box.enabled:
            _LOGGER.debug(
                "Wasp in Box sensor enabled for area %s, creating sensor", area_name
            )
            wasp_sensor = WaspInBoxSensor(
                area_handle=handle,
                config_entry=config_entry,
            )
            entities.append(wasp_sensor)
            _LOGGER.debug(
                "Created Wasp in Box sensor for area %s: %s",
                area_name,
                wasp_sensor.unique_id,
            )

        # Create Sleep Presence sensor if any people assigned to this area
        if area and area.config.area_id:
            people_for_area = coordinator.integration_config.get_people_for_area(
                area.config.area_id
            )
            if people_for_area:
                _LOGGER.debug(
                    "Sleep Presence sensor: %d people for area %s, creating sensor",
                    len(people_for_area),
                    area_name,
                )
                sleep_sensor = SleepPresenceSensor(
                    area_handle=handle,
                    config_entry=config_entry,
                    people=people_for_area,
                )
                entities.append(sleep_sensor)

    # Create "All Areas" aggregation occupancy sensor when areas exist
    if len(coordinator.get_area_names()) >= 1:
        _LOGGER.debug("Creating All Areas aggregation occupancy sensor")
        entities.append(
            Occupancy(
                all_areas=coordinator.get_all_areas(),
            )
        )

    async_add_entities(entities, update_before_add=False)
