"""Binary sensor entities for Area Occupancy Detection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import (
    ATTR_DOOR_STATE,
    ATTR_LAST_DOOR_TIME,
    ATTR_LAST_MOTION_TIME,
    ATTR_LAST_OCCUPIED_TIME,
    ATTR_MAX_DURATION,
    ATTR_MOTION_STATE,
    ATTR_MOTION_TIMEOUT,
    ATTR_VERIFICATION_DELAY,
    ATTR_VERIFICATION_PENDING,
    NAME_WASP_IN_BOX,
)
from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

NAME_BINARY_SENSOR = "Occupancy Status"

# Door state constants
DOOR_OPEN = STATE_ON
DOOR_CLOSED = STATE_OFF


class Occupancy(CoordinatorEntity[AreaOccupancyCoordinator], BinarySensorEntity):
    """Binary sensor for the occupancy status."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, entry_id: str) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_has_entity_name = True
        self._attr_unique_id = (
            f"{entry_id}_{NAME_BINARY_SENSOR.lower().replace(' ', '_')}"
        )
        self._attr_name = NAME_BINARY_SENSOR
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._attr_device_info: DeviceInfo | None = coordinator.device_info

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        # Let the coordinator know our entity_id
        self.coordinator.occupancy_entity_id = self.entity_id

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity which will be removed."""
        # Clear the entity_id from coordinator
        if self.coordinator.occupancy_entity_id == self.entity_id:
            self.coordinator.occupancy_entity_id = None
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
        return self.coordinator.occupied

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        _LOGGER.debug(
            "Occupancy sensor updating: occupied=%s, probability=%.3f",
            self.coordinator.occupied,
            self.coordinator.probability,
        )
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
        coordinator: AreaOccupancyCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        _LOGGER.debug(
            "Initializing WaspInBoxSensor for entry_id: %s", config_entry.entry_id
        )
        super().__init__()

        # Store references and configuration
        # Note: self.hass is automatically set by Home Assistant when entity is added
        self._coordinator = coordinator
        self._config = coordinator.config
        self._motion_timeout = self._config.wasp_in_box.motion_timeout
        self._weight = self._config.wasp_in_box.weight
        self._max_duration = self._config.wasp_in_box.max_duration
        self._verification_delay = self._config.wasp_in_box.verification_delay

        # Configure entity properties
        self._attr_has_entity_name = True
        self._attr_unique_id = (
            f"{config_entry.entry_id}_{NAME_WASP_IN_BOX.lower().replace(' ', '_')}"
        )
        self._attr_name = NAME_WASP_IN_BOX
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        self._attr_device_info = coordinator.device_info
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
            self._coordinator.wasp_entity_id = self.entity_id

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

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup when entity is removed."""
        _LOGGER.debug("Removing Wasp in Box sensor: %s", self.entity_id)

        # Clear the entity_id from coordinator and re-initialize entity manager
        if self._coordinator.wasp_entity_id == self.entity_id:
            self._coordinator.wasp_entity_id = None

        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        self._cancel_max_duration_timer()
        self._cancel_verification_timer()

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
        if not valid_entities:
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
        door_is_open = new_state == DOOR_OPEN
        door_was_closed = previous_door_state == DOOR_CLOSED

        if door_is_open and door_was_closed and self._state == STATE_ON:
            # Door opened while room was occupied - set to unoccupied
            _LOGGER.debug("Door opened while occupied - marking room as unoccupied")
            self._cancel_verification_timer()
            self._set_state(STATE_OFF)
        elif not door_is_open:  # Door closed (or closing)
            # Pattern A: Door closes with active motion
            if self._motion_state == STATE_ON:
                _LOGGER.debug(
                    "Door closed with motion detected - marking room as occupied"
                )
                self._set_state(STATE_ON)
            # Pattern B: Door closes with recent motion (within motion_timeout)
            elif self._last_motion_time:
                now = dt_util.utcnow()
                motion_age = (now - self._last_motion_time).total_seconds()
                if motion_age <= self._motion_timeout:
                    _LOGGER.debug(
                        "Door closed with recent motion (%.1fs ago) - marking room as occupied",
                        motion_age,
                    )
                    self._set_state(STATE_ON)
                else:
                    _LOGGER.debug(
                        "Door closed but motion is too old (%.1fs) - not marking as occupied",
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
        if new_state == STATE_ON and self._door_state == DOOR_CLOSED:
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


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: Any
) -> None:
    """Set up the Area Occupancy Detection binary sensors."""
    coordinator: AreaOccupancyCoordinator = config_entry.runtime_data

    # 1. Create the main sensor instance
    entities: list[BinarySensorEntity] = [
        Occupancy(coordinator=coordinator, entry_id=config_entry.entry_id)
    ]

    # 2. Create the Wasp in Box sensor if enabled
    if coordinator.config.wasp_in_box.enabled:
        _LOGGER.debug("Wasp in Box sensor enabled, creating sensor")
        wasp_sensor = WaspInBoxSensor(
            coordinator=coordinator, config_entry=config_entry
        )
        entities.append(wasp_sensor)
        _LOGGER.debug("Created Wasp in Box sensor: %s", wasp_sensor.unique_id)

    async_add_entities(entities, update_before_add=False)
