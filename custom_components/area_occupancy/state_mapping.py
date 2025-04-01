"""State definitions for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypedDict

from homeassistant.const import (
    STATE_CLOSED,
    STATE_IDLE,
    STATE_OFF,
    STATE_ON,
    STATE_OPEN,
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STANDBY,
)


@dataclass
class StateOption:
    """Represents a state option with its value and friendly name."""

    value: str
    name: str
    icon: str | None = None


class PlatformStates(TypedDict):
    """Type for platform states configuration."""

    options: list[StateOption]
    default: str


# Door states configuration
DOOR_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_OPEN, "Open", "mdi:door-open"),
        StateOption(STATE_CLOSED, "Closed", "mdi:door"),
    ],
    "default": STATE_CLOSED,
}

# Window states configuration
WINDOW_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_OPEN, "Open", "mdi:window-open"),
        StateOption(STATE_CLOSED, "Closed", "mdi:window-closed"),
    ],
    "default": STATE_CLOSED,
}

# Media player states configuration
MEDIA_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_PLAYING, "Playing", "mdi:play"),
        StateOption(STATE_PAUSED, "Paused", "mdi:pause"),
        StateOption(STATE_IDLE, "Idle", "mdi:sleep"),
        StateOption(STATE_OFF, "Off", "mdi:power"),
    ],
    "default": STATE_PLAYING,
}

# Appliance states configuration
APPLIANCE_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_ON, "On", "mdi:power"),
        StateOption(STATE_OFF, "Off", "mdi:power-off"),
        StateOption(STATE_STANDBY, "Standby", "mdi:power-sleep"),
    ],
    "default": STATE_ON,
}

# Light states configuration
LIGHT_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_ON, "On", "mdi:lightbulb"),
        StateOption(STATE_OFF, "Off", "mdi:lightbulb-off"),
    ],
    "default": STATE_ON,
}

# Motion sensor states configuration
MOTION_STATES: Final[PlatformStates] = {
    "options": [
        StateOption(STATE_ON, "Active", "mdi:motion-sensor"),
        StateOption(STATE_OFF, "Inactive", "mdi:motion-sensor-off"),
    ],
    "default": STATE_ON,
}


def get_state_options(platform_type: str) -> PlatformStates:
    """Get state options for a given platform type."""
    platform_map = {
        "door": DOOR_STATES,
        "window": WINDOW_STATES,
        "media": MEDIA_STATES,
        "appliance": APPLIANCE_STATES,
        "light": LIGHT_STATES,
        "motion": MOTION_STATES,
    }
    return platform_map.get(platform_type, MOTION_STATES)


def get_friendly_state_name(platform_type: str, state: str) -> str:
    """Get the friendly name for a state in a given platform type."""
    states = get_state_options(platform_type)
    for option in states["options"]:
        if option.value == state:
            return option.name
    return state


def get_state_icon(platform_type: str, state: str) -> str | None:
    """Get the icon for a state in a given platform type."""
    states = get_state_options(platform_type)
    for option in states["options"]:
        if option.value == state:
            return option.icon
    return None


def get_default_state(platform_type: str) -> str:
    """Get the default state for a given platform type."""
    states = get_state_options(platform_type)
    return states["default"]
