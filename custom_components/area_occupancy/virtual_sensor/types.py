"""Types for the virtual sensor framework."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class VirtualSensorStateEnum(Enum):
    """Enum for virtual sensor states."""

    UNKNOWN = "unknown"
    OCCUPIED = "occupied"
    UNOCCUPIED = "unoccupied"
    ERROR = "error"

    def __str__(self) -> str:
        """Return the string representation of the enum value."""
        return self.value


class VirtualSensorState(TypedDict, total=False):
    """State data for a virtual sensor."""

    state: str
    attributes: Dict[str, Any]
    last_updated: Optional[str]


class VirtualSensorConfig(TypedDict, total=False):
    """Configuration for a virtual sensor or list of sensors."""

    # Single sensor config
    type: str
    name: str
    update_interval: int
    weight: float
    enabled: bool
    options: Dict[str, Any]
    door_entity_id: Optional[str]
    motion_entity_id: Optional[str]
    motion_timeout: Optional[int]

    # List config
    virtual_sensors: List["VirtualSensorConfig"]


class VirtualSensorUpdateResult(TypedDict):
    """Result of a virtual sensor update."""

    success: bool
    state: Optional[VirtualSensorState]
    error: Optional[str]


class VirtualSensorEvent(TypedDict):
    """Event for a virtual sensor."""

    event_type: str
    data: Dict[str, Any]
    timestamp: str
