"""Types for the virtual sensor framework."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, TypedDict


class VirtualSensorStateEnum(Enum):
    """Enum for virtual sensor states."""

    UNKNOWN = "unknown"
    OCCUPIED = "occupied"
    UNOCCUPIED = "unoccupied"
    ERROR = "error"

    def __str__(self) -> str:
        """Return the string representation of the enum value."""
        return self.value


class VirtualSensorConfig(TypedDict, total=False):
    """Configuration for a virtual sensor."""

    type: str
    name: str
    update_interval: int
    weight: float
    enabled: bool
    options: Dict[str, Any]


class VirtualSensorStateData(TypedDict, total=False):
    """State data for a virtual sensor."""

    state: str
    attributes: Dict[str, Any]
    last_updated: str
    last_changed: str


class VirtualSensorEventData(TypedDict, total=False):
    """Event data for virtual sensor events."""

    entity_id: str
    old_state: Optional[VirtualSensorStateData]
    new_state: Optional[VirtualSensorStateData]


class VirtualSensorOptions(TypedDict, total=False):
    """Options for configuring a virtual sensor."""

    name: str
    update_interval: int
    weight: float
    enabled: bool
    type_specific: Dict[str, Any]


class VirtualSensorConfigEntry(TypedDict):
    """Configuration entry for a virtual sensor."""

    entry_id: str
    data: VirtualSensorConfig
    options: VirtualSensorOptions
    version: int
    domain: str
    title: str
    source: str
    system_options: Dict[str, Any]
    connection_class: str
    unique_id: Optional[str]
    disabled_by: Optional[str]


class VirtualSensorState(TypedDict):
    """State for a virtual sensor."""

    state: str
    attributes: Dict[str, Any]
    last_updated: str


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
