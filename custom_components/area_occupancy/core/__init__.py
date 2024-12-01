# custom_components/area_occupancy/core/__init__.py

"""Core components for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypeVar, Generic

from homeassistant.core import HomeAssistant


@dataclass(frozen=True)
class SensorState:
    """Immutable sensor state data."""

    entity_id: str
    state: str | float
    last_changed: datetime
    available: bool = True


@dataclass
class AreaConfig:
    """Area configuration with validation."""

    name: str
    motion_sensors: list[str]
    media_devices: list[str] = field(default_factory=list)
    appliances: list[str] = field(default_factory=list)
    illuminance_sensors: list[str] = field(default_factory=list)
    humidity_sensors: list[str] = field(default_factory=list)
    temperature_sensors: list[str] = field(default_factory=list)
    threshold: float = 0.5
    decay_enabled: bool = True
    decay_window: int = 600
    decay_type: str = "linear"
    history_period: int = 7

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.motion_sensors:
            raise ValueError("At least one motion sensor is required")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if self.decay_window < 60 or self.decay_window > 3600:
            raise ValueError("Decay window must be between 60 and 3600 seconds")
        if self.history_period < 1 or self.history_period > 30:
            raise ValueError("History period must be between 1 and 30 days")
        if self.decay_type not in ["linear", "exponential"]:
            raise ValueError("Decay type must be either 'linear' or 'exponential'")


@dataclass
class ProbabilityResult:
    """Result of probability calculations."""

    probability: float
    prior_probability: float
    active_triggers: list[str]
    sensor_probabilities: dict[str, float]
    decay_status: dict[str, float]
    confidence_score: float
    sensor_availability: dict[str, bool]
    device_states: dict[str, dict[str, str]]
    pattern_data: dict[str, Any] | None = None
    last_occupied: datetime | None = None
    state_duration: float | None = None
    occupancy_rate: float | None = None
    moving_average: float | None = None
    rate_of_change: float | None = None
    min_probability: float | None = None
    max_probability: float | None = None


T = TypeVar("T")


class StorageProvider(Protocol):
    """Protocol for storage providers."""

    async def async_load(self) -> dict[str, Any]:
        """Load stored data."""

    async def async_save(self, data: dict[str, Any]) -> None:
        """Save data to storage."""

    async def async_remove(self) -> None:
        """Remove stored data."""


class DataProvider(Protocol):
    """Protocol for data providers."""

    async def get_sensor_states(self) -> dict[str, SensorState]:
        """Get current states of all configured sensors."""

    async def get_historical_data(self) -> dict[str, Any]:
        """Get historical data for analysis."""
