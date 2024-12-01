# custom_components/area_occupancy/core/calculators/base.py

"""Base calculator protocol for Area Occupancy Detection."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from .. import SensorState

T = TypeVar("T")


class Calculator(Generic[T], ABC):
    """Base calculator protocol."""

    @abstractmethod
    def calculate(self, sensor_states: dict[str, SensorState]) -> T:
        """Calculate based on current sensor states."""


class AsyncCalculator(Generic[T], ABC):
    """Base async calculator protocol."""

    @abstractmethod
    async def async_calculate(self, sensor_states: dict[str, SensorState]) -> T:
        """Calculate based on current sensor states asynchronously."""


class BaseCalculator(ABC):
    """Base calculator implementation."""

    def __init__(self) -> None:
        """Initialize the calculator."""
        self._cache: dict[str, Any] = {}

    def _clear_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()

    def _get_cache_key(self, sensor_states: dict[str, SensorState]) -> str:
        """Generate a cache key from sensor states."""
        return "|".join(
            f"{entity_id}:{state.state}:{state.last_changed}"
            for entity_id, state in sorted(sensor_states.items())
        )

    def _validate_inputs(self, sensor_states: dict[str, SensorState]) -> bool:
        """Validate input sensor states."""
        if not sensor_states:
            return False

        return all(
            isinstance(state, SensorState) and state.entity_id and state.state
            for state in sensor_states.values()
        )
