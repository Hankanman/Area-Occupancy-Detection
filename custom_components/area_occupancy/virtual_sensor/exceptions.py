"""Exceptions for the virtual sensor framework."""

from __future__ import annotations

from typing import Any

from ..exceptions import AreaOccupancyError


class VirtualSensorError(AreaOccupancyError):
    """Base class for virtual sensor errors."""

    def __init__(self, message: str) -> None:
        """Initialize the error."""
        super().__init__(message)


class VirtualSensorConfigError(VirtualSensorError):
    """Error raised when a virtual sensor configuration is invalid."""

    def __init__(self, message: str, config: dict[str, Any]) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.config = config


class VirtualSensorTypeError(VirtualSensorError):
    """Error raised when a virtual sensor type is invalid."""

    def __init__(self, sensor_type: str) -> None:
        """Initialize the error."""
        super().__init__(f"Invalid virtual sensor type: {sensor_type}")
        self.sensor_type = sensor_type


class VirtualSensorStateError(VirtualSensorError):
    """Error raised when a virtual sensor state is invalid."""

    def __init__(self, message: str, state: str) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.state = state


class VirtualSensorUpdateError(VirtualSensorError):
    """Exception raised when sensor update fails."""

    def __init__(self, message: str, sensor_id: str) -> None:
        super().__init__(message)
        self.sensor_id = sensor_id


class VirtualSensorNotFoundError(VirtualSensorError):
    """Exception raised when sensor is not found."""

    def __init__(self, sensor_id: str) -> None:
        super().__init__(f"Virtual sensor not found: {sensor_id}")
        self.sensor_id = sensor_id


class VirtualSensorWeightError(VirtualSensorError):
    """Exception raised for invalid weight values."""

    def __init__(self, weight: float) -> None:
        super().__init__(f"Invalid weight value: {weight}")
        self.weight = weight


class VirtualSensorEventError(VirtualSensorError):
    """Error raised when a virtual sensor event is invalid."""

    def __init__(self, message: str, event_type: str) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.event_type = event_type
