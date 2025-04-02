"""Custom exceptions for Area Occupancy Detection."""

from homeassistant.exceptions import HomeAssistantError


class AreaOccupancyError(HomeAssistantError):
    """Base exception for Area Occupancy Detection errors."""


class ConfigurationError(AreaOccupancyError):
    """Raised when there is an error in the configuration."""


class StateError(AreaOccupancyError):
    """Raised when there is an error with sensor states."""


class CalculationError(AreaOccupancyError):
    """Raised when there is an error in probability calculations."""


class StorageError(AreaOccupancyError):
    """Raised when there is an error with storage operations."""


class PriorCalculationError(HomeAssistantError):
    """Error when prior probability calculation fails."""
