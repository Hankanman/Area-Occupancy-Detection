"""Integration-level configuration for Area Occupancy Detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry

from ..const import ANALYSIS_INTERVAL, DECAY_INTERVAL, SAVE_DEBOUNCE_SECONDS

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator


class IntegrationConfig:
    """Integration-level configuration for Area Occupancy Detection.

    This class manages global settings that apply to the entire integration,
    such as coordinator timing intervals, database behavior, and future
    cross-area coordination features.

    This is separate from AreaConfig, which handles per-area occupancy
    detection settings like sensors, weights, and thresholds.
    """

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the integration configuration.

        Args:
            coordinator: The coordinator instance
            config_entry: The Home Assistant config entry
        """
        self.coordinator = coordinator
        self.config_entry = config_entry
        self.hass = coordinator.hass

        # Integration identification
        self.integration_name = config_entry.title

        # Timing and performance settings
        self.analysis_interval = ANALYSIS_INTERVAL
        self.save_debounce = SAVE_DEBOUNCE_SECONDS
        self.decay_interval = DECAY_INTERVAL

        # Database and storage settings
        # These could be made configurable in the future if needed
        # self.database_retention_days = RETENTION_DAYS
        # self.enable_backups = True

        # Future: Cross-area coordination settings
        # self.person_tracking_enabled = False
        # self.area_transition_detection = False
        # self.global_occupancy_threshold = 0.5

    def __repr__(self) -> str:
        """Return a string representation of the integration config."""
        return f"IntegrationConfig(name={self.integration_name!r})"
