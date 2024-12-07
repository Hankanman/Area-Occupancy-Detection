"""Configuration management for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.const import CONF_NAME
from homeassistant.exceptions import HomeAssistantError

from .const import (
    CONF_MOTION_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_APPLIANCES,
    CONF_ILLUMINANCE_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_HISTORY_PERIOD,
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_DECAY_TYPE,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_MINIMUM_CONFIDENCE,
    CONF_AREA_ID,
    DEFAULT_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_TYPE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_MINIMUM_CONFIDENCE,
)
from .types import CoreConfig, OptionsConfig

_LOGGER = logging.getLogger(__name__)


class ConfigManager:
    """Manage Area Occupancy configuration validation and migration."""

    @staticmethod
    def validate_core_config(data: dict[str, Any]) -> CoreConfig:
        """Validate core configuration data."""
        if not data.get(CONF_NAME):
            raise HomeAssistantError("Name is required")

        if not data.get(CONF_MOTION_SENSORS):
            raise HomeAssistantError("At least one motion sensor is required")

        if not data.get(CONF_AREA_ID):
            raise HomeAssistantError("Area ID is required")

        return CoreConfig(
            name=data[CONF_NAME],
            area_id=data[CONF_AREA_ID],
            motion_sensors=data[CONF_MOTION_SENSORS],
        )

    @staticmethod
    def validate_options(data: dict[str, Any]) -> OptionsConfig:
        """Validate options configuration data."""

        options: OptionsConfig = {
            CONF_MEDIA_DEVICES: data.get(CONF_MEDIA_DEVICES, []),
            CONF_APPLIANCES: data.get(CONF_APPLIANCES, []),
            CONF_ILLUMINANCE_SENSORS: data.get(CONF_ILLUMINANCE_SENSORS, []),
            CONF_HUMIDITY_SENSORS: data.get(CONF_HUMIDITY_SENSORS, []),
            CONF_TEMPERATURE_SENSORS: data.get(CONF_TEMPERATURE_SENSORS, []),
            CONF_THRESHOLD: data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
            CONF_HISTORY_PERIOD: data.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD),
            CONF_DECAY_ENABLED: data.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
            CONF_DECAY_WINDOW: data.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
            CONF_DECAY_TYPE: data.get(CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE),
            CONF_HISTORICAL_ANALYSIS_ENABLED: data.get(
                CONF_HISTORICAL_ANALYSIS_ENABLED, DEFAULT_HISTORICAL_ANALYSIS_ENABLED
            ),
            CONF_MINIMUM_CONFIDENCE: data.get(
                CONF_MINIMUM_CONFIDENCE, DEFAULT_MINIMUM_CONFIDENCE
            ),
        }

        ConfigManager._validate_numeric_bounds(options)
        return options

    @staticmethod
    def _validate_numeric_bounds(options: OptionsConfig) -> None:
        """Validate numeric values are within acceptable bounds."""
        if not 0 <= options[CONF_THRESHOLD] <= 1:
            raise HomeAssistantError("Threshold must be between 0 and 1")

        if not 1 <= options[CONF_HISTORY_PERIOD] <= 30:
            raise HomeAssistantError("History period must be between 1 and 30 days")

        if not 60 <= options[CONF_DECAY_WINDOW] <= 3600:
            raise HomeAssistantError("Decay window must be between 60 and 3600 seconds")

        if not 0 <= options[CONF_MINIMUM_CONFIDENCE] <= 1:
            raise HomeAssistantError("Minimum confidence must be between 0 and 1")

    @staticmethod
    def migrate_legacy_config(
        config: dict[str, Any]
    ) -> tuple[CoreConfig, OptionsConfig]:
        """Migrate legacy configuration to new format."""
        try:
            core_config = CoreConfig(
                name=config[CONF_NAME],
                area_id=config.get(CONF_AREA_ID),
                motion_sensors=config[CONF_MOTION_SENSORS],
            )

            options_data = {
                k: v
                for k, v in config.items()
                if k
                not in [
                    CONF_NAME,
                    CONF_MOTION_SENSORS,
                    CONF_AREA_ID,
                ]
            }
            options_config = ConfigManager.validate_options(options_data)

            return core_config, options_config

        except Exception as err:
            _LOGGER.error("Error migrating config: %s", err)
            raise HomeAssistantError(f"Failed to migrate configuration: {err}") from err

    @staticmethod
    def merge_options(current: OptionsConfig, new: dict[str, Any]) -> OptionsConfig:
        """Merge new options with current config, validating result."""
        merged = {**current, **new}
        return ConfigManager.validate_options(merged)
