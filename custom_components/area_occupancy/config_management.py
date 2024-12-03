"""Configuration management for Area Occupancy Detection."""

from __future__ import annotations

import logging
from typing import Any, TypedDict
import voluptuous as vol

from homeassistant.const import CONF_NAME
from homeassistant.helpers import selector
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.media_player import DOMAIN as MEDIA_PLAYER_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN

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
    DEFAULT_THRESHOLD,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DECAY_TYPE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_MINIMUM_CONFIDENCE,
)

_LOGGER = logging.getLogger(__name__)


class CoreConfig(TypedDict):
    """Core configuration that cannot be changed after setup."""

    name: str
    motion_sensors: list[str]


class OptionsConfig(TypedDict, total=False):
    """Optional configuration that can be updated."""

    motion_sensors: list[str]
    media_devices: list[str]
    appliances: list[str]
    illuminance_sensors: list[str]
    humidity_sensors: list[str]
    temperature_sensors: list[str]
    threshold: float
    history_period: int
    decay_enabled: bool
    decay_window: int
    decay_type: str
    historical_analysis_enabled: bool
    minimum_confidence: float


class ConfigManager:
    """Manage Area Occupancy configuration."""

    @staticmethod
    def get_core_schema(data: dict[str, Any] | None = None) -> vol.Schema:
        """Get schema for core configuration."""
        data = data or {}
        return vol.Schema(
            {
                vol.Required(CONF_NAME, default=data.get(CONF_NAME, "")): str,
                vol.Required(
                    CONF_MOTION_SENSORS, default=data.get(CONF_MOTION_SENSORS, [])
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=BINARY_SENSOR_DOMAIN,
                        device_class="motion",
                        multiple=True,
                    ),
                ),
            }
        )

    @staticmethod
    def get_options_schema(options: dict[str, Any] | None = None) -> vol.Schema:
        """Get schema for options configuration."""
        options = options or {}
        return vol.Schema(
            {
                vol.Optional(
                    CONF_MEDIA_DEVICES, default=options.get(CONF_MEDIA_DEVICES, [])
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=[MEDIA_PLAYER_DOMAIN],
                        multiple=True,
                    ),
                ),
                vol.Optional(
                    CONF_APPLIANCES, default=options.get(CONF_APPLIANCES, [])
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=[BINARY_SENSOR_DOMAIN, SWITCH_DOMAIN],
                        device_class=["power", "plug", "outlet"],
                        multiple=True,
                    ),
                ),
                vol.Optional(
                    CONF_ILLUMINANCE_SENSORS,
                    default=options.get(CONF_ILLUMINANCE_SENSORS, []),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=SENSOR_DOMAIN,
                        device_class="illuminance",
                        multiple=True,
                    ),
                ),
                vol.Optional(
                    CONF_HUMIDITY_SENSORS,
                    default=options.get(CONF_HUMIDITY_SENSORS, []),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=SENSOR_DOMAIN,
                        device_class="humidity",
                        multiple=True,
                    ),
                ),
                vol.Optional(
                    CONF_TEMPERATURE_SENSORS,
                    default=options.get(CONF_TEMPERATURE_SENSORS, []),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=SENSOR_DOMAIN,
                        device_class="temperature",
                        multiple=True,
                    ),
                ),
                vol.Optional(
                    CONF_THRESHOLD,
                    default=options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        mode="slider",
                    ),
                ),
                vol.Optional(
                    CONF_HISTORY_PERIOD,
                    default=options.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        max=30,
                        step=1,
                        mode="slider",
                        unit_of_measurement="days",
                    ),
                ),
                vol.Optional(
                    CONF_HISTORICAL_ANALYSIS_ENABLED,
                    default=options.get(
                        CONF_HISTORICAL_ANALYSIS_ENABLED,
                        DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                    ),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_MINIMUM_CONFIDENCE,
                    default=options.get(
                        CONF_MINIMUM_CONFIDENCE, DEFAULT_MINIMUM_CONFIDENCE
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        mode="slider",
                    ),
                ),
                vol.Optional(
                    CONF_DECAY_ENABLED,
                    default=options.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_DECAY_WINDOW,
                    default=options.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=60,
                        max=3600,
                        step=60,
                        mode="slider",
                        unit_of_measurement="seconds",
                    ),
                ),
                vol.Optional(
                    CONF_DECAY_TYPE,
                    default=options.get(CONF_DECAY_TYPE, DEFAULT_DECAY_TYPE),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["linear", "exponential"],
                        mode="dropdown",
                        translation_key="decay_type",
                    ),
                ),
            }
        )

    @staticmethod
    def validate_core_config(data: dict[str, Any]) -> CoreConfig:
        """Validate core configuration data."""
        if not data.get(CONF_MOTION_SENSORS):
            raise vol.Invalid("At least one motion sensor is required")

        schema = ConfigManager.get_core_schema()
        validated = schema(data)

        return CoreConfig(
            name=validated[CONF_NAME],
            motion_sensors=validated[CONF_MOTION_SENSORS],
        )

    @staticmethod
    def validate_options(data: dict[str, Any]) -> OptionsConfig:
        """Validate options configuration data."""
        schema = ConfigManager.get_options_schema()
        return schema(data)

    @staticmethod
    def migrate_legacy_config(
        config: dict[str, Any]
    ) -> tuple[CoreConfig, OptionsConfig]:
        """Migrate legacy configuration to new format."""
        core_config: dict[str, Any] = {
            CONF_NAME: config[CONF_NAME],
            CONF_MOTION_SENSORS: config[CONF_MOTION_SENSORS],
        }

        options_config: dict[str, Any] = {
            k: v for k, v in config.items() if k not in [CONF_NAME, CONF_MOTION_SENSORS]
        }

        return (
            ConfigManager.validate_core_config(core_config),
            ConfigManager.validate_options(options_config),
        )
