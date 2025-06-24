"""Configuration model and manager for Area Occupancy Detection."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_HISTORY_PERIOD,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WASP_MAX_DURATION,
    CONF_WASP_MOTION_TIMEOUT,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_PURPOSE,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
)

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass
class Sensors:
    """Sensors configuration."""

    motion: list[str] = field(default_factory=list)
    primary_occupancy: str | None = None
    media: list[str] = field(default_factory=list)
    appliances: list[str] = field(default_factory=list)
    illuminance: list[str] = field(default_factory=list)
    humidity: list[str] = field(default_factory=list)
    temperature: list[str] = field(default_factory=list)
    doors: list[str] = field(default_factory=list)
    windows: list[str] = field(default_factory=list)

    def get_motion_sensors(self, coordinator: "AreaOccupancyCoordinator") -> list[str]:
        """Get motion sensors including wasp sensor if enabled and available.

        Args:
            coordinator: The coordinator instance to get wasp entity_id from

        Returns:
            list[str]: List of motion sensor entity_ids including wasp if applicable

        """

        motion_sensors = self.motion.copy()

        # Add wasp sensor if enabled and entity_id is available
        if (
            coordinator
            and coordinator.config.wasp_in_box.enabled
            and coordinator.wasp_entity_id
        ):
            motion_sensors.append(coordinator.wasp_entity_id)
            _LOGGER.debug(
                "Adding wasp sensor %s to motion sensors list",
                coordinator.wasp_entity_id,
            )

        return motion_sensors


@dataclass
class SensorStates:
    """Sensor states configuration."""

    door: list[str] = field(default_factory=lambda: [DEFAULT_DOOR_ACTIVE_STATE])
    window: list[str] = field(default_factory=lambda: [DEFAULT_WINDOW_ACTIVE_STATE])
    appliance: list[str] = field(
        default_factory=lambda: list(DEFAULT_APPLIANCE_ACTIVE_STATES)
    )
    media: list[str] = field(default_factory=lambda: list(DEFAULT_MEDIA_ACTIVE_STATES))


@dataclass
class Weights:
    """Weights configuration."""

    motion: float = DEFAULT_WEIGHT_MOTION
    media: float = DEFAULT_WEIGHT_MEDIA
    appliance: float = DEFAULT_WEIGHT_APPLIANCE
    door: float = DEFAULT_WEIGHT_DOOR
    window: float = DEFAULT_WEIGHT_WINDOW
    environmental: float = DEFAULT_WEIGHT_ENVIRONMENTAL
    wasp: float = DEFAULT_WASP_WEIGHT


@dataclass
class Decay:
    """Decay configuration."""

    enabled: bool = DEFAULT_DECAY_ENABLED
    half_life: int = DEFAULT_DECAY_HALF_LIFE


@dataclass
class History:
    """History configuration."""

    enabled: bool = DEFAULT_HISTORICAL_ANALYSIS_ENABLED
    period: int = DEFAULT_HISTORY_PERIOD


@dataclass
class WaspInBox:
    """Wasp in box configuration."""

    enabled: bool = False
    motion_timeout: int = DEFAULT_WASP_MOTION_TIMEOUT
    weight: float = DEFAULT_WASP_WEIGHT
    max_duration: int = DEFAULT_WASP_MAX_DURATION


@dataclass
class Config:
    """Configuration for Area Occupancy Detection."""

    name: str = "Area Occupancy"
    purpose: str = DEFAULT_PURPOSE
    area_id: str | None = None
    threshold: float = DEFAULT_THRESHOLD
    sensors: Sensors = field(default_factory=Sensors)
    sensor_states: SensorStates = field(default_factory=SensorStates)
    weights: Weights = field(default_factory=Weights)
    decay: Decay = field(default_factory=Decay)
    history: History = field(default_factory=History)
    wasp_in_box: WaspInBox = field(default_factory=WaspInBox)
    _raw: dict = field(default_factory=dict, repr=False)

    @property
    def start_time(self) -> datetime:
        """Return the start time of the history period."""
        return dt_util.utcnow() - timedelta(days=self.history.period)

    @property
    def end_time(self) -> datetime:
        """Return the end time of the history period."""
        return dt_util.utcnow()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create a config from a dictionary with validation."""
        # Validate threshold range
        threshold = float(data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)) / 100.0

        # Validate weights are positive
        weights_data = {}
        for weight_key, default_val in [
            (CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION),
            (CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA),
            (CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE),
            (CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR),
            (CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW),
            (CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL),
            (CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT),
        ]:
            weight_val = float(data.get(weight_key, default_val))
            if weight_val < 0:
                _LOGGER.warning(
                    "Invalid weight %s=%s, using default %s",
                    weight_key,
                    weight_val,
                    default_val,
                )
                weight_val = default_val
            weights_data[weight_key] = weight_val

        return cls(
            name=data.get(CONF_NAME, "Area Occupancy"),
            purpose=data.get(CONF_PURPOSE, DEFAULT_PURPOSE),
            area_id=data.get(CONF_AREA_ID),
            threshold=threshold,
            sensors=Sensors(
                motion=data.get(CONF_MOTION_SENSORS, []),
                primary_occupancy=data.get(CONF_PRIMARY_OCCUPANCY_SENSOR),
                media=data.get(CONF_MEDIA_DEVICES, []),
                appliances=data.get(CONF_APPLIANCES, []),
                illuminance=data.get(CONF_ILLUMINANCE_SENSORS, []),
                humidity=data.get(CONF_HUMIDITY_SENSORS, []),
                temperature=data.get(CONF_TEMPERATURE_SENSORS, []),
                doors=data.get(CONF_DOOR_SENSORS, []),
                windows=data.get(CONF_WINDOW_SENSORS, []),
            ),
            sensor_states=SensorStates(
                door=[data.get(CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE)],
                window=[
                    data.get(CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE)
                ],
                appliance=data.get(
                    CONF_APPLIANCE_ACTIVE_STATES, list(DEFAULT_APPLIANCE_ACTIVE_STATES)
                ),
                media=data.get(
                    CONF_MEDIA_ACTIVE_STATES, list(DEFAULT_MEDIA_ACTIVE_STATES)
                ),
            ),
            weights=Weights(
                motion=weights_data[CONF_WEIGHT_MOTION],
                media=weights_data[CONF_WEIGHT_MEDIA],
                appliance=weights_data[CONF_WEIGHT_APPLIANCE],
                door=weights_data[CONF_WEIGHT_DOOR],
                window=weights_data[CONF_WEIGHT_WINDOW],
                environmental=weights_data[CONF_WEIGHT_ENVIRONMENTAL],
                wasp=weights_data[CONF_WASP_WEIGHT],
            ),
            decay=Decay(
                enabled=bool(data.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)),
                half_life=int(data.get(CONF_DECAY_HALF_LIFE, DEFAULT_DECAY_HALF_LIFE)),
            ),
            history=History(
                enabled=bool(
                    data.get(
                        CONF_HISTORICAL_ANALYSIS_ENABLED,
                        DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                    )
                ),
                period=int(data.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD)),
            ),
            wasp_in_box=WaspInBox(
                enabled=bool(data.get(CONF_WASP_ENABLED, False)),
                motion_timeout=int(
                    data.get(CONF_WASP_MOTION_TIMEOUT, DEFAULT_WASP_MOTION_TIMEOUT)
                ),
                weight=float(data.get(CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT)),
                max_duration=int(
                    data.get(CONF_WASP_MAX_DURATION, DEFAULT_WASP_MAX_DURATION)
                ),
            ),
            _raw=data.copy(),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the config as a dictionary."""
        return self._raw.copy()


class ConfigManager:
    """Manages configuration for Area Occupancy Detection."""

    def __init__(self, coordinator: "AreaOccupancyCoordinator") -> None:
        """Initialize the config manager."""
        if coordinator.config_entry is None:
            raise ValueError("Config entry is required")
        self.coordinator = coordinator
        self.config_entry = coordinator.config_entry
        self._config = Config.from_dict(self._merge_entry(coordinator.config_entry))
        self._hass = coordinator.hass

        _LOGGER.debug("ConfigManager initialized with config: %s", self._config)

    @property
    def hass(self) -> HomeAssistant:
        """Get the Home Assistant instance."""
        if self._hass is None:
            raise RuntimeError("Home Assistant instance not set")
        return self._hass

    def set_hass(self, hass: HomeAssistant) -> None:
        """Set the Home Assistant instance."""
        self._hass = hass

    @staticmethod
    def _merge_entry(config_entry: ConfigEntry) -> dict[str, Any]:
        """Merge the config entry data and options."""
        merged = dict(config_entry.data)
        merged.update(config_entry.options)
        return merged

    @property
    def config(self) -> Config:
        """Get the config."""
        return self._config

    def update_from_entry(self, config_entry: ConfigEntry) -> None:
        """Update the config from a new config entry."""
        self.config_entry = config_entry
        self._config = Config.from_dict(self._merge_entry(config_entry))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key."""
        return getattr(self._config, key, default)

    async def update_config(self, options: dict[str, Any]) -> None:
        """Update configuration and persist to Home Assistant config entry.

        Args:
            options: Dictionary of configuration options to update

        Raises:
            ValueError: If any option values are invalid
            HomeAssistantError: If updating the config entry fails

        """
        try:
            # Create new options dict by merging existing with new options
            new_options = dict(self.config_entry.options)
            new_options.update(options)

            # Update the config entry in Home Assistant
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                options=new_options,
            )

            # Merge existing config entry with new options for internal state
            merged_data = self._merge_entry(self.config_entry)
            merged_data.update(options)

            # Create new config object with validation
            self._config = Config.from_dict(merged_data)

            # Request update since threshold affects occupied calculation
            await self.coordinator.async_request_refresh()

        except Exception as err:
            raise HomeAssistantError(f"Failed to update configuration: {err}") from err
