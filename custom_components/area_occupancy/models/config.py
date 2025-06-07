"""Configuration model and manager for Area Occupancy Detection."""

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import HomeAssistantError

from ..const import (
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_DECAY_ENABLED,
    CONF_DECAY_MIN_DELAY,
    CONF_DECAY_WINDOW,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HISTORICAL_ANALYSIS_ENABLED,
    CONF_HISTORY_PERIOD,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WASP_MAX_DURATION,
    CONF_WASP_MOTION_TIMEOUT,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_MIN_DELAY,
    DEFAULT_DECAY_WINDOW,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
    DEFAULT_HISTORY_PERIOD,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_LIGHT,
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
    lights: list[str] = field(default_factory=list)
    illuminance: list[str] = field(default_factory=list)
    humidity: list[str] = field(default_factory=list)
    temperature: list[str] = field(default_factory=list)
    doors: list[str] = field(default_factory=list)
    windows: list[str] = field(default_factory=list)


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
    light: float = DEFAULT_WEIGHT_LIGHT
    environmental: float = DEFAULT_WEIGHT_ENVIRONMENTAL
    wasp: float = DEFAULT_WASP_WEIGHT


@dataclass
class Decay:
    """Decay configuration."""

    enabled: bool = DEFAULT_DECAY_ENABLED
    window: int = DEFAULT_DECAY_WINDOW
    min_delay: int = DEFAULT_DECAY_MIN_DELAY
    history_period: int = DEFAULT_HISTORY_PERIOD
    historical_analysis_enabled: bool = DEFAULT_HISTORICAL_ANALYSIS_ENABLED


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
    area_id: str | None = None
    threshold: float = DEFAULT_THRESHOLD
    sensors: Sensors = field(default_factory=Sensors)
    sensor_states: SensorStates = field(default_factory=SensorStates)
    weights: Weights = field(default_factory=Weights)
    decay: Decay = field(default_factory=Decay)
    wasp_in_box: WaspInBox = field(default_factory=WaspInBox)
    _raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create a config from a dictionary."""
        return cls(
            name=data.get(CONF_NAME, "Area Occupancy"),
            area_id=data.get(CONF_AREA_ID),
            threshold=float(data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)),
            sensors=Sensors(
                motion=data.get(CONF_MOTION_SENSORS, []),
                primary_occupancy=data.get(CONF_PRIMARY_OCCUPANCY_SENSOR),
                media=data.get(CONF_MEDIA_DEVICES, []),
                appliances=data.get(CONF_APPLIANCES, []),
                lights=data.get(CONF_LIGHTS, []),
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
                motion=float(data.get(CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION)),
                media=float(data.get(CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA)),
                appliance=float(
                    data.get(CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE)
                ),
                door=float(data.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR)),
                window=float(data.get(CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW)),
                light=float(data.get(CONF_WEIGHT_LIGHT, DEFAULT_WEIGHT_LIGHT)),
                environmental=float(
                    data.get(CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL)
                ),
                wasp=float(data.get(CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT)),
            ),
            decay=Decay(
                enabled=bool(data.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)),
                window=int(data.get(CONF_DECAY_WINDOW, DEFAULT_DECAY_WINDOW)),
                min_delay=int(data.get(CONF_DECAY_MIN_DELAY, DEFAULT_DECAY_MIN_DELAY)),
                history_period=int(
                    data.get(CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD)
                ),
                historical_analysis_enabled=bool(
                    data.get(
                        CONF_HISTORICAL_ANALYSIS_ENABLED,
                        DEFAULT_HISTORICAL_ANALYSIS_ENABLED,
                    )
                ),
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

    def __init__(self, coordinator: "AreaOccupancyCoordinator"):
        """Initialize the config manager."""
        if coordinator.config_entry is None:
            raise ValueError("Config entry is required")
        self.config_entry = coordinator.config_entry
        self._config = Config.from_dict(self._merge_entry(coordinator.config_entry))
        self._hass = coordinator.hass

        _LOGGER.debug("ConfigManager initialized with config: %s", self._config)

    @property
    def hass(self):
        """Get the Home Assistant instance."""
        if self._hass is None:
            raise RuntimeError("Home Assistant instance not set")
        return self._hass

    def set_hass(self, hass):
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

    async def update_threshold(self, value: float) -> None:
        """Update the threshold value and sync with Home Assistant config entry.

        Args:
            value: New threshold value (0-100)

        Raises:
            ValueError: If the value is invalid
            HomeAssistantError: If updating the config entry fails

        """
        if not isinstance(value, (int, float)) or not 0 <= value <= 100:
            raise ValueError("Threshold must be a number between 0 and 100")

        try:
            # Create a new options dict with the updated threshold
            new_options = dict(self.config_entry.options)
            new_options[CONF_THRESHOLD] = value

            # Update the config entry
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                options=new_options,
            )

            # Update the Config object
            self._config.threshold = value
            self._config._raw[CONF_THRESHOLD] = value

        except Exception as err:
            raise HomeAssistantError(f"Failed to update threshold: {err}") from err

    def update_config(self, options: dict[str, Any]) -> None:
        """Update configuration from options."""
        self._config = Config.from_dict(self._merge_entry(self.config_entry))
        self._config._raw.update(options)
