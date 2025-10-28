"""Configuration model and manager for Area Occupancy Detection."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_ON
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
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_MOTION_TIMEOUT,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WASP_MAX_DURATION,
    CONF_WASP_MOTION_TIMEOUT,
    CONF_WASP_VERIFICATION_DELAY,
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
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_MOTION_TIMEOUT,
    DEFAULT_PURPOSE,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_VERIFICATION_DELAY,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    HA_RECORDER_DAYS,
)

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator


_LOGGER = logging.getLogger(__name__)


@dataclass
class Sensors:
    """Sensors configuration."""

    motion: list[str] = field(default_factory=list)
    motion_timeout: int = DEFAULT_MOTION_TIMEOUT
    primary_occupancy: str | None = None
    media: list[str] = field(default_factory=list)
    appliance: list[str] = field(default_factory=list)
    illuminance: list[str] = field(default_factory=list)
    humidity: list[str] = field(default_factory=list)
    temperature: list[str] = field(default_factory=list)
    door: list[str] = field(default_factory=list)
    window: list[str] = field(default_factory=list)

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
            and getattr(coordinator, "wasp_entity_id", None)
        ):
            wasp_id = coordinator.wasp_entity_id
            if wasp_id is not None:
                motion_sensors.append(wasp_id)
                _LOGGER.debug(
                    "Adding wasp sensor %s to motion sensors list",
                    wasp_id,
                )

        return motion_sensors


@dataclass
class SensorStates:
    """Sensor states configuration."""

    motion: list[str] = field(default_factory=lambda: [STATE_ON])
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
class WaspInBox:
    """Wasp in box configuration."""

    enabled: bool = False
    motion_timeout: int = DEFAULT_WASP_MOTION_TIMEOUT
    weight: float = DEFAULT_WASP_WEIGHT
    max_duration: int = DEFAULT_WASP_MAX_DURATION
    verification_delay: int = DEFAULT_WASP_VERIFICATION_DELAY


class Config:
    """Configuration for Area Occupancy Detection."""

    def __init__(self, coordinator: "AreaOccupancyCoordinator"):
        """Initialize the config from a coordinator."""
        self.coordinator = coordinator
        self.config_entry = coordinator.config_entry
        self.hass = coordinator.hass
        self.db = coordinator.db

        # Load configuration from the merged entry data
        if coordinator.config_entry is None:
            raise ValueError("Coordinator config_entry cannot be None")
        self._load_config(self._merge_entry(coordinator.config_entry))

    def _load_config(self, data: dict[str, Any]) -> None:
        """Load configuration from merged data.

        Args:
            data: Dictionary containing merged config entry data and options

        """
        # Validate threshold range
        threshold = float(data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)) / 100.0

        # Set all configuration attributes
        self.name = data.get(CONF_NAME, "Area Occupancy")
        self.purpose = data.get(CONF_PURPOSE, DEFAULT_PURPOSE)
        self.area_id = data.get(CONF_AREA_ID)
        self.threshold = threshold

        self.sensors = Sensors(
            motion=data.get(CONF_MOTION_SENSORS, []),
            motion_timeout=int(data.get(CONF_MOTION_TIMEOUT, DEFAULT_MOTION_TIMEOUT)),
            primary_occupancy=data.get(CONF_PRIMARY_OCCUPANCY_SENSOR),
            media=data.get(CONF_MEDIA_DEVICES, []),
            appliance=data.get(CONF_APPLIANCES, []),
            illuminance=data.get(CONF_ILLUMINANCE_SENSORS, []),
            humidity=data.get(CONF_HUMIDITY_SENSORS, []),
            temperature=data.get(CONF_TEMPERATURE_SENSORS, []),
            door=data.get(CONF_DOOR_SENSORS, []),
            window=data.get(CONF_WINDOW_SENSORS, []),
        )

        self.sensor_states = SensorStates(
            motion=[STATE_ON],  # Motion sensors default to STATE_ON
            door=[data.get(CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE)],
            window=[data.get(CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE)],
            appliance=data.get(
                CONF_APPLIANCE_ACTIVE_STATES, list(DEFAULT_APPLIANCE_ACTIVE_STATES)
            ),
            media=data.get(CONF_MEDIA_ACTIVE_STATES, list(DEFAULT_MEDIA_ACTIVE_STATES)),
        )

        self.weights = Weights(
            motion=data[CONF_WEIGHT_MOTION],
            media=data[CONF_WEIGHT_MEDIA],
            appliance=data[CONF_WEIGHT_APPLIANCE],
            door=data[CONF_WEIGHT_DOOR],
            window=data[CONF_WEIGHT_WINDOW],
            environmental=data[CONF_WEIGHT_ENVIRONMENTAL],
            wasp=data[CONF_WASP_WEIGHT],
        )

        self.decay = Decay(
            enabled=bool(data.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)),
            half_life=int(data.get(CONF_DECAY_HALF_LIFE, DEFAULT_DECAY_HALF_LIFE)),
        )

        self.wasp_in_box = WaspInBox(
            enabled=bool(data.get(CONF_WASP_ENABLED, False)),
            motion_timeout=int(
                data.get(CONF_WASP_MOTION_TIMEOUT, DEFAULT_WASP_MOTION_TIMEOUT)
            ),
            weight=float(data.get(CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT)),
            max_duration=int(
                data.get(CONF_WASP_MAX_DURATION, DEFAULT_WASP_MAX_DURATION)
            ),
            verification_delay=int(
                data.get(CONF_WASP_VERIFICATION_DELAY, DEFAULT_WASP_VERIFICATION_DELAY)
            ),
        )

    @property
    def start_time(self) -> datetime:
        """Return the start time of the history period (always 10 days ago)."""
        return dt_util.utcnow() - timedelta(days=HA_RECORDER_DAYS)

    @property
    def end_time(self) -> datetime:
        """Return the end time of the history period (now)."""
        return dt_util.utcnow()

    @property
    def entity_ids(self) -> list[str]:
        """Return the entity ids of the sensors."""
        return [
            *self.sensors.motion,
            *self.sensors.media,
            *self.sensors.appliance,
            *self.sensors.door,
            *self.sensors.window,
            *self.sensors.illuminance,
            *self.sensors.humidity,
            *self.sensors.temperature,
        ]

    def validate_entity_configuration(self) -> list[str]:
        """Validate entity configuration and return any issues found.

        Returns:
            List of validation error messages (empty if valid)

        """
        errors = []

        # Check for duplicate entity IDs
        all_entity_ids = self.entity_ids
        duplicates = {eid for eid in all_entity_ids if all_entity_ids.count(eid) > 1}
        if duplicates:
            errors.append(f"Duplicate entity IDs found: {duplicates}")

        # Check for empty entity lists
        if (
            not self.sensors.motion
            and not self.sensors.media
            and not self.sensors.appliance
        ):
            errors.append("No motion, media, or appliance sensors configured")

        # Validate individual sensor lists
        for sensor_type, entity_list in [
            ("motion", self.sensors.motion),
            ("media", self.sensors.media),
            ("appliance", self.sensors.appliance),
            ("door", self.sensors.door),
            ("window", self.sensors.window),
            ("illuminance", self.sensors.illuminance),
            ("humidity", self.sensors.humidity),
            ("temperature", self.sensors.temperature),
        ]:
            if entity_list and not all(
                isinstance(eid, str) and eid.strip() for eid in entity_list
            ):
                errors.append(f"Invalid {sensor_type} sensor entity IDs: {entity_list}")

        return errors

    @staticmethod
    def _merge_entry(config_entry: ConfigEntry) -> dict[str, Any]:
        """Merge the config entry data and options."""
        merged = dict(config_entry.data)
        merged.update(config_entry.options)
        return merged

    def update_from_entry(self, config_entry: ConfigEntry) -> None:
        """Update the config from a new config entry."""
        # Update the config entry reference
        self.config_entry = config_entry

        # Reload configuration from the merged entry data
        self._load_config(self._merge_entry(config_entry))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key."""
        return getattr(self, key, default)

    async def update_config(self, options: dict[str, Any]) -> None:
        """Update configuration and persist to Home Assistant config entry.

        Args:
            options: Dictionary of configuration options to update

        Raises:
            ValueError: If any option values are invalid
            HomeAssistantError: If updating the config entry fails

        """

        def _validate_config_entry() -> None:
            if self.config_entry is None:
                raise ValueError("Config entry is None")

        try:
            _validate_config_entry()
            # Create new options dict by merging existing with new options
            new_options = dict(self.config_entry.options)  # type: ignore[union-attr]
            new_options.update(options)

            # Update the config entry in Home Assistant
            self.hass.config_entries.async_update_entry(
                self.config_entry,  # type: ignore[arg-type]
                options=new_options,  # type: ignore[arg-type]
            )

            # Merge existing config entry with new options for internal state
            data = self._merge_entry(self.config_entry)  # type: ignore[arg-type]
            data.update(options)

            # Reload configuration with updated data
            self._load_config(data)

            # Request update since threshold affects occupied calculation
            # Only request refresh if setup is complete to avoid debouncer conflicts
            if self.coordinator.setup_complete:
                await self.coordinator.async_request_refresh()

        except Exception as err:
            raise HomeAssistantError(f"Failed to update configuration: {err}") from err
