"""Configuration model and manager for Area Occupancy Detection."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.util import dt as dt_util

from ..const import (
    ANALYSIS_INTERVAL,
    CONF_AIR_QUALITY_SENSORS,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_CO2_SENSORS,
    CONF_CO_SENSORS,
    CONF_COVER_ACTIVE_STATES,
    CONF_COVER_SENSORS,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MIN_PRIOR_OVERRIDE,
    CONF_MOTION_PROB_GIVEN_FALSE,
    CONF_MOTION_PROB_GIVEN_TRUE,
    CONF_MOTION_SENSORS,
    CONF_MOTION_TIMEOUT,
    CONF_PEOPLE,
    CONF_PERSON_CONFIDENCE_THRESHOLD,
    CONF_PERSON_DEVICE_TRACKER,
    CONF_PERSON_ENTITY,
    CONF_PERSON_SLEEP_AREA,
    CONF_PERSON_SLEEP_SENSOR,
    CONF_PM10_SENSORS,
    CONF_PM25_SENSORS,
    CONF_POWER_SENSORS,
    CONF_PRESSURE_SENSORS,
    CONF_PURPOSE,
    CONF_SLEEP_END,
    CONF_SLEEP_START,
    CONF_SOUND_PRESSURE_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_THRESHOLD,
    CONF_VOC_SENSORS,
    CONF_WASP_ENABLED,
    CONF_WASP_MAX_DURATION,
    CONF_WASP_MOTION_TIMEOUT,
    CONF_WASP_VERIFICATION_DELAY,
    CONF_WASP_WEIGHT,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_COVER,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_POWER,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DECAY_INTERVAL,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_COVER_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_MIN_PRIOR_OVERRIDE,
    DEFAULT_MOTION_PROB_GIVEN_FALSE,
    DEFAULT_MOTION_PROB_GIVEN_TRUE,
    DEFAULT_MOTION_TIMEOUT,
    DEFAULT_PURPOSE,
    DEFAULT_SLEEP_CONFIDENCE_THRESHOLD,
    DEFAULT_SLEEP_END,
    DEFAULT_SLEEP_START,
    DEFAULT_THRESHOLD,
    DEFAULT_WASP_MAX_DURATION,
    DEFAULT_WASP_MOTION_TIMEOUT,
    DEFAULT_WASP_VERIFICATION_DELAY,
    DEFAULT_WASP_WEIGHT,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_COVER,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_POWER,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
    HA_RECORDER_DAYS,
)
from .purpose import get_default_decay_half_life

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator


_LOGGER = logging.getLogger(__name__)


@dataclass
class PersonConfig:
    """Configuration for a person's sleep tracking."""

    person_entity: str  # e.g. "person.seb" — provides home/not_home state
    sleep_confidence_sensor: str  # e.g. "sensor.phone_seb_sleep_confidence"
    sleep_area_id: str  # e.g. "bedroom" — HA area ID
    confidence_threshold: int = DEFAULT_SLEEP_CONFIDENCE_THRESHOLD
    device_tracker: str | None = None  # optional override for home/away state


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
        coordinator: "AreaOccupancyCoordinator",
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
        self.decay_interval = DECAY_INTERVAL

        # Database and storage settings
        # These could be made configurable in the future if needed
        # self.database_retention_days = RETENTION_DAYS
        # self.enable_backups = True

        # Global Settings
        # self.sleep_start and self.sleep_end are now properties

        # Future: Cross-area coordination settings
        # self.person_tracking_enabled = False
        # self.area_transition_detection = False
        # self.global_occupancy_threshold = 0.5

    @property
    def sleep_start(self) -> str:
        """Get sleep start time from config entry options."""
        return self.config_entry.options.get(CONF_SLEEP_START, DEFAULT_SLEEP_START)

    @property
    def sleep_end(self) -> str:
        """Get sleep end time from config entry options."""
        return self.config_entry.options.get(CONF_SLEEP_END, DEFAULT_SLEEP_END)

    @property
    def people(self) -> list[PersonConfig]:
        """Get configured people from config entry options."""
        raw_people = self.config_entry.options.get(CONF_PEOPLE, [])
        result: list[PersonConfig] = []
        for person_data in raw_people:
            if not isinstance(person_data, dict):
                continue
            person_entity = person_data.get(CONF_PERSON_ENTITY)
            sleep_sensor = person_data.get(CONF_PERSON_SLEEP_SENSOR)
            sleep_area = person_data.get(CONF_PERSON_SLEEP_AREA)
            if not person_entity or not sleep_sensor or not sleep_area:
                _LOGGER.warning("Skipping incomplete person config: %s", person_data)
                continue
            try:
                threshold = int(
                    person_data.get(
                        CONF_PERSON_CONFIDENCE_THRESHOLD,
                        DEFAULT_SLEEP_CONFIDENCE_THRESHOLD,
                    )
                )
            except (ValueError, TypeError):
                _LOGGER.warning(
                    "Invalid confidence threshold for %s, using default",
                    person_entity,
                )
                threshold = DEFAULT_SLEEP_CONFIDENCE_THRESHOLD
            device_tracker = person_data.get(CONF_PERSON_DEVICE_TRACKER) or None
            result.append(
                PersonConfig(
                    person_entity=person_entity,
                    sleep_confidence_sensor=sleep_sensor,
                    sleep_area_id=sleep_area,
                    confidence_threshold=threshold,
                    device_tracker=device_tracker,
                )
            )
        return result

    def get_people_for_area(self, area_id: str) -> list[PersonConfig]:
        """Get people configured for a specific area."""
        return [p for p in self.people if p.sleep_area_id == area_id]

    def __repr__(self) -> str:
        """Return a string representation of the integration config."""
        return f"IntegrationConfig(name={self.integration_name!r})"


@dataclass
class Sensors:
    """Sensors configuration."""

    motion: list[str] = field(default_factory=list)
    motion_timeout: int = DEFAULT_MOTION_TIMEOUT
    motion_prob_given_true: float = DEFAULT_MOTION_PROB_GIVEN_TRUE
    motion_prob_given_false: float = DEFAULT_MOTION_PROB_GIVEN_FALSE
    media: list[str] = field(default_factory=list)
    appliance: list[str] = field(default_factory=list)
    illuminance: list[str] = field(default_factory=list)
    humidity: list[str] = field(default_factory=list)
    temperature: list[str] = field(default_factory=list)
    co2: list[str] = field(default_factory=list)
    co: list[str] = field(default_factory=list)
    sound_pressure: list[str] = field(default_factory=list)
    pressure: list[str] = field(default_factory=list)
    air_quality: list[str] = field(default_factory=list)
    voc: list[str] = field(default_factory=list)
    pm25: list[str] = field(default_factory=list)
    pm10: list[str] = field(default_factory=list)
    power: list[str] = field(default_factory=list)
    door: list[str] = field(default_factory=list)
    window: list[str] = field(default_factory=list)
    cover: list[str] = field(default_factory=list)
    _parent_config: "AreaConfig | None" = field(default=None, repr=False, compare=False)

    def get_motion_sensors(self, coordinator: "AreaOccupancyCoordinator") -> list[str]:
        """Get motion sensors including wasp sensor if enabled and available.

        Args:
            coordinator: The coordinator instance to get wasp entity_id from

        Returns:
            list[str]: List of motion sensor entity_ids including wasp if applicable

        """

        motion_sensors = self.motion.copy()

        # Add wasp sensor if enabled and entity_id is available
        # Use parent config if available, otherwise fall back to checking coordinator
        if self._parent_config and hasattr(self._parent_config, "wasp_in_box"):
            wasp_enabled = self._parent_config.wasp_in_box.enabled
        else:
            # Fallback for cases where parent config isn't set
            wasp_enabled = False

        if wasp_enabled:
            # In multi-area architecture, wasp_entity_id is stored per area
            wasp_id = None
            if (
                self._parent_config
                and hasattr(self._parent_config, "area_name")
                and self._parent_config.area_name
                and self._parent_config.area_name in coordinator.areas
            ):
                # Get wasp_entity_id from the area's data
                area_data = coordinator.areas[self._parent_config.area_name]
                wasp_id = getattr(area_data, "wasp_entity_id", None)

            if wasp_id is not None:
                motion_sensors.append(wasp_id)
                _LOGGER.debug(
                    "Adding wasp sensor %s to motion sensors list",
                    wasp_id,
                )

        return motion_sensors

    def get_sleep_sensors(self, coordinator: "AreaOccupancyCoordinator") -> list[str]:
        """Get sleep presence sensors assigned to this area.

        Args:
            coordinator: The coordinator instance

        Returns:
            list[str]: List of sleep sensor entity_ids for this area
        """
        sleep_sensors: list[str] = []
        if (
            self._parent_config
            and hasattr(self._parent_config, "area_name")
            and self._parent_config.area_name
            and self._parent_config.area_name in coordinator.areas
        ):
            area_data = coordinator.areas[self._parent_config.area_name]
            sleep_id = getattr(area_data, "sleep_entity_id", None)
            if sleep_id is not None:
                sleep_sensors.append(sleep_id)
                _LOGGER.debug(
                    "Adding sleep sensor %s to sleep sensors list",
                    sleep_id,
                )
        return sleep_sensors


@dataclass
class SensorStates:
    """Sensor states configuration."""

    motion: list[str] = field(default_factory=lambda: [STATE_ON])
    door: list[str] = field(default_factory=lambda: [DEFAULT_DOOR_ACTIVE_STATE])
    window: list[str] = field(default_factory=lambda: [DEFAULT_WINDOW_ACTIVE_STATE])
    cover: list[str] = field(default_factory=lambda: list(DEFAULT_COVER_ACTIVE_STATES))
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
    cover: float = DEFAULT_WEIGHT_COVER
    environmental: float = DEFAULT_WEIGHT_ENVIRONMENTAL
    power: float = DEFAULT_WEIGHT_POWER
    wasp: float = DEFAULT_WASP_WEIGHT


@dataclass
class DecayConfig:
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


class AreaConfig:
    """Configuration for Area Occupancy Detection."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
        area_name: str | None = None,
        area_data: dict[str, Any] | None = None,
    ):
        """Initialize the config from a coordinator.

        Args:
            coordinator: The coordinator instance
            area_name: Optional area name identifier (for multi-area support)
            area_data: Optional area-specific configuration data (if None, uses config_entry)
        """
        self.coordinator = coordinator
        self.config_entry = coordinator.config_entry
        self.hass = coordinator.hass
        self.db = coordinator.db
        self.area_name = area_name  # Area identifier for multi-area support

        # Load configuration from the merged entry data or provided area_data
        if area_data is not None:
            self._load_config(area_data)
        else:
            if coordinator.config_entry is None:
                raise ValueError("Coordinator config_entry cannot be None")
            merged = self._merge_entry(coordinator.config_entry)

            # Check if we have CONF_AREAS format (multi-area)
            if CONF_AREAS in merged and isinstance(merged[CONF_AREAS], list):
                # Validate area_name is provided for multi-area config
                if area_name is None:
                    raise ValueError(
                        "area_name is required when using multi-area configuration format"
                    )
                # Extract area data for this specific area
                area_data = self._extract_area_data_from_areas_list(
                    merged[CONF_AREAS], area_name, coordinator.hass
                )
                if area_data:
                    self._load_config(area_data)
                else:
                    # Area not found in config - log warning and load empty/default config
                    # to avoid silently ingesting top-level CONF_AREAS structure
                    _LOGGER.warning(
                        "Area '%s' not found in configuration. Loading default config.",
                        area_name,
                    )
                    self._load_config({})
            else:
                # No areas found in config
                self._load_config({})

    def _load_config(self, data: dict[str, Any]) -> None:
        """Load configuration from merged data.

        Args:
            data: Dictionary containing merged config entry data and options

        """
        # Validate threshold range
        threshold = float(data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)) / 100.0

        # Set all configuration attributes
        # Area name is resolved from area_id when needed (via coordinator)
        self.purpose = data.get(CONF_PURPOSE, DEFAULT_PURPOSE)
        # Get area_id from data
        self.area_id = data.get(CONF_AREA_ID)
        if not self.area_id:
            _LOGGER.warning(
                "Area config missing area_id for area '%s'.",
                self.area_name,
            )
        # Resolve area name from area_id (canonical source)
        # This ensures the device name matches the Home Assistant area name
        if self.area_id:
            area_reg = ar.async_get(self.hass)
            area_entry = area_reg.async_get_area(self.area_id)
            if area_entry:
                self.name = area_entry.name
            else:
                # Fallback to area_name if area_id not found in registry
                _LOGGER.warning(
                    "Area ID '%s' not found in registry for area '%s', using fallback name.",
                    self.area_id,
                    self.area_name,
                )
                self.name = self.area_name or "Unknown Area"
        else:
            # Fallback to area_name if area_id not available
            self.name = self.area_name or "Unknown Area"
        self.threshold = threshold

        self.sensors = Sensors(
            motion=data.get(CONF_MOTION_SENSORS, []),
            motion_timeout=int(data.get(CONF_MOTION_TIMEOUT, DEFAULT_MOTION_TIMEOUT)),
            motion_prob_given_true=float(
                data.get(CONF_MOTION_PROB_GIVEN_TRUE, DEFAULT_MOTION_PROB_GIVEN_TRUE)
            ),
            motion_prob_given_false=float(
                data.get(CONF_MOTION_PROB_GIVEN_FALSE, DEFAULT_MOTION_PROB_GIVEN_FALSE)
            ),
            media=data.get(CONF_MEDIA_DEVICES, []),
            appliance=data.get(CONF_APPLIANCES, []),
            illuminance=data.get(CONF_ILLUMINANCE_SENSORS, []),
            humidity=data.get(CONF_HUMIDITY_SENSORS, []),
            temperature=data.get(CONF_TEMPERATURE_SENSORS, []),
            co2=data.get(CONF_CO2_SENSORS, []),
            co=data.get(CONF_CO_SENSORS, []),
            sound_pressure=data.get(CONF_SOUND_PRESSURE_SENSORS, []),
            pressure=data.get(CONF_PRESSURE_SENSORS, []),
            air_quality=data.get(CONF_AIR_QUALITY_SENSORS, []),
            voc=data.get(CONF_VOC_SENSORS, []),
            pm25=data.get(CONF_PM25_SENSORS, []),
            pm10=data.get(CONF_PM10_SENSORS, []),
            power=data.get(CONF_POWER_SENSORS, []),
            door=data.get(CONF_DOOR_SENSORS, []),
            window=data.get(CONF_WINDOW_SENSORS, []),
            cover=data.get(CONF_COVER_SENSORS, []),
            _parent_config=self,
        )

        self.sensor_states = SensorStates(
            motion=[STATE_ON],  # Motion sensors default to STATE_ON
            door=[data.get(CONF_DOOR_ACTIVE_STATE, DEFAULT_DOOR_ACTIVE_STATE)],
            window=[data.get(CONF_WINDOW_ACTIVE_STATE, DEFAULT_WINDOW_ACTIVE_STATE)],
            cover=data.get(CONF_COVER_ACTIVE_STATES, list(DEFAULT_COVER_ACTIVE_STATES)),
            appliance=data.get(
                CONF_APPLIANCE_ACTIVE_STATES, list(DEFAULT_APPLIANCE_ACTIVE_STATES)
            ),
            media=data.get(CONF_MEDIA_ACTIVE_STATES, list(DEFAULT_MEDIA_ACTIVE_STATES)),
        )

        self.weights = Weights(
            motion=data.get(CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION),
            media=data.get(CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA),
            appliance=data.get(CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE),
            door=data.get(CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR),
            window=data.get(CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW),
            cover=data.get(CONF_WEIGHT_COVER, DEFAULT_WEIGHT_COVER),
            environmental=data.get(
                CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL
            ),
            power=data.get(CONF_WEIGHT_POWER, DEFAULT_WEIGHT_POWER),
            wasp=data.get(CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT),
        )

        # Resolve half-life: if 0 or not set, use purpose-based value
        half_life_value = int(data.get(CONF_DECAY_HALF_LIFE, DEFAULT_DECAY_HALF_LIFE))
        if half_life_value == 0:
            half_life_value = int(get_default_decay_half_life(self.purpose))

        self.decay = DecayConfig(
            half_life=half_life_value,
            enabled=bool(data.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED)),
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

        self.min_prior_override = float(
            data.get(CONF_MIN_PRIOR_OVERRIDE, DEFAULT_MIN_PRIOR_OVERRIDE)
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
            *self.sensors.cover,
            *self.sensors.illuminance,
            *self.sensors.humidity,
            *self.sensors.temperature,
            *self.sensors.co2,
            *self.sensors.co,
            *self.sensors.sound_pressure,
            *self.sensors.pressure,
            *self.sensors.air_quality,
            *self.sensors.voc,
            *self.sensors.pm25,
            *self.sensors.pm10,
            *self.sensors.power,
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
            ("co2", self.sensors.co2),
            ("co", self.sensors.co),
            ("sound_pressure", self.sensors.sound_pressure),
            ("pressure", self.sensors.pressure),
            ("air_quality", self.sensors.air_quality),
            ("voc", self.sensors.voc),
            ("pm25", self.sensors.pm25),
            ("pm10", self.sensors.pm10),
            ("power", self.sensors.power),
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

    @staticmethod
    def _extract_area_data_from_areas_list(
        areas_list: list[dict[str, Any]],
        area_name: str | None,
        hass: HomeAssistant,
    ) -> dict[str, Any] | None:
        """Extract area data from CONF_AREAS list for a specific area.

        Args:
            areas_list: List of area configuration dictionaries
            area_name: Area name to find (resolved from area_id)
            hass: Home Assistant instance for resolving area names

        Returns:
            Area configuration dictionary if found, None otherwise
        """
        if not area_name:
            return None

        area_reg = ar.async_get(hass)

        # Try to find area by matching area_name with resolved area names
        for area_data in areas_list:
            area_id = area_data.get(CONF_AREA_ID)
            if area_id:
                # Resolve area name from ID
                area_entry = area_reg.async_get_area(area_id)
                if area_entry and area_entry.name == area_name:
                    return area_data

        # Fallback: if no match found, return None
        return None

    def update_from_entry(self, config_entry: ConfigEntry) -> None:
        """Update the config from a new config entry."""
        # Update the config entry reference
        self.config_entry = config_entry

        # Reload configuration from the merged entry data
        merged = self._merge_entry(config_entry)

        # Check if we have CONF_AREAS format (multi-area)
        if CONF_AREAS in merged and isinstance(merged[CONF_AREAS], list):
            # Validate area_name is provided for multi-area config
            if self.area_name is None:
                raise ValueError(
                    "area_name is required when using multi-area configuration format"
                )
            # Extract area data for this specific area
            area_data = self._extract_area_data_from_areas_list(
                merged[CONF_AREAS], self.area_name, self.hass
            )
            if area_data:
                self._load_config(area_data)
            else:
                # Area not found in config - log warning and load empty/default config
                # to avoid silently ingesting top-level CONF_AREAS structure
                _LOGGER.warning(
                    "Area '%s' not found in configuration. Loading default config.",
                    self.area_name,
                )
                self._load_config({})
        else:
            # No CONF_AREAS format found - load empty/default config
            _LOGGER.warning(
                "Configuration must contain CONF_AREAS list. Loading default config."
            )
            self._load_config({})

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

        def _validate_area_name_for_multi_area(data: dict[str, Any]) -> None:
            """Validate area_name is provided when using multi-area configuration format."""
            if CONF_AREAS in data and isinstance(data[CONF_AREAS], list):
                if self.area_name is None:
                    raise ValueError(
                        "area_name is required when using multi-area configuration format"
                    )

        def _validate_area_found_in_list(area_id: str, area_updated: bool) -> None:
            """Validate that area was found in CONF_AREAS list."""
            if not area_updated:
                _LOGGER.warning(
                    "Area with ID '%s' not found in CONF_AREAS list when updating config",
                    area_id,
                )
                raise ValueError(
                    f"Area with ID '{area_id}' not found in CONF_AREAS list"
                )

        def _validate_area_id_available(area_id: str | None) -> None:
            """Validate that area_id is available for config update."""
            if not area_id:
                _LOGGER.warning(
                    "Area ID not available, cannot update area in CONF_AREAS list"
                )
                raise ValueError("Area ID not available for config update")

        def _validate_multi_area_format(
            new_options: dict[str, Any],
        ) -> None:
            """Validate that configuration is in multi-area format."""
            if CONF_AREAS not in new_options or not isinstance(
                new_options.get(CONF_AREAS), list
            ):
                raise ValueError(
                    "Configuration must be in multi-area format (CONF_AREAS list)"
                )

        try:
            _validate_config_entry()
            # Create new options dict by merging existing with new options
            new_options = dict(self.config_entry.options)  # type: ignore[union-attr]

            # Check if we're in multi-area format and need to update area within CONF_AREAS list
            _validate_multi_area_format(new_options)

            # Multi-area format: update the specific area within CONF_AREAS list
            areas_list = new_options[CONF_AREAS].copy()
            area_updated = False

            # Find the area to update by matching area_id
            area_id = self.area_id
            _validate_area_id_available(area_id)

            for i, area_data in enumerate(areas_list):
                if area_data.get(CONF_AREA_ID) == area_id:
                    # Merge new options into existing area config
                    updated_area = dict(area_data)
                    updated_area.update(options)
                    areas_list[i] = updated_area
                    area_updated = True
                    break

            _validate_area_found_in_list(area_id, area_updated)
            # Update CONF_AREAS list with modified area
            new_options[CONF_AREAS] = areas_list

            # Update the config entry in Home Assistant
            self.hass.config_entries.async_update_entry(
                self.config_entry,  # type: ignore[arg-type]
                options=new_options,  # type: ignore[arg-type]
            )

            # Merge existing config entry with new options for internal state
            data = self._merge_entry(self.config_entry)  # type: ignore[arg-type]
            # For multi-area format, data already has updated area from new_options

            # Validate area_name for multi-area config before processing
            _validate_area_name_for_multi_area(data)

            # Check if we have CONF_AREAS format (multi-area)
            if CONF_AREAS in data and isinstance(data[CONF_AREAS], list):
                # Extract area data for this specific area
                area_data = self._extract_area_data_from_areas_list(
                    data[CONF_AREAS], self.area_name, self.hass
                )
                if area_data:
                    # Reload configuration with extracted area data
                    self._load_config(area_data)
                else:
                    # Area not found in config - log warning and load empty/default config
                    # to avoid silently ingesting top-level CONF_AREAS structure
                    _LOGGER.warning(
                        "Area '%s' not found in configuration. Loading default config.",
                        self.area_name,
                    )
                    self._load_config({})
            else:
                # No CONF_AREAS format found - log warning and load empty/default config
                _LOGGER.warning(
                    "Configuration must contain CONF_AREAS list. Loading default config."
                )
                self._load_config({})

            # Request update since threshold affects occupied calculation
            # Only request refresh if setup is complete to avoid debouncer conflicts
            if self.coordinator.setup_complete:
                await self.coordinator.async_request_refresh()

        except Exception as err:
            raise HomeAssistantError(f"Failed to update configuration: {err}") from err
