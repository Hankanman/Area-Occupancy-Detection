"""Type definitions for Area Occupancy Detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from homeassistant.util import dt as dt_util

from .const import (
    CONF_APPLIANCES,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_WINDOW_SENSORS,
    MAX_PRIOR,
    MAX_PROBABILITY,
    MIN_PRIOR,
    MIN_PROBABILITY,
)

if TYPE_CHECKING:
    from .probabilities import Probabilities


_LOGGER = logging.getLogger(__name__)


class EntityType(StrEnum):
    """Enum representing the different types of entities used."""

    MOTION = "motion"
    MEDIA = "media"
    APPLIANCE = "appliance"
    DOOR = "door"
    WINDOW = "window"
    LIGHT = "light"
    ENVIRONMENTAL = "environmental"


# Unified configuration type
class Config(TypedDict):
    """Unified configuration for the integration.

    Required fields:
        name: The display name for the area
        area_id: The unique identifier for the area
        motion_sensors: List of motion sensor entity IDs to monitor

    Optional fields:
        media_devices: List of media player entity IDs to monitor
        appliances: List of appliance entity IDs to monitor
        illuminance_sensors: List of illuminance sensor entity IDs
        humidity_sensors: List of humidity sensor entity IDs
        temperature_sensors: List of temperature sensor entity IDs
        door_sensors: List of door sensor entity IDs
        window_sensors: List of window sensor entity IDs
        lights: List of light entity IDs to monitor
        threshold: Probability threshold for occupancy detection
        history_period: Number of days of historical data to analyze
        decay_enabled: Whether to enable probability decay
        decay_window: Time window for probability decay in minutes
        decay_min_delay: Minimum delay before decay starts in minutes
        historical_analysis_enabled: Whether to enable historical data analysis
    """

    # Required fields
    name: str
    area_id: str
    motion_sensors: list[str]

    # Optional fields
    media_devices: NotRequired[list[str]]
    appliances: NotRequired[list[str]]
    illuminance_sensors: NotRequired[list[str]]
    humidity_sensors: NotRequired[list[str]]
    temperature_sensors: NotRequired[list[str]]
    door_sensors: NotRequired[list[str]]
    window_sensors: NotRequired[list[str]]
    lights: NotRequired[list[str]]
    threshold: NotRequired[float]
    history_period: NotRequired[int]
    decay_enabled: NotRequired[bool]
    decay_window: NotRequired[int]
    decay_min_delay: NotRequired[int]
    historical_analysis_enabled: NotRequired[bool]


class SensorProbability(TypedDict):
    """Probability details for a single sensor.

    Required fields:
        probability: Raw probability value (0-1)
        weight: Weight factor for this sensor (0-1)
        weighted_probability: Probability after applying weight
    """

    probability: float
    weight: float
    weighted_probability: float


@dataclass
class SensorCalculation:
    """Data class to hold sensor calculation results."""

    weighted_probability: float
    is_active: bool
    details: SensorProbability

    @classmethod
    def empty(cls) -> SensorCalculation:
        """Create an empty sensor calculation with zero values.

        Returns:
            A SensorCalculation instance with zero values

        """
        return cls(
            weighted_probability=0.0,
            is_active=False,
            details={
                "probability": 0.0,
                "weight": 0.0,
                "weighted_probability": 0.0,
            },
        )


@dataclass
class ProbabilityState:
    """State of probability calculations."""

    probability: float = field(default=0.0)
    previous_probability: float = field(default=0.0)
    threshold: float = field(default=0.5)
    prior_probability: float = field(default=0.0)
    sensor_probabilities: dict[str, SensorProbability] = field(default_factory=dict)
    decay_status: float = field(default=0.0)
    current_states: dict[str, SensorInfo] = field(default_factory=dict)
    previous_states: dict[str, SensorInfo] = field(default_factory=dict)
    is_occupied: bool = field(default=False)
    decaying: bool = field(default=False)
    decay_start_time: datetime | None = field(default=None)
    decay_start_probability: float | None = field(default=None)

    @property
    def active_triggers(self) -> list[str]:
        """Get list of active triggers from sensor probabilities."""
        return list(self.sensor_probabilities.keys())

    def __post_init__(self) -> None:
        """Validate probability values."""
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
        if not 0 <= self.previous_probability <= 1:
            raise ValueError("Previous probability must be between 0 and 1")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if not 0 <= self.prior_probability <= 1:
            raise ValueError("Prior probability must be between 0 and 1")
        # Decay status is 0-100
        if not 0 <= self.decay_status <= 100:
            raise ValueError("Decay status must be between 0 and 100")

    def update(
        self,
        probability: float | None = None,
        previous_probability: float | None = None,
        threshold: float | None = None,
        prior_probability: float | None = None,
        sensor_probabilities: dict[str, SensorProbability] | None = None,
        decay_status: float | None = None,
        current_states: dict[str, SensorInfo] | None = None,
        previous_states: dict[str, SensorInfo] | None = None,
        is_occupied: bool | None = None,
        decaying: bool | None = None,
        decay_start_time: datetime | None = None,
        decay_start_probability: float | None = None,
    ) -> None:
        """Update the state with new values while maintaining the same instance."""
        if probability is not None:
            self.probability = max(0.0, min(probability, 1.0))
        if previous_probability is not None:
            self.previous_probability = max(0.0, min(previous_probability, 1.0))
        if threshold is not None:
            self.threshold = max(0.0, min(threshold, 1.0))
        if prior_probability is not None:
            self.prior_probability = max(0.0, min(prior_probability, 1.0))
        if sensor_probabilities is not None:
            self.sensor_probabilities = dict(sensor_probabilities)
        if decay_status is not None:
            self.decay_status = max(0.0, min(decay_status, 100.0))
        if current_states is not None:
            self.current_states = dict(current_states)
        if previous_states is not None:
            self.previous_states = dict(previous_states)
        if is_occupied is not None:
            self.is_occupied = is_occupied
        if decaying is not None:
            self.decaying = decaying
        if decay_start_time is not None:
            self.decay_start_time = decay_start_time
        if decay_start_probability is not None:
            self.decay_start_probability = decay_start_probability

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return {
            "probability": self.probability,
            "previous_probability": self.previous_probability,
            "threshold": self.threshold,
            "prior_probability": self.prior_probability,
            "sensor_probabilities": self.sensor_probabilities,
            "decay_status": self.decay_status,
            "current_states": self.current_states,
            "previous_states": self.previous_states,
            "is_occupied": self.is_occupied,
            "decaying": self.decaying,
            "decay_start_time": self.decay_start_time.isoformat()
            if self.decay_start_time
            else None,
            "decay_start_probability": self.decay_start_probability,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbabilityState:
        """Create a ProbabilityState from a dictionary."""
        decay_start_time_str = data.get("decay_start_time")
        decay_start_time = (
            dt_util.parse_datetime(str(decay_start_time_str))
            if decay_start_time_str
            else None
        )
        return cls(
            probability=float(data["probability"]),
            previous_probability=float(data["previous_probability"]),
            threshold=float(data["threshold"]),
            prior_probability=float(data["prior_probability"]),
            sensor_probabilities=data["sensor_probabilities"],
            decay_status=float(data["decay_status"]),
            current_states=data.get("current_states", {}),
            previous_states=data.get("previous_states", {}),
            is_occupied=bool(data["is_occupied"]),
            decaying=bool(data.get("decaying", False)),
            decay_start_time=decay_start_time,
            decay_start_probability=data.get("decay_start_probability"),
        )


@dataclass
class PriorData:
    """Holds prior probability data for an entity or type."""

    prior: float = field(default=MIN_PROBABILITY)
    prob_given_true: float | None = field(default=None)
    prob_given_false: float | None = field(default=None)
    last_updated: str | None = field(default=None)

    def __post_init__(self):
        """Validate probabilities."""
        if not MIN_PRIOR <= self.prior <= MAX_PRIOR:
            raise ValueError(f"Prior must be between {MIN_PRIOR} and {MAX_PRIOR}")
        if (
            self.prob_given_true is not None
            and not MIN_PROBABILITY <= self.prob_given_true <= MAX_PROBABILITY
        ):
            raise ValueError(
                f"prob_given_true must be between {MIN_PROBABILITY} and {MAX_PROBABILITY}"
            )
        if (
            self.prob_given_false is not None
            and not MIN_PROBABILITY <= self.prob_given_false <= MAX_PROBABILITY
        ):
            raise ValueError(
                f"prob_given_false must be between {MIN_PROBABILITY} and {MAX_PROBABILITY}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert PriorData to a dictionary."""
        data = {
            "prior": self.prior,
            "last_updated": self.last_updated,
        }
        if self.prob_given_true is not None:
            data["prob_given_true"] = self.prob_given_true
        if self.prob_given_false is not None:
            data["prob_given_false"] = self.prob_given_false
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorData:
        """Create PriorData from a dictionary."""
        # Handle potential None values during deserialization
        prior = data.get("prior")
        p_true = data.get("prob_given_true")
        p_false = data.get("prob_given_false")
        last_updated = data.get("last_updated")

        return cls(
            prior=float(prior) if prior is not None else MIN_PROBABILITY,
            prob_given_true=float(p_true) if p_true is not None else None,
            prob_given_false=float(p_false) if p_false is not None else None,
            last_updated=str(last_updated) if last_updated is not None else None,
        )


@dataclass
class PriorState:
    """State of prior probability calculations."""

    # Overall prior probability for the area
    overall_prior: float = field(default=MIN_PROBABILITY)

    # Prior probabilities by sensor type (simple float for easy access)
    motion_prior: float = field(default=MIN_PROBABILITY)
    media_prior: float = field(default=MIN_PROBABILITY)
    appliance_prior: float = field(default=MIN_PROBABILITY)
    door_prior: float = field(default=MIN_PROBABILITY)
    window_prior: float = field(default=MIN_PROBABILITY)
    light_prior: float = field(default=MIN_PROBABILITY)
    environmental_prior: float = field(default=MIN_PROBABILITY)

    # Individual entity priors using PriorData
    entity_priors: dict[str, PriorData] = field(default_factory=dict)

    # Type-level priors with full probability data using PriorData
    type_priors: dict[str, PriorData] = field(default_factory=dict)

    # Analysis period in days
    analysis_period: int = field(default=7)

    def __post_init__(self) -> None:
        """Validate prior values."""
        for attr_name in [
            "overall_prior",
            "motion_prior",
            "media_prior",
            "appliance_prior",
            "door_prior",
            "window_prior",
            "light_prior",
            "environmental_prior",
        ]:
            value = getattr(self, attr_name)
            if not MIN_PROBABILITY <= value <= MAX_PROBABILITY:
                raise ValueError(
                    f"{attr_name} must be between {MIN_PROBABILITY} and {MAX_PROBABILITY}"
                )

    def update(
        self,
        overall_prior: float | None = None,
        motion_prior: float | None = None,
        media_prior: float | None = None,
        appliance_prior: float | None = None,
        door_prior: float | None = None,
        window_prior: float | None = None,
        light_prior: float | None = None,
        environmental_prior: float | None = None,
        entity_priors: dict[str, PriorData] | None = None,
        type_priors: dict[str, PriorData] | None = None,
        analysis_period: int | None = None,
    ) -> None:
        """Update the state with new values while maintaining the same instance."""
        if overall_prior is not None:
            self.overall_prior = max(
                MIN_PROBABILITY, min(overall_prior, MAX_PROBABILITY)
            )
        if motion_prior is not None:
            self.motion_prior = max(MIN_PROBABILITY, min(motion_prior, MAX_PROBABILITY))
        if media_prior is not None:
            self.media_prior = max(MIN_PROBABILITY, min(media_prior, MAX_PROBABILITY))
        if appliance_prior is not None:
            self.appliance_prior = max(
                MIN_PROBABILITY, min(appliance_prior, MAX_PROBABILITY)
            )
        if door_prior is not None:
            self.door_prior = max(MIN_PROBABILITY, min(door_prior, MAX_PROBABILITY))
        if window_prior is not None:
            self.window_prior = max(MIN_PROBABILITY, min(window_prior, MAX_PROBABILITY))
        if light_prior is not None:
            self.light_prior = max(MIN_PROBABILITY, min(light_prior, MAX_PROBABILITY))
        if environmental_prior is not None:
            self.environmental_prior = max(
                MIN_PROBABILITY, min(environmental_prior, MAX_PROBABILITY)
            )
        if entity_priors is not None:
            self.entity_priors = entity_priors
        if type_priors is not None:
            self.type_priors = type_priors
        if analysis_period is not None:
            self.analysis_period = analysis_period

        # Note: overall_prior should be recalculated by the caller after updates

    def update_entity_prior(
        self,
        entity_id: str,
        prob_given_true: float,
        prob_given_false: float,
        prior: float,
        timestamp: str,
    ) -> None:
        """Update prior for a specific entity."""
        # Ensure probabilities are valid before creating PriorData
        if not MIN_PROBABILITY <= prior <= MAX_PROBABILITY:
            raise ValueError(f"Invalid prior {prior} for {entity_id}")
        if not MIN_PROBABILITY <= prob_given_true <= MAX_PROBABILITY:
            raise ValueError(
                f"Invalid prob_given_true {prob_given_true} for {entity_id}"
            )
        if not MIN_PROBABILITY <= prob_given_false <= MAX_PROBABILITY:
            raise ValueError(
                f"Invalid prob_given_false {prob_given_false} for {entity_id}"
            )

        self.entity_priors[entity_id] = PriorData(
            prior=prior,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            last_updated=timestamp,
        )

    def update_type_prior(
        self,
        sensor_type: str,
        prior_value: float,
        timestamp: str,
        prob_given_true: float | None = None,
        prob_given_false: float | None = None,
    ) -> None:
        """Update prior for a specific sensor type.

        Args:
            sensor_type: The type of sensor to update (str value of EntityType)
            prior_value: The prior probability value
            timestamp: The timestamp of the update
            prob_given_true: Optional P(sensor active | area occupied)
            prob_given_false: Optional P(sensor active | area not occupied)

        """
        # Update the simple prior attributes (motion_prior, etc.)
        prior_attr_map = {
            EntityType.MOTION.value: "motion_prior",
            EntityType.MEDIA.value: "media_prior",
            EntityType.APPLIANCE.value: "appliance_prior",
            EntityType.DOOR.value: "door_prior",
            EntityType.WINDOW.value: "window_prior",
            EntityType.LIGHT.value: "light_prior",
            EntityType.ENVIRONMENTAL.value: "environmental_prior",
        }
        if sensor_type in prior_attr_map:
            attr_name = prior_attr_map[sensor_type]
            setattr(
                self, attr_name, max(MIN_PROBABILITY, min(prior_value, MAX_PROBABILITY))
            )
        else:
            _LOGGER.warning(
                "Attempted to update prior for unknown sensor type: %s", sensor_type
            )

        # Create or update type prior entry in the type_priors dict
        self.type_priors[sensor_type] = PriorData(
            prior=max(MIN_PROBABILITY, min(prior_value, MAX_PROBABILITY)),
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            last_updated=timestamp,
        )

        # Caller is responsible for recalculating overall_prior after all updates

    def calculate_overall_prior(self, probabilities: Probabilities) -> float:
        """Calculate the overall prior from sensor type priors that have configured sensors.

        Args:
            probabilities: Probabilities instance from coordinator

        Returns:
            Average prior probability from configured sensor types

        """
        # Get all configured entity types from the probabilities instance
        configured_types = set()
        for entity_id in probabilities.entity_types:
            if entity_type := probabilities.get_entity_type(entity_id):
                configured_types.add(entity_type.value)  # Use .value for comparison

        # Only include priors for configured types from self.type_priors
        valid_priors = []
        for sensor_type, prior_data in self.type_priors.items():
            # Check if the sensor_type string is in the set of configured type values
            if sensor_type in configured_types and prior_data.prior > MIN_PROBABILITY:
                valid_priors.append(prior_data.prior)

        if not valid_priors:
            return MIN_PROBABILITY  # Return min instead of 0.0

        return sum(valid_priors) / len(valid_priors)

    def initialize_from_defaults(self, probabilities: Probabilities) -> None:
        """Initialize the state from default type priors.

        Args:
            probabilities: The probabilities configuration instance

        """
        timestamp = dt_util.utcnow().isoformat()

        # Get configured sensor types
        configured_types = set()
        for entity_id in probabilities.entity_types:
            if entity_type := probabilities.get_entity_type(entity_id):
                configured_types.add(entity_type.value)  # Use .value

        # Get initial default data (now returns dict[str, PriorData])
        initial_type_data = probabilities.get_initial_type_priors()

        # Initialize only configured type priors using PriorData
        for sensor_type_str, default_prior_data in initial_type_data.items():
            # Use default PriorData object's prior if type configured, else 0.0
            prior_value = (
                default_prior_data.prior if sensor_type_str in configured_types else 0.0
            )
            prob_true = default_prior_data.prob_given_true
            prob_false = default_prior_data.prob_given_false

            # update_type_prior now updates self.type_priors with PriorData
            self.update_type_prior(
                sensor_type_str,
                prior_value,
                timestamp,
                prob_true,
                prob_false,
            )

        # Calculate and set the overall prior *after* all types are processed
        overall_prior = self.calculate_overall_prior(probabilities)
        # Use update method to set only overall_prior safely
        self.update(overall_prior=overall_prior)

    def to_dict(self) -> dict[str, Any]:
        """Convert the PriorState dataclass to a dictionary for storage."""
        return {
            "overall_prior": self.overall_prior,
            "motion_prior": self.motion_prior,
            "media_prior": self.media_prior,
            "appliance_prior": self.appliance_prior,
            "door_prior": self.door_prior,
            "window_prior": self.window_prior,
            "light_prior": self.light_prior,
            "environmental_prior": self.environmental_prior,
            # Serialize PriorData objects
            "entity_priors": {
                k: v.to_dict() for k, v in self.entity_priors.items() if v
            },
            "type_priors": {k: v.to_dict() for k, v in self.type_priors.items() if v},
            "analysis_period": self.analysis_period,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorState:
        """Create a PriorState from a dictionary."""
        return cls(
            overall_prior=float(data.get("overall_prior", MIN_PROBABILITY)),
            motion_prior=float(data.get("motion_prior", MIN_PROBABILITY)),
            media_prior=float(data.get("media_prior", MIN_PROBABILITY)),
            appliance_prior=float(data.get("appliance_prior", MIN_PROBABILITY)),
            door_prior=float(data.get("door_prior", MIN_PROBABILITY)),
            window_prior=float(data.get("window_prior", MIN_PROBABILITY)),
            light_prior=float(data.get("light_prior", MIN_PROBABILITY)),
            environmental_prior=float(data.get("environmental_prior", MIN_PROBABILITY)),
            # Deserialize PriorData objects
            entity_priors={
                k: PriorData.from_dict(v)
                for k, v in data.get("entity_priors", {}).items()
                if isinstance(v, dict)  # Add check for dict type
            },
            type_priors={
                k: PriorData.from_dict(v)
                for k, v in data.get("type_priors", {}).items()
                if isinstance(v, dict)  # Add check for dict type
            },
            analysis_period=int(data.get("analysis_period", 7)),
        )


class SensorInfo(TypedDict):
    """Type for sensor state information."""

    state: str | None
    last_changed: str
    availability: bool


class ProbabilityConfig(TypedDict):
    """Configuration for default probabilities and weights per sensor type."""

    prob_given_true: float
    prob_given_false: float
    default_prior: float
    weight: float
    active_states: set[str]


class TimeInterval(TypedDict):
    """Time interval and duration data.

    Required fields:
        start: Start time of the interval
        end: End time of the interval
        state: State value during the interval
    """

    start: datetime
    end: datetime
    state: Any


class ProbabilityAttributes(TypedDict):
    """Attributes for the probability sensor."""

    active_triggers: NotRequired[list[str]]
    sensor_probabilities: NotRequired[set[str]]
    threshold: NotRequired[str]


class PriorsAttributes(TypedDict):
    """Attributes for the priors sensor.

    Optional fields:
        motion: Motion sensor prior probability
        media: Media device prior probability
        appliance: Appliance prior probability
        door: Door sensor prior probability
        window: Window sensor prior probability
        light: Light prior probability
        environmental: Environmental prior probability
        last_updated: Last update timestamp
        total_period: Total analysis period
    """

    motion: NotRequired[str]
    media: NotRequired[str]
    appliance: NotRequired[str]
    door: NotRequired[str]
    window: NotRequired[str]
    light: NotRequired[str]
    environmental: NotRequired[str]
    last_updated: NotRequired[str]
    total_period: NotRequired[str]


@dataclass
class SensorInputs:
    """Container for all sensor inputs used in area occupancy detection.

    This class handles the storage, validation, and management of all sensor inputs
    used in area occupancy detection. It provides methods for validating entity IDs,
    managing sensor lists, and retrieving combined sensor lists.

    Attributes:
        motion_sensors: List of motion sensor entity IDs
        primary_sensor: Primary occupancy sensor entity ID
        media_devices: List of media device entity IDs
        appliances: List of appliance entity IDs
        illuminance_sensors: List of illuminance sensor entity IDs
        humidity_sensors: List of humidity sensor entity IDs
        temperature_sensors: List of temperature sensor entity IDs
        door_sensors: List of door sensor entity IDs
        window_sensors: List of window sensor entity IDs
        lights: List of light entity IDs

    """

    motion_sensors: list[str]
    primary_sensor: str
    media_devices: list[str]
    appliances: list[str]
    illuminance_sensors: list[str]
    humidity_sensors: list[str]
    temperature_sensors: list[str]
    door_sensors: list[str]
    window_sensors: list[str]
    lights: list[str]

    @staticmethod
    def is_valid_entity_id(entity_id: str) -> bool:
        """Check if an entity ID has valid format.

        Args:
            entity_id: Entity ID to validate

        Returns:
            True if valid, False otherwise

        """
        if not isinstance(entity_id, str):
            return False
        parts = entity_id.split(".")
        return len(parts) == 2 and all(parts)

    @classmethod
    def validate_entity(cls, conf_key: str, config: dict, default: str = "") -> str:
        """Validate a single entity ID from configuration.

        Args:
            conf_key: Configuration key to validate
            config: Configuration dictionary containing the entity ID
            default: Default value if not found

        Returns:
            Validated entity ID

        Raises:
            ValueError: If entity ID format is invalid

        """
        entity_id = config.get(conf_key, default)
        if entity_id and not cls.is_valid_entity_id(entity_id):
            raise ValueError(f"Invalid entity ID format for {conf_key}: {entity_id}")
        return entity_id

    @classmethod
    def validate_entity_list(
        cls, conf_key: str, config: dict, default: list | None = None
    ) -> list[str]:
        """Validate a list of entity IDs from configuration.

        Args:
            conf_key: Configuration key to validate
            config: Configuration dictionary containing the entity IDs
            default: Default value if not found, defaults to empty list

        Returns:
            List of validated entity IDs

        Raises:
            TypeError: If configuration value is not a list
            ValueError: If any entity ID format is invalid

        """
        if default is None:
            default = []
        entity_ids = config.get(conf_key, default)
        if not isinstance(entity_ids, list):
            raise TypeError(f"Configuration {conf_key} must be a list")

        for entity_id in entity_ids:
            if not cls.is_valid_entity_id(entity_id):
                raise ValueError(f"Invalid entity ID format in {conf_key}: {entity_id}")
        return entity_ids

    @classmethod
    def from_config(cls, config: dict) -> SensorInputs:
        """Create a SensorInputs instance from a configuration dictionary.

        Args:
            config: Configuration dictionary containing sensor settings

        Returns:
            Initialized SensorInputs instance

        Raises:
            ValueError: If required configuration is missing or invalid
            TypeError: If configuration values have wrong types

        """
        return cls(
            motion_sensors=cls.validate_entity_list(CONF_MOTION_SENSORS, config, []),
            primary_sensor=cls.validate_entity(
                CONF_PRIMARY_OCCUPANCY_SENSOR, config, ""
            ),
            media_devices=cls.validate_entity_list(CONF_MEDIA_DEVICES, config, []),
            appliances=cls.validate_entity_list(CONF_APPLIANCES, config, []),
            illuminance_sensors=cls.validate_entity_list(
                CONF_ILLUMINANCE_SENSORS, config, []
            ),
            humidity_sensors=cls.validate_entity_list(
                CONF_HUMIDITY_SENSORS, config, []
            ),
            temperature_sensors=cls.validate_entity_list(
                CONF_TEMPERATURE_SENSORS, config, []
            ),
            door_sensors=cls.validate_entity_list(CONF_DOOR_SENSORS, config, []),
            window_sensors=cls.validate_entity_list(CONF_WINDOW_SENSORS, config, []),
            lights=cls.validate_entity_list(CONF_LIGHTS, config, []),
        )

    def get_all_sensors(self) -> list[str]:
        """Get a list of all sensor entity IDs.

        Returns:
            List of all sensor entity IDs

        """
        all_sensors = []
        for sensor_list in [
            self.motion_sensors,
            self.media_devices,
            self.appliances,
            self.illuminance_sensors,
            self.humidity_sensors,
            self.temperature_sensors,
            self.door_sensors,
            self.window_sensors,
            self.lights,
        ]:
            all_sensors.extend(sensor_list)

        # Add primary sensor if not already included
        if self.primary_sensor not in all_sensors:
            all_sensors.append(self.primary_sensor)

        return all_sensors


@dataclass
class TypeAggregate:
    """Type for aggregating sensor data."""

    priors: list[float] = field(default_factory=list)
    p_true: list[float] = field(default_factory=list)
    p_false: list[float] = field(default_factory=list)
    count: int = 0


class InstanceData(TypedDict):
    """TypedDict for stored instance data."""

    name: str | None
    prior_state: dict[str, Any] | None  # Store as dict for JSON compatibility
    last_updated: str


class StoredData(TypedDict):
    """TypedDict for the overall structure stored in the JSON file."""

    instances: dict[str, InstanceData]


@dataclass
class OccupancyCalculationResult:
    """Holds the results of the occupancy probability calculation."""

    calculated_probability: float
    prior_probability: float
    sensor_probabilities: dict[str, SensorProbability]


@dataclass
class LoadedInstanceData:
    """Holds data loaded for a specific instance from storage."""

    name: str | None
    prior_state: PriorState | None
    last_updated: str | None
