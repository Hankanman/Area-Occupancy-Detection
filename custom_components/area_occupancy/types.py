"""Type definitions for Area Occupancy Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict

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
    MIN_PROBABILITY,
)

EntityType = Literal[
    "motion",
    "media",
    "appliance",
    "door",
    "window",
    "light",
    "environmental",
]


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


class DeviceInfo(TypedDict):
    """Device information for Home Assistant device registry.

    Required fields:
        identifiers: Dictionary of identifiers for the device
        name: Display name of the device
        manufacturer: Name of the device manufacturer
        model: Model name of the device
        sw_version: Software version running on the device
    """

    identifiers: dict[str, str]
    name: str
    manufacturer: str
    model: str
    sw_version: str


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
    details: dict[str, float]

    @classmethod
    def empty(cls) -> SensorCalculation:
        """Create an empty sensor calculation with zero values.

        Returns:
            A SensorCalculation instance with zero values

        """
        return cls(
            weighted_probability=0.0,
            is_active=False,
            details={"probability": 0.0, "weight": 0.0, "weighted_probability": 0.0},
        )


@dataclass
class ProbabilityState:
    """State of probability calculations."""

    probability: float = field(default=0.0)
    previous_probability: float = field(default=0.0)
    threshold: float = field(default=0.5)
    prior_probability: float = field(default=0.0)
    sensor_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    decay_status: float = field(default=0.0)
    current_states: dict[
        str, dict[Literal["state", "last_changed", "availability"], Any]
    ] = field(default_factory=dict)
    previous_states: dict[
        str, dict[Literal["state", "last_changed", "availability"], Any]
    ] = field(default_factory=dict)
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
        if not 0 <= self.decay_status <= 1:
            raise ValueError("Decay status must be between 0 and 1")

    def update(
        self,
        probability: float | None = None,
        previous_probability: float | None = None,
        threshold: float | None = None,
        prior_probability: float | None = None,
        sensor_probabilities: dict[str, dict[str, float]] | None = None,
        decay_status: float | None = None,
        current_states: dict[
            str, dict[Literal["state", "last_changed", "availability"], Any]
        ]
        | None = None,
        previous_states: dict[
            str, dict[Literal["state", "last_changed", "availability"], Any]
        ]
        | None = None,
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
            self.sensor_probabilities = sensor_probabilities
        if decay_status is not None:
            self.decay_status = max(0.0, min(decay_status, 100.0))
        if current_states is not None:
            self.current_states = current_states
        if previous_states is not None:
            self.previous_states = previous_states
        if is_occupied is not None:
            self.is_occupied = is_occupied
        if decaying is not None:
            self.decaying = decaying
        if decay_start_time is not None:
            self.decay_start_time = decay_start_time
        if decay_start_probability is not None:
            self.decay_start_probability = decay_start_probability

    def to_dict(
        self,
    ) -> dict[
        str,
        float | list[str] | dict[str, float | dict[str, Any] | bool] | str | None,
    ]:
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
    def from_dict(
        cls,
        data: dict[
            str,
            float | list[str] | dict[str, float | dict[str, Any] | bool] | str | None,
        ],
    ) -> ProbabilityState:
        """Create a ProbabilityState from a dictionary."""
        decay_start_time_str = data.get("decay_start_time")
        decay_start_time = (
            dt_util.parse_datetime(str(decay_start_time_str))
            if decay_start_time_str
            else None
        )
        decay_start_probability = data.get("decay_start_probability")
        return cls(
            probability=float(data["probability"]),
            previous_probability=float(data["previous_probability"]),
            threshold=float(data["threshold"]),
            prior_probability=float(data["prior_probability"]),
            sensor_probabilities=dict(data["sensor_probabilities"]),
            decay_status=float(data["decay_status"]),
            current_states=dict(data.get("current_states", {})),
            previous_states=dict(data.get("previous_states", {})),
            is_occupied=bool(data["is_occupied"]),
            decaying=bool(data.get("decaying", False)),
            decay_start_time=decay_start_time,
            decay_start_probability=decay_start_probability,
        )


@dataclass
class PriorState:
    """State of prior probability calculations."""

    # Overall prior probability for the area
    overall_prior: float = field(default=0.0)

    # Prior probabilities by sensor type
    motion_prior: float = field(default=0.0)
    media_prior: float = field(default=0.0)
    appliance_prior: float = field(default=0.0)
    door_prior: float = field(default=0.0)
    window_prior: float = field(default=0.0)
    light_prior: float = field(default=0.0)
    environmental_prior: float = field(default=0.0)

    # Individual entity priors
    entity_priors: dict[str, dict[str, float]] = field(default_factory=dict)

    # Type-level priors with full probability data
    type_priors: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Last updated timestamps
    last_updated: dict[str, str] = field(default_factory=dict)

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
            if not 0 <= value <= 1:
                raise ValueError(f"{attr_name} must be between 0 and 1")

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
        entity_priors: dict[str, dict[str, float]] | None = None,
        last_updated: dict[str, str] | None = None,
        analysis_period: int | None = None,
    ) -> None:
        """Update the state with new values while maintaining the same instance."""
        if overall_prior is not None:
            self.overall_prior = max(0.0, min(overall_prior, 1.0))
        if motion_prior is not None:
            self.motion_prior = max(0.0, min(motion_prior, 1.0))
        if media_prior is not None:
            self.media_prior = max(0.0, min(media_prior, 1.0))
        if appliance_prior is not None:
            self.appliance_prior = max(0.0, min(appliance_prior, 1.0))
        if door_prior is not None:
            self.door_prior = max(0.0, min(door_prior, 1.0))
        if window_prior is not None:
            self.window_prior = max(0.0, min(window_prior, 1.0))
        if light_prior is not None:
            self.light_prior = max(0.0, min(light_prior, 1.0))
        if environmental_prior is not None:
            self.environmental_prior = max(0.0, min(environmental_prior, 1.0))
        if entity_priors is not None:
            self.entity_priors = entity_priors
        if last_updated is not None:
            self.last_updated = last_updated
        if analysis_period is not None:
            self.analysis_period = analysis_period

    def update_entity_prior(
        self,
        entity_id: str,
        prob_given_true: float,
        prob_given_false: float,
        prior: float,
        timestamp: str,
    ) -> None:
        """Update prior for a specific entity."""
        self.entity_priors[entity_id] = {
            "prob_given_true": max(0.0, min(prob_given_true, 1.0)),
            "prob_given_false": max(0.0, min(prob_given_false, 1.0)),
            "prior": max(0.0, min(prior, 1.0)),
        }
        self.last_updated[entity_id] = timestamp

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
            sensor_type: The type of sensor to update
            prior_value: The prior probability value
            timestamp: The timestamp of the update
            prob_given_true: Optional P(sensor active | area occupied)
            prob_given_false: Optional P(sensor active | area not occupied)

        """
        # Update the prior value
        if sensor_type == "motion":
            self.motion_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "media":
            self.media_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "appliance":
            self.appliance_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "door":
            self.door_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "window":
            self.window_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "light":
            self.light_prior = max(0.0, min(prior_value, 1.0))
        elif sensor_type == "environmental":
            self.environmental_prior = max(0.0, min(prior_value, 1.0))

        # Store the update timestamp
        self.last_updated[sensor_type] = timestamp

        # Create or update type prior entry
        type_prior = {
            "prior": prior_value,
            "last_updated": timestamp,
        }

        # Add conditional probabilities if provided
        if prob_given_true is not None:
            type_prior["prob_given_true"] = max(0.0, min(prob_given_true, 1.0))
        if prob_given_false is not None:
            type_prior["prob_given_false"] = max(0.0, min(prob_given_false, 1.0))

        # Store the complete type prior
        self.type_priors[sensor_type] = type_prior

        # Recalculate overall prior
        self.overall_prior = self.calculate_overall_prior()

    def calculate_overall_prior(self, probabilities=None) -> float:
        """Calculate the overall prior from sensor type priors that have configured sensors.

        Args:
            probabilities: Optional probabilities instance from coordinator

        Returns:
            Average prior probability from configured sensor types

        """
        if not probabilities:
            # Fall back to simple calculation if no probabilities instance
            type_priors = [
                self.motion_prior,
                self.media_prior,
                self.appliance_prior,
                self.door_prior,
                self.window_prior,
                self.light_prior,
                self.environmental_prior,
            ]
            valid_priors = [p for p in type_priors if p > 0]
            if not valid_priors:
                return 0.0
            return sum(valid_priors) / len(valid_priors)

        # Get all configured entity types from entity_priors
        configured_types = set()
        for entity_id, prior_data in self.entity_priors.items():
            # Skip entities with no prior data
            if not prior_data:
                continue

            # Get entity type from probabilities
            if entity_type := probabilities.get_entity_type(entity_id):
                configured_types.add(entity_type)

        # Only include priors for configured types
        valid_priors = []
        if "motion" in configured_types and self.motion_prior > 0:
            valid_priors.append(self.motion_prior)
        if "media" in configured_types and self.media_prior > 0:
            valid_priors.append(self.media_prior)
        if "appliance" in configured_types and self.appliance_prior > 0:
            valid_priors.append(self.appliance_prior)
        if "door" in configured_types and self.door_prior > 0:
            valid_priors.append(self.door_prior)
        if "window" in configured_types and self.window_prior > 0:
            valid_priors.append(self.window_prior)
        if "light" in configured_types and self.light_prior > 0:
            valid_priors.append(self.light_prior)
        if "environmental" in configured_types and self.environmental_prior > 0:
            valid_priors.append(self.environmental_prior)

        if not valid_priors:
            return 0.0

        return sum(valid_priors) / len(valid_priors)

    def initialize_from_defaults(self, probabilities) -> None:
        """Initialize the state from default type priors.

        Args:
            probabilities: The probabilities configuration instance

        """
        timestamp = dt_util.utcnow().isoformat()

        # Get configured sensor types
        configured_types = set()
        for entity_id in probabilities.entity_types:
            if entity_type := probabilities.get_entity_type(entity_id):
                configured_types.add(entity_type)

        # Initialize only configured type priors
        for sensor_type, prior_data in probabilities.get_initial_type_priors().items():
            if sensor_type in configured_types:
                self.update_type_prior(
                    sensor_type, prior_data.get("prior", MIN_PROBABILITY), timestamp
                )
            else:
                # Set unconfigured types to 0
                self.update_type_prior(sensor_type, 0.0, timestamp)

        # Calculate and set the overall prior
        overall_prior = self.calculate_overall_prior(probabilities)
        self.update(overall_prior=overall_prior)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        # Build base dictionary
        result = {
            "overall_prior": self.overall_prior,
            "entity_priors": self.entity_priors,
            "last_updated": self.last_updated,
            "analysis_period": self.analysis_period,
        }

        # Include type priors that have non-zero values
        if self.motion_prior > 0:
            result["motion_prior"] = self.motion_prior
        if self.media_prior > 0:
            result["media_prior"] = self.media_prior
        if self.appliance_prior > 0:
            result["appliance_prior"] = self.appliance_prior
        if self.door_prior > 0:
            result["door_prior"] = self.door_prior
        if self.window_prior > 0:
            result["window_prior"] = self.window_prior
        if self.light_prior > 0:
            result["light_prior"] = self.light_prior
        if self.environmental_prior > 0:
            result["environmental_prior"] = self.environmental_prior

        # Include type_priors if not empty
        if self.type_priors:
            result["type_priors"] = self.type_priors

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorState:
        """Create a PriorState from a dictionary."""
        instance = cls(
            overall_prior=float(data.get("overall_prior", 0.0)),
            motion_prior=float(data.get("motion_prior", 0.0)),
            media_prior=float(data.get("media_prior", 0.0)),
            appliance_prior=float(data.get("appliance_prior", 0.0)),
            door_prior=float(data.get("door_prior", 0.0)),
            window_prior=float(data.get("window_prior", 0.0)),
            light_prior=float(data.get("light_prior", 0.0)),
            environmental_prior=float(data.get("environmental_prior", 0.0)),
            entity_priors=dict(data.get("entity_priors", {})),
            last_updated=dict(data.get("last_updated", {})),
            analysis_period=int(data.get("analysis_period", 7)),
        )

        # Restore type_priors if present
        if "type_priors" in data:
            instance.type_priors = dict(data["type_priors"])

        return instance


class SensorState(TypedDict):
    """Current state information for a sensor.

    Required fields:
        state: Current state value as a string
        attributes: Dictionary of state attributes
        availability: Whether the sensor is available

    Optional fields:
        last_updated: ISO formatted timestamp of last update
    """

    state: str
    attributes: dict[str, str | float | bool]
    availability: bool
    last_updated: NotRequired[str | None]


class ProbabilityConfig(TypedDict, total=False):
    """Configuration for probability calculations.

    Required fields:
        prob_given_true: Probability when condition is true/area is occupied
        prob_given_false: Probability when condition is false/area is not occupied
        prior: Prior probability value

    Optional fields:
        weight: Weight factor for sensor calculations
        last_updated: Timestamp of last update
    """

    prob_given_true: float
    prob_given_false: float
    prior: float
    weight: NotRequired[float]
    last_updated: NotRequired[str]


class TimeInterval(TypedDict, total=False):
    """Time interval and duration data.

    Required fields:
        start: Start time of the interval
        end: End time of the interval

    Optional fields:
        state: State value during the interval
        total_active_time: Total time active
        total_inactive_time: Total time inactive
        total_time: Total time period analyzed
    """

    start: datetime
    end: datetime
    state: NotRequired[Any]
    total_active_time: NotRequired[float]
    total_inactive_time: NotRequired[float]
    total_time: NotRequired[float]


class ProbabilityAttributes(TypedDict):
    """Attributes for the probability sensor.

    Optional fields:
        active_triggers: List of currently active triggers
        sensor_probabilities: Set of sensor probability strings
        threshold: Threshold value as string
    """

    active_triggers: NotRequired[list[str]]
    sensor_probabilities: NotRequired[set[str]]
    threshold: NotRequired[str]


class PriorsAttributes(TypedDict):
    """Attributes for the priors sensor.

    Optional fields:
        motion_prior: Motion sensor prior probability
        media_prior: Media device prior probability
        appliance_prior: Appliance prior probability
        door_prior: Door sensor prior probability
        window_prior: Window sensor prior probability
        light_prior: Light prior probability
        last_updated: Last update timestamp
        total_period: Total analysis period
    """

    motion_prior: NotRequired[str]
    media_prior: NotRequired[str]
    appliance_prior: NotRequired[str]
    door_prior: NotRequired[str]
    window_prior: NotRequired[str]
    light_prior: NotRequired[str]
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
