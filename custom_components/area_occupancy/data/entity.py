"""Entity model."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import MAX_WEIGHT, MIN_WEIGHT
from ..utils import clamp_probability, ensure_timezone_aware
from .decay import Decay
from .entity_type import DEFAULT_TYPES, EntityType, InputType

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from ..db import AreaOccupancyDB as DB

_LOGGER = logging.getLogger(__name__)


@dataclass
class Entity:
    """Type for sensor state information."""

    # --- Core Data ---
    entity_id: str
    type: EntityType
    prob_given_true: float
    prob_given_false: float
    decay: Decay
    hass: HomeAssistant | None = None
    state_provider: Callable[[str], Any] | None = None
    last_updated: datetime | None = None
    previous_evidence: bool | None = None

    def __post_init__(self) -> None:
        """Validate that either hass or state_provider is provided.

        Either hass or state_provider must be provided, and they are mutually
        exclusive. This ensures proper state retrieval behavior:

        - **HA-backed instances**: Provide `hass` to use Home Assistant's state
          registry for entity state retrieval. This is the standard usage pattern
          for entities managed within a Home Assistant integration.

        - **State-provider-only instances**: Provide `state_provider` (a callable
          that takes an entity_id and returns state) for testing or external state
          management scenarios where Home Assistant is not available.

        Raises:
            ValueError: If neither hass nor state_provider is provided, or if both
                are provided simultaneously.
        """
        if self.hass is None and self.state_provider is None:
            raise ValueError("Either hass or state_provider must be provided")
        if self.hass is not None and self.state_provider is not None:
            raise ValueError("Cannot provide both hass and state_provider")
        if self.last_updated is None:
            self.last_updated = dt_util.utcnow()

    @property
    def name(self) -> str | None:
        """Get the entity name from Home Assistant state or state provider."""
        if self.state_provider:
            state_obj = self.state_provider(self.entity_id)
            if state_obj and hasattr(state_obj, "name"):
                return state_obj.name
            return None
        ha_state = self.hass.states.get(self.entity_id)
        return ha_state.name if ha_state else None

    @property
    def available(self) -> bool:
        """Get the entity availability."""
        return self.state is not None

    @property
    def state(self) -> str | float | bool | None:
        """Get the entity state from Home Assistant or state provider."""
        if self.state_provider:
            state_obj = self.state_provider(self.entity_id)
            if state_obj is None:
                return None
            # Handle both object with .state attribute and direct value
            if hasattr(state_obj, "state"):
                state_value = state_obj.state
            else:
                state_value = state_obj
        else:
            ha_state = self.hass.states.get(self.entity_id)
            if ha_state is None:
                return None
            state_value = ha_state.state

        # Check if state is valid
        if state_value in [
            "unknown",
            "unavailable",
            None,
            "",
            "NaN",
        ]:
            return None
        return state_value

    @property
    def weight(self) -> float:
        """Get the entity weight."""
        return self.type.weight

    @property
    def evidence(self) -> bool | None:
        """Determine if entity is active.

        Returns:
            bool | None: True if entity is active, False if inactive, None if state unknown

        """
        if self.state is None:
            return None

        if self.active_states:
            return str(self.state) in self.active_states
        if self.active_range:
            min_val, max_val = self.active_range
            try:
                return min_val <= float(self.state) <= max_val
            except (ValueError, TypeError):
                return False

        return None

    @property
    def active(self) -> bool:
        """Get the entity active status."""
        return self.evidence or self.decay.is_decaying

    @property
    def active_states(self) -> list[str] | None:
        """Get the active states for the entity."""
        return self.type.active_states

    @property
    def active_range(self) -> tuple[float, float] | None:
        """Get the active range for the entity."""
        return self.type.active_range

    @property
    def decay_factor(self) -> float:
        """Get decay factor that considers current evidence state.

        Returns 1.0 if evidence is currently True, otherwise returns the normal decay factor.
        This prevents inconsistent states where evidence is True but decay is being applied.
        """
        if self.evidence is True:
            return 1.0
        return self.decay.decay_factor

    def update_likelihood(
        self, prob_given_true: float, prob_given_false: float
    ) -> None:
        """Update the likelihood of the entity."""
        self.prob_given_true = clamp_probability(prob_given_true)
        self.prob_given_false = clamp_probability(prob_given_false)
        self.last_updated = dt_util.utcnow()

    def update_decay(self, decay_start: datetime, is_decaying: bool) -> None:
        """Update the decay of the entity."""
        self.decay.decay_start = decay_start
        self.decay.is_decaying = is_decaying

    def has_new_evidence(self) -> bool:
        """Update decay on actual evidence transitions.

        Returns:
            bool: True if evidence transition occurred, False otherwise

        """
        # Pure calculation from current HA state
        current_evidence = self.evidence

        # Capture previous evidence before updating it
        previous_evidence = self.previous_evidence

        # Skip transition logic if current evidence is None (entity unavailable)
        if current_evidence is None or previous_evidence is None:
            # Update previous evidence even if skipping to prevent false transitions later
            self.previous_evidence = current_evidence
            return False

        # Fix inconsistent state: if evidence is True but decay is running, stop decay
        if current_evidence and self.decay.is_decaying:
            self.decay.stop_decay()

        # Check for evidence transitions
        transition_occurred = current_evidence != previous_evidence

        # Handle evidence transitions
        if transition_occurred:
            self.last_updated = dt_util.utcnow()
            if current_evidence:  # FALSE→TRUE transition
                self.decay.stop_decay()
            else:  # TRUE→FALSE transition
                # Evidence lost - start decay
                self.decay.start_decay()

        # Update previous evidence for next comparison
        self.previous_evidence = current_evidence
        return transition_occurred


class EntityFactory:
    """Factory for creating entities from various sources."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
        area_name: str,
    ) -> None:
        """Initialize the factory.

        Args:
            coordinator: The coordinator instance
            area_name: Area name for multi-area support
        """
        self.coordinator = coordinator
        self.area_name = area_name
        # Validate area_name exists and retrieve config from coordinator.areas
        if area_name not in coordinator.areas:
            available = list(coordinator.areas.keys())
            raise ValueError(
                f"Area '{area_name}' not found. "
                f"Available areas: {available if available else '(none)'}"
            )
        self.config = coordinator.areas[area_name].config

    def create_from_db(self, entity_obj: "DB.Entities") -> Entity:
        """Create entity from storage data.

        Args:
            entity_obj: SQLAlchemy Entities object from database

        Returns:
            Entity: Properly constructed Entity object with Python types

        Raises:
            ValueError: If required fields are missing or invalid
            TypeError: If type conversion fails
        """
        # Convert SQLAlchemy objects to Python types using to_dict()
        # This ensures we get proper Python types rather than SQLAlchemy Column objects
        entity_data = entity_obj.to_dict()

        # Extract and validate required string fields
        entity_id = str(entity_data["entity_id"])
        entity_type_str = str(entity_data["entity_type"])

        if not entity_id:
            raise ValueError("Entity ID cannot be empty")
        if not entity_type_str:
            raise ValueError("Entity type cannot be empty")

        # Convert numeric fields with validation
        try:
            db_weight = float(entity_data["weight"])
            prob_given_true = float(entity_data["prob_given_true"])
            prob_given_false = float(entity_data["prob_given_false"])
        except (TypeError, ValueError) as e:
            raise TypeError(f"Failed to convert numeric fields: {e}") from e

        # Convert boolean field
        is_decaying = bool(entity_data["is_decaying"])

        # Handle datetime fields - ensure they're proper Python datetime objects
        decay_start = entity_data["decay_start"]
        last_updated = entity_data["last_updated"]

        # Validate datetime objects are timezone-aware
        if decay_start is not None:
            decay_start = ensure_timezone_aware(decay_start)

        if last_updated is not None:
            last_updated = ensure_timezone_aware(last_updated)

        # Convert evidence field - handle None case
        previous_evidence = entity_data["evidence"]
        if previous_evidence is not None:
            previous_evidence = bool(previous_evidence)

        # Create the entity type directly
        input_type = InputType(entity_type_str)

        # Extract overrides from config
        config_weight = None
        active_states = None
        active_range = None

        weights = getattr(self.config, "weights", None)
        if weights:
            weight_attr = getattr(weights, input_type.value, None)
            if weight_attr is not None:
                config_weight = weight_attr

        sensor_states = getattr(self.config, "sensor_states", None)
        if sensor_states:
            states_attr = getattr(sensor_states, input_type.value, None)
            if states_attr is not None:
                active_states = states_attr

        range_config_attr = f"{input_type.value}_active_range"
        range_attr = getattr(self.config, range_config_attr, None)
        if range_attr is not None:
            active_range = range_attr

        entity_type = EntityType(
            input_type,
            weight=config_weight,
            active_states=active_states,
            active_range=active_range,
        )

        # DB weight should take priority over configured default
        try:
            if MIN_WEIGHT <= db_weight <= MAX_WEIGHT:
                entity_type.weight = db_weight
        except (TypeError, ValueError):
            # Weight is invalid, keep the default from EntityType initialization
            pass

        # Motion sensors use configured likelihoods (user-configurable per area)
        # Do not use learned likelihoods from database for motion sensors
        if input_type == InputType.MOTION:
            # Get configured values from area config, fall back to defaults if not configured
            motion_prob_given_true = getattr(
                self.config.sensors,
                "motion_prob_given_true",
                DEFAULT_TYPES[InputType.MOTION]["prob_given_true"],
            )
            motion_prob_given_false = getattr(
                self.config.sensors,
                "motion_prob_given_false",
                DEFAULT_TYPES[InputType.MOTION]["prob_given_false"],
            )
            prob_given_true = float(motion_prob_given_true)
            prob_given_false = float(motion_prob_given_false)
            _LOGGER.debug(
                "Using configured likelihoods for motion sensor %s: prob_given_true=%.2f, prob_given_false=%.2f",
                entity_id,
                prob_given_true,
                prob_given_false,
            )

        # Create decay object
        # Wasp-in-Box sensors should not have decay (immediate vacancy)
        half_life = self.config.decay.half_life

        area = self.coordinator.areas.get(self.area_name)
        is_wasp = area and area.wasp_entity_id == entity_id
        if is_wasp:
            half_life = 0.1  # Effectively zero decay (clears in <0.5s)

        # Get sleep settings from integration config
        # For WASP entities, bypass sleeping semantics to ensure immediate vacancy
        if is_wasp:
            purpose_for_decay = None
            sleep_start = None
            sleep_end = None
        else:
            sleep_start = getattr(
                self.coordinator.integration_config, "sleep_start", None
            )
            sleep_end = getattr(self.coordinator.integration_config, "sleep_end", None)
            purpose_for_decay = getattr(self.config, "purpose", None)

        decay = Decay(
            half_life=half_life,
            is_decaying=is_decaying,
            decay_start=decay_start,
            purpose=purpose_for_decay,
            sleep_start=sleep_start,
            sleep_end=sleep_end,
        )

        return Entity(
            entity_id=entity_id,
            type=entity_type,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            decay=decay,
            hass=self.coordinator.hass,
            last_updated=last_updated,
            previous_evidence=previous_evidence,
        )

    def create_from_config_spec(self, entity_id: str, input_type: str) -> Entity:
        """Create entity from configuration specification."""
        # Create the entity type directly
        input_type_enum = InputType(input_type)

        # Extract overrides from config
        weight = None
        active_states = None
        active_range = None

        weights = getattr(self.config, "weights", None)
        if weights:
            weight_attr = getattr(weights, input_type_enum.value, None)
            if weight_attr is not None:
                weight = weight_attr

        sensor_states = getattr(self.config, "sensor_states", None)
        if sensor_states:
            states_attr = getattr(sensor_states, input_type_enum.value, None)
            if states_attr is not None:
                active_states = states_attr

        range_config_attr = f"{input_type_enum.value}_active_range"
        range_attr = getattr(self.config, range_config_attr, None)
        if range_attr is not None:
            active_range = range_attr

        entity_type = EntityType(
            input_type_enum,
            weight=weight,
            active_states=active_states,
            active_range=active_range,
        )

        # Wasp-in-Box sensors should not have decay (immediate vacancy)
        half_life = self.config.decay.half_life
        area = self.coordinator.areas.get(self.area_name)
        is_wasp = area and area.wasp_entity_id == entity_id
        if is_wasp:
            half_life = 0.1  # Effectively zero decay (clears in <0.5s)

        # Get sleep settings from integration config
        # For WASP entities, bypass sleeping semantics to ensure immediate vacancy
        if is_wasp:
            purpose_for_decay = None
            sleep_start = None
            sleep_end = None
        else:
            sleep_start = getattr(
                self.coordinator.integration_config, "sleep_start", None
            )
            sleep_end = getattr(self.coordinator.integration_config, "sleep_end", None)
            purpose_for_decay = getattr(self.config, "purpose", None)

        decay = Decay(
            half_life=half_life,
            is_decaying=False,
            decay_start=dt_util.utcnow(),
            purpose=purpose_for_decay,
            sleep_start=sleep_start,
            sleep_end=sleep_end,
        )

        # Motion sensors use configured likelihoods (user-configurable per area)
        # Other sensors use defaults from EntityType
        if input_type_enum == InputType.MOTION:
            motion_prob_given_true = getattr(
                self.config.sensors,
                "motion_prob_given_true",
                entity_type.prob_given_true,
            )
            motion_prob_given_false = getattr(
                self.config.sensors,
                "motion_prob_given_false",
                entity_type.prob_given_false,
            )
            prob_given_true = float(motion_prob_given_true)
            prob_given_false = float(motion_prob_given_false)
        else:
            prob_given_true = entity_type.prob_given_true
            prob_given_false = entity_type.prob_given_false

        return Entity(
            entity_id=entity_id,
            type=entity_type,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            decay=decay,
            hass=self.coordinator.hass,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
        )

    def create_all_from_config(self) -> dict[str, Entity]:
        """Create all entities from current configuration."""
        entity_type_mapping = self.get_entity_type_mapping()
        entities = {}

        for entity_id, input_type in entity_type_mapping.items():
            _LOGGER.debug("Creating entity %s with type %s", entity_id, input_type)
            entities[entity_id] = self.create_from_config_spec(entity_id, input_type)

        return entities

    def get_entity_type_mapping(self) -> dict[str, str]:
        """Get entity type mapping for all configured entities.

        Returns a mapping of entity_id -> input_type string that can be used
        directly for entity creation.
        """
        specs = {}

        # Define sensor type mappings to eliminate repetition
        SENSOR_TYPE_MAPPING = {
            "motion": InputType.MOTION,
            "media": InputType.MEDIA,
            "appliance": InputType.APPLIANCE,
            "door": InputType.DOOR,
            "window": InputType.WINDOW,
            "illuminance": InputType.ILLUMINANCE,
            "humidity": InputType.HUMIDITY,
            "temperature": InputType.TEMPERATURE,
            "co2": InputType.CO2,
            "sound_pressure": InputType.SOUND_PRESSURE,
            "pressure": InputType.PRESSURE,
            "air_quality": InputType.AIR_QUALITY,
            "voc": InputType.VOC,
            "pm25": InputType.PM25,
            "pm10": InputType.PM10,
            "energy": InputType.ENERGY,
        }

        # Process each sensor type using the mapping
        for sensor_type, input_type in SENSOR_TYPE_MAPPING.items():
            sensor_list = getattr(self.config.sensors, sensor_type)

            # Special handling for motion sensors (includes wasp)
            if sensor_type == "motion":
                sensor_list = self.config.sensors.get_motion_sensors(self.coordinator)

            for entity_id in sensor_list:
                specs[entity_id] = input_type.value

        return specs


class EntityManager:
    """Manages entities with simplified creation and storage logic."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
        area_name: str | None = None,
    ) -> None:
        """Initialize the entity manager.

        Args:
            coordinator: The coordinator instance
            area_name: Required area name for multi-area support. Used to look up
                the area configuration from coordinator.areas.
        """
        self.coordinator = coordinator
        self.area_name = area_name
        # Validate area_name and retrieve config from coordinator.areas
        if not area_name:
            raise ValueError("Area name is required in multi-area architecture")
        if area_name not in coordinator.areas:
            available = list(coordinator.areas.keys())
            raise ValueError(
                f"Area '{area_name}' not found. "
                f"Available areas: {available if available else '(none)'}"
            )
        self.config = coordinator.areas[area_name].config
        self.hass = coordinator.hass
        self._factory = EntityFactory(coordinator, area_name=area_name)
        self._entities: dict[str, Entity] = self._factory.create_all_from_config()

    @property
    def entities(self) -> dict[str, Entity]:
        """Get the entities."""
        return self._entities

    def get_entities_by_input_type(
        self, input_type: "InputType"
    ) -> dict[str, "Entity"]:
        """Get entities filtered by InputType."""
        return {
            entity_id: entity
            for entity_id, entity in self._entities.items()
            if entity.type.input_type == input_type
        }

    @property
    def entity_ids(self) -> list[str]:
        """Get the entity IDs."""
        return list(self._entities.keys())

    @property
    def active_entities(self) -> list[Entity]:
        """Get the active entities."""
        return [
            entity
            for entity in self._entities.values()
            if entity.evidence or entity.decay.is_decaying
        ]

    @property
    def inactive_entities(self) -> list[Entity]:
        """Get the inactive entities."""
        return [
            entity
            for entity in self._entities.values()
            if not entity.evidence and not entity.decay.is_decaying
        ]

    @property
    def decaying_entities(self) -> list[Entity]:
        """Get the decaying entities."""
        return [
            entity for entity in self._entities.values() if entity.decay.is_decaying
        ]

    def get_entity(self, entity_id: str) -> Entity:
        """Get the entity from an entity ID."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found for entity: {entity_id}")
        return self._entities[entity_id]

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the manager."""
        self._entities[entity.entity_id] = entity

    async def cleanup(self) -> None:
        """Clean up resources and recreate from config.

        This method clears all entity references to release memory
        and prevent leaks when areas are removed or reconfigured.
        """
        _LOGGER.debug("Cleaning up EntityManager for area: %s", self.area_name)
        # Clear all entity references to release memory
        # This ensures entities and their internal state (decay, etc.) are released
        self._entities.clear()
        # Recreate entities from config (needed for reconfiguration scenarios)
        self._entities = self._factory.create_all_from_config()
        _LOGGER.debug("EntityManager cleanup completed for area: %s", self.area_name)
