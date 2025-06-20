"""Entity model."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from ..utils import bayesian_probability, validate_datetime
from .decay import Decay
from .entity_type import EntityType, InputType
from .prior import Prior

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass
class Entity:
    """Type for sensor state information."""

    # --- Public State ---
    entity_id: str
    probability: float
    type: EntityType
    prior: Prior
    decay: Decay
    state: str | float | bool | None = None
    evidence: bool = False
    available: bool = True
    # --- Internal State ---
    previous_evidence: bool = False
    previous_probability: float = field(default=0.0, init=False, repr=False)
    last_updated: datetime = field(default_factory=dt_util.utcnow)

    def __post_init__(self):
        """Post init."""
        # Validate last_updated
        self.last_updated = validate_datetime(self.last_updated)

    def update_probability(self) -> bool:
        """Calculate and update entity probability based on current state.

        Note: Caller must capture previous state before updating current state
        by calling _capture_previous_state() or manually setting previous_* fields.

        Returns:
            bool: True if probability changed significantly (> 0.001), False otherwise

        """
        # Update timestamp
        self.last_updated = dt_util.utcnow()

        # Handle state transitions and decay
        state_edge = self._handle_state_transition()

        # Update probability only if there's evidence (state change)
        if state_edge is not None:
            return self._calculate_new_probability(state_edge)

        return False

    def capture_previous_state(self) -> None:
        """Capture current state as previous for comparison.

        This should be called BEFORE updating the current state values.
        """
        self.previous_probability = self.probability
        self.previous_evidence = self.evidence

    def _handle_state_transition(self) -> bool | None:
        """Detect edge and (start/stop) decay.

        Returns:
            True   → rising edge  (OFF→ON)
            False  → falling edge (ON→OFF)
            None   → no change

        """
        if self.evidence == self.previous_evidence:
            return None  # nothing new

        state_edge = self.evidence

        if state_edge:  # rising edge → stop decay
            self.decay.is_decaying = False
        else:  # first falling edge of burst
            self.decay.start_decay()  # starts timer only once

        return state_edge

    def _calculate_new_probability(self, evidence: bool) -> bool:
        """Calculate new probability using Bayesian update.

        Args:
            evidence: Whether this is a rising (True) or falling (False) edge

        Returns:
            bool: True if probability changed significantly

        """
        # Calculate new probability
        self.probability = bayesian_probability(
            prior=self.probability,
            prob_given_true=self.prior.prob_given_true,
            prob_given_false=self.prior.prob_given_false,
            evidence=evidence,
            weight=self.type.weight,
            decay_factor=self.decay.decay_factor if self.decay.is_decaying else 1.0,
        )

        # Check if probability changed significantly
        probability_changed = abs(self.probability - self.previous_probability) > 0.001

        if probability_changed:
            _LOGGER.debug(
                "Entity %s probability changed from %.3f to %.3f (edge: %s)",
                self.entity_id,
                self.previous_probability,
                self.probability,
                "ON" if evidence else "OFF",
            )

        return probability_changed

    def stop_decay_completely(self) -> None:
        """Stop decay completely (public interface)."""
        self.decay.is_decaying = False

    def cleanup(self) -> None:
        """Clean up entity resources."""
        self.stop_decay_completely()

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for storage."""
        return {
            "entity_id": self.entity_id,
            "probability": self.probability,
            "type": self.type.to_dict(),
            "prior": self.prior.to_dict(),
            "decay": self.decay.to_dict(),
            "state": self.state,
            "evidence": self.evidence,
            "available": self.available,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create entity from dictionary."""
        entity = cls(
            entity_id=data["entity_id"],
            probability=float(data["probability"]),
            type=EntityType.from_dict(data["type"]),
            prior=Prior.from_dict(data["prior"]),
            decay=Decay.from_dict(data["decay"]),
            state=data["state"],
            evidence=data["evidence"],
            available=data["available"],
        )

        # Initialize the previous state fields to current values for restored entities
        entity.previous_probability = entity.probability
        entity.previous_evidence = entity.previous_evidence

        return entity


class EntityManager:
    """Manages entities."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
    ) -> None:
        """Initialize the entities."""
        self.coordinator = coordinator
        self.config = coordinator.config_manager.config
        self.hass = coordinator.hass
        self._entities: dict[str, Entity] = {}

    @property
    def entities(self) -> dict[str, Entity]:
        """Get the entities."""
        return self._entities

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

    def to_dict(self) -> dict[str, Any]:
        """Convert entity manager to dictionary for storage."""
        return {
            "entities": {
                entity_id: entity.to_dict()
                for entity_id, entity in self._entities.items()
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: "AreaOccupancyCoordinator"
    ) -> "EntityManager":
        """Create entity manager from dictionary.

        Only accepts the format: {"entities": {entity_id: entity_dict, ...}}
        """
        manager = cls(coordinator=coordinator)

        # Only accept new format with 'entities' key
        if "entities" not in data:
            raise ValueError(
                f"Invalid storage format: missing 'entities' key in data structure. "
                f"Available keys: {list(data.keys())}. "
                f"This should have been caught by storage validation."
            )

        entities_data = data["entities"]

        try:
            manager._entities = {
                entity_id: Entity.from_dict(entity)
                for entity_id, entity in entities_data.items()
            }
        except (KeyError, ValueError, TypeError) as err:
            raise ValueError(
                f"Failed to deserialize entity data: {err}. "
                f"Entity structure may be corrupted or incompatible."
            ) from err

        return manager

    async def async_initialize(self) -> None:
        """Initialize the entities with proper restoration support."""
        # If we have restored entities, update them with current config
        if self._entities:
            _LOGGER.debug(
                "Found %d restored entities, updating with current config",
                len(self._entities),
            )
            await self._update_entities_from_config()
        else:
            _LOGGER.debug(
                "No restored entities found, creating fresh entities from config"
            )
            self._entities = await self._create_entities_from_config()

        _LOGGER.debug("EntityManager initialized with %d entities", len(self._entities))

    async def _update_entities_from_config(self) -> None:
        """Update existing entities with current configuration."""
        config_entities = await self._get_config_entity_mapping()
        updated_entities: dict[str, Entity] = {}

        # Process existing entities and mark those that are retained
        self._process_existing_entities(config_entities, updated_entities)

        # Create new entities for those added to configuration
        await self._create_new_entities(config_entities, updated_entities)

        # Apply the updated entity collection
        self._entities = updated_entities
        _LOGGER.info("Entity update complete: %d total entities", len(self._entities))

    def _process_existing_entities(
        self,
        config_entities: dict[str, EntityType],
        updated_entities: dict[str, Entity],
    ) -> None:
        """Process existing restored entities, updating configuration and marking processed ones.

        Args:
            config_entities: Configuration entity mapping (modified in place to mark processed)
            updated_entities: Collection to add retained entities to

        """
        for entity_id, restored_entity in self._entities.items():
            if entity_id in config_entities:
                # Entity still exists in config - update type configuration
                current_type = config_entities[entity_id]

                # Update type configuration but preserve learned data
                restored_entity.type.weight = current_type.weight
                restored_entity.type.active_states = current_type.active_states
                restored_entity.type.active_range = current_type.active_range

                updated_entities[entity_id] = restored_entity
                del config_entities[entity_id]  # Mark as processed
            else:
                _LOGGER.info(
                    "Entity %s removed from configuration, dropping stored data",
                    entity_id,
                )

    async def _create_new_entities(
        self,
        config_entities: dict[str, EntityType],
        updated_entities: dict[str, Entity],
    ) -> None:
        """Create new entities for those added to configuration.

        Args:
            config_entities: Remaining unprocessed entities from configuration
            updated_entities: Collection to add new entities to

        """
        for entity_id, entity_type in config_entities.items():
            _LOGGER.info("Creating new entity %s", entity_id)
            prior = await self._calculate_initial_prior(entity_id, entity_type)
            updated_entities[entity_id] = await self.create_entity(
                entity_id=entity_id,
                entity_type=entity_type,
                prior=prior,
            )

    async def _get_config_entity_mapping(self) -> dict[str, EntityType]:
        """Get mapping of entity_id -> EntityType from current configuration."""
        # Build sensor type mappings from configuration
        type_mappings = self.build_sensor_type_mappings()

        # Create initial entity mapping from type mappings
        entity_mapping = self._build_entity_mapping_from_types(type_mappings)

        # Add primary sensor with special handling
        self._add_primary_sensor_to_mapping(entity_mapping)

        return entity_mapping

    def build_sensor_type_mappings(self) -> dict[InputType, list[str]]:
        """Build mapping of InputType -> list of entity IDs from configuration.

        Returns:
            Dictionary mapping sensor types to their configured entity IDs

        """
        return {
            InputType.MOTION: self.config.sensors.get_motion_sensors(self.coordinator),
            InputType.MEDIA: self.config.sensors.media,
            InputType.APPLIANCE: self.config.sensors.appliances,
            InputType.DOOR: self.config.sensors.doors,
            InputType.WINDOW: self.config.sensors.windows,
            InputType.LIGHT: self.config.sensors.lights,
            InputType.ENVIRONMENTAL: self.config.sensors.illuminance
            + self.config.sensors.humidity
            + self.config.sensors.temperature,
        }

    def _build_entity_mapping_from_types(
        self, type_mappings: dict[InputType, list[str]]
    ) -> dict[str, EntityType]:
        """Build entity_id -> EntityType mapping from type mappings.

        Args:
            type_mappings: Dictionary of InputType -> list of entity IDs

        Returns:
            Dictionary mapping entity IDs to their EntityType configurations

        """
        entity_mapping: dict[str, EntityType] = {}

        for input_type, entity_ids in type_mappings.items():
            entity_type = self.coordinator.entity_types.get_entity_type(input_type)
            for entity_id in entity_ids:
                entity_mapping[entity_id] = entity_type

        return entity_mapping

    def _add_primary_sensor_to_mapping(
        self, entity_mapping: dict[str, EntityType]
    ) -> None:
        """Add primary sensor to entity mapping with special motion-type handling.

        Args:
            entity_mapping: Entity mapping to modify in place

        Raises:
            ValueError: If primary occupancy sensor is not configured

        """
        primary_sensor = self.config.sensors.primary_occupancy
        if not primary_sensor:
            raise ValueError("Primary occupancy sensor must be configured")

        # Primary sensor always gets motion-type treatment (highest reliability)
        primary_entity_type = self.coordinator.entity_types.get_entity_type(
            InputType.MOTION
        )

        # Override any existing mapping (primary sensor takes precedence)
        entity_mapping[primary_sensor] = primary_entity_type

        _LOGGER.debug("Added primary sensor %s as motion-type entity", primary_sensor)

    async def _calculate_initial_prior(
        self, entity_id: str, entity_type: EntityType
    ) -> Prior:
        """Calculate initial prior for a new entity."""

        try:
            return await self.coordinator.priors.calculate(
                entity=self.get_entity(entity_id),
            )
        except (
            ValueError,
            KeyError,
            AttributeError,
            ConnectionError,
            TimeoutError,
        ) as err:
            _LOGGER.warning(
                "Failed to calculate initial prior for %s: %s, using defaults",
                entity_id,
                err,
            )
            # Return default prior
            return Prior(
                prob_given_true=entity_type.prob_true,
                prob_given_false=entity_type.prob_false,
                last_updated=dt_util.utcnow(),
            )

    async def reset_entities(self) -> None:
        """Reset entities to fresh state from configuration."""
        self._entities = await self._create_entities_from_config()

    async def create_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        state: str | float | bool | None = None,
        evidence: bool = False,
        available: bool = True,
        prior: Prior | None = None,
    ) -> Entity:
        """Create a new entity.

        Args:
            entity_id: The unique identifier for the entity
            entity_type: The type of entity
            state: The current state of the entity
            evidence: Whether the entity is active
            available: Whether the entity is available
            prior: The prior probability information

        Returns:
            The created Entity instance

        """
        # Get current state from Home Assistant if not provided
        state, available = self._resolve_entity_state(entity_id, state, available)

        # Create required components
        decay = self._create_decay_component()
        prior = self._create_prior_component(entity_type, prior)

        # Create the entity instance
        entity = self._instantiate_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            state=state,
            available=available,
            prior=prior,
            decay=decay,
        )

        # Finalize entity setup
        self._finalize_entity_setup(entity, state)

        return entity

    def _resolve_entity_state(
        self, entity_id: str, state: str | float | bool | None, available: bool
    ) -> tuple[str | float | bool | None, bool]:
        """Resolve entity state and availability from Home Assistant if needed.

        Args:
            entity_id: The entity ID
            state: Current state (if known)
            available: Current availability (if known)

        Returns:
            Tuple of (resolved_state, resolved_availability)

        """
        if state is not None:
            return state, available

        ha_state = self.hass.states.get(entity_id)
        if ha_state and ha_state.state not in ["unknown", "unavailable", None, ""]:
            _LOGGER.debug(
                "Resolved entity %s state from HA: %s", entity_id, ha_state.state
            )
            return ha_state.state, True
        _LOGGER.debug(
            "Entity %s has no valid current state in HA, using defaults", entity_id
        )
        return None, False

    def _create_decay_component(self) -> Decay:
        """Create a decay component with current configuration values.

        Returns:
            Configured Decay instance

        """
        return Decay(
            last_trigger_ts=dt_util.utcnow().timestamp(),
            half_life=self.config.decay.window,
        )

    def _create_prior_component(
        self, entity_type: EntityType, provided_prior: Prior | None
    ) -> Prior:
        """Create a prior component, using provided prior or defaults from entity type.

        Args:
            entity_type: The entity type configuration
            provided_prior: Pre-created prior (if any)

        Returns:
            Prior instance

        """
        if provided_prior is not None:
            return provided_prior

        return Prior(
            prob_given_true=entity_type.prob_true,
            prob_given_false=entity_type.prob_false,
            last_updated=dt_util.utcnow(),
        )

    def _instantiate_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        state: str | float | bool | None,
        available: bool,
        prior: Prior,
        decay: Decay,
    ) -> Entity:
        """Create the actual Entity instance.

        Args:
            entity_id: The entity ID
            entity_type: The entity type
            state: Current state
            available: Whether entity is available
            prior: Prior component
            decay: Decay component

        Returns:
            Entity instance

        """
        return Entity(
            entity_id=entity_id,
            type=entity_type,
            probability=0.01,  # Will be calculated in finalization
            prior=prior,
            decay=decay,
            state=state,
            evidence=False,  # Will be determined in finalization
            available=available,
            last_updated=dt_util.utcnow(),
        )

    def _finalize_entity_setup(
        self, entity: Entity, state: str | float | bool | None
    ) -> None:
        """Finalize entity setup by determining active state and calculating initial probability.

        Args:
            entity: The entity to finalize
            state: The current state

        """
        # Determine if entity is active
        entity.evidence = EntityManager.is_entity_active(entity, state)

        # Calculate initial probability
        entity.update_probability()

        # Log successful setup
        if state is not None:
            _LOGGER.debug(
                "Created entity %s: state=%s, evidence=%s, available=%s, probability=%.3f",
                entity.entity_id,
                state,
                entity.evidence,
                entity.available,
                entity.probability,
            )

    async def async_state_changed_listener(self, event) -> None:
        """Handle state changes for tracked entities."""
        try:
            entity_id = event.data.get("entity_id")
            new_state = event.data.get("new_state")

            if entity_id not in self._entities:
                return

            entity = self._entities[entity_id]

            # Capture previous state BEFORE updating current state
            entity.capture_previous_state()

            # Update entity state from Home Assistant
            is_available = bool(
                new_state
                and new_state.state not in ["unknown", "unavailable", None, ""]
            )
            current_state_val = new_state.state if is_available else None

            # Update entity properties
            entity.state = current_state_val
            entity.available = is_available

            # Determine if entity is active using helper method
            entity.evidence = EntityManager.is_entity_active(entity, current_state_val)

            # Update entity probabilities - previous state was captured before current state update
            probability_changed = entity.update_probability()

            # Trigger coordinator refresh if probability changed significantly
            if probability_changed:
                await self.coordinator.async_refresh()

        except Exception:
            _LOGGER.exception("Error processing state change for entity %s", entity_id)

    @staticmethod
    def is_entity_active(entity: "Entity", state: str | float | bool | None) -> bool:
        """Determine if entity is active based on state.

        Args:
            entity: The entity to check
            state: The current state value (can be None or any type)

        Returns:
            bool: True if entity should be considered active

        """
        if state is None:
            return False

        # Convert state to string for comparison
        state_str = str(state)

        entity_type = entity.type
        if entity_type.active_states is not None:
            return state_str in entity_type.active_states
        if entity_type.active_range is not None:
            try:
                value = float(state)  # Use original state for numeric conversion
                min_val, max_val = entity_type.active_range
            except (ValueError, TypeError):
                return False
            else:
                return min_val <= value <= max_val

        return False

    async def initialize_states(self) -> None:
        """Initialize entity states from current Home Assistant states."""
        _LOGGER.debug(
            "Initializing current states for %d entities", len(self._entities)
        )

        entities_updated = 0

        # Initialize each entity from current HA state
        for entity_id, entity in self._entities.items():
            if self._initialize_entity_from_ha_state(entity_id, entity):
                entities_updated += 1

        # Log completion statistics
        _LOGGER.debug(
            "Initialized current states for %d/%d entities",
            entities_updated,
            len(self._entities),
        )

    def _initialize_entity_from_ha_state(self, entity_id: str, entity: Entity) -> bool:
        """Initialize a single entity from current Home Assistant state.

        Args:
            entity_id: The entity ID to initialize
            entity: The entity instance to update

        Returns:
            bool: True if entity was successfully updated from HA state

        """
        ha_state = self.hass.states.get(entity_id)

        # Check if HA state is valid and apply appropriate initialization
        if ha_state and ha_state.state not in ["unknown", "unavailable", None, ""]:
            self._apply_entity_state_initialization(entity, ha_state, entity_id, True)
            return True

        self._apply_entity_state_initialization(entity, None, entity_id, False)
        return False

    def _apply_entity_state_initialization(
        self, entity: Entity, ha_state, entity_id: str, is_available: bool
    ) -> None:
        """Apply state initialization to entity for both available and unavailable cases.

        Args:
            entity: The entity to update
            ha_state: The HA state object (None if unavailable)
            entity_id: The entity ID (for logging)
            is_available: Whether the entity state is available

        """
        if is_available and ha_state:
            # Set entity state from current HA state
            entity.state = ha_state.state
            entity.available = True

            # Determine if entity is active using helper method
            entity.evidence = EntityManager.is_entity_active(entity, ha_state.state)

            # Set previous state to current for initialization (no transitions on startup)
            entity.previous_evidence = entity.evidence
            entity.previous_probability = entity.probability

            # Update entity probability (will detect no transition, but will update timestamp)
            entity.update_probability()
        else:
            # Entity not available in HA
            entity.available = False
            _LOGGER.debug(
                "Entity %s not available in HA, marked as unavailable", entity_id
            )

    async def _create_entities_from_config(self) -> dict[str, Entity]:
        """Create entities from current configuration."""
        # Use shared sensor mapping logic instead of duplicating it
        type_mappings = self.build_sensor_type_mappings()

        entities: dict[str, Entity] = {}

        # First pass: Create all entities with default priors
        for input_type, inputs in type_mappings.items():
            entity_type = self.coordinator.entity_types.get_entity_type(input_type)
            for input_entity_id in inputs:
                # Create entity with default priors from entity type
                default_prior = Prior(
                    prob_given_true=entity_type.prob_true,
                    prob_given_false=entity_type.prob_false,
                    last_updated=dt_util.utcnow(),
                )
                entities[input_entity_id] = await self.create_entity(
                    entity_id=input_entity_id,
                    entity_type=entity_type,
                    prior=default_prior,
                )

        # Second pass: Update priors for all entities now that they exist
        for entity in entities.values():
            learned_prior = await self.coordinator.priors.calculate(entity)
            entity.prior = learned_prior

        return entities

    def get_entity(self, entity_id: str) -> Entity:
        """Get the entity from an entity ID."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found for entity: {entity_id}")
        return self._entities[entity_id]

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the manager."""
        self._entities[entity.entity_id] = entity
        _LOGGER.debug("Added entity %s to entity manager", entity.entity_id)

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the manager."""
        if entity_id in self._entities:
            del self._entities[entity_id]
            _LOGGER.debug("Removed entity %s from entity manager", entity_id)

    def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up all entities
        for entity in self._entities.values():
            entity.cleanup()
