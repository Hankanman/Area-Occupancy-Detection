"""Entity model."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.const import ATTR_FRIENDLY_NAME
from homeassistant.util import dt as dt_util

from ..utils import bayesian_probability
from .decay import Decay
from .entity_type import EntityType, InputType
from .likelihood import Likelihood

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass
class Entity:
    """Type for sensor state information."""

    # --- Core Data ---
    entity_id: str
    type: EntityType
    likelihood: Likelihood
    decay: Decay
    coordinator: "AreaOccupancyCoordinator"
    name: str | None = None
    last_updated: datetime = field(default_factory=dt_util.utcnow)

    # Minimal state tracking for proper decay behavior
    _previous_evidence: bool = field(default=False, init=False, repr=False)
    _effective_probability: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        """Post init - initialize previous evidence and working probability."""
        # Initialize previous evidence to current evidence to avoid false transitions on startup
        state = self.coordinator.hass.states.get(self.entity_id)
        if state:
            self.name = state.attributes.get(ATTR_FRIENDLY_NAME)
        self._previous_evidence = self.evidence

        # Initialize working probability based on current evidence
        self._effective_probability = (
            self.likelihood.prob_given_true
            if self.evidence
            else self.likelihood.prob_given_false
        )

        _LOGGER.debug(
            "Created entity %s with initial evidence: %s, probability: %.3f",
            self.entity_id,
            self.evidence,
            self._effective_probability,
        )

    @property
    def probability(self) -> float:
        """Calculate probability using working probability and decay."""
        # Update timestamp
        self.last_updated = dt_util.utcnow()

        # If currently decaying, apply decay factor to working probability
        if self.decay.is_decaying:
            # Apply decay towards prob_given_false
            target = self.likelihood.prob_given_false
            decay_factor = self.decay.decay_factor

            # Decay working probability towards target
            self._effective_probability = (
                self._effective_probability * decay_factor + target * (1 - decay_factor)
            )

        return self._effective_probability

    @property
    def effective_probability(self) -> float:
        """Get the current effective probability during decay.

        This shows the "live" probability value that's between prob_given_true
        and prob_given_false when the entity is decaying. When not decaying,
        it returns the current working probability.

        Returns:
            Current effective probability considering decay state

        """
        return self._effective_probability

    @property
    def available(self) -> bool:
        """Get the entity availability."""
        return self.state is not None

    @property
    def state(self) -> str | float | bool | None:
        """Get the entity state."""
        ha_state = self.coordinator.hass.states.get(self.entity_id)

        # Check if HA state is valid
        if ha_state and ha_state.state not in [
            "unknown",
            "unavailable",
            None,
            "",
            "NaN",
        ]:
            return ha_state.state
        return None

    @property
    def evidence(self) -> bool:
        """Determine if entity is active.

        Returns:
            bool: True if entity is active, False otherwise

        """
        if self.state is None:
            return False

        # Convert state to string for comparison
        state_str = str(self.state)

        if self.type.active_states is not None:
            return state_str in self.type.active_states
        if self.type.active_range is not None:
            try:
                value = float(self.state)
                min_val, max_val = self.type.active_range
            except (ValueError, TypeError):
                return False
            else:
                return min_val <= value <= max_val

        return False

    @property
    def active_states(self) -> list[str]:
        """Get the active states for the entity."""
        return self.type.active_states or []

    def has_new_evidence(self) -> bool:
        """Update decay and probability on actual evidence transitions.

        Returns:
            bool: True if evidence transition occurred, False otherwise

        """
        current_evidence = self.evidence  # Pure calculation from current HA state

        if current_evidence != self._previous_evidence:
            if current_evidence:  # OFF→ON transition
                # Evidence appeared - jump probability up via Bayesian update
                old_prob = self._effective_probability
                self._effective_probability = bayesian_probability(
                    prior=self._effective_probability,
                    prob_given_true=self.likelihood.prob_given_true,
                    prob_given_false=self.likelihood.prob_given_false,
                    evidence=True,
                    weight=self.type.weight,
                    decay_factor=1.0,  # No decay on evidence appearance
                )
                self.decay.stop_decay()
                _LOGGER.debug(
                    "Entity %s: evidence transition OFF→ON, probability %.3f→%.3f, stopped decay",
                    self.entity_id,
                    old_prob,
                    self._effective_probability,
                )
            else:  # ON→OFF transition
                # Evidence lost - start decay from current probability towards prob_given_false
                # Working probability stays at current level and will decay over time
                self.decay.start_decay()
                _LOGGER.debug(
                    "Entity %s: evidence transition ON→OFF, probability %.3f, started decay",
                    self.entity_id,
                    self._effective_probability,
                )

            # Update previous evidence for next comparison
            self._previous_evidence = current_evidence
            return True

        return False  # No transition occurred

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for storage."""
        return {
            "entity_id": self.entity_id,
            "type": self.type.to_dict(),
            "likelihood": self.likelihood.to_dict(),
            "decay": self.decay.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: "AreaOccupancyCoordinator"
    ) -> "Entity":
        """Create entity from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            type=EntityType.from_dict(data["type"]),
            likelihood=Likelihood.from_dict(data["likelihood"]),
            decay=Decay.from_dict(data["decay"]),
            coordinator=coordinator,
        )


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

    async def __post_init__(self) -> None:
        """Post init - validate datetime."""
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

    @property
    def decaying_entities(self) -> list[Entity]:
        """Get the decaying entities."""
        return [
            entity for entity in self._entities.values() if entity.decay.is_decaying
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
                entity_id: Entity.from_dict(entity, coordinator)
                for entity_id, entity in entities_data.items()
            }
        except (KeyError, ValueError, TypeError) as err:
            raise ValueError(
                f"Failed to deserialize entity data: {err}. "
                f"Entity structure may be corrupted or incompatible."
            ) from err

        return manager

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

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._entities.clear()
        self._entities = await self._create_entities_from_config()

    async def create_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
    ) -> Entity:
        """Create a new entity.

        Args:
            entity_id: The unique identifier for the entity
            entity_type: The type of entity
            prior: The prior probability information

        Returns:
            The created Entity instance

        """
        # Create required components
        decay = Decay(
            last_trigger_ts=dt_util.utcnow().timestamp(),
            half_life=self.config.decay.half_life,
        )
        likelihood = Likelihood(
            prob_given_true=entity_type.prob_true,
            prob_given_false=entity_type.prob_false,
            last_updated=dt_util.utcnow(),
        )

        return Entity(
            entity_id=entity_id,
            type=entity_type,
            likelihood=likelihood,
            decay=decay,
            coordinator=self.coordinator,
        )

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

    async def async_state_changed_listener(self, event) -> None:
        """Handle state changes for tracked entities.

        With pure calculations, we check for evidence transitions to manage decay properly.
        """
        try:
            entity_id = event.data.get("entity_id")

            if entity_id not in self._entities:
                return

            entity = self._entities[entity_id]

            # Check for evidence transitions and update decay accordingly
            new_evidence = entity.has_new_evidence()

            if new_evidence:
                _LOGGER.debug(
                    "Evidence transition detected for %s, triggering refresh",
                    entity_id,
                )
                await self.coordinator.async_refresh()

        except Exception:
            _LOGGER.exception("Error processing state change for entity %s", entity_id)

    async def update_all_entity_likelihoods(
        self, history_period: int | None = None
    ) -> int:
        """Update all entity likelihoods.

        Returns:
            int: Number of entities updated

        """
        # Ensure area baseline prior is calculated first since likelihood calculations depend on it
        if self.coordinator.config.history.enabled:
            await self.coordinator.prior.calculate_area_baseline_prior(history_period)

        updated_count = 0
        for entity in self._entities.values():
            try:
                await entity.likelihood.update(self.coordinator, entity, history_period)
                updated_count += 1
                _LOGGER.debug(
                    "Updated likelihood for entity %s: prob_given_true=%.3f, prob_given_false=%.3f",
                    entity.entity_id,
                    entity.likelihood.prob_given_true,
                    entity.likelihood.prob_given_false,
                )
            except Exception as err:
                _LOGGER.warning(
                    "Failed to update likelihood for entity %s: %s",
                    entity.entity_id,
                    err,
                )

        _LOGGER.info(
            "Updated likelihoods for %d out of %d entities",
            updated_count,
            len(self._entities),
        )
        return updated_count

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
        # First pass: Create all new entities with default priors
        new_entities: dict[str, Entity] = {}
        for entity_id, entity_type in config_entities.items():
            _LOGGER.info("Creating new entity %s", entity_id)
            # Create entity with default priors from entity type
            new_entity = await self.create_entity(
                entity_id=entity_id,
                entity_type=entity_type,
            )
            new_entities[entity_id] = new_entity
            updated_entities[entity_id] = new_entity

        # Note: Likelihood updates will be performed separately after all entities are created
        # This avoids initialization issues during entity creation

    async def _get_config_entity_mapping(self) -> dict[str, EntityType]:
        """Get mapping of entity_id -> EntityType from current configuration."""
        # Build sensor type mappings from configuration
        type_mappings = self.build_sensor_type_mappings()

        # Create initial entity mapping from type mappings

        return self._build_entity_mapping_from_types(type_mappings)

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
                entities[input_entity_id] = await self.create_entity(
                    entity_id=input_entity_id,
                    entity_type=entity_type,
                )

        # Note: Likelihood calculations will be performed later in the coordinator setup
        # This avoids initialization issues during entity creation

        return entities
