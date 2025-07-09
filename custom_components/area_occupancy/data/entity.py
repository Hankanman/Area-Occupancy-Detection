"""Entity model."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

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
    last_updated: datetime
    previous_evidence: bool | None
    previous_probability: float

    @property
    def name(self) -> str | None:
        """Get the entity name from Home Assistant state."""
        ha_state = self.coordinator.hass.states.get(self.entity_id)
        return ha_state.name if ha_state else None

    @property
    def probability(self) -> float:
        """Calculate this entity's raw contribution to area probability.

        This shows what this entity would contribute if it were fully active,
        using Bayesian calculation without decay factor applied.
        """

        # Calculate effective Bayesian posterior with decay applied
        return bayesian_probability(
            prior=self.coordinator.area_prior,
            prob_given_true=self.likelihood.prob_given_true,
            prob_given_false=self.likelihood.prob_given_false,
            evidence=True,
            decay_factor=self.decay_factor,
        )

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

    def has_new_evidence(self) -> bool:
        """Update decay and probability on actual evidence transitions.

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
                # Evidence appeared - jump probability up via Bayesian update
                self.previous_probability = bayesian_probability(
                    prior=self.previous_probability,
                    prob_given_true=self.likelihood.prob_given_true,
                    prob_given_false=self.likelihood.prob_given_false,
                    evidence=True,
                    decay_factor=1.0,  # No decay on evidence appearance
                )
                self.decay.stop_decay()
            else:  # TRUE→FALSE transition
                # Evidence lost - start decay from current probability towards prob_given_false
                # Working probability stays at current level and will decay over time
                self.decay.start_decay()

        # Update previous evidence for next comparison
        self.previous_evidence = current_evidence
        return transition_occurred

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for storage."""
        return {
            "entity_id": self.entity_id,
            "type": self.type.to_dict(),
            "likelihood": self.likelihood.to_dict(),
            "decay": self.decay.to_dict(),
            "last_updated": self.last_updated.isoformat(),
            "previous_evidence": self.previous_evidence,
            "previous_probability": self.previous_probability,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: "AreaOccupancyCoordinator"
    ) -> "Entity":
        """Create entity from dictionary."""
        entity_type = EntityType.from_dict(data["type"])
        decay = Decay.from_dict(data["decay"])

        # Create likelihood with specific values instead of entity reference
        likelihood = Likelihood.from_dict(
            data["likelihood"],
            coordinator,
            entity_id=data["entity_id"],
            active_states=entity_type.active_states or [],
            default_prob_true=entity_type.prob_true,
            default_prob_false=entity_type.prob_false,
            weight=entity_type.weight,
        )

        # Create the entity
        return cls(
            entity_id=data["entity_id"],
            type=entity_type,
            likelihood=likelihood,
            decay=decay,
            coordinator=coordinator,
            last_updated=datetime.fromisoformat(data["last_updated"]),
            previous_evidence=data["previous_evidence"],
            previous_probability=data["previous_probability"],
        )


class EntityManager:
    """Manages entities."""

    def __init__(self, coordinator: "AreaOccupancyCoordinator") -> None:
        """Initialize the entities."""
        self.coordinator = coordinator
        self.config = coordinator.config_manager.config
        self.hass = coordinator.hass
        self._entities: dict[str, Entity] = {}

    async def __post_init__(self) -> None:
        """Post init - validate datetime."""
        if self._entities:
            await self._update_entities_from_config()
        else:
            self._entities = await self._create_entities_from_config()

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
            }
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

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the manager."""
        if entity_id in self._entities:
            del self._entities[entity_id]

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._entities.clear()
        self._entities = await self._create_entities_from_config()

    async def create_entity(self, entity_id: str, entity_type: EntityType) -> Entity:
        """Create a new entity.

        Args:
            entity_id: The unique identifier for the entity
            entity_type: The type of entity

        Returns:
            The created Entity instance

        """
        # Create required components
        decay = Decay(
            last_trigger_ts=dt_util.utcnow().timestamp(),
            half_life=self.config.decay.half_life,
        )
        likelihood = Likelihood(
            coordinator=self.coordinator,
            entity_id=entity_id,
            active_states=entity_type.active_states or [],
            default_prob_true=entity_type.prob_true,
            default_prob_false=entity_type.prob_false,
            weight=entity_type.weight,
        )

        return Entity(
            entity_id=entity_id,
            type=entity_type,
            likelihood=likelihood,
            decay=decay,
            coordinator=self.coordinator,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            previous_probability=0.0,
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
            InputType.ILLUMINANCE: self.config.sensors.illuminance,
            InputType.HUMIDITY: self.config.sensors.humidity,
            InputType.TEMPERATURE: self.config.sensors.temperature,
        }

    async def update_all_entity_likelihoods(
        self, history_period: int | None = None, force: bool = False
    ) -> int:
        """Update all entity likelihoods.

        Args:
            history_period: Period in days for historical data
            force: If True, bypass cache validation and force recalculation

        Returns:
            int: Number of entities updated

        """
        # Ensure area baseline prior is calculated first since likelihood calculations depend on it
        if self.coordinator.config.history.enabled:
            await self.coordinator.prior.update(
                force=force, history_period=history_period
            )

        if not self._entities:
            _LOGGER.debug("No entities to update likelihoods for")
            return 0

        # Process entities in parallel chunks to avoid blocking
        chunk_size = 5  # Process 5 entities at a time
        entity_list = list(self._entities.values())
        updated_count = 0

        for i in range(0, len(entity_list), chunk_size):
            chunk = entity_list[i : i + chunk_size]

            # Create tasks for parallel processing
            tasks = []
            for entity in chunk:
                task = self._update_entity_likelihood_safe(
                    entity, force=force, history_period=history_period
                )
                tasks.append(task)

            # Wait for all tasks in this chunk to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful updates
            for result in results:
                if isinstance(result, Exception):
                    _LOGGER.warning("Entity likelihood update failed: %s", result)
                elif isinstance(result, int):
                    updated_count += result

            # Yield control between chunks to avoid blocking
            await asyncio.sleep(0)

        _LOGGER.info(
            "Updated likelihoods for %d out of %d entities",
            updated_count,
            len(self._entities),
        )
        return updated_count

    async def _update_entity_likelihood_safe(
        self, entity: Entity, force: bool = False, history_period: int | None = None
    ) -> int:
        """Safely update a single entity's likelihood with error handling.

        Args:
            entity: The entity to update
            force: If True, bypass cache validation and force recalculation
            history_period: Period in days for historical data

        Returns:
            1 if successful, 0 if failed

        """
        try:
            await entity.likelihood.update(force=force, history_period=history_period)
        except (ValueError, TypeError) as err:
            _LOGGER.warning(
                "Failed to update likelihood for entity %s: %s",
                entity.entity_id,
                err,
            )
            return 0
        else:
            return 1

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
                entity_id=entity_id, entity_type=entity_type
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
                    entity_id=input_entity_id, entity_type=entity_type
                )

        # Note: Likelihood calculations will be performed later in the coordinator setup
        # This avoids initialization issues during entity creation

        return entities
