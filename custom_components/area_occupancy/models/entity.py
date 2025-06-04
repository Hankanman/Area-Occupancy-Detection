"""Entity model."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util
from ..coordinator import AreaOccupancyCoordinator
from .prior import Prior, PriorType
from .entity_type import EntityType, InputType
from ..utils import bayesian_update, validate_prob, validate_datetime


@dataclass
class Entity:
    """Type for sensor state information."""

    type: EntityType
    state: str | float | bool | None = None
    is_active: bool = False
    probability: float = 0.0
    weighted_probability: float = 0.0
    last_changed: datetime | None = None
    available: bool = True
    prior: Prior | None = None

    def __post_init__(self):
        """Post init."""
        self.probability = validate_prob(self.probability)
        self.weighted_probability = validate_prob(self.weighted_probability)
        self.last_changed = validate_datetime(self.last_changed)


class EntityManager:
    """Manages entities."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Initialize the entities."""
        self.coordinator = coordinator
        self.config = coordinator.config_manager.config
        self.hass = coordinator.hass
        self.storage = coordinator.storage
        self._entities: dict[str, Entity] = {}

    async def async_initialize(self) -> None:
        """Initialize the entities."""
        self._entities = await self.map_inputs_to_entities()

    @property
    def entities(self) -> dict[str, Entity]:
        """Get the entities."""
        return self._entities

    async def update_entities(self) -> None:
        """Update the entities."""
        self._entities = await self.map_inputs_to_entities()

    def create_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        state: str | float | bool | None = None,
        is_active: bool = False,
        probability: float = 0.0,
        weighted_probability: float = 0.0,
        last_changed: datetime | None = None,
        available: bool = True,
        prior: Prior | None = None,
    ) -> Entity:
        """Create a new entity and add it to the manager.
        
        Args:
            entity_id: The unique identifier for the entity
            entity_type: The type of entity
            state: The current state of the entity
            is_active: Whether the entity is active
            probability: The probability value
            weighted_probability: The weighted probability value
            last_changed: When the entity last changed state
            available: Whether the entity is available
            prior: The prior probability information
            
        Returns:
            The created Entity instance
            
        Raises:
            ValueError: If an entity with the given ID already exists
        """
        if entity_id in self._entities:
            raise ValueError(f"Entity already exists: {entity_id}")
            
        entity = Entity(
            type=entity_type,
            state=state,
            is_active=is_active,
            probability=validate_prob(probability),
            weighted_probability=validate_prob(weighted_probability),
            last_changed=validate_datetime(last_changed),
            available=available,
            prior=prior,
        )
        
        self._entities[entity_id] = entity
        return entity

    def update_entity_state(
        self,
        entity_id: str,
        state: str | float | bool | None = None,
        is_active: bool | None = None,
        probability: float | None = None,
        weighted_probability: float | None = None,
        last_changed: datetime | None = None,
        available: bool | None = None,
    ) -> Entity:
        """Update an entity's state.
        
        Args:
            entity_id: The ID of the entity to update
            state: New state value
            is_active: New active status
            probability: New probability value
            weighted_probability: New weighted probability value
            last_changed: New last changed timestamp
            available: New availability status
            
        Returns:
            The updated Entity instance
            
        Raises:
            ValueError: If the entity is not found
        """
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found: {entity_id}")

        entity = self._entities[entity_id]
        
        # Create a new Entity instance with updated values
        updated_entity = replace(
            entity,
            state=state if state is not None else entity.state,
            is_active=is_active if is_active is not None else entity.is_active,
            probability=validate_prob(probability) if probability is not None else entity.probability,
            weighted_probability=validate_prob(weighted_probability) if weighted_probability is not None else entity.weighted_probability,
            last_changed=validate_datetime(last_changed) if last_changed is not None else entity.last_changed,
            available=available if available is not None else entity.available,
        )
        
        self._entities[entity_id] = updated_entity
        return updated_entity

    def reset_entity_state(self, entity_id: str) -> None:
        """Reset a entity's state to defaults."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found: {entity_id}")

        entity = self._entities[entity_id]
        entity.state = None
        entity.is_active = False
        entity.probability = 0.0
        entity.weighted_probability = 0.0
        entity.last_changed = None
        entity.available = True

    def reset_all_entity_states(self) -> None:
        """Reset all entity states to defaults."""
        for entity_id in self._entities:
            self.reset_entity_state(entity_id)

    async def map_inputs_to_entities(self) -> dict[str, Entity]:
        """Map inputs to entities."""
        history_days = self.config.decay.history_period
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=history_days)
        primary_sensor = self.config.sensors.primary_occupancy
        if not primary_sensor:
            raise ValueError("Primary occupancy sensor must be configured")
        type_mappings = {
            InputType.MOTION: self.config.sensors.motion,
            InputType.MEDIA: self.config.sensors.media,
            InputType.APPLIANCE: self.config.sensors.appliances,
            InputType.DOOR: self.config.sensors.doors,
            InputType.WINDOW: self.config.sensors.windows,
            InputType.LIGHT: self.config.sensors.lights,
            InputType.ENVIRONMENTAL: self.config.sensors.illuminance
            + self.config.sensors.humidity
            + self.config.sensors.temperature,
        }
        entities: dict[str, Entity] = {}
        for input_type, inputs in type_mappings.items():
            entity_type = self.coordinator.entity_types.get_entity_type(
                input_type
            )
            for input in inputs:
                prior = await self.coordinator.priors.calculate(
                    entity_id=input,
                    entity_type=entity_type,
                    hass=self.hass,
                    primary_sensor=primary_sensor,
                    start_time=start_time,
                    end_time=end_time,
                    prior_type=PriorType.ENTITY,
                )
                entities[input] = self.create_entity(
                    entity_id=input,
                    entity_type=entity_type,
                    prior=prior,
                )
        return entities

    def get_entity_weight(self, entity_id: str) -> float:
        """Get the weight of a sensor."""
        return self.coordinator.entities.get_entity(entity_id).type.weight

    def get_entity_active_states(self, entity_id: str) -> set[str]:
        """Get the active states of an entity."""
        active_states = self.coordinator.entities.get_entity(entity_id).type.active_states
        return set(active_states) if active_states is not None else set()

    def get_entity(self, entity_id: str) -> Entity:
        """Get the entity from an entity ID."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found for entity: {entity_id}")
        return self._entities[entity_id]

    def calculate_sensor_probability(self, entity: Entity) -> Entity:
        """Calculate probability contribution from a single sensor using Bayesian inference."""
        if not entity.state or not entity.available:  # Ensure state exists
            return entity

        unweighted_prob = bayesian_update(
            entity.type.prior,
            entity.type.prob_true,
            entity.type.prob_false,
            entity.is_active,
        )
        weight = float(entity.type.weight)
        weighted_prob = unweighted_prob * weight

        return replace(
            entity,
            probability=unweighted_prob,
            weighted_probability=weighted_prob,
            is_active=True,
        )
