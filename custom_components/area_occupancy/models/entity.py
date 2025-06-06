"""Entity model."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import logging
from typing import Callable

from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util
from ..coordinator import AreaOccupancyCoordinator
from .prior import Prior, PriorType
from .entity_type import EntityType, InputType
from .probability import Probability
from .decay import Decay
from ..utils import bayesian_update, validate_prob, validate_datetime

_LOGGER = logging.getLogger(__name__)


@dataclass
class Entity:
    """Type for sensor state information."""

    entity_id: str
    type: EntityType
    probability: Probability
    prior: Prior
    decay: Decay
    state: str | float | bool | None = None
    is_active: bool = False
    last_changed: datetime | None = None
    available: bool = True

    def __post_init__(self):
        """Post init."""
        # Validate last_changed
        self.last_changed = validate_datetime(self.last_changed)

    def update_probability(
        self,
        probability: float,
        weighted_probability: float,
        decayed_probability: float,
        decay_factor: float,
        is_active: bool,
    ) -> None:
        """Update probability state.

        Args:
            probability: The raw probability value
            weighted_probability: The weighted probability value
            decayed_probability: The decayed probability value
            decay_factor: The decay factor applied
            is_active: Whether the entity is active
        """
        self.probability = Probability(
            entity_id=self.entity_id,
            probability=probability,
            weighted_probability=weighted_probability,
            decayed_probability=decayed_probability,
            decay_factor=decay_factor,
            last_updated=dt_util.utcnow(),
            is_active=is_active,
        )

    def update_decay(
        self,
        is_decaying: bool,
        decay_start_time: datetime | None,
        decay_start_probability: float | None,
        decay_window: int,
        decay_enabled: bool,
    ) -> None:
        """Update decay state.

        Args:
            is_decaying: Whether the entity is currently decaying
            decay_start_time: When decay started
            decay_start_probability: Probability when decay started
            decay_window: Decay window in seconds
            decay_enabled: Whether decay is enabled
        """
        self.decay = Decay(
            entity_id=self.entity_id,
            is_decaying=is_decaying,
            decay_start_time=decay_start_time,
            decay_start_probability=decay_start_probability,
            decay_window=decay_window,
            decay_enabled=decay_enabled,
        )


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
        self._remove_state_listener: CALLBACK_TYPE | None = None

    async def async_initialize(self) -> None:
        """Initialize the entities."""
        self._entities = await self.map_inputs_to_entities()
        self._setup_entity_tracking()

    @property
    def entities(self) -> dict[str, Entity]:
        """Get the entities."""
        return self._entities

    async def update_entities(self) -> None:
        """Update the entities."""
        self._entities = await self.map_inputs_to_entities()
        # Re-setup tracking after updating entities
        self._setup_entity_tracking()

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
            
        # Create probability instance first
        prob = Probability(
            entity_id=entity_id,
            probability=validate_prob(probability),
            weighted_probability=validate_prob(weighted_probability),
            decayed_probability=validate_prob(0.0),
            decay_factor=validate_prob(1.0),
            last_updated=validate_datetime(last_changed),
            is_active=is_active,
        )
        
        # Create decay instance
        decay = Decay(
            entity_id=entity_id,
            decay_enabled=True,
            decay_window=300,  # Default 5 minutes
        )
        
        # Create prior instance if not provided
        if prior is None:
            prior = Prior(
                prior=entity_type.prior,
                prob_given_true=entity_type.prob_true,
                prob_given_false=entity_type.prob_false,
                last_updated=dt_util.utcnow(),
                type=PriorType.ENTITY,
            )
            
        entity = Entity(
            entity_id=entity_id,
            type=entity_type,
            probability=prob,
            prior=prior,
            decay=decay,
            state=state,
            is_active=is_active,
            last_changed=validate_datetime(last_changed),
            available=available,
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
        
        # Create new probability instance first
        prob = Probability(
            entity_id=entity_id,
            probability=validate_prob(probability) if probability is not None else entity.probability.probability,
            weighted_probability=validate_prob(weighted_probability) if weighted_probability is not None else entity.probability.weighted_probability,
            decayed_probability=entity.probability.decayed_probability,
            decay_factor=entity.probability.decay_factor,
            last_updated=validate_datetime(last_changed) if last_changed is not None else entity.probability.last_updated,
            is_active=is_active if is_active is not None else entity.probability.is_active,
        )
        
        # Create a new Entity instance with updated values
        updated_entity = replace(
            entity,
            state=state if state is not None else entity.state,
            is_active=is_active if is_active is not None else entity.is_active,
            probability=prob,
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
        
        # Create new probability instance first
        prob = Probability(
            entity_id=entity_id,
            probability=0.0,
            weighted_probability=0.0,
            decayed_probability=0.0,
            decay_factor=1.0,
            last_updated=dt_util.utcnow(),
            is_active=False,
        )
        
        # Create new decay instance
        decay = Decay(
            entity_id=entity_id,
            decay_enabled=True,
            decay_window=300,  # Default 5 minutes
        )
        
        # Create new prior instance
        prior = Prior(
            prior=entity.type.prior,
            prob_given_true=entity.type.prob_true,
            prob_given_false=entity.type.prob_false,
            last_updated=dt_util.utcnow(),
            type=PriorType.ENTITY,
        )
        
        entity.state = None
        entity.is_active = False
        entity.probability = prob
        entity.decay = decay
        entity.prior = prior
        entity.last_changed = None
        entity.available = True

    def reset_all_entity_states(self) -> None:
        """Reset all entity states to defaults."""
        for entity_id in self._entities:
            self.reset_entity_state(entity_id)

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes."""
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        entities_to_track = list(self._entities.keys())
        if not entities_to_track:
            return

        @callback
        def async_state_changed_listener(event) -> None:
            """Handle state changes for tracked entities."""
            try:
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")
                old_state = event.data.get("old_state")

                if entity_id not in self._entities:
                    return

                # Update entity state
                is_available = bool(
                    new_state
                    and new_state.state not in ["unknown", "unavailable", None, ""]
                )
                current_state_val = new_state.state if is_available else None
                last_changed = (
                    new_state.last_changed
                    if new_state and new_state.last_changed
                    else dt_util.utcnow()
                )

                # Get entity type and active states
                entity = self._entities[entity_id]
                active_states = entity.type.active_states or []
                
                # Determine if entity is active based on its type and state
                is_active = False
                if current_state_val is not None:
                    if entity.type.active_states is not None:
                        is_active = current_state_val in active_states
                    elif entity.type.active_range is not None:
                        try:
                            state_val = float(current_state_val)
                            min_val, max_val = entity.type.active_range
                            is_active = min_val <= state_val <= max_val
                        except (ValueError, TypeError):
                            is_active = False

                # Store previous probability for decay calculation
                previous_probability = entity.probability.probability

                # Create new probability instance
                prob = Probability(
                    entity_id=entity_id,
                    probability=entity.probability.probability,  # Keep current until calculated
                    weighted_probability=entity.probability.weighted_probability,  # Keep current until calculated
                    decayed_probability=entity.probability.decayed_probability,  # Keep current until calculated
                    decay_factor=entity.probability.decay_factor,  # Keep current until calculated
                    last_updated=last_changed,
                    is_active=is_active,
                )

                # Create new decay instance
                decay = Decay(
                    entity_id=entity_id,
                    is_decaying=entity.decay.is_decaying,
                    decay_start_time=entity.decay.decay_start_time,
                    decay_start_probability=entity.decay.decay_start_probability,
                    decay_window=entity.decay.decay_window,
                    decay_enabled=entity.decay.decay_enabled,
                )

                # Create updated entity with all properties except Prior
                updated_entity = Entity(
                    entity_id=entity_id,
                    type=entity.type,  # Keep existing type
                    probability=prob,
                    prior=entity.prior,  # Keep existing prior
                    decay=decay,
                    state=current_state_val,
                    is_active=is_active,
                    last_changed=last_changed,
                    available=is_available,
                )

                # Calculate new probability
                updated_entity = self.calculate_sensor_probability(updated_entity)
                
                # Apply decay
                decayed_probability, decay_factor = self.coordinator.decay.update_decay(
                    entity_id=entity_id,
                    current_probability=updated_entity.probability.probability,
                    previous_probability=previous_probability,
                )
                
                # Update entity with decayed probability
                updated_entity.update_probability(
                    probability=updated_entity.probability.probability,
                    weighted_probability=updated_entity.probability.weighted_probability,
                    decayed_probability=decayed_probability,
                    decay_factor=decay_factor,
                    is_active=updated_entity.probability.is_active,
                )
                self._entities[entity_id] = updated_entity

            except Exception:
                _LOGGER.exception(
                    "Error processing state change for entity %s", entity_id
                )

        self._remove_state_listener = async_track_state_change_event(
            self.hass,
            entities_to_track,
            async_state_changed_listener,
        )

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

        # Update probability state
        entity.update_probability(
            probability=unweighted_prob,
            weighted_probability=weighted_prob,
            decayed_probability=weighted_prob,  # Initially same as weighted
            decay_factor=1.0,  # No decay initially
            is_active=True,
        )

        return entity
