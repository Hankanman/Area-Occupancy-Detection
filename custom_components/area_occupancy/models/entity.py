"""Entity model."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import CALLBACK_TYPE, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util

from ..utils import validate_datetime, validate_prob
from .decay import Decay
from .entity_type import EntityType, InputType
from .prior import Prior, PriorType
from .probability import Probability

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

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
    last_updated: datetime = field(default_factory=dt_util.utcnow)

    def __post_init__(self):
        """Post init."""
        # Validate last_changed
        self.last_changed = validate_datetime(self.last_changed)
        self.last_updated = validate_datetime(self.last_updated)

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for storage."""
        return {
            "entity_id": self.entity_id,
            "type": self.type.to_dict(),
            "probability": self.probability.to_dict(),
            "prior": self.prior.to_dict(),
            "decay": self.decay.to_dict(),
            "state": self.state,
            "is_active": self.is_active,
            "last_changed": self.last_changed,
            "available": self.available,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create entity from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            type=EntityType.from_dict(data["type"]),
            probability=Probability.from_dict(data["probability"]),
            prior=Prior.from_dict(data["prior"]),
            decay=Decay.from_dict(data["decay"]),
            state=data["state"],
            is_active=data["is_active"],
            last_changed=data["last_changed"],
            available=data["available"],
            last_updated=data["last_updated"],
        )

    async def async_update(self) -> None:
        """Update entity state and probabilities."""
        # Update last_updated timestamp
        self.last_updated = dt_util.utcnow()

        # Update probability based on current state
        self.probability.calculate_probability(
            prior=self.type.prior,
            prob_true=self.type.prob_true,
            prob_false=self.type.prob_false,
        )

        # Apply decay
        decayed_probability, decay_factor = self.decay.update_decay(
            current_probability=self.probability.probability,
            previous_probability=self.probability.decayed_probability,
        )

        # Update probability with decayed values
        self.probability.update(
            decayed_probability=decayed_probability,
            decay_factor=decay_factor,
            is_active=self.is_active,
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
        self.storage = coordinator.storage
        self._entities: dict[str, Entity] = {}
        self._remove_state_listener: CALLBACK_TYPE | None = None

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
        """Create entity manager from dictionary."""
        manager = cls(coordinator=coordinator)
        manager._entities = {
            entity_id: Entity.from_dict(entity)
            for entity_id, entity in data["entities"].items()
        }
        return manager

    async def async_initialize(self) -> None:
        """Initialize the entities."""
        self._entities = await self.map_inputs_to_entities()
        self._setup_entity_tracking()
        _LOGGER.debug("EntityManager initialized with entities: %s", self._entities)

    @property
    def entities(self) -> dict[str, Entity]:
        """Get the entities."""
        return self._entities

    async def reset_entities(self) -> None:
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
            last_changed: When the entity last changed state
            available: Whether the entity is available
            prior: The prior probability information

        Returns:
            The created Entity instance

        Raises:
            ValueError: If an entity with the given ID already exists

        """
        if entity_id in self._entities:
            return self._entities[entity_id]

        # Create probability instance first
        prob = Probability(
            probability=validate_prob(probability),
            decayed_probability=validate_prob(0.0),
            decay_factor=validate_prob(1.0),
            last_updated=validate_datetime(last_changed),
            is_active=is_active,
        )

        # Create decay instance
        decay = Decay(
            is_decaying=True,
            decay_start_time=dt_util.utcnow(),
            decay_start_probability=validate_prob(0.0),
            decay_factor=validate_prob(1.0),
            decay_window=self.config.decay.window,
            decay_enabled=self.config.decay.enabled,
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

                # Get the existing entity
                entity = self._entities[entity_id]

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

                # Update entity properties
                entity.state = current_state_val
                entity.last_changed = last_changed
                entity.available = is_available

                # Determine if entity is active based on its type and state
                active_states = entity.type.active_states or []
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

                # Calculate new probability
                entity.probability.calculate_probability(
                    prior=entity.type.prior,
                    prob_true=entity.type.prob_true,
                    prob_false=entity.type.prob_false,
                )

                # Apply decay using entity's decay instance
                decayed_probability, decay_factor = entity.decay.update_decay(
                    current_probability=entity.probability.probability,
                    previous_probability=previous_probability,
                )

                # Update probability with decayed values
                entity.probability.update(
                    decayed_probability=decayed_probability,
                    decay_factor=decay_factor,
                    is_active=is_active,
                )

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
            entity_type = self.coordinator.entity_types.get_entity_type(input_type)
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

    def get_entity(self, entity_id: str) -> Entity:
        """Get the entity from an entity ID."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found for entity: {entity_id}")
        return self._entities[entity_id]

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None
