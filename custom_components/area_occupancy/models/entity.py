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
        # Store previous probability for decay calculation
        previous_probability = self.probability.probability

        # Update last_updated timestamp
        self.last_updated = dt_util.utcnow()

        # Update probability based on current state
        self.probability.calculate_probability(
            prior=self.type.prior,
            prob_true=self.type.prob_true,
            prob_false=self.type.prob_false,
        )

        # Apply decay using previous probability
        decayed_probability, decay_factor = self.decay.update_decay(
            current_probability=self.probability.probability,
            previous_probability=previous_probability,
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

        # Check if data has the expected structure
        if "entities" not in data:
            raise ValueError(
                f"Invalid storage format: missing 'entities' key in data structure. "
                f"Available keys: {list(data.keys())}. "
                f"This should have been caught by storage validation."
            )

        try:
            manager._entities = {
                entity_id: Entity.from_dict(entity)
                for entity_id, entity in data["entities"].items()
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

        self._setup_entity_tracking()
        _LOGGER.debug("EntityManager initialized with %d entities", len(self._entities))

        # Request coordinator refresh to update storage with initial states
        self.coordinator.request_update()

    async def _update_entities_from_config(self) -> None:
        """Update existing entities with current configuration."""
        config_entities = await self._get_config_entity_mapping()
        updated_entities: dict[str, Entity] = {}

        # Process existing restored entities
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

        # Create new entities for any added to config
        for entity_id, entity_type in config_entities.items():
            _LOGGER.info("Creating new entity %s", entity_id)
            prior = await self._calculate_initial_prior(entity_id, entity_type)
            updated_entities[entity_id] = self._create_entity(
                entity_id=entity_id,
                entity_type=entity_type,
                prior=prior,
            )

        self._entities = updated_entities
        _LOGGER.info("Entity update complete: %d total entities", len(self._entities))

    async def _get_config_entity_mapping(self) -> dict[str, EntityType]:
        """Get mapping of entity_id -> EntityType from current configuration."""
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

        entity_mapping: dict[str, EntityType] = {}
        for input_type, entity_ids in type_mappings.items():
            entity_type = self.coordinator.entity_types.get_entity_type(input_type)
            for entity_id in entity_ids:
                entity_mapping[entity_id] = entity_type

        return entity_mapping

    async def _calculate_initial_prior(
        self, entity_id: str, entity_type: EntityType
    ) -> Prior:
        """Calculate initial prior for a new entity."""
        history_days = self.config.decay.history_period
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=history_days)
        primary_sensor = self.config.sensors.primary_occupancy

        if not primary_sensor:
            raise ValueError("Primary occupancy sensor must be configured")

        try:
            return await self.coordinator.priors.calculate(
                entity_id=entity_id,
                entity_type=entity_type,
                hass=self.hass,
                primary_sensor=primary_sensor,
                start_time=start_time,
                end_time=end_time,
                prior_type=PriorType.ENTITY,
            )
        except (
            ValueError,
            KeyError,
            AttributeError,
            ConnectionError,
            TimeoutError,
        ) as err:
            _LOGGER.warning(
                "Failed to calculate initial prior for %s: %s", entity_id, err
            )
            # Return default prior
            return Prior(
                prior=entity_type.prior,
                prob_given_true=entity_type.prob_true,
                prob_given_false=entity_type.prob_false,
                last_updated=dt_util.utcnow(),
                type=PriorType.ENTITY,
            )

    @property
    def entities(self) -> dict[str, Entity]:
        """Get the entities."""
        return self._entities

    async def reset_entities(self) -> None:
        """Reset entities to fresh state from configuration."""
        self._entities = await self._create_entities_from_config()
        self._setup_entity_tracking()

    def _create_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        state: str | float | bool | None = None,
        is_active: bool = False,
        last_changed: datetime | None = None,
        available: bool = True,
        prior: Prior | None = None,
    ) -> Entity:
        """Create a new entity.

        Args:
            entity_id: The unique identifier for the entity
            entity_type: The type of entity
            state: The current state of the entity
            is_active: Whether the entity is active
            last_changed: When the entity last changed state
            available: Whether the entity is available
            prior: The prior probability information

        Returns:
            The created Entity instance

        """
        # If no state provided, get current state from Home Assistant
        if state is None:
            ha_state = self.hass.states.get(entity_id)
            if ha_state and ha_state.state not in ["unknown", "unavailable", None, ""]:
                state = ha_state.state
                last_changed = ha_state.last_changed
                available = True

                # Determine if entity is active based on its type and current state
                if entity_type.active_states is not None:
                    is_active = state in entity_type.active_states
                elif entity_type.active_range is not None:
                    try:
                        state_val = float(state)
                        min_val, max_val = entity_type.active_range
                        is_active = min_val <= state_val <= max_val
                    except (ValueError, TypeError):
                        is_active = False

                _LOGGER.debug(
                    "Initialized entity %s with current HA state: state=%s, is_active=%s, available=%s",
                    entity_id,
                    state,
                    is_active,
                    available,
                )
            else:
                _LOGGER.debug(
                    "Entity %s has no valid current state in HA, using defaults",
                    entity_id,
                )
                available = False

        # Create probability instance
        prob = Probability(
            probability=validate_prob(0.0),
            decayed_probability=validate_prob(0.0),
            decay_factor=validate_prob(1.0),
            last_updated=validate_datetime(last_changed),
            is_active=is_active,
        )

        # Create decay instance
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
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

        return Entity(
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

    def _setup_entity_tracking(self) -> None:
        """Set up event listener to track entity state changes."""
        # Clean up existing listener
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        entities_to_track = list(self._entities.keys())
        if not entities_to_track:
            _LOGGER.debug("No entities to track, skipping state listener setup")
            return

        _LOGGER.debug(
            "Setting up state tracking for %d entities", len(entities_to_track)
        )

        @callback
        def async_state_changed_listener(event) -> None:
            """Handle state changes for tracked entities."""
            try:
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")

                if entity_id not in self._entities:
                    return

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
                is_active = False
                if current_state_val is not None:
                    if entity.type.active_states is not None:
                        is_active = current_state_val in entity.type.active_states
                    elif entity.type.active_range is not None:
                        try:
                            state_val = float(current_state_val)
                            min_val, max_val = entity.type.active_range
                            is_active = min_val <= state_val <= max_val
                        except (ValueError, TypeError):
                            is_active = False

                # Update is_active status
                entity.is_active = is_active

                # IMMEDIATE entity update for maximum responsiveness
                # Log the state change before updating
                old_state = entity.state
                old_is_active = entity.is_active
                _LOGGER.debug(
                    "State change detected for %s: %s -> %s (active: %s -> %s) - triggering immediate update",
                    entity_id,
                    old_state,
                    current_state_val,
                    old_is_active,
                    is_active,
                )
                self.hass.async_create_task(self._async_update_entity(entity_id))

            except Exception:
                _LOGGER.exception(
                    "Error processing state change for entity %s", entity_id
                )

        self._remove_state_listener = async_track_state_change_event(
            self.hass,
            entities_to_track,
            async_state_changed_listener,
        )

        # Initialize current states for all entities after setting up tracking
        self._initialize_current_states()

    def _initialize_current_states(self) -> None:
        """Initialize entity states from current Home Assistant states."""
        _LOGGER.debug(
            "Initializing current states for %d entities", len(self._entities)
        )

        entities_updated = 0
        significant_changes = False

        for entity_id, entity in self._entities.items():
            ha_state = self.hass.states.get(entity_id)
            if ha_state and ha_state.state not in ["unknown", "unavailable", None, ""]:
                # Update entity with current HA state
                old_state = entity.state
                old_is_active = entity.is_active
                old_available = entity.available

                entity.state = ha_state.state
                entity.last_changed = ha_state.last_changed
                entity.available = True

                # Determine if entity is active based on its type and state
                is_active = False
                if entity.type.active_states is not None:
                    is_active = ha_state.state in entity.type.active_states
                elif entity.type.active_range is not None:
                    try:
                        state_val = float(ha_state.state)
                        min_val, max_val = entity.type.active_range
                        is_active = min_val <= state_val <= max_val
                    except (ValueError, TypeError):
                        is_active = False

                entity.is_active = is_active
                entities_updated += 1

                # Check for significant changes
                if (
                    old_state != entity.state
                    or old_is_active != entity.is_active
                    or old_available != entity.available
                ):
                    significant_changes = True

                    # Log significant state changes in dev mode
                    if self.coordinator.dev_mode:
                        _LOGGER.debug(
                            "Updated entity %s current state: state %s->%s, active %s->%s, available %s->%s",
                            entity_id,
                            old_state,
                            entity.state,
                            old_is_active,
                            entity.is_active,
                            old_available,
                            entity.available,
                        )

                # Update entity probabilities synchronously (without triggering coordinator update yet)
                self.hass.async_create_task(entity.async_update())
            else:
                # Entity not available, mark as such
                if (
                    entity.available
                ):  # Only log if changing from available to unavailable
                    significant_changes = True
                entity.available = False
                _LOGGER.debug(
                    "Entity %s not available in HA, marked as unavailable", entity_id
                )

        _LOGGER.debug(
            "Initialized current states for %d/%d entities",
            entities_updated,
            len(self._entities),
        )

        # If any significant changes occurred, request a coordinator update
        if significant_changes:
            _LOGGER.debug(
                "Requesting coordinator update due to entity state initialization"
            )
            self.coordinator.request_update()

    async def _create_entities_from_config(self) -> dict[str, Entity]:
        """Create entities from current configuration."""
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
            for input_entity_id in inputs:
                prior = await self.coordinator.priors.calculate(
                    entity_id=input_entity_id,
                    entity_type=entity_type,
                    hass=self.hass,
                    primary_sensor=primary_sensor,
                    start_time=start_time,
                    end_time=end_time,
                    prior_type=PriorType.ENTITY,
                )
                entities[input_entity_id] = self._create_entity(
                    entity_id=input_entity_id,
                    entity_type=entity_type,
                    prior=prior,
                )
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

    async def _async_update_entity(self, entity_id: str) -> None:
        """Update a single entity."""
        if entity_id not in self._entities:
            return

        entity = self._entities[entity_id]
        try:
            # Store old probability to detect changes
            old_probability = entity.probability.decayed_probability
            old_active_state = entity.is_active

            await entity.async_update()
            _LOGGER.debug("Successfully updated entity %s", entity_id)

            # Check if significant change occurred
            new_probability = entity.probability.decayed_probability
            new_active_state = entity.is_active

            # IMMEDIATE coordinator update - bypass debouncing by forcing data update
            # This ensures sensors get updated immediately on every state change
            self.coordinator.request_update(force=True)

            if self.coordinator.dev_mode:
                _LOGGER.debug(
                    "Entity %s triggered IMMEDIATE coordinator update: "
                    "prob %.3f->%.3f, active %s->%s, state=%s, decay_factor=%.3f",
                    entity_id,
                    old_probability,
                    new_probability,
                    old_active_state,
                    new_active_state,
                    entity.state,
                    entity.probability.decay_factor,
                )
            else:
                _LOGGER.debug(
                    "Entity %s triggered IMMEDIATE coordinator update: "
                    "prob %.3f->%.3f, active %s->%s",
                    entity_id,
                    old_probability,
                    new_probability,
                    old_active_state,
                    new_active_state,
                )

        except (ValueError, AttributeError, RuntimeError) as err:
            _LOGGER.warning("Failed to update entity %s: %s", entity_id, err)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None
