"""Entity model."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, Optional

from homeassistant.core import CALLBACK_TYPE, callback
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.util import dt as dt_util

from ..utils import bayesian_probability, validate_datetime, validate_prob
from .decay import DECAY_INTERVAL, Decay
from .entity_type import EntityType, InputType
from .prior import Prior

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass
class Entity:
    """Type for sensor state information."""

    entity_id: str
    type: EntityType
    probability: float
    prior: Prior
    decay: Decay
    state: str | float | bool | None = None
    is_active: bool = False
    available: bool = True
    last_updated: datetime = field(default_factory=dt_util.utcnow)
    # Decay timer management - not serialized
    _coordinator: Optional["AreaOccupancyCoordinator"] = field(
        default=None, init=False, repr=False
    )
    _decay_timer: CALLBACK_TYPE | None = field(default=None, init=False, repr=False)
    previous_probability: float = field(default=0.0, init=False, repr=False)
    previous_is_active: bool = field(default=False, init=False, repr=False)
    previous_state: str | float | bool | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        """Post init."""
        # Validate last_updated
        self.last_updated = validate_datetime(self.last_updated)

    def set_coordinator(self, coordinator: "AreaOccupancyCoordinator") -> None:
        """Set the coordinator reference for timer management."""
        self._coordinator = coordinator

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for storage."""
        return {
            "entity_id": self.entity_id,
            "type": self.type.input_type.value,
            "probability": self.probability,
            "prior": self.prior.to_dict(),
            "decay": self.decay.to_dict(),
            "state": self.state,
            "is_active": self.is_active,
            "available": self.available,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], coordinator: "AreaOccupancyCoordinator"
    ) -> "Entity":
        """Create entity from dictionary."""
        input_type = InputType(data["type"])

        entity = cls(
            entity_id=data["entity_id"],
            type=coordinator.entity_types.get_entity_type(input_type),
            probability=data["probability"],
            prior=Prior.from_dict(data["prior"]),
            decay=Decay(
                is_decaying=False,
                decay_start_time=None,
                decay_start_probability=validate_prob(0.0),
                decay_window=coordinator.config.decay.window,
                decay_enabled=coordinator.config.decay.enabled,
                decay_factor=validate_prob(1.0),
            ),
            state=data["state"],
            is_active=data["is_active"],
            available=data["available"],
            last_updated=data["last_updated"],
        )
        entity.set_coordinator(coordinator)

        # Initialize the previous state fields to current values for restored entities
        entity.previous_probability = entity.probability
        entity.previous_is_active = entity.is_active

        # If entity was decaying when saved, restart the decay timer
        if entity.decay.is_decaying:
            entity.start_decay_timer()

        return entity

    def update_probability(self, *, preserve_previous_state: bool = False) -> None:
        """Calculate and update entity probability with decay and prior checks.

        This is the unified method for all probability updates to ensure consistency
        across entity creation, updates, and decay processes.

        The previous state values are stored in _previous_probability and _previous_is_active
        fields and are updated at the beginning of each call unless preserve_previous_state is True.

        Args:
            preserve_previous_state: If True, don't overwrite _previous_* fields (used when
                                   caller has already set them to capture state transitions)

        Returns:
            None

        """
        # Store current values as previous for comparison, unless caller wants to preserve them
        if not preserve_previous_state:
            self.previous_probability = self.probability
            self.previous_is_active = self.is_active

        # Update last_updated timestamp
        self.last_updated = dt_util.utcnow()

        # Step 1: Calculate probability based on current sensor state
        current_probability = bayesian_probability(
            prior=self.prior.prior,
            prob_given_true=self.prior.prob_given_true,
            prob_given_false=self.prior.prob_given_false,
            is_active=self.is_active,
        )

        # Step 2: Handle decay state transitions using decay module logic
        if self.decay.should_start_decay(self.previous_is_active, self.is_active):
            # Start decay using decay module
            self.decay.start_decay(self.previous_probability)

            # Start the timer for continuous updates
            self.start_decay_timer()

            # For the first decay cycle, maintain the previous probability
            final_probability = self.previous_probability

        elif self.decay.should_stop_decay(self.previous_is_active, self.is_active):
            # Stop decay using decay module
            self.decay.stop_decay()
            self._stop_decay_timer()
            final_probability = current_probability

        elif self.decay.is_decaying:
            # Step 3: Apply decay if active
            decayed_probability, decay_factor = self.decay.update_decay(
                current_probability=current_probability,
                previous_probability=self.previous_probability,
            )
            final_probability = decayed_probability

            # Check if decay completed using decay module
            if self.decay.is_decay_complete(final_probability):
                self._stop_decay_timer()
                # Stop decay and reset decay state
                self.decay.stop_decay()
                # Reset is_active to False when decay completes
                self.is_active = False

                # Force immediate coordinator update to reflect decay completion
                if self._coordinator:
                    self._coordinator.request_update(
                        force=True,
                        message=f"Decay completed for {self.entity_id}, forcing final update",
                    )

        else:
            # No decay, use current probability
            final_probability = current_probability

        # Step 4: Update entity probability
        probability_changed = abs(self.probability - final_probability) > 0.001
        self.probability = final_probability

        if probability_changed:
            if self._coordinator:
                self._coordinator.request_update(
                    force=True, message="Entity state changed, forcing update"
                )

    def start_decay_timer(self) -> None:
        """Start the decay timer that fires every second."""
        # Respect global decay enabled setting
        if not self.decay.decay_enabled:
            return

        if not self._coordinator or not self._coordinator.hass:
            _LOGGER.warning(
                "Cannot start decay timer for %s: no coordinator", self.entity_id
            )
            return

        if self._decay_timer is not None:
            # Timer already running, don't start another
            return

        self._schedule_next_decay_update()

    def _stop_decay_timer(self) -> None:
        """Stop the decay timer."""
        if self._decay_timer is not None:
            self._decay_timer()
            self._decay_timer = None

    def _schedule_next_decay_update(self) -> None:
        """Schedule the next decay update in DECAY_INTERVAL seconds."""
        if not self._coordinator or not self._coordinator.hass:
            return

        next_update = dt_util.utcnow() + timedelta(seconds=DECAY_INTERVAL)
        self._decay_timer = async_track_point_in_time(
            self._coordinator.hass, self._handle_decay_timer, next_update
        )

    async def _handle_decay_timer(self, _now: datetime) -> None:
        """Handle decay timer firing - update probability with decay."""
        self._decay_timer = None

        # Check if decay is globally disabled
        if not self.decay.decay_enabled:
            _LOGGER.debug(
                "Decay disabled globally, stopping timer for entity %s", self.entity_id
            )
            self.decay.stop_decay()
            return

        if not self.decay.is_decaying:
            # Decay was stopped, don't continue
            return

        try:
            # Update probability with decay using unified method
            self.update_probability()

            # Schedule next decay update if still decaying
            if self.decay.is_decaying:
                self._schedule_next_decay_update()

        except (ValueError, ZeroDivisionError, TypeError, AttributeError) as err:
            _LOGGER.error("Error in decay timer for entity %s: %s", self.entity_id, err)
            # Stop decay on error to prevent continuous failures
            self._stop_decay_timer()
            self.decay.stop_decay()

    def stop_decay_completely(self) -> None:
        """Stop decay and timer completely (public interface)."""
        self.decay.stop_decay()
        self._stop_decay_timer()

    def cleanup(self) -> None:
        """Clean up entity resources."""
        self._stop_decay_timer()


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
            if entity.is_active or entity.decay.is_decaying
        ]

    @property
    def inactive_entities(self) -> list[Entity]:
        """Get the inactive entities."""
        return [
            entity
            for entity in self._entities.values()
            if not entity.is_active and not entity.decay.is_decaying
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
                entity_id: Entity.from_dict(entity, coordinator)
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

        # Initialize all entities with current states
        self._initialize_current_states()

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

                # Update decay configuration from current config
                restored_entity.decay.decay_enabled = self.config.decay.enabled
                restored_entity.decay.decay_window = self.config.decay.window

                # If decay was disabled in config, stop any active decay
                if not self.config.decay.enabled and restored_entity.decay.is_decaying:
                    _LOGGER.info(
                        "Decay disabled in config, stopping active decay for entity %s",
                        entity_id,
                    )
                    restored_entity.stop_decay_completely()

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

        entity_mapping: dict[str, EntityType] = {}

        # Add the primary occupancy sensor as a motion-type entity with high weight
        # This is the most reliable indicator and should be included in probability calculation
        primary_entity_type = self.coordinator.entity_types.get_entity_type(
            InputType.MOTION
        )
        entity_mapping[primary_sensor] = primary_entity_type

        for input_type, entity_ids in type_mappings.items():
            entity_type = self.coordinator.entity_types.get_entity_type(input_type)
            for entity_id in entity_ids:
                # Avoid adding the primary sensor twice if it's also in the motion list
                if entity_id != primary_sensor:
                    entity_mapping[entity_id] = entity_type

        return entity_mapping

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
                prior=entity_type.prior,
                prob_given_true=entity_type.prob_true,
                prob_given_false=entity_type.prob_false,
                last_updated=dt_util.utcnow(),
            )

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

        # Create probability instance - will be calculated properly below
        prob = validate_prob(0.0)

        # Create decay instance using configuration values
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=validate_prob(0.0),
            decay_window=self.config.decay.window,
            decay_enabled=self.config.decay.enabled,
            decay_factor=validate_prob(1.0),
        )

        # Create prior instance if not provided
        if prior is None:
            prior = Prior(
                prior=entity_type.prior,
                prob_given_true=entity_type.prob_true,
                prob_given_false=entity_type.prob_false,
                last_updated=dt_util.utcnow(),
            )

        entity = Entity(
            entity_id=entity_id,
            type=entity_type,
            probability=prob,
            prior=prior,
            decay=decay,
            state=state,
            is_active=is_active,
            available=available,
            last_updated=dt_util.utcnow(),
        )
        entity.set_coordinator(self.coordinator)

        # Calculate initial probability using unified method
        entity.update_probability()

        # If entity was decaying when saved, restart the decay timer
        if entity.decay.is_decaying:
            entity.start_decay_timer()

        return entity

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
        async def async_state_changed_listener(event) -> None:
            """Handle state changes for tracked entities."""
            try:
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")

                if entity_id not in self._entities:
                    return

                entity = self._entities[entity_id]

                # Capture previous state for transition detection before any updates
                entity.previous_probability = entity.probability
                entity.previous_is_active = entity.is_active

                # Update entity state
                is_available = bool(
                    new_state
                    and new_state.state not in ["unknown", "unavailable", None, ""]
                )
                current_state_val = new_state.state if is_available else None

                # Update entity properties
                entity.state = current_state_val
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

                # Update entity probabilities using unified method with preserved previous state
                entity.update_probability(preserve_previous_state=True)

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

        for entity_id, entity in self._entities.items():
            ha_state = self.hass.states.get(entity_id)
            if ha_state and ha_state.state not in ["unknown", "unavailable", None, ""]:
                # Set entity state from current HA state
                entity.state = ha_state.state
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

                # Update entity probability based on current state
                entity.update_probability()
            else:
                # Entity not available in HA
                entity.available = False
                _LOGGER.debug(
                    "Entity %s not available in HA, marked as unavailable", entity_id
                )

        _LOGGER.debug(
            "Initialized current states for %d/%d entities",
            entities_updated,
            len(self._entities),
        )

    async def _create_entities_from_config(self) -> dict[str, Entity]:
        """Create entities from current configuration."""

        type_mappings = {
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

        entities: dict[str, Entity] = {}

        # First pass: Create all entities with default priors
        for input_type, inputs in type_mappings.items():
            entity_type = self.coordinator.entity_types.get_entity_type(input_type)
            for input_entity_id in inputs:
                # Create entity with default priors from entity type
                default_prior = Prior(
                    prior=entity_type.prior,
                    prob_given_true=entity_type.prob_true,
                    prob_given_false=entity_type.prob_false,
                    last_updated=dt_util.utcnow(),
                )
                entities[input_entity_id] = self._create_entity(
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
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Clean up all entity decay timers
        for entity in self._entities.values():
            entity.stop_decay_completely()
            entity.cleanup()
