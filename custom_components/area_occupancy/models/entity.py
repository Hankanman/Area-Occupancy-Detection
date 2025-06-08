"""Entity model."""

import asyncio
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
        # Race condition prevention
        self._pending_updates: dict[
            str, datetime
        ] = {}  # Track pending updates per entity
        self._update_tasks: dict[str, Any] = {}  # Track running update tasks per entity
        self._update_debounce_delay = 0.1  # 100ms debounce delay

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
        """Initialize the entities with proper restoration support."""
        # Check if we already have restored entities (from storage)
        if hasattr(self, "_entities") and self._entities:
            _LOGGER.debug(
                "Found %d restored entities, merging with current config",
                len(self._entities),
            )
            await self._merge_restored_with_config()
        else:
            _LOGGER.debug(
                "No restored entities found, creating fresh entities from config"
            )
            self._entities = await self.map_inputs_to_entities()

        self._setup_entity_tracking()
        _LOGGER.debug("EntityManager initialized with %d entities", len(self._entities))

    async def _merge_restored_with_config(self) -> None:
        """Merge restored entities with current configuration."""
        try:
            # Get current config entities
            config_entities = await self._get_config_entity_mapping()

            # Track what entities we need to keep, update, or create
            merged_entities: dict[str, Entity] = {}

            # 1. Process existing restored entities
            for entity_id, restored_entity in self._entities.items():
                if entity_id in config_entities:
                    # Entity still exists in config - keep with learned data but update type if needed
                    current_type = config_entities[entity_id]

                    # Update entity type configuration (weights, states) but preserve learned priors
                    if restored_entity.type.input_type == current_type.input_type:
                        # Same type - update configuration but keep learned data
                        restored_entity.type.weight = current_type.weight
                        restored_entity.type.active_states = current_type.active_states
                        restored_entity.type.active_range = current_type.active_range
                        # Keep learned priors: restored_entity.prior (contains learned data)
                        _LOGGER.debug(
                            "Updated config for existing entity %s", entity_id
                        )
                    else:
                        # Type changed - recreate with new type but try to preserve some data
                        _LOGGER.info(
                            "Entity type changed for %s: %s -> %s",
                            entity_id,
                            restored_entity.type.input_type,
                            current_type.input_type,
                        )
                        restored_entity = await self._recreate_entity_with_new_type(
                            entity_id, current_type, restored_entity
                        )

                    merged_entities[entity_id] = restored_entity
                    # Remove from config_entities so we don't create duplicate
                    del config_entities[entity_id]
                else:
                    # Entity removed from config - log but don't keep
                    _LOGGER.info(
                        "Entity %s removed from configuration, dropping stored data",
                        entity_id,
                    )

            # 2. Create new entities for any added to config
            for entity_id, entity_type in config_entities.items():
                _LOGGER.info(
                    "Creating new entity %s (type: %s)",
                    entity_id,
                    entity_type.input_type,
                )
                # Calculate fresh prior for new entity
                prior = await self._calculate_initial_prior(entity_id, entity_type)
                new_entity = self.create_entity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    prior=prior,
                )
                merged_entities[entity_id] = new_entity

            # 3. Replace entity dictionary
            added_count = len(config_entities)
            self._entities = merged_entities

            _LOGGER.info(
                "Entity merge complete: %d total entities, %d added from config",
                len(self._entities),
                added_count,
            )

        except (ValueError, KeyError, AttributeError, TypeError) as err:
            _LOGGER.error("Error merging restored entities with config: %s", err)
            # Fallback to fresh creation
            _LOGGER.warning("Falling back to fresh entity creation")
            self._entities = await self.map_inputs_to_entities()

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

    async def _recreate_entity_with_new_type(
        self, entity_id: str, new_type: EntityType, old_entity: Entity
    ) -> Entity:
        """Recreate entity with new type, preserving what data we can."""
        # Calculate new prior for the new type
        prior = await self._calculate_initial_prior(entity_id, new_type)

        # Create new entity but preserve some metadata
        return self.create_entity(
            entity_id=entity_id,
            entity_type=new_type,
            state=old_entity.state,  # Keep current state
            is_active=old_entity.is_active,  # Keep current active status
            last_changed=old_entity.last_changed,  # Keep last changed time
            available=old_entity.available,  # Keep availability
            prior=prior,  # Use new calculated prior
        )

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
            The created Entity instance, or existing one if already exists

        """
        # Return existing entity if it already exists
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

                # Update is_active status
                entity.is_active = is_active

                # Use coordinated update to prevent race conditions
                # This debounces rapid changes and ensures only one update per entity
                self.hass.async_create_task(self.coordinated_entity_update(entity_id))

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
                entities[input_entity_id] = self.create_entity(
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

    def get_entity_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics about all entities.

        Returns:
            Dictionary containing entity metrics and health information

        """
        now = dt_util.utcnow()
        entity_types: dict[str, dict[str, Any]] = {}
        probability_distribution: dict[str, int] = {
            "high_confidence": 0,  # > 0.8
            "medium_confidence": 0,  # 0.2 - 0.8
            "low_confidence": 0,  # < 0.2
        }
        last_update_age: dict[str, int] = {}

        metrics = {
            "total_entities": len(self._entities),
            "active_entities": sum(1 for e in self._entities.values() if e.is_active),
            "available_entities": sum(
                1 for e in self._entities.values() if e.available
            ),
            "unavailable_entities": sum(
                1 for e in self._entities.values() if not e.available
            ),
            "decaying_entities": sum(
                1 for e in self._entities.values() if e.decay.is_decaying
            ),
            "entity_types": entity_types,
            "probability_distribution": probability_distribution,
            "last_update_age": last_update_age,
        }

        # Analyze by entity type
        for entity in self._entities.values():
            entity_type = entity.type.input_type.value
            if entity_type not in entity_types:
                entity_types[entity_type] = {
                    "count": 0,
                    "active": 0,
                    "available": 0,
                    "avg_probability": 0.0,
                }

            type_metrics = entity_types[entity_type]
            type_metrics["count"] += 1
            if entity.is_active:
                type_metrics["active"] += 1
            if entity.available:
                type_metrics["available"] += 1
            type_metrics["avg_probability"] += entity.probability.decayed_probability

            # Probability distribution
            prob = entity.probability.decayed_probability
            if prob > 0.8:
                probability_distribution["high_confidence"] += 1
            elif prob > 0.2:
                probability_distribution["medium_confidence"] += 1
            else:
                probability_distribution["low_confidence"] += 1

            # Update age analysis
            if entity.last_updated:
                age_seconds = (now - entity.last_updated).total_seconds()
                age_category = (
                    "recent"
                    if age_seconds < 300
                    else "stale"
                    if age_seconds < 3600
                    else "very_stale"
                )
                last_update_age[age_category] = last_update_age.get(age_category, 0) + 1

        # Calculate averages
        for type_metrics in entity_types.values():
            if type_metrics["count"] > 0:
                type_metrics["avg_probability"] /= type_metrics["count"]

        return metrics

    def get_problematic_entities(self) -> dict[str, list[str]]:
        """Get entities that may need attention.

        Returns:
            Dictionary categorizing entities by potential issues

        """
        now = dt_util.utcnow()
        problems: dict[str, list[str]] = {
            "unavailable": [],
            "stale_updates": [],  # No updates > 1 hour
            "stuck_decay": [],  # Decaying for > 2 hours
            "extreme_probabilities": [],  # Very high/low probabilities
        }

        for entity_id, entity in self._entities.items():
            # Unavailable entities
            if not entity.available:
                problems["unavailable"].append(entity_id)

            # Stale updates
            if (
                entity.last_updated
                and (now - entity.last_updated).total_seconds() > 3600
            ):
                problems["stale_updates"].append(entity_id)

            # Stuck decay
            if entity.decay.is_decaying and entity.decay.decay_start_time:
                decay_duration = (now - entity.decay.decay_start_time).total_seconds()
                if decay_duration > 7200:  # 2 hours
                    problems["stuck_decay"].append(entity_id)

            # Extreme probabilities (might indicate sensor issues)
            prob = entity.probability.decayed_probability
            if prob > 0.95 or prob < 0.05:
                problems["extreme_probabilities"].append(entity_id)

        return problems

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._remove_state_listener is not None:
            self._remove_state_listener()
            self._remove_state_listener = None

        # Cancel any pending update tasks
        for entity_id, task in self._update_tasks.items():
            if not task.done():
                task.cancel()
                _LOGGER.debug("Cancelled pending update task for %s", entity_id)

        # Clear tracking dictionaries
        self._update_tasks.clear()
        self._pending_updates.clear()

    async def coordinated_entity_update(self, entity_id: str) -> None:
        """Coordinate entity updates to prevent race conditions.

        This method ensures:
        1. Only one update runs per entity at a time
        2. Rapid state changes are debounced
        3. The latest state is always used for calculations

        Args:
            entity_id: The entity ID to update

        """
        now = dt_util.utcnow()

        # Record this update request
        self._pending_updates[entity_id] = now

        # If there's already an update task running for this entity, let it handle this update
        if entity_id in self._update_tasks:
            _LOGGER.debug("Update already in progress for %s, debouncing", entity_id)
            return

        try:
            # Create and track the update task
            update_task = self.hass.async_create_task(
                self._debounced_entity_update(entity_id)
            )
            self._update_tasks[entity_id] = update_task
            await update_task

        except (asyncio.CancelledError, RuntimeError, ValueError) as err:
            _LOGGER.error("Error in coordinated update for %s: %s", entity_id, err)
        finally:
            # Clean up task tracking
            self._update_tasks.pop(entity_id, None)
            self._pending_updates.pop(entity_id, None)

    async def _debounced_entity_update(self, entity_id: str) -> None:
        """Debounced entity update that waits for rapid changes to settle.

        Args:
            entity_id: The entity ID to update

        """

        # Wait for debounce period to handle rapid state changes
        await asyncio.sleep(self._update_debounce_delay)

        # Check if there have been more recent update requests
        last_request_time = self._pending_updates.get(entity_id)
        if last_request_time:
            time_since_request = (dt_util.utcnow() - last_request_time).total_seconds()
            if time_since_request < self._update_debounce_delay:
                # More recent request, wait a bit more
                await asyncio.sleep(self._update_debounce_delay)

        # Perform the actual update
        if entity_id in self._entities:
            entity = self._entities[entity_id]
            try:
                await entity.async_update()
                _LOGGER.debug("Successfully updated entity %s", entity_id)

                # Invalidate coordinator's cached probability since entity probability changed
                self.coordinator.invalidate_probability_cache()

            except (ValueError, AttributeError, RuntimeError) as err:
                _LOGGER.warning("Failed to update entity %s: %s", entity_id, err)
