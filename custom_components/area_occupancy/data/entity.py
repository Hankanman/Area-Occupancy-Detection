"""Entity model."""

import bisect
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from ..db import AreaOccupancyDB
from ..utils import bayesian_probability
from .decay import Decay
from .entity_type import EntityType

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

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
            prob_given_true=self.prob_given_true,
            prob_given_false=self.prob_given_false,
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

    def update_likelihood(
        self, prob_given_true: float, prob_given_false: float
    ) -> None:
        """Update the likelihood of the entity."""
        self.prob_given_true = prob_given_true
        self.prob_given_false = prob_given_false
        self.last_updated = dt_util.utcnow()

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
                    prob_given_true=self.prob_given_true,
                    prob_given_false=self.prob_given_false,
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
            "prob_given_true": self.prob_given_true,
            "prob_given_false": self.prob_given_false,
            "decay": self.decay.to_dict(),
            "last_updated": self.last_updated.isoformat(),
            "previous_evidence": self.previous_evidence,
            "previous_probability": self.previous_probability,
        }


class EntityFactory:
    """Factory for creating entities from various sources."""

    def __init__(self, coordinator: "AreaOccupancyCoordinator") -> None:
        """Initialize the factory."""
        self.coordinator = coordinator
        self.config = coordinator.config

    def create_from_db(self, entity_obj: "AreaOccupancyDB.Entities") -> Entity:
        """Create entity from storage data."""
        # Find the existing entity type from the coordinator
        entity_type = self.coordinator.entity_types.get_entity_type(
            entity_obj.entity_type
        )

        return self._create_entity(
            entity_id=entity_obj.entity_id,
            entity_type=entity_type,
            stored_prob_given_true=entity_obj.prob_given_true,
            stored_prob_given_false=entity_obj.prob_given_false,
            stored_decay=None,  # Decay data not stored in Entities table
            last_updated=entity_obj.last_updated,
            previous_evidence=None,  # Not stored in Entities table
            previous_probability=0.0,  # Default value
        )

    def create_from_config_spec(self, entity_id: str, spec: dict[str, Any]) -> Entity:
        """Create entity from configuration specification."""
        entity_type = self.coordinator.entity_types.get_entity_type(spec["input_type"])

        # Override entity type with config-specific values
        entity_type.weight = spec.get("weight", entity_type.weight)
        entity_type.active_states = spec.get("active_states", entity_type.active_states)
        entity_type.active_range = spec.get("active_range", entity_type.active_range)

        return self._create_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            stored_prob_given_true=None,
            stored_prob_given_false=None,
            stored_decay=None,
            last_updated=None,
            previous_evidence=None,
            previous_probability=0.0,
        )

    def _create_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        stored_prob_given_true: float | None = None,
        stored_prob_given_false: float | None = None,
        stored_decay: dict[str, Any] | None = None,
        last_updated: str | None = None,
        previous_evidence: bool | None = None,
        previous_probability: float = 0.0,
    ) -> Entity:
        """Create entity with the given parameters."""

        # Create decay
        if stored_decay:
            decay = Decay.from_dict(stored_decay)
        else:
            decay = Decay(
                last_trigger_ts=dt_util.utcnow().timestamp(),
                half_life=self.config.decay.half_life,
            )

        # Create entity
        return Entity(
            entity_id=entity_id,
            type=entity_type,
            prob_given_true=stored_prob_given_true,
            prob_given_false=stored_prob_given_false,
            decay=decay,
            coordinator=self.coordinator,
            last_updated=(
                dt_util.parse_datetime(last_updated)
                if last_updated
                else dt_util.utcnow()
            ),
            previous_evidence=previous_evidence,
            previous_probability=previous_probability,
        )

    def create_all_from_config(self) -> dict[str, Entity]:
        """Create all entities from current configuration."""
        config_specs = self.config.get_entity_specifications(self.coordinator)
        entities = {}

        for entity_id, spec in config_specs.items():
            _LOGGER.debug("Creating entity %s from config spec", entity_id)
            entities[entity_id] = self.create_from_config_spec(entity_id, spec)

        return entities


class EntityManager:
    """Manages entities with simplified creation and storage logic."""

    def __init__(self, coordinator: "AreaOccupancyCoordinator") -> None:
        """Initialize the entity manager."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self.hass = coordinator.hass
        self._entities: dict[str, Entity] = {}
        self._factory = EntityFactory(coordinator)

    async def __post_init__(self) -> None:
        """Post init - initialize entities from storage or config."""
        if self._entities:
            await self._sync_with_config()
        else:
            await self._create_from_config()

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
        """Create entity manager from dictionary."""
        manager = cls(coordinator=coordinator)

        if "entities" not in data:
            raise ValueError(
                f"Invalid storage format: missing 'entities' key in data structure. "
                f"Available keys: {list(data.keys())}."
            )

        entities_data = data["entities"]

        try:
            manager._entities = {
                entity_id: manager._factory.create_from_db(entity)
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
        """Clean up resources and recreate from config."""
        self._entities.clear()
        await self._create_from_config()

    async def _create_from_config(self) -> None:
        """Create entities from current configuration."""
        # Validate configuration first
        validation_errors = self.config.validate_entity_configuration()
        if validation_errors:
            _LOGGER.warning(
                "Entity configuration validation issues: %s", validation_errors
            )

        # Create all entities using factory
        self._entities = self._factory.create_all_from_config()
        _LOGGER.info("Created %d entities from configuration", len(self._entities))

    async def _sync_with_config(self) -> None:
        """Sync existing entities with current configuration."""
        # Validate configuration first
        validation_errors = self.config.validate_entity_configuration()
        if validation_errors:
            _LOGGER.warning(
                "Entity configuration validation issues: %s", validation_errors
            )

        # Get required entities from config
        required_entities = self._factory.create_all_from_config()
        updated_entities: dict[str, Entity] = {}

        # Process existing entities
        for entity_id, existing_entity in self._entities.items():
            if entity_id in required_entities:
                # Entity still exists in config - update type configuration
                new_entity = required_entities[entity_id]

                # Update type configuration but preserve learned data
                existing_entity.type.weight = new_entity.type.weight
                existing_entity.type.active_states = new_entity.type.active_states
                existing_entity.type.active_range = new_entity.type.active_range

                updated_entities[entity_id] = existing_entity
                del required_entities[entity_id]  # Mark as processed
            else:
                _LOGGER.info(
                    "Entity %s removed from configuration, dropping stored data",
                    entity_id,
                )

        # Add new entities
        updated_entities.update(required_entities)

        self._entities = updated_entities
        _LOGGER.info("Entity sync complete: %d total entities", len(self._entities))

    def update_all_entity_likelihoods(self):
        """Compute P(sensor=true|occupied) and P(sensor=true|empty) per sensor.

        Use motion-based labels for 'occupied'.
        """
        db = self.coordinator.db
        entry_id = self.coordinator.entry_id
        sensor_ids = self.coordinator.config.sensors.motion

        with db.get_session() as session:
            # Get all sensor configs for this area
            sensors = session.query(db.Entities).filter_by(entry_id=entry_id).all()
            occupied_intervals = (
                session.query(db.Intervals.start_time, db.Intervals.end_time)
                .join(
                    db.Entities,
                    db.Intervals.entity_id == db.Entities.entity_id,
                )
                .filter(
                    db.Entities.entry_id == entry_id,
                    db.Intervals.entity_id.in_(sensor_ids),
                    db.Intervals.state == "on",
                )
                .order_by(db.Intervals.start_time)
                .all()
            )

            # Get truth timeline from motion sensors (sorted for binary search)
            occupied_times = [(start, end) for start, end in occupied_intervals]

            def is_occupied(ts):
                """Efficiently check if timestamp falls within any occupied interval using binary search."""
                if not occupied_times:
                    return False

                # Binary search to find the rightmost interval that starts <= ts
                idx = bisect.bisect_right([start for start, _ in occupied_times], ts)

                # Check if ts falls within the interval found
                if idx > 0:
                    start, end = occupied_times[idx - 1]
                    if start <= ts < end:
                        return True

                return False

            sensor_entity_ids = [entity.entity_id for entity in sensors]
            if not sensor_entity_ids:
                return

            # Get all intervals for all sensors in one query
            all_intervals = (
                session.query(db.Intervals)
                .filter(db.Intervals.entity_id.in_(sensor_entity_ids))
                .all()
            )

            # Group intervals by entity_id for processing
            intervals_by_entity = defaultdict(list)
            for interval in all_intervals:
                intervals_by_entity[interval.entity_id].append(interval)

            # Process each sensor's intervals
            now = dt_util.utcnow()
            for entity in sensors:
                # Get intervals for this specific sensor
                intervals = intervals_by_entity[entity.entity_id]

                # Count true readings during occupied vs empty
                true_occ = false_occ = true_empty = false_empty = 0

                for interval in intervals:
                    occ = is_occupied(interval.start_time)
                    if interval.state == "on":
                        if occ:
                            true_occ += interval.duration_seconds
                        else:
                            true_empty += interval.duration_seconds
                    elif occ:
                        false_occ += interval.duration_seconds
                    else:
                        false_empty += interval.duration_seconds

                # Avoid division by zero
                prob_given_true = (
                    true_occ / (true_occ + false_occ)
                    if (true_occ + false_occ) > 0
                    else 0.5
                )
                prob_given_false = (
                    true_empty / (true_empty + false_empty)
                    if (true_empty + false_empty) > 0
                    else 0.5
                )
                updated_entity = db.Entities(
                    entity_id=entity.entity_id,
                    prob_given_true=prob_given_true,
                    prob_given_false=prob_given_false,
                    last_updated=now,
                )
                session.add(updated_entity)
                self.get_entity(entity.entity_id).update_likelihood(
                    prob_given_true, prob_given_false
                )

            session.commit()
