"""Entity model."""

from dataclasses import dataclass
from datetime import timedelta

from homeassistant.util import dt as dt_util

from ..const import (
    CONF_APPLIANCES,
    CONF_DOOR_SENSORS,
    CONF_HISTORY_PERIOD,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_LIGHTS,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_WINDOW_SENSORS,
)
from ..coordinator import AreaOccupancyCoordinator
from .prior import Prior
from .entity_type import EntityType, InputTypeEnum
from ..utils import bayesian_update


@dataclass
class Entity:
    """Type for sensor state information."""

    type: EntityType
    state: str | float | bool | None = None
    is_active: bool = False
    probability: float = 0.0
    weighted_probability: float = 0.0
    last_changed: str | None = None
    available: bool = True
    prior: Prior | None = None


class EntityManager:
    """Manages entities."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Initialize the entities."""
        self.coordinator = coordinator
        self.config = coordinator.config
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

    def update_feature_state(
        self,
        entity_id: str,
        state: str | float | bool | None = None,
        is_active: bool | None = None,
        probability: float | None = None,
        weighted_probability: float | None = None,
        last_changed: str | None = None,
        available: bool | None = None,
    ) -> None:
        """Update a feature's state."""
        if entity_id not in self._entities:
            raise ValueError(f"Feature not found for entity: {entity_id}")

        entity = self._entities[entity_id]

        # Update only provided values
        if state is not None:
            entity.state = state
        if is_active is not None:
            entity.is_active = is_active
        if probability is not None:
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability value: {probability}")
            entity.probability = probability
        if weighted_probability is not None:
            if (
                not isinstance(weighted_probability, (int, float))
                or not 0 <= weighted_probability <= 1
            ):
                raise ValueError(
                    f"Invalid weighted probability value: {weighted_probability}"
                )
            entity.weighted_probability = weighted_probability
        if last_changed is not None:
            entity.last_changed = last_changed
        if available is not None:
            entity.available = available

    def reset_feature_state(self, entity_id: str) -> None:
        """Reset a feature's state to defaults."""
        if entity_id not in self._entities:
            raise ValueError(f"Feature not found for entity: {entity_id}")

        entity = self._entities[entity_id]
        entity.state = None
        entity.is_active = False
        entity.probability = 0.0
        entity.weighted_probability = 0.0
        entity.last_changed = None
        entity.available = True

    def reset_all_feature_states(self) -> None:
        """Reset all feature states to defaults."""
        for entity_id in self._entities:
            self.reset_feature_state(entity_id)

    async def map_inputs_to_entities(self) -> dict[str, Entity]:
        """Map inputs to entities."""
        history_days = self.config.get(CONF_HISTORY_PERIOD, 7)
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=history_days)
        primary_sensor = self.config.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
        if not primary_sensor:
            raise ValueError("Primary occupancy sensor must be configured")
        type_mappings = {
            InputTypeEnum.MOTION: self.config.get(CONF_MOTION_SENSORS, []),
            InputTypeEnum.MEDIA: self.config.get(CONF_MEDIA_DEVICES, []),
            InputTypeEnum.APPLIANCE: self.config.get(CONF_APPLIANCES, []),
            InputTypeEnum.DOOR: self.config.get(CONF_DOOR_SENSORS, []),
            InputTypeEnum.WINDOW: self.config.get(CONF_WINDOW_SENSORS, []),
            InputTypeEnum.LIGHT: self.config.get(CONF_LIGHTS, []),
            InputTypeEnum.ENVIRONMENTAL: self.config.get(CONF_ILLUMINANCE_SENSORS, [])
            + self.config.get(CONF_HUMIDITY_SENSORS, [])
            + self.config.get(CONF_TEMPERATURE_SENSORS, []),
        }
        entities: dict[str, Entity] = {}
        for input_type, inputs in type_mappings.items():
            feature_type = self.coordinator.feature_type_manager.get_feature_type(
                input_type
            )
            for input in inputs:
                entities[input] = Entity(
                    type=feature_type,
                    state=None,
                    is_active=False,
                    probability=0.0,
                    weighted_probability=0.0,
                    last_changed=None,
                    available=True,
                    prior=await self.coordinator.prior_manager.calculate(
                        entity_id=input,
                        hass=self.hass,
                        feature_type=feature_type,
                        primary_sensor=primary_sensor,
                        start_time=start_time,
                        end_time=end_time,
                    ),
                )
        return entities

    def get_entity_weight(self, entity_id: str) -> float:
        """Get the weight of a sensor."""
        return self.coordinator.feature_type_manager.get_feature_type(entity_id).weight

    def get_entity_active_states(self, entity_id: str) -> set[str]:
        """Get the active states of an entity."""
        active_states = self.coordinator.feature_type_manager.get_feature_type(
            entity_id
        ).active_states
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
            entity.prior,
            entity.type.prob_true,
            entity.type.prob_false,
            entity.is_active,
        )
        weight = float(entity.type.weight)
        weighted_prob = unweighted_prob * weight

        return entity.copy(
            probability=unweighted_prob,
            weighted_probability=weighted_prob,
            is_active=True,
        )
