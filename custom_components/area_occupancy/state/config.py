"""Configuration management for state handling."""

from typing import Any, Dict, Optional, cast

from ..types import EntityType, ProbabilityConfig


class SensorConfiguration:
    """Manages sensor configuration and type mapping."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize configuration from raw config."""
        self._config = config
        self._entity_types: Dict[str, EntityType] = {}
        self._sensor_weights: Dict[str, float] = {}
        self._probability_configs: Dict[str, ProbabilityConfig] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize configuration from raw config."""
        self._map_entity_types()
        self._calculate_weights()
        self._setup_probability_configs()

    def _map_entity_types(self) -> None:
        """Map entity IDs to their types based on configuration."""
        type_mappings = {
            EntityType.MOTION: self._config.get("motion_sensors", []),
            EntityType.MEDIA: self._config.get("media_devices", []),
            EntityType.APPLIANCE: self._config.get("appliances", []),
            EntityType.DOOR: self._config.get("door_sensors", []),
            EntityType.WINDOW: self._config.get("window_sensors", []),
            EntityType.LIGHT: self._config.get("lights", []),
            EntityType.ENVIRONMENTAL: self._config.get("environmental_sensors", []),
        }

        for entity_type, entity_ids in type_mappings.items():
            for entity_id in entity_ids:
                self._entity_types[entity_id] = entity_type

    def _calculate_weights(self) -> None:
        """Calculate sensor weights based on configuration."""
        # Default weight is 1.0
        default_weight = 1.0

        # Get custom weights from config if available
        custom_weights = self._config.get("sensor_weights", {})

        # Apply weights to all configured sensors
        for entity_id in self._entity_types:
            self._sensor_weights[entity_id] = custom_weights.get(
                entity_id, default_weight
            )

    def _setup_probability_configs(self) -> None:
        """Set up probability configurations for each sensor type."""
        default_config: ProbabilityConfig = {
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "default_prior": 0.5,
            "weight": 1.0,
            "active_states": {"on", "true", "1", "open", "detected"},
        }

        # Get custom configs from config if available
        custom_configs = self._config.get("probability_configs", {})

        # Set up configs for each entity type
        for entity_type in EntityType:
            type_config = custom_configs.get(entity_type.value, {})
            self._probability_configs[entity_type.value] = cast(
                ProbabilityConfig,
                {
                    "prob_given_true": type_config.get(
                        "prob_given_true", default_config["prob_given_true"]
                    ),
                    "prob_given_false": type_config.get(
                        "prob_given_false", default_config["prob_given_false"]
                    ),
                    "default_prior": type_config.get(
                        "default_prior", default_config["default_prior"]
                    ),
                    "weight": type_config.get("weight", default_config["weight"]),
                    "active_states": set(
                        type_config.get(
                            "active_states", default_config["active_states"]
                        )
                    ),
                },
            )

    def get_entity_type(self, entity_id: str) -> Optional[EntityType]:
        """Get the type of an entity."""
        return self._entity_types.get(entity_id)

    def get_sensor_weight(self, entity_id: str) -> float:
        """Get the weight of a sensor."""
        return self._sensor_weights.get(entity_id, 1.0)

    def get_probability_config(self, entity_type: str) -> ProbabilityConfig:
        """Get probability configuration for a sensor type."""
        default_config: ProbabilityConfig = {
            "prob_given_true": 0.8,
            "prob_given_false": 0.2,
            "default_prior": 0.5,
            "weight": 1.0,
            "active_states": {"on", "true", "1", "open", "detected"},
        }
        return self._probability_configs.get(entity_type, default_config)

    def is_active_state(self, entity_type: str, state: str) -> bool:
        """Check if a state is considered active for a sensor type."""
        config = self.get_probability_config(entity_type)
        return state.lower() in config.get("active_states", set())

    def get_all_entity_ids(self) -> list[str]:
        """Get all configured entity IDs."""
        return list(self._entity_types.keys())

    def get_entity_ids_by_type(self, entity_type: EntityType) -> list[str]:
        """Get all entity IDs of a specific type."""
        return [
            entity_id
            for entity_id, type_ in self._entity_types.items()
            if type_ == entity_type
        ]
