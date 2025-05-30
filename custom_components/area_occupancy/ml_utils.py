"""ML Utilities for Area Occupancy Detection.

This module provides common utilities and functions shared across the ML implementation
to reduce code duplication and improve maintainability.

Only functions that are actually reused by multiple modules are included here.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from .const import (
    CONF_ANALYSIS_METHOD,
    CONF_ML_CONFIDENCE_THRESHOLD,
    CONF_ML_ENABLED,
    DEFAULT_ANALYSIS_METHOD,
    DEFAULT_ML_CONFIDENCE_THRESHOLD,
    DEFAULT_ML_ENABLED,
)
from .types import AnalysisMethod, EntityType, SensorInputs

if TYPE_CHECKING:
    from .probabilities import Probabilities

_LOGGER = logging.getLogger(__name__)


class TemporalFeatureGenerator:
    """Utility class for generating temporal features.

    Used by:
    - ml_data_collector.py: prepare_ml_data()
    - calculate_prob.py: build_optimized_ml_features()
    - ml_feature_engineering.py: _add_temporal_features_to_dict()
    """

    @staticmethod
    def add_temporal_features_to_dict(
        features: dict[str, Any], timestamp: datetime | None = None
    ) -> None:
        """Add temporal features directly to feature dictionary.

        Args:
            features: Feature dictionary to modify in-place
            timestamp: Timestamp to use, defaults to current time

        """
        if timestamp is None:
            timestamp = dt_util.utcnow()

        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month

        # Basic temporal features
        features["hour_of_day"] = hour
        features["day_of_week"] = day_of_week
        features["month"] = month

        # Boolean features
        features["is_weekend"] = day_of_week >= 5
        features["is_workday"] = (day_of_week < 5) and (8 <= hour <= 17)

        # Time of day features
        features["is_night"] = (hour >= 22) or (hour <= 6)
        features["is_morning"] = 6 < hour <= 12
        features["is_afternoon"] = 12 < hour <= 18
        features["is_evening"] = 18 < hour < 22

        # Seasonal features
        features["is_winter"] = month in [12, 1, 2]
        features["is_spring"] = month in [3, 4, 5]
        features["is_summer"] = month in [6, 7, 8]
        features["is_fall"] = month in [9, 10, 11]


def build_optimized_ml_features(
    current_states: dict[str, Any],
    probabilities: "Probabilities",
    timestamp: datetime | None = None,
    state_manager: Any = None,
) -> dict[str, Any]:
    """Build ML features using optimized shared utilities.

    This function consolidates the feature building logic used across different
    ML modules to reduce code duplication.

    Used by:
    - calculate_prob.py: _prepare_ml_features()

    Args:
        current_states: Dictionary of current sensor states
        probabilities: Probabilities handler
        timestamp: Optional timestamp, defaults to current time
        state_manager: Unified state manager (required)

    Returns:
        Dictionary of ML features

    """
    try:
        if not state_manager:
            _LOGGER.error("StateManager is required for ML feature building")
            return {}

        # Initialize feature dictionary
        features = {}

        # Organize sensors by type
        sensor_groups = {entity_type.value: [] for entity_type in EntityType}

        for entity_id, state_info in current_states.items():
            entity_type = state_manager.get_entity_type(entity_id)
            if entity_type and _is_state_available(state_info):
                state_value = _get_state_value(state_info)

                # Use StateManager for active state determination
                is_active = state_manager.is_entity_active(entity_id, state_value)

                sensor_groups[entity_type.value].append(
                    {"entity_id": entity_id, "state": state_value, "active": is_active}
                )

        # Generate sensor type features
        for sensor_type, sensors in sensor_groups.items():
            count = len(sensors)
            active_count = sum(1 for s in sensors if s.get("active", False))

            features[f"{sensor_type}_count"] = count
            features[f"{sensor_type}_active_count"] = active_count
            features[f"{sensor_type}_active_duration"] = (
                active_count / count if count > 0 else 0.0
            )
            features[f"{sensor_type}_any_active"] = active_count > 0
            features[f"{sensor_type}_max_active"] = 1.0 if active_count > 0 else 0.0
            # state_changes default to 0 for real-time inference
            features[f"{sensor_type}_state_changes"] = 0

        # Add temporal features
        TemporalFeatureGenerator.add_temporal_features_to_dict(features, timestamp)

        # Add cross-sensor interaction features
        _add_cross_sensor_features(features)

        return features

    except Exception as err:
        _LOGGER.exception("Error preparing ML features: %s", err)
        return {}


# Helper functions used only internally by build_optimized_ml_features
def _is_state_available(state_info: Any) -> bool:
    """Check if state info indicates availability."""
    if isinstance(state_info, dict):
        return state_info.get("availability", False)
    return True  # Assume available if not dict format


def _get_state_value(state_info: Any) -> str | None:
    """Extract state value from state info."""
    if isinstance(state_info, dict):
        return state_info.get("state")
    return str(state_info) if state_info is not None else None


def _add_cross_sensor_features(features: dict[str, Any]) -> None:
    """Add cross-sensor interaction features to feature dictionary."""
    # Motion + Media interaction
    motion_any = features.get("motion_any_active", False)
    media_any = features.get("media_any_active", False)
    light_any = features.get("light_any_active", False)
    door_any = features.get("door_any_active", False)
    env_any = features.get("environmental_any_active", False)

    features["motion_and_media"] = motion_any and media_any
    features["motion_or_media"] = motion_any or media_any
    features["motion_and_light"] = motion_any and light_any
    features["door_and_motion"] = door_any and motion_any
    features["environmental_and_motion"] = env_any and motion_any

    # Activity intensity
    activity_durations = [
        features.get(f"{et}_active_duration", 0.0)
        for et in [entity_type.value for entity_type in EntityType]
    ]
    features["total_activity"] = sum(activity_durations)
    features["activity_diversity"] = sum(
        1 for duration in activity_durations if duration > 0
    )

    # State changes (always 0 for real-time inference)
    features["total_state_changes"] = 0


class MLConfigurationHelper:
    """Helper class for ML configuration access and validation."""

    @staticmethod
    def get_ml_config(config: dict[str, Any] | None) -> dict[str, Any]:
        """Extract ML configuration with defaults.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary with ML configuration values

        """
        if not config:
            config = {}

        return {
            "ml_enabled": config.get(CONF_ML_ENABLED, DEFAULT_ML_ENABLED),
            "analysis_method": MLConfigurationHelper.get_analysis_method(config),
            "ml_confidence_threshold": config.get(
                CONF_ML_CONFIDENCE_THRESHOLD, DEFAULT_ML_CONFIDENCE_THRESHOLD
            ),
        }

    @staticmethod
    def get_analysis_method(config: dict[str, Any]) -> AnalysisMethod:
        """Get analysis method from config with validation.

        Args:
            config: Configuration dictionary

        Returns:
            AnalysisMethod enum value

        """
        analysis_method_str = config.get(CONF_ANALYSIS_METHOD, DEFAULT_ANALYSIS_METHOD)
        try:
            return AnalysisMethod(analysis_method_str)
        except ValueError:
            _LOGGER.warning(
                "Invalid analysis method '%s', using deterministic", analysis_method_str
            )
            return AnalysisMethod.DETERMINISTIC


class EntityTypeMapper:
    """Helper class for entity type operations."""

    def __init__(self, probabilities: "Probabilities", state_manager: Any = None):
        """Initialize with probabilities handler and state manager."""
        self.probabilities = probabilities
        self.state_manager = state_manager

    def get_entity_types_mapping(self, entity_ids: list[str]) -> dict[str, EntityType]:
        """Get entity type mapping for multiple entities.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Dictionary mapping entity IDs to their types

        """
        if not self.state_manager:
            _LOGGER.error("StateManager is required for entity type mapping")
            return {}

        return {
            entity_id: entity_type
            for entity_id in entity_ids
            if (entity_type := self.state_manager.get_entity_type(entity_id))
            is not None
        }

    def group_entities_by_type(
        self, entity_states: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group entities by their types.

        Args:
            entity_states: Dictionary of entity states

        Returns:
            Dictionary mapping entity type names to lists of entity data

        """
        if not self.state_manager:
            _LOGGER.error("StateManager is required for entity type mapping")
            return {entity_type.value: [] for entity_type in EntityType}

        sensor_groups = {entity_type.value: [] for entity_type in EntityType}

        for entity_id, state_info in entity_states.items():
            entity_type = self.state_manager.get_entity_type(entity_id)
            if entity_type and self._is_state_available(state_info):
                state_value = self._get_state_value(state_info)
                is_active = self.is_entity_state_active(entity_id, state_value)

                sensor_groups[entity_type.value].append(
                    {"entity_id": entity_id, "state": state_value, "active": is_active}
                )

        return sensor_groups

    def is_entity_state_active(self, entity_id: str, state: str | None) -> bool:
        """Check if entity state is active using StateManager.

        Args:
            entity_id: Entity ID
            state: Entity state

        Returns:
            True if state is active for the entity type

        """
        if not self.state_manager:
            _LOGGER.error("StateManager is required for active state determination")
            return False

        return self.state_manager.is_entity_active(entity_id, state)

    @staticmethod
    def _is_state_available(state_info: Any) -> bool:
        """Check if state info indicates availability."""
        if isinstance(state_info, dict):
            return state_info.get("availability", False)
        return True  # Assume available if not dict format

    @staticmethod
    def _get_state_value(state_info: Any) -> str | None:
        """Extract state value from state info."""
        if isinstance(state_info, dict):
            return state_info.get("state")
        return str(state_info) if state_info is not None else None


class SensorFeatureGenerator:
    """Utility class for generating sensor-based features."""

    @staticmethod
    def generate_sensor_type_features(
        sensor_groups: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Generate features for each sensor type.

        Args:
            sensor_groups: Dictionary mapping sensor types to entity data

        Returns:
            Dictionary of sensor type features

        """
        features = {}

        for sensor_type, sensors in sensor_groups.items():
            count = len(sensors)
            active_count = sum(1 for s in sensors if s.get("active", False))

            features[f"{sensor_type}_count"] = count
            features[f"{sensor_type}_active_count"] = active_count
            features[f"{sensor_type}_active_duration"] = (
                active_count / count if count > 0 else 0.0
            )
            features[f"{sensor_type}_any_active"] = active_count > 0
            features[f"{sensor_type}_max_active"] = 1.0 if active_count > 0 else 0.0
            # state_changes default to 0 for real-time inference
            features[f"{sensor_type}_state_changes"] = 0

        return features

    @staticmethod
    def add_cross_sensor_features_to_dict(features: dict[str, Any]) -> None:
        """Add cross-sensor interaction features to feature dictionary.

        Args:
            features: Feature dictionary to modify in-place

        """
        # Motion + Media interaction
        motion_any = features.get("motion_any_active", False)
        media_any = features.get("media_any_active", False)
        light_any = features.get("light_any_active", False)
        door_any = features.get("door_any_active", False)
        env_any = features.get("environmental_any_active", False)

        features["motion_and_media"] = motion_any and media_any
        features["motion_or_media"] = motion_any or media_any
        features["motion_and_light"] = motion_any and light_any
        features["door_and_motion"] = door_any and motion_any
        features["environmental_and_motion"] = env_any and motion_any

        # Activity intensity
        activity_durations = [
            features.get(f"{et}_active_duration", 0.0)
            for et in [entity_type.value for entity_type in EntityType]
        ]
        features["total_activity"] = sum(activity_durations)
        features["activity_diversity"] = sum(
            1 for duration in activity_durations if duration > 0
        )

        # State changes (always 0 for real-time inference)
        features["total_state_changes"] = 0


class FeatureValidationHelper:
    """Helper class for feature validation operations."""

    @staticmethod
    def validate_feature_schema(
        features: dict[str, Any], expected_schema: dict[str, str]
    ) -> bool:
        """Validate that features match expected schema.

        Args:
            features: Feature dictionary to validate
            expected_schema: Expected feature names and types

        Returns:
            True if features are valid, False otherwise

        """
        try:
            for feature_name, expected_type in expected_schema.items():
                if feature_name not in features:
                    _LOGGER.warning("Missing feature: %s", feature_name)
                    return False

                value = features[feature_name]

                # Check type compatibility
                if expected_type == "float" and not isinstance(value, (int, float)):
                    _LOGGER.warning(
                        "Feature %s expected %s but got %s",
                        feature_name,
                        expected_type,
                        type(value).__name__,
                    )
                    return False
                elif expected_type == "bool" and not isinstance(value, bool):
                    _LOGGER.warning(
                        "Feature %s expected %s but got %s",
                        feature_name,
                        expected_type,
                        type(value).__name__,
                    )
                    return False

            return True

        except Exception as err:
            _LOGGER.exception("Error validating feature schema: %s", err)
            return False


class MLErrorHandler:
    """Centralized error handling for ML operations."""

    @staticmethod
    def log_and_return_none(
        operation: str, error: Exception, entity_id: str = ""
    ) -> None:
        """Log ML error and return None consistently.

        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            entity_id: Optional entity ID for context

        """
        entity_context = f" for {entity_id}" if entity_id else ""
        _LOGGER.warning("Error in %s%s: %s", operation, entity_context, error)

    @staticmethod
    def handle_feature_preparation_error(error: Exception) -> dict[str, Any]:
        """Handle feature preparation errors consistently.

        Args:
            error: The exception that occurred

        Returns:
            Empty feature dictionary

        """
        _LOGGER.exception("Error preparing ML features: %s", error)
        return {}


def create_sensor_data_mapping(
    sensor_inputs: SensorInputs, current_states: dict[str, str], timestamp: datetime
) -> list[dict[str, Any]]:
    """Create sensor data mapping using standardized approach.

    Args:
        sensor_inputs: Sensor input configuration
        current_states: Current sensor states
        timestamp: Timestamp for the data

    Returns:
        List of sensor data dictionaries

    """
    sensor_data = []

    # Define sensor type mappings for efficient processing
    sensor_mappings = [
        (sensor_inputs.motion_sensors, EntityType.MOTION, {"on", "detected"}),
        (sensor_inputs.media_devices, EntityType.MEDIA, {"playing", "paused"}),
        (sensor_inputs.door_sensors, EntityType.DOOR, {"open", "closed"}),
        (sensor_inputs.virtual_sensors, EntityType.WASP_IN_BOX, {"on", "detected"}),
        (sensor_inputs.lights, EntityType.LIGHT, {"on"}),
        (sensor_inputs.window_sensors, EntityType.WINDOW, {"open"}),
        (sensor_inputs.appliances, EntityType.APPLIANCE, {"on", "standby"}),
        (sensor_inputs.illuminance_sensors, EntityType.ENVIRONMENTAL, set()),
        (sensor_inputs.humidity_sensors, EntityType.ENVIRONMENTAL, set()),
        (sensor_inputs.temperature_sensors, EntityType.ENVIRONMENTAL, set()),
    ]

    # Process each sensor type efficiently
    for entities, entity_type, active_states in sensor_mappings:
        for entity_id in entities:
            state = current_states.get(entity_id)
            if state is not None:
                # For environmental sensors, active_duration is always 1.0
                if entity_type == EntityType.ENVIRONMENTAL:
                    active_duration = 1.0
                else:
                    active_duration = 1.0 if state in active_states else 0.0

                sensor_data.append(
                    {
                        "timestamp": timestamp,
                        "entity_id": entity_id,
                        "entity_type": entity_type.value,
                        "state": state,
                        "active_duration": active_duration,
                        "state_changes": 0,  # Not available for real-time
                        "available": True,
                    }
                )

    return sensor_data
