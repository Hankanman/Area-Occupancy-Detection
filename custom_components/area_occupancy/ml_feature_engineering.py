"""ML Feature Engineering for Area Occupancy Detection.

This module handles feature engineering for machine learning models,
transforming raw sensor data into features suitable for training and inference.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from homeassistant.util import dt as dt_util

from .ml_utils import TemporalFeatureGenerator
from .types import EntityType, SensorInputs

_LOGGER = logging.getLogger(__name__)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw sensor data into ML features (optimized version).

    Args:
        df: DataFrame with raw sensor data including:
            - timestamp: datetime column
            - entity_id: sensor entity ID
            - entity_type: EntityType enum value
            - state: sensor state
            - active_duration: proportion of time active in window
            - state_changes: number of state changes in window

    Returns:
        DataFrame with engineered features ready for ML training/inference

    """
    if df.empty:
        return pd.DataFrame()

    try:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Build features using vectorized operations
        result_df = _build_features_vectorized(df)

        # Add derived features in sequence
        result_df = _add_temporal_features_vectorized(result_df)
        result_df = _add_cross_sensor_features_vectorized(result_df)
        result_df = _add_historical_features_vectorized(result_df)

        _LOGGER.debug(
            "Built feature matrix with %d samples and %d features",
            len(result_df),
            len(result_df.columns) - 1,  # exclude timestamp
        )

        return result_df

    except Exception as err:
        _LOGGER.exception("Error building feature matrix: %s", err)
        return pd.DataFrame()


def _build_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Build features using vectorized operations for better performance.

    Args:
        df: Input DataFrame with sensor data

    Returns:
        DataFrame with base features aggregated by timestamp

    """
    # Pre-compute all entity types to avoid repeated lookups
    entity_types = [entity_type.value for entity_type in EntityType]

    # Group by timestamp once
    grouped = df.groupby("timestamp")

    # Pre-allocate result dictionary for better memory efficiency
    result_data = {
        "timestamp": [],
    }

    # Initialize all feature columns
    for entity_type in entity_types:
        result_data[f"{entity_type}_count"] = []
        result_data[f"{entity_type}_active_duration"] = []
        result_data[f"{entity_type}_state_changes"] = []
        result_data[f"{entity_type}_max_active"] = []
        result_data[f"{entity_type}_any_active"] = []

    # Process each timestamp group
    for timestamp, group in grouped:
        result_data["timestamp"].append(timestamp)

        # Create a lookup for this group's data by entity type
        type_data_lookup = {}
        for entity_type in entity_types:
            type_data = group[group["entity_type"] == entity_type]
            type_data_lookup[entity_type] = type_data

        # Vectorized feature calculation for all types
        for entity_type in entity_types:
            type_data = type_data_lookup[entity_type]

            if not type_data.empty:
                result_data[f"{entity_type}_count"].append(len(type_data))
                result_data[f"{entity_type}_active_duration"].append(
                    type_data["active_duration"].mean()
                )
                result_data[f"{entity_type}_state_changes"].append(
                    type_data["state_changes"].sum()
                )
                result_data[f"{entity_type}_max_active"].append(
                    type_data["active_duration"].max()
                )
                result_data[f"{entity_type}_any_active"].append(
                    (type_data["active_duration"] > 0).any()
                )
            else:
                # Default values for missing types
                result_data[f"{entity_type}_count"].append(0)
                result_data[f"{entity_type}_active_duration"].append(0.0)
                result_data[f"{entity_type}_state_changes"].append(0)
                result_data[f"{entity_type}_max_active"].append(0.0)
                result_data[f"{entity_type}_any_active"].append(False)

    return pd.DataFrame(result_data)


def _add_temporal_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features using vectorized operations.

    Args:
        df: DataFrame with timestamp column

    Returns:
        DataFrame with additional temporal features

    """
    if "timestamp" not in df.columns or df.empty:
        return df

    # All temporal features calculated in one pass using vectorized operations
    dt_accessor = df["timestamp"].dt

    # Basic temporal features (vectorized)
    df["hour_of_day"] = dt_accessor.hour
    df["day_of_week"] = dt_accessor.dayofweek
    df["month"] = dt_accessor.month

    # Boolean features (vectorized)
    df["is_weekend"] = dt_accessor.dayofweek >= 5
    df["is_workday"] = (dt_accessor.dayofweek < 5) & (dt_accessor.hour.between(8, 17))

    # Time of day features (vectorized with numpy where for better performance)
    hour = dt_accessor.hour
    df["is_night"] = (hour >= 22) | (hour <= 6)
    df["is_morning"] = (hour > 6) & (hour <= 12)
    df["is_afternoon"] = (hour > 12) & (hour <= 18)
    df["is_evening"] = (hour > 18) & (hour < 22)

    # Seasonal features (vectorized)
    month = dt_accessor.month
    df["is_winter"] = month.isin([12, 1, 2])
    df["is_spring"] = month.isin([3, 4, 5])
    df["is_summer"] = month.isin([6, 7, 8])
    df["is_fall"] = month.isin([9, 10, 11])

    return df


def _add_cross_sensor_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sensor interaction features using vectorized operations.

    Args:
        df: DataFrame with individual sensor features

    Returns:
        DataFrame with additional cross-sensor features

    """
    # Pre-check for required columns to avoid repeated lookups
    motion_any = "motion_any_active" in df.columns
    media_any = "media_any_active" in df.columns
    light_any = "light_any_active" in df.columns
    door_any = "door_any_active" in df.columns
    env_any = "environmental_any_active" in df.columns

    # Motion + Media interaction (vectorized)
    if motion_any and media_any:
        df["motion_and_media"] = df["motion_any_active"] & df["media_any_active"]
        df["motion_or_media"] = df["motion_any_active"] | df["media_any_active"]

    # Motion + Light interaction
    if motion_any and light_any:
        df["motion_and_light"] = df["motion_any_active"] & df["light_any_active"]

    # Door + Motion interaction
    if door_any and motion_any:
        df["door_and_motion"] = df["door_any_active"] & df["motion_any_active"]

    # Environmental + Motion interaction
    if env_any and motion_any:
        df["environmental_and_motion"] = (
            df["environmental_any_active"] & df["motion_any_active"]
        )

    # Activity intensity - vectorized sum and count
    activity_cols = [col for col in df.columns if col.endswith("_active_duration")]
    if activity_cols:
        df["total_activity"] = df[activity_cols].sum(axis=1)
        df["activity_diversity"] = (df[activity_cols] > 0).sum(axis=1)

    # State change intensity - vectorized sum
    change_cols = [col for col in df.columns if col.endswith("_state_changes")]
    if change_cols:
        df["total_state_changes"] = df[change_cols].sum(axis=1)

    return df


def _add_historical_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical/rolling window features using optimized operations.

    Args:
        df: DataFrame sorted by timestamp

    Returns:
        DataFrame with additional historical features

    """
    if len(df) < 2:
        return df

    # Sort by timestamp to ensure proper rolling windows
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rolling features with optimized window calculations
    rolling_configs = [
        ("motion_active_duration", "motion"),
        ("media_active_duration", "media"),
    ]

    for col_name, prefix in rolling_configs:
        if col_name in df.columns:
            series = df[col_name]
            # Calculate all rolling features at once
            rolling = series.rolling(5, min_periods=1)
            df[f"{prefix}_rolling_mean_5"] = rolling.mean()
            df[f"{prefix}_rolling_max_5"] = rolling.max()
            df[f"{prefix}_trend"] = series.diff()

    # Optimized time-since calculations using vectorized operations
    _add_time_since_features_vectorized(df, "motion_any_active", "time_since_motion")
    _add_time_since_features_vectorized(df, "media_any_active", "time_since_media")

    return df


def _add_time_since_features_vectorized(
    df: pd.DataFrame, active_col: str, output_col: str
) -> None:
    """Add time-since-last-active features using vectorized operations.

    Args:
        df: DataFrame to modify in-place
        active_col: Column name indicating activity
        output_col: Output column name for time-since feature

    """
    if active_col not in df.columns:
        return

    # Use cumulative operations for better performance than apply
    active_mask = df[active_col]
    if not active_mask.any():
        df[output_col] = df.index
        return

    # Get indices where active
    active_indices = df.index[active_mask]

    # Use numpy searchsorted for efficient lookup
    indices = df.index.values
    last_active_idx = np.searchsorted(active_indices, indices, side="right") - 1

    # Handle case where no previous active found
    last_active_idx = np.clip(last_active_idx, 0, len(active_indices) - 1)

    # Calculate time since last active
    df[output_col] = indices - active_indices[last_active_idx]

    # Set to current index for samples before first active
    first_active_idx = active_indices[0] if len(active_indices) > 0 else len(df)
    mask_before_first = indices < first_active_idx
    df.loc[mask_before_first, output_col] = indices[mask_before_first]


def prepare_inference_features(
    sensor_inputs: SensorInputs, current_states: dict[str, str]
) -> dict[str, Any]:
    """Prepare features for real-time inference (optimized version).

    Args:
        sensor_inputs: Sensor input configuration containing entity IDs
        current_states: Dictionary mapping entity IDs to their current states

    Returns:
        Dictionary of features ready for ML model inference

    """
    try:
        timestamp = dt_util.utcnow()

        # Pre-build sensor data using list comprehension for better performance
        sensor_data = _build_sensor_data_optimized(
            sensor_inputs, current_states, timestamp
        )

        if not sensor_data:
            return {}

        # Create DataFrame and engineer features
        df = pd.DataFrame(sensor_data)
        feature_df = build_feature_matrix(df)

        if feature_df.empty:
            return {}

        # Return the latest (only) row as a dictionary
        features = feature_df.iloc[-1].to_dict()

        # Remove timestamp as it's not needed for inference
        features.pop("timestamp", None)

        return features

    except Exception as err:
        _LOGGER.exception("Error preparing inference features: %s", err)
        return {}


def _build_sensor_data_optimized(
    sensor_inputs: SensorInputs, current_states: dict[str, str], timestamp: datetime
) -> list[dict[str, Any]]:
    """Build sensor data list using optimized approach.

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


def get_feature_importance_names() -> list[str]:
    """Get list of expected feature names for feature importance analysis.

    Returns:
        List of feature names that should be present in trained models

    """
    base_features = []

    # Entity type features
    for entity_type in EntityType:
        base_features.extend(
            [
                f"{entity_type.value}_count",
                f"{entity_type.value}_active_duration",
                f"{entity_type.value}_state_changes",
                f"{entity_type.value}_max_active",
                f"{entity_type.value}_any_active",
            ]
        )

    # Temporal features
    temporal_features = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_workday",
        "is_night",
        "is_morning",
        "is_afternoon",
        "is_evening",
        "month",
        "is_winter",
        "is_spring",
        "is_summer",
        "is_fall",
    ]

    # Cross-sensor features
    cross_features = [
        "motion_and_media",
        "motion_or_media",
        "motion_and_light",
        "door_and_motion",
        "environmental_and_motion",
        "total_activity",
        "activity_diversity",
        "total_state_changes",
    ]

    # Historical features
    historical_features = [
        "motion_rolling_mean_5",
        "motion_rolling_max_5",
        "motion_trend",
        "media_rolling_mean_5",
        "media_trend",
        "time_since_motion",
        "time_since_media",
    ]

    return base_features + temporal_features + cross_features + historical_features


def prepare_inference_features_batch(
    sensor_inputs: SensorInputs,
    current_states: dict[str, str],
    entity_manager: Any,
    probabilities_handler: Any = None,
) -> dict[str, Any]:
    """Prepare features for real-time inference using batch operations (most optimized version).

    This version uses batch operations and pre-computed mappings for maximum performance
    in real-time inference scenarios.

    Args:
        sensor_inputs: Sensor input configuration containing entity IDs
        current_states: Dictionary mapping entity IDs to their current states
        entity_manager: Unified entity manager (required)
        probabilities_handler: Optional probabilities handler (deprecated - kept for compatibility)

    Returns:
        Dictionary of features ready for ML model inference

    """
    try:
        if not entity_manager:
            _LOGGER.error("EntityManager is required for inference feature preparation")
            return {}

        timestamp = dt_util.utcnow()

        # Collect all entity IDs for batch processing
        all_entity_ids = []
        sensor_type_mapping = {}

        # Create mapping of entities to their types efficiently
        entity_mappings = [
            (sensor_inputs.motion_sensors, EntityType.MOTION),
            (sensor_inputs.media_devices, EntityType.MEDIA),
            (sensor_inputs.door_sensors, EntityType.DOOR),
            (sensor_inputs.virtual_sensors, EntityType.WASP_IN_BOX),
            (sensor_inputs.lights, EntityType.LIGHT),
            (sensor_inputs.window_sensors, EntityType.WINDOW),
            (sensor_inputs.appliances, EntityType.APPLIANCE),
            (sensor_inputs.illuminance_sensors, EntityType.ENVIRONMENTAL),
            (sensor_inputs.humidity_sensors, EntityType.ENVIRONMENTAL),
            (sensor_inputs.temperature_sensors, EntityType.ENVIRONMENTAL),
        ]

        for entities, entity_type in entity_mappings:
            for entity_id in entities:
                if entity_id in current_states:
                    all_entity_ids.append(entity_id)
                    sensor_type_mapping[entity_id] = entity_type

        if not all_entity_ids:
            return {}

        # Use EntityManager for batch processing
        relevant_states = {eid: current_states[eid] for eid in all_entity_ids}
        active_states = entity_manager.check_multiple_entities_active(relevant_states)

        # Build features using vectorized approach
        features = _build_inference_features_vectorized(
            timestamp, sensor_type_mapping, current_states, active_states
        )

        return features

    except Exception as err:
        _LOGGER.exception("Error preparing batch inference features: %s", err)
        return {}


def _build_inference_features_vectorized(
    timestamp: datetime,
    sensor_type_mapping: dict[str, EntityType],
    current_states: dict[str, str],
    active_states: dict[str, bool],
) -> dict[str, Any]:
    """Build inference features using vectorized operations.

    Args:
        timestamp: Current timestamp
        sensor_type_mapping: Mapping of entity IDs to their types
        current_states: Current sensor states
        active_states: Pre-computed active state information

    Returns:
        Dictionary of features

    """
    # Initialize feature dictionary with all entity types
    features = {}
    entity_types = [entity_type.value for entity_type in EntityType]

    # Initialize all features to zero/false
    for entity_type in entity_types:
        features[f"{entity_type}_count"] = 0
        features[f"{entity_type}_active_duration"] = 0.0
        features[f"{entity_type}_state_changes"] = 0
        features[f"{entity_type}_max_active"] = 0.0
        features[f"{entity_type}_any_active"] = False

    # Group entities by type for efficient processing
    type_groups = {}
    for entity_id, entity_type in sensor_type_mapping.items():
        if entity_type.value not in type_groups:
            type_groups[entity_type.value] = []
        type_groups[entity_type.value].append(entity_id)

    # Calculate features for each type
    for entity_type, entity_ids in type_groups.items():
        count = len(entity_ids)
        active_count = sum(1 for eid in entity_ids if active_states.get(eid, False))

        features[f"{entity_type}_count"] = count
        features[f"{entity_type}_active_duration"] = (
            active_count / count if count > 0 else 0.0
        )
        features[f"{entity_type}_max_active"] = 1.0 if active_count > 0 else 0.0
        features[f"{entity_type}_any_active"] = active_count > 0
        # state_changes remains 0 for real-time inference

    # Add temporal features efficiently
    _add_temporal_features_to_dict(features, timestamp)

    # Add cross-sensor features
    _add_cross_sensor_features_to_dict(features)

    return features


def _add_temporal_features_to_dict(
    features: dict[str, Any], timestamp: datetime
) -> None:
    """Add temporal features directly to feature dictionary."""
    # Use the shared temporal feature generator utility
    TemporalFeatureGenerator.add_temporal_features_to_dict(features, timestamp)


def _add_cross_sensor_features_to_dict(features: dict[str, Any]) -> None:
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
