"""ML Feature Engineering for Area Occupancy Detection.

This module handles feature engineering for machine learning models,
transforming raw sensor data into features suitable for training and inference.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from homeassistant.util import dt as dt_util

from .types import EntityType, SensorInputs

_LOGGER = logging.getLogger(__name__)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw sensor data into ML features.

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
        # Create feature matrix grouped by timestamp
        feature_rows = []

        for timestamp, group in df.groupby("timestamp"):
            # Convert pandas timestamp to datetime
            ts: datetime
            try:
                if isinstance(timestamp, datetime):
                    ts = timestamp
                else:
                    # Try to handle pandas Timestamp
                    try:
                        ts = timestamp.to_pydatetime()  # type: ignore[attr-defined]
                    except AttributeError:
                        # Fallback: convert to datetime
                        ts = pd.to_datetime(str(timestamp)).to_pydatetime()
            except (ValueError, AttributeError, TypeError):
                # Final fallback: use current time
                _LOGGER.warning(
                    "Failed to convert timestamp %s, using current time", timestamp
                )
                ts = datetime.now()

            features = _build_timestamp_features(ts, group)
            feature_rows.append(features)

        result_df = pd.DataFrame(feature_rows)

        # Add derived features
        result_df = _add_temporal_features(result_df)
        result_df = _add_cross_sensor_features(result_df)
        result_df = _add_historical_features(result_df)

        _LOGGER.debug(
            "Built feature matrix with %d samples and %d features",
            len(result_df),
            len(result_df.columns) - 1,  # exclude timestamp
        )

        return result_df

    except Exception as err:
        _LOGGER.exception("Error building feature matrix: %s", err)
        return pd.DataFrame()


def _build_timestamp_features(
    timestamp: datetime, group: pd.DataFrame
) -> dict[str, Any]:
    """Build features for a single timestamp from sensor group.

    Args:
        timestamp: Timestamp for this feature row
        group: DataFrame with sensor data for this timestamp

    Returns:
        Dictionary of features for this timestamp

    """
    features: dict[str, Any] = {"timestamp": timestamp}

    # Add base sensor features by entity type
    for entity_type in EntityType:
        type_data = group[group["entity_type"] == entity_type.value]

        if not type_data.empty:
            features[f"{entity_type.value}_count"] = len(type_data)
            features[f"{entity_type.value}_active_duration"] = type_data[
                "active_duration"
            ].mean()
            features[f"{entity_type.value}_state_changes"] = type_data[
                "state_changes"
            ].sum()
            features[f"{entity_type.value}_max_active"] = type_data[
                "active_duration"
            ].max()
            features[f"{entity_type.value}_any_active"] = (
                type_data["active_duration"] > 0
            ).any()
        else:
            features[f"{entity_type.value}_count"] = 0
            features[f"{entity_type.value}_active_duration"] = 0.0
            features[f"{entity_type.value}_state_changes"] = 0
            features[f"{entity_type.value}_max_active"] = 0.0
            features[f"{entity_type.value}_any_active"] = False

    return features


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to the feature matrix.

    Args:
        df: DataFrame with timestamp column

    Returns:
        DataFrame with additional temporal features

    """
    if "timestamp" not in df.columns:
        return df

    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Basic temporal features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["is_workday"] = (df["timestamp"].dt.dayofweek < 5) & (
        df["timestamp"].dt.hour.between(8, 17)
    )
    df["is_night"] = df["timestamp"].dt.hour.between(22, 6)
    df["is_morning"] = df["timestamp"].dt.hour.between(6, 12)
    df["is_afternoon"] = df["timestamp"].dt.hour.between(12, 18)
    df["is_evening"] = df["timestamp"].dt.hour.between(18, 22)

    # Seasonal features
    df["month"] = df["timestamp"].dt.month
    df["is_winter"] = df["timestamp"].dt.month.isin([12, 1, 2])
    df["is_spring"] = df["timestamp"].dt.month.isin([3, 4, 5])
    df["is_summer"] = df["timestamp"].dt.month.isin([6, 7, 8])
    df["is_fall"] = df["timestamp"].dt.month.isin([9, 10, 11])

    return df


def _add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sensor interaction features.

    Args:
        df: DataFrame with individual sensor features

    Returns:
        DataFrame with additional cross-sensor features

    """
    # Motion + Media interaction
    if "motion_any_active" in df.columns and "media_any_active" in df.columns:
        df["motion_and_media"] = df["motion_any_active"] & df["media_any_active"]
        df["motion_or_media"] = df["motion_any_active"] | df["media_any_active"]

    # Motion + Light interaction
    if "motion_any_active" in df.columns and "light_any_active" in df.columns:
        df["motion_and_light"] = df["motion_any_active"] & df["light_any_active"]

    # Door + Motion interaction (could indicate entry/exit)
    if "door_any_active" in df.columns and "motion_any_active" in df.columns:
        df["door_and_motion"] = df["door_any_active"] & df["motion_any_active"]

    # Environmental + Motion interaction
    if "environmental_any_active" in df.columns and "motion_any_active" in df.columns:
        df["environmental_and_motion"] = (
            df["environmental_any_active"] & df["motion_any_active"]
        )

    # Activity intensity (sum of all active durations)
    activity_cols = [col for col in df.columns if col.endswith("_active_duration")]
    if activity_cols:
        df["total_activity"] = df[activity_cols].sum(axis=1)
        df["activity_diversity"] = (df[activity_cols] > 0).sum(axis=1)

    # State change intensity
    change_cols = [col for col in df.columns if col.endswith("_state_changes")]
    if change_cols:
        df["total_state_changes"] = df[change_cols].sum(axis=1)

    return df


def _add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical/rolling window features.

    Args:
        df: DataFrame sorted by timestamp

    Returns:
        DataFrame with additional historical features

    """
    if len(df) < 2:
        return df

    # Sort by timestamp to ensure proper rolling windows
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rolling features for motion activity
    if "motion_active_duration" in df.columns:
        df["motion_rolling_mean_5"] = (
            df["motion_active_duration"].rolling(5, min_periods=1).mean()
        )
        df["motion_rolling_max_5"] = (
            df["motion_active_duration"].rolling(5, min_periods=1).max()
        )
        df["motion_trend"] = df["motion_active_duration"].diff()

    # Rolling features for media activity
    if "media_active_duration" in df.columns:
        df["media_rolling_mean_5"] = (
            df["media_active_duration"].rolling(5, min_periods=1).mean()
        )
        df["media_trend"] = df["media_active_duration"].diff()

    # Time since last significant activity
    if "motion_any_active" in df.columns:
        motion_active_indices = df[df["motion_any_active"]].index
        df["time_since_motion"] = df.index.to_series().apply(
            lambda x: x - motion_active_indices[motion_active_indices <= x].max()
            if not motion_active_indices[motion_active_indices <= x].empty
            else x
        )

    if "media_any_active" in df.columns:
        media_active_indices = df[df["media_any_active"]].index
        df["time_since_media"] = df.index.to_series().apply(
            lambda x: x - media_active_indices[media_active_indices <= x].max()
            if not media_active_indices[media_active_indices <= x].empty
            else x
        )

    return df


def prepare_inference_features(
    sensor_inputs: SensorInputs, current_states: dict[str, str]
) -> dict[str, Any]:
    """Prepare features for real-time inference from current sensor states.

    Args:
        sensor_inputs: Sensor input configuration containing entity IDs
        current_states: Dictionary mapping entity IDs to their current states

    Returns:
        Dictionary of features ready for ML model inference

    """
    try:
        # Convert current states to DataFrame-like structure
        timestamp = dt_util.utcnow()

        # Create a minimal dataframe for feature engineering
        sensor_data = []

        # Add motion sensor states
        for entity_id in sensor_inputs.motion_sensors:
            state = current_states.get(entity_id)
            if state is not None:
                sensor_data.append(
                    {
                        "timestamp": timestamp,
                        "entity_id": entity_id,
                        "entity_type": EntityType.MOTION.value,
                        "state": state,
                        "active_duration": 1.0 if state in ["on", "detected"] else 0.0,
                        "state_changes": 0,  # Not available for real-time
                        "available": True,
                    }
                )

        # Add media device states
        for entity_id in sensor_inputs.media_devices:
            state = current_states.get(entity_id)
            if state is not None:
                sensor_data.append(
                    {
                        "timestamp": timestamp,
                        "entity_id": entity_id,
                        "entity_type": EntityType.MEDIA.value,
                        "state": state,
                        "active_duration": 1.0
                        if state in ["playing", "paused"]
                        else 0.0,
                        "state_changes": 0,
                        "available": True,
                    }
                )

        # Add door sensor states
        for entity_id in sensor_inputs.door_sensors:
            state = current_states.get(entity_id)
            if state is not None:
                sensor_data.append(
                    {
                        "timestamp": timestamp,
                        "entity_id": entity_id,
                        "entity_type": EntityType.DOOR.value,
                        "state": state,
                        "active_duration": 1.0 if state in ["open", "closed"] else 0.0,
                        "state_changes": 0,
                        "available": True,
                    }
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
