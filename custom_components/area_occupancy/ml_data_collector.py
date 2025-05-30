"""ML Data Collector for Area Occupancy Detection.

This module handles historical data extraction and training set preparation
for machine learning models.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from homeassistant.core import HomeAssistant
from homeassistant.helpers.recorder import get_instance

from .const import ML_FEATURE_WINDOW_SIZE, ML_MIN_TRAINING_SAMPLES
from .types import EntityType, MLTrainingData

_LOGGER = logging.getLogger(__name__)


class TrainingSetBuilder:
    """Build training datasets for ML models."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the training set builder."""
        self.hass = hass
        self._recorder = get_instance(hass)

    async def snapshot(
        self,
        sensor_map: dict[str, EntityType],
        start: datetime,
        end: datetime,
        primary_sensor: str | None = None,
    ) -> pd.DataFrame | None:
        """Create a training dataset snapshot from historical data.

        Args:
            sensor_map: Mapping of entity IDs to sensor types
            start: Start time for data collection
            end: End time for data collection
            primary_sensor: Primary occupancy sensor for labeling (optional)

        Returns:
            DataFrame with features and labels, or None if insufficient data

        """
        if not self._recorder:
            _LOGGER.warning("Recorder not available, cannot build training set")
            return None

        try:
            # Get historical states for all sensors
            _LOGGER.debug(
                "Collecting historical data from %s to %s for %d sensors",
                start,
                end,
                len(sensor_map),
            )

            all_data = []

            # Collect data for each sensor
            for entity_id, entity_type in sensor_map.items():
                entity_data = await self._get_entity_history(entity_id, start, end)
                if entity_data is not None and not entity_data.empty:
                    entity_data["entity_id"] = entity_id
                    entity_data["entity_type"] = entity_type.value
                    all_data.append(entity_data)

            if not all_data:
                _LOGGER.warning("No historical data found for any sensors")
                return None

            # Combine all sensor data
            combined_df = pd.concat(all_data, ignore_index=True)

            # Align data to time windows
            windowed_df = self._create_time_windows(combined_df, ML_FEATURE_WINDOW_SIZE)

            # Create labels from primary sensor if available
            if primary_sensor:
                windowed_df = await self._add_occupancy_labels(
                    windowed_df, primary_sensor, start, end
                )
            else:
                # Use heuristic labeling based on motion sensors
                windowed_df = self._add_heuristic_labels(windowed_df, sensor_map)

            # Check if we have enough samples
            if len(windowed_df) < ML_MIN_TRAINING_SAMPLES:
                _LOGGER.warning(
                    "Insufficient training samples: %d < %d required",
                    len(windowed_df),
                    ML_MIN_TRAINING_SAMPLES,
                )
                return None

            _LOGGER.info(
                "Created training dataset with %d samples covering %s to %s",
                len(windowed_df),
                start,
                end,
            )

            return windowed_df

        except Exception as err:
            _LOGGER.exception("Error building training set: %s", err)
            return None

    async def _get_entity_history(
        self, entity_id: str, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        """Get historical states for a single entity."""
        try:
            # Query recorder for state history
            history = await self._recorder.async_add_executor_job(
                self._get_state_history_sync, entity_id, start, end
            )

            if not history:
                _LOGGER.debug("No history found for entity %s", entity_id)
                return None

            # Convert to DataFrame
            data = []
            for state in history:
                data.append(
                    {
                        "timestamp": state.last_changed,
                        "state": state.state,
                        "attributes": dict(state.attributes),
                        "available": state.state not in ["unknown", "unavailable"],
                    }
                )

            if not data:
                return None

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            return df

        except Exception as err:
            _LOGGER.warning("Error getting history for %s: %s", entity_id, err)
            return None

    def _get_state_history_sync(
        self, entity_id: str, start: datetime, end: datetime
    ) -> list[Any]:
        """Get state history from recorder synchronously."""
        from homeassistant.components.recorder.history import get_significant_states

        try:
            states = get_significant_states(
                self.hass,
                start,
                end,
                entity_ids=[entity_id],
                minimal_response=False,
                significant_changes_only=True,
            )
            return states.get(entity_id, [])
        except Exception as err:
            _LOGGER.warning("Error in sync history query for %s: %s", entity_id, err)
            return []

    def _create_time_windows(
        self, df: pd.DataFrame, window_minutes: int
    ) -> pd.DataFrame:
        """Create time-based windows for feature aggregation."""
        if df.empty:
            return df

        # Round timestamps to window boundaries
        df["window"] = df["timestamp"].dt.floor(f"{window_minutes}min")

        # Group by entity and window, aggregate states
        windowed_data = []

        for (entity_id, entity_type, window), group in df.groupby(
            ["entity_id", "entity_type", "window"]
        ):
            # Calculate aggregated features for this window
            features = {
                "timestamp": window,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "state_changes": len(group),
                "active_duration": self._calculate_active_duration(group),
                "final_state": group.iloc[-1]["state"],
                "available": group["available"].all(),
            }
            windowed_data.append(features)

        return pd.DataFrame(windowed_data)

    def _calculate_active_duration(self, group: pd.DataFrame) -> float:
        """Calculate how long sensors were active in a time window."""
        if group.empty:
            return 0.0

        # Simple heuristic: count "active" states
        active_states = ["on", "playing", "paused", "open", "closed"]
        active_count = sum(1 for state in group["state"] if state in active_states)

        return active_count / len(group) if len(group) > 0 else 0.0

    async def _add_occupancy_labels(
        self, df: pd.DataFrame, primary_sensor: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Add occupancy labels based on primary sensor."""
        if df.empty:
            return df

        # Get primary sensor history
        primary_history = await self._get_entity_history(primary_sensor, start, end)

        if primary_history is None or primary_history.empty:
            _LOGGER.warning("No primary sensor history available for labeling")
            return self._add_heuristic_labels(df, {})

        # Create occupancy labels by matching timestamps
        df["occupied"] = False

        for _, row in df.iterrows():
            window_start = row["timestamp"]
            window_end = window_start + pd.Timedelta(minutes=ML_FEATURE_WINDOW_SIZE)

            # Check if primary sensor was active during this window
            mask = (primary_history["timestamp"] >= window_start) & (
                primary_history["timestamp"] < window_end
            )
            window_states = primary_history[mask]

            if not window_states.empty:
                # Consider occupied if sensor was active during window
                active_states = ["on", "occupied", "detected"]
                df.loc[df["timestamp"] == window_start, "occupied"] = any(
                    state in active_states for state in window_states["state"]
                )

        return df

    def _add_heuristic_labels(
        self, df: pd.DataFrame, sensor_map: dict[str, EntityType]
    ) -> pd.DataFrame:
        """Add heuristic occupancy labels based on motion sensor activity."""
        if df.empty:
            return df

        # Group by timestamp and check for motion activity
        df["occupied"] = False

        for timestamp, group in df.groupby("timestamp"):
            # Look for motion sensors in this window
            motion_active = any(
                (
                    row["entity_type"] == EntityType.MOTION.value
                    and row["active_duration"] > 0.3
                )  # At least 30% active
                for _, row in group.iterrows()
            )

            # Additional heuristics for media activity
            media_active = any(
                (
                    row["entity_type"] == EntityType.MEDIA.value
                    and row["active_duration"] > 0.5
                )  # At least 50% active
                for _, row in group.iterrows()
            )

            # Label as occupied if motion or significant media activity
            df.loc[df["timestamp"] == timestamp, "occupied"] = (
                motion_active or media_active
            )

        return df

    def prepare_ml_data(self, df: pd.DataFrame) -> MLTrainingData | None:
        """Convert DataFrame to MLTrainingData format."""
        if df is None or df.empty:
            return None

        try:
            _LOGGER.debug("Starting prepare_ml_data with DataFrame shape %s", df.shape)

            # Extract features by entity type and timestamp
            features = {}
            labels = []
            timestamps = []
            entity_ids = []

            grouped = df.groupby("timestamp")
            _LOGGER.debug("Found %d unique timestamps to process", len(grouped))

            for timestamp, group in grouped:
                timestamp_features = {}

                # Aggregate features by entity type
                for entity_type in EntityType:
                    type_data = group[group["entity_type"] == entity_type.value]
                    if not type_data.empty:
                        timestamp_features[f"{entity_type.value}_count"] = len(
                            type_data
                        )
                        timestamp_features[f"{entity_type.value}_active_duration"] = (
                            type_data["active_duration"].mean()
                        )
                        timestamp_features[f"{entity_type.value}_state_changes"] = (
                            type_data["state_changes"].sum()
                        )
                    else:
                        timestamp_features[f"{entity_type.value}_count"] = 0
                        timestamp_features[f"{entity_type.value}_active_duration"] = 0.0
                        timestamp_features[f"{entity_type.value}_state_changes"] = 0

                # Add temporal features
                # Convert timestamp to datetime object
                dt: datetime
                try:
                    if isinstance(timestamp, datetime):
                        dt = timestamp
                    else:
                        # Try to handle pandas Timestamp
                        try:
                            dt = timestamp.to_pydatetime()  # type: ignore[attr-defined]
                        except AttributeError:
                            # Fallback: convert string to datetime
                            dt = pd.to_datetime(str(timestamp)).to_pydatetime()
                except (ValueError, AttributeError, TypeError):
                    # Fallback: use current time if conversion fails
                    _LOGGER.warning(
                        "Failed to convert timestamp %s, using current time", timestamp
                    )
                    dt = datetime.now()

                # Use shared temporal feature generation
                from .ml_utils import TemporalFeatureGenerator

                temporal_features = {}
                TemporalFeatureGenerator.add_temporal_features_to_dict(
                    temporal_features, dt
                )

                # Add basic temporal features (subset for training data)
                timestamp_features["hour_of_day"] = temporal_features["hour_of_day"]
                timestamp_features["day_of_week"] = temporal_features["day_of_week"]
                timestamp_features["is_weekend"] = temporal_features["is_weekend"]

                # Store features with timestamp as key
                timestamp_key = dt.isoformat()
                features[timestamp_key] = timestamp_features

                # Get label (use majority vote if multiple entities)
                occupied = group["occupied"].any()
                labels.append(occupied)

                # Convert timestamp to ISO format string for consistency
                timestamps.append(timestamp_key)
                entity_ids.append(",".join(group["entity_id"].unique()))

            _LOGGER.debug(
                "Prepared ML data: %d features, %d labels, %d timestamps",
                len(features),
                len(labels),
                len(timestamps),
            )
            _LOGGER.debug("Sample feature keys: %s", list(features.keys())[:3])
            _LOGGER.debug("Sample timestamp list: %s", timestamps[:3])

            return MLTrainingData(
                features=features,
                labels=labels,
                timestamps=timestamps,
                entity_ids=entity_ids,
            )

        except Exception as err:
            _LOGGER.exception("Error preparing ML data: %s", err)
            return None
