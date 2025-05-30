"""ML Model Manager for Area Occupancy Detection.

This module handles machine learning model training, validation, persistence,
and inference for occupancy detection.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    ML_MIN_TRAINING_SAMPLES,
    ML_MODEL_VERSION,
)
from .exceptions import CalculationError
from .ml_feature_engineering import validate_feature_schema
from .storage import AreaOccupancyStorage
from .types import MLModelMeta, MLPrediction, MLTrainingData

_LOGGER = logging.getLogger(__name__)

# Optional ML dependencies
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    _LOGGER.warning("LightGBM not available, ML features will be disabled")

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    _LOGGER.warning("scikit-learn not available, ML features will be disabled")


class ModelManager:
    """Manages ML models for occupancy detection."""

    def __init__(self, hass: HomeAssistant, storage: AreaOccupancyStorage) -> None:
        """Initialize the model manager.

        Args:
            hass: Home Assistant instance
            storage: Storage manager for persistence

        """
        self.hass = hass
        self.storage = storage
        self._model: lgb.LGBMClassifier | None = None
        self._model_meta: MLModelMeta | None = None
        self._feature_schema: dict[str, str] | None = None
        self._training_in_progress = False

    @property
    def has_model(self) -> bool:
        """Check if a trained model is available."""
        return self._model is not None and self._model_meta is not None

    @property
    def model_available(self) -> bool:
        """Check if ML dependencies are available."""
        return HAS_LIGHTGBM and HAS_SKLEARN

    async def async_load(self) -> MLModelMeta | None:
        """Load a previously trained model from storage.

        Returns:
            Model metadata if loaded successfully, None otherwise

        """
        if not self.model_available:
            _LOGGER.warning("ML dependencies not available, cannot load model")
            return None

        try:
            # Load model metadata from storage
            stored_data = await self.storage.async_load_ml_model()

            if not stored_data:
                _LOGGER.debug("No stored ML model found")
                return None

            model_path = stored_data.get("model_path")
            model_meta_dict = stored_data.get("metadata")

            if not model_path or not model_meta_dict:
                _LOGGER.warning("Invalid stored ML model data")
                return None

            # Load the actual model file
            if not Path(model_path).exists():
                _LOGGER.warning("Model file not found: %s", model_path)
                return None

            self._model = await self.hass.async_add_executor_job(
                joblib.load, model_path
            )

            # Reconstruct model metadata
            self._model_meta = MLModelMeta(
                model_version=model_meta_dict["model_version"],
                feature_schema=model_meta_dict["feature_schema"],
                training_samples=model_meta_dict["training_samples"],
                performance_metrics=model_meta_dict["performance_metrics"],
                created_at=model_meta_dict["created_at"],
                last_trained=model_meta_dict["last_trained"],
                confidence_threshold=model_meta_dict["confidence_threshold"],
            )

            self._feature_schema = self._model_meta.feature_schema

            _LOGGER.info(
                "Loaded ML model (version %d) with %d training samples",
                self._model_meta.model_version,
                self._model_meta.training_samples,
            )

            return self._model_meta

        except Exception as err:
            _LOGGER.exception("Error loading ML model: %s", err)
            self._model = None
            self._model_meta = None
            self._feature_schema = None
            return None

    async def async_train(self, training_data: MLTrainingData) -> MLModelMeta | None:
        """Train a new ML model with the provided data.

        Args:
            training_data: Training dataset

        Returns:
            Model metadata if training successful, None otherwise

        Raises:
            CalculationError: If training fails

        """
        if not self.model_available:
            raise CalculationError("ML dependencies not available for training")

        if self._training_in_progress:
            _LOGGER.warning("Training already in progress")
            return None

        if len(training_data.labels) < ML_MIN_TRAINING_SAMPLES:
            raise CalculationError(
                f"Insufficient training samples: {len(training_data.labels)} < {ML_MIN_TRAINING_SAMPLES}"
            )

        self._training_in_progress = True

        try:
            _LOGGER.info(
                "Starting ML model training with %d samples", len(training_data.labels)
            )

            # Prepare training data
            X, y, feature_schema = await self._prepare_training_data(training_data)

            if X.shape[0] == 0:
                raise CalculationError("No valid training data after preprocessing")

            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            model, performance_metrics = await self._train_model(
                X_train, y_train, X_val, y_val
            )

            # Create model metadata
            model_meta = MLModelMeta(
                model_version=ML_MODEL_VERSION,
                feature_schema=feature_schema,
                training_samples=len(training_data.labels),
                performance_metrics=performance_metrics,
                created_at=dt_util.utcnow().isoformat(),
                last_trained=dt_util.utcnow().isoformat(),
                confidence_threshold=0.7,  # Default threshold
            )

            # Save model
            model_path = await self._save_model(model, model_meta)

            # Update instance variables
            self._model = model
            self._model_meta = model_meta
            self._feature_schema = feature_schema

            _LOGGER.info(
                "Successfully trained ML model at %s - Accuracy: %.3f, F1: %.3f",
                model_path,
                performance_metrics.get("accuracy", 0.0),
                performance_metrics.get("f1", 0.0),
            )

            return model_meta

        except Exception as err:
            _LOGGER.exception("Error training ML model: %s", err)
            raise CalculationError(f"Model training failed: {err}") from err

        finally:
            self._training_in_progress = False

    async def async_predict(self, features: dict[str, Any]) -> MLPrediction | None:
        """Make a prediction using the trained model.

        Args:
            features: Feature dictionary for prediction

        Returns:
            Prediction result or None if prediction fails

        """
        if not self.has_model or not features:
            return None

        try:
            # Validate features against schema
            if self._feature_schema and not validate_feature_schema(
                features, self._feature_schema
            ):
                _LOGGER.warning("Feature validation failed for prediction")
                return None

            # Prepare feature array
            X = self._prepare_prediction_features(features)

            if X is None:
                return None

            # Make prediction - ensure model is available
            if self._model is None:
                return None

            proba = await self.hass.async_add_executor_job(self._model.predict_proba, X)

            # Convert to numpy array if needed
            if not isinstance(proba, np.ndarray):
                proba = np.array(proba)

            # Get probability for positive class (occupied)
            if proba.shape[1] == 2:
                probability = float(proba[0][1])  # Probability of class 1 (occupied)
            else:
                probability = float(proba[0][0])

            # Calculate confidence (distance from decision boundary)
            confidence = float(max(proba[0]) - min(proba[0]))

            # Get feature importance if available
            feature_importance = {}
            if self._model is not None and hasattr(self._model, "feature_importances_"):
                importance_values = self._model.feature_importances_
                feature_names = list(features.keys())
                feature_importance = dict(zip(feature_names, importance_values))

            # Ensure model_meta is available
            if self._model_meta is None:
                return None

            return MLPrediction(
                probability=probability,
                confidence=confidence,
                feature_importance=feature_importance,
                model_version=self._model_meta.model_version,
            )

        except Exception as err:
            _LOGGER.exception("Error making ML prediction: %s", err)
            return None

    async def async_check_drift(self, recent_features: list[dict[str, Any]]) -> bool:
        """Check for model drift using recent feature data.

        Args:
            recent_features: List of recent feature dictionaries

        Returns:
            True if drift detected, False otherwise

        """
        if not self.has_model or not recent_features:
            return False

        try:
            # Simple drift detection based on feature distribution changes
            # This is a basic implementation - could be enhanced with more sophisticated methods

            if len(recent_features) < 10:
                return False  # Need more data for drift detection

            # Calculate feature statistics
            recent_stats = self._calculate_feature_stats(recent_features)

            # Compare with training data statistics (if available)
            # For now, use simple thresholds
            drift_threshold = 0.3  # 30% change threshold

            for feature_name, stats in recent_stats.items():
                if stats.get("std", 0) > drift_threshold:
                    _LOGGER.warning(
                        "Potential drift detected in feature: %s", feature_name
                    )
                    return True

            return False

        except Exception as err:
            _LOGGER.exception("Error checking model drift: %s", err)
            return False

    async def _prepare_training_data(
        self, training_data: MLTrainingData
    ) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
        """Prepare training data for model training.

        Args:
            training_data: Raw training data

        Returns:
            Tuple of (X, y, feature_schema)

        """
        # Convert features to DataFrame
        feature_rows = []
        for timestamp in training_data.timestamps:
            if timestamp in training_data.features:
                feature_rows.append(training_data.features[timestamp])

        if not feature_rows:
            return np.array([]), np.array([]), {}

        df = pd.DataFrame(feature_rows)

        # Remove non-numeric columns and handle missing values
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(0)

        # Create feature schema
        feature_schema = dict.fromkeys(numeric_df.columns, "float")

        # Convert to numpy arrays
        X = numeric_df.values
        y = np.array(training_data.labels)

        return X, y, feature_schema

    async def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[lgb.LGBMClassifier, dict[str, float]]:
        """Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (trained_model, performance_metrics)

        """
        # Configure LightGBM parameters
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

        # Create and train model
        model = lgb.LGBMClassifier(**params)

        # Train in executor to avoid blocking event loop
        await self.hass.async_add_executor_job(
            model.fit,
            X_train,
            y_train,
        )

        # Calculate performance metrics
        y_pred = await self.hass.async_add_executor_job(model.predict, X_val)

        # Convert to numpy array if needed
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        }

        return model, metrics

    async def _save_model(self, model: lgb.LGBMClassifier, meta: MLModelMeta) -> str:
        """Save model to storage.

        Args:
            model: Trained model
            meta: Model metadata

        Returns:
            Path to saved model file

        """
        # Create models directory in Home Assistant config
        models_dir = Path(self.hass.config.config_dir) / "area_occupancy_models"
        models_dir.mkdir(exist_ok=True)

        # Save model file
        model_filename = f"model_v{meta.model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_path = models_dir / model_filename

        await self.hass.async_add_executor_job(joblib.dump, model, model_path)

        # Store metadata and path in storage
        await self.storage.async_save_ml_model(
            {
                "model_path": str(model_path),
                "metadata": {
                    "model_version": meta.model_version,
                    "feature_schema": meta.feature_schema,
                    "training_samples": meta.training_samples,
                    "performance_metrics": meta.performance_metrics,
                    "created_at": meta.created_at,
                    "last_trained": meta.last_trained,
                    "confidence_threshold": meta.confidence_threshold,
                },
            }
        )

        return str(model_path)

    def _prepare_prediction_features(
        self, features: dict[str, Any]
    ) -> np.ndarray | None:
        """Prepare features for prediction.

        Args:
            features: Feature dictionary

        Returns:
            Feature array ready for prediction or None if preparation fails

        """
        try:
            if not self._feature_schema:
                return None

            # Create feature array in correct order
            feature_array = []
            for feature_name in self._feature_schema:
                value = features.get(feature_name, 0.0)

                # Convert to float
                if isinstance(value, bool):
                    value = float(value)
                elif not isinstance(value, (int, float)):
                    value = 0.0

                feature_array.append(value)

            return np.array([feature_array])

        except Exception as err:
            _LOGGER.exception("Error preparing prediction features: %s", err)
            return None

    def _calculate_feature_stats(
        self, features_list: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Calculate statistics for feature drift detection.

        Args:
            features_list: List of feature dictionaries

        Returns:
            Dictionary mapping feature names to their statistics

        """
        if not features_list:
            return {}

        # Convert to DataFrame for easier statistics calculation
        df = pd.DataFrame(features_list)

        # Calculate statistics for numeric columns
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }

        return stats
