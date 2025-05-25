"""Machine learning models for environmental sensor analysis."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    ENVIRONMENTAL_MINIMUM_DATA_POINTS,
    ENVIRONMENTAL_ML_MODEL_VERSION,
    ENVIRONMENTAL_MODEL_RETRAIN_INTERVAL,
)
from .exceptions import MLModelError
from .types import MLPrediction, ModelPerformanceMetrics

_LOGGER = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError:
    _LOGGER.warning("scikit-learn not available, ML analysis will be disabled")
    SKLEARN_AVAILABLE = False


class MLModelManager:
    """Manages machine learning models for environmental analysis."""

    def __init__(self, hass: HomeAssistant, storage_path: Path) -> None:
        """Initialize the ML model manager."""
        self.hass = hass
        self.storage_path = storage_path
        self.models_path = storage_path / "ml_models"
        self.models_path.mkdir(exist_ok=True)

        self._current_model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._model_version: str = ENVIRONMENTAL_ML_MODEL_VERSION
        self._last_training: Optional[datetime] = None
        self._performance_metrics: Optional[ModelPerformanceMetrics] = None

        if not SKLEARN_AVAILABLE:
            _LOGGER.error("scikit-learn not available, ML features disabled")

    async def predict(self, features: Dict[str, float]) -> Optional[MLPrediction]:
        """Make a prediction using the current model."""
        if not SKLEARN_AVAILABLE or not self._current_model:
            return None

        try:
            # Prepare feature vector
            feature_vector = await self._prepare_feature_vector(features)
            if feature_vector is None:
                return None

            # Scale features
            if self._scaler:
                feature_vector = self._scaler.transform([feature_vector])
            else:
                feature_vector = [feature_vector]

            # Make prediction
            probability = self._current_model.predict_proba(feature_vector)[0][1]

            # Calculate confidence based on prediction certainty
            confidence = abs(probability - 0.5) * 2  # Convert to 0-1 scale

            return MLPrediction(
                probability=probability,
                confidence=confidence,
                model_version=self._model_version,
                feature_count=len(features),
            )

        except Exception as err:
            _LOGGER.error("Prediction failed: %s", err)
            return None

    async def train_model(
        self,
        training_data: List[Dict[str, Any]],
        force_retrain: bool = False,
    ) -> bool:
        """Train or retrain the ML model."""
        if not SKLEARN_AVAILABLE:
            _LOGGER.error("Cannot train model: scikit-learn not available")
            return False

        try:
            # Check if training is needed
            if not force_retrain and not self._should_retrain():
                return True

            # Validate training data
            if len(training_data) < ENVIRONMENTAL_MINIMUM_DATA_POINTS:
                _LOGGER.warning(
                    "Insufficient training data: %d points (minimum %d)",
                    len(training_data),
                    ENVIRONMENTAL_MINIMUM_DATA_POINTS,
                )
                return False

            # Prepare training data
            X, y = await self._prepare_training_data(training_data)
            if X is None or y is None:
                _LOGGER.error("Failed to prepare training data")
                return False

            # Train multiple models and select the best
            best_model, best_scaler, metrics = await self._train_and_evaluate_models(
                X, y
            )

            if best_model is None:
                _LOGGER.error("Model training failed")
                return False

            # Update current model
            self._current_model = best_model
            self._scaler = best_scaler
            self._performance_metrics = metrics
            self._last_training = dt_util.utcnow()

            # Save model to disk
            await self._save_model()

            if metrics is not None:
                _LOGGER.info(
                    "Model training completed successfully. Accuracy: %.3f",
                    metrics.accuracy,
                )
            else:
                _LOGGER.info(
                    "Model training completed successfully, but metrics are unavailable."
                )
            return True

        except Exception as err:
            _LOGGER.error("Model training failed: %s", err)
            raise MLModelError(f"Training failed: {err}") from err

    async def load_model(self) -> bool:
        """Load saved model from disk."""
        if not SKLEARN_AVAILABLE:
            return False

        try:
            model_file = self.models_path / "current_model.pkl"
            scaler_file = self.models_path / "scaler.pkl"
            metrics_file = self.models_path / "metrics.json"

            if not all(f.exists() for f in [model_file, scaler_file, metrics_file]):
                _LOGGER.info("No saved model found")
                return False

            # Load model components
            with open(model_file, "rb") as f:
                self._current_model = pickle.load(f)

            with open(scaler_file, "rb") as f:
                self._scaler = pickle.load(f)

            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                self._performance_metrics = ModelPerformanceMetrics(**metrics_data)

            _LOGGER.info("Model loaded successfully")
            return True

        except Exception as err:
            _LOGGER.error("Failed to load model: %s", err)
            return False

    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the current model."""
        if not self._current_model or not hasattr(
            self._current_model, "feature_importances_"
        ):
            return {}

        try:
            importances = self._current_model.feature_importances_
            # This would need feature names to be meaningful
            # For now, return generic feature importance
            return {
                f"feature_{i}": importance for i, importance in enumerate(importances)
            }
        except Exception as err:
            _LOGGER.error("Failed to get feature importance: %s", err)
            return {}

    async def _prepare_feature_vector(
        self, features: Dict[str, float]
    ) -> Optional[List[float]]:
        """Prepare feature vector for prediction."""
        try:
            # Sort features by key for consistent ordering
            sorted_features = sorted(features.items())
            feature_vector = []

            for key, value in sorted_features:
                # Validate that the value can be converted to float
                try:
                    float_value = float(value)
                    if not np.isfinite(float_value):  # Check for inf/nan
                        _LOGGER.warning("Invalid feature value for %s: %s", key, value)
                        return None
                    feature_vector.append(float_value)
                except (ValueError, TypeError) as e:
                    _LOGGER.warning(
                        "Cannot convert feature %s value %s to float: %s", key, value, e
                    )
                    return None

            return feature_vector
        except Exception as err:
            _LOGGER.error("Failed to prepare feature vector: %s", err)
            return None

    async def _prepare_training_data(
        self,
        training_data: List[Dict[str, Any]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for model training."""
        try:
            X = []
            y = []

            for sample in training_data:
                features = sample.get("features", {})
                label = sample.get("occupancy", 0)  # 0 or 1

                feature_vector = await self._prepare_feature_vector(features)
                if feature_vector is not None:
                    X.append(feature_vector)
                    y.append(label)

            if not X:
                return None, None

            return np.array(X), np.array(y)

        except Exception as err:
            _LOGGER.error("Failed to prepare training data: %s", err)
            return None, None

    async def _train_and_evaluate_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[ModelPerformanceMetrics]]:
        """Train and evaluate multiple models, return the best one."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define models to try
            models = {
                "random_forest": RandomForestClassifier(
                    n_estimators=50, random_state=42, max_depth=10
                ),
                "gradient_boosting": GradientBoostingClassifier(
                    n_estimators=50, random_state=42, max_depth=6
                ),
                "svm": SVC(probability=True, random_state=42, kernel="rbf"),
            }

            best_model = None
            best_scaler = None
            best_score = 0
            best_metrics = None

            for name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Evaluate model
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)

                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = model
                        best_scaler = scaler

                        # Calculate detailed metrics
                        best_metrics = ModelPerformanceMetrics(
                            accuracy=float(accuracy),
                            precision=float(
                                precision_score(y_test, y_pred, zero_division=0)
                            ),
                            recall=float(recall_score(y_test, y_pred, zero_division=0)),
                            f1_score=float(f1_score(y_test, y_pred, zero_division=0)),
                            model_type=name,
                            training_samples=int(len(X_train)),
                            test_samples=int(len(X_test)),
                        )

                    _LOGGER.debug("Model %s accuracy: %.3f", name, accuracy)

                except Exception as err:
                    _LOGGER.warning("Failed to train model %s: %s", name, err)

            return best_model, best_scaler, best_metrics

        except Exception as err:
            _LOGGER.error("Model training and evaluation failed: %s", err)
            return None, None, None

    async def _save_model(self) -> None:
        """Save current model to disk."""
        try:
            if (
                not self._current_model
                or not self._scaler
                or not self._performance_metrics
            ):
                return

            # Save model
            model_file = self.models_path / "current_model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(self._current_model, f)

            # Save scaler
            scaler_file = self.models_path / "scaler.pkl"
            with open(scaler_file, "wb") as f:
                pickle.dump(self._scaler, f)

            # Save metrics
            metrics_file = self.models_path / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(self._performance_metrics.__dict__, f, indent=2)

            _LOGGER.debug("Model saved successfully")

        except Exception as err:
            _LOGGER.error("Failed to save model: %s", err)

    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if self._last_training is None:
            return True

        time_since_training = dt_util.utcnow() - self._last_training
        return (
            time_since_training.total_seconds() >= ENVIRONMENTAL_MODEL_RETRAIN_INTERVAL
        )

    @property
    def is_available(self) -> bool:
        """Check if ML model is available."""
        return SKLEARN_AVAILABLE and self._current_model is not None

    @property
    def performance_metrics(self) -> Optional[ModelPerformanceMetrics]:
        """Get current model performance metrics."""
        return self._performance_metrics


class ModelTrainingScheduler:
    """Schedules and manages model training tasks."""

    def __init__(self, model_manager: MLModelManager, hass: HomeAssistant) -> None:
        """Initialize the training scheduler."""
        self.model_manager = model_manager
        self.hass = hass
        self._training_task: Optional[asyncio.Task] = None

    async def start_periodic_training(self) -> None:
        """Start periodic model training."""
        if self._training_task and not self._training_task.done():
            return

        self._training_task = asyncio.create_task(self._periodic_training_loop())

    async def stop_periodic_training(self) -> None:
        """Stop periodic model training."""
        if self._training_task:
            self._training_task.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await self._training_task

    async def _periodic_training_loop(self) -> None:
        """Periodic training loop."""
        while True:
            try:
                await asyncio.sleep(ENVIRONMENTAL_MODEL_RETRAIN_INTERVAL)

                # Collect training data (placeholder)
                training_data = await self._collect_training_data()

                if training_data:
                    await self.model_manager.train_model(training_data)

            except asyncio.CancelledError:
                break
            except Exception as err:
                _LOGGER.error("Periodic training failed: %s", err)

    async def _collect_training_data(self) -> List[Dict[str, Any]]:
        """Collect training data from Home Assistant history."""
        # This would integrate with Home Assistant's recorder component
        # to collect historical sensor data and occupancy states
        # For now, return empty list as placeholder
        return []
