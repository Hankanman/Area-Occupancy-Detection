"""Tests for ML models module."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.area_occupancy.ml_models import (
    MLModelManager,
    ModelTrainingScheduler,
)
from custom_components.area_occupancy.types import MLPrediction, ModelPerformanceMetrics


@pytest.fixture
def temp_storage_path():
    """Create temporary storage path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_sklearn():
    """Mock scikit-learn imports."""
    with patch.dict(
        "sys.modules",
        {
            "sklearn": Mock(),
            "sklearn.ensemble": Mock(),
            "sklearn.svm": Mock(),
            "sklearn.model_selection": Mock(),
            "sklearn.metrics": Mock(),
            "sklearn.preprocessing": Mock(),
        },
    ):
        yield


class TestMLModelManager:
    """Test MLModelManager class."""

    async def test_init(self, hass: HomeAssistant, temp_storage_path):
        """Test ML model manager initialization."""
        manager = MLModelManager(hass, temp_storage_path)

        assert manager.hass == hass
        assert manager.storage_path == temp_storage_path
        assert manager.models_path.exists()
        assert manager._current_model is None
        assert manager._scaler is None

    async def test_predict_no_model(self, hass: HomeAssistant, temp_storage_path):
        """Test prediction when no model is available."""
        manager = MLModelManager(hass, temp_storage_path)

        features = {"feature1": 0.5, "feature2": 0.7}
        result = await manager.predict(features)

        assert result is None

    @patch("custom_components.area_occupancy.ml_models.SKLEARN_AVAILABLE", True)
    async def test_predict_with_model(self, hass: HomeAssistant, temp_storage_path):
        """Test prediction with available model."""
        manager = MLModelManager(hass, temp_storage_path)

        # Mock model and scaler
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]  # Binary classification
        manager._current_model = mock_model

        mock_scaler = Mock()
        mock_scaler.transform.return_value = [[0.5, 0.7]]
        manager._scaler = mock_scaler

        features = {"feature1": 0.5, "feature2": 0.7}
        result = await manager.predict(features)

        assert isinstance(result, MLPrediction)
        assert result.probability == 0.7
        assert 0.0 <= result.confidence <= 1.0
        assert result.feature_count == 2

    async def test_predict_error_handling(self, hass: HomeAssistant, temp_storage_path):
        """Test prediction error handling."""
        manager = MLModelManager(hass, temp_storage_path)

        # Mock model that raises exception
        mock_model = Mock()
        mock_model.predict_proba.side_effect = Exception("Test error")
        manager._current_model = mock_model

        with patch(
            "custom_components.area_occupancy.ml_models.SKLEARN_AVAILABLE", True
        ):
            features = {"feature1": 0.5}
            result = await manager.predict(features)

            assert result is None

    @patch("custom_components.area_occupancy.ml_models.SKLEARN_AVAILABLE", False)
    async def test_train_model_no_sklearn(self, hass: HomeAssistant, temp_storage_path):
        """Test training when sklearn is not available."""
        manager = MLModelManager(hass, temp_storage_path)

        training_data = [{"features": {"f1": 0.5}, "occupancy": 1}] * 150
        result = await manager.train_model(training_data)

        assert result is False

    @patch("custom_components.area_occupancy.ml_models.SKLEARN_AVAILABLE", True)
    async def test_train_model_insufficient_data(
        self, hass: HomeAssistant, temp_storage_path
    ):
        """Test training with insufficient data."""
        manager = MLModelManager(hass, temp_storage_path)

        # Less than minimum required data points
        training_data = [{"features": {"f1": 0.5}, "occupancy": 1}] * 50
        result = await manager.train_model(training_data)

        assert result is False

    @patch("custom_components.area_occupancy.ml_models.SKLEARN_AVAILABLE", True)
    async def test_prepare_feature_vector(self, hass: HomeAssistant, temp_storage_path):
        """Test feature vector preparation."""
        manager = MLModelManager(hass, temp_storage_path)

        features = {"feature_b": 0.7, "feature_a": 0.5, "feature_c": 0.3}
        vector = await manager._prepare_feature_vector(features)

        # Should be sorted by key
        assert vector == [0.5, 0.7, 0.3]  # a, b, c

    async def test_prepare_feature_vector_error(
        self, hass: HomeAssistant, temp_storage_path
    ):
        """Test feature vector preparation error handling."""
        manager = MLModelManager(hass, temp_storage_path)

        # Invalid features that can't be converted
        features = {"feature1": 999.0}  # Use a float to match Dict[str, float]
        vector = await manager._prepare_feature_vector(features)

        assert vector == [999.0]

    async def test_save_and_load_model(self, hass: HomeAssistant, temp_storage_path):
        """Test model saving and loading."""
        manager = MLModelManager(hass, temp_storage_path)

        # Create mock model components
        mock_model = Mock()
        mock_scaler = Mock()
        mock_metrics = ModelPerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1_score=0.85,
            model_type="test",
            training_samples=100,
            test_samples=25,
        )

        manager._current_model = mock_model
        manager._scaler = mock_scaler
        manager._performance_metrics = mock_metrics

        # Save model
        with patch("pickle.dump"), patch("builtins.open", create=True):
            await manager._save_model()

        # Test load model with files present
        with (
            patch("pickle.load") as mock_load,
            patch("builtins.open", create=True),
            patch.object(Path, "exists", return_value=True),
        ):
            mock_load.side_effect = [mock_model, mock_scaler]

            # Mock json.load for metrics
            with patch("json.load", return_value=mock_metrics.__dict__):
                result = await manager.load_model()

                assert result is True

    async def test_load_model_no_files(self, hass: HomeAssistant, temp_storage_path):
        """Test loading model when files don't exist."""
        manager = MLModelManager(hass, temp_storage_path)

        result = await manager.load_model()
        assert result is False

    async def test_get_feature_importance(self, hass: HomeAssistant, temp_storage_path):
        """Test getting feature importance."""
        manager = MLModelManager(hass, temp_storage_path)

        # Test with model that has feature importance
        mock_model = Mock()
        mock_model.feature_importances_ = [0.3, 0.5, 0.2]
        manager._current_model = mock_model

        importance = await manager.get_feature_importance()
        assert len(importance) == 3
        assert "feature_0" in importance

        # Test with model without feature importance
        mock_model_no_importance = Mock(spec=[])  # No feature_importances_ attribute
        manager._current_model = mock_model_no_importance

        importance_empty = await manager.get_feature_importance()
        assert importance_empty == {}

    async def test_is_available_property(self, hass: HomeAssistant, temp_storage_path):
        """Test is_available property."""
        manager = MLModelManager(hass, temp_storage_path)

        # No model available
        with patch(
            "custom_components.area_occupancy.ml_models.SKLEARN_AVAILABLE", True
        ):
            assert manager.is_available is False

        # sklearn not available
        with patch(
            "custom_components.area_occupancy.ml_models.SKLEARN_AVAILABLE", False
        ):
            manager._current_model = Mock()
            assert manager.is_available is False

        # Model available
        with patch(
            "custom_components.area_occupancy.ml_models.SKLEARN_AVAILABLE", True
        ):
            manager._current_model = Mock()
            assert manager.is_available is True


class TestModelTrainingScheduler:
    """Test ModelTrainingScheduler class."""

    async def test_init(self, hass: HomeAssistant, temp_storage_path):
        """Test scheduler initialization."""
        mock_manager = Mock(spec=MLModelManager)
        scheduler = ModelTrainingScheduler(mock_manager, hass)

        assert scheduler.model_manager == mock_manager
        assert scheduler.hass == hass
        assert scheduler._training_task is None

    async def test_start_periodic_training(
        self, hass: HomeAssistant, temp_storage_path
    ):
        """Test starting periodic training."""
        mock_manager = Mock(spec=MLModelManager)
        scheduler = ModelTrainingScheduler(mock_manager, hass)

        with patch.object(scheduler, "_periodic_training_loop") as mock_loop:
            mock_loop.return_value = AsyncMock()

            await scheduler.start_periodic_training()

            assert scheduler._training_task is not None

    async def test_stop_periodic_training(self, hass: HomeAssistant, temp_storage_path):
        """Test stopping periodic training."""
        mock_manager = Mock(spec=MLModelManager)
        scheduler = ModelTrainingScheduler(mock_manager, hass)

        # Create a real async function that can be cancelled
        async def mock_training_task():
            """Mock training task that can be cancelled."""
            await asyncio.sleep(3600)  # Sleep for a long time

        # Start the actual task and then stop it
        task = asyncio.create_task(mock_training_task())
        scheduler._training_task = task

        # Stop the training - this should cancel the task
        await scheduler.stop_periodic_training()

        # Verify the task was cancelled
        assert task.cancelled()

    async def test_collect_training_data(self, hass: HomeAssistant, temp_storage_path):
        """Test training data collection."""
        mock_manager = Mock(spec=MLModelManager)
        scheduler = ModelTrainingScheduler(mock_manager, hass)

        # Currently returns empty list as placeholder
        data = await scheduler._collect_training_data()
        assert data == []
