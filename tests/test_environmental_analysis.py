"""Tests for environmental sensor analysis."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.area_occupancy.environmental_analysis import (
    EnvironmentalAnalyzer,
    FeatureExtractor,
    SensorDataProcessor,
    ThresholdAnalyzer,
)
from custom_components.area_occupancy.exceptions import EnvironmentalAnalysisError
from custom_components.area_occupancy.ml_models import MLModelManager
from custom_components.area_occupancy.types import (
    EnvironmentalConfig,
    EnvironmentalResult,
    EnvironmentalSensorConfig,
    SensorReading,
)


@pytest.fixture
def environmental_config():
    """Create test environmental configuration."""
    return EnvironmentalConfig(
        sensors={
            "sensor.co2": EnvironmentalSensorConfig(
                entity_id="sensor.co2",
                sensor_type="co2",
                analysis_method="hybrid",
                baseline_value=400.0,
                sensitivity=0.7,
                enabled=True,
            ),
            "sensor.temperature": EnvironmentalSensorConfig(
                entity_id="sensor.temperature",
                sensor_type="temperature",
                analysis_method="ml",
                baseline_value=20.0,
                sensitivity=0.5,
                enabled=True,
            ),
        },
        analysis_frequency=60,
        minimum_data_points=100,
        ml_confidence_threshold=0.6,
        deterministic_fallback=True,
    )


@pytest.fixture
def sensor_readings():
    """Create test sensor readings."""
    now = dt_util.utcnow()
    return {
        "sensor.co2": SensorReading(
            value=450.0,
            timestamp=now,
            entity_id="sensor.co2",
        ),
        "sensor.temperature": SensorReading(
            value=22.5,
            timestamp=now,
            entity_id="sensor.temperature",
        ),
    }


@pytest.fixture
def mock_model_manager():
    """Create mock ML model manager."""
    manager = Mock(spec=MLModelManager)
    manager.predict = AsyncMock(return_value=None)
    manager.is_available = False
    return manager


class TestEnvironmentalAnalyzer:
    """Test EnvironmentalAnalyzer class."""

    async def test_init(
        self, hass: HomeAssistant, environmental_config, mock_model_manager
    ):
        """Test analyzer initialization."""
        analyzer = EnvironmentalAnalyzer(hass, environmental_config, mock_model_manager)

        assert analyzer.hass == hass
        assert analyzer.config == environmental_config
        assert analyzer.model_manager == mock_model_manager
        assert analyzer._last_analysis is None

    async def test_analyze_occupancy_probability_deterministic(
        self,
        hass: HomeAssistant,
        environmental_config,
        mock_model_manager,
        sensor_readings,
    ):
        """Test deterministic analysis."""
        environmental_config.analysis_method = "deterministic"
        analyzer = EnvironmentalAnalyzer(hass, environmental_config, mock_model_manager)

        with patch.object(
            analyzer._sensor_processor, "process_readings"
        ) as mock_process:
            mock_process.return_value = {
                "sensor.co2": {
                    "current_value": 450.0,
                    "timestamp": dt_util.utcnow(),
                    "entity_id": "sensor.co2",
                    "historical_mean": 420.0,
                    "historical_std": 20.0,
                    "rate_of_change": 0.1,
                    "z_score": 1.5,
                },
            }

            result = await analyzer.analyze_occupancy_probability(sensor_readings)

            assert isinstance(result, EnvironmentalResult)
            assert result.method == "deterministic"
            assert 0.0 <= result.probability <= 1.0
            assert 0.0 <= result.confidence <= 1.0

    async def test_analyze_occupancy_probability_ml_fallback(
        self,
        hass: HomeAssistant,
        environmental_config,
        mock_model_manager,
        sensor_readings,
    ):
        """Test ML analysis with fallback to deterministic."""
        environmental_config.analysis_method = "hybrid"
        analyzer = EnvironmentalAnalyzer(hass, environmental_config, mock_model_manager)

        # Mock ML prediction to return low confidence
        from custom_components.area_occupancy.types import MLPrediction

        mock_model_manager.predict.return_value = MLPrediction(
            probability=0.7,
            confidence=0.3,  # Below threshold
            model_version="1.0.0",
            feature_count=5,
        )

        with patch.object(
            analyzer._sensor_processor, "process_readings"
        ) as mock_process:
            mock_process.return_value = {
                "sensor.co2": {
                    "current_value": 450.0,
                    "timestamp": dt_util.utcnow(),
                    "entity_id": "sensor.co2",
                    "historical_mean": 420.0,
                    "historical_std": 20.0,
                    "rate_of_change": 0.1,
                    "z_score": 1.5,
                },
            }

            result = await analyzer.analyze_occupancy_probability(sensor_readings)

            assert isinstance(result, EnvironmentalResult)
            assert result.method == "deterministic"  # Fell back to deterministic

    async def test_analyze_occupancy_probability_error_handling(
        self,
        hass: HomeAssistant,
        environmental_config,
        mock_model_manager,
        sensor_readings,
    ):
        """Test error handling in analysis."""
        analyzer = EnvironmentalAnalyzer(hass, environmental_config, mock_model_manager)

        with patch.object(
            analyzer._sensor_processor, "process_readings"
        ) as mock_process:
            mock_process.side_effect = Exception("Test error")

            with pytest.raises(EnvironmentalAnalysisError):
                await analyzer.analyze_occupancy_probability(sensor_readings)

    async def test_caching_behavior(
        self,
        hass: HomeAssistant,
        environmental_config,
        mock_model_manager,
        sensor_readings,
    ):
        """Test analysis result caching."""
        analyzer = EnvironmentalAnalyzer(hass, environmental_config, mock_model_manager)

        # First analysis
        with patch.object(
            analyzer._sensor_processor, "process_readings"
        ) as mock_process:
            mock_process.return_value = {
                "sensor.co2": {
                    "current_value": 450.0,
                    "timestamp": dt_util.utcnow(),
                    "entity_id": "sensor.co2",
                    "historical_mean": 420.0,
                    "historical_std": 20.0,
                    "rate_of_change": 0.1,
                    "z_score": 1.5,
                },
            }

            result1 = await analyzer.analyze_occupancy_probability(sensor_readings)

            # Second analysis should use cache (within frequency limit)
            result2 = await analyzer.analyze_occupancy_probability(sensor_readings)

            assert result1.probability == result2.probability
            assert mock_process.call_count == 1  # Only called once due to caching


class TestSensorDataProcessor:
    """Test SensorDataProcessor class."""

    async def test_process_readings(self, hass: HomeAssistant, sensor_readings):
        """Test processing sensor readings."""
        processor = SensorDataProcessor(hass)

        with patch.object(processor, "_get_historical_data") as mock_history:
            mock_history.return_value = [400.0, 410.0, 420.0, 430.0]

            result = await processor.process_readings(sensor_readings)

            assert "sensor.co2" in result
            assert "sensor.temperature" in result

            co2_data = result["sensor.co2"]
            assert co2_data["current_value"] == 450.0
            assert "historical_mean" in co2_data
            assert "z_score" in co2_data

    async def test_process_single_reading_none_value(self, hass: HomeAssistant):
        """Test processing reading with None value."""
        processor = SensorDataProcessor(hass)

        reading = SensorReading(
            value=None,
            timestamp=dt_util.utcnow(),
            entity_id="sensor.test",
        )

        result = await processor._process_single_reading("sensor.test", reading)
        assert result is None

    async def test_calculate_rate_of_change(self, hass: HomeAssistant):
        """Test rate of change calculation."""
        processor = SensorDataProcessor(hass)

        # Test with sufficient historical data
        historical_data = [400.0, 410.0, 420.0, 430.0]
        current_value = 450.0

        rate = processor._calculate_rate_of_change(historical_data, current_value)
        assert rate > 0  # Increasing trend

        # Test with empty historical data
        rate_empty = processor._calculate_rate_of_change([], current_value)
        assert rate_empty == 0.0

    async def test_calculate_z_score(self, hass: HomeAssistant):
        """Test z-score calculation."""
        processor = SensorDataProcessor(hass)

        # Test with normal distribution
        historical_data = [400.0, 410.0, 420.0, 430.0, 440.0]
        current_value = 450.0

        z_score = processor._calculate_z_score(historical_data, current_value)
        assert z_score > 0  # Above mean

        # Test with empty historical data
        z_score_empty = processor._calculate_z_score([], current_value)
        assert z_score_empty == 0.0


class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    async def test_extract_features(self):
        """Test feature extraction."""
        extractor = FeatureExtractor()

        processed_data = {
            "sensor.co2": {
                "current_value": 450.0,
                "z_score": 1.5,
                "rate_of_change": 0.1,
            },
            "sensor.temperature": {
                "current_value": 22.5,
                "z_score": 0.5,
                "rate_of_change": 0.05,
            },
        }

        features = await extractor.extract_features(processed_data)

        # Check time-based features
        assert "hour_of_day" in features
        assert "day_of_week" in features
        assert "is_weekend" in features

        # Check sensor features
        sensor_features = [k for k in features if k.startswith("sensor_")]
        assert len(sensor_features) > 0

        # Check cross-sensor features
        assert "sensor_variance" in features
        assert "sensor_correlation" in features

    async def test_normalize_value(self):
        """Test value normalization."""
        extractor = FeatureExtractor()

        # Test normal values
        assert extractor._normalize_value(50.0) == 0.5
        assert extractor._normalize_value(100.0) == 1.0
        assert extractor._normalize_value(0.0) == 0.0

        # Test boundary conditions
        assert extractor._normalize_value(-10.0) == 0.0  # Clipped to 0
        assert extractor._normalize_value(200.0) == 1.0  # Clipped to 1


class TestThresholdAnalyzer:
    """Test ThresholdAnalyzer class."""

    async def test_analyze(self, environmental_config):
        """Test threshold-based analysis."""
        analyzer = ThresholdAnalyzer(environmental_config)

        processed_data = {
            "sensor.co2": {
                "current_value": 500.0,  # Above baseline + 100
                "timestamp": dt_util.utcnow(),
                "entity_id": "sensor.co2",
            },
            "sensor.temperature": {
                "current_value": 22.0,  # Close to baseline
                "timestamp": dt_util.utcnow(),
                "entity_id": "sensor.temperature",
            },
        }

        result = await analyzer.analyze(processed_data)

        assert isinstance(result, EnvironmentalResult)
        assert result.method == "deterministic"
        assert 0.0 <= result.probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert "sensor.co2" in result.sensor_contributions
        assert "sensor.temperature" in result.sensor_contributions

    async def test_calculate_sensor_probability_co2(self, environmental_config):
        """Test CO2 sensor probability calculation."""
        analyzer = ThresholdAnalyzer(environmental_config)
        sensor_config = environmental_config.sensors["sensor.co2"]

        # Test high CO2 (above baseline + 100)
        data_high = {"current_value": 550.0}
        prob_high = analyzer._calculate_sensor_probability(sensor_config, data_high)
        assert prob_high >= 0.7

        # Test normal CO2 (near baseline)
        data_normal = {"current_value": 420.0}
        prob_normal = analyzer._calculate_sensor_probability(sensor_config, data_normal)
        assert prob_normal <= 0.5

    async def test_calculate_confidence(self, environmental_config):
        """Test confidence calculation."""
        analyzer = ThresholdAnalyzer(environmental_config)

        # Test high agreement (low variance)
        sensor_scores_agree = {"sensor1": 0.8, "sensor2": 0.8, "sensor3": 0.7}
        confidence_high = analyzer._calculate_confidence(sensor_scores_agree)

        # Test low agreement (high variance)
        sensor_scores_disagree = {"sensor1": 0.2, "sensor2": 0.8, "sensor3": 0.5}
        confidence_low = analyzer._calculate_confidence(sensor_scores_disagree)

        assert confidence_high > confidence_low

        # Test empty scores
        confidence_empty = analyzer._calculate_confidence({})
        assert confidence_empty == 0.1
