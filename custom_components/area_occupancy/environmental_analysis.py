"""Environmental sensor analysis for improved occupancy detection."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    ENVIRONMENTAL_ANALYSIS_FREQUENCY,
    ENVIRONMENTAL_ML_CONFIDENCE_THRESHOLD,
)
from .exceptions import EnvironmentalAnalysisError
from .ml_models import MLModelManager
from .types import EnvironmentalConfig, EnvironmentalResult, SensorReading

_LOGGER = logging.getLogger(__name__)


class EnvironmentalAnalyzer:
    """Main environmental analysis coordinator."""

    def __init__(
        self,
        hass: HomeAssistant,
        config: EnvironmentalConfig,
        model_manager: Optional[MLModelManager] = None,
    ) -> None:
        """Initialize the environmental analyzer."""
        self.hass = hass
        self.config = config
        self.model_manager = model_manager
        self._sensor_processor = SensorDataProcessor(hass)
        self._feature_extractor = FeatureExtractor()
        self._threshold_analyzer = ThresholdAnalyzer(config)
        self._last_analysis: Optional[datetime] = None
        self._analysis_cache: Dict[str, EnvironmentalResult] = {}

    async def analyze_occupancy_probability(
        self,
        sensor_readings: Dict[str, SensorReading],
    ) -> EnvironmentalResult:
        """Analyze environmental data to determine occupancy probability."""
        try:
            # Check if analysis is needed
            if not self._should_analyze():
                return self._get_cached_result()

            # Process sensor data
            processed_data = await self._sensor_processor.process_readings(
                sensor_readings
            )

            # Choose analysis method based on configuration and data availability
            if (
                self.config.analysis_method == "ml"
                or self.config.analysis_method == "hybrid"
            ):
                ml_result = await self._perform_ml_analysis(processed_data)
                if (
                    ml_result
                    and ml_result.confidence >= ENVIRONMENTAL_ML_CONFIDENCE_THRESHOLD
                ):
                    result = ml_result
                elif self.config.analysis_method == "hybrid":
                    result = await self._perform_deterministic_analysis(processed_data)
                else:
                    result = EnvironmentalResult(
                        probability=0.5,
                        confidence=0.1,
                        method="fallback",
                        sensor_contributions={},
                    )
            else:
                result = await self._perform_deterministic_analysis(processed_data)

            # Cache result
            self._analysis_cache["latest"] = result
            self._last_analysis = dt_util.utcnow()

            return result

        except Exception as err:
            _LOGGER.error("Environmental analysis failed: %s", err)
            raise EnvironmentalAnalysisError(f"Analysis failed: {err}") from err

    async def _perform_ml_analysis(
        self,
        processed_data: Dict[str, Any],
    ) -> Optional[EnvironmentalResult]:
        """Perform machine learning-based analysis."""
        try:
            # Check if ML model manager is available
            if self.model_manager is None:
                _LOGGER.warning("ML model manager not available")
                return None

            # Extract features for ML model
            features = await self._feature_extractor.extract_features(processed_data)

            if not features:
                _LOGGER.warning("No features available for ML analysis")
                return None

            # Get prediction from ML model
            prediction = await self.model_manager.predict(features)

            if prediction is None:
                _LOGGER.warning("ML model prediction failed")
                return None

            # Calculate sensor contributions
            sensor_contributions = await self._calculate_sensor_contributions(
                processed_data, features, prediction
            )

            return EnvironmentalResult(
                probability=prediction.probability,
                confidence=prediction.confidence,
                method="ml",
                sensor_contributions=sensor_contributions,
                model_version=prediction.model_version,
            )

        except Exception as err:
            _LOGGER.error("ML analysis failed: %s", err)
            return None

    async def _perform_deterministic_analysis(
        self,
        processed_data: Dict[str, Any],
    ) -> EnvironmentalResult:
        """Perform deterministic threshold-based analysis."""
        return await self._threshold_analyzer.analyze(processed_data)

    async def _calculate_sensor_contributions(
        self,
        processed_data: Dict[str, Any],
        features: Dict[str, float],
        prediction: Any,
    ) -> Dict[str, float]:
        """Calculate individual sensor contributions to the prediction."""
        contributions = {}

        for sensor_id, data in processed_data.items():
            if sensor_id in self.config.sensors:
                # Use feature importance or simple heuristics
                sensor_type = self.config.sensors[sensor_id].sensor_type
                contribution = await self._estimate_sensor_contribution(
                    sensor_type, data, features, prediction
                )
                contributions[sensor_id] = contribution

        return contributions

    async def _estimate_sensor_contribution(
        self,
        sensor_type: str,
        sensor_data: Dict[str, Any],
        features: Dict[str, float],
        prediction: Any,
    ) -> float:
        """Estimate individual sensor contribution."""
        # Simple heuristic based on sensor type and value deviation
        if sensor_type == "co2":
            baseline = 400
            current = sensor_data.get("current_value", baseline)
            deviation = abs(current - baseline) / baseline
            return min(deviation * 0.5, 1.0)

        elif sensor_type == "temperature":
            # Get baseline from sensor configuration
            sensor_configs = [
                config
                for config in self.config.sensors.values()
                if config.sensor_type == sensor_type
            ]
            baseline = sensor_configs[0].baseline_value if sensor_configs else 20.0
            current = sensor_data.get("current_value", baseline)
            deviation = abs(current - baseline) / 10.0  # 10 degree scale
            return min(deviation * 0.3, 1.0)

        # Default contribution calculation
        return 0.1

    def _should_analyze(self) -> bool:
        """Check if new analysis is needed."""
        if self._last_analysis is None:
            return True

        time_since_last = dt_util.utcnow() - self._last_analysis
        return time_since_last.total_seconds() >= ENVIRONMENTAL_ANALYSIS_FREQUENCY

    def _get_cached_result(self) -> EnvironmentalResult:
        """Get cached analysis result."""
        return self._analysis_cache.get(
            "latest",
            EnvironmentalResult(
                probability=0.5,
                confidence=0.1,
                method="cached",
                sensor_contributions={},
            ),
        )


class SensorDataProcessor:
    """Processes environmental sensor data."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the sensor data processor."""
        self.hass = hass

    async def process_readings(
        self,
        sensor_readings: Dict[str, SensorReading],
    ) -> Dict[str, Any]:
        """Process raw sensor readings into analysis-ready data."""
        processed_data = {}

        for sensor_id, reading in sensor_readings.items():
            try:
                processed = await self._process_single_reading(sensor_id, reading)
                if processed:
                    processed_data[sensor_id] = processed
            except Exception as err:
                _LOGGER.warning(
                    "Failed to process reading for sensor %s: %s", sensor_id, err
                )

        return processed_data

    async def _process_single_reading(
        self,
        sensor_id: str,
        reading: SensorReading,
    ) -> Optional[Dict[str, Any]]:
        """Process a single sensor reading."""
        if reading.value is None:
            return None

        # Get historical data for trend analysis
        historical_data = await self._get_historical_data(sensor_id)

        # Calculate features
        processed = {
            "current_value": reading.value,
            "timestamp": reading.timestamp,
            "entity_id": sensor_id,
            "historical_mean": np.mean(historical_data)
            if historical_data
            else reading.value,
            "historical_std": np.std(historical_data) if historical_data else 0,
            "rate_of_change": self._calculate_rate_of_change(
                historical_data, reading.value
            ),
            "z_score": self._calculate_z_score(historical_data, reading.value),
        }

        return processed

    async def _get_historical_data(self, sensor_id: str) -> List[float]:
        """Get historical data for a sensor."""
        # This would integrate with Home Assistant's recorder component
        # For now, return empty list as placeholder
        return []

    def _calculate_rate_of_change(
        self, historical_data: List[float], current_value: float
    ) -> float:
        """Calculate rate of change from historical data."""
        if not historical_data or len(historical_data) < 2:
            return 0.0

        recent_value = historical_data[-1]
        return (current_value - recent_value) / max(abs(recent_value), 1.0)

    def _calculate_z_score(
        self, historical_data: List[float], current_value: float
    ) -> float:
        """Calculate z-score for current value."""
        if not historical_data:
            return 0.0

        mean = np.mean(historical_data)
        std = np.std(historical_data)

        if std == 0:
            return 0.0

        return float((current_value - mean) / std)


class FeatureExtractor:
    """Extracts features for machine learning analysis."""

    async def extract_features(
        self, processed_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract features from processed sensor data."""
        features = {}

        # Time-based features
        now = dt_util.utcnow()
        features["hour_of_day"] = now.hour / 24.0
        features["day_of_week"] = now.weekday() / 7.0
        features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0

        # Sensor-based features
        for sensor_id, data in processed_data.items():
            prefix = f"sensor_{hash(sensor_id) % 1000}"  # Short unique prefix

            features[f"{prefix}_value"] = self._normalize_value(data["current_value"])
            features[f"{prefix}_z_score"] = min(max(data["z_score"], -3), 3) / 3.0
            features[f"{prefix}_rate_change"] = min(max(data["rate_of_change"], -1), 1)

        # Cross-sensor features
        if len(processed_data) > 1:
            values = [data["current_value"] for data in processed_data.values()]
            features["sensor_variance"] = float(np.var(values)) if values else 0.0
            features["sensor_correlation"] = self._calculate_cross_correlation(
                processed_data
            )

        return features

    def _normalize_value(self, value: float) -> float:
        """Normalize sensor value to 0-1 range."""
        # Simple normalization, could be improved with sensor-specific ranges
        return min(max(value / 100.0, 0.0), 1.0)

    def _calculate_cross_correlation(self, processed_data: Dict[str, Any]) -> float:
        """Calculate cross-correlation between sensors."""
        # Simplified cross-correlation calculation
        z_scores = [data["z_score"] for data in processed_data.values()]
        if len(z_scores) < 2:
            return 0.0

        try:
            # Handle edge case where correlation matrix might be scalar
            corr_matrix = np.corrcoef(z_scores)
            if corr_matrix.ndim == 0:  # Scalar case (all values identical)
                return 1.0
            elif len(z_scores) == 2:
                return float(corr_matrix[0, 1])
            else:
                # For multiple sensors, return mean correlation
                n = corr_matrix.shape[0]
                # Get upper triangle (excluding diagonal)
                upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
                return float(np.mean(upper_triangle))
        except (ValueError, IndexError):
            # Fallback for any correlation calculation issues
            return 0.0


class ThresholdAnalyzer:
    """Deterministic threshold-based analysis."""

    def __init__(self, config: EnvironmentalConfig) -> None:
        """Initialize the threshold analyzer."""
        self.config = config

    async def analyze(self, processed_data: Dict[str, Any]) -> EnvironmentalResult:
        """Perform threshold-based analysis."""
        sensor_scores = {}
        total_weight = 0.0
        weighted_probability = 0.0

        for sensor_id, data in processed_data.items():
            sensor_config = self.config.sensors.get(sensor_id)
            if not sensor_config:
                continue

            # Calculate sensor-specific probability
            sensor_prob = self._calculate_sensor_probability(sensor_config, data)
            weight = sensor_config.sensitivity

            sensor_scores[sensor_id] = sensor_prob
            weighted_probability += sensor_prob * weight
            total_weight += weight

        # Calculate overall probability
        if total_weight > 0:
            final_probability = weighted_probability / total_weight
        else:
            final_probability = 0.5

        # Calculate confidence based on sensor agreement
        confidence = self._calculate_confidence(sensor_scores)

        return EnvironmentalResult(
            probability=final_probability,
            confidence=confidence,
            method="deterministic",
            sensor_contributions=sensor_scores,
        )

    def _calculate_sensor_probability(
        self,
        sensor_config: Any,
        data: Dict[str, Any],
    ) -> float:
        """Calculate probability for a single sensor."""
        current_value = data["current_value"]
        baseline = sensor_config.baseline_value
        sensor_type = sensor_config.sensor_type

        if sensor_type == "co2":
            # CO2 increases with occupancy
            if current_value > baseline + 100:
                return 0.8
            elif current_value > baseline + 50:
                return 0.7
            elif current_value > baseline + 20:
                return 0.6
            else:
                return 0.3

        elif sensor_type == "temperature":
            # Temperature can increase or decrease based on context
            deviation = abs(current_value - baseline)
            if deviation > 3.0:
                return 0.7
            elif deviation > 1.5:
                return 0.6
            else:
                return 0.4

        elif sensor_type == "humidity":
            # Humidity increases with occupancy
            if current_value > baseline + 10:
                return 0.7
            elif current_value > baseline + 5:
                return 0.6
            else:
                return 0.4

        elif sensor_type == "luminance":
            # Light changes indicate activity
            if abs(current_value - baseline) > baseline * 0.2:
                return 0.6
            else:
                return 0.4

        elif sensor_type == "sound":
            # Sound increases with occupancy
            if current_value > baseline + 10:
                return 0.8
            elif current_value > baseline + 5:
                return 0.6
            else:
                return 0.3

        elif sensor_type == "pressure":
            # Pressure changes can indicate movement
            if abs(current_value - baseline) > 1.0:
                return 0.6
            else:
                return 0.4

        return 0.5  # Default neutral probability

    def _calculate_confidence(self, sensor_scores: Dict[str, float]) -> float:
        """Calculate confidence based on sensor agreement."""
        if not sensor_scores:
            return 0.1

        scores = list(sensor_scores.values())
        mean_score = float(np.mean(scores))
        variance = float(np.var(scores))

        # Higher confidence when sensors agree (low variance)
        # and when they strongly indicate occupancy or non-occupancy
        agreement_confidence = 1.0 / (1.0 + variance * 4)  # Scale variance effect
        strength_confidence = abs(mean_score - 0.5) * 2  # Distance from neutral

        return min(agreement_confidence * strength_confidence, 1.0)
