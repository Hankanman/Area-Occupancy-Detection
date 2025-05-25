"""Environmental data storage and management."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    ENVIRONMENTAL_MAX_HISTORICAL_DAYS,
    ENVIRONMENTAL_STORAGE_VERSION,
)
from .exceptions import EnvironmentalStorageError
from .types import EnvironmentalConfig, EnvironmentalResult, SensorReading

_LOGGER = logging.getLogger(__name__)


class EnvironmentalStorage:
    """Manages environmental data storage and persistence."""

    def __init__(self, hass: HomeAssistant, storage_path: Path) -> None:
        """Initialize environmental storage."""
        self.hass = hass
        self.storage_path = storage_path
        self.data_path = storage_path / "environmental_data"
        self.data_path.mkdir(exist_ok=True)

        self._config_file = self.storage_path / "environmental_config.json"
        self._readings_file = self.data_path / "sensor_readings.json"
        self._results_file = self.data_path / "analysis_results.json"

        self._config_cache: Optional[EnvironmentalConfig] = None
        self._readings_cache: List[Dict[str, Any]] = []
        self._results_cache: List[Dict[str, Any]] = []

    async def save_config(self, config: EnvironmentalConfig) -> None:
        """Save environmental configuration."""
        try:
            config_data = {
                "version": ENVIRONMENTAL_STORAGE_VERSION,
                "sensors": {
                    sensor_id: {
                        "entity_id": sensor_config.entity_id,
                        "sensor_type": sensor_config.sensor_type,
                        "analysis_method": sensor_config.analysis_method,
                        "baseline_value": sensor_config.baseline_value,
                        "sensitivity": sensor_config.sensitivity,
                        "enabled": sensor_config.enabled,
                    }
                    for sensor_id, sensor_config in config.sensors.items()
                },
                "global_config": {
                    "analysis_frequency": config.analysis_frequency,
                    "minimum_data_points": config.minimum_data_points,
                    "ml_confidence_threshold": config.ml_confidence_threshold,
                    "deterministic_fallback": config.deterministic_fallback,
                    "max_historical_days": config.max_historical_days,
                },
                "created_at": dt_util.utcnow().isoformat(),
            }

            with open(self._config_file, "w") as f:
                json.dump(config_data, f, indent=2)

            self._config_cache = config
            _LOGGER.debug("Environmental config saved")

        except Exception as err:
            _LOGGER.error("Failed to save environmental config: %s", err)
            raise EnvironmentalStorageError(f"Config save failed: {err}") from err

    async def load_config(self) -> Optional[EnvironmentalConfig]:
        """Load environmental configuration."""
        try:
            if not self._config_file.exists():
                return None

            with open(self._config_file, "r") as f:
                config_data = json.load(f)

            # Validate version
            if config_data.get("version") != ENVIRONMENTAL_STORAGE_VERSION:
                _LOGGER.warning("Config version mismatch, migration may be needed")

            # Reconstruct config object (simplified)
            # In a real implementation, this would properly reconstruct the EnvironmentalConfig object
            self._config_cache = config_data  # Placeholder

            _LOGGER.debug("Environmental config loaded")
            return self._config_cache

        except Exception as err:
            _LOGGER.error("Failed to load environmental config: %s", err)
            return None

    async def save_sensor_readings(
        self,
        readings: Dict[str, SensorReading],
        occupancy_state: bool,
    ) -> None:
        """Save sensor readings with occupancy state."""
        try:
            timestamp = dt_util.utcnow().isoformat()

            reading_data = {
                "timestamp": timestamp,
                "occupancy": occupancy_state,
                "readings": {
                    sensor_id: {
                        "value": reading.value,
                        "timestamp": reading.timestamp.isoformat(),
                        "entity_id": reading.entity_id,
                    }
                    for sensor_id, reading in readings.items()
                },
            }

            # Add to cache
            self._readings_cache.append(reading_data)

            # Limit cache size - trim immediately when we exceed 1000
            if len(self._readings_cache) > 1000:
                # Keep only the most recent 500 entries
                self._readings_cache = self._readings_cache[-500:]

            # Save to file periodically
            await self._save_readings_to_file()

        except Exception as err:
            _LOGGER.error("Failed to save sensor readings: %s", err)

    async def save_analysis_result(self, result: EnvironmentalResult) -> None:
        """Save analysis result."""
        try:
            result_data = {
                "timestamp": dt_util.utcnow().isoformat(),
                "probability": result.probability,
                "confidence": result.confidence,
                "method": result.method,
                "sensor_contributions": result.sensor_contributions,
                "model_version": getattr(result, "model_version", None),
            }

            # Add to cache
            self._results_cache.append(result_data)

            # Limit cache size
            if len(self._results_cache) > 1000:
                self._results_cache = self._results_cache[-500:]

            # Save to file periodically
            await self._save_results_to_file()

        except Exception as err:
            _LOGGER.error("Failed to save analysis result: %s", err)

    async def get_historical_readings(
        self,
        sensor_id: str,
        hours_back: int = 24,
    ) -> List[Tuple[datetime, float]]:
        """Get historical readings for a sensor."""
        try:
            cutoff_time = dt_util.utcnow() - timedelta(hours=hours_back)
            historical_data = []
            seen_timestamps = set()

            # Check cache first
            for reading_data in self._readings_cache:
                timestamp = datetime.fromisoformat(reading_data["timestamp"])
                if timestamp < cutoff_time:
                    continue

                readings = reading_data.get("readings", {})
                if sensor_id in readings:
                    value = readings[sensor_id]["value"]
                    if value is not None:
                        timestamp_key = timestamp.isoformat()
                        if timestamp_key not in seen_timestamps:
                            historical_data.append((timestamp, value))
                            seen_timestamps.add(timestamp_key)

            # If cache doesn't have enough data, load from file
            if len(historical_data) < 10:
                file_data = await self._load_readings_from_file()
                for reading_data in file_data:
                    timestamp = datetime.fromisoformat(reading_data["timestamp"])
                    if timestamp < cutoff_time:
                        continue

                    readings = reading_data.get("readings", {})
                    if sensor_id in readings:
                        value = readings[sensor_id]["value"]
                        if value is not None:
                            timestamp_key = timestamp.isoformat()
                            if timestamp_key not in seen_timestamps:
                                historical_data.append((timestamp, value))
                                seen_timestamps.add(timestamp_key)

            # Sort by timestamp
            historical_data.sort(key=lambda x: x[0])
            return historical_data

        except Exception as err:
            _LOGGER.error("Failed to get historical readings: %s", err)
            return []

    async def get_training_data(
        self,
        days_back: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get training data for ML models."""
        try:
            cutoff_time = dt_util.utcnow() - timedelta(days=days_back)
            training_data = []

            # Load from file
            file_data = await self._load_readings_from_file()

            for reading_data in file_data:
                timestamp = datetime.fromisoformat(reading_data["timestamp"])
                if timestamp < cutoff_time:
                    continue

                # Extract features from readings
                features = await self._extract_features_from_reading(reading_data)
                if features:
                    training_data.append(
                        {
                            "timestamp": timestamp,
                            "features": features,
                            "occupancy": reading_data.get("occupancy", 0),
                        }
                    )

            return training_data

        except Exception as err:
            _LOGGER.error("Failed to get training data: %s", err)
            return []

    async def cleanup_old_data(self) -> None:
        """Clean up old data beyond retention period."""
        try:
            cutoff_time = dt_util.utcnow() - timedelta(
                days=ENVIRONMENTAL_MAX_HISTORICAL_DAYS
            )

            # Clean readings cache
            self._readings_cache = [
                reading
                for reading in self._readings_cache
                if datetime.fromisoformat(reading["timestamp"]) >= cutoff_time
            ]

            # Clean results cache
            self._results_cache = [
                result
                for result in self._results_cache
                if datetime.fromisoformat(result["timestamp"]) >= cutoff_time
            ]

            # Save cleaned data
            await self._save_readings_to_file()
            await self._save_results_to_file()

            _LOGGER.debug("Old data cleanup completed")

        except Exception as err:
            _LOGGER.error("Data cleanup failed: %s", err)

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "readings_count": len(self._readings_cache),
            "results_count": len(self._results_cache),
            "config_exists": False,
            "readings_file_size": 0,
            "results_file_size": 0,
            "total_size": 0,
        }

        import contextlib

        with contextlib.suppress(OSError):
            stats["config_exists"] = self._config_file.exists()

        try:
            if self._readings_file.exists():
                stats["readings_file_size"] = self._readings_file.stat().st_size
        except OSError:
            # File stat error - leave as 0
            pass

        try:
            if self._results_file.exists():
                stats["results_file_size"] = self._results_file.stat().st_size
        except OSError:
            # File stat error - leave as 0
            pass

        stats["total_size"] = stats["readings_file_size"] + stats["results_file_size"]

        return stats

    async def _save_readings_to_file(self) -> None:
        """Save readings cache to file."""
        try:
            with open(self._readings_file, "w") as f:
                json.dump(self._readings_cache, f)
        except Exception as err:
            _LOGGER.error("Failed to save readings to file: %s", err)

    async def _save_results_to_file(self) -> None:
        """Save results cache to file."""
        try:
            with open(self._results_file, "w") as f:
                json.dump(self._results_cache, f)
        except Exception as err:
            _LOGGER.error("Failed to save results to file: %s", err)

    async def _load_readings_from_file(self) -> List[Dict[str, Any]]:
        """Load readings from file."""
        try:
            if not self._readings_file.exists():
                return []

            with open(self._readings_file, "r") as f:
                return json.load(f)
        except Exception as err:
            _LOGGER.error("Failed to load readings from file: %s", err)
            return []

    async def _extract_features_from_reading(
        self,
        reading_data: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        """Extract features from a reading for training data."""
        try:
            timestamp = datetime.fromisoformat(reading_data["timestamp"])
            readings = reading_data.get("readings", {})

            features = {
                "hour_of_day": timestamp.hour / 24.0,
                "day_of_week": timestamp.weekday() / 7.0,
                "is_weekend": 1.0 if timestamp.weekday() >= 5 else 0.0,
            }

            # Add sensor features
            for sensor_id, reading in readings.items():
                value = reading.get("value")
                if value is not None:
                    # Simple feature extraction
                    prefix = f"sensor_{hash(sensor_id) % 1000}"
                    features[f"{prefix}_value"] = float(value)

            return (
                features if len(features) > 3 else None
            )  # Need more than just time features

        except Exception as err:
            _LOGGER.error("Failed to extract features: %s", err)
            return None


class EnvironmentalDataManager:
    """High-level data management for environmental analysis."""

    def __init__(self, storage: EnvironmentalStorage) -> None:
        """Initialize data manager."""
        self.storage = storage
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start_data_management(self) -> None:
        """Start data management tasks."""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def stop_data_management(self) -> None:
        """Stop data management tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def _periodic_cleanup(self) -> None:
        """Periodic data cleanup task."""
        while True:
            try:
                # Run cleanup every 24 hours
                await asyncio.sleep(24 * 3600)
                await self.storage.cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as err:
                _LOGGER.error("Periodic cleanup failed: %s", err)
