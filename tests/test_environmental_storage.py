"""Tests for environmental storage module."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.area_occupancy.environmental_storage import (
    EnvironmentalDataManager,
    EnvironmentalStorage,
)
from custom_components.area_occupancy.exceptions import EnvironmentalStorageError
from custom_components.area_occupancy.types import (
    EnvironmentalConfig,
    EnvironmentalResult,
    EnvironmentalSensorConfig,
    SensorReading,
)


@pytest.fixture
def temp_storage_path():
    """Create temporary storage path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


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
        },
        analysis_frequency=60,
        minimum_data_points=100,
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
    }


@pytest.fixture
def environmental_result():
    """Create test environmental result."""
    return EnvironmentalResult(
        probability=0.7,
        confidence=0.8,
        method="ml",
        sensor_contributions={"sensor.co2": 0.6},
        model_version="1.0.0",
    )


class TestEnvironmentalStorage:
    """Test EnvironmentalStorage class."""

    async def test_init(self, hass: HomeAssistant, temp_storage_path):
        """Test storage initialization."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        assert storage.hass == hass
        assert storage.storage_path == temp_storage_path
        assert storage.data_path.exists()
        assert storage._config_cache is None
        assert storage._readings_cache == []
        assert storage._results_cache == []

    async def test_save_config(
        self, hass: HomeAssistant, temp_storage_path, environmental_config
    ):
        """Test saving environmental configuration."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        await storage.save_config(environmental_config)

        assert storage._config_file.exists()
        assert storage._config_cache == environmental_config

        # Verify file contents
        with open(storage._config_file, "r") as f:
            config_data = json.load(f)

        assert "version" in config_data
        assert "sensors" in config_data
        assert "global_config" in config_data
        assert "sensor.co2" in config_data["sensors"]

    async def test_save_config_error(
        self, hass: HomeAssistant, temp_storage_path, environmental_config
    ):
        """Test save config error handling."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Make directory read-only to cause write error
        with (
            patch("builtins.open", side_effect=PermissionError("Permission denied")),
            pytest.raises(EnvironmentalStorageError),
        ):
            await storage.save_config(environmental_config)

    async def test_load_config(
        self, hass: HomeAssistant, temp_storage_path, environmental_config
    ):
        """Test loading environmental configuration."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # First save config
        await storage.save_config(environmental_config)

        # Clear cache
        storage._config_cache = None

        # Load config
        loaded_config = await storage.load_config()

        assert loaded_config is not None
        assert storage._config_cache is not None

    async def test_load_config_no_file(self, hass: HomeAssistant, temp_storage_path):
        """Test loading config when file doesn't exist."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        loaded_config = await storage.load_config()
        assert loaded_config is None

    async def test_load_config_error(self, hass: HomeAssistant, temp_storage_path):
        """Test load config error handling."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Create invalid JSON file
        with open(storage._config_file, "w") as f:
            f.write("invalid json")

        loaded_config = await storage.load_config()
        assert loaded_config is None

    async def test_save_sensor_readings(
        self, hass: HomeAssistant, temp_storage_path, sensor_readings
    ):
        """Test saving sensor readings."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        await storage.save_sensor_readings(sensor_readings, True)

        assert len(storage._readings_cache) == 1

        reading_data = storage._readings_cache[0]
        assert reading_data["occupancy"] is True
        assert "sensor.co2" in reading_data["readings"]
        assert reading_data["readings"]["sensor.co2"]["value"] == 450.0

    async def test_save_analysis_result(
        self, hass: HomeAssistant, temp_storage_path, environmental_result
    ):
        """Test saving analysis result."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        await storage.save_analysis_result(environmental_result)

        assert len(storage._results_cache) == 1

        result_data = storage._results_cache[0]
        assert result_data["probability"] == 0.7
        assert result_data["confidence"] == 0.8
        assert result_data["method"] == "ml"
        assert result_data["model_version"] == "1.0.0"

    async def test_get_historical_readings(
        self, hass: HomeAssistant, temp_storage_path, sensor_readings
    ):
        """Test getting historical readings."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Add some readings to cache
        await storage.save_sensor_readings(sensor_readings, True)
        await storage.save_sensor_readings(sensor_readings, False)

        historical_data = await storage.get_historical_readings("sensor.co2", 24)

        assert len(historical_data) == 2
        assert all(
            isinstance(item, tuple) and len(item) == 2 for item in historical_data
        )
        assert all(
            isinstance(item[0], datetime) and isinstance(item[1], (int, float))
            for item in historical_data
        )

    async def test_get_historical_readings_empty(
        self, hass: HomeAssistant, temp_storage_path
    ):
        """Test getting historical readings when none exist."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        historical_data = await storage.get_historical_readings(
            "sensor.nonexistent", 24
        )
        assert historical_data == []

    async def test_get_training_data(
        self, hass: HomeAssistant, temp_storage_path, sensor_readings
    ):
        """Test getting training data."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Add readings with features
        await storage.save_sensor_readings(sensor_readings, True)

        with patch.object(storage, "_load_readings_from_file") as mock_load:
            mock_load.return_value = storage._readings_cache

            training_data = await storage.get_training_data(30)

            # Currently returns empty due to feature extraction limitations
            assert isinstance(training_data, list)

    async def test_cleanup_old_data(
        self, hass: HomeAssistant, temp_storage_path, sensor_readings
    ):
        """Test cleaning up old data."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Add some readings
        await storage.save_sensor_readings(sensor_readings, True)

        # Manually add old reading
        old_timestamp = (dt_util.utcnow() - timedelta(days=100)).isoformat()
        old_reading = {
            "timestamp": old_timestamp,
            "occupancy": False,
            "readings": {},
        }
        storage._readings_cache.insert(0, old_reading)

        initial_count = len(storage._readings_cache)
        await storage.cleanup_old_data()

        # Old data should be removed
        assert len(storage._readings_cache) < initial_count

        # Verify remaining data is recent
        for reading in storage._readings_cache:
            timestamp = datetime.fromisoformat(reading["timestamp"])
            age = dt_util.utcnow() - timestamp
            assert age.days <= 90  # Within retention period

    async def test_get_storage_stats(
        self, hass: HomeAssistant, temp_storage_path, sensor_readings
    ):
        """Test getting storage statistics."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Add some data
        await storage.save_sensor_readings(sensor_readings, True)

        stats = await storage.get_storage_stats()

        assert isinstance(stats, dict)
        assert "readings_count" in stats
        assert "results_count" in stats
        assert "config_exists" in stats
        assert "total_size" in stats
        assert stats["readings_count"] >= 1

    async def test_get_storage_stats_error(
        self, hass: HomeAssistant, temp_storage_path
    ):
        """Test storage stats error handling."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Create a file to test with
        storage._readings_file.touch()

        with patch("pathlib.Path.stat", side_effect=OSError("File error")):
            stats = await storage.get_storage_stats()
            # Should return stats with 0 file sizes due to OSError handling
            assert stats["readings_file_size"] == 0
            assert stats["results_file_size"] == 0

    async def test_cache_size_limits(
        self, hass: HomeAssistant, temp_storage_path, sensor_readings
    ):
        """Test cache size limits are enforced."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Add many readings to exceed cache limit
        for i in range(1100):  # More than limit of 1000
            await storage.save_sensor_readings(sensor_readings, i % 2 == 0)

        # Cache should be limited to 500 (half of 1000 when limit exceeded)
        assert len(storage._readings_cache) == 500

    async def test_extract_features_from_reading(
        self, hass: HomeAssistant, temp_storage_path
    ):
        """Test extracting features from reading data."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        reading_data = {
            "timestamp": dt_util.utcnow().isoformat(),
            "occupancy": True,
            "readings": {
                "sensor.co2": {"value": 450.0},
                "sensor.temperature": {"value": 22.5},
            },
        }

        features = await storage._extract_features_from_reading(reading_data)

        assert features is not None
        assert "hour_of_day" in features
        assert "day_of_week" in features
        assert "is_weekend" in features
        # Should have sensor features too
        sensor_features = [k for k in features if k.startswith("sensor_")]
        assert len(sensor_features) > 0

    async def test_extract_features_insufficient_data(
        self, hass: HomeAssistant, temp_storage_path
    ):
        """Test feature extraction with insufficient data."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Reading with no sensor data
        reading_data = {
            "timestamp": dt_util.utcnow().isoformat(),
            "occupancy": True,
            "readings": {},
        }

        features = await storage._extract_features_from_reading(reading_data)
        assert features is None  # Should return None for insufficient features

    async def test_extract_features_error(self, hass: HomeAssistant, temp_storage_path):
        """Test feature extraction error handling."""
        storage = EnvironmentalStorage(hass, temp_storage_path)

        # Invalid reading data
        reading_data = {"invalid": "data"}

        features = await storage._extract_features_from_reading(reading_data)
        assert features is None


class TestEnvironmentalDataManager:
    """Test EnvironmentalDataManager class."""

    async def test_init(self, hass: HomeAssistant, temp_storage_path):
        """Test data manager initialization."""
        storage = EnvironmentalStorage(hass, temp_storage_path)
        manager = EnvironmentalDataManager(storage)

        assert manager.storage == storage
        assert manager._cleanup_task is None

    async def test_start_data_management(self, hass: HomeAssistant, temp_storage_path):
        """Test starting data management."""
        storage = EnvironmentalStorage(hass, temp_storage_path)
        manager = EnvironmentalDataManager(storage)

        with patch.object(manager, "_periodic_cleanup") as mock_cleanup:
            mock_cleanup.return_value = AsyncMock()

            await manager.start_data_management()

            assert manager._cleanup_task is not None

    async def test_stop_data_management(self, hass: HomeAssistant, temp_storage_path):
        """Test stopping data management."""
        storage = EnvironmentalStorage(hass, temp_storage_path)
        manager = EnvironmentalDataManager(storage)

        # Create a real asyncio task that we can cancel
        import asyncio

        async def long_running_task():
            """Sleep indefinitely to simulate a long-running task."""
            try:
                await asyncio.sleep(3600)  # Sleep for an hour
            except asyncio.CancelledError:
                raise

        # Create and assign the task
        task = asyncio.create_task(long_running_task())
        manager._cleanup_task = task

        # Stop should cancel and wait for the task
        await manager.stop_data_management()

        # Verify the task was cancelled
        assert task.cancelled() or task.done()

    async def test_stop_data_management_no_task(
        self, hass: HomeAssistant, temp_storage_path
    ):
        """Test stopping data management when no task exists."""
        storage = EnvironmentalStorage(hass, temp_storage_path)
        manager = EnvironmentalDataManager(storage)

        # Should not raise error when no task exists
        await manager.stop_data_management()
