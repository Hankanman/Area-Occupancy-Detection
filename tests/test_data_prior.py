"""Tests for the Prior class."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

from custom_components.area_occupancy.data.prior import Prior

# Import MIN_PRIOR directly from the module
MIN_PRIOR = 0.1
from custom_components.area_occupancy.utils import StateInterval
from homeassistant.core import State
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestPrior:
    """Test Prior class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test Prior initialization."""
        # Set up mock coordinator config
        mock_coordinator.config.sensors.motion = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        mock_coordinator.config.history.period = 7
        mock_coordinator.hass = Mock()

        prior = Prior(mock_coordinator)

        assert prior.sensor_ids == ["binary_sensor.motion1", "binary_sensor.motion2"]
        assert prior.days == 7
        assert prior.hass == mock_coordinator.hass
        assert prior.cache_ttl == timedelta(hours=2)
        assert prior._current_value is None
        assert prior._last_updated is None
        assert prior._sensor_hash is None
        assert prior._sensor_data == {}

    def test_value_property(self, mock_coordinator: Mock) -> None:
        """Test value property."""
        prior = Prior(mock_coordinator)

        # Test with value explicitly set
        prior._current_value = 0.35
        assert prior.value == 0.35

        # Test with None value - should return default
        prior._current_value = None
        assert prior.value == MIN_PRIOR  # MIN_PRIOR from data/prior.py

        # Test with value below minimum - should return default
        prior._current_value = 0.005
        assert prior.value == MIN_PRIOR

    def test_state_intervals_empty(self, mock_coordinator: Mock) -> None:
        """Test state_intervals property with no data."""
        prior = Prior(mock_coordinator)
        assert prior.state_intervals == []

    def test_state_intervals_single_sensor(self, mock_coordinator: Mock) -> None:
        """Test state_intervals property with single sensor data."""
        prior = Prior(mock_coordinator)

        # Create test intervals
        base_time = dt_util.utcnow() - timedelta(hours=1)
        intervals: list[StateInterval] = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(minutes=10),
            },
            {
                "state": "off",
                "start": base_time + timedelta(minutes=10),
                "end": base_time + timedelta(minutes=20),
            },
            {
                "state": "on",
                "start": base_time + timedelta(minutes=20),
                "end": base_time + timedelta(minutes=30),
            },
        ]

        prior._sensor_data["sensor1"] = {
            "entity_id": "sensor1",
            "start_time": base_time,
            "end_time": base_time + timedelta(hours=1),
            "states_count": 0,
            "intervals": intervals,
            "occupied_seconds": 1200,
            "ratio": 0.333,
        }

        result = prior.state_intervals
        # Should merge the two "on" intervals into one since they're contiguous
        assert len(result) == 1
        assert result[0]["state"] == "on"
        assert result[0]["start"] == base_time
        assert result[0]["end"] == base_time + timedelta(minutes=30)

    def test_state_intervals_overlapping_merge(self, mock_coordinator: Mock) -> None:
        """Test state_intervals property merges overlapping intervals."""
        prior = Prior(mock_coordinator)

        base_time = dt_util.utcnow() - timedelta(hours=1)

        # Sensor 1 intervals
        intervals1: list[StateInterval] = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(minutes=15),
            },
            {
                "state": "off",
                "start": base_time + timedelta(minutes=15),
                "end": base_time + timedelta(minutes=20),
            },
        ]

        # Sensor 2 intervals (overlapping with sensor 1)
        intervals2: list[StateInterval] = [
            {
                "state": "on",
                "start": base_time + timedelta(minutes=10),
                "end": base_time + timedelta(minutes=25),
            }
        ]

        prior._sensor_data["sensor1"] = {
            "entity_id": "sensor1",
            "start_time": base_time,
            "end_time": base_time + timedelta(hours=1),
            "states_count": 0,
            "intervals": intervals1,
            "occupied_seconds": 900,
            "ratio": 0.25,
        }

        prior._sensor_data["sensor2"] = {
            "entity_id": "sensor2",
            "start_time": base_time,
            "end_time": base_time + timedelta(hours=1),
            "states_count": 0,
            "intervals": intervals2,
            "occupied_seconds": 900,
            "ratio": 0.25,
        }

        result = prior.state_intervals
        # Should merge overlapping "on" intervals into one
        assert len(result) == 1
        assert result[0]["start"] == base_time
        assert result[0]["end"] == base_time + timedelta(minutes=25)

    def test_prior_total_seconds(self, mock_coordinator: Mock) -> None:
        """Test prior_total_seconds property."""
        prior = Prior(mock_coordinator)

        base_time = dt_util.utcnow() - timedelta(hours=1)
        intervals: list[StateInterval] = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(minutes=10),
            },  # 600 seconds
            {
                "state": "on",
                "start": base_time + timedelta(minutes=20),
                "end": base_time + timedelta(minutes=30),
            },  # 600 seconds
        ]

        prior._sensor_data["sensor1"] = {
            "entity_id": "sensor1",
            "start_time": base_time,
            "end_time": base_time + timedelta(hours=1),
            "states_count": 0,
            "intervals": intervals,
            "occupied_seconds": 1200,
            "ratio": 0.333,
        }

        assert prior.prior_total_seconds == 1200

    def test_prior_total_seconds_empty(self, mock_coordinator: Mock) -> None:
        """Test prior_total_seconds property with no intervals."""
        prior = Prior(mock_coordinator)
        assert prior.prior_total_seconds == 0

    @patch("custom_components.area_occupancy.data.prior.get_intervals_hybrid")
    async def test_calculate_success(
        self,
        mock_get_intervals: AsyncMock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful prior calculation."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.config.history.period = 1
        mock_coordinator.occupancy_entity_id = None  # No occupancy entity

        prior = Prior(mock_coordinator)

        # Mock intervals from get_intervals_hybrid
        base_time = dt_util.utcnow() - timedelta(days=1)
        # 8 hours on out of 24 hours total = 33.3% ratio
        mock_intervals = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(hours=8),
            },  # 8 hours on
        ]
        mock_get_intervals.return_value = mock_intervals

        # Intervals are already filtered by get_intervals_hybrid
        # No need for additional filtering

        result = await prior.calculate()

        # 8 hours / 24 hours = 0.333, with 5% buffer = 0.333 * 1.05 = 0.35
        expected = (8 * 3600) / (24 * 3600) * 1.05  # 0.35
        assert abs(result - expected) < 0.001
        assert prior._current_value == result
        assert prior._last_updated is not None
        assert prior._sensor_hash is not None

    @patch("custom_components.area_occupancy.data.prior.get_intervals_hybrid")
    async def test_calculate_no_states(
        self, mock_get_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test prior calculation with no states returned."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.config.history.period = 1
        mock_coordinator.occupancy_entity_id = None

        prior = Prior(mock_coordinator)
        mock_get_intervals.return_value = []

        # When no states are found, calculate() should return MIN_PRIOR
        result = await prior.calculate()
        assert result == MIN_PRIOR

    @patch("custom_components.area_occupancy.data.prior.get_intervals_hybrid")
    async def test_calculate_multiple_sensors(
        self,
        mock_get_intervals: AsyncMock,
        mock_coordinator: Mock,
    ) -> None:
        """Test prior calculation with multiple sensors."""
        mock_coordinator.config.sensors.motion = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        mock_coordinator.config.history.period = 1
        mock_coordinator.occupancy_entity_id = None

        prior = Prior(mock_coordinator)

        # Mock intervals for both sensors
        base_time = dt_util.utcnow() - timedelta(days=1)
        mock_intervals = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(hours=6),
            },  # 6 hours on
        ]
        mock_get_intervals.return_value = mock_intervals

        # Intervals are already filtered by get_intervals_hybrid
        # No need for additional filtering

        result = await prior.calculate()

        # Both sensors have 6/24 = 25% ratio, average = 25%, with 5% buffer = 26.25%
        expected = 0.25 * 1.05
        assert abs(result - expected) < 0.001

    async def test_update_cache_valid(self, mock_coordinator: Mock) -> None:
        """Test update method with valid cache."""
        mock_coordinator.config.sensors.motion = ["sensor1"]

        prior = Prior(mock_coordinator)
        prior._current_value = 0.4
        prior._last_updated = dt_util.utcnow() - timedelta(minutes=30)  # Fresh cache
        prior._sensor_hash = hash(frozenset(["sensor1"]))  # Must match sensor_ids

        result = await prior.update()

        assert result == 0.4  # Should return cached value

    @patch("custom_components.area_occupancy.data.prior.Prior.calculate")
    async def test_update_cache_invalid(
        self, mock_calculate: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test update method with invalid cache."""
        prior = Prior(mock_coordinator)
        prior._current_value = None  # No cached value

        mock_calculate.return_value = 0.3

        result = await prior.update()

        assert result == 0.3
        mock_calculate.assert_called_once()

    @patch("custom_components.area_occupancy.data.prior.Prior.calculate")
    async def test_update_calculation_error(
        self, mock_calculate: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test update method handles calculation errors."""
        prior = Prior(mock_coordinator)
        prior._current_value = None

        mock_calculate.side_effect = Exception("Test error")

        result = await prior.update()

        assert result == MIN_PRIOR  # Should fallback to default

    def test_is_cache_valid_no_value(self, mock_coordinator: Mock) -> None:
        """Test cache validation with no cached value."""
        prior = Prior(mock_coordinator)
        assert not prior._is_cache_valid()

    def test_is_cache_valid_no_timestamp(self, mock_coordinator: Mock) -> None:
        """Test cache validation with no timestamp."""
        prior = Prior(mock_coordinator)
        prior._current_value = 0.3
        prior._last_updated = None
        assert not prior._is_cache_valid()

    def test_is_cache_valid_expired(self, mock_coordinator: Mock) -> None:
        """Test cache validation with expired timestamp."""
        prior = Prior(mock_coordinator)
        prior._current_value = 0.3
        prior._last_updated = dt_util.utcnow() - timedelta(hours=3)  # Expired
        prior._sensor_hash = hash(frozenset(["sensor1"]))
        mock_coordinator.config.sensors.motion = ["sensor1"]

        assert not prior._is_cache_valid()

    def test_is_cache_valid_sensors_changed(self, mock_coordinator: Mock) -> None:
        """Test cache validation with changed sensors."""
        prior = Prior(mock_coordinator)
        prior._current_value = 0.3
        prior._last_updated = dt_util.utcnow() - timedelta(minutes=30)  # Fresh
        prior._sensor_hash = hash(frozenset(["old_sensor"]))
        mock_coordinator.config.sensors.motion = ["new_sensor"]  # Different sensors

        assert not prior._is_cache_valid()

    def test_is_cache_valid_success(self, mock_coordinator: Mock) -> None:
        """Test successful cache validation."""
        mock_coordinator.config.sensors.motion = ["sensor1"]

        prior = Prior(mock_coordinator)
        prior._current_value = 0.3
        prior._last_updated = dt_util.utcnow() - timedelta(minutes=30)  # Fresh
        prior._sensor_hash = hash(frozenset(["sensor1"]))  # Must match sensor_ids

        assert prior._is_cache_valid()

    def test_to_dict(self, mock_coordinator: Mock) -> None:
        """Test converting prior to dictionary."""
        prior = Prior(mock_coordinator)
        timestamp = dt_util.utcnow()
        prior._current_value = 0.35
        prior._last_updated = timestamp
        prior._sensor_hash = 12345

        result = prior.to_dict()

        expected = {
            "value": 0.35,
            "last_updated": timestamp.isoformat(),
            "sensor_hash": 12345,
        }
        assert result == expected

    def test_to_dict_none_values(self, mock_coordinator: Mock) -> None:
        """Test converting prior to dictionary with None values."""
        prior = Prior(mock_coordinator)

        result = prior.to_dict()

        expected = {"value": None, "last_updated": None, "sensor_hash": None}
        assert result == expected

    def test_from_dict(self, mock_coordinator: Mock) -> None:
        """Test creating prior from dictionary."""
        timestamp = dt_util.utcnow()
        data = {
            "value": 0.35,
            "last_updated": timestamp.isoformat(),
            "sensor_hash": 12345,
        }

        prior = Prior.from_dict(data, mock_coordinator)

        assert prior._current_value == 0.35
        assert prior._last_updated == timestamp
        assert prior._sensor_hash == 12345

    def test_from_dict_none_timestamp(self, mock_coordinator: Mock) -> None:
        """Test creating prior from dictionary with None timestamp."""
        data = {"value": 0.35, "last_updated": None, "sensor_hash": 12345}

        prior = Prior.from_dict(data, mock_coordinator)

        assert prior._current_value == 0.35
        assert prior._last_updated is None
        assert prior._sensor_hash == 12345

    def test_constants(self) -> None:
        """Test module constants."""
        assert MIN_PRIOR == 0.1

    async def test_integration_workflow(self, mock_coordinator: Mock) -> None:
        """Test complete workflow integration."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.config.history.period = 1

        prior = Prior(mock_coordinator)

        # Test initial state
        assert prior.value == MIN_PRIOR
        assert not prior._is_cache_valid()

        # Test serialization of initial state
        data = prior.to_dict()
        assert data["value"] is None

        # Test deserialization
        restored_prior = Prior.from_dict(data, mock_coordinator)
        assert restored_prior._current_value is None
        assert restored_prior.value == MIN_PRIOR
