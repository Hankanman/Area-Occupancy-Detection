"""Tests for prior module."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.data.prior import MIN_PRIOR, Prior, PriorData
from custom_components.area_occupancy.utils import TimeInterval
from homeassistant.core import State
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestPriorData:
    """Test PriorData dataclass."""

    def test_prior_data_initialization(self) -> None:
        """Test PriorData initialization."""
        start_time = dt_util.utcnow() - timedelta(hours=2)
        end_time = dt_util.utcnow()
        states = []
        intervals = []

        data = PriorData(
            entity_id="binary_sensor.motion",
            start_time=start_time,
            end_time=end_time,
            states=states,
            intervals=intervals,
            occupied_seconds=3600,
            ratio=0.5,
            total_on_intervals=10,
            filtered_short_intervals=1,
            filtered_long_intervals=2,
            valid_intervals=7,
            max_filtered_duration_seconds=25 * 3600,  # 25 hours stuck sensor
        )

        assert data.entity_id == "binary_sensor.motion"
        assert data.start_time == start_time
        assert data.end_time == end_time
        assert data.states == states
        assert data.intervals == intervals
        assert data.occupied_seconds == 3600
        assert data.ratio == 0.5
        assert data.total_on_intervals == 10
        assert data.filtered_short_intervals == 1
        assert data.filtered_long_intervals == 2
        assert data.valid_intervals == 7
        assert data.max_filtered_duration_seconds == 25 * 3600


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
        assert prior.value is None
        assert prior.last_updated is None
        assert prior.sensor_hash is None
        assert prior.data == {}

    def test_current_value_property(self, mock_coordinator: Mock) -> None:
        """Test current_value property."""
        prior = Prior(mock_coordinator)

        # Test with value explicitly set
        prior.value = 0.35
        assert prior.current_value == 0.35

        # Test with None value - should return default
        prior.value = None
        assert prior.current_value == MIN_PRIOR  # MIN_PRIOR from data/prior.py

        # Test with value below minimum - should return default
        prior.value = 0.005
        assert prior.current_value == MIN_PRIOR

    def test_prior_intervals_empty(self, mock_coordinator: Mock) -> None:
        """Test prior_intervals property with no data."""
        prior = Prior(mock_coordinator)
        assert prior.prior_intervals == []

    def test_prior_intervals_single_sensor(self, mock_coordinator: Mock) -> None:
        """Test prior_intervals property with single sensor data."""
        prior = Prior(mock_coordinator)

        # Create test intervals
        base_time = dt_util.utcnow() - timedelta(hours=1)
        intervals: list[TimeInterval] = [
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

        prior.data["sensor1"] = PriorData(
            entity_id="sensor1",
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            states=[],
            intervals=intervals,
            occupied_seconds=1200,
            ratio=0.333,
            total_on_intervals=2,
            filtered_short_intervals=0,
            filtered_long_intervals=0,
            valid_intervals=2,
            max_filtered_duration_seconds=None,
        )

        result = prior.prior_intervals
        assert len(result) == 2
        assert result[0]["state"] == "on"
        assert result[1]["state"] == "on"

    def test_prior_intervals_overlapping_merge(self, mock_coordinator: Mock) -> None:
        """Test prior_intervals property merges overlapping intervals."""
        prior = Prior(mock_coordinator)

        base_time = dt_util.utcnow() - timedelta(hours=1)

        # Sensor 1 intervals
        intervals1: list[TimeInterval] = [
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
        intervals2: list[TimeInterval] = [
            {
                "state": "on",
                "start": base_time + timedelta(minutes=10),
                "end": base_time + timedelta(minutes=25),
            }
        ]

        prior.data["sensor1"] = PriorData(
            entity_id="sensor1",
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            states=[],
            intervals=intervals1,
            occupied_seconds=900,
            ratio=0.25,
            total_on_intervals=1,
            filtered_short_intervals=0,
            filtered_long_intervals=0,
            valid_intervals=1,
            max_filtered_duration_seconds=None,
        )

        prior.data["sensor2"] = PriorData(
            entity_id="sensor2",
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            states=[],
            intervals=intervals2,
            occupied_seconds=900,
            ratio=0.25,
            total_on_intervals=1,
            filtered_short_intervals=0,
            filtered_long_intervals=0,
            valid_intervals=1,
            max_filtered_duration_seconds=None,
        )

        result = prior.prior_intervals
        # Should merge overlapping "on" intervals into one
        assert len(result) == 1
        assert result[0]["start"] == base_time
        assert result[0]["end"] == base_time + timedelta(minutes=25)

    def test_prior_total_seconds(self, mock_coordinator: Mock) -> None:
        """Test prior_total_seconds property."""
        prior = Prior(mock_coordinator)

        base_time = dt_util.utcnow() - timedelta(hours=1)
        intervals: list[TimeInterval] = [
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

        prior.data["sensor1"] = PriorData(
            entity_id="sensor1",
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            states=[],
            intervals=intervals,
            occupied_seconds=1200,
            ratio=0.333,
            total_on_intervals=2,
            filtered_short_intervals=0,
            filtered_long_intervals=0,
            valid_intervals=2,
            max_filtered_duration_seconds=None,
        )

        assert prior.prior_total_seconds == 1200

    def test_prior_total_seconds_empty(self, mock_coordinator: Mock) -> None:
        """Test prior_total_seconds property with no intervals."""
        prior = Prior(mock_coordinator)
        assert prior.prior_total_seconds == 0

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    @patch("custom_components.area_occupancy.data.prior.states_to_intervals")
    async def test_calculate_success(
        self,
        mock_states_to_intervals: AsyncMock,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful prior calculation."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.config.history.period = 1

        prior = Prior(mock_coordinator)

        # Mock states from recorder
        base_time = dt_util.utcnow() - timedelta(days=1)
        mock_states = [
            Mock(spec=State, entity_id="binary_sensor.motion1", state="on"),
            Mock(spec=State, entity_id="binary_sensor.motion1", state="off"),
        ]
        mock_get_states.return_value = mock_states

        # Mock intervals
        mock_intervals = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(hours=8),
            },  # 8 hours on
            {
                "state": "off",
                "start": base_time + timedelta(hours=8),
                "end": base_time + timedelta(hours=24),
            },  # 16 hours off
        ]
        mock_states_to_intervals.return_value = mock_intervals

        result = await prior.calculate()

        # 8 hours / 24 hours = 0.333, with 5% buffer = 0.333 * 1.05 = 0.35
        expected = (8 * 3600) / (24 * 3600) * 1.05  # 0.35
        assert abs(result - expected) < 0.001
        assert prior.value == result
        assert prior.last_updated is not None
        assert prior.sensor_hash is not None

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    async def test_calculate_no_states(
        self, mock_get_states: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test prior calculation with no states returned."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.config.history.period = 1

        prior = Prior(mock_coordinator)
        mock_get_states.return_value = []

        # When no states are found, calculate() will have empty self.data
        # and will cause ZeroDivisionError in the current implementation.
        # This should be handled, but for now we test the actual behavior
        with pytest.raises(ZeroDivisionError):
            await prior.calculate()

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    @patch("custom_components.area_occupancy.data.prior.states_to_intervals")
    async def test_calculate_multiple_sensors(
        self,
        mock_states_to_intervals: AsyncMock,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
    ) -> None:
        """Test prior calculation with multiple sensors."""
        mock_coordinator.config.sensors.motion = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        mock_coordinator.config.history.period = 1

        prior = Prior(mock_coordinator)

        # Mock states for both sensors
        mock_get_states.return_value = [Mock(spec=State)]

        # Different ratios for each sensor
        def mock_intervals_side_effect(*args, **kwargs):
            base_time = dt_util.utcnow() - timedelta(days=1)
            return [
                {
                    "state": "on",
                    "start": base_time,
                    "end": base_time + timedelta(hours=6),
                },  # 25% for sensor1
                {
                    "state": "off",
                    "start": base_time + timedelta(hours=6),
                    "end": base_time + timedelta(hours=24),
                },
            ]

        mock_states_to_intervals.side_effect = mock_intervals_side_effect

        result = await prior.calculate()

        # Both sensors have 25% ratio, average = 25%, with 5% buffer = 26.25%
        expected = 0.25 * 1.05
        assert abs(result - expected) < 0.001

    async def test_update_cache_valid(self, mock_coordinator: Mock) -> None:
        """Test update method with valid cache."""
        mock_coordinator.config.sensors.motion = ["sensor1"]

        prior = Prior(mock_coordinator)
        prior.value = 0.4
        prior.last_updated = dt_util.utcnow() - timedelta(minutes=30)  # Fresh cache
        prior.sensor_hash = hash(frozenset(["sensor1"]))  # Must match sensor_ids

        result = await prior.update()

        assert result == 0.4  # Should return cached value

    @patch("custom_components.area_occupancy.data.prior.Prior.calculate")
    async def test_update_cache_invalid(
        self, mock_calculate: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test update method with invalid cache."""
        prior = Prior(mock_coordinator)
        prior.value = None  # No cached value

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
        prior.value = None

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
        prior.value = 0.3
        prior.last_updated = None
        assert not prior._is_cache_valid()

    def test_is_cache_valid_expired(self, mock_coordinator: Mock) -> None:
        """Test cache validation with expired timestamp."""
        prior = Prior(mock_coordinator)
        prior.value = 0.3
        prior.last_updated = dt_util.utcnow() - timedelta(hours=3)  # Expired
        prior.sensor_hash = hash(frozenset(["sensor1"]))
        mock_coordinator.config.sensors.motion = ["sensor1"]

        assert not prior._is_cache_valid()

    def test_is_cache_valid_sensors_changed(self, mock_coordinator: Mock) -> None:
        """Test cache validation with changed sensors."""
        prior = Prior(mock_coordinator)
        prior.value = 0.3
        prior.last_updated = dt_util.utcnow() - timedelta(minutes=30)  # Fresh
        prior.sensor_hash = hash(frozenset(["old_sensor"]))
        mock_coordinator.config.sensors.motion = ["new_sensor"]  # Different sensors

        assert not prior._is_cache_valid()

    def test_is_cache_valid_success(self, mock_coordinator: Mock) -> None:
        """Test successful cache validation."""
        mock_coordinator.config.sensors.motion = ["sensor1"]

        prior = Prior(mock_coordinator)
        prior.value = 0.3
        prior.last_updated = dt_util.utcnow() - timedelta(minutes=30)  # Fresh
        prior.sensor_hash = hash(frozenset(["sensor1"]))  # Must match sensor_ids

        assert prior._is_cache_valid()

    def test_to_dict(self, mock_coordinator: Mock) -> None:
        """Test converting prior to dictionary."""
        prior = Prior(mock_coordinator)
        timestamp = dt_util.utcnow()
        prior.value = 0.35
        prior.last_updated = timestamp
        prior.sensor_hash = 12345

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

        assert prior.value == 0.35
        assert prior.last_updated == timestamp
        assert prior.sensor_hash == 12345

    def test_from_dict_none_timestamp(self, mock_coordinator: Mock) -> None:
        """Test creating prior from dictionary with None timestamp."""
        data = {"value": 0.35, "last_updated": None, "sensor_hash": 12345}

        prior = Prior.from_dict(data, mock_coordinator)

        assert prior.value == 0.35
        assert prior.last_updated is None
        assert prior.sensor_hash == 12345

    def test_constants(self) -> None:
        """Test module constants."""
        assert MIN_PRIOR == 0.1

    async def test_integration_workflow(self, mock_coordinator: Mock) -> None:
        """Test complete workflow integration."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.config.history.period = 1

        prior = Prior(mock_coordinator)

        # Test initial state
        assert prior.current_value == MIN_PRIOR
        assert not prior._is_cache_valid()

        # Test serialization of initial state
        data = prior.to_dict()
        assert data["value"] is None

        # Test deserialization
        restored_prior = Prior.from_dict(data, mock_coordinator)
        assert restored_prior.value is None
        assert restored_prior.current_value == MIN_PRIOR
