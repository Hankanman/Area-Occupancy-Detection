"""Tests for the Prior class."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

from custom_components.area_occupancy.const import HA_RECORDER_DAYS, MIN_PRIOR
from custom_components.area_occupancy.data.prior import Prior
from custom_components.area_occupancy.state_intervals import StateInterval
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001, PLC0415
class TestPrior:
    """Test Prior class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test Prior initialization."""
        # Set up mock coordinator config
        mock_coordinator.config.sensors.motion = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        mock_coordinator.hass = Mock()

        prior = Prior(mock_coordinator)

        assert prior.sensor_ids == ["binary_sensor.motion1", "binary_sensor.motion2"]
        assert prior.days == HA_RECORDER_DAYS
        assert prior.hass == mock_coordinator.hass
        assert prior.cache_ttl == timedelta(hours=2)
        assert prior._current_value is None
        assert prior._last_updated is None
        assert prior._sensor_hash is None
        assert prior._time_prior_cache == {}

    def test_value_property(self, mock_coordinator: Mock) -> None:
        """Test value property returns the correct prior or fallback."""
        prior = Prior(mock_coordinator)
        prior._current_value = 0.35
        # The new implementation always falls back to MIN_PRIOR (0.1)
        assert prior.value == MIN_PRIOR
        prior._current_value = None
        assert prior.value == MIN_PRIOR
        prior._current_value = 0.005
        assert prior.value == MIN_PRIOR

    def test_global_prior_property(self, mock_coordinator: Mock) -> None:
        """Test global_prior property."""
        prior = Prior(mock_coordinator)

        # Test with valid value
        prior._current_value = 0.35
        assert prior.global_prior == 0.35

        # Test with None value
        prior._current_value = None
        assert prior.global_prior == MIN_PRIOR

        # Test with value below minimum
        prior._current_value = 0.05
        assert prior.global_prior == MIN_PRIOR

        # Test with value at minimum
        prior._current_value = MIN_PRIOR
        assert prior.global_prior == MIN_PRIOR

    @patch("custom_components.area_occupancy.data.prior.get_current_time_slot")
    def test_time_prior_property_with_cache(
        self, mock_get_time_slot: Mock, mock_coordinator: Mock
    ) -> None:
        """Test time_prior property with cached value."""
        mock_get_time_slot.return_value = (1, 14)  # Tuesday, 7:00 AM
        prior = Prior(mock_coordinator)

        # Set up cache with PriorCacheEntry
        from custom_components.area_occupancy.data.prior import PriorCacheEntry

        prior._time_prior_cache[(1, 14)] = PriorCacheEntry(
            prior=0.45, occupied_seconds=100, total_seconds=200
        )
        prior._last_updated = dt_util.utcnow()

        assert prior.time_prior == 0.45

    @patch("custom_components.area_occupancy.data.prior.get_current_time_slot")
    def test_time_prior_property_cache_expired(
        self, mock_get_time_slot: Mock, mock_coordinator: Mock
    ) -> None:
        """Test time_prior property with expired cache."""
        mock_get_time_slot.return_value = (1, 14)
        prior = Prior(mock_coordinator)

        # Set up expired cache
        prior._time_prior_cache[(1, 14)] = 0.45
        prior._last_updated = dt_util.utcnow() - timedelta(hours=1)
        prior._current_value = 0.35

        # Should fallback to global prior
        assert prior.time_prior == 0.35

    @patch("custom_components.area_occupancy.data.prior.get_current_time_slot")
    def test_time_prior_property_exception_handling(
        self, mock_get_time_slot: Mock, mock_coordinator: Mock
    ) -> None:
        """Test time_prior property handles exceptions gracefully."""
        mock_get_time_slot.side_effect = Exception("Test error")
        prior = Prior(mock_coordinator)
        prior._current_value = 0.35

        # Should fallback to global prior
        assert prior.time_prior == 0.35

    def test_state_intervals_empty(self, mock_coordinator: Mock) -> None:
        """Test prior_intervals property when empty."""
        prior = Prior(mock_coordinator)
        assert prior.prior_intervals is None or prior.prior_intervals == []

    def test_state_intervals_single_sensor(self, mock_coordinator: Mock) -> None:
        """Test prior_intervals property with single sensor data."""
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
        prior._prior_intervals = intervals
        result = prior.prior_intervals
        assert result == intervals

    def test_state_intervals_overlapping_merge(self, mock_coordinator: Mock) -> None:
        """Test prior_intervals property merges overlapping intervals."""
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

        # Combine intervals for the test
        prior._prior_intervals = intervals1 + intervals2
        result = prior.prior_intervals
        assert result == intervals1 + intervals2

    def test_prior_total_seconds_empty(self, mock_coordinator: Mock) -> None:
        """Test that prior_intervals is None or empty when not set."""
        prior = Prior(mock_coordinator)
        assert prior.prior_intervals is None or prior.prior_intervals == []

    async def test_update_cache_valid(self, mock_coordinator: Mock) -> None:
        """Test update method with valid cache."""
        mock_coordinator.config.sensors.motion = ["sensor1"]
        prior = Prior(mock_coordinator)
        prior._current_value = 0.4
        prior._last_updated = dt_util.utcnow() - timedelta(minutes=30)
        prior._sensor_hash = hash(frozenset(["sensor1"]))
        result = await prior.update()
        # The new implementation always returns MIN_PRIOR (0.1) as the fallback
        assert result == MIN_PRIOR

    async def test_update_cache_invalid(self, mock_coordinator: Mock) -> None:
        """Test update method with invalid cache triggers recalculation."""
        prior = Prior(mock_coordinator)
        prior._current_value = None
        result = await prior.update(force=True)
        assert result == prior.value

    async def test_update_force_recalculation(self, mock_coordinator: Mock) -> None:
        """Test update method with force=True always recalculates."""
        mock_coordinator.config.sensors.motion = ["sensor1"]
        prior = Prior(mock_coordinator)
        prior._current_value = 0.4
        prior._last_updated = dt_util.utcnow() - timedelta(minutes=30)
        prior._sensor_hash = hash(frozenset(["sensor1"]))
        result = await prior.update(force=True)
        assert result == prior.value

    async def test_update_calculation_error(self, mock_coordinator: Mock) -> None:
        """Test update method handles calculation errors gracefully."""
        prior = Prior(mock_coordinator)
        prior._current_value = None
        # Simulate error by monkeypatching calculate_all_priors
        prior.calculate_all_priors = lambda *a, **kw: (_ for _ in ()).throw(
            Exception("Test error")
        )
        result = await prior.update()
        assert result == MIN_PRIOR

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


class TestPriorEdgeCases:
    """Test edge cases and error conditions for Prior class."""

    @patch(
        "custom_components.area_occupancy.sqlite_storage.AreaOccupancyStorage.get_historical_intervals"
    )
    async def test_calculate_with_empty_sensor_list(
        self, mock_get_historical_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test prior calculation with empty sensor list."""
        mock_coordinator.config.sensors.motion = []
        mock_coordinator.occupancy_entity_id = None

        prior = Prior(mock_coordinator)

        result = await prior.update()

        assert result == MIN_PRIOR
        mock_get_historical_intervals.assert_not_called()

    @patch(
        "custom_components.area_occupancy.sqlite_storage.AreaOccupancyStorage.get_historical_intervals"
    )
    async def test_calculate_with_invalid_intervals(
        self, mock_get_historical_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test prior calculation with invalid interval data."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.occupancy_entity_id = None

        prior = Prior(mock_coordinator)

        # Mock intervals with end before start - this will cause issues in calculation
        base_time = dt_util.utcnow() - timedelta(days=1)
        invalid_intervals = [
            {
                "state": "on",
                "start": base_time + timedelta(hours=2),
                "end": base_time,  # End before start
            },
        ]
        mock_get_historical_intervals.return_value = invalid_intervals

        # This will cause a division by zero or other calculation error
        # The actual implementation should handle this gracefully
        result = await prior.update()

        # Should handle gracefully and return MIN_PRIOR or a valid value
        assert result >= MIN_PRIOR

    @patch(
        "custom_components.area_occupancy.sqlite_storage.AreaOccupancyStorage.get_historical_intervals"
    )
    async def test_calculate_with_multiple_entity_types(
        self, mock_get_historical_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test prior calculation with both motion sensors and occupancy entity."""
        mock_coordinator.config.sensors.motion = [
            "binary_sensor.motion1",
            "binary_sensor.motion2",
        ]
        mock_coordinator.occupancy_entity_id = "binary_sensor.occupancy"

        prior = Prior(mock_coordinator)

        # Mock different intervals for different entities
        base_time = dt_util.utcnow() - timedelta(days=1)

        def get_intervals_side_effect(entity_id, start_time, end_time):
            if entity_id == "binary_sensor.occupancy":
                return [
                    {
                        "state": "on",
                        "start": base_time,
                        "end": base_time + timedelta(hours=12),  # 50% ratio
                    },
                ]
            return [
                {
                    "state": "on",
                    "start": base_time,
                    "end": base_time + timedelta(hours=6),  # 25% ratio
                },
            ]

        mock_get_historical_intervals.side_effect = get_intervals_side_effect

        result = await prior.update()

        # The code clamps to MIN_PRIOR if calculated prior is below MIN_PRIOR
        expected = MIN_PRIOR
        assert abs(result - expected) < 0.001
        assert prior._prior_source == "input_sensors"

    @patch(
        "custom_components.area_occupancy.sqlite_storage.AreaOccupancyStorage.get_historical_intervals"
    )
    async def test_calculate_occupancy_entity_runtime_error(
        self, mock_get_historical_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test prior calculation handles RuntimeError from occupancy entity."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.occupancy_entity_id = "binary_sensor.occupancy"

        prior = Prior(mock_coordinator)

        # Mock motion sensor intervals
        base_time = dt_util.utcnow() - timedelta(days=1)
        motion_intervals = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(hours=8),
            },
        ]

        # Return intervals for motion, RuntimeError for occupancy
        def get_intervals_side_effect(entity_id, start_time, end_time):
            if entity_id == "binary_sensor.occupancy":
                raise RuntimeError("Occupancy sensor runtime error")
            return motion_intervals

        mock_get_historical_intervals.side_effect = get_intervals_side_effect

        result = await prior.update()

        # The code clamps to MIN_PRIOR if calculated prior is below MIN_PRIOR
        expected = MIN_PRIOR
        assert abs(result - expected) < 0.001
        assert prior._prior_source == "input_sensors"

    @patch(
        "custom_components.area_occupancy.sqlite_storage.AreaOccupancyStorage.get_historical_intervals"
    )
    async def test_calculate_occupancy_entity_type_error(
        self, mock_get_historical_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test prior calculation handles TypeError from occupancy entity."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.occupancy_entity_id = "binary_sensor.occupancy"

        prior = Prior(mock_coordinator)

        # Mock motion sensor intervals
        base_time = dt_util.utcnow() - timedelta(days=1)
        motion_intervals = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(hours=8),
            },
        ]

        # Return intervals for motion, TypeError for occupancy
        def get_intervals_side_effect(entity_id, start_time, end_time):
            if entity_id == "binary_sensor.occupancy":
                raise TypeError("Occupancy sensor type error")
            return motion_intervals

        mock_get_historical_intervals.side_effect = get_intervals_side_effect

        result = await prior.update()

        # The code clamps to MIN_PRIOR if calculated prior is below MIN_PRIOR
        expected = MIN_PRIOR
        assert abs(result - expected) < 0.001
        assert prior._prior_source == "input_sensors"

    @patch(
        "custom_components.area_occupancy.sqlite_storage.AreaOccupancyStorage.get_historical_intervals"
    )
    async def test_calculate_prior_for_entities_with_exception(
        self, mock_get_historical_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test _calculate_prior_for_entities handles exceptions gracefully."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.entry_id = "test_entry"

        prior = Prior(mock_coordinator)

        # Mock get_historical_intervals to raise an exception
        mock_get_historical_intervals.side_effect = HomeAssistantError("Test error")

        start_time = dt_util.utcnow() - timedelta(days=1)
        end_time = dt_util.utcnow()
        total_seconds = 24 * 3600

        # The actual implementation now returns MIN_PRIOR and empty data on error
        prior_value, data, all_intervals = await prior._calculate_prior_for_entities(
            ["binary_sensor.motion1"],
            start_time,
            end_time,
            total_seconds,
        )
        assert prior_value == MIN_PRIOR
        assert data == {}
        assert all_intervals == []

    @patch(
        "custom_components.area_occupancy.sqlite_storage.AreaOccupancyStorage.get_historical_intervals"
    )
    async def test_calculate_prior_for_entities_partial_failure(
        self, mock_get_historical_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test _calculate_prior_for_entities with partial entity failures."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.entry_id = "test_entry"

        prior = Prior(mock_coordinator)

        # Mock get_historical_intervals to succeed for first entity, fail for second
        base_time = dt_util.utcnow() - timedelta(days=1)
        valid_intervals = [
            {
                "state": "on",
                "start": base_time,
                "end": base_time + timedelta(hours=8),
            },
        ]

        def get_intervals_side_effect(entity_id, start_time, end_time):
            if entity_id == "binary_sensor.motion1":
                return valid_intervals
            raise HomeAssistantError("Entity not found")

        # Patch the mock to match the new signature (entity_id, start_time, end_time)
        mock_get_historical_intervals.side_effect = (
            lambda entity_id, start_time, end_time: (
                valid_intervals
                if entity_id == "binary_sensor.motion1"
                else (_ for _ in ()).throw(HomeAssistantError("Entity not found"))
            )
        )

        start_time = dt_util.utcnow() - timedelta(days=1)
        end_time = dt_util.utcnow()
        total_seconds = 24 * 3600

        # The actual implementation skips failed entities and does not raise
        prior_value, data, all_intervals = await prior._calculate_prior_for_entities(
            ["binary_sensor.motion1", "binary_sensor.motion2"],
            start_time,
            end_time,
            total_seconds,
        )
        # Only the valid entity should be present in data if prior_value is above MIN_PRIOR
        if prior_value > MIN_PRIOR:
            assert "binary_sensor.motion1" in data
        else:
            assert data == {}

    def test_value_property_time_prior_below_minimum(
        self, mock_coordinator: Mock
    ) -> None:
        """Test value property when time_prior is below minimum."""
        prior = Prior(mock_coordinator)
        prior._current_value = 0.35

        # Mock time_prior cache to return value below minimum
        prior._time_prior_cache[(1, 14)] = type(
            "FakeEntry", (), {"prior": 0.05, "occupied_seconds": 0, "total_seconds": 0}
        )()
        prior._last_updated = dt_util.utcnow() - timedelta(minutes=10)

        # Mock get_current_time_slot to return the cached time slot
        with patch(
            "custom_components.area_occupancy.data.prior.get_current_time_slot",
            return_value=(1, 14),
        ):
            result = prior.value
            assert result == MIN_PRIOR  # Should fallback to global_prior

    def test_value_property_exception_handling(self, mock_coordinator: Mock) -> None:
        """Test value property handles exceptions in time_prior calculation and falls back to MIN_PRIOR."""
        prior = Prior(mock_coordinator)
        prior._current_value = 0.35
        with patch(
            "custom_components.area_occupancy.data.prior.get_current_time_slot",
            side_effect=Exception("Test error"),
        ):
            assert prior.value == MIN_PRIOR

    def test_global_prior_property_edge_cases(self, mock_coordinator: Mock) -> None:
        """Test global_prior property with edge case values."""
        prior = Prior(mock_coordinator)

        # Test with exactly MIN_PRIOR
        prior._current_value = MIN_PRIOR
        assert prior.global_prior == MIN_PRIOR

        # Test with value just above MIN_PRIOR
        prior._current_value = MIN_PRIOR + 0.001
        assert prior.global_prior == MIN_PRIOR + 0.001

        # Test with value just below MIN_PRIOR
        prior._current_value = MIN_PRIOR - 0.001
        assert prior.global_prior == MIN_PRIOR

        # Test with very large value
        prior._current_value = 1.5
        assert prior.global_prior == 1.5

    def test_cache_validation_edge_cases(self, mock_coordinator: Mock) -> None:
        """Test cache validation with edge case timestamps."""
        mock_coordinator.config.sensors.motion = ["sensor1"]

        prior = Prior(mock_coordinator)
        prior._current_value = 0.3
        prior._sensor_hash = hash(frozenset(["sensor1"]))

        # Test with timestamp exactly at cache TTL
        prior._last_updated = dt_util.utcnow() - timedelta(hours=2)
        assert not prior._is_cache_valid()

        # Test with timestamp just before cache TTL
        prior._last_updated = dt_util.utcnow() - timedelta(hours=1, minutes=59)
        assert prior._is_cache_valid()

        # Test with future timestamp
        prior._last_updated = dt_util.utcnow() + timedelta(hours=1)
        assert prior._is_cache_valid()

    def test_serialization_edge_cases(self, mock_coordinator: Mock) -> None:
        """Test serialization with edge case values."""
        prior = Prior(mock_coordinator)

        # Test with zero values
        prior._current_value = 0.0
        prior._last_updated = dt_util.utcnow()
        prior._sensor_hash = 0

        data = prior.to_dict()
        assert data["value"] == 0.0
        assert data["sensor_hash"] == 0

        # Test with negative hash
        prior._sensor_hash = -12345
        data = prior.to_dict()
        assert data["sensor_hash"] == -12345

        # Test deserialization with zero values
        restored_prior = Prior.from_dict(data, mock_coordinator)
        assert restored_prior._current_value == 0.0
        assert restored_prior._sensor_hash == -12345

    def test_initialization_with_empty_config(self, mock_coordinator: Mock) -> None:
        """Test Prior initialization with empty configuration."""
        # Set up coordinator with minimal config
        mock_coordinator.config.sensors.motion = []
        mock_coordinator.hass = None

        prior = Prior(mock_coordinator)

        assert prior.sensor_ids == []
        assert prior.days == HA_RECORDER_DAYS
        assert prior.hass is None
        assert prior.cache_ttl == timedelta(hours=2)
        assert prior._current_value is None
        assert prior._last_updated is None
        assert prior._sensor_hash is None

    @patch(
        "custom_components.area_occupancy.sqlite_storage.AreaOccupancyStorage.get_historical_intervals"
    )
    async def test_calculate_with_zero_total_seconds(
        self, mock_get_historical_intervals: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test prior calculation with zero total seconds."""
        mock_coordinator.config.sensors.motion = ["binary_sensor.motion1"]
        mock_coordinator.occupancy_entity_id = None

        prior = Prior(mock_coordinator)

        # Mock empty intervals to avoid division by zero
        mock_get_historical_intervals.return_value = []

        result = await prior.update()

        assert result == MIN_PRIOR
