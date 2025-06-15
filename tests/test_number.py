"""Tests for number module."""

from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.number import Threshold, async_setup_entry
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError


class TestThreshold:
    """Test Threshold number entity."""

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.threshold = 0.6
        coordinator.available = True
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }
        coordinator.async_update_threshold = AsyncMock()
        return coordinator

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test Threshold entity initialization."""
        entity = Threshold(mock_coordinator, "test_entry")

        assert entity.coordinator == mock_coordinator
        assert entity._entry_id == "test_entry"
        assert entity.unique_id == "test_entry_threshold"
        assert entity.name == "Threshold"

    def test_entity_properties(self, mock_coordinator: Mock) -> None:
        """Test entity properties."""
        entity = Threshold(mock_coordinator, "test_entry")

        # Test native_min_value
        assert entity.native_min_value == 1.0

        # Test native_max_value
        assert entity.native_max_value == 99.0

        # Test native_step
        assert entity.native_step == 1.0

        # Test native_unit_of_measurement
        assert entity.native_unit_of_measurement == "%"

        # Test mode
        assert entity.mode == "slider"

    def test_native_value_property(self, mock_coordinator: Mock) -> None:
        """Test native_value property."""
        entity = Threshold(mock_coordinator, "test_entry")

        # Test with coordinator threshold
        mock_coordinator.threshold = 0.75
        assert entity.native_value == 75.0  # Converted to percentage

        # Test with different threshold
        mock_coordinator.threshold = 0.5
        assert entity.native_value == 50.0

        # Test with edge values
        mock_coordinator.threshold = 0.01
        assert entity.native_value == 1.0

        mock_coordinator.threshold = 0.99
        assert entity.native_value == 99.0

    async def test_async_set_native_value_valid(self, mock_coordinator: Mock) -> None:
        """Test setting valid native value."""
        entity = Threshold(mock_coordinator, "test_entry")

        await entity.async_set_native_value(75.0)

        # Should call coordinator's async_update_threshold with decimal value
        mock_coordinator.async_update_threshold.assert_called_once_with(0.75)

    async def test_async_set_native_value_edge_cases(
        self, mock_coordinator: Mock
    ) -> None:
        """Test setting edge case values."""
        entity = Threshold(mock_coordinator, "test_entry")

        # Test minimum value
        await entity.async_set_native_value(1.0)
        mock_coordinator.async_update_threshold.assert_called_with(0.01)

        # Test maximum value
        mock_coordinator.async_update_threshold.reset_mock()
        await entity.async_set_native_value(99.0)
        mock_coordinator.async_update_threshold.assert_called_with(0.99)

        # Test mid-range value
        mock_coordinator.async_update_threshold.reset_mock()
        await entity.async_set_native_value(50.0)
        mock_coordinator.async_update_threshold.assert_called_with(0.5)

    async def test_async_set_native_value_invalid_low(
        self, mock_coordinator: Mock
    ) -> None:
        """Test setting value below minimum."""
        entity = Threshold(mock_coordinator, "test_entry")

        with pytest.raises(
            ServiceValidationError, match="Threshold must be between 1 and 99"
        ):
            await entity.async_set_native_value(0.5)

        # Should not call coordinator
        mock_coordinator.async_update_threshold.assert_not_called()

    async def test_async_set_native_value_invalid_high(
        self, mock_coordinator: Mock
    ) -> None:
        """Test setting value above maximum."""
        entity = Threshold(mock_coordinator, "test_entry")

        with pytest.raises(
            ServiceValidationError, match="Threshold must be between 1 and 99"
        ):
            await entity.async_set_native_value(100.0)

        # Should not call coordinator
        mock_coordinator.async_update_threshold.assert_not_called()

    async def test_async_set_native_value_coordinator_error(
        self, mock_coordinator: Mock
    ) -> None:
        """Test handling coordinator errors."""
        entity = Threshold(mock_coordinator, "test_entry")

        # Mock coordinator to raise an exception
        mock_coordinator.async_update_threshold.side_effect = Exception("Update failed")

        with pytest.raises(Exception, match="Update failed"):
            await entity.async_set_native_value(75.0)

    def test_icon_property(self, mock_coordinator: Mock) -> None:
        """Test icon property."""
        entity = Threshold(mock_coordinator, "test_entry")

        assert entity.icon == "mdi:percent"

    def test_entity_category(self, mock_coordinator: Mock) -> None:
        """Test entity category."""
        entity = Threshold(mock_coordinator, "test_entry")

        assert entity.entity_category == "config"

    def test_available_property(self, mock_coordinator: Mock) -> None:
        """Test available property."""
        entity = Threshold(mock_coordinator, "test_entry")

        # Test when coordinator is available
        mock_coordinator.available = True
        assert entity.available is True

        # Test when coordinator is not available
        mock_coordinator.available = False
        assert entity.available is False

    def test_device_info_property(self, mock_coordinator: Mock) -> None:
        """Test device_info property."""
        entity = Threshold(mock_coordinator, "test_entry")

        expected_device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }

        assert entity.device_info == expected_device_info

    async def test_value_conversion_precision(self, mock_coordinator: Mock) -> None:
        """Test value conversion precision."""
        entity = Threshold(mock_coordinator, "test_entry")

        # Test various precision scenarios
        test_cases = [
            (33.33, 0.3333),  # Should round to 0.33
            (66.67, 0.6667),  # Should round to 0.67
            (12.5, 0.125),
            (87.5, 0.875),
        ]

        for percentage, expected_decimal in test_cases:
            mock_coordinator.async_update_threshold.reset_mock()
            await entity.async_set_native_value(percentage)

            # Check that the call was made with properly converted value
            called_value = mock_coordinator.async_update_threshold.call_args[0][0]
            assert (
                abs(called_value - expected_decimal) < 0.01
            )  # Allow small floating point differences


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry"
        entry.runtime_data = Mock()  # Mock coordinator
        return entry

    async def test_async_setup_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful setup entry."""
        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        # Should add threshold entity
        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 1
        assert isinstance(entities[0], Threshold)

    async def test_async_setup_entry_with_coordinator_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup entry with coordinator data."""
        # Set up coordinator with specific data
        mock_coordinator = Mock()
        mock_coordinator.threshold = 0.8
        mock_config_entry.runtime_data = mock_coordinator

        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        # Verify entity was created with correct coordinator
        entities = mock_async_add_entities.call_args[0][0]
        threshold_entity = entities[0]
        assert threshold_entity.coordinator == mock_coordinator
        assert threshold_entity._entry_id == "test_entry"


class TestThresholdIntegration:
    """Test Threshold entity integration scenarios."""

    @pytest.fixture
    def comprehensive_threshold(self, mock_coordinator: Mock) -> Threshold:
        """Create a comprehensive threshold entity for testing."""
        return Threshold(mock_coordinator, "test_entry")

    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        """Create a comprehensive mock coordinator."""
        coordinator = Mock()
        coordinator.threshold = 0.6
        coordinator.available = True
        coordinator.device_info = {
            "identifiers": {("area_occupancy", "test_entry")},
            "name": "Test Area",
        }
        coordinator.async_update_threshold = AsyncMock()
        return coordinator

    async def test_threshold_update_workflow(
        self, comprehensive_threshold: Threshold, mock_coordinator: Mock
    ) -> None:
        """Test complete threshold update workflow."""
        entity = comprehensive_threshold

        # Initial state
        assert entity.native_value == 60.0

        # Update threshold
        await entity.async_set_native_value(75.0)

        # Verify coordinator was called
        mock_coordinator.async_update_threshold.assert_called_once_with(0.75)

        # Simulate coordinator update
        mock_coordinator.threshold = 0.75

        # Verify new value
        assert entity.native_value == 75.0

    async def test_multiple_threshold_updates(
        self, comprehensive_threshold: Threshold, mock_coordinator: Mock
    ) -> None:
        """Test multiple threshold updates."""
        entity = comprehensive_threshold

        # Perform multiple updates
        updates = [25.0, 50.0, 75.0, 90.0]
        expected_decimals = [0.25, 0.5, 0.75, 0.9]

        for percentage, expected_decimal in zip(
            updates, expected_decimals, strict=False
        ):
            mock_coordinator.async_update_threshold.reset_mock()
            await entity.async_set_native_value(percentage)
            mock_coordinator.async_update_threshold.assert_called_once_with(
                expected_decimal
            )

    def test_threshold_boundary_validation(
        self, comprehensive_threshold: Threshold
    ) -> None:
        """Test threshold boundary validation."""
        entity = comprehensive_threshold

        # Test valid boundaries
        assert entity.native_min_value == 1.0
        assert entity.native_max_value == 99.0

        # Test that values at boundaries are valid
        # (actual validation happens in async_set_native_value)
        assert entity.native_min_value <= 1.0
        assert entity.native_max_value >= 99.0

    async def test_error_recovery(
        self, comprehensive_threshold: Threshold, mock_coordinator: Mock
    ) -> None:
        """Test error recovery scenarios."""
        entity = comprehensive_threshold

        # Test coordinator error followed by successful update
        mock_coordinator.async_update_threshold.side_effect = [
            Exception("Temporary error"),
            None,  # Success on second call
        ]

        # First call should raise exception
        with pytest.raises(Exception, match="Temporary error"):
            await entity.async_set_native_value(75.0)

        # Reset side effect for second call
        mock_coordinator.async_update_threshold.side_effect = None

        # Second call should succeed
        await entity.async_set_native_value(80.0)
        mock_coordinator.async_update_threshold.assert_called_with(0.8)

    def test_state_consistency(
        self, comprehensive_threshold: Threshold, mock_coordinator: Mock
    ) -> None:
        """Test state consistency across different coordinator states."""
        entity = comprehensive_threshold

        # Test various coordinator threshold values
        test_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        expected_percentages = [10.0, 25.0, 50.0, 75.0, 90.0]

        for threshold, expected_percentage in zip(
            test_thresholds, expected_percentages, strict=False
        ):
            mock_coordinator.threshold = threshold
            assert entity.native_value == expected_percentage

    async def test_concurrent_updates(
        self, comprehensive_threshold: Threshold, mock_coordinator: Mock
    ) -> None:
        """Test handling of concurrent updates."""
        entity = comprehensive_threshold

        # Mock async_update_threshold to track calls
        call_count = 0
        original_method = mock_coordinator.async_update_threshold

        async def track_calls(value):
            nonlocal call_count
            call_count += 1
            await original_method(value)

        mock_coordinator.async_update_threshold = AsyncMock(side_effect=track_calls)

        # Perform concurrent-like updates (sequential in test)
        await entity.async_set_native_value(30.0)
        await entity.async_set_native_value(70.0)

        # Both calls should have been made
        assert call_count == 2
        assert mock_coordinator.async_update_threshold.call_count == 2
