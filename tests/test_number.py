"""Tests for number module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.number import Threshold, async_setup_entry
from homeassistant.components.number import NumberMode
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.exceptions import ServiceValidationError


class TestThreshold:
    """Test Threshold number entity."""

    def test_initialization(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test Threshold entity initialization."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        assert entity.coordinator == mock_coordinator_with_threshold
        assert entity.unique_id == "test_entry_occupancy_threshold"
        assert entity.name == "Threshold"

    def test_entity_properties(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test entity properties."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        # Test native_min_value
        assert entity.native_min_value == 1.0

        # Test native_max_value
        assert entity.native_max_value == 99.0

        # Test native_step
        assert entity.native_step == 1.0

        # Test native_unit_of_measurement
        assert entity.native_unit_of_measurement == PERCENTAGE

        # Test mode
        assert entity.mode == NumberMode.BOX

    def test_native_value_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test native_value property."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        # Test with coordinator threshold (stored as decimal, returned as percentage)
        mock_coordinator_with_threshold.threshold = 0.75
        assert entity.native_value == 75.0  # Converted to percentage

        # Test with different threshold
        mock_coordinator_with_threshold.threshold = 0.5
        assert entity.native_value == 50.0

        # Test with edge values
        mock_coordinator_with_threshold.threshold = 0.01
        assert entity.native_value == 1.0

        mock_coordinator_with_threshold.threshold = 0.99
        assert entity.native_value == 99.0

    async def test_async_set_native_value_valid(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test setting valid native value."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        await entity.async_set_native_value(75.0)

        # Should call coordinator's config_manager.update_config with threshold value
        mock_coordinator_with_threshold.config_manager.update_config.assert_called_once_with(
            {"threshold": 75.0}
        )

    async def test_async_set_native_value_edge_cases(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test setting edge case values."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        # Test minimum value
        await entity.async_set_native_value(1.0)
        mock_coordinator_with_threshold.config_manager.update_config.assert_called_with(
            {"threshold": 1.0}
        )

        # Test maximum value
        mock_coordinator_with_threshold.config_manager.update_config.reset_mock()
        await entity.async_set_native_value(99.0)
        mock_coordinator_with_threshold.config_manager.update_config.assert_called_with(
            {"threshold": 99.0}
        )

        # Test mid-range value
        mock_coordinator_with_threshold.config_manager.update_config.reset_mock()
        await entity.async_set_native_value(50.0)
        mock_coordinator_with_threshold.config_manager.update_config.assert_called_with(
            {"threshold": 50.0}
        )

    async def test_async_set_native_value_invalid_low(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test setting value below minimum."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        with pytest.raises(
            ServiceValidationError, match="Threshold value must be between 1.0 and 99.0"
        ):
            await entity.async_set_native_value(0.5)

        # Should not call coordinator
        mock_coordinator_with_threshold.config_manager.update_config.assert_not_called()

    async def test_async_set_native_value_invalid_high(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test setting value above maximum."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        with pytest.raises(
            ServiceValidationError, match="Threshold value must be between 1.0 and 99.0"
        ):
            await entity.async_set_native_value(100.0)

        # Should not call coordinator
        mock_coordinator_with_threshold.config_manager.update_config.assert_not_called()

    async def test_async_set_native_value_coordinator_error(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test handling coordinator errors."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        # Mock coordinator to raise an exception
        mock_coordinator_with_threshold.config_manager.update_config.side_effect = (
            Exception("Update failed")
        )

        with pytest.raises(Exception, match="Update failed"):
            await entity.async_set_native_value(75.0)

    def test_entity_category(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test entity category."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        assert entity.entity_category == EntityCategory.CONFIG

    def test_available_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test available property."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        # Test when coordinator is available (CoordinatorEntity uses last_update_success)
        mock_coordinator_with_threshold.last_update_success = True
        assert entity.available is True

        # Test when coordinator is not available
        mock_coordinator_with_threshold.last_update_success = False
        assert entity.available is False

    def test_device_info_property(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test device_info property."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        expected_device_info = mock_coordinator_with_threshold.device_info
        assert entity.device_info == expected_device_info

    async def test_value_conversion_precision(
        self, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test value conversion precision."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        # Test various precision scenarios - coordinator expects percentage values
        test_cases = [
            (33.33, 33.33),  # Percentage to percentage
            (66.67, 66.67),
            (12.5, 12.5),
            (87.5, 87.5),
        ]

        for percentage, expected_percentage in test_cases:
            mock_coordinator_with_threshold.config_manager.update_config.reset_mock()
            await entity.async_set_native_value(percentage)

            # Check that the call was made with the percentage value
            called_value = (
                mock_coordinator_with_threshold.config_manager.update_config.call_args[
                    0
                ][0]["threshold"]
            )
            assert called_value == expected_percentage


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful setup entry."""
        mock_async_add_entities = Mock()

        # Set up runtime_data with a mock coordinator
        mock_coordinator = Mock()
        mock_coordinator.device_info = {"test": "device_info"}
        mock_config_entry.runtime_data = mock_coordinator

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
        mock_coordinator.device_info = {"test": "device_info"}
        mock_config_entry.runtime_data = mock_coordinator

        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        # Verify entity was created with correct coordinator
        entities = mock_async_add_entities.call_args[0][0]
        threshold_entity = entities[0]
        assert threshold_entity.coordinator == mock_coordinator
        assert threshold_entity.unique_id == "test_entry_id_occupancy_threshold"


class TestThresholdIntegration:
    """Test Threshold entity integration scenarios."""

    @pytest.fixture
    def comprehensive_threshold(
        self, mock_coordinator_with_threshold: Mock
    ) -> Threshold:
        """Create a comprehensive threshold entity for testing."""
        return Threshold(mock_coordinator_with_threshold, "test_entry")

    async def test_threshold_update_workflow(
        self, comprehensive_threshold: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test complete threshold update workflow."""
        entity = comprehensive_threshold

        # Initial state
        assert entity.native_value == 60.0

        # Update threshold
        await entity.async_set_native_value(75.0)

        # Verify coordinator was called with percentage
        mock_coordinator_with_threshold.config_manager.update_config.assert_called_once_with(
            {"threshold": 75.0}
        )

        # Simulate config update effect on coordinator
        mock_coordinator_with_threshold.config.threshold = 0.75
        mock_coordinator_with_threshold.threshold = 0.75

        # Verify coordinator state was updated
        assert mock_coordinator_with_threshold.threshold == 0.75  # Converted to decimal
        assert entity.native_value == 75.0  # Still in percentage

    async def test_multiple_threshold_updates(
        self, comprehensive_threshold: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test multiple threshold updates."""
        entity = comprehensive_threshold

        # Perform multiple updates - coordinator expects percentages
        updates = [25.0, 50.0, 75.0, 90.0]
        expected_percentages = [25.0, 50.0, 75.0, 90.0]

        for percentage, expected_percentage in zip(
            updates, expected_percentages, strict=False
        ):
            mock_coordinator_with_threshold.config_manager.update_config.reset_mock()
            await entity.async_set_native_value(percentage)
            mock_coordinator_with_threshold.config_manager.update_config.assert_called_once_with(
                {"threshold": expected_percentage}
            )

            # Simulate config update effect on coordinator
            mock_coordinator_with_threshold.config.threshold = (
                expected_percentage / 100.0
            )
            mock_coordinator_with_threshold.threshold = expected_percentage / 100.0

            # Verify entity value
            assert entity.native_value == expected_percentage

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
        self, comprehensive_threshold: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test error recovery scenarios."""
        entity = comprehensive_threshold

        # Test coordinator error followed by successful update
        mock_coordinator_with_threshold.config_manager.update_config.side_effect = [
            Exception("Temporary error"),
            None,  # Success on second call
        ]

        # First call should raise exception
        with pytest.raises(Exception, match="Temporary error"):
            await entity.async_set_native_value(75.0)

        # Second call should succeed
        await entity.async_set_native_value(75.0)

        # Simulate config update effect on coordinator
        mock_coordinator_with_threshold.config.threshold = 0.75
        mock_coordinator_with_threshold.threshold = 0.75

        # Verify final state
        mock_coordinator_with_threshold.config_manager.update_config.assert_called_with(
            {"threshold": 75.0}
        )
        assert entity.native_value == 75.0

    def test_state_consistency(
        self, comprehensive_threshold: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test state consistency across different coordinator states."""
        entity = comprehensive_threshold

        # Test various coordinator threshold values
        test_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        expected_percentages = [10.0, 25.0, 50.0, 75.0, 90.0]

        for threshold, expected_percentage in zip(
            test_thresholds, expected_percentages, strict=False
        ):
            mock_coordinator_with_threshold.threshold = threshold
            assert entity.native_value == expected_percentage

    async def test_concurrent_updates(
        self, comprehensive_threshold: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test handling of concurrent updates."""
        entity = comprehensive_threshold

        # Mock config_manager.update_config to track calls
        call_count = 0
        original_method = mock_coordinator_with_threshold.config_manager.update_config

        async def track_calls(value):
            nonlocal call_count
            call_count += 1
            await original_method(value)

        mock_coordinator_with_threshold.config_manager.update_config = AsyncMock(
            side_effect=track_calls
        )

        # Perform concurrent-like updates (sequential in test)
        await entity.async_set_native_value(30.0)
        await entity.async_set_native_value(70.0)

        # Both calls should have been made
        assert call_count == 2

        # Verify final state
        mock_coordinator_with_threshold.config.threshold = 0.7
        mock_coordinator_with_threshold.threshold = 0.7
        assert entity.native_value == 70.0
