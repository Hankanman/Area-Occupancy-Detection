"""Tests for number module."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.number import Threshold, async_setup_entry
from homeassistant.components.number import NumberMode
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.exceptions import ServiceValidationError


class TestThreshold:
    """Test Threshold number entity."""

    @pytest.fixture
    def threshold_entity(self, mock_coordinator_with_threshold: Mock) -> Threshold:
        """Create a threshold entity for testing."""
        return Threshold(mock_coordinator_with_threshold, "test_entry")

    def test_initialization(self, threshold_entity: Threshold) -> None:
        """Test Threshold entity initialization."""
        assert threshold_entity.coordinator is not None
        assert threshold_entity.unique_id == "test_entry_occupancy_threshold"
        assert threshold_entity.name == "Threshold"

    def test_entity_properties(self, threshold_entity: Threshold) -> None:
        """Test entity properties."""
        assert threshold_entity.native_min_value == 1.0
        assert threshold_entity.native_max_value == 99.0
        assert threshold_entity.native_step == 1.0
        assert threshold_entity.native_unit_of_measurement == PERCENTAGE
        assert threshold_entity.mode == NumberMode.BOX
        assert threshold_entity.entity_category == EntityCategory.CONFIG

    @pytest.mark.parametrize(
        ("coordinator_threshold", "expected_percentage"),
        [
            (0.75, 75.0),
            (0.5, 50.0),
            (0.01, 1.0),
            (0.99, 99.0),
            (0.1, 10.0),
            (0.25, 25.0),
            (0.9, 90.0),
        ],
    )
    def test_native_value_property(
        self,
        threshold_entity: Threshold,
        mock_coordinator_with_threshold: Mock,
        coordinator_threshold: float,
        expected_percentage: float,
    ) -> None:
        """Test native_value property with various threshold values."""
        mock_coordinator_with_threshold.threshold = coordinator_threshold
        assert threshold_entity.native_value == expected_percentage

    @pytest.mark.parametrize(
        "percentage_value",
        [1.0, 25.0, 50.0, 75.0, 99.0],
    )
    async def test_async_set_native_value_valid(
        self,
        threshold_entity: Threshold,
        mock_coordinator_with_threshold: Mock,
        percentage_value: float,
    ) -> None:
        """Test setting valid native values."""
        await threshold_entity.async_set_native_value(percentage_value)
        mock_coordinator_with_threshold.config.update_config.assert_called_once_with(
            {"threshold": percentage_value}
        )

    @pytest.mark.parametrize(
        ("invalid_value", "expected_error"),
        [
            (0.5, "Threshold value must be between 1.0 and 99.0"),
            (100.0, "Threshold value must be between 1.0 and 99.0"),
            (-1.0, "Threshold value must be between 1.0 and 99.0"),
        ],
    )
    async def test_async_set_native_value_invalid(
        self,
        threshold_entity: Threshold,
        mock_coordinator_with_threshold: Mock,
        invalid_value: float,
        expected_error: str,
    ) -> None:
        """Test setting invalid native values."""
        with pytest.raises(ServiceValidationError, match=expected_error):
            await threshold_entity.async_set_native_value(invalid_value)
        mock_coordinator_with_threshold.config.update_config.assert_not_called()

    async def test_async_set_native_value_coordinator_error(
        self, threshold_entity: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test handling coordinator errors."""
        mock_coordinator_with_threshold.config.update_config.side_effect = Exception(
            "Update failed"
        )
        with pytest.raises(Exception, match="Update failed"):
            await threshold_entity.async_set_native_value(75.0)

    @pytest.mark.parametrize(
        ("last_update_success", "expected_available"),
        [(True, True), (False, False)],
    )
    def test_available_property(
        self,
        threshold_entity: Threshold,
        mock_coordinator_with_threshold: Mock,
        last_update_success: bool,
        expected_available: bool,
    ) -> None:
        """Test available property based on coordinator state."""
        mock_coordinator_with_threshold.last_update_success = last_update_success
        assert threshold_entity.available is expected_available

    def test_device_info_property(
        self, threshold_entity: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test device_info property."""
        expected_device_info = mock_coordinator_with_threshold.device_info
        assert threshold_entity.device_info == expected_device_info

    @pytest.mark.parametrize(
        "percentage_value",
        [33.33, 66.67, 12.5, 87.5],
    )
    async def test_value_conversion_precision(
        self,
        threshold_entity: Threshold,
        mock_coordinator_with_threshold: Mock,
        percentage_value: float,
    ) -> None:
        """Test value conversion precision."""
        await threshold_entity.async_set_native_value(percentage_value)
        called_value = mock_coordinator_with_threshold.config.update_config.call_args[
            0
        ][0]["threshold"]
        assert called_value == percentage_value

    async def test_multiple_threshold_updates(
        self, threshold_entity: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test multiple threshold updates in sequence."""
        updates = [25.0, 50.0, 75.0, 90.0]

        for percentage in updates:
            mock_coordinator_with_threshold.config.update_config.reset_mock()
            await threshold_entity.async_set_native_value(percentage)
            mock_coordinator_with_threshold.config.update_config.assert_called_once_with(
                {"threshold": percentage}
            )

    async def test_error_recovery(
        self, threshold_entity: Threshold, mock_coordinator_with_threshold: Mock
    ) -> None:
        """Test error recovery scenarios."""
        # Test coordinator error followed by successful update
        mock_coordinator_with_threshold.config.update_config.side_effect = [
            Exception("Temporary error"),
            None,  # Success on second call
        ]

        # First call should raise exception
        with pytest.raises(Exception, match="Temporary error"):
            await threshold_entity.async_set_native_value(75.0)

        # Second call should succeed
        await threshold_entity.async_set_native_value(75.0)
        mock_coordinator_with_threshold.config.update_config.assert_called_with(
            {"threshold": 75.0}
        )


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful setup entry."""
        mock_async_add_entities = Mock()
        mock_coordinator = Mock()
        mock_coordinator.device_info = {"test": "device_info"}
        mock_config_entry.runtime_data = mock_coordinator

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 1
        assert isinstance(entities[0], Threshold)

    async def test_async_setup_entry_with_coordinator_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup entry with coordinator data."""
        mock_coordinator = Mock()
        mock_coordinator.threshold = 0.8
        mock_coordinator.device_info = {"test": "device_info"}
        mock_config_entry.runtime_data = mock_coordinator
        mock_async_add_entities = Mock()

        await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

        entities = mock_async_add_entities.call_args[0][0]
        threshold_entity = entities[0]
        assert threshold_entity.coordinator == mock_coordinator
        assert threshold_entity.unique_id == "test_entry_id_occupancy_threshold"
