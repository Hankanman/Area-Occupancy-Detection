"""Tests for number module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import CONF_THRESHOLD
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.number import Threshold, async_setup_entry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful setup entry."""
        mock_async_add_entities = Mock()
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator_with_areas

        await async_setup_entry(hass, mock_config_entry, mock_async_add_entities)

        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 1
        assert isinstance(entities[0], Threshold)

    async def test_async_setup_entry_with_coordinator_data(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test setup entry with coordinator data."""
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator_with_areas
        mock_config_entry.entry_id = "test_entry_id"
        mock_async_add_entities = Mock()

        await async_setup_entry(hass, mock_config_entry, mock_async_add_entities)

        entities = mock_async_add_entities.call_args[0][0]
        threshold_entity = entities[0]
        assert threshold_entity.coordinator == coordinator_with_areas
        # unique_id format uses entry_id, device_id, and entity_name
        entry_id = coordinator_with_areas.entry_id
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area(area_name)
        device_id = next(iter(area.device_info()["identifiers"]))[1]
        expected_unique_id = f"{entry_id}_{device_id}_occupancy_threshold"
        assert threshold_entity.unique_id == expected_unique_id


class TestThreshold:
    """Test Threshold entity."""

    def test_native_value_converts_zero_to_percentage(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value converts internal 0.0 to 0%."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock area.threshold() to return 0.0
        area = coordinator_with_areas.get_area(area_name)
        with patch.object(area, "threshold", return_value=0.0):
            assert threshold_entity.native_value == 0.0

    def test_native_value_converts_one_to_percentage(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value converts internal 1.0 to 100%."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock area.threshold() to return 1.0
        area = coordinator_with_areas.get_area(area_name)
        with patch.object(area, "threshold", return_value=1.0):
            assert threshold_entity.native_value == 100.0

    def test_native_value_converts_mid_value_to_percentage(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value converts internal 0.5 to 50%."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock area.threshold() to return 0.5
        area = coordinator_with_areas.get_area(area_name)
        with patch.object(area, "threshold", return_value=0.5):
            assert threshold_entity.native_value == 50.0

    def test_native_value_with_missing_area(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value returns 0.0 when area is missing."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock _get_area to return None
        with patch.object(threshold_entity, "_get_area", return_value=None):
            assert threshold_entity.native_value == 0.0

    async def test_async_set_native_value_validates_minimum(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value raises error for values < 1.0."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Test with value below minimum (0.0)
        with pytest.raises(ServiceValidationError) as exc_info:
            await threshold_entity.async_set_native_value(0.0)
        assert "must be between" in str(exc_info.value)

        # Test with value just below minimum (0.9)
        with pytest.raises(ServiceValidationError) as exc_info:
            await threshold_entity.async_set_native_value(0.9)
        assert "must be between" in str(exc_info.value)

    async def test_async_set_native_value_validates_maximum(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value raises error for values > 99.0."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Test with value above maximum (100.0)
        with pytest.raises(ServiceValidationError) as exc_info:
            await threshold_entity.async_set_native_value(100.0)
        assert "must be between" in str(exc_info.value)

        # Test with value just above maximum (99.1)
        with pytest.raises(ServiceValidationError) as exc_info:
            await threshold_entity.async_set_native_value(99.1)
        assert "must be between" in str(exc_info.value)

    async def test_async_set_native_value_stores_percentage_directly(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value stores percentage value directly (not divided by 100)."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        area = coordinator_with_areas.get_area(area_name)
        mock_update_config = AsyncMock()
        with patch.object(area.config, "update_config", mock_update_config):
            # Test with valid percentage value (50.0)
            await threshold_entity.async_set_native_value(50.0)

            # Verify update_config was called with percentage value directly (not divided)
            mock_update_config.assert_called_once_with({CONF_THRESHOLD: 50.0})

    async def test_async_set_native_value_stores_boundary_values(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value stores boundary values (1.0 and 99.0) correctly."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        area = coordinator_with_areas.get_area(area_name)
        mock_update_config = AsyncMock()
        with patch.object(area.config, "update_config", mock_update_config):
            # Test minimum valid value (1.0)
            await threshold_entity.async_set_native_value(1.0)
            mock_update_config.assert_called_with({CONF_THRESHOLD: 1.0})

            # Reset mock for next test
            mock_update_config.reset_mock()

            # Test maximum valid value (99.0)
            await threshold_entity.async_set_native_value(99.0)
            mock_update_config.assert_called_with({CONF_THRESHOLD: 99.0})

    async def test_async_set_native_value_with_missing_area(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value handles missing area gracefully."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock _get_area to return None
        with patch.object(threshold_entity, "_get_area", return_value=None):
            # Should not raise an error, just log warning and return
            await threshold_entity.async_set_native_value(50.0)
            # No exception should be raised
