"""Tests for number module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import (
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_THRESHOLD,
)
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
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test successful setup entry."""
        mock_async_add_entities = Mock()
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator

        await async_setup_entry(hass, mock_config_entry, mock_async_add_entities)

        mock_async_add_entities.assert_called_once()
        entities = mock_async_add_entities.call_args[0][0]
        assert len(entities) == 1
        assert isinstance(entities[0], Threshold)

    async def test_async_setup_entry_with_coordinator_data(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test setup entry with coordinator data."""
        # Use real coordinator
        mock_config_entry.runtime_data = coordinator
        mock_async_add_entities = Mock()

        await async_setup_entry(hass, mock_config_entry, mock_async_add_entities)

        entities = mock_async_add_entities.call_args[0][0]
        threshold_entity = entities[0]
        assert threshold_entity.coordinator == coordinator
        # unique_id format uses entry_id, device_id, and entity_name
        entry_id = coordinator.entry_id
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        device_id = next(iter(area.device_info()["identifiers"]))[1]
        expected_unique_id = f"{entry_id}_{device_id}_occupancy_threshold"
        assert threshold_entity.unique_id == expected_unique_id


class TestThreshold:
    """Test Threshold entity."""

    def test_native_value_converts_zero_to_percentage(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value converts internal 0.0 to 0%."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock area.threshold() to return 0.0
        area = coordinator.get_area(area_name)
        with patch.object(area, "threshold", return_value=0.0):
            assert threshold_entity.native_value == 0.0

    def test_native_value_converts_one_to_percentage(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value converts internal 1.0 to 100%."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock area.threshold() to return 1.0
        area = coordinator.get_area(area_name)
        with patch.object(area, "threshold", return_value=1.0):
            assert threshold_entity.native_value == 100.0

    def test_native_value_converts_mid_value_to_percentage(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value converts internal 0.5 to 50%."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock area.threshold() to return 0.5
        area = coordinator.get_area(area_name)
        with patch.object(area, "threshold", return_value=0.5):
            assert threshold_entity.native_value == 50.0

    def test_native_value_with_missing_area(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test native_value returns 0.0 when area is missing."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock _get_area to return None
        with patch.object(threshold_entity, "_get_area", return_value=None):
            assert threshold_entity.native_value == 0.0

    async def test_async_set_native_value_validates_minimum(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value raises error for values < 1.0."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
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
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value raises error for values > 99.0."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
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
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value stores percentage value directly (not divided by 100)."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        area = coordinator.get_area(area_name)
        mock_update_config = AsyncMock()
        with patch.object(area.config, "update_config", mock_update_config):
            # Test with valid percentage value (50.0)
            await threshold_entity.async_set_native_value(50.0)

            # Verify update_config was called with percentage value directly (not divided)
            mock_update_config.assert_called_once_with({CONF_THRESHOLD: 50.0})

    async def test_async_set_native_value_stores_boundary_values(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value stores boundary values (1.0 and 99.0) correctly."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        area = coordinator.get_area(area_name)
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
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value handles missing area gracefully."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Mock _get_area to return None
        with patch.object(threshold_entity, "_get_area", return_value=None):
            # Should not raise an error, just log warning and return
            await threshold_entity.async_set_native_value(50.0)
            # No exception should be raised

    async def test_async_set_native_value_triggers_hot_reload(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test that threshold update triggers coordinator hot reload via config entry update."""
        area_name = coordinator.get_area_names()[0]
        handle = coordinator.get_area_handle(area_name)
        threshold_entity = Threshold(area_handle=handle)

        # Ensure config entry options has CONF_AREAS format (required for update_config)
        if CONF_AREAS not in coordinator.config_entry.options:
            # Get current area config from data and copy to options
            areas_list = coordinator.config_entry.data.get(CONF_AREAS, [])
            coordinator.config_entry.options = {
                CONF_AREAS: [area.copy() for area in areas_list]
            }

        # Mock config entry update to verify it's called with properly structured options
        with patch.object(
            hass.config_entries, "async_update_entry"
        ) as mock_update_entry:
            # Set threshold value
            await threshold_entity.async_set_native_value(75.0)

            # Verify config entry was updated
            mock_update_entry.assert_called_once()
            call_args = mock_update_entry.call_args
            assert call_args is not None
            updated_options = call_args[1]["options"]

            # Verify options structure for multi-area format
            if CONF_AREAS in updated_options:
                areas_list = updated_options[CONF_AREAS]
                assert isinstance(areas_list, list)
                # Verify threshold was updated in the correct area
                area = coordinator.get_area(area_name)
                area_id = area.config.area_id if area else None
                if area_id:
                    area_found = False
                    for area_data in areas_list:
                        if area_data.get(CONF_AREA_ID) == area_id:
                            assert area_data.get(CONF_THRESHOLD) == 75.0
                            area_found = True
                            break
                    assert area_found, (
                        "Area should be found and updated in CONF_AREAS list"
                    )
