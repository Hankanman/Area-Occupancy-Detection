"""Tests for number module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.number import Threshold, async_setup_entry
from homeassistant.components.number import NumberMode
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError


class TestThreshold:
    """Test Threshold number entity."""

    @pytest.fixture
    def threshold_entity(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> Threshold:
        """Create a threshold entity for testing."""
        area_name = coordinator_with_areas.get_area_names()[0]
        handle = coordinator_with_areas.get_area_handle(area_name)
        return Threshold(area_handle=handle)

    def test_initialization(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test Threshold entity initialization."""
        assert threshold_entity.coordinator is not None
        # unique_id uses entry_id, device_id, and entity_name
        area_name = coordinator_with_areas.get_area_names()[0]
        entry_id = coordinator_with_areas.entry_id
        area = coordinator_with_areas.get_area(area_name)
        device_id = next(iter(area.device_info()["identifiers"]))[1]
        expected_unique_id = f"{entry_id}_{device_id}_occupancy_threshold"
        assert threshold_entity.unique_id == expected_unique_id
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
        coordinator_with_areas: AreaOccupancyCoordinator,
        coordinator_threshold: float,
        expected_percentage: float,
    ) -> None:
        """Test native_value property with various threshold values."""
        # Mock area.threshold method
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area(area_name)
        area.threshold = Mock(return_value=coordinator_threshold)
        assert threshold_entity.native_value == expected_percentage

    @pytest.mark.parametrize(
        "percentage_value",
        [1.0, 25.0, 50.0, 75.0, 99.0],
    )
    async def test_async_set_native_value_valid(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
        percentage_value: float,
    ) -> None:
        """Test setting valid native values."""
        # Access config via area
        area_name = threshold_entity.coordinator.get_area_names()[0]
        area = threshold_entity.coordinator.get_area(area_name)
        with patch.object(area.config, "update_config") as mock_update_config:
            await threshold_entity.async_set_native_value(percentage_value)
            mock_update_config.assert_called_once_with({"threshold": percentage_value})

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
        coordinator_with_areas: AreaOccupancyCoordinator,
        invalid_value: float,
        expected_error: str,
    ) -> None:
        """Test setting invalid native values."""
        area_name = threshold_entity.coordinator.get_area_names()[0]
        area = threshold_entity.coordinator.get_area(area_name)
        with patch.object(area.config, "update_config") as mock_update_config:
            with pytest.raises(ServiceValidationError, match=expected_error):
                await threshold_entity.async_set_native_value(invalid_value)
            mock_update_config.assert_not_called()

    async def test_async_set_native_value_coordinator_error(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test handling coordinator errors."""
        area_name = threshold_entity.coordinator.get_area_names()[0]
        area = threshold_entity.coordinator.get_area(area_name)
        with (
            patch.object(
                area.config, "update_config", side_effect=Exception("Update failed")
            ),
            pytest.raises(Exception, match="Update failed"),
        ):
            await threshold_entity.async_set_native_value(75.0)

    @pytest.mark.parametrize(
        ("last_update_success", "expected_available"),
        [(True, True), (False, False)],
    )
    def test_available_property(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
        last_update_success: bool,
        expected_available: bool,
    ) -> None:
        """Test available property based on coordinator state."""
        threshold_entity.coordinator.last_update_success = last_update_success
        assert threshold_entity.available is expected_available

    def test_device_info_property(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test device_info property."""
        # device_info is now accessed directly from Area
        area_name = threshold_entity.coordinator.get_area_names()[0]
        area = threshold_entity.coordinator.get_area(area_name)
        expected_device_info = area.device_info()
        assert threshold_entity.device_info == expected_device_info

    @pytest.mark.parametrize(
        "percentage_value",
        [33.33, 66.67, 12.5, 87.5],
    )
    async def test_value_conversion_precision(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
        percentage_value: float,
    ) -> None:
        """Test value conversion precision."""
        area_name = threshold_entity.coordinator.get_area_names()[0]
        area = threshold_entity.coordinator.get_area(area_name)
        with patch.object(area.config, "update_config") as mock_update_config:
            await threshold_entity.async_set_native_value(percentage_value)
            called_value = mock_update_config.call_args[0][0]["threshold"]
            assert called_value == percentage_value

    async def test_multiple_threshold_updates(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test multiple threshold updates in sequence."""
        updates = [25.0, 50.0, 75.0, 90.0]

        area_name = threshold_entity.coordinator.get_area_names()[0]
        area = threshold_entity.coordinator.get_area(area_name)
        with patch.object(area.config, "update_config") as mock_update_config:
            for percentage in updates:
                mock_update_config.reset_mock()
                await threshold_entity.async_set_native_value(percentage)
                mock_update_config.assert_called_once_with({"threshold": percentage})

    async def test_error_recovery(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test error recovery scenarios."""
        # Test coordinator error followed by successful update
        area_name = threshold_entity.coordinator.get_area_names()[0]
        area = threshold_entity.coordinator.get_area(area_name)
        with patch.object(
            area.config,
            "update_config",
            side_effect=[
                Exception("Temporary error"),
                None,  # Success on second call
            ],
        ) as mock_update_config:
            # First call should raise exception
            with pytest.raises(Exception, match="Temporary error"):
                await threshold_entity.async_set_native_value(75.0)

            # Second call should succeed
            await threshold_entity.async_set_native_value(75.0)
            assert mock_update_config.call_count == 2
            mock_update_config.assert_called_with({"threshold": 75.0})

    async def test_async_set_native_value_missing_area(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test async_set_native_value when area is missing."""

        # Mock get_area to return None
        with patch.object(threshold_entity.coordinator, "get_area", return_value=None):
            # Should handle gracefully and return without error
            await threshold_entity.async_set_native_value(75.0)

    async def test_async_set_native_value_update_config_error(
        self,
        threshold_entity: Threshold,
        coordinator_with_areas: AreaOccupancyCoordinator,
    ) -> None:
        """Test async_set_native_value when update_config raises error."""
        area_name = threshold_entity.coordinator.get_area_names()[0]
        area = threshold_entity.coordinator.get_area(area_name)

        # Mock update_config to raise error
        with (
            patch.object(
                area.config, "update_config", side_effect=RuntimeError("Update failed")
            ),
            pytest.raises(RuntimeError, match="Update failed"),
        ):
            # Should propagate error
            await threshold_entity.async_set_native_value(75.0)


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
