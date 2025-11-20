"""Tests for number module."""

from __future__ import annotations

from unittest.mock import Mock

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.number import Threshold, async_setup_entry
from homeassistant.core import HomeAssistant


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
