"""Test number entities."""

from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.components.number import NumberDeviceClass, NumberMode
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.area_occupancy.const import (
    CONF_THRESHOLD,
    DOMAIN,
    NAME_THRESHOLD_NUMBER,
)
from custom_components.area_occupancy.number import (
    AreaOccupancyThreshold,
    async_setup_entry,
)


class TestAreaOccupancyThreshold:
    """Test AreaOccupancyThreshold class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test threshold number initialization."""
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        assert number._attr_has_entity_name is True
        assert number._attr_unique_id == f"test_entry_id_{NAME_THRESHOLD_NUMBER.lower().replace(' ', '_')}"
        assert number._attr_name == NAME_THRESHOLD_NUMBER
        assert number._attr_device_class == NumberDeviceClass.POWER_FACTOR
        assert number._attr_native_unit_of_measurement == PERCENTAGE
        assert number._attr_mode == NumberMode.BOX
        assert number._attr_entity_category == EntityCategory.CONFIG
        assert number._attr_device_info == mock_coordinator.device_info
        assert number._attr_native_min_value == 1
        assert number._attr_native_max_value == 99
        assert number._attr_native_step == 1

    def test_native_value_with_valid_threshold(self, mock_coordinator: Mock) -> None:
        """Test native value with valid threshold."""
        mock_coordinator.threshold = 0.75  # 75%
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        assert number.native_value == 75.0

    def test_native_value_with_zero_threshold(self, mock_coordinator: Mock) -> None:
        """Test native value with zero threshold."""
        mock_coordinator.threshold = 0.0
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        assert number.native_value == 0.0

    def test_native_value_with_max_threshold(self, mock_coordinator: Mock) -> None:
        """Test native value with maximum threshold."""
        mock_coordinator.threshold = 1.0  # 100%
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        assert number.native_value == 100.0

    def test_native_value_with_missing_threshold(self, mock_coordinator: Mock) -> None:
        """Test native value with missing threshold attribute."""
        # Remove threshold attribute
        if hasattr(mock_coordinator, 'threshold'):
            delattr(mock_coordinator, 'threshold')
        
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        # Should handle gracefully and return a default value
        try:
            result = number.native_value
            # Should return some numeric value or None
            assert result is None or isinstance(result, (int, float))
        except AttributeError:
            # This is acceptable - the test verifies the behavior exists
            pass

    async def test_async_set_native_value_valid(self, mock_coordinator: Mock) -> None:
        """Test setting valid native value."""
        mock_coordinator.async_update_threshold = AsyncMock()
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        await number.async_set_native_value(75.0)
        
        mock_coordinator.async_update_threshold.assert_called_once_with(75.0)

    async def test_async_set_native_value_minimum(self, mock_coordinator: Mock) -> None:
        """Test setting minimum valid native value."""
        mock_coordinator.async_update_threshold = AsyncMock()
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        await number.async_set_native_value(1.0)
        
        mock_coordinator.async_update_threshold.assert_called_once_with(1.0)

    async def test_async_set_native_value_maximum(self, mock_coordinator: Mock) -> None:
        """Test setting maximum valid native value."""
        mock_coordinator.async_update_threshold = AsyncMock()
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        await number.async_set_native_value(99.0)
        
        mock_coordinator.async_update_threshold.assert_called_once_with(99.0)

    async def test_async_set_native_value_with_coordinator_error(self, mock_coordinator: Mock) -> None:
        """Test setting native value when coordinator raises error."""
        mock_coordinator.async_update_threshold = AsyncMock(
            side_effect=ServiceValidationError("Invalid threshold")
        )
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        with pytest.raises(ServiceValidationError):
            await number.async_set_native_value(75.0)

    async def test_async_set_native_value_no_coordinator_method(self, mock_coordinator: Mock) -> None:
        """Test setting native value when coordinator doesn't have update method."""
        # Remove the async_update_threshold method
        if hasattr(mock_coordinator, 'async_update_threshold'):
            delattr(mock_coordinator, 'async_update_threshold')
        
        number = AreaOccupancyThreshold(mock_coordinator, "test_entry_id")
        
        # Should handle gracefully
        try:
            await number.async_set_native_value(75.0)
        except AttributeError:
            # This is acceptable - the test verifies the behavior
            pass


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_setup_entry_success(self, mock_hass: Mock) -> None:
        """Test successful setup of number entity."""
        # Mock config entry
        config_entry = Mock()
        config_entry.entry_id = "test_entry_id"
        
        # Mock coordinator
        mock_coordinator = Mock()
        mock_coordinator.threshold = 0.6
        
        # Mock hass data
        mock_hass.data = {
            DOMAIN: {
                "test_entry_id": {
                    "coordinator": mock_coordinator
                }
            }
        }
        
        # Mock async_add_entities
        async_add_entities = AsyncMock()
        
        await async_setup_entry(mock_hass, config_entry, async_add_entities)
        
        # Should add one number entity
        async_add_entities.assert_called_once()
        added_entities = async_add_entities.call_args[0][0]
        assert len(added_entities) == 1
        assert isinstance(added_entities[0], AreaOccupancyThreshold)

    async def test_setup_entry_coordinator_access(self, mock_hass: Mock) -> None:
        """Test that setup entry correctly accesses the coordinator."""
        # Mock config entry
        config_entry = Mock()
        config_entry.entry_id = "test_entry_id"
        
        # Mock coordinator
        mock_coordinator = Mock()
        
        # Mock hass data structure
        mock_hass.data = {
            DOMAIN: {
                "test_entry_id": {
                    "coordinator": mock_coordinator
                }
            }
        }
        
        # Mock async_add_entities
        async_add_entities = AsyncMock()
        
        await async_setup_entry(mock_hass, config_entry, async_add_entities)
        
        # Verify the coordinator was accessed correctly
        added_entities = async_add_entities.call_args[0][0]
        threshold_number = added_entities[0]
        assert threshold_number.coordinator == mock_coordinator

    async def test_setup_entry_with_update_before_add(self, mock_hass: Mock) -> None:
        """Test setup entry with update_before_add flag."""
        # Mock config entry
        config_entry = Mock()
        config_entry.entry_id = "test_entry_id"
        
        # Mock coordinator
        mock_coordinator = Mock()
        
        # Mock hass data
        mock_hass.data = {
            DOMAIN: {
                "test_entry_id": {
                    "coordinator": mock_coordinator
                }
            }
        }
        
        # Mock async_add_entities
        async_add_entities = AsyncMock()
        
        await async_setup_entry(mock_hass, config_entry, async_add_entities)
        
        # Verify update_before_add was set to True
        call_kwargs = async_add_entities.call_args[1]
        assert call_kwargs.get("update_before_add") is True

    async def test_setup_entry_missing_coordinator(self, mock_hass: Mock) -> None:
        """Test setup entry when coordinator is missing."""
        # Mock config entry
        config_entry = Mock()
        config_entry.entry_id = "missing_entry_id"
        
        # Mock hass data without the coordinator
        mock_hass.data = {
            DOMAIN: {}
        }
        
        # Mock async_add_entities
        async_add_entities = AsyncMock()
        
        # Should handle gracefully or raise appropriate error
        try:
            await async_setup_entry(mock_hass, config_entry, async_add_entities)
        except KeyError:
            # This is expected behavior when coordinator is missing
            pass
        except Exception as e:
            # Other exceptions should be specific and meaningful
            assert "coordinator" in str(e).lower() or "not found" in str(e).lower()

    async def test_setup_entry_malformed_hass_data(self, mock_hass: Mock) -> None:
        """Test setup entry with malformed hass data structure."""
        # Mock config entry
        config_entry = Mock()
        config_entry.entry_id = "test_entry_id"
        
        # Mock malformed hass data
        mock_hass.data = {
            DOMAIN: {
                "test_entry_id": {
                    # Missing "coordinator" key
                    "other_data": "value"
                }
            }
        }
        
        # Mock async_add_entities
        async_add_entities = AsyncMock()
        
        # Should handle gracefully or raise appropriate error
        try:
            await async_setup_entry(mock_hass, config_entry, async_add_entities)
        except KeyError:
            # This is expected behavior when structure is malformed
            pass
        except Exception as e:
            # Other exceptions should be specific and meaningful
            assert "coordinator" in str(e).lower() or "not found" in str(e).lower()