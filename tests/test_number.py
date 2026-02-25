"""Tests for number module."""

from __future__ import annotations

from asyncio import Lock
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import CONF_THRESHOLD
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.number import Threshold, async_setup_entry
from homeassistant.components.number import NumberMode
from homeassistant.components.sensor import SensorStateClass
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from tests.conftest import create_test_area


# ruff: noqa: SLF001, TID251
@pytest.fixture
def threshold_entity(coordinator: AreaOccupancyCoordinator) -> Threshold:
    """Create a Threshold entity for testing."""
    area_name = coordinator.get_area_names()[0]
    handle = coordinator.get_area_handle(area_name)
    return Threshold(area_handle=handle)


@pytest.fixture
def threshold_entity_with_hass(
    hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
) -> Threshold:
    """Create a Threshold entity with hass assigned for testing."""
    area_name = coordinator.get_area_names()[0]
    handle = coordinator.get_area_handle(area_name)
    entity = Threshold(area_handle=handle)
    entity.hass = hass
    return entity


@pytest.fixture
def setup_config_entry_for_device_registry(
    hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
) -> None:
    """Register config entry in hass.config_entries for device registry tests."""
    if not hasattr(coordinator.config_entry, "setup_lock"):
        coordinator.config_entry.setup_lock = Lock()
    hass.config_entries._entries[coordinator.entry_id] = coordinator.config_entry


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_creates_entities_for_all_areas(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test setup entry creates threshold entities for all areas with correct properties."""
        # Arrange: Create multiple areas
        create_test_area(
            coordinator,
            area_name="Kitchen",
            entity_ids=["binary_sensor.kitchen_motion"],
        )
        create_test_area(
            coordinator,
            area_name="Living Room",
            entity_ids=["binary_sensor.living_motion"],
        )

        mock_config_entry.runtime_data = coordinator
        mock_async_add_entities = Mock()

        # Act: Set up entry
        await async_setup_entry(hass, mock_config_entry, mock_async_add_entities)

        # Assert: Verify entities were created for each area (one call per area)
        area_count = len(coordinator.get_area_names())
        assert mock_async_add_entities.call_count == area_count
        entities = []
        for call_args in mock_async_add_entities.call_args_list:
            entities.extend(call_args[0][0])
        assert len(entities) == area_count

        # Verify all entities are Threshold instances
        for entity in entities:
            assert isinstance(entity, Threshold)
            assert entity.coordinator == coordinator

        # Verify unique_id format for first entity
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        device_id = next(iter(area.device_info()["identifiers"]))[1]
        expected_unique_id = f"{coordinator.entry_id}_{device_id}_occupancy_threshold"
        assert entities[0].unique_id == expected_unique_id


class TestThreshold:
    """Test Threshold entity."""

    def test_initialization_attributes(self, threshold_entity: Threshold) -> None:
        """Test that all initialization attributes are set correctly."""
        # Assert: Verify all __init__ attributes
        assert threshold_entity._attr_has_entity_name is True
        assert threshold_entity._attr_name == "Threshold"
        assert threshold_entity._attr_native_min_value == 1.0
        assert threshold_entity._attr_native_max_value == 99.0
        assert threshold_entity._attr_native_step == 1.0
        assert threshold_entity._attr_mode == NumberMode.BOX
        assert threshold_entity._attr_native_unit_of_measurement == PERCENTAGE
        assert threshold_entity._attr_entity_category == EntityCategory.CONFIG
        assert threshold_entity._attr_state_class == SensorStateClass.MEASUREMENT
        assert threshold_entity._attr_device_info is not None
        assert threshold_entity.unique_id is not None

    @pytest.mark.parametrize(
        ("internal_value", "expected_percentage"),
        [
            (0.0, 0.0),  # Zero case
            (0.5, 50.0),  # Mid value
            (1.0, 100.0),  # Maximum value
            (0.25, 25.0),  # Quarter value
            (0.75, 75.0),  # Three-quarter value
        ],
    )
    def test_native_value_converts_to_percentage(
        self,
        threshold_entity: Threshold,
        coordinator: AreaOccupancyCoordinator,
        internal_value: float,
        expected_percentage: float,
    ) -> None:
        """Test native_value converts internal threshold (0.0-1.0) to percentage (0-100)."""
        # Arrange
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Act & Assert
        with patch.object(area, "threshold", return_value=internal_value):
            assert threshold_entity.native_value == expected_percentage

    def test_native_value_with_missing_area(self, threshold_entity: Threshold) -> None:
        """Test native_value returns 0.0 when area is missing."""
        # Act & Assert
        with patch.object(threshold_entity, "_get_area", return_value=None):
            assert threshold_entity.native_value == 0.0

    def test_get_area_resolves_area_handle(
        self, threshold_entity: Threshold, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test _get_area method resolves area handle correctly."""
        # Arrange
        area_name = coordinator.get_area_names()[0]

        # Act
        area = threshold_entity._get_area()

        # Assert
        assert area is not None
        assert area.area_name == area_name

    def test_get_area_returns_none_when_area_missing(
        self, threshold_entity: Threshold
    ) -> None:
        """Test _get_area returns None when area handle resolves to None."""
        # Act & Assert
        with patch.object(
            threshold_entity._area_handle,
            "resolve",
            return_value=None,
        ):
            assert threshold_entity._get_area() is None

    @pytest.mark.parametrize(
        "invalid_value",
        [
            0.0,  # zero
            0.9,  # just below minimum
            99.1,  # just above maximum
            100.0,  # above maximum
            -1.0,  # negative
        ],
        ids=[
            "zero",
            "just_below_minimum",
            "just_above_maximum",
            "above_maximum",
            "negative",
        ],
    )
    async def test_async_set_native_value_validates_range(
        self,
        threshold_entity: Threshold,
        invalid_value: float,
    ) -> None:
        """Test async_set_native_value raises ServiceValidationError for values outside 1.0-99.0 range."""
        # Act & Assert
        with pytest.raises(ServiceValidationError) as exc_info:
            await threshold_entity.async_set_native_value(invalid_value)
        assert "must be between" in str(exc_info.value)

    async def test_async_set_native_value_stores_percentage_directly(
        self, threshold_entity: Threshold, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test async_set_native_value stores percentage value directly (not divided by 100)."""
        # Arrange
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        mock_update_config = AsyncMock()

        # Act
        with patch.object(area.config, "update_config", mock_update_config):
            await threshold_entity.async_set_native_value(50.0)

        # Assert: Verify update_config was called with percentage value directly (not divided)
        mock_update_config.assert_called_once_with({CONF_THRESHOLD: 50.0})

    async def test_async_set_native_value_with_missing_area(
        self, threshold_entity: Threshold
    ) -> None:
        """Test async_set_native_value handles missing area gracefully without raising exception."""
        # Act & Assert: Should not raise an error, just log warning and return
        with patch.object(threshold_entity, "_get_area", return_value=None):
            await threshold_entity.async_set_native_value(50.0)
            # No exception should be raised

    async def test_async_added_to_hass_device_area_assignment(
        self,
        hass: HomeAssistant,
        threshold_entity_with_hass: Threshold,
        device_registry,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test async_added_to_hass assigns device to Home Assistant area when area_id is configured."""
        # Arrange
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.area_id = "test_area_id"

        # Register config entry in hass.config_entries so device registry can link to it
        if not hasattr(coordinator.config_entry, "setup_lock"):
            coordinator.config_entry.setup_lock = Lock()
        hass.config_entries._entries[coordinator.entry_id] = coordinator.config_entry

        # Create device in registry
        device_entry = device_registry.async_get_or_create(
            config_entry_id=coordinator.entry_id,
            identifiers=threshold_entity_with_hass.device_info["identifiers"],
            name=f"{area_name} Occupancy",
        )
        # Initially device has no area_id
        assert device_entry.area_id is None

        # Act
        with patch(
            "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass"
        ) as mock_parent:
            await threshold_entity_with_hass.async_added_to_hass()
            mock_parent.assert_called_once()

        # Assert: Device should now have area_id assigned
        device_entry = device_registry.async_get_device(
            identifiers=threshold_entity_with_hass.device_info["identifiers"]
        )
        assert device_entry is not None
        assert device_entry.area_id == "test_area_id"

    async def test_async_added_to_hass_no_area_id_configured(
        self,
        threshold_entity_with_hass: Threshold,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test async_added_to_hass does not assign area when area_id is not configured."""
        # Arrange
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.area_id = None  # No area_id configured

        # Act
        with patch(
            "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass"
        ) as mock_parent:
            await threshold_entity_with_hass.async_added_to_hass()
            mock_parent.assert_called_once()

        # Assert: Device area assignment should not have been called
        # (no exception raised, but area_id assignment skipped)

    async def test_async_added_to_hass_device_already_has_area(
        self,
        hass: HomeAssistant,
        threshold_entity_with_hass: Threshold,
        device_registry,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test async_added_to_hass does not update device if it already has correct area_id."""
        # Arrange
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.area_id = "test_area_id"

        # Register config entry in hass.config_entries so device registry can link to it
        if not hasattr(coordinator.config_entry, "setup_lock"):
            coordinator.config_entry.setup_lock = Lock()
        hass.config_entries._entries[coordinator.entry_id] = coordinator.config_entry

        # Create device in registry with area_id already set
        device_entry = device_registry.async_get_or_create(
            config_entry_id=coordinator.entry_id,
            identifiers=threshold_entity_with_hass.device_info["identifiers"],
            name=f"{area_name} Occupancy",
        )
        device_registry.async_update_device(device_entry.id, area_id="test_area_id")
        # Re-fetch device to get updated area_id
        device_entry = device_registry.async_get_device(
            identifiers=threshold_entity_with_hass.device_info["identifiers"]
        )
        assert device_entry is not None
        assert device_entry.area_id == "test_area_id"

        # Mock update_device to verify it's not called
        mock_update_device = patch.object(
            device_registry, "async_update_device"
        ).start()

        # Act
        with patch(
            "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass"
        ):
            await threshold_entity_with_hass.async_added_to_hass()

        # Assert: update_device should not be called since area_id matches
        mock_update_device.assert_not_called()
        patch.stopall()

    async def test_async_added_to_hass_device_not_in_registry(
        self,
        threshold_entity_with_hass: Threshold,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        """Test async_added_to_hass handles gracefully when device doesn't exist in registry."""
        # Arrange
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.area_id = "test_area_id"

        # Mock device registry to return None (device doesn't exist)
        mock_registry = Mock()
        mock_registry.async_get_device.return_value = None

        # Act & Assert: Should handle gracefully without crashing
        with (
            patch(
                "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass"
            ),
            patch(
                "custom_components.area_occupancy.number.dr.async_get",
                return_value=mock_registry,
            ),
        ):
            await threshold_entity_with_hass.async_added_to_hass()
            # No exception should be raised
