"""Tests for __init__.py module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.area_occupancy import async_setup_entry, async_unload_entry
from custom_components.area_occupancy.const import DOMAIN, PLATFORMS


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @pytest.fixture
    def mock_config_entry(self) -> MockConfigEntry:
        """Create a mock config entry."""
        return MockConfigEntry(
            domain=DOMAIN,
            data={
                "name": "Test Area",
                "motion_sensors": ["binary_sensor.motion"],
                "primary_occupancy_sensor": "binary_sensor.motion",
            },
            options={},
            entry_id="test_entry_id",
            title="Test Area",
        )

    async def test_setup_entry_success(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test successful config entry setup."""
        with patch("custom_components.area_occupancy.AreaOccupancyCoordinator") as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.async_setup = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator
            
            # Mock platform setup
            with patch.object(hass.config_entries, "async_forward_entry_setups") as mock_forward:
                mock_forward.return_value = True
                
                # Add config entry to hass
                mock_config_entry.add_to_hass(hass)
                
                result = await async_setup_entry(hass, mock_config_entry)
                
                assert result is True
                
                # Verify coordinator was created and set up
                mock_coordinator_class.assert_called_once_with(hass, mock_config_entry)
                mock_coordinator.async_setup.assert_called_once()
                
                # Verify platforms were set up
                mock_forward.assert_called_once_with(mock_config_entry, PLATFORMS)
                
                # Verify coordinator was stored in hass data
                assert DOMAIN in hass.data
                assert mock_config_entry.entry_id in hass.data[DOMAIN]

    async def test_setup_entry_coordinator_setup_failure(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test config entry setup when coordinator setup fails."""
        with patch("custom_components.area_occupancy.AreaOccupancyCoordinator") as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.async_setup = AsyncMock(
                side_effect=ConfigEntryNotReady("Setup failed")
            )
            mock_coordinator_class.return_value = mock_coordinator
            
            mock_config_entry.add_to_hass(hass)
            
            with pytest.raises(ConfigEntryNotReady):
                await async_setup_entry(hass, mock_config_entry)

    async def test_setup_entry_platform_setup_failure(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test config entry setup when platform setup fails."""
        with patch("custom_components.area_occupancy.AreaOccupancyCoordinator") as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.async_setup = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator
            
            # Mock platform setup failure
            with patch.object(hass.config_entries, "async_forward_entry_setups") as mock_forward:
                mock_forward.side_effect = Exception("Platform setup failed")
                
                mock_config_entry.add_to_hass(hass)
                
                result = await async_setup_entry(hass, mock_config_entry)
                
                # Should return False on platform setup failure
                assert result is False

    async def test_setup_entry_with_existing_data(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test config entry setup when hass data already exists."""
        # Pre-populate hass data
        hass.data[DOMAIN] = {}
        
        with patch("custom_components.area_occupancy.AreaOccupancyCoordinator") as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.async_setup = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator
            
            with patch.object(hass.config_entries, "async_forward_entry_setups") as mock_forward:
                mock_forward.return_value = True
                
                mock_config_entry.add_to_hass(hass)
                
                result = await async_setup_entry(hass, mock_config_entry)
                
                assert result is True
                assert mock_config_entry.entry_id in hass.data[DOMAIN]


class TestAsyncUnloadEntry:
    """Test async_unload_entry function."""

    @pytest.fixture
    def mock_config_entry(self) -> MockConfigEntry:
        """Create a mock config entry."""
        return MockConfigEntry(
            domain=DOMAIN,
            data={
                "name": "Test Area",
                "motion_sensors": ["binary_sensor.motion"],
            },
            options={},
            entry_id="test_entry_id",
            title="Test Area",
        )

    async def test_unload_entry_success(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test successful config entry unload."""
        # Set up hass data with coordinator
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        
        hass.data[DOMAIN] = {
            mock_config_entry.entry_id: mock_coordinator
        }
        
        # Mock platform unload
        with patch.object(hass.config_entries, "async_unload_platforms") as mock_unload:
            mock_unload.return_value = True
            
            mock_config_entry.add_to_hass(hass)
            
            result = await async_unload_entry(hass, mock_config_entry)
            
            assert result is True
            
            # Verify platforms were unloaded
            mock_unload.assert_called_once_with(mock_config_entry, PLATFORMS)
            
            # Verify coordinator was shut down
            mock_coordinator.async_shutdown.assert_called_once()
            
            # Verify coordinator was removed from hass data
            assert mock_config_entry.entry_id not in hass.data[DOMAIN]

    async def test_unload_entry_platform_unload_failure(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test config entry unload when platform unload fails."""
        # Set up hass data with coordinator
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        
        hass.data[DOMAIN] = {
            mock_config_entry.entry_id: mock_coordinator
        }
        
        # Mock platform unload failure
        with patch.object(hass.config_entries, "async_unload_platforms") as mock_unload:
            mock_unload.return_value = False
            
            mock_config_entry.add_to_hass(hass)
            
            result = await async_unload_entry(hass, mock_config_entry)
            
            assert result is False
            
            # Coordinator should not be shut down if platform unload fails
            mock_coordinator.async_shutdown.assert_not_called()

    async def test_unload_entry_no_coordinator(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test config entry unload when no coordinator exists."""
        # Set up empty hass data
        hass.data[DOMAIN] = {}
        
        # Mock platform unload
        with patch.object(hass.config_entries, "async_unload_platforms") as mock_unload:
            mock_unload.return_value = True
            
            mock_config_entry.add_to_hass(hass)
            
            result = await async_unload_entry(hass, mock_config_entry)
            
            assert result is True
            
            # Verify platforms were still unloaded
            mock_unload.assert_called_once_with(mock_config_entry, PLATFORMS)

    async def test_unload_entry_coordinator_shutdown_error(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test config entry unload when coordinator shutdown fails."""
        # Set up hass data with coordinator
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock(side_effect=Exception("Shutdown failed"))
        
        hass.data[DOMAIN] = {
            mock_config_entry.entry_id: mock_coordinator
        }
        
        # Mock platform unload
        with patch.object(hass.config_entries, "async_unload_platforms") as mock_unload:
            mock_unload.return_value = True
            
            mock_config_entry.add_to_hass(hass)
            
            # Should still succeed even if coordinator shutdown fails
            result = await async_unload_entry(hass, mock_config_entry)
            
            assert result is True
            
            # Coordinator should still be removed from hass data
            assert mock_config_entry.entry_id not in hass.data[DOMAIN]

    async def test_unload_entry_no_domain_data(
        self, 
        hass: HomeAssistant, 
        mock_config_entry: MockConfigEntry
    ) -> None:
        """Test config entry unload when domain data doesn't exist."""
        # No hass data for domain
        
        # Mock platform unload
        with patch.object(hass.config_entries, "async_unload_platforms") as mock_unload:
            mock_unload.return_value = True
            
            mock_config_entry.add_to_hass(hass)
            
            result = await async_unload_entry(hass, mock_config_entry)
            
            assert result is True
            
            # Verify platforms were still unloaded
            mock_unload.assert_called_once_with(mock_config_entry, PLATFORMS)