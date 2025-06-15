"""Tests for __init__.py module."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from custom_components.area_occupancy import (
    async_setup_entry,
    async_setup,
    async_remove_entry,
    async_reload_entry,
    async_unload_entry,
)
from custom_components.area_occupancy.const import DOMAIN


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.data = {}
        hass.async_create_task = Mock()
        hass.config_entries = Mock()
        hass.config_entries.async_forward_entry_setups = AsyncMock()
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry_id"
        entry.data = {"name": "Test Area"}
        entry.runtime_data = None
        entry.add_update_listener = Mock()
        return entry

    async def test_async_setup_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful setup of config entry."""
        with patch(
            "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator"
        ) as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.async_config_entry_first_refresh = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            result = await async_setup_entry(mock_hass, mock_config_entry)

            assert result is True
            assert DOMAIN in mock_hass.data
            assert mock_config_entry.entry_id in mock_hass.data[DOMAIN]
            mock_coordinator.async_config_entry_first_refresh.assert_called_once()
            mock_hass.config_entries.async_forward_entry_setups.assert_called_once()

    async def test_async_setup_entry_coordinator_init_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup failure during coordinator initialization."""
        with patch(
            "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator",
            side_effect=Exception("Init failed"),
        ):
            with pytest.raises(ConfigEntryNotReady):
                await async_setup_entry(mock_hass, mock_config_entry)

    async def test_async_setup_entry_refresh_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup failure during first refresh."""
        with patch(
            "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator"
        ) as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.async_config_entry_first_refresh = AsyncMock(
                side_effect=Exception("Refresh failed")
            )
            mock_coordinator_class.return_value = mock_coordinator

            with pytest.raises(ConfigEntryNotReady):
                await async_setup_entry(mock_hass, mock_config_entry)


class TestAsyncSetup:
    """Test async_setup function."""

    async def test_async_setup_success(self) -> None:
        """Test successful setup."""
        mock_hass = Mock(spec=HomeAssistant)
        config = {}

        result = await async_setup(mock_hass, config)

        assert result is True


class TestAsyncRemoveEntry:
    """Test async_remove_entry function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.data = {DOMAIN: {"test_entry_id": Mock()}}
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry_id"
        return entry

    async def test_async_remove_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful removal of config entry."""
        mock_coordinator = Mock()
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator

        with patch(
            "custom_components.area_occupancy.storage.StorageManager"
        ) as mock_storage_class:
            mock_storage = Mock()
            mock_storage.async_remove_instance = AsyncMock()
            mock_storage_class.return_value = mock_storage

            await async_remove_entry(mock_hass, mock_config_entry)

            mock_storage.async_remove_instance.assert_called_once_with("test_entry_id")

    async def test_async_remove_entry_no_coordinator(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test removal when coordinator doesn't exist."""
        mock_hass.data[DOMAIN] = {}

        # Should not raise an exception
        await async_remove_entry(mock_hass, mock_config_entry)


class TestAsyncReloadEntry:
    """Test async_reload_entry function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.config_entries = Mock()
        hass.config_entries.async_reload = AsyncMock(return_value=True)
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry_id"
        return entry

    async def test_async_reload_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful reload of config entry."""
        result = await async_reload_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_hass.config_entries.async_reload.assert_called_once_with(
            mock_config_entry.entry_id
        )


class TestAsyncUnloadEntry:
    """Test async_unload_entry function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.data = {DOMAIN: {"test_entry_id": Mock()}}
        hass.config_entries = Mock()
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry_id"
        return entry

    async def test_async_unload_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful unload of config entry."""
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_coordinator.async_shutdown.assert_called_once()
        mock_hass.config_entries.async_unload_platforms.assert_called_once()

    async def test_async_unload_entry_platform_unload_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test unload when platform unload fails."""
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator
        mock_hass.config_entries.async_unload_platforms = AsyncMock(return_value=False)

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is False
        mock_coordinator.async_shutdown.assert_called_once()

    async def test_async_unload_entry_no_coordinator(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test unload when coordinator doesn't exist."""
        mock_hass.data[DOMAIN] = {}

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_hass.config_entries.async_unload_platforms.assert_called_once()


class TestEntryUpdated:
    """Test _async_entry_updated function."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.data = {DOMAIN: {"test_entry_id": Mock()}}
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.entry_id = "test_entry_id"
        return entry

    async def test_async_entry_updated_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful entry update."""
        mock_coordinator = Mock()
        mock_coordinator.async_update_options = AsyncMock()
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator

        from custom_components.area_occupancy import _async_entry_updated

        await _async_entry_updated(mock_hass, mock_config_entry)

        mock_coordinator.async_update_options.assert_called_once_with(
            mock_config_entry.options
        )

    async def test_async_entry_updated_no_coordinator(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test entry update when coordinator doesn't exist."""
        mock_hass.data[DOMAIN] = {}

        from custom_components.area_occupancy import _async_entry_updated

        # Should not raise an exception
        await _async_entry_updated(mock_hass, mock_config_entry) 