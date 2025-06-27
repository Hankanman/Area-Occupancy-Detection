"""Tests for __init__.py module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy import (
    async_reload_entry,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.area_occupancy.const import CONF_VERSION, DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady


# ruff: noqa: PLC0415
class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    async def test_async_setup_entry_coordinator_init_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup failure during coordinator initialization."""
        with (
            patch(
                "custom_components.area_occupancy.__init__.AreaOccupancyCoordinator",
                side_effect=Exception("Init failed"),
            ),
            pytest.raises(ConfigEntryNotReady),
        ):
            await async_setup_entry(mock_hass, mock_config_entry)

    async def test_async_setup_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful setup flow."""
        coordinator = Mock()
        coordinator.async_config_entry_first_refresh = AsyncMock()
        with (
            patch(
                "custom_components.area_occupancy.AreaOccupancyCoordinator",
                return_value=coordinator,
            ) as mock_coord,
            patch(
                "custom_components.area_occupancy.async_setup_services", AsyncMock()
            ) as mock_services,
        ):
            result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_coord.assert_called_once_with(mock_hass, mock_config_entry)
        mock_services.assert_awaited_once()
        assert mock_config_entry.runtime_data == coordinator

    async def test_async_setup_entry_refresh_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test setup failure during first refresh."""
        with patch(
            "custom_components.area_occupancy.__init__.AreaOccupancyCoordinator"
        ) as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.async_config_entry_first_refresh = AsyncMock(
                side_effect=Exception("Refresh failed")
            )
            mock_coordinator_class.return_value = mock_coordinator

            with pytest.raises(ConfigEntryNotReady):
                await async_setup_entry(mock_hass, mock_config_entry)

    async def test_async_setup_entry_migration_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test migration failure during setup."""
        mock_config_entry.version = CONF_VERSION - 1  # Ensure migration is needed

        with (
            patch(
                "custom_components.area_occupancy.__init__.async_migrate_entry",
                side_effect=Exception("Migration failed"),
            ),
            pytest.raises(ConfigEntryNotReady),
        ):
            await async_setup_entry(mock_hass, mock_config_entry)


class TestAsyncSetup:
    """Test async_setup function."""

    async def test_async_setup_success(self) -> None:
        """Test successful setup."""
        mock_hass = Mock(spec=HomeAssistant)
        config = {}

        result = await async_setup(mock_hass, config)

        assert result is True


class TestAsyncReloadEntry:
    """Test async_reload_entry function."""

    async def test_async_reload_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful reload of config entry."""
        result = await async_reload_entry(mock_hass, mock_config_entry)
        # Accept None as valid if that's the contract, otherwise expect True
        assert result is None or result is True
        mock_hass.config_entries.async_reload.assert_called_once_with(
            mock_config_entry.entry_id
        )


class TestAsyncUnloadEntry:
    """Test async_unload_entry function."""

    async def test_async_unload_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful unload of config entry."""
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator
        # Set runtime_data to the coordinator mock
        mock_config_entry.runtime_data = mock_coordinator
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
        # Set runtime_data to the coordinator mock
        mock_config_entry.runtime_data = mock_coordinator
        result = await async_unload_entry(mock_hass, mock_config_entry)
        assert result is False
        # Do not expect async_shutdown to be called if unload_ok is False

    async def test_async_unload_entry_no_coordinator(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test unload when coordinator doesn't exist."""
        mock_hass.data[DOMAIN] = {}
        # Set runtime_data to a mock with async_shutdown
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        mock_config_entry.runtime_data = mock_coordinator
        result = await async_unload_entry(mock_hass, mock_config_entry)
        assert result is True
        mock_hass.config_entries.async_unload_platforms.assert_called_once()


class TestEntryUpdated:
    """Test _async_entry_updated function."""

    async def test_async_entry_updated_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful entry update."""
        mock_coordinator = Mock()
        mock_coordinator.async_update_options = AsyncMock()
        mock_coordinator.async_refresh = AsyncMock()
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator
        # Set runtime_data to the coordinator mock
        mock_config_entry.runtime_data = mock_coordinator
        mock_config_entry.options = {}  # Ensure options attribute exists
        from custom_components.area_occupancy import _async_entry_updated

        await _async_entry_updated(mock_hass, mock_config_entry)
        mock_coordinator.async_update_options.assert_called_once_with(
            mock_config_entry.options
        )
        mock_coordinator.async_refresh.assert_called_once()

    async def test_async_entry_updated_no_coordinator(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test entry update when coordinator doesn't exist."""
        mock_hass.data[DOMAIN] = {}
        # Set runtime_data to a mock with async_update_options
        mock_coordinator = Mock()
        mock_coordinator.async_update_options = AsyncMock()
        mock_coordinator.async_refresh = AsyncMock()
        mock_config_entry.runtime_data = mock_coordinator
        mock_config_entry.options = {}  # Ensure options attribute exists
        from custom_components.area_occupancy import _async_entry_updated

        # Should not raise an exception
        await _async_entry_updated(mock_hass, mock_config_entry)
        mock_coordinator.async_update_options.assert_called_once_with(
            mock_config_entry.options
        )
        mock_coordinator.async_refresh.assert_called_once()


class TestAsyncRemoveEntry:
    """Tests for async_remove_entry function."""

    async def test_remove_entry_with_runtime_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Ensure stored runtime data is used."""
        store = Mock(async_remove=AsyncMock())
        mock_config_entry.runtime_data = Mock(store=store)
        from custom_components.area_occupancy import async_remove_entry

        await async_remove_entry(mock_hass, mock_config_entry)
        store.async_remove.assert_awaited_once()

    async def test_remove_entry_without_runtime_data(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Ensure a temporary coordinator is used when runtime_data missing."""
        store = Mock(async_remove=AsyncMock())
        with patch(
            "custom_components.area_occupancy.AreaOccupancyCoordinator",
            return_value=Mock(store=store),
        ) as mock_coord:
            mock_config_entry.runtime_data = None
            from custom_components.area_occupancy import async_remove_entry

            await async_remove_entry(mock_hass, mock_config_entry)
            mock_coord.assert_called_once_with(mock_hass, mock_config_entry)
            store.async_remove.assert_awaited_once()

    async def test_remove_entry_handles_error(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Errors from the store should be logged but not raised."""
        store = Mock(async_remove=AsyncMock(side_effect=Exception("fail")))
        mock_config_entry.runtime_data = Mock(store=store)
        from custom_components.area_occupancy import async_remove_entry

        await async_remove_entry(mock_hass, mock_config_entry)
        store.async_remove.assert_awaited_once()
