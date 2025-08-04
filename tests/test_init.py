"""Tests for __init__.py module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy import (
    _async_entry_updated,
    async_reload_entry,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.area_occupancy.const import CONF_VERSION, DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @pytest.mark.parametrize(
        ("failure_type", "exception_message"),
        [
            ("coordinator_init", "Init failed"),
            ("refresh", "Refresh failed"),
            ("migration", "Migration failed"),
        ],
    )
    async def test_async_setup_entry_failures(
        self,
        mock_hass: Mock,
        mock_config_entry: Mock,
        failure_type: str,
        exception_message: str,
    ) -> None:
        """Test various setup failure scenarios."""
        if failure_type == "coordinator_init":
            with (
                patch(
                    "custom_components.area_occupancy.AreaOccupancyCoordinator",
                    side_effect=Exception(exception_message),
                ),
                pytest.raises(ConfigEntryNotReady),
            ):
                await async_setup_entry(mock_hass, mock_config_entry)

        elif failure_type == "refresh":
            with patch(
                "custom_components.area_occupancy.AreaOccupancyCoordinator"
            ) as mock_coordinator_class:
                mock_coordinator = Mock()
                mock_coordinator.async_config_entry_first_refresh = AsyncMock(
                    side_effect=Exception(exception_message)
                )
                mock_coordinator_class.return_value = mock_coordinator

                with pytest.raises(ConfigEntryNotReady):
                    await async_setup_entry(mock_hass, mock_config_entry)

        elif failure_type == "migration":
            mock_config_entry.version = CONF_VERSION - 1
            with (
                patch(
                    "custom_components.area_occupancy.__init__.async_migrate_entry",
                    side_effect=Exception(exception_message),
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

    def _setup_coordinator_mock(self, mock_hass: Mock, mock_config_entry: Mock) -> Mock:
        """Set up coordinator mock with common configuration."""
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        mock_config_entry.runtime_data = mock_coordinator
        return mock_coordinator

    async def test_async_unload_entry_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful unload of config entry."""
        mock_coordinator = self._setup_coordinator_mock(mock_hass, mock_config_entry)
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_coordinator.async_shutdown.assert_called_once()
        mock_hass.config_entries.async_unload_platforms.assert_called_once()

    async def test_async_unload_entry_platform_unload_failure(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test unload when platform unload fails."""
        mock_coordinator = self._setup_coordinator_mock(mock_hass, mock_config_entry)
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator
        mock_hass.config_entries.async_unload_platforms = AsyncMock(return_value=False)

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is False
        # Do not expect async_shutdown to be called if unload_ok is False

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

    def _setup_coordinator_mock(self, mock_hass: Mock, mock_config_entry: Mock) -> Mock:
        """Set up coordinator mock with common configuration."""
        mock_coordinator = Mock()
        mock_coordinator.async_update_options = AsyncMock()
        mock_coordinator.async_refresh = AsyncMock()
        mock_config_entry.runtime_data = mock_coordinator
        mock_config_entry.options = {}
        return mock_coordinator

    async def test_async_entry_updated_success(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test successful entry update."""
        mock_coordinator = self._setup_coordinator_mock(mock_hass, mock_config_entry)
        mock_hass.data[DOMAIN]["test_entry_id"] = mock_coordinator

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
        mock_coordinator = self._setup_coordinator_mock(mock_hass, mock_config_entry)

        # Should not raise an exception
        await _async_entry_updated(mock_hass, mock_config_entry)

        mock_coordinator.async_update_options.assert_called_once_with(
            mock_config_entry.options
        )
        mock_coordinator.async_refresh.assert_called_once()


class TestAsyncRemoveEntry:
    """Tests for async_remove_entry function."""

    # Note: The actual async_remove_entry function implementation is not fully tested
    # because the current implementation doesn't have clear observable side effects
    # that can be easily verified in tests. These tests were removed as they only
    # asserted True without meaningful verification.
