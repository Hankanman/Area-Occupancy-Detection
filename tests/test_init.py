"""Tests for __init__.py module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import text

from custom_components.area_occupancy import (
    _async_entry_updated,
    async_reload_entry,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.area_occupancy.const import CONF_VERSION, DOMAIN as DOMAIN_CONST
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
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
        hass: HomeAssistant,
        mock_config_entry: Mock,
        failure_type: str,
        exception_message: str,
    ) -> None:
        """Test various setup failure scenarios."""
        # Ensure hass.data[DOMAIN] doesn't exist initially
        if DOMAIN_CONST in hass.data:
            del hass.data[DOMAIN_CONST]

        if failure_type == "coordinator_init":
            with (
                patch(
                    "custom_components.area_occupancy.AreaOccupancyCoordinator",
                    side_effect=Exception(exception_message),
                ),
                pytest.raises(ConfigEntryNotReady),
            ):
                await async_setup_entry(hass, mock_config_entry)

        elif failure_type == "refresh":
            with patch(
                "custom_components.area_occupancy.AreaOccupancyCoordinator"
            ) as mock_coordinator_class:
                mock_coordinator = Mock()
                mock_coordinator.async_config_entry_first_refresh = AsyncMock(
                    side_effect=Exception(exception_message)
                )
                # Mock get_area_names to return a list (needed for logging at end of setup)
                mock_coordinator.get_area_names = Mock(return_value=["Test Area"])
                mock_coordinator_class.return_value = mock_coordinator

                with pytest.raises(ConfigEntryNotReady):
                    await async_setup_entry(hass, mock_config_entry)

        elif failure_type == "migration":
            object.__setattr__(mock_config_entry, "version", CONF_VERSION - 1)
            with (
                patch(
                    "custom_components.area_occupancy.__init__.async_migrate_entry",
                    side_effect=Exception(exception_message),
                ),
                pytest.raises(ConfigEntryNotReady),
            ):
                await async_setup_entry(hass, mock_config_entry)

    async def test_async_setup_entry_success(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test successful setup flow with real database initialization."""
        # Ensure hass.data[DOMAIN] doesn't exist initially
        if DOMAIN_CONST in hass.data:
            del hass.data[DOMAIN_CONST]

        # Use real coordinator to test actual database initialization
        coordinator = AreaOccupancyCoordinator(hass, mock_config_entry)
        # Mock get_area_names to return a list
        coordinator.get_area_names = Mock(return_value=["Test Area"])

        with (
            patch.object(
                coordinator, "async_config_entry_first_refresh", new=AsyncMock()
            ),
            patch(
                "custom_components.area_occupancy.AreaOccupancyCoordinator",
                return_value=coordinator,
            ) as mock_coord,
            patch(
                "custom_components.area_occupancy.async_setup_services", AsyncMock()
            ) as mock_services,
            patch.object(
                hass.config_entries, "async_forward_entry_setups", new=AsyncMock()
            ),
        ):
            result = await async_setup_entry(hass, mock_config_entry)

        assert result is True
        mock_coord.assert_called_once_with(hass, mock_config_entry)

        # Verify database initialization was called and completed successfully
        # Since we're in test environment with AREA_OCCUPANCY_AUTO_INIT_DB=1,
        # the database should already be initialized in the coordinator's __init__
        assert coordinator.db is not None

        # Verify the database file exists and is accessible
        assert coordinator.db.db_path is not None

        # Verify coordinator is stored in hass.data[DOMAIN]
        assert hass.data[DOMAIN_CONST] == coordinator
        assert coordinator.db.db_path.exists()

        # Verify we can create a session without errors (indicates tables exist)
        try:
            with coordinator.db.get_session() as session:
                # Simple query to verify database is functional
                result = session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                )
                tables = result.fetchall()
                # Should have at least one table (areas, entities, etc.)
                assert len(tables) > 0
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"Database initialization failed - cannot query database: {e}")

        mock_services.assert_awaited_once()
        # Coordinator is now stored in hass.data[DOMAIN] instead of runtime_data
        assert hass.data[DOMAIN_CONST] == coordinator

    async def test_async_setup_entry_migration_updates_version(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test that entry version is updated after successful migration."""
        # Ensure hass.data[DOMAIN] doesn't exist initially
        if DOMAIN_CONST in hass.data:
            del hass.data[DOMAIN_CONST]

        # Set entry to an old version to trigger migration
        old_version = CONF_VERSION - 1
        object.__setattr__(mock_config_entry, "version", old_version)

        # Use real coordinator to test actual database initialization
        coordinator = AreaOccupancyCoordinator(hass, mock_config_entry)
        # Mock get_area_names to return a list
        coordinator.get_area_names = Mock(return_value=["Test Area"])

        # Mock async_update_entry to verify it's called
        mock_update_entry = Mock()

        with (
            patch.object(
                coordinator, "async_config_entry_first_refresh", new=AsyncMock()
            ),
            patch(
                "custom_components.area_occupancy.AreaOccupancyCoordinator",
                return_value=coordinator,
            ),
            patch("custom_components.area_occupancy.async_setup_services", AsyncMock()),
            patch.object(
                hass.config_entries, "async_forward_entry_setups", new=AsyncMock()
            ),
            patch.object(
                hass.config_entries,
                "async_update_entry",
                new=mock_update_entry,
            ),
        ):
            result = await async_setup_entry(hass, mock_config_entry)

        assert result is True
        # Verify async_update_entry was called with the correct version
        mock_update_entry.assert_called_once_with(
            mock_config_entry, version=CONF_VERSION
        )


class TestAsyncSetup:
    """Test async_setup function."""

    async def test_async_setup_success(self, hass: HomeAssistant) -> None:
        """Test successful setup."""
        config: dict[str, Any] = {}

        result = await async_setup(hass, config)

        assert result is True


class TestAsyncReloadEntry:
    """Test async_reload_entry function."""

    async def test_async_reload_entry_success(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test successful reload of config entry."""
        # Mock async_reload since hass is now a real instance
        hass.config_entries.async_reload = AsyncMock(return_value=None)
        result = await async_reload_entry(hass, mock_config_entry)
        # Accept None as valid if that's the contract, otherwise expect True
        assert result is None or result is True
        hass.config_entries.async_reload.assert_called_once_with(
            mock_config_entry.entry_id
        )


class TestAsyncUnloadEntry:
    """Test async_unload_entry function."""

    def _setup_coordinator_mock(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> Mock:
        """Set up coordinator mock with common configuration."""
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        mock_config_entry.runtime_data = mock_coordinator
        return mock_coordinator

    async def test_async_unload_entry_success(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test successful unload of config entry."""
        mock_coordinator = self._setup_coordinator_mock(hass, mock_config_entry)
        # Coordinator is now stored directly in hass.data[DOMAIN], not as a dict
        hass.data[DOMAIN_CONST] = mock_coordinator
        # Mock async_unload_platforms since hass is now a real instance
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

        result = await async_unload_entry(hass, mock_config_entry)

        assert result is True
        mock_coordinator.async_shutdown.assert_called_once()
        hass.config_entries.async_unload_platforms.assert_called_once()

    async def test_async_unload_entry_platform_unload_failure(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test unload when platform unload fails."""
        mock_coordinator = self._setup_coordinator_mock(hass, mock_config_entry)
        # Coordinator is now stored directly in hass.data[DOMAIN], not as a dict
        hass.data[DOMAIN_CONST] = mock_coordinator
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=False)

        result = await async_unload_entry(hass, mock_config_entry)

        assert result is False
        # Do not expect async_shutdown to be called if unload_ok is False

    async def test_async_unload_entry_no_coordinator(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test unload when coordinator doesn't exist."""
        # Coordinator is now stored directly in hass.data[DOMAIN], not as a dict
        # So we need to ensure it doesn't exist or is None
        if DOMAIN_CONST in hass.data:
            del hass.data[DOMAIN_CONST]
        # Mock async_unload_platforms since hass is now a real instance
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

        result = await async_unload_entry(hass, mock_config_entry)

        assert result is True
        hass.config_entries.async_unload_platforms.assert_called_once()


class TestEntryUpdated:
    """Test _async_entry_updated function."""

    def _setup_coordinator_mock(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> Mock:
        """Set up coordinator mock with common configuration."""
        mock_coordinator = Mock()
        mock_coordinator.async_update_options = AsyncMock()
        mock_coordinator.async_refresh = AsyncMock()
        mock_config_entry.runtime_data = mock_coordinator
        object.__setattr__(mock_config_entry, "options", {})
        return mock_coordinator

    async def test_async_entry_updated_success(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test successful entry update."""
        mock_coordinator = self._setup_coordinator_mock(hass, mock_config_entry)
        # Coordinator is now stored directly in hass.data[DOMAIN], not as a dict
        hass.data[DOMAIN_CONST] = mock_coordinator

        await _async_entry_updated(hass, mock_config_entry)

        mock_coordinator.async_update_options.assert_called_once_with(
            mock_config_entry.options
        )
        mock_coordinator.async_refresh.assert_called_once()

    async def test_async_entry_updated_no_coordinator(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test entry update when coordinator doesn't exist."""
        # Coordinator is now stored directly in hass.data[DOMAIN], not as a dict
        # Ensure it doesn't exist
        if DOMAIN_CONST in hass.data:
            del hass.data[DOMAIN_CONST]
        mock_coordinator = self._setup_coordinator_mock(hass, mock_config_entry)

        # Should not raise an exception
        await _async_entry_updated(hass, mock_config_entry)

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
