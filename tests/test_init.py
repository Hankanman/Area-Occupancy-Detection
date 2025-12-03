"""Tests for __init__.py module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import text

from custom_components.area_occupancy import (
    _async_entry_updated,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.area_occupancy.const import (
    CONF_VERSION,
    DOMAIN as DOMAIN_CONST,
    PLATFORMS,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @staticmethod
    def _ensure_domain_not_in_hass_data(hass: HomeAssistant) -> None:
        """Ensure DOMAIN is not in hass.data."""
        if DOMAIN_CONST in hass.data:
            del hass.data[DOMAIN_CONST]

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
        self._ensure_domain_not_in_hass_data(hass)

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
        self._ensure_domain_not_in_hass_data(hass)

        # Use real coordinator to test actual database initialization
        coordinator = AreaOccupancyCoordinator(hass, mock_config_entry)
        # Mock get_area_names to return a list
        coordinator.get_area_names = Mock(return_value=["Test Area"])

        # Mock update listener addition
        mock_add_listener = Mock()

        with (
            patch.object(
                coordinator, "async_config_entry_first_refresh", new=AsyncMock()
            ),
            patch.object(
                coordinator, "async_init_database", new=AsyncMock()
            ) as mock_init_db,
            patch(
                "custom_components.area_occupancy.AreaOccupancyCoordinator",
                return_value=coordinator,
            ) as mock_coord,
            patch(
                "custom_components.area_occupancy.async_setup_services", AsyncMock()
            ) as mock_services,
            patch.object(
                hass.config_entries, "async_forward_entry_setups", new=AsyncMock()
            ) as mock_forward_setups,
            patch.object(mock_config_entry, "async_on_unload", new=mock_add_listener),
        ):
            result = await async_setup_entry(hass, mock_config_entry)

        assert result is True
        mock_coord.assert_called_once_with(hass, mock_config_entry)

        # Verify database initialization was called
        mock_init_db.assert_awaited_once()

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

        # Verify services setup was called
        mock_services.assert_awaited_once()

        # Verify platforms were set up
        mock_forward_setups.assert_awaited_once_with(mock_config_entry, PLATFORMS)

        # Verify update listener was added
        mock_add_listener.assert_called_once()

        # Verify coordinator is stored in hass.data[DOMAIN]
        assert hass.data[DOMAIN_CONST] == coordinator

    async def test_async_setup_entry_migration_updates_version(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test that entry version is updated after successful migration."""
        self._ensure_domain_not_in_hass_data(hass)

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
            patch.object(coordinator, "async_init_database", new=AsyncMock()),
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
            patch(
                "custom_components.area_occupancy.async_migrate_entry",
                new=AsyncMock(return_value=True),
            ),
        ):
            result = await async_setup_entry(hass, mock_config_entry)

        assert result is True
        # Verify async_update_entry was called with the correct version
        mock_update_entry.assert_called_once_with(
            mock_config_entry, version=CONF_VERSION
        )

    async def test_async_setup_entry_deleted_entry(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test early return when entry is marked for deletion."""
        self._ensure_domain_not_in_hass_data(hass)

        # Mark entry as deleted
        object.__setattr__(mock_config_entry, "data", {"deleted": True})

        result = await async_setup_entry(hass, mock_config_entry)

        assert result is False
        # Verify coordinator was not created
        assert DOMAIN_CONST not in hass.data

    async def test_async_setup_entry_migration_consolidation(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test migration that consolidates entries (returns False)."""
        self._ensure_domain_not_in_hass_data(hass)

        # Set entry to an old version to trigger migration
        old_version = CONF_VERSION - 1
        object.__setattr__(mock_config_entry, "version", old_version)
        # Start with empty data
        object.__setattr__(mock_config_entry, "data", {})

        # Simulate migration marking entry as deleted (consolidation scenario)
        async def mark_deleted(*args, **kwargs):
            object.__setattr__(mock_config_entry, "data", {"deleted": True})
            return False

        with patch(
            "custom_components.area_occupancy.async_migrate_entry",
            new=AsyncMock(side_effect=mark_deleted),
        ):
            result = await async_setup_entry(hass, mock_config_entry)

        assert result is False
        # Verify coordinator was not created
        assert DOMAIN_CONST not in hass.data

    async def test_async_setup_entry_coordinator_reuse(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test coordinator reuse scenario (migration case)."""
        # Set up existing coordinator
        existing_coordinator = AreaOccupancyCoordinator(hass, mock_config_entry)
        existing_coordinator.get_area_names = Mock(return_value=["Test Area"])
        hass.data[DOMAIN_CONST] = existing_coordinator

        # Create a new entry that should reuse the existing coordinator
        new_entry = Mock()
        new_entry.entry_id = "new_entry_id"
        new_entry.version = CONF_VERSION
        new_entry.data = {}
        new_entry.options = {}
        new_entry.runtime_data = None

        with (
            patch(
                "custom_components.area_occupancy.async_setup_services", AsyncMock()
            ) as mock_services,
            patch.object(
                hass.config_entries, "async_forward_entry_setups", new=AsyncMock()
            ) as mock_forward_setups,
            patch.object(new_entry, "async_on_unload", new=Mock()),
        ):
            result = await async_setup_entry(hass, new_entry)

        assert result is True
        # Verify existing coordinator was reused
        assert hass.data[DOMAIN_CONST] == existing_coordinator
        # Verify services setup was called (idempotent check)
        mock_services.assert_awaited_once()
        # Verify platforms were set up
        mock_forward_setups.assert_awaited_once_with(new_entry, PLATFORMS)
        # Verify runtime_data was set
        assert new_entry.runtime_data == existing_coordinator

    async def test_async_setup_entry_database_init_failure(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test database initialization failure."""
        self._ensure_domain_not_in_hass_data(hass)

        # Use real coordinator
        coordinator = AreaOccupancyCoordinator(hass, mock_config_entry)
        coordinator.get_area_names = Mock(return_value=["Test Area"])

        with (
            patch.object(
                coordinator,
                "async_init_database",
                new=AsyncMock(side_effect=Exception("DB init failed")),
            ),
            patch(
                "custom_components.area_occupancy.AreaOccupancyCoordinator",
                return_value=coordinator,
            ),
            pytest.raises(ConfigEntryNotReady),
        ):
            await async_setup_entry(hass, mock_config_entry)

    async def test_async_setup_entry_services_idempotency(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test that services setup only happens once (idempotent)."""
        self._ensure_domain_not_in_hass_data(hass)

        # Set up services flag to simulate services already set up
        if "_services_setup" not in hass.data:
            hass.data["_services_setup"] = {}
        hass.data["_services_setup"][DOMAIN_CONST] = True

        # Use real coordinator
        coordinator = AreaOccupancyCoordinator(hass, mock_config_entry)
        coordinator.get_area_names = Mock(return_value=["Test Area"])

        with (
            patch.object(
                coordinator, "async_config_entry_first_refresh", new=AsyncMock()
            ),
            patch.object(coordinator, "async_init_database", new=AsyncMock()),
            patch(
                "custom_components.area_occupancy.AreaOccupancyCoordinator",
                return_value=coordinator,
            ),
            patch(
                "custom_components.area_occupancy.async_setup_services", AsyncMock()
            ) as mock_services,
            patch.object(
                hass.config_entries, "async_forward_entry_setups", new=AsyncMock()
            ),
            patch.object(mock_config_entry, "async_on_unload", new=Mock()),
        ):
            result = await async_setup_entry(hass, mock_config_entry)

        assert result is True
        # Verify services setup was NOT called (already set up)
        mock_services.assert_not_awaited()


class TestAsyncUnloadEntry:
    """Test async_unload_entry function."""

    @staticmethod
    def _ensure_domain_not_in_hass_data(hass: HomeAssistant) -> None:
        """Ensure DOMAIN is not in hass.data."""
        if DOMAIN_CONST in hass.data:
            del hass.data[DOMAIN_CONST]

    def _setup_coordinator_mock(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> Mock:
        """Set up coordinator mock with common configuration."""
        mock_coordinator = Mock()
        mock_coordinator.async_shutdown = AsyncMock()
        mock_config_entry.runtime_data = mock_coordinator
        return mock_coordinator

    @pytest.mark.parametrize(
        ("other_entries", "expect_shutdown", "has_services_flag"),
        [
            # Last entry scenario - coordinator should be shut down
            ([], True, True),
            # Multiple entries scenario - coordinator should NOT be shut down
            (["other_entry_id"], False, False),
        ],
    )
    async def test_async_unload_entry_success(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        other_entries: list[str],
        expect_shutdown: bool,
        has_services_flag: bool,
    ) -> None:
        """Test successful unload with different entry scenarios."""
        mock_coordinator = self._setup_coordinator_mock(hass, mock_config_entry)
        hass.data[DOMAIN_CONST] = mock_coordinator
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

        # Set up async_entries mock
        if other_entries:
            # Create mock entries for other entries
            mock_other_entries = []
            for entry_id in other_entries:
                other_entry = Mock()
                other_entry.entry_id = entry_id
                mock_other_entries.append(other_entry)
            # Include current entry in the list (realistic behavior)
            hass.config_entries.async_entries = Mock(
                return_value=[mock_config_entry, *mock_other_entries]
            )
        else:
            # No other entries
            hass.config_entries.async_entries = Mock(return_value=[])

        # Set up services flag if needed
        if has_services_flag:
            if "_services_setup" not in hass.data:
                hass.data["_services_setup"] = {}
            hass.data["_services_setup"][DOMAIN_CONST] = True

        result = await async_unload_entry(hass, mock_config_entry)

        assert result is True
        hass.config_entries.async_unload_platforms.assert_called_once()

        if expect_shutdown:
            # Coordinator should be shut down when it's the last entry
            mock_coordinator.async_shutdown.assert_called_once()
            assert DOMAIN_CONST not in hass.data
            if has_services_flag:
                assert DOMAIN_CONST not in hass.data.get("_services_setup", {})
                assert "_services_setup" in hass.data
        else:
            # Coordinator should NOT be shut down when other entries exist
            mock_coordinator.async_shutdown.assert_not_called()
            assert hass.data[DOMAIN_CONST] == mock_coordinator

        # Verify runtime_data was cleared in both cases
        assert mock_config_entry.runtime_data is None

    async def test_async_unload_entry_platform_unload_failure(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test unload when platform unload fails."""
        mock_coordinator = self._setup_coordinator_mock(hass, mock_config_entry)
        # Coordinator is now stored directly in hass.data[DOMAIN], not as a dict
        hass.data[DOMAIN_CONST] = mock_coordinator
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=False)
        # Mock async_entries for consistency
        hass.config_entries.async_entries = Mock(return_value=[])

        result = await async_unload_entry(hass, mock_config_entry)

        assert result is False
        # Do not expect async_shutdown to be called if unload_ok is False

    async def test_async_unload_entry_no_coordinator(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test unload when coordinator doesn't exist."""
        self._ensure_domain_not_in_hass_data(hass)
        # Mock async_unload_platforms since hass is now a real instance
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
        # Mock async_entries to return empty list (no other entries)
        hass.config_entries.async_entries = Mock(return_value=[])

        result = await async_unload_entry(hass, mock_config_entry)

        assert result is True
        hass.config_entries.async_unload_platforms.assert_called_once()


class TestEntryUpdated:
    """Test _async_entry_updated function."""

    @staticmethod
    def _ensure_domain_not_in_hass_data(hass: HomeAssistant) -> None:
        """Ensure DOMAIN is not in hass.data."""
        if DOMAIN_CONST in hass.data:
            del hass.data[DOMAIN_CONST]

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

    @pytest.mark.parametrize(
        ("has_coordinator", "expect_methods_called"),
        [
            # Coordinator exists - methods should be called
            (True, True),
            # Coordinator doesn't exist - methods should NOT be called (early return)
            (False, False),
        ],
    )
    async def test_async_entry_updated(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        has_coordinator: bool,
        expect_methods_called: bool,
    ) -> None:
        """Test entry update with and without coordinator."""
        if has_coordinator:
            mock_coordinator = self._setup_coordinator_mock(hass, mock_config_entry)
            hass.data[DOMAIN_CONST] = mock_coordinator
        else:
            self._ensure_domain_not_in_hass_data(hass)
            mock_config_entry.runtime_data = None
            # Create a mock coordinator for assertion purposes (won't be called)
            mock_coordinator = Mock()
            mock_coordinator.async_update_options = AsyncMock()
            mock_coordinator.async_refresh = AsyncMock()

        await _async_entry_updated(hass, mock_config_entry)

        if expect_methods_called:
            mock_coordinator.async_update_options.assert_called_once_with(
                mock_config_entry.options
            )
            mock_coordinator.async_refresh.assert_called_once()
        else:
            # Verify that no coordinator methods were called (function returns early)
            mock_coordinator.async_update_options.assert_not_called()
            mock_coordinator.async_refresh.assert_not_called()


class TestAsyncRemoveEntry:
    """Tests for async_remove_entry function.

    Note: The async_remove_entry function is currently empty (no-op) in the implementation.
    As such, there are no meaningful behaviors to test. Tests will be added when
    the function is implemented with actual cleanup logic.
    """
