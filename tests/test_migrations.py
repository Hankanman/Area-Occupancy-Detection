"""Tests for migrations.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.const import (
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_VERSION,
    CONF_VERSION_MINOR,
)
from custom_components.area_occupancy.migrations import (
    async_migrate_entry,
    async_reset_database_if_needed,
)
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant


class TestAsyncMigrateEntry:
    """Test async_migrate_entry function."""

    @pytest.fixture
    def mock_config_entry_v1_0(self, mock_config_entry: Mock) -> Mock:
        """Create a mock config entry at version 1.0."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = mock_config_entry.entry_id
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}
        return entry

    @pytest.fixture
    def mock_config_entry_current(self, mock_config_entry: Mock) -> Mock:
        """Create a mock config entry at current version."""
        entry = Mock(spec=ConfigEntry)
        entry.version = CONF_VERSION
        entry.minor_version = CONF_VERSION_MINOR
        entry.entry_id = mock_config_entry.entry_id
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }
        entry.options = {}
        return entry

    async def test_async_migrate_entry_v1_0_to_current(
        self, hass: HomeAssistant, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration from version 1.0 to current."""
        result = await async_migrate_entry(hass, mock_config_entry_v1_0)
        assert result is True

    async def test_async_migrate_entry_already_current(
        self, hass: HomeAssistant, mock_config_entry_current: Mock
    ) -> None:
        """Test migration when already at current version."""
        result = await async_migrate_entry(hass, mock_config_entry_current)
        assert result is True

    async def test_async_migrate_entry_future_version(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration from future version."""
        mock_entry = Mock(spec=ConfigEntry)
        mock_entry.version = CONF_VERSION + 1
        mock_entry.minor_version = 0
        mock_entry.entry_id = "test_entry_id"
        mock_entry.state = ConfigEntryState.LOADED
        mock_entry.data = {}
        mock_entry.options = {}

        result = await async_migrate_entry(hass, mock_entry)
        assert result is True


class TestAsyncResetDatabaseIfNeeded:
    """Test async_reset_database_if_needed function."""

    async def test_async_reset_database_if_needed_old_version(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database reset when version is old (< 13)."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        # Call with old version
        await async_reset_database_if_needed(hass, 12)

        # Database file should be deleted
        assert not db_path.exists()

    async def test_async_reset_database_if_needed_current_version(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test database reset skipped for version >= 13."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        # Call with current version
        await async_reset_database_if_needed(hass, 13)

        # Database file should still exist (not deleted)
        assert db_path.exists()

    async def test_async_reset_database_if_needed_no_db_file(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test when database file doesn't exist."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()

        # Call with old version but no database file
        await async_reset_database_if_needed(hass, 12)

        # Should complete without error
        # No database file to delete, so nothing happens

    async def test_async_reset_database_if_needed_deletes_wal_files(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test that WAL files are also deleted."""
        # Set up temporary config directory
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        wal_path = storage_dir / "area_occupancy.db-wal"
        shm_path = storage_dir / "area_occupancy.db-shm"
        db_path.write_bytes(b"SQLite format 3")
        wal_path.write_bytes(b"WAL data")
        shm_path.write_bytes(b"SHM data")

        # Call with old version
        await async_reset_database_if_needed(hass, 12)

        # All files should be deleted
        assert not db_path.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()


class TestAsyncMigrateEntryAdditional:
    """Additional tests for async_migrate_entry function."""

    async def test_async_migrate_entry_database_reset(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test migration resets database for old versions."""
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        result = await async_migrate_entry(hass, entry)
        assert result is True
        # Database should be deleted for version < 13
        assert not db_path.exists()

    async def test_async_migrate_entry_no_database_reset_for_current_version(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Test migration does not reset database for current version."""
        hass.config.config_dir = str(tmp_path)
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        entry = Mock(spec=ConfigEntry)
        entry.version = CONF_VERSION
        entry.minor_version = CONF_VERSION_MINOR
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        result = await async_migrate_entry(hass, entry)
        assert result is True
        # Database should still exist for version >= 13
        assert db_path.exists()
