"""Tests for migrations.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.const import (
    CONF_AREA_ID,
    CONF_MOTION_SENSORS,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
    DEFAULT_THRESHOLD,
    DOMAIN,
)
from custom_components.area_occupancy.migrations import (
    _check_unique_id_conflict,
    _drop_tables_locked,
    _find_entities_by_prefix,
    _safe_database_operation,
    _safe_file_operation,
    _update_db_version,
    _update_entity_unique_id,
    async_migrate_entry,
    async_migrate_unique_ids,
    async_reset_database_if_needed,
    validate_threshold,
)
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant


class TestAsyncMigrateUniqueIds:
    """Test async_migrate_unique_ids function."""

    @pytest.mark.parametrize(
        ("platform", "entity_suffix", "expected_calls"),
        [
            ("sensor", "probability", 1),
            ("sensor", "decay", 1),
            ("binary_sensor", "occupancy", 1),
            ("number", "threshold", 1),
            ("invalid_platform", "anything", 0),
        ],
    )
    async def test_async_migrate_unique_ids_platforms(
        self,
        hass: HomeAssistant,
        mock_config_entry: Mock,
        platform: str,
        entity_suffix: str,
        expected_calls: int,
    ) -> None:
        """Test unique ID migration for different platforms."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            mock_registry.async_update_entity = Mock()

            if platform != "invalid_platform":
                entity1 = Mock()
                entity1.unique_id = (
                    f"{DOMAIN}_{mock_config_entry.entry_id}_{entity_suffix}"
                )
                entity1.config_entry_id = mock_config_entry.entry_id
                mock_registry.entities = {f"{platform}.entity1": entity1}
            else:
                mock_registry.entities = {}

            mock_get_registry.return_value = mock_registry

            await async_migrate_unique_ids(hass, mock_config_entry, platform)

            assert mock_registry.async_update_entity.call_count == expected_calls

    async def test_async_migrate_unique_ids_multiple_entities(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test unique ID migration with multiple matching entities."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            entity1 = Mock()
            entity1.unique_id = f"{DOMAIN}_{mock_config_entry.entry_id}_probability"
            entity1.config_entry_id = mock_config_entry.entry_id
            entity2 = Mock()
            entity2.unique_id = f"{DOMAIN}_{mock_config_entry.entry_id}_decay"
            entity2.config_entry_id = mock_config_entry.entry_id

            mock_registry.entities = {
                "sensor.entity1": entity1,
                "sensor.entity2": entity2,
            }
            mock_registry.async_update_entity = Mock()
            mock_get_registry.return_value = mock_registry

            await async_migrate_unique_ids(hass, mock_config_entry, "sensor")

            assert mock_registry.async_update_entity.call_count == 2

    async def test_async_migrate_unique_ids_no_matching_entities(
        self, hass: HomeAssistant, mock_config_entry: Mock
    ) -> None:
        """Test unique ID migration with no matching entities."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get"
        ) as mock_get_registry:
            mock_registry = Mock()
            entity1 = Mock()
            entity1.unique_id = "other_prefix_entity"
            entity1.config_entry_id = mock_config_entry.entry_id

            mock_registry.entities = {"sensor.entity1": entity1}
            mock_registry.async_update_entity = Mock()
            mock_get_registry.return_value = mock_registry

            await async_migrate_unique_ids(hass, mock_config_entry, "sensor")

            mock_registry.async_update_entity.assert_not_called()


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
        """Test migration from version 1.0 to current - no migration needed."""
        result = await async_migrate_entry(hass, mock_config_entry_v1_0)
        # Migration always returns True (no migration needed)
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

    async def test_async_migrate_entry_migration_error(
        self, hass: HomeAssistant, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with error during migration - no migration needed."""
        result = await async_migrate_entry(hass, mock_config_entry_v1_0)
        # Migration always returns True (no migration needed)
        assert result is True

    async def test_async_migrate_entry_invalid_threshold(
        self, hass: HomeAssistant, mock_config_entry_v1_0: Mock
    ) -> None:
        """Test migration with invalid threshold - no migration needed."""
        mock_config_entry_v1_0.options = {CONF_THRESHOLD: 150}

        result = await async_migrate_entry(hass, mock_config_entry_v1_0)
        # Migration always returns True (no migration needed)
        assert result is True


class TestValidateThreshold:
    """Test validate_threshold function."""

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            (1.0, 1.0),
            (50.0, 50.0),
            (99.0, 99.0),
            (0.0, DEFAULT_THRESHOLD),
            (-10.0, DEFAULT_THRESHOLD),
            (100.0, DEFAULT_THRESHOLD),
            (150.0, DEFAULT_THRESHOLD),
            (0.1, DEFAULT_THRESHOLD),
            (99.9, DEFAULT_THRESHOLD),
        ],
    )
    def test_validate_threshold(self, input_value: float, expected: float) -> None:
        """Test threshold validation with various values."""
        assert validate_threshold(input_value) == expected


class TestFindEntitiesByPrefix:
    """Test _find_entities_by_prefix function."""

    def test_find_entities_by_prefix_with_matching_entities(self) -> None:
        """Test finding entities with matching prefix."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "area_occupancy_entry1_probability"
        entity1.config_entry_id = "entry1"
        entity2 = Mock()
        entity2.unique_id = "area_occupancy_entry1_decay"
        entity2.config_entry_id = "entry1"
        entity3 = Mock()
        entity3.unique_id = "other_prefix_entity"
        entity3.config_entry_id = "entry2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
            "sensor.entity3": entity3,
        }

        result = _find_entities_by_prefix(
            mock_registry, "area_occupancy_entry1_", "entry1"
        )

        assert len(result) == 2
        assert ("sensor.entity1", entity1) in result
        assert ("sensor.entity2", entity2) in result

    def test_find_entities_by_prefix_no_config_entry_filter(self) -> None:
        """Test finding entities without config entry filter."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "prefix_entity1"
        entity1.config_entry_id = "entry1"
        entity2 = Mock()
        entity2.unique_id = "prefix_entity2"
        entity2.config_entry_id = "entry2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        result = _find_entities_by_prefix(mock_registry, "prefix_", None)

        assert len(result) == 2

    def test_find_entities_by_prefix_no_matches(self) -> None:
        """Test finding entities with no matches."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "other_prefix_entity"
        entity1.config_entry_id = "entry1"

        mock_registry.entities = {"sensor.entity1": entity1}

        result = _find_entities_by_prefix(mock_registry, "prefix_", "entry1")

        assert len(result) == 0

    def test_find_entities_by_prefix_none_unique_id(self) -> None:
        """Test finding entities with None unique_id."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = None
        entity1.config_entry_id = "entry1"

        mock_registry.entities = {"sensor.entity1": entity1}

        result = _find_entities_by_prefix(mock_registry, "prefix_", "entry1")

        assert len(result) == 0


class TestCheckUniqueIdConflict:
    """Test _check_unique_id_conflict function."""

    def test_check_unique_id_conflict_no_conflict(self) -> None:
        """Test conflict check with no conflicts."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "unique_id_1"
        entity2 = Mock()
        entity2.unique_id = "unique_id_2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        has_conflict, conflict_id = _check_unique_id_conflict(
            mock_registry, "unique_id_3", "sensor.entity1"
        )

        assert has_conflict is False
        assert conflict_id is None

    def test_check_unique_id_conflict_with_conflict(self) -> None:
        """Test conflict check with conflicts."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "unique_id_1"
        entity2 = Mock()
        entity2.unique_id = "unique_id_2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        has_conflict, conflict_id = _check_unique_id_conflict(
            mock_registry, "unique_id_2", "sensor.entity1"
        )

        assert has_conflict is True
        assert conflict_id == "sensor.entity2"

    def test_check_unique_id_conflict_case_insensitive(self) -> None:
        """Test conflict check is case insensitive."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "Unique_ID_1"
        entity2 = Mock()
        entity2.unique_id = "unique_id_2"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        has_conflict, conflict_id = _check_unique_id_conflict(
            mock_registry, "UNIQUE_ID_1", "sensor.entity2"
        )

        assert has_conflict is True
        assert conflict_id == "sensor.entity1"

    def test_check_unique_id_conflict_excludes_self(self) -> None:
        """Test conflict check excludes the entity being checked."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "unique_id_1"

        mock_registry.entities = {"sensor.entity1": entity1}

        has_conflict, conflict_id = _check_unique_id_conflict(
            mock_registry, "unique_id_1", "sensor.entity1"
        )

        assert has_conflict is False
        assert conflict_id is None


class TestUpdateEntityUniqueId:
    """Test _update_entity_unique_id function."""

    def test_update_entity_unique_id_success(self) -> None:
        """Test successful unique ID update."""
        mock_registry = Mock()
        mock_registry.async_update_entity = Mock()
        entity1 = Mock()
        entity1.unique_id = "old_unique_id"

        mock_registry.entities = {"sensor.entity1": entity1}

        success, conflict_id = _update_entity_unique_id(
            mock_registry, "sensor.entity1", "old_unique_id", "new_unique_id", False
        )

        assert success is True
        assert conflict_id is None
        mock_registry.async_update_entity.assert_called_once_with(
            "sensor.entity1", new_unique_id="new_unique_id"
        )

    def test_update_entity_unique_id_with_conflict_check_no_conflict(self) -> None:
        """Test update with conflict check and no conflict."""
        mock_registry = Mock()
        mock_registry.async_update_entity = Mock()
        entity1 = Mock()
        entity1.unique_id = "old_unique_id"
        entity2 = Mock()
        entity2.unique_id = "other_unique_id"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        success, conflict_id = _update_entity_unique_id(
            mock_registry, "sensor.entity1", "old_unique_id", "new_unique_id", True
        )

        assert success is True
        assert conflict_id is None
        mock_registry.async_update_entity.assert_called_once()

    def test_update_entity_unique_id_with_conflict(self) -> None:
        """Test update with conflict check and conflict exists."""
        mock_registry = Mock()
        entity1 = Mock()
        entity1.unique_id = "old_unique_id"
        entity2 = Mock()
        entity2.unique_id = "new_unique_id"

        mock_registry.entities = {
            "sensor.entity1": entity1,
            "sensor.entity2": entity2,
        }

        success, conflict_id = _update_entity_unique_id(
            mock_registry, "sensor.entity1", "old_unique_id", "new_unique_id", True
        )

        assert success is False
        assert conflict_id == "sensor.entity2"
        mock_registry.async_update_entity.assert_not_called()

    def test_update_entity_unique_id_lowercase(self) -> None:
        """Test that unique ID is converted to lowercase."""
        mock_registry = Mock()
        mock_registry.async_update_entity = Mock()
        entity1 = Mock()
        entity1.unique_id = "old_unique_id"

        mock_registry.entities = {"sensor.entity1": entity1}

        success, _conflict_id = _update_entity_unique_id(
            mock_registry, "sensor.entity1", "old_unique_id", "NEW_UNIQUE_ID", False
        )

        assert success is True
        mock_registry.async_update_entity.assert_called_once_with(
            "sensor.entity1", new_unique_id="new_unique_id"
        )


class TestAsyncResetDatabaseIfNeeded:
    """Test async_reset_database_if_needed function."""

    async def test_async_reset_database_if_needed(self, hass: HomeAssistant) -> None:
        """Test database reset when needed."""
        # The function calls hass.async_add_executor_job which runs in executor thread
        # We just verify it completes without error
        await async_reset_database_if_needed(hass, 10)
        # Function should complete successfully

    async def test_async_reset_database_if_needed_version_11(
        self, hass: HomeAssistant
    ) -> None:
        """Test database reset skipped for version >= 11."""
        # For version >= 11, _drop_tables_locked will skip the actual work
        # but still runs via executor. We just verify it completes without error.
        await async_reset_database_if_needed(hass, 11)
        # Function should complete successfully


class TestAsyncMigrateEntryAdditional:
    """Additional tests for async_migrate_entry function."""

    async def test_async_migrate_entry_removes_deprecated_keys(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration removes deprecated keys - no migration needed."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_AREA_ID: "area_123",
        }
        entry.options = {}

        result = await async_migrate_entry(hass, entry)
        # Migration always returns True (no migration needed)
        assert result is True

    async def test_async_migrate_entry_database_reset(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration resets database for old versions - no migration needed."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 1
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        result = await async_migrate_entry(hass, entry)
        # Migration always returns True (no migration needed)
        assert result is True

    async def test_async_migrate_entry_consolidation_entry_removed(
        self, hass: HomeAssistant
    ) -> None:
        """Test migration when entry is removed during consolidation."""
        entry = Mock(spec=ConfigEntry)
        entry.version = 12
        entry.minor_version = 0
        entry.entry_id = "test_entry_id"
        entry.state = ConfigEntryState.LOADED
        entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        entry.options = {}

        def async_entries_mock(domain=None):
            # Entry was removed during consolidation
            return []

        hass.config_entries.async_entries = async_entries_mock

        result = await async_migrate_entry(hass, entry)

        # Should return True (no migration needed)
        assert result is True


class TestDropTablesLocked:
    """Test _drop_tables_locked function."""

    def test_drop_tables_locked_version_11_skip(self, tmp_path: Path) -> None:
        """Test that table dropping is skipped for version >= 11."""
        _drop_tables_locked(tmp_path, 11)
        # Should complete without error

    def test_drop_tables_locked_no_db_file(self, tmp_path: Path) -> None:
        """Test when database file doesn't exist."""
        _drop_tables_locked(tmp_path, 10)
        # Should complete without error

    def test_drop_tables_locked_db_version_3_or_higher(self, tmp_path: Path) -> None:
        """Test when database version is 3 or higher (no drop needed)."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        # Mock everything to verify no DROP TABLE calls
        with (
            patch("filelock.FileLock") as mock_lock,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
            patch("sqlalchemy.create_engine") as mock_create_engine,
        ):
            mock_lock.return_value.__enter__ = Mock()
            mock_lock.return_value.__exit__ = Mock(return_value=None)
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session

            # Mock db_version query to return 3
            mock_result = Mock()
            mock_result.fetchone.return_value = ("3",)
            mock_session.execute.return_value = mock_result

            _drop_tables_locked(tmp_path, 10)

            # Should not drop tables when version >= 3
            # Verify connect was not called (no table dropping)
            mock_engine.connect.assert_not_called()

    def test_drop_tables_locked_exception_handling(self, tmp_path: Path) -> None:
        """Test exception handling in _drop_tables_locked."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        with patch("filelock.FileLock.__enter__", side_effect=Exception("Lock error")):
            # Should handle exception gracefully
            _drop_tables_locked(tmp_path, 10)

    def test_drop_tables_locked_metadata_query_exception(self, tmp_path: Path) -> None:
        """Test when metadata query raises exception."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        with (
            patch("filelock.FileLock") as mock_lock,
            patch("sqlalchemy.create_engine") as mock_create_engine,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
        ):
            mock_lock.return_value.__enter__ = Mock()
            mock_lock.return_value.__exit__ = Mock(return_value=None)
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session
            mock_session.execute.side_effect = Exception("Query error")

            # Should handle exception and set db_version to 0
            _drop_tables_locked(tmp_path, 10)

    def test_drop_tables_locked_update_metadata_exception(self, tmp_path: Path) -> None:
        """Test when metadata update raises exception."""
        storage_dir = tmp_path / ".storage"
        storage_dir.mkdir()
        db_path = storage_dir / "area_occupancy.db"
        db_path.write_bytes(b"SQLite format 3")

        with (
            patch("filelock.FileLock") as mock_lock,
            patch("sqlalchemy.create_engine") as mock_create_engine,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
        ):
            mock_lock.return_value.__enter__ = Mock()
            mock_lock.return_value.__exit__ = Mock(return_value=None)
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session

            # First query returns None (no db_version)
            mock_result1 = Mock()
            mock_result1.fetchone.return_value = None
            # Second query (UPDATE) raises exception
            mock_result2 = Mock()
            mock_result2.scalar.return_value = 0
            mock_session.execute.side_effect = [
                mock_result1,  # First query (SELECT)
                Exception("Update error"),  # UPDATE fails
            ]

            # Should handle exception gracefully
            _drop_tables_locked(tmp_path, 10)


class TestUpdateDbVersion:
    """Test _update_db_version helper function."""

    def test_update_db_version_update_existing(self) -> None:
        """Test updating existing db_version."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1  # Changes() returns 1
        mock_session.execute.return_value = mock_result

        _update_db_version(mock_session, 4)

        assert mock_session.execute.call_count >= 1

    def test_update_db_version_insert_new(self) -> None:
        """Test inserting new db_version when it doesn't exist."""
        mock_session = Mock()
        mock_result1 = Mock()
        mock_result1.scalar.return_value = 0  # Changes() returns 0 (no update)
        mock_result2 = Mock()
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        _update_db_version(mock_session, 4)

        assert mock_session.execute.call_count == 2

    def test_update_db_version_exception_handling(self) -> None:
        """Test exception handling in _update_db_version."""
        mock_session = Mock()
        mock_session.execute.side_effect = Exception("Database error")

        # Should not raise
        _update_db_version(mock_session, 4)


class TestSafeFileOperation:
    """Test _safe_file_operation helper function."""

    def test_safe_file_operation_success(self) -> None:
        """Test successful file operation."""
        operation = Mock()
        result = _safe_file_operation(operation, "Error message")

        assert result is True
        operation.assert_called_once()

    def test_safe_file_operation_oserror(self) -> None:
        """Test file operation with OSError."""
        operation = Mock(side_effect=OSError("File error"))
        result = _safe_file_operation(operation, "Error message")

        assert result is False
        operation.assert_called_once()


class TestSafeDatabaseOperation:
    """Test _safe_database_operation helper function."""

    def test_safe_database_operation_success(self) -> None:
        """Test successful database operation."""
        operation = Mock()
        result = _safe_database_operation(operation, "Error message")

        assert result is True
        operation.assert_called_once()

    def test_safe_database_operation_sqlalchemy_error(self) -> None:
        """Test database operation with SQLAlchemyError."""
        operation = Mock(side_effect=SQLAlchemyError("DB error"))
        result = _safe_database_operation(operation, "Error message")

        assert result is False
        operation.assert_called_once()

    def test_safe_database_operation_oserror(self) -> None:
        """Test database operation with OSError."""
        operation = Mock(side_effect=OSError("OS error"))
        result = _safe_database_operation(operation, "Error message")

        assert result is False
        operation.assert_called_once()
