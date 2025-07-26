"""Tests for schema module."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.schema import (
    AreaEntityConfigRecord,
    AreaOccupancyRecord,
    AreaTimePriorRecord,
    EntityRecord,
    SchemaConverter,
    area_entity_config_table,
    area_occupancy_table,
    area_time_priors_table,
    entities_table,
    indexes,
    metadata,
    metadata_table,
    state_intervals_table,
)
from custom_components.area_occupancy.state_intervals import StateInterval


# ruff: noqa: PLC0415
class TestDataModels:
    """Test data model classes."""

    def test_area_occupancy_record_defaults(self) -> None:
        """Test AreaOccupancyRecord with default values."""
        record = AreaOccupancyRecord()

        assert record.entry_id == ""
        assert record.area_name == ""
        assert record.purpose == ""
        assert record.threshold == 0.0
        assert isinstance(record.created_at, datetime)
        assert isinstance(record.updated_at, datetime)

    def test_area_occupancy_record_with_values(self) -> None:
        """Test AreaOccupancyRecord with specific values."""
        now = datetime.now()
        record = AreaOccupancyRecord(
            entry_id="test_entry",
            area_name="Test Area",
            purpose="living_room",
            threshold=0.7,
            created_at=now,
            updated_at=now,
        )

        assert record.entry_id == "test_entry"
        assert record.area_name == "Test Area"
        assert record.purpose == "living_room"
        assert record.threshold == 0.7
        assert record.created_at == now
        assert record.updated_at == now

    def test_entity_record_defaults(self) -> None:
        """Test EntityRecord with default values."""
        record = EntityRecord()

        assert record.entity_id == ""
        assert isinstance(record.last_seen, datetime)
        assert isinstance(record.created_at, datetime)

    def test_entity_record_with_values(self) -> None:
        """Test EntityRecord with specific values."""
        now = datetime.now()
        record = EntityRecord(
            entity_id="sensor.test",
            last_seen=now,
            created_at=now,
        )

        assert record.entity_id == "sensor.test"
        assert record.last_seen == now
        assert record.created_at == now

    def test_entity_record_domain_property(self) -> None:
        """Test EntityRecord domain property."""
        record = EntityRecord(entity_id="sensor.test")
        assert record.domain == "sensor"

        record = EntityRecord(entity_id="binary_sensor.motion")
        assert record.domain == "binary_sensor"

        record = EntityRecord(entity_id="invalid")
        assert record.domain == "unknown"

    def test_area_entity_config_record_defaults(self) -> None:
        """Test AreaEntityConfigRecord with default values."""
        record = AreaEntityConfigRecord()

        assert record.entry_id == ""
        assert record.entity_id == ""
        assert record.entity_type == ""
        assert record.weight == 1.0
        assert record.prob_given_true == 0.5
        assert record.prob_given_false == 0.1
        assert isinstance(record.last_updated, datetime)

    def test_area_entity_config_record_with_values(self) -> None:
        """Test AreaEntityConfigRecord with specific values."""
        now = datetime.now()
        record = AreaEntityConfigRecord(
            entry_id="test_entry",
            entity_id="sensor.test",
            entity_type="motion",
            weight=2.0,
            prob_given_true=0.8,
            prob_given_false=0.2,
            last_updated=now,
        )

        assert record.entry_id == "test_entry"
        assert record.entity_id == "sensor.test"
        assert record.entity_type == "motion"
        assert record.weight == 2.0
        assert record.prob_given_true == 0.8
        assert record.prob_given_false == 0.2
        assert record.last_updated == now

    def test_area_time_prior_record_defaults(self) -> None:
        """Test AreaTimePriorRecord with default values."""
        record = AreaTimePriorRecord()

        assert record.entry_id == ""
        assert record.day_of_week == 0
        assert record.time_slot == 0
        assert record.prior_value == 0.1
        assert record.data_points == 0
        assert isinstance(record.last_updated, datetime)

    def test_area_time_prior_record_with_values(self) -> None:
        """Test AreaTimePriorRecord with specific values."""
        now = datetime.now()
        record = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=1,  # Tuesday
            time_slot=14,  # 7:00-7:29
            prior_value=0.3,
            data_points=10,
            last_updated=now,
        )

        assert record.entry_id == "test_entry"
        assert record.day_of_week == 1
        assert record.time_slot == 14
        assert record.prior_value == 0.3
        assert record.data_points == 10
        assert record.last_updated == now

    def test_area_time_prior_record_time_range_property(self) -> None:
        """Test AreaTimePriorRecord time_range property."""
        # Test 7:00-7:29 slot
        record = AreaTimePriorRecord(time_slot=14)
        assert record.time_range == (7, 0)
        assert record.end_time_range == (7, 30)

        # Test 7:30-7:59 slot
        record = AreaTimePriorRecord(time_slot=15)
        assert record.time_range == (7, 30)
        assert record.end_time_range == (8, 0)

        # Test 23:30-23:59 slot (edge case)
        record = AreaTimePriorRecord(time_slot=47)
        assert record.time_range == (23, 30)
        assert record.end_time_range == (0, 0)

    def test_area_time_prior_record_edge_cases(self) -> None:
        """Test AreaTimePriorRecord edge cases."""
        # Test midnight slot (0:00-0:29)
        record = AreaTimePriorRecord(time_slot=0)
        assert record.time_range == (0, 0)
        assert record.end_time_range == (0, 30)

        # Test 23:00-23:29 slot
        record = AreaTimePriorRecord(time_slot=46)
        assert record.time_range == (23, 0)
        assert record.end_time_range == (23, 30)


class TestSchemaConverter:
    """Test SchemaConverter class."""

    def test_row_to_area_occupancy(self) -> None:
        """Test row_to_area_occupancy method."""
        mock_row = Mock()
        mock_row.entry_id = "test_entry"
        mock_row.area_name = "Test Area"
        mock_row.purpose = "living_room"
        mock_row.threshold = 0.7
        mock_row.created_at = datetime.now()
        mock_row.updated_at = datetime.now()

        result = SchemaConverter.row_to_area_occupancy(mock_row)

        assert isinstance(result, AreaOccupancyRecord)
        assert result.entry_id == "test_entry"
        assert result.area_name == "Test Area"
        assert result.purpose == "living_room"
        assert result.threshold == 0.7

    def test_area_occupancy_to_dict(self) -> None:
        """Test area_occupancy_to_dict method."""
        now = datetime.now()
        record = AreaOccupancyRecord(
            entry_id="test_entry",
            area_name="Test Area",
            purpose="living_room",
            threshold=0.7,
            created_at=now,
            updated_at=now,
        )

        result = SchemaConverter.area_occupancy_to_dict(record)

        assert isinstance(result, dict)
        assert result["entry_id"] == "test_entry"
        assert result["area_name"] == "Test Area"
        assert result["purpose"] == "living_room"
        assert result["threshold"] == 0.7
        assert result["created_at"] == now
        assert result["updated_at"] == now

    def test_row_to_entity(self) -> None:
        """Test row_to_entity method."""
        mock_row = Mock()
        mock_row.entity_id = "sensor.test"
        mock_row.last_seen = datetime.now()
        mock_row.created_at = datetime.now()

        result = SchemaConverter.row_to_entity(mock_row)

        assert isinstance(result, EntityRecord)
        assert result.entity_id == "sensor.test"

    def test_entity_to_dict(self) -> None:
        """Test entity_to_dict method."""
        now = datetime.now()
        record = EntityRecord(
            entity_id="sensor.test",
            last_seen=now,
            created_at=now,
        )

        result = SchemaConverter.entity_to_dict(record)

        assert isinstance(result, dict)
        assert result["entity_id"] == "sensor.test"
        assert result["last_seen"] == now
        assert result["created_at"] == now

    def test_row_to_area_entity_config(self) -> None:
        """Test row_to_area_entity_config method."""
        mock_row = Mock()
        mock_row.entry_id = "test_entry"
        mock_row.entity_id = "sensor.test"
        mock_row.entity_type = "motion"
        mock_row.weight = 2.0
        mock_row.prob_given_true = 0.8
        mock_row.prob_given_false = 0.2
        mock_row.last_updated = datetime.now()

        result = SchemaConverter.row_to_area_entity_config(mock_row)

        assert isinstance(result, AreaEntityConfigRecord)
        assert result.entry_id == "test_entry"
        assert result.entity_id == "sensor.test"
        assert result.entity_type == "motion"
        assert result.weight == 2.0
        assert result.prob_given_true == 0.8
        assert result.prob_given_false == 0.2

    def test_area_entity_config_to_dict(self) -> None:
        """Test area_entity_config_to_dict method."""
        now = datetime.now()
        record = AreaEntityConfigRecord(
            entry_id="test_entry",
            entity_id="sensor.test",
            entity_type="motion",
            weight=2.0,
            prob_given_true=0.8,
            prob_given_false=0.2,
            last_updated=now,
        )

        result = SchemaConverter.area_entity_config_to_dict(record)

        assert isinstance(result, dict)
        assert result["entry_id"] == "test_entry"
        assert result["entity_id"] == "sensor.test"
        assert result["entity_type"] == "motion"
        assert result["weight"] == 2.0
        assert result["prob_given_true"] == 0.8
        assert result["prob_given_false"] == 0.2
        assert result["last_updated"] == now

    def test_row_to_area_time_prior(self) -> None:
        """Test row_to_area_time_prior method."""
        mock_row = Mock()
        mock_row.entry_id = "test_entry"
        mock_row.day_of_week = 1
        mock_row.time_slot = 14
        mock_row.prior_value = 0.3
        mock_row.data_points = 10
        mock_row.last_updated = datetime.now()

        result = SchemaConverter.row_to_area_time_prior(mock_row)

        assert isinstance(result, AreaTimePriorRecord)
        assert result.entry_id == "test_entry"
        assert result.day_of_week == 1
        assert result.time_slot == 14
        assert result.prior_value == 0.3
        assert result.data_points == 10

    def test_area_time_prior_to_dict(self) -> None:
        """Test area_time_prior_to_dict method."""
        now = datetime.now()
        record = AreaTimePriorRecord(
            entry_id="test_entry",
            day_of_week=1,
            time_slot=14,
            prior_value=0.3,
            data_points=10,
            last_updated=now,
        )

        result = SchemaConverter.area_time_prior_to_dict(record)

        assert isinstance(result, dict)
        assert result["entry_id"] == "test_entry"
        assert result["day_of_week"] == 1
        assert result["time_slot"] == 14
        assert result["prior_value"] == 0.3
        assert result["data_points"] == 10
        assert result["last_updated"] == now

    def test_row_to_state_interval(self) -> None:
        """Test row_to_state_interval method."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)

        mock_row = Mock()
        mock_row.entity_id = "sensor.test"
        mock_row.state = "on"
        mock_row.start_time = start_time
        mock_row.end_time = end_time
        mock_row.duration_seconds = 300.0

        result = SchemaConverter.row_to_state_interval(mock_row)

        # StateInterval is a TypedDict, so check it's a dict with expected keys
        assert isinstance(result, dict)
        assert "entity_id" in result
        assert "state" in result
        assert "start" in result
        assert "end" in result
        assert result["entity_id"] == "sensor.test"
        assert result["state"] == "on"
        assert result["start"] == start_time
        assert result["end"] == end_time

    def test_state_interval_to_dict(self) -> None:
        """Test state_interval_to_dict method."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)

        interval = StateInterval(
            entity_id="sensor.test",
            state="on",
            start=start_time,
            end=end_time,
        )

        result = SchemaConverter.state_interval_to_dict(interval)

        assert isinstance(result, dict)
        assert result["entity_id"] == "sensor.test"
        assert result["state"] == "on"
        assert result["start_time"] == start_time
        assert result["end_time"] == end_time
        assert "duration_seconds" in result
        assert "created_at" in result


class TestDatabaseSchema:
    """Test database schema definitions."""

    def test_metadata_contains_all_tables(self) -> None:
        """Test that metadata contains all expected tables."""
        expected_tables = {
            "entities",
            "state_intervals",
            "metadata",
            "area_occupancy",
            "area_entity_config",
            "area_time_priors",
        }

        actual_tables = set(metadata.tables.keys())
        assert expected_tables.issubset(actual_tables)

    def test_entities_table_structure(self) -> None:
        """Test entities table structure."""
        table = entities_table

        # Check columns exist
        assert "entity_id" in table.columns
        assert "last_seen" in table.columns
        assert "created_at" in table.columns

        # Check primary key
        assert table.primary_key.columns[0].name == "entity_id"

    def test_state_intervals_table_structure(self) -> None:
        """Test state_intervals table structure."""
        table = state_intervals_table

        # Check columns exist
        assert "id" in table.columns
        assert "entity_id" in table.columns
        assert "state" in table.columns
        assert "start_time" in table.columns
        assert "end_time" in table.columns
        assert "duration_seconds" in table.columns
        assert "created_at" in table.columns

        # Check primary key
        assert table.primary_key.columns[0].name == "id"

        # Check foreign key
        fk_constraints = list(table.foreign_keys)
        assert len(fk_constraints) == 1
        assert fk_constraints[0].column.name == "entity_id"

    def test_area_occupancy_table_structure(self) -> None:
        """Test area_occupancy table structure."""
        table = area_occupancy_table

        # Check columns exist
        assert "entry_id" in table.columns
        assert "area_name" in table.columns
        assert "purpose" in table.columns
        assert "threshold" in table.columns
        assert "created_at" in table.columns
        assert "updated_at" in table.columns

        # Check primary key
        assert table.primary_key.columns[0].name == "entry_id"

    def test_area_entity_config_table_structure(self) -> None:
        """Test area_entity_config table structure."""
        table = area_entity_config_table

        # Check columns exist
        assert "entry_id" in table.columns
        assert "entity_id" in table.columns
        assert "entity_type" in table.columns
        assert "weight" in table.columns
        assert "prob_given_true" in table.columns
        assert "prob_given_false" in table.columns
        assert "last_updated" in table.columns

        # Check composite primary key
        pk_columns = [col.name for col in table.primary_key.columns]
        assert "entry_id" in pk_columns
        assert "entity_id" in pk_columns

        # Check foreign keys
        fk_constraints = list(table.foreign_keys)
        assert len(fk_constraints) == 2

    def test_area_time_priors_table_structure(self) -> None:
        """Test area_time_priors table structure."""
        table = area_time_priors_table

        # Check columns exist
        assert "entry_id" in table.columns
        assert "day_of_week" in table.columns
        assert "time_slot" in table.columns
        assert "prior_value" in table.columns
        assert "data_points" in table.columns
        assert "last_updated" in table.columns

        # Check composite primary key
        pk_columns = [col.name for col in table.primary_key.columns]
        assert "entry_id" in pk_columns
        assert "day_of_week" in pk_columns
        assert "time_slot" in pk_columns

    def test_indexes_exist(self) -> None:
        """Test that expected indexes are defined."""
        expected_index_names = {
            "idx_state_intervals_entity",
            "idx_state_intervals_entity_time",
            "idx_state_intervals_end_time",
            "idx_area_entity_entry",
            "idx_area_time_priors_entry",
            "idx_area_time_priors_day_slot",
            "idx_area_time_priors_last_updated",
        }

        actual_index_names = {index.name for index in indexes}
        assert expected_index_names.issubset(actual_index_names)

        for index in indexes:
            assert index.table in metadata.tables.values()

    def test_metadata_table_structure(self) -> None:
        """Test metadata table structure."""
        table = metadata_table

        # Check columns exist
        assert "key" in table.columns
        assert "value" in table.columns

        # Check primary key
        assert table.primary_key.columns[0].name == "key"


class TestDatabaseVersion:
    """Test database version handling."""

    def test_db_version_constant(self) -> None:
        """Test that DB_VERSION is defined and is an integer."""
        from custom_components.area_occupancy.schema import DB_VERSION

        assert isinstance(DB_VERSION, int)
        assert DB_VERSION > 0

    def test_get_create_table_ddl(self) -> None:
        """Test get_create_table_ddl function."""
        from custom_components.area_occupancy.schema import get_create_table_ddl

        ddl = get_create_table_ddl("sqlite")
        assert isinstance(ddl, str)
        assert len(ddl) > 0

        # Should contain table creation statements
        assert "CREATE TABLE" in ddl.upper()

        # Test with invalid dialect
        with pytest.raises(Exception):  # noqa: B017
            get_create_table_ddl("invalid_dialect")
