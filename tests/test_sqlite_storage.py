"""Additional tests exercising sqlite_storage with a real database."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
import sqlalchemy as sa

from custom_components.area_occupancy.schema import (
    AreaEntityConfigRecord,
    AreaOccupancyRecord,
    AreaTimePriorRecord,
)
from custom_components.area_occupancy.sqlite_storage import AreaOccupancyStorage
from custom_components.area_occupancy.state_intervals import StateInterval
from homeassistant.util import dt as dt_util


@pytest.fixture
def real_storage(tmp_path, mock_hass: Mock) -> AreaOccupancyStorage:
    """Return storage using a real sqlite database."""
    mock_hass.config.config_dir = str(tmp_path)
    # Execute jobs synchronously
    mock_hass.async_add_executor_job.side_effect = lambda func, *a, **kw: func(*a, **kw)

    # Create mock coordinator with entity IDs for the updated import method
    mock_coordinator = Mock()
    mock_coordinator.occupancy_entity_id = "binary_sensor.test_occupancy"
    mock_coordinator.wasp_entity_id = "binary_sensor.test_wasp"

    storage = AreaOccupancyStorage(hass=mock_hass, entry_id="test_entry")
    storage.coordinator = mock_coordinator
    return storage


async def test_enable_wal_mode_error(real_storage: AreaOccupancyStorage):
    """_enable_wal_mode logs when sqlite call fails."""
    err = sa.exc.SQLAlchemyError("fail")
    with (
        patch.object(real_storage.engine, "connect", side_effect=err),
        patch("custom_components.area_occupancy.sqlite_storage._LOGGER.debug") as log,
    ):
        real_storage._enable_wal_mode()  # noqa: SLF001
        log.assert_called_with("Failed to enable WAL mode: %s", err)


async def test_vacuum_database_executes(real_storage: AreaOccupancyStorage):
    """VACUUM command is executed on the connection."""
    mock_conn = Mock()
    context = Mock(
        __enter__=Mock(return_value=mock_conn), __exit__=Mock(return_value=None)
    )
    with patch.object(real_storage.engine, "connect", return_value=context):
        await real_storage.vacuum_database()
        mock_conn.execute.assert_called_once()
        assert "VACUUM" in str(mock_conn.execute.call_args.args[0])


async def test_import_intervals_from_recorder(real_storage: AreaOccupancyStorage):
    """Importing intervals stores them via batch method."""
    # Mock get_intervals_from_recorder to avoid actual recorder calls
    with (
        patch(
            "custom_components.area_occupancy.sqlite_storage.get_intervals_from_recorder",
            return_value=[],
        ),
    ):
        await real_storage.import_intervals_from_recorder(["sensor.demo"], days=1)

        # Check that import stats includes all processed entities
        # The method now processes: input entities + occupancy + wasp entities
        expected_entities = {
            "sensor.demo",
            "binary_sensor.test_occupancy",
            "binary_sensor.test_wasp",
        }

        # Verify all expected entities were processed
        for entity_id in expected_entities:
            assert entity_id in real_storage.import_stats
            assert (
                real_storage.import_stats[entity_id] == 0
            )  # No intervals returned by mock


async def test_import_intervals_from_recorder_with_none_entities(
    real_storage: AreaOccupancyStorage,
):
    """Test import when coordinator entity IDs are None."""
    # Set coordinator entity IDs to None
    real_storage.coordinator.occupancy_entity_id = None
    real_storage.coordinator.wasp_entity_id = None

    with (
        patch(
            "custom_components.area_occupancy.sqlite_storage.get_intervals_from_recorder",
            return_value=[],
        ),
        patch.object(
            real_storage,
            "save_state_intervals_batch",
            return_value=0,
        ),
    ):
        await real_storage.import_intervals_from_recorder(["sensor.demo"], days=1)

        # Should only process the input entity when coordinator entities are None
        assert "sensor.demo" in real_storage.import_stats
        assert len(real_storage.import_stats) == 1


async def test_historical_intervals_and_cleanup(real_storage: AreaOccupancyStorage):
    """Save intervals, query them and cleanup old ones."""
    await real_storage.async_initialize()
    base = dt_util.utcnow() - timedelta(hours=1)
    intervals = [
        StateInterval(
            entity_id="sensor.demo",
            state="on",
            start=base,
            end=base + timedelta(minutes=5),
        ),
        StateInterval(
            entity_id="sensor.demo",
            state="off",
            start=base + timedelta(minutes=10),
            end=base + timedelta(minutes=15),
        ),
    ]
    await real_storage.save_state_intervals_batch(intervals)

    result = await real_storage.get_historical_intervals(
        "sensor.demo",
        start_time=base - timedelta(minutes=1),
        end_time=base + timedelta(minutes=20),
        limit=1,
        page_size=1,
    )
    assert len(result) == 1

    # Add an old interval and cleanup
    old_time = dt_util.utcnow() - timedelta(days=400)
    await real_storage.save_state_intervals_batch(
        [
            StateInterval(
                entity_id="sensor.demo",
                state="idle",
                start=old_time,
                end=old_time + timedelta(minutes=1),
            )
        ]
    )
    deleted = await real_storage.cleanup_old_intervals(retention_days=365)
    assert deleted == 1
    count = await real_storage.get_total_intervals_count()
    assert count == 2  # two recent remain after cleanup


async def test_historical_intervals_with_state_filter(
    real_storage: AreaOccupancyStorage,
):
    """Test querying historical intervals with state filter."""
    await real_storage.async_initialize()
    base = dt_util.utcnow() - timedelta(hours=1)
    intervals = [
        StateInterval(
            entity_id="sensor.demo",
            state="on",
            start=base,
            end=base + timedelta(minutes=5),
        ),
        StateInterval(
            entity_id="sensor.demo",
            state="off",
            start=base + timedelta(minutes=10),
            end=base + timedelta(minutes=15),
        ),
        StateInterval(
            entity_id="sensor.demo",
            state="on",
            start=base + timedelta(minutes=20),
            end=base + timedelta(minutes=25),
        ),
    ]
    await real_storage.save_state_intervals_batch(intervals)

    # Query with state filter for "on" states only
    result = await real_storage.get_historical_intervals(
        "sensor.demo",
        start_time=base - timedelta(minutes=1),
        end_time=base + timedelta(minutes=30),
        state_filter="on",
    )

    # Should only return the "on" state intervals
    assert len(result) == 2
    for interval in result:
        assert interval["state"] == "on"

    # Query with state filter for "off" states only
    result_off = await real_storage.get_historical_intervals(
        "sensor.demo",
        start_time=base - timedelta(minutes=1),
        end_time=base + timedelta(minutes=30),
        state_filter="off",
    )

    # Should only return the "off" state interval
    assert len(result_off) == 1
    assert result_off[0]["state"] == "off"


async def test_stats_with_real_data(real_storage: AreaOccupancyStorage):
    """Stats reflect stored records."""
    await real_storage.async_initialize()
    area = AreaOccupancyRecord(
        entry_id="test_entry",
        area_name="Area",
        purpose="test",
        threshold=0.4,
        created_at=dt_util.utcnow(),
        updated_at=dt_util.utcnow(),
    )
    await real_storage.save_area_occupancy(area)
    config = AreaEntityConfigRecord(
        entry_id="test_entry",
        entity_id="sensor.demo",
        entity_type="motion",
        weight=1.0,
        prob_given_true=0.9,
        prob_given_false=0.1,
        last_updated=dt_util.utcnow(),
    )
    await real_storage.save_area_entity_config(config)
    prior = AreaTimePriorRecord(
        entry_id="test_entry",
        day_of_week=1,
        time_slot=12,
        prior_value=0.5,
        data_points=1,
        last_updated=dt_util.utcnow(),
    )
    await real_storage.save_time_priors_batch([prior])
    stats = await real_storage.get_stats()
    assert stats["area_occupancy_count"] == 1
    assert stats["area_entity_config_entry_test_entry"] == 1
    assert stats["area_time_priors_entry_test_entry"] == 1
