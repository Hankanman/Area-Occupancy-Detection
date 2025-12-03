"""Tests for database state synchronization."""

from datetime import timedelta
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import sqlalchemy as sa

from custom_components.area_occupancy.const import (
    MAX_INTERVAL_SECONDS,
    MIN_INTERVAL_SECONDS,
    RETENTION_DAYS,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.db.sync import (
    _get_existing_interval_keys,
    _get_existing_numeric_sample_keys,
    _states_to_intervals,
    _states_to_numeric_samples,
    sync_states,
)
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from tests.conftest import create_test_area  # noqa: TID251


class TestStatesToIntervals:
    """Test _states_to_intervals function."""

    def test_states_to_intervals_single_state(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test converting single state to interval."""
        db = coordinator.db
        start = dt_util.utcnow()
        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "off", last_changed=start),
            ]
        }
        end_time = start + timedelta(seconds=20)

        intervals = _states_to_intervals(db, states, end_time)
        assert len(intervals) == 1
        assert intervals[0]["entity_id"] == "binary_sensor.motion"
        assert intervals[0]["state"] == "off"
        assert intervals[0]["start_time"] == start
        assert intervals[0]["end_time"] == end_time
        assert intervals[0]["duration_seconds"] == (end_time - start).total_seconds()

    def test_states_to_intervals_multiple_states(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test converting multiple states to intervals."""
        db = coordinator.db
        start = dt_util.utcnow()
        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "off", last_changed=start),
                State(
                    "binary_sensor.motion",
                    "on",
                    last_changed=start + timedelta(seconds=10),
                ),
            ]
        }
        end_time = start + timedelta(seconds=20)

        intervals = _states_to_intervals(db, states, end_time)
        assert len(intervals) == 2
        assert intervals[0]["state"] == "off"
        assert intervals[0]["start_time"] == start
        assert intervals[0]["end_time"] == start + timedelta(seconds=10)
        assert intervals[1]["state"] == "on"
        assert intervals[1]["start_time"] == start + timedelta(seconds=10)
        assert intervals[1]["end_time"] == end_time

    def test_states_to_intervals_filters_invalid_states(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that invalid states are filtered out."""
        db = coordinator.db
        start = dt_util.utcnow()
        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "unknown", last_changed=start),
                State(
                    "binary_sensor.motion",
                    "on",
                    last_changed=start + timedelta(seconds=10),
                ),
            ]
        }
        end_time = start + timedelta(seconds=20)

        intervals = _states_to_intervals(db, states, end_time)
        # Should filter out "unknown" state
        assert len(intervals) == 1
        assert intervals[0]["state"] == "on"

    def test_states_to_intervals_filters_retention_period(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that states older than RETENTION_DAYS are filtered out."""
        db = coordinator.db
        now = dt_util.utcnow()
        # Create state older than retention period
        old_state_time = now - timedelta(days=RETENTION_DAYS + 1)
        recent_state_time = now - timedelta(days=1)

        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "on", last_changed=old_state_time),
                State("binary_sensor.motion", "off", last_changed=recent_state_time),
            ]
        }
        end_time = now

        intervals = _states_to_intervals(db, states, end_time)
        # Should filter out state older than retention period
        assert len(intervals) == 1
        assert intervals[0]["state"] == "off"
        assert intervals[0]["start_time"] == recent_state_time

    def test_states_to_intervals_filters_max_duration_on(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that 'on' states exceeding MAX_INTERVAL_SECONDS are filtered out."""
        db = coordinator.db
        start = dt_util.utcnow()
        # Create 'on' state with duration exceeding MAX_INTERVAL_SECONDS
        end_time = start + timedelta(seconds=MAX_INTERVAL_SECONDS + 100)

        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "on", last_changed=start),
            ]
        }

        intervals = _states_to_intervals(db, states, end_time)
        # Should filter out 'on' state exceeding MAX_INTERVAL_SECONDS
        assert len(intervals) == 0

    def test_states_to_intervals_filters_min_duration_non_on(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that non-'on' states below MIN_INTERVAL_SECONDS are filtered out."""
        db = coordinator.db
        start = dt_util.utcnow()
        # Create 'off' state with duration below MIN_INTERVAL_SECONDS
        end_time = start + timedelta(seconds=MIN_INTERVAL_SECONDS - 1)

        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "off", last_changed=start),
            ]
        }

        intervals = _states_to_intervals(db, states, end_time)
        # Should filter out non-'on' state below MIN_INTERVAL_SECONDS
        assert len(intervals) == 0

    def test_states_to_intervals_includes_max_duration_on_boundary(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that 'on' states with duration exactly equal to MAX_INTERVAL_SECONDS are included."""
        db = coordinator.db
        start = dt_util.utcnow()
        # Create 'on' state with duration exactly equal to MAX_INTERVAL_SECONDS
        end_time = start + timedelta(seconds=MAX_INTERVAL_SECONDS)

        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "on", last_changed=start),
            ]
        }

        intervals = _states_to_intervals(db, states, end_time)
        # Should include 'on' state with duration equal to MAX_INTERVAL_SECONDS
        assert len(intervals) == 1
        assert intervals[0]["state"] == "on"
        assert intervals[0]["duration_seconds"] == MAX_INTERVAL_SECONDS

    def test_states_to_intervals_includes_min_duration_non_on_boundary(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that non-'on' states with duration exactly equal to MIN_INTERVAL_SECONDS are included."""
        db = coordinator.db
        start = dt_util.utcnow()
        # Create 'off' state with duration exactly equal to MIN_INTERVAL_SECONDS
        end_time = start + timedelta(seconds=MIN_INTERVAL_SECONDS)

        states = {
            "binary_sensor.motion": [
                State("binary_sensor.motion", "off", last_changed=start),
            ]
        }

        intervals = _states_to_intervals(db, states, end_time)
        # Should include non-'on' state with duration equal to MIN_INTERVAL_SECONDS
        assert len(intervals) == 1
        assert intervals[0]["state"] == "off"
        assert intervals[0]["duration_seconds"] == MIN_INTERVAL_SECONDS

    def test_states_to_intervals_handles_empty_states(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that empty state list returns empty intervals."""
        db = coordinator.db
        states = {"binary_sensor.motion": []}
        end_time = dt_util.utcnow()

        intervals = _states_to_intervals(db, states, end_time)
        assert len(intervals) == 0

    def test_states_to_intervals_sorts_by_last_changed(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that states are properly sorted by last_changed before processing."""
        db = coordinator.db
        start = dt_util.utcnow()
        # Create states out of order
        states = {
            "binary_sensor.motion": [
                State(
                    "binary_sensor.motion",
                    "on",
                    last_changed=start + timedelta(seconds=20),
                ),
                State("binary_sensor.motion", "off", last_changed=start),
                State(
                    "binary_sensor.motion",
                    "on",
                    last_changed=start + timedelta(seconds=10),
                ),
            ]
        }
        end_time = start + timedelta(seconds=30)

        intervals = _states_to_intervals(db, states, end_time)
        # Should be sorted by last_changed
        assert len(intervals) == 3
        assert intervals[0]["state"] == "off"
        assert intervals[0]["start_time"] == start
        assert intervals[1]["state"] == "on"
        assert intervals[1]["start_time"] == start + timedelta(seconds=10)
        assert intervals[2]["state"] == "on"
        assert intervals[2]["start_time"] == start + timedelta(seconds=20)


class TestSyncStates:
    """Test sync_states function."""

    @pytest.mark.asyncio
    async def test_sync_states_creates_intervals(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that sync_states creates intervals in the database."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Arrange: Set up area and entity
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        area = db.coordinator.get_area(area_name)
        motion_entity = SimpleNamespace(
            entity_id="binary_sensor.motion",
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area.entities.add_entity(motion_entity)

        now = dt_util.utcnow()
        mock_states = [
            State(
                "binary_sensor.motion", "on", last_changed=now - timedelta(minutes=5)
            ),
        ]

        def mock_get_significant_states(*args, **kwargs):
            return {"binary_sensor.motion": mock_states}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_significant_states",
            mock_get_significant_states,
        )

        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync
        await sync_states(db)

        # Assert: Verify interval was created
        with db.get_session() as session:
            intervals = (
                session.query(db.Intervals)
                .filter_by(entity_id="binary_sensor.motion")
                .all()
            )
            assert len(intervals) == 1
            assert intervals[0].entity_id == "binary_sensor.motion"
            assert intervals[0].state == "on"

    @pytest.mark.asyncio
    async def test_sync_states_sets_correct_fields(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that sync_states sets all required fields correctly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Arrange: Set up area and entity
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        area = db.coordinator.get_area(area_name)
        motion_entity = SimpleNamespace(
            entity_id="binary_sensor.motion",
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area.entities.add_entity(motion_entity)

        now = dt_util.utcnow()
        mock_states = [
            State(
                "binary_sensor.motion", "on", last_changed=now - timedelta(minutes=5)
            ),
        ]

        def mock_get_significant_states(*args, **kwargs):
            return {"binary_sensor.motion": mock_states}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_significant_states",
            mock_get_significant_states,
        )

        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync
        await sync_states(db)

        # Assert: Verify all fields are set correctly
        with db.get_session() as session:
            interval = (
                session.query(db.Intervals)
                .filter_by(entity_id="binary_sensor.motion")
                .first()
            )
            assert interval is not None
            assert interval.entity_id == "binary_sensor.motion"
            assert interval.state == "on"
            assert interval.area_name == area_name
            assert interval.entry_id == db.coordinator.entry_id
            assert interval.aggregation_level == "raw"
            assert interval.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_sync_states_handles_duplicate_intervals(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that duplicate intervals are not inserted twice."""

        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Arrange: Set up area and entity
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        area = db.coordinator.get_area(area_name)
        motion_entity = SimpleNamespace(
            entity_id="binary_sensor.motion",
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area.entities.add_entity(motion_entity)

        # Use fixed times to ensure duplicate detection works
        fixed_now = dt_util.utcnow()
        start_time = fixed_now - timedelta(minutes=5)
        end_time = fixed_now

        mock_states = [
            State("binary_sensor.motion", "on", last_changed=start_time),
        ]

        def mock_get_significant_states(*args, **kwargs):
            return {"binary_sensor.motion": mock_states}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_significant_states",
            mock_get_significant_states,
        )

        # Mock get_latest_interval to return a fixed time before our test data
        def mock_get_latest_interval(db_instance):
            return start_time - timedelta(hours=1)

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.queries.get_latest_interval",
            mock_get_latest_interval,
        )

        # Mock dt_util.utcnow to return fixed end_time
        def mock_utcnow():
            return end_time

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.dt_util.utcnow",
            mock_utcnow,
        )

        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync twice
        await sync_states(db)
        await sync_states(db)

        # Assert: Verify only one interval exists (duplicate was skipped)
        with db.get_session() as session:
            intervals = (
                session.query(db.Intervals)
                .filter_by(entity_id="binary_sensor.motion")
                .all()
            )
            assert len(intervals) == 1

    @pytest.mark.asyncio
    async def test_sync_states_handles_empty_entity_ids(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that sync_states returns early when no entities are configured."""
        db = coordinator.db

        # Arrange: Clear all entities from areas
        for area_name in db.coordinator.get_area_names():
            area = db.coordinator.get_area(area_name)
            if area:
                area.entities.entities.clear()

        # Mock recorder (should not be called)
        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock()
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync with no entities
        await sync_states(db)

        # Assert: Recorder should not be called
        mock_recorder.async_add_executor_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_states_handles_empty_states(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that sync_states returns early when recorder returns no states."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Arrange: Set up area and entity
        db.save_area_data(area_name)
        area = db.coordinator.get_area(area_name)
        motion_entity = SimpleNamespace(
            entity_id="binary_sensor.motion",
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area.entities.add_entity(motion_entity)

        def mock_get_significant_states(*args, **kwargs):
            return {}  # Empty states

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_significant_states",
            mock_get_significant_states,
        )

        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync
        await sync_states(db)

        # Assert: No intervals should be created
        with db.get_session() as session:
            intervals = session.query(db.Intervals).all()
            assert len(intervals) == 0

    @pytest.mark.asyncio
    async def test_sync_states_skips_entity_not_in_area(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that entities not found by find_area_for_entity are skipped."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Arrange: Set up area but don't add entity to area's EntityManager
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        # Note: Entity is NOT added to area.entities, so find_area_for_entity will return None

        now = dt_util.utcnow()
        mock_states = [
            State(
                "binary_sensor.motion", "on", last_changed=now - timedelta(minutes=5)
            ),
        ]

        def mock_get_significant_states(*args, **kwargs):
            return {"binary_sensor.motion": mock_states}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_significant_states",
            mock_get_significant_states,
        )

        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync
        await sync_states(db)

        # Assert: No intervals should be created (entity not in area)
        with db.get_session() as session:
            intervals = (
                session.query(db.Intervals)
                .filter_by(entity_id="binary_sensor.motion")
                .all()
            )
            assert len(intervals) == 0

    @pytest.mark.asyncio
    async def test_sync_states_multi_area_isolation(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that multiple areas with different entities are properly isolated."""
        db = coordinator.db
        area1_name = db.coordinator.get_area_names()[0]

        # Arrange: Create second area
        area2_name = "Kitchen"
        area2 = create_test_area(
            coordinator,
            area_name=area2_name,
            entity_ids=["binary_sensor.kitchen_motion"],
            threshold=0.5,
        )

        # Set area_id for area2 so save_area_data works
        area2.config.area_id = "kitchen"

        db.save_area_data(area1_name)
        db.save_area_data(area2_name)

        # Set up entities for both areas
        area1 = db.coordinator.get_area(area1_name)
        motion1_entity = SimpleNamespace(
            entity_id="binary_sensor.motion",
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area1.entities.add_entity(motion1_entity)

        motion2_entity = SimpleNamespace(
            entity_id="binary_sensor.kitchen_motion",
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area2.entities.add_entity(motion2_entity)

        # Also create entities in database for foreign key constraints
        with db.get_session() as session:
            entity1 = db.Entities(
                entity_id="binary_sensor.motion",
                entry_id=db.coordinator.entry_id,
                area_name=area1_name,
                entity_type="motion",
            )
            entity2 = db.Entities(
                entity_id="binary_sensor.kitchen_motion",
                entry_id=db.coordinator.entry_id,
                area_name=area2_name,
                entity_type="motion",
            )
            session.add(entity1)
            session.add(entity2)
            session.commit()

        now = dt_util.utcnow()
        mock_states = {
            "binary_sensor.motion": [
                State(
                    "binary_sensor.motion",
                    "on",
                    last_changed=now - timedelta(minutes=5),
                ),
            ],
            "binary_sensor.kitchen_motion": [
                State(
                    "binary_sensor.kitchen_motion",
                    "off",
                    last_changed=now - timedelta(minutes=3),
                ),
            ],
        }

        def mock_get_significant_states(*args, **kwargs):
            return mock_states

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_significant_states",
            mock_get_significant_states,
        )

        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync
        await sync_states(db)

        # Assert: Verify intervals are isolated by area
        with db.get_session() as session:
            area1_intervals = (
                session.query(db.Intervals).filter_by(area_name=area1_name).all()
            )
            area2_intervals = (
                session.query(db.Intervals).filter_by(area_name=area2_name).all()
            )

            assert len(area1_intervals) == 1
            assert area1_intervals[0].entity_id == "binary_sensor.motion"
            assert area1_intervals[0].area_name == area1_name

            assert len(area2_intervals) == 1
            assert area2_intervals[0].entity_id == "binary_sensor.kitchen_motion"
            assert area2_intervals[0].area_name == area2_name

    @pytest.mark.asyncio
    async def test_sync_states_records_numeric_samples(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that numeric states are stored in NumericSamples."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        area = db.coordinator.get_area(area_name)
        numeric_entity_id = "sensor.numeric"
        numeric_entity = SimpleNamespace(
            entity_id=numeric_entity_id,
            type=SimpleNamespace(input_type=InputType.TEMPERATURE),
        )
        area.entities.add_entity(numeric_entity)

        now = dt_util.utcnow()
        mock_states = [
            State(
                numeric_entity_id,
                "23.5",
                {"unit_of_measurement": "°C"},
                last_changed=now - timedelta(minutes=5),
            ),
            State(
                numeric_entity_id,
                "24.1",
                {"unit_of_measurement": "°C"},
                last_changed=now - timedelta(minutes=3),
            ),
        ]

        def mock_get_significant_states(*args, **kwargs):
            return {numeric_entity_id: mock_states}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_significant_states",
            mock_get_significant_states,
        )

        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        await sync_states(db)

        with db.get_session() as session:
            samples = (
                session.query(db.NumericSamples)
                .filter_by(entity_id=numeric_entity_id)
                .all()
            )
            assert len(samples) == 2

    @pytest.mark.asyncio
    async def test_sync_states_handles_duplicate_numeric_samples(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that duplicate numeric samples are not inserted twice."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        area = db.coordinator.get_area(area_name)
        numeric_entity_id = "sensor.numeric"
        numeric_entity = SimpleNamespace(
            entity_id=numeric_entity_id,
            type=SimpleNamespace(input_type=InputType.TEMPERATURE),
        )
        area.entities.add_entity(numeric_entity)

        now = dt_util.utcnow()
        timestamp = now - timedelta(minutes=5)
        mock_states = [
            State(
                numeric_entity_id,
                "23.5",
                {"unit_of_measurement": "°C"},
                last_changed=timestamp,
            ),
        ]

        def mock_get_significant_states(*args, **kwargs):
            return {numeric_entity_id: mock_states}

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_significant_states",
            mock_get_significant_states,
        )

        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync twice
        await sync_states(db)
        await sync_states(db)

        # Assert: Verify only one sample exists (duplicate was skipped)
        with db.get_session() as session:
            samples = (
                session.query(db.NumericSamples)
                .filter_by(entity_id=numeric_entity_id)
                .all()
            )
            assert len(samples) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error_class", "error_message", "use_bulk_insert_mock"),
        [
            (sa.exc.SQLAlchemyError, "Database error", True),
            (HomeAssistantError, "HA error", False),
            (TimeoutError, "Timeout", False),
            (OSError, "OS error", False),
            (RuntimeError, "Runtime error", False),
        ],
        ids=[
            "sqlalchemy_error",
            "homeassistant_error",
            "timeout_error",
            "os_error",
            "runtime_error",
        ],
    )
    async def test_sync_states_logs_errors(
        self,
        coordinator: AreaOccupancyCoordinator,
        monkeypatch,
        caplog,
        error_class,
        error_message,
        use_bulk_insert_mock,
    ):
        """Test that various errors are logged and not raised."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Arrange: Set up area and entity
        db.save_area_data(area_name)
        area = db.coordinator.get_area(area_name)
        motion_entity = SimpleNamespace(
            entity_id="binary_sensor.motion",
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area.entities.add_entity(motion_entity)

        # Set up mock recorder
        mock_recorder = Mock()
        if use_bulk_insert_mock:
            # SQLAlchemyError: Mock get_significant_states to return data, then mock bulk_insert_mappings
            def mock_get_significant_states(*args, **kwargs):
                return {
                    "binary_sensor.motion": [
                        State(
                            "binary_sensor.motion", "on", last_changed=dt_util.utcnow()
                        )
                    ]
                }

            monkeypatch.setattr(
                "custom_components.area_occupancy.db.sync.get_significant_states",
                mock_get_significant_states,
            )

            mock_recorder.async_add_executor_job = AsyncMock(
                side_effect=lambda func: func()
            )

            # Mock bulk_insert_mappings to raise SQLAlchemyError
            def mock_bulk_insert_mappings(self, mapper, mappings, *args, **kwargs):
                raise error_class(error_message)

            monkeypatch.setattr(
                sa.orm.Session,
                "bulk_insert_mappings",
                mock_bulk_insert_mappings,
            )
        else:
            # Other errors: Raise from async_add_executor_job
            mock_recorder.async_add_executor_job = AsyncMock(
                side_effect=error_class(error_message)
            )

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # Act: Run sync
        with caplog.at_level(logging.ERROR):
            await sync_states(db)

        # Assert: Error should be logged, not raised
        assert "Failed to sync states" in caplog.text
        assert error_message in caplog.text


class TestIntervalLookup:
    """Tests for helper functions used during sync."""

    def test_get_existing_interval_keys_batches(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that existing intervals are found in batch-sized queries."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        interval_defs = []
        with db.get_session() as session:
            for idx in range(2):
                start = now + timedelta(minutes=idx)
                end = start + timedelta(minutes=5)
                session.add(
                    db.Intervals(
                        entry_id=db.coordinator.entry_id,
                        area_name=area_name,
                        entity_id=f"binary_sensor.motion_{idx}",
                        state="on",
                        start_time=start,
                        end_time=end,
                        duration_seconds=(end - start).total_seconds(),
                    )
                )
                interval_defs.append((f"binary_sensor.motion_{idx}", start, end))
            session.commit()

        interval_keys = set(interval_defs)
        # Add a non-existing key to ensure it's ignored
        interval_keys.add(
            (
                "binary_sensor.motion_new",
                now + timedelta(hours=1),
                now + timedelta(hours=1, minutes=5),
            )
        )

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync._INTERVAL_LOOKUP_BATCH",
            1,
        )

        def normalize(dt_obj):
            return dt_obj.replace(tzinfo=None)

        with db.get_session() as session:
            existing = _get_existing_interval_keys(session, db, interval_keys)

        expected = {
            (entity_id, normalize(start), normalize(end))
            for entity_id, start, end in interval_defs
        }
        assert existing == expected

    @pytest.mark.parametrize(
        "lookup_function",
        [
            _get_existing_interval_keys,
            _get_existing_numeric_sample_keys,
        ],
        ids=["interval_keys", "numeric_sample_keys"],
    )
    def test_get_existing_keys_empty_set(
        self, coordinator: AreaOccupancyCoordinator, lookup_function
    ):
        """Test that empty keys return empty set for both interval and numeric sample lookups."""
        db = coordinator.db
        with db.get_session() as session:
            existing = lookup_function(session, db, set())
        assert existing == set()


class TestNumericSampleLookup:
    """Tests for numeric sample lookup functions."""

    def test_get_existing_numeric_sample_keys_batches(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that existing numeric samples are found in batch-sized queries."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        sample_defs = []
        with db.get_session() as session:
            for idx in range(2):
                timestamp = now + timedelta(minutes=idx)
                session.add(
                    db.NumericSamples(
                        entry_id=db.coordinator.entry_id,
                        area_name=area_name,
                        entity_id=f"sensor.temp_{idx}",
                        timestamp=timestamp,
                        value=20.0 + idx,
                        state=str(20.0 + idx),
                    )
                )
                sample_defs.append((f"sensor.temp_{idx}", timestamp))
            session.commit()

        sample_keys = set(sample_defs)
        # Add a non-existing key to ensure it's ignored
        sample_keys.add(("sensor.temp_new", now + timedelta(hours=1)))

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync._NUMERIC_SAMPLE_LOOKUP_BATCH",
            1,
        )

        def normalize(dt_obj):
            return dt_obj.replace(tzinfo=None)

        with db.get_session() as session:
            existing = _get_existing_numeric_sample_keys(session, db, sample_keys)

        expected = {
            (entity_id, normalize(timestamp)) for entity_id, timestamp in sample_defs
        }
        assert existing == expected


class TestStatesToNumericSamples:
    """Tests for _states_to_numeric_samples function."""

    def test_states_to_numeric_samples_filters_invalid_floats(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that invalid float conversions are skipped."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        area = db.coordinator.get_area(area_name)
        numeric_entity_id = "sensor.temperature"
        numeric_entity = SimpleNamespace(
            entity_id=numeric_entity_id,
            type=SimpleNamespace(input_type=InputType.TEMPERATURE),
        )
        area.entities.add_entity(numeric_entity)

        states = {
            numeric_entity_id: [
                State(numeric_entity_id, "23.5", last_changed=dt_util.utcnow()),
                State(numeric_entity_id, "invalid", last_changed=dt_util.utcnow()),
                State(numeric_entity_id, "24.1", last_changed=dt_util.utcnow()),
            ]
        }

        samples = _states_to_numeric_samples(db, states)
        # Should filter out "invalid" state
        assert len(samples) == 2
        assert all(sample["value"] in (23.5, 24.1) for sample in samples)

    def test_states_to_numeric_samples_filters_non_numeric_entities(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that non-numeric entities are filtered out."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        area = db.coordinator.get_area(area_name)
        # Add motion entity (not numeric)
        motion_entity_id = "binary_sensor.motion"
        motion_entity = SimpleNamespace(
            entity_id=motion_entity_id,
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area.entities.add_entity(motion_entity)

        states = {
            motion_entity_id: [
                State(motion_entity_id, "on", last_changed=dt_util.utcnow()),
            ]
        }

        samples = _states_to_numeric_samples(db, states)
        # Should filter out non-numeric entity
        assert len(samples) == 0
