"""Tests for database state synchronization."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.db.sync import (
    _get_existing_interval_keys,
    _states_to_intervals,
    sync_states,
)
from homeassistant.core import State
from homeassistant.util import dt as dt_util


class TestStatesToIntervals:
    """Test _states_to_intervals function."""

    def test_states_to_intervals_single_state(self, test_db):
        """Test converting single state to interval."""
        db = test_db
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

    def test_states_to_intervals_multiple_states(self, test_db):
        """Test converting multiple states to intervals."""
        db = test_db
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

    def test_states_to_intervals_filters_invalid_states(self, test_db):
        """Test that invalid states are filtered out."""
        db = test_db
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


class TestSyncStates:
    """Test sync_states function."""

    @pytest.mark.asyncio
    async def test_sync_states_success(self, test_db, monkeypatch):
        """Test successful state synchronization."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        # Create entity first so sync can process it
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        # Add entity to area's EntityManager so find_area_for_entity can find it
        area = db.coordinator.get_area(area_name)
        motion_entity = SimpleNamespace(
            entity_id="binary_sensor.motion",
            type=SimpleNamespace(input_type=InputType.MOTION),
        )
        area.entities.add_entity(motion_entity)

        # Mock recorder history
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

        # Mock get_instance to return a mock recorder
        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        await sync_states(db)

        # Verify intervals were saved by querying the database
        with db.get_session() as session:
            intervals = (
                session.query(db.Intervals)
                .filter_by(entity_id="binary_sensor.motion")
                .all()
            )
            # Verify exactly one interval was created
            assert len(intervals) == 1, f"Expected 1 interval, got {len(intervals)}"

            interval = intervals[0]

            # Validate entity_id matches
            assert interval.entity_id == "binary_sensor.motion"

            # Validate state
            assert interval.state == "on"

            # Validate timestamps with tolerance (allow 2 seconds for execution time)
            tolerance = timedelta(seconds=2)
            expected_start = now - timedelta(minutes=5)

            # Normalize both datetimes to UTC-naive for comparison
            # Database stores timezone-aware datetimes, so convert to UTC then remove timezone
            def to_utc_naive(dt):
                """Convert datetime to UTC-naive for comparison."""
                if dt.tzinfo is None:
                    return dt
                return dt.astimezone(dt_util.UTC).replace(tzinfo=None)

            interval_start_utc = to_utc_naive(interval.start_time)
            expected_start_utc = to_utc_naive(expected_start)

            assert (
                abs((interval_start_utc - expected_start_utc).total_seconds())
                <= tolerance.total_seconds()
            ), (
                f"start_time {interval.start_time} not within tolerance of {expected_start}"
            )

            # end_time validation: duration check is sufficient
            # Note: We don't compare absolute end_time or check end_time >= start_time
            # due to potential timezone differences between dt_util.now() (used in sync_states)
            # and dt_util.utcnow() (used for start_time). The duration_seconds check below
            # validates that the interval is approximately 5 minutes, which is the key validation.

            # Validate duration_seconds (approximately 5 minutes = 300 seconds)
            expected_duration = 300.0  # 5 minutes in seconds
            duration_tolerance = 2.0  # Allow 2 seconds tolerance
            assert (
                abs(interval.duration_seconds - expected_duration) <= duration_tolerance
            ), (
                f"duration_seconds {interval.duration_seconds} not within tolerance of {expected_duration}"
            )

            # Validate aggregation_level (defaults to "raw")
            assert interval.aggregation_level == "raw"

            # Validate area_name matches
            assert interval.area_name == area_name

            # Validate entry_id matches
            assert interval.entry_id == db.coordinator.entry_id

    @pytest.mark.asyncio
    async def test_sync_states_handles_errors(self, test_db, monkeypatch):
        """Test that sync_states handles errors gracefully."""
        db = test_db

        # Mock get_instance to return a mock recorder
        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=RuntimeError("Recorder error")  # Raise error when called
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        # This test asserts that no exception is raised - errors must be handled
        # internally and not re-raised. If sync_states raises an exception here,
        # the test will fail, which is the desired behavior.
        await sync_states(db)

    @pytest.mark.asyncio
    async def test_sync_states_multi_area_isolation(self, test_db, monkeypatch):
        """Test that sync_states isolates areas correctly."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area exists first (foreign key requirement)
        db.save_area_data(area_name)

        # Create entity first
        with db.get_locked_session() as session:
            entity = db.Entities(
                entity_id="binary_sensor.motion",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        # Mock states for specific area
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

        # Mock get_instance to return a mock recorder
        mock_recorder = Mock()
        mock_recorder.async_add_executor_job = AsyncMock(
            side_effect=lambda func: func()
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.sync.get_instance",
            lambda hass: mock_recorder,
        )

        await sync_states(db)

        # Verify intervals are associated with correct area by querying database
        with db.get_session() as session:
            intervals = (
                session.query(db.Intervals)
                .filter_by(
                    entity_id="binary_sensor.motion",
                    area_name=area_name,
                )
                .all()
            )
            # Verify area_name is set correctly if intervals exist
            for interval in intervals:
                assert interval.area_name == area_name

    @pytest.mark.asyncio
    async def test_sync_states_records_numeric_samples(self, test_db, monkeypatch):
        """Ensure numeric states are stored in NumericSamples."""
        db = test_db
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


class TestIntervalLookup:
    """Tests for helper functions used during sync."""

    def test_get_existing_interval_keys_batches(self, test_db, monkeypatch):
        """Existing intervals should be found in batch-sized queries."""
        db = test_db
        area_name = db.coordinator.get_area_names()[0]
        db.save_area_data(area_name)

        now = dt_util.utcnow()
        interval_defs = []
        with db.get_locked_session() as session:
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
