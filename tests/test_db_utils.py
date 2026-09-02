"""Tests for database utility functions."""

from contextlib import contextmanager
from datetime import UTC, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy.exc import SQLAlchemyError

from custom_components.area_occupancy.const import (
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_WINDOW_ACTIVE_STATE,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.db.utils import (
    area_active_states_by_type,
    entity_active_states,
    is_active_state,
    is_intervals_empty,
    is_timestamp_in_prepared_intervals,
    is_timestamp_occupied,
    is_valid_state,
    prepare_occupied_intervals,
    resolve_active_states,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


class TestIsValidState:
    """Test is_valid_state function."""

    def test_valid_states(self):
        """Test that valid states return True."""
        assert is_valid_state("on") is True
        assert is_valid_state("off") is True
        assert is_valid_state("playing") is True
        assert is_valid_state("idle") is True
        assert is_valid_state(0) is True
        assert is_valid_state(1) is True
        assert is_valid_state(25.5) is True

    def test_invalid_states(self):
        """Test that invalid states return False."""
        assert is_valid_state("unknown") is False
        assert is_valid_state("unavailable") is False
        assert is_valid_state(None) is False
        assert is_valid_state("") is False
        assert is_valid_state("NaN") is False


class TestIsIntervalsEmpty:
    """Test is_intervals_empty function."""

    def test_empty_intervals(self, coordinator: AreaOccupancyCoordinator):
        """Test is_intervals_empty with empty intervals table."""
        db = coordinator.db
        result = is_intervals_empty(db)
        assert result is True

    def test_non_empty_intervals(self, coordinator: AreaOccupancyCoordinator):
        """Test is_intervals_empty with non-empty intervals table."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        end = dt_util.utcnow()
        start = end - timedelta(seconds=60)

        # Ensure area and entity exist first (foreign key requirements)
        db.save_area_data(area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_session() as session:
            interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion",
                state="on",
                start_time=start,
                end_time=end,
                duration_seconds=60,
                aggregation_level="raw",
            )
            session.add(interval)
            session.commit()

        result = is_intervals_empty(db)
        assert result is False

    def test_no_such_table_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test is_intervals_empty when table doesn't exist."""
        db = coordinator.db

        @contextmanager
        def mock_session():
            class MockSession:
                def query(self, *args):
                    raise SQLAlchemyError("no such table: intervals")

                def close(self):
                    pass

            yield MockSession()

        monkeypatch.setattr(db, "get_session", mock_session)
        result = is_intervals_empty(db)
        assert result is True  # Should return True when table doesn't exist

    def test_other_sqlalchemy_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test is_intervals_empty with other SQLAlchemy error."""
        db = coordinator.db

        @contextmanager
        def mock_session():
            class MockSession:
                def query(self, *args):
                    raise SQLAlchemyError("Connection error")

                def close(self):
                    pass

            yield MockSession()

        monkeypatch.setattr(db, "get_session", mock_session)
        result = is_intervals_empty(db)
        assert result is True  # Should return True as fallback

    def test_home_assistant_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test is_intervals_empty with HomeAssistantError."""
        db = coordinator.db

        @contextmanager
        def mock_session():
            raise HomeAssistantError("Database error")

        monkeypatch.setattr(db, "get_session", mock_session)
        result = is_intervals_empty(db)
        assert result is True  # Should return True as fallback


class TestPreparedIntervalLookup:
    """Tests for ``prepare_occupied_intervals`` + ``is_timestamp_in_prepared_intervals``.

    Behaviour parity with the legacy O(N) ``is_timestamp_occupied`` helper
    is the core invariant — the bisect-based lookup is only useful if it
    answers the same questions for the inputs the analysis pipeline
    produces (sorted, non-overlapping intervals).
    """

    def test_empty_inputs(self) -> None:
        starts, ends = prepare_occupied_intervals([])
        assert starts == [] and ends == []
        assert (
            is_timestamp_in_prepared_intervals(dt_util.utcnow(), starts, ends) is False
        )

    def test_membership_matches_legacy_helper(self) -> None:
        """Every probe lands the same answer through both code paths.

        Covers the boundary cases the legacy tests already enforce
        (start inclusive, end exclusive, gap, before, after) plus a
        deliberately-shuffled input order so the sort step inside
        ``prepare_occupied_intervals`` is exercised.
        """
        now = dt_util.utcnow()
        intervals = [
            (now + timedelta(hours=2), now + timedelta(hours=3)),  # second
            (now, now + timedelta(hours=1)),  # first
            (now + timedelta(hours=5), now + timedelta(hours=6)),  # third
        ]
        starts, ends = prepare_occupied_intervals(intervals)
        # Sort guarantee — bisect relies on it.
        assert starts == sorted(starts)

        probes = [
            now - timedelta(seconds=1),
            now,  # at start (inclusive)
            now + timedelta(minutes=30),  # inside first
            now + timedelta(hours=1),  # at end (exclusive)
            now + timedelta(hours=1, minutes=30),  # in gap
            now + timedelta(hours=2, minutes=30),  # inside second
            now + timedelta(hours=4),  # in gap
            now + timedelta(hours=5, minutes=30),  # inside third
            now + timedelta(hours=10),  # past end
        ]
        for probe in probes:
            assert is_timestamp_in_prepared_intervals(
                probe, starts, ends
            ) == is_timestamp_occupied(probe, intervals), probe

    def test_naive_inputs_normalised_to_utc(self) -> None:
        """Naive datetimes are interpreted as UTC, matching ``to_utc``.

        The hot loop callers in ``analyze_correlation`` mostly pass
        already-aware datetimes (``from_db_utc`` re-attaches UTC), but
        this guards against a regression that breaks parity with the
        legacy helper for ``tzinfo=None`` inputs.
        """
        naive_now = dt_util.utcnow().replace(tzinfo=None)
        intervals = [(naive_now, naive_now + timedelta(hours=1))]
        starts, ends = prepare_occupied_intervals(intervals)
        # Endpoints are stored UTC-aware after normalisation.
        assert starts[0].tzinfo is not None
        assert ends[0].tzinfo is not None
        # Probe with a non-UTC aware timestamp that converts to a UTC
        # instant inside the interval — the helper must convert the
        # probe to UTC before comparison, otherwise a Sydney clock at
        # 12:30 wouldn't match a UTC interval covering ~04:30 UTC.
        plus_two = timezone(timedelta(hours=2))
        # naive_now + 30min, interpreted as UTC, then re-expressed as +02:00
        probe = (
            (naive_now + timedelta(minutes=30)).replace(tzinfo=UTC).astimezone(plus_two)
        )
        assert is_timestamp_in_prepared_intervals(probe, starts, ends) is True


class TestActiveStateResolution:
    """Active-state resolution shared by the ground-truth and live paths.

    Issue #520: ``build_presence_query`` resolved active states from
    ``DEFAULT_TYPES`` while ``Entity.evidence`` resolved them from the area's
    config. These helpers exist so both paths read one definition and cannot
    drift apart again.
    """

    def test_configured_states_win(self) -> None:
        """A non-empty configured list overrides the type default."""
        sensor_states = SimpleNamespace(media=["playing"])
        assert resolve_active_states(sensor_states, InputType.MEDIA) == {"playing"}

    def test_empty_configured_list_falls_back_to_default(self) -> None:
        """An empty list means "use defaults", matching ``EntityType``."""
        sensor_states = SimpleNamespace(media=[])
        assert resolve_active_states(sensor_states, InputType.MEDIA) == set(
            DEFAULT_MEDIA_ACTIVE_STATES
        )

    def test_missing_field_falls_back_to_default(self) -> None:
        """``SensorStates`` has no 'sleep' field, so sleep uses its default."""
        sensor_states = SimpleNamespace(media=["playing"])
        assert resolve_active_states(sensor_states, InputType.SLEEP) == {"on"}

    def test_no_sensor_states_falls_back_to_default(self) -> None:
        """A missing config object still resolves to the type default."""
        assert resolve_active_states(None, InputType.MEDIA) == set(
            DEFAULT_MEDIA_ACTIVE_STATES
        )

    def test_by_type_reads_area_config(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Per-area resolution reflects that area's configured states."""
        area_name = coordinator.get_area_names()[0]
        coordinator.areas[area_name].config.sensor_states.media = ["playing"]

        resolved = area_active_states_by_type(
            coordinator, area_name, (InputType.MEDIA, InputType.SLEEP)
        )

        assert resolved[InputType.MEDIA] == {"playing"}
        assert resolved[InputType.SLEEP] == {"on"}

    def test_by_type_unknown_area_is_empty(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """An unknown area resolves nothing rather than guessing defaults."""
        assert (
            area_active_states_by_type(coordinator, "No Such Area", (InputType.MEDIA,))
            == {}
        )

    def test_entity_map_matches_live_entity_states(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """The per-entity map is the same data ``Entity.evidence`` reads."""
        resolved = entity_active_states(coordinator_with_sensors)
        area = coordinator_with_sensors.get_area(
            coordinator_with_sensors.get_area_names()[0]
        )
        for entity_id, entity in area.entities.entities.items():
            if entity.active_states:
                assert resolved[entity_id] == set(entity.active_states)

    def test_entity_map_tolerates_entity_without_active_states(self) -> None:
        """An entity object that doesn't expose active_states is skipped.

        Numeric sensors resolve an active *range* rather than a state list,
        and duck-typed stand-ins may carry neither. One unusual entity must
        not take the whole sync down; ``db.sync`` falls back to its historic
        "on" semantics for anything missing here.
        """
        entity = SimpleNamespace(entity_id="sensor.co2")
        area = SimpleNamespace(
            entities=SimpleNamespace(entities={"sensor.co2": entity})
        )
        coordinator = SimpleNamespace(areas={"Office": area})

        assert entity_active_states(coordinator) == {}

    def test_is_active_state_plain_match(self) -> None:
        """A state present in the set is active."""
        assert is_active_state("paused", {"playing", "paused"}) is True
        assert is_active_state("idle", {"playing", "paused"}) is False

    def test_is_active_state_empty_set_is_never_active(self) -> None:
        """No configured states means nothing counts as active."""
        assert is_active_state("on", set()) is False

    def test_is_active_state_maps_binary_to_semantic(self) -> None:
        """A window configured as 'open' matches the recorded 'on'.

        Binary sensors record on/off; semantic configs say open/closed. The
        live evidence path maps between them, so ground truth must too.
        """
        semantic = {DEFAULT_WINDOW_ACTIVE_STATE}
        assert semantic == {"open"}
        assert is_active_state("on", semantic) is True
        assert is_active_state("off", semantic) is False
